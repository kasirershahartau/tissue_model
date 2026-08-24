# Running the tissue-model simulations on Azure

A practical, step-by-step guide for running `create_random_arrays`,
`initialize_differentiated_arrays`, `find_mechanical_parameters` and
`find_psigma` on Azure, driven by [`azure_run.py`](azure_run.py).

---

## 0. Which setup? (decision in one line)

- **Single Spot VM** — recommended for everything. Simplest, works for all four
  methods, uses the existing `ProcessPoolExecutor` parallelism. **Start here.**
- **Azure Batch (many Spot nodes)** — only worth it when you need to generate
  *hundreds* of arrays (`create_random_arrays` / `initialize_differentiated_arrays`),
  which are pure fan-out. The adaptive fitters (`find_mechanical_parameters`,
  `find_psigma`) interleave compute with decisions and belong on one multi-core VM.

The simulations are **single-threaded, CPU-bound, low-memory, GPU-free** → use a
**compute-optimized Fsv2 VM** as a **Spot** instance (70–90 % cheaper; runs are
independent and resumable, so eviction is safe).

### ⚠️ Do not over-size the VM for `find-mech`

`find_mechanical_parameters` is **not** embarrassingly parallel:

```
for each of n_calls candidate points:        <- STRICTLY SEQUENTIAL (Bayesian)
    evaluate n_sheets sheets in parallel     <- only ~10 wide
        each sheet runs base, then ablation  <- sequential pair
```

`bayesian_optimization.minimize` proposes one point at a time and needs its
score before proposing the next, so **concurrency is capped at `n_sheets` (10),
not at the vCPU count**. A 72-vCPU VM leaves 62 cores idle and finishes *no
sooner* than a 16-vCPU one — while costing 4.5× more.

Measured over the 5976 archived fit runs (`results/fit_*/debug.log`):

| quantity | value |
|---|---|
| mean single simulation | ~0.4 h (median 0.31, p90 0.88) |
| mean sheet (base + ablation) | 38.7 min |
| **slowest of 10 sheets** (what an evaluation waits for) | **83.1 min** |
| core utilisation | **~47 %** (straggler factor ~2.1) |

For the current job (`n_calls=60`, `n_sheets=10`, ablation on):

| | |
|---|---|
| total CPU work | **~390 core-hours** |
| wall-clock on any VM with ≥10 usable cores | **~80 h (3.4 days)** |
| compute cost | ~$55 PAYG / **~$11 Spot** |

**So moving this fit to the cloud as-is buys ≈1× speedup** — only the per-core
clock differs. See [§12](#12-how-much-speedup-should-i-expect) for what actually
does make it faster.

---

## 1. Before you start — estimate the cost (local, no Azure needed)

The cost estimator in `azure_run.py` uses only the Python standard library, so
run it on your own machine first:

```bash
# 1a. Measure ONE simulation locally and time it (gives you hours-per-sim).
#     e.g. build a single random array and time it:
conda run -n tyssue python azure_run.py random-arrays --indices 0 --workers 1

# 1b. Plug that number in (0.32 h is this project's measured mean) to size a job:
python azure_run.py estimate find-mech --n-calls 60 --n-sheets 10 --ablation \
    --hours-per-sim 0.32 --vm Standard_F16s_v2
python azure_run.py estimate find-psigma --n-grid 11 --n-refine 2 --n-sheets 10 --hours-per-sim 0.32
python azure_run.py estimate random-arrays --n 10 --hours-per-sim 0.32
```

It prints the simulation count, total core-hours, single-VM wall-clock,
**core utilisation**, and **PAYG vs Spot** cost. A full fitting campaign is
**tens of dollars on Spot**. Omit `--hours-per-sim` for a scaling table.

The estimator models each method's *real* parallel structure, including the
sequential Bayesian loop and the straggler penalty — for `find-mech` it also
warns when the chosen VM has more cores than the job can ever use, and prints a
`--batch-k` what-if table. Compare `--vm Standard_F16s_v2` with
`--vm Standard_F72s_v2`: identical wall-clock, 4.5× the cost.

---

## 2. Prerequisites

- An Azure subscription with permission to create VMs (your lab account).
- **Spot quota** for the Fsv2 family in your region (Portal → *Quotas* →
  "Total Regional Spot vCPUs" and "Standard FSv2 Family vCPUs"; request an
  increase if it's 0).
- The **Azure CLI** installed locally and `az login` done.
- An **SSH** client.
- A region close to the lab for latency / data residency, e.g. `israelcentral`
  or `westeurope` (set `LOC` below).

---

## 3. Provision a Spot VM

```bash
RG=Simulations          # resource group
LOC=israelcentral            # or israelcentral
VM=SimulationsVM
# 16 vCPU is ENOUGH for find-mech with n_sheets=10 (see §0 -- concurrency is
# capped at n_sheets, so a bigger VM is pure cost). Go bigger only for the
# fan-out generators, or to run E17.5 and P0 on one machine.
SIZE=Standard_F16s_v2       # 16 vCPU / 32 GB

az group create -n $RG -l $LOC

# Spot VM, Deallocate on eviction so it can be restarted and the run resumed.
# Larger OS disk because results (HDF5 histories + gifs) accumulate.
az vm create -g $RG -n $VM \
  --image Ubuntu2204 --size $SIZE \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --os-disk-size-gb 128 \
  --admin-username azureuser --generate-ssh-keys

# Safety net: auto-shutdown at 02:00 UTC so a forgotten VM can't bill overnight.
az vm auto-shutdown -g $RG -n $VM --time 0200

# Note the public IP that `az vm create` prints (IP=...).
```

> **Linux vs Windows VM.** Ubuntu is ~40 % cheaper (no Windows licence) and is
> recommended. This codebase has a few **hard-coded Windows paths** (Step 5) that
> need a one-line fix on Linux. If you'd rather change nothing, you can instead
> create a Windows VM (`--image Win2022Datacenter`), replicate the
> `C:\Users\...` folder layout, and use `conda run -n tyssue` exactly as on your
> dev machine — at higher cost.

---

## 4. Base setup on the VM (needed for ALL methods)

SSH in (`ssh azureuser@<IP>`), then:

```bash
# 4a. System tools. ImageMagick is needed by run()'s final gif step.
sudo apt-get update && sudo apt-get install -y git imagemagick tmux

# 4b. Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/mc.sh
bash ~/mc.sh -b -p ~/miniconda
source ~/miniconda/etc/profile.d/conda.sh
conda config --add channels conda-forge

# 4c. Recreate the env. EASIEST: export it from your dev machine first
#     (on your machine:  conda env export -n tyssue > environment.yml ),
#     copy environment.yml up, then:
conda env create -f environment.yml          # creates the `tyssue` env
# --- or build it explicitly if you don't have the yml: ---
# conda create -n tyssue -c conda-forge python=3.10 tyssue numpy scipy pandas \
#     matplotlib statsmodels scikit-posthocs h5py odfpy
```

Get the simulation code onto the VM (from your machine):

```bash
# from your LOCAL machine, in the tissue_model folder.
# The VENDORED tyssue in ./tyssue/src MUST come along -- see 4d.
scp -r .  azureuser@<IP>:~/tissue_model
```

```bash
# 4d. TWO MANDATORY ENV VARS. Add both to ~/.bashrc.
cd ~/tissue_model

# (i) The vendored tyssue. This project does NOT work with the stock PyPI
#     tyssue -- it fails with "change of datatype in edge table". This has
#     bitten us for real; it is not optional.
export PYTHONPATH="$PWD/tyssue/src"

# (ii) Results directory. The default is the WINDOWS path D:\Kasirer\results,
#      which is meaningless on Linux, so every read/write would land somewhere
#      wrong or fail.
export TISSUE_RESULTS_DIR="$HOME/results"
mkdir -p "$TISSUE_RESULTS_DIR"
```

**Sanity check** the env on the VM. This verifies the LAPACK/`svd` backend *and*
that the vendored tyssue is the one being imported:

```bash
conda activate tyssue
cd ~/tissue_model
python -c "import numpy as np, tyssue, scipy; print('svd', np.linalg.svd(np.random.rand(6,3))[1].shape, 'ok'); print('tyssue from', tyssue.__file__)"
# ^ 'tyssue from' MUST print a path under ~/tissue_model/tyssue/src/
```

`azure_run.py find-mech` also checks both of these at startup and prints a loud
warning if either is wrong.

> **Always run inside the activated env** (`conda activate tyssue`, or
> `conda run -n tyssue ...`). Launching a bare `python` can miss the env's
> linear-algebra libraries and crash `statsmodels`/`numpy.linalg`.

At this point `create_random_arrays` and `initialize_differentiated_arrays`
(pure simulation, no experimental data) are ready to run — skip to Step 6.

---

## 5. Extra setup for the FITTING / ANALYSIS steps

Needed by anything that computes a p value against experiments: the
**initial-morphology fit** (`find_best_timepoint_for_random_arrays`, Step 7.2),
`find-mech`, and `find-psigma`. (The two generators, `random-arrays` and
`init-diff`, do **not** need this.)

These call `compare_*` in `post_processing.py`, which reads the **experimental
data** and imports the **`statistical_analysis`** module from the sibling
`tissue_image_processing` repo. Both locations are configurable via environment
variables (they default to the local Windows checkout), so on the VM you just
copy the folders up and point two env vars at them — **no file editing**:

```bash
# 5a. Copy the experimental data and the analyzer repo up (from your machine):
scp -r "/path/to/Experimental Data"  azureuser@<IP>:~/experimental_data
scp -r "/path/to/tissue_image_processing/tissue_analyzing_tool"  azureuser@<IP>:~/tissue_analyzing_tool
```

```bash
# 5b. On the VM: point post_processing.py at those folders (add to ~/.bashrc to
#     make them persistent across sessions):
export EXPERIMENTAL_DATA_DIR="$HOME/experimental_data"      # -> experimental_results_folder
export TISSUE_ANALYZER_PATH="$HOME/tissue_analyzing_tool"   # where statistical_analysis lives
```

> Both default to the original `C:\Users\Kasirer\...` paths when the env vars are
> unset, so nothing changes on your local machine.

---

## 6. Run a method (inside `tmux`, inside the env)

Runs take hours, so launch them in **`tmux`** so they survive an SSH
disconnect. `--workers` defaults to all vCPUs.

```bash
tmux new -s sim                 # detach with Ctrl-b d ; reattach: tmux attach -t sim
conda activate tyssue
cd ~/tissue_model
```

```bash
# Generators (pure simulation) -------------------------------------------------
python azure_run.py random-arrays --n 10
python azure_run.py random-arrays --indices 0 3 7          # only these indices
python azure_run.py init-diff --stage E17.5 --gammaSC 0.01 --gammaHC-ratio 10 --alphaHC-ratio 1 --psigma 0
python azure_run.py init-diff --stage P0    --indices 0 1 2

# Mechanical-parameter fit (Bayesian optimization), per stage ------------------
# ALL defaults mirror run_model.__main__, so this reproduces the local fit:
#   fitted:  gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index
#   fixed:   gammaHC_ratio=1.0, bending=0.02 (replaces line tension),
#            qst base/ablation = 0.03/0.02, type_by=delta_level
# ALWAYS --dry-run first: it prints the fully resolved call and runs nothing.
python azure_run.py find-mech --stage E17.5 --dry-run
python azure_run.py find-mech --stage E17.5
python azure_run.py find-mech --stage P0

# Override anything you need, e.g. a cheaper smoke test:
python azure_run.py find-mech --stage E17.5 --n-calls 4 --n-initial-points 3 --n-sheets 2

# Revert to the HISTORICAL parameterisation (single shape index, fitted
# gammaHC_ratio, line tension instead of bending). Passing --shape-index-bounds
# automatically drops the per-type pair, keeping exactly 4 fitted parameters:
python azure_run.py find-mech --stage E17.5 \
    --shape-index-bounds 1.1 1.4 --gammaHC-ratio-bounds 1.0 1.4 \
    --line-tension 0.05 --no-bending

# Shared psigma fit (coarse-to-fine line search over BOTH stages) --------------
python azure_run.py find-psigma \
    --mechanical-params '{"E17.5":[0.01,10,1],"P0":[0.02,8,1.5]}' \
    --psigma-bounds 0 2 --n-grid 11 --n-refine 2 --n-sheets 10
```

The intermediate **initial-morphology fitting** step (matching number-of-neighbors
per cell, which produces the `*_for_E17` / `*_for_P0` folders) lives in
`post_processing.py` and has no CLI subcommand; run it inline:

```bash
python -c "import post_processing as pp; \
pp.find_best_timepoint_for_random_arrays(indices=range(10)); \
pp.store_best_time_point_for_random_arrays()"
```

---

## 7. Recommended end-to-end pipeline order

```
1. random-arrays                      # generate raw random arrays
2. find_best_timepoint_for_random_arrays + store_best_time_point_for_random_arrays
                                       # fit initial MORPHOLOGY -> *_for_E17 / *_for_P0
3. find-mech  --stage E17.5            # fit MECHANICS per stage
   find-mech  --stage P0
4. find-psigma --mechanical-params ... # fit the single SHARED psigma
   (optional) init-diff                # produce final differentiated arrays at the fitted params
```

Use the **same array indices** throughout (`--indices`) so a chosen subset of
arrays flows cleanly from generation through every fit.

---

## 8. Monitor a running job

```bash
tmux attach -t sim                                # watch live console output
tail -f $TISSUE_RESULTS_DIR/<run-name>/debug.log  # per-run debug log
ls $TISSUE_RESULTS_DIR                            # results appear per simulation

# progress of a find-mech fit: one folder pair (base + _abl) per sheet per call
ls -d $TISSUE_RESULTS_DIR/fit_* | wc -l
```

---

## 9. Retrieve results and STOP billing

```bash
# from your LOCAL machine ($TISSUE_RESULTS_DIR on the VM, ~/results by default):
azcopy copy "azureuser@<IP>:~/results" "." --recursive   # or scp -r

# then, on Azure (compute billing stops on deallocate; disk is kept):
az vm deallocate -g $RG -n $VM
# fully tear down when done:
az group delete -n $RG --yes
```

Results can be large (HDF5 histories + gifs); `tar czf results.tgz results`
before downloading if bandwidth matters.

---

## 10. Optional — scale the generators with Azure Batch

For *hundreds* of arrays, fan out one Batch task per index instead of one VM.
`azure_run.py` includes a template `submit_batch_fanout(method, indices, ...)`
that emits, per index `i`, a task running e.g.
`python azure_run.py random-arrays --indices i --workers 1` on a pool of Spot
nodes. It needs `pip install azure-batch azure-storage-blob azure-identity`, a
Batch + Storage account, and a node start-task that recreates the `tyssue` env
and stages the code. It's a template (not run against a live account) — fill in
your account env vars (`AZ_BATCH_ACCOUNT`/`AZ_BATCH_KEY`/`AZ_BATCH_URL`) and the
start-task. Keep the fitters (`find-mech`, `find-psigma`) on a single VM.

---

## 11. Troubleshooting (gotchas this project actually hits)

| Symptom | Cause & fix |
|---|---|
| `ValueError: change of datatype in edge table` | The **stock PyPI tyssue** is being imported instead of the vendored one. `export PYTHONPATH=$PWD/tyssue/src` (Step 4d-i). This is the single most common setup failure. |
| Results written to a strange path, or `FileNotFoundError` on `D:\Kasirer\results` | `TISSUE_RESULTS_DIR` not set on Linux — the default is a Windows path (Step 4d-ii). |
| Fit dies with `not enough values to unpack (expected N, got M)` | **`run_model.py` was edited while the fit was running.** The parent keeps the old task-builder while newly spawned workers import the new unpack signature. Never edit the tree during a run — copy it and edit the copy. Completed run folders are reused, so restarting is cheap. |
| Fit found the same best point as a previous run and did almost no work | Working as intended: run folders are content-hashed, so matching parameter points are reused. Change a parameter that is folded into the hash (or delete the folders) to force fresh runs. |
| `numpy.linalg`/`statsmodels` hard-crash (e.g. `0xc06d007f` on Windows) | Running a bare `python` without env activation. **Always** `conda activate tyssue` / `conda run -n tyssue`. The Step-4 `svd` check verifies the backend. |
| `FileNotFoundError` on `...Experimental Data` or `ImportError: statistical_analysis` | The Step-5 env vars aren't set on the VM. `export EXPERIMENTAL_DATA_DIR=...` and `export TISSUE_ANALYZER_PATH=...` (point them at the copied folders). Affects the analysis steps: morphology fit (7.2), `find-mech`, `find-psigma` — not the generators. |
| Run finishes but no `movie.gif` / a gif error in the log | ImageMagick missing. `sudo apt-get install -y imagemagick` (Step 4a). The history is still saved regardless. |
| VM disappears mid-run (Spot eviction) | Expected. Restart with `az vm start -g $RG -n $VM`; reruns reuse existing result folders, and `run(continue_existing_run=True, ...)` resumes a partial archive. |
| `psigma` seems to have no effect | `psigma` only matters with stress-dependent differentiation. `find-psigma`'s worker already passes `stress_dependent=True`; if you call `run()` directly, set it too. |
| SSH drops kill the job | Launch inside `tmux` (Step 6). |

---

## 12. How much speedup should I expect?

Short answer: **for one `find-mech` fit, moving to the cloud unchanged gives
≈1×.** The bottleneck is not core count.

### Where the ~80 h actually goes

The job is only **~390 core-hours** of real CPU work. If it could be spread
freely across cores it would finish in a few hours. It doesn't, because:

1. **The Bayesian loop is sequential** — 60 evaluations, each needing the
   previous score. Only the 10 sheets *within* an evaluation run concurrently.
2. **Each evaluation waits for its slowest sheet** — mean sheet 38.7 min but
   max-of-10 83.1 min, so **~47 % of the 10 cores sit idle**.

Net: ~60 × 1.4 h ≈ **80 h**, on 10 effective cores, on *any* VM size. Your local
16-core machine already saturates this, which is exactly why the local run takes
~80–100 h.

### What each option actually buys

| Option | Speedup (one fit) | Effort |
|---|---|---|
| Same code on a bigger cloud VM | **~1×** (only per-core clock; Fsv2 ≈ a modern desktop) | none |
| Run E17.5 **and** P0 on two VMs at once | 1× each, but **2× throughput** — both stages in ~80 h instead of ~160 h, and your workstation is free | none |
| Launch the 25 random initial points concurrently | ~1.5–1.7× (42 % of the budget has *no* sequential dependency, so this costs nothing in sample efficiency) | small |
| **Batch BO** (`--batch-k K`: K candidate points in flight) | **K=4 → ~3×** (~27 h, 40 vCPU); **K=6 → ~4.3×** (~19 h, 60 vCPU) | moderate — needs a batch-capable optimizer |
| Both of the above | **~6×** → ~13 h | moderate |

Speedups above are projected by bootstrapping the archived run-time
distribution; run `azure_run.py estimate find-mech --batch-k K ...` to reproduce
them. They flatten past K≈6 because the straggler penalty grows with batch size
(waiting on the slowest of 60 is worse than the slowest of 10).

### Recommendation

- **Right now:** let the local fit finish, and put the *other* stage on one
  16-vCPU Spot VM (~$11). That is the whole 2× with zero code risk.
- **Before the next fitting campaign:** add batch proposals to
  `bayesian_optimization.minimize`. That is the only change that turns money
  into wall-clock here — and given that both previous fits found their best
  point during the *random* initial phase, batching should cost little or
  nothing in fit quality.
- **Cost is never the constraint:** the entire fit is ~$11 on Spot. Do not
  optimise for it; optimise for wall-clock and for not corrupting a run.

Re-running after restart:
source ~/miniconda/etc/profile.d/conda.sh
conda activate tyssue
---Launch both stages----
tmux new -s e17
~/tissue_model/run_mech_fit.sh E17.5
Detach with Ctrl-b d, then:

tmux new -s p0
~/tissue_model/run_mech_fit.sh P0
Monitor: tmux attach -t e17, tail -f ~/mech_fit_E17.5.log, or ls -d ~/results/fit_*pa0.466* | wc -l.

-----When done — retrieve and stop billing-----
scp azureuser@<IP>:"~/results/*_optimization_*" "D:\Kasirer\results\"
az vm deallocate -g Simulations -n SimFit

scp azureuser@<IP>:/home/azureuser/results/grid_fit_mechanics_v2_E17.5.json D:/Kasirer/results/

-----P0 step-5 scans (v2 fit) — launch in tmux-----
Upload (from the local tissue_model folder):
scp -i SimulationsVM_key.pem p0_rgamma_scan.py p0_gamma_scan.py p0_from_e17_stiffness.py post_processing.py run_p0_scan.sh azureuser@<IP>:~/tissue_v2/
ssh azureuser@<IP> chmod +x ~/tissue_v2/run_p0_scan.sh

THE TREE MATTERS: the v2 work lives in ~/tissue_v2 (~/tissue_model and
~/tissue_full are older checkouts and have no grid_fit_mechanics_v2.py).
run_p0_scan.sh runs from ITS OWN directory, so invoke the copy in the tree
you mean, and it checks its siblings are present before starting.

run_p0_scan.sh sets conda + PYTHONPATH + the three env vars itself (a fresh tmux
pane does NOT source ~/.bashrc) and pre-flights the workbook, the E17.5 grid
JSON and every experimental term before starting.

tmux new -s p0scan
~/tissue_v2/run_p0_scan.sh rgamma --dry-run    # check, then Ctrl-c and:
~/tissue_v2/run_p0_scan.sh rgamma              # 70 tasks
Detach with Ctrl-b d. Reattach: tmux attach -t p0scan

  rgamma  step 5c diagnostic: R_alpha pinned by the stress ratio, R_gamma swept
  gamma   step 5b: R fixed, 10 gammaSC values (90 tasks)

Monitor: tmux attach -t p0scan, or tail -f ~/p0_rgamma_scan.log
Retrieve:
scp azureuser@<IP>:/home/azureuser/results/p0_rgamma_scan.json D:/Kasirer/results/
