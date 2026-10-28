# Running the tissue-model simulations on Azure

A practical, step-by-step guide for running `create_random_arrays`,
`initialize_differentiated_arrays`, `find_mechanical_parameters` and
`find_psigma` on Azure, driven by [`azure_run.py`](azure_run.py).

---

## 0. Which setup? (decision in one line)

- **Single pay-as-you-go VM** — recommended for everything. Simplest, works for
  all four methods, uses the existing `ProcessPoolExecutor` parallelism.
  **Start here.**
- **Azure Batch (many nodes)** — only worth it when you need to generate
  *hundreds* of arrays (`create_random_arrays` / `initialize_differentiated_arrays`),
  which are pure fan-out. The adaptive fitters (`find_mechanical_parameters`,
  `find_psigma`) interleave compute with decisions and belong on one multi-core VM.

The simulations are **single-threaded, CPU-bound, low-memory, GPU-free** → use a
**compute-optimized Fsv2 VM**, pay-as-you-go.

> **Why not Spot?** It is 70–90 % cheaper and the runs are resumable, so on
> paper eviction is harmless. In practice our Spot VMs were deallocated
> repeatedly and the babysitting cost far more than the compute saved — a whole
> fit is tens of dollars either way. Use PAYG; `--priority Spot` is a knowing
> trade of your time for a few dollars.

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
| compute cost | **~$55 PAYG** (~$11 on Spot, not recommended — see §0) |

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
**core utilisation**, and cost. A full fitting campaign is **tens of dollars**
PAYG. It always prints a PAYG line and a Spot line — read the PAYG one; the
Spot line assumes an 80 % saving we no longer take (`--spot-discount 0`
collapses it onto PAYG if you would rather not see it). Omit `--hours-per-sim`
for a scaling table.

The estimator models each method's *real* parallel structure, including the
sequential Bayesian loop and the straggler penalty — for `find-mech` it also
warns when the chosen VM has more cores than the job can ever use, and prints a
`--batch-k` what-if table. Compare `--vm Standard_F16s_v2` with
`--vm Standard_F72s_v2`: identical wall-clock, 4.5× the cost.

---

> **Which shell each block is for.** `bat` blocks run **locally, in Windows
> Command Prompt** (`az`, `scp`, `ssh`). `bash` blocks run **on the VM**, which
> is Ubuntu — paste those after `ssh`.
>
> Mixing them up is the usual first stumble: `RG=Simulations` in cmd fails with
> *"'RG' is not recognized as an internal or external command"*, because the
> syntax differs on every axis that matters here.
>
> | | set a variable | use it | line continuation | comment |
> |---|---|---|---|---|
> | cmd.exe (local) | `set RG=Simulations` (no spaces around `=`) | `%RG%` | `^` | `rem` |
> | bash (on the VM) | `RG=Simulations` | `$RG` | `\` | `#` |
>
> The local blocks keep each `az` call on one line so there are no
> continuations to get wrong, and they use `%IP%` rather than a literal
> `<IP>` — cmd reads `<` and `>` as redirection, so a pasted `<IP>` errors
> rather than prompting you to substitute it.
>
> From **PowerShell** instead: `$RG = "Simulations"`, used as `$RG`, with a
> backtick for continuation.

## 2. Prerequisites

- An Azure subscription with permission to create VMs (your lab account).
- **vCPU quota** for the Fsv2 family in your region (Portal → *Quotas* →
  "Standard FSv2 Family vCPUs"; request an increase if it is below the core
  count you plan to rent — 48 for the pT re-fit in §9).
- The **Azure CLI** installed locally and `az login` done.
- An **SSH** client.
- A region close to the lab for latency / data residency, e.g. `israelcentral`
  or `westeurope` (set `LOC` below).

---

## 3. Provision a VM

```bat
rem 16 vCPU is ENOUGH for find-mech with n_sheets=10 (see section 0 --
rem concurrency is capped at n_sheets, so a bigger VM is pure cost). Go
rem bigger only for the fan-out generators, or to run both stages at once.
set RG=Simulations
set LOC=israelcentral
set VM=SimulationsVM
set SIZE=Standard_F16s_v2

az group create -n %RG% -l %LOC%

rem Pay-as-you-go (see section 0 on Spot). Larger OS disk because results
rem (HDF5 histories + gifs) accumulate. For the pT re-fit, put them on a
rem data disk instead -- 9.2.
az vm create -g %RG% -n %VM% --image Ubuntu2204 --size %SIZE% --os-disk-size-gb 128 --admin-username azureuser --generate-ssh-keys

rem Safety net so a forgotten VM cannot bill overnight. NOTE: auto-shutdown is
rem a DevTest Labs feature and is missing in some regions (israelcentral among
rem them) -- if this errors, use the self-deallocating setup in 9.1 / 9.6.
az vm auto-shutdown -g %RG% -n %VM% --time 0200

rem Keep the publicIpAddress it prints; later steps use %IP%.
set IP=20.0.0.1
```

> **Linux vs Windows VM.** Ubuntu is ~40 % cheaper (no Windows licence) and is
> recommended. This codebase has a few **hard-coded Windows paths** (Step 5) that
> need a one-line fix on Linux. If you'd rather change nothing, you can instead
> create a Windows VM (`--image Win2022Datacenter`), replicate the
> `C:\Users\...` folder layout, and use `conda run -n tyssue` exactly as on your
> dev machine — at higher cost.

---

## 4. Base setup on the VM (needed for ALL methods)

SSH in (`ssh azureuser@%IP%`), then:

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
#     matplotlib pytables openpyxl
# plus, ONLY if you also want the hierarchical p-value tooling on the VM
# (statistical_analysis, §5 -- the pT sweep does not use it):
#     statsmodels scikit-posthocs odfpy
```

Get the simulation code onto the VM (from your machine):

```bat
rem From your LOCAL machine, in the tissue_model folder. Stage a filtered copy
rem first -- do NOT `scp -r .` the folder as it stands (see the warning below).
rem The VENDORED tyssue in .\tyssue\src MUST come along -- see 4d.
set IP=20.0.0.1
robocopy . %TEMP%\tm_upload /E /XD .git __pycache__ .idea .pytest_cache /XF *.pem /NFL /NDL /NJH /NJS
scp -r %TEMP%\tm_upload azureuser@%IP%:~/tissue_model
rd /s /q %TEMP%\tm_upload
```

> **Why not plain `scp -r .`?** Two reasons, one of them a security one.
>
> * It copies **`SimulationsVM_key.pem`** — a private SSH key — onto the VM,
>   where it would also survive into any snapshot or image of that machine.
>   That key is gitignored precisely because it should not travel, and nothing
>   on the VM needs it.
> * It is 187 MB / 1233 files, of which **110 MB is `.git`**, plus test
>   fixtures (`*.hf5`), figures and `__pycache__`. The filtered copy above is
>   **74 MB / 476 files** and carries everything the sweep and the §9.5 tests
>   need, the vendored `tyssue/src` included.
>
> **Do not reach for `git archive` instead.** It ships the last *commit*, so
> any uncommitted fix — the edge-tension correction and the `find_psigma`
> changes among them — would be silently left behind, and the VM would run the
> old model while looking perfectly healthy.

```bash
# 4d. TWO MANDATORY ENV VARS. Add both to ~/.bashrc.
cd ~/tissue_model

# (i) The vendored tyssue. This project does NOT work with the stock PyPI
#     tyssue -- it fails with "change of datatype in edge table". This has
#     bitten us for real; it is not optional.
export PYTHONPATH="$PWD/tyssue/src"

# (ii) Results directory. The default is the WINDOWS path D:\kasirer\model_results,
#      which is meaningless on Linux, so every read/write would land somewhere
#      wrong or fail.
export TISSUE_RESULTS_DIR="$HOME/results"
mkdir -p "$TISSUE_RESULTS_DIR"
```

> **`pytables`, not `h5py`.** Every history archive is read and written through
> `pandas.HDFStore`, which is backed by **PyTables** — nothing in this codebase
> imports `h5py` at all. The list used to name `h5py`; if conda happens not to
> pull PyTables in as a `tyssue` dependency, every run dies at its first save.
> `openpyxl` is for the `.xlsx` the post-processing writes (§9.7).
>
> `statsmodels` / `scikit-posthocs` / `odfpy` are only needed by
> `statistical_analysis` (the hierarchical p-value compare, §5). The pT sweep
> and `build_run_table` run without them — verified by importing the whole
> find-psigma path on a machine that has none of the three.
>
> Expect a slow solve either way: `tyssue` drags in a notebook/visualisation
> stack (ipyvolume, pythreejs, vispy, bqplot, ipywidgets) that a headless VM
> never uses but that must still import cleanly.

**Sanity check** the env on the VM. This verifies the LAPACK/`svd` backend *and*
that the vendored tyssue is the one being imported:

```bash
conda activate tyssue
cd ~/tissue_model
python -c "import numpy as np, tyssue, scipy; print('svd', np.linalg.svd(np.random.rand(6,3))[1].shape, 'ok'); print('tyssue from', tyssue.__file__)"
# ^ 'tyssue from' MUST print a path under ~/tissue_model/tyssue/src/

# And the HDF5 backend, which is how EVERY run saves its history. Without
# PyTables this raises here, in a second, instead of after the first run.
python -c "import pandas as pd, numpy as np, tempfile, os; p=os.path.join(tempfile.mkdtemp(),'t.h5'); pd.DataFrame({'a':np.arange(3.0)}).to_hdf(p,key='x'); print('HDFStore ok', pd.read_hdf(p,key='x').shape)"
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

```bat
rem 5a. Copy the experimental data and the analyzer repo up (from your machine):
scp -r "C:\path\to\Experimental Data" azureuser@%IP%:~/experimental_data
scp -r "C:\path\to\tissue_image_processing\tissue_analyzing_tool" azureuser@%IP%:~/tissue_analyzing_tool
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

Both steps use the production binning scheme, `post_processing.NEIGHBOR_BINS`
(merged tails: `<=4, 5, 6, 7, >=8`), and talk to each other through that
scheme's own chosen-frame file, `pval_results_mergedtails.txt`. Pass
`bins=pp._NEIGHBOR_BINS` to both calls for the single-count 3–9 alternative;
it has its own file (`pval_results.txt`) and its own per-run traces, so the two
can coexist without either reading the other's frames.

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

```bat
rem from your LOCAL machine (TISSUE_RESULTS_DIR on the VM, ~/results by default):
scp -r azureuser@%IP%:~/results .

rem compute billing stops on deallocate; the disk is kept:
az vm deallocate -g %RG% -n %VM%
rem fully tear down when done:
az group delete -n %RG% --yes
```

Results can be large (HDF5 histories + gifs); `tar czf results.tgz results`
before downloading if bandwidth matters.

---

## 10. Optional — scale the generators with Azure Batch

For *hundreds* of arrays, fan out one Batch task per index instead of one VM.
`azure_run.py` includes a template `submit_batch_fanout(method, indices, ...)`
that emits, per index `i`, a task running e.g.
`python azure_run.py random-arrays --indices i --workers 1` on a pool of
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
| Results written to a strange path, or `FileNotFoundError` on `D:\kasirer\model_results` | `TISSUE_RESULTS_DIR` not set on Linux — the default is a Windows path (Step 4d-ii). |
| Fit dies with `not enough values to unpack (expected N, got M)` | **`run_model.py` was edited while the fit was running.** The parent keeps the old task-builder while newly spawned workers import the new unpack signature. Never edit the tree during a run — copy it and edit the copy. Completed run folders are reused, so restarting is cheap. |
| Fit found the same best point as a previous run and did almost no work | Working as intended: run folders are content-hashed, so matching parameter points are reused. Change a parameter that is folded into the hash (or delete the folders) to force fresh runs. |
| `numpy.linalg`/`statsmodels` hard-crash (e.g. `0xc06d007f` on Windows) | Running a bare `python` without env activation. **Always** `conda activate tyssue` / `conda run -n tyssue`. The Step-4 `svd` check verifies the backend. |
| `FileNotFoundError` on `...Experimental Data` or `ImportError: statistical_analysis` | The Step-5 env vars aren't set on the VM. `export EXPERIMENTAL_DATA_DIR=...` and `export TISSUE_ANALYZER_PATH=...` (point them at the copied folders). Affects the analysis steps: morphology fit (7.2), `find-mech`, `find-psigma` — not the generators. |
| Run finishes but no `movie.gif` / a gif error in the log | ImageMagick missing. `sudo apt-get install -y imagemagick` (Step 4a). The history is still saved regardless. |
| VM disappears mid-run | Should not happen on PAYG; it is what drove us off Spot. Either way, restart with `az vm start -g $RG -n $VM`: reruns reuse existing result folders, and `run(continue_existing_run=True, ...)` resumes a partial archive. |
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
  16-vCPU VM (~$55 PAYG). That is the whole 2× with zero code risk.
- **Before the next fitting campaign:** add batch proposals to
  `bayesian_optimization.minimize`. That is the only change that turns money
  into wall-clock here — and given that both previous fits found their best
  point during the *random* initial phase, batching should cost little or
  nothing in fit quality.
- **Cost is never the constraint:** the entire fit is ~$55 PAYG. Do not
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
scp azureuser@%IP%:"~/results/*_optimization_*" "D:\Kasirer\results\"
az vm deallocate -g Simulations -n SimFit

scp azureuser@%IP%:/home/azureuser/results/grid_fit_mechanics_v2_E17.5.json D:/Kasirer/results/

-----P0 step-5 scans (v2 fit) — launch in tmux-----
Upload (from the local tissue_model folder):
scp -i SimulationsVM_key.pem p0_rgamma_scan.py p0_gamma_scan.py p0_from_e17_stiffness.py post_processing.py run_p0_scan.sh azureuser@%IP%:~/tissue_v2/
ssh azureuser@%IP% chmod +x ~/tissue_v2/run_p0_scan.sh

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
scp azureuser@%IP%:/home/azureuser/results/p0_rgamma_scan.json D:/Kasirer/results/

---

## 9. The pT re-fit (phase 1) — step by step

The psigma sweep differs from `find-mech` in two ways that change the recipe:
it is **storage-heavy** (~0.9 GB per run against 15 MB) and, since the pass is
now batched, it **actually uses all the cores you rent**. A dedicated data disk
and a bigger VM are both justified here; §0's "16 vCPU is enough" does not
apply.

Budget: **~8 h, ~$30** (F48s_v2 PAYG + a 2 TiB disk for two days).

### 9.1 Provision — non-Spot, with a data disk

```bat
set RG=Simulations
set LOC=israelcentral
set VM=PsigmaVM
set SIZE=Standard_F48s_v2

az group create -n %RG% -l %LOC%

rem NO --priority Spot: an 8-hour run gets evicted repeatedly, and the
rem compute is only ~$16. A 2 TiB P40 data disk comes up as LUN 0.
az vm create -g %RG% -n %VM% --image Ubuntu2204 --size %SIZE% --os-disk-size-gb 128 --data-disk-sizes-gb 2048 --admin-username azureuser --generate-ssh-keys

rem Keep the publicIpAddress it prints -- every later step needs it.
rem Substitute the real address; do NOT type angle brackets, cmd reads
rem < and > as redirection.
set IP=20.0.0.1
```

**Then give the VM permission to stop itself.** `az vm auto-shutdown` is a
DevTest Labs feature and does not exist in every region (israelcentral is one
of the gaps), so instead of a wall-clock shutdown the VM deallocates itself the
moment the sweep ends — better anyway, since it stops billing when the work
stops rather than at a guessed hour. A system-assigned identity, scoped to this
one VM:

```bat
rem Capture the identity's object id and the VM's resource id into variables.
rem Typed at the prompt this is %i; inside a .bat file it must be %%i.
for /f "usebackq delims=" %i in (`az vm identity assign -g %RG% -n %VM% --query systemAssignedIdentity -o tsv`) do set OID=%i
for /f "usebackq delims=" %i in (`az vm show -g %RG% -n %VM% --query id -o tsv`) do set VMID=%i

echo %OID%
echo %VMID%

az role assignment create --assignee-object-id %OID% --assignee-principal-type ServicePrincipal --role "Virtual Machine Contributor" --scope %VMID%
```

Both `echo`s must print a value before you run the role assignment. An empty
`%OID%` means the identity was assigned on an earlier attempt, in which case
read it back with
`az vm show -g %RG% -n %VM% --query identity.principalId -o tsv`.

**Careful:** shutting the guest down (`sudo poweroff`) leaves the VM *Stopped*,
not *Stopped (deallocated)*, and Azure keeps billing compute. Only
`az vm deallocate` stops the meter — which is what §9.6 chains to the sweep.

### 9.2 Format and mount the data disk

```bash
ssh azureuser@%IP%
lsblk                                   # the 2 TiB disk is usually /dev/sdc
sudo mkfs.ext4 -F /dev/sdc
sudo mkdir -p /data && sudo mount /dev/sdc /data
sudo chown -R azureuser:azureuser /data
# survive a reboot
echo "UUID=$(sudo blkid -s UUID -o value /dev/sdc) /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
df -h /data                             # must show ~2.0T available
```

### 9.3 Base setup

Follow **§4** (apt packages, Miniconda, the `tyssue` env, `scp` the repo). Then
set **three** environment variables instead of two — add all three to
`~/.bashrc`:

```bash
cd ~/tissue_model
export PYTHONPATH="$PWD:$PWD/tyssue/src"
export TISSUE_RESULTS_DIR=/data/results          # the DATA DISK, not $HOME
export EXPERIMENTAL_DATA_DIR=$HOME/experimental  # scoring reads this
mkdir -p "$TISSUE_RESULTS_DIR"
```

`$HOME/results` would fill the 128 GB OS disk after ~130 runs. The sweep needs
~1.1 TB.

### 9.4 Upload the inputs (~1.2 GB total)

From your local machine:

```bat
rem The 20 fitted initial morphologies the runs start from (0.2 GB).
rem cmd does NOT expand wildcards for scp, so loop over the folders. At the
rem prompt this is %d; inside a .bat file it must be %%d.
for /d %d in (D:\Kasirer\results\random_periodic_array*_for_E17) do scp -r "%d" azureuser@%IP%:/data/results/
for /d %d in (D:\Kasirer\results\random_periodic_array*_for_P0) do scp -r "%d" azureuser@%IP%:/data/results/

rem the experimental measurements the scoring compares against (0.9 GB)
scp -r "C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data" azureuser@%IP%:~/experimental
```

Confirm all twenty landed before going on:

```bat
ssh azureuser@%IP% "ls -d /data/results/*_for_* | wc -l"
```

Do **not** upload the pT = 0 baseline runs (54 GB). pT = 0 is excluded from the
sweep and merged locally afterwards — which is also why `--psigma-bounds`
starts at 0.06: the refinement window is clamped to the lower bound, so no pass
can wander onto zero and re-simulate a baseline that is not there.

### 9.5 Verify before spending anything

```bash
conda activate tyssue
cd ~/tissue_model

# (a) the vendored tyssue is the one imported (§4)
python -c "import tyssue; print(tyssue.__file__)"

# (b) the parts of the fit we changed
python -m pytest test_periodic.py -k "FindPsigma or EdgeTension" -q

# (c) end-to-end plumbing on a THROWAWAY results dir, so a deliberately
#     truncated run can never be reused by the real sweep (the folder name
#     does not encode t_end)
TISSUE_RESULTS_DIR=/data/smoke python azure_run.py find-psigma \
  --mechanical-params '{"E17.5":{"gamma_sc":0.0105,"R_gamma":6.746124660721616,"R_alpha":3.5,"A0":0.741812389049632},"P0":{"gamma_sc":0.0105,"R_gamma":3.850521945713907,"R_alpha":1.757,"A0":0.7585423232230211}}' \
  --psigma-bounds 0.09 0.09 --n-grid 1 --n-refine 0 \
  --n-sheets 1 --n-repeats 1 --t-end 2 --save-interval 0.5 --workers 2
```

(c) must print the resolved mechanics per stage (`gammaHC_ratio=6.74612` for
E17.5, `3.85052` for P0 — if it says 1.0, the mechanical-params JSON did not
parse), run 2 simulations, score them, and write `psigma_scores_*.json`. The
SCORE is meaningless at `t_end 2`; only the plumbing is being tested. Then:

```bash
rm -rf /data/smoke          # nothing in it is reusable
```

### 9.6 The real sweep

```bash
tmux new -s psigma            # survives the ssh session dropping
conda activate tyssue && cd ~/tissue_model

{ python azure_run.py find-psigma \
    --mechanical-params '{"E17.5":{"gamma_sc":0.0105,"R_gamma":6.746124660721616,"R_alpha":3.5,"A0":0.741812389049632},"P0":{"gamma_sc":0.0105,"R_gamma":3.850521945713907,"R_alpha":1.757,"A0":0.7585423232230211}}' \
    --psigma-bounds 0.06 0.12 --n-grid 7 --n-refine 2 \
    --n-sheets 10 --n-repeats 3 --workers 46 \
    2>&1 | tee /data/psigma_sweep.log ; } ; \
  az vm deallocate -g Simulations -n PsigmaVM --no-wait
```

`--workers 46`, not 48: two cores are left for the OS, the HDF5 writes and the
scoring parent. Detach with `ctrl-b d`, reattach with `tmux attach -t psigma`.

The trailing `az vm deallocate` runs **however the sweep exits** — finished,
crashed or killed — so compute billing stops with the work rather than at a
guessed hour. Nothing is lost if it fires early: the log and every result sit
on the data disk, which survives deallocation, and `az vm start` plus the same
command resumes from the completed folders. Note the `;` before it rather than
`&&`: `&&` would skip the deallocate exactly when the sweep failed and you most
want the machine off. Run §9.7 first if you want the tables built on the VM, or
simply start it again afterwards — that part is a minute's work.

Watch, from a second shell:

```bash
tail -f /data/psigma_sweep.log | grep -E "^\[find_psigma\]"
df -h /data                   # the number to watch; ~1.1 TB expected
```

Each pass prints `pass: N psigma x 2 stage(s) x 10 array(s) x 3 repeat(s)`, and
scores are written to `/data/results/psigma_scores_*.json` after **every**
point — so an interruption costs only the point in flight, and re-running the
same command resumes (completed folders are reused, and the score cache skips
re-scoring points whose runs have not changed).

### 9.7 Post-process ON the VM, then download only the small files

```bash
python build_run_table.py --pattern 'fullmodel_v2*_ps*' --workers 46
```

(`--pattern`, not `--prefix`; the glob keeps it to the swept points, since the
pT = 0 baseline is not on this machine.)

Then bring back the tables, not the histories (1.1 TB of egress would cost more
than the whole campaign):

```bat
rem from your local machine. The * is expanded by the REMOTE shell, so it
rem works from cmd even though cmd itself does not glob.
scp azureuser@%IP%:/data/results/psigma_scores_*.json .
scp azureuser@%IP%:/data/results/fullmodel_runs.csv .
scp azureuser@%IP%:/data/psigma_sweep.log .
```

The pT = 0 baseline is merged here, locally, where its 60 runs already live. It
is 3 repeats too, so it joins the sweep on equal footing — do not merge a
baseline scored on a different repeat count.

### 9.8 Tear down (the disk bills while the VM is off)

```bat
rem deallocate stops COMPUTE only -- the disk keeps billing
az vm deallocate -g %RG% -n %VM%
rem this removes the VM AND the disk
az group delete -n %RG% --yes
```

A 2 TiB disk left behind is ~$6.5/day whether or not the VM runs — more than
the sweep itself if forgotten for a week. Snapshot first if you want the
histories kept.
