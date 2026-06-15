# Running the tissue-model simulations on Azure

A practical, step-by-step guide for running `create_random_arrays`,
`initialize_differentiated_arrays`, `find_mechanical_parameters` and
`find_psigma` on Azure, driven by [`azure_run.py`](azure_run.py).

---

## 0. Which setup? (decision in one line)

- **Single large Spot VM** — recommended for everything. Simplest, works for all
  four methods, uses the existing `ProcessPoolExecutor` parallelism across the
  VM's cores. **Start here.**
- **Azure Batch (many Spot nodes)** — only worth it when you need to generate
  *hundreds* of arrays (`create_random_arrays` / `initialize_differentiated_arrays`),
  which are pure fan-out. The adaptive fitters (`find_mechanical_parameters`,
  `find_psigma`) interleave compute with decisions and belong on one multi-core VM.

The simulations are **single-threaded, CPU-bound, low-memory, GPU-free** → use a
**compute-optimized Fsv2 VM** as a **Spot** instance (70–90 % cheaper; runs are
independent and resumable, so eviction is safe).

---

## 1. Before you start — estimate the cost (local, no Azure needed)

The cost estimator in `azure_run.py` uses only the Python standard library, so
run it on your own machine first:

```bash
# 1a. Measure ONE simulation locally and time it (gives you hours-per-sim).
#     e.g. build a single random array and time it:
conda run -n tyssue python azure_run.py random-arrays --indices 0 --workers 1

# 1b. Plug that number in (say 0.5 h) to size each job:
python azure_run.py estimate find-mech   --n-calls 40 --n-sheets 10 --ablation --hours-per-sim 0.5 --vm Standard_F32s_v2
python azure_run.py estimate find-psigma --n-grid 11 --n-refine 2 --n-sheets 10 --hours-per-sim 0.5
python azure_run.py estimate random-arrays --n 10 --hours-per-sim 0.5
```

It prints the simulation count, idealized core-hours, single-VM wall-clock, and
**PAYG vs Spot** dollar cost. A full fitting campaign is typically **tens of
dollars on Spot**. Omit `--hours-per-sim` to get a scaling table instead.

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
RG=tissue-model-rg          # resource group
LOC=westeurope              # or israelcentral
VM=tissue-sim
SIZE=Standard_F32s_v2       # 32 vCPU / 64 GB; use F72s_v2 for 72 vCPU

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
# from your LOCAL machine, in the tissue_model folder:
scp -r .  azureuser@<IP>:~/tissue_model
```

**Sanity check** the env on the VM (this is also the test that the
LAPACK/`svd` backend works — see Troubleshooting):

```bash
conda activate tyssue
cd ~/tissue_model
python -c "import numpy as np, tyssue, scipy; print('svd', np.linalg.svd(np.random.rand(6,3))[1].shape, 'ok')"
```

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
python azure_run.py find-mech --stage E17.5 --n-calls 40 --n-sheets 10 \
    --ablated-cells 12 13 14 15 --post-ablation-frame 4
python azure_run.py find-mech --stage P0    --n-calls 40 --n-sheets 10

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
tmux attach -t sim                     # watch live console output
tail -f ~/tissue_model/results/<run-name>/debug.log   # per-run debug log
ls ~/tissue_model/results              # results appear per simulation
```

---

## 9. Retrieve results and STOP billing

```bash
# from your LOCAL machine:
azcopy copy "azureuser@<IP>:~/tissue_model/results" "." --recursive   # or scp -r

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
| `numpy.linalg`/`statsmodels` hard-crash (e.g. `0xc06d007f` on Windows) | Running a bare `python` without env activation. **Always** `conda activate tyssue` / `conda run -n tyssue`. The Step-4 `svd` check verifies the backend. |
| `FileNotFoundError` on `...Experimental Data` or `ImportError: statistical_analysis` | The Step-5 env vars aren't set on the VM. `export EXPERIMENTAL_DATA_DIR=...` and `export TISSUE_ANALYZER_PATH=...` (point them at the copied folders). Affects the analysis steps: morphology fit (7.2), `find-mech`, `find-psigma` — not the generators. |
| Run finishes but no `movie.gif` / a gif error in the log | ImageMagick missing. `sudo apt-get install -y imagemagick` (Step 4a). The history is still saved regardless. |
| VM disappears mid-run (Spot eviction) | Expected. Restart with `az vm start -g $RG -n $VM`; reruns reuse existing result folders, and `run(continue_existing_run=True, ...)` resumes a partial archive. |
| `psigma` seems to have no effect | `psigma` only matters with stress-dependent differentiation. `find-psigma`'s worker already passes `stress_dependent=True`; if you call `run()` directly, set it too. |
| SSH drops kill the job | Launch inside `tmux` (Step 6). |
```
