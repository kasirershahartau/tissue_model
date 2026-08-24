"""azure_run.py -- run the tissue-model simulation methods on Azure.

This is both (1) a command-line *driver* you execute ON an Azure VM to run any
of the four simulation entry points with full multi-core parallelism, and (2) a
*cost estimator* you can run anywhere (no heavy deps) to size a job before
paying for it. An optional Azure Batch fan-out template is included at the
bottom for scaling the embarrassingly-parallel generators across many nodes.

------------------------------------------------------------------------------
WHY F-SERIES + SPOT
------------------------------------------------------------------------------
The simulations are single-threaded, CPU-bound, low-memory, GPU-free, and
mutually independent. The cheapest way to run many of them is therefore a
*compute-optimized* VM with many vCPUs, run as a *Spot* instance:

  * Compute-optimized  -> Azure **Fsv2** family (high vCPU-to-RAM ratio, fast
    cores). Good sizes: Standard_F32s_v2 (32 vCPU / 64 GB) or
    Standard_F72s_v2 (72 vCPU / 144 GB).
  * Spot pricing       -> typically 70-90% cheaper than pay-as-you-go (PAYG).
    Safe here because runs are independent and resumable.

General-purpose Dsv5/Dasv5 work too (similar $/vCPU); avoid GPU (N-series) and
memory-optimized (E-series) -- you would pay for hardware this job never uses.

------------------------------------------------------------------------------
HOW FAST CAN find-mech ACTUALLY GO?  (read this before sizing a VM)
------------------------------------------------------------------------------
`find-mech` is NOT embarrassingly parallel, and this dominates every sizing
decision. Its structure is:

    for each of n_calls candidate parameter points:      <- STRICTLY SEQUENTIAL
        evaluate n_sheets initial sheets in parallel     <- only ~10-wide
            each sheet runs base, then ablation          <- sequential pair

`bayesian_optimization.minimize` proposes ONE point at a time and must see its
score before proposing the next, so the outer loop cannot be parallelized as
written. Concurrency is therefore capped at `n_sheets` (10), NOT at the vCPU
count -- a 72-vCPU VM would leave 62 cores idle and finish no sooner than a
16-vCPU one.

Worse, an evaluation only finishes when its SLOWEST sheet does. Measured over
5976 archived fit runs: mean sheet (base+ablation) 38.7 min, but max-of-10
83.1 min -- so ~47% core utilisation, and a straggler factor of ~2.1.

Consequences for a 60-call, 10-sheet, ablation-on fit:

    total CPU      ~ 390 core-hours       <- the real size of the job
    wall-clock     ~ 83 h on ANY VM with >= 10 usable cores
    cost           ~ $16 PAYG / ~$3 Spot of actual compute

So: moving this fit to the cloud AS IS buys ~1x speedup (only the per-core
clock differs). The cloud wins in three other ways, in increasing order of
effort:

  1. FREE     -- run E17.5 and P0 on two VMs at once: both stages done in the
                 time of one, and your workstation stays free.
  2. CHEAP    -- `--batch-k K` evaluates K candidate points concurrently
                 (K*10 sims in flight). Measured projection: K=4 on 40 vCPU
                 -> ~27 h (3.1x); K=6 on 60 vCPU -> ~19 h (4.3x). Requires the
                 batch-capable optimizer; see `estimate --batch-k`.
  3. FREE-ish -- the first `n_initial_points` (25 of 60) are RANDOM and carry
                 no sequential dependency at all, so they can all be launched
                 at once with no loss of sample efficiency.

------------------------------------------------------------------------------
QUICK START -- single Spot VM
------------------------------------------------------------------------------
# 0) one-time: install the Azure CLI and `az login`
RG=tissue-model-rg ; LOC=westeurope ; VM=tissue-sim

az group create -n $RG -l $LOC

# 16 vCPU is enough for a stock (sequential-BO) find-mech with n_sheets=10.
# Only go bigger if you use --batch-k, or run several stages on one VM.
# Deallocate so a preempted Spot VM can be restarted and the run resumed.
az vm create -g $RG -n $VM \
  --image Ubuntu2204 --size Standard_F16s_v2 \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --admin-username azureuser --generate-ssh-keys

# auto-shutdown at 02:00 UTC so a forgotten VM doesn't bill overnight
az vm auto-shutdown -g $RG -n $VM --time 0200

# 1) copy the code up (or `git clone` on the VM). The VENDORED tyssue in
#    ./tyssue/src is REQUIRED -- the stock PyPI tyssue raises
#    "change of datatype in edge table". Make sure that directory comes along.
scp -r .  azureuser@<PUBLIC_IP>:~/tissue_model

# 2) on the VM: install miniconda + the env, then run a method
ssh azureuser@<PUBLIC_IP>
#   (install miniconda, recreate the `tyssue` env from your environment.yml,
#    `conda activate tyssue`), then:
cd ~/tissue_model
export PYTHONPATH=$PWD/tyssue/src            # MANDATORY: use the vendored tyssue
export TISSUE_RESULTS_DIR=$HOME/results      # MANDATORY on Linux: the default
                                             # is the Windows path D:\Kasirer\results
python azure_run.py find-mech --stage E17.5      # defaults mirror run_model.__main__

# 3) pull results back and STOP/DELETE the VM so billing stops
azcopy copy "azureuser@<PUBLIC_IP>:~/results" "." --recursive   # or scp
az vm deallocate -g $RG -n $VM     # stop billing compute (keeps the disk)
az group delete -n $RG --yes       # tear everything down

------------------------------------------------------------------------------
GOTCHAS THAT HAVE ACTUALLY BITTEN THIS PROJECT
------------------------------------------------------------------------------
* PYTHONPATH must point at ./tyssue/src. Running against the installed tyssue
  fails with "change of datatype in edge table".
* TISSUE_RESULTS_DIR must be set on Linux (the default is a Windows D:\ path).
* Do NOT edit run_model.py while a fit is running. The parent process keeps the
  old task-builder while freshly spawned workers import the new unpack, and the
  fit dies with "not enough values to unpack". Copy the tree, edit the copy.
* Re-running a fit REUSES existing run folders whose hash matches, so an
  interrupted Spot run resumes cheaply -- keep the results directory on a disk
  that survives deallocation (or sync to Blob).

------------------------------------------------------------------------------
COST MODEL
------------------------------------------------------------------------------
Cost is dominated by compute = (#simulations) x (hours/sim) x ($/vCPU-hour),
because each simulation pins ~1 vCPU. Storage (HDF5 histories, gifs) on Blob is
~$0.02/GB-month and egress is ~$0.08/GB after the free 100 GB/month -- both
negligible next to compute for typical jobs. Note you pay for EVERY core for
the whole wall-clock, busy or not, so an oversized VM on a 10-wide job is pure
waste; `estimate` prints the utilisation so you can see it.

Use the `estimate` sub-command below to plug in YOUR measured hours-per-sim
(run one sim locally first and time it) and get a dollar figure + wall-clock.
"""

import argparse
import json
import math
import os
import sys


# --------------------------------------------------------------------------- #
# Approximate prices (Linux, PAYG, region-dependent -- VERIFY in the Azure     #
# pricing calculator). Roughly ~$0.042 per vCPU-hour for Fsv2 PAYG.            #
# --------------------------------------------------------------------------- #
VM_REFERENCE = {
    #  size               : (vCPUs, approx PAYG $/hour)
    "Standard_F8s_v2":  (8,  0.338),
    "Standard_F16s_v2": (16, 0.677),
    "Standard_F32s_v2": (32, 1.353),
    "Standard_F48s_v2": (48, 2.030),
    "Standard_F72s_v2": (72, 3.046),
}
DEFAULT_VM = "Standard_F32s_v2"

# Per-stage initial guess, in the order of the DEFAULT fitted parameters:
# (gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index). Kept in sync with
# run_model.__main__ so a cloud run starts from the same point as a local one.
_X0_PRESETS = {
    "E17.5": [0.05, 1.03, 1.20, 1.26],
    "P0":    [0.10, 1.04, 1.28, 1.32],
}


# --------------------------------------------------------------------------- #
# Cost estimation (pure stdlib -- safe to run without tyssue / numpy / azure)  #
# --------------------------------------------------------------------------- #
def estimate_num_simulations(method, n=10, n_sheets=10, n_calls=40, ablation=False,
                             n_grid=11, n_refine=2, n_stages=2):
    """Rough count of ``run()`` simulations a method launches.

    These mirror the loop structure of the corresponding functions in
    ``run_model.py`` (upper bounds; in-process caching / existing-folder reuse
    only makes the real count smaller)."""
    if method == "random-arrays":
        return n
    if method == "init-diff":
        return n_sheets
    if method == "find-mech":
        # n_calls candidate parameter sets, each evaluated over n_sheets, with
        # a second (ablation) simulation per sheet when cells are ablated.
        return n_calls * n_sheets * (2 if ablation else 1)
    if method == "find-psigma":
        # ~n_grid points per line-search pass, (1 + n_refine) passes, times
        # sheets and stages.
        return n_grid * (1 + n_refine) * n_sheets * n_stages
    raise ValueError("unknown method %r" % method)


# Measured over 5976 archived fit runs: one BO evaluation blocks on the SLOWEST
# of its 10 sheets, and max-of-10 / mean = 83.1 min / 38.7 min ~= 2.1. Ignoring
# this underestimates find-mech wall-clock by more than 2x.
DEFAULT_STRAGGLER_FACTOR = 2.1
_STRAGGLER_REF_TASKS = 10        # the factor above was measured at 10 sheets


def _straggler(n_tasks, base=DEFAULT_STRAGGLER_FACTOR):
    """Straggler penalty for blocking on the slowest of ``n_tasks``.

    The penalty GROWS with the batch (max-of-40 > max-of-10), which is exactly
    what erodes the speedup from batching, so a constant factor would overstate
    the benefit of a big VM. Bootstrapping the archived run-time distribution
    gives max/mean ~ 2.15 at 10 tasks and ~2.81 at 40, i.e. roughly
    base + 0.48*ln(n/10), saturating near 3.
    """
    if n_tasks <= 1:
        return 1.0
    return max(1.0, min(3.0, base + 0.48 * math.log(n_tasks / float(_STRAGGLER_REF_TASKS))))


def estimate_wall_clock(method, hours_per_sim, cores, num_sims=None, n_sheets=10,
                        n_calls=40, ablation=False, batch_k=1,
                        straggler_factor=DEFAULT_STRAGGLER_FACTOR):
    """Wall-clock hours, honouring each method's REAL parallel structure.

    ``find-mech`` is the important case: its Bayesian-optimization loop is
    SEQUENTIAL, so concurrency is capped at ``n_sheets * batch_k`` sheet-tasks
    no matter how many vCPUs the VM has. The generators are pure fan-out and do
    scale with ``cores``.
    """
    if method == "find-mech":
        sims_per_task = 2 if ablation else 1        # base, then ablation
        k = max(1, int(batch_k))
        tasks_per_round = n_sheets * k              # dispatched together
        concurrency = max(1, min(cores, tasks_per_round))
        waves = math.ceil(tasks_per_round / concurrency)
        round_h = (waves * sims_per_task * hours_per_sim
                   * _straggler(concurrency, straggler_factor))
        return math.ceil(n_calls / k) * round_h, concurrency
    concurrency = max(1, min(cores, num_sims or 1))
    waves = math.ceil((num_sims or 1) / concurrency)
    return waves * hours_per_sim * _straggler(concurrency, straggler_factor), concurrency


def estimate_cost(num_sims, hours_per_sim, vm=DEFAULT_VM, spot_discount=0.8,
                  vcpu_payg=0.042, method="find-mech", n_sheets=10, n_calls=40,
                  ablation=False, batch_k=1,
                  straggler_factor=DEFAULT_STRAGGLER_FACTOR):
    """Return a dict with core-hours, single-VM wall-clock, cost and utilisation.

    ``spot_discount`` is the fractional saving of Spot vs PAYG (0.8 == 80% off).
    """
    cores, vm_payg_hr = VM_REFERENCE.get(vm, (32, 1.353))
    core_hours = num_sims * hours_per_sim
    wall_clock_h, concurrency = estimate_wall_clock(
        method, hours_per_sim, cores, num_sims=num_sims, n_sheets=n_sheets,
        n_calls=n_calls, ablation=ablation, batch_k=batch_k,
        straggler_factor=straggler_factor)
    # You pay for every core for the whole wall-clock, busy or not.
    payg_cost = wall_clock_h * vm_payg_hr
    spot_cost = payg_cost * (1.0 - spot_discount)
    return {
        "num_sims": num_sims,
        "core_hours": core_hours,
        "idealized_payg": core_hours * vcpu_payg,
        "vm": vm,
        "vm_cores": cores,
        "concurrency": concurrency,
        "utilisation": core_hours / (wall_clock_h * cores) if wall_clock_h else 0.0,
        "wall_clock_h": wall_clock_h,
        "vm_payg_cost": payg_cost,
        "vm_spot_cost": spot_cost,
    }


def _cmd_estimate(args):
    num = estimate_num_simulations(
        args.method, n=args.n, n_sheets=args.n_sheets, n_calls=args.n_calls,
        ablation=args.ablation, n_grid=args.n_grid, n_refine=args.n_refine,
        n_stages=args.n_stages)
    kw = dict(vm=args.vm, spot_discount=args.spot_discount, method=args.method,
              n_sheets=args.n_sheets, n_calls=args.n_calls,
              ablation=args.ablation, batch_k=args.batch_k,
              straggler_factor=args.straggler_factor)
    print("Method: %s" % args.method)
    print("Estimated simulations: %d  (upper bound; folder reuse only shrinks it)" % num)
    hours = args.hours_per_sim
    if hours is None:
        print("\nProvide --hours-per-sim (measure one run locally first) for a $ figure.")
        print("Measured on this project: ~0.4 h/sim (median 0.3, p90 0.9).")
        print("Core-hours and cost scale linearly with it:")
        for h in (0.1, 0.25, 0.4, 1.0, 2.0):
            c = estimate_cost(num, h, **kw)
            print("  %4.2f h/sim -> %7.0f core-h | %s wall-clock %6.1f h | "
                  "PAYG $%7.0f | Spot $%6.0f"
                  % (h, c["core_hours"], c["vm"], c["wall_clock_h"],
                     c["vm_payg_cost"], c["vm_spot_cost"]))
        return
    c = estimate_cost(num, hours, **kw)
    print("Total CPU work: %.0f core-hours" % c["core_hours"])
    print("On one %s (%d vCPU):" % (c["vm"], c["vm_cores"]))
    print("  concurrent sims ~ %d  ->  %.0f%% core utilisation"
          % (c["concurrency"], 100 * c["utilisation"]))
    print("  wall-clock      ~ %.1f h  (%.1f days)"
          % (c["wall_clock_h"], c["wall_clock_h"] / 24.0))
    print("  PAYG cost       ~ $%.0f" % c["vm_payg_cost"])
    print("  Spot cost       ~ $%.0f  (assuming %.0f%% off)"
          % (c["vm_spot_cost"], args.spot_discount * 100))
    if args.method == "find-mech":
        if c["concurrency"] < c["vm_cores"]:
            print("\n  NOTE: only %d of %d cores can ever be busy -- the Bayesian loop is"
                  "\n  sequential, so concurrency is capped at n_sheets*batch_k. A smaller"
                  "\n  VM would finish just as fast and cost less."
                  % (c["concurrency"], c["vm_cores"]))
        print("\n  Batch-BO what-if (K candidate points evaluated concurrently):")
        base_wall = estimate_wall_clock(
            "find-mech", hours, c["vm_cores"], n_sheets=args.n_sheets,
            n_calls=args.n_calls, ablation=args.ablation, batch_k=1,
            straggler_factor=args.straggler_factor)[0]
        for k in (1, 2, 4, 6, 8):
            need = args.n_sheets * k
            w = estimate_wall_clock(
                "find-mech", hours, need, n_sheets=args.n_sheets,
                n_calls=args.n_calls, ablation=args.ablation, batch_k=k,
                straggler_factor=args.straggler_factor)[0]
            print("    K=%d needs %3d vCPU -> %6.1f h  (%.1fx)"
                  % (k, need, w, base_wall / w if w else 0.0))
    print("\n(Storage + egress are typically a few $ on top; verify prices for your region.)")


# --------------------------------------------------------------------------- #
# Method runners (import run_model lazily so `estimate` needs no heavy deps)   #
# --------------------------------------------------------------------------- #
def _ensure_repo_cwd():
    """Run from the directory holding this file so ``results/`` lands beside
    the code, regardless of where the driver was launched."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _cmd_random_arrays(args):
    _ensure_repo_cwd()
    from run_model import create_random_arrays
    names = create_random_arrays(n=args.n, indices=args.indices, n_workers=args.workers)
    print("Created %d arrays:\n%s" % (len(names), "\n".join(map(str, names))))


def _cmd_init_diff(args):
    _ensure_repo_cwd()
    from full_model import initialize_differentiated_arrays
    names = initialize_differentiated_arrays(
        args.gammaSC, args.gammaHC_ratio, args.alphaHC_ratio, args.psigma,
        stage=args.stage, indices=args.indices, n_arrays=args.n_sheets,
        n_workers=args.workers)
    print("Differentiated %d arrays:\n%s" % (len(names), "\n".join(map(str, names))))


def _check_environment():
    """Fail fast on the two mistakes that silently ruin a cloud run."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    vendored = os.path.join(here, "tyssue", "src")
    if os.path.isdir(vendored):
        try:
            import tyssue
            used = os.path.dirname(os.path.dirname(os.path.abspath(tyssue.__file__)))
            if os.path.normcase(used) != os.path.normcase(vendored):
                print("WARNING: using tyssue from %s, NOT the vendored %s.\n"
                      "         Expect 'change of datatype in edge table'. Set:\n"
                      "           export PYTHONPATH=%s"
                      % (used, vendored, vendored), file=sys.stderr)
        except ImportError:
            pass
    from post_processing import RESULTS_DIR
    if not os.path.isdir(RESULTS_DIR):
        print("WARNING: results dir %r does not exist. On Linux set:\n"
              "           export TISSUE_RESULTS_DIR=$HOME/results"
              % RESULTS_DIR, file=sys.stderr)
    else:
        n = len(glob.glob(os.path.join(RESULTS_DIR, "random_periodic_array*")))
        print("results dir: %s (%d initial-array folders)" % (RESULTS_DIR, n))


def _bounds(v):
    """argparse pair -> tuple, or None when the parameter is to be left out."""
    return None if v is None else tuple(v)


def _cmd_find_mech(args):
    _ensure_repo_cwd()
    _check_environment()
    from run_model import find_mechanical_parameters
    initial_sheets = args.initial_sheets
    if initial_sheets is None and args.indices is None:
        # mirror run_model.__main__: random_periodic_array{i}_for_{E17|P0}
        suffix = "E17" if args.stage == "E17.5" else "P0"
        initial_sheets = ["random_periodic_array%d_for_%s" % (i, suffix)
                          for i in range(args.n_sheets)]
    # The per-type shape index is the DEFAULT parameterisation, but supplying a
    # single --shape-index-bounds means the caller is reverting to the historical
    # box; applying the per-type defaults on top would fit six parameters.
    if (args.shape_index_bounds is None and args.hc_shape_index_bounds is None
            and args.sc_shape_index_bounds is None):
        args.hc_shape_index_bounds = (1.05, 1.55)
        args.sc_shape_index_bounds = (1.05, 1.55)
    bending = None if args.no_bending else args.bending
    x0 = args.x0
    if x0 is None and args.stage in _X0_PRESETS:
        # only safe if the ACTIVE parameter set is the default 4-parameter one
        default_box = (args.gammaSC_bounds is not None
                       and args.alphaHC_ratio_bounds is not None
                       and args.hc_shape_index_bounds is not None
                       and args.sc_shape_index_bounds is not None
                       and args.gammaHC_ratio_bounds is None
                       and args.shape_index_bounds is None)
        if default_box:
            x0 = _X0_PRESETS[args.stage]
            print("x0 preset for %s: %s" % (args.stage, x0))
    kwargs = dict(
        initial_sheets=initial_sheets,
        indices=args.indices, n_sheets=args.n_sheets,
        gammaSC_bounds=_bounds(args.gammaSC_bounds),
        gammaHC_ratio_bounds=_bounds(args.gammaHC_ratio_bounds),
        alphaHC_ratio_bounds=_bounds(args.alphaHC_ratio_bounds),
        shape_index_bounds=_bounds(args.shape_index_bounds),
        hc_shape_index_bounds=_bounds(args.hc_shape_index_bounds),
        sc_shape_index_bounds=_bounds(args.sc_shape_index_bounds),
        gammaHC_ratio_fixed=args.gammaHC_ratio_fixed,
        ablated_cells=tuple(args.ablated_cells),
        post_ablation_frame=args.post_ablation_frame,
        n_calls=args.n_calls, n_initial_points=args.n_initial_points,
        n_workers=args.workers, random_state=args.seed,
        x0=x0, type_by=args.type_by,
        use_saved_threshold=args.use_saved_threshold,
        fix_threshold=args.fix_threshold,
        max_wall_seconds=args.max_wall_seconds,
        min_progress_rate=args.min_progress_rate,
        rerun_stalled_runs=args.rerun_stalled_runs,
        base_quasi_static_threshold=args.base_qst,
        ablation_quasi_static_threshold=args.ablation_qst,
        line_tension=args.line_tension, bending=bending)
    print("fitting %s | %d sheets | n_calls=%d (%d random) | bending=%s | "
          "line_tension=%s | qst base/abl=%.3f/%.3f"
          % (args.stage, args.n_sheets, args.n_calls, args.n_initial_points,
             bending, args.line_tension, args.base_qst, args.ablation_qst))
    if args.dry_run:
        print("\n--dry-run: resolved find_mechanical_parameters(%r, ...) kwargs:" % args.stage)
        for k in sorted(kwargs):
            print("    %-34s %r" % (k, kwargs[k]))
        return
    best, result = find_mechanical_parameters(args.stage, **kwargs)
    # The fitted parameters depend on which bounds were supplied, so report by
    # name rather than assuming the historical (gammaSC, gammaHC, alphaHC).
    names = [n for n, b in (("gammaSC", args.gammaSC_bounds),
                            ("gammaHC_ratio", args.gammaHC_ratio_bounds),
                            ("alphaHC_ratio", args.alphaHC_ratio_bounds),
                            ("shape_index", args.shape_index_bounds),
                            ("hc_shape_index", args.hc_shape_index_bounds),
                            ("sc_shape_index", args.sc_shape_index_bounds))
             if b is not None]
    print("Best mechanical params:")
    for n, v in zip(names, tuple(best)):
        print("  %-16s %.4f" % (n, v))
    if getattr(result, "fun", None) is not None:
        print("  objective (sum z^2) %.3f" % result.fun)


def _cmd_find_psigma(args):
    _ensure_repo_cwd()
    from full_model import find_psigma
    mechanical_params = {stage: tuple(vals) for stage, vals in json.loads(args.mechanical_params).items()}
    best, scores = find_psigma(
        mechanical_params, indices=args.indices,
        psigma_bounds=tuple(args.psigma_bounds), n_grid=args.n_grid,
        n_refine=args.n_refine, n_initial_sheets=args.n_sheets,
        n_workers=args.workers)
    print("Best shared psigma: %.2f" % best)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # estimate ------------------------------------------------------------- #
    e = sub.add_parser("estimate", help="estimate #sims, wall-clock and $ cost")
    e.add_argument("method", choices=["random-arrays", "init-diff", "find-mech", "find-psigma"])
    e.add_argument("--hours-per-sim", type=float, default=None,
                   help="measured wall-clock of ONE simulation (run one locally and time it)")
    e.add_argument("--n", type=int, default=10, help="random-arrays: number of arrays")
    e.add_argument("--n-sheets", type=int, default=10, help="sheets per stage")
    e.add_argument("--n-calls", type=int, default=40, help="find-mech: BO evaluations")
    e.add_argument("--ablation", action="store_true", help="find-mech: ablation term enabled")
    e.add_argument("--n-grid", type=int, default=11, help="find-psigma: grid points per pass")
    e.add_argument("--n-refine", type=int, default=2, help="find-psigma: refinement passes")
    e.add_argument("--n-stages", type=int, default=2, help="find-psigma: number of stages")
    e.add_argument("--vm", default=DEFAULT_VM, choices=list(VM_REFERENCE))
    e.add_argument("--spot-discount", type=float, default=0.8, help="fractional Spot saving (0.8=80%%)")
    e.add_argument("--batch-k", dest="batch_k", type=int, default=1,
                   help="find-mech: candidate points evaluated CONCURRENTLY. "
                        "1 (default) = the stock sequential Bayesian loop.")
    e.add_argument("--straggler-factor", dest="straggler_factor", type=float,
                   default=DEFAULT_STRAGGLER_FACTOR,
                   help="max-of-N / mean run time; %.1f measured on 5976 archived runs"
                        % DEFAULT_STRAGGLER_FACTOR)
    e.set_defaults(func=_cmd_estimate)

    # shared: worker count for the runners
    def add_workers(parser):
        parser.add_argument("--workers", type=int, default=None,
                            help="parallel processes (default: all vCPUs)")

    # random-arrays -------------------------------------------------------- #
    ra = sub.add_parser("random-arrays", help="create_random_arrays")
    ra.add_argument("--n", type=int, default=None,
                    help="build arrays range(n) (shorthand for --indices 0..n-1)")
    ra.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit array indices to build, e.g. one per Batch task")
    add_workers(ra)
    ra.set_defaults(func=_cmd_random_arrays)

    # init-diff ------------------------------------------------------------ #
    idf = sub.add_parser("init-diff", help="initialize_differentiated_arrays")
    idf.add_argument("--stage", default="E17.5", choices=["E17.5", "P0"])
    idf.add_argument("--gammaSC", type=float, default=0.01)
    idf.add_argument("--gammaHC-ratio", dest="gammaHC_ratio", type=float, default=10.0)
    idf.add_argument("--alphaHC-ratio", dest="alphaHC_ratio", type=float, default=1.0)
    idf.add_argument("--psigma", type=float, default=0.0)
    idf.add_argument("--indices", type=int, nargs="+", default=None,
                     help="explicit array indices to differentiate (default: range(--n-sheets))")
    idf.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    add_workers(idf)
    idf.set_defaults(func=_cmd_init_diff)

    # find-mech ------------------------------------------------------------ #
    # Defaults MIRROR run_model.__main__ (the current type-dependent shape-index
    # parameterisation: gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index,
    # with gammaHC_ratio fixed at 1.0 and bending replacing line tension), so
    # `find-mech --stage E17.5` reproduces the local run exactly.
    fm = sub.add_parser("find-mech", help="find_mechanical_parameters (Bayesian opt)")
    fm.add_argument("--stage", required=True, choices=["E17.5", "P0"])
    fm.add_argument("--initial-sheets", dest="initial_sheets", nargs="+", default=None,
                    help="explicit initial-sheet names (default: random_periodic_array{i}_for_<E17|P0>)")
    fm.add_argument("--indices", type=int, nargs="+", default=None,
                    help="array indices for the initial sheets (default: range(--n-sheets))")
    fm.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    fm.add_argument("--ablated-cells", dest="ablated_cells", type=int, nargs="*",
                    default=[337, 304, 65, 114])
    fm.add_argument("--post-ablation-frame", dest="post_ablation_frame", type=int, default=-1)
    fm.add_argument("--n-calls", dest="n_calls", type=int, default=60)
    fm.add_argument("--n-initial-points", dest="n_initial_points", type=int, default=25)
    fm.add_argument("--seed", type=int, default=0)

    # --- search box. A bounds pair that is omitted (None) drops that parameter
    #     from the fit; gammaHC_ratio is fixed rather than fitted by default.
    fm.add_argument("--gammaSC-bounds", dest="gammaSC_bounds", type=float, nargs=2,
                    default=(0.02, 0.15))
    fm.add_argument("--alphaHC-ratio-bounds", dest="alphaHC_ratio_bounds", type=float,
                    nargs=2, default=(1.00, 1.15))
    fm.add_argument("--gammaHC-ratio-bounds", dest="gammaHC_ratio_bounds", type=float,
                    nargs=2, default=None,
                    help="omit (default) to FIX gammaHC_ratio instead of fitting it")
    fm.add_argument("--gammaHC-ratio-fixed", dest="gammaHC_ratio_fixed", type=float,
                    default=1.0, help="value used when --gammaHC-ratio-bounds is omitted")
    # These default to the per-type pair (1.05, 1.55) ONLY when
    # --shape-index-bounds is absent; see _cmd_find_mech. Encoding that as an
    # argparse default would silently give six fitted parameters on a revert.
    fm.add_argument("--hc-shape-index-bounds", dest="hc_shape_index_bounds", type=float,
                    nargs=2, default=None, help="default (1.05, 1.55)")
    fm.add_argument("--sc-shape-index-bounds", dest="sc_shape_index_bounds", type=float,
                    nargs=2, default=None, help="default (1.05, 1.55)")
    fm.add_argument("--shape-index-bounds", dest="shape_index_bounds", type=float,
                    nargs=2, default=None,
                    help="single type-INdependent shape index; use INSTEAD of the "
                         "--hc/--sc pair to revert to the historical parameterisation")
    fm.add_argument("--x0", type=float, nargs="+", default=None,
                    help="initial guess, in the order of the ACTIVE bounds above "
                         "(default: the per-stage x0 from run_model.__main__)")

    # --- mechanics / cost knobs
    fm.add_argument("--bending", type=float, default=0.02,
                    help="BoundaryBending kappa (curvature penalty at virtual vertices)")
    fm.add_argument("--no-bending", dest="no_bending", action="store_true",
                    help="disable the bending effector entirely (not the same as 0.0)")
    fm.add_argument("--line-tension", dest="line_tension", type=float, default=None,
                    help="omit (default): no LineTension effector -- bending replaces it")
    fm.add_argument("--base-qst", dest="base_qst", type=float, default=0.03,
                    help="quasi-static threshold for base runs")
    fm.add_argument("--ablation-qst", dest="ablation_qst", type=float, default=0.02,
                    help="quasi-static threshold for ablation runs")

    # --- cell typing / thresholds
    fm.add_argument("--type-by", dest="type_by", default="delta_level",
                    choices=["atoh_level", "delta_level"])
    fm.add_argument("--use-saved-threshold", dest="use_saved_threshold",
                    action="store_true", default=True,
                    help="per-sheet delta threshold from the array folder (default on)")
    fm.add_argument("--no-saved-threshold", dest="use_saved_threshold",
                    action="store_false")
    fm.add_argument("--fix-threshold", dest="fix_threshold", type=float, default=None)

    # --- stall guards (a wedged sim otherwise burns the whole Spot budget)
    fm.add_argument("--max-wall-seconds", dest="max_wall_seconds", type=float,
                    default=10000)
    fm.add_argument("--min-progress-rate", dest="min_progress_rate", type=float,
                    default=1e-4)
    fm.add_argument("--rerun-stalled-runs", dest="rerun_stalled_runs",
                    action="store_true")
    fm.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="resolve and print the call, run nothing -- check the "
                         "config BEFORE paying for a VM")
    add_workers(fm)
    fm.set_defaults(func=_cmd_find_mech)

    # find-psigma ---------------------------------------------------------- #
    fp = sub.add_parser("find-psigma", help="find_psigma (shared E17.5/P0 fit)")
    fp.add_argument("--mechanical-params", dest="mechanical_params", required=True,
                    help='JSON, e.g. \'{"E17.5":[0.01,10,1],"P0":[0.02,8,1.5]}\'')
    fp.add_argument("--psigma-bounds", dest="psigma_bounds", type=float, nargs=2, default=(0.0, 2.0))
    fp.add_argument("--n-grid", dest="n_grid", type=int, default=11)
    fp.add_argument("--n-refine", dest="n_refine", type=int, default=2)
    fp.add_argument("--indices", type=int, nargs="+", default=None,
                    help="array indices for the initial sheets (default: range(--n-sheets))")
    fp.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    add_workers(fp)
    fp.set_defaults(func=_cmd_find_psigma)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


# --------------------------------------------------------------------------- #
# OPTIONAL: Azure Batch fan-out template (scale the parallel GENERATORS across #
# many Spot nodes). Best for `random-arrays` / `init-diff`, which are pure     #
# fan-out. `find-mech` (adaptive Bayesian opt) and `find-psigma` (adaptive     #
# line search) interleave compute with decisions, so they belong on a single  #
# multi-core VM via the runners above.                                         #
#                                                                             #
# Requires: pip install azure-batch azure-storage-blob azure-identity         #
# and a Batch account + linked Storage account. This is a TEMPLATE -- fill in  #
# your account URLs/keys (via env vars) and a node start-task that installs    #
# the conda env. It has not been run against a live account here.             #
# --------------------------------------------------------------------------- #
def submit_batch_fanout(method, indices, stage="E17.5", pool_id="tissue-pool",
                        vm_size="Standard_F16s_v2", spot_nodes=10):
    """Submit one Batch task per array index for a fan-out generator method.

    Parameters
    ----------
    method : {"random-arrays", "init-diff"}
        Which generator to fan out (one task per array index).
    indices : sequence of int
        The array indices to build -- one Batch task each (e.g. ``range(100)``).
    stage : str
        For ``init-diff``: which stage's initial morphology to differentiate.
    spot_nodes : int
        Low-priority (Spot) nodes in the auto-scaling pool.
    """
    indices = list(indices)
    import datetime
    import azure.batch as batch
    import azure.batch.models as batchmodels
    from azure.batch.batch_auth import SharedKeyCredentials

    account = os.environ["AZ_BATCH_ACCOUNT"]
    key = os.environ["AZ_BATCH_KEY"]
    url = os.environ["AZ_BATCH_URL"]
    client = batch.BatchServiceClient(SharedKeyCredentials(account, key), batch_url=url)

    # A pool of Spot ("low priority") compute nodes. The start task should set
    # up the conda `tyssue` env and stage the code (e.g. clone the repo or
    # download an application package); shown here as a placeholder command.
    image = batchmodels.ImageReference(
        publisher="canonical", offer="0001-com-ubuntu-server-jammy",
        sku="22_04-lts", version="latest")
    pool = batchmodels.PoolAddParameter(
        id=pool_id,
        vm_size=vm_size,
        target_low_priority_nodes=spot_nodes,   # Spot nodes (preemptible, cheap)
        target_dedicated_nodes=0,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=image,
            node_agent_sku_id="batch.node.ubuntu 22.04"),
        start_task=batchmodels.StartTask(
            command_line="/bin/bash -c 'echo set-up-conda-env-and-code-here'",
            wait_for_success=True,
            user_identity=batchmodels.UserIdentity(
                auto_user=batchmodels.AutoUserSpecification(
                    elevation_level=batchmodels.ElevationLevel.admin,
                    scope=batchmodels.AutoUserScope.pool))))
    try:
        client.pool.add(pool)
    except batchmodels.BatchErrorException:
        pass  # pool already exists

    job_id = "%s-%s" % (method, datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    client.job.add(batchmodels.JobAddParameter(
        id=job_id, pool_info=batchmodels.PoolInformation(pool_id=pool_id)))

    # One task per array INDEX. Each task runs ONE simulation (--indices i,
    # --workers 1) so Batch -- not a local pool -- provides the parallelism
    # across nodes, and each array maps to a distinct, addressable folder.
    tasks = []
    for i in indices:
        if method == "random-arrays":
            sub = "random-arrays --indices %d --workers 1" % i
        else:  # init-diff
            sub = "init-diff --stage %s --indices %d --workers 1" % (stage, i)
        cmd = ("/bin/bash -c 'cd $AZ_BATCH_TASK_WORKING_DIR && "
               "conda run -n tyssue python azure_run.py %s'" % sub)
        tasks.append(batchmodels.TaskAddParameter(id="task-%d" % i, command_line=cmd))
    client.task.add_collection(job_id, tasks)
    print("Submitted Batch job %s with %d tasks on pool %s (%d Spot nodes)."
          % (job_id, len(indices), pool_id, spot_nodes))
    return job_id


if __name__ == "__main__":
    main()
