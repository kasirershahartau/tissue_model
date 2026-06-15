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
QUICK START -- single large Spot VM (recommended for all four methods)
------------------------------------------------------------------------------
# 0) one-time: install the Azure CLI and `az login`
RG=tissue-model-rg ; LOC=westeurope ; VM=tissue-sim

az group create -n $RG -l $LOC

# A 32-vCPU compute-optimized Spot VM (Deallocate so a preempted VM can be
# restarted and the run resumed). Add a data disk if results get large.
az vm create -g $RG -n $VM \
  --image Ubuntu2204 --size Standard_F32s_v2 \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --admin-username azureuser --generate-ssh-keys

# auto-shutdown at 02:00 UTC so a forgotten VM doesn't bill overnight
az vm auto-shutdown -g $RG -n $VM --time 0200

# 1) copy the code up (or `git clone` on the VM)
scp -r .  azureuser@<PUBLIC_IP>:~/tissue_model

# 2) on the VM: install miniconda + the env, then run a method
ssh azureuser@<PUBLIC_IP>
#   (install miniconda, recreate the `tyssue` env from your environment.yml,
#    `conda activate tyssue`), then e.g.:
cd ~/tissue_model
python azure_run.py find-mech --stage E17.5 --n-calls 40 --n-sheets 10 --ablated-cells 12 13 14 15

# 3) pull results back and STOP/DELETE the VM so billing stops
azcopy copy "azureuser@<PUBLIC_IP>:~/tissue_model/results" "." --recursive   # or scp
az vm deallocate -g $RG -n $VM     # stop billing compute (keeps the disk)
az group delete -n $RG --yes       # tear everything down

------------------------------------------------------------------------------
COST MODEL
------------------------------------------------------------------------------
Cost is dominated by compute = (#simulations) x (hours/sim) x ($/vCPU-hour),
because each simulation pins ~1 vCPU. Storage (HDF5 histories, gifs) on Blob is
~$0.02/GB-month and egress is ~$0.08/GB after the free 100 GB/month -- both
negligible next to compute for typical jobs.

Use the `estimate` sub-command below to plug in YOUR measured seconds-per-sim
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


def estimate_cost(num_sims, hours_per_sim, vm=DEFAULT_VM, spot_discount=0.8,
                  vcpu_payg=0.042):
    """Return a dict with idealized core-hours and single-VM wall-clock/cost.

    ``spot_discount`` is the fractional saving of Spot vs PAYG (0.8 == 80% off).
    """
    cores, vm_payg_hr = VM_REFERENCE.get(vm, (32, 1.353))
    core_hours = num_sims * hours_per_sim
    # On one VM you pay for every core for the whole wall-clock, busy or not.
    wall_clock_h = math.ceil(num_sims / cores) * hours_per_sim
    payg_cost = wall_clock_h * vm_payg_hr
    spot_cost = payg_cost * (1.0 - spot_discount)
    return {
        "num_sims": num_sims,
        "core_hours": core_hours,
        "idealized_payg": core_hours * vcpu_payg,
        "vm": vm,
        "vm_cores": cores,
        "wall_clock_h": wall_clock_h,
        "vm_payg_cost": payg_cost,
        "vm_spot_cost": spot_cost,
    }


def _cmd_estimate(args):
    num = estimate_num_simulations(
        args.method, n=args.n, n_sheets=args.n_sheets, n_calls=args.n_calls,
        ablation=args.ablation, n_grid=args.n_grid, n_refine=args.n_refine,
        n_stages=args.n_stages)
    hours = args.hours_per_sim if args.hours_per_sim is not None else None
    print("Method: %s" % args.method)
    print("Estimated simulations: %d" % num)
    if hours is None:
        print("\nProvide --hours-per-sim (measure one run locally first) for a $ figure.")
        print("Core-hours and cost scale linearly with it:")
        for h in (0.1, 0.25, 0.5, 1.0, 2.0):
            c = estimate_cost(num, h, vm=args.vm, spot_discount=args.spot_discount)
            print("  %4.2f h/sim -> %7.0f core-h | %s wall-clock %5.1f h | "
                  "PAYG $%7.0f | Spot $%6.0f"
                  % (h, c["core_hours"], c["vm"], c["wall_clock_h"],
                     c["vm_payg_cost"], c["vm_spot_cost"]))
        return
    c = estimate_cost(num, hours, vm=args.vm, spot_discount=args.spot_discount)
    print("Core-hours (idealized): %.0f" % c["core_hours"])
    print("On one %s (%d vCPU):" % (c["vm"], c["vm_cores"]))
    print("  wall-clock ~ %.1f h" % c["wall_clock_h"])
    print("  PAYG cost  ~ $%.0f" % c["vm_payg_cost"])
    print("  Spot cost  ~ $%.0f  (assuming %.0f%% off)"
          % (c["vm_spot_cost"], args.spot_discount * 100))
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
    from run_model import initialize_differentiated_arrays
    names = initialize_differentiated_arrays(
        args.gammaSC, args.gammaHC_ratio, args.alphaHC_ratio, args.psigma,
        stage=args.stage, indices=args.indices, n_arrays=args.n_sheets,
        n_workers=args.workers)
    print("Differentiated %d arrays:\n%s" % (len(names), "\n".join(map(str, names))))


def _cmd_find_mech(args):
    _ensure_repo_cwd()
    from run_model import find_mechanical_parameters
    best, result = find_mechanical_parameters(
        args.stage, initial_sheets=args.initial_sheets,
        indices=args.indices, n_sheets=args.n_sheets,
        gammaSC_bounds=tuple(args.gammaSC_bounds),
        gammaHC_ratio_bounds=tuple(args.gammaHC_ratio_bounds),
        alphaHC_ratio_bounds=tuple(args.alphaHC_ratio_bounds),
        ablated_cells=args.ablated_cells, post_ablation_frame=args.post_ablation_frame,
        n_calls=args.n_calls, n_initial_points=args.n_initial_points,
        n_workers=args.workers, random_state=args.seed)
    print("Best mechanical params (gammaSC, gammaHC_ratio, alphaHC_ratio): %s" % (tuple(best),))


def _cmd_find_psigma(args):
    _ensure_repo_cwd()
    from run_model import find_psigma
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
    fm = sub.add_parser("find-mech", help="find_mechanical_parameters (Bayesian opt)")
    fm.add_argument("--stage", required=True, choices=["E17.5", "P0"])
    fm.add_argument("--initial-sheets", dest="initial_sheets", nargs="+", default=None,
                    help="explicit initial-sheet names (default: random_periodic_array{indices}_for_<stage>)")
    fm.add_argument("--indices", type=int, nargs="+", default=None,
                    help="array indices for the initial sheets (default: range(--n-sheets))")
    fm.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    fm.add_argument("--ablated-cells", dest="ablated_cells", type=int, nargs="*", default=[])
    fm.add_argument("--post-ablation-frame", dest="post_ablation_frame", type=int, default=4)
    fm.add_argument("--n-calls", dest="n_calls", type=int, default=40)
    fm.add_argument("--n-initial-points", dest="n_initial_points", type=int, default=10)
    fm.add_argument("--gammaSC-bounds", dest="gammaSC_bounds", type=float, nargs=2, default=(0.001, 0.1))
    fm.add_argument("--gammaHC-ratio-bounds", dest="gammaHC_ratio_bounds", type=float, nargs=2, default=(1.0, 20.0))
    fm.add_argument("--alphaHC-ratio-bounds", dest="alphaHC_ratio_bounds", type=float, nargs=2, default=(0.1, 5.0))
    fm.add_argument("--seed", type=int, default=0)
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
