"""COMBINED-stage psigma sweep at K = -0.080, run to t_end = 100.

    python run_psigma_sweep.py --dry-run     # plan + disk estimate, runs nothing
    python run_psigma_sweep.py               # (or: run_sweep_both.bat)

WHY THIS SWEEP. The per-stage sweeps at K=-0.080 found the SAME optimum for both
stages, which the earlier "the two stages want opposite things" reading missed
because psigma 0.015-0.045 sit on a hump rather than on a trend:

    psigma    E17.5 total   P0 total   combined
    0.000        125.8        467.9      593.7
    0.015        512.0        453.3      965.2
    0.030        823.6        441.8     1265.4
    0.045        715.0        353.9     1068.9
    0.060         82.0         55.7      137.7   <- both stages minimise here
    0.075          ?          650.8         ?
    0.090          ?          840.0         ?

So a SHARED psigma is viable after all, at 4.3x better than baseline.

WHY IT MUST BE RE-RUN. Every one of those numbers was measured at t_end=50, and
101 of 210 runs never reached steady state - monotonically worse with psigma,
and TOTAL at the optimum:

    settled/10   psigma  0     0.015  0.030  0.045  0.060  0.075  0.090
    E17.5                6/10   9/10   5/10   0/10   0/10    -      -
    P0                  10/10   9/10   4/10   2/10   0/10   0/10   0/10

The two arms are therefore not truncated equally: the psigma=0.060 runs are cut
off far earlier relative to their own steady state than the baseline they beat.
Truncation suppresses measured differentiation, and the model's known failure is
OVER-differentiating, so truncation systematically FLATTERS high psigma. Until
the runs converge, the optimum is confounded with where the t=50 wall fell.

Median settle time climbs 20 -> 28 -> 34 (E17.5) and 30 -> 32 -> 42 -> 44 (P0)
with maxima at 49.9/49.7 - clipped by the wall, not intrinsic - so t_end=100
should capture most of the remaining approach.

WHAT ACTUALLY RUNS. run() extends any COMPLETED-but-unsettled run in place from
its last frame (t_end is cumulative on a resume), leaves settled runs untouched,
and simulates missing points from scratch. Nothing is discarded and nothing
settled is recomputed. --dry-run prints the exact plan first.

Scores land in <results>/psigma_scores_E17.5_P0_ks-0.080.json, a different file
from the t_end=50 sweeps, so the old numbers remain available for comparison.
"""
import argparse
import os

import numpy as np

from run_model import _psigma_tag, _classify_existing_run, _reached_steady_state, RESULTS_DIR
from full_model import find_psigma

# best-fit mechanics per stage: (gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index)
MECH = {
    "E17.5": (0.2461, 1.00, 4.86, 5.72),      # 5-term mechanical score 1.076
    "P0":    (0.2298, 1.00, 5.1487, 5.6706),  # 5-term mechanical score 10.723
}
PREFERRED_AREA = 0.593 * np.pi / 4            # 0.4657, set by the ablation data
STRESS_SHIFT = -0.080                         # K
STRESS_HILL_EXPONENT = None                   # None = 3 (the default)

# 0, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090 - the step the existing runs used,
# so every folder tag matches one already on disk (where it exists).
PSIGMA_BOUNDS = (0.0, 0.090)
PSIGMA_N_GRID = 7

T_END = 100
N_WORKERS = int(os.environ.get("TISSUE_FIT_WORKERS", "10"))
GB_PER_UNIT_TIME = 2.6 / 50.0                 # measured: 2.6 GB for a t=50 run


def folder(stage, psigma):
    suffix = "E17" if stage == "E17.5" else "P0"
    if psigma == 0.0:
        return "fullmodel_random_periodic_array%%d_for_%s" % suffix
    return "fullmodel_ps%s_ks%.3f_random_periodic_array%%d_for_%s" % (
        _psigma_tag(psigma), STRESS_SHIFT, suffix)


def plan(grid, n_arrays=10):
    """What each (psigma, stage, array) will do: skip / extend / fresh."""
    rows, gb = [], 0.0
    for psigma in grid:
        for stage in MECH:
            counts = {"settled (skip)": 0, "extend to 100": 0, "fresh run": 0}
            for i in range(n_arrays):
                d = os.path.join(RESULTS_DIR, folder(stage, psigma) % i)
                if not os.path.isdir(d):
                    counts["fresh run"] += 1
                    gb += T_END * GB_PER_UNIT_TIME
                elif _classify_existing_run(d) == "completed" and _reached_steady_state(d):
                    counts["settled (skip)"] += 1
                else:
                    counts["extend to 100"] += 1
                    gb += (T_END - 50) * GB_PER_UNIT_TIME
            rows.append((psigma, stage, counts))
    return rows, gb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the disk estimate, then exit")
    a = ap.parse_args()

    grid = [round(float(p), 5) for p in np.linspace(*PSIGMA_BOUNDS, PSIGMA_N_GRID)]
    print("COMBINED psigma sweep | stages %s | K=%.3f | m=%s | t_end=%d | %d workers"
          % (", ".join(MECH), STRESS_SHIFT,
             STRESS_HILL_EXPONENT if STRESS_HILL_EXPONENT else "3 (default)",
             T_END, N_WORKERS), flush=True)
    print("psigma grid: %s\n" % grid)

    rows, gb = plan(grid)
    print("  %-8s %-6s %s" % ("psigma", "stage", "action"))
    tot = {"settled (skip)": 0, "extend to 100": 0, "fresh run": 0}
    for psigma, stage, counts in rows:
        print("  %-8.3f %-6s %s" % (psigma, stage, "  ".join(
            "%s=%d" % (k, v) for k, v in counts.items() if v)))
        for k, v in counts.items():
            tot[k] += v
    print("\n  TOTAL: %s" % "  ".join("%s=%d" % (k, v) for k, v in tot.items()))
    free_gb = 0.0
    try:
        free_gb = os.statvfs(RESULTS_DIR).f_bavail * os.statvfs(RESULTS_DIR).f_frsize / 2**30
    except (AttributeError, OSError):
        import shutil as _sh
        free_gb = _sh.disk_usage(RESULTS_DIR).free / 2**30
    print("  estimated NEW disk: ~%.0f GB   (free now: %.0f GB)" % (gb, free_gb))
    if gb > free_gb * 0.8:
        print("  *** WARNING: that is more than 80%% of free space ***")
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was run.")

    best, scores = find_psigma(
        MECH,
        psigma_bounds=PSIGMA_BOUNDS, n_grid=PSIGMA_N_GRID, n_refine=0,
        preferred_area=PREFERRED_AREA, stress_shift=STRESS_SHIFT,
        stress_hill_exponent=STRESS_HILL_EXPONENT,
        notch_sensitivity=0.1, repressor_sensitivity=0.3,
        atoh_sensitivity=0.355079, bending=0.02,
        quasi_static_threshold=0.03, initial_notch_delta_level=0.01,
        t_end=T_END, dt=0.01, save_interval=0.1,
        n_arrays=10, n_workers=N_WORKERS, plot=True, save_json=True)
    print("\nbest psigma = %.5g" % best)
