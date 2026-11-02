"""Extra REPEATS of the full model at chosen psigma values (simulate only, no scoring).

    python run_psigma_repeats.py --dry-run
    python run_psigma_repeats.py --workers 5

WHY. Each psigma currently has ONE run per initial array — 10 points per stage.
Across the 0.001 grid the scores swung by an order of magnitude between adjacent
steps (P0 score2: 38 -> 28 -> 4.7 -> 25 -> 25 -> 14 -> 12 -> 226), which is
run-to-run scatter, not a psigma dependence. More repeats is the only way to tell
a real window from a fluctuation.

HOW THE REPEATS ARE KEPT APART. The run folder encodes psigma, K and the initial
array, but nothing that distinguishes one realisation from another, so a second
run would collide with the first. Each repeat therefore gets its own NAME PREFIX
(fullmodel_v2 / fullmodel_v2r2 / fullmodel_v2r3). Repeat 1 is what already
exists, so only 2 and 3 are simulated here.

THE REPEATS ARE GENUINELY INDEPENDENT: the initial notch/delta/repressor levels
are drawn from np.random.rand on the UNSEEDED global RNG (there is no
np.random.seed anywhere in the codebase), so each fresh worker process starts
from a different initial condition on the same array. That is precisely the
variable the scatter comes from.

Scoring is deliberately NOT done here — pooling all repeats into one comparison
is the point, and that is score_psigma_pooled.py.
"""
import argparse
import os

from post_processing import RESULTS_DIR
# REPEAT_PREFIX is imported, not redefined: find_psigma runs repeats itself
# now, and two copies of this map would be two ways to name the same run.
from full_model import run_full_model_arrays, REPEAT_PREFIX
import grid_fit_mechanics_v2 as g2
from run_fitted_full_model import (best_point, ATOH_SENSITIVITY, NOTCH_SENSITIVITY,
                                   REPRESSOR_SENSITIVITY, INITIAL_LI_LEVEL,
                                   SHAPE_INDEX, BENDING, LINE_TENSION)

STAGES = ("E17.5", "P0")
PSIGMA = [0.0, 0.160, 0.161, 0.162, 0.163, 0.164, 0.165]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, nargs="+", default=PSIGMA)
    ap.add_argument("--repeats", type=int, nargs="+", default=[2, 3],
                    help="which repeat indices to simulate (1 already exists)")
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0)
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--t-end", dest="t_end", type=float, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--l0-ratio", dest="l0_ratio", type=float, default=1.0,
                    help="L0/<perimeter>: the lateral-inhibition length scale as a "
                         "multiple of the mean initial perimeter. 1.0 (default) is "
                         "the historical L0 == perimeter assumption; see "
                         "lateral_inhibition_parameters.md. A non-default value "
                         "gets its own _l0r<value> folders.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fits = {s: best_point(s) for s in a.stage}
    n = len(a.repeats) * len(a.psigma) * len(a.stage) * a.n_arrays
    print("=" * 78)
    print("psigma REPEATS  |  repeats %s  |  %d psigma  |  %d stage(s)  |  %d workers"
          % (a.repeats, len(a.psigma), len(a.stage), a.workers))
    print("=" * 78)
    for s in a.stage:
        f = fits[s]
        print("  %-6s gammaSC=%.4g R_gamma=%.3f R_alpha=%.3f A0=%.5f"
              % (s, f["gamma_sc"], f["R_gamma"], f["R_alpha"], f["A0"]))
    print("  psigma: %s" % ", ".join("%.3f" % p for p in a.psigma))
    print("  prefixes: %s" % ", ".join(REPEAT_PREFIX[r] for r in a.repeats))
    print("  l0_ratio: %.4g%s" % (a.l0_ratio,
                                  "  (L0 == average perimeter)" if a.l0_ratio == 1.0
                                  else "  -> folders tagged _l0r%.4g" % a.l0_ratio))
    print("\n  up to %d run(s) (completed ones are reused)" % n)
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    for rep in a.repeats:
        prefix = REPEAT_PREFIX[rep]
        for psigma in a.psigma:
            for stage in a.stage:
                f = fits[stage]
                print("\n" + "-" * 78)
                print("  repeat %d | %s | psigma = %.3f" % (rep, stage, psigma), flush=True)
                run_full_model_arrays(
                    stage,
                    gammaSC=f["gamma_sc"], gammaHC_ratio=f["R_gamma"],
                    alphaHC_ratio=f["R_alpha"],
                    hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
                    atoh_sensitivity=ATOH_SENSITIVITY,
                    notch_sensitivity=NOTCH_SENSITIVITY,
                    repressor_sensitivity=REPRESSOR_SENSITIVITY,
                    bending=BENDING, line_tension=LINE_TENSION,
                    quasi_static_threshold=g2.BASE_QST,
                    preferred_area=f["A0"],
                    psigma=psigma, stress_shift=a.stress_shift,
                    initial_notch_delta_level=INITIAL_LI_LEVEL,
                    t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
                    n_arrays=a.n_arrays, n_workers=a.workers,
                    reuse_existing_run=True, name_prefix=prefix,
                    l0_ratio=a.l0_ratio)
    print("\nall repeats done")


if __name__ == "__main__":
    main()
