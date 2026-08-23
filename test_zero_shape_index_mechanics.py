"""Can PURE perimeter contractility (p0 = 0) fit the mechanics at the NEW A0?

    python test_zero_shape_index_mechanics.py --dry-run    # show the plan only
    python test_zero_shape_index_mechanics.py              # run + score
    python test_zero_shape_index_mechanics.py --arrays 0 1 2 --workers 3

WHY. The model began with perimeter CONTRACTILITY (shape index p0 = 0, i.e.
Gamma/2 * P^2) and moved to perimeter ELASTICITY (Gamma/2 * (P - P0)^2) because
cells came out too round. But that verdict was reached while A0 was set to
values that inflated the cells or left them roughly unchanged. A0 has since been
re-derived from the circular-ablation shrinkage data and is now well BELOW the
actual cell area (0.466 vs ~0.766), which by itself supplies tension and
elongates cells. So the original objection may no longer hold: p0 = 0 might
reach the right roundness at the new A0. This script tests exactly that.

Note that p0 = 0 does not remove an effector - ContractilityPerimeterElasticity
computes Gamma/2 * (P - P0)^2, so P0 = 0 reduces it to Gamma/2 * P^2, which IS
the original contractility model. Nothing else about the mechanics changes.

WHAT IT RUNS. One (or more) E17.5 initial arrays through the SAME evaluator the
mechanical fit uses, `_evaluate_mechanics_for_sheet`, so the numbers are directly
comparable to the fit's:

  * no differentiation, lateral-inhibition state seeded from the array's stored
    {notch,delta,repressor}_levels.npy (both handled inside the evaluator);
  * per-sheet HC/SC delta threshold from threshold.npy, which also drives the
    simulation's atoh_sensitivity, exactly as in the fit;
  * the un-ablated steady state gives roundness + shrinkage, and a second run
    with the fit's ablated cells gives the two ablation terms.

SCORE. All five terms: hc_roundness, sc_roundness, shrinkage, hc_ablation,
sc_ablation. Each is z = (mean_model - mean_exp) / SEM_exp against the
experimental repeats, and the objective is sum(z^2) - identical to the fit's.

ONE-ARRAY CAVEAT. The fit averages ten arrays before taking that mean; with one
array the model mean is noisier, so a single score is indicative, not decisive.
The SCALE is comparable (the denominator is the experimental SEM either way), so
a result near the best fit's ~1.08 would be interesting and a result in the tens
or hundreds would not be a sampling artefact. Add --arrays to firm it up.
"""
import argparse
import os

import numpy as np

from post_processing import (initial_morphology_name,
                             compare_pooled_model_mechanics_to_experiments)
from run_model import (MECHANICS_TERMS, _MECHANICS_ROUNDNESS_TERMS,
                       _MECHANICS_ABLATION_TERMS, _WORST_CASE_NSIGMA,
                       _evaluate_mechanics_for_sheet, _load_saved_threshold)

# --- the hypothesis under test ---------------------------------------------
GAMMA_SC = 1.0
GAMMA_HC_RATIO = 2.0
ALPHA_HC_RATIO = 2.0
SHAPE_INDEX = 0.0            # p0 = 0 -> pure contractility
HC_SHAPE_INDEX = 0.0
SC_SHAPE_INDEX = 0.0
PREFERRED_AREA = 0.593 * np.pi / 4      # 0.4657 - the A0 from the shrinkage data

# --- held identical to run_refit.py so the score is comparable --------------
STAGE = "E17.5"
ABLATED_CELLS = (337, 304, 65, 114)
POST_ABLATION_FRAME = -1
TYPE_BY = "delta_level"
BASE_QST = 0.03
ABLATION_QST = 0.02
LINE_TENSION = None
BENDING = 0.02
MIN_PROGRESS_RATE = 1e-4
MAX_WALL_SECONDS = 10000
PROGRESS_WINDOW_SECONDS = 30.0
RERUN_STALLED = False

# For reference when reading the output.
BEST_FIT_REFERENCE = ("current best E17.5 fit (p0 != 0): gammaSC=0.2461, "
                      "gammaHC_ratio=1.0, alphaHC=1.0, hc_p0=4.86, sc_p0=5.72, "
                      "5-term score 1.076")


def build_task(initial):
    """The exact argument tuple find_mechanical_parameters builds per sheet."""
    return (GAMMA_SC, GAMMA_HC_RATIO, ALPHA_HC_RATIO, initial, STAGE,
            list(ABLATED_CELLS), POST_ABLATION_FRAME,
            _load_saved_threshold(initial),        # use_saved_threshold=True
            TYPE_BY,
            MAX_WALL_SECONDS, MIN_PROGRESS_RATE, PROGRESS_WINDOW_SECONDS,
            RERUN_STALLED, SHAPE_INDEX,
            BASE_QST, ABLATION_QST, LINE_TENSION,
            HC_SHAPE_INDEX, SC_SHAPE_INDEX, BENDING, PREFERRED_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=int, nargs="+", default=[0],
                    help="E17.5 initial-array indices (default: just array 0)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    initials = [initial_morphology_name(i, STAGE) for i in a.arrays]
    print("=" * 78)
    print("p0 = 0 (pure contractility) at the shrinkage-derived A0")
    print("=" * 78)
    print("  gammaSC        = %.4g" % GAMMA_SC)
    print("  gammaHC_ratio  = %.4g" % GAMMA_HC_RATIO)
    print("  alphaHC_ratio  = %.4g" % ALPHA_HC_RATIO)
    print("  shape indices  = %.4g / %.4g / %.4g  (shape_index / hc / sc)"
          % (SHAPE_INDEX, HC_SHAPE_INDEX, SC_SHAPE_INDEX))
    print("  A0             = %.4f  (0.593*pi/4)" % PREFERRED_AREA)
    print("  target P0      = %.4g  (= p0 * sqrt(A0)) -> perimeter term is "
          "Gamma/2 * P^2" % (HC_SHAPE_INDEX * np.sqrt(PREFERRED_AREA)))
    print("  stage          = %s   arrays = %s" % (STAGE, initials))
    print("  ablated cells  = %s" % (ABLATED_CELLS,))
    for init in initials:
        print("  saved delta threshold for %-32s = %s"
              % (init, _load_saved_threshold(init)))
    print("\n  for reference: %s" % BEST_FIT_REFERENCE)
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    tasks = [build_task(init) for init in initials]
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            details_list = list(ex.map(_evaluate_mechanics_for_sheet, tasks))
    else:
        details_list = [_evaluate_mechanics_for_sheet(t) for t in tasks]

    model_terms = {term: [] for term in MECHANICS_TERMS}
    n_ok = 0
    for init, details in zip(initials, details_list):
        if details is None:
            print("\n  %-40s DEGENERATED (dropped)" % init)
            continue
        n_ok += 1
        for term in MECHANICS_TERMS:
            if details[term] is not None:
                model_terms[term].append(details[term])
    if not n_ok:
        raise SystemExit("every sheet degenerated - no score can be formed.")

    active = list(_MECHANICS_ROUNDNESS_TERMS) + ["shrinkage"] + list(_MECHANICS_ABLATION_TERMS)
    zscores = compare_pooled_model_mechanics_to_experiments(model_terms, STAGE)

    print("\n" + "=" * 78)
    print("5-TERM MECHANICAL SCORE  (%d/%d sheet(s) contributed)" % (n_ok, len(initials)))
    print("=" * 78)
    print("  %-14s %10s %12s %10s" % ("term", "n-sigma", "z^2", "n cells"))
    total = 0.0
    for term in active:
        z = zscores.get(term, float("nan"))
        if not np.isfinite(z):
            z = _WORST_CASE_NSIGMA          # same penalty the fit applies
            note = "  (no usable data -> worst case)"
        else:
            note = ""
        contrib = float(z * z)
        total += contrib
        n_cells = int(sum(len(x) for x in model_terms[term]))
        print("  %-14s %+10.3f %12.4g %10d%s" % (term, z, contrib, n_cells, note))
    print("  %-14s %10s %12.4g" % ("TOTAL", "", total))
    print("\n  best-fit reference: %s" % BEST_FIT_REFERENCE)
    if n_ok == 1:
        print("  NOTE: one array only - the model mean is noisy. Re-run with "
              "--arrays 0 1 2 ... before drawing a conclusion.")


if __name__ == "__main__":
    main()
