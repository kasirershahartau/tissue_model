"""Mechanical-parameter RE-FIT after the L0-normalization fix.

    python run_refit.py E17.5
    python run_refit.py P0

Runs identically locally and on the Azure VM (only the two env vars differ):

    # local (Git Bash)                    # Azure VM
    PYTHONPATH=./tyssue/src               export PYTHONPATH=$HOME/tissue_model/tyssue/src
    TISSUE_RESULTS_DIR defaults to D:\\..  export TISSUE_RESULTS_DIR=$HOME/results

WHAT CHANGED vs the previous fits
--------------------------------
* Mechanical parameters are no longer L0-normalized. The preferred area used to
  be inflated to pi/4 * L0^2 (~9.4, i.e. ~12x the real ~0.79 cell area), which
  left every cell compressed — the tissue would EXPAND on a cut, opposite to the
  experiment's ~10% shrinkage.
* PREFERRED AREA is now set from the CIRCULAR-ABLATION data: 0.593 * pi/4, the
  value that reproduces the measured 7.66% linear shrinkage of a cut disc while
  ALSO matching both roundness targets (confirmed by simulation: 7.65%,
  HC 0.792 / SC 0.646 vs experimental 0.804 / 0.649).
* Because the preferred area moved, the SHAPE INDEX is on a new scale:
  P0 = shape_index * sqrt(A0). The index is defined against the PREFERRED area,
  not the actual one, so with A0 (0.466) well below the actual cell area (~0.766)
  a LARGER index is needed for the same target perimeter — hence ~4.9-5.7 rather
  than the ~4.0 measured as P/sqrt(A) on real cells. A0 and the shape index must
  move TOGETHER: lowering A0 alone over-rounds the cells (SC roundness 0.80 vs
  0.649), and raising the index restores it while A0 supplies the tension.
* ALL FOUR bounds are WIDENED. gammaSC's best sat high in the old box; alphaHC
  was PINNED AT ITS UPPER BOUND in the P0 cloud fit; the shape-index boxes are
  re-centred on the confirmed 4.86 / 5.72.
* Every run folder carries the preferred area in its name/hash (``_pa0.466``),
  so these fits CANNOT collide with — or reuse — any earlier folders.

Everything else is unchanged from the last fit: 4 fitted parameters, bending
0.02, no line tension, quasi-static thresholds 0.03 (base) / 0.02 (ablation),
delta-level typing with the per-sheet saved threshold, the same ablated cells.
"""
import os
import sys
import numpy as np

from run_model import find_mechanical_parameters

# PREFERRED AREA — set by the CIRCULAR-ABLATION data, not guessed.
# 28 ablations (60 um initial radius) give 7.66% +- 0.61% LINEAR shrinkage, and
# E17.5 / P0 are statistically indistinguishable (7.51% vs 7.81%), so ONE value
# serves both stages. A simulated sweep over A0 (re-equilibrating the tissue at
# each value, then relaxing it affinely) plus a direct 2-array confirmation gives
#     A0 = 0.593*pi/4 = 0.4659  with  hc_p0 = 4.86, sc_p0 = 5.72
#       -> shrinkage 7.65% (exp 7.66%)
#       -> roundness HC 0.792 (exp 0.804), SC 0.646 (exp 0.649)
# i.e. the cut response AND both roundness targets at once. See
# SHRINKAGE_ESTIMATE_METHOD.md.
#
# WHY the shape indices are ~4.9-5.7 and no longer ~4.0: the model's shape index
# is P0/sqrt(A0) — defined against the PREFERRED area, not the actual one. With
# A0 now well below the actual cell area (0.466 vs ~0.766), a LARGER index is
# needed for the same target perimeter. Lowering A0 alone over-rounds the cells
# (SC roundness 0.80 vs 0.649); raising the index restores it while the smaller
# A0 supplies the tension. The two must move together.
PREFERRED_AREA = 0.593 * np.pi / 4        # 0.4659 -> 7.65% shrinkage on cut

# alphaHC_ratio is now FIXED at 1.0, not fitted. The free fit drove it to 0.748
# (HC SOFTER than SC), which contradicts the biology — hair cells are stiffer.
# Fixing it leaves THREE fitted parameters against FIVE observables.
ALPHA_HC_FIXED = 1.0

# Per-stage boxes, re-centred on what the first full re-fit found.
# x0 = [gammaSC, hc_shape_index, sc_shape_index]  (alphaHC no longer fitted).
_CFG = {
    # E17.5 — objective 1.771 on the 4-term objective, nothing near a bound.
    # Seeded at its own best; gammaSC 0.246 already gives ~7.65% shrinkage.
    "E17.5": dict(x0=[0.2461, 4.86, 5.72]),
    # P0 — its 4-term best (gammaSC 0.052) gave 13.88% shrinkage vs 7.66%
    # measured. gammaSC is the lever (it sets how hard the perimeter term
    # RESISTS the area-driven contraction): ~0.161 restores the target at P0's
    # other parameters, so that is the seed.
    "P0": dict(x0=[0.1613, 5.50, 5.74]),
}

# TIGHTENED. The shape indices are well pinned by roundness — both stages landed
# in 4.86-5.50 (HC) and 5.72-5.74 (SC) — so the boxes only need to bracket that
# with margin for the shift caused by fixing alphaHC. gammaSC is now constrained
# from two directions (roundness AND the new shrinkage term), so its old
# (0.005, 0.60) range is far wider than useful; a tighter 3-D box means the
# random initial design samples the live region much more densely, which is what
# actually finds the optimum here (the GP phase contributed nothing in either fit).
GAMMA_SC_BOUNDS = (0.05, 0.35)
HC_SHAPE_INDEX_BOUNDS = (4.60, 5.90)
SC_SHAPE_INDEX_BOUNDS = (5.20, 6.30)

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "E17.5"
    if stage not in _CFG:
        raise SystemExit("stage must be one of: %s" % ", ".join(_CFG))
    cfg = _CFG[stage]
    suffix = "E17" if stage == "E17.5" else "P0"
    initial_sheets = ["random_periodic_array%d_for_%s" % (i, suffix) for i in range(10)]

    print("RE-FIT %s | preferred_area=%.4f (0.593*pi/4) | sqrt(A0)=%.4f | "
          "P0 targets HC=%.3f SC=%.3f"
          % (stage, PREFERRED_AREA, np.sqrt(PREFERRED_AREA),
             cfg["x0"][1] * np.sqrt(PREFERRED_AREA), cfg["x0"][2] * np.sqrt(PREFERRED_AREA)),
          flush=True)

    # Worker cap. Default (None) lets the fit size its own pools: n_sheets (10)
    # per evaluation, and — in the CLOUD bundle only — all cores for the parallel
    # initial design. SET THIS when two fits share one VM: each initial design
    # would otherwise claim every core, so two stages at once oversubscribe both
    # CPU and RAM (~1 GB per simulation) and the kernel OOM-kills a worker, which
    # surfaces as BrokenProcessPool. Half the cores each is the safe split.
    n_workers = os.environ.get("TISSUE_FIT_WORKERS")
    n_workers = int(n_workers) if n_workers else None
    if n_workers:
        print("worker cap: TISSUE_FIT_WORKERS=%d" % n_workers, flush=True)

    find_mechanical_parameters(
        stage, initial_sheets=initial_sheets, n_sheets=10, n_workers=n_workers,
        gammaSC_bounds=GAMMA_SC_BOUNDS,
        gammaHC_ratio_bounds=None,          # -> fixed at 1.0 (HC and SC share contractility)
        alphaHC_ratio_bounds=None,          # -> FIXED at ALPHA_HC_FIXED, not fitted
        alphaHC_ratio_fixed=ALPHA_HC_FIXED,
        hc_shape_index_bounds=HC_SHAPE_INDEX_BOUNDS,
        sc_shape_index_bounds=SC_SHAPE_INDEX_BOUNDS,
        preferred_area=PREFERRED_AREA,
        ablated_cells=(337, 304, 65, 114), post_ablation_frame=-1,
        # Wider box -> more coverage. On the CLOUD bundle the initial design is
        # the parallel phase, so the extra init points are nearly free there.
        # 3 fitted parameters in a tighter box -> fewer calls needed than the
        # 4-parameter search. Init is the parallel phase on the cloud bundle.
        n_calls=50, n_initial_points=30, random_state=0,
        x0=cfg["x0"], use_saved_threshold=True, type_by="delta_level",
        min_progress_rate=1e-4, max_wall_seconds=10000, rerun_stalled_runs=False,
        base_quasi_static_threshold=0.03,
        ablation_quasi_static_threshold=0.02,
        line_tension=None,
        bending=0.02,
    )
