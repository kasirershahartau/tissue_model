"""PER-STAGE psigma sweep: find what EACH stage wants, separately.

    python run_psigma_sweep_stage.py E17.5
    python run_psigma_sweep_stage.py P0

WHY SEPARATELY. Both stages at K = -0.080, all points now run:

    psigma    E17.5 total   P0 total
    0.000        125.8        468.0
    0.015        512.0        453.3
    0.030        823.6        441.8
    0.045        715.0        353.9
    0.060          ?           55.7   <- P0's optimum, 8.4x better than baseline
    0.075                     650.8   <- score1 collapses (550): too little
    0.090                     840.0      differentiation to match any frame

The two stages want OPPOSITE things. Score 3 drives all of it, and the reason is
in the experimental data: for support cells with 0 HC neighbours the measured
differentiation fraction is ~20.9% at E17.5 but only ~4.3% at P0, while the model
produces ~46% at BOTH. So P0 needs roughly 10x more blocking than E17.5 — but
the gate cannot tell the stages apart, so a SHARED psigma applies nearly the same
suppression to both: too much for E17.5, not enough for P0.

MEASURED, not assumed (plot_hill_gate.py, at the frames the scores actually use
- t0 ~ 10.1 for E17.5 and ~ 10.8 for P0, NOT the early transient):

  * stage selectivity is ~1.07 at K=-0.080 and ~1.11 at K=-0.060. Raising K buys
    essentially nothing: the stages are the same stress trajectory slightly time-
    shifted, so what looked like separation early on was a difference in TIMING
    sampled during a fast transient. Real selectivity needs K >= -0.030, which
    hard-zeroes 25-32% of the tissue via max(s-K, 0) - that is the K=-0.060
    failure (55% zeroed at t=5, E17.5 score2 6.7 -> 52 even at psigma=0.001).
  * the gate does NOT single out isolated SCs. At the scoring frame SC-0/SC-1/
    SC-2 sit within ~10% of each other and SC-1 is lowest. What it separates
    strongly is HC from SC (gate 0.79 vs ~0.45 at E17.5, psigma=0.060). So the
    score-3 gain is a global HC/SC positive feedback, and the 0-neighbour bucket
    improves most only because it is the most over-differentiated to begin with.

WHAT THIS GRID TESTS. P0 has a strong optimum at psigma=0.060 that was never run
for E17.5. Since the two stages' gate values are close, the question is whether
that optimum transfers or whether E17.5's monotone degradation (125.8 -> 512 ->
824 -> 715) simply continues. The E17.5 grid therefore reaches 0.060, reusing the
0 / 0.015 / 0.030 / 0.045 runs already on disk - 10 new simulations, not 50.

If it does not transfer, a SHARED psigma is dead and psigma must be fitted per
stage (defensible: the mechanical parameters already are).

Outputs (stage-tagged, so the two sweeps do not overwrite each other):
    <results>/psigma_scores_<stage>.json   written after EVERY point
    <results>/psigma_fit_<stage>.png
"""
import sys
import numpy as np

from run_model import find_psigma

MECH = {
    "E17.5": (0.2461, 1.00, 4.86, 5.72),
    "P0":    (0.2298, 1.00, 5.1487, 5.6706),
}
PREFERRED_AREA = 0.593 * np.pi / 4
STRESS_SHIFT = -0.08                       # K, unchanged from the first sweep

# find_psigma builds an EVEN grid from (low, high) with n_grid points.
# Grids chosen to REUSE the points the first sweep already ran (0.015, 0.030,
# 0.045 at K=-0.08), so only the genuinely new values cost simulation time.
GRID = {
    # Reach P0's optimum (0.060) with the SAME step the existing runs used, so
    # 0 / 0.015 / 0.030 / 0.045 are all reused and only 0.060 is simulated.
    "E17.5": ((0.0, 0.060), 5),
    # Complete: every point already on disk, so this re-scores for free and
    # regenerates the per-term breakdown.
    "P0":    ((0.0, 0.090), 7),
}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "E17.5"
    if stage not in MECH:
        raise SystemExit("stage must be one of: %s" % ", ".join(MECH))
    bounds, n_grid = GRID[stage]
    print("PER-STAGE psigma sweep | %s | psigma %s in %d steps | K=%.3f"
          % (stage, bounds, n_grid, STRESS_SHIFT), flush=True)

    best, scores = find_psigma(
        {stage: MECH[stage]},
        psigma_bounds=bounds, n_grid=n_grid, n_refine=0,
        preferred_area=PREFERRED_AREA, stress_shift=STRESS_SHIFT,
        notch_sensitivity=0.1, repressor_sensitivity=0.3,
        atoh_sensitivity=0.355079, bending=0.02,
        quasi_static_threshold=0.03, initial_notch_delta_level=0.01,
        t_end=100, dt=0.01, save_interval=0.1,
        n_arrays=10, plot=True, save_json=True)
    print("\nbest psigma for %s = %.4f" % (stage, best))
