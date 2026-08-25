"""TEST 2 - raise the sharp-angle collapse threshold to 0.4 rad (no line tension).

Re-runs the full best-fit E17.5 point (10 base + 10 ablation simulations) with
the in-simulation corner-collapse threshold raised from the default 0.1 rad
(5.7 deg) to 0.4 rad (23 deg) - so the solver straightens sharper spikes every
step as it runs, instead of only near-degenerate ones. Nothing else changes.
Reports how the four n-sigma terms and the objective move vs the original best
fit.

Run:
    PYTHONPATH=./tyssue/src python test_sharp_angle.py
Results go into 'SA04test_a*' folders (separate from the real best-fit runs);
each base folder's finale.png shows the resulting morphology.
"""
from mod_test_common import run_modification_test

SHARP_ANGLE_THRESHOLD = 0.8  # radians (46 deg); sim default is 0.1 (5.7 deg)
N_WORKERS = 10

if __name__ == "__main__":
    run_modification_test("SA08test", sharp_angle_threshold=SHARP_ANGLE_THRESHOLD,
                          line_tension=None, n_workers=N_WORKERS)
