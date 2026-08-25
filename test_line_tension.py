"""TEST 1 - add a small line tension.

Re-runs the full best-fit E17.5 point (10 base + 10 ablation simulations) with a
small edge LINE TENSION added (nothing else changed), then reports how the four
n-sigma terms and the objective move relative to the original best fit. A line
tension penalises jagged, high-curvature bonds, so it should smooth the spiky
boundaries; the straightening pre-test predicted this also nudges HC/SC roundness
UP toward experiment.

Run:
    PYTHONPATH=./tyssue/src python test_line_tension.py
Results go into 'LTtest_a*' folders (separate from the real best-fit runs); each
base folder's finale.png shows the resulting morphology.
"""
from mod_test_common import run_modification_test

# Small line tension for every cell-type pair. Tune here if you want to sweep it
# (the fit's SC perimeter contractility is ~0.045, so ~0.01 is "small").
LINE_TENSION = 0.05
N_WORKERS = 10

if __name__ == "__main__":
    run_modification_test("LT2test", sharp_angle_threshold=0.1,
                          line_tension=LINE_TENSION, n_workers=N_WORKERS)
