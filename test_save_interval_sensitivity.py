"""Does a coarser save_interval change the differentiation scores?

    python test_save_interval_sensitivity.py                 # P0 @ psigma=0.060
    python test_save_interval_sensitivity.py --stage E17.5 --psigma 0.045

A longer sweep (t_end 250) is only affordable at a coarser save_interval - at
0.1 it would need ~1.7 TB. Before paying that in score fidelity, measure it.

HOW. The three scoring functions
(_best_matching_frame_by_neighbor_pairs, calc_HC_neighbors_at_differentiation,
calc_percentage_of_differentiating_by_initial_neighbors) all enumerate frames
through post_processing.get_time_points, and none reads history.time_stamps
directly. Subsampling THAT function is therefore an exact stand-in for having
recorded fewer frames: retrieve() is asked only for stamps that exist, so every
returned sheet is bit-identical to the real one.

stride 5 on a 0.1 archive == save_interval 0.5, and so on.

WHAT TO WATCH. score3 needs only t0 and the final frame, so it should be nearly
stride-independent. score1 picks the best-matching frame from a coarser grid -
small shifts expected. score2 is the exposed one: it dates each cell's
differentiation to the frame of its last threshold crossing, so the HC-neighbour
count is read at a coarser moment.
"""
import argparse
import time

import numpy as np

import post_processing as pp
from post_processing import full_model_run_names
from run_model import _psigma_tag


def run_names(stage, psigma, stress_shift, n_arrays):
    if psigma == 0.0:
        return full_model_run_names(stage, n_arrays=n_arrays)
    suffix = "E17" if stage == "E17.5" else "P0"
    return ["fullmodel_ps%s_ks%.3f_random_periodic_array%d_for_%s"
            % (_psigma_tag(psigma), stress_shift, i, suffix) for i in range(n_arrays)]


def score_at_stride(stage, names, stride, type_by, threshold):
    """Score as if only every ``stride``-th frame had been recorded."""
    original = pp.get_time_points

    def subsampled(history):
        stamps = original(history)
        keep = stamps[::stride]
        # The LAST frame must survive: scores 2 and 3 both read the final state,
        # and dropping it would compare different end times across strides.
        if keep[-1] != stamps[-1]:
            keep = np.append(keep, stamps[-1])
        return keep

    pp.get_time_points = subsampled
    try:
        t0 = time.time()
        res = pp.compare_full_model_differentiation_to_experiments(
            stage, model_names=names, type_by=type_by, threshold=threshold)
        return res, time.time() - t0
    finally:
        pp.get_time_points = original


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="P0", choices=["E17.5", "P0"])
    ap.add_argument("--psigma", type=float, default=0.060)
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=-0.080)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=0.355079)
    a = ap.parse_args()

    names = run_names(a.stage, a.psigma, a.stress_shift, a.n_arrays)
    print("save_interval sensitivity | %s | psigma=%.4f | %d runs"
          % (a.stage, a.psigma, len(names)), flush=True)
    print("archives were written at save_interval 0.1, so stride s == "
          "save_interval 0.1*s\n", flush=True)

    rows = []
    for stride in a.strides:
        res, secs = score_at_stride(a.stage, names, stride, a.type_by, a.threshold)
        rows.append((stride, res, secs))
        print("  stride %-3d (save_interval %.2f): total=%.3f  [%.0fs]"
              % (stride, 0.1 * stride, res["total"], secs), flush=True)

    base = rows[0][1]
    print("\n%s" % ("=" * 78))
    print("  %-8s %-10s %10s %10s %10s %11s" %
          ("stride", "save_int", "score1", "score2", "score3", "TOTAL"))
    for stride, res, _s in rows:
        print("  %-8d %-10.2f %10.3f %10.3f %10.3f %11.3f"
              % (stride, 0.1 * stride, res["score1"], res["score2"],
                 res["score3"], res["total"]))
    print("\n  relative change vs stride 1 (the archives' true resolution):")
    print("  %-8s %-10s %10s %10s %10s %11s" %
          ("stride", "save_int", "score1", "score2", "score3", "TOTAL"))
    for stride, res, _s in rows[1:]:
        def rel(k):
            b = base[k]
            return float("nan") if b == 0 else 100.0 * (res[k] - b) / abs(b)
        print("  %-8d %-10.2f %9.1f%% %9.1f%% %9.1f%% %10.1f%%"
              % (stride, 0.1 * stride, rel("score1"), rel("score2"),
                 rel("score3"), rel("total")))
    print("\n  scoring cost: %s"
          % "  ".join("stride %d = %.0fs" % (s, t) for s, _r, t in rows))


if __name__ == "__main__":
    main()
