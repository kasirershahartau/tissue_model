"""Score the v2 full-model runs on the 3 differentiation scores, matched and crossed.

    python score_v2_full_model.py                       # all 4 combinations
    python score_v2_full_model.py --runs E17.5 --exp P0

The three scores are the SAME ones used for the pre-v2 model
(`compare_full_model_differentiation_to_experiments`):

  1. neighbour-pair composition at the best-matching initial frame
  2. HC-neighbour count AT DIFFERENTIATION, as % of all differentiating cells
  3. % of initial SCs that differentiate, grouped by initial HC-neighbour count

Running every (runs, exp) combination gives step 3 and half of step 4 in one
pass: the diagonal is the matched score, and the off-diagonal scores each stage's
runs against the OTHER stage's experiments. That half of the cross-mechanics test
needs NO new simulations — only the array-swap half does.

Scoring is slow (each case re-reads 10 histories and hunts the best-matching
frame), so results are written after EVERY case rather than at the end.
"""
import argparse
import json
import os

from post_processing import (RESULTS_DIR, full_model_run_names,
                             compare_full_model_differentiation_to_experiments as compare3)

OUT = "v2_differentiation_scores.json"
STAGES = ("E17.5", "P0")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(STAGES), choices=list(STAGES),
                    help="which stage's RUNS to score")
    ap.add_argument("--exp", nargs="+", default=list(STAGES), choices=list(STAGES),
                    help="which stage's EXPERIMENTS to score against")
    ap.add_argument("--run-prefix", dest="run_prefix", default="fullmodel_v2")
    ap.add_argument("--threshold", type=float, default=0.355079)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_path = a.out or os.path.join(RESULTS_DIR, OUT)
    results = {}
    if os.path.isfile(out_path):
        try:
            results = json.load(open(out_path))
        except (OSError, ValueError):
            results = {}

    for run_stage in a.runs:
        names = full_model_run_names(run_stage, run_prefix=a.run_prefix)
        missing = [n for n in names
                   if not os.path.isdir(os.path.join(RESULTS_DIR, n))]
        if missing:
            print("!! %s: %d run folder(s) missing, e.g. %s"
                  % (run_stage, len(missing), missing[0]), flush=True)
            continue
        for exp_stage in a.exp:
            tag = "runs=%s|exp=%s" % (run_stage, exp_stage)
            print("\n" + "#" * 74, flush=True)
            print("### %s   %s" % (tag, "(MATCHED)" if run_stage == exp_stage
                                   else "(CROSSED)"), flush=True)
            print("#" * 74, flush=True)
            try:
                res = compare3(exp_stage, model_names=names, threshold=a.threshold)
            except Exception as exc:                     # noqa: BLE001
                print("FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
                res = None
            results[tag] = res
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=1, default=float)
            print("  -> written to %s" % out_path, flush=True)

    print("\n" + "=" * 74)
    print("SUMMARY — 3 differentiation scores (lower is better)")
    print("=" * 74)
    print("  %-22s %10s %10s %10s %10s %10s"
          % ("case", "score1", "score2", "score3", "s1+s2", "total"))
    for tag, v in sorted(results.items()):
        if not v:
            print("  %-22s   failed" % tag); continue
        s = [v.get("score%d" % i, float("nan")) for i in (1, 2, 3)]
        print("  %-22s %10.4g %10.4g %10.4g %10.4g %10.4g"
              % (tag, s[0], s[1], s[2], s[0] + s[1], v.get("total", sum(s))))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
