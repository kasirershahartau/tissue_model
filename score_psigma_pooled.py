"""Score the psigma sweep with REPEATS AVERAGED PER ARRAY (10 data points, not 30).

    python score_psigma_pooled.py --dry-run
    python score_psigma_pooled.py --psigma 0 0.160 0.161 0.162 0.163 0.164 0.165

Each initial array contributes ONE data point, the mean of its repeats. That is
deliberate: SEM_sim sits in the n-sigma denominator, so treating 30 runs as 30
independent points would shrink it by ~sqrt(3) and inflate every score by up to
3x in chi^2 — not because the model got worse, but because the uncertainty got
smaller. Averaging within an array removes the lateral-inhibition seed noise
(which is what the repeats re-roll) while keeping the array-to-array spread,
which is the variation that genuinely belongs in the SEM. The result stays on
the same scale as the single-repeat scores, so old and new numbers compare
directly.

Missing repeats are tolerated: a group scores from whatever runs exist, so a
partially finished sweep still reports.
"""
import argparse
import json
import os

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             compare_full_model_differentiation_to_experiments as compare3)
from run_model import _psigma_tag
from run_psigma_repeats import REPEAT_PREFIX

STAGES = ("E17.5", "P0")
PSIGMA = [0.0, 0.160, 0.161, 0.162, 0.163, 0.164, 0.165]


def run_name(stage, psigma, prefix, i, stress_shift=0.0):
    """Folder name as _run_full_model_one builds it (psigma=0 carries no tag)."""
    init = initial_morphology_name(i, stage)
    if float(psigma) == 0.0:
        return "%s_%s" % (prefix, init)
    return "%s_ps%s_ks%.3f_%s" % (prefix, _psigma_tag(psigma), stress_shift, init)


def groups_for(stage, psigma, repeats, n_arrays, stress_shift=0.0):
    """[[repeat runs of array 0], [array 1], ...], keeping only what exists."""
    out, missing = [], 0
    for i in range(n_arrays):
        grp = []
        for r in repeats:
            nm = run_name(stage, psigma, REPEAT_PREFIX[r], i, stress_shift)
            if os.path.isdir(os.path.join(RESULTS_DIR, nm)):
                grp.append(nm)
            else:
                missing += 1
        if grp:
            out.append(grp)
    return out, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, nargs="+", default=PSIGMA)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0)
    ap.add_argument("--out", default=None,
                    help="output JSON. The default name encodes the STAGE and K "
                         "but NOT which repeats were used, so scoring a single "
                         "repeat into it would overwrite the pooled result.")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("POOLED psigma SCORES  |  repeats %s averaged per array" % a.repeats)
    print("=" * 78)
    for stage in a.stage:
        out = (a.out if a.out else
               os.path.join(RESULTS_DIR, "psigma_scores_v2pooled_%s_ks%.3f.json"
                            % (stage, a.stress_shift)))
        scores = {}
        if os.path.isfile(out):
            try:
                scores = json.load(open(out))
            except (OSError, ValueError):
                scores = {}
        for psigma in a.psigma:
            key = "%.5f" % psigma
            grps, missing = groups_for(stage, psigma, a.repeats, a.n_arrays,
                                       a.stress_shift)
            have = sum(len(g) for g in grps)
            state = ("cached" if key in scores and not a.rescore
                     else "TO SCORE" if grps else "no runs")
            print("  %-6s psigma %-7.3f %2d array(s), %2d run(s)%s   %s"
                  % (stage, psigma, len(grps), have,
                     " (%d missing)" % missing if missing else "", state))
            if a.dry_run or state != "TO SCORE":
                continue
            try:
                res = compare3(stage, model_groups=grps)
                res["n_arrays"] = len(grps)
                res["n_runs"] = have
            except Exception as exc:                       # noqa: BLE001
                print("    FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
                res = None
            scores[key] = res
            with open(out, "w") as fh:
                json.dump(scores, fh, indent=1, default=float)
        if a.dry_run:
            continue
        print("\n  %s  (lower is better)" % stage)
        print("    %-8s %9s %9s %9s %9s %9s %8s"
              % ("psigma", "score1", "score2", "score3", "s1+s2", "total", "runs"))
        for k in sorted(scores, key=float):
            v = scores[k]
            if not v:
                print("    %-8.3g   failed" % float(k)); continue
            s = [v.get("score%d" % i, float("nan")) for i in (1, 2, 3)]
            print("    %-8.3g %9.4g %9.4g %9.4g %9.4g %9.4g %8s"
                  % (float(k), s[0], s[1], s[2], s[0] + s[1],
                     v.get("total", sum(s)), v.get("n_runs", "?")))
        print("\n  wrote %s\n" % out)


if __name__ == "__main__":
    main()
