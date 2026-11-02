"""Rank a psigma sweep by a chosen subset of the comparison's sub-scores.

    python rank_psigma_scores.py psigma_scores_E17.5_P0_ks0.000_grouped.json
    python rank_psigma_scores.py scores.json --terms score1 score2 score3

Works from a stored scores json ALONE -- no run folders, no re-scoring. Every
point's score1/score2/score3 are already in the file, so changing which of them
the objective sums is arithmetic, not simulation.

WHY IT MATTERS. ``compare_full_model_differentiation_to_experiments`` returns
score1 (HC:HC and HC:SC neighbour pairs at t0), score2 (HC neighbours at
differentiation) and score3 (% of initial SCs differentiating, by initial
HC-neighbour count). The campaign is fitted on score1+score2 -- four terms per
stage. score3 is reported but not fitted: it is far larger than the other two
(79% of the 7-term total at the optimum), so minimising the sum of all three
selects a different psigma.

Writes <input>_ranked_<terms>.csv and .png next to the json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALL_TERMS = ("score1", "score2", "score3")


def ranked(scores, stages, terms):
    rows = []
    for key, entry in scores.items():
        row = {"psigma": float(key)}
        for s in stages:
            v = entry.get(s)
            if not v:
                continue
            row["%s_obj" % s] = float(sum(v[t] for t in terms))
            for t in ALL_TERMS:
                row["%s_%s" % (s, t)] = float(v[t])
        row["objective"] = float(sum(row.get("%s_obj" % s, 0.0) for s in stages))
        row["all_terms"] = float(sum(entry[s]["total"] for s in stages
                                     if s in entry))
        rows.append(row)
    return sorted(rows, key=lambda r: r["psigma"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path")
    ap.add_argument("--terms", nargs="+", default=["score1", "score2"],
                    choices=list(ALL_TERMS))
    ap.add_argument("--stages", nargs="+", default=None,
                    help="default: whatever the json says")
    a = ap.parse_args()

    d = json.load(open(a.json_path))
    stages = a.stages or d.get("stages") or ["E17.5", "P0"]
    rows = ranked(d["scores"], stages, a.terms)
    if not rows:
        raise SystemExit("no scored points in %s" % a.json_path)

    label = "+".join(t.replace("score", "s") for t in a.terms)
    print("  objective = %s, summed over %s" % (label, ", ".join(stages)))
    head = "  %-9s" % "psigma" + "".join("%12s" % ("%s %s" % (s, label))
                                         for s in stages)
    print(head + "%12s%12s" % ("objective", "all terms"))
    best = min(rows, key=lambda r: r["objective"])
    for r in rows:
        line = "  %-9.5g" % r["psigma"]
        line += "".join("%12.3f" % r.get("%s_obj" % s, float("nan"))
                        for s in stages)
        line += "%12.3f%12.3f" % (r["objective"], r["all_terms"])
        if r is best:
            line += "   <- BEST"
        print(line)

    print("\n  BEST psigma = %.5g   objective (%s) = %.3f   [%s]"
          % (best["psigma"], label, best["objective"],
             ", ".join("%s=%.1f" % (s, best.get("%s_obj" % s, float("nan")))
                       for s in stages)))
    base = next((r for r in rows if r["psigma"] == 0.0), None)
    if base:
        print("  pT = 0 baseline = %.3f  ->  %.1fx better"
              % (base["objective"], base["objective"] / best["objective"]))
    else:
        print("  (no pT = 0 point in this file; score it where the baseline"
              " runs live and compare)")

    stem = os.path.splitext(a.json_path)[0] + "_ranked_" + "".join(
        t[-1] for t in a.terms)
    cols = (["psigma"] + ["%s_obj" % s for s in stages]
            + ["objective", "all_terms"]
            + ["%s_%s" % (s, t) for s in stages for t in ALL_TERMS])
    with open(stem + ".csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("%.6g" % r.get(c, float("nan")) for c in cols) + "\n")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = [r["psigma"] for r in rows]
    for s in stages:
        ax.plot(x, [r.get("%s_obj" % s, np.nan) for r in rows], "--o",
                markersize=4, label="%s  (%s)" % (s, label))
    ax.plot(x, [r["objective"] for r in rows], "k-o", lw=1.8, markersize=5,
            label="objective = %s, both stages" % label)
    ax.axvline(best["psigma"], color="grey", ls=":")
    ax.annotate("best %.5g\n%.1f" % (best["psigma"], best["objective"]),
                (best["psigma"], best["objective"]),
                textcoords="offset points", xytext=(8, 10), fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("pT (mechanosensitivity)")
    ax.set_ylabel("differentiation score (chi^2; lower is better)")
    ax.set_title("pT sweep ranked on %s\n(score3 computed but not fitted)"
                 % label, fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    print("\n  wrote %s.csv / .png / .svg" % stem)


if __name__ == "__main__":
    main()
