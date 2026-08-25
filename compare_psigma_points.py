"""Rank psigma points at a COMMON repeat count, per stage and summed.

    python compare_psigma_points.py --repeats 5
    python compare_psigma_points.py --repeats 5 --psigma 0.162 0.163 0.164 0.165

Reads fullmodel_runs.pkl and re-aggregates; nothing is recomputed.

WHY --repeats EXISTS AND WHY IT IS NOT OPTIONAL IN PRACTICE. The score is an
n-sigma and SEM_sim sits in its denominator. With one repeat the ten data points
are ten individual runs; with five they are ten array MEANS, whose spread is
smaller, so the SEM shrinks and any systematic offset shows up as a larger z.
P0 at psigma 0.162 went 6.05 -> 13.45 -> 29.11 at 3, 5 and 10 repeats without a
single model value changing. So a table mixing 1-repeat and 10-repeat points
ranks the under-sampled ones first, which is an artefact and not a result.
Restricting every point to the same number of repeats removes it. Points with
fewer repeats than requested are listed separately rather than silently compared.

A point whose score has an undefined term (no differentiation events anywhere,
which happens once the pattern collapses) sums to zero under the scoring
convention and would rank BEST. Those are excluded and reported.
"""
import argparse
import os

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_fullmodel_table import psigma_table

STAGES = ("E17.5", "P0")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5,
                    help="use repeats 1..N of every psigma (default 5)")
    ap.add_argument("--psigma", type=float, nargs="+", default=None,
                    help="restrict to these psigma values")
    ap.add_argument("--drop-collapsed", dest="drop_collapsed", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    runs = pd.read_pickle(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    # incomplete runs are dropped when the table is written, so the column is
    # absent from the saved file
    if "error" in runs.columns:
        runs = runs[runs["error"].fillna("") == ""]
    if a.psigma is not None:
        want = {round(float(p), 6) for p in a.psigma}
        runs = runs[runs["psigma"].round(6).isin(want)]

    have = (runs.groupby(["stage", "psigma"])["repeat"].nunique()
            .groupby("psigma").min())
    enough = sorted(have[have >= a.repeats].index)
    short = sorted(have[have < a.repeats].index)

    sel = runs[(runs["repeat"] <= a.repeats)
               & (runs["psigma"].isin(enough))]
    tab = psigma_table(sel, drop_collapsed=a.drop_collapsed)

    print("=" * 96)
    print("psigma COMPARISON at %d repeat(s) per array%s"
          % (a.repeats, "  (collapsed runs dropped)" if a.drop_collapsed else ""))
    print("=" * 96)
    if short:
        print("  not enough repeats, excluded: %s"
              % ", ".join("%.3f (%d)" % (p, have[p]) for p in short))
    dead = tab[(tab["score1_undefined_terms"] > 0) | (tab["score2_undefined_terms"] > 0)]
    if len(dead):
        print("  undefined score term (no events), excluded: %s"
              % ", ".join("%s %.3f" % (r["stage"], r["psigma"])
                          for _i, r in dead.iterrows()))
    tab = tab[(tab["score1_undefined_terms"] == 0) & (tab["score2_undefined_terms"] == 0)]
    if not len(tab):
        raise SystemExit("nothing left to compare")

    for stage in STAGES:
        g = tab[tab["stage"] == stage].sort_values("score2")
        if not len(g):
            continue
        print("\n  %s — ranked by score 2" % stage)
        print("    %-8s %10s %10s %10s %6s %7s %9s"
              % ("psigma", "score2", "score1", "s1+s2", "runs", "arrays", "collapsed"))
        for _i, r in g.head(a.top).iterrows():
            print("    %-8.3f %10.3f %10.3f %10.3f %6d %7d %9d"
                  % (r["psigma"], r["score2"], r["score1"],
                     r["score1_plus_score2"], r["n_runs"], r["n_arrays"],
                     r["n_collapsed"]))

    piv = tab.pivot_table(index="psigma", columns="stage",
                          values=["score1", "score2", "n_runs", "n_collapsed"])
    both = piv.dropna(subset=[("score2", s) for s in STAGES])
    if not len(both):
        print("\n  no psigma is usable at both stages")
        return
    s2 = both[("score2", "E17.5")] + both[("score2", "P0")]
    s1 = both[("score1", "E17.5")] + both[("score1", "P0")]
    print("\n  SUMMED OVER STAGES, ranked by summed score 2")
    print("    %-8s %9s %9s %9s %9s %9s %9s %8s"
          % ("psigma", "sum s2", "s2 E17.5", "s2 P0", "sum s1", "s1 E17.5",
             "s1 P0", "runs/st"))
    for ps in s2.sort_values().head(a.top).index:
        print("    %-8.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %8d"
              % (ps, s2[ps], both.loc[ps, ("score2", "E17.5")],
                 both.loc[ps, ("score2", "P0")], s1[ps],
                 both.loc[ps, ("score1", "E17.5")], both.loc[ps, ("score1", "P0")],
                 both.loc[ps, ("n_runs", "E17.5")]))
    print("\n  SUMMED OVER STAGES, ranked by score 1 + score 2 (all four terms)")
    tot = s1 + s2
    for ps in tot.sort_values().head(a.top).index:
        print("    psigma %.3f   total %8.3f   (s1 %7.3f  s2 %8.3f)"
              % (ps, tot[ps], s1[ps], s2[ps]))

    out = os.path.join(RESULTS_DIR, "psigma_comparison_%drep.csv" % a.repeats)
    tab.to_csv(out, index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
