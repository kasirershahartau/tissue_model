"""Score 1, score 2 and their sum for every model version in the figure.

    python compare_scores_by_source.py
    python compare_scores_by_source.py --terms      # the four chi^2 terms too

The sources are the ones plot_neighbor_pairs.py draws, minus the experiments —
an experiment cannot be scored against itself, the score IS the comparison to it.

WHY RECOMPUTE. Both pipelines already store a score1/score2, but they were
computed by different code against differently-loaded targets. Scoring all six
here from the per-run tables, with one definition and one set of experimental
targets, is what makes the rows comparable to each other. Each score is the sum
of two chi^2 terms, chi^2 = n-sigma^2 with

    n-sigma = (mean_sim - mean_exp) / sqrt(SEM_sim^2 + SEM_exp^2)

    score 1   composition:  HC:HC and HC:SC share of the cell-cell contacts
    score 2   who differentiates: % of differentiating cells with 0, and with 1,
              HC neighbour at the moment they differentiated

taken over per-array means (the model) and per-movie values (the experiment), so
the SEMs describe array-to-array and movie-to-movie variation. --check prints the
value each pipeline stored next to the recomputed one.
"""
import argparse
import os

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR, _nsigma_and_chi2
from build_experimental_tables import read_table
from build_fullmodel_table import exp_targets
from plot_neighbor_pairs import LI_DIR, SOURCES, _plain, _select

# (label, column in the runs tables, index into exp_targets) per score
TERMS = {1: [("HC:HC", "pct_HCHC_contacts_t0", 0),
             ("HC:SC", "pct_HCSC_contacts_t0", 1)],
         2: [("0 HC nb", "pct_events_0_HC_neighbours", 2),
             ("1 HC nb", "pct_events_1_HC_neighbours", 3)]}


def _per_array(g, col):
    """One value per initial array: the mean over that array's repeats."""
    v = g.groupby("initial_array")[col].mean().to_numpy(float)
    return v[np.isfinite(v)]


def score_sources(li_dir=LI_DIR, results_dir=RESULTS_DIR):
    li = pd.read_pickle(os.path.join(li_dir, "runs.pkl"))
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))

    rows = []
    for stage, kind, label, sel in SOURCES:
        if kind == "exp":
            continue
        g = _select(li if kind == "li" else full, stage, sel)
        if not len(g):
            raise SystemExit("no runs for %s / %s" % (stage, sel))
        targets = exp_targets(stage)
        row = dict(stage=stage, model="lateral inhibition only" if kind == "li"
                   else "full model", source=_plain(label), n_runs=len(g),
                   n_arrays=int(g["initial_array"].nunique()))
        for score, terms in TERMS.items():
            total = 0.0
            for name, col, which in terms:
                z, chi2, ms, me = _nsigma_and_chi2(_per_array(g, col),
                                                  targets[which])
                row["%s_nsigma" % name] = z
                row["%s_chi2" % name] = chi2
                row["%s_sim" % name] = ms
                row["%s_exp" % name] = me
                total += 0.0 if not np.isfinite(chi2) else chi2
            row["score%d" % score] = total
        row["score1_plus_score2"] = row["score1"] + row["score2"]
        rows.append(row)
    return pd.DataFrame(rows)


def _stored(table, sel, cols):
    """The score this source's own pipeline saved, for comparison."""
    for c in cols:
        if c in table.columns:
            m = np.ones(len(table), bool)
            for col, want in sel.items():
                m &= np.isclose(table[col].astype(float), want)
            v = table[m][c]
            if len(v):
                yield c, float(v.iloc[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terms", action="store_true",
                    help="also print the four chi^2 terms the scores add up")
    ap.add_argument("--check", action="store_true",
                    help="compare with the score each pipeline stored")
    ap.add_argument("--li-dir", default=LI_DIR)
    a = ap.parse_args()

    df = score_sources(li_dir=a.li_dir)

    print("  %-6s %-34s %6s %8s %9s %11s"
          % ("stage", "source", "runs", "score 1", "score 2", "score 1 + 2"))
    print("  " + "-" * 78)
    for _i, r in df.iterrows():
        print("  %-6s %-34s %6d %8.2f %9.2f %11.2f"
              % (r["stage"], r["source"], r["n_runs"],
                 r["score1"], r["score2"], r["score1_plus_score2"]))

    if a.terms:
        print("\n  the chi^2 terms each score adds up (model vs experiment):")
        print("  %-6s %-34s %-9s %9s %9s %9s"
              % ("stage", "source", "term", "model", "exp", "chi^2"))
        for _i, r in df.iterrows():
            for score in (1, 2):
                for name, _c, _w in TERMS[score]:
                    print("  %-6s %-34s %-9s %9.3f %9.3f %9.2f"
                          % (r["stage"], r["source"], name, r["%s_sim" % name],
                             r["%s_exp" % name], r["%s_chi2" % name]))

    if a.check:
        print("\n  what each pipeline stored for the same point:")
        li_sum = pd.read_pickle(os.path.join(a.li_dir, "morphology_summary.pkl"))
        ps = read_table(os.path.join(RESULTS_DIR, "fullmodel_psigma.pkl"))
        for stage, kind, label, sel in SOURCES:
            if kind == "exp":
                continue
            table = li_sum if kind == "li" else ps
            table = table[table["stage"] == stage]
            got = dict(_stored(table, sel, ("score1", "score2",
                                            "score1+score2", "score1_plus_score2")))
            mine = df[(df.stage == stage) & (df.source == _plain(label))].iloc[0]
            print("   %-6s %-34s stored %s" % (stage, _plain(label),
                  "  ".join("%s=%.2f" % (k, v) for k, v in got.items())))
            print("   %-6s %-34s here   score1=%.2f  score2=%.2f"
                  % ("", "", mine["score1"], mine["score2"]))


if __name__ == "__main__":
    main()
