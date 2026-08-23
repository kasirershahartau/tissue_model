"""Score 1, score 2 and their sum against psigma, at a common 3 repeats.

    python plot_scores_3rep.py
    python plot_scores_3rep.py --zoom 0.160 0.165 --repeats 3

Writes <results>/scores_3rep.png (full range), scores_3rep_zoom.png (0.16-0.165)
and scores_3rep.csv with the plotted values.

EVERY POINT RESTS ON THE SAME NUMBER OF REPEATS. That is the whole point of this
figure: the score is an n-sigma with SEM_sim in its denominator, so a point
measured at ten repeats scores higher than the same model at one, and a curve
mixing them shows structure that is an artefact of sampling. Points that do not
have --repeats repeats are dropped and listed, rather than plotted alongside.

Three panels: score 1 (neighbour pairs at t0), score 2 (HC neighbours at
differentiation), and their sum. Each carries both stages plus the sum over
stages, which is the quantity a joint two-stage fit actually minimises.

Points where a score term is undefined — no differentiation events anywhere,
which happens once the tissue collapses — are marked on the psigma axis rather
than drawn at zero, since the summing convention would otherwise rank a dead
parameter point as a perfect fit.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from post_processing import RESULTS_DIR
from build_fullmodel_table import psigma_table

COLOUR = {"E17.5": "tab:blue", "P0": "tab:red", "sum": "k"}
# (column, title, acceptance line). 8 is the working threshold for a single
# score at a single stage; the sum of two such scores therefore gets 16.
PANELS = (("score1", "Score 1 — neighbour pairs at $t_0$", 8.0),
          ("score2", "Score 2 — HC neighbours at differentiation", 8.0),
          ("total", "Score 1 + score 2", 16.0))
TICK_FULL = 0.01                  # x grid interval on the full-range figure
LABEL_FULL = 0.05                 # ... labelled every 0.05 so they do not collide
TICK_ZOOM = 0.001                 # the fine grid belongs only to the zoom
# The full-range figure carries the 0.01-spaced points only. The 0.161-0.165
# cluster is five points inside one tick and turns the curve into a spike there;
# it is what the zoom exists to show.
FULL_POINTS = (0.0, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18)


def build(repeats):
    runs = pd.read_pickle(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    runs = runs[runs["error"].fillna("") == ""]
    runs = runs[runs["repeat"] <= repeats]
    have = (runs.groupby(["stage", "psigma"])["repeat"].nunique()
            .groupby("psigma").min())
    enough = sorted(have[have >= repeats].index)
    short = sorted(have[have < repeats].index)
    tab = psigma_table(runs[runs["psigma"].isin(enough)])
    tab["total"] = tab["score1"] + tab["score2"]
    tab["undefined"] = (tab["score1_undefined_terms"] + tab["score2_undefined_terms"])
    return tab, short, have


def series(tab, stage, col):
    s = tab[tab["stage"] == stage].sort_values("psigma")
    return s[s["undefined"] == 0], s[s["undefined"] > 0]


def draw(tab, xlim, path, repeats, title_extra=""):
    stages = [s for s in ("E17.5", "P0") if s in set(tab["stage"])]
    piv = tab.pivot_table(index="psigma", columns="stage",
                          values=["score1", "score2", "total", "undefined"])
    both = piv.dropna(subset=[("total", s) for s in stages])

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    for ax, (col, title, cut) in zip(axes, PANELS):
        for stage in stages:
            good, dead = series(tab, stage, col)
            if xlim:
                good = good[good["psigma"].between(*xlim)]
                dead = dead[dead["psigma"].between(*xlim)]
            ax.plot(good["psigma"], good[col], "-o", color=COLOUR[stage], ms=5,
                    lw=1.6, label=stage)
            for k, x in enumerate(dead["psigma"]):
                ax.axvline(x, color=COLOUR[stage], ls=":", lw=1.1, alpha=0.55,
                           label="%s: term undefined" % stage if k == 0 else None)
        if len(both):
            tot = sum(both[(col, s)] for s in stages)
            keep = (sum(both[("undefined", s)] for s in stages) == 0)
            t = tot[keep]
            if xlim:
                t = t[(t.index >= xlim[0]) & (t.index <= xlim[1])]
            ax.plot(t.index, t.values, "--s", color=COLOUR["sum"], ms=4, lw=1.4,
                    alpha=0.8, label="sum over stages")
        ax.axhline(cut, color="0.35", ls="--", lw=1.3,
                   label="$\\chi^2$ = %g" % cut)
        ax.set_yscale("log")
        ax.set_xlabel("$p_\\sigma$")
        ax.set_ylabel("%s  ($\\chi^2$, lower is better)"
                      % ("score 1 + score 2" if col == "total" else col))
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        if xlim:
            ax.set_xlim(*xlim)
        if xlim:
            ax.xaxis.set_major_locator(MultipleLocator(TICK_ZOOM))
        else:
            # 0.01 as the visible interval, but a label every 0.01 collides at
            # this width, so the labels go every 0.05 and 0.01 stays as the grid
            ax.xaxis.set_major_locator(MultipleLocator(LABEL_FULL))
            ax.xaxis.set_minor_locator(MultipleLocator(TICK_FULL))
            ax.grid(alpha=0.18, which="minor", axis="x")
        ax.tick_params(axis="x", labelsize=8 if xlim else 9)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Full-model scores against $p_\\sigma$ — every point at %d repeats "
                 "x 10 arrays per stage%s" % (repeats, title_extra), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--zoom", type=float, nargs=2, default=[0.160, 0.165])
    a = ap.parse_args()

    tab, short, have = build(a.repeats)
    print("psigma at %d repeats: %s" % (a.repeats, sorted(tab["psigma"].unique())))
    if short:
        print("excluded (fewer than %d repeats): %s"
              % (a.repeats, ", ".join("%.3f (%d)" % (p, have[p]) for p in short)))
    tab.to_csv(os.path.join(RESULTS_DIR, "scores_3rep.csv"), index=False)

    coarse = tab[tab["psigma"].round(6).isin([round(p, 6) for p in FULL_POINTS])]
    missing = sorted(set(FULL_POINTS) - set(coarse["psigma"].round(6)))
    if missing:
        print("full-range figure: no 3-repeat data at %s"
              % ", ".join("%.3f" % p for p in missing))
    draw(coarse, None, os.path.join(RESULTS_DIR, "scores_3rep.png"), a.repeats)
    draw(tab, tuple(a.zoom),
         os.path.join(RESULTS_DIR, "scores_3rep_zoom.png"), a.repeats,
         title_extra="   [zoom %.3f – %.3f]" % tuple(a.zoom))

    print("\n  best by score1+score2 summed over stages:")
    piv = tab.pivot_table(index="psigma", columns="stage", values="total")
    if piv.shape[1] == 2:
        tot = piv.sum(axis=1).sort_values()
        for ps in tot.index[:5]:
            print("    psigma %.3f   total %8.3f" % (ps, tot[ps]))


if __name__ == "__main__":
    main()
