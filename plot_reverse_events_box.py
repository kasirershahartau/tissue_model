"""Reverse differentiation per run, model by model.

    python plot_reverse_events_box.py
    python plot_reverse_events_box.py --tight-y --out somewhere/fig

How often a hair cell turns back into a support cell in one run, for the same
six model versions the isolated-SC figure draws, in the same colours. There is
no experimental box: nothing in the movies counts reversals.

WHERE THE NUMBERS COME FROM
    full model   fullmodel_runs.pkl, n_reverse_events — a cell that is a
                 non-boundary SC in the final frame and was a HC earlier in the
                 window, counted once. Collapsed runs are excluded: there the
                 whole hair-cell population reverts at once, which is a
                 different phenomenon and swamps the column.
    LI only      <li dir>/new/trajectory_groups.csv, normal_reverse_-
                 differentiating_n, one row per (array, repeat).

A CAVEAT ON THE TWO SOURCES. They were produced by different code from different
trajectories, so the counts are comparable only as far as the two definitions
agree — both mean "cells that ended below threshold having been above it". The
lateral-inhibition table also carries a weird_reverse_differentiating_n for
cells whose trajectory was irregular; only the normal one is drawn. And it has
10 repeats per array against the full model's 3, so its boxes rest on more
runs, though the per-run quantity plotted is the same either way.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import read_table
from plot_neighbor_pairs import (LI_DIR, STYLE, STYLE_LABEL, POINT_SIZE,
                                 POINT_COLOUR, MEAN_MARKERSIZE, STAT_COLOUR,
                                 ERROR_CAPSIZE, ERROR_LINEWIDTH, VIOLIN_W,
                                 _mean_sem, _x, _tight_limits)

LI_CSV = os.path.join(LI_DIR, "new", "trajectory_groups.csv")
# whiskers at fixed percentiles rather than matplotlib's 1.5*IQR fence, so the
# caption can state what they mean without it depending on each source's spread
WHISKER_PERCENTILES = (10, 90)
Y_TICK_STEP = 4
LI_COLUMN = "normal_reverse_differentiating_n"
# (stage, kind, label, selector) left to right; _x adds the block gap to P0
SOURCES = [
    ("E17.5", "li", "lateral inhibition\nonly\npS=0.1, pR=0.3",
     dict(dev_stage="E17", pS=0.1, pR=0.3)),
    ("E17.5", "full", "full model\n$p_T$=0", dict(psigma=0.0)),
    ("E17.5", "full", "full model\n$p_T$=0.162", dict(psigma=0.162)),
    ("P0", "li", "lateral inhibition\nonly\npS=0.15, pR=0.25",
     dict(dev_stage="P0", pS=0.15, pR=0.25)),
    ("P0", "full", "full model\n$p_T$=0", dict(psigma=0.0)),
    ("P0", "full", "full model\n$p_T$=0.162", dict(psigma=0.162)),
]
PAIR = "reverse"                # _draw filters on this; there is only one here


def _draw_box(ax, data, points, rng, mean_markersize=MEAN_MARKERSIZE):
    """A box per source, with its runs over it and the mean and its SEM on top.

    The box is the distribution's own summary — median and quartiles, whiskers
    at WHISKER_PERCENTILES — drawn in the source's colours. The black star and whisker are
    the same statistic every other figure here reports: the mean over ARRAYS and
    its SEM, which is not the median the box marks and can sit well away from it
    when a few runs carry most of the events. Outliers are not drawn as fliers
    because every run is already a point.
    """
    for d in data:
        x = _x(d["index"], d["stage"])
        v = d["per_run"] if points == "run" else d["per_unit"]
        ax.boxplot([v], positions=[x], widths=VIOLIN_W, patch_artist=True,
                   manage_ticks=False, showfliers=False,
                   whis=WHISKER_PERCENTILES,
                   boxprops=dict(facecolor=d["face"], edgecolor=d["edge"],
                                 linewidth=1.6, zorder=2),
                   medianprops=dict(color=d["edge"], linewidth=2.0, zorder=3),
                   whiskerprops=dict(color=d["edge"], linewidth=1.6, zorder=2),
                   capprops=dict(color=d["edge"], linewidth=1.6, zorder=2))
        if len(v):
            jit = rng.uniform(-VIOLIN_W * 0.28, VIOLIN_W * 0.28, size=len(v))
            ax.scatter(x + jit, v, s=POINT_SIZE["model"], marker=".",
                       color=POINT_COLOUR, alpha=0.8, linewidths=0, zorder=4)
        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=5)
        ax.plot(x, d["mean"], marker="*", markersize=mean_markersize,
                color=STAT_COLOUR, linestyle="none", zorder=6)


def gather(li_csv=LI_CSV, results_dir=RESULTS_DIR):
    """One row per model version: its per-array means and its per-run values."""
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    full = full[full["n_reverse_events"].notna()
                & ~full["collapsed"].astype(bool)]
    if not os.path.isfile(li_csv):
        raise SystemExit("no %s" % li_csv)
    li = pd.read_csv(li_csv)

    out = []
    for i, (stage, kind, label, sel) in enumerate(SOURCES):
        if kind == "li":
            m = li["dev_stage"] == sel["dev_stage"]
            for col in ("pS", "pR"):
                m &= np.isclose(li[col].astype(float), sel[col])
            g = li[m]
            per_run = g[LI_COLUMN].to_numpy(float)
            per_unit = g.groupby("array_id")[LI_COLUMN].mean().to_numpy(float)
        else:
            g = full[(full["stage"] == stage)
                     & np.isclose(full["psigma"].astype(float), sel["psigma"])]
            per_run = g["n_reverse_events"].to_numpy(float)
            per_unit = (g.groupby("initial_array")["n_reverse_events"]
                        .mean().to_numpy(float))
        if not len(per_run):
            raise SystemExit("no runs for %s / %s" % (stage, sel))
        m_, s_ = _mean_sem(per_unit)
        face, edge = STYLE[(True, stage)]           # every source here is a model
        out.append(dict(index=i, stage=stage, kind=kind, label=label,
                        pair=PAIR, face=face, edge=edge, mean=m_, sem=s_,
                        per_unit=per_unit, per_run=per_run,
                        n_units=len(per_unit), unit="arrays", n_runs=len(per_run)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", choices=("run", "array", "none"), default="run",
                    help="what each scattered point is (default: one model run)")
    ap.add_argument("--tight-y", action="store_true",
                    help="crop to the data instead of starting at 0")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--li-csv", default=LI_CSV)
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    data = gather(li_csv=a.li_csv)

    print("  reverse differentiation events per run\n")
    print("  %-6s %-34s %8s %8s %8s %6s %6s"
          % ("stage", "source", "mean", "SEM", "median", "arrays", "runs"))
    for d in data:
        v = d["per_run"]
        print("  %-6s %-34s %8.2f %8.2f %8.1f %6d %6d"
              % (d["stage"], d["label"].replace("\n", " ").replace("$p_T$", "pT"),
                 d["mean"], d["sem"], float(np.median(v)),
                 d["n_units"], d["n_runs"]))

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    _draw_box(ax, data, a.points, np.random.default_rng(a.seed),
              mean_markersize=a.mean_markersize)
    ax.set_ylabel("reverse differentiation events per run", fontsize=10.5)
    lo, hi = _tight_limits(data, PAIR)
    ax.set_ylim(lo if a.tight_y else 0, hi)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))

    ax.set_xticks([_x(i, s[0]) for i, s in enumerate(SOURCES)])
    ax.set_xticklabels([s[2] for s in SOURCES], fontsize=8.5)
    ax.set_xlim(_x(0, SOURCES[0][0]) - 0.75,
                _x(len(SOURCES) - 1, SOURCES[-1][0]) + 0.75)

    for stage in ("E17.5", "P0"):
        xs = [_x(i, s[0]) for i, s in enumerate(SOURCES) if s[0] == stage]
        ax.annotate(stage, xy=(np.mean(xs), 1.03),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=12, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[(True, s)][0],
                             edgecolor=STYLE[(True, s)][1], linewidth=1.6)
               for s in ("E17.5", "P0")]
    names = [STYLE_LABEL[(True, s)] for s in ("E17.5", "P0")]
    if a.points != "none":
        handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                                  color=POINT_COLOUR, alpha=0.8,
                                  markersize=np.sqrt(POINT_SIZE["model"]) * 1.8))
        names.append("one model run" if a.points == "run" else "one array")
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=len(names),
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    # the key goes on its own line: as one line it ran off both edges
    fig.suptitle("Hair cells that turned back into support cells\n"
                 "box = quartiles with %d-%d%% whiskers;"
                 "  star = mean over arrays, black whisker = SEM"
                 % WHISKER_PERCENTILES,
                 fontsize=11, y=0.99, va="top", linespacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.845, bottom=0.24)

    stem = os.path.splitext(a.out or os.path.join(RESULTS_DIR,
                                                  "reverse_events_box"))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
