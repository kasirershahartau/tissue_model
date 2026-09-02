"""Isolated support cells per model version, as violins.

    python plot_isolated_sc_violin.py                 # out of the SCs
    python plot_isolated_sc_violin.py --of cells      # out of every cell
    python plot_isolated_sc_violin.py --points array --tight-y

An SC with no HC touching it is a hole in the pattern: lateral inhibition is
supposed to leave every support cell next to a hair cell. The percentage of such
cells among all SCs at the end of the run is the statistic that replaced score 3;
--of cells counts them against every cell instead.

The model versions are the ones plot_neighbor_pairs.py draws — this figure shares
its sources, colours and styling, so the two sit together — but WITHOUT the
experimental bars: this is a comparison among the models.

WHAT THE SHAPES ARE. The violin is a kernel density of the scattered points, one
per model run by default (--points array for the per-array means). The star is
the mean and the whisker the SEM, both over initial arrays: a run is not an
independent sample, an array is.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR, _nsigma_and_chi2
from build_experimental_tables import read_table
from plot_neighbor_pairs import (LI_DIR, SOURCES, STYLE, STYLE_LABEL, POINT_SIZE,
                                 STAT_COLOUR,
                                 POINT_COLOUR, MEAN_MARKERSIZE, BLOCK_GAP,
                                 _mean_sem, _plain, _select, _draw, _x,
                                 _tight_limits)

# Out of what: the same isolated cells counted against two denominators. As a
# share of the SCs it says how well the pattern served the support cells; as a
# share of every cell it is the fraction of the tissue that is a hole, which
# moves with the HC fraction as well.
DENOMINATOR = {
    "SC": ("pct_SC_no_HC_neighbour_of_SC",
           "% of SCs with no HC neighbour", "isolated_sc_violin"),
    "cells": ("pct_SC_no_HC_neighbour_of_all_cells",
              "% of all cells that are SCs with no HC neighbour",
              "isolated_sc_of_all_cells_violin"),
}
# The model counts its isolated SCs at the END of the run, so the experimental
# target is the last recorded frame (+24h); "first" compares against the frame-1
# segmentation instead, which is what the saved psigma table used.
TARGET_FRAME = {"final": "%s_final", "first": "%s"}
PLOT_SOURCES = SOURCES          # models and the experiment they aim at
PAIR = "isolated SC"            # _draw filters on this; there is only one here


def gather(column, exp_term, li_dir=LI_DIR, results_dir=RESULTS_DIR):
    """One row per source: its per-array (or per-movie) values and its per-run ones.

    The experiment enters as a bar, as in plot_neighbor_pairs — three movies are
    too few for a kernel density — and is read at whichever frame ``exp_term``
    names, so the bar and the n-sigma printed below it are the same number.
    """
    li = pd.read_pickle(os.path.join(li_dir, "runs.pkl"))
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    exp = pd.read_pickle(os.path.join(results_dir, "fullmodel_experiment.pkl"))

    out = []
    for i, (stage, kind, label, sel) in enumerate(PLOT_SOURCES):
        if kind == "exp":
            g = exp[(exp["stage"] == stage) & (exp["term"] == exp_term)]
            if not len(g):
                raise SystemExit("no experimental term %s for %s; rebuild with"
                                 " build_experimental_tables.py"
                                 % (exp_term, stage))
            per_unit = per_run = g["value"].to_numpy(float)
            unit, n_runs = "movies", len(g)
        else:
            g = _select(li if kind == "li" else full, stage, sel)
            if not len(g):
                raise SystemExit("no runs for %s / %s" % (stage, sel))
            per_unit = g.groupby("initial_array")[column].mean().to_numpy(float)
            per_run = g[column].to_numpy(float)
            unit, n_runs = "arrays", len(g)
        m, s = _mean_sem(per_unit)
        face, edge = STYLE[(kind != "exp", stage)]
        out.append(dict(index=i, stage=stage, kind=kind, label=label,
                        pair=PAIR, face=face, edge=edge, mean=m, sem=s,
                        per_unit=per_unit, per_run=per_run,
                        n_units=len(per_unit), unit=unit, n_runs=n_runs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--of", choices=tuple(DENOMINATOR), default="SC",
                    help="denominator: out of the SCs (default) or of all cells")
    ap.add_argument("--target-frame", choices=tuple(TARGET_FRAME), default="final",
                    help="experimental frame to score against (default: the last"
                         " recorded one, matching the model's end of run)")
    ap.add_argument("--points", choices=("run", "array", "none"), default="run",
                    help="what each scattered point is (default: one model run)")
    ap.add_argument("--tight-y", action="store_true",
                    help="crop to the data instead of starting at 0")
    ap.add_argument("--exp-no-bar", dest="exp_no_bar", action="store_true",
                    help="draw the experiment as star + SEM + points only, with no bar behind them")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--li-dir", default=LI_DIR)
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    column, ylabel, default_stem = DENOMINATOR[a.of]
    exp_term = TARGET_FRAME[a.target_frame] % column
    data = gather(column, exp_term, li_dir=a.li_dir)
    # the experiment's own rows are the target every model is scored against
    targets = {d["stage"]: d["per_unit"] for d in data if d["kind"] == "exp"}
    if a.exp_no_bar:
        for d in data:
            if d["kind"] == "exp":
                d["draw_shape"] = False

    print("  %s\n  experimental target: %s\n" % (ylabel, exp_term))
    print("  %-6s %-34s %8s %8s %6s %6s %9s %8s %8s"
          % ("stage", "source", "mean", "SEM", "arrays", "runs",
             "exp mean", "n-sigma", "z^2"))
    for d in data:
        if d["kind"] == "exp":      # the target itself: nothing to score it against
            print("  %-6s %-34s %8.3f %8.3f %6d %6d"
                  % (d["stage"], _plain(d["label"]), d["mean"], d["sem"],
                     d["n_units"], d["n_runs"]))
            continue
        z, chi2, _ms, me = _nsigma_and_chi2(d["per_unit"], targets[d["stage"]])
        print("  %-6s %-34s %8.3f %8.3f %6d %6d %9.3f %8.3f %8.3f"
              % (d["stage"], _plain(d["label"]), d["mean"], d["sem"],
                 d["n_units"], d["n_runs"], me, z, chi2))

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    _draw(ax, data, PAIR, a.points, np.random.default_rng(a.seed),
          mean_markersize=a.mean_markersize)
    ax.set_ylabel(ylabel, fontsize=10.5)
    if a.tight_y:
        ax.set_ylim(*_tight_limits(data, PAIR))
    else:
        ax.set_ylim(0, _tight_limits(data, PAIR)[1])

    ax.set_xticks([_x(i, s[0]) for i, s in enumerate(PLOT_SOURCES)])
    ax.set_xticklabels([s[2] for s in PLOT_SOURCES], fontsize=8.5)
    ax.set_xlim(_x(0, PLOT_SOURCES[0][0]) - 0.75,
                _x(len(PLOT_SOURCES) - 1, PLOT_SOURCES[-1][0]) + 0.75)

    for stage in ("E17.5", "P0"):
        xs = [_x(i, s[0]) for i, s in enumerate(PLOT_SOURCES) if s[0] == stage]
        ax.annotate(stage, xy=(np.mean(xs), 1.03),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=12, fontweight="bold")

    keys = [k for k in STYLE_LABEL if k[0] or not a.exp_no_bar]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in keys]
    names = [STYLE_LABEL[k] for k in keys]
    if a.exp_no_bar:
        handles.append(plt.Line2D([], [], marker="*", linestyle="none",
                                  color=STAT_COLOUR,
                                  markersize=a.mean_markersize))
        names.append("experiment (mean)")
    if a.points != "none":
        for kind, name in (("model", "one model run" if a.points == "run"
                            else "one array"), ("exp", "one movie")):
            handles.append(plt.Line2D(
                [], [], marker=".", linestyle="none", color=POINT_COLOUR,
                alpha=0.8, markersize=np.sqrt(POINT_SIZE[kind]) * 1.8))
            names.append(name)
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Support cells left with no hair-cell neighbour"
                 "  (star = mean over arrays, whisker = SEM)",
                 fontsize=11.5, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.24)

    stem = os.path.splitext(a.out or os.path.join(RESULTS_DIR, default_stem))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
