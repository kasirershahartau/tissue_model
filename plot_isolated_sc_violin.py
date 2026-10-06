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
# The chosen gate value is marked with a hatch rather than a colour of its own,
# since colour here already says model-or-experiment and which stage — the same
# convention as plot_neighbor_pair_bars.
HATCH = "/"
HATCH_PSIGMA = 0.162


def gather(column, exp_term, li_dir=LI_DIR, results_dir=RESULTS_DIR,
           sources=PLOT_SOURCES, model_violin=False):
    """One row per source: its per-array (or per-movie) values and its per-run ones.

    EVERYTHING IS A BAR by default, unlike plot_neighbor_pairs where the models
    get violins: the experiment because three movies are too few for a kernel
    density, the models because at E17.5 they all sit within 0.1% of zero and a
    violin there is a sliver that says nothing. ``model_violin`` asks for the
    density anyway; _draw still falls back to a bar for a source whose values
    are all identical, where a kernel density does not exist. The experiment is
    read at whichever frame ``exp_term`` names, so the bar and the n-sigma
    printed below it are the same number.
    """
    li = pd.read_pickle(os.path.join(li_dir, "runs.pkl"))
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    exp = pd.read_pickle(os.path.join(results_dir, "fullmodel_experiment.pkl"))

    out = []
    for i, (stage, kind, label, sel) in enumerate(sources):
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
        hatched = (kind != "exp"
                   and np.isclose(sel.get("psigma", np.nan), HATCH_PSIGMA))
        out.append(dict(index=i, stage=stage, kind=kind, label=label,
                        hatch=HATCH if hatched else None,
                        pair=PAIR, face=face, edge=edge, mean=m, sem=s,
                        per_unit=per_unit, per_run=per_run,
                        # bars throughout unless asked otherwise: at E17.5 every
                        # model sits within 0.1% of zero, where a kernel density
                        # is a sliver carrying no information, and a bar beside
                        # the experiment makes the size of the miss legible
                        as_violin=bool(model_violin) and kind != "exp",
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
    ap.add_argument("--scored", action="store_true",
                    help="print each model's chi^2 against its stage's"
                         " experiment over its bar")
    ap.add_argument("--model-violin", dest="model_violin", action="store_true",
                    help="draw the models as violins instead of bars; a source"
                         " with no spread still falls back to a bar")
    ap.add_argument("--drop-li", dest="drop_li", action="store_true",
                    help="leave out the lateral-inhibition-only rows, keeping"
                         " the full model against the experiment")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--li-dir", default=LI_DIR)
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    column, ylabel, default_stem = DENOMINATOR[a.of]
    exp_term = TARGET_FRAME[a.target_frame] % column
    # dropping a source re-indexes the rest, so the axis closes the gap rather
    # than leaving an empty slot where lateral inhibition used to be
    sources = [s for s in PLOT_SOURCES if not (a.drop_li and s[1] == "li")]
    data = gather(column, exp_term, li_dir=a.li_dir, sources=sources,
                  model_violin=a.model_violin)
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
        d["z2"] = chi2          # the figure prints the number this line reports
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

    if a.scored:
        # the number goes above everything that source drew, so it never lands
        # on a point or a whisker; the extra headroom is for the topmost one
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
        span = ax.get_ylim()[1] - lo
        for d in data:
            if d["kind"] == "exp" or not np.isfinite(d.get("z2", np.nan)):
                continue
            v = d["per_run"] if a.points == "run" else d["per_unit"]
            over = [d["mean"] + (d["sem"] if np.isfinite(d["sem"]) else 0.0)]
            if len(v):
                over.append(float(np.max(v)))
            ax.annotate("%.3g" % d["z2"],
                        xy=(_x(d["index"], d["stage"]),
                            max(over) + 0.015 * span),
                        ha="center", va="bottom", fontsize=9,
                        color=STAT_COLOUR, zorder=6)

    ax.set_xticks([_x(i, s[0]) for i, s in enumerate(sources)])
    ax.set_xticklabels([s[2] for s in sources], fontsize=8.5)
    ax.set_xlim(_x(0, sources[0][0]) - 0.75,
                _x(len(sources) - 1, sources[-1][0]) + 0.75)

    for stage in ("E17.5", "P0"):
        xs = [_x(i, s[0]) for i, s in enumerate(sources) if s[0] == stage]
        ax.annotate(stage, xy=(np.mean(xs), 1.03),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=12, fontweight="bold")

    keys = [k for k in STYLE_LABEL if k[0] or not a.exp_no_bar]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in keys]
    names = [STYLE_LABEL[k] for k in keys]
    if any(d.get("hatch") for d in data):
        # the swatch is small, so one stroke can fall outside it entirely:
        # repeating the pattern guarantees the key is not drawn blank
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="0.25", linewidth=1.6,
                                     hatch=HATCH * 3))
        names.append(r"full model, $p_T$ = %g" % HATCH_PSIGMA)
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

    # the variants get their own names: without this both --exp-no-bar and
    # --scored would overwrite the plain figure's files
    if a.out:
        stem = os.path.splitext(a.out)[0]
    else:
        stem = os.path.join(RESULTS_DIR, default_stem
                            + ("_fullonly" if a.drop_li else "")
                            + ("_nobar" if a.exp_no_bar else "")
                            + ("_scored" if a.scored else ""))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
