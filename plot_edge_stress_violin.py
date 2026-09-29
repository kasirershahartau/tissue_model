"""Interface stress by the cell types it separates, HC:SC against SC:SC.

    python plot_edge_stress_violin.py
    python plot_edge_stress_violin.py --layout single
    python plot_edge_stress_violin.py --frame initial

TWO LAYOUTS, written to different names so both can exist.

``panels`` (the default, edge_stress_<frame>.png): one panel per pT value,
joined on a shared border and sharing a y axis so the two are directly
comparable, and within each panel HC:SC beside SC:SC at both stages. The panels
carry no titles — the pT labels are added by hand — so the order is the one in
PANELS, top first.

``single`` (edge_stress_<frame>_single.png): all eight distributions in one
axes, in SINGLE_ORDER — HC:SC then SC:SC for each pT, the two pT blocks of a
stage together, E17.5 before P0. Here the pT IS labelled, under each block,
because nothing else distinguishes two blocks of the same colour;
--no-pt-labels drops it if you would rather add it by hand.

WHAT A VIOLIN IS. The distribution over NEIGHBOUR PAIRS, pooled across runs,
from build_edge_stress_table: for each pair, sum(edge_stress * length) over the
mesh edges the two cells share, divided by L0. The black star and whisker are
the mean over ARRAYS and its SEM — the hierarchy every score here uses — so they
are the numbers to quote and the violin is the shape behind them.

Only the contractility effector set is drawn, the one run_model.stress_effectors
gates on. Only the model appears: the experiment has no stress measurement.

HC:HC IS NOT DRAWN. It exists (269 pairs at E17.5, 470 at P0 for pT = 0.162) but
was left out at request; the tables still carry it.

WHY --frame initial DROPS THE HC:SC VIOLINS. At t = 0 no cell is above the delta
threshold, so every pair is SC:SC and only those positions are filled. The
default t0 is the fitted frame, where all classes exist.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from post_processing import RESULTS_DIR
from build_experimental_tables import read_table
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_SIZE, POINT_COLOUR,
                                 MEAN_MARKERSIZE, STAT_COLOUR, ERROR_CAPSIZE,
                                 ERROR_LINEWIDTH, VIOLIN_W, BLOCK_GAP,
                                 _mean_sem, _x)

PANELS = (0.0, 0.162)          # one panel per pT value
# (stage, class) left to right within a panel; the P0 block is pushed right by _x
SOURCES = [("E17.5", "HC:SC"), ("E17.5", "SC:SC"),
           ("P0", "HC:SC"), ("P0", "SC:SC")]
EFFECTORS = "contractility"
# --layout single puts all eight distributions in one row, in this block order;
# within each block HC:SC comes first, then SC:SC
SINGLE_ORDER = [(0.0, "E17.5"), (0.162, "E17.5"), (0.0, "P0"), (0.162, "P0")]
MAX_POINTS = 400        # per violin; the full set is tens of thousands
# smaller than the shared model size: these clouds are dense enough that the
# standard dot fills the violin and hides its shape
POINT_S = POINT_SIZE["model"] * 0.5
Y_TICK_STEP = 0.1


def gather(frame, results_dir=RESULTS_DIR):
    """{pT: [one row per (stage, class)]} with pair values and per-array means."""
    detail = pd.read_pickle(os.path.join(results_dir, "fullmodel_edge_stress.pkl"))
    runs = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    if "effectors" in detail.columns:
        detail = detail[detail["effectors"] == EFFECTORS]
    detail = detail[detail["frame"] == frame]
    if not len(detail):
        raise SystemExit("no rows for frame=%s" % frame)
    detail = detail.merge(runs[["model_name", "stage", "psigma", "initial_array"]],
                          on="model_name", how="left")

    out = {}
    for pt in PANELS:
        rows = []
        for i, (stage, kl) in enumerate(SOURCES):
            g = detail[(detail["stage"] == stage)
                       & np.isclose(detail["psigma"].astype(float), pt)
                       & (detail["edge_class"] == kl)]
            if not len(g):
                continue
            per_pair = g["stress"].to_numpy(float)
            per_unit = g.groupby("initial_array")["stress"].mean().to_numpy(float)
            m, s = _mean_sem(per_unit)
            face, edge = STYLE[(True, stage)]
            rows.append(dict(index=i, stage=stage, pT=pt, edge_class=kl,
                             face=face, edge=edge, mean=m, sem=s,
                             per_unit=per_unit, per_run=per_pair,
                             n_pairs=len(per_pair), n_arrays=len(per_unit)))
        out[pt] = rows
    return out


def single_row_layout(data):
    """Every distribution in one row: HC:SC then SC:SC, per pT, per stage.

    Order is the one SINGLE_ORDER gives. The two classes of a (stage, pT) block
    touch; blocks are separated by BLOCK_GAP and the two stages by twice that
    again, so the eye groups the pairs the way the labels do.
    """
    rows = []
    for b, (pt, stage) in enumerate(SINGLE_ORDER):
        for j, kl in enumerate(("HC:SC", "SC:SC")):
            found = [d for d in data[pt]
                     if d["stage"] == stage and d["edge_class"] == kl]
            if not found:
                continue
            d = dict(found[0])
            d["x"] = (2 * b + j + BLOCK_GAP * b
                      + (2 * BLOCK_GAP if stage == "P0" else 0.0))
            d["block"] = b
            rows.append(d)
    return rows


def _draw(ax, rows, rng, mean_markersize, max_points, point_size=POINT_S):
    for d in rows:
        x = d.get("x", _x(d["index"], d["stage"]))
        v = d["per_run"]
        if len(v) > 1 and np.ptp(v) > 0:
            body = ax.violinplot([v], positions=[x], widths=VIOLIN_W,
                                 showmeans=False, showextrema=False,
                                 showmedians=False)["bodies"][0]
            body.set_facecolor(d["face"]); body.set_edgecolor(d["edge"])
            body.set_linewidth(1.6); body.set_alpha(1.0); body.set_zorder(2)
        # a violin can hold tens of thousands of pairs; a sample keeps the cloud
        # readable and the SVG small
        show = v if len(v) <= max_points else rng.choice(v, max_points,
                                                         replace=False)
        jit = rng.uniform(-VIOLIN_W * 0.28, VIOLIN_W * 0.28, size=len(show))
        ax.scatter(x + jit, show, s=point_size, marker=".",
                   color=POINT_COLOUR, alpha=0.5, linewidths=0, zorder=3)
        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=4)
        ax.plot(x, d["mean"], marker="*", markersize=mean_markersize,
                color=STAT_COLOUR, linestyle="none", zorder=5)


def _finish(ax, ticks, labels, step):
    """Shared axis dressing: y ticks, x ticks, room at the ends."""
    ax.set_ylabel("interface stress", fontsize=10.5)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(min(ticks) - 0.75, max(ticks) + 0.75)


def draw_panels(data, a):
    """One panel per pT, stacked and joined on a shared border."""
    # sharey: both panels hold both classes, so one scale makes the pT values
    # directly comparable
    # hspace=0: the panels share one boundary line rather than floating apart
    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, sharey=True,
                             figsize=(9.0, 7.6), gridspec_kw=dict(hspace=0))
    for ax, pt in zip(axes, PANELS):
        _draw(ax, data[pt], np.random.default_rng(a.seed), a.mean_markersize,
              a.max_points, a.point_size)
        ax.set_ylabel("interface stress", fontsize=10.5)
        ax.yaxis.set_major_locator(MultipleLocator(a.y_tick_step))
    # no per-panel title: the pT labels are added by hand afterwards. Every
    # other spine stays, so the pair is a closed box; the one that goes is the
    # LOWER panel's top, because the shared border is already drawn by the
    # upper panel's bottom spine and two coincident lines render heavier. The
    # upper panel also loses the x ticks that would poke through that border.
    axes[0].tick_params(axis="x", bottom=False)
    axes[-1].spines["top"].set_visible(False)

    ticks = [_x(i, s[0]) for i, s in enumerate(SOURCES)]
    _finish(axes[-1], ticks, [kl for _s, kl in SOURCES], a.y_tick_step)
    for stage in ("E17.5", "P0"):
        xs = [t for t, s in zip(ticks, SOURCES) if s[0] == stage]
        axes[0].annotate(stage, xy=(np.mean(xs), 1.04),
                         xycoords=("data", "axes fraction"), ha="center",
                         va="bottom", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.115, right=0.98, top=0.875, bottom=0.135)
    return fig, axes[0]


def draw_single(data, a):
    """All eight distributions in one axes, grouped by stage then by pT."""
    rows = single_row_layout(data)
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    _draw(ax, rows, np.random.default_rng(a.seed), a.mean_markersize,
          a.max_points, a.point_size)
    _finish(ax, [d["x"] for d in rows], [d["edge_class"] for d in rows],
            a.y_tick_step)

    # two label rows under the axis: the pT of each block, then the stage
    # spanning its two blocks. Both are pushed below the class labels so the
    # grouping reads without a legend.
    for b, (pt, stage) in enumerate(SINGLE_ORDER):
        xs = [d["x"] for d in rows if d["block"] == b]
        if not xs:
            continue
        if a.pt_labels:
            ax.annotate(r"$p_T$ = %g" % pt, xy=(np.mean(xs), -0.075),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="top", fontsize=11)
    for stage in ("E17.5", "P0"):
        xs = [d["x"] for d in rows if d["stage"] == stage]
        if xs:
            ax.annotate(stage, xy=(np.mean(xs), 1.02),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="bottom", fontsize=12, fontweight="bold")
    # top leaves the stage labels clear of the two-line suptitle; bottom leaves
    # a band for the class ticks, the pT row under them and the legend below
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.245)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", choices=("t0", "initial"), default="t0")
    ap.add_argument("--layout", choices=("panels", "single"), default="panels",
                    help="panels (default): one panel per pT, stacked. "
                         "single: all eight distributions in one axes")
    ap.add_argument("--no-pt-labels", dest="pt_labels", action="store_false",
                    help="single layout: drop the pT labels under the blocks")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--point-size", dest="point_size", type=float,
                    default=POINT_S, help="scatter marker area (default %g)"
                                          % POINT_S)
    ap.add_argument("--y-tick-step", dest="y_tick_step", type=float,
                    default=Y_TICK_STEP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = gather(a.frame)
    print("  frame %s, %s effectors\n" % (a.frame, EFFECTORS))
    print("  %-7s %-6s %-7s %10s %10s %8s %7s"
          % ("pT", "stage", "class", "mean", "SEM", "pairs", "arrays"))
    for pt in PANELS:
        for d in data[pt]:
            print("  %-7.3f %-6s %-7s %10.4f %10.4f %8d %7d"
                  % (pt, d["stage"], d["edge_class"], d["mean"], d["sem"],
                     d["n_pairs"], d["n_arrays"]))

    if a.layout == "single":
        fig, top_ax = draw_single(data, a)
    else:
        fig, top_ax = draw_panels(data, a)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[(True, s)][0],
                             edgecolor=STYLE[(True, s)][1], linewidth=1.6)
               for s in ("E17.5", "P0")]
    names = [STYLE_LABEL[(True, s)] for s in ("E17.5", "P0")]
    handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                              color=POINT_COLOUR, alpha=0.5,
                              markersize=np.sqrt(a.point_size) * 1.8))
    names.append("one neighbour pair (sampled)")
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Mechanical stress on a cell-cell interface at %s\n"
                 "sum(stress x length) / $L_0$ per neighbour pair;"
                 "  star = mean over arrays, whisker = SEM" % a.frame,
                 fontsize=11, y=0.99, va="top", linespacing=1.6)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.875, bottom=0.135)

    # the two layouts write to different names, so making one does not
    # overwrite the other
    stem = os.path.splitext(a.out or os.path.join(
        RESULTS_DIR, "edge_stress_%s%s"
        % (a.frame, "_single" if a.layout == "single" else "")))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
