"""Neighbour-pair composition as bars: every model against the experiment.

    python plot_neighbor_pair_bars.py
    python plot_neighbor_pair_bars.py --figure hchc
    python plot_neighbor_pair_bars.py --exp-source ours

Score 1 asks what fraction of the cell-cell contacts are HC:HC and what fraction
HC:SC. HC:HC is the informative one — lateral inhibition should keep hair cells
apart, so touching HCs are the pattern's failures.

TWO figures, split by model family the same way plot_hc_neighbors_bars is:

    li      lateral inhibition only against the experiment
    full    the full model at pT = 0 and 0.162 against it

Each holds BOTH pair types side by side — HC:HC left, HC:SC right, with a y axis
each, because HC:HC sits below 2% and HC:SC near 45% and one scale would flatten
HC:HC. Each panel holds both stages: one block per stage, the family's bars then
the experiment inside it, colour carrying the stage. pT = 0.162 is hatched, since
it shares its stage's colour with pT = 0.

WHICH EXPERIMENTAL TARGET. The two tables used to disagree — about 10% on HC:HC
and 0.9% on HC:SC — because our cell mask kept cells the reference pipeline
drops. post_processing.experimental_cell_mask now implements that rule (invalid
cells, edge cells, and neighbours of invalid cells, the last tested against the
``neighbors`` column rather than the contact matrix), and the two agree to 1e-9
at both stages on both means AND both SEMs.

So --exp-source now picks only WHERE the same numbers are read from:

    ours   fullmodel_experiment.pkl (default) — carries the three PER-MOVIE
           values, so the experimental bar gets its scattered points
    li     lateral_inhibition_tables.xlsx, morphology_summary — mean and SEM
           only, so that bar is drawn without points. Kept as a cross-check
           against the independent pipeline.

THE MODEL SIDE is identical either way: plot_neighbor_pairs.gather, i.e.
pct_HCHC_contacts_t0 / pct_HCSC_contacts_t0 averaged within an initial array,
then mean +- SEM across arrays.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from post_processing import RESULTS_DIR
import plot_neighbor_pairs as P
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_COLOUR, POINT_SIZE,
                                 STAT_COLOUR, ERROR_CAPSIZE, ERROR_LINEWIDTH,
                                 BAR_W, LI_DIR)

STAGES = ("E17.5", "P0")
STAGE_GAP = 0.8                  # bars inside a stage touch; the blocks do not
# pT = 0.162 shares its stage's colour with pT = 0, so it is hatched to tell the
# two apart without leaving the one-colour-per-stage palette
HATCHED_PSIGMA = 0.162
HATCH = "/"
# blank space, in bar widths, left at the right-hand end of every panel but the
# last, so the next panel's y tick labels have somewhere to sit
LABEL_ROOM = 1.9
LI_WORKBOOK = "lateral_inhibition_tables.xlsx"
def hatch_for(index):
    """'/' for the pT = 0.162 full model, None otherwise."""
    _stage, kind, _label, sel = P.SOURCES[index]
    if kind == "full" and np.isclose(float(sel["psigma"]), HATCHED_PSIGMA):
        return HATCH
    return None


def short_label(index):
    """The x label for a source, from its entry in plot_neighbor_pairs.SOURCES.

    Built from the KIND and the selector, not from the display label — that one
    carries the pS/pR values and a newline, so matching on it silently fails.
    """
    _stage, kind, _label, sel = P.SOURCES[index]
    if kind == "li":
        return "LI only"
    if kind == "exp":
        return "exp"
    return r"$p_T$=%g" % float(sel["psigma"])
# Two figures, split by model family the way plot_hc_neighbors_bars is: lateral
# inhibition alone in one, the two full-model pT values in the other.
#
# BOTH PAIR TYPES SHARE A FIGURE, side by side with A Y AXIS EACH. They cannot
# share one: HC:HC sits below 2% and HC:SC near 45%, so a common scale would
# flatten HC:HC to a line on the floor. plot_neighbor_pairs splits them for the
# same reason.
PAIRS = [("HC:HC", "HCHC"), ("HC:SC", "HCSC")]      # left panel first
FIGURES = {
    "li": dict(kinds=("li", "exp"), stem="neighbor_pairs_li_vs_exp",
               title="Neighbour-pair composition: "
                     "lateral inhibition only against the experiment"),
    "full": dict(kinds=("full", "exp"), stem="neighbor_pairs_fullmodel_vs_exp",
                 title="Neighbour-pair composition: "
                       "the full model against the experiment"),
}


def li_experimental_target(column, li_dir=LI_DIR):
    """{stage: (mean %, SEM %)} from the lateral-inhibition workbook.

    The columns hold FRACTIONS, so they are scaled to percent here to match the
    model columns, which are already percentages.
    """
    w = pd.read_excel(os.path.join(li_dir, LI_WORKBOOK), "morphology_summary")
    out = {}
    for stage in STAGES:
        r = w[w["stage"] == stage].iloc[0]
        out[stage] = (100.0 * float(r["%s_fraction_exp_mean" % column]),
                      100.0 * float(r["%s_fraction_exp_sem" % column]))
    return out


def gather(which, exp_source, li_dir=LI_DIR):
    """One row per (pair, stage, source), in the order the bars are drawn."""
    spec = FIGURES[which]
    data = P.gather(li_dir=li_dir)

    rows = []
    for pi, (pair, column) in enumerate(PAIRS):
        target = (li_experimental_target(column, li_dir)
                  if exp_source == "li" else None)
        for si, stage in enumerate(STAGES):
            inside = [d for d in data
                      if d["stage"] == stage and d["pair"] == pair
                      and d["kind"] in spec["kinds"]]
            inside.sort(key=lambda d: d["index"])    # SOURCES order
            for ki, d in enumerate(inside):
                is_exp = d["kind"] == "exp"
                mean, sem, pts = d["mean"], d["sem"], d["per_unit"]
                if is_exp and target is not None:
                    mean, sem = target[stage]
                    pts = np.array([], float)  # no per-movie values in that table
                face, edge = STYLE[(not is_exp, stage)]
                rows.append(dict(pair=pair, pair_index=pi, stage=stage,
                                 stage_index=si, source_index=ki,
                                 key=d["kind"], label=P._plain(d["label"]),
                                 short=short_label(d["index"]),
                                 hatch=hatch_for(d["index"]),
                                 face=face, edge=edge, mean=mean, sem=sem,
                                 points=np.asarray(pts, float),
                                 n_units=d["n_units"], n_runs=d["n_runs"],
                                 z=np.nan, z2=np.nan))
    # z of each model bar against the experiment of its own PAIR and stage
    exp = {(r["pair"], r["stage"]): r for r in rows if r["key"] == "exp"}
    for r in rows:
        if r["key"] == "exp":
            continue
        e = exp[(r["pair"], r["stage"])]
        den = np.hypot(r["sem"], e["sem"])
        if np.isfinite(den) and den > 0:
            r["z"] = (r["mean"] - e["mean"]) / den
            r["z2"] = r["z"] ** 2
    return rows


BOTTOM_PAD = 0.08         # of the visible span, left below the lowest point
TOP_PAD = 0.22            # ... and left above the highest, as white space


def tighten_bottom(ax, rows, pad=BOTTOM_PAD, top_pad=TOP_PAD):
    """Start the axis just under the lowest thing drawn, not at zero.

    NOTE this truncates the bars: with the baseline cut off, a bar's LENGTH no
    longer encodes its value and only the top edge is readable. That is the
    point here — the differences between these sources are small next to their
    absolute size — but it has to be said in the caption.

    The floor is the lowest of every bar's mean minus its SEM and every
    scattered point, so nothing drawn falls outside the axis. The ceiling is
    matplotlib's own, raised by ``top_pad`` of the span so the tallest whisker
    or scattered point is not left touching the frame.
    """
    lows = []
    for r in rows:
        sem = r["sem"] if np.isfinite(r["sem"]) else 0.0
        lows.append(float(r["mean"]) - sem)
        if len(r["points"]):
            lows.append(float(np.min(r["points"])))
    if not lows:
        return
    lo, hi = min(lows), ax.get_ylim()[1]
    bottom = lo - pad * max(hi - lo, 1e-12)
    if lo >= 0.0:
        # a percentage cannot be negative, so the pad never opens empty space
        # below zero
        bottom = max(bottom, 0.0)
    ax.set_ylim(bottom, hi + top_pad * max(hi - bottom, 1e-12))


def draw(rows, which, exp_source, rng, out_dir, scores=False, bare=False):
    spec = FIGURES[which]
    n_src = max(r["source_index"] for r in rows) + 1
    block = n_src + STAGE_GAP
    right = (len(STAGES) - 1) * block + n_src - 1
    # Every panel but the last ends in blank space for the NEXT panel's y tick
    # labels — see the comment where the limits are applied — so the panels do
    # not span the same number of x units. Giving them equal figure width would
    # then render the same bar narrower on the wider panel, so the widths are
    # made PROPORTIONAL TO THE SPANS instead: one data unit is the same number
    # of pixels everywhere, and every bar comes out the same width.
    lims = [(-0.75, right + (0.75 if i == len(PAIRS) - 1 else LABEL_ROOM))
            for i in range(len(PAIRS))]
    spans = [hi - lo for lo, hi in lims]
    # one panel per pair type, side by side, each keeping its own y scale. No
    # sharex: they sit in separate columns, so each carries its own stage ticks.
    fig, axes = plt.subplots(1, len(PAIRS),
                             figsize=(2.4 + 1.9 * len(STAGES) * n_src, 5.6),
                             gridspec_kw=dict(width_ratios=spans))
    axes = np.atleast_1d(axes)

    def xpos(r):
        return r["stage_index"] * block + r["source_index"]

    for pi, (pair, _col) in enumerate(PAIRS):
        ax = axes[pi]
        mine = [r for r in rows if r["pair_index"] == pi]
        for r in mine:
            x = xpos(r)
            ax.bar(x, r["mean"], BAR_W, color=r["face"], edgecolor=r["edge"],
                   linewidth=1.6, hatch=r["hatch"], zorder=2)
            ax.errorbar(x, r["mean"], yerr=r["sem"], fmt="none",
                        ecolor=STAT_COLOUR, elinewidth=ERROR_LINEWIDTH,
                        capsize=ERROR_CAPSIZE, capthick=ERROR_LINEWIDTH, zorder=4)
            v = r["points"]
            if len(v):
                jit = rng.uniform(-BAR_W * 0.26, BAR_W * 0.26, size=len(v))
                ax.scatter(x + jit, v,
                           s=POINT_SIZE["exp" if r["key"] == "exp" else "model"],
                           marker=".", color=POINT_COLOUR, alpha=0.8,
                           linewidths=0, zorder=3)
        tighten_bottom(ax, mine)
        if scores:
            # headroom is per panel, since the two carry different scales, and
            # it is a fraction of the SPAN rather than of the top — the axis no
            # longer starts at zero, so scaling the top would not be a margin
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.14 * (hi - lo))
            span = ax.get_ylim()[1] - lo
            for r in mine:
                if not np.isfinite(r["z2"]):
                    continue
                over = [r["mean"] + (r["sem"] if np.isfinite(r["sem"]) else 0.0)]
                if len(r["points"]):
                    over.append(float(np.max(r["points"])))
                ax.annotate("%.3g" % r["z2"],
                            xy=(xpos(r), max(over) + 0.015 * span), ha="center",
                            va="bottom", fontsize=9, color=STAT_COLOUR, zorder=6)
        if not bare:
            ax.set_ylabel("%% of %s contacts" % pair, fontsize=11)

    # One tick per STAGE GROUP, centred, carrying the stage name — the only
    # thing the x axis groups by here. Naming each bar is left to the legend:
    # colour separates model from experiment, the hatch separates the two pT
    # values. Side by side, so every panel needs its own copy.
    centres = [si * block + (n_src - 1) / 2.0 for si in range(len(STAGES))]
    for i, ax in enumerate(axes):
        ax.set_xticks(centres)
        # The panels are flush, so the right panel's y tick labels are drawn
        # over its neighbour's territory. Every panel but the last therefore
        # ends in blank space wide enough to hold them, instead of its bars
        # running up to the shared border. The BARE version keeps that space
        # even though nothing is drawn in it — it is where the ticks and the
        # axis label go when they are added by hand.
        ax.set_xlim(*lims[i])
        if bare:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            continue
        ax.set_xticklabels(list(STAGES), fontsize=12.5, fontweight="bold")
        if i:
            # the joined panels put this 0 right beside its neighbour's own
            # zero, on the shared border, where it reads as a stray digit
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _p: "" if v == 0 else "%g" % v))
    handles, names = [], []
    for is_model in (True, False):
        for stage in STAGES:
            face, edge = STYLE[(is_model, stage)]
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=face,
                                         edgecolor=edge, linewidth=1.6))
            names.append(STYLE_LABEL[(is_model, stage)])
    # the hatch is a second channel on top of the stage colour, so its key is a
    # neutral swatch rather than one per stage
    if any(r["hatch"] for r in rows):
        # the key swatch is small enough that a single "/" can fall entirely
        # outside it and render blank, so the key is drawn denser than the bars
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="0.25", linewidth=1.6,
                                     hatch=HATCH * 3))
        names.append(r"full model, $p_T$ = 0.162")
    handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                              color=POINT_COLOUR, alpha=0.8, markersize=9))
    names.append("one initial array"
                 + ("" if exp_source == "li" else " / one movie"))
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    second = "bar = mean, whisker = SEM"
    if exp_source == "li":
        second += ";  experimental target read from the lateral-inhibition " \
                  "tables, which carry no per-movie values"
    if scores:
        second += "\nnumber over a model bar is $z^2$ against its stage's experiment"
    fig.suptitle(spec["title"] + "\n" + second, fontsize=11.5, y=0.985,
                 va="top", linespacing=1.5)
    # wspace 0: the panels meet on one line, which is the HC:HC panel's right
    # border and the HC:SC panel's y axis at once
    fig.subplots_adjust(left=0.07, right=0.985,
                        top=0.855 if not scores else 0.805, bottom=0.215,
                        wspace=0.0)

    # the default source keeps the plain name; the cross-check gets a suffix
    stem = os.path.join(out_dir, spec["stem"]
                        + ("_litarget" if exp_source == "li" else "")
                        + ("_scored" if scores else "")
                        + ("_bare" if bare else ""))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=tuple(FIGURES) + ("all",), default="all",
                    help="li, full, or all")
    ap.add_argument("--exp-source", dest="exp_source", choices=("ours", "li"),
                    default="ours",
                    help="where to read the (now identical) experimental target: "
                         "ours keeps the per-movie points, li is the cross-check")
    ap.add_argument("--li-dir", default=LI_DIR)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for which in (sorted(FIGURES) if a.figure == "all" else (a.figure,)):
        rows = gather(which, a.exp_source, a.li_dir)
        print("\n  %s   (experimental target: %s)"
              % (FIGURES[which]["title"], a.exp_source))
        print("  %-6s %-6s %-26s %9s %9s %7s %7s %8s %8s"
              % ("pair", "stage", "source", "mean %", "SEM", "units", "runs",
                 "z", "z^2"))
        for r in rows:
            print("  %-6s %-6s %-26s %9.4f %9.4f %7d %7d %8s %8s"
                  % (r["pair"], r["stage"], r["label"][:26], r["mean"], r["sem"],
                     r["n_units"], r["n_runs"],
                     "%+.3f" % r["z"] if np.isfinite(r["z"]) else "-",
                     "%.3g" % r["z2"] if np.isfinite(r["z2"]) else "-"))
        # labelled and bare, scored and not: the bare pair carries no axis
        # label and no tick labels, for a composite figure whose labels are
        # added by hand
        for scores in (False, True):
            for bare in (False, True):
                stem = draw(rows, which, a.exp_source,
                            np.random.default_rng(a.seed), a.out_dir, scores,
                            bare)
                print("  wrote %s.png / .svg" % stem)


if __name__ == "__main__":
    main()
