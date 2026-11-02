"""Two supplementary figures for the mechanical fit.

    python plot_mechanical_fit_supplement.py
    python plot_mechanical_fit_supplement.py --only grid --terms roundness_ratio ablation_ratio

Both read the scan JSONs the fit itself wrote, through
plot_mechanical_fit_history.load_scan -- nothing is re-simulated.

GRID (fit_supp_e17_grid) -- the initial 2-D E17.5 fit, step 4: a 5x5 grid over
R and gamma_SC, with alpha and gamma still COUPLED (R = alpha_HC =
gamma_HC/gamma_SC), one heat map per scored term and one for their sum.

BOUNDARY (fit_supp_boundary_<stage>) -- the walk along gamma_SC that chose each
stage's point, one panel per stage: every term and the sum against gamma_SC,
with the lowest sum marked.

WHAT A "SCORE" IS HERE. Each term contributes chi^2 = z^2 to the objective, and
the objective is their sum, so the per-term maps add up to the combined one and
share its units. That is why one colour scale serves all of them: a term drawn
pale is a term that is costing the fit something.

THREE TERMS, NOT TWO. The fit scores roundness, area change near the ablation
AND shrinkage, so "one per score plus the combined" is four panels and four
lines. ``--terms`` selects a subset if a figure should show fewer.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from post_processing import RESULTS_DIR
from plot_mechanical_fit_history import (load_scan, TERMS, TERM_LABEL,
                                         TERM_COLOUR, CHOSEN, save)

CMAP = "plasma_r"             # yellow = low score, dark blue = high
# The box round the winning cell. With the scale reversed the winner always
# sits on the LIGHT end, so black reads against it; under the un-reversed
# plasma it sat on dark blue and had to be bright green instead.
BEST_BOX_COLOUR = "black"
COMBINED_LABEL = r"combined, $\sum z^2$"
COMBINED_COLOUR = "black"
# the gamma_SC walk that picked each stage's point: A0 solved self-consistently
# (steps 6 and 5e). p0_boundary_scan is the earlier A0 = pi/4 ceiling walk and
# exists for P0 only, so it cannot make a per-stage pair; --scan reaches it.
STAGE_SCAN = {"E17.5": "step6", "P0": "step5e"}


def chi2_frame(scan, terms):
    """The scan with a chi^2 column per term and the combined one."""
    g = load_scan(scan).copy()
    for t in terms:
        g["chi2_" + t] = g["z_" + t] ** 2
    # recomputed rather than read from "objective": with --terms the combined
    # map must be the sum of the terms SHOWN, or the panels would not add up
    g["chi2_combined"] = g[["chi2_" + t for t in terms]].sum(axis=1, min_count=1)
    return g


def strip(ax):
    """Leave the data and the frame, take every word off the axes.

    For a panel whose labels are added by hand afterwards: tick MARKS stay, so
    the positions are still there to label, but nothing is written.
    """
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _cell_text(ax, x, y, value):
    ax.text(x, y, "%.3g" % value, ha="center", va="center", fontsize=8.0,
            color="white",
            path_effects=[pe.withStroke(linewidth=1.9, foreground="0.15")])


def grid_figure(out_dir, terms, scan="step4", annotate=False, ncol=2,
                bare=False):
    """One heat map per term plus the combined, over (gamma_SC, R)."""
    g = chi2_frame(scan, terms)
    ys = np.array(sorted(g["R_alpha"].unique()), float)      # R, coupled here
    xs = np.array(sorted(g["gamma_sc"].unique()), float)
    yi = {v: i for i, v in enumerate(ys)}
    xi = {v: i for i, v in enumerate(xs)}

    cols = ["chi2_" + t for t in terms] + ["chi2_combined"]
    labels = [TERM_LABEL[t] for t in terms] + ["combined"]
    grids = []
    for col in cols:
        grid = np.full((len(ys), len(xs)), np.nan)
        for _i, r in g.iterrows():
            grid[yi[r["R_alpha"]], xi[r["gamma_sc"]]] = r[col]
        grids.append(grid)

    # ONE scale for every panel, so the maps are comparable by eye and the
    # combined one is visibly the sum: from 0 to the largest value drawn
    top = float(np.nanmax([np.nanmax(gr) for gr in grids]))
    norm = plt.Normalize(vmin=0.0, vmax=top)

    n = len(cols)
    nrow = int(np.ceil(n / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.55 * ncol, 3.75 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    best = g.loc[g["chi2_combined"].idxmin()]
    for ax, grid, label in zip(axes, grids, labels):
        mesh = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1),
                             np.ma.masked_invalid(grid), norm=norm, cmap=CMAP,
                             edgecolors="white", linewidth=0.8)
        if annotate:
            for i in range(len(ys)):
                for j in range(len(xs)):
                    if np.isfinite(grid[i, j]):
                        _cell_text(ax, j + 0.5, i + 0.5, grid[i, j])
        ax.set_xticks(np.arange(len(xs)) + 0.5)
        ax.set_xticklabels(["%g" % v for v in xs], fontsize=8.5, rotation=45)
        ax.set_yticks(np.arange(len(ys)) + 0.5)
        ax.set_yticklabels(["%g" % v for v in ys], fontsize=9)
        ax.set_title(label, fontsize=10.5)
        # the winner of the combined map, boxed on every panel so each term can
        # be read at the point the fit actually chose
        ax.add_patch(plt.Rectangle((xi[best["gamma_sc"]], yi[best["R_alpha"]]),
                                   1, 1, fill=False,
                                   edgecolor=BEST_BOX_COLOUR,
                                   linewidth=2.8, zorder=5))
    for k, ax in enumerate(axes):
        if k >= n:
            ax.axis("off")
            continue
        if k % ncol == 0:
            ax.set_ylabel(r"$R = \alpha_{HC} = \gamma_{HC}/\gamma_{SC}$",
                          fontsize=10.5)
        # an empty slot below would otherwise leave its neighbour's x axis bare
        if k >= n - ncol:
            ax.set_xlabel(r"$\gamma_{SC}$", fontsize=11)

    print("  grid: %d points, chi^2 scale 0 .. %.3f" % (len(g), top))
    print("  best combined: R = %g, gamma_SC = %g, chi^2 = %.3f"
          % (best["R_alpha"], best["gamma_sc"], best["chi2_combined"]))
    for t in terms:
        print("     %-18s %.3f" % (TERM_LABEL[t], best["chi2_" + t]))
    if bare:
        for ax in axes[:n]:
            strip(ax)
    else:
        fig.suptitle("Initial 2-D fit, E17.5: $\\alpha$ and $\\gamma$ coupled"
                     "   (boxed = lowest combined $\\chi^2$)",
                     fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.105, right=0.865, top=0.885, bottom=0.095,
                        wspace=0.10, hspace=0.30)
    # its own axes, placed AFTER subplots_adjust: a colorbar attached to the
    # panels is laid out before that call and ends up drawn over the last one
    cax = fig.add_axes([0.885, 0.095, 0.018, 0.79])
    cbar = fig.colorbar(mesh, cax=cax)
    if bare:
        # the bar itself is the scale and stays; only its words go
        cax.set_yticklabels([])
    else:
        cbar.set_label(r"$\chi^2 = z^2$  (lower is better)", fontsize=10)
    stem = "fit_supp_e17_grid" + ("_bare" if bare else "")
    return save(fig, os.path.join(out_dir, stem))


def boundary_figure(out_dir, terms, stages=("E17.5", "P0"), log_y=False,
                    scans=None, mark_chosen=False, mark_terms=None,
                    legend=True, bare=False, stacked=True):
    """Per stage: every term and the combined score against gamma_SC.

    THE MARKED POINT IS NOT THE MINIMUM OF THE COMBINED CURVE. It minimises
    ``mark_terms`` -- roundness and shrinkage by default, leaving the ablation
    term out. That is the criterion the manuscript's points satisfy: ablation
    barely moves along this walk (8.6-9.6 at P0, 1.01-1.17 at E17.5), so it
    adds an almost constant offset that cannot discriminate between points but
    can shift the apparent minimum, which is what it does at E17.5.
    """
    scans = scans or STAGE_SCAN
    if stacked:
        # sharex, because the two stages were walked over DIFFERENT gamma_SC
        # grids (E17.5 reaches 0.0175, P0 reaches down to 0.0075): on a common
        # axis the overlapping points line up, at the price of a little empty
        # space at one end of each panel. y stays per-panel -- E17.5 spans 1-3,
        # P0 spans 0-83.
        # hspace 0: the panels meet on one line, which is the upper panel's
        # bottom border and the lower panel's top at once. The lower panel's
        # own top spine goes, so that line is drawn once, and the upper panel
        # loses the x ticks that would poke through it.
        fig, axes = plt.subplots(len(stages), 1, figsize=(6.6, 3.8 * len(stages)),
                                 sharex=True, gridspec_kw=dict(hspace=0.0))
        for ax in np.atleast_1d(axes).ravel()[:-1]:
            ax.tick_params(axis="x", bottom=False)
        for ax in np.atleast_1d(axes).ravel()[1:]:
            ax.spines["top"].set_visible(False)
    else:
        fig, axes = plt.subplots(1, len(stages), figsize=(6.0 * len(stages), 4.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, stage in zip(axes, stages):
        g = chi2_frame(scans[stage], terms).sort_values("gamma_sc")
        x = g["gamma_sc"].to_numpy(float)
        for t in terms:
            ax.plot(x, g["chi2_" + t], "-o", ms=5, lw=1.5,
                    color=TERM_COLOUR[t], label=TERM_LABEL[t], zorder=3)
        ax.plot(x, g["chi2_combined"], "-o", ms=5, lw=2.0,
                color=COMBINED_COLOUR, label=COMBINED_LABEL, zorder=4)

        crit_cols = ["chi2_" + t for t in (mark_terms or terms)
                     if "chi2_" + t in g.columns]
        crit = g[crit_cols].sum(axis=1, min_count=1).to_numpy(float)
        k = int(np.nanargmin(crit))
        # the dot sits on the COMBINED curve at the selected gamma_SC, so the
        # figure marks a point rather than a value of the criterion -- which is
        # a sum over a subset and is not one of the curves drawn
        bx = float(x[k]); by = float(g["chi2_combined"].to_numpy(float)[k])
        ax.axvline(bx, color=COMBINED_COLOUR, ls="--", lw=1.1, alpha=0.75,
                   zorder=1)
        ax.plot([bx], [by], marker="o", ms=13, color=COMBINED_COLOUR,
                markeredgecolor="white", markeredgewidth=1.2, linestyle="none",
                zorder=6,
                label="best: $\\gamma_{SC}$ = %g, $\\chi^2$ = %.3g" % (bx, by))
        print("  %-6s %d points, gamma_SC = %s"
              % (stage, len(x), ", ".join("%g" % v for v in x)))
        print("  %-6s marked gamma_SC = %g  (lowest %s = %.4f)"
              % (stage, bx, " + ".join(t.split("_")[0] for t in
                                       (mark_terms or terms)), crit[k]))
        print("     combined there %.3f   (lowest combined is %.3f at %g)"
              % (by, float(np.nanmin(g["chi2_combined"])),
                 float(x[int(np.nanargmin(g["chi2_combined"].to_numpy(float)))])))
        for t in terms:
            print("     %-18s %.3f" % (TERM_LABEL[t], g["chi2_" + t].iloc[k]))

        chosen = CHOSEN.get(stage, {}).get("gamma_sc")
        if chosen is not None and not np.isclose(chosen, bx):
            print("     NOTE the manuscript's point is gamma_SC = %g, not the"
                  " scan minimum" % chosen)
            if mark_chosen:
                ax.axvline(chosen, color="0.45", ls=":", lw=1.2, zorder=1)
                ax.annotate("chosen for the\nfull model", xy=(chosen, 0.97),
                            xycoords=("data", "axes fraction"), ha="center",
                            va="top", fontsize=8.5, color="0.35")
        if log_y:
            ax.set_yscale("log")
        if stacked and not bare:
            # attached panels share a border, so a title above the lower one
            # would land on that line; the stage goes inside its own panel.
            # It is an annotation rather than a title, so strip() cannot take
            # it off later -- hence the bare check HERE.
            ax.annotate(stage, xy=(0.025, 0.93), xycoords="axes fraction",
                        ha="left", va="top", fontsize=12, fontweight="bold")
        elif not stacked:
            ax.set_title(stage, fontsize=12, fontweight="bold")
    if stacked:
        axes[-1].set_xlabel(r"$\gamma_{SC}$", fontsize=11.5)
        for ax in axes:
            ax.set_ylabel(r"$\chi^2 = z^2$", fontsize=11.5)
    else:
        for ax in axes:
            ax.set_xlabel(r"$\gamma_{SC}$", fontsize=11.5)
        axes[0].set_ylabel(r"$\chi^2 = z^2$  (lower is better)", fontsize=11.5)

    # ONE legend for the figure, under the panels. Per-panel legends repeated
    # the same four term entries and, at P0, sat on top of the curves climbing
    # the right-hand side -- there is no free corner there.
    # Never on the bare version, whatever the caller asked for.
    if legend and not bare:
        handles = [plt.Line2D([], [], color=TERM_COLOUR[t], marker="o", ms=5,
                              lw=1.5) for t in terms]
        names = [TERM_LABEL[t] for t in terms]
        handles.append(plt.Line2D([], [], color=COMBINED_COLOUR, marker="o",
                                  ms=5, lw=2.0))
        names.append(COMBINED_LABEL)
        handles.append(plt.Line2D([], [], color=COMBINED_COLOUR, marker="o",
                                  ms=11, linestyle="none",
                                  markeredgecolor="white", markeredgewidth=1.2))
        # the term KEY's first word, not the label's last: TERM_LABEL splits to
        # "ratio" for roundness_ratio, which reads as nonsense in a sum
        names.append("lowest %s" % " + ".join(t.split("_")[0]
                                              for t in (mark_terms or terms)))
        fig.legend(handles, names, frameon=False, fontsize=8.5,
                   ncol=2 if stacked else len(handles), loc="lower center",
                   bbox_to_anchor=(0.5, 0.004))

    if bare:
        for ax in axes:
            strip(ax)
        if stacked:
            fig.subplots_adjust(left=0.075, right=0.98, top=0.98, bottom=0.05,
                                hspace=0.0)
        else:
            fig.subplots_adjust(left=0.055, right=0.985, top=0.965,
                                bottom=0.075, wspace=0.12)
    else:
        fig.suptitle("Walk along $\\gamma_{SC}$ with $A_0$ solved: the score of"
                     " every term, and their sum", fontsize=12, y=0.985)
        if stacked:
            fig.subplots_adjust(left=0.125, right=0.98, top=0.925, bottom=0.185,
                                hspace=0.0)
        else:
            fig.subplots_adjust(left=0.075, right=0.985, top=0.855,
                                bottom=0.115, wspace=0.18)
    stem = "fit_supp_boundary" + ("_bare" if bare else "")
    return save(fig, os.path.join(out_dir, stem))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("grid", "boundary"), default=None)
    ap.add_argument("--terms", nargs="+", default=list(TERMS),
                    choices=list(TERMS),
                    help="which scored terms to draw (default: all three)")
    ap.add_argument("--log-y", dest="log_y", action="store_true",
                    help="log y on the boundary panels")
    ap.add_argument("--mark-terms", dest="mark_terms", nargs="+",
                    default=["roundness_ratio", "shrinkage"],
                    choices=list(TERMS),
                    help="which terms the marked point minimises on the"
                         " boundary panels (default: roundness + shrinkage,"
                         " the criterion the chosen points satisfy)")
    ap.add_argument("--no-legend", dest="legend", action="store_false",
                    help="drop the legend from the labelled boundary panels"
                         " (the bare ones never carry one)")
    ap.add_argument("--side-by-side", dest="stacked", action="store_false",
                    help="boundary panels in one row instead of stacked")
    ap.add_argument("--annotate", action="store_true",
                    help="print each cell's value on the heat maps")
    ap.add_argument("--ncol", type=int, default=2,
                    help="heat-map panels per row (default 2)")
    ap.add_argument("--mark-chosen", dest="mark_chosen", action="store_true",
                    help="also mark the gamma_SC the full model runs at, when"
                         " it differs from the scan minimum")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    a = ap.parse_args()

    # each figure is written twice: labelled, and bare for hand annotation
    if a.only != "boundary":
        print("\nInitial 2-D E17.5 grid")
        for bare in (False, True):
            print("  wrote %s" % grid_figure(a.out_dir, a.terms,
                                             annotate=a.annotate, ncol=a.ncol,
                                             bare=bare))
    if a.only != "grid":
        print("\nGamma_SC walk, per stage")
        for bare in (False, True):
            print("  wrote %s" % boundary_figure(a.out_dir, a.terms,
                                                 log_y=a.log_y,
                                                 mark_chosen=a.mark_chosen,
                                                 mark_terms=a.mark_terms,
                                                 legend=a.legend, bare=bare,
                                                 stacked=a.stacked))


if __name__ == "__main__":
    main()
