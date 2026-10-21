"""Score 1, score 2 and their total against pT, summed over both stages.

    python plot_supp_pt_scores.py
    python plot_supp_pt_scores.py --repeats 3 --zoom 0.1595 0.1705

The supplementary panel that justifies the chosen gate value. Three curves --
score 1 (neighbour pairs at t0), score 2 (HC neighbours at differentiation) and
their sum -- each SUMMED OVER E17.5 AND P0, which is the quantity the choice was
made on.

EVERY POINT RESTS ON THE SAME NUMBER OF REPEATS, three per initial array. The
score is an n-sigma with SEM_sim in its denominator, so a point measured at ten
repeats scores higher than the same model at one, and a curve mixing them shows
structure that is an artefact of sampling. pT values without three repeats at
BOTH stages are dropped and listed, never plotted alongside.

NO ERROR BARS. A score is one chi^2 computed from the arrays POOLED -- the
array-to-array SEM is inside it, in the denominator of its own n-sigma, and
there is no second spread to draw around it.

THE INSET IS THE REFINED SEARCH, the dense cluster between 0.160 and 0.170 that
the coarse sweep pointed at. It is linear where the main axes are logarithmic:
the whole point of the zoom is the ordering of points that differ by a factor
of three, which a log axis flattens.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import RESULTS_DIR
from plot_scores_3rep import build

# (column, label, colour) -- score 1 and 2 in their own colours, the sum in
# black, as on the mechanical-fit panels
SERIES = [("score1", "score 1 — neighbour pairs at $t_0$", "#1f77b4"),
          ("score2", "score 2 — HC neighbours at differentiation", "#9467bd"),
          ("total", "total", "black")]
ZOOM = (0.1595, 0.1705)
BEST_COLOUR = "#00c020"       # the chosen point, against a black total curve


def summed(repeats):
    """One row per pT: each score summed over the stages that have it.

    A pT is kept only where BOTH stages were measured and no term is
    undefined -- an undefined term contributes 0 under the summing convention,
    which would make a dead point look like a good one.
    """
    tab, short, have = build(repeats)
    piv = tab.pivot_table(index="psigma", columns="stage",
                          values=[c for c, _l, _c in SERIES] + ["undefined"])
    stages = sorted({s for _v, s in piv.columns})
    out = {}
    keep = []
    for ps in piv.index:
        if any(np.isnan(piv[(c, s)][ps]) for c, _l, _c in SERIES
               for s in stages):
            continue
        if sum(piv[("undefined", s)][ps] for s in stages) > 0:
            continue
        keep.append(float(ps))
    out["psigma"] = np.array(keep, float)
    for c, _l, _col in SERIES:
        out[c] = np.array([sum(piv[(c, s)][ps] for s in stages)
                           for ps in keep], float)
    return out, short, have, stages


def draw_series(ax, d, mask=None, ms=5.0, lw=1.6):
    x = d["psigma"] if mask is None else d["psigma"][mask]
    for c, label, colour in SERIES:
        y = d[c] if mask is None else d[c][mask]
        ax.plot(x, y, "-o", ms=ms, lw=lw if c != "total" else lw + 0.5,
                color=colour, label=label, zorder=4 if c == "total" else 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--zoom", type=float, nargs=2, default=list(ZOOM))
    ap.add_argument("--linear-y", dest="log_y", action="store_false",
                    help="linear main axes (default: log, the range is 20-450)")
    ap.add_argument("--no-mark", dest="mark", action="store_false",
                    help="do not mark the lowest total")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    a = ap.parse_args()

    d, short, have, stages = summed(a.repeats)
    print("  stages summed: %s" % ", ".join(stages))
    print("  %d pT values at %d repeats: %s"
          % (len(d["psigma"]), a.repeats,
             ", ".join("%g" % v for v in d["psigma"])))
    if short:
        print("  dropped (fewer than %d repeats): %s"
              % (a.repeats, ", ".join("%g (%d)" % (p, have[p]) for p in short)))
    k = int(np.argmin(d["total"]))
    best = float(d["psigma"][k])
    print("\n  %7s %9s %9s %9s" % ("pT", "score1", "score2", "total"))
    for i, ps in enumerate(d["psigma"]):
        print("  %7.3f %9.3f %9.3f %9.3f%s"
              % (ps, d["score1"][i], d["score2"][i], d["total"][i],
                 "   <- lowest total" if i == k else ""))

    def mark_on(axis, ms, lw, label=None):
        if not a.mark:
            return
        axis.axvline(best, color="black", ls="--", lw=lw, alpha=0.75, zorder=1)
        axis.plot([best], [d["total"][k]], marker="o", ms=ms,
                  color=BEST_COLOUR, markeredgecolor="black",
                  markeredgewidth=1.0, linestyle="none", zorder=6, label=label)

    def dress_inset(axis):
        axis.set_xlim(*a.zoom)
        # explicit ticks: the default locator crowds five labels into this
        # width and clips the last one against the frame
        axis.set_xticks([0.160, 0.163, 0.166, 0.169])
        axis.grid(alpha=0.22)

    zoom_mask = (d["psigma"] >= a.zoom[0]) & (d["psigma"] <= a.zoom[1])

    # three files: the labelled figure with its inset, the main curve alone
    # with nothing written on it, and the zoom alone with nothing written on it
    for kind in ("full", "bare", "inset_bare"):
        bare = kind != "full"
        if kind == "inset_bare":
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            draw_series(ax, d, mask=zoom_mask, ms=5.0, lw=1.6)
            mark_on(ax, 11, 1.1)
            dress_inset(ax)
        else:
            fig, ax = plt.subplots(figsize=(8.2, 5.4))
            draw_series(ax, d)
            if a.log_y:
                ax.set_yscale("log")
            if kind == "full":
                # headroom for the inset. Every free corner of this plot is
                # next to the dense cluster the inset magnifies, so the inset
                # went ON TOP of it; stretching the axis instead keeps the
                # curves clear. Only needed when the inset is actually drawn.
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi * 7.0 if a.log_y else hi * 1.9)
            mark_on(ax, 12, 1.1, "lowest total: $p_T$ = %g" % best)
            ax.grid(alpha=0.22, which="both")

        if kind == "full":
            inset = ax.inset_axes([0.085, 0.605, 0.46, 0.355])
            draw_series(inset, d, mask=zoom_mask, ms=4.0, lw=1.3)
            mark_on(inset, 9, 1.0)
            dress_inset(inset)
            inset.tick_params(labelsize=7.5)
            inset.set_title("refined search", fontsize=8.5)
            ax.set_xlabel("$p_T$", fontsize=12)
            ax.set_ylabel("$\\chi^2$ summed over stages  (lower is better)",
                          fontsize=11.5)
            ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
            fig.suptitle("Score against $p_T$, both stages summed"
                         "   (every point at %d repeats x 10 arrays per stage)"
                         % a.repeats, fontsize=11.5, y=0.975)
            fig.subplots_adjust(left=0.095, right=0.985, top=0.905,
                                bottom=0.105)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            fig.subplots_adjust(left=0.065, right=0.985, top=0.98, bottom=0.065)

        stem = os.path.join(a.out_dir, "supp_pt_scores"
                            + {"full": "", "bare": "_bare",
                               "inset_bare": "_inset_bare"}[kind])
        for ext in ("png", "svg"):
            fig.savefig("%s.%s" % (stem, ext), dpi=200)
        plt.close(fig)
        print("  wrote %s.png / .svg" % stem)


if __name__ == "__main__":
    main()
