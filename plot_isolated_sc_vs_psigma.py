"""SCs with no HC neighbour against psigma, model vs experiment, both stages.

    python plot_isolated_sc_vs_psigma.py

Writes <results>/isolated_sc_vs_psigma.png and isolated_sc_vs_psigma.csv.

This is the statistic that replaced score 3. Unlike scores 1 and 2 it has a
DIRECT experimental value — the same frame-1 valid-cell population score 1's
target comes from — so the model curve can be read against a horizontal band
rather than against another model. Left panel expresses it as a percentage of
SCs, right panel as a percentage of all cells; they carry the same information
and differ only by the HC fraction.

WHICH FRAME. The experiment is one fixed frame and score 1 matches the model's
t0 to it, so t0 is the like-for-like comparison; the FINAL frame is where the
model settles. Both are drawn (final solid, t0 dashed) because they say different
things: at psigma = 0 the model starts above the experimental level at t0 and
then fills in almost completely, while the stress gate arrests that.

REPEATS: solid curves are repeat 1 only, so every psigma is on the same footing —
see plot_scores_vs_psigma.py for why that matters. Points that include collapsed
runs (HC fraction < 0.10, where every surviving SC is trivially isolated) are
ringed, since one collapsed run drags the mean by ~1/n.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR
from build_fullmodel_table import psigma_table, exp_isolated_sc
from plot_scores_vs_psigma import read_runs_csv, COLOUR, ZOOM

OUT_PNG = "isolated_sc_vs_psigma.png"
OUT_CSV = "isolated_sc_vs_psigma.csv"


def main():
    df = read_runs_csv()
    ok = df[df["error"].fillna("") == ""]
    ok = ok.assign(iso_t0=100.0 * ok["n_SC_no_HC_neighbour_t0"] / ok["n_SC_t0"],
                   iso_t0_cells=100.0 * ok["n_SC_no_HC_neighbour_t0"] / ok["n_cells_t0"])
    print("%d run(s), %d psigma value(s)" % (len(ok), ok["psigma"].nunique()))

    base = psigma_table(ok[ok["repeat"] == 1]).assign(series="repeat 1")
    allr = psigma_table(ok).assign(series="all repeats")
    # t0 values are not in the psigma table; aggregate them the same way
    # (per-array mean first, then across arrays)
    t0 = (ok[ok["repeat"] == 1].groupby(["stage", "psigma", "initial_array"])
          [["iso_t0", "iso_t0_cells"]].mean().groupby(["stage", "psigma"]).mean()
          .reset_index())
    pd.concat([base, allr]).to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    panels = (("iso_SC_of_SC", "iso_t0", "% of SCs with no HC neighbour", 0),
              ("iso_SC_of_all_cells", "iso_t0_cells",
               "% of ALL cells that are SCs with no HC neighbour", 1))
    for col, t0col, ylab, k in panels:
      for row, zoom in ((0, False), (1, True)):
        ax = axes[row][k]
        for stage in ("E17.5", "P0"):
            c = COLOUR[stage]
            b = base[base["stage"] == stage].sort_values("psigma")
            if not len(b):
                continue
            exp = exp_isolated_sc(stage)[k]
            me = float(np.mean(exp)); se = float(np.std(exp, ddof=1) / np.sqrt(len(exp)))
            ax.axhspan(me - se, me + se, color=c, alpha=0.13, zorder=0)
            ax.axhline(me, color=c, lw=1.2, ls="-", alpha=0.55, zorder=1)
            ax.annotate("%s experiment %.2f%%" % (stage, me),
                        (0.207 if zoom else 0.355, me),
                        fontsize=7.5, color=c, va="bottom", ha="right")

            ax.errorbar(b["psigma"], b["%s_mean" % col], yerr=b["%s_sem" % col],
                        fmt="-o", color=c, ms=4.5, lw=1.6, capsize=2.5,
                        label="%s, final frame" % stage, zorder=3)
            g = t0[t0["stage"] == stage].sort_values("psigma")
            ax.plot(g["psigma"], g[t0col], "--s", color=c, ms=3.5, lw=1.1,
                    alpha=0.65, mfc="none", label="%s, at $t_0$" % stage, zorder=2)
            # collapsed runs make every surviving SC trivially isolated
            bad = b[b["n_collapsed"] > 0]
            ax.plot(bad["psigma"], bad["%s_mean" % col], "o", mfc="none", mec="k",
                    ms=10, mew=1.2, zorder=4)
        ax.set_xlabel("$p_\\sigma$")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        if zoom:
            # the collapse plateau at ~100% squashes the range that matters
            ax.set_xlim(0.125, 0.212)
            ax.set_ylim(0, 22)
            ax.set_title("zoom: $p_\\sigma$ 0.125 – 0.21, collapse plateau clipped",
                         fontsize=9)
        else:
            ax.set_xlim(-0.012, 0.40)
            ax.legend(fontsize=8, loc="upper left")
    axes[0][0].set_title("Isolated SCs vs $p_\\sigma$ — fraction of SCs", fontsize=11)
    axes[0][1].set_title("Isolated SCs vs $p_\\sigma$ — fraction of all cells", fontsize=11)
    fig.suptitle("SCs with no HC neighbour: model against experiment "
                 "(black ring = the point contains a collapsed run)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(RESULTS_DIR, OUT_PNG), dpi=170, bbox_inches="tight")
    print("wrote %s\n      %s" % (os.path.join(RESULTS_DIR, OUT_PNG),
                                  os.path.join(RESULTS_DIR, OUT_CSV)))

    print("\n  where the FINAL-frame curve crosses the experimental level "
          "(repeat 1, %SCs)")
    for stage in ("E17.5", "P0"):
        b = base[(base["stage"] == stage) & (base["n_collapsed"] == 0)]
        b = b.sort_values("psigma")
        exp = exp_isolated_sc(stage)[0]
        me = float(np.mean(exp))
        x = b["psigma"].to_numpy(float); y = b["iso_SC_of_SC_mean"].to_numpy(float)
        hits = [(x[i], x[i + 1]) for i in range(len(x) - 1)
                if (y[i] - me) * (y[i + 1] - me) < 0]
        print("    %-6s experiment %.2f%%   model range %.2f-%.2f%%   crossings: %s"
              % (stage, me, np.nanmin(y), np.nanmax(y),
                 ", ".join("%.3f-%.3f" % h for h in hits) if hits else "none"))


if __name__ == "__main__":
    main()
