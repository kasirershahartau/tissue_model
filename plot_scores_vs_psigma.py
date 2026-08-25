"""Score 1 and score 2 against psigma, one subplot each, both stages.

    python plot_scores_vs_psigma.py

Reads fullmodel_runs.csv (built by build_fullmodel_table.py) and writes
<results>/scores_vs_psigma.png plus scores_vs_psigma.csv with the plotted values.

REPEATS ARE THE TRAP HERE. The sweep has 1 repeat at most psigma values, 3 at
0.160-0.165, 5 at 0.163, and 10 at 0 and 0.162. The score is an n-sigma, and
SEM_sim sits in its denominator, so adding repeats tightens the per-array means
and INFLATES the score wherever the model is systematically off — P0's score 2 at
psigma 0.162 went 6.05 -> 13.45 -> 29.11 at 3, 5 and 10 repeats without a single
model value changing. Plotting those numbers on one curve would show a spike at
0.162 that is an artefact of how much data it has.

So the SOLID CURVE IS REPEAT 1 ONLY, uniform across every psigma and therefore
comparable left to right. The extra repeats are drawn as open markers at the same
psigma, connected by a vertical line to their repeat-1 point: the gap between
them is the size of the repeat artefact, not a change in the model. Read the
solid curve for shape and the open markers for how firm a given point is.

Per-array averaging is used throughout (10 arrays = 10 data points), as in
score_psigma_pooled.py.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR
from build_fullmodel_table import psigma_table

OUT_PNG = "scores_vs_psigma.png"
OUT_CSV = "scores_vs_psigma.csv"
COLOUR = {"E17.5": "tab:blue", "P0": "tab:red"}
ZOOM = (0.1585, 0.1715)


def read_runs_csv():
    """Tolerate a torn final line, which happens if the builder is still running."""
    path = os.path.join(RESULTS_DIR, "fullmodel_runs.csv")
    with open(path) as fh:
        lines = fh.read().splitlines()
    ncol = lines[0].count(",") + 1
    if lines and lines[-1].count(",") + 1 != ncol:
        lines = lines[:-1]
        print("  (dropped a torn final line - the builder is still writing)")
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines) + "\n"))
    return df.drop_duplicates("model_name", keep="last")


def main():
    df = read_runs_csv()
    # incomplete runs are dropped when the table is written, so the column
    # may be absent from the saved file
    ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    print("%d run(s), %d psigma value(s), stages %s"
          % (len(ok), ok["psigma"].nunique(), sorted(ok["stage"].dropna().unique())))

    base = psigma_table(ok[ok["repeat"] == 1])          # comparable everywhere
    allr = psigma_table(ok)                             # every repeat that exists
    base = base.assign(series="repeat 1")
    allr = allr.assign(series="all repeats")
    pd.concat([base, allr]).to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)

    def draw(ax, score, zoom=False):
        undef = "%s_undefined_terms" % score
        dead = {}
        for stage in ("E17.5", "P0"):
            b = base[base["stage"] == stage].sort_values("psigma")
            if zoom:
                b = b[b["psigma"].between(*ZOOM)]
            if not len(b):
                continue
            c = COLOUR[stage]
            good = b[b[undef] == 0]
            ax.plot(good["psigma"], good[score], "-o", color=c, ms=4.5, lw=1.6,
                    label="%s (1 repeat)" % stage, zorder=3)
            # A point with an undefined term sums to 0 under the scoring
            # convention, so drawing it on the curve would show a dead parameter
            # point as a perfect fit. Mark its psigma instead of inventing a y.
            dead[stage] = b[b[undef] > 0]["psigma"].to_numpy(float)
            a = allr[(allr["stage"] == stage) & (allr["n_repeats"] > 1)].sort_values("psigma")
            if zoom:
                a = a[a["psigma"].between(*ZOOM)]
            for _i, r in a.iterrows():
                m = b[np.isclose(b["psigma"], r["psigma"])]
                if len(m):
                    ax.plot([r["psigma"]] * 2, [float(m[score].iloc[0]), r[score]],
                            "-", color=c, lw=0.9, alpha=0.55, zorder=2)
                ax.plot(r["psigma"], r[score], "o", mfc="none", mec=c, ms=8,
                        mew=1.5, zorder=4)
                ax.annotate("%dx" % int(r["n_repeats"]), (r["psigma"], r[score]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7, color=c)
        for stage, xs in dead.items():
            for k, x in enumerate(xs):
                ax.axvline(x, color=COLOUR[stage], lw=1.2, alpha=0.55, zorder=1,
                           ls="--" if stage == "E17.5" else ":",
                           label=("%s: no events, term undefined" % stage
                                  if k == 0 else None))
        ax.set_yscale("log")
        if zoom:
            ax.set_xlim(*ZOOM)
        ax.set_xlabel("$p_\\sigma$")
        ax.set_ylabel("%s  ($\\chi^2$)" % score)
        ax.grid(alpha=0.25, which="both")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6),
                             gridspec_kw=dict(height_ratios=[1.35, 1.0]))
    for col, (score, title) in enumerate(
            (("score1", "Score 1 — neighbour pairs at $t_0$"),
             ("score2", "Score 2 — HC neighbours at differentiation"))):
        draw(axes[0][col], score)
        axes[0][col].set_title(title, fontsize=11)
        axes[0][col].legend(fontsize=8, loc="best")
        # the sweep is dense between 0.16 and 0.17, where the full range hides it
        draw(axes[1][col], score, zoom=True)
        axes[1][col].set_title("zoom: $p_\\sigma$ 0.159 – 0.171", fontsize=9)

    n_col = int(allr["n_collapsed"].sum()) if len(allr) else 0
    fig.suptitle("Full-model scores against $p_\\sigma$ — open markers are the same "
                 "point with more repeats (%d collapsed run(s) included)" % n_col,
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(RESULTS_DIR, OUT_PNG)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote %s\n      %s" % (out, os.path.join(RESULTS_DIR, OUT_CSV)))

    for score in ("score1", "score2"):
        undef = "%s_undefined_terms" % score
        print("\n  %s (repeat 1, points with an undefined term excluded)" % score)
        for stage in ("E17.5", "P0"):
            b = base[(base["stage"] == stage) & (base[undef] == 0)].sort_values("psigma")
            skipped = base[(base["stage"] == stage) & (base[undef] > 0)]["psigma"]
            if len(b):
                lo = b.loc[b[score].idxmin()]
                print("    %-6s best at psigma %.3f -> %.3f   (range %.2f - %.2f)%s"
                      % (stage, lo["psigma"], lo[score], b[score].min(), b[score].max(),
                         "   excluded: %s" % np.round(skipped.to_numpy(float), 3).tolist()
                         if len(skipped) else ""))


if __name__ == "__main__":
    main()
