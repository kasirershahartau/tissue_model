"""Plot the three differentiation scores against psigma, per stage (v2 model).

    python plot_psigma_v2.py
    python plot_psigma_v2.py --zoom 0.13 0.175

Top row is the whole swept range, bottom row zooms on the fine grid. The y axis
is LOG because the scores span 0.05 to 1800 — collapse is three orders of
magnitude away from a good fit, and a linear axis would hide everything below it.

Two things the figure is meant to make obvious:

* the DASHED LINE at 8 is the "good" threshold. A psigma is only useful if
  score1 AND score2 sit under it for BOTH stages.
* points where runs COLLAPSED are ringed. Near threshold, whether a given run
  dies is a coin flip on its initial LI seed, so those points are drawn from a
  much wider distribution than the line through them suggests. The scatter
  between adjacent 0.001 steps is the real noise floor.

psigma = 0 comes from the matched cases of v2_differentiation_scores.json, so the
baseline is read rather than retyped.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR

STAGES = ("E17.5", "P0")
COLORS = {"score1": "tab:blue", "score2": "tab:orange", "score3": "tab:green"}
LABELS = {"score1": "score 1  (neighbour pairs at $t_0$)",
          "score2": "score 2  (HC neighbours at differentiation)",
          "score3": "score 3  (% of SCs differentiating, by HC neighbours)"}


def load(stage):
    """[(psigma, {score: value}, n_dead)] including the psigma = 0 baseline."""
    pts = []
    base_path = os.path.join(RESULTS_DIR, "v2_differentiation_scores.json")
    if os.path.isfile(base_path):
        b = json.load(open(base_path)).get("runs=%s|exp=%s" % (stage, stage))
        if b:
            pts.append((0.0, {k: b[k] for k in ("score1", "score2", "score3")}, 0))
    p = os.path.join(RESULTS_DIR, "psigma_scores_v2_%s_ks0.000.json" % stage)
    if os.path.isfile(p):
        for k, v in json.load(open(p)).items():
            if not v:
                continue
            n_dead = len(v.get("dead_runs") or [])
            pts.append((float(k), {s: v.get(s, np.nan) for s in COLORS}, n_dead))
    return sorted(pts)


def draw(ax, pts, xlim=None, show_dead=True):
    x = np.array([p[0] for p in pts])
    for s in ("score1", "score2", "score3"):
        y = np.array([p[1].get(s, np.nan) for p in pts], float)
        # A score of exactly 0 is NOT a perfect fit: it is the degenerate case
        # where the tissue produced no differentiating cells, sim% came out nan,
        # and nan was scored as zero. On a log axis those would plunge to the
        # bottom and read as the best points on the plot, so they are broken out
        # of the line and marked instead.
        degen = y <= 0
        yy = np.where(degen, np.nan, y)
        ax.plot(x, yy, "o-", color=COLORS[s], ms=5, lw=1.6, label=LABELS[s])
        if degen.any():
            ax.plot(x[degen], np.full(degen.sum(), 2e-2), "X", ms=9,
                    color=COLORS[s], mec="k", mew=0.8, zorder=6,
                    label="degenerate (no differentiating cells)"
                    if s == "score2" else None)
    if show_dead:
        dead = [(p[0], max(p[1].get(s, 0) for s in COLORS)) for p in pts if p[2]]
        if dead:
            ax.plot([d[0] for d in dead], [d[1] for d in dead], "o", ms=13,
                    mfc="none", mec="crimson", mew=1.6, zorder=5,
                    label="run(s) collapsed here")
    ax.axhline(8, color="0.35", ls="--", lw=1.4, zorder=0)
    ax.annotate('"good" = 8', (ax.get_xlim()[0], 8), fontsize=8, color="0.35",
                va="bottom", ha="left", xytext=(2, 2), textcoords="offset points")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    if xlim:
        ax.set_xlim(*xlim)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoom", type=float, nargs=2, default=[0.128, 0.175])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "psigma_v2_scores.png"))
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for col, stage in enumerate(STAGES):
        pts = load(stage)
        if not pts:
            continue
        draw(axes[0][col], pts)
        axes[0][col].set_title("%s — full range" % stage, fontsize=11)
        zoom = [p for p in pts if a.zoom[0] <= p[0] <= a.zoom[1]]
        draw(axes[1][col], zoom, xlim=a.zoom)
        axes[1][col].set_title("%s — fine grid (0.001 steps)" % stage, fontsize=11)
        for r in (0, 1):
            axes[r][col].set_xlabel("$p_\\sigma$")
            axes[r][col].set_ylabel("score  (lower is better)")
        print("%-6s %d points, psigma %.3g..%.3g"
              % (stage, len(pts), pts[0][0], pts[-1][0]))

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Differentiation scores vs mechanosensitivity — v2 model, K = 0",
                 y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
