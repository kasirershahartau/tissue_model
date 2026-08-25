"""Plot psigma sweep scores from the psigma_scores_*.json files.

    python plot_psigma_scores.py                       # every psigma_scores_*.json
    python plot_psigma_scores.py P0 E17.5_P0           # named tags only

These are the DIFFERENTIATION scores of compare_full_model_differentiation_to_
experiments - NOT the mechanical fit's roundness/ablation/shrinkage terms. The
mechanical parameters are already fixed per stage before this sweep starts; the
sweep varies only psigma. The three terms are:

    1. neighbour-pair composition (HC:HC%, HC:SC%) at the best-matching initial
       frame - i.e. does the sheet reproduce the experimental starting state
    2. HC-neighbour count AT DIFFERENTIATION, as % of all differentiating cells
       (buckets 0 and 1)
    3. % of initial SCs that differentiate, grouped by their initial
       HC-neighbour count (buckets 0, 1, >=2)

Each bucket is chi^2 of an n-sigma pooling 10 sims against 3 experiments.

For each file one figure is written to the results dir:

    top row     combined objective (all stages summed) and the score1+score2
                partial sum - score3 is the term most directly targeted by
                mechanosensitivity, so seeing the sweep without it shows
                whether psigma pays for that gain elsewhere.
    bottom row  one panel per stage with the individual terms.

y axes are log because the terms span three orders of magnitude; a run that
blows up (score1 ~ 500 when differentiation collapses) would otherwise flatten
everything else onto the axis.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import RESULTS_DIR

SCORE_KEYS = ("score1", "score2", "score3")
SCORE_LABELS = {
    "score1": "score1  neighbour pairs at best-matching initial frame",
    "score2": "score2  HC nb at differentiation (% of differentiating)",
    "score3": "score3  % of initial SCs differentiating, by initial HC nb",
}
SCORE_COLOURS = {"score1": "#1b7837", "score2": "#2166ac", "score3": "#b2182b"}
STAGE_MARKERS = {"E17.5": "o", "P0": "s"}


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    psigmas = np.array(sorted(float(p) for p in data["scores"]))
    keys = sorted(data["scores"], key=float)
    return data, psigmas, [data["scores"][k] for k in keys]


def _decorate(ax, psigmas, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$p_\sigma$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both", linewidth=0.5)
    if len(psigmas) > 1:
        span = np.ptp(psigmas)
        ax.set_xlim(psigmas.min() - 0.02 * span, psigmas.max() + 0.02 * span)
    ax.legend(fontsize=7, framealpha=0.9)


def plot_file(path):
    data, psigmas, rows = load(path)
    stages = data["stages"]
    K = data.get("stress_shift", 0.0)
    m = data.get("stress_hill_exponent") or 3
    # Name the figure after the JSON file, not the stage set: the filename now
    # carries K (and m), so sweeps at different gate shapes cannot collide.
    tag = os.path.basename(path)[len("psigma_scores_"):-len(".json")]

    total = np.array([r["objective"] for r in rows])
    # score1+score2 summed over stages == objective without the SC-ablation term
    partial = np.array([sum(r[s]["score1"] + r[s]["score2"] for s in stages)
                        for r in rows])

    fig = plt.figure(figsize=(5.5 * max(len(stages), 2), 8.0))
    gs = fig.add_gridspec(2, len(stages) if len(stages) > 1 else 2,
                          hspace=0.32, wspace=0.24)

    # --- top: combined objective, with and without score3 --------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(psigmas, total, "-o", color="k", lw=2, ms=7,
            label="total objective (score1+2+3)")
    ax.plot(psigmas, partial, "--^", color="#762a83", lw=2, ms=7,
            label="score1 + score2 only (drops the term psigma targets)")
    for y, c in ((total, "k"), (partial, "#762a83")):
        if len(y) > 1:
            i = int(np.argmin(y))
            ax.plot(psigmas[i], y[i], "*", color=c, ms=18, zorder=5)
            ax.annotate(r"best $p_\sigma$=%g" % psigmas[i],
                        (psigmas[i], y[i]), textcoords="offset points",
                        xytext=(6, -13), fontsize=8, color=c)
    if len(psigmas) > 1:
        ax.axhline(total[0], color="grey", ls=":", lw=1)
        ax.annotate(r"$p_\sigma$=0 baseline", (psigmas.max(), total[0]),
                    textcoords="offset points", xytext=(-80, 4),
                    fontsize=7, color="grey")
    _decorate(ax, psigmas, "differentiation objective, %s   (K=%.3f, m=%d)"
              % (" + ".join(stages), K, m), "objective  (lower = better)")

    # --- bottom: individual terms, one panel per stage ------------------------
    for j, stage in enumerate(stages):
        ax = fig.add_subplot(gs[1, j])
        for key in SCORE_KEYS:
            y = np.array([r[stage][key] for r in rows])
            ax.plot(psigmas, y, "-", marker=STAGE_MARKERS.get(stage, "o"),
                    color=SCORE_COLOURS[key], lw=1.8, ms=6,
                    label=SCORE_LABELS[key])
        y = np.array([r[stage]["total"] for r in rows])
        ax.plot(psigmas, y, "-", color="k", lw=1.2, alpha=0.55,
                label="%s total" % stage)
        _decorate(ax, psigmas, "%s - individual terms" % stage, "score")
    if len(stages) == 1:
        fig.add_subplot(gs[1, 1]).axis("off")

    if len(psigmas) == 1:
        fig.text(0.5, 0.02, "only one $p_\\sigma$ point in this file - "
                            "sweep was not run / not finished",
                 ha="center", fontsize=9, color="#b2182b")

    out = os.path.join(RESULTS_DIR, "psigma_scores_%s.png" % tag)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== %s  (K=%.3f, m=%d, %d points) ===" % (tag, K, m, len(psigmas)))
    hdr = "  %-8s" % "psigma"
    for stage in stages:
        hdr += "|  %-9s %-9s %-9s %-9s " % tuple("%s.%s" % (stage[:4], k)
                                                 for k in SCORE_KEYS + ("tot",))
    print(hdr + "|  %-10s %-10s" % ("TOTAL", "s1+s2"))
    for p, r, t, q in zip(psigmas, rows, total, partial):
        line = "  %-8.4f" % p
        for stage in stages:
            line += "|  %-9.3f %-9.3f %-9.3f %-9.3f " % (
                r[stage]["score1"], r[stage]["score2"],
                r[stage]["score3"], r[stage]["total"])
        print(line + "|  %-10.3f %-10.3f" % (t, q))
    print("  -> %s" % out)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = [os.path.join(RESULTS_DIR, "psigma_scores_%s.json" % a)
                 for a in sys.argv[1:]]
    else:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              "psigma_scores_*.json")))
    for path in paths:
        if os.path.isfile(path):
            plot_file(path)
        else:
            print("missing: %s" % path)
