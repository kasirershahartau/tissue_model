"""One figure per scoring window: every score vs psigma, both stages.

    python plot_window_figures.py
    python plot_window_figures.py --windows 2 10 --outdir D:/Kasirer/results/figs

Reads the cache written by sweep_scoring_windows.py, so it needs no simulation
and no re-scoring - run it any time, including while the sweep is still going
(missing points are simply left out of the curves).

Each figure carries 10 lines: score1 / score2 / score3 / (s1+s2) / (s1+s2+s3),
for E17.5 (solid, circles) and P0 (dashed, squares). Colour encodes WHICH score,
line style encodes WHICH stage, so the eye can compare a term across stages or
the terms within a stage.

The y axis is log: the terms span score1 ~ 0.001 to score3 ~ 700, and on a
linear axis score3 flattens everything else into the baseline. Non-positive
values (an exactly-zero chi^2) cannot be drawn on a log axis and are clipped to
a floor, marked with an open symbol so they are not read as real values.

READ THE FIGURES WITHIN A WINDOW, NOT ACROSS THEM. A shorter window contains
fewer differentiation events, so its SEMs differ and chi^2 shrinks mechanically.
Comparing psigma inside one panel is meaningful; comparing panel to panel is not.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR

CACHE = "window_score_cache.json"
SERIES = [
    ("score1", "score1  (neighbour pairs at t0)", "#1b7837"),
    ("score2", "score2  (HC nb at differentiation)", "#2166ac"),
    ("score3", "score3  (% of initial SCs differentiating)", "#b2182b"),
    ("s12",    "score1 + score2", "#762a83"),
    ("total",  "score1 + score2 + score3", "#000000"),
]
STAGE_STYLE = {"E17.5": ("-", "o"), "P0": ("--", "s")}
FLOOR = 1e-3


def load(stress_shift):
    """cache -> {(stage, psigma, window): result}"""
    with open(os.path.join(RESULTS_DIR, CACHE)) as fh:
        raw = json.load(fh)
    out = {}
    for key, val in raw.items():
        stage, ps, ks, w = key.split("|")
        if abs(float(ks.split("=")[1]) - stress_shift) > 1e-9:
            continue
        out[(stage, float(ps.split("=")[1]), float(w.split("=")[1]))] = val["result"]
    return out


def value(res, name):
    if name == "s12":
        return res["score1"] + res["score2"]
    return res[name]


def make_figure(data, window, outdir, stress_shift, omit=(), suffix=None):
    stages = sorted({k[0] for k in data if k[2] == window})
    if not stages:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    n_lines = 0
    for stage in stages:
        ls, marker = STAGE_STYLE.get(stage, ("-", "o"))
        pss = sorted(p for (s, p, w) in data if s == stage and w == window)
        if not pss:
            continue
        for name, label, colour in [x for x in SERIES if x[0] not in omit]:
            y = np.array([value(data[(stage, p, window)], name) for p in pss], float)
            drawn = np.where(y > 0, y, FLOOR)
            ax.plot(pss, drawn, ls, color=colour, marker=marker, ms=6, lw=1.9,
                    label="%s - %s" % (stage, label))
            # mark clipped (non-positive) points so they aren't misread
            bad = y <= 0
            if bad.any():
                ax.plot(np.array(pss)[bad], drawn[bad], marker, color=colour,
                        ms=10, mfc="none", mew=1.6)
            n_lines += 1

    ax.set_xlabel(r"$p_\sigma$  (mechanosensitivity)")
    ax.set_ylabel(r"$\chi^2$  (lower = better)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.set_title("Scoring window = %g  (K = %.3f)\n"
                 "solid + circles = E17.5,  dashed + squares = P0"
                 % (window, stress_shift), fontsize=12)
    # With only the composite curves left there is room to breathe; with all
    # ten a small two-column legend is the only thing that fits.
    if n_lines <= 4:
        ax.legend(fontsize=10, ncol=1, framealpha=0.95, loc="upper left")
    else:
        ax.legend(fontsize=7, ncol=2, framealpha=0.92, loc="upper left")

    # mark the best psigma per stage on the trustworthy objective (s1+s2)
    notes = []
    for stage in stages:
        pss = sorted(p for (s, p, w) in data if s == stage and w == window)
        vals = [(p, value(data[(stage, p, window)], "s12")) for p in pss]
        if vals:
            best = min(vals, key=lambda kv: kv[1])
            ax.axvline(best[0], color=("#444444" if stage == "E17.5" else "#999999"),
                       ls=":", lw=1.2)
            notes.append("%s best (s1+s2): psigma=%.3f" % (stage, best[0]))
    if notes:
        ax.text(0.99, 0.02, "   |   ".join(notes), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="#333333")

    if suffix is None:
        suffix = "" if not omit else "_no_" + "_".join(sorted(omit))
    out = os.path.join(outdir, "psigma_scores_window_%g%s.png" % (window, suffix))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=float, nargs="+", default=None,
                    help="default: every window present in the cache")
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=-0.080)
    ap.add_argument("--omit", nargs="*", default=[],
                    choices=[x[0] for x in SERIES],
                    help="series to leave out, e.g. --omit score1. The composite "
                         "curves (s12, total) still INCLUDE the omitted term - only "
                         "its own line is dropped, which lets the y axis focus on "
                         "the informative range instead of being stretched by "
                         "score1's near-zero values.")
    ap.add_argument("--suffix", default=None,
                    help="override the auto filename suffix, e.g. --suffix _composites")
    ap.add_argument("--outdir", default=os.path.join(RESULTS_DIR, "window_figures"))
    a = ap.parse_args()

    data = load(a.stress_shift)
    if not data:
        raise SystemExit("no cached scores for K=%.3f" % a.stress_shift)
    windows = a.windows if a.windows else sorted({k[2] for k in data})
    os.makedirs(a.outdir, exist_ok=True)

    print("%d cached score(s) | %d window(s)" % (len(data), len(windows)))
    for w in windows:
        pts = [(s, p) for (s, p, ww) in data if ww == w]
        out = make_figure(data, w, a.outdir, a.stress_shift, tuple(a.omit), a.suffix)
        if out is None:
            print("  window %-5g no data" % w)
            continue
        stages = sorted({s for s, _ in pts})
        counts = "  ".join("%s:%d psigma" % (s, sum(1 for x, _ in pts if x == s))
                           for s in stages)
        print("  window %-5g %-34s -> %s" % (w, counts, os.path.basename(out)))
    print("\nfigures in %s" % a.outdir)


if __name__ == "__main__":
    main()
