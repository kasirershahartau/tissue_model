"""The three mechanical fit observables at the best-fitting parameters.

    python plot_mechanics_fit_violin.py
    python plot_mechanics_fit_violin.py --zero-baseline --out somewhere/fig

Three panels, one per term the mechanical fit was scored on, sharing an x axis of
model and experiment side by side for each stage. Styling follows
plot_neighbor_pairs.py: the star is the mean and the whisker the SEM, models are
violins, and an experiment is a bar unless it has at least VIOLIN_MIN_N repeats —
the two ratios rest on 2-3 movies, too few for a kernel density to mean anything,
while shrinkage has 14 cut discs and gets a violin like the models.

WHICH MODEL. The best-fitting parameter point of each stage, by total chi^2 over
the three terms, taken from mechanics_points.pkl. Points where a term could not
be measured are excluded first: their undefined terms sum to a total chi^2 of
zero, which would otherwise rank a point that produced no data as the best fit.
Each surviving point carries exactly one run per initial array, and those runs
are the violin.

THE AXES START AT THE DATA, NOT AT ZERO. Two of the three terms are HC-over-SC
ratios, where the meaningful reference is 1 (hair cells and support cells
behaving alike), not 0. --zero-baseline restores a zero origin.

A NOTE ON THE ERROR BARS. The fit's own z divides by the EXPERIMENTAL SEM alone,
because the model side enters as a single pooled mean. The model SEM drawn here
is therefore descriptive — it says how much the arrays disagreed — and is not
what the chi^2 was computed with.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_SIZE, POINT_COLOUR,
                                 MEAN_MARKERSIZE, _mean_sem, _draw,
                                 _x, _tight_limits)

# (term, model column in mechanics_runs, y label, y tick step) top to bottom
TERMS = [("roundness_ratio", "roundness_ratio_mean",
          "HC : SC roundness ratio", 0.05),
         ("ablation_ratio", "ablation_ratio_mean",
          "HC : SC area change ratio\nnear the ablation", 0.1),
         ("shrinkage", "shrinkage_pct",
          "linear shrinkage of a cut disc (%)", 5)]
# (stage, kind, label) left to right; _x adds BLOCK_GAP to the P0 pair
SOURCES = [("E17.5", "model", "model\nE17.5"),
           ("E17.5", "exp", "experiment\nE17.5"),
           ("P0", "model", "model\nP0"),
           ("P0", "exp", "experiment\nP0")]
# an experimental group this size or larger is drawn as a violin rather than a
# bar: below it the kernel density would be drawing a curve through noise
VIOLIN_MIN_N = 10


def best_point(points, stage):
    """The lowest total chi^2 among the points where every term was measured."""
    g = points[points["stage"] == stage]
    usable = g[(g["n_sheets"] > 0) & np.all(
        [np.isfinite(g["%s_chi2" % t]) for t, _c, _l, _s in TERMS], axis=0)]
    if not len(usable):
        raise SystemExit("no fully measured parameter point for %s" % stage)
    return usable.sort_values("total_chi2").iloc[0], len(g) - len(usable)


def runs_of(runs, stage, point):
    """The runs belonging to one parameter point.

    The point stores the two stiffness RATIOS and the free parameters; a run
    stores the absolute stiffnesses. Matching on all four pins the point down to
    its own runs — gammaSC and A0 alone are shared across scans.
    """
    m = ((runs["stage"] == stage)
         & np.isclose(runs["gammaSC"], point["gammaSC"])
         & np.isclose(runs["A0"], point["A0"], rtol=1e-6)
         & np.isclose(runs["alphaHC"] / runs["alphaSC"], point["R_alpha"],
                      rtol=1e-3)
         & np.isclose(runs["gammaHC"] / runs["gammaSC"], point["R_gamma"],
                      rtol=1e-3))
    return runs[m]


def _usable(runs, term):
    """Rows of ``runs`` whose value for this term means anything.

    In 32 of the 474 ablation runs in the table the post-ablation frame is
    identical to the pre-ablation one — every cell's area is unchanged, so every
    ratio is exactly 1 and the HC-over-SC ratio is a meaningless 1.0. The fit
    skipped the ablation term for those sheets (see mechanics_eval, "Skip ONLY
    the ablation term") while keeping their perfectly good roundness, and
    dropping them here reproduces the stored chi^2 to eight decimals. The base
    run is fine, so the filter applies to this ONE term, not to the run.
    """
    if term != "ablation_ratio":
        return np.ones(len(runs), bool)
    if "ablation_measured" in runs.columns:      # patched or rebuilt table
        return runs["ablation_measured"].astype(bool).to_numpy()
    return ~(np.isclose(runs["hc_area_change_mean"], 1.0)
             & np.isclose(runs["sc_area_change_mean"], 1.0))


def gather(results_dir=RESULTS_DIR):
    points = pd.read_pickle(os.path.join(results_dir, "mechanics_points.pkl"))
    runs = pd.read_pickle(os.path.join(results_dir, "mechanics_runs.pkl"))
    exp = pd.read_pickle(os.path.join(results_dir, "mechanics_experiment.pkl"))

    chosen = {}
    for stage in ("E17.5", "P0"):
        point, skipped = best_point(points, stage)
        chosen[stage] = (point, runs_of(runs, stage, point), skipped)

    out = []
    for i, (stage, kind, label) in enumerate(SOURCES):
        point, g, _skipped = chosen[stage]
        for term, col, _ylabel, _step in TERMS:
            if kind == "exp":
                v = exp[(exp["stage"] == stage)
                        & (exp["term"] == term)]["value"].to_numpy(float)
            else:
                v = g[_usable(g, term)][col].dropna().to_numpy(float)
            m, s = _mean_sem(v)
            face, edge = STYLE[(kind == "model", stage)]
            # The shrinkage experiment has 14 cut discs — enough for a density,
            # unlike the 2-3 movies behind the two ratios — so it is drawn like
            # a model, violin and small points.
            violin = kind == "model" or len(v) >= VIOLIN_MIN_N
            out.append(dict(index=i, stage=stage, kind=kind, label=label,
                            pair=term, face=face, edge=edge, mean=m, sem=s,
                            per_unit=v, per_run=v, n_units=len(v),
                            as_violin=violin,
                            point_size=POINT_SIZE["model" if violin else "exp"],
                            unit="movies/repeats" if kind == "exp" else "arrays",
                            n_runs=len(v)))
    return out, chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zero-baseline", action="store_true",
                    help="start each panel at 0 instead of at its data")
    ap.add_argument("--points", choices=("run", "array", "none"), default="run",
                    help="one point per run; for this figure a run IS an array")
    ap.add_argument("--exp-no-bar", dest="exp_no_bar", action="store_true",
                    help="draw the experiment as star + SEM + points only, with no bar behind them")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    data, chosen = gather()
    if a.exp_no_bar:
        # only the rows that would be BARS lose their shape; the shrinkage
        # experiment has 14 repeats and keeps its violin
        for d in data:
            if d["kind"] == "exp" and not d.get("as_violin", False):
                d["draw_shape"] = False

    for stage, (point, g, skipped) in chosen.items():
        print("  %s best point: R_alpha=%.3f  R_gamma=%.4f  gammaSC=%.4f  A0=%.4f"
              % (stage, point["R_alpha"], point["R_gamma"], point["gammaSC"],
                 point["A0"]))
        print("     total chi^2 %.3f = %s   (%d run(s); %d point(s) skipped as"
              " not fully measured)"
              % (point["total_chi2"],
                 " + ".join("%.3f" % point["%s_chi2" % t] for t, _c, _l, _s in TERMS),
                 len(g), skipped))
    print("\n  %-22s %-6s %-11s %8s %8s %5s"
          % ("term", "stage", "source", "mean", "SEM", "n"))
    for d in data:
        print("  %-22s %-6s %-11s %8.3f %8.3f %5d"
              % (d["pair"], d["stage"], d["kind"], d["mean"], d["sem"],
                 d["n_units"]))

    fig, axes = plt.subplots(len(TERMS), 1, sharex=True, figsize=(8.5, 9.5),
                             gridspec_kw=dict(hspace=0))
    for ax, (term, _col, ylabel, step) in zip(axes, TERMS):
        _draw(ax, data, term, a.points, np.random.default_rng(a.seed),
              mean_markersize=a.mean_markersize)
        ax.set_ylabel(ylabel, fontsize=10)
        # one tick per TERMS step, so the three panels are not each labelled at
        # whatever resolution matplotlib picked for their very different ranges
        ax.yaxis.set_major_locator(MultipleLocator(step))
        lo, hi = _tight_limits(data, term)
        ax.set_ylim(0 if a.zero_baseline else lo, hi)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False)

    axis = axes[-1]
    axis.set_xticks([_x(i, s[0]) for i, s in enumerate(SOURCES)])
    axis.set_xticklabels([s[2] for s in SOURCES], fontsize=9.5)
    axis.set_xlim(_x(0, SOURCES[0][0]) - 0.75,
                  _x(len(SOURCES) - 1, SOURCES[-1][0]) + 0.75)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in STYLE_LABEL]
    names = list(STYLE_LABEL.values())
    if a.points != "none":
        handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                                  color=POINT_COLOUR, alpha=0.8,
                                  markersize=np.sqrt(POINT_SIZE["model"]) * 1.8))
        names.append("one array / repeat")
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Mechanical fit at the best-fitting parameters"
                 "  (star = mean, whisker = SEM)", fontsize=11.5, y=0.985)
    fig.subplots_adjust(left=0.145, right=0.98, top=0.955, bottom=0.155)

    stem = os.path.splitext(a.out or os.path.join(RESULTS_DIR,
                                                  "mechanics_fit_violin"))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
