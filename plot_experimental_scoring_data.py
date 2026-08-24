"""Plot every experimental number the mechanical fit is scored against.

    python plot_experimental_scoring_data.py
    python plot_experimental_scoring_data.py --with-model    # overlay best fits

Three terms x two stages. Each panel shows, per stage:

  * every individual measurement, jittered and coloured BY EXPERIMENT;
  * each experiment's MEAN as a black diamond;
  * the grand mean (mean of those diamonds) as a line, with a band at +/- SEM.

The band is the point of the figure. The objective is z = (model - mean)/SEM
with SEM taken over EXPERIMENT means, so that band — not the spread of
individual cells — is the whole scale of the score. It explains why P0 scores an
order of magnitude worse than E17.5 on roundness despite a similar absolute
miss: P0's three experiments happen to agree to 0.5% of the mean while E17.5's
disagree by 4.3%, so the same discrepancy is divided by a 9x smaller number.

Note the terms are not sampled alike. Roundness pools hundreds of cells per
experiment; the ablation ratio has 2-8; shrinkage stores one value per ablation
(14 of them), so there its "experiment means" are single ablations.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import (RESULTS_DIR, load_experimental_results,
                             _finite_arrays, _MECHANICS_EXPERIMENTAL_TYPE)

STAGES = ("E17.5", "P0")
LABEL = {
    "roundness_ratio": ("HC / SC roundness ratio", "HC roundness / mean SC roundness"),
    "ablation_ratio": ("HC / SC area change after ablation",
                       "HC area change / mean SC area change"),
    "shrinkage": ("cut shrinkage", "linear shrinkage (%)"),
}
# Best-fit model means, backed out of the stored z as mean_exp + z*SEM_exp.
MODEL_JSON = {"E17.5": "grid_fit_mechanics_v2_E17.5.json",
              "P0": "p0_from_e17_stiffness.json"}


def stats_for(term, stage):
    """(list of per-experiment arrays, per-experiment means, grand mean, SEM)."""
    arrays = _finite_arrays(load_experimental_results(stage, _MECHANICS_EXPERIMENTAL_TYPE[term]))
    means = np.array([a.mean() for a in arrays], dtype=float)
    sem = float(means.std(ddof=1) / np.sqrt(means.size)) if means.size > 1 else float("nan")
    return arrays, means, float(means.mean()), sem


def best_model_means(stage):
    """{term: model mean} for the best point of that stage's fit, or {}."""
    import json
    path = os.path.join(RESULTS_DIR, MODEL_JSON[stage])
    if not os.path.isfile(path):
        return {}
    try:
        pts = json.load(open(path))["points"]
    except (OSError, ValueError, KeyError):
        return {}
    best = min(pts.values(), key=lambda v: v["objective"])
    out = {}
    for term, z in (best.get("z") or {}).items():
        _a, _m, mu, sem = stats_for(term, stage)
        if np.isfinite(z) and np.isfinite(sem):
            out[term] = mu + z * sem
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                  "experimental_scoring_data.png"))
    ap.add_argument("--with-model", dest="with_model", action="store_true",
                    help="overlay each stage's best-fit model mean as a dashed line")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    model = {s: best_model_means(s) for s in STAGES} if a.with_model else {}
    terms = list(_MECHANICS_EXPERIMENTAL_TYPE)
    fig, axes = plt.subplots(1, len(terms), figsize=(13.5, 4.8))

    print("%-16s %-7s %9s %9s %8s %9s   per-experiment means"
          % ("term", "stage", "mean", "SEM", "n_exp", "SEM/mean"))
    for ax, term in zip(np.atleast_1d(axes), terms):
        for xi, stage in enumerate(STAGES):
            arrays, means, mu, sem = stats_for(term, stage)
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(arrays)))
            for j, arr in enumerate(arrays):
                # jitter width shrinks with the number of experiments so the
                # per-experiment clusters stay visually separate
                off = (j - (len(arrays) - 1) / 2.0) * (0.5 / max(len(arrays), 1))
                x = xi + off + rng.normal(0, 0.018, arr.size)
                ax.plot(x, arr, ".", ms=3, alpha=0.35, color=colors[j], zorder=1)
                ax.plot(xi + off, means[j], "D", ms=7, color=colors[j],
                        mec="k", mew=1.0, zorder=3)
            if np.isfinite(sem):
                ax.axhspan(mu - sem, mu + sem, xmin=0.06 + 0.5 * xi, xmax=0.44 + 0.5 * xi,
                           color="0.35", alpha=0.25, zorder=0)
            ax.hlines(mu, xi - 0.42, xi + 0.42, color="k", lw=2, zorder=4)
            ax.annotate("%.3f\n$\\pm$%.3f" % (mu, sem), (xi + 0.44, mu),
                        va="center", ha="left", fontsize=8)
            if a.with_model and term in model.get(stage, {}):
                ax.hlines(model[stage][term], xi - 0.42, xi + 0.42, color="crimson",
                          lw=2, ls="--", zorder=4)
            print("%-16s %-7s %9.4f %9.4f %8d %8.1f%%   %s"
                  % (term, stage, mu, sem, means.size, 100 * sem / abs(mu),
                     np.array2string(np.round(means, 3), max_line_width=200)))
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES)
        ax.set_xlim(-0.6, len(STAGES) - 0.25)
        ax.set_title(LABEL[term][0], fontsize=10)
        ax.set_ylabel(LABEL[term][1], fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        if term != "shrinkage":
            ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=0)

    handles = [plt.Line2D([], [], marker=".", ls="", color="0.4", alpha=0.5,
                          label="individual measurements"),
               plt.Line2D([], [], marker="D", ls="", color="w", mec="k",
                          label="experiment mean"),
               plt.Line2D([], [], color="k", lw=2, label="grand mean"),
               plt.Rectangle((0, 0), 1, 1, color="0.35", alpha=0.25,
                             label="$\\pm$ SEM (the scoring scale)")]
    if a.with_model:
        handles.append(plt.Line2D([], [], color="crimson", lw=2, ls="--",
                                  label="best-fit model mean"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Experimental data behind the 3 mechanical-fit scoring terms", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
