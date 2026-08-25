"""Plot each fit measurement's n-sigma score against each fitting parameter.

Four panels — one per fitting parameter (gammaSC, gammaHC_ratio, alphaHC_ratio,
shape_index). In each panel the standardized discrepancy ``z = (model - exp)/SE``
of every measurement (HC/SC roundness, HC/SC area-change-after-ablation) is drawn
against that parameter, one COLOUR per measurement.

Note this is a MARGINAL projection of the Bayesian-optimization trace: each point
is one evaluation and the other three parameters vary across points, so read
trends, not exact 1-D curves. Degenerate evaluations (no usable model data -> z
is NaN) are skipped. The +-2 sigma band (a term "matched" within experimental
uncertainty) and z=0 are drawn for reference.

    python plot_nsigma_vs_params.py                 # E17.5, saves a PNG next to the trace
    python plot_nsigma_vs_params.py --stage P0 --ylim -15 15
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_processing import RESULTS_DIR
from v1_bayesian_fit import load_mechanical_optimization_trace

PARAMS = ["gammaSC", "gammaHC_ratio", "alphaHC_ratio", "shape_index"]
# One colour per measurement (kept consistent across all four panels).
MEASURES = {
    "hc_roundness": ("HC roundness", "#2a78d6"),
    "sc_roundness": ("SC roundness", "#1baf7a"),
    "hc_ablation":  ("HC ablation",  "#eda100"),
    "sc_ablation":  ("SC ablation",  "#e0663b"),
}


def plot_nsigma_vs_params(stage="E17.5", results_dir=RESULTS_DIR,
                          connect=True, ylim=None, save_path=None, show=True):
    tr = load_mechanical_optimization_trace(stage, results_dir=results_dir)
    params = [p for p in PARAMS if p in tr.columns]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, param in zip(axes, params):
        x_all = tr[param].to_numpy(dtype=float)
        for key, (label, color) in MEASURES.items():
            col = "nsigma_" + key
            if col not in tr.columns:
                continue
            z = tr[col].to_numpy(dtype=float)
            good = np.isfinite(x_all) & np.isfinite(z)
            xg, zg = x_all[good], z[good]
            if xg.size == 0:
                continue
            if connect:                       # sort by the parameter to guide the eye
                order = np.argsort(xg)
                ax.plot(xg[order], zg[order], "-o", ms=4, lw=1.1,
                        color=color, alpha=0.85, label=label)
            else:
                ax.scatter(xg, zg, s=24, color=color, alpha=0.85, label=label)
        ax.axhspan(-2, 2, color="0.5", alpha=0.10, zorder=0)   # matched band
        ax.axhline(0, color="0.45", lw=0.9, zorder=1)
        ax.set_xlabel(param)
        ax.set_ylabel(r"n-sigma   $z=(\mathrm{model}-\mathrm{exp})/\mathrm{SE}$")
        ax.set_title(param)
        ax.grid(alpha=0.15)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].legend(title="measurement", fontsize=9, loc="best", framealpha=0.9)
    fig.suptitle("n-sigma vs fitting parameters — %s   "
                 "(±2σ band shaded; marginal projection, other params vary)"
                 % stage, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is None:
        save_path = os.path.join(results_dir, "%s_nsigma_vs_params.png" % stage)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    print("saved", save_path)
    if show:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="E17.5")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--scatter", action="store_true",
                    help="plot unconnected markers instead of sort-by-x lines")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="clip the y-axis (e.g. --ylim -15 15) to zoom the "
                         "informative band; fluid-regime outliers go off-scale")
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    plot_nsigma_vs_params(stage=args.stage, results_dir=args.results_dir,
                          connect=not args.scatter, ylim=args.ylim,
                          save_path=args.save_path, show=not args.no_show)
