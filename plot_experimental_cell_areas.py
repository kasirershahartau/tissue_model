"""Experimental cell areas at +24h: support cells against hair cells.

    python plot_experimental_cell_areas.py
    python plot_experimental_cell_areas.py --pixels --max-points 0

One violin per (stage, cell type), from the LAST recorded frame of each of the
three experiments per stage — the same +24h frames the roundness terms are
measured on (``_experimental_late_cells_info_path``), so this figure and those
numbers describe the same tissue.

WHO IS COUNTED: valid cells that are not edge cells. An edge cell is clipped by
the field of view, so its area is not the cell's area and would drag the
distribution down. NOTE this is neither of the two masks used elsewhere --
``load_experimental_results`` takes valid cells ALONE for the roundness terms
(roundness is a ratio, so clipping hurts it less), and ``experimental_cell_mask``
additionally drops the neighbours of invalid cells, which matters for counting
contacts but not for measuring one cell's own area.

THE SHAPE AND THE STATISTIC COME FROM DIFFERENT LEVELS, as everywhere else here.
The violin and the dots are individual CELLS, pooled over the three experiments
-- the distribution actually measured -- while the star and whisker are the mean
and SEM over the THREE EXPERIMENTS, which is the independent unit.
``--experiment-means`` adds the three per-experiment means as larger dots.

``--pvalue`` tests HC against SC within each stage with the HIERARCHICAL
two-sample comparison the rest of the manuscript uses (``calc_pval`` with
``continues=True``, since an area is continuous).

EVERY CELL IS AN OBSERVATION there, not a per-experiment mean. The lists handed
over are the cells grouped by experiment, and HierarchicalTwoSamplesCompare
flattens them into one row per cell carrying a ``replicate`` label, then fits
``mixedlm("measurement ~ stage", groups=replicate)`` -- an LMM with the
experiment as a random intercept. So the thousands of cells all count, while
the between-experiment variation is absorbed by the random effect instead of
being mistaken for evidence.

NOTE the library labels replicates per SIDE (``R0_k`` for one group, ``R1_k``
for the other), so experiment k contributes two unrelated random intercepts
rather than one shared by its own HCs and SCs. The pairing between a movie's HC
and SC populations is therefore not exploited; the test is the weaker for it,
not the more permissive.

It needs statistical_analysis (statsmodels, scikit_posthocs); where that does
not import the figure is drawn without it and the reason is printed.

AREAS ARE CONVERTED TO MICRONS SQUARED at PIXEL_UM per pixel (0.1, confirmed
against the tif at 10.06 px per micron); ``--pixels`` leaves them in px^2.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR, _experimental_late_cells_info_path
from compare_ablation_distance_to_experiment import PIXEL_UM
from plot_neighbor_pairs import (STYLE, POINT_SIZE, POINT_COLOUR, STAT_COLOUR,
                                 MEAN_MARKERSIZE, ERROR_CAPSIZE,
                                 ERROR_LINEWIDTH, VIOLIN_W, _mean_sem, _x)

STAGES = ("E17.5", "P0")
EXPERIMENTS = (1, 2, 3)
TYPES = (("SC", 0), ("HC", 1))      # left to right within a stage block
MAX_POINTS = 100                    # cells sampled per violin; 0 draws all


def hierarchical_pvalue(group_a, group_b):
    """HC against SC by the manuscript's own hierarchical test.

    Each argument is a list of per-experiment arrays of INDIVIDUAL CELLS.
    HierarchicalTwoSamplesCompare flattens them to one row per cell with a
    ``replicate`` label and fits an LMM with that label as a random intercept,
    so every cell is an observation and the experiment is the grouping -- see
    the module docstring for what the library's per-side labelling costs.

    Returns (p, None) or (None, reason).
    """
    try:
        from post_processing import calc_pval
    except ImportError as exc:
        return None, ("statistical_analysis did not import (%s); it needs "
                      "statsmodels and scikit_posthocs" % exc)
    try:
        # copied: calc_pval clips its inputs in place when maximal_n is given,
        # and there is no reason to hand it the arrays the figure is drawn from
        p = calc_pval([np.array(v, dtype=float) for v in group_a],
                      [np.array(v, dtype=float) for v in group_b],
                      maximal_n=None, continues=True)
    except ImportError as exc:
        return None, ("statistical_analysis did not import (%s); it needs "
                      "statsmodels and scikit_posthocs" % exc)
    except Exception as exc:                                # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)
    return float(p), None


def counted_areas(stage, experiment, pixels=False):
    """(SC areas, HC areas) of one +24h frame, valid non-edge cells only."""
    ci = pd.read_pickle(_experimental_late_cells_info_path(stage, experiment))
    valid = ci["valid"].to_numpy().astype(bool)
    edge = np.zeros(valid.shape, bool)
    if "edge_cell" in ci:
        edge = (pd.to_numeric(ci["edge_cell"], errors="coerce")
                .fillna(0).to_numpy() != 0)
    keep = valid & ~edge
    area = pd.to_numeric(ci["area"], errors="coerce").to_numpy(float)
    if not pixels:
        area = area * PIXEL_UM ** 2
    is_hc = ci["type"].to_numpy() == 1
    out = {}
    for name, code in TYPES:
        m = keep & (is_hc if code == 1 else ~is_hc)
        v = area[m]
        out[name] = v[np.isfinite(v)]
    return out


def gather(pixels=False):
    """One row per (stage, type): pooled cells, and the per-experiment means."""
    rows = []
    for s, stage in enumerate(STAGES):
        per_exp = {name: [] for name, _c in TYPES}
        pooled = {name: [] for name, _c in TYPES}
        n_cells = {name: [] for name, _c in TYPES}
        for e in EXPERIMENTS:
            got = counted_areas(stage, e, pixels)
            for name, _c in TYPES:
                per_exp[name].append(float(np.mean(got[name])))
                pooled[name].append(got[name])
                n_cells[name].append(got[name].size)
        for t, (name, _c) in enumerate(TYPES):
            units = np.asarray(per_exp[name], float)
            m, sem = _mean_sem(units)
            face, edge = STYLE[(False, stage)]
            rows.append(dict(index=len(TYPES) * s + t, stage=stage, cell=name,
                             face=face, edge=edge, mean=m, sem=sem,
                             per_unit=units,
                             per_cell=np.concatenate(pooled[name]),
                             # kept un-pooled for the hierarchical test
                             per_experiment=list(pooled[name]),
                             n_cells=list(n_cells[name])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pixels", action="store_true",
                    help="leave areas in px^2 instead of um^2")
    ap.add_argument("--max-points", dest="max_points", type=int,
                    default=MAX_POINTS,
                    help="cells sampled per violin, each drawn as one dot"
                         " (default %d; 0 draws every cell, which is ~8800"
                         " points and a heavy svg). The sample is for DRAWING"
                         " only -- the p value always uses every cell"
                         % MAX_POINTS)
    ap.add_argument("--experiment-means", dest="exp_means", action="store_true",
                    help="also mark the three per-experiment means")
    ap.add_argument("--pvalue", action="store_true",
                    help="hierarchical HC-vs-SC p value per stage: an LMM over"
                         " individual cells with the experiment as a random"
                         " intercept")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    unit = "px$^2$" if a.pixels else "$\\mu$m$^2$"
    data = gather(a.pixels)
    print("  areas at +24h, valid non-edge cells, in %s\n"
          % ("px^2" if a.pixels else "um^2"))
    print("  %-6s %-4s %10s %9s %8s %10s %10s"
          % ("stage", "cell", "mean", "SEM", "cells", "median", "per-exp"))
    for d in data:
        print("  %-6s %-4s %10.2f %9.2f %8d %10.2f   %s"
              % (d["stage"], d["cell"], d["mean"], d["sem"],
                 d["per_cell"].size, float(np.median(d["per_cell"])),
                 ", ".join("%.1f" % v for v in d["per_unit"])))
    for stage in STAGES:
        hc = [d for d in data if d["stage"] == stage and d["cell"] == "HC"][0]
        sc = [d for d in data if d["stage"] == stage and d["cell"] == "SC"][0]
        print("  %-6s HC/SC of the per-experiment means: %s  (mean %.3f)"
              % (stage, ", ".join("%.3f" % r for r in hc["per_unit"] / sc["per_unit"]),
                 float(np.mean(hc["per_unit"] / sc["per_unit"]))))

    pvals = {}
    if a.pvalue:
        said = False
        for stage in STAGES:
            hc = [d for d in data if d["stage"] == stage and d["cell"] == "HC"][0]
            sc = [d for d in data if d["stage"] == stage and d["cell"] == "SC"][0]
            p, why = hierarchical_pvalue(hc["per_experiment"],
                                         sc["per_experiment"])
            pvals[stage] = p
            print("  %-6s HC vs SC, LMM over %d cells in %d experiments:"
                  " p = %s"
                  % (stage, hc["per_cell"].size + sc["per_cell"].size,
                     len(EXPERIMENTS), "%.4g" % p if p is not None else "-"))
            if why and not said:
                print("     (no p value -- %s)" % why)
                said = True

    rng = np.random.default_rng(a.seed)
    for bare in (False, True):
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        for d in data:
            x = _x(d["index"], d["stage"])
            v = d["per_cell"]
            body = ax.violinplot([v], positions=[x], widths=VIOLIN_W,
                                 showmeans=False, showextrema=False,
                                 showmedians=False)["bodies"][0]
            body.set_facecolor(d["face"]); body.set_edgecolor(d["edge"])
            body.set_linewidth(1.6); body.set_alpha(1.0); body.set_zorder(2)
            # one dot per cell; a sample of them unless --max-points 0
            show = v
            if 0 < a.max_points < v.size:
                show = rng.choice(v, a.max_points, replace=False)
            if show.size:
                ax.scatter(x + rng.uniform(-VIOLIN_W * 0.30, VIOLIN_W * 0.30,
                                           size=show.size), show,
                           s=POINT_SIZE["model"] * 0.5, marker=".",
                           color=POINT_COLOUR, alpha=0.45, linewidths=0,
                           zorder=3)
            if a.exp_means:
                ax.scatter(np.full(d["per_unit"].size, x), d["per_unit"],
                           s=POINT_SIZE["exp"], marker=".", color=POINT_COLOUR,
                           alpha=0.9, linewidths=0, zorder=4)
            ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none",
                        ecolor=STAT_COLOUR, elinewidth=ERROR_LINEWIDTH,
                        capsize=ERROR_CAPSIZE, capthick=ERROR_LINEWIDTH,
                        zorder=5)
            ax.plot(x, d["mean"], marker="*", markersize=a.mean_markersize,
                    color=STAT_COLOUR, linestyle="none", zorder=6)

        ticks = [_x(d["index"], d["stage"]) for d in data]
        ax.set_xticks(ticks)
        ax.set_xlim(min(ticks) - 0.7, max(ticks) + 0.7)
        lo = min(float(np.min(d["per_cell"])) for d in data)
        hi = max(float(np.max(d["per_cell"])) for d in data)
        top = hi + (0.16 if pvals else 0.06) * (hi - lo)
        ax.set_ylim(max(0.0, lo - 0.04 * (hi - lo)), top)

        # a bracket spanning each stage's pair, with its p value over it
        for stage in STAGES:
            p = pvals.get(stage)
            if p is None:
                continue
            xs = [_x(d["index"], d["stage"]) for d in data
                  if d["stage"] == stage]
            y = hi + 0.045 * (hi - lo)
            tick = 0.018 * (hi - lo)
            ax.plot([min(xs), min(xs), max(xs), max(xs)],
                    [y - tick, y, y, y - tick], "-", color=STAT_COLOUR,
                    lw=1.2, zorder=6)
            if not bare:
                ax.annotate("p = %.3g" % p, xy=(np.mean(xs), y + tick * 0.5),
                            ha="center", va="bottom", fontsize=10,
                            color=STAT_COLOUR, zorder=6)

        if bare:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            fig.subplots_adjust(left=0.075, right=0.985, top=0.98, bottom=0.06)
        else:
            ax.set_xticklabels([d["cell"] for d in data], fontsize=11)
            ax.set_ylabel("cell area (%s)" % unit, fontsize=11.5)
            for stage in STAGES:
                xs = [_x(d["index"], d["stage"]) for d in data
                      if d["stage"] == stage]
                ax.annotate(stage, xy=(np.mean(xs), 1.02),
                            xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=12, fontweight="bold")
            fig.suptitle("Experimental cell area at +24h, valid non-edge cells"
                         "\nviolin = every counted cell, star = mean over the"
                         " %d experiments, whisker = SEM" % len(EXPERIMENTS),
                         fontsize=11.5, y=0.985, va="top", linespacing=1.5)
            fig.subplots_adjust(left=0.105, right=0.985, top=0.835, bottom=0.075)

        stem = os.path.join(a.out_dir,
                            "experimental_cell_areas" + ("_bare" if bare else ""))
        for ext in ("png", "svg"):
            fig.savefig("%s.%s" % (stem, ext), dpi=200)
        plt.close(fig)
        print("\n  wrote %s.png / .svg" % stem)


if __name__ == "__main__":
    main()
