"""Mean number of neighbours a cell has: the initial arrays against the experiment.

    python plot_initial_array_neighbors.py
    python plot_initial_array_neighbors.py --scored

The initial arrays are built by letting a random tiling relax with divisions and
no differentiation, then keeping the frame whose neighbour statistics best match
the experiment — one frame per stage. This figure checks what that selection
bought: the mean number of neighbours per cell in each kept frame, against the
same quantity in the first frame of each movie.

WHO IS COUNTED. On the model side every face of a periodic sheet, since there is
no boundary to exclude. On the experimental side experimental_cell_mask — valid,
not an edge cell, and not touching an invalid one — because a clipped or
half-unknown neighbourhood undercounts neighbours by construction. Both sides
count a neighbour the way the pair statistics do: a non-zero entry of the
contact matrix.

A POINT IS AN INITIAL ARRAY on the model side and A MOVIE on the experimental
one. The star and whisker are the mean and SEM over those, and the chi^2 is
(difference / combined SEM)^2, as everywhere else here.

MEAN NEIGHBOURS IS NOT 6. For an infinite planar tiling Euler's formula forces
exactly 6, and a periodic sheet has no boundary to escape it with — but a cell
pair that meets along two separate edges is one neighbour here and two in the
formula, so the measured mean sits slightly below 6 on both sides.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, experimental_results_folder,
                             experimental_cell_mask, _nsigma_and_chi2)
from run_model import load_sheet_from_file
from ablate_single_hc import BOX
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_SIZE, POINT_COLOUR,
                                 STAT_COLOUR, MEAN_MARKERSIZE, ERROR_CAPSIZE,
                                 ERROR_LINEWIDTH, BAR_W, BLOCK_GAP, _mean_sem,
                                 _x)

STAGES = ("E17.5", "P0")
SUFFIX = {"E17.5": "E17", "P0": "P0"}       # the folder name's stage tag
ARRAYS = range(10)                          # random_periodic_array0..9
EXPERIMENTS = (1, 2, 3)                     # ..._experiment<N>_..._frame_1


def model_mean_neighbours(folder, results_dir=RESULTS_DIR):
    """Mean neighbours per cell in one saved initial array.

    The contact matrix is the sheet's own, so a contact that wraps around the
    periodic box counts like any other — which is why this is not read off
    num_sides, a count of EDGES that includes every virtual vertex.
    """
    sheet = load_sheet_from_file(os.path.join(results_dir, folder),
                                 force_periodic_box=BOX)
    sheet.geom.update_all(sheet)
    A = np.asarray(sheet.get_contact_matrix()) > 0
    np.fill_diagonal(A, False)
    per_cell = A.sum(axis=1).astype(float)
    return float(per_cell.mean()), per_cell


def experimental_mean_neighbours(stage, experiment):
    """Mean neighbours per counted cell in the first frame of one movie."""
    prefix = "E17" if stage == "E17.5" else "P0"
    base = os.path.join(experimental_results_folder, stage)
    ci = pd.read_pickle(os.path.join(
        base, "%s_experiment%d_cells_info_frame_1" % (prefix, experiment)))
    cm = np.load(os.path.join(
        base, "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, experiment)))
    keep = experimental_cell_mask(ci, cm)
    A = np.asarray(cm)[np.ix_(keep, keep)] > 0
    np.fill_diagonal(A, False)
    per_cell = A.sum(axis=1).astype(float)
    return float(per_cell.mean()), per_cell


def gather(arrays=ARRAYS, results_dir=RESULTS_DIR):
    """One row per (stage, side), with its per-unit means."""
    out = []
    for i, stage in enumerate(STAGES):
        model, missing = [], []
        for k in arrays:
            folder = "random_periodic_array%d_for_%s" % (k, SUFFIX[stage])
            if not os.path.isdir(os.path.join(results_dir, folder)):
                missing.append(folder)
                continue
            model.append(model_mean_neighbours(folder, results_dir)[0])
        if missing:
            print("  missing: %s" % ", ".join(missing))
        exp = [experimental_mean_neighbours(stage, e)[0] for e in EXPERIMENTS]
        for kind, vals, unit in (("model", model, "arrays"),
                                 ("exp", exp, "movies")):
            m, s = _mean_sem(np.asarray(vals, float))
            face, edge = STYLE[(kind == "model", stage)]
            out.append(dict(index=2 * i + (kind == "exp"), stage=stage,
                            kind=kind, face=face, edge=edge, mean=m, sem=s,
                            per_unit=np.asarray(vals, float), unit=unit,
                            label="initial arrays" if kind == "model"
                                  else "experiment"))
    return out


def per_array(stage, arrays=ARRAYS, results_dir=RESULTS_DIR):
    """One row per saved array of a stage: its mean, and its own chi^2.

    SCORED ALONE, each against the three movies. One array is ONE model number,
    so there is no model SEM to add and the denominator is the experimental
    spread by itself -- which is NOT how the pooled figure scores, where ten
    arrays supply a (very small) SEM of their own. The two numbers are therefore
    not comparable, and the pooled one is the stricter test.
    """
    exp_vals = [experimental_mean_neighbours(stage, e)[0] for e in EXPERIMENTS]
    rows = []
    for k in arrays:
        folder = "random_periodic_array%d_for_%s" % (k, SUFFIX[stage])
        if not os.path.isdir(os.path.join(results_dir, folder)):
            continue
        m, per_cell = model_mean_neighbours(folder, results_dir)
        _z, chi2, _ms, _me = _nsigma_and_chi2([m], exp_vals)
        rows.append(dict(array=k, mean=m, n_cells=per_cell.size, z2=chi2,
                         folder=folder))
    return rows, np.asarray(exp_vals, float)


def draw_one(ax, row, exp_vals, stage, ylim):
    """One array against the experiment: two bars, the array's own chi^2 above."""
    m_face, m_edge = STYLE[(True, stage)]
    e_face, e_edge = STYLE[(False, stage)]
    e_mean, e_sem = _mean_sem(exp_vals)
    ax.bar(0, row["mean"], BAR_W, color=m_face, edgecolor=m_edge, linewidth=1.5,
           zorder=2)
    ax.bar(1, e_mean, BAR_W, color=e_face, edgecolor=e_edge, linewidth=1.5,
           zorder=2)
    ax.errorbar(1, e_mean, yerr=e_sem, fmt="none", ecolor=STAT_COLOUR,
                elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)
    ax.scatter(np.full(exp_vals.size, 1.0), exp_vals, s=POINT_SIZE["exp"],
               marker=".", color=POINT_COLOUR, alpha=0.8, linewidths=0, zorder=3)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.65, 1.65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["array\n%d" % row["array"], "exp"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    top = max(row["mean"], e_mean + e_sem, float(np.max(exp_vals)))
    ax.annotate("%.3g" % row["z2"], xy=(0.5, top + 0.04 * (ylim[1] - ylim[0])),
                ha="center", va="bottom", fontsize=9, color=STAT_COLOUR,
                zorder=6)


def draw_per_array(stage, rows, exp_vals, out_dir, individual=False):
    """A grid of one panel per array, and optionally one file per array too."""
    e_mean, e_sem = _mean_sem(exp_vals)
    lo = min([r["mean"] for r in rows] + [e_mean - e_sem, float(exp_vals.min())])
    hi = max([r["mean"] for r in rows] + [e_mean + e_sem, float(exp_vals.max())])
    pad = 0.25 * (hi - lo)
    ylim = (lo - pad, hi + 0.75 * (hi - lo))        # room for the chi^2 label

    ncol = 5
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.6 * nrow),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        draw_one(ax, row, exp_vals, stage, ylim)
    for ax in axes[len(rows):]:
        ax.axis("off")
    for k, ax in enumerate(axes[:len(rows)]):
        if k % ncol == 0:
            ax.set_ylabel("mean neighbours", fontsize=9)
    fig.suptitle("%s: each initial array against the experiment, scored alone"
                 "   (number = $\\chi^2$)" % stage, fontsize=11.5, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.11,
                        wspace=0.12, hspace=0.42)
    stem = os.path.join(out_dir, "initial_array_neighbors_per_array_%s"
                        % stage.replace(".", ""))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    print("  wrote %s.png / .svg" % stem)

    if individual:
        for row in rows:
            f, ax = plt.subplots(figsize=(2.6, 3.4))
            draw_one(ax, row, exp_vals, stage, ylim)
            ax.set_ylabel("mean neighbours", fontsize=9)
            f.suptitle("%s, array %d" % (stage, row["array"]), fontsize=10)
            f.subplots_adjust(left=0.3, right=0.96, top=0.9, bottom=0.14)
            one = os.path.join(out_dir, "initial_array_neighbors_%s_array%d"
                               % (stage.replace(".", ""), row["array"]))
            for ext in ("png", "svg"):
                f.savefig("%s.%s" % (one, ext), dpi=200)
            plt.close(f)
        print("  wrote %d individual figures" % len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-array", dest="per_array", action="store_true",
                    help="one scored bar plot per saved array instead of the"
                         " pooled figure")
    ap.add_argument("--individual", action="store_true",
                    help="with --per-array, also write one file per array")
    ap.add_argument("--scored", action="store_true",
                    help="print the model's chi^2 against the experiment over"
                         " its bar")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    if a.per_array:
        for stage in STAGES:
            rows, exp_vals = per_array(stage)
            e_mean, e_sem = _mean_sem(exp_vals)
            print("\n  %s: experiment %.4f +- %.4f over %d movies"
                  % (stage, e_mean, e_sem, exp_vals.size))
            print("  %-7s %9s %7s %10s" % ("array", "mean", "cells", "chi^2"))
            for r in rows:
                print("  %-7d %9.4f %7d %10.3f"
                      % (r["array"], r["mean"], r["n_cells"], r["z2"]))
            draw_per_array(stage, rows, exp_vals, RESULTS_DIR, a.individual)
        return

    data = gather()
    targets = {d["stage"]: d["per_unit"] for d in data if d["kind"] == "exp"}
    print("\n  mean number of neighbours per cell\n")
    print("  %-6s %-16s %8s %8s %7s %9s %9s"
          % ("stage", "source", "mean", "SEM", "units", "n-sigma", "chi^2"))
    for d in data:
        if d["kind"] == "exp":
            print("  %-6s %-16s %8.3f %8.3f %3d %-6s"
                  % (d["stage"], d["label"], d["mean"], d["sem"],
                     len(d["per_unit"]), d["unit"]))
            continue
        z, chi2, _ms, _me = _nsigma_and_chi2(d["per_unit"], targets[d["stage"]])
        d["z2"] = chi2
        print("  %-6s %-16s %8.3f %8.3f %3d %-6s %9.3f %9.3f"
              % (d["stage"], d["label"], d["mean"], d["sem"],
                 len(d["per_unit"]), d["unit"], z, chi2))

    rng = np.random.default_rng(a.seed)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for d in data:
        x = _x(d["index"], d["stage"])
        ax.bar(x, d["mean"], BAR_W, color=d["face"], edgecolor=d["edge"],
               linewidth=1.6, zorder=2)
        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=4)
        v = d["per_unit"]
        jit = rng.uniform(-BAR_W * 0.28, BAR_W * 0.28, size=len(v))
        ax.scatter(x + jit, v,
                   s=POINT_SIZE["model" if d["kind"] == "model" else "exp"],
                   marker=".", color=POINT_COLOUR, alpha=0.8, linewidths=0,
                   zorder=3)

    # the bars differ by ~1%, so a zero baseline would show four identical
    # columns; the axis starts just under the lowest thing drawn instead, which
    # truncates them — only the top edge is readable then
    lows = [float(np.min(d["per_unit"])) for d in data]
    lows += [d["mean"] - (d["sem"] or 0.0) for d in data]
    highs = [float(np.max(d["per_unit"])) for d in data]
    highs += [d["mean"] + (d["sem"] or 0.0) for d in data]
    lo, hi = min(lows), max(highs)
    pad = 0.25 * (hi - lo)
    ax.set_ylim(lo - pad, hi + (0.55 if a.scored else 0.3) * (hi - lo))
    ax.set_ylabel("mean neighbours per cell", fontsize=11)

    if a.scored:
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for d in data:
            if "z2" not in d or not np.isfinite(d["z2"]):
                continue
            top = max(d["mean"] + (d["sem"] or 0.0), float(np.max(d["per_unit"])))
            ax.annotate("%.3g" % d["z2"],
                        xy=(_x(d["index"], d["stage"]), top + 0.03 * span),
                        ha="center", va="bottom", fontsize=9.5,
                        color=STAT_COLOUR, zorder=6)

    ticks = [_x(d["index"], d["stage"]) for d in data]
    ax.set_xticks(ticks)
    ax.set_xticklabels([d["label"] for d in data], fontsize=10)
    ax.set_xlim(min(ticks) - 0.7, max(ticks) + 0.7)
    for stage in STAGES:
        xs = [_x(d["index"], d["stage"]) for d in data if d["stage"] == stage]
        ax.annotate(stage, xy=(np.mean(xs), 1.02),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=12, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in STYLE_LABEL]
    names = [STYLE_LABEL[k].replace("model", "initial array")
             for k in STYLE_LABEL]
    for kind, name in (("model", "one initial array"), ("exp", "one movie")):
        handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                                  color=POINT_COLOUR, alpha=0.8,
                                  markersize=np.sqrt(POINT_SIZE[kind]) * 1.8))
        names.append(name)
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Neighbours per cell in the selected initial arrays\n"
                 "bar = mean over arrays / movies, whisker = SEM",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.855, bottom=0.235)

    stem = os.path.splitext(a.out or os.path.join(
        RESULTS_DIR, "initial_array_neighbors%s"
        % ("_scored" if a.scored else "")))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
