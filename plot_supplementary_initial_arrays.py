"""The three supplementary items that document the initial-array selection.

    python plot_supplementary_initial_arrays.py
    python plot_supplementary_initial_arrays.py --seed 7 --linear-y

  1. supp_table_initial_arrays.csv          rank, stage, cells, score
  2. supp_fig_score_vs_time.png/.svg        score against time, one array per
                                            stage, best frame starred
  3. supp_fig_neighbour_hist_<stage>.*      the ensemble's neighbour-count
                                            distribution beside the experiment

EACH IS WRITTEN TWICE, labelled and BARE, as the other supplement scripts do.
The bare figures go through ``plot_mechanical_fit_supplement.strip`` -- the same
helper the mechanical-fit supplement uses, so "bare" means one thing in this
project -- which leaves the data, the frame and the tick MARKS and takes every
word off: no axis labels, no tick labels, no legend, no annotation, no grid.
They are for setting the type by hand. The bare table is the four requested
columns alone, without the identifier columns the labelled one carries.

NOTHING IS RE-SCORED HERE. Every number comes out of the per-array
``z2_vs_time_mergedtails.npz`` traces that the selection itself wrote, read
through the selection's own functions -- ``_z2_trace_path`` to find them and
``ensemble_neighbor_z2`` to pool them. A figure script that recomputed the
score would be reporting a second opinion, and the bins are the one thing most
easily got wrong: the production scheme is ``NEIGHBOR_BINS``, tails MERGED
(``<=4, 5, 6, 7, >=8``, five bins, four degrees of freedom), not the
single-count 3..9 buckets that the older ``plot_initial_array_neighbor_hist``
and ``plot_array_selection_score`` draw. Those two scripts predate the merged
scheme and the z^2 rule; they are left alone rather than retrofitted, because
the figures they make belong to the superseded p-value selection.

THE TWO SCORES THESE FIGURES SHOW ARE NOT THE SAME NUMBER, and the table and
the histograms deliberately report different ones:

  stage 1, per array  z^2 against the EXPERIMENTAL SEM alone. One frame is one
                      realisation, so the model contributes no spread. This is
                      what chose each array's frame, and it is the table's
                      "score" column and the curve in figure 2.
  stage 2, ensemble   the ten chosen frames treated as the model's replicates,
                      so BOTH sides carry a standard error. This is the
                      histograms' annotation, and it is the number to quote for
                      how well the arrays match the experiment -- the per-array
                      scores cannot be averaged into it.

A caveat that belongs in the caption, not just here: the arrays were SELECTED
by minimising the stage-1 score, so the ensemble they form beats a random ten
by construction. The selection-proof figure is the greedy floor (about 8 at
P0), which the P0 ensemble essentially reaches.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import (RESULTS_DIR, NEIGHBOR_BINS, _neighbor_trace_tag,
                             _z2_trace_path, initial_morphology_name,
                             chosen_array_run_name, ensemble_neighbor_z2,
                             _STAGE_COLOUR, _SELECTED_COLOUR, _POINT_MS)
from plot_neighbor_pairs import STYLE, STYLE_LABEL
# the project's own definition of "bare", not a second one
from plot_mechanical_fit_supplement import strip

STAGES = ("E17.5", "P0")
RANKS = range(10)
# The array shown in figure 2, per stage. A fixed default rather than a random
# draw, so re-running reproduces the manuscript's figure; --random redraws.
FIGURE_ARRAYS = {"E17.5": 5, "P0": 5}
# Figure 2's x limit. The two stages' arrays were not all built to the same
# length (pool arrays 0-217 ran to t=25, the rest to 15), so the panel is
# clipped to the shorter one; see fig_score_vs_time.
X_MAX = 15.0
SEED_FOLDER = "initial arrays seed data"
STAR_MS = 20.0            # a star reads smaller than a disc of the same size
# the greedy floor quoted in the caveat above, per stage: the best joint score
# reachable by picking ten arrays, which is what the ensemble should be
# compared against rather than against zero
GREEDY_FLOOR = {"E17.5": None, "P0": 8.0}


def load_trace(rank, stage, results_dir=RESULTS_DIR):
    """One array's stored score trace, plus the frame the selection chose.

    ``chosen`` is located by TIME against the recorded selection rather than by
    re-taking the argmin, so the table reports the frame that was really
    extracted. The two are then compared, and any disagreement is returned for
    the caller to print -- a selection that moved is worth seeing, not papering
    over.
    """
    run = chosen_array_run_name(rank, stage)
    cache = _z2_trace_path(run, _neighbor_trace_tag()) + ".npz"
    if not os.path.isfile(cache):
        raise SystemExit("no stored trace for %s:\n  %s\nRun the selection "
                         "(find_best_timepoint_according_to_n_neighbors) first."
                         % (run, cache))
    with np.load(cache) as z:
        out = {"time": z["time"], "z2": z["z2_%s" % stage],
               "fractions": z["fractions"], "n_cells": z["n_cells"],
               "labels": [str(s) for s in z["bin_labels"]]}
    out["run"] = run
    out["extract"] = initial_morphology_name(rank, stage)
    out["argmin"] = int(np.argmin(out["z2"]))
    return out


def recorded_selection(results_dir=RESULTS_DIR):
    """chosen_arrays.json: what the selection actually extracted, by rank."""
    path = os.path.join(results_dir, SEED_FOLDER, "chosen_arrays.json")
    with open(path) as fh:
        doc = json.load(fh)
    return {(stage, r["rank"]): r for stage, rows in doc.items() for r in rows}


def build_table(results_dir=RESULTS_DIR):
    rows, notes = [], []
    rec = recorded_selection(results_dir)
    for stage in STAGES:
        for rank in RANKS:
            tr = load_trace(rank, stage, results_dir)
            r = rec[(stage, rank)]
            k = int(np.argmin(np.abs(tr["time"] - r["time_point"])))
            if k != tr["argmin"]:
                notes.append("%s rank %d: the recorded frame (t=%.3f) is not "
                             "the trace minimum (t=%.3f)"
                             % (stage, rank, tr["time"][k],
                                tr["time"][tr["argmin"]]))
            if not np.isclose(tr["z2"][k], r["stage1_z2"], rtol=1e-6):
                notes.append("%s rank %d: trace says z2=%.6f at the recorded "
                             "frame, chosen_arrays.json says %.6f"
                             % (stage, rank, tr["z2"][k], r["stage1_z2"]))
            rows.append({"rank": rank, "stage": stage,
                         "n_cells": int(tr["n_cells"][k]),
                         "score": float(tr["z2"][k]),
                         "frame_time": float(tr["time"][k]),
                         "extract": tr["extract"], "run": tr["run"],
                         "pool_array": r["source_index"]})
    return rows, notes


def write_table(rows, results_dir=RESULTS_DIR):
    # The four columns asked for come first; the rest identify the array, which
    # a supplementary table needs if anyone is to find it again. The BARE table
    # is those four alone.
    cols = ["rank", "stage", "n_cells", "score", "frame_time", "extract",
            "run", "pool_array"]
    bare_cols = ["rank", "stage", "n_cells", "score"]
    path = os.path.join(results_dir, "supp_table_initial_arrays.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    bare_path = os.path.join(results_dir, "supp_table_initial_arrays_bare.csv")
    with open(bare_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=bare_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("\n  %-5s %-7s %7s %12s %10s  %s"
          % ("rank", "stage", "cells", "score", "frame t", "extract"))
    for r in rows:
        print("  %-5d %-7s %7d %12.3f %10.3f  %s"
              % (r["rank"], r["stage"], r["n_cells"], r["score"],
                 r["frame_time"], r["extract"]))
    for stage in STAGES:
        c = [r["n_cells"] for r in rows if r["stage"] == stage]
        s = [r["score"] for r in rows if r["stage"] == stage]
        print("  %-7s cells %d-%d (mean %.0f), score %.2f-%.2f"
              % (stage, min(c), max(c), np.mean(c), min(s), max(s)))
    print("\nwrote %s" % path)
    print("wrote %s  (the four columns alone)" % bare_path)
    return path


def fig_score_vs_time(picked, log_y=True, results_dir=RESULTS_DIR, out=None,
                      bare=False, x_max=X_MAX):
    """Score against simulation time, one array per stage on one axes.

    ``x_max`` CLIPS THE VIEW, it does not truncate the data: a curve that runs
    past it is drawn to the frame edge and simply continues out of sight. The
    default exists because the two stages' arrays were not built to the same
    length -- pool arrays 0-217 ran to t=25 and the rest to 15 -- so without it
    one curve stops in mid-panel and the asymmetry reads as a property of the
    tissue rather than of the build. Every chosen frame sits near t=10, well
    inside the window, so nothing the figure is about is cut off.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for stage in STAGES:
        rank, tr = picked[stage]
        k = tr["argmin"]
        ax.plot(tr["time"], tr["z2"], "-o", lw=1.6, ms=_POINT_MS,
                color=_STAGE_COLOUR[stage], zorder=3,
                label="%s, array %d  (%d cells)"
                      % (stage, rank, int(tr["n_cells"][k])))
        ax.axvline(tr["time"][k], color="k", ls="--", lw=1.1, alpha=0.75,
                   zorder=1)
        ax.plot([tr["time"][k]], [tr["z2"][k]], marker="*", ms=STAR_MS,
                color=_SELECTED_COLOUR[stage], markeredgecolor="white",
                markeredgewidth=0.9, linestyle="none", zorder=6,
                label="best frame: t = %.2f, $z^2$ = %.2f"
                      % (tr["time"][k], tr["z2"][k]))
    if log_y:
        ax.set_yscale("log")
    if x_max:
        ax.set_xlim(-0.02 * x_max, 1.02 * x_max)
    ax.set_xlabel("simulation time")
    ax.set_ylabel(r"$\sum_k z_k^2$ against the experiment  (lower is better)")
    if bare:
        # strip() after the labels are set, so the labelled and bare figures
        # differ in nothing but the words -- same axes, same limits, same
        # curves, so one can be traced onto the other
        strip(ax)
    else:
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=9, loc="best")
    fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.11)
    stem = os.path.splitext(out or os.path.join(
        results_dir, "supp_fig_score_vs_time"))[0] + ("_bare" if bare else "")
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("wrote %s.%s" % (stem, ext))
    plt.close(fig)
    return stem


def fig_neighbour_hist(stage, results_dir=RESULTS_DIR, out=None, bare=False,
                       res=None):
    """The ensemble's neighbour-count distribution beside the experiment.

    Both sides are means of independent realisations -- ten arrays, three
    movies -- so both get a standard error, which is exactly the pairing the
    stage-2 score uses. Computed by ``ensemble_neighbor_z2`` itself, so the
    bars and the annotated z^2 cannot disagree.
    """
    if res is None:
        runs = [chosen_array_run_name(r, stage) for r in RANKS]
        res = ensemble_neighbor_z2(runs, stage, verbose=True)
    labels = res["labels"]
    x = np.arange(len(labels), dtype=float)
    w = 0.38

    m_face, m_edge = STYLE[(True, stage)]
    e_face, e_edge = STYLE[(False, stage)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - w / 2, 100.0 * res["model_mean"], w,
           yerr=100.0 * res["model_sem"], capsize=3.5,
           facecolor=m_face, edgecolor=m_edge, linewidth=1.3,
           error_kw=dict(ecolor=m_edge, lw=1.1),
           label="%s  (n = %d arrays)" % (STYLE_LABEL[(True, stage)],
                                          res["n_runs"]))
    ax.bar(x + w / 2, 100.0 * res["exp_mean"], w,
           yerr=100.0 * res["exp_sem"], capsize=3.5,
           facecolor=e_face, edgecolor=e_edge, linewidth=1.3,
           error_kw=dict(ecolor=e_edge, lw=1.1),
           label="%s  (n = %d movies)" % (STYLE_LABEL[(False, stage)],
                                          res["n_exp"]))
    ax.set_xticks(x)
    # neighbor_bin_labels spells the open bins "<=4" / ">=8" for the console;
    # on a figure they should be the real symbols
    ax.set_xticklabels([l.replace("<=", r"$\leq$").replace(">=", r"$\geq$")
                        for l in labels])
    ax.set_xlabel("number of neighbours")
    ax.set_ylabel("cells  (%)")
    dof = len(labels) - 1
    if bare:
        strip(ax)
    else:
        note = "overall $z^2$ = %.2f on %d dof" % (res["z2"], dof)
        floor = GREEDY_FLOOR.get(stage)
        if floor is not None:
            note += "\n(selection floor $\\approx$ %.0f)" % floor
        ax.annotate(note, xy=(0.98, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9.5)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.22, axis="y")
    fig.subplots_adjust(left=0.105, right=0.98, top=0.97, bottom=0.12)

    tag = "E17" if stage == "E17.5" else "P0"
    stem = (os.path.splitext(out or os.path.join(
        results_dir, "supp_fig_neighbour_hist_%s" % tag))[0]
        + ("_bare" if bare else ""))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("wrote %s.%s" % (stem, ext))
    plt.close(fig)
    return stem, res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=int, nargs=2,
                    default=[FIGURE_ARRAYS["E17.5"], FIGURE_ARRAYS["P0"]],
                    metavar=("E17_RANK", "P0_RANK"),
                    help="the ranks shown in figure 2 (default: %d and %d)"
                         % (FIGURE_ARRAYS["E17.5"], FIGURE_ARRAYS["P0"]))
    ap.add_argument("--random", action="store_true",
                    help="draw figure 2's two arrays at random instead, with"
                         " --seed; the draw is printed so it is reproducible")
    ap.add_argument("--seed", type=int, default=20261108,
                    help="seed for --random")
    ap.add_argument("--x-max", dest="x_max", type=float, default=X_MAX,
                    help="clip figure 2's time axis here (default %g); 0 shows"
                         " each curve to its own end" % X_MAX)
    ap.add_argument("--linear-y", dest="linear_y", action="store_true",
                    help="linear y on figure 2; the default is log, because the"
                         " two stages' minima differ by a factor of ~50")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    a = ap.parse_args()

    print("bins: %s   (%d bins, %d dof)"
          % (NEIGHBOR_BINS, len(NEIGHBOR_BINS), len(NEIGHBOR_BINS) - 1))

    print("\n=== 1. table ===")
    rows, notes = build_table(a.results_dir)
    write_table(rows, a.results_dir)
    if notes:
        print("\n  DISAGREEMENT with chosen_arrays.json:")
        for n in notes:
            print("    " + n)
    else:
        print("  every row matches chosen_arrays.json (frame and score)")

    print("\n=== 2. score against time ===")
    if a.random:
        rng = np.random.default_rng(a.seed)
        chosen_ranks = {s: int(rng.integers(0, len(list(RANKS))))
                        for s in STAGES}
        print("  arrays drawn with seed %d: %s" % (a.seed, chosen_ranks))
    else:
        chosen_ranks = dict(zip(STAGES, a.arrays))
        print("  arrays: %s" % chosen_ranks)
    picked = {s: (chosen_ranks[s], load_trace(chosen_ranks[s], s, a.results_dir))
              for s in STAGES}
    for s in STAGES:
        rank, tr = picked[s]
        k = tr["argmin"]
        print("    %-6s array %d (%s): %d frames, t=%.2f..%.2f, best t=%.3f"
              " z2=%.3f, %d cells"
              % (s, rank, tr["run"], tr["time"].size, tr["time"][0],
                 tr["time"][-1], tr["time"][k], tr["z2"][k],
                 int(tr["n_cells"][k])))
    if a.x_max:
        past = {s: int(np.sum(tr["time"] > a.x_max))
                for s, (_r, tr) in picked.items()}
        print("    x axis clipped at t=%g; frames out of view: %s"
              % (a.x_max, past))
    for bare in (False, True):
        fig_score_vs_time(picked, log_y=not a.linear_y,
                          results_dir=a.results_dir, bare=bare,
                          x_max=a.x_max)

    print("\n=== 3. neighbour-count distributions ===")
    for stage in STAGES:
        # scored once, drawn twice: the labelled and bare panels must show the
        # same bars, not two independent calls that could diverge
        runs = [chosen_array_run_name(r, stage) for r in RANKS]
        res = ensemble_neighbor_z2(runs, stage, verbose=True)
        for bare in (False, True):
            fig_neighbour_hist(stage, a.results_dir, bare=bare, res=res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
