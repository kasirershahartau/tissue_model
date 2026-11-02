"""How the initial-array time point is chosen: score against time, per stage.

    python plot_array_selection_score.py
    python plot_array_selection_score.py --run random_periodic_array99 --every 5

An initial array is built by letting a random tiling relax with divisions and no
differentiation, and then KEEPING ONE FRAME per stage — the one whose tissue
best matches that stage's experiment. This figure is that selection: the score
of every frame against each stage, and the minimum each stage picks.

WHAT IS COUNTED is the number of neighbours a cell has, from the sheet's own
contact matrix on the model side and from experimental_cell_mask plus the
segmentation's contact matrix on the experimental side, so both sides count a
neighbour the same way (see plot_initial_array_neighbors).

TWO SCORES, and the default is the one that can actually choose a frame.

  distribution  the share of cells with 3, 4, ... 9 neighbours is one bucket
                each, scored against the three movies, and the chi^2 values are
                summed. This is the manuscript's own scoring shape -- scores 1
                and 2 are sums of per-bucket chi^2 -- and it is the closest
                thing to the rule the arrays were really selected by, which was
                the p value of the whole distribution.
  mean          the mean number of neighbours per cell, one bucket. It is
                MONOTONE in time over this run and so has no interior minimum:
                the model never relaxes down to the experimental mean, and the
                best frame is simply the last. Kept because it is the quantity
                plot_initial_array_neighbors compares, not because it selects.

THE SCORE IS chi^2 = ((model - experiment) / SEM_experiment)^2 per bucket. A
single frame is ONE model number, so there is no model SEM to add: the
denominator is the spread over the three movies alone. That is the convention
the mechanical fit uses, where the model also enters as a single pooled value.
A bucket on which the three movies agree exactly has no spread and no defined
chi^2; those are counted and left out of the sum.

The original runs' p-value curves did not survive, so this is the selection as
it would be done today, on a freshly built array.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             _nsigma_and_chi2)
from plot_initial_array_neighbors import (STAGES, EXPERIMENTS,
                                          experimental_mean_neighbours)
from plot_neighbor_pairs import STYLE

RUN = "random_periodic_array99"
# the stage's own colour, as everywhere else: the model face of that stage
COLOUR = {s: STYLE[(True, s)][1] for s in STAGES}
# The chosen frame is marked in a colour that CONTRASTS with its curve, so the
# marker does not read as part of it; these are the two stages' experiment
# colours in plot_neighbor_pairs' STYLE table, against the model colours the
# curves use. The dashed line dropped to each marker stays black.
SELECTED_COLOUR = {"E17.5": "blue", "P0": "red"}
POINT_MS = 3.2            # a dot per scored frame, over the line
SELECTED_MS = 9.0         # ... and a bigger one where the minimum is
# neighbour counts outside this range are clipped into the end buckets; the
# original selection used the same ceiling of 9
BUCKETS = (3, 4, 5, 6, 7, 8, 9)


def frame_neighbour_counts(sheet):
    """Neighbours per cell in one retrieved frame, one entry per cell."""
    A = np.asarray(sheet.get_contact_matrix()) > 0
    np.fill_diagonal(A, False)
    return A.sum(axis=1).astype(float)


def bucket_shares(per_cell):
    """% of cells with each neighbour count in BUCKETS (ends absorb the tails)."""
    c = np.clip(np.asarray(per_cell, float), BUCKETS[0], BUCKETS[-1])
    return np.array([100.0 * np.mean(c == k) for k in BUCKETS])


def score_frame(per_cell, exp_shares, exp_means, quantity):
    """(chi^2, number of undefined buckets) for one frame against one stage."""
    if quantity == "mean":
        _z, chi2, _ms, _me = _nsigma_and_chi2([float(np.mean(per_cell))],
                                              exp_means)
        return (chi2 if np.isfinite(chi2) else np.nan), int(not np.isfinite(chi2))
    mine = bucket_shares(per_cell)
    total, undefined = 0.0, 0
    for b in range(len(BUCKETS)):
        _z, chi2, _ms, _me = _nsigma_and_chi2([mine[b]], exp_shares[:, b])
        if np.isfinite(chi2):
            total += chi2
        else:
            # the three movies agree exactly on this bucket: no spread, so no
            # scale to measure a difference against
            undefined += 1
    return total, undefined


def trace(run, every=1, results_dir=RESULTS_DIR):
    """(times, per-cell neighbour counts) over the frames of one array run."""
    history = load_history_file(run)
    times = list(get_time_points(history))
    print("  %s: %d frames, t = %.3f .. %.3f"
          % (run, len(times), times[0], times[-1]))
    times = times[::every]
    out_t, out_v = [], []
    for k, t in enumerate(times):
        sheet = history.retrieve(t)
        sheet.arrange_sheet_from_history()
        try:
            v = frame_neighbour_counts(sheet)
        except Exception as exc:                            # noqa: BLE001
            print("    t=%.3f skipped (%s: %s)" % (t, type(exc).__name__, exc))
            continue
        out_t.append(float(t))
        out_v.append(v)
        if k % 25 == 0:
            print("    t=%7.3f  %d cells, mean neighbours %.4f"
                  % (t, v.size, v.mean()), flush=True)
    return np.array(out_t), out_v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN, help="array run folder to score")
    ap.add_argument("--every", type=int, default=1,
                    help="score every Nth frame (default: all of them)")
    ap.add_argument("--quantity", choices=("distribution", "mean"),
                    default="distribution",
                    help="sum of per-bucket chi^2 over the neighbour-count"
                         " distribution (default), or the mean alone")
    ap.add_argument("--log-y", dest="log_y", action="store_true",
                    help="log y axis; the minima are deep")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t, v = trace(a.run, a.every)
    if not len(t):
        raise SystemExit("no scorable frame in %s" % a.run)

    exp_cells = {s: [experimental_mean_neighbours(s, e)[1] for e in EXPERIMENTS]
                 for s in STAGES}
    exp_means = {s: np.array([float(np.mean(c)) for c in exp_cells[s]])
                 for s in STAGES}
    exp_shares = {s: np.array([bucket_shares(c) for c in exp_cells[s]])
                  for s in STAGES}

    score, undef = {}, {}
    for s in STAGES:
        # one model value per frame: _nsigma_and_chi2 gives it a zero SEM, so
        # the denominator is the experimental spread alone
        pairs = [score_frame(c, exp_shares[s], exp_means[s], a.quantity)
                 for c in v]
        score[s] = np.array([p[0] for p in pairs], float)
        undef[s] = int(np.max([p[1] for p in pairs])) if pairs else 0

    print("\n  scoring: %s" % a.quantity)
    print("  %-6s %9s %9s %10s %10s %9s"
          % ("stage", "exp mean", "exp SEM", "best t", "best chi^2", "undefined"))
    best = {}
    for s in STAGES:
        k = int(np.nanargmin(score[s]))
        best[s] = (t[k], score[s][k], float(np.mean(v[k])))
        print("  %-6s %9.4f %9.4f %10.3f %10.4f %6d/%d"
              % (s, exp_means[s].mean(),
                 exp_means[s].std(ddof=1) / np.sqrt(exp_means[s].size),
                 t[k], score[s][k], undef[s],
                 len(BUCKETS) if a.quantity == "distribution" else 1))
        print("           model mean neighbours there: %.4f  (%d cells)"
              % (np.mean(v[k]), v[k].size))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for s in STAGES:
        ax.plot(t, score[s], "-o", lw=1.6, ms=POINT_MS, color=COLOUR[s],
                label=s, zorder=3)
        bt, bs, _bv = best[s]
        ax.axvline(bt, color="k", ls="--", lw=1.1, alpha=0.75, zorder=1)
        ax.plot([bt], [bs], marker="o", ms=SELECTED_MS,
                color=SELECTED_COLOUR[s], markeredgecolor="white",
                markeredgewidth=0.9, linestyle="none", zorder=6,
                label="%s: best frame, t = %.2f" % (s, bt))
    if a.log_y:
        ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("$\\chi^2$ against the experiment  (lower is better)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9, loc="best")
    fig.suptitle("Choosing the frame an initial array is taken from (%s)\n"
                 "%s, scored against each stage's movies"
                 % (a.run, "neighbour-count distribution, chi^2 summed over"
                           " buckets" if a.quantity == "distribution"
                    else "mean neighbours per cell"),
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.855, bottom=0.11)

    stem = os.path.splitext(a.out or os.path.join(
        RESULTS_DIR, "array_selection_score_%s%s"
        % (a.run, "_mean" if a.quantity == "mean" else "")))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))
    np.savez(stem + ".npz", time=t,
             mean_neighbours=np.array([float(np.mean(c)) for c in v]),
             **{"chi2_%s" % s: score[s] for s in STAGES})
    print("wrote %s.npz" % stem)


if __name__ == "__main__":
    main()
