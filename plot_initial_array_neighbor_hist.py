"""Neighbour-count distributions: each initial array beside the experiment.

    python plot_initial_array_neighbor_hist.py
    python plot_initial_array_neighbor_hist.py --individual

One panel per saved initial array, per stage: the share of cells with 3, 4, ...
9 neighbours, drawn as interleaved bars — the array in the stage's model colour,
the experiment beside it in the experimental one. This is the comparison the
arrays were actually SELECTED on: the frame kept for a stage was the one whose
whole neighbour-count distribution best matched that stage's movies, so the
distribution, not its mean, is what a reader should see.

COUNTS ARE CLIPPED INTO 3..9, the ceiling the original selection used
(max_number_of_neighbors=9). The end buckets therefore absorb the tails.

WHO IS COUNTED. Every face of a periodic sheet on the model side, since there is
no boundary. On the experimental side there are two populations, and they are
not the same tissue:

  default       experimental_cell_mask — valid, not an edge cell, not touching
                an invalid one. The manuscript's corrected population.
  --legacy-mask every valid cell, edge cells included, straight from
                load_experimental_results. This is what the ORIGINAL selection
                saw, so it is the population its p values belong to. An edge
                cell is clipped by the field of view, so its neighbourhood is
                truncated and this distribution sits lower.

Figures built with --legacy-mask are written under their own "_legacy" names, so
the two can exist side by side.

WHAT IS ANNOTATED. Two numbers, and they answer different questions.

  chi^2  summed over the seven buckets. On a per-array panel each bucket's
         model value is scored against the three movies with the experimental
         SEM as the only denominator, since one array is a single number; on
         the --combined panel both spreads enter, the arrays' and the movies'.
  p      the hierarchical two-sample test the arrays were ORIGINALLY selected
         with -- calc_pval(dist1, dist2, maximal_n=9, continues=False), the
         call find_best_timepoint_according_to_n_neighbors made. Each argument
         is a list of samples, one per repeat: a per-array panel passes that
         one array, and --combined passes all ten, so the arrays become the
         model group's repeats against the three movies.

The p value needs statistical_analysis (statsmodels, scikit_posthocs). Where
that does not import the panels carry chi^2 alone and the reason is printed
once; --no-pvalue skips the test deliberately. To reproduce the p values the
selection was really run with, use --legacy-mask as well: that is the
population load_experimental_results hands the test.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import (RESULTS_DIR, _nsigma_and_chi2,
                             load_experimental_results)
from run_model import load_sheet_from_file
from ablate_single_hc import BOX
from plot_initial_array_neighbors import (STAGES, SUFFIX, ARRAYS, EXPERIMENTS,
                                          experimental_mean_neighbours)
from plot_array_selection_score import BUCKETS, bucket_shares
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_COLOUR, POINT_SIZE,
                                 STAT_COLOUR, _mean_sem)

BAR_W = 0.38                    # two bars per bucket, just clear of each other
MAX_N = BUCKETS[-1]             # the original selection's ceiling


def experimental_counts(stage, legacy=False):
    """Neighbours per counted cell, one array per movie.

    ``legacy`` takes load_experimental_results' own population — every valid
    cell, edge cells included — which is what the original p-value selection
    was run on. Otherwise the corrected mask, via experimental_mean_neighbours.
    """
    if legacy:
        return [np.asarray(v, float) for v in
                load_experimental_results(stage, "number of neighbors")]
    return [experimental_mean_neighbours(stage, e)[1] for e in EXPERIMENTS]


def model_counts(folder, results_dir=RESULTS_DIR):
    """Neighbours per cell in one saved array, from the sheet's contact matrix."""
    sheet = load_sheet_from_file(os.path.join(results_dir, folder),
                                 force_periodic_box=BOX)
    sheet.geom.update_all(sheet)
    A = np.asarray(sheet.get_contact_matrix()) > 0
    np.fill_diagonal(A, False)
    return A.sum(axis=1).astype(float)


def pvalue(model_samples, experiments):
    """The selection's own hierarchical p value, exactly as it was computed.

    ``calc_pval(dist1, dist2, maximal_n=9, continues=False)`` is the call
    find_best_timepoint_according_to_n_neighbors made through
    compare_to_experimental_results, and each argument is a LIST OF SAMPLES --
    one per repeat of that group. The original had a single model run, so it
    passed ``[model_results]``; pass one entry per array here and the same
    hierarchical test treats the arrays as that group's repeats, which is what
    the combined figure averages over.

    EVERYTHING IS COPIED FIRST because calc_pval clips IN PLACE
    (``np.clip(..., out=dist1[i])``), so handing it the arrays the figure is
    drawn from would silently rewrite them.

    Returns (p, None) or (None, reason).
    """
    try:
        from post_processing import calc_pval
    except ImportError as exc:
        return None, ("statistical_analysis did not import (%s); it needs "
                      "statsmodels and scikit_posthocs" % exc)
    try:
        p = calc_pval([np.array(m, dtype=float) for m in model_samples],
                      [np.array(e, dtype=float) for e in experiments],
                      maximal_n=MAX_N, continues=False)
    except ImportError as exc:                              # raised inside
        return None, ("statistical_analysis did not import (%s); it needs "
                      "statsmodels and scikit_posthocs" % exc)
    except Exception as exc:                                # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)
    return float(p), None


def scores(model, experiments):
    """(chi^2 summed over buckets, per-bucket model %, per-movie exp %)."""
    mine = bucket_shares(model)
    theirs = np.array([bucket_shares(e) for e in experiments])
    total = 0.0
    for b in range(len(BUCKETS)):
        _z, chi2, _ms, _me = _nsigma_and_chi2([mine[b]], theirs[:, b])
        if np.isfinite(chi2):
            total += chi2
    return total, mine, theirs


def draw_combined(ax, stage, model_rows, theirs, chi2, ymax, pval=None,
                  annotate=False, label_inside=False):
    """Every array of a stage averaged into one histogram, beside the movies'.

    A BIN'S HEIGHT IS THE MEAN OVER ARRAYS and its whisker their SEM, so the
    array-to-array spread is what the model's error bar shows — the same
    hierarchy the scores use, where an array is the independent unit. The
    experiment's bar is the mean over movies with their SEM, unchanged.
    """
    m_face, m_edge = STYLE[(True, stage)]
    e_face, e_edge = STYLE[(False, stage)]
    x = np.arange(len(BUCKETS), dtype=float)
    for off, rows, face, edge in ((-BAR_W / 2, model_rows, m_face, m_edge),
                                  (+BAR_W / 2, theirs, e_face, e_edge)):
        mean = rows.mean(axis=0)
        sem = (rows.std(axis=0, ddof=1) / np.sqrt(rows.shape[0])
               if rows.shape[0] > 1 else np.zeros(rows.shape[1]))
        ax.bar(x + off, mean, BAR_W, color=face, edgecolor=edge, linewidth=1.4,
               zorder=2)
        ax.errorbar(x + off, mean, yerr=sem, fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=4)
        for row in rows:
            ax.scatter(x + off, row, s=POINT_SIZE["model"], marker=".",
                       color=POINT_COLOUR, alpha=0.75, linewidths=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUCKETS], fontsize=9.5)
    ax.set_ylim(0, ymax)
    # a little more than half a bar at each end, so the outermost bars clear
    # the frame the two panels share
    ax.set_xlim(-0.7, len(BUCKETS) - 1 + 0.7)
    ax.set_xlabel("number of neighbours", fontsize=10.5)
    if annotate:
        note = "$\\chi^2$ = %.3g" % chi2
        if pval is not None:
            note += "\np = %.3g" % pval
        ax.annotate(note, xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=10, color=STAT_COLOUR)
    if label_inside:
        # stacked panels share a border, so a title above the lower one would
        # land on that line; the stage goes inside its own panel instead
        ax.annotate(stage, xy=(0.03, 0.94), xycoords="axes fraction",
                    ha="left", va="top", fontsize=12, fontweight="bold")
    else:
        ax.set_title(stage, fontsize=12, fontweight="bold")


def draw_one(ax, stage, mine, theirs, label, chi2, pval, ymax):
    """Interleaved bars for one array: its distribution against the movies'."""
    m_face, m_edge = STYLE[(True, stage)]
    e_face, e_edge = STYLE[(False, stage)]
    x = np.arange(len(BUCKETS), dtype=float)
    ax.bar(x - BAR_W / 2, mine, BAR_W, color=m_face, edgecolor=m_edge,
           linewidth=1.3, zorder=2, label="array")
    e_mean = theirs.mean(axis=0)
    e_sem = theirs.std(axis=0, ddof=1) / np.sqrt(theirs.shape[0])
    ax.bar(x + BAR_W / 2, e_mean, BAR_W, color=e_face, edgecolor=e_edge,
           linewidth=1.3, zorder=2, label="experiment")
    ax.errorbar(x + BAR_W / 2, e_mean, yerr=e_sem, fmt="none",
                ecolor=STAT_COLOUR, elinewidth=1.3, capsize=3, capthick=1.3,
                zorder=4)
    for row in theirs:
        ax.scatter(x + BAR_W / 2, row, s=POINT_SIZE["model"], marker=".",
                   color=POINT_COLOUR, alpha=0.8, linewidths=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUCKETS], fontsize=8)
    ax.set_ylim(0, ymax)
    ax.tick_params(axis="y", labelsize=8)
    note = "$\\chi^2$ = %.3g" % chi2
    if pval is not None:
        note += "\np = %.3g" % pval
    ax.annotate(note, xy=(0.97, 0.95), xycoords="axes fraction", ha="right",
                va="top", fontsize=8.5, color=STAT_COLOUR)
    ax.set_title(label, fontsize=9.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays", type=int, nargs="+", default=list(ARRAYS))
    ap.add_argument("--individual", action="store_true",
                    help="also write one file per array")
    ap.add_argument("--combined", action="store_true",
                    help="one panel per stage with the arrays averaged into a"
                         " single histogram (whisker = SEM over arrays)"
                         " instead of a panel per array")
    ap.add_argument("--no-pvalue", dest="want_p", action="store_false",
                    help="skip the hierarchical p value even if it is available")
    ap.add_argument("--stacked", action="store_true",
                    help="with --combined, put the stages one above the other"
                         " instead of side by side")
    ap.add_argument("--score-labels", dest="score_labels", action="store_true",
                    help="write chi^2 and p onto the combined panels; off by"
                         " default, since the scores are reported in the text."
                         " They are printed to the console either way")
    ap.add_argument("--legacy-mask", dest="legacy", action="store_true",
                    help="count every valid experimental cell, edge cells"
                         " included — the population the original selection"
                         " and its p values used")
    ap.add_argument("--results-dir", dest="results_dir", default=RESULTS_DIR,
                    help="where the random_periodic_array<n>_for_<stage>"
                         " folders live (default %(default)s)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the figures (default: --results-dir)")
    a = ap.parse_args()
    if a.out_dir is None:
        a.out_dir = a.results_dir

    said = False
    tag = "_legacy" if a.legacy else ""
    pooled = {}
    missing = []
    for stage in STAGES:
        exp_cells = experimental_counts(stage, a.legacy)
        panels = []
        print("\n  %s   (%s mask: %s cells per movie, mean %.4f)"
              % (stage, "legacy" if a.legacy else "corrected",
                 "/".join(str(c.size) for c in exp_cells),
                 float(np.mean([c.mean() for c in exp_cells]))))
        print("  %-7s %7s %10s %12s" % ("array", "cells", "chi^2", "p"))
        for k in a.arrays:
            folder = "random_periodic_array%d_for_%s" % (k, SUFFIX[stage])
            if not os.path.isdir(os.path.join(a.results_dir, folder)):
                # NOT silent: an empty table with no explanation is what a
                # bare `continue` here used to produce
                missing.append(folder)
                continue
            mc = model_counts(folder, a.results_dir)
            chi2, mine, theirs = scores(mc, exp_cells)
            p, why = (pvalue([mc], exp_cells) if a.want_p else (None, None))
            if why and not said:
                print("  (no p value -- %s)" % why)
                said = True
            panels.append(dict(array=k, mine=mine, theirs=theirs, chi2=chi2,
                               p=p, counts=mc, n_cells=mc.size))
            print("  %-7d %7d %10.3f %12s"
                  % (k, mc.size, chi2, "%.4g" % p if p is not None else "-"))
        if missing:
            print("  (%d array folder(s) not found under %s: %s)"
                  % (len(missing), a.results_dir, ", ".join(missing)))
            missing = []
        if not panels:
            want = ["random_periodic_array%d_for_%s" % (k, SUFFIX[stage])
                    for k in a.arrays]
            raise SystemExit(
                "no initial-array folder found for %s under\n    %s\n"
                "  looked for: %s%s\n"
                "  Point --results-dir at the directory that holds them (or set"
                " TISSUE_RESULTS_DIR)."
                % (stage, a.results_dir, ", ".join(want[:3]),
                   " ..." if len(want) > 3 else ""))
        pooled[stage] = (np.array([d["mine"] for d in panels]),
                         panels[0]["theirs"],
                         [d["counts"] for d in panels], exp_cells)
        if a.combined:
            continue

        ymax = 1.18 * max(max(d["mine"].max(), d["theirs"].max())
                          for d in panels)
        ncol = 5
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.9 * nrow),
                                 sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, d in zip(axes, panels):
            draw_one(ax, stage, d["mine"], d["theirs"], "array %d" % d["array"],
                     d["chi2"], d["p"], ymax)
        for ax in axes[len(panels):]:
            ax.axis("off")
        for k, ax in enumerate(axes[:len(panels)]):
            if k % ncol == 0:
                ax.set_ylabel("% of cells", fontsize=9)
            if k >= len(panels) - ncol:
                ax.set_xlabel("neighbours", fontsize=9)

        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[(m, stage)][0],
                                 edgecolor=STYLE[(m, stage)][1], linewidth=1.4)
                   for m in (True, False)]
        names = ["initial array", "experiment (mean of %d movies)"
                 % len(EXPERIMENTS)]
        handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                                  color=POINT_COLOUR, alpha=0.8,
                                  markersize=np.sqrt(POINT_SIZE["model"]) * 1.8))
        names.append("one movie")
        fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, 0.004))
        fig.suptitle("%s: neighbour-count distribution of each initial array,"
                     " against the experiment%s"
                     % (stage, "   [every valid cell, edge cells included]"
                        if a.legacy else ""), fontsize=12, y=0.985)
        fig.subplots_adjust(left=0.055, right=0.99, top=0.885, bottom=0.115,
                            wspace=0.12, hspace=0.42)
        stem = os.path.join(a.out_dir, "initial_array_neighbor_hist_%s%s"
                            % (stage.replace(".", ""), tag))
        for ext in ("png", "svg"):
            fig.savefig("%s.%s" % (stem, ext), dpi=200)
        plt.close(fig)
        print("  wrote %s.png / .svg" % stem)

        if a.individual:
            for d in panels:
                f, ax = plt.subplots(figsize=(4.0, 3.4))
                draw_one(ax, stage, d["mine"], d["theirs"],
                         "%s, array %d" % (stage, d["array"]), d["chi2"],
                         d["p"], ymax)
                ax.set_ylabel("% of cells", fontsize=9.5)
                ax.set_xlabel("number of neighbours", fontsize=9.5)
                f.subplots_adjust(left=0.145, right=0.97, top=0.9, bottom=0.145)
                one = os.path.join(a.out_dir,
                                   "initial_array_neighbor_hist_%s_array%d%s"
                                   % (stage.replace(".", ""), d["array"], tag))
                for ext in ("png", "svg"):
                    f.savefig("%s.%s" % (one, ext), dpi=200)
                plt.close(f)
            print("  wrote %d individual figures" % len(panels))

    if not a.combined or not pooled:
        return

    # the averaged histogram: one panel per stage, both on one y scale
    stages = [s for s in STAGES if s in pooled]
    ymax = 1.2 * max(max(m.max(), t.max()) for m, t, _c, _e in pooled.values())
    # The panels meet on ONE line, drawn once: side by side that line is the
    # left panel's right border and the right panel's y axis, so the right
    # panel's own left spine goes; stacked it is the upper panel's bottom
    # border and the lower panel's top, so the lower panel's top spine goes.
    if a.stacked:
        fig, axes = plt.subplots(len(stages), 1, figsize=(6.6, 3.5 * len(stages)),
                                 sharex=True, sharey=True,
                                 gridspec_kw=dict(hspace=0.0))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[:-1]:
            ax.tick_params(axis="x", bottom=False)
        for ax in axes[1:]:
            ax.spines["top"].set_visible(False)
    else:
        fig, axes = plt.subplots(1, len(stages), figsize=(5.2 * len(stages), 4.6),
                                 sharey=True, gridspec_kw=dict(wspace=0.0))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[1:]:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)
    print("\n  averaged over arrays, chi^2 per bucket with BOTH spreads;"
          "\n  p over the same two groups, the arrays as the model's repeats:")
    for ax, stage in zip(axes, stages):
        model_rows, theirs, counts, exp_cells = pooled[stage]
        total = 0.0
        for b in range(len(BUCKETS)):
            _z, chi2, _ms, _me = _nsigma_and_chi2(model_rows[:, b], theirs[:, b])
            if np.isfinite(chi2):
                total += chi2
        p, why = (pvalue(counts, exp_cells) if a.want_p else (None, None))
        draw_combined(ax, stage, model_rows, theirs, total, ymax, p,
                      a.score_labels, a.stacked)
        print("    %-6s %d arrays vs %d movies   chi^2 = %8.3f   p = %s"
              % (stage, model_rows.shape[0], theirs.shape[0], total,
                 "%.4g" % p if p is not None else "-"))
        if why:
            print("           (no p value -- %s)" % why)
    if a.stacked:
        # one x label under the pair, a y label on each panel
        for ax in axes[:-1]:
            ax.set_xlabel("")
        for ax in axes:
            ax.set_ylabel("% of cells", fontsize=11)
    else:
        axes[0].set_ylabel("% of cells", fontsize=11)

    handles, names = [], []
    for stage in stages:
        for m, what in ((True, "initial arrays"), (False, "experiment")):
            handles.append(plt.Rectangle((0, 0), 1, 1,
                                         facecolor=STYLE[(m, stage)][0],
                                         edgecolor=STYLE[(m, stage)][1],
                                         linewidth=1.4))
            names.append("%s, %s" % (what, stage))
    handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                              color=POINT_COLOUR, alpha=0.8,
                              markersize=np.sqrt(POINT_SIZE["model"]) * 1.8))
    names.append("one array / one movie")
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.004))
    # the stacked figure is much narrower, so the title breaks onto its own
    # lines rather than running off both sides
    legacy_note = "every valid cell, edge cells included"
    if a.stacked:
        title = ("Neighbour-count distribution: the initial arrays\n"
                 "averaged, against the experiment\n"
                 "bar = mean over arrays / movies, whisker = SEM")
        if a.legacy:
            title += "\n[%s]" % legacy_note
        size = 10.5
    else:
        title = ("Neighbour-count distribution: the initial arrays averaged,"
                 " against the experiment%s\n"
                 "bar = mean over arrays / movies, whisker = SEM"
                 % ("   [%s]" % legacy_note if a.legacy else ""))
        size = 12
    fig.suptitle(title, fontsize=size, y=0.985, va="top", linespacing=1.5)
    if a.stacked:
        fig.subplots_adjust(left=0.125, right=0.98,
                            top=0.855 if a.legacy else 0.875, bottom=0.165,
                            hspace=0.0)
    else:
        fig.subplots_adjust(left=0.085, right=0.985, top=0.835, bottom=0.215,
                            wspace=0.0)
    stem = os.path.join(a.out_dir, "initial_array_neighbor_hist_combined%s%s"
                        % ("_stacked" if a.stacked else "", tag))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    print("  wrote %s.png / .svg" % stem)


if __name__ == "__main__":
    main()
