"""HC neighbours at differentiation: model against experiment, bucket by bucket.

    python plot_hc_neighbors_bars.py
    python plot_hc_neighbors_bars.py --figure li

Score 2 asks how many hair-cell neighbours a cell already had at the moment it
differentiated. Lateral inhibition predicts none, so the 0 bucket is the
prediction and the 1 and 2+ buckets are its failures.

Two figures, each a single panel holding both stages:

    li     lateral-inhibition-only model against the experiment   (4 bars/bucket)
    full   the full model at pT = 0 and the fitted pT against the
           experiment                                             (6 bars/bucket)

COLOUR CARRIES THE STAGE, POSITION CARRIES THE MODEL. The palette is the one the
other manuscript figures use — turquoise/green for a model at E17.5,
orange/orangered at P0, cyan/blue for the experiment at E17.5, pink/red at P0.
So in the full-model figure pT = 0 and the fitted pT share a colour within a stage
and are told apart by their x label, exactly as in the neighbour-pair figure.

Within a bucket every bar touches its neighbour; buckets are separated by
BUCKET_GAP. Order inside a bucket is E17.5's sources then P0's, models before the
experiment.

ONLY THE 0 AND 1 BUCKETS ARE DRAWN, because those are the only two score 2 was
ever built from. The percentages are still out of ALL differentiation events, so
the two bars of a source do not add to 100 — the missing remainder is the 2+
cells, which the model records and the experiment's target never did.

EVERY PERCENTAGE IS COMPUTED FROM COUNTS, not read from a pct column, so the
sources stay comparable: fullmodel_runs carries exact counts 0/1/2/3 (they sum
to n_differentiation_events on every row) while the lateral-inhibition table
carries 0/1/2plus.

VERIFIED AGAINST THE PUBLISHED TABLES. The z values here reproduce
fullmodel_tables.xlsx sheet "pT" (ev0_nsigma, ev1_nsigma, score2) to machine
precision at both stages and both pT, and the LI values reproduce
old/lateral_inhibition_tables.xlsx morphology_summary to four decimals.

THE HIERARCHY. One model point is one INITIAL ARRAY: each run's percentage is
computed, those are averaged within the array, and the bar is the mean over
arrays with its SEM. Repeats of one array share an initial condition and are not
independent, which is the same rule every other score here uses. One experimental
point is one biological repeat, taken as it is. A run in which nothing
differentiated has no percentage and is dropped, rather than counted as 0%.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import read_table, experimental_results_folder
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_COLOUR, POINT_SIZE,
                                 STAT_COLOUR, ERROR_CAPSIZE, ERROR_LINEWIDTH,
                                 BAR_W, _mean_sem)
from best_pt import BEST_PT, BEST_PT_LABEL

# The LI tables briefly lived in old/ and new/ subfolders holding DIFFERENT
# datasets of identical size (same 100 runs over 10 arrays, different numbers).
# The user deleted new/ on 2026-10-04 and moved the correct files — the ones
# lateral_inhibition_tables.xlsx was built from — back here. This runs.pkl
# reproduces that workbook's score 2 exactly, so the subfolder is gone for good.
LI_DIR = r"D:\Kasirer\results\lateral inhibition only results"
LI_PARAMS = {"E17.5": dict(pS=0.1, pR=0.3), "P0": dict(pS=0.15, pR=0.25)}
STAGES = ("E17.5", "P0")
# Score 2 is built from the 0 and 1 buckets ONLY — see fullmodel_tables.xlsx
# sheet "pT" (ev0_nsigma, ev1_nsigma, score2 = ev0^2 + ev1^2) and the LI
# workbook's morphology_summary, which carries an exp mean/SEM for 0 and 1 and
# none for 2+. The 2+ bucket is recorded for the model but was never scored, so
# it is left out here too and these figures match the cited numbers exactly.
BUCKETS = ("0", "1")                  # HC neighbours at differentiation
BUCKET_LABEL = {"0": "0", "1": "1", "2+": "2 or more"}
BUCKET_GAP = 0.8                      # between buckets; bars inside one touch
STAGE_GAP = BUCKET_GAP                # the stage blocks are spaced like buckets
# the fitted pT shares its stage's colour with pT = 0, so it is hatched to tell
# the two apart without leaving the one-colour-per-stage palette.
# The key is "ptbest", not the value: it used to be "pt162", so the re-fit would
# have needed the string changed in four places or the series would vanish.
HATCH = {"ptbest": "/"}

# (key, x-axis label) in the order they are drawn WITHIN one stage
FIGURES = {
    "li": dict(stem="hc_neighbors_li_vs_exp",
               title="HC neighbours at differentiation: "
                     "lateral inhibition only against the experiment",
               sources=[("li", "LI only"), ("exp", "exp")]),
    "full": dict(stem="hc_neighbors_fullmodel_vs_exp",
                 title="HC neighbours at differentiation: "
                       "the full model against the experiment",
                 sources=[("pt0", "$p_T$=0"),
                          ("ptbest", "$p_T$=%s" % BEST_PT_LABEL),
                          ("exp", "exp")]),
}


def _bucket_counts(frame, total_col="n_differentiation_events"):
    """{bucket: per-row count} from whatever bucket columns the table carries.

    2+ is always derived from the total rather than summed from named columns,
    so a table that stops at 3 and a table that stops at "2plus" give the same
    answer, and a run with 4 HC neighbours could not be silently dropped.
    """
    n0 = frame["n_events_0_HC_neighbours"].to_numpy(float)
    n1 = frame["n_events_1_HC_neighbours"].to_numpy(float)
    total = frame[total_col].to_numpy(float)
    return {"0": n0, "1": n1, "2+": total - n0 - n1}, total


def model_points(frame, bucket):
    """Per-array percentages: per run first, then averaged within the array."""
    counts, total = _bucket_counts(frame)
    pct = np.where(total > 0,
                   100.0 * counts[bucket] / np.where(total > 0, total, 1), np.nan)
    per_run = pd.DataFrame(dict(initial_array=frame["initial_array"].to_numpy(),
                                pct=pct)).dropna(subset=["pct"])
    if not len(per_run):
        return np.array([], float), 0
    return (per_run.groupby("initial_array")["pct"].mean().to_numpy(float),
            len(per_run))


def experiment_points(stage, bucket):
    """One percentage per biological repeat, from the raw per-cell counts.

    fullmodel_experiment.pkl records only the 0 and 1 rows (score 2 needs no
    more), so the counts themselves are read instead — the same arrays that
    table was built from.
    """
    out = []
    for i in (0, 1, 2):
        path = os.path.join(experimental_results_folder,
                            "%s differentiating cells_experiment%d.npy" % (stage, i))
        if not os.path.isfile(path):
            continue
        counts = np.asarray(np.load(path), float)
        counts = counts[np.isfinite(counts)]
        if not counts.size:
            continue                    # nothing differentiated: no percentage
        hit = (counts == 0) if bucket == "0" else (
            (counts == 1) if bucket == "1" else (counts >= 2))
        out.append(100.0 * float(np.mean(hit)))
    return np.asarray(out, float), len(out)


def gather(which, results_dir=RESULTS_DIR, li_dir=LI_DIR):
    """One row per (stage, bucket, source), in the order the bars are drawn.

    Stage-major: the whole E17.5 block — all three buckets — comes before the
    whole P0 block, and the two are separated by STAGE_GAP.
    """
    li = pd.read_pickle(os.path.join(li_dir, "runs.pkl"))
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    # collapsed runs are a different outcome, not a noisy sample of this one —
    # the same exclusion every per-point aggregate in this project makes
    full = full[~full["collapsed"].astype(bool)]

    sources = FIGURES[which]["sources"]
    rows = []
    for si, stage in enumerate(STAGES):
        for bi, bucket in enumerate(BUCKETS):
            for ki, (key, short) in enumerate(sources):
                if key == "exp":
                    vals, n_raw = experiment_points(stage, bucket)
                    unit = "repeats"
                else:
                    if key == "li":
                        p = LI_PARAMS[stage]
                        g = li[(li["stage"] == stage)
                               & np.isclose(li["pS"].astype(float), p["pS"])
                               & np.isclose(li["pR"].astype(float), p["pR"])]
                    else:
                        pt = 0.0 if key == "pt0" else BEST_PT
                        g = full[(full["stage"] == stage)
                                 & np.isclose(full["psigma"].astype(float), pt)]
                    if not len(g):
                        raise SystemExit("no runs for %s / %s" % (stage, key))
                    vals, n_raw = model_points(g, bucket)
                    unit = "arrays"
                m, s = _mean_sem(vals)
                face, edge = STYLE[(key != "exp", stage)]
                rows.append(dict(bucket=bucket, bucket_index=bi, stage=stage,
                                 stage_index=si, key=key, short=short,
                                 source_index=ki,
                                 face=face, edge=edge, mean=m, sem=s,
                                 points=vals, unit=unit, n_units=len(vals),
                                 n_runs=n_raw, z=np.nan, z2=np.nan))
    return _score(rows)


def _score(rows):
    """z^2 of each model bar against the experiment bar beside it.

    z = (mean_model - mean_exp) / sqrt(SEM_model^2 + SEM_exp^2), the same
    n-sigma every other model-against-experiment score in this project uses —
    both sides carry sampling error, so both SEMs enter. It is NOT the mechanics
    fit's z, which divides by SEM_exp alone because there the model side was a
    single pooled value.

    The comparison is strictly within a (stage, bucket): a model bar is scored
    against the experiment of its own stage and its own neighbour count.
    """
    exp = {(r["stage"], r["bucket"]): r for r in rows if r["key"] == "exp"}
    for r in rows:
        if r["key"] == "exp":
            continue
        e = exp.get((r["stage"], r["bucket"]))
        if e is None:
            continue
        den = np.hypot(r["sem"], e["sem"])
        if np.isfinite(den) and den > 0:
            r["z"] = (r["mean"] - e["mean"]) / den
            r["z2"] = r["z"] ** 2
    return rows


BOTTOM_PAD = 0.08         # of the visible span, left below the lowest point


def tighten_bottom(ax, rows, pad=BOTTOM_PAD):
    """Start the axis just under the lowest thing drawn, not at zero.

    NOTE this truncates the bars: with the baseline cut off, a bar's LENGTH no
    longer encodes its value and only the top edge is readable. That is the
    point here — the differences between these sources are small next to their
    absolute size — but it has to be said in the caption.

    The floor is the lowest of every bar's mean minus its SEM and every
    scattered point, so nothing drawn falls outside the axis.
    """
    lows = []
    for r in rows:
        sem = r["sem"] if np.isfinite(r["sem"]) else 0.0
        lows.append(float(r["mean"]) - sem)
        if len(r["points"]):
            lows.append(float(np.min(r["points"])))
    if not lows:
        return
    lo, hi = min(lows), ax.get_ylim()[1]
    bottom = lo - pad * max(hi - lo, 1e-12)
    if lo >= 0.0:
        # a percentage cannot be negative, so the pad never opens empty space
        # below zero. Where runs put 0% in a bucket — common here — the lowest
        # point IS zero and the axis simply starts there, unzoomed.
        bottom = max(bottom, 0.0)
    ax.set_ylim(bottom, hi)


def draw(rows, which, rng, out_dir, scores=False):
    """A single panel, one block per stage: buckets along x, sources inside.

    ``scores`` writes each model bar's z^2 against the experiment beside it,
    above whichever is higher — the error bar or the highest scattered point —
    and is saved under a separate name so the clean figure stays clean.
    """
    n_src = len(FIGURES[which]["sources"])
    # gaps go BETWEEN buckets, not after the last one — counting a trailing gap
    # in the block width would add it to STAGE_GAP and make the stage break
    # wider than the bucket breaks it is supposed to match
    block = len(BUCKETS) * n_src + (len(BUCKETS) - 1) * BUCKET_GAP
    slots = len(STAGES) * len(BUCKETS) * n_src
    fig, ax = plt.subplots(figsize=(3.2 + 0.70 * slots, 5.4))

    def xpos(r):
        return (r["stage_index"] * (block + STAGE_GAP)
                + r["bucket_index"] * (n_src + BUCKET_GAP) + r["source_index"])

    for r in rows:
        x = xpos(r)
        ax.bar(x, r["mean"], BAR_W, color=r["face"], edgecolor=r["edge"],
               linewidth=1.6, hatch=HATCH.get(r["key"]), zorder=2)
        ax.errorbar(x, r["mean"], yerr=r["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=4)
        v = r["points"]
        if len(v):
            jit = rng.uniform(-BAR_W * 0.26, BAR_W * 0.26, size=len(v))
            ax.scatter(x + jit, v,
                       s=POINT_SIZE["exp" if r["key"] == "exp" else "model"],
                       marker=".", color=POINT_COLOUR, alpha=0.8, linewidths=0,
                       zorder=3)

    tighten_bottom(ax, rows)
    if scores:
        # headroom first, so the labels are placed against the final scale, and
        # as a fraction of the SPAN — the axis no longer starts at zero, so
        # scaling the top would not be a margin
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
        span = ax.get_ylim()[1] - lo
        for r in rows:
            if not np.isfinite(r.get("z2", np.nan)):
                continue
            over = [r["mean"] + (r["sem"] if np.isfinite(r["sem"]) else 0.0)]
            if len(r["points"]):
                over.append(float(np.max(r["points"])))
            ax.annotate("%.3g" % r["z2"], xy=(xpos(r), max(over) + 0.015 * span),
                        ha="center", va="bottom", fontsize=8.5,
                        color=STAT_COLOUR, zorder=6)

    # one tick per BAR naming the model, and the bucket annotated under the block
    # One tick per BAR GROUP, carrying only the neighbour count. Which bar is
    # which model is left to the legend: colour separates model from experiment,
    # and the hatch separates the two pT values.
    centres = {}
    for si in range(len(STAGES)):
        for bi in range(len(BUCKETS)):
            centres[(si, bi)] = (si * (block + STAGE_GAP)
                                 + bi * (n_src + BUCKET_GAP) + (n_src - 1) / 2.0)
    keys = sorted(centres)
    ax.set_xticks([centres[k] for k in keys])
    ax.set_xticklabels([BUCKET_LABEL[BUCKETS[k[1]]] for k in keys], fontsize=11)
    first, last = min(centres.values()), max(centres.values())
    ax.set_xlim(first - (n_src - 1) / 2.0 - 0.75,
                last + (n_src - 1) / 2.0 + 0.75)
    ax.set_xlabel("HC neighbours at differentiation", fontsize=11.5)
    # the stage names sit over their block, the way every other figure here
    # labels a stage
    for si, stage in enumerate(STAGES):
        xs = [centres[(si, bi)] for bi in range(len(BUCKETS))]
        ax.annotate(stage, xy=(float(np.mean(xs)), 1.02),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("% of differentiating cells", fontsize=11)

    handles, names = [], []
    for is_model in (True, False):
        for stage in STAGES:
            face, edge = STYLE[(is_model, stage)]
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=face,
                                         edgecolor=edge, linewidth=1.6))
            names.append(STYLE_LABEL[(is_model, stage)])
    # the hatch is a second channel on top of the stage colour, so its key is a
    # neutral swatch rather than one per stage. Only shown when a bar uses it.
    if any(HATCH.get(r["key"]) for r in rows):
        # the key swatch is small enough that a single "/" can fall entirely
        # outside it and render blank, so the key is drawn denser than the bars
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="0.25", linewidth=1.6,
                                     hatch=HATCH["ptbest"] * 3))
        names.append(r"full model, $p_T$ = %s" % BEST_PT_LABEL)
    handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                              color=POINT_COLOUR, alpha=0.8, markersize=9))
    names.append("one initial array (model) / one repeat (experiment)")
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))
    # on its own line: as one line it overran both edges of the narrower
    # two-bar figure
    second = ("bar = mean, whisker = SEM\nnumber over a model bar is "
              r"$z^2$ against its own stage and bucket"
              if scores else "bar = mean, whisker = SEM")
    fig.suptitle(FIGURES[which]["title"] + "\n" + second,
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.085, right=0.985,
                        top=0.80 if scores else 0.845, bottom=0.215)

    stem = os.path.join(out_dir,
                        FIGURES[which]["stem"] + ("_scored" if scores else ""))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", choices=("li", "full", "both"), default="both")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--li-dir", default=LI_DIR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for which in (("li", "full") if a.figure == "both" else (a.figure,)):
        rows = gather(which, li_dir=a.li_dir)
        print("\n  %s" % FIGURES[which]["title"])
        print("  %-6s %-10s %-10s %9s %9s %7s %7s %8s %8s"
              % ("stage", "bucket", "source", "mean %", "SEM", "units", "runs",
                 "z", "z^2"))
        for r in rows:
            print("  %-6s %-10s %-10s %9.2f %9.2f %7d %7d %8s %8s"
                  % (r["stage"], BUCKET_LABEL[r["bucket"]],
                     r["short"].replace("$p_T$", "pT"), r["mean"], r["sem"],
                     r["n_units"], r["n_runs"],
                     "%+.2f" % r["z"] if np.isfinite(r["z"]) else "-",
                     "%.3g" % r["z2"] if np.isfinite(r["z2"]) else "-"))
        # summed over the two buckets this IS score 2, so the print doubles as a
        # check against fullmodel_tables.xlsx / lateral_inhibition_tables.xlsx
        for stage in STAGES:
            for key, short in FIGURES[which]["sources"]:
                if key == "exp":
                    continue
                tot = np.nansum([r["z2"] for r in rows
                                 if r["key"] == key and r["stage"] == stage])
                print("    score 2 (sum of the two z^2)  %-6s %-10s %9.4f"
                      % (stage, short.replace("$p_T$", "pT"), tot))
        for scores in (False, True):
            stem = draw(rows, which, np.random.default_rng(a.seed), a.out_dir,
                        scores)
            print("  wrote %s.png / .svg" % stem)


if __name__ == "__main__":
    main()
