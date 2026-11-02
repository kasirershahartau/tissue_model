"""Distance from the ablation, model against experiment, in cell diameters.

    python plot_ablation_distance_vs_experiment.py
    python plot_ablation_distance_vs_experiment.py --quantity ratio --scored

Reads the three tables compare_ablation_distance_to_experiment.py writes and
draws them: one violin per pT value, the experiment beside them, P0 only (the
experimental ablation series is P0).

WHY THIS IS NOT plot_ablation_distances.py. That figure shows every EVENT's raw
distance in LATTICE UNITS, split into the events the ablation added and the ones
the control had instead. The experiment supplies neither: it measures
micrometres, and one mean distance per biological repeat with no control
subtraction. The only axis the two sides share is CELL DIAMETERS — "how many
cells away did this cell differentiate" — with each side supplying its own
equivalent-circle diameter, so nothing has to be calibrated between them. That
conversion, and the scores below, are the comparison script's; this file only
draws them.

TWO QUANTITIES, both scored there:

  distance  the mean distance of a differentiation event, in cell diameters.
            It carries the size of the domain — a cell can only be as far away
            as the periodic box or the field of view allows.
  ratio     that distance over the mean distance to every SC present before the
            ablation, which is where a cell would be if nothing drew it towards
            the wound. The domain size divides out.

THE CLOUD AND THE STATISTIC COME FROM DIFFERENT LEVELS, on both sides. Every
drawn point is ONE DIFFERENTIATION EVENT — the distribution the measurement
actually produced — while the star and whisker stay the mean and SEM over the
INDEPENDENT UNITS: initial arrays for the model (runs sharing an array are
averaged first, since they share an initial condition), biological repeats for
the experiment. Those units are what the z was computed over, so the star and
whisker are comparable across the figure even though the clouds are far wider.

The chi^2 printed by --scored is read from the summary table, not recomputed, so
the figure cannot drift from the number the comparison reports.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from best_pt import BEST_PT
from build_experimental_tables import read_table
from compare_ablation_distance_to_experiment import EXP_XLSX, SHEETS
from plot_neighbor_pairs import (STYLE, STYLE_LABEL, POINT_SIZE, POINT_COLOUR,
                                 STAT_COLOUR, MEAN_MARKERSIZE, ERROR_CAPSIZE,
                                 ERROR_LINEWIDTH, BAR_W, VIOLIN_W, _mean_sem)

PREFIX = "ablation_distance_vs_experiment"
STAGE = "P0"                    # the experimental ablation series is P0 only
PT = (0.0, BEST_PT)             # one model source per gate value, left to right
# ...but overridable with --pt, because the campaign ended up with THREE
# candidate gate values worth drawing side by side: 0 (no gate), 0.08556
# (matches the experimental differentiation rate) and 0.08667 (the fitted
# optimum). A module constant could only ever show two of them.
# (model column, experimental column, summary chi^2 column, y label, stem suffix)
QUANTITY = {
    "distance": ("diff_diameters", "dist_diameters", "z2_diameters",
                 "distance from the ablated cell (cell diameters)", ""),
    "ratio": ("ratio", "ratio", "z2_ratio",
              "distance / mean distance to an SC", "_ratio"),
}
# the chosen gate value is marked with a hatch rather than a colour of its own,
# as in plot_neighbor_pair_bars, plot_isolated_sc_violin and the stress figures
HATCH = "/"
HATCH_PT = BEST_PT
# space between the frame and the outer edge of the end shapes, in the same
# units as the positions; a bar is BAR_W wide, so this is the pad past its edge
# plus half a bar
X_PAD = 0.6
# room under the data, as a fraction of the drawn range. The axis dips below
# zero to make it, which puts no negative label on the axis — the tick locator
# steps in 5s from 0 — but does stop the lowest violin sitting on the frame.
BOTTOM_PAD = 0.06


def experimental_events(stage, group="diff"):
    """Every experimental differentiation event: its distance and its repeat.

    The same workbook, sheet and columns compare_ablation_distance_to_experiment
    reads — this is the step before the per-repeat mean it scores with. One
    biological repeat is one (Date, Position), written once and then left blank
    down the rows that belong to it.
    """
    raw = pd.read_excel(EXP_XLSX, SHEETS[(stage, group)], header=None)
    hdr = [str(v).strip() for v in raw.iloc[0].tolist()]
    df = raw.iloc[1:].copy()
    df.columns = hdr
    dcol = [c for c in df.columns if c.startswith("Distance")][0]
    d = pd.to_numeric(df[dcol], errors="coerce")
    rep = (df["Date"].ffill().astype(str) + " @ "
           + df["Position"].ffill().astype(str))
    m = d.notna()
    return pd.DataFrame(dict(repeat=rep[m].to_numpy(),
                             dist_um=d[m].to_numpy(float)))


def gather(quantity, results_dir=RESULTS_DIR, stage=STAGE, pts=None):
    """One row per source: its per-unit values, its mean and SEM, and its chi^2."""
    model_col, exp_col, z2_col, _ylabel, _suffix = QUANTITY[quantity]
    summary = read_table(os.path.join(results_dir, PREFIX + ".pkl"))
    runs = read_table(os.path.join(results_dir, PREFIX + "_runs.pkl"))
    exp = pd.read_pickle(os.path.join(results_dir, PREFIX + "_exp.pkl"))
    # one row per differentiation event, joined to its run by folder: the runs
    # table carries that run's cell diameter and reference distance, which are
    # what turn a lattice-unit distance into the figure's units
    events = read_table(os.path.join(results_dir, "ablation_events.pkl"))
    summary = summary[summary["stage"] == stage]
    runs = runs[runs["stage"] == stage]
    exp = exp[(exp["stage"] == stage) & (exp["unit"] == "repeat")]
    if not len(exp):
        raise SystemExit("no experimental repeats for %s in %s_exp.pkl"
                         % (stage, PREFIX))

    pts = tuple(PT if pts is None else pts)
    out = []
    for i, pt in enumerate(pts):
        g = runs[np.isclose(runs["psigma"].astype(float), pt)]
        if not len(g):
            raise SystemExit("no model ablations at pT=%g" % pt)
        # an array is the independent unit: repeats on one share its initial
        # condition, so they are averaged before the mean and SEM, exactly as
        # the comparison script does before scoring
        per_unit = g.groupby("initial_array")[model_col].mean().to_numpy(float)
        # every event of those runs, each divided by ITS OWN run's scale —
        # cell diameter, or the reference SC distance for the ratio — so a
        # point means the same thing it does in the per-run average above
        scale_col = "cell_diameter" if quantity == "distance" else "ref_mean"
        scale = g.set_index("run_folder")[scale_col]
        ev = events[events["run_folder"].isin(set(g["run_folder"]))]
        per_point = (ev["distance"].to_numpy(float)
                     / ev["run_folder"].map(scale).to_numpy(float))
        per_point = per_point[np.isfinite(per_point)]
        row = summary[np.isclose(summary["psigma"].astype(float), pt)]
        m, s = _mean_sem(per_unit)
        face, edge = STYLE[(True, stage)]
        out.append(dict(index=i, kind="model", label=r"full model" "\n"
                                                    r"$p_T$=%g" % pt,
                        hatch=HATCH if np.isclose(pt, HATCH_PT) else None,
                        face=face, edge=edge, mean=m, sem=s, per_unit=per_unit,
                        per_point=per_point,
                        z2=float(row[z2_col].iloc[0]) if len(row) else np.nan,
                        n_units=len(per_unit), unit="arrays", n_runs=len(g)))
    per_unit = exp[exp_col].to_numpy(float)
    m, s = _mean_sem(per_unit)
    face, edge = STYLE[(False, stage)]
    # the experiment's own events, the same treatment as the model's above
    ev = experimental_events(stage)
    if quantity == "ratio":
        # each event against its own repeat's reference distance, the pairing
        # the per-repeat ratio already uses
        ref = dict(zip(exp["label"].astype(str), exp["ref_um"].astype(float)))
        denom = ev["repeat"].astype(str).map(ref).to_numpy(float)
    else:
        denom = float(exp["cell_diameter_um"].iloc[0])
    per_point = ev["dist_um"].to_numpy(float) / denom
    per_point = per_point[np.isfinite(per_point)]
    out.append(dict(index=len(pts), kind="exp", label="experiment", hatch=None,
                    face=face, edge=edge, mean=m, sem=s, per_unit=per_unit,
                    per_point=per_point,
                    z2=np.nan, n_units=len(per_unit), unit="repeats",
                    n_runs=len(per_point)))
    return out


def draw(ax, data, rng, mean_markersize, exp_no_bar):
    """A violin per source, with its points; the star and whisker over them."""
    for d in data:
        x = float(d["index"])
        # the drawn cloud is per_point where a source has one (the experiment's
        # individual events); everywhere else it is the scoring unit itself
        v = d.get("per_point")
        v = d["per_unit"] if v is None else v
        is_exp = d["kind"] == "exp"
        shape = ("none" if is_exp and exp_no_bar
                 else "violin" if len(v) > 1 and np.ptp(v) > 0 else "bar")
        width = BAR_W
        if shape == "bar":
            ax.bar(x, d["mean"], BAR_W, color=d["face"], edgecolor=d["edge"],
                   linewidth=1.6, hatch=d["hatch"], zorder=2)
        elif shape == "violin":
            width = VIOLIN_W
            body = ax.violinplot([v], positions=[x], widths=width,
                                 showmeans=False, showextrema=False,
                                 showmedians=False)["bodies"][0]
            body.set_facecolor(d["face"]); body.set_edgecolor(d["edge"])
            body.set_linewidth(1.6); body.set_alpha(1.0); body.set_zorder(2)
            if d["hatch"]:
                body.set_hatch(d["hatch"])
        if shape != "bar":
            # a bar shows its mean as the bar top; anything else needs it drawn
            ax.plot(x, d["mean"], marker="*", markersize=mean_markersize,
                    color=STAT_COLOUR, linestyle="none", zorder=5)
        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=4)
        if len(v):
            jit = rng.uniform(-width * 0.28, width * 0.28, size=len(v))
            # the big dot is for a handful of movies; an event cloud gets the
            # ordinary one or it fills the violin and hides its shape
            dense = d.get("per_point") is not None
            ax.scatter(x + jit, v,
                       s=POINT_SIZE["exp" if is_exp and not dense else "model"],
                       marker=".", color=POINT_COLOUR, alpha=0.8, linewidths=0,
                       zorder=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quantity", choices=tuple(QUANTITY), default="distance",
                    help="distance in cell diameters (default), or that distance"
                         " over the reference SC distance")
    ap.add_argument("--scored", action="store_true",
                    help="print each model's chi^2 against the experiment over"
                         " its violin")
    ap.add_argument("--exp-no-bar", dest="exp_no_bar", action="store_true",
                    help="draw the experiment as star + SEM + points only")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE)
    ap.add_argument("--bottom-pad", dest="bottom_pad", type=float,
                    default=BOTTOM_PAD,
                    help="room under the data, as a fraction of the range"
                         " (default %g)" % BOTTOM_PAD)
    ap.add_argument("--x-pad", dest="x_pad", type=float, default=X_PAD,
                    help="space left of the first shape and right of the last"
                         " (default %g; a bar is %g wide)" % (X_PAD, BAR_W))
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--pt", type=float, nargs="+", default=None,
                    help="gate values to draw, left to right (default %s)" % (PT,))
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    _mc, _ec, _zc, ylabel, suffix = QUANTITY[a.quantity]
    data = gather(a.quantity, stage=a.stage, pts=a.pt)

    print("  %s, %s\n" % (a.stage, ylabel))
    print("  %-22s %9s %9s %12s %8s %9s"
          % ("source", "mean", "SEM", "stat units", "events", "z^2"))
    for d in data:
        drawn_n = len(d["per_point"]) if d.get("per_point") is not None \
            else len(d["per_unit"])
        print("  %-22s %9.3f %9.3f %4d %-7s %8d %9s"
              % (d["label"].replace("\n", " "), d["mean"], d["sem"],
                 d["n_units"], d["unit"], drawn_n,
                 "%.3g" % d["z2"] if np.isfinite(d["z2"]) else "-"))

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    draw(ax, data, np.random.default_rng(a.seed), a.mean_markersize,
         a.exp_no_bar)
    ax.set_ylabel(ylabel, fontsize=10.5)
    drawn = {d["index"]: (d["per_point"] if d.get("per_point") is not None
                          else d["per_unit"]) for d in data}
    top = max(np.nanmax(v) if len(v) else 0.0 for v in drawn.values())
    top = max(top, max(d["mean"] + (d["sem"] or 0.0) for d in data))
    hi = top * (1.28 if a.scored else 1.12)
    ax.set_ylim(-a.bottom_pad * hi, hi)

    if a.scored:
        span = ax.get_ylim()[1]
        for d in data:
            if not np.isfinite(d["z2"]):
                continue
            over = [d["mean"] + (d["sem"] if np.isfinite(d["sem"]) else 0.0)]
            if len(drawn[d["index"]]):
                over.append(float(np.max(drawn[d["index"]])))
            ax.annotate("%.3g" % d["z2"],
                        xy=(d["index"], max(over) + 0.015 * span), ha="center",
                        va="bottom", fontsize=9.5, color=STAT_COLOUR, zorder=6)

    ax.set_xticks([d["index"] for d in data])
    ax.set_xticklabels([d["label"] for d in data], fontsize=9.5)
    ax.set_xlim(-a.x_pad, len(data) - 1 + a.x_pad)

    keys = [(True, a.stage)] + ([] if a.exp_no_bar else [(False, a.stage)])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in keys]
    names = [STYLE_LABEL[k] for k in keys]
    if any(d["hatch"] for d in data):
        # the swatch is small, so one stroke can fall outside it entirely:
        # repeating the pattern guarantees the key is not drawn blank
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor="0.25", linewidth=1.6,
                                     hatch=HATCH * 3))
        names.append(r"$p_T$ = %g" % HATCH_PT)
    if a.exp_no_bar:
        handles.append(plt.Line2D([], [], marker="*", linestyle="none",
                                  color=STAT_COLOUR,
                                  markersize=a.mean_markersize))
        names.append("experiment (mean)")
    handles.append(plt.Line2D([], [], marker=".", linestyle="none",
                              color=POINT_COLOUR, alpha=0.8,
                              markersize=np.sqrt(POINT_SIZE["model"]) * 1.8))
    names.append("one differentiation event")
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Distance of an ablation-induced differentiation event (%s)\n"
                 "star = mean, whisker = SEM" % a.stage,
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.135, right=0.98, top=0.86, bottom=0.26)

    stem = os.path.splitext(a.out or os.path.join(
        RESULTS_DIR, "ablation_distance_vs_experiment%s%s"
        % (suffix, "_scored" if a.scored else "")))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
