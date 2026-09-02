"""Face stress over time by cell group, both psigma values in one figure.

    python plot_face_stress_groups.py
    python plot_face_stress_groups.py --effectors all --t-min 2

The stress the Hill gate reads, split into the four groups differentiation
distinguishes: support cells with 0, 1 and >= 2 hair-cell neighbours, and the
hair cells themselves. Groups are reassigned at every sampled frame, so a cell
moves between them as it differentiates.

WHERE THE NUMBERS COME FROM. face_stress_ps0_groups.py does the expensive part —
walking every archive frame by frame — and leaves per-run, per-array and summary
pickles per psigma value. This reads those summaries, joins them into one table
with a psigma column, adds it to the full-model workbook, and draws the figure.
Re-run that script first if the archives change; nothing here re-reads them.

WHAT A CURVE IS. Mean over arrays of (mean over that array's repeats of the mean
stress over the group's cells), so the band is the SEM ACROSS ARRAYS with the
lateral-inhibition seed noise averaged out first — not cell-to-cell spread, which
is far larger and would say nothing about reproducibility. Early on a group may be
missing from some arrays (no HCs yet, no SC with two HC neighbours); those means
are over a subset and are drawn as faded markers without a band rather than
silently mixed into the curve.

The dashed line on the psigma > 0 panels is the gate's half-max: a group above it
has its differentiation more than half enabled by stress, below it less than half.
It belongs only on the contractility panels, since that is the effector set
run_model.stress_effectors actually gates on.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import add_sheets
from face_stress_ps0_groups import GROUPS, COLOUR, base_name

PSIGMAS = (0.0, 0.162)
STAGES = ("E17.5", "P0")
SHEET = "face_stress"
PKL = "fullmodel_face_stress"


def gather(psigmas=PSIGMAS, results_dir=RESULTS_DIR):
    """The per-psigma summaries as one table, the parameter as a pT column."""
    frames = []
    for ps in psigmas:
        path = os.path.join(results_dir, "%s_summary.pkl" % base_name(ps))
        if not os.path.isfile(path):
            raise SystemExit("no %s — run face_stress_ps0_groups.py --psigma %g"
                             % (os.path.basename(path), ps))
        f = pd.read_pickle(path)
        f.insert(0, "pT", float(ps))      # the manuscript's name for it
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["pT", "stage", "effectors", "group", "time"]
                           ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--effectors", default="contractility",
                    choices=("contractility", "all"),
                    help="which stress the figure shows (the gate reads"
                         " contractility)")
    ap.add_argument("--t-min", dest="t_min", type=float, default=2.0,
                    help="ignore the settling transient before this time")
    ap.add_argument("--no-sheet", dest="sheet", action="store_false",
                    help="draw the figure without touching the workbook")
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    a = ap.parse_args()

    table = gather()
    print("  combined summary: %d rows  (psigma %s, stages %s, effectors %s)"
          % (len(table), sorted(table.pT.unique()),
             sorted(table.stage.unique()), sorted(table.effectors.unique())))

    if a.sheet:
        table.to_pickle(os.path.join(RESULTS_DIR, PKL + ".pkl"))
        if add_sheets(os.path.join(RESULTS_DIR, "fullmodel_tables.xlsx"),
                      {SHEET: table}):
            print("  added sheet '%s' to fullmodel_tables.xlsx and wrote %s.pkl"
                  % (SHEET, PKL))

    show = table[(table["effectors"] == a.effectors)
                 & (table["time"] >= a.t_min)]
    # One y axis for all four panels: the quantity is the same everywhere, so a
    # shared scale is what lets the stages and the two pT values be read against
    # each other. Limited to the data — a zero origin would push every curve into
    # the top third of the panel.
    lo = float((show["mean"] - show["sem"].fillna(0.0)).min())
    hi = float((show["mean"] + show["sem"].fillna(0.0)).max())
    pad = 0.06 * (hi - lo)

    fig, axes = plt.subplots(len(PSIGMAS), len(STAGES), squeeze=False,
                             figsize=(11.0, 7.6), sharex="col", sharey=True)
    for r, ps in enumerate(PSIGMAS):
        for c, stage in enumerate(STAGES):
            ax = axes[r][c]
            panel = show[(show["pT"] == ps) & (show["stage"] == stage)]
            full = panel["n_arrays"].max()
            for label, _ct, _bins in GROUPS:
                s = panel[panel["group"] == label].sort_values("time")
                if not len(s):
                    continue
                solid, partial = s[s["n_arrays"] >= full], s[s["n_arrays"] < full]
                ax.plot(solid["time"], solid["mean"], "-", color=COLOUR[label],
                        lw=1.8, label=label, zorder=3)
                ax.fill_between(solid["time"], solid["mean"] - solid["sem"],
                                solid["mean"] + solid["sem"],
                                color=COLOUR[label], alpha=0.22, linewidth=0,
                                zorder=2)
                if len(partial):
                    ax.plot(partial["time"], partial["mean"], "o", ms=3.0,
                            color=COLOUR[label], alpha=0.30, zorder=3)
            if ps > 0 and a.effectors == "contractility":
                ax.axhline(ps, color="k", lw=1.4, ls="--", zorder=4,
                           label=r"$p_T$ = %.3f (gate half-max)" % ps)
            ax.set_ylim(lo - pad, hi + pad)
            # Which panel is which is left to the caption: no titles. The shared
            # scale means only the left column needs an axis.
            if c == 0:
                ax.set_ylabel("mean face stress (%s)" % a.effectors, fontsize=10)
            if r == len(PSIGMAS) - 1:
                ax.set_xlabel("simulation time", fontsize=10)
            else:                       # no ticks poking through a shared border
                ax.tick_params(axis="x", bottom=False)

    # hspace/wspace = 0: every panel shares its border with its neighbours
    fig.subplots_adjust(left=0.085, right=0.915, top=0.985, bottom=0.085,
                        hspace=0, wspace=0)

    stem = os.path.splitext(a.out or os.path.join(RESULTS_DIR,
                                                  "face_stress_groups"))[0]
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
        print("\nwrote %s.%s" % (stem, ext))


if __name__ == "__main__":
    main()
