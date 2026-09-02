"""Neighbour-pair composition: every model against the experiment, one figure.

    python plot_neighbor_pairs.py
    python plot_neighbor_pairs.py --points array --tight-y --out D:/somewhere/fig.png

Score 1 asks what fraction of the cell-cell contacts in the tissue are HC:HC and
what fraction are HC:SC. HC:HC is the informative one — lateral inhibition should
keep hair cells apart, so touching HCs are the pattern's failures — and HC:SC
says how much of the tissue boundary is HC-to-SC at all.

The two live on very different scales (HC:HC below 2%, HC:SC around 45%), so they
get a panel each with its own y axis. Both start at zero, as bars should; pass
--tight-y to crop each panel to its data instead, which magnifies the differences
at the cost of a truncated baseline.

VIOLINS FOR THE MODELS, BARS FOR THE EXPERIMENT. A model contributes tens of runs,
enough for the shape of the distribution to mean something; an experiment
contributes three movies, where a violin would be drawing a curve through noise.
The violin is a kernel density of exactly the points scattered over it, evaluated
between the lowest and the highest of them, so the shape starts and ends on real
runs and never suggests values nothing reached.

WHAT THE STATISTIC IS. The marker and error bar are computed exactly as score 1
computes them: average within an initial array (or, for the experiment, take each
movie as it is), then mean +- SEM across arrays/movies. The scattered points are
the individual runs by default, which show the spread WITHIN arrays too — a wider
point cloud than the error bar is expected, not a contradiction. --points array
plots the per-array means instead, the quantity the error bar actually describes,
and narrows the violin to match.

SOURCES. Read from the saved tables, not recomputed:
    lateral inhibition only   <li dir>/runs.pkl          (pS, pR)
    full model                <results>/fullmodel_runs.pkl        (psigma)
    experiment                <results>/fullmodel_experiment.pkl  (3 movies)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import read_table

LI_DIR = r"D:\Kasirer\results\lateral inhibition only results"
# (stage, kind, label, selector) in the order they are plotted. The stage blocks
# are separated on the axis; within a block the order is the one asked for:
# lateral inhibition only, then the full model at the two psigma values, then
# the experiment it is all being compared to.
SOURCES = [
    ("E17.5", "li", "lateral inhibition\nonly\npS=0.1, pR=0.3", dict(pS=0.1, pR=0.3)),
    ("E17.5", "full", "full model\n" r"p$\sigma$=0", dict(psigma=0.0)),
    ("E17.5", "full", "full model\n" r"p$\sigma$=0.162", dict(psigma=0.162)),
    ("E17.5", "exp", "experiment", {}),
    ("P0", "li", "lateral inhibition\nonly\npS=0.15, pR=0.25", dict(pS=0.15, pR=0.25)),
    ("P0", "full", "full model\n" r"p$\sigma$=0", dict(psigma=0.0)),
    ("P0", "full", "full model\n" r"p$\sigma$=0.162", dict(psigma=0.162)),
    ("P0", "exp", "experiment", {}),
]
# (panel title, model/LI column, experimental term)
PAIRS = [("HC:HC", "pct_HCHC_contacts_t0", "pct_HCHC_contacts"),
         ("HC:SC", "pct_HCSC_contacts_t0", "pct_HCSC_contacts")]
# Colour carries what the bar IS — model or experiment, and which stage — so the
# same source keeps its colour in both panels: (face, edge) per (is model, stage).
STYLE = {(True, "E17.5"): ("turquoise", "green"),
         (True, "P0"): ("orange", "orangered"),
         (False, "E17.5"): ("cyan", "blue"),
         (False, "P0"): ("pink", "red")}
STYLE_LABEL = {(True, "E17.5"): "model, E17.5", (True, "P0"): "model, P0",
               (False, "E17.5"): "experiment, E17.5",
               (False, "P0"): "experiment, P0"}
POINT_COLOUR = "gray"
POINT_SIZE = {"model": 40, "exp": 100}  # 3 movies deserve bigger dots than 100 runs
                                        # scatter's s is an AREA in points^2
STAT_COLOUR = "black"           # the mean star and its SEM whisker
MEAN_MARKERSIZE = 10            # Line2D points (a diameter), not scatter's s
ERROR_CAPSIZE = 10
ERROR_LINEWIDTH = 2
BAR_W = 1.0                     # neighbouring bars touch
# Every violin is drawn to the same maximum width, so a narrow distribution
# becomes a full-width flat shape and neighbours merge into each other. 80% of
# the bar width leaves a visible gap between them.
VIOLIN_W = 0.8 * BAR_W
BLOCK_GAP = 0.4                 # extra space between the E17.5 and P0 blocks


def _mean_sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else np.nan,
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def _plain(label):
    """The axis label as readable text — the console has no mathtext."""
    return (label.replace("\n", " ").replace(r"p$\sigma$", "psigma")
            .replace("$", ""))


def _select(df, stage, sel):
    """Rows of one stage matching the float parameters of a source."""
    m = df["stage"] == stage
    for col, want in sel.items():
        m &= np.isclose(df[col].astype(float), want)
    return df[m]


def _x(index, stage):
    return index + (BLOCK_GAP if stage == "P0" else 0)


def gather(li_dir=LI_DIR, results_dir=RESULTS_DIR):
    """Per source and pair: the per-array values, and the per-run values.

    Returns a list of dicts, one per (source, pair), carrying both levels so the
    caller can draw the statistic from one and the point cloud from the other.
    """
    li = pd.read_pickle(os.path.join(li_dir, "runs.pkl"))
    full = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    exp = pd.read_pickle(os.path.join(results_dir, "fullmodel_experiment.pkl"))

    out = []
    for i, (stage, kind, label, sel) in enumerate(SOURCES):
        for pair, col, term in PAIRS:
            if kind == "exp":
                g = exp[(exp["stage"] == stage) & (exp["term"] == term)]
                # a movie is already one value: no within-unit averaging to do
                per_unit = g["value"].to_numpy(float)
                per_run = per_unit
                unit, n_runs = "movies", len(g)
            else:
                df = li if kind == "li" else full
                g = _select(df, stage, sel)
                if not len(g):
                    raise SystemExit("no runs for %s / %s" % (stage, sel or kind))
                per_unit = g.groupby("initial_array")[col].mean().to_numpy(float)
                per_run = g[col].to_numpy(float)
                unit, n_runs = "arrays", len(g)
            m, s = _mean_sem(per_unit)
            style = STYLE[(kind != "exp", stage)]
            out.append(dict(index=i, stage=stage, kind=kind, label=label,
                            pair=pair, face=style[0], edge=style[1],
                            mean=m, sem=s, per_unit=per_unit, per_run=per_run,
                            n_units=len(per_unit), unit=unit, n_runs=n_runs))
    return out


def _draw(ax, data, pair, points, rng, mean_markersize=MEAN_MARKERSIZE):
    """One panel: the violin (or bar), the mean with its SEM, and the points."""
    for d in data:
        if d["pair"] != pair:
            continue
        x = _x(d["index"], d["stage"])
        is_exp = d["kind"] == "exp"
        v = d["per_run"] if points == "run" else d["per_unit"]
        # A model gets a violin and an experiment a bar, unless the row says
        # otherwise: an experiment with enough repeats has a distribution worth
        # showing, and asks for the smaller points that go with a crowded shape.
        as_violin = d.get("as_violin", not is_exp)
        point_size = d.get("point_size",
                           POINT_SIZE["exp" if is_exp else "model"])
        # ``draw_shape=False`` leaves the statistic to speak for itself: star,
        # whisker and points, with no filled shape behind them.
        shape = ("none" if not d.get("draw_shape", True)
                 else "violin" if (as_violin and len(v) > 1 and np.ptp(v) > 0)
                 else "bar")
        width = BAR_W
        if shape == "bar":
            ax.bar(x, d["mean"], BAR_W, color=d["face"], edgecolor=d["edge"],
                   linewidth=1.6, zorder=2)
        elif shape == "violin":
            width = VIOLIN_W
            body = ax.violinplot([v], positions=[x], widths=width,
                                 showmeans=False, showextrema=False,
                                 showmedians=False)["bodies"][0]
            body.set_facecolor(d["face"])
            body.set_edgecolor(d["edge"])
            body.set_linewidth(1.6)
            body.set_alpha(1.0)
            body.set_zorder(2)
        if shape != "bar":
            # a bar shows its mean as the bar top; anything else needs it drawn
            ax.plot(x, d["mean"], marker="*", markersize=mean_markersize,
                    color=STAT_COLOUR, linestyle="none", zorder=5)
        ax.errorbar(x, d["mean"], yerr=d["sem"], fmt="none", ecolor=STAT_COLOUR,
                    elinewidth=ERROR_LINEWIDTH, capsize=ERROR_CAPSIZE,
                    capthick=ERROR_LINEWIDTH, zorder=4)
        if len(v):
            # jitter stays well inside whichever shape was drawn, so a point
            # never reads as its neighbour's
            jit = rng.uniform(-width * 0.28, width * 0.28, size=len(v))
            ax.scatter(x + jit, v, s=point_size, marker=".", color=POINT_COLOUR,
                       alpha=0.8, linewidths=0, zorder=3)


def _tight_limits(data, pair, pad=0.10):
    """y range covering the bars, their error bars and every plotted point."""
    vals = []
    for d in data:
        if d["pair"] != pair:
            continue
        vals.append(np.concatenate([d["per_run"], d["per_unit"],
                                    [d["mean"] - (d["sem"] or 0),
                                     d["mean"] + (d["sem"] or 0)]]))
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    span = v.max() - v.min()
    return v.min() - pad * span, v.max() + pad * span


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", choices=("run", "array", "none"), default="run",
                    help="what each scattered point is (default: one model run)")
    ap.add_argument("--tight-y", action="store_true",
                    help="crop each panel to its data instead of starting at 0")
    ap.add_argument("--exp-no-bar", dest="exp_no_bar", action="store_true",
                    help="draw the experiment as star + SEM + points only, with no bar behind them")
    ap.add_argument("--mean-markersize", type=float, default=MEAN_MARKERSIZE,
                    help="size of the star marking a violin's mean, in Line2D"
                         " points (a diameter, not scatter's s)")
    ap.add_argument("--li-dir", default=LI_DIR)
    ap.add_argument("--out", default=None,
                    help="output path; the extension is ignored, both .png and"
                         " .svg are written")
    ap.add_argument("--seed", type=int, default=0, help="jitter seed")
    a = ap.parse_args()

    data = gather(li_dir=a.li_dir)
    if a.exp_no_bar:
        for d in data:
            if d["kind"] == "exp":
                d["draw_shape"] = False

    print("  %-6s %-34s %-6s %8s %8s %6s %6s"
          % ("stage", "source", "pair", "mean", "SEM", "units", "runs"))
    for d in data:
        print("  %-6s %-34s %-6s %8.3f %8.3f %6d %6d"
              % (d["stage"], _plain(d["label"]), d["pair"],
                 d["mean"], d["sem"], d["n_units"], d["n_runs"]))

    # hspace=0: the panels share one boundary line rather than floating apart
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11.5, 8.0),
                             gridspec_kw=dict(hspace=0))
    for ax, (pair, _c, _t) in zip(axes, PAIRS):
        _draw(ax, data, pair, a.points, np.random.default_rng(a.seed),
              mean_markersize=a.mean_markersize)
        ax.set_ylabel("%% of %s contacts" % pair, fontsize=10.5)
        if a.tight_y:
            ax.set_ylim(*_tight_limits(data, pair))
        else:
            ax.set_ylim(0, _tight_limits(data, pair)[1])
    # every panel closed on all four sides, and no ticks hanging off the shared
    # boundary into the panel below it
    axes[0].tick_params(axis="x", bottom=False)

    axis = axes[-1]
    axis.set_xticks([_x(i, s[0]) for i, s in enumerate(SOURCES)])
    axis.set_xticklabels([s[2] for s in SOURCES], fontsize=8.5)
    axis.set_xlim(_x(0, SOURCES[0][0]) - 0.75,
                  _x(len(SOURCES) - 1, SOURCES[-1][0]) + 0.75)

    # name the two stage blocks above the figure rather than repeating them in
    # every tick label
    for stage in ("E17.5", "P0"):
        xs = [_x(i, s[0]) for i, s in enumerate(SOURCES) if s[0] == stage]
        axes[0].annotate(stage, xy=(np.mean(xs), 1.10),
                         xycoords=("data", "axes fraction"), ha="center",
                         va="bottom", fontsize=12, fontweight="bold")

    # with --exp-no-bar nothing coloured is drawn for the experiment, so its
    # swatches would point at shapes that are not there
    keys = [k for k in STYLE_LABEL if k[0] or not a.exp_no_bar]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=STYLE[k][0],
                             edgecolor=STYLE[k][1], linewidth=1.6)
               for k in keys]
    names = [STYLE_LABEL[k] for k in keys]
    if a.exp_no_bar:
        handles.append(plt.Line2D([], [], marker="*", linestyle="none",
                                  color=STAT_COLOUR,
                                  markersize=a.mean_markersize))
        names.append("experiment (mean)")
    if a.points != "none":
        for kind, name in (("model", "one model run" if a.points == "run"
                            else "one array"), ("exp", "one movie")):
            handles.append(plt.Line2D(
                [], [], marker=".", linestyle="none", color=POINT_COLOUR,
                alpha=0.8, markersize=np.sqrt(POINT_SIZE[kind]) * 1.8))
            names.append(name)
    # below the panels: inside them it would sit on the E17.5 bars
    fig.legend(handles, names, frameon=False, fontsize=9.5, ncol=len(names),
               loc="lower center", bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Neighbour-pair composition of the tissue"
                 "  (violin = model runs, %s;"
                 "  star = mean over arrays/movies, whisker = SEM)"
                 % ("experiment as points" if a.exp_no_bar
                    else "bar = experiment"),
                 fontsize=11.5, y=0.985)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.885, bottom=0.185)

    # PNG to look at, SVG to edit and to submit — same figure, one render
    stem = os.path.splitext(a.out or os.path.join(RESULTS_DIR,
                                                  "neighbor_pairs"))[0]
    for ext in ("png", "svg"):
        path = "%s.%s" % (stem, ext)
        fig.savefig(path, dpi=200)
        print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
