"""Face stress over time at psigma = 0, by SC HC-neighbour count and for HCs.

    python face_stress_ps0_groups.py --interval 0.5 --workers 4
    python face_stress_ps0_groups.py --repeats 1 2 3 --stage P0

Writes to <results>/:
    face_stress_ps0.png            the figure
    face_stress_ps0_tables.xlsx    sheets: per_run, per_array, summary
    face_stress_ps0_{per_run,per_array,summary}.pkl

FOUR GROUPS, reassigned at every sampled frame: SC with 0 HC neighbours, SC with
1, SC with >= 2, and HC (any neighbour count). HCs are pooled across their own
neighbour bins by an n-weighted mean of the per-bin means, which is exactly the
mean over all HCs in that frame.

THE AVERAGING IS HIERARCHICAL, in the order asked for:
  1. per RUN (one array, one repeat): mean stress over the cells of each group,
     at each sampled time — this is what face_stress_over_time.one_run returns;
  2. per ARRAY: mean over that array's repeats;
  3. across ARRAYS: mean of those per-array curves, and the SEM over arrays.
So the error bars describe array-to-array variation, with the lateral-inhibition
seed noise averaged out first. They are NOT cell-to-cell spread, which is much
larger and would say nothing about how reproducible the curve is.

A COMMON TIME WINDOW. Runs stop at different times (psigma = 0 E17.5 reaches
t ~ 23, P0 ~ 14), so a grid point that only the long runs reach would be an
average over a shifting subset of arrays and its SEM would not be comparable to
its neighbours'. The summary is therefore truncated to the last time every array
still contributes; the per-run and per-array sheets keep everything.

BOTH EFFECTOR SETS are reported, as in face_stress_over_time:
  contractility - ContractilityPerimeterElasticity alone, which is what
                  run_model.stress_effectors gates on, so it is the stress
                  psigma actually compares against;
  all           - plus area elasticity and bending: the total mechanical stress,
                  an order of magnitude larger and dominated by the area term.
The figure shows both, one row each.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR
from face_stress_over_time import one_run
from build_run_table import array_of
from run_psigma_repeats import REPEAT_PREFIX
from score_psigma_pooled import run_name

STAGES = ("E17.5", "P0")
N_ARRAYS = 10
# label -> (cell_type, hc_neighbour bins to pool)
GROUPS = (("SC, 0 HC nb", "SC", (0,)),
          ("SC, 1 HC nb", "SC", (1,)),
          ("SC, >=2 HC nb", "SC", (2,)),
          ("HC (any nb)", "HC", (0, 1, 2)))
COLOUR = {"SC, 0 HC nb": "tab:blue", "SC, 1 HC nb": "tab:orange",
          "SC, >=2 HC nb": "tab:green", "HC (any nb)": "tab:red"}


def base_name(psigma):
    return "face_stress_ps0" if float(psigma) == 0.0 else "face_stress_ps%.3f" % psigma


def collect(stage, psigma, repeats, interval, type_by, threshold, workers):
    """Per-run group means: one row per (run, time, effector set, group)."""
    tasks, meta = [], {}
    for rep in repeats:
        for i in range(N_ARRAYS):
            nm = run_name(stage, psigma, REPEAT_PREFIX[rep], i)
            if not os.path.isdir(os.path.join(RESULTS_DIR, nm)):
                continue
            tasks.append((nm, interval, type_by, threshold))
            meta[nm] = (rep, array_of(nm))
    print("  %s: %d run(s) over repeats %s" % (stage, len(tasks), list(repeats)),
          flush=True)
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        results = []
        per_pool = max(1, 3 * workers)                  # recycled pool
        for s in range(0, len(tasks), per_pool):
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results.extend(ex.map(one_run, tasks[s:s + per_pool]))
            print("    %d/%d" % (min(s + per_pool, len(tasks)), len(tasks)), flush=True)
    else:
        results = [one_run(t) for t in tasks]

    rows = []
    for name, run_rows, err in results:
        if err:
            print("    FAILED %-46s %s" % (name[:46], err), flush=True)
            continue
        rep, arr = meta[name]
        for (t, eff, ctype, nb, n, mean, std, fpos) in run_rows:
            if eff == "geometry":
                continue
            rows.append(dict(stage=stage, model_name=name, repeat=rep,
                             initial_array=arr, time=t, effectors=eff,
                             cell_type=ctype, hc_neighbors=nb, n_cells=n,
                             mean_stress=mean, std_stress=std, frac_positive=fpos))
    return pd.DataFrame(rows)


def to_groups(df):
    """Collapse the six (cell_type, bin) rows into the four requested groups.

    HC bins are pooled by an n-weighted mean, which reproduces the plain mean
    over every HC in the frame; the SC groups are single bins and pass through."""
    out = []
    for label, ctype, bins in GROUPS:
        sel = df[(df["cell_type"] == ctype) & (df["hc_neighbors"].isin(bins))].copy()
        if not len(sel):
            continue
        sel["w"] = sel["mean_stress"] * sel["n_cells"]
        g = (sel.groupby(["stage", "model_name", "repeat", "initial_array",
                          "time", "effectors"], as_index=False)
             .agg(w=("w", "sum"), n_cells=("n_cells", "sum")))
        g["mean_stress"] = g["w"] / g["n_cells"]
        g["group"] = label
        out.append(g.drop(columns=["w"]))
    return pd.concat(out, ignore_index=True)


def aggregate(per_run):
    """per-repeat -> per-array -> across-array mean and SEM."""
    per_array = (per_run.groupby(["stage", "effectors", "group", "initial_array",
                                  "time"], as_index=False)
                 .agg(mean_stress=("mean_stress", "mean"),
                      n_repeats=("repeat", "nunique"),
                      n_cells=("n_cells", "mean")))
    summary = (per_array.groupby(["stage", "effectors", "group", "time"],
                                 as_index=False)
               .agg(mean=("mean_stress", "mean"),
                    sd=("mean_stress", lambda s: s.std(ddof=1)),
                    n_arrays=("initial_array", "nunique"),
                    n_cells=("n_cells", "mean")))
    summary["sem"] = summary["sd"] / np.sqrt(summary["n_arrays"])
    return per_array, summary


def common_window(summary):
    """Keep only times where every array still contributes, per stage."""
    keep = []
    for stage, g in summary.groupby("stage"):
        full = g["n_arrays"].max()
        ok = g[g["n_arrays"] >= full]["time"]
        t_max = ok.max() if len(ok) else np.nan
        keep.append(g[g["time"] <= t_max])
        print("  %-6s all %d arrays present up to t = %.2f (of %.2f sampled)"
              % (stage, full, t_max, g["time"].max()))
    return pd.concat(keep, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, default=0.0,
                    help="which sweep point to read (default 0). A non-zero value "
                         "also draws the gate threshold as a dashed line, since "
                         "psigma IS the stress at which the Hill gate is half open.")
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=0.355079)
    ap.add_argument("--effectors", nargs="+", default=["contractility"],
                    choices=["contractility", "all"],
                    help="which effector set(s) to PLOT; both are always computed "
                         "and saved (default: contractility, the set the stress "
                         "gate actually reads)")
    ap.add_argument("--t-min", dest="t_min", type=float, default=2.0,
                    help="start the plot here, past the initial homogeneous "
                         "relaxation (default 2.0)")
    ap.add_argument("--reuse", action="store_true",
                    help="re-plot from the saved per_run table, sampling nothing")
    a = ap.parse_args()

    out_base = base_name(a.psigma)
    per_run_path = os.path.join(RESULTS_DIR, out_base + "_per_run.pkl")
    if a.reuse and os.path.isfile(per_run_path):
        per_run = pd.read_pickle(per_run_path)
        print("re-using %s: %d row(s)" % (os.path.basename(per_run_path), len(per_run)))
    else:
        print("=" * 78)
        print("FACE STRESS OVER TIME | psigma = %.3f | repeats %s | every %.2g t"
              % (a.psigma, a.repeats, a.interval))
        print("=" * 78)
        raw = pd.concat([collect(s, a.psigma, a.repeats, a.interval, a.type_by,
                                 a.threshold, a.workers) for s in a.stage],
                        ignore_index=True)
        if not len(raw):
            raise SystemExit("no data collected")
        per_run = to_groups(raw)
        per_run.to_pickle(per_run_path)

    per_array, summary = aggregate(per_run)
    print()
    summary = common_window(summary)

    # Mark what the figure shows so the sheet and the figure cannot drift apart,
    # without throwing away the rows the figure leaves out.
    full_by_stage = summary.groupby("stage")["n_arrays"].transform("max")
    summary["plotted"] = ((summary["time"] >= a.t_min)
                          & (summary["effectors"].isin(a.effectors))
                          & (summary["n_arrays"] >= full_by_stage))

    for name, frame in (("per_run", per_run), ("per_array", per_array),
                        ("summary", summary)):
        frame.to_pickle(os.path.join(RESULTS_DIR, "%s_%s.pkl" % (out_base, name)))
    xlsx = os.path.join(RESULTS_DIR, out_base + "_tables.xlsx")
    try:
        with pd.ExcelWriter(xlsx) as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            per_array.to_excel(w, sheet_name="per_array", index=False)
            per_run.to_excel(w, sheet_name="per_run", index=False)
    except Exception as exc:                            # noqa: BLE001
        print("  xlsx failed (%s: %s); pickles are written" % (type(exc).__name__, exc))

    stages = [s for s in STAGES if s in set(summary["stage"])]
    effs = list(a.effectors)
    plotted = summary[summary["time"] >= a.t_min]
    fig, axes = plt.subplots(len(effs), len(stages),
                             figsize=(6.6 * len(stages), 4.8 * len(effs)),
                             squeeze=False)
    for r, eff in enumerate(effs):
        for c, stage in enumerate(stages):
            ax = axes[r][c]
            full_n = summary[summary["stage"] == stage]["n_arrays"].max()
            for label, _ct, _b in GROUPS:
                s = plotted[(plotted["stage"] == stage) & (plotted["effectors"] == eff)
                            & (plotted["group"] == label)].sort_values("time")
                if not len(s):
                    continue
                # Early on a group may not exist in every array — no HCs yet, or
                # no SC with two HC neighbours. Those means are over a SUBSET of
                # arrays, so their SEM is not comparable to the rest and they are
                # drawn faded and without error bars rather than silently mixed in.
                solid = s[s["n_arrays"] >= full_n]
                partial = s[s["n_arrays"] < full_n]
                ax.errorbar(solid["time"], solid["mean"], yerr=solid["sem"], fmt="o-",
                            color=COLOUR[label], ms=3.5, lw=1.2, capsize=2.5,
                            elinewidth=0.9, label=label)
                if len(partial):
                    ax.plot(partial["time"], partial["mean"], "o", ms=3.0,
                            color=COLOUR[label], alpha=0.30)
            ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
            if a.psigma > 0 and eff == "contractility":
                # psigma is the Hill gate's half-max, so a group above this line
                # has its differentiation rate more than half enabled by stress
                # and a group below it less than half. The gate reads the
                # contractility stress, so the line belongs only on that panel.
                ax.axhline(a.psigma, color="k", lw=1.3, ls="--", alpha=0.8,
                           label="$p_\\sigma$ = %.3f (gate half-max)" % a.psigma)
            ax.set_xlabel("time")
            ax.set_ylabel("mean face stress (%s)" % eff)
            ax.set_title("%s — %s effectors" % (stage, eff), fontsize=11)
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("Face stress from $t$ = %g at $p_\\sigma$ = %.3f — mean over arrays "
                 "of (mean over %d repeats), error bars = SEM over %d arrays"
                 % (a.t_min, a.psigma, len(a.repeats),
                    int(summary["n_arrays"].max())), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS_DIR, out_base + ".png"), dpi=170,
                bbox_inches="tight")

    print("\n  summary %d rows; groups %s"
          % (len(summary), sorted(summary["group"].unique())))
    print("\nwrote %s" % os.path.join(RESULTS_DIR, out_base + ".png"))
    print("      %s" % xlsx)
    print("      %s_{per_run,per_array,summary}.pkl" % out_base)


if __name__ == "__main__":
    main()
