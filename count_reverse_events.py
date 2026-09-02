"""Reverse differentiation (HC -> SC) across every full-model run.

    python count_reverse_events.py --workers 4        # full pass (resumable)
    python count_reverse_events.py --limit 8          # quick check
    python count_reverse_events.py --merge-only       # just update the tables

Counts only reversals that STICK: a cell is a reverse event if it is a
non-boundary SC in the final frame and was a HC earlier in the window. That is
the forward rule of build_fullmodel_table.differentiation_events with the
direction flipped, so the two counts are defined the same way and can be put in
one row. Transient dips below threshold that recover are deliberately not
counted (in the sample that motivated this there were none: every reversal seen
was permanent).

WINDOW. From the run's own t0 — the frame the scoring matches to the experiment —
to the last frame, the same window the forward events use. Note the t0 read back
from the saved table can sit a few ulp below the history stamp it names, so the
frame is matched with a tolerance; without it the t0 frame is dropped and every
crossing in the first interval is missed (see count_reverse_differentiation).

Writes to <results>/:
    fullmodel_reverse_runs.pkl     one row per run
    fullmodel_reverse_events.pkl   one row per reversal
    fullmodel_reverse_runs.csv     incremental; an interrupted pass resumes here
and merges into the published tables:
    fullmodel_runs.pkl / sheet "runs"   + n_reverse_events, reverse_per_100_events
    fullmodel_pT.pkl   / sheet "pT"     + the per-point mean and SEM, over the
                                          runs that did NOT collapse (see merge)
    fullmodel_tables.xlsx sheet "reverse_events"  the per-event detail
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR, load_history_file, get_time_points
from build_experimental_tables import read_table, to_output_names, add_sheets
from count_reverse_differentiation import (TYPE_BY, THRESHOLD, T0_TOL,
                                           mirrored_reverse)

RUNS_PKL = "fullmodel_reverse_runs"
EVENTS_PKL = "fullmodel_reverse_events"


def one_run(args):
    """(row, events) for one run; ``error`` is set instead of raising."""
    name, t0, threshold = args
    row = dict(model_name=name, t0=float(t0), error="")
    try:
        history = load_history_file(name)
        stamps = np.asarray(get_time_points(history), float)
        stamps = stamps[stamps >= float(t0) - T0_TOL]
        frames = []
        for t in stamps:
            s = history.retrieve(float(t))
            s.arrange_sheet_from_history()
            frames.append(s.face_df.set_index("id")[TYPE_BY])
        events = mirrored_reverse(history, stamps, frames, threshold)
        row.update(n_frames=int(stamps.size),
                   t_window_start=float(stamps[0]) if stamps.size else np.nan,
                   t_window_end=float(stamps[-1]) if stamps.size else np.nan,
                   n_reverse_events=len(events))
        rows = [dict(model_name=name, cell_id=cid, t_reverted=t,
                     dt_since_t0=t - float(t0)) for cid, t in events]
        return row, rows
    except Exception as exc:                            # noqa: BLE001
        row.update(n_frames=0, t_window_start=np.nan, t_window_end=np.nan,
                   n_reverse_events=np.nan,
                   error="%s: %s" % (type(exc).__name__, exc))
        return row, []


def compute(runs, workers, limit, threshold, csv_path, ev_csv_path):
    """Parallel pass over the runs, appending each result as it lands.

    Both the per-run row and its events go to disk immediately, so an interrupted
    pass resumes without losing either.
    """
    done = set()
    if os.path.isfile(csv_path):
        done = set(pd.read_csv(csv_path)["model_name"])
        print("  resuming: %d run(s) already measured" % len(done))
    todo = [(r["model_name"], r["t0"], threshold) for _i, r in runs.iterrows()
            if r["model_name"] not in done]
    if limit:
        todo = todo[:limit]
    print("  %d run(s) to measure on %d worker(s)" % (len(todo), workers))
    if not todo:
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_run, t): t[0] for t in todo}
        for k, fut in enumerate(as_completed(futures), start=1):
            row, evs = fut.result()
            pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False,
                                       header=not os.path.isfile(csv_path))
            if evs:
                pd.DataFrame(evs).to_csv(ev_csv_path, mode="a", index=False,
                                         header=not os.path.isfile(ev_csv_path))
            if k % 25 == 0 or k == len(todo):
                print("   %4d/%d" % (k, len(todo)), flush=True)


def merge(reverse_runs, events, results_dir=RESULTS_DIR):
    """Fold the counts into the published run and per-point tables."""
    runs = read_table(os.path.join(results_dir, "fullmodel_runs.pkl"))
    counts = reverse_runs.set_index("model_name")["n_reverse_events"]
    runs["n_reverse_events"] = runs["model_name"].map(counts)
    fwd = runs["n_differentiation_events"].replace(0, np.nan)
    runs["reverse_per_100_events"] = 100.0 * runs["n_reverse_events"] / fwd

    ps = read_table(os.path.join(results_dir, "fullmodel_psigma.pkl"))
    # COLLAPSED RUNS ARE EXCLUDED from the per-point numbers. When the pattern
    # collapses the whole hair-cell population reverts at once — 372 reversals in
    # one run against a median of 1 — so a single collapsed array swamps its
    # column and hides the real trend: at E17.5 pT=0.150 three collapsed repeats
    # carried 1105 of the point's 1130 reversals. Collapsed runs keep their own
    # per-run count in the runs sheet; only the aggregate drops them, exactly as
    # psigma_table's drop_collapsed does for the scores.
    healthy = runs[~runs["collapsed"].astype(bool)]
    # per array first, then across arrays — the same hierarchy every other
    # per-point number in this table uses
    agg = []
    for (stage, psigma), g in healthy.groupby(["stage", "psigma"]):
        per_array = g.groupby("initial_array")["n_reverse_events"].mean()
        v = per_array.to_numpy(float); v = v[np.isfinite(v)]
        agg.append(dict(stage=stage, psigma=float(psigma),
                        n_reverse_events_mean=float(v.mean()) if v.size else np.nan,
                        n_reverse_events_sem=(float(v.std(ddof=1) / np.sqrt(v.size))
                                              if v.size > 1 else np.nan),
                        n_reverse_events_total=float(g["n_reverse_events"].sum()),
                        n_runs_with_reverse=int((g["n_reverse_events"] > 0).sum()),
                        n_runs_reverse_measured=int(g["n_reverse_events"].notna().sum())))
    agg = pd.DataFrame(agg)
    ps = ps.drop(columns=[c for c in agg.columns
                          if c in ps.columns and c not in ("stage", "psigma")])
    ps = ps.merge(agg, on=["stage", "psigma"], how="left")

    runs_out, ps_out = to_output_names(runs), to_output_names(ps)
    runs_out.to_pickle(os.path.join(results_dir, "fullmodel_runs.pkl"))
    ps_out.to_pickle(os.path.join(results_dir, "fullmodel_psigma.pkl"))
    events.to_pickle(os.path.join(results_dir, EVENTS_PKL + ".pkl"))
    add_sheets(os.path.join(results_dir, "fullmodel_tables.xlsx"),
               {"runs": runs_out, "pT": ps_out, "reverse_events": events})
    return runs_out, ps_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--merge-only", action="store_true",
                    help="skip the pass and merge what the CSV already holds")
    a = ap.parse_args()

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    csv_path = os.path.join(RESULTS_DIR, RUNS_PKL + ".csv")
    ev_csv_path = os.path.join(RESULTS_DIR, EVENTS_PKL + ".csv")
    print("full-model runs: %d" % len(runs))

    if not a.merge_only:
        compute(runs, a.workers, a.limit, a.threshold, csv_path, ev_csv_path)

    if not os.path.isfile(csv_path):
        raise SystemExit("no results yet")
    reverse_runs = pd.read_csv(csv_path).drop_duplicates("model_name", keep="last")
    reverse_runs.to_pickle(os.path.join(RESULTS_DIR, RUNS_PKL + ".pkl"))

    bad = reverse_runs["error"].fillna("").astype(str) != ""
    print("\n  measured %d run(s); %d failed" % (len(reverse_runs), int(bad.sum())))
    if bad.any():
        for _i, r in reverse_runs[bad].head(5).iterrows():
            print("     %s: %s" % (r["model_name"][-44:], r["error"][:70]))

    events = (pd.read_csv(ev_csv_path).drop_duplicates()
              if os.path.isfile(ev_csv_path)
              else pd.DataFrame(columns=["model_name", "cell_id", "t_reverted",
                                         "dt_since_t0"]))
    print("  %d reverse event(s) in %d run(s)"
          % (len(events), events["model_name"].nunique() if len(events) else 0))

    runs_out, ps_out = merge(reverse_runs, events)
    print("  merged into fullmodel_runs (%d x %d) and fullmodel_pT (%d x %d)"
          % (*runs_out.shape, *ps_out.shape))
    # the reverse columns are over the runs that did not collapse, so the run
    # count printed beside them must be that one, not the point's total
    print("\n  %-6s %-7s %6s %8s %10s %10s %8s"
          % ("stage", "pT", "runs", "healthy", "reverse", "per run", "with rev"))
    for _i, r in ps_out.sort_values(["stage", "pT"]).iterrows():
        if not np.isfinite(r.get("n_reverse_events_total", np.nan)):
            print("  %-6s %-7.3f %6d %8d %10s"
                  % (r["stage"], r["pT"], r["n_runs"], 0, "all collapsed"))
            continue
        print("  %-6s %-7.3f %6d %8d %10.0f %10.2f %8d"
              % (r["stage"], r["pT"], r["n_runs"], r["n_runs_reverse_measured"],
                 r["n_reverse_events_total"], r["n_reverse_events_mean"],
                 r["n_runs_with_reverse"]))


if __name__ == "__main__":
    main()
