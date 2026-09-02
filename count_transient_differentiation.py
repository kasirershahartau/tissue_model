"""How many "differentiation events" are really a hair cell recovering?

    python count_transient_differentiation.py --workers 4
    python count_transient_differentiation.py --pT 0 0.162 --stage P0

THE FALSE POSITIVE. build_fullmodel_table.differentiation_events takes every
non-boundary HC of the final frame and walks back through its uninterrupted run
of HC frames; the first of those frames is the differentiation. A cell already
HC at t0 is meant to be skipped — the walk reaches frame 0 and the cell is
dropped — but that only works if it stayed a HC throughout. A cell that was HC
at t0, dipped below the threshold, and came back stops the walk at the
RE-crossing, so it is recorded as a differentiation it never underwent.

This counts those: cells the forward rule calls an event whose delta was already
above threshold at t0. They are the mirror image of the reverse events — same
transient, but ending as a HC rather than an SC, which is why the reverse count
never sees them.

Nothing here is merged into the tables. It measures how much the published
forward counts overstate differentiation, and the answer belongs in the text
before it belongs in a column.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type)
from build_experimental_tables import read_table
from count_reverse_differentiation import TYPE_BY, THRESHOLD, T0_TOL


def mirrored_forward(history, stamps, frames, threshold, type_by=TYPE_BY):
    """The forward rule's events, each tagged with the cell's state at t0.

    This keeps the rule as it was BEFORE the transient guard — final-frame
    non-boundary HCs, walked back through their uninterrupted HC run, dropping
    only those whose walk reaches frame 0 — because the recoveries are what it
    exists to count. build_fullmodel_table.differentiation_events now rejects
    them outright, so the two deliberately disagree by exactly ``n_transient``.

    Each event carries whether the cell was ALREADY a HC in the first frame of
    the window; flagged means a recovery, not a differentiation.
    """
    if stamps.size == 0:
        return []
    final = history.retrieve(float(stamps[-1]))
    final.arrange_sheet_from_history()
    _idx, final_hc_ids = get_non_boundary_cell_ids_from_type(
        final, cell_type="HC", type_by=type_by, threshold=threshold)

    def is_hc(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v)) \
            and v > threshold

    out, last = [], stamps.size - 1
    for cid in final_hc_ids:
        f = last
        while f > 0 and is_hc(frames[f - 1].get(cid)):
            f -= 1
        if f == 0:                       # HC for the whole window: not an event
            continue
        out.append((int(cid), float(stamps[f]), bool(is_hc(frames[0].get(cid)))))
    return out


def one_run(args):
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
        events = mirrored_forward(history, stamps, frames, threshold)
        transient = [e for e in events if e[2]]
        row.update(n_frames=int(stamps.size), n_forward_events=len(events),
                   n_transient=len(transient),
                   n_true_differentiation=len(events) - len(transient),
                   transient_cell_ids=";".join(str(e[0]) for e in transient))
    except Exception as exc:                            # noqa: BLE001
        row.update(n_frames=0, n_forward_events=np.nan, n_transient=np.nan,
                   n_true_differentiation=np.nan, transient_cell_ids="",
                   error="%s: %s" % (type(exc).__name__, exc))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pT", type=float, nargs="+", default=[0.0, 0.162])
    ap.add_argument("--stage", nargs="+", default=["E17.5", "P0"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="fullmodel_transient_differentiation.csv")
    a = ap.parse_args()

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    sel = runs[runs["stage"].isin(a.stage)
               & runs["psigma"].apply(lambda p: any(np.isclose(p, v)
                                                    for v in a.pT))]
    # collapsed runs are excluded for the same reason the reverse aggregates
    # exclude them: the whole population changes state at once
    sel = sel[~sel["collapsed"].astype(bool)]
    if a.limit:
        sel = sel.head(a.limit)
    print("measuring %d run(s) on %d worker(s)" % (len(sel), a.workers))

    path = a.out if os.path.isabs(a.out) else os.path.join(RESULTS_DIR, a.out)
    # each row is appended as it lands, so a pass over the whole sweep survives
    # an interruption instead of starting again
    done = set()
    if os.path.isfile(path):
        done = set(pd.read_csv(path)["model_name"])
        print("  resuming: %d run(s) already measured" % len(done))
    todo = [(r["model_name"], r["t0"], a.threshold) for _i, r in sel.iterrows()
            if r["model_name"] not in done]
    print("  %d run(s) left to measure" % len(todo))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one_run, t) for t in todo]
        for k, fut in enumerate(as_completed(futures), start=1):
            pd.DataFrame([fut.result()]).to_csv(
                path, mode="a", index=False, header=not os.path.isfile(path))
            if k % 20 == 0 or k == len(todo):
                print("   %4d/%d" % (k, len(todo)), flush=True)

    rows = pd.read_csv(path).drop_duplicates("model_name", keep="last")
    out = rows.merge(
        sel[["model_name", "stage", "psigma", "initial_array", "repeat",
             "n_differentiation_events"]], on="model_name", how="left")
    out = out.rename(columns={"psigma": "pT"})
    bad = out["error"].fillna("").astype(str) != ""
    if bad.any():
        print("  %d run(s) failed" % int(bad.sum()))
    ok = out[~bad]

    print("\n  %-6s %-6s %5s %10s %11s %11s %9s"
          % ("stage", "pT", "runs", "forward", "transient", "true diff", "% transient"))
    for stage in a.stage:
        for pt in a.pT:
            g = ok[(ok.stage == stage) & np.isclose(ok.pT, pt)]
            if not len(g):
                continue
            f = float(g.n_forward_events.sum()); t = float(g.n_transient.sum())
            print("  %-6s %-6.3f %5d %10.0f %11.0f %11.0f %8.2f%%"
                  % (stage, pt, len(g), f, t, f - t, 100 * t / f if f else np.nan))
            print("       per run: forward %.2f, transient %.2f; runs with one %d"
                  % (g.n_forward_events.mean(), g.n_transient.mean(),
                     int((g.n_transient > 0).sum())))
    # the table's own forward count should equal the rule reproduced here
    d = (ok["n_forward_events"] - ok["n_differentiation_events"]).abs()
    print("\n  reproduces the table's n_differentiation_events on %d of %d run(s)"
          % (int((d == 0).sum()), len(ok)))

    # the CSV stays the raw append log that resume reads; the joined table,
    # which carries the stage and pT each run belongs to, goes beside it
    pkl = os.path.splitext(path)[0] + ".pkl"
    out.to_pickle(pkl)
    print("wrote %s\n      %s" % (path, pkl))


if __name__ == "__main__":
    main()
