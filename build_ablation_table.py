"""Distance-from-ablation data -> three tables (Excel + pandas pickles).

    python build_ablation_table.py                     # ablate_from_strict.json
    python build_ablation_table.py --json ablate_psigma_repeats_v2.json

Writes to <results>/:
    ablation_tables.xlsx     sheets: overall, runs, events, plus any
                             distance_vs_experiment* sheet already in the
                             workbook (see ABLATION_SHEETS)
    ablation_overall.pkl     one row per (stage, psigma)
    ablation_runs.pkl        one row per ablation
    ablation_events.pkl      one row per differentiating cell

THE SOURCE. ablate_from_strict.py forks each array at a time t* after which the
UNABLATED tissue produced zero differentiation events for 5 time units, then
ablates one random HC. The event-free window is the counterfactual, so every
event in these tables is caused by the ablation and no control column is needed.
Runs that died are carried in ``runs`` with their error string and excluded from
every statistic.

TWO KINDS OF MEAN, both in ``overall``. ``dist_mean`` pools all events and
answers "how far is a typical event"; ``run_dist_mean`` averages each run's mean
first and answers "how far does a typical ABLATION reach". They differ whenever
runs contribute unequal numbers of events, which is exactly the case here (0-1
events per run at psigma 0 against 5-12 at 0.162), and the run-level one is the
right denominator for comparing psigma values.

DISTANCES are periodic minimum-image, measured on the PRE-ablation frame, and
cells are tracked by ``id`` — ``unique_id`` is recompacted when a face is
removed, which is precisely what an ablation does. ``distance_rel_half_box``
divides by the half-box diagonal, so 1.0 means "as far away as the geometry
allows": that is what distinguishes a local response from a tissue-wide one.

EACH EVENT ALSO CARRIES ITS PRE-ABLATION CONTEXT — HC-neighbour count, area, and
whether the cell touched the ablated one. The HC-neighbour count is the same
quantity score 2 is built from, so these rows say whether ablation-triggered
differentiation obeys the same neighbour rule as spontaneous differentiation.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, get_non_boundary_cell_ids_from_type,
                             calc_contact_with_neighbors_from_type)
from run_model import load_sheet_from_file
from ablate_single_hc import TYPE_BY, THRESHOLD, BOX
from build_experimental_tables import carried_over_sheets, to_output_names

# added by compare_ablation_distance_to_experiment.py: (pickle, sheet)
ABLATION_SHEETS = [("ablation_distance_vs_experiment", "distance_vs_experiment"),
                   ("ablation_distance_vs_experiment_runs",
                    "distance_vs_experiment_runs"),
                   ("ablation_distance_vs_experiment_exp",
                    "distance_vs_experiment_exp")]

IN_JSON = "ablate_from_strict.json"
CUTOFFS = (1.5, 2.5, 4.0, 6.0)
MAX_NB = 6                        # see build_fullmodel_table: exhaustive by design


def _mean_sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else np.nan,
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def pre_context(rec):
    """Per-cell pre-ablation context, keyed by ``id``, plus frame-level numbers.

    One sheet load per ablation. Everything here is measured BEFORE the cell was
    removed, so it describes the tissue the ablation acted on rather than its
    response."""
    pre = load_sheet_from_file(os.path.join(RESULTS_DIR, rec["source"]),
                              time_point=rec["t_strict"], force_periodic_box=BOX)
    pre.geom.update_all(pre)
    all_idx, _ = get_non_boundary_cell_ids_from_type(
        pre, cell_type="all", type_by=TYPE_BY, threshold=THRESHOLD)
    hc_idx, _ = get_non_boundary_cell_ids_from_type(
        pre, cell_type="HC", type_by=TYPE_BY, threshold=THRESHOLD)
    n_hc_nb, _ = calc_contact_with_neighbors_from_type(
        pre, "all", "HC", type_by=TYPE_BY, threshold=THRESHOLD)
    ids = pre.face_df["id"].to_numpy(int)
    per_cell = {}
    for pos, k in enumerate(np.asarray(all_idx, int)):
        per_cell[int(ids[k])] = dict(n_HC_neighbours_pre=int(np.asarray(n_hc_nb)[pos]),
                                     area_pre=float(pre.face_df["area"].to_numpy(float)[k]),
                                     delta_pre=float(pre.face_df[TYPE_BY].to_numpy(float)[k]))
    # direct neighbours of the ablated cell, by id
    label = int(rec["ablated_label"])
    ablated_pos = int(np.flatnonzero(pre.face_df.index.values == label)[0])
    A = np.asarray(pre.get_contact_matrix()) > 0
    np.fill_diagonal(A, False)
    touching = {int(ids[j]) for j in np.flatnonzero(A[ablated_pos])}
    Lx = float(getattr(pre, "Lx", BOX[0])); Ly = float(getattr(pre, "Ly", BOX[1]))
    frame = dict(ablated_cell_id=int(ids[ablated_pos]),
                 ablated_x=float(pre.face_df.loc[label, "x"]),
                 ablated_y=float(pre.face_df.loc[label, "y"]),
                 ablated_area=float(pre.face_df.loc[label, "area"]),
                 n_ablated_neighbours=len(touching),
                 n_cells_pre=int(np.size(all_idx)), n_HC_pre=int(np.size(hc_idx)),
                 n_SC_pre=int(np.size(all_idx) - np.size(hc_idx)),
                 Lx=Lx, Ly=Ly, half_box_diagonal=0.5 * float(np.hypot(Lx, Ly)))
    return per_cell, touching, frame


def build(records, with_context=True):
    ev_rows, run_rows = [], []
    for rec in records:
        base = dict(stage=rec.get("stage"), psigma=rec.get("psigma"),
                    initial_array=rec.get("array"), repeat=rec.get("repeat"),
                    run_folder=rec.get("folder"),
                    source_run=rec.get("source"),
                    # a time on the SOURCE run's clock: where the fork was taken.
                    # NOT comparable with t_differentiated, which is the fork's own.
                    t_strict_in_source=rec.get("t_strict"),
                    ablated_label=rec.get("ablated_label"), seed=rec.get("seed"),
                    error=rec.get("error") or "")
        if rec.get("error"):
            run_rows.append(dict(base, n_events=np.nan))
            continue
        events = rec.get("events") or []
        per_cell, touching, frame = ({}, set(), {})
        if with_context:
            try:
                per_cell, touching, frame = pre_context(rec)
            except Exception as exc:                    # noqa: BLE001
                base["error"] = "context: %s: %s" % (type(exc).__name__, exc)
        base.update(frame)
        half = frame.get("half_box_diagonal", np.nan)
        for k, e in enumerate(sorted(events, key=lambda x: x["t_differentiated"])):
            cid = int(e["cell_id"])
            ctx = per_cell.get(cid, {})
            ev_rows.append(dict(
                base, cell_id=cid, event_order=k + 1,
                distance=float(e["distance"]),
                distance_rel_half_box=(float(e["distance"]) / half
                                       if np.isfinite(half) and half else np.nan),
                # The fork restarts its clock at 0 and the ablation happens at
                # its t = 0, so this IS the time since the ablation. It used to
                # subtract t_strict, which belongs to the SOURCE run's clock —
                # mixing the two made almost every value negative.
                t_differentiated=float(e["t_differentiated"]),
                dt_since_ablation=float(e["t_differentiated"]),
                touched_ablated_cell=cid in touching,
                n_HC_neighbours_pre=ctx.get("n_HC_neighbours_pre", np.nan),
                area_pre=ctx.get("area_pre", np.nan),
                delta_pre=ctx.get("delta_pre", np.nan)))
        d = np.array([e["distance"] for e in events], float)
        row = dict(base, n_events=len(events), n_pre_sc=rec.get("n_pre_sc"))
        m, s = _mean_sem(d)
        row.update(dist_mean=m, dist_sem=s,
                   dist_median=float(np.median(d)) if d.size else np.nan,
                   dist_min=float(d.min()) if d.size else np.nan,
                   dist_max=float(d.max()) if d.size else np.nan,
                   events_per_pre_SC=(len(events) / rec["n_pre_sc"]
                                      if rec.get("n_pre_sc") else np.nan))
        for c in CUTOFFS:
            row["n_events_within_%.1f" % c] = int((d <= c).sum()) if d.size else 0
            row["pct_events_within_%.1f" % c] = (100.0 * float((d <= c).mean())
                                                 if d.size else np.nan)
        run_rows.append(row)

    runs = pd.DataFrame(run_rows)
    events = pd.DataFrame(ev_rows)

    over_rows = []
    ok = runs[runs["error"].fillna("") == ""] if len(runs) else runs
    for (stage, ps), g in (ok.groupby(["stage", "psigma"]) if len(ok) else []):
        ge = events[(events["stage"] == stage) & (events["psigma"] == ps)]
        d = ge["distance"].to_numpy(float)
        row = dict(stage=stage, psigma=float(ps), n_ablations=len(g),
                   n_failed=int((runs[(runs["stage"] == stage) &
                                      (runs["psigma"] == ps)]["error"].fillna("") != "").sum()),
                   n_events=int(len(ge)))
        m, s = _mean_sem(g["n_events"])
        row.update(events_per_ablation_mean=m, events_per_ablation_sem=s)
        m, s = _mean_sem(d)                      # pooled over events
        row.update(dist_mean=m, dist_sem=s,
                   dist_median=float(np.median(d)) if d.size else np.nan,
                   dist_min=float(d.min()) if d.size else np.nan,
                   dist_max=float(d.max()) if d.size else np.nan)
        m, s = _mean_sem(g["dist_mean"])         # each ablation counts once
        row.update(run_dist_mean=m, run_dist_sem=s)
        # A THIRD mean, now that each array carries several ablations: average
        # within the array first, then across arrays. This is the convention used
        # everywhere else in the project, and it is the one whose SEM describes
        # array-to-array variation rather than which cell happened to be removed.
        # It differs from run_dist_mean whenever arrays contribute unequal numbers
        # of usable ablations, which they do once a run dies.
        by_arr = g.groupby("initial_array")
        for lab, col in (("array_dist", "dist_mean"),
                         ("array_events", "n_events"),
                         ("array_pct_within_2.5", "pct_events_within_2.5")):
            m, s = _mean_sem(by_arr[col].mean())
            row["%s_mean" % lab] = m
            row["%s_sem" % lab] = s
        row["n_arrays"] = int(g["initial_array"].nunique())
        row["ablations_per_array"] = (len(g) / max(row["n_arrays"], 1))
        m, s = _mean_sem(g["events_per_pre_SC"])
        row.update(events_per_pre_SC_mean=m, events_per_pre_SC_sem=s)
        if len(ge) and "distance_rel_half_box" in ge:
            m, s = _mean_sem(ge["distance_rel_half_box"])
            row.update(dist_rel_half_box_mean=m, dist_rel_half_box_sem=s)
        for c in CUTOFFS:
            row["n_events_within_%.1f" % c] = int((d <= c).sum()) if d.size else 0
            row["pct_events_within_%.1f" % c] = (100.0 * float((d <= c).mean())
                                                 if d.size else np.nan)
            m, s = _mean_sem(g["pct_events_within_%.1f" % c])
            row["run_pct_within_%.1f_mean" % c] = m
            row["run_pct_within_%.1f_sem" % c] = s
        if len(ge) and "n_HC_neighbours_pre" in ge:
            nb = ge["n_HC_neighbours_pre"].to_numpy(float)
            nb = nb[np.isfinite(nb)]
            for k, lab in ((0, "0"), (1, "1"), (2, "2plus")):
                sel = (nb == k) if k < 2 else (nb >= 2)
                row["pct_events_%s_HC_neighbours_pre" % lab] = (100.0 * float(sel.mean())
                                                                if nb.size else np.nan)
            # the 2+ bucket resolved into exact counts; the per-event raw values
            # are already in the events sheet, so this costs nothing to add
            for k in range(MAX_NB + 1):
                sel = (nb == k)
                row["n_events_exactly_%d_HC_neighbours_pre" % k] = int(sel.sum())
                row["pct_events_exactly_%d_HC_neighbours_pre" % k] = (
                    100.0 * float(sel.mean()) if nb.size else np.nan)
            over = (nb > MAX_NB)
            row["n_events_more_than_%d_HC_neighbours_pre" % MAX_NB] = int(over.sum())
            row["pct_events_more_than_%d_HC_neighbours_pre" % MAX_NB] = (
                100.0 * float(over.mean()) if nb.size else np.nan)
            row["pct_events_touching_ablated"] = (
                100.0 * float(ge["touched_ablated_cell"].astype(bool).mean())
                if len(ge) else np.nan)
        m, s = _mean_sem(g["t_strict_in_source"])
        row.update(t_strict_in_source_mean=m, t_strict_in_source_sem=s)
        over_rows.append(row)
    overall = pd.DataFrame(over_rows)
    if len(overall):
        overall = overall.sort_values(["stage", "psigma"]).reset_index(drop=True)
    return overall, runs, events


def _prune_for_output(overall, runs, events):
    """Trim the saved tables: failed ablations out, dead and duplicate columns out.

    An ablation whose run died carries no events and no measurements, so it is
    dropped as a ROW rather than flagged — and the error column then has nothing
    to say. The record of what failed stays in ablate_from_strict.json.

    ``t_differentiated`` and ``dt_since_ablation`` are now the same number (the
    fork's clock starts at the ablation), so only the named one is kept.

    The per-count HC-neighbour columns beyond 2 are zero for every event, and the
    2plus bucket is redundant once the exact counts are there.
    """
    n_before = len(runs)
    if "error" in runs.columns:
        runs = runs[runs["error"].fillna("").astype(str) == ""].copy()
    dropped = n_before - len(runs)
    kept = set(runs["run_folder"])
    events = events[events["run_folder"].isin(kept)].copy()

    # box geometry is a property of the setup, identical on every row
    dead = ["error", "Lx", "Ly", "t_differentiated"]
    runs = runs.drop(columns=[c for c in dead if c in runs.columns])
    events = events.drop(columns=[c for c in dead if c in events.columns])

    ren = {}
    for k in (2, 3):
        for pre in ("n", "pct"):
            src = "%s_events_exactly_%d_HC_neighbours_pre" % (pre, k)
            if src in overall.columns:
                ren[src] = "%s_events_%d_HC_neighbours_pre" % (pre, k)
    overall = overall.rename(columns=ren)
    drop_o = []
    for pre in ("n", "pct"):
        drop_o += ["%s_events_2plus_HC_neighbours_pre" % pre,
                   "%s_events_more_than_%d_HC_neighbours_pre" % (pre, MAX_NB)]
        drop_o += ["%s_events_exactly_%d_HC_neighbours_pre" % (pre, k)
                   for k in (0, 1) + tuple(range(4, MAX_NB + 1))]
    overall = overall.drop(columns=[c for c in drop_o if c in overall.columns])

    print("\n  pruned: %d failed ablation(s) dropped; overall -> %d cols, "
          "runs -> %d cols, events -> %d cols"
          % (dropped, overall.shape[1], runs.shape[1], events.shape[1]))
    return overall, runs, events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=IN_JSON,
                    help="results JSON, absolute or relative to the results dir")
    ap.add_argument("--out-prefix", dest="prefix", default="ablation",
                    help="output base name; use another to keep an older set")
    ap.add_argument("--no-context", dest="context", action="store_false",
                    help="skip the pre-ablation sheet loads (no neighbour/area fields)")
    a = ap.parse_args()

    path = a.json if os.path.isabs(a.json) else os.path.join(RESULTS_DIR, a.json)
    if not os.path.isfile(path):
        raise SystemExit("no such file: %s" % path)
    records = json.load(open(path))
    print("%s: %d record(s)" % (os.path.basename(path), len(records)), flush=True)

    overall, runs, events = build(records, with_context=a.context)

    overall, runs, events = _prune_for_output(overall, runs, events)
    # published under the manuscript's name for the parameter
    overall, runs, events = (to_output_names(f)
                             for f in (overall, runs, events))

    for name, frame in (("overall", overall), ("runs", runs), ("events", events)):
        frame.to_pickle(os.path.join(RESULTS_DIR, "%s_%s.pkl" % (a.prefix, name)))
    xlsx = os.path.join(RESULTS_DIR, "%s_tables.xlsx" % a.prefix)
    try:
        with pd.ExcelWriter(xlsx) as writer:
            overall.to_excel(writer, sheet_name="overall", index=False)
            runs.to_excel(writer, sheet_name="runs", index=False)
            events.to_excel(writer, sheet_name="events", index=False)
            # The comparison against the experiment is computed elsewhere, by
            # compare_ablation_distance_to_experiment.py, and adds its sheets to
            # this workbook. Rewriting the workbook would drop them, so they are
            # carried over from the pickles it left behind.
            for name, frame in carried_over_sheets(ABLATION_SHEETS):
                frame.to_excel(writer, sheet_name=name, index=False)
                print("  carried over sheet %s (%d rows)" % (name, len(frame)))
    except Exception as exc:                            # noqa: BLE001
        print("  xlsx failed (%s: %s); the pickles are written"
              % (type(exc).__name__, exc))

    print("\n  overall %4d rows x %3d cols" % overall.shape)
    print("  runs    %4d rows x %3d cols" % runs.shape)
    print("  events  %4d rows x %3d cols" % events.shape)
    if len(overall):
        print("\n  %-6s %-7s %5s %7s %9s %9s %9s %9s"
              % ("stage", "psigma", "abl", "events", "ev/abl", "dist mean",
                 "dist med", "<=2.5%"))
        for _i, r in overall.iterrows():
            print("  %-6s %-7.3f %5d %7d %9.2f %9.2f %9.2f %9.0f"
                  % (r["stage"], r["psigma"], r["n_ablations"], r["n_events"],
                     r["events_per_ablation_mean"], r["dist_mean"],
                     r["dist_median"], r["pct_events_within_2.5"]))
    print("\nwrote %s" % xlsx)
    print("      %s_overall.pkl, %s_runs.pkl, %s_events.pkl"
          % (a.prefix, a.prefix, a.prefix))


if __name__ == "__main__":
    main()
