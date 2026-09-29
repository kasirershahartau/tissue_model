"""Mechanical stress on a cell-cell interface, split by the types it separates.

    python build_edge_stress_table.py --workers 4
    python build_edge_stress_table.py --limit 4          # quick check

Every neighbour pair in the pT = 0 and pT = 0.162 runs, at two moments,
classified as HC:HC, HC:SC or SC:SC by the types of the two cells.

THE TWO MOMENTS ARE NOT 0 AND 0. ``initial`` is the first recorded frame, t = 0.
``t0`` is the run's own t0 from the runs table — the frame whose neighbour-pair
composition best matches the experiment, which lands around t = 8-16 depending
on the run, and is the frame every score is measured at. They are different
frames of the same simulation, not two names for the start.

WHAT THE STRESS IS. Per NEIGHBOUR PAIR, normalised the way the face stress is:

    sum over the mesh edges shared by the two cells of (edge_stress * length)
    divided by L0, the mean cell perimeter of the run's first frame

InnerEarModel.get_edge_stress supplies the per-edge term and already returns each
edge summed with its opposite — the same coupling the face stress uses, on the
grounds that the two half-edges are one physical junction carrying both. So no
second summing is needed there.

The aggregation to a pair matters because virtual vertices subdivide every
interface: a sheet carries ~33 half-edges per face, so one cell-cell contact is
made of about 5 or 6 mesh edges and there are ~8300 of them against ~1500
contacts. A per-edge distribution would be counting mesh resolution, not
tissue. Each pair is formed once, from the half-edges whose label is smaller
than their opposite's, so nothing is double counted.

THE INITIAL FRAME HAS NO HAIR CELLS. At t = 0 delta spans 0.000-0.010, far below
the 0.355 threshold: lateral inhibition builds the pattern during the run. So
the initial frame yields SC:SC pairs only, and its value is a baseline for the
tissue before any differentiation rather than a comparison between classes.

ONE EFFECTOR SET. ContractilityPerimeterElasticity alone — what
run_model.stress_effectors gates on, so the stress pT is actually compared
against. The all-effectors total is dominated by the area term and answers a
different question; face_stress_over_time computes it if it is ever wanted.

Writes to <results>/:
    fullmodel_edge_stress.pkl          one row per (run, frame, set, pair)
    fullmodel_edge_stress_runs.pkl     one row per (run, frame, set, class)
    fullmodel_edge_stress_summary.pkl  one row per (stage, frame, set, class)
and adds the summary and per-run sheets to fullmodel_tables.xlsx.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR, load_history_file, get_time_points
from build_experimental_tables import read_table, to_output_names, add_sheets
from face_stress_over_time import _Shim, EFFECTOR_SETS
from count_reverse_differentiation import TYPE_BY, THRESHOLD, T0_TOL

PT = (0.0, 0.162)
CLASSES = ("HC:HC", "HC:SC", "SC:SC")
# contractility only: that is the set run_model.stress_effectors gates on, so
# the one pT is compared against. The all-effectors total is dominated by the
# area term and answers a different question; face_stress_over_time still
# computes it if it is ever wanted.
STRESS_SETS = {"contractility": EFFECTOR_SETS["contractility"]}
RUNS_PKL = "fullmodel_edge_stress"


def junction_rows(sheet, threshold, type_by=TYPE_BY):
    """Per half-edge kept: its row, class, length and the two faces it separates.

    A junction is a half-edge and its opposite; it is kept once, under the
    smaller of the two labels. ``opposite`` holds an EDGE label, so the cell on
    the far side is reached by resolving it through edge_df — comparing it to a
    face label would be comparing unrelated index spaces.
    """
    edge_df = sheet.edge_df
    idx = edge_df.index.to_numpy()
    opp = edge_df["opposite"].to_numpy()
    face = edge_df["face"].to_numpy()
    paired = opp >= 0
    keep = paired & (idx < opp)            # each junction once
    if not keep.any():
        e = np.array([], int)
        return e, np.array([], object), np.array([], float), e, e

    pos = edge_df.index.get_indexer(opp[keep])
    is_hc = (sheet.face_df[type_by] > threshold)
    face_a, face_b = face[keep], face[pos]
    a = is_hc.reindex(face_a).to_numpy(bool)
    b = is_hc.reindex(face_b).to_numpy(bool)
    n_hc = a.astype(int) + b.astype(int)
    klass = np.array(CLASSES, object)[2 - n_hc]     # 2->HC:HC, 1->HC:SC, 0->SC:SC
    return (np.flatnonzero(keep), klass, edge_df["length"].to_numpy(float)[keep],
            np.minimum(face_a, face_b), np.maximum(face_a, face_b))


def one_run(args):
    """Per-neighbour-pair rows for one run at both frames, for both effector sets."""
    name, t0, threshold = args
    out, err = [], ""
    try:
        history = load_history_file(name)
        stamps = np.asarray(get_time_points(history), float)
        first = history.retrieve(float(stamps[0]))
        first.arrange_sheet_from_history(); first.geom.update_all(first)
        # the model's own length_normalization_factor: the mean cell perimeter
        # of the FIRST frame, exactly what the face stress divides by
        L0 = float(np.mean(first.face_df["perimeter"].to_numpy(float)))

        wanted = {"initial": float(stamps[0]),
                  "t0": float(stamps[stamps >= float(t0) - T0_TOL][0])}
        for frame, t in wanted.items():
            sheet = history.retrieve(t)
            sheet.arrange_sheet_from_history()
            sheet.geom.update_all(sheet)
            where, klass, length, face_a, face_b = junction_rows(sheet, threshold)
            if not where.size:
                continue
            for set_name, effectors in STRESS_SETS.items():
                stress = np.asarray(_Shim(sheet).get_edge_stress(effectors).values,
                                    float)[where]
                # sum(stress * length) / L0 over the mesh edges of one interface —
                # the face-stress construction restricted to a single neighbour
                edges = pd.DataFrame(dict(face_a=face_a, face_b=face_b,
                                          edge_class=klass, length=length,
                                          weighted=stress * length))
                pair = edges.groupby(["face_a", "face_b", "edge_class"],
                                     sort=False).agg(
                    stress=("weighted", lambda w: w.sum() / L0),
                    contact_length=("length", "sum"),
                    n_edges=("length", "size")).reset_index()
                pair.insert(0, "effectors", set_name)
                pair.insert(0, "time", t)
                pair.insert(0, "frame", frame)
                pair.insert(0, "model_name", name)
                pair["L0"] = L0
                out.append(pair)
    except Exception as exc:                                # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)
    frame = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return name, frame, err


def summarise(detail, runs):
    """Per (run, frame, set, class), then per (stage, frame, set, class)."""
    key = ["model_name", "frame", "effectors", "edge_class"]
    per_run = detail.groupby(key)["stress"].agg(
        n_pairs="size", stress_mean="mean", stress_median="median",
        stress_sd="std", stress_min="min", stress_max="max").reset_index()
    per_run = per_run.merge(
        runs[["model_name", "stage", "psigma", "initial_array", "repeat"]],
        on="model_name", how="left")

    rows = []
    for (stage, psigma, frame, eff, kl), g in per_run.groupby(
            ["stage", "psigma", "frame", "effectors", "edge_class"]):
        # per array first, then across arrays — the hierarchy the scores use
        v = g.groupby("initial_array")["stress_mean"].mean().to_numpy(float)
        v = v[np.isfinite(v)]
        rows.append(dict(stage=stage, psigma=float(psigma), frame=frame,
                         effectors=eff, edge_class=kl,
                         n_arrays=v.size, n_runs=len(g),
                         n_pairs_total=int(g["n_pairs"].sum()),
                         stress_mean=float(v.mean()) if v.size else np.nan,
                         stress_sem=(float(v.std(ddof=1) / np.sqrt(v.size))
                                     if v.size > 1 else np.nan)))
    return per_run, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pT", type=float, nargs="+", default=list(PT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-xlsx", dest="xlsx", action="store_false")
    a = ap.parse_args()

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    want = runs["psigma"].astype(float).apply(
        lambda p: any(np.isclose(p, v) for v in a.pT))
    sel = runs[want & ~runs["collapsed"].astype(bool)]
    if a.limit:
        sel = sel.head(a.limit)
    print("pT %s: %d run(s) on %d worker(s)"
          % (", ".join("%.3f" % v for v in a.pT), len(sel), a.workers))

    parts, failed = [], []
    todo = [(r["model_name"], r["t0"], a.threshold) for _i, r in sel.iterrows()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(one_run, t) for t in todo]
        for k, fut in enumerate(as_completed(futures), start=1):
            name, frame, err = fut.result()
            if err:
                failed.append((name, err))
            elif len(frame):
                parts.append(frame)
            if k % 10 == 0 or k == len(todo):
                print("   %3d/%d" % (k, len(todo)), flush=True)
    if failed:
        print("  %d run(s) failed; first: %s" % (len(failed), failed[0][1][:70]))
    if not parts:
        raise SystemExit("nothing measured")

    detail = pd.concat(parts, ignore_index=True)
    per_run, summary = summarise(detail, runs)
    for name, f in ((RUNS_PKL, detail), (RUNS_PKL + "_runs", per_run),
                    (RUNS_PKL + "_summary", summary)):
        to_output_names(f).to_pickle(os.path.join(RESULTS_DIR, name + ".pkl"))
    if a.xlsx:
        add_sheets(os.path.join(RESULTS_DIR, "fullmodel_tables.xlsx"),
                   {"edge_stress": to_output_names(summary),
                    "edge_stress_runs": to_output_names(per_run)})

    print("\n  detail %d rows, per-run %d rows, summary %d rows"
          % (len(detail), len(per_run), len(summary)))
    print("\n  %-6s %-8s %-14s %-7s %8s %10s %10s"
          % ("stage", "pT", "frame", "class", "pairs", "mean", "SEM"))
    shown = to_output_names(summary)
    for _i, r in shown.sort_values(["stage", "pT", "frame", "edge_class"]).iterrows():
        print("  %-6s %-7.3f %-8s %-7s %8d %10.4f %10.4f"
              % (r["stage"], r["pT"], r["frame"], r["edge_class"],
                 r["n_pairs_total"], r["stress_mean"], r["stress_sem"]))


if __name__ == "__main__":
    main()
