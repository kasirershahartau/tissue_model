"""One row per v2 MECHANICAL-FIT simulation: parameters + every measured term.

    python build_mechanics_table.py --limit 8      # quick check
    python build_mechanics_table.py --workers 3

Writes, into <results>/:
    mechanics_runs.xlsx / .pkl     one row per base run (the detail)
    mechanics_points.xlsx / .pkl   one row per parameter POINT (the scores)
plus an incremental mechanics_runs.csv, appended after every run, so a crashed
pass resumes instead of restarting.

WHY TWO TABLES. The measured quantities — roundness, area change, shrinkage —
are per-run. The SCORES are not: an n-sigma is
(mean_model - mean_exp)/sqrt(SEM^2 + SEM^2) over the POOLED set of ~10 sheets, so
it exists only for a parameter point. The per-run table repeats its point's
scores on every row for convenience, but they are a point-level property; the
points table is the one to read when comparing parameter sets.

TWO SEMs, DELIBERATELY DISTINCT:
  * *_sem in the runs table   spread over CELLS within that run
  * *_sem in the points table spread over RUNS (this is the one in the score)

PROVENANCE. A0 means different things across the fit's history — the closed-form
step-1 value in the coupled grid, pi/4 on the boundary scan, the exact
self-consistent value afterwards — so each point carries the scan it came from.

Identification: v2 runs are those with bending 0 and shape index 0 (pure
contractility). Each ablation run names its base run as its initial_sheet_name,
which is how the two halves are paired. A0 comes from
``preferred_area_override``, NOT from ``preferred_area`` — the latter is the
untouched pi/4 default and would be wrong for every row.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type,
                             calc_roundness_for_type, calc_area_change_after_ablation,
                             relaxed_cut_scale, _hc_over_mean_sc,
                             load_experimental_results, _finite_arrays,
                             _MECHANICS_EXPERIMENTAL_TYPE)
from build_experimental_tables import carried_over_sheets, MECHANICS_SHEETS
from build_run_table import read_parameters as read_params
from run_model import _reached_steady_state

TYPE_BY = "delta_level"
RUNS_BASE = "mechanics_runs"
POINTS_BASE = "mechanics_points"
SCAN_FILES = {
    "grid_E17.5": "grid_fit_mechanics_v2_E17.5.json",
    "stress_P0": "p0_from_e17_stiffness.json",
    "rgamma_P0": "p0_rgamma_scan.json",
    "boundary_P0": "p0_boundary_scan.json",
    "selfconsistent_E17.5": "e17_selfconsistent_scan.json",
    "selfconsistent_P0": "p0_selfconsistent_scan.json",
}


def lit(v, default=None):
    try:
        return ast.literal_eval(v)
    except Exception:                                   # noqa: BLE001
        return default if default is not None else v


def is_v2(params):
    """v2 = pure contractility: BOTH per-type shape indices 0, and no bending.

    Test hc_shape_index / sc_shape_index, NOT shape_index. The global
    ``shape_index`` is 0.0 in v1 runs too — v1 carries its target perimeter in
    the per-type fields (3.83 / 4.56), leaving the global one unused. Checking
    the global field is therefore vacuous and would let every pre-bending v1 run
    through.
    """
    def zero(key):
        v = lit(params.get(key, "1"), None)
        try:
            return v is not None and abs(float(v)) < 1e-12
        except (TypeError, ValueError):
            return False
    return zero("hc_shape_index") and zero("sc_shape_index") and zero("bending")


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def ablation_measured(hc_area_change, sc_area_change):
    """Did the ablation actually move anything?

    In some archives the post-ablation frame IS the pre-ablation frame, so every
    cell's area is unchanged: each area-change ratio is exactly 1, and the
    HC-over-mean-SC ratio comes out as a meaningless 1.0 with zero spread. That
    is not a measurement of anything, and averaging it in pulls a parameter
    point's ablation ratio towards 1.

    The mechanical fit already skipped those sheets for the ablation term while
    keeping their perfectly good roundness (see mechanics_eval), which is why
    dropping them reproduces the stored per-term chi^2 exactly. This is the same
    test, in one place, for the builder and for patch_mechanics_ablation.py.
    """
    for v in (hc_area_change, sc_area_change):
        if v is None or not len(np.asarray(v, float)):
            return False
    return not (np.allclose(np.asarray(hc_area_change, float), 1.0)
                and np.allclose(np.asarray(sc_area_change, float), 1.0))


def ablation_partners():
    """{base run folder: ablation run folder} — each _abl names its base run."""
    out = {}
    for d in os.listdir(RESULTS_DIR):
        if not d.startswith("fit_") or not d.endswith("_abl"):
            continue
        try:
            p = read_params(d)
        except Exception:                               # noqa: BLE001
            continue
        base = lit(p.get("initial_sheet_name", ""), "")
        if base:
            out[base] = d
    return out


def scan_points():
    """[(scan, stage, R_alpha, R_gamma, gammaSC, A0, z-dict, n_sheets)] from every scan."""
    import json
    rows = []
    for scan, fn in SCAN_FILES.items():
        path = os.path.join(RESULTS_DIR, fn)
        if not os.path.isfile(path):
            continue
        blob = json.load(open(path))
        stage = blob.get("stage") or ("E17.5" if "E17" in scan else "P0")
        top_ra = blob.get("R_alpha")
        for key, v in (blob.get("points") or {}).items():
            if not v:
                continue
            ra = v.get("R_alpha", top_ra)
            rg = gs = None
            if "Rg=" in key:
                rg = float(key.split("Rg=")[1].split("|")[0])
            if "gSC=" in key:
                gs = float(key.split("gSC=")[1])
            if key.startswith("R=") and "|" in key:      # coupled grid: R=..|gSC=..
                ra = rg = float(key.split("|")[0].split("=")[1])
            elif key.startswith("R=") and "|" not in key:  # stress scan: R=..
                ra = rg = float(key.split("=")[1])
            rows.append(dict(scan=scan, stage=stage, R_alpha=ra, R_gamma=rg,
                             gamma_sc=v.get("gamma_sc", gs), A0=v.get("A0"),
                             z=v.get("z") or {}, n_sheets=v.get("n_sheets_ok")))
    return rows


def exp_targets():
    """{(stage, term): (mean, sem)} — so every z in the table is reproducible."""
    out = {}
    for stage in ("E17.5", "P0"):
        for term, etype in _MECHANICS_EXPERIMENTAL_TYPE.items():
            try:
                e = _finite_arrays(load_experimental_results(stage, etype))
                m = np.array([x.mean() for x in e], float)
                out[(stage, term)] = (float(m.mean()),
                                      float(m.std(ddof=1) / np.sqrt(m.size)))
            except Exception:                           # noqa: BLE001
                out[(stage, term)] = (float("nan"), float("nan"))
    return out


def per_cell_roundness(sheet, th, cell_type):
    """One row per cell: face id, index label, roundness. Same selection and
    ordering as calc_roundness_for_type, but keeping the identity of each cell."""
    _idx, labels = get_non_boundary_cell_ids_from_type(
        sheet, cell_type=cell_type, type_by=TYPE_BY, threshold=th)
    if len(labels) == 0:
        return []
    roundness = sheet.get_face_roundness()
    area = sheet.get_face_area()
    ids = sheet.face_df.loc[labels, "id"].to_numpy()
    return [dict(face_id=int(i), face_index=int(l), cell_type=cell_type,
                 roundness=float(roundness.loc[l]), area=float(area.loc[l]))
            for i, l in zip(ids, labels)]


def per_cell_ablation(abl_folder, ablated_cells, th):
    """One row per (ablated cell, neighbour): area before, after, and the ratio.

    calc_area_change_after_ablation unions the neighbourhoods of all ablated
    cells, so a cell touching two of them is counted once and its attribution is
    lost. Here each (ablated cell, neighbour) PAIR gets a row, so a cell bordering
    two ablated cells appears twice — ``n_ablated_neighbours`` says so, and the
    summary columns in the runs table still use the union, matching the score."""
    history = load_history_file(abl_folder)
    initial = history.retrieve(0)
    initial.arrange_sheet_from_history()
    final = history.retrieve(float(np.max(get_time_points(history))))
    final.arrange_sheet_from_history()
    a0 = initial.get_face_area()
    a1 = final.get_face_area()
    id_of = initial.face_df["id"]
    # id -> face_df label, per frame, restricted to faces that still HAVE an area.
    # An ablated face keeps its row in face_df but loses its edges, so it drops
    # out of get_face_area() while still appearing in face_df and in any "alive"
    # set derived from it. Keying off the area index is what makes a removed cell
    # actually absent here.
    ini_label_of_id = {int(i): lab for lab, i in initial.face_df["id"].items()
                       if lab in a0.index}
    fin_label_of_id = {int(i): lab for lab, i in final.face_df["id"].items()
                       if lab in a1.index}
    alive = set(fin_label_of_id)
    rows, seen = [], {}
    for ablated in ablated_cells:
        try:
            neigh = np.setdiff1d(initial.get_neighbors(ablated), ablated)
        except Exception:                               # noqa: BLE001
            continue
        for lab in neigh:
            cid = int(id_of.loc[lab])
            seen[cid] = seen.get(cid, 0) + 1
    for ablated in ablated_cells:
        try:
            neigh = np.setdiff1d(initial.get_neighbors(ablated), ablated)
        except Exception:                               # noqa: BLE001
            continue
        ids = initial.face_df.loc[neigh, "id"].to_numpy()
        for ctype in ("HC", "SC"):
            # NOTE the second return value is persistent IDs, not face_df labels.
            # They coincide in a freshly built sheet, which is why treating them
            # as labels worked until a run had faces REMOVED — an ablation run
            # always does, so every lookup below goes through the id->label maps.
            _i, cell_ids = get_non_boundary_cell_ids_from_type(
                initial, cell_type=ctype, type_by=TYPE_BY, threshold=th,
                only_for_these_cells=ids)
            for cid in cell_ids:
                cid = int(cid)
                if cid not in alive:
                    continue
                ini_lab, fin_lab = ini_label_of_id.get(cid), fin_label_of_id.get(cid)
                if ini_lab is None or fin_lab is None:
                    continue
                before, after = float(a0.loc[ini_lab]), float(a1.loc[fin_lab])
                rows.append(dict(ablated_cell=int(ablated), face_id=cid,
                                 face_index=int(ini_lab), cell_type=ctype,
                                 area_before=before, area_after=after,
                                 area_ratio=after / before if before else float("nan"),
                                 n_ablated_neighbours=seen.get(cid, 1)))
    return rows


def one_run(args):
    folder, abl_folder = args
    try:
        p = read_params(folder)
        init = lit(p.get("initial_sheet_name", ""), "")
        stage = "E17.5" if init.endswith("_for_E17") else "P0"
        gsc = float(lit(p.get("gammaSC", "nan"), float("nan")))
        rg = float(lit(p.get("gammaHC_ratio", "nan"), float("nan")))
        ra = float(lit(p.get("alphaHC_ratio", "nan"), float("nan")))
        A0 = float(lit(p.get("preferred_area_override", "nan"), float("nan")))
        th = float(lit(p.get("atoh_sensitivity", "nan"), float("nan")))

        history = load_history_file(folder)
        t_final = float(np.max(get_time_points(history)))
        sheet = history.retrieve(t_final)
        sheet.arrange_sheet_from_history()
        sheet.geom.update_all(sheet)

        hc_r = np.asarray(calc_roundness_for_type(sheet, cell_type="HC",
                                                  type_by=TYPE_BY, threshold=th), float)
        sc_r = np.asarray(calc_roundness_for_type(sheet, cell_type="SC",
                                                  type_by=TYPE_BY, threshold=th), float)
        ratio = _hc_over_mean_sc(hc_r, sc_r)
        lam = float(relaxed_cut_scale(sheet.face_df))
        hc_idx, _ = get_non_boundary_cell_ids_from_type(sheet, "HC", type_by=TYPE_BY,
                                                        threshold=th)
        sc_idx, _ = get_non_boundary_cell_ids_from_type(sheet, "SC", type_by=TYPE_BY,
                                                        threshold=th)
        n_hc, n_sc = len(hc_idx), len(sc_idx)

        hc_a = sc_a = a_ratio = None
        cells = []
        if abl_folder:
            ap = read_params(abl_folder)
            cells = list(lit(ap.get("ablated_cells", "[]"), []) or [])
            hc_a, sc_a = calc_area_change_after_ablation(
                load_history_file(abl_folder), abl_folder, ablated_cells=list(cells),
                end_time=-1, type_by=TYPE_BY, threshold=th)
            if not ablation_measured(hc_a, sc_a):
                # The archive's post-ablation frame is the pre-ablation one, so
                # every area is unchanged and the ratio would be a meaningless
                # exactly-1.0. Record nothing rather than that; see
                # ablation_measured.
                hc_a = sc_a = None
            else:
                a_ratio = _hc_over_mean_sc(hc_a, sc_a)

        row = dict(
            run_folder=folder, ablation_folder=abl_folder or "", stage=stage,
            initial_array=init, t_final=t_final,
            steady_state=bool(_reached_steady_state(os.path.join(RESULTS_DIR, folder))),
            gammaSC=gsc, gammaHC=gsc * rg, gammaHC_ratio=rg,
            alphaSC=1.0, alphaHC=ra,
            A0=A0, A0_over_quarter_pi=A0 / (np.pi / 4) if np.isfinite(A0) else np.nan,
            atoh_threshold=th,
            shape_index=float(lit(p.get("shape_index", "nan"), float("nan"))),
            bending=float(lit(p.get("bending", "nan"), float("nan")) or 0.0),
            quasi_static_threshold=float(lit(p.get("quasi_static_threshold", "nan"),
                                             float("nan"))),
            n_HC=n_hc, n_SC=n_sc,
            hc_fraction=n_hc / (n_hc + n_sc) if (n_hc + n_sc) else np.nan,
            hc_roundness_mean=mean(hc_r), hc_roundness_sem=sem(hc_r),
            sc_roundness_mean=mean(sc_r), sc_roundness_sem=sem(sc_r),
            roundness_ratio_mean=mean(ratio) if ratio is not None else np.nan,
            roundness_ratio_sem=sem(ratio) if ratio is not None else np.nan,
            ablation_measured=bool(abl_folder) and a_ratio is not None,
            n_HC_near_ablation=len(hc_a) if hc_a is not None else 0,
            n_SC_near_ablation=len(sc_a) if sc_a is not None else 0,
            hc_area_change_mean=mean(hc_a) if hc_a is not None else np.nan,
            hc_area_change_sem=sem(hc_a) if hc_a is not None else np.nan,
            sc_area_change_mean=mean(sc_a) if sc_a is not None else np.nan,
            sc_area_change_sem=sem(sc_a) if sc_a is not None else np.nan,
            ablation_ratio_mean=mean(a_ratio) if a_ratio is not None else np.nan,
            ablation_ratio_sem=sem(a_ratio) if a_ratio is not None else np.nan,
            lambda_linear=lam, shrinkage_pct=100.0 * (1.0 - lam),
            ablated_cells=";".join(str(int(c)) for c in cells),
            n_ablated_cells=len(cells),
            error="")
        # per-cell detail, tagged with the run so the long tables stay joinable
        tag = dict(run_folder=folder, stage=stage, initial_array=init,
                   gammaSC=gsc, gammaHC_ratio=rg, alphaHC=ra, A0=A0)
        cells_hc = [dict(tag, **r) for r in per_cell_roundness(sheet, th, "HC")]
        cells_sc = [dict(tag, **r) for r in per_cell_roundness(sheet, th, "SC")]
        abl_rows = []
        if abl_folder and cells:
            try:
                abl_rows = [dict(tag, ablation_folder=abl_folder, **r)
                            for r in per_cell_ablation(abl_folder, cells, th)]
            except Exception as exc:                    # noqa: BLE001
                row["error"] = "per_cell_ablation: %s" % type(exc).__name__
        return row, cells_hc, cells_sc, abl_rows
    except Exception as exc:                            # noqa: BLE001
        return (dict(run_folder=folder, error="%s: %s" % (type(exc).__name__, exc)),
                [], [], [])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="first N runs only (a check)")
    ap.add_argument("--workers", type=int, default=3,
                    help="keep low while simulations are running — this is read-heavy")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    partners = ablation_partners()
    bases = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        if not d.startswith("fit_") or d.endswith("_abl"):
            continue
        try:
            if is_v2(read_params(d)):
                bases.append(d)
        except Exception:                               # noqa: BLE001
            continue
    if a.limit:
        bases = bases[:a.limit]
    csv_path = os.path.join(RESULTS_DIR, RUNS_BASE + ".csv")
    done = set()
    if os.path.isfile(csv_path) and not a.no_resume:
        try:
            done = set(pd.read_csv(csv_path)["run_folder"])
        except Exception:                               # noqa: BLE001
            done = set()
    todo = [(b, partners.get(b, "")) for b in bases if b not in done]
    print("v2 mechanical-fit runs: %d base, %d with an ablation partner; %d to do"
          % (len(bases), sum(1 for b in bases if b in partners), len(todo)), flush=True)

    # Each long table gets its own incremental CSV so a crashed pass resumes with
    # its per-cell detail intact, not just the summary.
    side = {k: os.path.join(RESULTS_DIR, "%s_%s.csv" % (RUNS_BASE, k))
            for k in ("hc_roundness", "sc_roundness", "hc_ablation", "sc_ablation")}
    if a.no_resume:
        for f in list(side.values()) + [csv_path]:
            if os.path.isfile(f):
                os.remove(f)

    _cols = {}

    def append(path, records):
        """Append rows, ALIGNED to the columns the file already has.

        A run that fails returns a two-key row while a good one returns ~40.
        Appending it as-is writes two fields into a wide CSV, and read_csv then
        maps them POSITIONALLY onto the first columns — which is how an exception
        message ended up in ``ablation_folder``, the second column. Reindexing
        first leaves the missing fields empty instead."""
        if not records:
            return
        frame = pd.DataFrame(records)
        if path not in _cols:
            if os.path.isfile(path):
                try:
                    _cols[path] = list(pd.read_csv(path, nrows=0).columns)
                except Exception:                       # noqa: BLE001
                    _cols[path] = list(frame.columns)
            else:
                _cols[path] = list(frame.columns)
        extra = [c for c in frame.columns if c not in _cols[path]]
        if extra:                       # a later row introduced a new field
            _cols[path] = _cols[path] + extra
        frame.reindex(columns=_cols[path]).to_csv(
            path, mode="a", index=False, header=not os.path.isfile(path))

    rows = []
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        per_pool = max(1, 3 * a.workers)
        for s in range(0, len(todo), per_pool):
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for r, chc, csc, cab in ex.map(one_run, todo[s:s + per_pool]):
                    rows.append(r)
                    append(csv_path, [r])
                    append(side["hc_roundness"], chc)
                    append(side["sc_roundness"], csc)
                    append(side["hc_ablation"],
                           [x for x in cab if x["cell_type"] == "HC"])
                    append(side["sc_ablation"],
                           [x for x in cab if x["cell_type"] == "SC"])
            print("  %d/%d" % (min(s + per_pool, len(todo)), len(todo)), flush=True)
    df = pd.read_csv(csv_path) if os.path.isfile(csv_path) else pd.DataFrame(rows)

    # ---- attach the point-level scores -----------------------------------
    tgt = exp_targets()
    pts = scan_points()
    prows = []
    for p in pts:
        row = dict(scan=p["scan"], stage=p["stage"], R_alpha=p["R_alpha"],
                   R_gamma=p["R_gamma"], gammaSC=p["gamma_sc"], A0=p["A0"],
                   n_sheets=p["n_sheets"])
        total = 0.0
        for term in _MECHANICS_EXPERIMENTAL_TYPE:
            z = p["z"].get(term, np.nan)
            row["%s_nsigma" % term] = z
            row["%s_chi2" % term] = z * z if np.isfinite(z) else np.nan
            m, s = tgt.get((p["stage"], term), (np.nan, np.nan))
            row["%s_exp_mean" % term] = m
            row["%s_exp_sem" % term] = s
            if np.isfinite(z):
                total += z * z
        row["total_chi2"] = total
        prows.append(row)
    pdf = pd.DataFrame(prows)

    # Runs that produced nothing are dropped as ROWS, not flagged. 25 of them hold
    # only debug.log and parameters.txt — no history.hf5, so the simulation never
    # produced output — and 4 more have a history that cannot be read. None carry
    # a single measurement, so they are absent from the per-cell sheets already
    # and contribute nothing to any score. With them gone the error column has
    # nothing left to say and goes too.
    n_before = len(df)
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
        df = df.drop(columns=["error"])
    if n_before != len(df):
        print("  dropped %d run(s) with no usable output" % (n_before - len(df)))

    # Columns dropped from the SAVED runs sheet only; the CSV cache keeps them.
    # gammaHC_ratio is gammaHC/gammaSC and both are already columns;
    # A0_over_quarter_pi is A0 rescaled; the rest are fixed across the v2 fit
    # (shape index 0, no bending, one atoh threshold) and so carry no contrast.
    df = df.drop(columns=[c for c in ("gammaHC_ratio", "A0_over_quarter_pi",
                                      "atoh_threshold", "shape_index", "bending")
                          if c in df.columns])

    sheets = {"runs": df, "points": pdf}
    for key, path in side.items():
        sheets[key] = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()

    # one pickle per table, and ONE workbook with every table as a sheet
    for name, frame in sheets.items():
        frame.to_pickle(os.path.join(RESULTS_DIR, "mechanics_%s.pkl" % name))
    xlsx = os.path.join(RESULTS_DIR, "mechanics_tables.xlsx")
    try:
        with pd.ExcelWriter(xlsx) as writer:
            for name, frame in sheets.items():
                # Excel caps a sheet at 1,048,576 rows and the per-cell tables can
                # approach that, so truncate loudly rather than fail at the end.
                f = frame
                if len(f) > 1000000:
                    print("  !! %s has %d rows; writing the first 1,000,000 to xlsx"
                          " (the pickle keeps all of them)" % (name, len(f)))
                    f = f.iloc[:1000000]
                f.to_excel(writer, sheet_name=name[:31], index=False)
            # the experimental targets the fit was scored against, written by
            # build_experimental_tables.py; carried over so rewriting this
            # workbook does not drop them
            for name, frame in carried_over_sheets(MECHANICS_SHEETS):
                frame.to_excel(writer, sheet_name=name, index=False)
                print("  carried over sheet %s (%d rows)" % (name, len(frame)))
    except Exception as exc:                            # noqa: BLE001
        print("  xlsx failed (%s: %s); pickles and CSVs are written"
              % (type(exc).__name__, exc))
    print()
    for name, frame in sheets.items():
        print("  %-14s %8d rows x %3d cols"
              % (name, len(frame), frame.shape[1] if len(frame) else 0))
    print()
    print("wrote %s" % xlsx)
    print("       mechanics_{%s}.pkl" % ",".join(sheets))


if __name__ == "__main__":
    main()
