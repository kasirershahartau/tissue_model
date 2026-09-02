"""Distance from the ablation: model vs experiment, scored as z^2.

    python compare_ablation_distance_to_experiment.py
    python compare_ablation_distance_to_experiment.py --stage E17.5

Writes three tables, as pickles and as sheets added to ablation_tables.xlsx:

    ablation_distance_vs_experiment.pkl        one row per (stage, psigma): the
                                               two scores and everything they
                                               were computed from
    ablation_distance_vs_experiment_runs.pkl   one row per model ablation
    ablation_distance_vs_experiment_exp.pkl    the experimental side: one row per
                                               movie (cell size) and per
                                               biological repeat (distances)

THE UNIT PROBLEM. The experiment measures micrometres; the model measures
lattice units. A z on raw distances would mostly be measuring the conversion.

The fix is to express both in CELL DIAMETERS — "how many cells away from the
ablation did this cell differentiate" is the same question on both sides, and
needs no calibration between them. Each side supplies its own diameter, measured
the same way: the equivalent-circle diameter 2*sqrt(area/pi), averaged over cells.

  experiment  the segmented cells_info table of the first frame of each P0
              movie, valid non-edge cells, area in pixels at PIXEL_UM per pixel
  model       the pre-ablation frame of the source run, non-boundary faces

The three P0 movies cannot be matched to the three biological repeats of the
distance file (the repeats are keyed by date and position, the movies by index),
so their mean diameter is used as one scale for all three. They agree to ~3%, and
--scale-sensitivity prints what the extremes would do to the answer.

A CAVEAT ON THE ABSOLUTE DISTANCE. It carries the size of the domain: a cell can
only be as far from the ablation as the field of view (experiment) or the
periodic box (model) allows, so part of any difference is geometry rather than
biology. The reference SC distance — the mean over every SC present just before
the ablation, i.e. where a cell would be if nothing drew it towards the ablation
— measures exactly that, and is printed alongside for both sides. The ratio
diff/reference divides it out, and is scored second for that reason.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import read_table, to_output_names
from run_model import load_sheet_from_file
from ablate_single_hc import TYPE_BY, THRESHOLD, BOX, min_image_distance
from post_processing import get_non_boundary_cell_ids_from_type

PAPER = (r"C:\Users\Kasirer\Phd\mouse_ear_project\papers"
         r"\Dynamic lateral inhibition in the utricle")
EXP_XLSX = os.path.join(PAPER, "Raw Data",
                        "distance_from_ablation_raw_data(figure 2).xlsx")
CELLS_INFO_DIR = os.path.join(PAPER, "Experimental Data")
PIXEL_UM = 0.1                          # confirmed by the tif: 10.06 px per micron
SHEETS = {("E17.5", "diff"): "E17.5 differentiating cells",
          ("E17.5", "ref"): "E17.5 reference_SC",
          ("P0", "diff"): "P0 differentiating cells",
          ("P0", "ref"): "P0 reference_SC"}


def _mean_sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else np.nan,
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def _equivalent_diameter(area):
    """Mean over cells of the diameter of a circle with the same area."""
    a = np.asarray(area, float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.mean(2 * np.sqrt(a / np.pi))), int(a.size)


def experimental_diameter(stage):
    """Cell diameter (um) from the first frame of each movie of that stage.

    Returns one row per movie: the segmentation's own cell areas, in pixels at
    PIXEL_UM per pixel, over the cells it marked valid and away from the image
    border (a border cell is cut off, so its area is not the cell's).
    """
    pat = os.path.join(CELLS_INFO_DIR, stage,
                       "%s_experiment*_cells_info_frame_1" % stage)
    rows = []
    for path in sorted(glob.glob(pat)):
        ci = pd.read_pickle(path)
        keep = ci["valid"].astype(bool)
        if "edge_cell" in ci.columns:   # touching the image border: area truncated
            keep &= pd.to_numeric(ci["edge_cell"], errors="coerce").fillna(0) == 0
        area_um2 = pd.to_numeric(ci.loc[keep, "area"], errors="coerce") * PIXEL_UM ** 2
        d, n = _equivalent_diameter(area_um2)
        rows.append(dict(label=os.path.basename(path), n_cells=n,
                         cell_diameter_um=d))
    if not rows:
        raise SystemExit("no cells_info tables matching %s" % pat)
    return pd.DataFrame(rows)


def experimental_distances(stage):
    """{group: mean distance in um per biological repeat, indexed by repeat}."""
    out = {}
    for group in ("diff", "ref"):
        raw = pd.read_excel(EXP_XLSX, SHEETS[(stage, group)], header=None)
        hdr = [str(v).strip() for v in raw.iloc[0].tolist()]
        df = raw.iloc[1:].copy()
        df.columns = hdr
        dcol = [c for c in df.columns if c.startswith("Distance")][0]
        d = pd.to_numeric(df[dcol], errors="coerce")
        # one biological repeat = one (Date, Position); written once, then blank
        rep = (df["Date"].ffill().astype(str) + " @ "
               + df["Position"].ffill().astype(str))
        m = d.notna()
        out[group] = d[m].groupby(rep[m]).mean().astype(float)
    return out


def model_scales(rec_folder, t_strict, ablated_label):
    """Cell diameter, and mean distance to every non-boundary SC, before the ablation.

    Both in lattice units, from the same frame the ablation was taken from. The
    reference distance is the null "how far away is a cell anyway", measured with
    the periodic minimum image exactly as the event distances were.
    """
    pre = load_sheet_from_file(os.path.join(RESULTS_DIR, rec_folder),
                               time_point=t_strict, force_periodic_box=BOX)
    pre.geom.update_all(pre)
    all_idx, _ = get_non_boundary_cell_ids_from_type(pre, cell_type="all")
    diameter, n_cells = _equivalent_diameter(
        pre.face_df["area"].to_numpy(float)[np.asarray(all_idx, int)])

    sc_idx, _ = get_non_boundary_cell_ids_from_type(
        pre, cell_type="SC", type_by=TYPE_BY, threshold=THRESHOLD)
    xy = pre.face_df[["x", "y"]].to_numpy(float)[np.asarray(sc_idx, int)]
    ref = pre.face_df.loc[int(ablated_label), ["x", "y"]].to_numpy(float)
    Lx = float(getattr(pre, "Lx", BOX[0])); Ly = float(getattr(pre, "Ly", BOX[1]))
    d = min_image_distance(xy, ref, Lx, Ly)
    return diameter, n_cells, float(np.mean(d)), int(np.size(d))


def _z(model_mean, model_sem, e_mean, e_sem):
    den = np.sqrt(model_sem ** 2 + e_sem ** 2)
    return (model_mean - e_mean) / den if den > 0 else np.nan


def _score(name, summary, model_col, exp_col):
    e_mean = summary[exp_col + "_mean"].iloc[0]
    e_sem = summary[exp_col + "_sem"].iloc[0]
    print("\n  %s" % name)
    print("    experiment: %.3f +- %.3f  (%d repeats)"
          % (e_mean, e_sem, summary["exp_n_repeats"].iloc[0]))
    print("    %-8s %8s %10s %10s %10s" % ("psigma", "arrays", "model", "z", "z^2"))
    for _i, r in summary.iterrows():
        z = _z(r[model_col + "_mean"], r[model_col + "_sem"], e_mean, e_sem)
        print("    %-8.3f %8d %10.3f %10.3f %10.3f"
              % (r["psigma"], r["n_arrays"], r[model_col + "_mean"], z, z * z))


def _write(frames, prefix, xlsx):
    """Pickle each frame and add it to the workbook, replacing an older sheet."""
    frames = {k: to_output_names(v) for k, v in frames.items()}
    for name, frame in frames.items():
        path = os.path.join(RESULTS_DIR, "%s_%s.pkl" % (prefix, name)
                            if name else "%s.pkl" % prefix)
        frame.to_pickle(path)
    try:
        mode = "a" if os.path.isfile(xlsx) else "w"
        kw = dict(if_sheet_exists="replace") if mode == "a" else {}
        with pd.ExcelWriter(xlsx, engine="openpyxl", mode=mode, **kw) as writer:
            for name, frame in frames.items():
                sheet = "distance_vs_experiment"
                if name:
                    sheet += "_" + name
                frame.to_excel(writer, sheet_name=sheet[:31], index=False)
    except Exception as exc:                                # noqa: BLE001
        print("  xlsx failed (%s: %s); the pickles are written"
              % (type(exc).__name__, exc))
        return
    print("  added to %s: %s" % (os.path.basename(xlsx),
                                 ", ".join("distance_vs_experiment"
                                           + ("_" + n if n else "")
                                           for n in frames)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="P0")
    ap.add_argument("--scale-sensitivity", action="store_true",
                    help="also score with the smallest and largest movie diameter")
    ap.add_argument("--out-prefix", dest="prefix",
                    default="ablation_distance_vs_experiment")
    ap.add_argument("--xlsx", default="ablation_tables.xlsx",
                    help="workbook to add the sheets to, relative to the results dir")
    a = ap.parse_args()

    print("=" * 78)
    print("DISTANCE FROM ABLATION IN CELL DIAMETERS - model vs experiment (%s)"
          % a.stage)
    print("=" * 78)
    print("  experimental cell diameter (first frame, valid non-edge cells,"
          " %.2f um per pixel):" % PIXEL_UM)
    movies = experimental_diameter(a.stage)
    for _i, m in movies.iterrows():
        print("     %-40s %5d cells   diameter %.3f um"
              % (m["label"], m["n_cells"], m["cell_diameter_um"]))
    dias = movies["cell_diameter_um"].to_numpy(float)
    e_dia = float(dias.mean())
    print("     mean over movies: %.3f um  (spread %.3f-%.3f, %.1f%%)"
          % (e_dia, dias.min(), dias.max(), 100 * (dias.max() - dias.min()) / e_dia))

    exp = experimental_distances(a.stage)
    e_diff_um, e_ref_um = exp["diff"], exp["ref"]
    e_diff = e_diff_um / e_dia
    e_ref = e_ref_um / e_dia
    e_ratio = e_diff_um / e_ref_um              # paired: same repeat, same ablation
    print("\n  experiment, per biological repeat:")
    print("     differentiating um : %s" % np.round(e_diff_um, 2).tolist())
    print("     differentiating  D : %s" % np.round(e_diff, 2).tolist())
    print("     reference SC     D : %s" % np.round(e_ref, 2).tolist())
    print("     ratio diff/ref     : %s" % np.round(e_ratio, 3).tolist())

    runs = read_table(os.path.join(RESULTS_DIR, "ablation_runs.pkl"))
    events = read_table(os.path.join(RESULTS_DIR, "ablation_events.pkl"))
    runs = runs[runs["stage"] == a.stage]

    rows = []
    for _i, r in runs.iterrows():
        ev = events[events["run_folder"] == r["run_folder"]]
        if not len(ev):
            continue                            # no events -> no mean distance
        try:
            dia, n_cells, ref_mean, n_sc = model_scales(
                r["source_run"], r["t_strict_in_source"], r["ablated_label"])
        except Exception as exc:                # noqa: BLE001
            print("   pre-ablation frame failed for %s: %s"
                  % (r["run_folder"][-40:], exc))
            continue
        diff_mean = float(ev["distance"].mean())
        rows.append(dict(psigma=r["psigma"], initial_array=r["initial_array"],
                         run_folder=r["run_folder"], n_events=len(ev),
                         cell_diameter=dia, n_cells=n_cells,
                         diff_mean=diff_mean, ref_mean=ref_mean, n_sc=n_sc,
                         diff_diameters=diff_mean / dia,
                         ref_diameters=ref_mean / dia,
                         ratio=diff_mean / ref_mean))
    mod = pd.DataFrame(rows)
    if not len(mod):
        raise SystemExit("no model ablations with events for %s" % a.stage)

    cols = ["diff_mean", "ref_mean", "cell_diameter",
            "diff_diameters", "ref_diameters", "ratio"]
    em, es = _mean_sem(e_diff)
    rm, rs = _mean_sem(e_ratio)
    print("\n  model, by psigma (arrays averaged first, then across arrays):")
    print("    %-8s %5s %7s %8s %9s %9s %9s" %
          ("psigma", "runs", "arrays", "diam", "diff D", "ref D", "ratio"))
    sum_rows = []
    for ps, g in mod.groupby("psigma"):
        # one array is one independent sample, so it is averaged before the SEM:
        # repeats on the same array share an initial condition
        per_array = g.groupby("initial_array")[cols].mean()
        dm, dm_sem = _mean_sem(per_array["diff_diameters"])
        rd, _ = _mean_sem(per_array["ref_diameters"])
        ra, ra_sem = _mean_sem(per_array["ratio"])
        dia = float(per_array["cell_diameter"].mean())
        print("    %-8.3f %5d %7d %8.3f %9.2f %9.2f %9.3f"
              % (ps, len(g), len(per_array), dia, dm, rd, ra))
        sum_rows.append(dict(
            stage=a.stage, psigma=ps,
            n_ablations_with_events=len(g), n_arrays=len(per_array),
            n_events=int(g["n_events"].sum()),
            model_cell_diameter=dia,
            model_dist_lattice=float(per_array["diff_mean"].mean()),
            model_ref_lattice=float(per_array["ref_mean"].mean()),
            model_dist_diameters_mean=dm, model_dist_diameters_sem=dm_sem,
            model_ref_diameters_mean=rd,
            model_ratio_mean=ra, model_ratio_sem=ra_sem,
            exp_cell_diameter_um=e_dia,
            exp_cell_diameter_min_um=float(dias.min()),
            exp_cell_diameter_max_um=float(dias.max()),
            exp_n_movies=len(movies), exp_n_repeats=int(e_diff.size),
            exp_dist_um_mean=float(np.mean(e_diff_um)),
            exp_dist_diameters_mean=em, exp_dist_diameters_sem=es,
            exp_ref_diameters_mean=float(np.mean(e_ref)),
            exp_ratio_mean=rm, exp_ratio_sem=rs,
            z_diameters=_z(dm, dm_sem, em, es),
            z_ratio=_z(ra, ra_sem, rm, rs),
            # what one lattice unit is worth, if the two cell sizes are the same cell
            um_per_lattice_unit=e_dia / dia if dia else np.nan))
    summary = pd.DataFrame(sum_rows)
    summary["z2_diameters"] = summary["z_diameters"] ** 2
    summary["z2_ratio"] = summary["z_ratio"] ** 2

    print("\n  z = (model - experiment) / sqrt(SEM_model^2 + SEM_exp^2)")
    _score("distance in cell diameters", summary,
           "model_dist_diameters", "exp_dist_diameters")
    _score("ratio diff/reference  (domain size divided out)", summary,
           "model_ratio", "exp_ratio")

    if a.scale_sensitivity:
        print("\n  the cell diameter is one common scale, so it shifts the"
              " experiment as a whole:")
        for label, d in (("smallest movie", dias.min()), ("largest movie", dias.max())):
            m, s = _mean_sem(e_diff_um / d)
            print("     %-16s diameter %.3f um -> experiment %.3f +- %.3f"
                  % (label, d, m, s))
            for _i, r in summary.iterrows():
                z = _z(r["model_dist_diameters_mean"],
                       r["model_dist_diameters_sem"], m, s)
                print("        psigma %.3f  z^2 = %.3f" % (r["psigma"], z * z))

    # the experimental side, as measured: movie rows carry the cell size, repeat
    # rows the distances (divided by the mean diameter over the movies)
    exp_rows = [dict(stage=a.stage, unit="movie", label=m["label"],
                     n_cells=m["n_cells"], cell_diameter_um=m["cell_diameter_um"])
                for _i, m in movies.iterrows()]
    for label in e_diff_um.index:
        exp_rows.append(dict(
            stage=a.stage, unit="repeat", label=str(label),
            cell_diameter_um=e_dia,
            dist_um=float(e_diff_um[label]), ref_um=float(e_ref_um[label]),
            dist_diameters=float(e_diff[label]), ref_diameters=float(e_ref[label]),
            ratio=float(e_ratio[label])))
    exp_table = pd.DataFrame(exp_rows)

    mod.insert(0, "stage", a.stage)
    _write({"": summary, "runs": mod, "exp": exp_table}, a.prefix,
           os.path.join(RESULTS_DIR, a.xlsx))
    print("  summary %d rows x %d cols, runs %d x %d, exp %d x %d"
          % (*summary.shape, *mod.shape, *exp_table.shape))


if __name__ == "__main__":
    main()
