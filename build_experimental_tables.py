"""The experimental data the model is scored against, as tables.

    python build_experimental_tables.py

Every number in the fullmodel and mechanics workbooks is a comparison against
these values, but the values themselves lived only inside the loaders. This
writes them out, one row per experimental unit, so a reader can see what the
model was asked to reproduce without running anything.

Adds to <results>/:
    fullmodel_tables.xlsx    sheets: experiment, experiment_summary
    mechanics_tables.xlsx    sheets: experiment, experiment_summary
    fullmodel_experiment.pkl, fullmodel_experiment_summary.pkl
    mechanics_experiment.pkl, mechanics_experiment_summary.pkl

WHAT A ROW IS. One biological unit — a movie, or an ablation repeat — and one
quantity. The detail tables are long rather than wide on purpose: the score-1
quantities come from the frame-1 segmentations (experiments 1-3) and the score-2
quantities from separate per-experiment differentiation files (experiments 0-2),
and nothing establishes that experiment 1 of one is experiment 0 of the other.
Putting them in one row would assert a pairing the scoring never uses — every
comparison treats each list as an unordered set of replicates.

HOW THE SUMMARY IS USED. Both stages of scoring average within a unit first, then
across units, so ``mean`` and ``sem`` here are over the units. The full-model
n-sigma combines both sides, sqrt(SEM_sim^2 + SEM_exp^2); the mechanical fit
divides by the experimental SEM alone (the model side is a single pooled mean).
"""
import argparse
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, experimental_results_folder,
                             load_experimental_results,
                             _exp_neighbor_pair_percentages,
                             _experimental_late_cells_info_path)

# (pickle base name, sheet name) for the tables this script adds to a workbook.
# The builders re-attach them when they rewrite their workbook, so the sheets
# survive a rebuild -- see carried_over_sheets.
FULLMODEL_SHEETS = [("fullmodel_experiment", "experiment"),
                    ("fullmodel_experiment_summary", "experiment_summary"),
                    # written by plot_face_stress_groups.py
                    ("fullmodel_face_stress", "face_stress")]
MECHANICS_SHEETS = [("mechanics_experiment", "experiment"),
                    ("mechanics_experiment_summary", "experiment_summary")]
STAGES = ("E17.5", "P0")


def _mean_sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else np.nan,
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def _row(stage, term, used_by, unit, value, n, n_units):
    return dict(stage=stage, term=term, used_by=used_by, unit=unit,
                value=float(value), n=int(n), n_units=n_units)


def _isolated_sc(cells_info, contact_matrix):
    """(isolated SCs, valid SCs, valid cells) in one segmented frame.

    An isolated SC is a support cell with no hair cell touching it — a hole in
    the pattern. Same valid-cell mask and type column as the neighbour-pair
    percentages, so the counts refer to the same population.
    """
    valid = cells_info.valid.values.astype(bool)
    is_hc = (cells_info.type.values == 1)[valid]
    A = np.asarray(contact_matrix)[np.ix_(valid, valid)] > 0
    np.fill_diagonal(A, False)
    iso = int(((~is_hc) & (A[:, is_hc].sum(axis=1) == 0)).sum())
    return iso, int((~is_hc).sum()), int(valid.sum())


# ------------------------------------------------------------- full model ----

def fullmodel_experimental_table():
    """The targets of score 1, score 2 and the isolated-SC statistic.

    Score 1 compares the composition of the tissue: what fraction of the
    cell-cell contacts are HC:HC and what fraction HC:SC, in the segmented frame.
    Score 2 compares who differentiates: the percentage of differentiating cells
    that had 0, and that had 1, HC neighbour when they did. The isolated-SC
    percentage replaced the old score 3 and appears twice: on the frame-1
    segmentation score 1 is matched to, and on the last recorded frame, which is
    the frame the model's own end-of-run number should be compared against.
    """
    rows = []
    for stage in STAGES:
        prefix = "E17" if stage == "E17.5" else "P0"
        for e in (1, 2, 3):
            base = "%s_experiment%d" % (prefix, e)
            ci = pd.read_pickle(os.path.join(experimental_results_folder, stage,
                                             base + "_cells_info_frame_1"))
            cm = np.load(os.path.join(experimental_results_folder, stage,
                                      base + "_contact_matrix_frame_1.npy"))
            hchc, hcsc, _scsc = _exp_neighbor_pair_percentages(ci, cm)
            iso, n_sc, n_valid = _isolated_sc(ci, cm)

            for term, value, n in (
                    ("pct_HCHC_contacts", hchc, n_valid),
                    ("pct_HCSC_contacts", hcsc, n_valid)):
                rows.append(_row(stage, term, "score 1", base, value, n,
                                 "valid cells"))
            rows.append(_row(stage, "pct_SC_no_HC_neighbour_of_SC", "isolated SC",
                             base, 100.0 * iso / max(n_sc, 1), n_sc, "valid SCs"))
            rows.append(_row(stage, "pct_SC_no_HC_neighbour_of_all_cells",
                             "isolated SC", base, 100.0 * iso / max(n_valid, 1),
                             n_valid, "valid cells"))

            # The same statistic at the LAST recorded frame (+24h). The model
            # measures its isolated SCs at the end of the run, so this is the
            # like-for-like target; the frame-1 rows above are what score 1's
            # composition is matched on.
            late_ci = pd.read_pickle(_experimental_late_cells_info_path(stage, e))
            late_cm = np.load(_experimental_late_cells_info_path(
                stage, e, "contact_matrix"))
            iso, n_sc, n_valid = _isolated_sc(late_ci, late_cm)
            late = os.path.basename(_experimental_late_cells_info_path(stage, e))
            rows.append(_row(stage, "pct_SC_no_HC_neighbour_of_SC_final",
                             "isolated SC, final frame", late,
                             100.0 * iso / max(n_sc, 1), n_sc, "valid SCs"))
            rows.append(_row(stage, "pct_SC_no_HC_neighbour_of_all_cells_final",
                             "isolated SC, final frame", late,
                             100.0 * iso / max(n_valid, 1), n_valid,
                             "valid cells"))

        for i in (0, 1, 2):
            counts = np.asarray(np.load(os.path.join(
                experimental_results_folder,
                "%s differentiating cells_experiment%d.npy" % (stage, i))), float)
            if counts.size == 0:
                continue                        # nothing differentiated: no target
            unit = "%s differentiating cells_experiment%d" % (stage, i)
            for k in (0, 1):
                rows.append(_row(stage, "pct_events_%d_HC_neighbours" % k,
                                 "score 2", unit, 100.0 * np.mean(counts == k),
                                 counts.size, "differentiating cells"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- mechanics ----

def mechanics_experimental_table():
    """The three fit terms, plus the absolute quantities the ratios are built from.

    The fit is scored on ratios — HC over the mean SC — because they are
    dimensionless and so comparable to the model without a length calibration.
    The absolute roundness and area-change values are carried alongside as
    context: they are what the ratio was formed from, and the mechanics runs
    table stores the model's own absolute values next to them.
    """
    rows = []
    for stage in STAGES:
        for term, exp_type, used_by, unit_kind in (
                ("roundness_ratio", "HC to SC roundness ratio", "fit term", "HCs"),
                ("hc_roundness", "HC roundness", "context", "HCs"),
                ("sc_roundness", "SC roundness", "context", "SCs"),
                ("ablation_ratio", "HC to SC area change ratio after ablation",
                 "fit term", "ablation-adjacent HCs"),
                ("shrinkage", "cut shrinkage", "fit term", "cut discs")):
            try:
                arrays = load_experimental_results(stage, exp_type)
            except Exception as exc:            # noqa: BLE001
                print("  %s / %s unavailable (%s: %s)"
                      % (stage, term, type(exc).__name__, exc))
                continue
            for i, arr in enumerate(arrays, start=1):
                a = np.asarray(arr, float)
                a = a[np.isfinite(a)]
                if not a.size:
                    continue
                # The fit averages within a repeat before comparing repeats, so
                # the repeat's value is the mean of its cells.
                rows.append(_row(stage, term, used_by,
                                 "experiment %d" % i, a.mean(), a.size, unit_kind))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ output ----

def summarize(detail):
    """Mean and SEM over the experimental units — the target the score sees."""
    rows = []
    for (stage, term), g in detail.groupby(["stage", "term"], sort=False):
        m, s = _mean_sem(g["value"])
        rows.append(dict(stage=stage, term=term, used_by=g["used_by"].iloc[0],
                         n_experiments=len(g), mean=m, sem=s,
                         min=float(g["value"].min()), max=float(g["value"].max())))
    return pd.DataFrame(rows)


def add_sheets(xlsx, frames):
    """Write each frame into the workbook, replacing a sheet of the same name."""
    try:
        mode = "a" if os.path.isfile(xlsx) else "w"
        kw = dict(if_sheet_exists="replace") if mode == "a" else {}
        with pd.ExcelWriter(xlsx, engine="openpyxl", mode=mode, **kw) as writer:
            for sheet, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet[:31], index=False)
    except Exception as exc:                    # noqa: BLE001
        print("  %s failed (%s: %s); the pickles are written"
              % (os.path.basename(xlsx), type(exc).__name__, exc))
        return False
    return True


def to_output_names(frame):
    """The frame as it should be PUBLISHED: psigma written as pT.

    The inverse of read_table. Applied by the builders just before they save, so
    a rebuild produces the manuscript's name without any internal renaming.
    """
    return frame.rename(columns={c: c.replace("psigma", "pT")
                                 for c in frame.columns if "psigma" in c})


def read_table(path):
    """Read a saved table, with the parameter under the name the code uses.

    The deliverable tables call the Hill gate's half-max ``pT`` (the manuscript's
    name, applied by rename_psigma_to_pT.py); every script here calls it
    ``psigma``. Mapping it back on load keeps one name inside the code and the
    other in what gets published, instead of renaming hundreds of references.
    """
    frame = pd.read_pickle(path)
    return frame.rename(columns={c: c.replace("pT", "psigma")
                                 for c in frame.columns if "pT" in c})


def carried_over_sheets(specs):
    """(sheet, frame) for tables another script added to a workbook.

    A builder rewrites its whole workbook, which would drop sheets it did not
    produce. Calling this while writing re-attaches them from the pickles they
    were saved as; a sheet whose pickle is missing is simply skipped.
    """
    out = []
    for pkl, sheet in specs:
        path = os.path.join(RESULTS_DIR, pkl + ".pkl")
        if os.path.isfile(path):
            out.append((sheet[:31], pd.read_pickle(path)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-xlsx", dest="xlsx", action="store_false",
                    help="write the pickles only")
    a = ap.parse_args()

    for label, build, specs, workbook in (
            ("full model", fullmodel_experimental_table, FULLMODEL_SHEETS,
             "fullmodel_tables.xlsx"),
            ("mechanics", mechanics_experimental_table, MECHANICS_SHEETS,
             "mechanics_tables.xlsx")):
        print("%s:" % label)
        detail = build()
        summary = summarize(detail)
        frames = {}
        for (pkl, sheet), frame in zip(specs, (detail, summary)):
            frame.to_pickle(os.path.join(RESULTS_DIR, pkl + ".pkl"))
            frames[sheet] = frame
        for _i, r in summary.iterrows():
            print("   %-6s %-38s %-11s %d exp  %9.3f +- %.3f"
                  % (r["stage"], r["term"], r["used_by"], r["n_experiments"],
                     r["mean"], r["sem"]))
        if a.xlsx and add_sheets(os.path.join(RESULTS_DIR, workbook), frames):
            print("   -> %s: %s (%d rows), %s (%d rows)"
                  % (workbook, specs[0][1], len(detail), specs[1][1], len(summary)))
        print()


if __name__ == "__main__":
    main()
