"""Rename the psigma parameter to pT in the saved tables.

    python rename_psigma_to_pT.py --dry-run
    python rename_psigma_to_pT.py

The Hill gate's half-max is called pT in the manuscript. This renames it in the
DELIVERABLES — the pickles and the workbook sheets — and in the sheet that is
itself named after the parameter. The code keeps calling it psigma internally;
readers normalise the name on load (see build_experimental_tables.read_table),
and the two builders rename it again when they next write their output, so a
rebuild does not undo this.

Any column whose name contains "psigma" is renamed, not just the bare one, so a
derived column would be caught too.
"""
import argparse
import os
import shutil

import pandas as pd

from post_processing import RESULTS_DIR

OLD, NEW = "psigma", "pT"
# pickle base name -> (workbook, sheet). The sheet named after the parameter is
# renamed as well; every other sheet keeps its name and gets new column headers.
TABLES = [
    ("fullmodel_runs", "fullmodel_tables.xlsx", "runs"),
    ("fullmodel_psigma", "fullmodel_tables.xlsx", "psigma"),
    ("fullmodel_events", "fullmodel_tables.xlsx", "events"),
    ("fullmodel_face_stress", "fullmodel_tables.xlsx", "face_stress"),
    ("ablation_overall", "ablation_tables.xlsx", "overall"),
    ("ablation_runs", "ablation_tables.xlsx", "runs"),
    ("ablation_events", "ablation_tables.xlsx", "events"),
    ("ablation_distance_vs_experiment", "ablation_tables.xlsx",
     "distance_vs_experiment"),
    ("ablation_distance_vs_experiment_runs", "ablation_tables.xlsx",
     "distance_vs_experiment_runs"),
]
SHEET_RENAME = {"psigma": "pT"}


def renamed(columns):
    return {c: c.replace(OLD, NEW) for c in columns if OLD in c}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args()

    by_workbook = {}
    for base, workbook, sheet in TABLES:
        path = os.path.join(RESULTS_DIR, base + ".pkl")
        if not os.path.isfile(path):
            print("  %-40s missing, skipped" % (base + ".pkl"))
            continue
        frame = pd.read_pickle(path)
        cols = renamed(frame.columns)
        target = SHEET_RENAME.get(sheet, sheet)
        print("  %-40s %-22s -> %-14s %s"
              % (base + ".pkl", sheet, target,
                 ", ".join("%s->%s" % kv for kv in cols.items()) or "no column"))
        if not cols and target == sheet:
            continue
        frame = frame.rename(columns=cols)
        if not a.dry_run:
            if not a.no_backup and not os.path.isfile(path + ".bak"):
                shutil.copy2(path, path + ".bak")
            frame.to_pickle(path)
        by_workbook.setdefault(workbook, []).append((sheet, target, frame))

    if a.dry_run:
        print("\ndry run: nothing written")
        return

    for workbook, entries in by_workbook.items():
        xlsx = os.path.join(RESULTS_DIR, workbook)
        if not os.path.isfile(xlsx):
            print("  %s missing; pickles are renamed" % workbook)
            continue
        try:
            # rename the worksheet in place first — writing a frame under the new
            # name would otherwise leave the old sheet behind
            import openpyxl
            wb = openpyxl.load_workbook(xlsx)
            for sheet, target, _f in entries:
                if sheet != target and sheet in wb.sheetnames:
                    wb[sheet].title = target
            wb.save(xlsx)
            wb.close()
            with pd.ExcelWriter(xlsx, engine="openpyxl", mode="a",
                                if_sheet_exists="replace") as writer:
                for _sheet, target, frame in entries:
                    frame.to_excel(writer, sheet_name=target[:31], index=False)
            print("  %s: %d sheet(s) rewritten" % (workbook, len(entries)))
        except Exception as exc:                            # noqa: BLE001
            print("  %s failed (%s: %s); the pickles are renamed"
                  % (workbook, type(exc).__name__, exc))


if __name__ == "__main__":
    main()
