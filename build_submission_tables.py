"""Write the two deliverable workbooks: the full sweep in one, the chosen pT in the other.

    python build_submission_tables.py

These are the raw-data tables that go to the journal beside the manuscript.
They are assembled here rather than by re-running the two builders, because
each builder would destroy something the submission needs:

  fullmodel_tables.xlsx   EVERY pT in the sweep, not just the two with run
                          folders on this machine. build_fullmodel_table
                          filters its output with ``isin(all_names)`` -- the
                          run FOLDERS present -- which is right when it
                          re-measures archives, but here it would throw away
                          the 900 rows the cloud measured and whose histories
                          were never downloaded. The widened
                          ``fullmodel_runs.pkl`` holds them all.

  ablation_tables.xlsx    ONLY the chosen pT, as the manuscript reports it.
                          Filtered from the existing three-pT pickles rather
                          than rebuilt, so the pT = 0 and 0.08667 rows stay in
                          those pickles and the comparison FIGURES -- which
                          need pT = 0 beside the chosen value -- keep working.

Both get psigma -> pT through ``to_output_names``, columns and sheet names
alike: the manuscript calls the gate pT and the submitted tables must agree
with it.

The events sheet necessarily covers only the pT values with local archives
(pT = 0 and the chosen value); see add_fullmodel_events.py. Existing workbooks
are backed up to ``*.pre_tablerewrite`` before the first overwrite.
"""
import os
import shutil

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from build_experimental_tables import (read_table, to_output_names,
                                       carried_over_sheets, FULLMODEL_SHEETS)
from build_fullmodel_table import psigma_table
from build_ablation_table import ABLATION_SHEETS
from best_pt import BEST_PT


def backup(name):
    p = os.path.join(RESULTS_DIR, name)
    b = p + ".pre_tablerewrite"
    if os.path.isfile(p) and not os.path.isfile(b):
        shutil.copy2(p, b)
        print("  backed up %s" % os.path.basename(b))


def pt_col(frame):
    for c in ("pT", "psigma"):
        if c in frame.columns:
            return c
    return None


def fullmodel():
    print("=" * 78)
    print("fullmodel_tables.xlsx -- every pT in the sweep")
    print("=" * 78)
    backup("fullmodel_tables.xlsx")

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    c = pt_col(runs)
    print("  runs table: %d rows, %d pT value(s)"
          % (len(runs), runs[c].nunique()))
    # drop_collapsed=False: the published objective was computed with the
    # collapsed runs kept, and this sheet has to match the fit it reports on.
    # The FIGURES drop them, each saying so where it does.
    pdf = psigma_table(runs, drop_collapsed=False)
    print("  per-point table: %d rows" % len(pdf))

    runs_out, pdf_out = (to_output_names(f) for f in (runs, pdf))
    events = pd.DataFrame()
    ev_p = os.path.join(RESULTS_DIR, "fullmodel_events.pkl")
    if os.path.isfile(ev_p):
        events = to_output_names(read_table(ev_p))

    xlsx = os.path.join(RESULTS_DIR, "fullmodel_tables.xlsx")
    with pd.ExcelWriter(xlsx) as w:
        runs_out.to_excel(w, sheet_name="runs", index=False)
        pdf_out.to_excel(w, sheet_name="pT", index=False)
        if len(events):
            events.to_excel(w, sheet_name="events", index=False)
            print("  events sheet: %d rows" % len(events))
        for name, frame in carried_over_sheets(FULLMODEL_SHEETS):
            to_output_names(frame).to_excel(w, sheet_name=name, index=False)
            print("  carried over %s (%d rows)" % (name, len(frame)))
    print("  wrote %s" % xlsx)
    print("    pT values: %s" % sorted(set(pdf_out[pt_col(pdf_out)])))


def ablation():
    print()
    print("=" * 78)
    print("ablation_tables.xlsx -- only pT = %g" % BEST_PT)
    print("=" * 78)
    backup("ablation_tables.xlsx")

    sheets = {}
    for stem, sheet in (("ablation_overall", "overall"),
                        ("ablation_runs", "runs"),
                        ("ablation_events", "events")):
        p = os.path.join(RESULTS_DIR, stem + ".pkl")
        if not os.path.isfile(p):
            print("  %s.pkl missing, skipped" % stem)
            continue
        f = read_table(p)
        col = pt_col(f)
        n0 = len(f)
        if col:
            f = f[np.isclose(f[col].astype(float), BEST_PT)]
        sheets[sheet] = to_output_names(f)
        print("  %-18s %4d of %4d rows kept" % (sheet, len(f), n0))

    xlsx = os.path.join(RESULTS_DIR, "ablation_tables.xlsx")
    with pd.ExcelWriter(xlsx) as w:
        for sheet, f in sheets.items():
            f.to_excel(w, sheet_name=sheet, index=False)
        for name, frame in carried_over_sheets(ABLATION_SHEETS):
            f = frame
            col = pt_col(f)
            if col:
                f = f[np.isclose(f[col].astype(float), BEST_PT)]
            to_output_names(f).to_excel(w, sheet_name=name, index=False)
            print("  carried over %-28s %4d of %4d rows kept"
                  % (name, len(f), len(frame)))
    print("  wrote %s" % xlsx)


def check():
    print()
    print("=" * 78)
    print("any 'psigma' left in either workbook?")
    for fn in ("fullmodel_tables.xlsx", "ablation_tables.xlsx"):
        bad = []
        sheets = pd.read_excel(os.path.join(RESULTS_DIR, fn), sheet_name=None)
        for sheet, f in sheets.items():
            hits = [c for c in f.columns if "psigma" in str(c).lower()]
            if hits:
                bad.append("%s:%s" % (sheet, ",".join(hits)))
        print("  %-26s %s" % (fn, "; ".join(bad) if bad else "none - all pT"))


if __name__ == "__main__":
    fullmodel()
    ablation()
    check()
