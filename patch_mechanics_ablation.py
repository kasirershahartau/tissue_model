"""Retire the fabricated ablation ratios in the saved mechanics tables.

    python patch_mechanics_ablation.py --dry-run     # report, change nothing
    python patch_mechanics_ablation.py

In some ablation archives the post-ablation frame IS the pre-ablation frame:
every cell's area is unchanged, so each area-change ratio is exactly 1 and the
HC-over-mean-SC ratio is a meaningless 1.0 with zero spread. build_mechanics_table
recorded that 1.0 as if it were a measurement, and anything averaging
``ablation_ratio_mean`` pulls its answer towards 1.

The mechanical fit itself excluded those sheets from the ablation term while
keeping their valid roundness (mechanics_eval, "Skip ONLY the ablation term"),
and dropping them here reproduces the per-term chi^2 stored in
mechanics_points.pkl to eight decimals. So the fit was right and the table is
wrong; this patches the table to agree.

WHAT IT CHANGES. In mechanics_runs.pkl, for the affected runs only, the six
ablation summary fields become NaN — the ablation was not measured, and NaN is
what "not measured" looks like. Every other field, roundness and shrinkage
included, is untouched: the base run is fine, only its ablation partner is
degenerate. A new ``ablation_measured`` column records the reason so the rows
stay explicable.

WHAT IT DOES NOT CHANGE. The per-cell tables keep their rows. Those are faithful
records — the areas really are equal — so they are flagged with the same column
rather than deleted, and can be filtered on it.

The builder now applies the same rule (see ablation_measured), so a future
rebuild produces this directly and running this script again is a no-op.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR

# zeroed out for a run whose ablation did not move anything
ABLATION_FIELDS = ["ablation_ratio_mean", "ablation_ratio_sem",
                   "hc_area_change_mean", "hc_area_change_sem",
                   "sc_area_change_mean", "sc_area_change_sem"]
PER_CELL = ["mechanics_hc_ablation", "mechanics_sc_ablation"]


def degenerate_runs(runs):
    """Runs whose ablation left every HC and every SC area exactly unchanged.

    Read off the saved summary rather than the per-cell tables: both mean area
    changes are exactly 1.0 only when nothing moved. Already-patched rows carry
    NaN there and so are not selected again.
    """
    return (np.isclose(runs["hc_area_change_mean"], 1.0)
            & np.isclose(runs["sc_area_change_mean"], 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copies (rebuilding these costs hours)")
    ap.add_argument("--xlsx", default="mechanics_tables.xlsx",
                    help="workbook whose sheets are refreshed, in the results dir")
    a = ap.parse_args()

    runs_path = os.path.join(RESULTS_DIR, "mechanics_runs.pkl")
    runs = pd.read_pickle(runs_path)
    bad = degenerate_runs(runs)
    print("mechanics_runs.pkl: %d runs, %d with an ablation measured, "
          "%d degenerate" % (len(runs), runs["ablation_ratio_mean"].notna().sum(),
                             int(bad.sum())))
    if bad.any():
        print("  by stage: %s"
              % runs[bad].groupby("stage").size().to_dict())
    if not bad.any():
        print("  nothing to patch")
        return

    frames = {"mechanics_runs": runs}
    for name in PER_CELL:
        frames[name] = pd.read_pickle(os.path.join(RESULTS_DIR, name + ".pkl"))

    folders = set(runs.loc[bad, "run_folder"])
    runs.loc[bad, ABLATION_FIELDS] = np.nan
    runs["ablation_measured"] = runs["ablation_ratio_mean"].notna()
    for name in PER_CELL:
        f = frames[name]
        f["ablation_measured"] = ~f["run_folder"].isin(folders)
        print("  %s: %d of %d rows flagged not measured"
              % (name, int((~f["ablation_measured"]).sum()), len(f)))

    if a.dry_run:
        print("\ndry run: nothing written")
        return

    for name, frame in frames.items():
        path = os.path.join(RESULTS_DIR, name + ".pkl")
        if not a.no_backup and not os.path.isfile(path + ".bak"):
            shutil.copy2(path, path + ".bak")
        frame.to_pickle(path)
        print("  wrote %s.pkl%s" % (name, "" if a.no_backup else " (.bak kept)"))

    xlsx = os.path.join(RESULTS_DIR, a.xlsx)
    if os.path.isfile(xlsx):
        try:
            # only the three touched sheets are rewritten; rebuilding the
            # workbook from scratch would mean re-reading every archive
            with pd.ExcelWriter(xlsx, engine="openpyxl", mode="a",
                                if_sheet_exists="replace") as writer:
                for name, frame in frames.items():
                    sheet = name.replace("mechanics_", "")
                    frame.to_excel(writer, sheet_name=sheet[:31], index=False)
                    print("  refreshed sheet %s" % sheet)
        except Exception as exc:                            # noqa: BLE001
            print("  xlsx failed (%s: %s); the pickles are patched"
                  % (type(exc).__name__, exc))


if __name__ == "__main__":
    main()
