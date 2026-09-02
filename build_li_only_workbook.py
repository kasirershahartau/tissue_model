"""Collect the lateral-inhibition-only tables into one workbook.

    python build_li_only_workbook.py
    python build_li_only_workbook.py --dir "D:/Kasirer/results/lateral inhibition only results"

That model is the same lateral inhibition from the same initial arrays with the
mechanics switched off, so its tables are the natural control for ours. They
arrive as separate pickles; this writes them as sheets of a single workbook, in
the same shape as fullmodel_tables.xlsx — per-run first, then the aggregates,
then the per-event detail.

The data is NOT modified: no rows dropped, no columns pruned, no renaming. These
are someone else's results, and the point of the workbook is to read them, not to
reshape them. Where a column means the same thing as one of ours it already
carries the same name.
"""
import argparse
import os

import pandas as pd

DEFAULT_DIR = r"D:\Kasirer\results\lateral inhibition only results"
# per-run, then the two aggregation levels, then per-event — matching the order
# our own workbooks use, coarsest context first
ORDER = ["runs", "repeat_summary", "morphology_summary", "events"]
EXCEL_MAX_ROWS = 1_048_576


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None,
                    help="workbook path (default: <dir>/lateral_inhibition_tables.xlsx)")
    a = ap.parse_args()

    found = {}
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".pkl"):
            continue
        obj = pd.read_pickle(os.path.join(a.dir, fn))
        if not isinstance(obj, pd.DataFrame):
            print("  skipping %s: %s, not a DataFrame" % (fn, type(obj).__name__))
            continue
        found[os.path.splitext(fn)[0]] = obj

    names = [n for n in ORDER if n in found] + \
            [n for n in sorted(found) if n not in ORDER]
    if not names:
        raise SystemExit("no DataFrame pickles in %s" % a.dir)

    out = a.out or os.path.join(a.dir, "lateral_inhibition_tables.xlsx")
    with pd.ExcelWriter(out) as writer:
        for n in names:
            f = found[n]
            if len(f) > EXCEL_MAX_ROWS - 1:
                print("  %s has %d rows, past Excel's limit — writing the first %d"
                      % (n, len(f), EXCEL_MAX_ROWS - 1))
                f = f.head(EXCEL_MAX_ROWS - 1)
            f.to_excel(writer, sheet_name=n[:31], index=False)
            print("  %-20s %7d rows x %2d cols" % (n, *found[n].shape))

    print("\nwrote %s (%.1f MB)" % (out, os.path.getsize(out) / 2**20))


if __name__ == "__main__":
    main()
