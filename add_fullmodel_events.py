"""Add one pT's differentiation events to the full-model events table.

    python add_fullmodel_events.py                # the chosen pT
    python add_fullmodel_events.py --pt 0 --workers 6

``build_fullmodel_table`` writes events only for the runs it MEASURES. The
chosen pT's rows reached ``fullmodel_runs.pkl`` already measured, from the
cloud sweep, so its pass skipped them and the events sheet covered pT = 0
alone; the VM's own ``fullmodel_events.csv`` was never downloaded and the VM is
gone. All 60 runs at the chosen pT were re-created locally for the face-stress
rebuild, so the events can be recomputed from the archives that are here.

Computed with ``differentiation_events``, the same function
build_fullmodel_table uses, at each run's own ``t0`` taken from the runs table,
so the new rows are the same quantity measured the same way as the pT = 0 ones.

WRITES NOTHING BUT ``fullmodel_events.{pkl,csv}``. Re-running
build_fullmodel_table instead would refilter the runs table down to the runs
whose folders are on this machine and throw away the 900 sweep rows the cloud
measured, which is exactly what the widened table exists to keep.

Rows already present at this pT are replaced, not duplicated, so the script is
safe to re-run.
"""
import os
import sys

import numpy as np
import pandas as pd

from post_processing import RESULTS_DIR
from history_io import load_history_file
from build_experimental_tables import read_table
from build_fullmodel_table import differentiation_events, TYPE_BY, THRESHOLD
from best_pt import BEST_PT

OUT_PKL = os.path.join(RESULTS_DIR, "fullmodel_events.pkl")
OUT_CSV = os.path.join(RESULTS_DIR, "fullmodel_events.csv")


def one(args):
    name, stage, pt, array, repeat, t0 = args
    try:
        rows = differentiation_events(load_history_file(name), t0, THRESHOLD,
                                      type_by=TYPE_BY)
    except Exception as exc:                                   # noqa: BLE001
        return name, None, "%s: %s" % (type(exc).__name__, str(exc)[:80])
    for r in rows:
        r.update(model_name=name, stage=stage, psigma=pt,
                 initial_array=array, repeat=repeat, t0=t0)
    return name, rows, None


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", type=float, default=BEST_PT)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    col = "pT" if "pT" in runs.columns else "psigma"
    sel = runs[np.isclose(runs[col].astype(float), a.pt)]
    have = [r for _i, r in sel.iterrows()
            if os.path.isfile(os.path.join(RESULTS_DIR, r["model_name"],
                                           "history.hf5"))]
    print("pT = %g: %d row(s) in the table, %d with a local archive"
          % (a.pt, len(sel), len(have)))
    if not have:
        raise SystemExit("no local archives at that pT; nothing to do")

    old = read_table(OUT_PKL) if os.path.isfile(OUT_PKL) else pd.DataFrame()
    ocol = "pT" if "pT" in old.columns else "psigma"
    if len(old) and ocol in old.columns:
        already = set(old.loc[np.isclose(old[ocol].astype(float), a.pt),
                              "model_name"])
        if already:
            print("  %d run(s) already present at this pT; they will be"
                  " replaced" % len(already))

    tasks = [(r["model_name"], r["stage"], float(a.pt), int(r["initial_array"]),
              int(r["repeat"]), float(r["t0"])) for r in have]

    from concurrent.futures import ProcessPoolExecutor
    new, failed, done = [], [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, rows, err in ex.map(one, tasks):
            done += 1
            if err:
                failed.append((name, err))
            else:
                new.extend(rows)
            if done % 10 == 0:
                print("   %d/%d  (%d events so far)" % (done, len(tasks), len(new)),
                      flush=True)
    for name, err in failed:
        print("   FAILED %-56s %s" % (name, err))
    print("  %d event(s) from %d run(s)" % (len(new), len(tasks) - len(failed)))
    if not new:
        raise SystemExit("no events produced; nothing written")

    add = pd.DataFrame(new)
    if len(old):
        keep = old[~old["model_name"].isin(set(add["model_name"]))]
        out = pd.concat([keep, add], ignore_index=True)
    else:
        out = add
    out.to_pickle(OUT_PKL)
    out.to_csv(OUT_CSV, index=False)
    ocol = "pT" if "pT" in out.columns else "psigma"
    print("\nwrote %s: %d rows, pT %s"
          % (OUT_PKL, len(out), sorted(set(out[ocol].astype(float)))))
    for pt, g in out.groupby(ocol):
        print("   pT %-9g %5d event(s) over %3d run(s)"
              % (pt, len(g), g["model_name"].nunique()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
