"""One row per full-model run: parameters + the differentiation measurements.

    python build_run_table.py                    # every fullmodel_* run
    python build_run_table.py --workers 10
    python build_run_table.py --pattern "fullmodel_ps*_ks-0.080_*"

Writes <results>/full_model_runs.xlsx, plus an incremental
<results>/full_model_runs.csv that is appended after EVERY run. The CSV is the
crash-safe record: re-running skips whatever is already in it (--no-resume to
start over), because a full pass is a few hours of history-file scanning.

The measured columns are exactly the per-run quantities that
compare_full_model_differentiation_to_experiments pools into scores 1-3, taken
from the SAME functions so the table and the scores cannot drift:

  chosen initial frame      _best_matching_frame_by_neighbor_pairs - the frame
                            whose (HC:HC%, HC:SC%) is closest to the experimental
                            frame-1 target for that stage. Everything below is
                            measured from it.
  % HC:HC / HC:SC / SC:SC   _sim_neighbor_pair_percentages at that frame (the
                            three are complementary and sum to 100)
  # differentiating cells   cells that became HC after the chosen frame
  ...out of differentiating calc_HC_neighbors_at_differentiation, binned 0/1/>=2
                            - HC-neighbour count AT the moment of differentiation
  ...out of SCs with the    calc_percentage_of_differentiating_by_initial_neighbors
  same count initially      - denominator is SCs in that bin at the chosen frame

Parameters come from each run's own parameters.txt, not from the sweep scripts'
constants, so a row always describes the run that actually happened.
"""
import argparse
import fnmatch
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, load_history_file,
                             experimental_results_folder,
                             _exp_neighbor_pair_percentages,
                             _sim_neighbor_pair_percentages,
                             _best_matching_frame_by_neighbor_pairs,
                             calc_HC_neighbors_at_differentiation,
                             calc_percentage_of_differentiating_by_initial_neighbors)

COLUMNS = [
    "model name", "stage", "initial array",
    "pS", "pR", "gammaSC", "alphaHC ratio", "A0", "HC p0", "SC p0",
    # psigma / K are not in the requested list but vary across these runs and are
    # parameters like the rest - without them the rows are only distinguishable
    # by parsing the model name.
    "psigma", "K (stress shift)",
    "chosen initial frame",
    "% HC:HC bonds in chosen initial frame",
    "% HC:SC bonds in chosen initial frame",
    "% SC:SC bonds in chosen initial frame",
    "# differentiating cells",
    "% differentiating cells with 0 HC neighbors out of differentiating cells",
    "% differentiating cells with 1 HC neighbors out of differentiating cells",
    "% differentiating cells with 2 or more HC neighbors out of differentiating cells",
    "% differentiating cells with 0 HC neighbors out of SCs with 0 HC neighbors at chosen initial frame",
    "% differentiating cells with 1 HC neighbors out of SCs with 1 HC neighbors at chosen initial frame",
    "% differentiating cells with 2 or more HC neighbors out of SCs with 2 or more HC neighbors at chosen initial frame",
]


def read_parameters(name):
    """parameters.txt -> {key: raw string}. Values may be dicts/lists/quoted."""
    path = os.path.join(RESULTS_DIR, name, "parameters.txt")
    out = {}
    with open(path) as fh:
        for line in fh:
            key, sep, val = line.partition(":")
            if sep and not key.startswith(" "):
                out[key.strip()] = val.strip()
    return out


def _num(params, key):
    try:
        return float(params[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def stage_of(name):
    if name.endswith("_for_E17"):
        return "E17.5"
    if name.endswith("_for_P0"):
        return "P0"
    return None


def array_of(name):
    import re
    m = re.search(r"random_periodic_array(\d+)_for_", name)
    return int(m.group(1)) if m else np.nan


_TARGETS = {}


def experimental_targets(stage):
    """Mean experimental (HC:HC%, HC:SC%) at frame 1 - the frame-matching target."""
    if stage in _TARGETS:
        return _TARGETS[stage]
    prefix = "E17" if stage == "E17.5" else "P0"
    hchc, hcsc = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        a, b, _ = _exp_neighbor_pair_percentages(ci, cm)
        hchc.append(a); hcsc.append(b)
    _TARGETS[stage] = (float(np.nanmean(hchc)), float(np.nanmean(hcsc)))
    return _TARGETS[stage]


def one_run(args):
    name, type_by, threshold, max_nb = args
    try:
        stage = stage_of(name)
        if stage is None:
            return name, None, "cannot infer stage from name"
        params = read_parameters(name)
        row = {
            "model name": name,
            "stage": stage,
            "initial array": array_of(name),
            "pS": _num(params, "notch_sensitivity"),
            "pR": _num(params, "repressor_sensitivity"),
            "gammaSC": _num(params, "gammaSC"),
            "alphaHC ratio": _num(params, "alphaHC_ratio"),
            "A0": _num(params, "preferred_area_override"),
            "HC p0": _num(params, "hc_shape_index"),
            "SC p0": _num(params, "sc_shape_index"),
            "psigma": _num(params, "psigma"),
            # K only acts when the gate is on; report it as NaN otherwise so a
            # psigma=0 baseline is not mistaken for a run at that threshold.
            "K (stress shift)": (_num(params, "stress_shift")
                                 if str(params.get("stress_dependent")) == "True"
                                 else np.nan),
        }

        history = load_history_file(name)
        tgt_hchc, tgt_hcsc = experimental_targets(stage)
        t0, _, _ = _best_matching_frame_by_neighbor_pairs(
            history, tgt_hchc, tgt_hcsc, type_by, threshold)

        sheet = history.retrieve(float(t0))
        sheet.arrange_sheet_from_history()
        hchc, hcsc, scsc = _sim_neighbor_pair_percentages(sheet, type_by, threshold)
        row["chosen initial frame"] = float(t0)
        row["% HC:HC bonds in chosen initial frame"] = hchc
        row["% HC:SC bonds in chosen initial frame"] = hcsc
        row["% SC:SC bonds in chosen initial frame"] = scsc

        counts = np.asarray(calc_HC_neighbors_at_differentiation(
            history, initial_time_point=t0, type_by=type_by,
            threshold=threshold), dtype=float)
        counts = counts[~np.isnan(counts)]
        n = counts.size
        row["# differentiating cells"] = int(n)
        for nb, lab in ((0, "0"), (1, "1"), (2, "2 or more")):
            sel = (counts == nb) if nb < 2 else (counts >= 2)
            row["%% differentiating cells with %s HC neighbors out of "
                "differentiating cells" % lab] = (100.0 * sel.mean() if n else np.nan)

        pct = np.asarray(calc_percentage_of_differentiating_by_initial_neighbors(
            history, t0, max_number_of_neighbors=max_nb, type_by=type_by,
            threshold=threshold), dtype=float)
        for nb, lab in ((0, "0"), (1, "1"), (2, "2 or more")):
            row["%% differentiating cells with %s HC neighbors out of SCs with %s "
                "HC neighbors at chosen initial frame" % (lab, lab)] = (
                    float(pct[nb]) if nb < pct.size else np.nan)
        return name, row, None
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the table
        return name, None, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="fullmodel_*",
                    help="glob over run-folder names (default every fullmodel_*)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=0.355079)
    ap.add_argument("--max-neighbors", dest="max_nb", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "full_model_runs.xlsx"))
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the existing CSV and recompute every run")
    a = ap.parse_args()

    names = sorted(n for n in os.listdir(RESULTS_DIR)
                   if fnmatch.fnmatch(n, a.pattern)
                   and os.path.isdir(os.path.join(RESULTS_DIR, n))
                   and os.path.isfile(os.path.join(RESULTS_DIR, n, "parameters.txt")))
    csv_path = os.path.splitext(a.out)[0] + ".csv"
    done = pd.DataFrame(columns=COLUMNS)
    if os.path.isfile(csv_path):
        if a.no_resume:
            os.remove(csv_path)          # else the appends below would extend it
            print("--no-resume: discarded the previous %s" % os.path.basename(csv_path))
        else:
            done = pd.read_csv(csv_path)
            names = [n for n in names if n not in set(done["model name"])]
            print("resuming: %d rows already in %s" % (len(done), csv_path))
    print("%d run(s) to measure | %d worker(s)" % (len(names), a.workers), flush=True)

    rows, failed = [], []
    tasks = [(n, a.type_by, a.threshold, a.max_nb) for n in names]
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=a.workers)
        it = pool.map(one_run, tasks)
    else:
        pool, it = None, (one_run(t) for t in tasks)
    try:
        for i, (name, row, err) in enumerate(it, 1):
            if err:
                failed.append((name, err))
                print("  [%3d/%3d] %-58s FAILED %s" % (i, len(tasks), name[:58], err),
                      flush=True)
                continue
            rows.append(row)
            # Append after every run: a full pass is hours, and a crash at run
            # 190 must not throw away the first 189.
            pd.DataFrame([row], columns=COLUMNS).to_csv(
                csv_path, mode="a", header=not os.path.isfile(csv_path), index=False)
            print("  [%3d/%3d] %-58s t0=%5.2f  ndiff=%4d" %
                  (i, len(tasks), name[:58], row["chosen initial frame"],
                   row["# differentiating cells"]), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    parts = [p for p in (done, pd.DataFrame(rows, columns=COLUMNS)) if len(p)]
    df = (pd.concat(parts, ignore_index=True) if parts
          else pd.DataFrame(columns=COLUMNS))[COLUMNS]
    df = df.sort_values(["stage", "psigma", "K (stress shift)", "initial array"],
                        na_position="first").reset_index(drop=True)

    with pd.ExcelWriter(a.out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="full model runs", index=False)
        ws = xl.sheets["full model runs"]
        ws.freeze_panes = "C2"
        for j, col in enumerate(df.columns, 1):
            width = min(max(len(str(col)), 12) + 2, 46)
            ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
            ws.cell(row=1, column=j).alignment = \
                __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[1].height = 78
    print("\nwrote %s  (%d rows x %d cols)" % (a.out, len(df), len(df.columns)))
    print("incremental csv: %s" % csv_path)
    if failed:
        print("\n%d run(s) FAILED:" % len(failed))
        for n, e in failed:
            print("  %-58s %s" % (n[:58], e))


if __name__ == "__main__":
    main()
