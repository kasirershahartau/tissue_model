"""Calibrate simulation time against real time using the Notch-inhibition experiment.

    python calibrate_time_by_notch_inhibition.py --dry-run
    python calibrate_time_by_notch_inhibition.py --workers 4 --t-end 50

WHY THIS ANCHOR. The scoring window t0 -> t_score must correspond to the
experiment's 48 h window, so simulation time needs a conversion factor. The two obvious
anchors are both unusable with this data:

  * counting differentiation events - the imaging misses events, so the count is
    a lower bound of unknown tightness;
  * matching initial-frame to final-frame composition - the field of view drifts
    during acquisition, so the two frames do not cover the same cells. (An
    earlier attempt at this returned a 0.1-0.3 unit window, i.e. one to three
    frames, which is exactly the degenerate answer that assumption produces.)

The Notch-inhibition experiment avoids both. Blocking repressor production gives
a SATURATING endpoint - "after 48 h all cells are HC" - that needs neither cell
tracking nor a stable field of view: you only have to see that everything went
HC. Its intermediate landmark ("many differentiating HCs at ~24 h") is a free
consistency check.

WHAT THIS RUNS. The drug is applied at the START of the observation, so each
calibration run FORKS the corresponding baseline run at its own t0 - the frame
whose neighbour-pair composition best matches the experimental frame 1 - and
continues from there with notch_inhibition=True. The fork's clock restarts at 0,
so the reported time IS elapsed time since the drug, directly comparable to the
experiment's hours.

  t(all HC)  <->  48 h        =>  1 sim unit = 48 / t_allHC  hours
  experimental window is 48 h =>  scoring window = t_allHC sim units

PSIGMA. --psigma also runs the inhibition under mechanosensitivity. This is an
INDEPENDENT constraint on psigma, needing no score: the experiment says every
cell goes HC under repressor block, so any psigma that prevents the model from
reaching all-HC is ruled out by that observation alone.

CAREFUL: a fork must not re-seed the lateral-inhibition state. run() only clamps
randomize_notch_delta_levels for a RESUME, and run_full_model_arrays passes
`randomize_notch_delta_levels=(not continue_existing_run)` - which is True for a
fork. So this calls run() directly with randomize_notch_delta_levels=False,
leaving the forked snapshot's notch/delta/repressor intact.
"""
import argparse
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, experimental_results_folder as EXP,
                             load_history_file, get_time_points,
                             _exp_neighbor_pair_percentages,
                             _best_matching_frame_by_neighbor_pairs,
                             get_non_boundary_cell_ids_from_type,
                             full_model_run_names)
from run_model import run, _psigma_tag

MECH = {
    "E17.5": (0.2461, 1.00, 4.86, 5.72),
    "P0":    (0.2298, 1.00, 5.1487, 5.6706),
}
PREFERRED_AREA = 0.593 * np.pi / 4
ATOH_SENSITIVITY = 0.355079
TYPE_BY, THRESHOLD = "delta_level", 0.355079

# The observation window is 48 h, and the calibration says 48 h == 2.0 sim units
# (t(all HC) at psigma=0). The DIAGNOSTIC quantity is the SC fraction still left
# at that moment, because the experiment reports it per stage:
#   E17.5 - every cell becomes HC          -> ~0% SC remaining
#   P0    - about 10% of cells stay SC     -> ~10% SC remaining
# With repressor production zeroed, the only thing that can still hold a cell
# back is the stress gate, so this residual is a DIRECT readout of psigma - and
# it needs no scoring and no assumption about the scoring window.
T_48H = 2.0
EXPERIMENTAL_SC_REMAINING = {"E17.5": 0.0, "P0": 10.0}


def experimental_initial_target(stage):
    prefix = "E17" if stage == "E17.5" else "P0"
    hchc, hcsc = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(EXP, stage,
             "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(EXP, stage,
             "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        a, b, _ = _exp_neighbor_pair_percentages(ci, cm)
        hchc.append(a); hcsc.append(b)
    return float(np.nanmean(hchc)), float(np.nanmean(hcsc))


def calib_name(stage, i, psigma=0.0, stress_shift=-0.080):
    """psigma=0 keeps the original (untagged) name so the first runs are reused."""
    suffix = "E17" if stage == "E17.5" else "P0"
    if float(psigma) == 0.0:
        return "notchinhib_random_periodic_array%d_for_%s" % (i, suffix)
    return "notchinhib_ps%s_ks%.3f_random_periodic_array%d_for_%s" % (
        _psigma_tag(float(psigma)), stress_shift, i, suffix)


def hc_fraction_curve(name):
    """(times, HC fraction among non-boundary cells) for a calibration run."""
    history = load_history_file(name)
    ts, fr = [], []
    for t in get_time_points(history):
        sheet = history.retrieve(float(t))
        sheet.arrange_sheet_from_history()
        all_idx, _ = get_non_boundary_cell_ids_from_type(
            sheet, "all", type_by=TYPE_BY, threshold=THRESHOLD)
        if all_idx.size == 0:
            continue
        hc_idx, _ = get_non_boundary_cell_ids_from_type(
            sheet, "HC", type_by=TYPE_BY, threshold=THRESHOLD)
        ts.append(float(t)); fr.append(len(hc_idx) / float(all_idx.size))
    return np.array(ts), np.array(fr)


def first_time_above(ts, fr, level):
    hit = np.flatnonzero(fr >= level)
    return float(ts[hit[0]]) if hit.size else None


def fraction_at(ts, fr, t):
    """HC fraction at the recorded frame nearest ``t`` (None if the run is short)."""
    if ts.size == 0 or ts[-1] < t - 1e-9:
        return None
    return float(fr[int(np.argmin(np.abs(ts - t)))])


def one_run(args):
    stage, i, t_end, save_interval, dry, psigma, stress_shift = args
    gammaSC, alphaHC, hc_p0, sc_p0 = MECH[stage]
    src = full_model_run_names(stage)[i]
    tgt = experimental_initial_target(stage)
    history = load_history_file(src)
    t0, _, _ = _best_matching_frame_by_neighbor_pairs(history, tgt[0], tgt[1],
                                                      TYPE_BY, THRESHOLD)
    name = calib_name(stage, i, psigma, stress_shift)
    if dry:
        return name, t0, None
    try:
        run(gammaSC, 1.0, alphaHC, float(psigma),
            stress_dependent=(float(psigma) != 0.0), stress_shift=stress_shift,
            initial_sheet_name=src, continue_from_time=float(t0),
            continue_existing_run=False,          # FORK, not resume
            randomize_notch_delta_levels=False,   # keep the snapshot's LI state
            notch_inhibition=True,
            name=name, t_end=t_end, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,            # we want the full trajectory
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=hc_p0, sc_shape_index=sc_p0, bending=0.02,
            quasi_static_threshold=0.03, preferred_area_override=PREFERRED_AREA,
            reuse_existing_run=True)
        return name, t0, None
    except Exception as exc:  # noqa: BLE001
        return name, t0, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["E17.5", "P0"])
    ap.add_argument("--arrays", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--t-end", dest="t_end", type=float, default=50)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all-hc-level", dest="level", type=float, default=0.99,
                    help="HC fraction counted as 'all cells are HC' (default 0.99)")
    ap.add_argument("--psigma", type=float, nargs="+", default=[0.0],
                    help="mechanosensitivity values to test under inhibition")
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=-0.080)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [(s, i, a.t_end, a.save_interval, a.dry_run, ps, a.stress_shift)
             for s in a.stages for ps in a.psigma for i in a.arrays]
    print("Notch-inhibition time calibration | %d run(s) | fork at each run's own t0"
          % len(tasks), flush=True)
    if a.dry_run:
        for t in tasks:
            name, t0, _ = one_run(t)
            print("  psigma=%-7.3f %-56s fork from t0=%.2f" % (t[5], name, t0))
        raise SystemExit("\n--dry-run: nothing was run.")

    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one_run, t): t for t in tasks}
            for n, f in enumerate(as_completed(futs), 1):
                name, t0, err = f.result()
                print("  [%2d/%2d] %-46s %s" % (n, len(tasks), name,
                      "OK (t0=%.2f)" % t0 if err is None else "FAILED " + err), flush=True)
    else:
        for n, t in enumerate(tasks, 1):
            name, t0, err = one_run(t)
            print("  [%2d/%2d] %-46s %s" % (n, len(tasks), name,
                  "OK (t0=%.2f)" % t0 if err is None else "FAILED " + err), flush=True)

    print()
    print("=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    for stage in a.stages:
        print()
        print("=== %s ===" % stage)
        target = EXPERIMENTAL_SC_REMAINING[stage]
        print("  experiment: %.0f%% of cells remain SC at 48 h" % target)
        print("  %-8s %-6s %9s %9s %10s %12s"
              % ("psigma", "array", "t(50%HC)", "t(90%HC)", "t(all HC)",
                 "SC%% left@48h"))
        for psigma in a.psigma:
            t_all, t_half, peaks = [], [], []
            for i in a.arrays:
                name = calib_name(stage, i, psigma, a.stress_shift)
                if not os.path.isdir(os.path.join(RESULTS_DIR, name)):
                    continue
                try:
                    ts, fr = hc_fraction_curve(name)
                except Exception as exc:  # noqa: BLE001
                    print("  %-8.3f %-6d  curve failed: %s" % (psigma, i, exc))
                    continue
                t50 = first_time_above(ts, fr, 0.50)
                t90 = first_time_above(ts, fr, 0.90)
                tall = first_time_above(ts, fr, a.level)
                hc48 = fraction_at(ts, fr, T_48H)
                sc48 = None if hc48 is None else 100.0 * (1.0 - hc48)
                if sc48 is not None:
                    peaks.append(sc48)
                print("  %-8.3f %-6d %9s %9s %10s %11s"
                      % (psigma, i, "-" if t50 is None else "%.2f" % t50,
                         "-" if t90 is None else "%.2f" % t90,
                         "-" if tall is None else "%.2f" % tall,
                         "-" if sc48 is None else "%.1f%%" % sc48))
                if tall is not None:
                    t_all.append(tall)
                if t50 is not None:
                    t_half.append(t50)
            if peaks:
                med = float(np.median(peaks))
                print("    -> psigma=%.3f: SC remaining at 48 h = %.1f%%  "
                      "(experiment %.0f%%, gap %+.1f pts)"
                      % (psigma, med, target, med - target))
            if not peaks:
                continue
            if t_all:
                m = float(np.median(t_all))
                print("    -> psigma=%.3f: t(all HC) median=%.2f  <->  48 h  "
                      "=> 1 unit=%.1f h; SCORING WINDOW (48 h) = %.2f units"
                      % (psigma, m, 48.0 / m, m))
                if t_half:
                    h = float(np.median(t_half))
                    print("       cross-check t(50%% HC)=%.2f = %.0f h"
                          % (h, h * 48.0 / m))
            else:
                print("    -> psigma=%.3f: NEVER reached %.0f%% HC "
                      "(min SC remaining %.1f%%) by t_end=%g"
                      % (psigma, 100 * a.level, min(peaks), a.t_end))
                print("       the inhibition experiment says every cell goes HC, so "
                      "this psigma is RULED OUT by that observation.")


if __name__ == "__main__":
    main()
