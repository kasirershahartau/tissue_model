"""Single-HC ablation: which cells differentiate afterwards, and how far away?

    python ablate_single_hc.py --dry-run
    python ablate_single_hc.py --workers 4
    python ablate_single_hc.py --analyse-only        # re-score existing runs

Mirrors the experiment: ablate ONE hair cell, watch for differentiation events
over the following window, then trace each differentiating cell back to the
frame JUST BEFORE the ablation and measure the distance from its centroid to the
ablated cell's centroid.

SETUP. Each run forks the psigma=0 full model at its LAST recorded frame - the
steady state - picks one non-boundary HC uniformly at random (seeded per array,
so the choice is reproducible), ablates it, and continues for --t-end simulated
units. One repeat per array: 10 for E17.5, 10 for P0.

The fork's clock restarts at 0, so its t=0 frame IS the pre-ablation state, and
every reported time is elapsed time since ablation.

DISTANCES ARE PERIODIC. The sheet is a periodic box, so a naive centroid
difference can be a whole box-width wrong for a cell near an edge. All distances
use the minimum image convention, matching topological_events._min_image_midpoint:

    d = c_i - c_ablated ;  d -= (Lx, Ly) * round(d / (Lx, Ly))

DIFFERENTIATION is SC -> HC: a cell whose delta_level is below the threshold at
t=0 and above it at some later frame.

CELLS ARE TRACKED BY ``id``, NOT ``unique_id``. When a face is REMOVED - which
is exactly what an ablation does - ``unique_id`` is compacted and renumbered, so
the same ``unique_id`` refers to a different cell before and after. Measured on
one ablation run: tracking by ``unique_id`` mislabels 210 cells (delta differing
by >0.25) against 6 under ``id``. In the no-ablation control, where nothing is
removed, the two agree exactly (5 and 5) - which is why the artefact appeared
only in the ablated runs and produced ~50 phantom "differentiation events" per
ablation, all at the frame where the cell was dropped.

CAREFUL: a fork must not re-seed the lateral-inhibition state - run() only
clamps randomize_notch_delta_levels for a RESUME, so this calls run() directly
with it False, leaving the forked snapshot's notch/delta/repressor intact.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type,
                             full_model_run_names)
from run_model import run, load_sheet_from_file

# v2 MECHANICS, read from the self-consistent fit rather than retyped:
# (gammaSC, R_gamma, R_alpha, A0). Shape index 0 (pure contractility) and
# bending 0 everywhere - see MECHANICAL_FIT_V2.md.
def _v2_mech():
    import json
    from post_processing import RESULTS_DIR as _RD
    out = {}
    for stage, fn in (("E17.5", "e17_selfconsistent_scan.json"),
                      ("P0", "p0_selfconsistent_scan.json")):
        data = json.load(open(os.path.join(_RD, fn)))
        pts = data["points"]
        best = min(pts.values(),
                   key=lambda v: sum((v["z"] or {}).get(t, 1e9) ** 2
                                     for t in ("roundness_ratio", "shrinkage")))
        # per-point "R_alpha" was only added when the R_alpha scan was wired in,
        # so the scans that produced these files carry it at file level only.
        out[stage] = (best["gamma_sc"], best["R_gamma"],
                      best.get("R_alpha", data["R_alpha"]), best["A0"])
    return out


MECH = _v2_mech()
SHAPE_INDEX = 0.0
BENDING = 0.0
RUN_PREFIX = "fullmodel_v2"
ATOH_SENSITIVITY = 0.355079
TYPE_BY, THRESHOLD = "delta_level", 0.355079
BOX = (20, 20)                       # nx*distx, ny*disty
# _v2 in the name: the run FOLDERS were renamed for v2 but these output files
# were not, so the first v2 run silently overwrote the pre-v2 results. The v1
# run folders (ablate1hc_array*) survive, so those numbers can be re-derived.
OUT_JSON = "single_hc_ablation_v2.json"
OUT_JSON_CTRL = "single_hc_ablation_v2_control.json"


def min_image_distance(c, ref, Lx, Ly):
    """Distance from each row of ``c`` to ``ref`` across a periodic box."""
    d = np.asarray(c, float) - np.asarray(ref, float)
    periods = np.array([Lx, Ly], float)
    d -= periods * np.round(d / periods)
    return np.hypot(d[:, 0], d[:, 1])


def pick_hc(sheet, seed):
    """(face LABEL, centroid) of one uniformly-random non-boundary HC.

    Returns the label because the ablation handler indexes with .loc, while
    get_non_boundary_cell_ids_from_type hands back POSITIONAL indices."""
    hc_pos, _ = get_non_boundary_cell_ids_from_type(
        sheet, "HC", type_by=TYPE_BY, threshold=THRESHOLD)
    if hc_pos.size == 0:
        raise RuntimeError("no non-boundary HC to ablate")
    label = int(sheet.face_df.index.values[hc_pos[np.random.RandomState(seed).randint(hc_pos.size)]])
    centroid = sheet.face_df.loc[label, ["x", "y"]].to_numpy(float)
    return label, centroid


def source_state(stage, i):
    """Pre-ablation sheet: the source run's last frame, plus its time."""
    src = full_model_run_names(stage, run_prefix=RUN_PREFIX)[i]
    history = load_history_file(src)
    t_last = float(np.max(get_time_points(history)))
    sheet = load_sheet_from_file(os.path.join(RESULTS_DIR, src),
                                 time_point=t_last, force_periodic_box=BOX)
    sheet.geom.update_all(sheet)
    return src, t_last, sheet


def ablation_run_name(stage, i, label, control=False):
    """(name passed to run(), resulting folder).

    CONTROL runs fork the identical steady state and cover the identical span
    with NOTHING ablated, so run() does not append its "ablated_<cell>" suffix.
    They measure the BACKGROUND rate, which is not assumed to be zero: forking
    recomputes the lateral-inhibition length_normalization_factor from the
    loaded frame (settled mean perimeter ~3.77) instead of the original run's
    first frame (~3.46), shifting the LI coupling by ~9% and potentially moving
    every cell at once. Whatever that produces shows up here and must be
    subtracted from the ablation counts."""
    suffix = "E17" if stage == "E17.5" else "P0"
    if control:
        base = "ctrl1hc_v2_array%d_for_%s" % (i, suffix)
        return base, base
    # run() appends "ablated_<cell>" itself when name AND ablated_cells are given.
    return ("ablate1hc_v2_array%d_for_%s" % (i, suffix),
            "ablate1hc_v2_array%d_for_%sablated_%d" % (i, suffix, label))


def one_run(args):
    stage, i, t_end, save_interval, seed, dry, control = args
    try:
        src, t_last, sheet = source_state(stage, i)
        label, centroid = pick_hc(sheet, seed + i)
        base, folder = ablation_run_name(stage, i, label, control)
        if dry:
            return dict(stage=stage, array=i, source=src, t_last=t_last,
                        ablated_label=label, folder=folder, control=control,
                        error=None)
        gammaSC, R_gamma, R_alpha, A0 = MECH[stage]
        # ORDER: run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, ...)
        # gamma ratio SECOND, alpha ratio THIRD - they differ in v2.
        run(gammaSC, R_gamma, R_alpha, 0.0,
            initial_sheet_name=src, continue_from_time=t_last,
            continue_existing_run=False,           # FORK from the steady state
            randomize_notch_delta_levels=False,    # keep its LI state
            ablated_cells=([] if control else [label]), name=base,
            t_end=t_end, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,
            # Without these a pathological step hangs forever instead of failing:
            # two runs sat 16 h inside solve_ivp before this was added.
            max_wall_seconds=3600, min_progress_rate=1e-4,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX, bending=BENDING,
            quasi_static_threshold=0.03, preferred_area_override=A0,
            reuse_existing_run=True)
        return dict(stage=stage, array=i, source=src, t_last=t_last,
                    ablated_label=label, folder=folder, control=control, error=None)
    except Exception as exc:  # noqa: BLE001
        return dict(stage=stage, array=i, folder=None, control=control, error="%s: %s"
                    % (type(exc).__name__, exc))


def analyse(rec, pre_state=None):
    """Differentiation events after the ablation, with pre-ablation distances.

    ``pre_state`` is the (name, time, sheet) triple of the run this fork came
    from. It is a parameter so callers that fork something other than the
    psigma=0 baseline — e.g. the psigma/repeat sweep — can supply their own
    source instead of having it looked up from the default prefix."""
    stage, i, folder, label = rec["stage"], rec["array"], rec["folder"], rec["ablated_label"]
    _src, _t, pre = pre_state if pre_state is not None else source_state(stage, i)
    ref = pre.face_df.loc[label, ["x", "y"]].to_numpy(float)
    Lx, Ly = float(getattr(pre, "Lx", BOX[0])), float(getattr(pre, "Ly", BOX[1]))
    # pre-ablation centroid + delta of every cell, keyed by unique_id
    pre_xy, pre_delta = {}, {}
    for uid, x, y, d in zip(pre.face_df["id"].to_numpy(),
                            pre.face_df["x"].to_numpy(float),
                            pre.face_df["y"].to_numpy(float),
                            pre.face_df[TYPE_BY].to_numpy(float)):
        pre_xy[int(uid)] = (float(x), float(y)); pre_delta[int(uid)] = float(d)
    ablated_uid = int(pre.face_df.loc[label, "id"])

    history = load_history_file(folder)
    stamps = get_time_points(history)
    became_hc = {}                      # uid -> time of first crossing
    for t in stamps:
        s = history.retrieve(float(t)); s.arrange_sheet_from_history()
        uids = s.face_df["id"].to_numpy()
        delta = s.face_df[TYPE_BY].to_numpy(float)
        for uid, d in zip(uids, delta):
            uid = int(uid)
            if uid == ablated_uid or uid in became_hc:
                continue
            if d > THRESHOLD and pre_delta.get(uid, 1.0) <= THRESHOLD:
                became_hc[uid] = float(t)

    events = []
    for uid, t in sorted(became_hc.items(), key=lambda kv: kv[1]):
        if uid not in pre_xy:
            continue
        dist = float(min_image_distance(np.array([pre_xy[uid]]), ref, Lx, Ly)[0])
        events.append({"cell_id": uid, "t_differentiated": t, "distance": dist})
    rec = dict(rec)
    rec.update(n_events=len(events), events=events,
               n_pre_sc=int(sum(1 for v in pre_delta.values() if v <= THRESHOLD)))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(MECH))
    ap.add_argument("--arrays", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--t-end", dest="t_end", type=float, default=5.0)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--control", action="store_true",
                    help="fork the SAME steady state and run the SAME span with "
                         "NOTHING ablated - the background differentiation rate")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [(s, i, a.t_end, a.save_interval, a.seed,
              a.dry_run or a.analyse_only, a.control)
             for s in a.stages for i in a.arrays]
    print("single-HC %s | %d run(s) | t_end=%g | seed=%d"
          % ("CONTROL (nothing ablated)" if a.control else "ablation",
             len(tasks), a.t_end, a.seed), flush=True)

    if a.dry_run:
        for t in tasks:
            r = one_run(t)
            print("  %-6s array%-2d  ablate face %-5s at t=%.2f -> %s"
                  % (r["stage"], r["array"], r.get("ablated_label"),
                     r.get("t_last", float("nan")), r.get("folder")))
        raise SystemExit("\n--dry-run: nothing was run.")

    if a.workers > 1 and len(tasks) > 1 and not a.analyse_only:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(one_run, t) for t in tasks]
            recs = []
            for n, f in enumerate(as_completed(futs), 1):
                r = f.result(); recs.append(r)
                print("  [%2d/%2d] %-6s array%-2d %s" % (n, len(tasks), r["stage"],
                      r["array"], "OK" if not r["error"] else "FAILED " + r["error"]),
                      flush=True)
    else:
        recs = [one_run(t) for t in tasks]

    print("\nanalysing...", flush=True)
    out = []
    for r in sorted(recs, key=lambda z: (z["stage"], z["array"])):
        if r["error"] or not r.get("folder"):
            print("  %-6s array%-2d SKIP (%s)" % (r["stage"], r["array"], r["error"]))
            continue
        try:
            rr = analyse(r)
        except Exception as exc:  # noqa: BLE001
            print("  %-6s array%-2d analysis failed: %s" % (r["stage"], r["array"], exc))
            continue
        out.append(rr)
        ds = [e["distance"] for e in rr["events"]]
        print("  %-6s array%-2d ablated uid-face %-5d : %d event(s)%s"
              % (rr["stage"], rr["array"], rr["ablated_label"], rr["n_events"],
                 "  distances " + ", ".join("%.2f" % d for d in sorted(ds)) if ds else ""))

    with open(os.path.join(RESULTS_DIR,
              OUT_JSON_CTRL if a.control else OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n" + "=" * 70)
    for stage in a.stages:
        rows = [r for r in out if r["stage"] == stage]
        if not rows:
            continue
        n_ev = [r["n_events"] for r in rows]
        allel = [e["distance"] for r in rows for e in r["events"]]
        print("%s : %d ablation(s), %d differentiation event(s) total, "
              "%.2f per ablation" % (stage, len(rows), sum(n_ev), np.mean(n_ev)))
        if allel:
            q = np.percentile(allel, [25, 50, 75])
            print("   distance from ablated HC (pre-ablation frame): "
                  "min=%.2f  Q1=%.2f  median=%.2f  Q3=%.2f  max=%.2f"
                  % (min(allel), q[0], q[1], q[2], max(allel)))
    print("\nwrote %s" % os.path.join(RESULTS_DIR,
          OUT_JSON_CTRL if a.control else OUT_JSON))


if __name__ == "__main__":
    main()
