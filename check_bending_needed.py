"""Is BoundaryBending still needed once the model is pure contractility?

    python check_bending_needed.py --dry-run
    python check_bending_needed.py --array 0

Bending was added because, with LineTension removed, the virtual vertices along
each bond were free to buckle and cell outlines came out visibly wavy;
BoundaryBending penalises CURVATURE at those vertices and made outlines 10-100x
smoother at unchanged roundness. The v2 model changes the perimeter term from
elasticity (Gamma/2 (P-P0)^2) back to plain contractility (Gamma/2 P^2), which
penalises PERIMETER LENGTH directly — and a length penalty already opposes
buckling, since a wavy bond is longer than a straight one. So bending may now be
redundant. This measures that instead of assuming it.

METRIC. Boundary waviness at the virtual vertices: for each interior virtual
vertex on a bond, the SAGITTA (its perpendicular offset from the straight line
joining its two neighbours) divided by that line's CHORD length. 0 = perfectly
straight bond; the value quoted when bending was introduced was 0.05 wavy vs
0.005 straight.

Runs the SAME array at the SAME v2 parameters twice, differing only in bending,
and reports waviness plus HC/SC roundness so a smoothness gain can be weighed
against any shape change.
"""
import argparse
import os

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             load_history_file, get_time_points,
                             calc_roundness_for_type, load_experimental_results)
from run_model import run, _load_saved_threshold

STAGE = "E17.5"
TYPE_BY = "delta_level"
# v2 defaults; only `bending` differs between the two runs.
GAMMA_SC = 0.006
R = 2.5
SHAPE_INDEX = 0.0
BASE_QST = 0.03


def measured_lambda(stage):
    arrays = [np.asarray(a, float) for a in load_experimental_results(stage, "cut shrinkage")]
    per_repeat = [a[np.isfinite(a)].mean() for a in arrays if np.isfinite(a).any()]
    return 1.0 - float(np.mean(per_repeat)) / 100.0


def waviness(sheet):
    """Sagitta/chord at every interior virtual vertex, as an array.

    A bond runs from one real (3-way) vertex to the next through a chain of
    virtual vertices. For each virtual vertex with a predecessor and successor
    along its own bond, the sagitta is its perpendicular distance from the
    straight line joining those two, normalised by that line's length.
    """
    ev = sheet.edge_df[["srce", "trgt", "face"]].to_numpy()
    xy = sheet.vert_df[["x", "y"]].to_numpy(float)
    is_virtual = (sheet.vert_df.get("is_virtual",
                  np.zeros(len(sheet.vert_df), bool)).to_numpy().astype(bool)
                  if "is_virtual" in sheet.vert_df.columns else None)
    # successor map along each face's edge chain
    nxt = {}
    for s, t, _f in ev:
        nxt.setdefault(int(s), int(t))
    prv = {t: s for s, t in nxt.items()}
    out = []
    Lx = float(getattr(sheet, "Lx", 20.0)); Ly = float(getattr(sheet, "Ly", 20.0))
    per = np.array([Lx, Ly])
    for v in range(len(xy)):
        if is_virtual is not None and not is_virtual[v]:
            continue
        a, c = prv.get(v), nxt.get(v)
        if a is None or c is None:
            continue
        p0, p1, p2 = xy[a], xy[v], xy[c]
        d1 = p1 - p0; d1 -= per * np.round(d1 / per)      # periodic
        d2 = p2 - p0; d2 -= per * np.round(d2 / per)
        chord = np.hypot(*d2)
        if chord <= 1e-12:
            continue
        sag = abs(d1[0] * d2[1] - d1[1] * d2[0]) / chord   # |cross| / |chord|
        out.append(sag / chord)
    return np.array(out)


def one(array_idx, bending, t_end, save_interval, dry):
    initial = initial_morphology_name(array_idx, STAGE)
    lam = measured_lambda(STAGE)
    A0 = (np.pi / 4.0) * (lam ** 2 + 8.0 * GAMMA_SC)
    name = "bendcheck_b%.3f_%s" % (bending, initial)
    if dry:
        return name, A0, None
    run(GAMMA_SC, R, R, 0.0, initial, name=name,
        no_differentiation=True, end_on_steady_state=True,
        t_end=t_end, dt=0.01, divisions=False,
        shape_index=SHAPE_INDEX, hc_shape_index=SHAPE_INDEX,
        sc_shape_index=SHAPE_INDEX, bending=bending, line_tension=None,
        quasi_static_threshold=BASE_QST, preferred_area_override=A0,
        atoh_sensitivity=_load_saved_threshold(initial),
        save_interval=save_interval, reuse_existing_run=True,
        max_wall_seconds=7200, min_progress_rate=1e-4)
    return name, A0, None


def report(name):
    history = load_history_file(name)
    t = float(np.max(get_time_points(history)))
    sheet = history.retrieve(t)
    sheet.arrange_sheet_from_history()
    sheet.geom.update_all(sheet)
    w = waviness(sheet)
    hc = calc_roundness_for_type(sheet, cell_type="HC", type_by=TYPE_BY,
                                 threshold=_LOADED_THRESHOLD)
    sc = calc_roundness_for_type(sheet, cell_type="SC", type_by=TYPE_BY,
                                 threshold=_LOADED_THRESHOLD)
    return dict(t=t, n_wav=w.size,
                wav_mean=float(np.mean(w)) if w.size else float("nan"),
                wav_p95=float(np.percentile(w, 95)) if w.size else float("nan"),
                hc=float(np.mean(hc)) if len(hc) else float("nan"),
                sc=float(np.mean(sc)) if len(sc) else float("nan"),
                ratio=(float(np.mean(hc)) / float(np.mean(sc))
                       if len(hc) and len(sc) and np.mean(sc) > 0 else float("nan")))


_LOADED_THRESHOLD = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", type=int, default=0)
    ap.add_argument("--bending", type=float, nargs="+", default=[0.0, 0.02])
    ap.add_argument("--t-end", dest="t_end", type=float, default=60)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    _LOADED_THRESHOLD = _load_saved_threshold(initial_morphology_name(a.array, STAGE))
    print("bending check | %s array%d | gammaSC=%.4g R=%.3g p0=0 | bending %s"
          % (STAGE, a.array, GAMMA_SC, R, a.bending))
    names = []
    for b in a.bending:
        nm, A0, _ = one(a.array, b, a.t_end, a.save_interval, a.dry_run)
        print("  bending=%-6.3g -> %-52s (A0=%.5f)" % (b, nm, A0), flush=True)
        names.append((b, nm))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    print("\n  %-8s %8s %8s %12s %12s %9s %9s %8s"
          % ("bending", "t_end", "n_vv", "waviness", "waviness p95",
             "HC round", "SC round", "HC/SC"))
    for b, nm in names:
        try:
            r = report(nm)
        except Exception as exc:                      # noqa: BLE001
            print("  %-8.3g  report failed: %s" % (b, exc)); continue
        print("  %-8.3g %8.2f %8d %12.5f %12.5f %9.4f %9.4f %8.4f"
              % (b, r["t"], r["n_wav"], r["wav_mean"], r["wav_p95"],
                 r["hc"], r["sc"], r["ratio"]))
    print("\n  reference: when bending was introduced, wavy ~0.05 vs straight ~0.005")
    print("  if bending=0 already sits near the straight value, it is redundant here.")
