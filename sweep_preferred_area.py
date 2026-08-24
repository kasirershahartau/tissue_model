"""Preferred-area SWEEP: relax the tissue at each candidate A0, then measure the
cut-circle shrinkage from the RE-EQUILIBRATED geometry (the affine estimate held
geometry fixed, so it under/over-shoots once cells re-arrange).

Also reports roundness at each A0, since lowering A0 lowers the target perimeter
P0 = shape_index*sqrt(A0) and makes cells rounder — which is what would fight the
roundness terms in the fit.

Target: 7.66% linear shrinkage (28 circular ablations, 60 um initial radius).
"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

MODEL = r"C:/Users/Kasirer/Phd/mouse_ear_project/tissue_model"
sys.path.insert(0, MODEL)
sys.path.insert(0, os.path.join(MODEL, "tyssue", "src"))
os.environ.setdefault("TISSUE_RESULTS_DIR", r"D:/Kasirer/results")

# best E17.5 re-fit point so far
GSC, AHC, HC_P0, SC_P0 = 0.2461, 1.0633, 3.9460, 4.6440
FRACS = [0.70, 0.75, 0.79, 0.83, 0.87, 0.90]      # of pi/4
ARRAYS = [0, 1]
TARGET_SHRINK = 7.66
TB, THRESH = "delta_level", 0.355079


def one(args):
    frac, idx = args
    import run_model as rm
    a0 = frac * np.pi / 4
    init = "random_periodic_array%d_for_E17" % idx
    li = rm._li_levels_kwargs_for_initial_sheet(init)
    thr = float(rm._load_saved_threshold(init))
    name = "pasweep_%03d_a%d" % (round(frac * 100), idx)
    try:
        rm.run(GSC, 1.0, AHC, 0, initial_sheet_name=init, name=name,
               no_differentiation=True, end_on_steady_state=True, t_end=25, dt=0.01,
               shape_index=0.0, hc_shape_index=HC_P0, sc_shape_index=SC_P0,
               bending=0.02, line_tension=None, quasi_static_threshold=0.03,
               atoh_sensitivity=thr, preferred_area_override=a0,
               save_interval=10.0, reuse_existing_run=True, **li)
        return frac, idx, name, None
    except Exception as exc:
        return frac, idx, name, "%s: %s" % (type(exc).__name__, exc)


def measure(name):
    from post_processing import (load_history_file, get_time_points,
                                 get_non_boundary_cell_ids_from_type,
                                 calc_roundness_for_type)
    h = load_history_file(name); t = get_time_points(h)
    s = h.retrieve(float(t[-1])); s.arrange_sheet_from_history(); s.geom.update_all(s)
    fd = s.face_df
    A = fd["area"].to_numpy(float); P = fd["perimeter"].to_numpy(float)
    A0 = fd["prefered_area"].to_numpy(float); P0 = fd["prefered_perimeter"].to_numpy(float)
    K = fd["area_elasticity"].to_numpy(float); G = fd["contractility"].to_numpy(float)

    def dE(l):
        return np.sum(K * (l * l * A - A0) * 2 * l * A + G * (l * P - P0) * P)
    lo, hi = 0.3, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dE(lo) * dE(mid) <= 0: hi = mid
        else: lo = mid
    lam = 0.5 * (lo + hi)
    hc = float(np.mean(calc_roundness_for_type(s, "HC", type_by=TB, threshold=THRESH)))
    sc = float(np.mean(calc_roundness_for_type(s, "SC", type_by=TB, threshold=THRESH)))
    return lam, hc, sc, float(np.mean(A))


if __name__ == "__main__":
    tasks = [(f, i) for f in FRACS for i in ARRAYS]
    print("preferred-area sweep: %d runs (%d fractions x %d arrays)"
          % (len(tasks), len(FRACS), len(ARRAYS)), flush=True)
    done = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as ex:
        for r in ex.map(one, tasks):
            done.append(r)
            print("  finished frac=%.2f array%d %s"
                  % (r[0], r[1], "" if r[3] is None else "FAILED " + r[3]), flush=True)

    print("\n" + "=" * 74)
    print("SWEEP RESULT   (experiment: %.2f%% linear shrinkage)" % TARGET_SHRINK)
    print("=" * 74)
    print("  %-7s %-9s %-9s %-9s %-16s %s"
          % ("A0/pi4", "A0", "lambda*", "shrink%", "roundness HC/SC", "mean area"))
    rows = []
    for frac in FRACS:
        vals = [measure(n) for (f, i, n, e) in done if f == frac and e is None]
        if not vals:
            print("  %-7.2f (all runs failed)" % frac); continue
        lam = float(np.mean([v[0] for v in vals]))
        hc = float(np.mean([v[1] for v in vals])); sc = float(np.mean([v[2] for v in vals]))
        ar = float(np.mean([v[3] for v in vals]))
        shrink = 100 * (1 - lam)
        rows.append((frac, shrink))
        print("  %-7.2f %-9.4f %-9.4f %-9.2f %.3f / %.3f      %.4f"
              % (frac, frac * np.pi / 4, lam, shrink, hc, sc, ar))
    print("\n  experimental roundness: HC 0.804  SC 0.649")
    if len(rows) >= 2:
        fr = np.array([r[0] for r in rows]); sh = np.array([r[1] for r in rows])
        o = np.argsort(sh)
        best = float(np.interp(TARGET_SHRINK, sh[o], fr[o]))
        print("\n  -> interpolated A0 for %.2f%% shrinkage: %.3f * pi/4 = %.4f"
              % (TARGET_SHRINK, best, best * np.pi / 4))
