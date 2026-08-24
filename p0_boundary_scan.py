"""Step 5d — P0 along the A0 = pi/4 BOUNDARY: gammaSC swept, R_gamma maximal.

    python p0_boundary_scan.py --dry-run
    python p0_boundary_scan.py --workers 6         # locally
    python p0_boundary_scan.py --workers 30        # 32-core Azure VM

Step 5c showed the objective falling monotonically towards the A0 < pi/4 ceiling
at every gammaSC, so the optimum always sits ON that ceiling. Rather than pay for
a full 2-D (gammaSC, R_gamma) grid whose interior is known to be worse, this
walks the ceiling itself: for each gammaSC, R_gamma is set to the largest value
still satisfying A0 < pi/4.

WHAT THE CURVE ACTUALLY IS. On the ceiling avg_gamma/avg_alpha = (1-lambda^2)/8,
so the step-1 preferred area collapses to

    A0 = (pi/4)(lambda^2 + 8*(1-lambda^2)/8) = pi/4      exactly, for EVERY point

and, because avg_alpha is pinned by R_alpha, avg_gamma = (1-lambda^2)/8 *
avg_alpha is constant too. So this is not a ragged boundary: it is the 1-D family
at FIXED preferred area and FIXED mean contractility, along which only the HC/SC
contractility SPLIT changes. Every point is equally consistent with the measured
shrinkage (step 1) and with the measured stress ratio (R_alpha untouched).

A0 = pi/4 means the preferred area equals the packed cell area to 0.3%
(400/508 = 0.7874 vs 0.7854), i.e. the area term is nearly relaxed and
essentially ALL the tissue tension comes from contractility.

THE RANGE. R_gamma(gammaSC) rises as gammaSC falls, and reaches 1 exactly at

    gammaSC_max = (1-lambda^2)/8 * avg_alpha

which is the "even at R_gamma = 1 the ceiling is reached" end of the family.
Above it the ceiling would demand R_gamma < 1, i.e. HC LESS round than SC, which
step 5c ruled out. Points are placed geometrically in gammaSC, because
R_gamma ~ 1/gammaSC and roundness ~ ln(R_gamma), so log spacing gives roughly
even coverage in the quantity being fitted.

PREDICTION TO TEST. Step 5c fitted roundness = a + b*ln(R_gamma) with b = 0.1272
(and the E17.5 grid gives 0.1275 out to R = 5), which puts the target 1.1955
near gammaSC ~ 0.0105, R_gamma ~ 4. A monotone crossing near there confirms the
law; a crossing somewhere else, or none, means the log extrapolation broke.

A tiny margin keeps A0 strictly BELOW pi/4 rather than exactly on it, and
R_gamma is snapped DOWN onto an already-simulated value when one is within 1%
(snapping down can only lower A0, so it cannot violate the bound) — that reuses
the step-5c run at gammaSC = 0.0175 instead of repeating it.
"""
import argparse
import json
import os

import numpy as np

from post_processing import RESULTS_DIR, initial_morphology_name
import grid_fit_mechanics_v2 as g2
from grid_fit_mechanics_v2 import measured_lambda, score_point
from p0_from_e17_stiffness import hc_fraction, e17_candidates, OUT as STEP5_OUT
from p0_gamma_scan import load_step5, derived_r, snap_to_step5
from p0_rgamma_scan import (build_task, gamma_over_alpha, experimental_stats,
                            OUT as RGAMMA_OUT)

STAGE = "P0"
OUT = "p0_boundary_scan.json"

# Geometric in gammaSC (see the docstring), denser through the predicted
# crossing at ~0.0105, and 0.0175 included so step 5c's point is reused.
GRID_GAMMA_SC = [0.005, 0.006, 0.0075, 0.009, 0.0105,
                 0.0125, 0.0145, 0.0175, 0.021, 0.026]


def rgamma_on_boundary(lam, gamma_sc, R_alpha, f_hc, margin):
    """R_gamma putting avg_gamma/avg_alpha at (1-margin)*(1-lam^2)/8."""
    target = (1.0 - lam ** 2) / 8.0 * (1.0 - margin)
    return ((target / gamma_sc) * (R_alpha * f_hc + 1 - f_hc) - (1 - f_hc)) / f_hc


def gamma_sc_max(lam, R_alpha, f_hc, margin):
    """gammaSC at which the boundary demands R_gamma = 1 (the family's end)."""
    return (1.0 - lam ** 2) / 8.0 * (1.0 - margin) * (R_alpha * f_hc + 1 - f_hc)


def point_key(R_alpha, R_gamma, gamma_sc):
    """Identical to the step-5c key, so its finished points can be read back."""
    return "Ra=%.4f|Rg=%.4f|gSC=%.5f" % (R_alpha, R_gamma, gamma_sc)


def load_points(path):
    if not os.path.isfile(path):
        return {}
    try:
        return json.load(open(path)).get("points", {})
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--tasks-per-pool", dest="tasks_per_pool", type=int, default=None,
                    help="rebuild the worker pool every N tasks so memory is released "
                         "(default 3x--workers); see run_task_pool")
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, nargs="+",
                    default=GRID_GAMMA_SC)
    ap.add_argument("--r-alpha", dest="r_alpha", type=float, default=None,
                    help="default: derived from the best-fitting E17.5 grid point")
    ap.add_argument("--margin", type=float, default=2e-3,
                    help="keep A0 this fraction below pi/4 (default 0.002)")
    ap.add_argument("--step5-json", dest="step5_json",
                    default=os.path.join(RESULTS_DIR, STEP5_OUT))
    ap.add_argument("--no-reuse", action="store_true",
                    help="do not snap onto / reuse step-5c's simulated points")
    ap.add_argument("--min-sheets", dest="min_sheets", type=int, default=None)
    ap.add_argument("--keep-partial", dest="keep_partial", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if any(g <= 0 for g in a.gamma_sc):
        raise SystemExit("gammaSC must be > 0")
    lam, pct = measured_lambda(STAGE)
    fP, nP = hc_fraction(STAGE, a.n_sheets)
    step5 = load_step5(a.step5_json)

    print("=" * 78)
    print("STEP 5d | %s along A0 = pi/4 | %d gammaSC values | %d sheets | %d workers"
          % (STAGE, len(a.gamma_sc), a.n_sheets, a.workers))
    print("=" * 78)
    print("  model: FaceContractility (p0=0), NO bending, alphaSC=1")
    print("  measured shrinkage %.4f%% -> lambda=%.6f ; f_HC=%.4f (%d arrays)"
          % (pct, lam, fP, nP))

    best_RE, best_g, best_obj = e17_candidates(1)[0]
    if a.r_alpha is not None:
        R_alpha, src = a.r_alpha, "supplied on the command line"
    else:
        R_alpha = derived_r(1)[0][0]
        src = "derived from the best E17.5 point via the stress ratio"
    R_alpha = snap_to_step5([R_alpha], step5)[0]
    avg_alpha = R_alpha * fP + 1 - fP
    print("\n  R_alpha PINNED at %.4f  (%s)" % (R_alpha, src))

    # Everything below is a consequence of sitting on the ceiling.
    A0 = (np.pi / 4.0) * (lam ** 2 + 8 * (1 - lam ** 2) / 8 * (1 - a.margin))
    avg_gamma = (1 - lam ** 2) / 8.0 * (1 - a.margin) * avg_alpha
    g_max = gamma_sc_max(lam, R_alpha, fP, a.margin)
    print("  on this curve:  A0 = %.5f (%.4f x pi/4, constant)   avg_alpha = %.4f"
          "   avg_gamma = %.5f (constant)" % (A0, A0 / (np.pi / 4), avg_alpha, avg_gamma))
    print("  R_gamma reaches 1 at gammaSC = %.5f — the end of the family" % g_max)

    too_big = [g for g in a.gamma_sc if g > g_max]
    if too_big:
        raise SystemExit("these gammaSC exceed %.5f, where the ceiling would need "
                         "R_gamma < 1 (HC less round than SC): %s" % (g_max, too_big))

    prev = {} if a.no_reuse else load_points(os.path.join(RESULTS_DIR, RGAMMA_OUT))
    prev_rg = {}
    for k, v in prev.items():
        try:
            g = float(k.split("gSC=")[1]); rg = float(k.split("Rg=")[1].split("|")[0])
        except (IndexError, ValueError):
            continue
        prev_rg.setdefault(round(g, 8), []).append(rg)

    out_path = os.path.join(RESULTS_DIR, OUT)
    done = {}
    if os.path.isfile(out_path) and not a.no_resume:
        done = dict(load_points(out_path))
        min_sheets = (a.min_sheets if a.min_sheets is not None
                      else max(1, int(np.ceil(0.6 * a.n_sheets))))
        if not a.keep_partial:
            bad = [k for k, v in done.items()
                   if not v.get("reused") and (v.get("n_sheets_ok") or 0) < min_sheets]
            for k in bad:
                del done[k]
            if bad:
                print("  discarding %d incomplete point(s) so they re-run" % len(bad))

    initials = [initial_morphology_name(i, STAGE) for i in range(a.n_sheets)]
    print("\n  %-10s %10s %13s %11s %s"
          % ("gammaSC", "R_gamma", "gammaHC", "A0/(pi/4)", "state"))
    tasks, plan = [], []
    for g in sorted(a.gamma_sc):
        Rg = rgamma_on_boundary(lam, g, R_alpha, fP, a.margin)
        # Snap DOWN onto an already-simulated R_gamma when one is within 1%.
        # Downwards only: a smaller R_gamma lowers avg_gamma, hence lowers A0, so
        # the bound cannot be violated by the reuse.
        note = ""
        for cand in sorted(prev_rg.get(round(g, 8), []), reverse=True):
            if 0 < Rg - cand <= 0.01 * Rg:
                note = " (snapped from %.4f to reuse step 5c)" % Rg
                Rg = cand
                break
        key = point_key(R_alpha, Rg, g)
        rec = done.get(key)
        if rec is None and key in prev and not a.no_reuse:
            rec = dict(prev[key]); rec["reused"] = True
            done[key] = rec
            note = note or " (reused from step 5c)"
        a0 = (np.pi / 4.0) * (lam ** 2 + 8 * gamma_over_alpha(g, Rg, R_alpha, fP))
        state = ("step 5c" if rec and rec.get("reused")
                 else "cached" if rec else "TO RUN")
        print("  %-10.4g %10.4f %13.5f %11.5f %s%s"
              % (g, Rg, g * Rg, a0 / (np.pi / 4), state, note))
        plan.append((g, Rg, key))
        if rec is None:
            for initial in initials:
                tasks.append((key, build_task(initial, g, Rg, R_alpha, a0)))
    print("\n  %d point(s) to run x %d sheet(s) = %d task(s); each = 1 base + 1"
          " ablation run" % (len(tasks) // max(len(initials), 1), len(initials),
                             len(tasks)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")
    if not tasks:
        print("  every point already cached.")

    results = {}
    if tasks:
        def handle(key, details, n, total):
            results.setdefault(key, []).append(details)
            if len(results[key]) != len(initials):
                return
            z, obj, n_ok, n_cells = score_point(STAGE, results[key])
            done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok,
                         "n_cells": n_cells}
            with open(out_path, "w") as fh:
                json.dump({"stage": STAGE, "lambda": lam, "shrinkage_pct": pct,
                           "f_HC": fP, "R_alpha": R_alpha, "A0": A0,
                           "margin": a.margin, "points": done}, fh, indent=1)
            print("  [%4d/%4d] %-36s DONE  objective=%.4g  (%d/%d sheets)"
                  % (n, total, key, obj, n_ok, len(initials)), flush=True)

        # Pools are RECYCLED (see run_task_pool): a single long-lived pool ran
        # this machine out of memory and killed every point after the first ~34
        # tasks, including the most benign parameter set in the scan.
        g2.run_task_pool(tasks, a.workers, handle, tasks_per_pool=a.tasks_per_pool)

    # ----- report ---------------------------------------------------------
    mu_r, sem_r = experimental_stats("roundness_ratio", STAGE)
    print("\n" + "=" * 78)
    print("A0 = pi/4 boundary, R_alpha = %.4f  (avg_gamma fixed at %.5f)"
          % (R_alpha, avg_gamma))
    print("=" * 78)
    print("  %-10s %9s %10s %10s %9s %9s %9s %7s"
          % ("gammaSC", "R_gamma", "objective", "roundness", "round_z", "abl_z",
             "shrink_z", "sheets"))
    curve = []
    for g, Rg, key in plan:
        rec = done.get(key)
        if rec is None:
            print("  %-10.4g %9.4f %10s" % (g, Rg, "-"))
            continue
        z = rec.get("z") or {}
        zr = z.get("roundness_ratio", float("nan"))
        rnd = mu_r + zr * sem_r if np.isfinite(zr) else float("nan")
        if np.isfinite(rnd):
            curve.append((g, Rg, rnd, rec["objective"]))
        print("  %-10.4g %9.4f %10.4g %10.4f %9.2f %9.2f %9.2f %7s"
              % (g, Rg, rec["objective"], rnd, zr,
                 z.get("ablation_ratio", float("nan")),
                 z.get("shrinkage", float("nan")), rec.get("n_sheets_ok")))
    print("\n  experimental target: roundness %.4f +/- %.4f" % (mu_r, sem_r))
    if len(curve) >= 2:
        best = min(curve, key=lambda c: abs(c[2] - mu_r))
        print("  closest point: gammaSC=%.4g R_gamma=%.3f -> roundness %.4f (%+.1f sigma)"
              % (best[0], best[1], best[2], (best[2] - mu_r) / sem_r))
        below = [c for c in curve if c[2] < mu_r]
        above = [c for c in curve if c[2] >= mu_r]
        if below and above:
            lo, hi = max(below, key=lambda c: c[2]), min(above, key=lambda c: c[2])
            t = (mu_r - lo[2]) / (hi[2] - lo[2])
            print("  target is BRACKETED: crossing at gammaSC ~ %.5f, R_gamma ~ %.2f"
                  % (lo[0] + t * (hi[0] - lo[0]), lo[1] + t * (hi[1] - lo[1])))
        else:
            print("  target NOT bracketed — the curve stays %s it across the whole family"
                  % ("below" if not above else "above"))
        best_obj = min(curve, key=lambda c: c[3])
        print("  best objective %.4g at gammaSC=%.4g, R_gamma=%.3f"
              % (best_obj[3], best_obj[0], best_obj[1]))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
