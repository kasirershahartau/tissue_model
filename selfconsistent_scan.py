"""Step 5e/6 — EITHER STAGE with A0 solved self-consistently. (Was p0_selfconsistent_scan.py.)

    python selfconsistent_scan.py --stage P0     --workers 6
    python selfconsistent_scan.py --stage E17.5  --workers 6
    python selfconsistent_scan.py --stage E17.5  --r-alpha 2.0   # sensitivity check

WHY. Step 1 removes A0 analytically by idealising EVERY cell as a circle of
diameter 1 (A = pi/4, P = pi). That holds while alpha and gamma are coupled, but
decoupling makes HC and SC geometries diverge and the idealisation fails: at
gammaSC = 0.005, R_gamma = 9.1 the measured geometry was A_HC 0.672 / A_SC 0.805
and P_HC 2.970 / P_SC 3.687, and the shrinkage term — which step 1 is supposed to
satisfy by construction — collapsed to z = -3.3.

THE EXACT CONDITION. Minimising E = sum_i [a_i/2 (l^2 A_i - A0)^2 + g_i/2 (l P_i)^2]
over the affine factor l WITHOUT assuming identical cells gives

    A0 = lam^2 * sum(a_i A_i^2)/sum(a_i A_i)  +  sum(g_i P_i^2)/(2 sum(a_i A_i))

which reduces to (pi/4)(lam^2 + 8 avg_gamma/avg_alpha) when every A_i, P_i is
equal, so it GENERALISES step 1 rather than replacing it. A_i and P_i come from
the run, so it is solved by iteration: A0 -> run -> measure -> A0. On P0 it
converged in TWO passes at every point and pulled shrinkage z from -3.3..-1.9
back to -0.23..-0.03, while moving roundness by at most 0.0034. A0 sets
shrinkage; R_gamma sets roundness; the two are effectively orthogonal.

WHERE THE STAGES DIFFER — R_alpha.
  P0    : DERIVED. The measured stress ratio fixes avg_alpha_P0/avg_alpha_E17.5,
          and with E17.5's value assumed gives R_alpha = 1.757.
  E17.5 : NOT DETERMINED BY THESE OBSERVABLES. Roundness is gamma-driven (alpha
          contributes ~4%), shrinkage is absorbed into A0, and the only
          alpha-sensitive term is the ablation ratio, which no parameter moves.
          The default 3.5 is carried over from the COUPLED grid, where R was
          fitting roundness and was therefore really an R_gamma. Treat it as an
          assumption, not a result, and use --r-alpha to test what it changes.

That asymmetry matters because R_alpha(E17.5) feeds R_alpha(P0) through the
stress ratio: if this scan is re-run at a different R_alpha, P0's should follow.

EXPECT E17.5 TO BE MUCH LESS DISCRIMINATING. Its roundness target is
1.2453 +/- 0.0541 (4.3%) against P0's 1.1955 +/- 0.0059 (0.5%). Since roundness
moves ~0.105 per ln(R_gamma) along this family, one SEM is a factor 1.67 in
R_gamma at E17.5 versus 1.06 at P0 — so E17.5 pins R_gamma roughly 10x more
loosely, and the two stages' R_gamma may well not separate.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name, load_history_file,
                             get_time_points, get_non_boundary_cell_ids_from_type)
from run_model import _load_saved_threshold, _short_run_folder_name
import grid_fit_mechanics_v2 as g2
from grid_fit_mechanics_v2 import measured_lambda, score_point
from p0_from_e17_stiffness import hc_fraction, OUT as STEP5_OUT
from p0_gamma_scan import load_step5, derived_r, snap_to_step5
from p0_rgamma_scan import experimental_stats
from p0_boundary_scan import rgamma_on_boundary

STAGES = ("E17.5", "P0")
# P0's name is kept as-is so its finished run still resumes.
OUT = {"P0": "p0_selfconsistent_scan.json", "E17.5": "e17_selfconsistent_scan.json"}

# Each grid brackets that stage's predicted crossing. E17.5 needs a HIGHER
# R_gamma (target 1.2453 vs 1.1955) hence a LOWER gammaSC at the same R_alpha,
# but its larger avg_alpha pushes back, so the two windows end up similar.
GRID = {"P0":    [0.0075, 0.009, 0.0105, 0.0125, 0.0145],
        "E17.5": [0.009, 0.0105, 0.0125, 0.0145, 0.0175]}

# Carried over from the coupled E17.5 grid. NOT a fitted value — see the docstring.
E17_R_ALPHA = 3.5


def task(initial, stage, gamma_sc, R_gamma, R_alpha, A0, ablated):
    """Task tuple. NOTE the order: (gammaSC, gammaHC_ratio, alphaHC_ratio, ...)
    — gamma second, alpha third; see p0_rgamma_scan.build_task."""
    return (float(gamma_sc), float(R_gamma), float(R_alpha), initial, stage,
            list(ablated), g2.POST_ABLATION_FRAME,
            _load_saved_threshold(initial), g2.TYPE_BY,
            g2.MAX_WALL_SECONDS, g2.MIN_PROGRESS_RATE, g2.PROGRESS_WINDOW_SECONDS,
            g2.RERUN_STALLED, g2.SHAPE_INDEX,
            g2.BASE_QST, g2.ABLATION_QST, g2.LINE_TENSION,
            g2.SHAPE_INDEX, g2.SHAPE_INDEX, g2.BENDING, float(A0))


def base_folder(initial, gamma_sc, R_gamma, R_alpha, A0):
    """Deterministic folder of the un-ablated run, so it can be re-opened."""
    return _short_run_folder_name(
        initial, gamma_sc, R_gamma, R_alpha, 0,
        atoh_sensitivity=_load_saved_threshold(initial), shape_index=g2.SHAPE_INDEX,
        quasi_static_threshold=g2.BASE_QST, line_tension=g2.LINE_TENSION,
        bending=g2.BENDING, hc_shape_index=g2.SHAPE_INDEX,
        sc_shape_index=g2.SHAPE_INDEX, preferred_area=A0)


def sheet_sums(folder, gamma_sc, R_gamma, R_alpha, threshold):
    """(sum a_i A_i^2, sum a_i A_i, sum g_i P_i^2) at the run's steady state."""
    history = load_history_file(folder)
    sheet = history.retrieve(float(np.max(get_time_points(history))))
    sheet.arrange_sheet_from_history()
    sheet.geom.update_all(sheet)
    area = sheet.get_face_area().to_numpy(float)
    per = sheet.face_df["perimeter"].to_numpy(float)
    hc, _ = get_non_boundary_cell_ids_from_type(sheet, "HC", type_by=g2.TYPE_BY,
                                                threshold=threshold)
    sc, _ = get_non_boundary_cell_ids_from_type(sheet, "SC", type_by=g2.TYPE_BY,
                                                threshold=threshold)
    idx = np.concatenate([np.asarray(hc, int), np.asarray(sc, int)])
    is_hc = np.isin(idx, np.asarray(hc, int))
    alpha = np.where(is_hc, R_alpha, 1.0)
    gamma = np.where(is_hc, gamma_sc * R_gamma, gamma_sc)
    A, P = area[idx], per[idx]
    return float((alpha * A * A).sum()), float((alpha * A).sum()), float((gamma * P * P).sum())


def exact_a0(sums, lam):
    """A0 from the exact stationarity condition (see the module docstring)."""
    num_aa, num_a, num_g = sums
    return lam ** 2 * num_aa / num_a + num_g / (2.0 * num_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="P0", choices=list(STAGES))
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--tasks-per-pool", dest="tasks_per_pool", type=int, default=None)
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, nargs="+", default=None)
    ap.add_argument("--r-gamma", dest="r_gamma", type=float, default=None,
                    help="fix R_gamma instead of deriving it from the A0=pi/4 family "
                         "— use with a list of --r-alpha to scan alpha at fixed gamma")
    ap.add_argument("--r-alpha", dest="r_alpha", type=float, nargs="+", default=None,
                    help="one value, or several to SCAN R_alpha against the ablation "
                         "term. P0 defaults to the stress-ratio value; E17.5 to %.2f, "
                         "which is an ASSUMPTION (see the docstring)" % E17_R_ALPHA)
    ap.add_argument("--margin", type=float, default=2e-3,
                    help="only sets the STARTING A0 (the step-5d boundary value)")
    ap.add_argument("--a0-iters", dest="a0_iters", type=int, default=3)
    ap.add_argument("--a0-tol", dest="a0_tol", type=float, default=2e-3)
    ap.add_argument("--relax", type=float, default=1.0,
                    help="damping on the A0 update; <1 if it oscillates")
    ap.add_argument("--out", default=None, help="override the output JSON name")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    stage = a.stage
    gammas = sorted(a.gamma_sc if a.gamma_sc is not None else GRID[stage])
    lam, pct = measured_lambda(stage)
    f_hc, n_arr = hc_fraction(stage, a.n_sheets)

    if a.r_alpha is not None:
        R_alphas, src = list(a.r_alpha), "supplied on the command line"
    elif stage == "P0":
        ra = derived_r(1)[0][0]
        R_alphas = snap_to_step5([ra], load_step5(os.path.join(RESULTS_DIR, STEP5_OUT)))
        src = "derived from the best E17.5 point via the stress ratio"
    else:
        R_alphas, src = [E17_R_ALPHA], "carried over from the coupled grid — AN ASSUMPTION"
    scanning_alpha = len(R_alphas) > 1

    print("=" * 78)
    print("SELF-CONSISTENT A0 | %s | %d point(s) | %d sheets | %d workers"
          % (stage, len(gammas), a.n_sheets, a.workers))
    print("=" * 78)
    print("  measured shrinkage %.4f%% -> lambda=%.6f ; f_HC=%.4f (%d arrays)"
          % (pct, lam, f_hc, n_arr))
    print("  R_alpha = %s  (%s)"
          % (", ".join("%.4f" % r for r in R_alphas), src))
    if scanning_alpha:
        # The ablation ratio is the ONLY alpha-sensitive term, and on the E17.5
        # coupled grid it saturates: the model value moves -0.060 per unit R at
        # R~1.5 but only -0.0008 by R~4. So expect a LOWER BOUND on R_alpha, not
        # a point estimate — and read the flat region as "anything above here".
        print("  SCANNING R_alpha against the ablation term (the only alpha-sensitive")
        print("       term). It saturates, so expect a lower bound, not an optimum.")
    elif stage != "P0" and a.r_alpha is None:
        print("  NOTE roundness is gamma-driven, so R_alpha is NOT determined by the")
        print("       roundness term. Pass several --r-alpha to set it from ablation.")
    mu_r, sem_r = experimental_stats("roundness_ratio", stage)
    print("  roundness target %.4f +/- %.4f  (SEM is %.1f%% of the mean)"
          % (mu_r, sem_r, 100 * sem_r / mu_r))

    initials = [initial_morphology_name(i, stage) for i in range(a.n_sheets)]
    # A point is a (gammaSC, R_gamma, R_alpha) triple. R_gamma either comes from
    # the A0=pi/4 family (the step-5d construction) or is pinned with --r-gamma,
    # which is what an R_alpha scan wants: hold the roundness solution fixed and
    # move only the alpha contrast.
    points = [(g, a.r_gamma if a.r_gamma is not None
               else rgamma_on_boundary(lam, g, ra, f_hc, a.margin), ra)
              for ra in R_alphas for g in gammas]
    A0_start = (np.pi / 4) * (lam ** 2 + 8 * ((1 - lam ** 2) / 8 * (1 - a.margin)))

    # Resume state is read BEFORE the dry-run exit, so --dry-run can show which
    # points are already done rather than only listing the plan.
    out_path = os.path.join(RESULTS_DIR, a.out or OUT[stage])
    done = {}
    if os.path.isfile(out_path) and not a.no_resume:
        try:
            done = dict(json.load(open(out_path)).get("points", {}))
        except (OSError, ValueError):
            done = {}
    enough = max(1, int(0.6 * a.n_sheets))

    def point_key(g, Rg, ra):
        return "Ra=%.4f|Rg=%.4f|gSC=%.5f" % (ra, Rg, g)

    def cached(g, Rg, ra):
        rec = done.get(point_key(g, Rg, ra))
        return rec if rec and (rec.get("n_sheets_ok") or 0) >= enough else None

    print("\n  %-9s %-9s %-9s %11s %s"
          % ("gammaSC", "R_gamma", "R_alpha", "A0 start", "state"))
    todo = 0
    for g, Rg, ra in points:
        rec = cached(g, Rg, ra)
        todo += rec is None
        print("  %-9.4g %-9.3f %-9.3f %11.5f %s"
              % (g, Rg, ra, A0_start, "TO RUN" if rec is None else
                 "cached (A0=%.5f, objective=%.4g)" % (rec.get("A0", float("nan")),
                                                       rec["objective"])))
    bad = [(g, Rg) for g, Rg, _ra in points if Rg <= 1]
    if bad:
        raise SystemExit("these points need R_gamma <= 1 (HC less round than SC): %s"
                         % [g for g, _ in bad])
    print("\n  %d point(s) to run; cost up to %d base passes + 1 scored pass each"
          " = up to %d runs" % (todo, a.a0_iters, todo * (a.a0_iters + 2) * len(initials)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    for g, Rg, R_alpha in points:
        key = point_key(g, Rg, R_alpha)
        if cached(g, Rg, R_alpha) is not None:   # already reported in the plan
            continue
        print("\n  " + "-" * 74)
        print("  gammaSC=%.4g  R_gamma=%.3f  R_alpha=%.3f" % (g, Rg, R_alpha))
        A0, trail = A0_start, [A0_start]
        for it in range(a.a0_iters):
            # BASE runs only: no ablated cells, so the ablation half is skipped.
            tasks = [(key, task(init, stage, g, Rg, R_alpha, A0, [])) for init in initials]
            g2.run_task_pool(tasks, a.workers, lambda *_a: None,
                             tasks_per_pool=a.tasks_per_pool, quiet=True)
            tot, n_ok = np.zeros(3), 0
            for init in initials:
                folder = base_folder(init, g, Rg, R_alpha, A0)
                if not os.path.isdir(os.path.join(RESULTS_DIR, folder)):
                    continue
                try:
                    tot += np.array(sheet_sums(folder, g, Rg, R_alpha,
                                               _load_saved_threshold(init)))
                    n_ok += 1
                except Exception as exc:              # noqa: BLE001
                    print("      %s while measuring %s" % (type(exc).__name__, folder[-12:]))
            if n_ok == 0:
                print("      no usable base runs — giving up on this point")
                A0 = None
                break
            A0_next = A0 + a.relax * (exact_a0(tot, lam) - A0)
            rel = abs(A0_next - A0) / A0
            print("    pass %d: %2d/%d sheets -> A0 %.5f -> %.5f  (%+.2f%%, %.4f x pi/4)"
                  % (it + 1, n_ok, len(initials), A0, A0_next, 100 * (A0_next - A0) / A0,
                     A0_next / (np.pi / 4)), flush=True)
            trail.append(A0_next)
            A0 = A0_next
            if rel < a.a0_tol:
                print("    converged (moved less than %.2g)" % a.a0_tol)
                break
        if A0 is None:
            continue

        results = []
        tasks = [(key, task(init, stage, g, Rg, R_alpha, A0, g2.ABLATED_CELLS))
                 for init in initials]
        g2.run_task_pool(tasks, a.workers, lambda _k, d, n, t: results.append(d),
                         tasks_per_pool=a.tasks_per_pool, quiet=True)
        z, obj, n_ok, n_cells = score_point(stage, results)
        done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok, "n_cells": n_cells,
                     "A0": A0, "A0_trail": trail, "gamma_sc": g, "R_gamma": Rg,
                     "R_alpha": R_alpha}
        with open(out_path, "w") as fh:
            json.dump({"stage": stage, "lambda": lam, "shrinkage_pct": pct,
                       "f_HC": f_hc, "R_alpha": R_alphas[0], "points": done}, fh, indent=1)
        print("    SCORED objective=%.4g  (%d/%d sheets)  A0=%.5f"
              % (obj, n_ok, len(initials), A0), flush=True)

    # ----- report ---------------------------------------------------------
    mu_a, sem_a = experimental_stats("ablation_ratio", stage)
    print("\n" + "=" * 78)
    print("%s  |  SELF-CONSISTENT A0" % stage)
    print("=" * 78)
    print("  %-9s %-8s %-8s %9s %9s %10s %9s %7s %7s %7s %6s"
          % ("gammaSC", "R_gamma", "R_alpha", "A0", "A0/(pi/4)", "objective",
             "roundness", "rnd_z", "abl_z", "shr_z", "n"))
    curve, alpha_curve = [], []
    for g, Rg, ra in points:
        rec = done.get(point_key(g, Rg, ra))
        if rec is None:
            print("  %-9.4g %-8.3f %-8.3f %9s" % (g, Rg, ra, "-")); continue
        z = rec.get("z") or {}
        zr = z.get("roundness_ratio", float("nan"))
        za = z.get("ablation_ratio", float("nan"))
        rnd = mu_r + zr * sem_r if np.isfinite(zr) else float("nan")
        if np.isfinite(rnd):
            curve.append((g, Rg, rnd))
        if np.isfinite(za):
            alpha_curve.append((ra, za, mu_a + za * sem_a))
        print("  %-9.4g %-8.3f %-8.3f %9.5f %9.4f %10.4g %9.4f %7.2f %7.2f %7.2f %6s"
              % (g, Rg, ra, rec["A0"], rec["A0"] / (np.pi / 4), rec["objective"], rnd,
                 zr, za, z.get("shrinkage", float("nan")), rec.get("n_sheets_ok")))

    if scanning_alpha and len(alpha_curve) >= 2:
        alpha_curve.sort()
        print("\n  ABLATION vs R_alpha (the only alpha-sensitive term)")
        print("    %-9s %10s %9s %s" % ("R_alpha", "model", "z", "gain over previous"))
        prev = None
        for ra, za, m in alpha_curve:
            gain = "" if prev is None else "%+.4f per unit R_alpha" % ((m - prev[1]) / (ra - prev[0]))
            print("    %-9.3f %10.4f %9.3f  %s" % (ra, m, za, gain))
            prev = (ra, m)
        best = min(alpha_curve, key=lambda c: abs(c[1]))
        # "flat" = the remaining gain is small next to the experimental SEM
        flat = [c for c in alpha_curve
                if abs(c[1]) - abs(best[1]) < 0.1]
        print("\n    best |z| = %.2f at R_alpha = %.3f" % (abs(best[1]), best[0]))
        if len(flat) > 1:
            print("    but R_alpha >= %.3f is INDISTINGUISHABLE (all within 0.1 sigma of"
                  " the best) — read this as a LOWER BOUND, not a point estimate"
                  % min(c[0] for c in flat))
        print("    experimental target %.4f +/- %.4f ; model floor is ~1/(1+d_SC), so"
              " |z| cannot reach 0 here" % (mu_a, sem_a))
    elif len(curve) >= 2:
        below = [c for c in curve if c[2] < mu_r]
        above = [c for c in curve if c[2] >= mu_r]
        if below and above:
            lo, hi = max(below, key=lambda c: c[2]), min(above, key=lambda c: c[2])
            t = (mu_r - lo[2]) / (hi[2] - lo[2])
            gx = lo[0] + t * (hi[0] - lo[0])
            Rx = float(np.exp(np.log(lo[1]) + t * (np.log(hi[1]) - np.log(lo[1]))))
            print("\n  target BRACKETED: crossing at gammaSC ~ %.5f, R_gamma ~ %.2f"
                  % (gx, Rx))
            xs = np.log([c[1] for c in curve]); ys = np.array([c[2] for c in curve])
            b = np.polyfit(xs, ys, 1)[0]
            print("  1 SEM of roundness = a factor %.2f in R_gamma (slope %.4f per ln R)"
                  % (np.exp(sem_r / abs(b)), b))
        else:
            print("\n  target NOT bracketed — curve stays %s it"
                  % ("below" if not above else "above"))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
