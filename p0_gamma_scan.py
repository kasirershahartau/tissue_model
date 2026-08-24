"""Step 5b — P0 scan over 10 gammaSC values at FIXED R.

    python p0_gamma_scan.py --dry-run
    python p0_gamma_scan.py --workers 30            # 32-core Azure VM
    python p0_gamma_scan.py --R 1.757 1.166         # scan at both step-5 R values

Step 5 held gammaSC at the value carried over from E17.5 (0.0175) and varied R
over the two derived candidates. That fit was not good enough, so this scans the
ONE parameter step 5 never moved: gammaSC, over its whole admissible band.

WHY R CAN STAY FIXED WHILE gammaSC MOVES. R_P0 is DERIVED, not fitted: the
measured stress/viscosity ratio pins avg_alpha, because substituting the step-1
A0 into sigma = alpha*(A - A0) + 2*pi*gamma makes the gamma terms cancel exactly,

    sigma = (pi/4) * (1 - lambda^2) * avg_alpha

That derivation contains no gammaSC at all. Scanning gammaSC therefore does NOT
invalidate the R it produced — the two parameters are separable in this model,
which is exactly what makes a 1-D scan legitimate rather than a shortcut.

A0 IS NOT FIXED ACROSS THE SCAN. A0 = (pi/4)(lambda_P0^2 + 8*gammaSC) is
recomputed at every point, so this sweeps A0 across its whole admissible band
[0.6693, 0.7854) together with the contractility. gammaSC and A0 cannot be
separated here; a point in this scan is a joint (contractility, preferred area)
setting, and the band is narrow by construction.

THE GRID. 10 values spanning (0, (1-lambda_P0^2)/8 = 0.018755), the bound that
enforces A0 < pi/4. Spacing is 0.002 up to 0.016 and then finer, because the
E17.5 grid bottomed out ON the upper boundary — if P0 does the same, the extra
resolution up there is where it will show. 0.0175 is included deliberately: it
is the step-5 point, so it is seeded from that run instead of re-simulated.

WHICH R. Always DERIVED from the best-fitting point of the E17.5 grid, through
the stress ratio: R_E -> avg_alpha_E17.5 -> k*avg_alpha -> R_P0. The step-5 JSON
is read only to display its result and to seed rows it already simulated — its
objectives do NOT choose R, because the two R values it tried were themselves
derived from E17.5 and picking between them by score would be fitting R by the
back door. --top carries more than one E17.5 candidate; --R overrides outright.

When a derived R agrees with a step-5 R to within 0.5%, the stored value is
adopted verbatim so that already-simulated row can be reused (the two differ
only in the 4th decimal, from the VM having recomputed the same formula).
"""
import argparse
import json
import os

import numpy as np

from post_processing import RESULTS_DIR, initial_morphology_name
from grid_fit_mechanics_v2 import (measured_lambda, preferred_area, build_task,
                                   score_point)
from p0_from_e17_stiffness import (hc_fraction, stress_ratio, e17_candidates,
                                   OUT as STEP5_OUT)

STAGE = "P0"
OUT = "p0_gamma_scan.json"

# See the docstring: even 0.002 steps, then finer against the A0 < pi/4 edge.
GRID_GAMMA_SC = [0.002, 0.004, 0.006, 0.008, 0.010,
                 0.012, 0.014, 0.016, 0.0175, 0.0185]


def point_key(R, gamma_sc):
    """Same key format as the E17.5 grid, so the two JSONs read alike."""
    return "R=%.4f|gSC=%.5f" % (R, gamma_sc)


def load_step5(path):
    """The finished step-5 run as {R: record}, or {} if it has not landed."""
    if not os.path.isfile(path):
        return {}
    try:
        pts = json.load(open(path)).get("points", {})
    except (OSError, ValueError):
        return {}
    out = {}
    for key, rec in pts.items():
        try:
            out[float(key.split("=", 1)[1])] = rec
        except (IndexError, ValueError):
            continue
    return out


def derived_r(top=1):
    """R_P0 for the best `top` E17.5 points, via the stress ratio (step-5 formula).

    Returns [(R_P0, R_E, sigma_R_P0), ...] and prints the derivation, so the R
    the scan runs at is always traceable to a specific E17.5 grid point.
    """
    lamE, _ = measured_lambda("E17.5")
    lamP, _ = measured_lambda(STAGE)
    fE, _ = hc_fraction("E17.5")
    fP, _ = hc_fraction(STAGE)
    ratio, sratio, stats, wb = stress_ratio()
    corr = (1 - lamE ** 2) / (1 - lamP ** 2)
    k, sk = ratio * corr, sratio * corr
    print("\n  R derived from the E17.5 fit via the measured stress ratio")
    print("    workbook %s" % wb)
    print("    stress/eta  E17.5 %.5f+/-%.5f (n=%d)   P0 %.5f+/-%.5f (n=%d)"
          % (stats["E17.5"][0], stats["E17.5"][1], stats["E17.5"][2],
             stats["P0"][0], stats["P0"][1], stats["P0"][2]))
    print("    ratio %.4f x shrinkage correction %.4f -> k = %.4f +/- %.4f"
          % (ratio, corr, k, sk))
    print("    f_HC  E17.5 %.4f   P0 %.4f" % (fE, fP))
    out = []
    for RE, g, obj in e17_candidates(top):
        aE = 1 + (RE - 1) * fE
        RP = 1 + (k * aE - 1) / fP
        sRP = (aE / fP) * sk
        print("    E17.5 best%s R_E=%-5.2f (gSC=%.4g, obj=%.4g) -> avg_a=%.4f"
              " -> R_P0=%.4f +/- %.4f"
              % ("" if not out else " #%d" % (len(out) + 1), RE, g, obj, aE, RP, sRP))
        if RP <= 1:
            print("      SKIPPED: violates R > 1")
            continue
        out.append((RP, RE, sRP))
    if not out:
        raise SystemExit("no feasible R_P0 > 1 was derived; pass --R explicitly")
    return out


def snap_to_step5(Rs, step5, tol=5e-3):
    """Adopt a step-5 R when the derived one matches it, so its row can be reused.

    The VM recomputed the same formula and landed a few 1e-4 away (e.g. 1.7570 vs
    1.7573). Without snapping, the seed key would miss and 10 sheets would be
    re-simulated for a physically identical point.
    """
    out = []
    for R in Rs:
        near = [r for r in step5 if abs(r - R) <= tol * max(R, 1.0)]
        if near:
            r5 = min(near, key=lambda r: abs(r - R))
            if abs(r5 - R) > 1e-9:
                print("  R %.4f snapped to the step-5 value %.4f (delta %.1e) so its"
                      " simulated row is reusable" % (R, r5, abs(r5 - R)))
            out.append(r5)
        else:
            out.append(R)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30,
                    help="30 on a 32-core VM leaves 2 cores for the parent + I/O")
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, nargs="+",
                    default=GRID_GAMMA_SC)
    ap.add_argument("--R", type=float, nargs="+", default=None,
                    help="default: derived from the best-fitting E17.5 grid point")
    ap.add_argument("--top", type=int, default=1,
                    help="how many E17.5 points to derive an R from (default 1: the best)")
    ap.add_argument("--step5-json", dest="step5_json",
                    default=os.path.join(RESULTS_DIR, STEP5_OUT))
    ap.add_argument("--no-seed", action="store_true",
                    help="re-run points step 5 already covered instead of reusing them")
    ap.add_argument("--keep-partial", dest="keep_partial", action="store_true",
                    help="keep cached points that lost sheets instead of re-running them")
    ap.add_argument("--min-sheets", dest="min_sheets", type=int, default=None,
                    help="re-run a cached point below this many good sheets "
                         "(default 60%% of --n-sheets: a wipeout, not a stray loss)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if any(g <= 0 for g in a.gamma_sc):
        raise SystemExit("gammaSC must be > 0")
    lam, pct = measured_lambda(STAGE)
    gamma_max = (1.0 - lam ** 2) / 8.0
    bad = [g for g in a.gamma_sc if g >= gamma_max]
    if bad:
        raise SystemExit("A0 < pi/4 requires gammaSC < %.6f for %s (shrinkage "
                         "%.4f%%); these violate it: %s" % (gamma_max, STAGE, pct, bad))

    step5 = load_step5(a.step5_json)

    print("=" * 78)
    print("STEP 5b | %s gammaSC scan | %d gammaSC values | %d sheets | %d workers"
          % (STAGE, len(a.gamma_sc), a.n_sheets, a.workers))
    print("=" * 78)
    print("  model: FaceContractility (p0=0), NO bending, alphaSC=1")
    print("  measured shrinkage %.4f%%  ->  lambda=%.6f" % (pct, lam))
    print("  A0 = (pi/4)(lambda^2 + 8*gammaSC); A0 < pi/4 => gammaSC < %.6f"
          % gamma_max)
    if step5:
        print("\n  step 5 (gammaSC fixed at %.4g, R varied):"
              % step5[list(step5)[0]].get("gamma_sc", float("nan")))
        for R in sorted(step5):
            z = step5[R].get("z") or {}
            print("    R=%-8.3f objective=%-9.4g round %+.2f  abl %+.2f  shrink %+.2f"
                  % (R, step5[R]["objective"],
                     z.get("roundness_ratio", float("nan")),
                     z.get("ablation_ratio", float("nan")),
                     z.get("shrinkage", float("nan"))))
    # R is DERIVED from the E17.5 fit, never chosen by P0 score — see the docstring.
    if a.R is not None:
        Rs, source = list(a.R), "supplied on the command line"
    else:
        Rs = [r for r, _RE, _s in derived_r(a.top)]
        source = "derived from the best %d E17.5 point(s)" % a.top
    Rs = snap_to_step5(Rs, step5)
    print("\n  R held at %s  (%s)"
          % (", ".join("%.4f" % r for r in Rs), source))

    out_path = os.path.join(RESULTS_DIR, OUT)
    done = {}
    if os.path.isfile(out_path) and not a.no_resume:
        try:
            done = dict(json.load(open(out_path)).get("points", {}))
        except (OSError, ValueError):
            done = {}
        # A point whose sheets mostly FAILED is stored like any other (objective
        # inf or the worst-case constant), so plain resume would treat a wipeout
        # as finished. That is exactly how the R_gamma scan nearly lost 3 points
        # to MemoryError. Drop the partial ones; completed sheets are reused from
        # disk, so retrying is cheap.
        # A WIPEOUT threshold, not "all 10": some sheets fail for real reasons at
        # stiff parameters, so requiring every sheet would re-run such a point on
        # every resume without ever satisfying it.
        min_sheets = (a.min_sheets if a.min_sheets is not None
                      else max(1, int(np.ceil(0.6 * a.n_sheets))))
        if not a.keep_partial:
            partial = [k for k, v in done.items()
                       if not v.get("from_step5")
                       and (v.get("n_sheets_ok") or 0) < min_sheets]
            for k in partial:
                del done[k]
            if partial:
                print("  discarding %d incomplete point(s) so they re-run: %s"
                      % (len(partial), ", ".join(sorted(partial))))
    # Step 5 already simulated (R, 0.0175) on these same sheets at these same
    # settings, so that point is a free row rather than 10 repeated runs.
    n_seeded = 0
    if step5 and not a.no_seed:
        for R, rec in step5.items():
            g = rec.get("gamma_sc")
            if g is None:
                continue
            key = point_key(R, g)
            if key in done or R not in Rs:
                continue
            if not any(point_key(R, gg) == key for gg in a.gamma_sc):
                continue
            done[key] = {"z": rec.get("z"), "objective": rec["objective"],
                         "n_sheets_ok": rec.get("n_sheets_ok"), "from_step5": True}
            n_seeded += 1
    if n_seeded:
        print("  seeded %d point(s) from step 5 (identical parameters)" % n_seeded)

    initials = [initial_morphology_name(i, STAGE) for i in range(a.n_sheets)]
    print("\n  %-10s %10s %11s %12s %s"
          % ("gammaSC", "A0", "A0/(pi/4)", "A0/actual", "state"))
    tasks = []
    for R in Rs:
        if len(Rs) > 1:
            print("  R = %.4f" % R)
        for g in a.gamma_sc:
            A0 = preferred_area(lam, g)
            key = point_key(R, g)
            rec = done.get(key)
            state = ("step 5" if rec and rec.get("from_step5")
                     else "cached" if rec else "TO RUN")
            print("  %-10.4g %10.5f %11.4f %11.3fx %s"
                  % (g, A0, A0 / (np.pi / 4), A0 / (400.0 / 508.0), state))
            if rec is None:
                for initial in initials:
                    tasks.append((key, build_task(STAGE, initial, R, g, A0)))
    n_points = len(tasks) // max(len(initials), 1)
    print("\n  %d point(s) to run x %d sheet(s) = %d task(s); each = 1 base + 1"
          " ablation run" % (n_points, len(initials), len(tasks)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")
    if not tasks:
        print("  every point already cached.")

    # One flat pool over (point, sheet) so workers never idle between points.
    results = {}
    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from mechanics_eval import _evaluate_mechanics_for_sheet
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = {ex.submit(_evaluate_mechanics_for_sheet, t): key
                       for key, t in tasks}
            for n, fut in enumerate(as_completed(futures), 1):
                key = futures[fut]
                try:
                    details = fut.result()
                except Exception as exc:      # noqa: BLE001 - one bad sheet only
                    print("  [%4d/%4d] %-24s sheet FAILED %s: %s"
                          % (n, len(tasks), key, type(exc).__name__, exc), flush=True)
                    details = None
                results.setdefault(key, []).append(details)
                if len(results[key]) == len(initials):
                    z, obj, n_ok, n_cells = score_point(STAGE, results[key])
                    done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok,
                                 "n_cells": n_cells}
                    with open(out_path, "w") as fh:
                        json.dump({"stage": STAGE, "lambda": lam,
                                   "shrinkage_pct": pct, "R": Rs,
                                   "points": done}, fh, indent=1)
                    print("  [%4d/%4d] %-24s DONE  objective=%.4g  (%d/%d sheets)"
                          % (n, len(tasks), key, obj, n_ok, len(initials)), flush=True)

    # ----- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("OBJECTIVE (sum z^2 over the 3 terms) vs gammaSC")
    print("=" * 78)
    for R in Rs:
        print("  R = %.4f" % R)
        print("    %-10s %10s %11s %10s %10s %10s"
              % ("gammaSC", "A0", "objective", "round_z", "abl_z", "shrink_z"))
        for g in a.gamma_sc:
            rec = done.get(point_key(R, g))
            if rec is None:
                print("    %-10.4g %10.5f %11s" % (g, preferred_area(lam, g), "-"))
                continue
            z = rec.get("z") or {}
            print("    %-10.4g %10.5f %11.4g %10s %10s %10s"
                  % (g, preferred_area(lam, g), rec["objective"],
                     "%+.2f" % z.get("roundness_ratio", float("nan")),
                     "%+.2f" % z.get("ablation_ratio", float("nan")),
                     "%+.2f" % z.get("shrinkage", float("nan"))))
    ranked = sorted(((v["objective"], k) for k, v in done.items()))
    if ranked:
        print("\n  best: %s  objective=%.4g" % (ranked[0][1], ranked[0][0]))
        edge = point_key(Rs[0], max(a.gamma_sc))
        if ranked[0][1] == edge:
            print("  NOTE the optimum sits ON the upper gammaSC edge, as E17.5 did —"
                  "\n       the A0 < pi/4 constraint is what is binding, not the grid.")
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
