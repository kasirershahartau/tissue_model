"""Step 5 — P0 at FIXED gammaSC, with R set by the measured stiffness ratio.

    python p0_from_e17_stiffness.py --dry-run
    python p0_from_e17_stiffness.py --workers 30        # 32-core Azure VM
    python p0_from_e17_stiffness.py --r-p0 1.76 1.17    # bypass the derivation

Instead of fitting P0 from scratch, alphaHC for P0 is DERIVED from the
circular-ablation stiffness data and only the two E17.5 candidates are carried
over. gammaSC is held at its E17.5 value, so this run varies R alone.

THE DERIVATION. The isotropic stress in this model is
sigma = dE/dA = alpha*(A - A0) + 2*pi*gamma per cell. Summing over cells and
substituting the step-1 A0, the gamma terms CANCEL exactly:

    sigma = (pi/4) * (1 - lambda^2) * avg_alpha

so the measured stress/viscosity fixes avg_alpha once the shrinkage is known.
With alphaSC = 1 and alphaHC = R, avg_alpha = 1 + (R - 1) * f_HC, hence

    k    = (stress_P0 / stress_E17.5) * (1 - lam_E^2) / (1 - lam_P^2)
    R_P0 = 1 + [ k * (1 + (R_E - 1) * f_E) - 1 ] / f_P0

The (1 - lambda^2) correction is what distinguishes STRESS from Young's
modulus: the areal modulus is K = alpha*A and carries no such factor. The
stress column is used here because it is the more reliable measurement (SEM 18%
vs 23%), at the cost of needing that correction.

f_HC IS NOT ITERATED. The mechanical runs use no_differentiation=True, so cell
types never change during a run and the HC fraction is fixed by each initial
array plus its saved delta threshold. It is measured here, not assumed.

A0 IS NOT QUITE FIXED across the stages even at fixed gammaSC:
A0 = (pi/4)(lambda^2 + 8*gammaSC) uses each stage's OWN measured shrinkage, and
lambda differs (0.9249 E17.5 vs 0.9219 P0), so A0 moves 0.78185 -> 0.77752.
Forcing E17.5's A0 onto P0 would break P0's shrinkage term, so the stage's own
lambda is used.

FEASIBILITY. P0 is measured SOFTER, so avg_alpha must fall; since
avg_alpha = 1 + (R-1)*f_HC bottoms out at 1, R_P0 > 1 requires R_E above a
threshold (~2.22 for the stress route). Any candidate that fails it is reported
and skipped rather than run with an invalid R.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, resolve_circular_ablation_file,
                             initial_morphology_name, load_history_file,
                             get_time_points, get_non_boundary_cell_ids_from_type)
from run_model import _load_saved_threshold
from grid_fit_mechanics_v2 import (measured_lambda, preferred_area, build_task,
                                   score_point, OUT_JSON)

STAGES = ("E17.5", "P0")
OUT = "p0_from_e17_stiffness.json"


def hc_fraction(stage, n_sheets=10):
    """Mean HC fraction over a stage's initial arrays (fixed; see module docs)."""
    out = []
    for i in range(n_sheets):
        name = initial_morphology_name(i, stage)
        if not os.path.isdir(os.path.join(RESULTS_DIR, name)):
            continue
        history = load_history_file(name)
        sheet = history.retrieve(float(np.max(get_time_points(history))))
        sheet.arrange_sheet_from_history()
        threshold = _load_saved_threshold(name)
        all_idx, _ = get_non_boundary_cell_ids_from_type(
            sheet, "all", type_by="delta_level", threshold=threshold)
        hc_idx, _ = get_non_boundary_cell_ids_from_type(
            sheet, "HC", type_by="delta_level", threshold=threshold)
        if all_idx.size:
            out.append(len(hc_idx) / float(all_idx.size))
    if not out:
        raise SystemExit("no initial arrays found for %s" % stage)
    return float(np.mean(out)), len(out)


def stress_ratio():
    """(P0/E17.5 stress-over-viscosity, its propagated SEM) from the workbook."""
    # resolve_circular_ablation_file tries both layouts: the local "Raw Data"
    # sibling folder and the VM's copy INSIDE the experimental data dir.
    path = resolve_circular_ablation_file()
    if not os.path.isfile(path):
        raise SystemExit("circular-ablation workbook not found at %s (set "
                         "CIRCULAR_ABLATION_FILE, or pass --k / --r-p0)" % path)
    d = pd.read_excel(path, sheet_name="Overall data")
    col = "Stress over viscosity (1/min)"
    if col not in d.columns:
        raise SystemExit("column %r missing from %s — pass --k or --r-p0 instead"
                         % (col, path))
    stats = {}
    for stage in STAGES:
        v = pd.to_numeric(d[d["Stage"] == stage][col], errors="coerce").dropna()
        stats[stage] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), v.size)
    (mE, sE, nE), (mP, sP, nP) = stats["E17.5"], stats["P0"]
    ratio = mP / mE
    return ratio, ratio * float(np.hypot(sE / mE, sP / mP)), stats, path


def e17_candidates(top=2):
    """(R_E, gammaSC, objective) for the best `top` points of the E17.5 grid."""
    path = os.path.join(RESULTS_DIR, OUT_JSON % "E17.5")
    if not os.path.isfile(path):
        raise SystemExit("E17.5 grid result not found at %s" % path)
    pts = json.load(open(path))["points"]
    rows = []
    for key, val in pts.items():
        m = re.match(r"R=([\d.]+)\|gSC=([\d.]+)", key)
        rows.append((float(m.group(1)), float(m.group(2)), val["objective"]))
    rows.sort(key=lambda r: r[2])
    return rows[:top]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--top", type=int, default=2,
                    help="how many E17.5 points to carry over (default 2: best + runner-up)")
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, default=None,
                    help="override the gammaSC carried from E17.5")
    ap.add_argument("--k", type=float, default=None,
                    help="override the derived avg_alpha ratio")
    ap.add_argument("--r-p0", dest="r_p0", type=float, nargs="+", default=None,
                    help="skip the derivation and use these R values directly")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lamE, pctE = measured_lambda("E17.5")
    lamP, pctP = measured_lambda("P0")
    fE, nE = hc_fraction("E17.5", a.n_sheets)
    fP, nP = hc_fraction("P0", a.n_sheets)

    print("=" * 78)
    print("STEP 5 | P0 at fixed gammaSC, R derived from the measured stress ratio")
    print("=" * 78)
    print("  shrinkage   E17.5 %.4f%% (lam %.6f)   P0 %.4f%% (lam %.6f)"
          % (pctE, lamE, pctP, lamP))
    print("  f_HC        E17.5 %.4f (%d arrays)     P0 %.4f (%d arrays)"
          % (fE, nE, fP, nP))

    cands = e17_candidates(a.top)
    print("\n  E17.5 candidates carried over:")
    for i, (RE, g, obj) in enumerate(cands, 1):
        print("    %d) R_E=%-6.2f gammaSC=%-8.4g objective=%.4g" % (i, RE, g, obj))

    gamma_sc = a.gamma_sc if a.gamma_sc is not None else cands[0][1]
    if any(abs(c[1] - gamma_sc) > 1e-12 for c in cands):
        print("    NOTE the candidates do not share one gammaSC; using %.4g for all"
              % gamma_sc)

    # A0 uses P0's OWN lambda — see module docstring.
    gamma_max = (1.0 - lamP ** 2) / 8.0
    if gamma_sc >= gamma_max:
        raise SystemExit("gammaSC=%.5f violates A0 < pi/4 for P0 (limit %.6f)"
                         % (gamma_sc, gamma_max))
    A0 = preferred_area(lamP, gamma_sc)
    print("\n  gammaSC held at %.4g  ->  A0(P0) = %.5f   (A0(E17.5) was %.5f;"
          " they differ only through each stage's own shrinkage)"
          % (gamma_sc, A0, preferred_area(lamE, gamma_sc)))
    print("  P0 bound: gammaSC < %.6f — satisfied" % gamma_max)

    if a.r_p0 is not None:
        points = [(r, None, None) for r in a.r_p0]
        print("\n  R values supplied directly: %s" % a.r_p0)
    else:
        if a.k is not None:
            k, sk = a.k, 0.0
            print("\n  avg_alpha ratio k supplied directly: %.4f" % k)
        else:
            ratio, sratio, stats, wb = stress_ratio()
            corr = (1 - lamE ** 2) / (1 - lamP ** 2)
            k, sk = ratio * corr, sratio * corr
            print("\n  workbook: %s" % wb)
            print("  stress/viscosity  E17.5 %.5f+/-%.5f (n=%d)   P0 %.5f+/-%.5f (n=%d)"
                  % (stats["E17.5"][0], stats["E17.5"][1], stats["E17.5"][2],
                     stats["P0"][0], stats["P0"][1], stats["P0"][2]))
            print("  ratio %.4f +/- %.4f  x shrinkage correction %.4f  ->  k = %.4f +/- %.4f"
                  % (ratio, sratio, corr, k, sk))
        need = 1 + (1.0 / k - 1.0) / fE
        print("  feasibility: R_P0 > 1 requires R_E > %.3f" % need)
        points = []
        for RE, g, obj in cands:
            aE = 1 + (RE - 1) * fE
            RP = 1 + (k * aE - 1) / fP
            sRP = (aE / fP) * sk
            if RP <= 1:
                print("    R_E=%.2f -> R_P0=%.3f  SKIPPED (violates R > 1)" % (RE, RP))
                continue
            points.append((RP, RE, sRP))
            print("    R_E=%-6.2f -> avg_a_E=%.4f -> avg_a_P0=%.4f -> R_P0=%.3f +/- %.3f"
                  % (RE, aE, k * aE, RP, sRP))
    if not points:
        raise SystemExit("no feasible R_P0 — see the feasibility line above.")

    initials = [initial_morphology_name(i, "P0") for i in range(a.n_sheets)]
    tasks = [("R=%.4f" % RP, build_task("P0", init, RP, gamma_sc, A0))
             for RP, _RE, _s in points for init in initials]
    print("\n  %d point(s) x %d sheet(s) = %d task(s); each = 1 base + 1 ablation run"
          % (len(points), len(initials), len(tasks)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from run_model import _evaluate_mechanics_for_sheet
    results, done = {}, {}
    out_path = os.path.join(RESULTS_DIR, OUT)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(_evaluate_mechanics_for_sheet, t): key for key, t in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                details = fut.result()
            except Exception as exc:                    # noqa: BLE001
                print("  [%3d/%3d] %-12s sheet FAILED %s: %s"
                      % (n, len(tasks), key, type(exc).__name__, exc), flush=True)
                details = None
            results.setdefault(key, []).append(details)
            if len(results[key]) == len(initials):
                z, obj, n_ok, n_cells = score_point("P0", results[key])
                done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok,
                             "gamma_sc": gamma_sc, "A0": A0}
                with open(out_path, "w") as fh:
                    json.dump({"stage": "P0", "lambda": lamP, "gamma_sc": gamma_sc,
                               "A0": A0, "f_HC": {"E17.5": fE, "P0": fP},
                               "points": done}, fh, indent=1)
                print("  [%3d/%3d] %-12s DONE objective=%.4g (%d/%d sheets)"
                      % (n, len(tasks), key, obj, n_ok, len(initials)), flush=True)

    print("\n" + "=" * 78)
    print("  %-10s %-9s %11s %10s %10s %10s" % ("R_P0", "from R_E", "objective",
                                                "round_z", "abl_z", "shrink_z"))
    for RP, RE, _s in points:
        rec = done.get("R=%.4f" % RP)
        if rec is None:
            continue
        z = rec.get("z") or {}
        print("  %-10.3f %-9s %11.4g %10s %10s %10s"
              % (RP, "-" if RE is None else "%.2f" % RE, rec["objective"],
                 "%+.2f" % z.get("roundness_ratio", float("nan")),
                 "%+.2f" % z.get("ablation_ratio", float("nan")),
                 "%+.2f" % z.get("shrinkage", float("nan"))))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
