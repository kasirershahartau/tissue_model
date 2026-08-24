"""Step 4 — 5x5 GRID search of the v2 mechanical fit (E17.5), no BO.

    python grid_fit_mechanics_v2.py --dry-run
    python grid_fit_mechanics_v2.py --workers 30          # 32-core Azure VM
    python grid_fit_mechanics_v2.py --stage P0 --workers 30

THE v2 MODEL. Face CONTRACTILITY (Gamma/2 * P^2, i.e. shape index p0 = 0) and
NO bending. Two free parameters:

    R        = alphaHC = gammaHC / gammaSC     (alphaSC = 1 by definition)
    gammaSC

A0 is NOT fitted. Step 1 gives it in closed form from the measured cut
shrinkage, and step 2's single-ratio assumption (alphaHC/alphaSC ==
gammaHC/gammaSC) makes avg_gamma/avg_alpha collapse to gammaSC exactly,
independent of the HC/SC counts:

    A0 = (pi/4) * (lambda^2 + 8 * gammaSC),     lambda = 1 - shrinkage%/100

so A0 depends on gammaSC ALONE — no self-consistency iteration is needed. The
measured shrinkage is read from the experimental data at run time, not hardcoded.

Sanity check that the derivation rests on: the idealisation "every cell is a
circle of diameter 1" puts A = pi/4 = 0.7854 against an actual packed area of
400/508 = 0.7874, a 0.3% match.

SCORE (step 3): three terms — HC/SC roundness ratio, HC/SC ablation area-change
ratio, and shrinkage — each z = (mean_model - mean_exp) / SEM_exp with the
experimental mean and SEM taken over PER-EXPERIMENT means. Objective = sum z^2.

PARALLELISM. The unit of work is one (grid point, initial sheet) pair, and each
pair runs a base simulation plus an ablation simulation. All 25 x n_sheets pairs
are flattened into ONE pool so 30 workers stay saturated instead of idling at
the end of each grid point — the same reason the cloud BO used a flat batch.
Results are written after every completed point, so an interrupted search keeps
what it finished and a re-run skips it.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             load_experimental_results,
                             compare_pooled_model_mechanics_to_experiments)
from run_model import MECHANICS_TERMS, _MECHANICS_ROUNDNESS_TERMS, _MECHANICS_ABLATION_TERMS, _WORST_CASE_NSIGMA, _load_saved_threshold
from mechanics_eval import _evaluate_mechanics_for_sheet

# --- the 5x5 grid ----------------------------------------------------------
# R > 1 (HC stiffer than SC), geometric with ratio ~sqrt(2).
GRID_R = [1.25, 1.75, 2.5, 3.5, 5.0]
# gammaSC is bounded by the requirement A0 < pi/4 (the cell must be STRETCHED
# relative to its preferred area, which is what puts the tissue under tension):
#
#     (pi/4)(lambda^2 + 8*gammaSC) < pi/4   <=>   gammaSC < (1 - lambda^2)/8
#
# For E17.5 (shrinkage 7.5076%) that is gammaSC < 0.018064; for P0 < 0.018755.
# The bound is computed per stage at run time and enforced, so the grid can
# never silently violate it.
#
# LINEAR spacing, because A0 is linear in gammaSC — this sweeps A0 evenly across
# its whole admissible band. Note that band is narrow: A0 is confined to
# [0.6719, 0.7854), i.e. 0.855 to 1.0 x (pi/4). The model CANNOT reach the
# previous fit's A0 = 0.466; if the score bottoms out against the low-gammaSC
# edge, that is a statement about the model, not about the grid.
GRID_GAMMA_SC = [0.002, 0.006, 0.010, 0.014, 0.0175]

# --- held fixed (matches run_refit.py so scores stay comparable) ------------
ABLATED_CELLS = (337, 304, 65, 114)
POST_ABLATION_FRAME = -1
TYPE_BY = "delta_level"
BASE_QST = 0.03
ABLATION_QST = 0.02
LINE_TENSION = None
BENDING = 0.0                     # v2: NO bending elasticity
SHAPE_INDEX = 0.0                 # v2: p0 = 0 -> pure contractility
MIN_PROGRESS_RATE = 1e-4
MAX_WALL_SECONDS = 10000
PROGRESS_WINDOW_SECONDS = 30.0
RERUN_STALLED = False

OUT_JSON = "grid_fit_mechanics_v2_%s.json"


def measured_lambda(stage):
    """Linear shrinkage RATIO from the experiments (the data stores a %)."""
    arrays = [np.asarray(a, float) for a in load_experimental_results(stage, "cut shrinkage")]
    per_repeat = [a[np.isfinite(a)].mean() for a in arrays if np.isfinite(a).any()]
    pct = float(np.mean(per_repeat))
    return 1.0 - pct / 100.0, pct


def preferred_area(lam, gamma_sc):
    """Step-1 closed form: A0 = (pi/4)(lambda^2 + 8*gammaSC)."""
    return (np.pi / 4.0) * (lam ** 2 + 8.0 * float(gamma_sc))


def build_task(stage, initial, R, gamma_sc, A0):
    """The argument tuple _evaluate_mechanics_for_sheet expects.

    alphaHC_ratio and gammaHC_ratio are BOTH R — that is the step-2 assumption."""
    return (float(gamma_sc), float(R), float(R), initial, stage,
            list(ABLATED_CELLS), POST_ABLATION_FRAME,
            _load_saved_threshold(initial), TYPE_BY,
            MAX_WALL_SECONDS, MIN_PROGRESS_RATE, PROGRESS_WINDOW_SECONDS,
            RERUN_STALLED, SHAPE_INDEX,
            BASE_QST, ABLATION_QST, LINE_TENSION,
            SHAPE_INDEX, SHAPE_INDEX, BENDING, float(A0))


def run_task_pool(tasks, workers, handle, tasks_per_pool=None, quiet=False):
    """Run ``(key, args)`` tasks through worker pools, RECYCLING THE WORKERS.

    Worker memory grows across tasks and is never returned, so a long flat pool
    dies of MemoryError partway through — and it dies on whatever was scheduled
    LAST, which looks exactly like a parameter-dependent failure. Two runs hit
    this: the R_gamma scan lost its last 3 points, and the boundary scan lost 6
    (60 of 62 failures were MemoryError, including gammaSC=0.026 / R_gamma=1.01,
    the most benign point in the whole scan — proof it is not the parameters).

    ``ProcessPoolExecutor(max_tasks_per_child=...)`` would be the clean fix but
    it needs Python 3.11; this runs on 3.10. So the pool is torn down and rebuilt
    every ``tasks_per_pool`` tasks, which forces the workers to exit and hand
    their memory back. Rebuilding costs a few seconds against tasks that take
    minutes, so the overhead is negligible.

    ``handle(key, details, n, total)`` is called once per finished task, with
    ``details=None`` if it raised. Tasks are consumed IN ORDER, so batches line
    up with grid points when the caller groups its tasks that way.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from mechanics_eval import _evaluate_mechanics_for_sheet

    if tasks_per_pool is None:
        tasks_per_pool = max(1, 3 * workers)
    total, n = len(tasks), 0
    for start in range(0, total, tasks_per_pool):
        batch = tasks[start:start + tasks_per_pool]
        if not quiet and total > tasks_per_pool:
            print("  -- worker pool %d/%d (tasks %d-%d of %d) --"
                  % (start // tasks_per_pool + 1,
                     (total + tasks_per_pool - 1) // tasks_per_pool,
                     start + 1, start + len(batch), total), flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_evaluate_mechanics_for_sheet, t): key
                       for key, t in batch}
            for fut in as_completed(futures):
                key = futures[fut]
                n += 1
                try:
                    details = fut.result()
                except Exception as exc:      # noqa: BLE001 - one bad sheet only
                    print("  [%4d/%4d] %-36s sheet FAILED %s: %s"
                          % (n, total, key, type(exc).__name__, exc), flush=True)
                    details = None
                handle(key, details, n, total)


def score_point(stage, details_list):
    """Pool one grid point's sheets -> {term: z}, objective, n contributing."""
    model_terms = {t: [] for t in MECHANICS_TERMS}
    n_ok = 0
    for details in details_list:
        if details is None:
            continue
        n_ok += 1
        for t in MECHANICS_TERMS:
            if details.get(t) is not None:
                model_terms[t].append(details[t])
    if n_ok == 0:
        return None, float("inf"), 0, {}
    z = compare_pooled_model_mechanics_to_experiments(model_terms, stage)
    active = (list(_MECHANICS_ROUNDNESS_TERMS) + ["shrinkage"]
              + list(_MECHANICS_ABLATION_TERMS))
    obj = 0.0
    for t in active:
        zz = z.get(t, float("nan"))
        if not np.isfinite(zz):
            zz = _WORST_CASE_NSIGMA          # degenerate -> large but finite
        obj += float(zz * zz)
    n_cells = {t: int(sum(len(a) for a in model_terms[t])) for t in MECHANICS_TERMS}
    return z, obj, n_ok, n_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="E17.5", choices=["E17.5", "P0"])
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30,
                    help="30 on a 32-core VM leaves 2 cores for the parent + I/O")
    ap.add_argument("--R", type=float, nargs="+", default=GRID_R)
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, nargs="+",
                    default=GRID_GAMMA_SC)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if any(r <= 1 for r in a.R):
        raise SystemExit("R must be > 1 (HC stiffer than SC)")
    if any(g <= 0 for g in a.gamma_sc):
        raise SystemExit("gammaSC must be > 0")

    lam, pct = measured_lambda(a.stage)
    # A0 < pi/4  <=>  gammaSC < (1 - lambda^2)/8. Enforced rather than assumed,
    # because the bound depends on the stage's measured shrinkage.
    gamma_max = (1.0 - lam ** 2) / 8.0
    bad = [g for g in a.gamma_sc if g >= gamma_max]
    if bad:
        raise SystemExit(
            "A0 < pi/4 requires gammaSC < %.6f for %s (shrinkage %.4f%%); "
            "these violate it: %s" % (gamma_max, a.stage, pct, bad))
    initials = [initial_morphology_name(i, a.stage) for i in range(a.n_sheets)]
    points = [(R, g) for R in a.R for g in a.gamma_sc]

    out_path = os.path.join(RESULTS_DIR, OUT_JSON % a.stage)
    done = {}
    if os.path.isfile(out_path) and not a.no_resume:
        try:
            done = {k: v for k, v in json.load(open(out_path)).get("points", {}).items()}
        except (OSError, ValueError):
            done = {}

    print("=" * 78)
    print("v2 GRID FIT  |  %s  |  %d x %d grid  |  %d sheets  |  %d workers"
          % (a.stage, len(a.R), len(a.gamma_sc), len(initials), a.workers))
    print("=" * 78)
    print("  model: FaceContractility (p0=0), NO bending, alphaSC=1")
    print("  measured shrinkage %.4f%%  ->  lambda=%.6f" % (pct, lam))
    print("  A0 = (pi/4)(lambda^2 + 8*gammaSC) = %.4f + %.4f*gammaSC"
          % (np.pi / 4 * lam ** 2, np.pi / 4 * 8))
    print("  A0 < pi/4 (cells stretched) => gammaSC < %.6f ; A0 confined to "
          "[%.4f, %.4f)" % (gamma_max, np.pi / 4 * lam ** 2, np.pi / 4))
    print()
    print("  %-8s %-10s %10s %11s %12s %s"
          % ("R", "gammaSC", "A0", "A0/(pi/4)", "A0/actual", "state"))
    tasks = []
    for R, g in points:
        A0 = preferred_area(lam, g)
        key = "R=%.4f|gSC=%.5f" % (R, g)
        state = "cached" if key in done else "TO RUN"
        print("  %-8.3f %-10.4g %10.5f %11.4f %11.3fx %s"
              % (R, g, A0, A0 / (np.pi / 4), A0 / (400.0 / 508.0), state))
        if key not in done:
            for initial in initials:
                tasks.append((key, build_task(a.stage, initial, R, g, A0)))
    print()
    print("  %d point(s) to run x %d sheet(s) = %d task(s); each task = 1 base run"
          " + 1 ablation run" % (len(tasks) // max(len(initials), 1), len(initials),
                                 len(tasks)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")
    if not tasks:
        print("  every point already cached.")

    # One flat pool over (point, sheet) so workers never idle between points.
    results = {}
    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
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
                    z, obj, n_ok, n_cells = score_point(a.stage, results[key])
                    done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok,
                                 "n_cells": n_cells}
                    with open(out_path, "w") as fh:
                        json.dump({"stage": a.stage, "lambda": lam,
                                   "shrinkage_pct": pct, "points": done}, fh, indent=1)
                    print("  [%4d/%4d] %-24s DONE  objective=%.4g  (%d/%d sheets)"
                          % (n, len(tasks), key, obj, n_ok, len(initials)), flush=True)

    # ----- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("OBJECTIVE (sum z^2 over the 3 terms) — rows R, cols gammaSC")
    print("=" * 78)
    print("  %-8s" % "R" + "".join("%12s" % ("%.4g" % g) for g in a.gamma_sc))
    for R in a.R:
        row = "  %-8.3f" % R
        for g in a.gamma_sc:
            rec = done.get("R=%.4f|gSC=%.5f" % (R, g))
            row += "%12s" % ("-" if rec is None else "%.4g" % rec["objective"])
        print(row)
    ranked = sorted(((v["objective"], k) for k, v in done.items()))
    if ranked:
        print("\n  best 5 points:")
        print("    %-24s %12s %10s %10s %10s"
              % ("point", "objective", "roundness", "ablation", "shrinkage"))
        for obj, key in ranked[:5]:
            z = done[key].get("z") or {}
            print("    %-24s %12.4g %10s %10s %10s"
                  % (key, obj,
                     "%+.2f" % z.get("roundness_ratio", float("nan")),
                     "%+.2f" % z.get("ablation_ratio", float("nan")),
                     "%+.2f" % z.get("shrinkage", float("nan"))))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
