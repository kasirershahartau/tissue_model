"""Step 5c — DIAGNOSTIC: does HC/SC roundness respond to R_gamma or to R_alpha?

    python p0_rgamma_scan.py --dry-run
    python p0_rgamma_scan.py --workers 30           # 32-core Azure VM

Step 5 fitted P0 badly: objective 263.9, essentially ALL of it the roundness
term (z = -15.98). Roundness on the E17.5 grid is set almost entirely by R
(0.10-0.18 across the R range, versus 0.0065 across the whole gammaSC range),
and hitting P0's target needs R ~ 3.9 while the measured stress ratio derives
R_P0 = 1.757. Under step 2's single-ratio assumption those cannot both hold.

THE WAY OUT. The stress constrains avg_alpha ALONE:

    sigma = (pi/4) * (1 - lambda^2) * avg_alpha

so pinning R_alpha = 1.757 keeps the stress match EXACTLY while R_gamma is free
to supply the roundness contrast. But the existing grid moved alpha and gamma
together, so it cannot say which of them roundness actually responds to. This
scan answers that one question, and nothing else: R_alpha and gammaSC fixed,
R_gamma swept.

  * If roundness barely moves -> it is alpha-driven, decoupling buys nothing,
    and the conflict between the stress data and the roundness data is real.
  * If roundness tracks R_gamma -> decoupling is the fix, and the follow-up is a
    2-D (gammaSC, R_gamma) grid at fixed R_alpha.

STEP 1 SURVIVES THE DECOUPLING. Its derivation never assumed alpha and gamma
were proportional — it minimises E = sum_i [alpha_i/2 (lam^2 A - A0)^2 +
gamma_i/2 (lam P)^2] over lam with every cell a circle of diameter 1, giving
A0 = lam^2 A + (P^2/2A)*(sum gamma / sum alpha). What the single-ratio
assumption bought was that sum(gamma)/sum(alpha) collapsed to gammaSC with the
HC/SC counts cancelling. Decoupled it does not cancel, but it is still closed
form, because f_HC is FIXED and measured (no_differentiation=True):

    avg_gamma/avg_alpha = gammaSC * (R_gamma*f + 1 - f) / (R_alpha*f + 1 - f)
    A0 = (pi/4) * (lambda^2 + 8 * avg_gamma/avg_alpha)

A0 is therefore recomputed at every point, which keeps the SHRINKAGE term
matched all the way along the sweep. A0 co-varying with R_gamma is not a
confound to remove — it is what holds shrinkage fixed while the contrast moves.

R_gamma > 1 IS REQUIRED, not merely conventional. FaceContractility penalises
PERIMETER (Gamma/2 * P^2), so a cell with higher Gamma shrinks its perimeter
towards the minimum-perimeter shape at fixed area — a circle. gammaHC > gammaSC
is therefore precisely what makes HC ROUNDER than SC. R_gamma < 1 predicts the
opposite ordering, a roundness ratio BELOW 1, when the measurement is ~1.20 for
both stages. Such points cannot be part of any acceptable fit, so they are
rejected (--allow-rgamma-below-1 overrides, for diagnostics only).

THE CEILING SQUEEZES THE GRID. A0 < pi/4 requires
avg_gamma/avg_alpha < (1-lambda^2)/8 = 0.018755, which at gammaSC = 0.0175 caps
R_gamma at about 1.95. Together with R_gamma > 1 that leaves the narrow window
(1, 1.95) — and note R_alpha = 1.757 already sits inside it, so the sweep
resolves a modest contrast range rather than a wide lever arm. If roundness does
not visibly move across this window the result is suggestive rather than
conclusive, and the honest follow-up is to LOWER gammaSC (which raises the
ceiling: R_gamma ~ 3.9 needs gammaSC ~ 0.0102) rather than to read a null into
a narrow window. --allow-large-a0 lifts the A0 check for a deliberate look past
it.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             load_experimental_results, _finite_arrays,
                             _MECHANICS_EXPERIMENTAL_TYPE)
from run_model import _load_saved_threshold
import grid_fit_mechanics_v2 as g2
from grid_fit_mechanics_v2 import measured_lambda, score_point
from p0_from_e17_stiffness import hc_fraction, e17_candidates, OUT as STEP5_OUT
from p0_gamma_scan import load_step5, derived_r, snap_to_step5

STAGE = "P0"
OUT = "p0_rgamma_scan.json"

# Confined to (1, ceiling): R_gamma > 1 because HC must come out ROUNDER than
# SC, and R_gamma < 1.95 because of A0 < pi/4 at gammaSC = 0.0175. Four points
# below R_alpha = 1.757 and two squeezed above it, plus the free anchor.
GRID_R_GAMMA = [1.1, 1.25, 1.4, 1.55, 1.85, 1.94]


def build_task(initial, gamma_sc, R_gamma, R_alpha, A0):
    """The tuple _evaluate_mechanics_for_sheet expects.

    THE ORDER IS A TRAP: the worker unpacks (gammaSC, gammaHC_ratio,
    alphaHC_ratio, ...) — GAMMA ratio second, ALPHA ratio third.
    grid_fit_mechanics_v2.build_task passes R in both slots, so the order is
    invisible there (and its comment lists them the other way round). Here the
    two differ, so getting it backwards would silently swap the experiment.
    """
    return (float(gamma_sc), float(R_gamma), float(R_alpha), initial, STAGE,
            list(g2.ABLATED_CELLS), g2.POST_ABLATION_FRAME,
            _load_saved_threshold(initial), g2.TYPE_BY,
            g2.MAX_WALL_SECONDS, g2.MIN_PROGRESS_RATE, g2.PROGRESS_WINDOW_SECONDS,
            g2.RERUN_STALLED, g2.SHAPE_INDEX,
            g2.BASE_QST, g2.ABLATION_QST, g2.LINE_TENSION,
            g2.SHAPE_INDEX, g2.SHAPE_INDEX, g2.BENDING, float(A0))


def gamma_over_alpha(gamma_sc, R_gamma, R_alpha, f_hc):
    """avg_gamma/avg_alpha with the ratios decoupled (alphaSC = 1)."""
    return gamma_sc * (R_gamma * f_hc + 1 - f_hc) / (R_alpha * f_hc + 1 - f_hc)


def preferred_area(lam, gamma_sc, R_gamma, R_alpha, f_hc):
    """Step-1 A0, decoupled form."""
    return (np.pi / 4.0) * (lam ** 2 + 8.0 * gamma_over_alpha(gamma_sc, R_gamma,
                                                              R_alpha, f_hc))


def rgamma_ceiling(lam, gamma_sc, R_alpha, f_hc):
    """Largest R_gamma with A0 < pi/4, i.e. avg_gamma/avg_alpha < (1-lam^2)/8."""
    gmax = (1.0 - lam ** 2) / 8.0
    return ((gmax / gamma_sc) * (R_alpha * f_hc + 1 - f_hc) - (1 - f_hc)) / f_hc


def experimental_stats(term, stage):
    """(grand mean, SEM) so a stored z can be turned back into a model mean."""
    arrays = _finite_arrays(load_experimental_results(stage, _MECHANICS_EXPERIMENTAL_TYPE[term]))
    means = np.array([a.mean() for a in arrays], dtype=float)
    return float(means.mean()), float(means.std(ddof=1) / np.sqrt(means.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=30,
                    help="30 on a 32-core VM leaves 2 cores for the parent + I/O")
    ap.add_argument("--r-gamma", dest="r_gamma", type=float, nargs="+",
                    default=GRID_R_GAMMA)
    ap.add_argument("--r-alpha", dest="r_alpha", type=float, default=None,
                    help="default: derived from the best-fitting E17.5 grid point")
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, default=None,
                    help="default: the gammaSC of the best-fitting E17.5 grid point")
    ap.add_argument("--step5-json", dest="step5_json",
                    default=os.path.join(RESULTS_DIR, STEP5_OUT))
    ap.add_argument("--allow-large-a0", dest="allow_large_a0", action="store_true",
                    help="lift the A0 < pi/4 check (see the docstring's CEILING note)")
    ap.add_argument("--allow-rgamma-below-1", dest="allow_small_rgamma",
                    action="store_true",
                    help="permit R_gamma <= 1, which makes HC LESS round than SC")
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--keep-partial", dest="keep_partial", action="store_true",
                    help="keep cached points that lost sheets instead of re-running them")
    ap.add_argument("--min-sheets", dest="min_sheets", type=int, default=None,
                    help="re-run a cached point below this many good sheets "
                         "(default 60%% of --n-sheets: a wipeout, not a stray loss)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lam, pct = measured_lambda(STAGE)
    fP, nP = hc_fraction(STAGE, a.n_sheets)
    step5 = load_step5(a.step5_json)

    print("=" * 78)
    print("STEP 5c | %s R_gamma diagnostic | %d R_gamma values | %d sheets | %d workers"
          % (STAGE, len(a.r_gamma), a.n_sheets, a.workers))
    print("=" * 78)
    print("  model: FaceContractility (p0=0), NO bending, alphaSC=1")
    print("  measured shrinkage %.4f%% -> lambda=%.6f ; f_HC=%.4f (%d arrays)"
          % (pct, lam, fP, nP))

    # gammaSC and R_alpha both come from the E17.5 fit — neither is fitted here.
    best_RE, best_g, best_obj = e17_candidates(1)[0]
    gamma_sc = a.gamma_sc if a.gamma_sc is not None else best_g
    print("  gammaSC held at %.4g (from the best E17.5 point: R_E=%.2f, obj=%.4g)"
          % (gamma_sc, best_RE, best_obj))
    if a.r_alpha is not None:
        R_alpha, src = a.r_alpha, "supplied on the command line"
    else:
        R_alpha = derived_r(1)[0][0]
        src = "derived from the best E17.5 point via the stress ratio"
    R_alpha = snap_to_step5([R_alpha], step5)[0]
    print("\n  R_alpha PINNED at %.4f  (%s)" % (R_alpha, src))
    print("  -> avg_alpha = %.4f is unchanged along the sweep, so the measured"
          " stress ratio stays matched at every point"
          % (R_alpha * fP + 1 - fP))

    # Contractility penalises perimeter, so gammaHC > gammaSC is what makes HC
    # rounder than SC. R_gamma <= 1 predicts a roundness ratio below 1 against a
    # measured ~1.20 — outside any acceptable fit, not merely a worse one.
    small = [g for g in a.r_gamma if g <= 1]
    if small and not a.allow_small_rgamma:
        raise SystemExit(
            "R_gamma must be > 1 (HC rounder than SC); these are not: %s\n"
            "pass --allow-rgamma-below-1 only for a deliberate diagnostic" % small)
    if small:
        print("  --allow-rgamma-below-1: %s predict HC LESS round than SC" % small)

    ceiling = rgamma_ceiling(lam, gamma_sc, R_alpha, fP)
    print("  A0 < pi/4 caps R_gamma at %.4f at this gammaSC" % ceiling)
    over = [g for g in a.r_gamma if g >= ceiling]
    if over and not a.allow_large_a0:
        raise SystemExit(
            "these R_gamma exceed the A0 < pi/4 ceiling of %.4f: %s\n"
            "lower gammaSC to buy headroom, or pass --allow-large-a0 deliberately"
            % (ceiling, over))
    if over:
        print("  --allow-large-a0: %s exceed the ceiling and will run with A0 >= pi/4"
              % over)

    # The R_gamma == R_alpha point IS the coupled step-5 point (the decoupled A0
    # formula reduces to gammaSC there), so it comes free as the sweep's anchor.
    r_gammas = sorted(set(list(a.r_gamma) + [R_alpha]))

    out_path = os.path.join(RESULTS_DIR, OUT)
    done = {}
    if os.path.isfile(out_path) and not a.no_resume:
        try:
            done = dict(json.load(open(out_path)).get("points", {}))
        except (OSError, ValueError):
            done = {}
        # A point whose sheets mostly FAILED is stored like any other (objective
        # inf or the worst-case constant), so plain resume would treat a wipeout
        # as finished and never retry it. That is how a run that lost 3 points to
        # MemoryError would silently stay lost. Drop the partial ones so a re-run
        # with fewer workers picks them up; completed sheets are reused from disk,
        # so retrying is cheap.
        #
        # The threshold is a WIPEOUT threshold, not "all 10". Some sheets fail for
        # real reasons at stiff parameters — the virtual-vertex collapse that
        # empties a face (IndexError) and the occasional dt-floor stall — so a
        # point can be stuck at 8/10 forever. Requiring all 10 would re-run such a
        # point on every resume without ever satisfying it.
        min_sheets = (a.min_sheets if a.min_sheets is not None
                      else max(1, int(np.ceil(0.6 * a.n_sheets))))
        if not a.keep_partial:
            partial = {k: v for k, v in done.items()
                       if not v.get("from_step5")
                       and (v.get("n_sheets_ok") or 0) < min_sheets}
            for k in partial:
                del done[k]
            if partial:
                print("  discarding %d incomplete point(s) so they re-run: %s"
                      % (len(partial), ", ".join("%s (%s/%d sheets)"
                         % (k.split("|")[1], partial[k].get("n_sheets_ok"), a.n_sheets)
                         for k in sorted(partial))))

    def key_of(Rg):
        return "Ra=%.4f|Rg=%.4f|gSC=%.5f" % (R_alpha, Rg, gamma_sc)

    if step5 and not a.no_seed:
        rec = step5.get(R_alpha)
        if (rec is not None and abs(rec.get("gamma_sc", -1) - gamma_sc) < 1e-12
                and key_of(R_alpha) not in done):
            done[key_of(R_alpha)] = {"z": rec.get("z"), "objective": rec["objective"],
                                     "n_sheets_ok": rec.get("n_sheets_ok"),
                                     "from_step5": True}
            print("  seeded the anchor R_gamma=%.4f from step 5 (same parameters)"
                  % R_alpha)

    initials = [initial_morphology_name(i, STAGE) for i in range(a.n_sheets)]
    print("\n  %-10s %12s %10s %11s %s"
          % ("R_gamma", "avg_g/avg_a", "A0", "A0/(pi/4)", "state"))
    tasks = []
    for Rg in r_gammas:
        A0 = preferred_area(lam, gamma_sc, Rg, R_alpha, fP)
        key = key_of(Rg)
        rec = done.get(key)
        state = ("step 5 (anchor)" if rec and rec.get("from_step5")
                 else "cached" if rec else "TO RUN")
        print("  %-10.4g %12.6f %10.5f %11.4f %s"
              % (Rg, gamma_over_alpha(gamma_sc, Rg, R_alpha, fP), A0,
                 A0 / (np.pi / 4), state))
        if rec is None:
            for initial in initials:
                tasks.append((key, build_task(initial, gamma_sc, Rg, R_alpha, A0)))
    print("\n  %d point(s) to run x %d sheet(s) = %d task(s); each = 1 base + 1"
          " ablation run" % (len(tasks) // max(len(initials), 1), len(initials),
                             len(tasks)))
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
                    print("  [%4d/%4d] %-34s sheet FAILED %s: %s"
                          % (n, len(tasks), key, type(exc).__name__, exc), flush=True)
                    details = None
                results.setdefault(key, []).append(details)
                if len(results[key]) == len(initials):
                    z, obj, n_ok, n_cells = score_point(STAGE, results[key])
                    done[key] = {"z": z, "objective": obj, "n_sheets_ok": n_ok,
                                 "n_cells": n_cells}
                    with open(out_path, "w") as fh:
                        json.dump({"stage": STAGE, "lambda": lam, "shrinkage_pct": pct,
                                   "f_HC": fP, "gamma_sc": gamma_sc,
                                   "R_alpha": R_alpha, "points": done}, fh, indent=1)
                    print("  [%4d/%4d] %-34s DONE  objective=%.4g  (%d/%d sheets)"
                          % (n, len(tasks), key, obj, n_ok, len(initials)), flush=True)

    # ----- report + verdict -----------------------------------------------
    mu_r, sem_r = experimental_stats("roundness_ratio", STAGE)
    print("\n" + "=" * 78)
    print("R_gamma sweep at R_alpha = %.4f, gammaSC = %.4g" % (R_alpha, gamma_sc))
    print("=" * 78)
    print("  %-9s %9s %11s %11s %10s %10s %10s"
          % ("R_gamma", "A0", "objective", "roundness", "round_z", "abl_z", "shrink_z"))
    roundness = {}
    for Rg in r_gammas:
        rec = done.get(key_of(Rg))
        A0 = preferred_area(lam, gamma_sc, Rg, R_alpha, fP)
        if rec is None:
            print("  %-9.4g %9.5f %11s" % (Rg, A0, "-"))
            continue
        z = rec.get("z") or {}
        zr = z.get("roundness_ratio", float("nan"))
        if np.isfinite(zr):
            roundness[Rg] = mu_r + zr * sem_r
        print("  %-9.4g %9.5f %11.4g %11s %10s %10s %10s"
              % (Rg, A0, rec["objective"],
                 "%.4f" % roundness.get(Rg, float("nan")), "%+.2f" % zr,
                 "%+.2f" % z.get("ablation_ratio", float("nan")),
                 "%+.2f" % z.get("shrinkage", float("nan"))))
    print("\n  experimental target: roundness ratio %.4f +/- %.4f (SEM)" % (mu_r, sem_r))
    if len(roundness) >= 2:
        span = max(roundness.values()) - min(roundness.values())
        dRg = max(roundness) - min(roundness)
        print("  roundness spans %.4f over R_gamma %.3g..%.3g (%.4f per unit R_gamma)"
              % (span, min(roundness), max(roundness), span / dRg if dRg else float("nan")))
        print("  for reference the COUPLED grid moved it 0.0464 per unit R"
              " (1.094@R=1.75 -> 1.183@R=3.5)")
        print("\n  VERDICT: %s" % (
            "gamma-driven — decoupling is the fix; follow up with a 2-D "
            "(gammaSC, R_gamma) grid at fixed R_alpha, since lower gammaSC is what "
            "buys R_gamma headroom." if span / max(dRg, 1e-9) > 0.015 else
            "alpha-driven — roundness barely responds to R_gamma, so decoupling "
            "does NOT resolve the stress-vs-roundness conflict. Reconsider the "
            "stress->avg_alpha mapping or the f_HC ~ 0.50 of the arrays."))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
