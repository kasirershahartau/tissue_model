"""psigma sweep on the v2 model: mechanosensitivity against the 3 differentiation scores.

    python run_psigma_sweep_v2.py --dry-run
    python run_psigma_sweep_v2.py --stage E17.5 --workers 6
    python run_psigma_sweep_v2.py --psigma 0.15 0.25 0.35      # coarse bracket first

WHY NOT find_psigma. That function's contract is v1: mechanical_params is
(gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index) with gammaHC_ratio
FIXED at 1.0, and preferred_area is ONE value shared by both stages. v2 breaks
both — R_gamma is 6.746 / 3.851 and A0 is 0.74181 / 0.75854 — so this runs the
same loop with the v2 parameterisation instead of bending the old one.

THE GRID IS AN ORDER OF MAGNITUDE ABOVE THE v1 SWEEP (which used 0-0.09). The
gate is hill(max(sigma - K, 0), psigma) = s^m/(psigma^m + s^m), so psigma is a
HALF-MAX and only bites when comparable to the stress it sees. Measured on the
v2 runs at each stage's chosen initial frame (contractility effector, the set
run_model actually gates on):

    E17.5  t0 = 8.42   SC 0.250   HC 0.354
    P0     t0 = 9.21   SC 0.204   HC 0.251

At the old psigma ~ 0.05 the gate is ~0.96 for everything — no effect at all.

MEASURED RESULT (grid 0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.25 / 0.35), total score:

    E17.5   106   113   104    52.1   71.6   377    1806
    P0      188   200   278   155     412    747    1513

BOTH stages minimise at psigma = 0.15, and the window is NARROW: no benefit at
or below 0.10, collapse above. Score 1 carries it — 22.34 -> 1.00 (E17.5) and
15.40 -> 0.055 (P0) — i.e. the gate fixes the HC:HC deficit (the pattern was too
perfectly salt-and-pepper) at UNCHANGED HC density (~0.31 either way). It
changes WHICH cells differentiate, not how many.

TWO EARLIER CLAIMS IN THIS FILE WERE WRONG, corrected here:
  * "0.20-0.30 is where it discriminates" — no; 0.20 already collapses P0.
  * "the v1 lockout cannot occur with K = 0" — it does. Collapse is not caused
    by max(sigma - K, 0) zeroing cells; it is the gate suppressing delta
    production globally enough that lateral inhibition never bootstraps out of
    its initial overshoot. K only sets how abruptly that happens.

COLLAPSE THRESHOLDS scale with each stage's stress, as a gate-value threshold
should: P0 dies between 0.15 and 0.20 (all 10 runs, HC fraction 0.006-0.029),
E17.5 between 0.20 and 0.25. Ratio of thresholds ~1.29 against a stress ratio of
1.23. So the shared psigma is squeezed from above by P0 and from below by the
onset of any benefit at all.

DEAD RUNS ARE RETRIED ONCE (--dead-below, --no-retry). The LI seed is drawn from
np.random.rand on the UNSEEDED global RNG, so a re-run genuinely re-rolls the
initial condition; near threshold that coin flip decides the outcome. A run that
dies twice is recorded in the JSON under "flagged_dead_twice" and kept.

SCORING HAZARD, visible in the numbers above: when the tissue dies, score 2
reads a spurious 0 (sim% is nan, scored as zero) and P0's score 3 IMPROVES
(93.9 -> 7.1) because 0% differentiation is closer to P0's measured 4.33% than
the model's real 47% is. On scores 2+3 alone a dead tissue is the global
optimum for P0; only score 1 rejects it.

CAVEAT carried over from the stress analysis: SC-0 is NOT the lowest-stress
group (E17.5 t0: SC0 0.2190 vs SC1 0.2167; P0: 0.1884 vs 0.1800), so any gain
will come from global HC/SC suppression, not from sparing isolated SCs.

psigma = 0 reuses the existing fullmodel_v2 baseline for free.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, full_model_run_names, load_history_file,
                             get_time_points,
                             compare_full_model_differentiation_to_experiments as compare3)
from run_model import run_full_model_arrays
import grid_fit_mechanics_v2 as g2
from run_fitted_full_model import (best_point, NAME_PREFIX, ATOH_SENSITIVITY,
                                   NOTCH_SENSITIVITY, REPRESSOR_SENSITIVITY,
                                   INITIAL_LI_LEVEL, SHAPE_INDEX, BENDING, LINE_TENSION)

STAGES = ("E17.5", "P0")
# Centred on the measured stress scale; see the module docstring.
GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def final_hc_fraction(name):
    """HC fraction at a run's last frame. Well below the ~0.31 baseline means
    the run collapsed to all-SC — differentiation never started."""
    history = load_history_file(name)
    t = np.asarray(get_time_points(history), float)
    sheet = history.retrieve(float(t[-1]))
    sheet.arrange_sheet_from_history()
    d = sheet.face_df["delta_level"].to_numpy(float)
    return float((d > ATOH_SENSITIVITY).mean())


def retry_dead(stage, names, fit, psigma, a):
    """Re-run any array that ended dead, ONCE. Returns (dead, revived, flagged).

    Worth doing because the initial notch/delta/repressor levels come from
    np.random.rand on the UNSEEDED global RNG (there is no np.random.seed
    anywhere in the codebase), so a fresh process re-rolls the initial
    condition rather than reproducing the same death. Near the collapse
    threshold that is exactly the coin-flip that decides the outcome.

    The dead attempt is RENAMED, not deleted, so the evidence survives — a run
    that dies twice at the same psigma is a result, not a glitch.
    """
    dead = []
    for i, nm in enumerate(names):
        try:
            frac = final_hc_fraction(nm)
        except Exception as exc:                        # noqa: BLE001
            print("    [dead-check] %s unreadable (%s)" % (nm[-30:], type(exc).__name__),
                  flush=True)
            continue
        if frac < a.dead_below:
            dead.append((i, nm, frac))
    revived, flagged = [], []
    for i, nm, frac in dead:
        src = os.path.join(RESULTS_DIR, nm)
        dst = src + "__dead1"
        n = 2
        while os.path.exists(dst):
            dst = src + "__dead%d" % n
            n += 1
        try:
            os.rename(src, dst)
        except OSError as exc:
            print("    [retry] cannot set aside %s (%s); skipping" % (nm[-30:], exc),
                  flush=True)
            continue
        print("    [retry] array %d was dead (HC frac %.3f); re-running once"
              % (i, frac), flush=True)
        run_full_model_arrays(
            stage, gammaSC=fit["gamma_sc"], gammaHC_ratio=fit["R_gamma"],
            alphaHC_ratio=fit["R_alpha"],
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
            atoh_sensitivity=a.atoh if hasattr(a, "atoh") else ATOH_SENSITIVITY,
            notch_sensitivity=NOTCH_SENSITIVITY,
            repressor_sensitivity=REPRESSOR_SENSITIVITY,
            bending=BENDING, line_tension=LINE_TENSION,
            quasi_static_threshold=g2.BASE_QST, preferred_area=fit["A0"],
            psigma=psigma, stress_shift=a.stress_shift, stress_hill_exponent=a.m,
            initial_notch_delta_level=INITIAL_LI_LEVEL,
            t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
            indices=[i], n_workers=1,
            reuse_existing_run=False, name_prefix=a.name_prefix)
        try:
            new = final_hc_fraction(nm)
        except Exception as exc:                        # noqa: BLE001
            print("    [retry] array %d unreadable after re-run (%s)"
                  % (i, type(exc).__name__), flush=True)
            new = float("nan")
        if not (new >= a.dead_below):
            flagged.append(dict(array=i, first=frac, second=new))
            print("    [retry] array %d DIED AGAIN (HC frac %.3f) — FLAGGED, continuing"
                  % (i, new), flush=True)
        else:
            revived.append(dict(array=i, first=frac, second=new))
            print("    [retry] array %d recovered (HC frac %.3f)" % (i, new), flush=True)
    return [dict(array=i, hc_fraction=f) for i, _n, f in dead], revived, flagged


def out_path(stage, stress_shift, m):
    tag = "%s_ks%.3f" % (stage, stress_shift)
    if m is not None:
        tag += "_m%d" % m
    return os.path.join(RESULTS_DIR, "psigma_scores_v2_%s.json" % tag)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, nargs="+", default=GRID)
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0,
                    help="K. 0 in v2: all stress is positive, nothing to shift.")
    ap.add_argument("--hill-exponent", dest="m", type=int, default=None,
                    help="Hill exponent; None uses the model default")
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--t-end", dest="t_end", type=float, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--name-prefix", dest="name_prefix", default=NAME_PREFIX)
    ap.add_argument("--rescore", action="store_true",
                    help="re-evaluate the REQUESTED psigma values even if cached "
                         "(other points in the file are kept)")
    ap.add_argument("--dead-below", dest="dead_below", type=float, default=0.10,
                    help="final HC fraction below this = collapsed (baseline ~0.31)")
    ap.add_argument("--no-retry", dest="no_retry", action="store_true",
                    help="do not re-run collapsed runs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fits = {s: best_point(s) for s in a.stage}
    print("=" * 78)
    print("psigma SWEEP (v2)  |  K = %.3f  |  Hill m = %s  |  prefix %s_"
          % (a.stress_shift, a.m if a.m is not None else "default", a.name_prefix))
    print("=" * 78)
    for s in a.stage:
        f = fits[s]
        print("  %-6s gammaSC=%.4g  R_gamma=%.3f  R_alpha=%.3f  A0=%.5f"
              % (s, f["gamma_sc"], f["R_gamma"], f["R_alpha"], f["A0"]))
    print("  psigma grid: %s" % ", ".join("%.3g" % p for p in a.psigma))

    plan = []
    for stage in a.stage:
        done = {}
        p = out_path(stage, a.stress_shift, a.m)
        if os.path.isfile(p):
            try:
                done = json.load(open(p))
            except (OSError, ValueError):
                done = {}
        todo = [ps for ps in a.psigma
                if a.rescore or "%.5f" % ps not in done]
        plan.append((stage, todo, len(done)))
        print("  %-6s %d cached, %d to run: %s"
              % (stage, len(done), len(todo), ", ".join("%.3g" % x for x in todo)))
    n_sims = sum(len(t) for _s, t, _d in plan) * a.n_arrays
    print("\n  up to %d full-model run(s) + %d scoring pass(es)"
          % (n_sims, sum(len(t) for _s, t, _d in plan)))
    print("  (psigma = 0 reuses the existing %s baseline and costs nothing)"
          % a.name_prefix)
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    for stage, todo, _n in plan:
        f = fits[stage]
        p = out_path(stage, a.stress_shift, a.m)
        # Start from whatever is on disk even with --rescore: only the swept
        # keys get overwritten, so re-scoring one psigma cannot delete the rest.
        scores = {}
        if os.path.isfile(p):
            try:
                scores = json.load(open(p))
            except (OSError, ValueError):
                scores = {}
        for psigma in todo:
            print("\n" + "-" * 78)
            print("  %s  psigma = %.4g" % (stage, psigma), flush=True)
            names = run_full_model_arrays(
                stage,
                gammaSC=f["gamma_sc"], gammaHC_ratio=f["R_gamma"],
                alphaHC_ratio=f["R_alpha"],
                hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
                atoh_sensitivity=ATOH_SENSITIVITY,
                notch_sensitivity=NOTCH_SENSITIVITY,
                repressor_sensitivity=REPRESSOR_SENSITIVITY,
                bending=BENDING, line_tension=LINE_TENSION,
                quasi_static_threshold=g2.BASE_QST,
                preferred_area=f["A0"],
                psigma=psigma, stress_shift=a.stress_shift,
                stress_hill_exponent=a.m,
                initial_notch_delta_level=INITIAL_LI_LEVEL,
                t_end=a.t_end, dt=a.dt,
                save_interval=a.save_interval,
                n_arrays=a.n_arrays, n_workers=a.workers,
                reuse_existing_run=True, name_prefix=a.name_prefix)
            dead = revived = flagged = []
            if not a.no_retry:
                dead, revived, flagged = retry_dead(stage, names, f, psigma, a)
                if dead:
                    print("    %d dead, %d recovered, %d still dead"
                          % (len(dead), len(revived), len(flagged)), flush=True)
            try:
                res = compare3(stage, model_names=names)
            except Exception as exc:                       # noqa: BLE001
                print("  scoring FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
                res = None
            if res is not None:
                res = dict(res)
                res["dead_runs"] = dead
                res["revived_runs"] = revived
                res["flagged_dead_twice"] = flagged
            scores["%.5f" % psigma] = res
            with open(p, "w") as fh:
                json.dump(scores, fh, indent=1, default=float)
            print("  -> %s" % p, flush=True)

        print("\n" + "=" * 78)
        print("%s  |  psigma sweep (lower is better)" % stage)
        print("=" * 78)
        print("  %-9s %10s %10s %10s %10s %10s"
              % ("psigma", "score1", "score2", "score3", "s1+s2", "total"))
        for key in sorted(scores, key=float):
            v = scores[key]
            if not v:
                print("  %-9.4g   failed" % float(key)); continue
            s = [v.get("score%d" % i, float("nan")) for i in (1, 2, 3)]
            print("  %-9.4g %10.4g %10.4g %10.4g %10.4g %10.4g"
                  % (float(key), s[0], s[1], s[2], s[0] + s[1],
                     v.get("total", sum(s))))


if __name__ == "__main__":
    main()
