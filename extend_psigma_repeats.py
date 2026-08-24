"""Extend psigma = 0 / 0.162 / 0.163 to 5 repeats, retry dead runs, score them out.

    python extend_psigma_repeats.py --dry-run
    python extend_psigma_repeats.py --workers 4

WHY THESE THREE psigma. P0's score 2 reads 6.05 at psigma 0.162 and 35.49 at
0.163 — a 6x jump across one 0.001 step, which persisted after averaging three
repeats. The HC fractions rule out collapse as the cause: all 30 runs at both
values are alive and equally suppressed (mean 0.2637 vs 0.2596). So the jump is
variance in WHICH cells differentiate, and the only way to pin it down is more
realisations. psigma = 0 comes along as the no-mechanosensitivity reference.

WHAT IT DOES, in order:

  1. SIMULATE repeats 4 and 5 (prefixes fullmodel_v2r4 / _v2r5).
  2. CHECK every repeat 1-5 for collapse — final HC fraction below --dead-below
     (default 0.10, against a ~0.31 baseline). The strict "< 1%" test has
     undercounted twice: runs sitting at 2-3% are collapsed but would pass it.
  3. RETRY each dead run ONCE. The dead attempt is renamed, not deleted, so a
     run that dies twice stays on disk as evidence. Re-running genuinely
     re-rolls the outcome because the lateral-inhibition seed comes from
     np.random.rand on the UNSEEDED global RNG.
  4. FLAG runs that die twice, and EXCLUDE them from scoring rather than
     averaging their zeros into their array's group.

WHY EXCLUSION MATTERS. Score 2 already skips a run with no differentiating
cells, but scores 1 and 3 do NOT — they append unconditionally, so a dead run
contributes HC:HC = 0, HC:SC = 0 and 0% differentiating. Under grouping those
zeros are averaged into that array's mean at 1/N weight and quietly drag the
score. E17.5 repeat 2 already has two such runs (arrays 0 and 4) at both 0.162
and 0.163.

The result is written separately from the 3-repeat pooled scores so the two can
be compared rather than one overwriting the other.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             compare_full_model_differentiation_to_experiments as compare3)
from full_model import run_full_model_arrays
import grid_fit_mechanics_v2 as g2
from run_fitted_full_model import (best_point, ATOH_SENSITIVITY, NOTCH_SENSITIVITY,
                                   REPRESSOR_SENSITIVITY, INITIAL_LI_LEVEL,
                                   SHAPE_INDEX, BENDING, LINE_TENSION)
from run_psigma_repeats import REPEAT_PREFIX
from score_psigma_pooled import run_name

STAGES = ("E17.5", "P0")
PSIGMA = [0.0, 0.162, 0.163]
TH = 0.355079
OUT = "psigma_scores_v2_5rep_%s_ks%.3f.json"


def hc_fraction(name):
    """Final-frame HC fraction; well below the ~0.31 baseline means collapsed."""
    history = load_history_file(name)
    t = np.asarray(get_time_points(history), float)
    sheet = history.retrieve(float(t[-1]))
    sheet.arrange_sheet_from_history()
    d = sheet.face_df["delta_level"].to_numpy(float)
    return float((d > TH).mean())


def simulate(stage, psigma, repeat, fit, a):
    run_full_model_arrays(
        stage, gammaSC=fit["gamma_sc"], gammaHC_ratio=fit["R_gamma"],
        alphaHC_ratio=fit["R_alpha"],
        hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
        atoh_sensitivity=ATOH_SENSITIVITY, notch_sensitivity=NOTCH_SENSITIVITY,
        repressor_sensitivity=REPRESSOR_SENSITIVITY,
        bending=BENDING, line_tension=LINE_TENSION,
        quasi_static_threshold=g2.BASE_QST, preferred_area=fit["A0"],
        psigma=psigma, stress_shift=a.stress_shift,
        initial_notch_delta_level=INITIAL_LI_LEVEL,
        t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
        n_arrays=a.n_arrays, n_workers=a.workers,
        reuse_existing_run=True, name_prefix=REPEAT_PREFIX[repeat])


def retry_one(stage, psigma, repeat, i, fit, a):
    """Set the dead run aside and re-run that single array. -> new HC fraction."""
    nm = run_name(stage, psigma, REPEAT_PREFIX[repeat], i, a.stress_shift)
    src = os.path.join(RESULTS_DIR, nm)
    dst, n = src + "__dead1", 2
    while os.path.exists(dst):
        dst = src + "__dead%d" % n
        n += 1
    try:
        os.rename(src, dst)
    except OSError as exc:
        print("      cannot set aside %s (%s)" % (nm[-40:], exc), flush=True)
        return float("nan")
    run_full_model_arrays(
        stage, gammaSC=fit["gamma_sc"], gammaHC_ratio=fit["R_gamma"],
        alphaHC_ratio=fit["R_alpha"],
        hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
        atoh_sensitivity=ATOH_SENSITIVITY, notch_sensitivity=NOTCH_SENSITIVITY,
        repressor_sensitivity=REPRESSOR_SENSITIVITY,
        bending=BENDING, line_tension=LINE_TENSION,
        quasi_static_threshold=g2.BASE_QST, preferred_area=fit["A0"],
        psigma=psigma, stress_shift=a.stress_shift,
        initial_notch_delta_level=INITIAL_LI_LEVEL,
        t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
        indices=[i], n_workers=1,
        reuse_existing_run=False, name_prefix=REPEAT_PREFIX[repeat])
    try:
        return hc_fraction(nm)
    except Exception:                                    # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, nargs="+", default=PSIGMA)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--simulate", type=int, nargs="+", default=[4, 5],
                    help="repeat indices to simulate; the rest must already exist")
    ap.add_argument("--dead-below", dest="dead_below", type=float, default=0.10)
    ap.add_argument("--no-retry", dest="no_retry", action="store_true")
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0)
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-end", dest="t_end", type=float, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fits = {s: best_point(s) for s in a.stage}
    print("=" * 78)
    print("EXTEND TO %d REPEATS | psigma %s | %s"
          % (len(a.repeats), ", ".join("%.3f" % p for p in a.psigma),
             ", ".join(a.stage)))
    print("=" * 78)
    print("  simulate repeats %s ; check %s ; dead below HC fraction %.2f"
          % (a.simulate, a.repeats, a.dead_below))
    print("  new runs: %d" % (len(a.simulate) * len(a.psigma) * len(a.stage) * a.n_arrays))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    # ---- 1. simulate the new repeats ------------------------------------
    for rep in a.simulate:
        for psigma in a.psigma:
            for stage in a.stage:
                print("\n" + "-" * 78)
                print("  simulate repeat %d | %s | psigma %.3f" % (rep, stage, psigma),
                      flush=True)
                simulate(stage, psigma, rep, fits[stage], a)

    # ---- 2/3/4. liveness, retry, score with the dead excluded ------------
    report = {}
    for stage in a.stage:
        out = os.path.join(RESULTS_DIR, OUT % (stage, a.stress_shift))
        scores = {}
        if os.path.isfile(out):
            try:
                scores = json.load(open(out))
            except (OSError, ValueError):
                scores = {}
        for psigma in a.psigma:
            print("\n" + "=" * 78)
            print("  %s  psigma %.3f  — liveness" % (stage, psigma), flush=True)
            groups, flagged, revived = [], [], []
            for i in range(a.n_arrays):
                grp = []
                for rep in a.repeats:
                    nm = run_name(stage, psigma, REPEAT_PREFIX[rep], i, a.stress_shift)
                    if not os.path.isdir(os.path.join(RESULTS_DIR, nm)):
                        continue
                    try:
                        frac = hc_fraction(nm)
                    except Exception as exc:             # noqa: BLE001
                        print("    array %d repeat %d unreadable (%s)"
                              % (i, rep, type(exc).__name__), flush=True)
                        continue
                    if frac >= a.dead_below:
                        grp.append(nm)
                        continue
                    if a.no_retry:
                        flagged.append(dict(array=i, repeat=rep, first=frac))
                        continue
                    print("    array %d repeat %d dead (HC %.3f); retrying once"
                          % (i, rep, frac), flush=True)
                    new = retry_one(stage, psigma, rep, i, fits[stage], a)
                    if new >= a.dead_below:
                        revived.append(dict(array=i, repeat=rep, first=frac, second=new))
                        grp.append(nm)
                        print("      recovered (HC %.3f)" % new, flush=True)
                    else:
                        flagged.append(dict(array=i, repeat=rep, first=frac, second=new))
                        print("      DIED AGAIN (HC %.3f) — FLAGGED, excluded from scoring"
                              % new, flush=True)
                if grp:
                    groups.append(grp)
            n_runs = sum(len(g) for g in groups)
            print("    %d array(s), %d usable run(s); %d revived, %d excluded"
                  % (len(groups), n_runs, len(revived), len(flagged)), flush=True)
            if not groups:
                scores["%.5f" % psigma] = None
                continue
            try:
                res = compare3(stage, model_groups=groups)
                res.update(n_arrays=len(groups), n_runs=n_runs,
                           revived=revived, flagged_dead_twice=flagged)
            except Exception as exc:                     # noqa: BLE001
                print("    scoring FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
                res = None
            scores["%.5f" % psigma] = res
            with open(out, "w") as fh:
                json.dump(scores, fh, indent=1, default=float)
        report[stage] = (scores, out)

    print("\n" + "=" * 78)
    print("5-REPEAT SCORES (dead runs excluded)")
    print("=" * 78)
    for stage, (scores, out) in report.items():
        print("  %s" % stage)
        print("    %-8s %9s %9s %9s %9s %9s %6s %8s"
              % ("psigma", "score1", "score2", "score3", "s1+s2", "total",
                 "runs", "excluded"))
        for k in sorted(scores, key=float):
            v = scores[k]
            if not v:
                print("    %-8.3g   failed" % float(k)); continue
            s = [v.get("score%d" % i, float("nan")) for i in (1, 2, 3)]
            print("    %-8.3g %9.4g %9.4g %9.4g %9.4g %9.4g %6s %8d"
                  % (float(k), s[0], s[1], s[2], s[0] + s[1], v.get("total", sum(s)),
                     v.get("n_runs", "?"), len(v.get("flagged_dead_twice") or [])))
        print("    wrote %s\n" % out)


if __name__ == "__main__":
    main()
