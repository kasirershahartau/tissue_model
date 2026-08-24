"""Re-run collapsed / unreadable full-model runs ONCE, then report what survived.

    python revive_dead_runs.py --psigma 0.164 0.165 --repeats 4 5 --dry-run
    python revive_dead_runs.py --psigma 0.164 0.165 --repeats 4 5 --workers 4

WHY A RETRY IS LEGITIMATE. The initial notch/delta/repressor levels come from
np.random.rand on the UNSEEDED global RNG, so a fresh process re-rolls the
initial condition instead of reproducing the same death. Near the collapse
threshold that roll is exactly what decides the outcome, so a dead run is one
draw from the outcome distribution rather than a broken simulation.

WHICH MEANS THE RETRY IS NOT FREE OF BIAS, AND THE REPORT SAYS SO. Retrying only
the deaths and keeping whatever comes back censors the collapse rate downwards:
at a psigma where the tissue genuinely collapses 20% of the time, retrying until
it doesn't makes the point look healthier than it is. That is why each attempt is
RENAMED to __deadN rather than deleted and the summary prints how many arrays
died, how many were revived and how many died twice — a run that dies twice is a
result. Use the flagged count when describing how close a psigma sits to
collapse; use the revived runs only for the scores.

DEAD means the final-frame HC fraction is below --dead-below (default 0.10,
against a healthy ~0.25-0.31), or the history cannot be read at all.
"""
import argparse
import json
import os

import numpy as np

from post_processing import RESULTS_DIR, initial_morphology_name
from run_model import _psigma_tag
from full_model import run_full_model_arrays
from run_psigma_sweep_v2 import final_hc_fraction
from run_psigma_repeats import REPEAT_PREFIX, STAGES
from run_fitted_full_model import (best_point, ATOH_SENSITIVITY, NOTCH_SENSITIVITY,
                                   REPRESSOR_SENSITIVITY, INITIAL_LI_LEVEL,
                                   SHAPE_INDEX, BENDING, LINE_TENSION)
import grid_fit_mechanics_v2 as g2

OUT_JSON = "revive_dead_runs.json"


def run_name(stage, psigma, prefix, i, stress_shift=0.0):
    """Folder name as _run_full_model_one builds it (psigma=0 carries no tag)."""
    init = initial_morphology_name(i, stage)
    if float(psigma) == 0.0:
        return "%s_%s" % (prefix, init)
    return "%s_ps%s_ks%.3f_%s" % (prefix, _psigma_tag(psigma), stress_shift, init)


def state_of(name, dead_below):
    """('alive'|'dead'|'missing', hc_fraction)."""
    if not os.path.isdir(os.path.join(RESULTS_DIR, name)):
        return "missing", np.nan
    try:
        frac = final_hc_fraction(name)
    except Exception:                                   # noqa: BLE001
        return "dead", np.nan                           # unreadable counts as dead
    return ("dead" if frac < dead_below else "alive"), frac


def set_aside(name):
    """Rename to __deadN so the evidence survives. Returns the new name."""
    src = os.path.join(RESULTS_DIR, name)
    n = 1
    while os.path.exists(src + "__dead%d" % n):
        n += 1
    dst = src + "__dead%d" % n
    os.rename(src, dst)
    return os.path.basename(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--psigma", type=float, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0)
    ap.add_argument("--dead-below", dest="dead_below", type=float, default=0.10)
    ap.add_argument("--max-dead-fraction", dest="max_dead_fraction", type=float,
                    default=1.0,
                    help="skip a point whose dead fraction reaches this. Past ~half "
                         "dead, collapse IS the measurement there, and retrying "
                         "only the deaths would censor the very quantity being "
                         "measured.")
    ap.add_argument("--dead-scope", dest="dead_scope", default="psigma",
                    choices=["psigma", "stage"],
                    help="'psigma' (default): if EITHER stage trips the threshold, "
                         "skip that psigma at both stages — the point cannot serve "
                         "a joint two-stage fit, so reviving the healthier stage "
                         "buys nothing. 'stage': judge each stage on its own.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--t-end", dest="t_end", type=float, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fits = {s: best_point(s) for s in a.stage}
    print("=" * 78)
    print("REVIVE DEAD RUNS  |  psigma %s  |  repeats %s  |  stages %s"
          % (a.psigma, a.repeats, a.stage))
    print("  dead = final HC fraction < %.2f, or an unreadable history" % a.dead_below)
    print("=" * 78)

    # ---- survey ---------------------------------------------------------
    todo, report, tally = {}, [], {}
    for stage in a.stage:
        for psigma in a.psigma:
            for rep in a.repeats:
                prefix = REPEAT_PREFIX[rep]
                dead = []
                for i in range(a.n_arrays):
                    nm = run_name(stage, psigma, prefix, i, a.stress_shift)
                    st, frac = state_of(nm, a.dead_below)
                    # a folder that was never run is not evidence of collapse
                    if st != "missing":
                        t = tally.setdefault((stage, psigma), [0, 0])
                        t[1] += 1
                        t[0] += int(st == "dead")
                    if st != "alive":
                        dead.append((i, nm, st, frac))
                if dead:
                    todo[(stage, psigma, rep)] = dead
                    for i, nm, st, frac in dead:
                        print("  %-6s ps=%.3f r%-2d array%-2d  %-8s hc=%s"
                              % (stage, psigma, rep, i, st,
                                 "%.3f" % frac if np.isfinite(frac) else "unreadable"))

    print("\n  dead fraction per (stage, psigma):   [scope: %s]" % a.dead_scope)
    tripped, by_psigma = set(), {}
    for (stage, psigma), (n_dead_g, n_tot) in tally.items():
        if n_tot and n_dead_g / n_tot >= a.max_dead_fraction:
            tripped.add((stage, psigma))
        by_psigma.setdefault(psigma, []).append(stage)
    # In 'psigma' scope one stage tripping condemns the point for BOTH: the
    # sweep exists to fit the two stages jointly, so a psigma that collapses at
    # either one is not a candidate and reviving the healthier stage there would
    # spend runs on a point that cannot be used.
    skipped = set()
    for stage, psigma in tally:
        if (stage, psigma) in tripped or (
                a.dead_scope == "psigma"
                and any((s, psigma) in tripped for s in by_psigma[psigma])):
            skipped.add((stage, psigma))
    for (stage, psigma), (n_dead_g, n_tot) in sorted(tally.items(), key=lambda kv: str(kv[0])):
        frac = n_dead_g / n_tot if n_tot else 0.0
        note = ""
        if (stage, psigma) in skipped:
            note = ("SKIP (collapse is the measurement here)"
                    if (stage, psigma) in tripped
                    else "SKIP (the other stage collapses at this psigma)")
        print("    %-6s %.3f  %2d/%2d dead = %4.0f%%   %s"
              % (stage, psigma, n_dead_g, n_tot, 100 * frac, note))
    for key in list(todo):
        if (key[0], key[1]) in skipped:
            del todo[key]
    n_dead = sum(len(v) for v in todo.values())
    print("\n  %d run(s) to revive across %d (stage, psigma, repeat) group(s)"
          % (n_dead, len(todo)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was set aside or re-run.")
    if not n_dead:
        print("  nothing dead — no work to do")
        return

    # ---- set aside and re-run -------------------------------------------
    for (stage, psigma, rep), dead in sorted(todo.items(), key=lambda kv: str(kv[0])):
        prefix = REPEAT_PREFIX[rep]
        moved = []
        for i, nm, st, frac in dead:
            if st == "missing":
                continue                                # nothing to move; just re-run
            try:
                moved.append((i, nm, set_aside(nm)))
            except OSError as exc:
                print("    cannot set aside %s (%s); skipping" % (nm[-40:], exc))
        print("\n" + "-" * 78)
        print("  re-running %s psigma %.3f repeat %d (%d array(s))"
              % (stage, psigma, rep, len(dead)), flush=True)
        f = fits[stage]
        # reuse_existing_run keeps the survivors and only the set-aside arrays
        # are actually simulated again
        run_full_model_arrays(
            stage, gammaSC=f["gamma_sc"], gammaHC_ratio=f["R_gamma"],
            alphaHC_ratio=f["R_alpha"],
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=NOTCH_SENSITIVITY,
            repressor_sensitivity=REPRESSOR_SENSITIVITY,
            bending=BENDING, line_tension=LINE_TENSION,
            quasi_static_threshold=g2.BASE_QST, preferred_area=f["A0"],
            psigma=psigma, stress_shift=a.stress_shift,
            initial_notch_delta_level=INITIAL_LI_LEVEL,
            t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
            n_arrays=a.n_arrays, n_workers=a.workers,
            reuse_existing_run=True, name_prefix=prefix)
        for i, nm, old in moved:
            st, frac = state_of(nm, a.dead_below)
            report.append(dict(stage=stage, psigma=psigma, repeat=rep, array=i,
                               name=nm, set_aside_as=old, outcome=st,
                               hc_fraction=None if not np.isfinite(frac) else frac))

    with open(os.path.join(RESULTS_DIR, OUT_JSON), "w") as fh:
        json.dump(report, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("OUTCOME  (a run that dies twice is a result, not a glitch)")
    print("=" * 78)
    print("  %-6s %-8s %7s %8s %9s %9s" % ("stage", "psigma", "repeats",
                                           "died", "revived", "died again"))
    keys = sorted({(r["stage"], r["psigma"]) for r in report})
    for stage, psigma in keys:
        g = [r for r in report if r["stage"] == stage and r["psigma"] == psigma]
        rev = sum(1 for r in g if r["outcome"] == "alive")
        print("  %-6s %-8.3f %7s %8d %9d %9d"
              % (stage, psigma, ",".join(str(r) for r in sorted({x["repeat"] for x in g})),
                 len(g), rev, len(g) - rev))
    print("\n  NOTE: retrying only the deaths censors the collapse rate downwards.")
    print("        Quote the 'died' column when describing how close a psigma is")
    print("        to collapse; use the revived runs only for the scores.")
    print("\nwrote %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
