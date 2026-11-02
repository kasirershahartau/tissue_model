"""Re-score an existing psigma sweep, grouping each array's REPEATS.

    python rescore_psigma.py --workers 46
    python rescore_psigma.py --psigma 0.08667 0.09 --workers 8   # just these

WHY THIS EXISTS. ``compare_full_model_differentiation_to_experiments`` takes
either ``model_names`` (every run an independent data point) or
``model_groups`` (a list of per-ARRAY groups whose repeats are averaged into
one point first). Repeats re-roll the lateral-inhibition seed on the SAME
initial sheet, so they are repeated measurements of one tissue, not samples of
tissue-to-tissue variation. Scored flat, SEM_sim shrinks by ~sqrt(n_repeats)
and every chi^2 inflates by up to that squared -- the first cloud sweep came
back about 3x worse than the archived numbers from this alone.

``find_psigma`` passed a flat list until it was fixed; this re-scores the runs
that sweep already produced, so NO SIMULATION IS REPEATED. It reads the run
folders, never writes to them.

WHAT IT WRITES, into TISSUE_RESULTS_DIR:
    psigma_scores_<tag>_grouped.json   same shape as find_psigma's own json
and prints the per-point table, the best point, and -- when the sweep's
original json is present -- an old/new column so the inflation is visible.

The pT = 0 baseline is included automatically if its folders are there (they
carry no _ps tag). On the VM they usually are not: score it where it lives and
merge afterwards.
"""
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             compare_full_model_differentiation_to_experiments)
from run_model import _psigma_tag
from full_model import REPEAT_PREFIX, DEFAULT_OBJECTIVE_TERMS

STAGES = ("E17.5", "P0")
DEFAULT_THRESHOLD = 0.355079


def run_name(stage, psigma, prefix, i, stress_shift=0.0):
    """Folder name as _run_full_model_one builds it (psigma=0 carries no tag)."""
    init = initial_morphology_name(i, stage)
    if float(psigma) == 0.0:
        return "%s_%s" % (prefix, init)
    return "%s_ps%s_ks%.3f_%s" % (prefix, _psigma_tag(psigma), stress_shift, init)


def discover_psigma(stress_shift=0.0):
    """Every psigma with at least one run folder on disk, 0.0 included when the
    untagged baseline folders are present."""
    found = set()
    pat = re.compile(r"_ps([0-9.]+)_ks%.3f_" % stress_shift)
    for name in os.listdir(RESULTS_DIR):
        m = pat.search(name)
        if m:
            found.add(round(float(m.group(1)), 5))
        elif re.match(r"^fullmodel_v2(r\d+)?_random_periodic_array\d+_for_",
                      name):
            found.add(0.0)
    return sorted(found)


def groups_for(stage, psigma, repeats, n_arrays, stress_shift=0.0):
    """[[array 0's repeats], [array 1's repeats], ...], keeping what exists."""
    groups, missing = [], 0
    for i in range(n_arrays):
        grp = [run_name(stage, psigma, REPEAT_PREFIX[r], i, stress_shift)
               for r in repeats]
        here = [n for n in grp
                if os.path.isfile(os.path.join(RESULTS_DIR, n, "history.hf5"))]
        missing += len(grp) - len(here)
        if here:
            groups.append(here)
    return groups, missing


def _score_one(args):
    stage, psigma, groups, type_by, threshold, max_nb = args
    try:
        res = compare_full_model_differentiation_to_experiments(
            stage, model_groups=groups, type_by=type_by, threshold=threshold,
            max_number_of_neighbors=max_nb)
        return (stage, psigma, res, None)
    except Exception as exc:                                    # noqa: BLE001
        return (stage, psigma, None, "%s: %s" % (type(exc).__name__, exc))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psigma", type=float, nargs="+", default=None,
                    help="values to score (default: every one found on disk)")
    ap.add_argument("--stage", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=0.0)
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--max-neighbors", dest="max_nb", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--terms", nargs="+", default=list(DEFAULT_OBJECTIVE_TERMS),
                    choices=["score1", "score2", "score3"],
                    help="which sub-scores the OBJECTIVE sums (default "
                         "score1 score2 -- the four terms per stage the "
                         "campaign is fitted on). All three are always "
                         "computed and printed; score3 dwarfs the others "
                         "(79%% of the 7-term total at the s1+s2 optimum), so "
                         "including it changes which psigma wins.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    psigmas = a.psigma if a.psigma is not None else discover_psigma(a.stress_shift)
    if not psigmas:
        raise SystemExit("no run folders found under %s" % RESULTS_DIR)

    print("re-scoring %d psigma value(s) x %d stage(s), repeats %s, from %s"
          % (len(psigmas), len(a.stage), a.repeats, RESULTS_DIR), flush=True)

    tasks, skipped = [], []
    for psigma in psigmas:
        for stage in a.stage:
            groups, missing = groups_for(stage, psigma, a.repeats, a.n_arrays,
                                         a.stress_shift)
            if not groups:
                skipped.append((stage, psigma, "no runs"))
                continue
            if missing:
                print("  note: %s psigma=%.5g is missing %d of %d runs"
                      % (stage, psigma, missing, a.n_arrays * len(a.repeats)),
                      flush=True)
            tasks.append((stage, psigma, groups, a.type_by, a.threshold, a.max_nb))

    n_workers = max(1, min(a.workers, len(tasks)))
    print("  %d point(s) to score on %d worker(s)" % (len(tasks), n_workers),
          flush=True)
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            done = list(pool.map(_score_one, tasks))
    else:
        done = [_score_one(t) for t in tasks]

    scores, failed = {}, []
    for stage, psigma, res, err in done:
        if err:
            failed.append((stage, psigma, err))
            continue
        scores.setdefault(psigma, {})[stage] = res
    for psigma, entry in scores.items():
        entry["objective"] = float(sum(entry[s][t] for s in a.stage
                                       if s in entry for t in a.terms))
        entry["all_terms_total"] = float(sum(entry[s]["total"] for s in a.stage
                                             if s in entry))

    tag = "_".join(a.stage) + "_ks%.3f" % a.stress_shift

    print("\n" + "=" * 86)
    print("GROUPED scores (repeats of an array averaged into one point)")
    print("  objective = %s, summed over stages; all-terms total shown too"
          % " + ".join(a.terms))
    print("=" * 86)
    print("  %-9s %10s %10s %12s %12s"
          % ("psigma", *a.stage[:2], "objective", "all terms"))
    for psigma in sorted(scores):
        e = scores[psigma]
        cells = ["%10.3f" % sum(e[s][t] for t in a.terms) if s in e
                 else "%10s" % "-" for s in a.stage[:2]]
        print("  %-9.5g %s %12.3f %12.3f"
              % (psigma, " ".join(cells), e["objective"], e["all_terms_total"]))

    if scores:
        best = min(scores, key=lambda k: scores[k]["objective"])
        print("\n  BEST psigma = %.5g   objective (%s) = %.3f   [%s]"
              % (best, "+".join(a.terms), scores[best]["objective"],
                 ", ".join("%s=%.1f"
                           % (s, sum(scores[best][s][t] for t in a.terms))
                           for s in a.stage if s in scores[best])))
        if 0.0 in scores:
            base = scores[0.0]["objective"]
            verdict = "IMPROVES" if scores[best]["objective"] < base else "does NOT improve"
            print("  vs the pT = 0 baseline %.3f: mechanosensitivity %s the fit"
                  % (base, verdict))
        else:
            print("  (no pT = 0 baseline on this machine -- merge it where it lives)")

    for stage, psigma, why in skipped:
        print("  skipped %s psigma=%.5g: %s" % (stage, psigma, why))
    for stage, psigma, err in failed:
        print("  FAILED  %s psigma=%.5g: %s" % (stage, psigma, err))

    out = a.out or os.path.join(RESULTS_DIR,
                                "psigma_scores_%s_grouped.json" % tag)
    with open(out, "w") as fh:
        json.dump({"stages": list(a.stage), "stress_shift": a.stress_shift,
                   "repeats": a.repeats, "n_arrays": a.n_arrays,
                   "grouping": "per-array, repeats averaged",
                   "objective_terms": list(a.terms),
                   "scores": {("%.5f" % k): v for k, v in scores.items()}},
                  fh, indent=1)
    print("\n  wrote %s" % out)


if __name__ == "__main__":
    main()
