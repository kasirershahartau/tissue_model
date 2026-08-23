"""Score every psigma over a range of SCORING WINDOWS, and cache everything.

    python sweep_scoring_windows.py --dry-run
    python sweep_scoring_windows.py
    python sweep_scoring_windows.py --windows 2 10 30 --psigma 0 0.06

THE QUESTION. Can the model reproduce the E17.5-vs-P0 difference in
differentiation pattern from the mechanical difference alone? Every comparison
so far has been confounded by WHEN the model was scored: runs were scored at
their last recorded frame, so points that ran longer differentiated more, and
the apparent benefit of psigma shrank as t_end grew (4.3x at t=50, 2.1x at
t=100). Neither the Notch-inhibition experiment nor the measured differentiation
percentages can pin the window down, so instead of guessing it, sweep it.

WHAT A WINDOW MEANS. t0 is the frame whose neighbour-pair composition best
matches the experimental frame 1 (this is score 1, unchanged). A window W scores
as if the run had ENDED at t0 + W: scores 2 and 3 both read "the final frame",
so capping the frames at t0 + W makes that final frame a fixed distance past t0,
identically for every psigma. That removes the truncation bias by construction -
all points are compared over the same amount of simulated development.

HOW. post_processing's three scoring functions
(_best_matching_frame_by_neighbor_pairs, calc_HC_neighbors_at_differentiation,
calc_percentage_of_differentiating_by_initial_neighbors) all enumerate frames
through get_time_points and none touches history.time_stamps directly, so
capping THAT function is an exact stand-in for a shorter run - every sheet
returned is bit-identical to the real one.

Capping does not disturb t0: t0 is the argmin over all frames, and it lies
inside [0, t0 + W], so it stays the argmin of the capped set.

CACHING (two layers, both keyed on a size+mtime fingerprint of the runs, so a
re-simulated or extended run invalidates itself automatically):
    <results>/t0_cache.json             per-run t0 - the expensive full scan
    <results>/window_score_cache.json   per (stage, psigma, K, window) score
Results are also written incrementally to <results>/scoring_window_sweep.json
after EVERY point, so an interrupted sweep keeps everything it finished.
"""
import argparse
import json
import os

import numpy as np

import post_processing as pp
from post_processing import (RESULTS_DIR, load_history_file, full_model_run_names,
                             _exp_neighbor_pair_percentages,
                             _best_matching_frame_by_neighbor_pairs)
from run_model import _psigma_tag, _history_fingerprint
import pandas as pd

MECH_STAGES = ("E17.5", "P0")
PSIGMAS = [0.0, 0.015, 0.030, 0.045, 0.060, 0.075]
WINDOWS = [2, 5, 10, 15, 20, 25, 30, 40, 50]
STRESS_SHIFT = -0.080
TYPE_BY, THRESHOLD, MAX_NB = "delta_level", 0.355079, 2

T0_CACHE = "t0_cache.json"
SCORE_CACHE = "window_score_cache.json"
OUT_JSON = "scoring_window_sweep.json"


# --------------------------------------------------------------------------- #
def _load(fname):
    try:
        with open(os.path.join(RESULTS_DIR, fname)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _store(fname, obj):
    path = os.path.join(RESULTS_DIR, fname)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def run_names(stage, psigma):
    if float(psigma) == 0.0:
        return full_model_run_names(stage)
    suffix = "E17" if stage == "E17.5" else "P0"
    return ["fullmodel_ps%s_ks%.3f_random_periodic_array%d_for_%s"
            % (_psigma_tag(float(psigma)), STRESS_SHIFT, i, suffix) for i in range(10)]


def experimental_target(stage, _cache={}):
    if stage in _cache:
        return _cache[stage]
    prefix = "E17" if stage == "E17.5" else "P0"
    a, b = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(pp.experimental_results_folder, stage,
             "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(pp.experimental_results_folder, stage,
             "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        x, y, _ = _exp_neighbor_pair_percentages(ci, cm)
        a.append(x); b.append(y)
    _cache[stage] = (float(np.nanmean(a)), float(np.nanmean(b)))
    return _cache[stage]


def get_t0(stage, name, cache):
    """t0 for one run, cached against its history fingerprint."""
    fp = _history_fingerprint([name])[0]
    hit = cache.get(name)
    if hit is not None and hit.get("fingerprint") == fp:
        return float(hit["t0"])
    tgt = experimental_target(stage)
    history = load_history_file(name)
    t0, _, _ = _best_matching_frame_by_neighbor_pairs(
        history, tgt[0], tgt[1], TYPE_BY, THRESHOLD)
    cache[name] = {"fingerprint": fp, "t0": float(t0)}
    _store(T0_CACHE, cache)
    return float(t0)


def score_with_window(stage, names, t0_by_name, window):
    """Score as if every run had ended at its own t0 + window."""
    original = pp.get_time_points

    def capped(history):
        stamps = original(history)
        name = os.path.basename(os.path.dirname(str(history.hf5file)))
        t0 = t0_by_name.get(name)
        if t0 is None:
            return stamps
        keep = stamps[stamps <= t0 + window + 1e-9]
        # Never hand back an empty/1-frame set: below t0 the scores are undefined.
        return keep if keep.size >= 2 else stamps[:2]

    pp.get_time_points = capped
    try:
        return pp.compare_full_model_differentiation_to_experiments(
            stage, model_names=names, type_by=TYPE_BY, threshold=THRESHOLD,
            max_number_of_neighbors=MAX_NB)
    finally:
        pp.get_time_points = original


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psigma", type=float, nargs="+", default=PSIGMAS)
    ap.add_argument("--windows", type=float, nargs="+", default=WINDOWS)
    ap.add_argument("--stages", nargs="+", default=list(MECH_STAGES))
    ap.add_argument("--stress-shift", dest="stress_shift", type=float, default=STRESS_SHIFT)
    ap.add_argument("--no-cache", action="store_true", help="recompute everything")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    globals()["STRESS_SHIFT"] = a.stress_shift

    t0_cache = {} if a.no_cache else _load(T0_CACHE)
    sc_cache = {} if a.no_cache else _load(SCORE_CACHE)
    results = _load(OUT_JSON) if not a.no_cache else {}
    results.setdefault("stress_shift", a.stress_shift)
    results.setdefault("scores", {})

    print("scoring-window sweep | psigma %s | windows %s | K=%.3f"
          % (a.psigma, a.windows, a.stress_shift), flush=True)
    plan = []
    for stage in a.stages:
        for ps in a.psigma:
            names = run_names(stage, ps)
            missing = [n for n in names if not os.path.isdir(os.path.join(RESULTS_DIR, n))]
            if missing:
                print("  SKIP %-6s psigma=%.3f : %d/%d run folders missing"
                      % (stage, ps, len(missing), len(names)))
                continue
            plan.append((stage, ps, names))
    print("  %d (stage, psigma) combination(s) x %d window(s) = %d score(s)"
          % (len(plan), len(a.windows), len(plan) * len(a.windows)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was computed.")

    for stage, ps, names in plan:
        print("\n=== %s  psigma=%.3f ===" % (stage, ps), flush=True)
        t0s = {}
        for n in names:
            t0s[n] = get_t0(stage, n, t0_cache)
        print("  t0: mean %.2f  (range %.2f - %.2f)"
              % (np.mean(list(t0s.values())), min(t0s.values()), max(t0s.values())),
              flush=True)
        fp = _history_fingerprint(names)
        for w in a.windows:
            key = "%s|ps=%.5f|ks=%.4f|w=%.3f" % (stage, ps, a.stress_shift, w)
            hit = sc_cache.get(key)
            if hit is not None and hit.get("fingerprint") == fp:
                res = hit["result"]
                print("  window %-5g cached   total=%9.3f  (s1=%.2f s2=%.2f s3=%.2f)"
                      % (w, res["total"], res["score1"], res["score2"], res["score3"]),
                      flush=True)
            else:
                res = score_with_window(stage, names, t0s, float(w))
                sc_cache[key] = {"fingerprint": fp, "result": res}
                _store(SCORE_CACHE, sc_cache)
                print("  window %-5g COMPUTED total=%9.3f  (s1=%.2f s2=%.2f s3=%.2f)"
                      % (w, res["total"], res["score1"], res["score2"], res["score3"]),
                      flush=True)
            results["scores"].setdefault(stage, {}).setdefault("%.5f" % ps,
                                                               {})["%.3f" % w] = res
            _store(OUT_JSON, results)

    # ----- report ---------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("TOTAL SCORE  (rows = scoring window, cols = psigma)")
    print("=" * 78)
    for stage in a.stages:
        got = results["scores"].get(stage, {})
        if not got:
            continue
        pss = sorted(got, key=float)
        print("\n=== %s ===" % stage)
        print("  %-8s %s" % ("window", "".join("%11s" % ("ps=%.3f" % float(p)) for p in pss)))
        for w in a.windows:
            row = "  %-8g" % w
            for p in pss:
                r = got[p].get("%.3f" % w)
                row += "%11s" % ("-" if r is None else "%.1f" % r["total"])
            print(row)
        print("  best psigma per window:")
        for w in a.windows:
            vals = [(float(p), got[p]["%.3f" % w]["total"])
                    for p in pss if got[p].get("%.3f" % w)]
            if vals:
                best = min(vals, key=lambda kv: kv[1])
                print("    window %-5g -> psigma=%.3f (total %.1f);  psigma=0 gives %.1f"
                      % (w, best[0], best[1], dict(vals).get(0.0, float("nan"))))
    print("\nresults: %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
