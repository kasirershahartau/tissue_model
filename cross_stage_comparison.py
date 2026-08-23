"""Do the fitted mechanical parameters HELP the differentiation fit, or hurt it?

    python cross_stage_comparison.py --dry-run
    python cross_stage_comparison.py --workers 4
    python cross_stage_comparison.py --score-only     # skip simulation

All at psigma = 0, so mechanosensitivity plays no part: this isolates whether
the mechanical difference between the stages carries any of the differentiation
difference.

A 2x2 of (initial arrays) x (mechanical parameters), each scored against BOTH
stages' experiments - 8 comparisons:

    arrays   mechanics   scored vs E17.5     scored vs P0
    E17.5    E17.5       native  (have it)   CROSS
    P0       P0          CROSS               native  (have it)
    E17.5    P0          new run             new run
    P0       E17.5       new run             new run

The native diagonal is what every previous score reported. The off-diagonal
answers the real question: if E17.5's mechanics are doing useful work, then
E17.5-arrays + E17.5-mechanics should beat E17.5-arrays + P0-mechanics when
judged against E17.5 data. If the two are indistinguishable, the fitted
mechanics contribute nothing to differentiation; if the crossed one wins, they
actively hurt.

Splitting arrays from mechanics matters because the two stages differ in BOTH:
their initial morphologies are separate sheets, and only gammaSC /
hc_shape_index / sc_shape_index differ mechanically (alphaHC_ratio is 1.0 for
both). Without the cross, an array effect would be indistinguishable from a
mechanics effect.

SCORING. Each comparison uses the fixed-window scoring established by
sweep_scoring_windows.py: t0 is the frame best matching the TARGET stage's
experimental frame 1, and the run is scored as if it ended at t0 + window. t0
therefore depends on which stage you score against, so the t0 cache is keyed by
(target stage, run) - not by run alone.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import post_processing as pp
from post_processing import (RESULTS_DIR, load_history_file, full_model_run_names,
                             initial_morphology_name,
                             _exp_neighbor_pair_percentages,
                             _best_matching_frame_by_neighbor_pairs)
from run_model import run, _history_fingerprint

MECH = {                       # (gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index)
    "E17.5": (0.2461, 1.00, 4.86, 5.72),
    "P0":    (0.2298, 1.00, 5.1487, 5.6706),
}
PREFERRED_AREA = 0.593 * np.pi / 4
ATOH_SENSITIVITY = 0.355079
TYPE_BY, THRESHOLD, MAX_NB = "delta_level", 0.355079, 2
T0_CACHE = "cross_t0_cache.json"
SCORE_CACHE = "cross_score_cache.json"
OUT_JSON = "cross_stage_comparison.json"
SHORT = {"E17.5": "E17", "P0": "P0"}


def _load(f):
    try:
        with open(os.path.join(RESULTS_DIR, f)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _store(f, obj):
    p = os.path.join(RESULTS_DIR, f)
    try:
        tmp = p + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1)
        os.replace(tmp, p)
    except OSError:
        pass


def configs():
    """(label, array_stage, mech_stage, run_names, needs_simulation)."""
    out = []
    for st in ("E17.5", "P0"):
        out.append(("arr%s_mech%s" % (SHORT[st], SHORT[st]), st, st,
                    full_model_run_names(st), False))
    for arr_st, mech_st in (("E17.5", "P0"), ("P0", "E17.5")):
        names = ["crossmodel_mech%s_%s" % (SHORT[mech_st], initial_morphology_name(i, arr_st))
                 for i in range(10)]
        out.append(("arr%s_mech%s" % (SHORT[arr_st], SHORT[mech_st]),
                    arr_st, mech_st, names, True))
    return out


def experimental_target(stage, _c={}):
    if stage in _c:
        return _c[stage]
    pre = SHORT[stage]
    a, b = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(pp.experimental_results_folder, stage,
             "%s_experiment%d_cells_info_frame_1" % (pre, e)))
        cm = np.load(os.path.join(pp.experimental_results_folder, stage,
             "%s_experiment%d_contact_matrix_frame_1.npy" % (pre, e)))
        x, y, _ = _exp_neighbor_pair_percentages(ci, cm)
        a.append(x); b.append(y)
    _c[stage] = (float(np.nanmean(a)), float(np.nanmean(b)))
    return _c[stage]


def get_t0(target_stage, name, cache):
    """t0 depends on WHICH experiment you align to, so key by (stage, run)."""
    key = "%s|%s" % (target_stage, name)
    fp = _history_fingerprint([name])[0]
    hit = cache.get(key)
    if hit is not None and hit.get("fingerprint") == fp:
        return float(hit["t0"])
    tgt = experimental_target(target_stage)
    t0, _, _ = _best_matching_frame_by_neighbor_pairs(
        load_history_file(name), tgt[0], tgt[1], TYPE_BY, THRESHOLD)
    cache[key] = {"fingerprint": fp, "t0": float(t0)}
    _store(T0_CACHE, cache)
    return float(t0)


def score(target_stage, names, t0s, window):
    original = pp.get_time_points

    def capped(history):
        stamps = original(history)
        t0 = t0s.get(os.path.basename(os.path.dirname(str(history.hf5file))))
        if t0 is None:
            return stamps
        keep = stamps[stamps <= t0 + window + 1e-9]
        return keep if keep.size >= 2 else stamps[:2]

    pp.get_time_points = capped
    try:
        return pp.compare_full_model_differentiation_to_experiments(
            target_stage, model_names=names, type_by=TYPE_BY,
            threshold=THRESHOLD, max_number_of_neighbors=MAX_NB)
    finally:
        pp.get_time_points = original


def one_sim(args):
    arr_stage, mech_stage, i, t_end, save_interval = args
    gammaSC, alphaHC, hc_p0, sc_p0 = MECH[mech_stage]
    initial = initial_morphology_name(i, arr_stage)
    name = "crossmodel_mech%s_%s" % (SHORT[mech_stage], initial)
    try:
        run(gammaSC, 1.0, alphaHC, 0.0, initial_sheet_name=initial, name=name,
            no_differentiation=False, stress_dependent=False,
            end_on_steady_state=True, t_end=t_end, dt=0.01, divisions=False,
            shape_index=0.0, hc_shape_index=hc_p0, sc_shape_index=sc_p0,
            bending=0.02, quasi_static_threshold=0.03,
            preferred_area_override=PREFERRED_AREA, save_interval=save_interval,
            atoh_sensitivity=ATOH_SENSITIVITY, notch_sensitivity=0.1,
            repressor_sensitivity=0.3,
            randomize_notch_delta_levels=True, initial_notch_delta_level=0.01,
            reuse_existing_run=True)
        return name, None
    except Exception as exc:  # noqa: BLE001
        return name, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=float, nargs="+", default=[2, 5, 10])
    ap.add_argument("--t-end", dest="t_end", type=float, default=30)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfgs = configs()
    todo = [(c[1], c[2], i, a.t_end, a.save_interval)
            for c in cfgs if c[4]
            for i in range(10)
            if not os.path.isdir(os.path.join(RESULTS_DIR, c[3][i]))]
    print("cross-stage comparison (psigma=0) | %d config(s) x 2 target stage(s) "
          "x %d window(s)" % (len(cfgs), len(a.windows)))
    for label, arr, mech, names, new in cfgs:
        have = sum(os.path.isdir(os.path.join(RESULTS_DIR, n)) for n in names)
        print("  %-16s arrays=%-6s mechanics=%-6s  %d/10 run folders present%s"
              % (label, arr, mech, have, "  (NEW)" if new else ""))
    print("  -> %d simulation(s) needed" % len(todo))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was run.")

    if todo and not a.score_only:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(one_sim, t) for t in todo]
            for n, f in enumerate(as_completed(futs), 1):
                nm, err = f.result()
                print("  [%2d/%2d] %-52s %s" % (n, len(todo), nm[:52],
                      "OK" if err is None else "FAILED " + err), flush=True)

    t0c, scc, out = _load(T0_CACHE), _load(SCORE_CACHE), {}
    for label, arr, mech, names, _new in cfgs:
        present = [n for n in names if os.path.isdir(os.path.join(RESULTS_DIR, n))]
        if len(present) < len(names):
            print("\n%s: only %d/%d runs present - skipping"
                  % (label, len(present), len(names)))
            continue
        for target in ("E17.5", "P0"):
            t0s = {n: get_t0(target, n, t0c) for n in names}
            fp = _history_fingerprint(names)
            for w in a.windows:
                key = "%s|vs=%s|w=%.3f" % (label, target, w)
                hit = scc.get(key)
                if hit is not None and hit.get("fingerprint") == fp:
                    res = hit["result"]
                else:
                    res = score(target, names, t0s, float(w))
                    scc[key] = {"fingerprint": fp, "result": res}
                    _store(SCORE_CACHE, scc)
                out.setdefault(label, {}).setdefault(target, {})["%.3f" % w] = res
                print("  %-16s vs %-6s w=%-4g  s1=%7.2f s2=%8.2f s3=%9.2f  "
                      "s1+s2=%8.2f  TOTAL=%9.2f"
                      % (label, target, w, res["score1"], res["score2"],
                         res["score3"], res["score1"] + res["score2"], res["total"]),
                      flush=True)
    _store(OUT_JSON, out)

    print("\n" + "=" * 92)
    print("s1+s2   (rows = arrays x mechanics, cols = experiment scored against)")
    print("=" * 92)
    for w in a.windows:
        print("\n--- window %g ---" % w)
        print("  %-18s %14s %14s" % ("arrays x mechanics", "vs E17.5", "vs P0"))
        for label, arr, mech, _n, _new in cfgs:
            row = "  %-18s" % label
            for target in ("E17.5", "P0"):
                r = out.get(label, {}).get(target, {}).get("%.3f" % w)
                row += "%14s" % ("-" if r is None else
                                 "%.2f" % (r["score1"] + r["score2"]))
            print(row)
        # the comparison the experiment is actually about
        for target in ("E17.5", "P0"):
            native = "arr%s_mech%s" % (SHORT[target], SHORT[target])
            other = "P0" if target == "E17.5" else "E17.5"
            crossed = "arr%s_mech%s" % (SHORT[target], SHORT[other])
            n_r = out.get(native, {}).get(target, {}).get("%.3f" % w)
            c_r = out.get(crossed, {}).get(target, {}).get("%.3f" % w)
            if n_r and c_r:
                nv = n_r["score1"] + n_r["score2"]
                cv = c_r["score1"] + c_r["score2"]
                verdict = ("own mechanics BETTER" if nv < cv else
                           "own mechanics WORSE" if nv > cv else "tie")
                print("    vs %-6s: same arrays, own mechanics %.2f vs other's %.2f"
                      "  -> %s" % (target, nv, cv, verdict))
    print("\nwrote %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
