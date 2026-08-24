"""Find a STRICT steady state: a window with ZERO differentiation events.

    python strict_steady_state.py --dry-run
    python strict_steady_state.py --workers 4
    python strict_steady_state.py --analyse-only     # rescan existing runs

THE POINT. The solver's steady-state criterion passes runs whose lateral
inhibition is still turning over: continued 30 time units with nothing ablated,
psigma = 0 drifts -0.4% in HC fraction (rate decaying to 5e-05) but psigma =
0.162 drifts -1.4% with the rate PLATEAUING at 3-7e-04. So the ablation controls
produced 2.43 and 10.57 background events per run — differentiation that would
have happened anyway. A strict steady state removes that: if the unablated tissue
produces no events in a window, every event after an ablation is caused by it,
and no control run is needed.

HOW. Rather than fork-check-fork-check in 5-unit steps, each array is continued
ONCE for --max-extra time units and the trajectory is scanned offline for the
first window of --window units containing zero differentiation events. That is
equivalent to the stepwise procedure, costs one run instead of up to twenty, and
yields the whole event timeline instead of just the verdict.

An array with no clean window by --max-extra is FLAGGED as failed rather than
quietly used. Expect that at psigma = 0.162: its drift rate does not decay, so a
genuinely event-free window may not exist.

A differentiation event is a cell below the delta threshold at the window start
and above it at some later frame in that window, tracked by ``id`` (``unique_id``
is recompacted whenever a face is removed).

Output: strict_steady_state.json — per array, the strict time t*, the event
timeline, and whether it converged. ablate_from_strict.py consumes that.
"""
import argparse
import json
import os

import numpy as np

from post_processing import RESULTS_DIR, load_history_file, get_time_points
from run_model import run
from ablate_single_hc import (MECH, SHAPE_INDEX, BENDING, ATOH_SENSITIVITY,
                              TYPE_BY, THRESHOLD)
from run_psigma_repeats import REPEAT_PREFIX
from score_psigma_pooled import run_name

OUT_JSON = "strict_steady_state.json"


def fork_name(stage, psigma, i):
    suffix = "E17" if stage == "E17.5" else "P0"
    return "strictss_ps%.3f_array%d_for_%s" % (psigma, i, suffix)


def one_run(args):
    stage, i, psigma, max_extra, save_interval, dry = args
    src = run_name(stage, psigma, REPEAT_PREFIX[1], i)
    name = fork_name(stage, psigma, i)
    rec = dict(stage=stage, array=i, psigma=psigma, source=src, folder=name, error=None)
    try:
        t_last = float(np.max(get_time_points(load_history_file(src))))
        rec["source_t_end"] = t_last
        if dry:
            return rec
        gammaSC, R_gamma, R_alpha, A0 = MECH[stage]
        run(gammaSC, R_gamma, R_alpha, psigma,
            initial_sheet_name=src, continue_from_time=t_last,
            continue_existing_run=False,           # fork; leaves the source intact
            randomize_notch_delta_levels=False,    # keep the evolved LI state
            stress_dependent=(float(psigma) != 0.0),
            ablated_cells=[], name=name,
            t_end=max_extra, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,             # we decide when it is steady
            max_wall_seconds=14400, min_progress_rate=1e-4,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX, bending=BENDING,
            quasi_static_threshold=0.03, preferred_area_override=A0,
            reuse_existing_run=True)
        return rec
    except Exception as exc:                       # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        return rec


def event_times(folder):
    """[(time, n_new_HC)] — cells crossing the threshold upward, by ``id``."""
    history = load_history_file(folder)
    stamps = np.asarray(get_time_points(history), float)
    seen, out, prev = {}, [], None
    for t in stamps:
        s = history.retrieve(float(t))
        s.arrange_sheet_from_history()
        ids = s.face_df["id"].to_numpy(int)
        delta = s.face_df[TYPE_BY].to_numpy(float)
        cur = dict(zip(ids.tolist(), delta.tolist()))
        if prev is not None:
            n = 0
            for cid, d in cur.items():
                if d > THRESHOLD and prev.get(cid, 1.0) <= THRESHOLD and cid not in seen:
                    seen[cid] = float(t); n += 1
            if n:
                out.append((float(t), n))
        prev = cur
    return out, float(stamps[-1])


def window_counts(events, t_end, window):
    """Differentiation events per consecutive window: [(t_start, n), ...].

    The full profile, not just the first quiet window — it shows whether events
    are thinning towards zero (a convergent tail) or holding a steady rate (the
    tissue is still turning over and no strict steady state exists)."""
    edges = np.arange(0.0, t_end + window, window)
    out = []
    for lo in edges[:-1]:
        hi = lo + window
        out.append((float(lo), int(sum(n for t, n in events if lo < t <= hi))))
    return out


def first_quiet_window(events, t_end, window):
    """Earliest t with no event in (t, t+window]. None if there is none."""
    times = [e[0] for e in events]
    t = 0.0
    while t + window <= t_end + 1e-9:
        if not any(t < x <= t + window for x in times):
            return t
        # jump past the offending event rather than crawling
        t = min(x for x in times if x > t)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["P0"], choices=["E17.5", "P0"])
    ap.add_argument("--psigma", type=float, nargs="+", default=[0.0, 0.162])
    ap.add_argument("--arrays", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--window", type=float, default=5.0,
                    help="a window this long with zero events = strict steady state")
    ap.add_argument("--max-extra", dest="max_extra", type=float, default=100.0,
                    help="give up after this many extra time units and flag failed")
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", dest="analyse_only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [(st, i, ps, a.max_extra, a.save_interval, a.dry_run)
             for st in a.stage for ps in a.psigma for i in a.arrays]
    print("=" * 78)
    print("STRICT STEADY STATE | %s | psigma %s | %d run(s) x %g time units"
          % (", ".join(a.stage), a.psigma, len(tasks), a.max_extra))
    print("=" * 78)
    print("  criterion: no differentiation event in any %g-unit window" % a.window)
    free = None
    try:
        import shutil
        free = shutil.disk_usage(RESULTS_DIR).free / 1e9
        print("  disk free: %.1f GB" % free)
    except Exception:
        pass
    if a.dry_run:
        for t in tasks[:6]:
            r = one_run(t)
            print("  %-6s ps=%-6.3f array%-2d fork %s from t=%.2f"
                  % (r["stage"], r["psigma"], r["array"], r["folder"],
                     r.get("source_t_end", float("nan"))))
        print("  ... %d total" % len(tasks))
        raise SystemExit("\n--dry-run: nothing was run.")

    if a.analyse_only:
        recs = [one_run(t[:5] + (True,)) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        recs, per_pool = [], max(1, 3 * a.workers)
        for s in range(0, len(tasks), per_pool):    # recycled pool, see run_task_pool
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                recs.extend(ex.map(one_run, tasks[s:s + per_pool]))
            try:
                import shutil
                print("  -- pool done; disk free %.1f GB --"
                      % (shutil.disk_usage(RESULTS_DIR).free / 1e9), flush=True)
            except Exception:
                pass

    out = []
    for rec in recs:
        if rec.get("error"):
            out.append(rec); continue
        try:
            ev, t_end = event_times(rec["folder"])
            t_star = first_quiet_window(ev, t_end, a.window)
            rec = dict(rec, n_events=sum(n for _t, n in ev), events=ev,
                       t_end=t_end, t_strict=t_star, converged=t_star is not None,
                       window_counts=window_counts(ev, t_end, a.window))
        except Exception as exc:                    # noqa: BLE001
            rec = dict(rec, error="scan: %s: %s" % (type(exc).__name__, exc))
        out.append(rec)
    with open(os.path.join(RESULTS_DIR, OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("STRICT STEADY STATE per array")
    print("=" * 78)
    print("  %-6s %-8s %6s %8s %9s %10s %s"
          % ("stage", "psigma", "array", "events", "t_end", "t_strict", "verdict"))
    for ps in a.psigma:
        ok = 0
        for r in out:
            if r.get("psigma") != ps:
                continue
            if r.get("error"):
                print("  %-6s %-8.3f %6d   %s" % (r["stage"], ps, r["array"], r["error"][:50]))
                continue
            good = r.get("converged")
            ok += bool(good)
            print("  %-6s %-8.3f %6d %8d %9.1f %10s %s"
                  % (r["stage"], ps, r["array"], r["n_events"], r["t_end"],
                     "%.1f" % r["t_strict"] if good else "-",
                     "strict" if good else "FAILED (no quiet window)"))
        print("    -> psigma %.3f: %d of %d arrays reached a strict steady state"
              % (ps, ok, len([r for r in out if r.get("psigma") == ps])))
        rows = [r for r in out if r.get("psigma") == ps and r.get("window_counts")]
        if rows:
            n_win = max(len(r["window_counts"]) for r in rows)
            tot = [0] * n_win
            for r in rows:
                for k, (_lo, n) in enumerate(r["window_counts"]):
                    tot[k] += n
            print("    events per %g-unit window, summed over %d array(s):"
                  % (a.window, len(rows)))
            for k in range(0, n_win, 4):
                chunk = ["%5.0f-%-5.0f %4d"
                         % (k2 * a.window, (k2 + 1) * a.window, tot[k2])
                         for k2 in range(k, min(k + 4, n_win))]
                print("      " + "   ".join(chunk))
        print()
    try:
        import shutil
        print("  disk free now: %.1f GB" % (shutil.disk_usage(RESULTS_DIR).free / 1e9))
    except Exception:
        pass
    print("wrote %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
