"""Repair runs that were resumed BEFORE the LI-preservation fix.

Background
----------
Before the fix, a resume (``continue_existing_run=True``, e.g. via the fit's
``rerun_stalled_runs`` path) RE-SEEDED the lateral-inhibition levels
(notch/delta/repressor) from the initial-sheet ``*_levels.npy`` arrays, instead
of preserving the levels already evolved in the loaded archive. When the
archive's LI differed from those arrays, every cell's delta changed at the
resume frame -> ``atoh = increasing_hill(delta)`` changed -> roughly half the
cells flipped HC<->SC mid-trajectory. The fix makes a resume PRESERVE the
archive's LI (see ``run_model.run``: the seed arrays are forced to ``None`` when
``continue_existing_run`` is True).

What this script does
---------------------
Finds every run that was resumed (has a ``parameters_continue*.txt``) and
re-resumes it from the LAST CLEAN snapshot BEFORE the first resume time. That
matters: the snapshot AT the recorded ``continue_from_time`` was itself
overwritten by the corrupting resume, so continuing from it would just preserve
the corruption. Continuing from the previous snapshot truncates the whole
corrupted tail and replaces it with a clean, LI-consistent continuation (the
fixed code preserves the archive's original LI at that point).

Idempotent: a ``.li_reseed_fixed`` marker is written on success and such runs
are skipped on re-runs.

Usage
-----
    # DRY RUN — just list the runs that would be repaired:
    python fix_corrupted_resumed_runs.py

    # actually re-resume them (long-running: each re-simulates to t_end):
    python fix_corrupted_resumed_runs.py --fix

    # bound each run's wall time / require progress so a re-stall bails:
    python fix_corrupted_resumed_runs.py --fix --max-wall-seconds 1800 --min-progress-rate 1e-3
"""
import os
import sys
import ast
import argparse
import inspect

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_processing import RESULTS_DIR
from run_model import run, _STALL_SIGNATURES
from tyssue import HistoryHdf5
from virtual_sheet import VirtualSheet

MARKER = ".li_reseed_fixed"


def _parse_parameters(path):
    """Parse a run() parameter dump (``var: repr(value)`` per line) into a dict,
    keeping only the literal-eval-able values (floats, bools, ints, lists,
    strings, None) — enough to reconstruct the run() call."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            key, sep, val = line.partition(":")
            if not sep:
                continue
            try:
                out[key.strip()] = ast.literal_eval(val.strip())
            except (ValueError, SyntaxError):
                pass  # non-literal (objects / arrays) — run() default is used
    return out


def _first_resume_time(folder):
    """The ``continue_from_time`` recorded by the FIRST resume, or None."""
    p = os.path.join(folder, "parameters_continue1.txt")
    if not os.path.isfile(p):
        return None
    return _parse_parameters(p).get("continue_from_time")


def _last_snapshot_before(folder, t):
    """Largest archive snapshot time strictly < ``t`` — the last frame recorded
    by the ORIGINAL run before the corrupting resume overwrote from ``t`` on.
    None if the archive is missing or has nothing before ``t``."""
    arch = os.path.join(folder, "history.hf5")
    if not os.path.isfile(arch):
        return None
    hist = HistoryHdf5.from_archive(arch, eptm_class=VirtualSheet)
    ts = np.asarray(hist.time_stamps, float)
    clean = ts[ts < t - 1e-9]
    return float(clean.max()) if len(clean) else None


def find_resumed_runs(results_dir):
    """Every run folder that was resumed (has parameters_continue1.txt) and is
    not yet marked fixed. Returns a list of dicts with the resume time and the
    clean time to re-resume from."""
    found = []
    for name in sorted(os.listdir(results_dir)):
        folder = os.path.join(results_dir, name)
        if not os.path.isdir(folder):
            continue
        if not os.path.isfile(os.path.join(folder, "parameters_continue1.txt")):
            continue
        if os.path.isfile(os.path.join(folder, MARKER)):
            continue
        t_resume = _first_resume_time(folder)
        t_safe = _last_snapshot_before(folder, t_resume) if t_resume is not None else None
        found.append({"name": name, "folder": folder,
                      "resume_time": t_resume, "resume_from": t_safe})
    return found


def _rebuild_run_kwargs(folder):
    """Reconstruct the run() call from the ORIGINAL ``parameters.txt``: take every
    value whose key matches a run() parameter, then override to cleanly resume
    the run's OWN archive."""
    params = _parse_parameters(os.path.join(folder, "parameters.txt"))
    accepted = set(inspect.signature(run).parameters)
    kwargs = {k: v for k, v in params.items() if k in accepted}
    bare = os.path.basename(folder)
    kwargs["name"] = bare              # write back into the same folder
    kwargs["initial_sheet_name"] = bare  # load the run's OWN archive
    kwargs["continue_existing_run"] = True
    # Keep the fit's reuse/rerun-stalled dispatch out of the way — we resume
    # explicitly.
    kwargs.pop("reuse_existing_run", None)
    kwargs.pop("rerun_stalled_runs", None)
    return kwargs


def fix_run(info, max_wall_seconds=None, min_progress_rate=None):
    """Re-resume one run from its last clean snapshot with the fixed code.

    Returns ``"completed"`` if the re-resume ran to the end, or ``"stalled"`` if
    it bailed on the non-progress guard. A stall is NOT a failure: the archive
    is now LI-consistent (no mid-run flip) up to the stall — the honest result
    for a parameter region that genuinely stalls (these runs were resumed in the
    first place BECAUSE they stalled). Either way a marker is written so the run
    isn't reprocessed. A non-stall exception propagates."""
    kwargs = _rebuild_run_kwargs(info["folder"])
    kwargs["continue_from_time"] = info["resume_from"]
    if max_wall_seconds is not None:
        kwargs["max_wall_seconds"] = max_wall_seconds
    if min_progress_rate is not None:
        kwargs["min_progress_rate"] = min_progress_rate
    outcome = "completed"
    try:
        run(**kwargs)
    except RuntimeError as exc:
        if not any(sig in str(exc) for sig in _STALL_SIGNATURES):
            raise
        outcome = "stalled"
    open(os.path.join(info["folder"], MARKER), "w").close()
    return outcome


def _fix_one_worker(task):
    """Picklable ProcessPoolExecutor unit: repair one run, never raise (so one
    bad run can't tear down the pool). Returns ``(name, status_string)``."""
    info, max_wall_seconds, min_progress_rate = task
    try:
        outcome = fix_run(info, max_wall_seconds, min_progress_rate)
        return (info["name"], "FIXED (%s)" % outcome)
    except Exception as exc:  # noqa: BLE001
        return (info["name"], "FAILED %s: %s" % (type(exc).__name__, exc))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true",
                    help="actually re-resume (default: dry-run, list only)")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--max-wall-seconds", type=float, default=None,
                    help="hard wall-clock cap per run (bails a re-stall)")
    ap.add_argument("--min-progress-rate", type=float, default=None,
                    help="floor on sim-time advanced per wall-second (bails a re-stall)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel processes (default: min(#runs, cpu_count); 1 = serial)")
    args = ap.parse_args(argv)

    runs = find_resumed_runs(args.results_dir)
    print("Found %d resumed-before-fix run(s) in %s:\n" % (len(runs), args.results_dir))
    for r in runs:
        note = "" if r["resume_from"] is not None else "  [NO CLEAN SNAPSHOT — will skip]"
        print("  %-52s resumed@%s  ->  re-resume from %s%s"
              % (r["name"], r["resume_time"], r["resume_from"], note))
    if not runs:
        return
    if not args.fix:
        print("\nDRY RUN. Re-run with --fix to actually re-resume them.")
        return

    to_fix = [r for r in runs if r["resume_from"] is not None]
    skipped = [r for r in runs if r["resume_from"] is None]
    for r in skipped:
        print("SKIP %s: no clean snapshot before t=%s (resumed from the start)."
              % (r["name"], r["resume_time"]))
    if not to_fix:
        return

    n_workers = args.workers or min(len(to_fix), os.cpu_count() or 1)
    tasks = [(r, args.max_wall_seconds, args.min_progress_rate) for r in to_fix]
    print("\n--- re-resuming %d run(s) across %d process(es) ---" % (len(to_fix), n_workers),
          flush=True)

    # Each re-resume writes into its OWN run folder, so they're independent —
    # run them in parallel processes (one full simulation each).
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_fix_one_worker, tasks))
    else:
        results = [_fix_one_worker(t) for t in tasks]

    print("\n=== results ===")
    fixed = failed = 0
    for name, status in results:
        print("  %-52s %s" % (name, status))
        if status.startswith("FIXED"):
            fixed += 1
        else:
            failed += 1
    print("\nDone. fixed=%d failed=%d skipped=%d" % (fixed, failed, len(skipped)))


if __name__ == "__main__":
    main()
