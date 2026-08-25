"""Compact mechanical-fit history archives down to only the frame(s) the p-value
calculation actually reads, to reclaim disk space.

A finished ``fit_*`` run stores a full ``history.hf5`` (e.g. 184 frames, ~900 MB),
but the mechanics comparison only ever reads:

* **base (un-ablated) runs** — the LAST (steady-state) frame: HC/SC area ratio and
  HC/SC roundness come from ``extract_model_mechanics`` -> ``history.retrieve(max
  time)``. So only the last frame is kept.
* **ablation runs** (folder name ends ``_abl``) — the FIRST *and* LAST frame:
  ``calc_area_change_after_ablation`` reads ``history.retrieve(0)`` (the just-
  ablated state) AND ``history.retrieve(last)`` (the relaxed state) and returns
  their area ratio. Dropping the first frame would make every area ratio 1.0 —
  silently wrong — so BOTH frames are kept for ``_abl`` runs.

Only ``fit_*`` folders are touched. Differentiation / initial-morphology runs
(``random_periodic_array*`` etc.) trace cells across many frames (HC-neighbours-
at-differentiation) and are FORK sources, so they keep their full history.

Why a rewrite (not an in-place delete): PyTables/HDF5 ``remove`` marks rows free
but does NOT shrink the file, and ``ptrepack`` isn't installed. So each file is
rewritten to a small temp holding only the kept frame(s), then atomically renamed
over the original. The temp is tiny (~1/frame-count of the original), so this
reclaims space even when the disk is nearly full. ``retrieve`` snaps to the
nearest recorded time and ``from_archive`` reads the LAST vert row, so the kept
frames are written in ascending-time order (max-time frame last) and every reader
still resolves the same sheets. Each rewrite is verified (correct times present,
last row == max time) BEFORE the original is replaced; a failure leaves the
original untouched.

Usage
-----
    # DRY RUN — report per-run frame counts + projected space freed:
    python compact_history_files.py

    # test on the first few before committing to all:
    python compact_history_files.py --limit 5 --apply

    # actually compact every fit_* history file:
    python compact_history_files.py --apply

    # a few parallel workers (only helps on an SSD; default serial):
    python compact_history_files.py --apply --workers 4
"""
import os
import sys
import glob
import shutil
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_processing import RESULTS_DIR

# The history elements that carry a per-frame ``time`` column. Anything else in
# the archive (e.g. a ``settings`` Series) is copied through verbatim.
TIME_TABLES = ("vert", "edge", "face")


def _unique_times(store):
    """Distinct frame times in the archive, read cheaply from the vert table's
    ``time`` column only (the same source HistoryHdf5.time_stamps uses)."""
    return pd.unique(store.select("vert", columns=["time"])["time"].values)


def frames_to_keep(folder_name, times):
    """Times to keep for this run. Ablation (`_abl`) runs need the first frame
    (retrieve(0)) AND the last; base runs need only the last."""
    times = np.asarray(times, dtype=float)
    tmax = float(times.max())
    if folder_name.endswith("_abl"):
        return sorted({float(times.min()), tmax})
    return [tmax]


def _verify(path, keep):
    """Reopen the rewritten file and assert it holds EXACTLY the kept frame times
    and that the last vert row is the max time (what from_archive reads as the
    current/last frame). Raises on any mismatch so the caller keeps the original."""
    keep_r = set(np.round(np.asarray(keep, float), 9))
    with pd.HDFStore(path, "r") as s:
        keys = [k.strip("/") for k in s.keys()]
        present = [k for k in TIME_TABLES if k in keys]
        if "vert" not in present:
            raise ValueError("no vert table after rewrite")
        for key in present:
            t = pd.unique(s.select(key, columns=["time"])["time"].values)
            if set(np.round(t.astype(float), 9)) != keep_r:
                raise ValueError("%s times %s != kept %s" % (key, list(t), keep))
        last = float(s.select("vert", columns=["time"]).iloc[-1]["time"])
        if round(last, 9) != round(float(max(keep)), 9):
            raise ValueError("last vert time %.6g != max kept %.6g" % (last, max(keep)))


def compact_one(folder, apply=False, work_dir=None, complib="blosc", complevel=5):
    """Compact one run's history.hf5. Returns (name, old_bytes, new_bytes, status).
    Never raises: a per-file failure is reported and the original is left intact.

    ``work_dir`` — where the small compacted file is written before it replaces
    the original. Give a directory on a DRIVE WITH FREE SPACE when the results
    drive is full (an in-place temp can't be created at 0 bytes free). The flow
    then is: write the compact file to ``work_dir`` and verify it, DELETE the
    original (freeing the big file), move the compact file into place, re-verify.
    The original is deleted only AFTER a verified compact file exists elsewhere,
    so a crash leaves the compact recoverable in ``work_dir`` (and you have the
    backup). With ``work_dir=None`` an in-place temp + atomic os.replace is used
    (needs free space on the results drive)."""
    name = os.path.basename(folder)
    path = os.path.join(folder, "history.hf5")
    if not os.path.isfile(path):
        return (name, 0, 0, "SKIP no history.hf5")
    old_size = os.path.getsize(path)
    try:
        with pd.HDFStore(path, "r") as store:
            keys = [k.strip("/") for k in store.keys()]
            if "vert" not in keys:
                return (name, old_size, old_size, "SKIP no vert table")
            times = _unique_times(store)
            keep = frames_to_keep(name, times)
            if len(times) <= len(keep):
                return (name, old_size, old_size,
                        "SKIP already %d frame(s)" % len(times))
    except Exception as exc:  # noqa: BLE001
        return (name, old_size, old_size, "FAILED read %s: %s" % (type(exc).__name__, exc))

    projected = int(old_size * (len(keep) / max(len(times), 1)))
    if not apply:
        return (name, old_size, projected,
                "DRY keep %s of %d frames" % ([round(t, 4) for t in keep], len(times)))

    if work_dir is not None:
        tmp = os.path.join(work_dir, name + ".compact.hf5")
    else:
        tmp = path + ".compact.tmp"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        with pd.HDFStore(path, "r") as src, \
                pd.HDFStore(tmp, "w", complib=complib, complevel=complevel) as dst:
            for key in [k.strip("/") for k in src.keys()]:
                if key not in TIME_TABLES:
                    dst.put(key, src[key])   # settings etc. — copy verbatim
                    continue
                # Kept rows in ASCENDING time order -> the max-time frame's rows
                # land LAST, so from_archive's iloc[-1] resolves to it.
                parts = [src.select(key, where="time == %.17g" % t) for t in keep]
                df = pd.concat(parts) if parts else src.select(key, stop=0)
                dst.put(key, df, format="table", data_columns=["time"])
        _verify(tmp, keep)
    except Exception as exc:  # noqa: BLE001
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return (name, old_size, old_size, "FAILED %s: %s" % (type(exc).__name__, exc))

    new_size = os.path.getsize(tmp)
    if work_dir is None:
        os.replace(tmp, path)   # atomic on the same volume; frees the original
        return (name, old_size, new_size, "OK kept %d frame(s)" % len(keep))

    # Cross-volume (results drive full): free the original, then move the small
    # compact file into its place and re-verify.
    try:
        os.remove(path)                 # frees the big original -> room on the drive
        shutil.move(tmp, path)          # copy compact back (drive now has room)
        _verify(path, keep)             # confirm the moved file is intact
    except Exception as exc:  # noqa: BLE001
        # The verified compact file may still be sitting in work_dir — surface it
        # so it can be moved into place by hand (the backup also has the original).
        return (name, old_size, old_size,
                "FAILED move/verify %s: %s (compact kept at %s)"
                % (type(exc).__name__, exc, tmp))
    return (name, old_size, new_size, "OK kept %d frame(s)" % len(keep))


def _worker(task):
    folder, apply, work_dir = task
    return compact_one(folder, apply=apply, work_dir=work_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually rewrite (default: dry-run, report only)")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--pattern", default="fit_*",
                    help="folder glob under results-dir (default: fit_*)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N runs (test on a few first)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes (default 1; >1 only helps on an SSD)")
    ap.add_argument("--work-dir", default=None,
                    help="stage the small compact file here (a dir on a drive WITH "
                         "free space) when the results drive is full; the original "
                         "is deleted only after a verified compact exists here")
    ap.add_argument("--min-age-minutes", type=float, default=0.0,
                    help="skip any run whose history.hf5 was modified within the last "
                         "N minutes. Use this to compact SAFELY while a fit is still "
                         "RUNNING: an actively-written run has a fresh mtime, so a "
                         "generous value (e.g. 30) guarantees the compactor never "
                         "touches a history the fit may still be writing.")
    args = ap.parse_args(argv)

    work_dir = args.work_dir
    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        free = shutil.disk_usage(work_dir).free
        print("work-dir: %s  (%.1f GB free)" % (work_dir, free / 1e9))

    folders = sorted(
        d for d in glob.glob(os.path.join(args.results_dir, args.pattern))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "history.hf5")))

    # SAFETY when a fit is still running: drop runs whose history.hf5 was touched
    # within --min-age-minutes. An in-progress simulation writes its history
    # continuously, so a fresh mtime means "the fit may still hold this file";
    # compacting it could collide with the writer (HDF5 locking) and crash the sim.
    if args.min_age_minutes > 0:
        import time
        cutoff = time.time() - args.min_age_minutes * 60.0
        before = len(folders)
        folders = [d for d in folders
                   if os.path.getmtime(os.path.join(d, "history.hf5")) <= cutoff]
        print("skipping %d run(s) modified within the last %.0f min "
              "(in-progress — left untouched)"
              % (before - len(folders), args.min_age_minutes), flush=True)

    if args.limit is not None:
        folders = folders[:args.limit]

    print("%s  %d history file(s) under %s matching %r"
          % ("APPLY" if args.apply else "DRY-RUN", len(folders),
             args.results_dir, args.pattern), flush=True)

    tasks = [(f, args.apply, work_dir) for f in folders]
    if args.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results_iter = ex.map(_worker, tasks)
    else:
        results_iter = (_worker(t) for t in tasks)

    old_tot = new_tot = 0
    n_ok = n_skip = n_fail = 0
    n_done = 0
    total = len(tasks)
    for name, old, new, status in results_iter:
        n_done += 1
        old_tot += old
        new_tot += new
        if status.startswith("OK"):
            n_ok += 1
        elif status.startswith("FAILED"):
            n_fail += 1
            print("  FAIL %-58s %s" % (name[:58], status), flush=True)
        else:
            n_skip += 1
        # Progress heartbeat every 50 runs (and at the end) so a long background
        # run is monitorable and shows cumulative space freed.
        if args.apply and (n_done % 50 == 0 or n_done == total):
            print("  ... %d/%d  ok=%d skip=%d fail=%d  freed so far=%.2f GB"
                  % (n_done, total, n_ok, n_skip, n_fail,
                     (old_tot - new_tot) / 1e9), flush=True)

    freed = old_tot - new_tot
    print("\nruns: %s=%d  skipped=%d  failed=%d"
          % ("compacted" if args.apply else "would-compact", n_ok, n_skip, n_fail))
    print("current total : %.2f GB" % (old_tot / 1e9))
    print("%s: %.2f GB   (%s %.2f GB)"
          % ("after         " if args.apply else "projected     ", new_tot / 1e9,
             "freed" if args.apply else "would free", freed / 1e9))
    if not args.apply:
        print("\nDRY-RUN only. Re-run with --apply (optionally --limit N first).")


if __name__ == "__main__":
    main()
