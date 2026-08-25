"""Rebuild the full-model movie gifs that are missing or stale.

    python make_missing_gifs.py --dry-run        # what would be built
    python make_missing_gifs.py --workers 6
    python make_missing_gifs.py --pattern "fullmodel_ps0.060_*" --frames 60

WHY THIS EXISTS. The gif the simulation writes for itself never appeared on
Windows: tyssue's create_gif finishes with

    subprocess.run(["convert", (graph_dir / "movie_*.png").as_posix(), output])

and on Windows ``convert`` resolves through CreateProcess, which searches
System32 BEFORE PATH — so the FAT->NTFS ``convert.exe`` runs instead of
ImageMagick's and answers "Invalid drive specification.". With no ``check=True``
the failure is silent, so runs completed "successfully" with no gif. That is
fixed in post_processing.create_gif_safe, which now assembles the gif with
Pillow and never shells out; this script just applies it in bulk.

STALENESS. A gif is rebuilt when it is missing, or older than the run's
history.hf5 — which is exactly the case for every run the psigma sweep extended
from t=50 to t=100, since resuming rewrites the archive.

CONCURRENCY. Rendering is ~3 s per frame (matplotlib, not the encoder), so a
100-frame gif is ~5 minutes. Runs are processed in parallel; keep --workers
modest if a sweep is using the machine. Runs whose archive changed in the last
--min-age minutes are SKIPPED, so a history still being written by a live sweep
is never read mid-write.
"""
import argparse
import fnmatch
import os
import time
import warnings

warnings.filterwarnings("ignore")

from post_processing import RESULTS_DIR, redraw


def produced_path(run, save_name):
    """Where redraw(run, save_name) actually writes its gif."""
    return os.path.join(RESULTS_DIR, run, "%s_movie.gif" % save_name)


def target_path(run, gif_name):
    """Where we want the gif to END UP. redraw can only write
    ``<save_name>_movie.gif``, so overwriting the simulation's own
    ``movie.gif`` means renaming afterwards."""
    return os.path.join(RESULTS_DIR, run, gif_name)


def classify(run, gif_name, min_age_s):
    """-> (action, reason). action in {"build", "skip"}."""
    hist = os.path.join(RESULTS_DIR, run, "history.hf5")
    if not os.path.isfile(hist):
        return "skip", "no history.hf5"
    age = time.time() - os.path.getmtime(hist)
    if age < min_age_s:
        return "skip", "archive written %.0f min ago (run may be live)" % (age / 60)
    gif = target_path(run, gif_name)
    if not os.path.isfile(gif):
        return "build", "missing"
    if os.path.getmtime(gif) < os.path.getmtime(hist):
        return "build", "stale (older than history.hf5)"
    return "skip", "up to date"


def one_run(args):
    run, save_name, frames, color_by, gif_name = args
    t0 = time.time()
    try:
        redraw(run, save_name, movie=True,
               maximal_number_of_frames_to_save=frames, color_by=color_by)
        produced, target = produced_path(run, save_name), target_path(run, gif_name)
        if not os.path.isfile(produced):
            raise RuntimeError("redraw reported success but %s is absent"
                               % os.path.basename(produced))
        if os.path.abspath(produced) != os.path.abspath(target):
            os.replace(produced, target)     # atomic overwrite of the old gif
        return run, time.time() - t0, None
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the batch
        return run, time.time() - t0, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="fullmodel_*",
                    help="glob over run-folder names (default every fullmodel_*)")
    ap.add_argument("--save-name", dest="save_name", default="movie",
                    help="redraw's output prefix; it also writes "
                         "<save-name>_initial.png / _finale.png")
    ap.add_argument("--gif-name", dest="gif_name", default="movie.gif",
                    help="final gif FILENAME inside each run folder (default "
                         "movie.gif - the one the simulation itself failed to "
                         "write). redraw's output is renamed onto this.")
    ap.add_argument("--frames", type=int, default=100,
                    help="max frames in the gif (default 100)")
    ap.add_argument("--color-by", dest="color_by", default="atoh")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-age", dest="min_age", type=float, default=10.0,
                    help="skip runs whose history.hf5 changed within this many "
                         "minutes - protects against reading a live run (default 10)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs = sorted(n for n in os.listdir(RESULTS_DIR)
                  if fnmatch.fnmatch(n, a.pattern)
                  and os.path.isdir(os.path.join(RESULTS_DIR, n)))
    todo, skipped = [], {}
    for run in runs:
        action, reason = classify(run, a.gif_name, a.min_age * 60)
        if action == "build":
            todo.append((run, reason))
        else:
            skipped[reason] = skipped.get(reason, 0) + 1

    print("%d run(s) matched '%s'" % (len(runs), a.pattern))
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print("  skip  %-42s %d" % (reason, n))
    build_reasons = {}
    for _run, reason in todo:
        build_reasons[reason] = build_reasons.get(reason, 0) + 1
    for reason, n in sorted(build_reasons.items(), key=lambda kv: -kv[1]):
        print("  BUILD %-42s %d" % (reason, n))
    if not todo:
        print("\nnothing to do.")
        return
    print("\n-> %d gif(s) to build, %d frame(s) each, %d worker(s)"
          % (len(todo), a.frames, a.workers))
    print("   rough estimate: %.1f h wall-clock at ~3 s/frame"
          % (len(todo) * a.frames * 3.0 / 3600.0 / max(a.workers, 1)))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was built.")

    tasks = [(run, a.save_name, a.frames, a.color_by, a.gif_name)
             for run, _ in todo]
    # Report each run AS IT FINISHES. list(executor.map(...)) blocks until the
    # whole batch is done, which on a multi-hour job means no progress and no
    # sight of a failure pattern until the very end.
    results = []
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = {ex.submit(one_run, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                run, secs, err = results[-1]
                print("  [%3d/%3d] %-52s %s (%.0fs)"
                      % (i, len(tasks), run[:52],
                         "OK" if err is None else "FAILED " + err, secs), flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            results.append(one_run(t))
            run, secs, err = results[-1]
            print("  [%3d/%3d] %-52s %s (%.0fs)"
                  % (i, len(tasks), run[:52],
                     "OK" if err is None else "FAILED " + err, secs), flush=True)

    built = [r for r in results if r[2] is None]
    failed = [r for r in results if r[2] is not None]
    print("\nbuilt %d, failed %d" % (len(built), len(failed)))
    if failed:
        print("failures:")
        for run, _s, err in failed:
            print("  %-58s %s" % (run[:58], err))


if __name__ == "__main__":
    main()
