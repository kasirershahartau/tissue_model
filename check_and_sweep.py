"""Verify the downloaded full-model histories, drop any that are corrupt, then
launch the psigma sweep.

    python check_and_sweep.py              # check, delete corrupt, run the sweep
    python check_and_sweep.py --dry-run    # report only, delete nothing, no sweep
    python check_and_sweep.py --no-sweep   # check + delete, but stop there

Why: a ``history.hf5`` whose transfer was cut short still has a plausible name
and size, but fails to open — and inside the sweep that surfaces as a crash
hours in. Worse, a file that opens but is TRUNCATED can yield a sheet that
looks valid, so this retrieves the last frame rather than only listing the
time stamps.

A folder that fails is DELETED, which is the right outcome: the sweep then
re-runs that array from scratch instead of resuming from a broken archive.
Only ``fullmodel_*`` folders are ever touched — the ``random_periodic_array*``
inputs and any ``fit_*`` results are left alone.
"""
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tyssue", "src"))

from post_processing import RESULTS_DIR, load_history_file, get_time_points  # noqa: E402


def check(folder_name):
    """(ok, detail). Opens the archive AND materialises its last frame."""
    try:
        history = load_history_file(folder_name)
        stamps = get_time_points(history)
        if len(stamps) == 0:
            return False, "no frames"
        sheet = history.retrieve(float(stamps[-1]))
        sheet.arrange_sheet_from_history()
        if sheet.face_df.shape[0] == 0:
            return False, "last frame has no faces"
        return True, "%d frames, t=%.1f, %d cells" % (
            len(stamps), float(stamps[-1]), sheet.face_df.shape[0])
    except Exception as exc:  # noqa: BLE001 - any failure means unusable
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:60])


def main():
    dry = "--dry-run" in sys.argv
    no_sweep = "--no-sweep" in sys.argv or dry

    folders = sorted(os.path.basename(p) for p in
                     glob.glob(os.path.join(RESULTS_DIR, "fullmodel_*"))
                     if os.path.isdir(p))
    if not folders:
        print("no fullmodel_* folders under %s" % RESULTS_DIR)
        print("the sweep would run every stage from scratch - stopping so that is a "
              "deliberate choice, not an accident.")
        return 1

    print("checking %d full-model archives under %s\n" % (len(folders), RESULTS_DIR),
          flush=True)
    bad = []
    for name in folders:
        ok, detail = check(name)
        print("  %-5s %-56s %s" % ("OK" if ok else "BAD", name[:56], detail), flush=True)
        if not ok:
            bad.append(name)

    print("\n%d ok, %d corrupt" % (len(folders) - len(bad), len(bad)))
    if bad and not dry:
        for name in bad:
            path = os.path.join(RESULTS_DIR, name)
            # belt and braces: never delete outside the fullmodel_* namespace
            if os.path.basename(path).startswith("fullmodel_") and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                print("  deleted %s" % name)
    elif bad:
        print("  (--dry-run: nothing deleted)")

    remaining = len([p for p in glob.glob(os.path.join(RESULTS_DIR, "fullmodel_*"))
                     if os.path.isdir(p)])
    print("\n%d full-model folders remain (these will be reused/resumed; anything "
          "missing re-runs from scratch)" % remaining)

    if no_sweep:
        return 0
    print("\n=== launching the psigma sweep ===\n", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(HERE, "tyssue", "src")
    return subprocess.call([sys.executable, os.path.join(HERE, "run_psigma_sweep.py")],
                           cwd=HERE, env=env)


if __name__ == "__main__":
    sys.exit(main())
