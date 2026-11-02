"""Extract the reproducibility package for a set of runs — without the histories.

    python extract_li_seeds.py --pattern "fullmodel_v2*" --out /data/reproduce
    python extract_li_seeds.py --pattern "fullmodel_v2*" --out D:\\pkg --workers 6

WHY THIS EXISTS. A run is NOT reproducible from its ``parameters.txt`` alone.
Every base run is started with ``randomize_notch_delta_levels=True`` and
``notch_levels: None``, and the draw goes through the UNSEEDED global
``np.random`` (inner_ear_model, ``_seed_li_levels``). Re-running the recorded
parameters therefore gives a statistically equivalent trajectory, not the same
one. The drawn state survives in exactly one place: frame 0 of ``history.hf5``.

So the archive that actually reproduces the work is

    parameters.txt + l0.txt + the frame-0 LI arrays        (~15 KB per run)

rather than the history (~600 MB per run). This writes that, plus a manifest
recording the git commit the runs were produced by.

FORKS DO NOT NEED THIS. A strict steady state or an ablation inherits its LI
state from the archive it forks, so it is fully determined by its source run,
``continue_from_time`` and ``ablated_cells`` — all three already in its
``parameters.txt``. Point ``--pattern`` at the base runs.

REPLAYING a packaged run: pass the three arrays back in and turn the draw off,

    levels = {k: np.load(os.path.join(pkg, run, k + "_levels.npy"))
              for k in ("notch", "delta", "repressor")}
    run(..., randomize_notch_delta_levels=False,
        notch_levels=levels["notch"], delta_levels=levels["delta"],
        repressor_levels=levels["repressor"])

the arrays being ordered by the frame-0 ``id`` column, which ``ids.npy`` stores
beside them so the mapping can be checked rather than assumed.
"""
import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from history_io import RESULTS_DIR, load_history_file, get_time_points

LEVELS = ("notch_level", "delta_level", "repressor_level")
EXTRAS = ("parameters.txt", "l0.txt")
# Small per-run products worth keeping: they are inputs to the figures and cost
# a few MB in total, unlike the history.
FIGURE_FILES = ("initial.png", "finale.png", "delta_initial.png", "delta_finale.png")


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
        return {"commit": out.stdout.strip() or None,
                "dirty": bool(dirty.stdout.strip())}
    except Exception:                                          # noqa: BLE001
        return {"commit": None, "dirty": None}


def one_run(args):
    name, out_root, want_figures = args
    src = os.path.join(RESULTS_DIR, name)
    dst = os.path.join(out_root, name)
    rec = {"run": name, "error": None}
    try:
        os.makedirs(dst, exist_ok=True)
        for fn in EXTRAS + (FIGURE_FILES if want_figures else ()):
            p = os.path.join(src, fn)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(dst, fn))
        # Runs older than the l0.txt record have no file to copy. Resolve the
        # value the same way a fork would and write it, so the package does not
        # depend on the histories still existing to answer the question.
        if not os.path.isfile(os.path.join(dst, "l0.txt")):
            from run_model import _recorded_L0
            l0 = _recorded_L0(name)
            if l0 is not None:
                with open(os.path.join(dst, "l0.txt"), "w") as fh:
                    fh.write("%.12g\n" % l0)
                rec["l0_resolved"] = True

        history = load_history_file(name)
        stamps = np.asarray(get_time_points(history), float)
        t0 = float(np.min(stamps))
        sheet = history.retrieve(t0)
        sheet.arrange_sheet_from_history()
        face = sheet.face_df
        rec.update(t0=t0, n_faces=int(len(face)), n_frames=int(stamps.size),
                   t_end=float(np.max(stamps)))
        for col in LEVELS:
            if col in face.columns:
                np.save(os.path.join(dst, col + "s.npy"),
                        face[col].to_numpy(float))
            else:
                rec.setdefault("missing", []).append(col)
        # the identity the arrays are ordered by, so a replay can verify the
        # mapping instead of trusting positional order
        if "id" in face.columns:
            np.save(os.path.join(dst, "ids.npy"), face["id"].to_numpy())
        if "notch_sensitivity" in face.columns:
            v = face["notch_sensitivity"].to_numpy(float)
            if np.ptp(v) > 0:        # only when it was actually randomised
                np.save(os.path.join(dst, "notch_sensitivity.npy"), v)
    except Exception as exc:                                   # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", nargs="+", required=True,
                    help="glob(s) over run folder names, e.g. 'fullmodel_v2*'")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--figures", action="store_true",
                    help="also copy the four small pngs per run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    names = sorted(d for d in os.listdir(RESULTS_DIR)
                   if os.path.isfile(os.path.join(RESULTS_DIR, d, "history.hf5"))
                   and any(fnmatch.fnmatch(d, p) for p in a.pattern))
    print("results dir : %s" % RESULTS_DIR)
    print("matched     : %d run(s)" % len(names))
    if not names:
        raise SystemExit("nothing matched %s" % a.pattern)
    if a.dry_run:
        for n in names[:10]:
            print("   ", n)
        raise SystemExit("... %d total\n\n--dry-run: nothing written." % len(names))

    os.makedirs(a.out, exist_ok=True)
    tasks = [(n, a.out, a.figures) for n in names]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs = list(ex.map(one_run, tasks))
    else:
        recs = [one_run(t) for t in tasks]

    bad = [r for r in recs if r.get("error")]
    for r in bad[:10]:
        print("  FAILED %-60s %s" % (r["run"], r["error"][:70]))
    manifest = {"results_dir": RESULTS_DIR, "git": git_commit(),
                "pattern": a.pattern, "n_runs": len(recs),
                "n_failed": len(bad), "runs": recs}
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=float)

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(a.out) for f in fn)
    print("\n%d packaged, %d failed" % (len(recs) - len(bad), len(bad)))
    print("git commit  : %s%s" % (manifest["git"]["commit"],
                                  "  (WORKING TREE DIRTY)" if manifest["git"]["dirty"] else ""))
    print("package size: %.1f MB in %s" % (total / 1e6, a.out))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
