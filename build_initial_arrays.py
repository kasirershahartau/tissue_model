"""Build the pool of disordered initial configurations, and choose twenty.

Everything needed to RUN the initial-configuration step, as one pipeline.
:mod:`plot_initial_arrays` draws the figures and tables from what this leaves
behind; nothing here plots, and nothing there simulates.

    python build_initial_arrays.py build     --arrays 0 467 --workers 6
    python build_initial_arrays.py score     --workers 6
    python build_initial_arrays.py select
    python build_initial_arrays.py seed-data
    python build_initial_arrays.py prune     --execute
    python build_initial_arrays.py finalise  --execute

RUN THEM IN THAT ORDER. ``seed-data`` copies each run's parameters.txt, so it
has to precede ``prune``, which deletes the runs it copied them from;
``finalise`` renames and relocates the winners, so everything addressing them
by pool name has to be finished first.

WHAT THE STEPS DO

  build      Simulate N random arrays, seeded, in a recycled process pool.
             Each array starts from a REGULAR periodic lattice and is
             disordered by divisions plus the per-cell parameter re-draws that
             ``random_forces`` makes every 5 time units.
  score      For every array, score every frame against both stages'
             experimental neighbour-number distributions and cache the trace
             (``z2_vs_time_mergedtails.npz``). This is the expensive read step
             and it is what every later decision is made from.
  select     Rank the arrays by their own best frame, keep the best ten per
             stage, and write each one out as a single-frame extract.
  seed-data  Copy the parameters and cross-check every recorded seed against
             the rule, so what each array WAS is on record without its history.
  prune      Preserve the small artefacts, then delete the non-winners.
  finalise   Rename the twenty winners by RANK and gather them, with
             everything else from this step, under one results folder.

TWO THINGS THAT ARE EASY TO GET WRONG

  * An array contributes ONE frame, its own best. Taking the ten best
    (array, frame) pairs globally would let several near-adjacent frames of one
    lucky array fill the set, and those are the same tissue -- the ten initial
    conditions exist to be independent.
  * An array whose minimum sits at the END of its trace is EXCLUDED, not
    trusted: the trace was still descending when the run stopped, so its score
    is only an upper bound on what that array would have reached.

The number to quote for a selected set is the BOUND -- the worst z^2 among the
ten kept -- because it states the criterion and the threshold it reached. The
mean of the ten is not quoted: the minimum of a large sample is a biased
statistic and averaging over selected items hides that.
"""
import argparse
import ast
import csv
import json
import os
import re
import shutil
import sys
import time

import numpy as np

from history_io import load_history_file, save_data_of_a_given_time_point
from post_processing import (RESULTS_DIR, INITIAL_CONFIG_DIR,
                             INITIAL_CONFIG_RUNS, INITIAL_CONFIG_SEED_DATA,
                             initial_morphology_name, chosen_array_run_name,
                             find_best_timepoint_according_to_n_neighbors,
                             _frame_is_fold_free, closest_fold_free_frames)
from run_model import (random_array_variant_name, array_seed,
                       DEFAULT_ARRAY_SEED)

STAGES = ("E17.5", "P0")
TRACE = "z2_vs_time_mergedtails.npz"
DIVISION_AREA = 1.1
KEEP = 10
#: Appended to the variant name, to keep a seeded build off the folder names of
#: the earlier unseeded sets, which must not be overwritten.
#:
#: "Seeded" does NOT mean reproducible, and nothing here should imply it does.
#: Measured 2026-11-19: array 0 built twice with the same seed and identical
#: settings gave two different tissues. The runs agree through the regular
#: lattice and the first parameter re-draw, then separate at t~5.2 when
#: divisions begin -- a last-bit difference changes which cell divides first,
#: that division consumes a random angle, and the streams never resynchronise.
#: The seed fixes the DRAWS, so the pool reproduces as a distribution; no
#: individual array can be recovered, so every run folder is irreplaceable.
NAME_SUFFIX = "_seeded"
_HEAD = "random_periodic_array"


def pool_name(index, division_area=DIVISION_AREA, name_suffix=NAME_SUFFIX):
    """The pool folder for one array: the ONE place this name is built.

    ``random_array_variant_name`` already carries the division-area tag
    (``_da1p1``), so the suffix here adds only the seeding marker. Building the
    name in two places is how the first version of this file produced
    ``random_periodic_array0_da1p1_da1p1_seeded``.
    """
    return random_array_variant_name(index, division_area=division_area) \
        + name_suffix


def pool_regex(division_area=DIVISION_AREA, name_suffix=NAME_SUFFIX):
    """The matching pattern, derived from :func:`pool_name` rather than typed
    out again, so a build at another division area is still recognised by the
    steps that follow it."""
    tail = pool_name(0, division_area, name_suffix)[len(_HEAD) + 1:]
    return re.compile(r"^%s(\d+)%s$" % (_HEAD, re.escape(tail)))


POOL = pool_regex()

#: Nothing in the pool needs frames past t ~ 13: across the scored arrays the
#: latest best frame of an array good enough to be SELECTED is t=12.3 (P0) and
#: t=10.8 (E17.5). 15 leaves ~2.7 units of margin and halves both the wall
#: clock and the disk against the 25 the first arrays were built with.
T_END = 15.0

#: Stall guards. Both default OFF in run(); a batch build needs them most,
#: because a wedged array pins its worker for as long as the batch lives (one
#: batch of six sat 4.6 h with no solver step completing). They save the
#: history before raising, so a stopped array still leaves a usable truncated
#: trace and the loop records it as CRASHED and carries on. 1e-4 sim-time/s is
#: ~40x below healthy; the window is widened from the 30 s default because a
#: legitimate slow patch -- dt pinned near the floor through a topology event --
#: can sit under the floor for tens of seconds without being wedged.
MAX_WALL_SECONDS = 10800
MIN_PROGRESS_RATE = 1e-4
PROGRESS_WINDOW_SECONDS = 300.0

#: Arrays a worker may build before it is replaced. A worker grows by order a
#: gigabyte per array and never gives it back, so a long batch on one pool ends
#: up thrashing: three batches wedged this way, six workers holding 15-16 GB
#: each on a 32 GB box. Python 3.10 has no ``max_tasks_per_child``, so the
#: workers are recycled by hand -- a fresh executor per chunk.
ARRAYS_PER_WORKER = 4

#: Fields lifted out of parameters.txt into arrays.csv. Everything else stays
#: in the copied file; the point of the copy is that nothing is lost to curation.
LIFT = ("seed", "division_area", "division_rule", "t_end", "dt", "nx", "ny",
        "max_bond_length", "min_bond_length", "intercalation_length",
        "delamination_area", "delamination_rate", "line_tension")
SMALL = (TRACE, "z2_vs_time_mergedtails.png", "z2_vs_time_mergedtails.svg")


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def pool_members(results_dir=RESULTS_DIR, with_history=True):
    """Pool folders on disk, in index order."""
    out = []
    for d in sorted(os.listdir(results_dir)):
        if not POOL.match(d):
            continue
        if with_history and not os.path.isfile(
                os.path.join(results_dir, d, "history.hf5")):
            continue
        out.append(d)
    return sorted(out, key=lambda d: int(POOL.match(d).group(1)))


def read_trace(folder, stage, results_dir=RESULTS_DIR):
    """An array's best frame for one stage, or None when it has no usable trace.

    ``at_end`` flags a trace that was still descending when the run stopped;
    such an array is excluded from the selection rather than trusted.
    """
    p = os.path.join(results_dir, folder, TRACE)
    if not os.path.isfile(p):
        return None
    with np.load(p, allow_pickle=True) as z:
        key = "z2_%s" % stage
        if key not in z:
            return None
        tr = np.asarray(z[key], float)
        t = np.asarray(z["time"], float)
    good = np.isfinite(tr)
    if not good.any():
        return None
    k = int(np.argmin(np.where(good, tr, np.inf)))
    return dict(folder=folder, z2=float(tr[k]), t=float(t[k]),
                t_end=float(t.max()), at_end=bool(abs(t[k] - t.max()) < 0.3),
                trace_t=t, trace_z2=tr)


def read_params(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k not in LIFT:
                continue
            try:
                out[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                out[k] = v
    return out


def seed_dir(results_dir=RESULTS_DIR):
    return os.path.join(results_dir, INITIAL_CONFIG_SEED_DATA)


def dir_bytes(p):
    tot = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try:
                tot += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return tot


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def cmd_build(a):
    from concurrent.futures import ProcessPoolExecutor
    from run_model import _create_one_random_array_star
    import pandas as pd

    indices = list(range(a.arrays[0], a.arrays[1] + 1))
    names = [pool_name(i, a.division_area, a.name_suffix) for i in indices]
    # the later steps find the pool by pattern, so a non-default build has to
    # be matched by one too, or score/select would silently see nothing
    global POOL
    POOL = pool_regex(a.division_area, a.name_suffix)
    print("building %d array(s), division_area=%g, t_end=%g, base seed %d"
          % (len(indices), a.division_area, a.t_end, DEFAULT_ARRAY_SEED))
    print("  guards: %s"
          % ("max_wall=%gs, min_rate=%g over %gs"
             % (a.max_wall_seconds, a.min_progress_rate, PROGRESS_WINDOW_SECONDS)
             if a.guards else "OFF"))
    if a.dry_run:
        for i, nm in list(zip(indices, names))[:10]:
            print("   %-38s seed %d" % (nm, array_seed(i)))
        print("--dry-run: nothing built.")
        return 0

    # NOT create_random_arrays: its executor.map re-raises the first failure,
    # and ~1 build in 8 dies at the dt floor ("[...] not in index"). That would
    # let one known-flaky run take out every healthy one still queued, so the
    # outcomes are collected per array instead.
    tasks = [{"index": i, "name": nm, "division_area": a.division_area,
              "t_end": a.t_end, "seed": array_seed(i),
              "max_wall_seconds": a.max_wall_seconds if a.guards else None,
              "min_progress_rate": a.min_progress_rate if a.guards else None,
              "progress_window_seconds": PROGRESS_WINDOW_SECONDS}
             for i, nm in zip(indices, names)]
    per_chunk = (len(tasks) if a.arrays_per_worker <= 0
                 else max(1, a.workers * a.arrays_per_worker))
    ok, bad = [], []
    for c0 in range(0, len(tasks), per_chunk):
        chunk = tasks[c0:c0 + per_chunk]
        print("  -- chunk %d/%d: %d array(s), fresh pool of %d"
              % (c0 // per_chunk + 1, (len(tasks) + per_chunk - 1) // per_chunk,
                 len(chunk), a.workers), flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = {ex.submit(_create_one_random_array_star, t): t
                       for t in chunk}
            for f, t in futures.items():
                try:
                    f.result()
                    ok.append(t["name"])
                except Exception as exc:                        # noqa: BLE001
                    bad.append((t["name"], "%s: %s"
                                % (type(exc).__name__, str(exc)[:110])))

    # A clean return does NOT mean an array was built: run() prints
    # "Directory ... already exists" and returns at once when the folder is
    # there, so a stub left by a killed batch silently absorbs its index and
    # the future succeeds. That cost two arrays of a 36-array restart, both
    # reported as built. Verify the artefact, not the call.
    verified, missing, short = [], [], []
    for nm in ok:
        p = os.path.join(RESULTS_DIR, nm, "history.hf5")
        if not os.path.isfile(p):
            missing.append(nm)
            continue
        try:
            with pd.HDFStore(p, "r") as store:      # 'face' is the small table
                tmax = float(store.select("face", columns=["time"])["time"].max())
        except Exception as exc:                            # noqa: BLE001
            short.append((nm, "unreadable: %s" % type(exc).__name__))
            continue
        (verified if tmax >= a.t_end - 0.5 else short).append(
            nm if tmax >= a.t_end - 0.5 else (nm, "trace ends at t=%.2f" % tmax))
    print("\nbuilt %d of %d (%d verified to t>=%g)"
          % (len(ok), len(tasks), len(verified), a.t_end - 0.5))
    for nm, err in bad:
        print("   CRASHED   %-38s %s" % (nm, err))
    for nm in missing:
        print("   NO HISTORY %-38s" % nm)
    for nm, why in short:
        print("   TRUNCATED %-38s %s" % (nm, why))
    return 0


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------

def _score_one(name):
    try:
        find_best_timepoint_according_to_n_neighbors(name, reuse=True)
        return (name, None)
    except Exception as exc:                                    # noqa: BLE001
        return (name, "%s: %s" % (type(exc).__name__, str(exc)[:90]))


def cmd_score(a):
    members = pool_members()
    todo = [m for m in members
            if not os.path.isfile(os.path.join(RESULTS_DIR, m, TRACE))]
    print("pool: %d array(s) with a history; %d scored, %d to score"
          % (len(members), len(members) - len(todo), len(todo)))
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        done = 0
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for name, err in ex.map(_score_one, todo):
                done += 1
                if err:
                    print("   FAILED %-44s %s" % (name, err), flush=True)
                elif done % 10 == 0:
                    print("   scored %d/%d" % (done, len(todo)), flush=True)
    report_pool(members)
    return 0


def report_pool(members, keep=KEEP):
    for stage in STAGES:
        rows = [r for r in (read_trace(m, stage) for m in members) if r]
        usable = sorted((r for r in rows if not r["at_end"]),
                        key=lambda r: r["z2"])
        print("=" * 74)
        print("  %s: %d scored, %d usable, %d dropped (minimum at trace end)"
              % (stage, len(rows), len(usable), len(rows) - len(usable)))
        if not usable:
            continue
        allz = np.array([r["z2"] for r in usable])
        print("     pool: min %.1f  median %.1f  below 20: %d/%d"
              % (allz.min(), np.median(allz), int((allz < 20).sum()), allz.size))
        print("     BOUND (worst of the best %d): z2 = %.2f"
              % (min(keep, len(usable)), usable[:keep][-1]["z2"]))


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------

def pick_fold_free(history, row, log):
    """The chosen frame, or its best fold-free neighbour if it self-intersects.

    A self-intersecting frame is unusable as an initial condition and the
    neighbour-count selection knows nothing about folds, so every chosen frame
    is checked. The fallback prefers whichever side has the lower z^2 on the
    SAME trace that picked it, and any fallback is reported.
    """
    t_best = row["t"]
    if _frame_is_fold_free(history, t_best):
        return t_best, False
    log.append("      FOLDED at t=%.4f -- looking for a fold-free neighbour"
               % t_best)
    t_before, t_after = closest_fold_free_frames(history, t_best)
    cands = [t for t in (t_before, t_after) if t is not None]
    if not cands:
        raise RuntimeError("no fold-free frame anywhere in %s" % row["folder"])

    def z_at(t):
        return float(row["trace_z2"][int(np.argmin(np.abs(row["trace_t"] - t)))])

    best = min(cands, key=z_at)
    log.append("      -> t=%.4f (z2 %.2f) instead of t=%.4f (z2 %.2f)"
               % (best, z_at(best), t_best, row["z2"]))
    return best, True


def cmd_select(a):
    members = pool_members()
    chosen = {}
    for stage in STAGES:
        rows = [r for r in (read_trace(m, stage) for m in members) if r]
        usable = sorted((r for r in rows if not r["at_end"]),
                        key=lambda r: r["z2"])
        chosen[stage] = usable[:a.keep]
    print("chosen per stage (rank <- pool folder, z2, t):")
    for stage in STAGES:
        print("  %s" % stage)
        for r, row in enumerate(chosen[stage]):
            print("     %2d <- %-40s z2 %7.2f  t %6.2f"
                  % (r, row["folder"], row["z2"], row["t"]))
    overlap = ({r["folder"] for r in chosen[STAGES[0]]}
               & {r["folder"] for r in chosen[STAGES[1]]})
    print("  arrays chosen by BOTH stages: %d" % len(overlap))
    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    mapping = {}
    for stage in STAGES:
        mapping[stage] = []
        print("\n%s" % stage)
        for rank, row in enumerate(chosen[stage]):
            log, t0 = [], time.time()
            history = load_history_file(row["folder"])
            t_use, fell_back = pick_fold_free(history, row, log)
            out = initial_morphology_name(rank, stage)
            dst = os.path.join(RESULTS_DIR, out)
            if os.path.isdir(dst):
                # save_data_of_a_given_time_point writes INTO an existing
                # directory, which would leave a mix of two selections' files
                if not a.overwrite:
                    raise SystemExit(
                        "%s already exists; pass --overwrite to replace the "
                        "previous selection" % dst)
                shutil.rmtree(dst)
            save_data_of_a_given_time_point(history, t_use, out)
            for line in log:
                print(line, flush=True)
            print("   %-26s <- %-40s t=%8.4f  z2 %7.2f  (%.0fs)"
                  % (out, row["folder"], t_use, row["z2"], time.time() - t0),
                  flush=True)
            mapping[stage].append(dict(
                rank=rank, extract=out, source=row["folder"],
                pool_folder=row["folder"],
                run_folder=os.path.basename(chosen_array_run_name(rank, stage)),
                source_index=int(POOL.match(row["folder"]).group(1)),
                time_point=t_use, stage1_z2=row["z2"],
                used_fold_free_fallback=fell_back))
            del history

    out_path = os.path.join(seed_dir(), "chosen_arrays.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(mapping, fh, indent=2)
    print("\nwrote %s" % out_path)
    return 0


# --------------------------------------------------------------------------
# seed-data
# --------------------------------------------------------------------------

def cmd_seed_data(a):
    out_dir = seed_dir()
    par_dir = os.path.join(out_dir, "parameters")
    os.makedirs(par_dir, exist_ok=True)

    chosen = {}
    cpath = os.path.join(out_dir, "chosen_arrays.json")
    if os.path.isfile(cpath):
        with open(cpath) as fh:
            for stage, rows in json.load(fh).items():
                for r in rows:
                    chosen.setdefault(r["source_index"], []).append(
                        "%s#%d" % (stage, r["rank"]))

    rows, missing_params, mismatch = [], [], []
    for d in pool_members(with_history=False):
        i = int(POOL.match(d).group(1))
        folder = os.path.join(RESULTS_DIR, d)
        ppath = os.path.join(folder, "parameters.txt")
        p = read_params(ppath)
        if p:
            shutil.copyfile(ppath, os.path.join(par_dir, "%s.txt" % d))
        else:
            missing_params.append(d)

        expected = int(array_seed(i))
        recorded = p.get("seed")
        if recorded is not None and int(recorded) != expected:
            mismatch.append((d, recorded, expected))
        row = {"index": i, "folder": d, "seed_recorded": recorded,
               "seed_expected": expected,
               "seed_matches": recorded is not None and int(recorded) == expected,
               "has_history": os.path.isfile(os.path.join(folder, "history.hf5")),
               "selected_as": ";".join(chosen.get(i, [])),
               "kept_as": ""}
        for k in LIFT:
            if k != "seed":
                row[k] = p.get(k)
        row["trace_t_end"] = None
        for stage in STAGES:
            r = read_trace(d, stage)
            row["best_z2_%s" % stage] = round(r["z2"], 4) if r else None
            row["best_t_%s" % stage] = round(r["t"], 4) if r else None
            if r:
                row["trace_t_end"] = round(r["t_end"], 4)
        rows.append(row)

    if not rows:
        raise SystemExit("no pool folders found under %s" % RESULTS_DIR)
    with open(os.path.join(out_dir, "arrays.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "arrays.json"), "w") as fh:
        json.dump({"base_seed": int(DEFAULT_ARRAY_SEED),
                   "seed_rule": "numpy SeedSequence([base_seed, index])"
                                ".generate_state(1)[0]",
                   "n_arrays": len(rows), "arrays": rows}, fh, indent=2)
    print("wrote %s" % out_dir)
    print("  arrays: %d   with history: %d   parameters copied: %d"
          % (len(rows), sum(r["has_history"] for r in rows),
             len(rows) - len(missing_params)))
    print("  seed cross-check: %d match, %d MISMATCH, %d without parameters.txt"
          % (sum(r["seed_matches"] for r in rows), len(mismatch),
             len(missing_params)))
    for d, rec, exp in mismatch[:10]:
        print("     MISMATCH %-44s recorded %s, expected %s" % (d, rec, exp))
    return 1 if mismatch else 0


# --------------------------------------------------------------------------
# prune
# --------------------------------------------------------------------------

def cmd_prune(a):
    out_dir = seed_dir()
    trace_dir = os.path.join(out_dir, "score_traces")
    os.makedirs(trace_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chosen_arrays.json")) as fh:
        sel = json.load(fh)
    winners = {r["pool_folder"] for rows in sel.values() for r in rows}
    if len(winners) != len(STAGES) * a.keep:
        raise SystemExit("expected %d winners, found %d -- refusing to run"
                         % (len(STAGES) * a.keep, len(winners)))

    members = pool_members(with_history=False)
    victims = [d for d in members if d not in winners]
    print("pool %d, winners %d, to delete %d"
          % (len(members), len(winners), len(victims)))

    # copy the small files for EVERY array, winners included: one place to look
    copied = absent = 0
    for d in members:
        for fn in SMALL:
            src = os.path.join(RESULTS_DIR, d, fn)
            if not os.path.isfile(src):
                absent += 1
                continue
            dst = os.path.join(trace_dir, "%s.%s" % (d, fn.rsplit(".", 1)[1]))
            if not (os.path.isfile(dst)
                    and os.path.getsize(dst) == os.path.getsize(src)):
                shutil.copyfile(src, dst)
            copied += 1
    print("small files copied: %d (absent at source: %d) -> %s"
          % (copied, absent, trace_dir))

    freed = 0
    if not a.execute:
        for d in victims:
            freed += dir_bytes(os.path.join(RESULTS_DIR, d))
        print("\nDRY RUN. Would delete %d folder(s), freeing %.1f GB."
              % (len(victims), freed / 1e9))
        return 0

    deleted = skipped = 0
    for d in victims:
        src_npz = os.path.join(RESULTS_DIR, d, TRACE)
        if os.path.isfile(src_npz) and not os.path.isfile(
                os.path.join(trace_dir, "%s.npz" % d)):
            print("   SKIP (trace not preserved): %s" % d, flush=True)
            skipped += 1
            continue
        p = os.path.join(RESULTS_DIR, d)
        freed += dir_bytes(p)
        shutil.rmtree(p)
        deleted += 1
        if deleted % 50 == 0:
            print("   deleted %d/%d (%.0f GB)" % (deleted, len(victims),
                                                  freed / 1e9), flush=True)
    print("\ndeleted %d folder(s), skipped %d, freed %.1f GB"
          % (deleted, skipped, freed / 1e9))
    return 0


# --------------------------------------------------------------------------
# finalise
# --------------------------------------------------------------------------

def cmd_finalise(a):
    """Rename the winners by rank and gather them under the results folder.

    After this the pool index is gone from the folder name, so the mapping is
    written back into chosen_arrays.json (``run_folder`` / ``pool_folder``) and
    arrays.csv (``kept_as``). The files INSIDE each run keep the old name:
    parameters.txt and debug.log are the verbatim record of the run() call and
    are not rewritten.
    """
    out_dir = seed_dir()
    cpath = os.path.join(out_dir, "chosen_arrays.json")
    with open(cpath) as fh:
        chosen = json.load(fh)

    plan, missing = [], []
    for stage, rows in chosen.items():
        for r in rows:
            src = os.path.join(RESULTS_DIR, r["pool_folder"])
            dst = os.path.join(RESULTS_DIR, chosen_array_run_name(r["rank"], stage))
            if not os.path.isdir(src):
                if os.path.isdir(dst):
                    continue                       # already finalised
                missing.append(r["pool_folder"])
                continue
            if os.path.exists(dst):
                raise SystemExit("target already exists: %s" % dst)
            plan.append((stage, r["rank"], src, dst))
    if missing:
        raise SystemExit("source run folder(s) missing: %s" % missing[:5])
    if not plan:
        print("nothing to do: every winner is already in place.")
        return 0
    for stage, rank, src, dst in plan:
        print("  %-7s #%d  %-40s -> %s"
              % (stage, rank, os.path.basename(src),
                 os.path.relpath(dst, RESULTS_DIR)))
    if not a.execute:
        print("\nDRY RUN. %d folder(s) would move." % len(plan))
        return 0

    os.makedirs(os.path.join(RESULTS_DIR, INITIAL_CONFIG_RUNS), exist_ok=True)
    for _stage, _rank, src, dst in plan:
        os.rename(src, dst)
    bad = [d for _s, _r, _src, d in plan
           if not os.path.isfile(os.path.join(d, "history.hf5"))]
    if bad:
        raise SystemExit("history.hf5 missing after the move: %s" % bad)
    print("\nmoved %d run(s); history.hf5 present in all of them" % len(plan))

    by_pool = {}
    for stage, rows in chosen.items():
        for r in rows:
            r["run_folder"] = os.path.basename(chosen_array_run_name(r["rank"], stage))
            by_pool[r["pool_folder"]] = r["run_folder"]
    with open(cpath, "w") as fh:
        json.dump(chosen, fh, indent=2)
    csv_path = os.path.join(out_dir, "arrays.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        cols = list(rows[0].keys())
        if "kept_as" not in cols:
            cols.insert(cols.index("selected_as") + 1, "kept_as")
        for row in rows:
            row["kept_as"] = by_pool.get(row["folder"], "")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print("recorded the mapping in chosen_arrays.json and arrays.csv")
    return 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="simulate the pool")
    b.add_argument("--arrays", type=int, nargs=2, default=[0, 9],
                   metavar=("FIRST", "LAST"), help="inclusive index range")
    b.add_argument("--workers", type=int, default=6,
                   help="6, not 12: the results drive is a spinning disk")
    b.add_argument("--division-area", dest="division_area", type=float,
                   default=DIVISION_AREA)
    b.add_argument("--t-end", dest="t_end", type=float, default=T_END)
    b.add_argument("--name-suffix", dest="name_suffix", default=NAME_SUFFIX)
    b.add_argument("--arrays-per-worker", dest="arrays_per_worker", type=int,
                   default=ARRAYS_PER_WORKER,
                   help="0 disables worker recycling")
    b.add_argument("--max-wall-seconds", dest="max_wall_seconds", type=float,
                   default=MAX_WALL_SECONDS)
    b.add_argument("--min-progress-rate", dest="min_progress_rate", type=float,
                   default=MIN_PROGRESS_RATE)
    b.add_argument("--no-guards", dest="guards", action="store_false")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score every frame of every array")
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(func=cmd_score)

    e = sub.add_parser("select", help="keep the best ten per stage, extract them")
    e.add_argument("--keep", type=int, default=KEEP)
    e.add_argument("--overwrite", action="store_true",
                   help="replace existing extracts instead of refusing")
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=cmd_select)

    d = sub.add_parser("seed-data", help="parameters + seed cross-check")
    d.set_defaults(func=cmd_seed_data)

    p = sub.add_parser("prune", help="keep the traces, delete the non-winners")
    p.add_argument("--keep", type=int, default=KEEP)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_prune)

    f = sub.add_parser("finalise", help="rename the winners by rank and gather")
    f.add_argument("--execute", action="store_true")
    f.set_defaults(func=cmd_finalise)

    a = ap.parse_args()
    print("results root: %s" % RESULTS_DIR)
    print("this step's folder: %s\n" % os.path.join(RESULTS_DIR,
                                                    INITIAL_CONFIG_DIR))
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
