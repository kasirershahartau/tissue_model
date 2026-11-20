"""The live state a resume needs and the history archive does not carry.

WHY THIS EXISTS
---------------
``run(seed=...)`` makes a FRESH run reproducible: both generators are seeded
from the recorded seed and the trajectory replays bit for bit. A RESUME could
not, because a run is more than its sheet. Reloading frame ``t0`` from
``history.hf5`` restores the geometry exactly (measured: vertex positions, edge
topology, edge row order and the ``order`` column are bit-identical in every
frame), but the solver, the event manager and the model also carry state that
has no column in the archive:

  * **Both RNG streams, mid-stream.** Seeding from the top of the stream at a
    resume is not the state the original run was in at ``t0`` -- it had already
    drawn millions of values. The virtual-vertex shuffles alone consume ~100
    words per step.
  * **The event order.** ``EventManager`` holds the SHUFFLED order of the next
    step's events, plus its clock. Rebuilding the manager gives the
    construction order, not the order the original run was about to use.
  * **The countdowns.** ``random_forces`` re-draws the per-cell mechanical
    parameters every ``wait_time`` units (``InnerEarModel.time_to_random``) and
    the random division rule fires every ``interval`` units
    (``TopologicalEventsHandler.time_to_division``, with its own PCG64 stream).
    Both reset to a full period on a resume, so the redraw phase shifts.
  * **The adaptive step.** The solver's ``dt`` shrinks on a rejected step and
    ratchets back. A resume restarts at the nominal dt, so it takes a different
    first step from the one the original run took at ``t0``.
  * **The save phase and the steady-state streak.** Snapshot times would land
    on a different grid, and a run that had already accumulated N steady steps
    would need ``steady_state_min_steps`` more.
  * **The cell labels and the measured length unit** -- see
    ``capture_identity`` and ``capture_model``.

So each recorded frame gets one line of a JSONL sidecar beside the archive
(``history_resume.jsonl``), and a resume restores from the line at its resume
time. ~8 KB per frame (576 KB for a 71-frame run, against 50 MB of history);
most of it is the two generators' 624- and 625-word states, which are
high-entropy and do not compress.

MEASURED, cross-process: an 8x8 array built to t=7 and resumed from t=3.51 is
BIT-IDENTICAL over all 36 remaining frames, and the full model (differentiation
+ quasi-static mechanics) resumed from t=2 is bit-identical over all 16 frames
and stops at the same steady-state frame. With the sidecar deleted the array
resume ends with 64 cells where the original has 82.

WHAT IS NOT IN HERE, DELIBERATELY
---------------------------------
Anything the archive already holds: the LI levels, the geometry, and the
per-cell mechanical columns, which ``run`` puts back from the loaded frame
after ``InnerEarModel.__init__`` has reset them (``_restore_derived_columns``).
This file is only for state that would otherwise be lost.

OLDER RUNS. Every run recorded before this existed has no sidecar. Resuming one
still works and still does NOT reproduce the original continuation; the resume
logs a warning saying so rather than failing. Compaction
(``compact_history_files.py``) does not touch the sidecar, and it does not have
to: a resume time is snapped to a surviving archive stamp, and the sidecar's
line for that frame is still there.

NOT ALWAYS WANTED. ``simulate(restore_resume_state=False)`` continues from the
same frame with a FRESH stream, which is the point of
``run_model.restore_array_run`` (replace a folded tail) and of recovering a run
that died at the dt floor -- reproducing that continuation would reproduce the
crash. That mode is reproducible too, from the run's own recorded seed.
"""
import base64
import json
import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)

#: Sidecar format version. Bumped if the schema changes incompatibly; a line
#: written by a newer format is ignored rather than mis-restored.
FORMAT = 1

_SUFFIX = "_resume.jsonl"

#: ``__name__`` given to the one-shot differentiation the solver appends in
#: quasi-static mode. The factory's inner function is called ``differentiation``
#: -- the same name as the per-step one appended at setup -- so without a
#: distinct name the two are indistinguishable in a recorded queue, and the dt
#: baked into the one-shot (``current_t - last_differentiation_t``, not the
#: nominal step) would be lost on a restore.
QUASI_STATIC_DIFFERENTIATION = "quasi_static_differentiation"


def sidecar_path(history_file):
    """Path of the resume sidecar beside the archive at ``history_file``.

    ``.../run/history.hf5`` -> ``.../run/history_resume.jsonl``. Derived from
    the archive's own name (rather than fixed per folder) so an archive and its
    sidecar cannot be paired up wrongly.
    """
    if not history_file:
        return None
    return os.path.splitext(str(history_file))[0] + _SUFFIX


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
def _b64(values):
    """Pack a sequence of uint32 words into base64.

    Both generators' states are a few hundred 32-bit words; as a JSON list of
    integers they cost ~4x what they do packed, and the file is written once
    per recorded frame.
    """
    return base64.b64encode(
        np.asarray(values, dtype="<u4").tobytes()).decode("ascii")


def _unb64(text):
    return np.frombuffer(base64.b64decode(text), dtype="<u4")


def capture_rngs():
    """The state of BOTH generators this project draws from.

    numpy's global MT19937 carries the draws; Python's stdlib ``random``
    carries the event ORDER (``EventManager.update`` shuffles with it). See the
    seeding note in ``run_model.run`` -- missing either one is enough to make a
    run irreproducible.
    """
    npy = np.random.get_state()
    pyr = random.getstate()
    return {
        "numpy": {"kind": npy[0], "keys": _b64(npy[1]), "pos": int(npy[2]),
                  "has_gauss": int(npy[3]), "cached_gaussian": float(npy[4])},
        "python": {"version": int(pyr[0]), "keys": _b64(pyr[1]),
                   "gauss_next": pyr[2]},
    }


def restore_rngs(state):
    """Put both generators back where ``capture_rngs`` found them."""
    notes = []
    npy = state.get("numpy")
    if npy:
        if npy.get("kind") != np.random.get_state()[0]:
            notes.append("numpy bit generator is %s but the sidecar recorded "
                         "%s; NOT restored" % (np.random.get_state()[0],
                                               npy.get("kind")))
        else:
            np.random.set_state((
                npy["kind"], _unb64(npy["keys"]).astype("uint32"),
                int(npy["pos"]), int(npy["has_gauss"]),
                float(npy["cached_gaussian"]),
            ))
            notes.append("numpy MT19937 restored at stream position %d"
                         % int(npy["pos"]))
    pyr = state.get("python")
    if pyr:
        random.setstate((
            int(pyr["version"]),
            tuple(int(v) for v in _unb64(pyr["keys"])),
            pyr["gauss_next"],
        ))
        notes.append("stdlib random restored (event order)")
    return notes


# --------------------------------------------------------------------------
# the event manager's queues
# --------------------------------------------------------------------------
def _entry(behavior, kwargs):
    rec = {"name": behavior.__name__}
    simple = {k: v for k, v in (kwargs or {}).items()
              if isinstance(v, (int, float, str, bool, type(None)))}
    if simple:
        rec["kwargs"] = simple
    # The quasi-static one-shot differentiation carries the dt it was built
    # with on the function object (see solvers.IVPSolver.solve), because it is
    # baked into the closure rather than passed as a kwarg.
    dt = getattr(behavior, "_resume_dt", None)
    if dt is not None:
        rec["closure_dt"] = float(dt)
    return rec


def capture_manager(manager):
    """The clock and BOTH queues, in order.

    ``current`` is the (already shuffled) order the next step will execute in;
    ``next`` is what has been appended for the step after. Both are needed: at
    the initial record of a run the events sit in ``next`` and only ``wait`` is
    in ``current``.
    """
    if manager is None:
        return None
    return {
        "clock": int(manager.clock),
        "current": [_entry(b, kw) for b, kw in manager.current],
        "next": [_entry(b, kw) for b, kw in manager.next],
    }


def restore_manager(manager, state, inner=None, atoh_by_repressor=False):
    """Rebuild ``manager.current`` / ``manager.next`` in the recorded order.

    A queue entry holds a CLOSURE, which cannot be serialised, so the recorded
    order is replayed by NAME against the behaviours the freshly built manager
    already holds -- every behaviour in this model re-appends itself, so the
    live queues only ever contain those same objects (plus ``wait``, and the
    quasi-static one-shot, which is rebuilt from its recorded dt).

    Entries that share a name are assigned in construction order. That is exact
    except for a run with SEVERAL ablated cells, where the per-cell ``ablation``
    closures are interchangeable in effect (each tags its own cell, in one step,
    by column write) but could be permuted among themselves.

    A behaviour the manager holds that the record does not mention is dropped
    (and logged) rather than re-appended -- see the note at the bottom.
    """
    if manager is None or not state:
        return ["no manager state recorded"]
    from tyssue.behaviors.event_manager import wait as _wait

    pool = {}
    for behavior, kwargs in list(manager.current) + list(manager.next):
        pool.setdefault(behavior.__name__, []).append((behavior, kwargs))
    pool.setdefault("wait", [(_wait, {"face_id": -1, "n_steps": 1})])

    notes = []
    taken = {}

    def build(entry):
        name = entry["name"]
        kwargs = dict(entry.get("kwargs") or {})
        if name == QUASI_STATIC_DIFFERENTIATION:
            if inner is None:
                notes.append("cannot rebuild the pending %s (no model)" % name)
                return None
            behavior = (inner.lateral_inhibition_model
                        .get_length_dependent_differentiation_function(
                            dt=float(entry.get("closure_dt", 0.0)),
                            quasi_static=True,
                            atoh_by_repressor=atoh_by_repressor))
            behavior.__name__ = QUASI_STATIC_DIFFERENTIATION
            behavior._resume_dt = float(entry.get("closure_dt", 0.0))
            return (behavior, kwargs)
        candidates = pool.get(name)
        if not candidates:
            notes.append("recorded event %r is not in this run's manager; "
                         "skipped" % name)
            return None
        i = taken.get(name, 0)
        taken[name] = i + 1
        if i >= len(candidates):
            notes.append("recorded %d %r event(s) but the manager holds %d; "
                         "reusing the last" % (i + 1, name, len(candidates)))
            i = len(candidates) - 1
        behavior, built_kwargs = candidates[i]
        return (behavior, kwargs or dict(built_kwargs))

    for queue, key in ((manager.current, "current"), (manager.next, "next")):
        queue.clear()
        for entry in state.get(key, ()):
            item = build(entry)
            if item is not None:
                queue.append(item)

    # A behaviour the freshly built manager holds that the recorded queues do
    # NOT mention is DROPPED, and said so in the log. It is a one-shot that
    # already fired in the original run: the per-cell ``ablation`` behaviours
    # tag their cell once and never re-append, and in quasi-static mode so does
    # the differentiation appended at setup (the solver appends its own
    # one-shots afterwards). Re-appending those would re-tag cells that are
    # already dying and inject a differentiation step the original run never
    # took. The cost is that a resume which switches a behaviour ON that the
    # original run did not have would not get it -- that resume is already a
    # different simulation, which is what the log line is for. ``wait`` is
    # excluded: it is a no-op placeholder the manager seeds itself with.
    for name, candidates in pool.items():
        if name == "wait":
            continue
        n_unused = len(candidates) - taken.get(name, 0)
        if n_unused > 0:
            notes.append("%d %r event(s) are not in the recorded queue "
                         "(already fired before the resume point); dropped"
                         % (n_unused, name))

    manager.clock = int(state.get("clock", 0))
    notes.append("event queues restored (clock=%d, %d current, %d next): %s"
                 % (manager.clock, len(manager.current), len(manager.next),
                    ", ".join(b.__name__ for b, _ in manager.current) or "-"))
    return notes


# --------------------------------------------------------------------------
# the model's own countdowns
# --------------------------------------------------------------------------
def capture_model(inner):
    """The timers, private streams and measured constants that live on the
    model rather than on the sheet.

    ``time_to_random`` paces the ``random_forces`` re-draw;
    ``time_to_division`` and ``_division_rng`` pace the random division rule.
    Each exists only when the corresponding feature is switched on.

    ``length_normalization_factor`` (L0) is here because it is MEASURED, not
    configured: a resume inherits it through ``l0.txt``, which recorded it to
    12 significant digits, so the resumed run's L0 differed from the original's
    in the last bits. Nothing depends on that until the first
    ``set_random_parameters`` draw (A0 = N(mean, std) * L0**2) -- and then
    every preferred area in the tissue is slightly off and the trajectories
    separate. Measured: a resume agreed with the uninterrupted run frame for
    frame until the re-draw at t=5 and diverged from exactly there.
    """
    out = {}
    if inner is None:
        return out
    if hasattr(inner, "time_to_random"):
        out["time_to_random"] = float(inner.time_to_random)
    if getattr(inner, "length_normalization_factor", None) is not None:
        out["length_normalization_factor"] = float(
            inner.length_normalization_factor)
    handler = getattr(inner, "topological_events_handler", None)
    if handler is not None:
        if hasattr(handler, "time_to_division"):
            out["time_to_division"] = float(handler.time_to_division)
        rng = getattr(handler, "_division_rng", None)
        if rng is not None:
            out["division_rng"] = rng.bit_generator.state
    return out


def restore_model(inner, state):
    notes = []
    if inner is None or not state:
        return notes
    l0 = state.get("length_normalization_factor")
    if l0 is not None:
        was = float(getattr(inner, "length_normalization_factor", float("nan")))
        inner.length_normalization_factor = float(l0)
        li = getattr(inner, "lateral_inhibition_model", None)
        if li is not None:
            # It holds its own copy, taken at construction.
            li.length_normalization_factor = float(l0)
        if repr(was) != repr(float(l0)):
            notes.append("length unit L0 restored exactly: %r (was %r, the "
                         "value l0.txt could carry)" % (float(l0), was))
    if "time_to_random" in state and hasattr(inner, "time_to_random"):
        inner.time_to_random = float(state["time_to_random"])
        notes.append("random_forces countdown restored to %.6g"
                     % inner.time_to_random)
    handler = getattr(inner, "topological_events_handler", None)
    if handler is not None:
        if "time_to_division" in state and hasattr(handler, "time_to_division"):
            handler.time_to_division = float(state["time_to_division"])
            notes.append("random-division countdown restored to %.6g"
                         % handler.time_to_division)
        rng = getattr(handler, "_division_rng", None)
        if rng is not None and "division_rng" in state:
            try:
                rng.bit_generator.state = state["division_rng"]
                notes.append("random-division RNG restored")
            except (ValueError, KeyError, TypeError) as exc:
                notes.append("random-division RNG NOT restored (%s)" % exc)
    return notes


# --------------------------------------------------------------------------
# cell labels
# --------------------------------------------------------------------------
def capture_identity(eptm):
    """``unique_id`` and the counter that issues the next one.

    tyssue's ``Epithelium.__init__`` OVERWRITES ``unique_id`` with the row
    index and resets ``specs['face']['unique_id_max']`` to the row count
    (core/objects.py:157) -- and a resume loads its sheet through that
    constructor, so the archived labels are gone before anything can read them
    and the next daughter would be numbered from a different base.

    The labels do NOT affect the trajectory: the contact matrix and the state
    vectors it multiplies are both indexed by this same column, so any
    relabelling cancels out. But they key ``labels.npy``,
    ``contact_matrix.npy`` and ``cell_info.pkl``, so without this a resumed run
    hands downstream analysis a different labelling of the same cells. Faces
    only -- nothing reads the vert / edge ``unique_id``.
    """
    if eptm is None or "unique_id" not in eptm.face_df.columns:
        return None
    return {
        "index": _b64(eptm.face_df.index.to_numpy()),
        "unique_id": _b64(eptm.face_df["unique_id"].to_numpy()),
        "unique_id_max": int(eptm.specs.get("face", {}).get(
            "unique_id_max", eptm.face_df.shape[0])),
    }


def restore_identity(eptm, state):
    if eptm is None or not state:
        return []
    if "unique_id" not in eptm.face_df.columns:
        return ["sheet has no unique_id column; labels not restored"]
    index = _unb64(state["index"])
    live = eptm.face_df.index.to_numpy()
    if live.size != index.size or not np.array_equal(
            live.astype("<u4", copy=False), index):
        return ["face index differs from the recorded frame; unique_id NOT "
                "restored (%d live vs %d recorded rows)"
                % (live.size, index.size)]
    # Keep the dtype: HistoryHdf5 captured the column dtypes when it was built
    # and record() raises on any change.
    eptm.face_df["unique_id"] = _unb64(state["unique_id"]).astype(
        eptm.face_df["unique_id"].dtype)
    eptm.specs.setdefault("face", {})["unique_id_max"] = int(
        state["unique_id_max"])
    return ["cell labels restored (unique_id_max=%d)"
            % state["unique_id_max"]]


# --------------------------------------------------------------------------
# capture / apply
# --------------------------------------------------------------------------
def capture(solver, loop_state=None):
    """One sidecar record: both RNGs, the event queues, the model timers, the
    cell labels and the solver's loop scalars."""
    state = {"format": FORMAT}
    state["rngs"] = capture_rngs()
    state["manager"] = capture_manager(getattr(solver, "manager", None))
    state["model"] = capture_model(getattr(solver, "inner", None))
    state["identity"] = capture_identity(getattr(solver, "eptm", None))
    state["loop"] = dict(loop_state or {})
    return state


def apply(solver, state, atoh_by_repressor=False):
    """Restore everything in ``state`` onto a freshly built solver/manager.

    Call AFTER the manager is fully populated and BEFORE ``solve``. The loop
    scalars are stashed on the solver for ``solve`` to pick up once it has
    derived ``initial_dt`` / ``dt_min`` from its nominal arguments -- a resume
    restores the adaptive dt, NOT the nominal one, so the floor and the ratchet
    ceiling stay as the caller asked.
    """
    notes = []
    if not state:
        return notes
    fmt = int(state.get("format", 0))
    if fmt > FORMAT:
        return ["resume state is format %d, this code reads %d; ignored"
                % (fmt, FORMAT)]
    notes += restore_rngs(state.get("rngs") or {})
    notes += restore_manager(getattr(solver, "manager", None),
                             state.get("manager"),
                             inner=getattr(solver, "inner", None),
                             atoh_by_repressor=atoh_by_repressor)
    notes += restore_model(getattr(solver, "inner", None),
                           state.get("model") or {})
    notes += restore_identity(getattr(solver, "eptm", None),
                              state.get("identity"))
    loop = state.get("loop") or {}
    if loop:
        solver._resume_loop_state = dict(loop)
        notes.append("solver loop state restored (%s)"
                     % ", ".join("%s=%.6g" % (k, v) for k, v in
                                 sorted(loop.items())))
    return notes


# --------------------------------------------------------------------------
# file I/O
# --------------------------------------------------------------------------
def write(path, time, state):
    """Append one record. Never raises: losing a sidecar line must not kill a
    simulation that is otherwise fine."""
    if not path:
        return
    rec = dict(state)
    rec["t"] = float(time)
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":"), default=float))
            fh.write("\n")
    except Exception as exc:                                   # noqa: BLE001
        log.warning("could not write resume state at t=%g to %s: %s",
                    time, path, exc)


def _iter_records(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue              # a half-written tail line; skip it
            if "t" in rec:
                yield rec


def read(path, time, tol=1e-9):
    """The record AT ``time``, or None.

    Exact (within ``tol``), deliberately, not merely nearest. The caller has
    already snapped its resume time to a real archive stamp
    (``run_model._nearest_archive_time``) and the same float is written here
    through ``repr``, so the matching line is there to the bit when it exists
    at all. Taking the nearest line instead would silently pair frame N's
    sheet with frame N-1's RNG state in the one case where they disagree: a
    process killed between ``history.record`` and this file's append, which
    leaves a frame with no line of its own. Returning None instead sends the
    caller down the "no resume state" path, which says so loudly.

    ``<=`` keeps the LAST record at a repeated stamp, matching ``record()``,
    which replaces the row at an existing time.
    """
    if not path or not os.path.isfile(path):
        return None
    best = None
    for rec in _iter_records(path):
        if best is None or abs(rec["t"] - time) <= abs(best["t"] - time):
            best = rec
    if best is None:
        return None
    if abs(best["t"] - time) > tol:
        log.warning("no resume state at t=%r in %s; the nearest is t=%r, "
                    "which belongs to a different frame", time, path,
                    best["t"])
        return None
    return best


def truncate(path, time, tol=1e-9):
    """Drop every record after ``time`` and de-duplicate what is left.

    The mirror of ``_rewrite_history_for_resume`` for the sidecar: the resumed
    run appends from ``time`` onward, so records past it belong to the
    abandoned tail. De-duplication keeps the LAST record at each stamp, so
    repeated resumes don't grow the file.
    """
    if not path or not os.path.isfile(path):
        return 0
    kept = {}
    order = []
    for rec in _iter_records(path):
        if rec["t"] > float(time) + tol:
            continue
        key = round(rec["t"], 12)
        if key not in kept:
            order.append(key)
        kept[key] = rec
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for key in order:
            fh.write(json.dumps(kept[key], separators=(",", ":"),
                                default=float))
            fh.write("\n")
    os.replace(tmp, path)
    return len(order)
