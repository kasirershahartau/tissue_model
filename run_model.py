import numpy as np
import atexit
import hashlib
import logging
import os, shutil, sys
from tyssue import HistoryHdf5
from matplotlib import pyplot as plt
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel, ContractilityPerimeterElasticity, BoundaryBending
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility
from post_processing import (extract_model_mechanics,
                             compare_pooled_model_mechanics_to_experiments,
                             compare_differentiation_to_experiments,
                             compare_full_model_differentiation_to_experiments,
                             create_gif_safe,
                             random_array_name, initial_morphology_name, _STAGE_SHEET_SUFFIX,
                             RESULTS_DIR)

# --------------------------------------------------------------------------- #
# Debug logging                                                               #
# --------------------------------------------------------------------------- #
# Loggers that the periodic / inner-ear pipeline writes to. These names are
# the dotted module paths Python's logging library uses for
# ``logging.getLogger(__name__)`` from inside each file. Keeping the list as
# a module-level constant makes attach AND detach (in the run() teardown)
# use the same target set.
_DEBUG_LOG_TARGETS = (
    "",  # root — catches everything via propagation
    "tyssue",
    "tyssue.solvers.viscous",  # the tyssue EulerSolver / our IVPSolver log
    "solvers",
    "topological_events",
    "virtual_sheet",
    "periodic_sheet",
    "inner_ear_model",
    "lateral_inhibition_model",
)


class _FlushFileHandler(logging.FileHandler):
    """A ``FileHandler`` that calls ``self.flush()`` after every emitted
    record so the log file stays current. With a vanilla
    ``FileHandler``, Python's internal buffer holds onto the most
    recent N records and only writes them out at shutdown — if the
    process dies via uncaught exception that's still fine (interpreter
    shutdown flushes), but on a hard crash (segfault in a C extension,
    Ctrl+C while inside a numpy routine, ...) the tail of the log can
    be lost. Per-record flushing makes the file usable as a running
    transcript.
    """

    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            # Never let logging hide a real failure
            pass


def _enable_debug_log(log_path, append=False):
    """Attach a DEBUG-level ``_FlushFileHandler`` to all loggers in
    :data:`_DEBUG_LOG_TARGETS`, writing to ``log_path``.

    Returns the handler so the caller can flush / close / detach it in
    a ``finally`` block. Also registers an ``atexit`` callback that
    flushes the file on interpreter exit, so a partial run still
    leaves a usable log even if the caller forgot to clean up.

    Parameters
    ----------
    append : bool, default False
        When ``True``, open the log file in append mode so the
        original failed-run transcript is preserved and the resumed
        run's transcript lands underneath it. ``False`` (the default)
        truncates the file — the normal "fresh run" behaviour.
    """
    handler = _FlushFileHandler(
        log_path, mode=("a" if append else "w"), encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
        "%(filename)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Set the level on every target logger so their DEBUG/INFO records are
    # created, but attach the handler to the ROOT logger ONLY. Every record
    # propagates UP to root, so a single handler there captures each one
    # exactly once. Attaching the SAME handler to several loggers that lie in
    # one propagation chain — e.g. "tyssue.solvers.viscous" -> "tyssue" -> root,
    # all present in _DEBUG_LOG_TARGETS — made each such record be emitted once
    # per handler along the chain (3 duplicate lines for every solver warning,
    # 2 for a virtual_sheet/topological_events record).
    for logger_name in _DEBUG_LOG_TARGETS:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    # Keep matplotlib's chatty DEBUG/INFO (font-manager "findfont" spam, backend
    # probing, ...) OUT of the file: root is at DEBUG and matplotlib propagates
    # to it, so without this the log fills with matplotlib noise. WARNING+ from
    # matplotlib still gets through.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    root_logger = logging.getLogger("")
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)

    def _final_flush():
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    atexit.register(_final_flush)

    return handler


def _disable_debug_log(handler):
    """Detach ``handler`` from every logger in
    :data:`_DEBUG_LOG_TARGETS` (the symmetric undo of
    :func:`_enable_debug_log`) and close the underlying file. Safe to
    call even if some loggers don't currently hold the handler."""
    for logger_name in _DEBUG_LOG_TARGETS:
        lg = logging.getLogger(logger_name)
        if handler in lg.handlers:
            lg.removeHandler(handler)
    try:
        handler.flush()
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass

def initialize_sheet(nx, ny, distx=1, disty=1, max_bond_length=0.5, min_bond_length=0.05, periodic=True):
    sheet = VirtualSheet.planar_virtual_sheet_2d(
        'basic2D',  # a name or identifier for this sheet
        nx=nx,  # approximate number of cells on the x axis
        ny=ny,  # approximae number of cells along the y axis
        distx=distx,  # distance between 2 cells along x
        disty=disty,  # distance between 2 cells along y
        minimal_bond_length=min_bond_length,
        maximal_bond_length=max_bond_length,
        periodic=periodic
    )
    return sheet


def load_sheet_from_file(initial_sheet_name, two_dim=True, time_point=None,
                         force_periodic_box=None):
    """Load a :class:`VirtualSheet` from a saved HDF5 history archive.

    Parameters
    ----------
    initial_sheet_name : str
        Path prefix (no ``.hf5`` extension) to the archive.
    two_dim : bool, default True
        Whether to drop the ``z`` / ``sz`` / ... columns and put the
        sheet back into 2D mode.
    time_point : float, optional
        Specific time stamp to retrieve. When ``None`` (the default)
        the LAST recorded time point is used — the historical
        behaviour. Pass an explicit time when resuming a partially-
        completed run from a chosen moment instead of just the tail.
    force_periodic_box : (Lx, Ly), optional
        Fallback box dimensions used only when the archive lacks the
        ``_periodic_flag`` metadata (legacy archives written before
        periodicity was stashed on every snapshot). Lets a periodic
        run be resumed / forked from such an archive without loading
        it as non-periodic. A stored flag always takes precedence.
    """
    history = HistoryHdf5.from_archive(os.path.join(initial_sheet_name, "history.hf5"), eptm_class=VirtualSheet)
    if time_point is None:
        time_point = np.max(history.time_stamps)
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history(two_dim, force_periodic_box=force_periodic_box)
    sheet.initiate_edge_order()
    return sheet


# Historical default for the Atoh1 Hill delta half-max (a.k.a. patoh). Shared by
# run() (its ``atoh_sensitivity`` default) and _short_run_folder_name (only a
# NON-default value is folded into the folder hash) so the two never drift.
_DEFAULT_ATOH_SENSITIVITY = 0.377
# Historical mechanical steady-state cutoff. Runs made at this value keep their
# existing folder names (and cached results); a run at any OTHER threshold gets a
# DISTINCT folder so reuse_existing_run can't wrongly reuse a differently-relaxed
# archive at the same (sheet, parameters).
_DEFAULT_QUASI_STATIC_THRESHOLD = 0.01


def _short_run_folder_name(initial_sheet_name, gammaSC, gammaHC_ratio,
                           alphaHC_ratio, psigma, ablated_cells=None,
                           atoh_sensitivity=_DEFAULT_ATOH_SENSITIVITY,
                           shape_index=0.0,
                           quasi_static_threshold=_DEFAULT_QUASI_STATIC_THRESHOLD,
                           line_tension=None,
                           hc_shape_index=None, sc_shape_index=None,
                           bending=None, preferred_area=None, stress_shift=0.0):
    """Build a SHORT, deterministic, collision-resistant results-folder name.

    The previous scheme embedded the full ``initial_sheet_name`` verbatim::

        periodic_from<initial>_gammaSC-..._gammaHC_ratio-..._..._psigma-...

    When one run was forked from another — the ablation run forks from the
    un-ablated run, and the parameter-fit runs fork from the fitted-morphology
    sheets — the parent's already-long name became part of the child's, so the
    names grew without bound and eventually overran the Windows 260-char path
    limit.

    Here the UNBOUNDED parts (the initial-sheet identity and the ablation
    list) are folded into a short hash, while the fitted parameters stay
    human-readable. The result has a small, fixed length no matter how the run
    was produced, and is still unique per
    ``(initial sheet, parameters, ablation)`` combination.

    gammaSC is formatted at ``%.4f`` (everything else stays ``%.2f``). The v2
    contractility fit sweeps gammaSC over ~0.002-0.018, where two decimals
    collapsed three grid points onto the same token: only ``preferred_area``
    (itself a function of gammaSC) kept the folders apart, which is far too
    thin a margin to rely on. Four decimals separates gammaSC values down to
    0.0001 in the name AND in the hash.

    NOTE this changes every fit_* folder name, so previously-computed runs are
    no longer matched by ``reuse_existing_run``. That is intended here: the v2
    model (p0=0, no bending) shares no parameter point with the old runs anyway,
    since their names carry p0hc/p0sc and bending tags.

    NOTE ALSO ``find_mechanical_parameters`` still caches objective values at
    ``round(value, 2)`` per parameter. That is a BO-era cache and it would
    collapse gammaSC values finer than 0.01 — irrelevant to the grid search
    (which caches per grid point by its own key), but it must be raised if the
    Bayesian path is ever used on this parameter range.
    """
    ablation_tag = "_".join(str(int(c)) for c in ablated_cells) if ablated_cells else ""
    # Canonical, fully-distinguishing description — only ever HASHED, never
    # used verbatim, so its length is irrelevant.
    canonical = "from=%s|gSC=%.4f|gHC=%.2f|aHC=%.2f|ps=%.2f|abl=%s" % (
        initial_sheet_name or "", gammaSC, gammaHC_ratio, alphaHC_ratio,
        psigma, ablation_tag,
    )
    # Only fold a NON-default atoh_sensitivity into the hash: runs made with the
    # historical default keep their existing folder names (and cached results),
    # while a fit that drives atoh_sensitivity from a loaded delta threshold gets
    # a DISTINCT folder instead of colliding with — and, via reuse_existing_run,
    # wrongly reusing — a default-atoh run at the same (sheet, gSC, gHC, aHC).
    if abs(atoh_sensitivity - _DEFAULT_ATOH_SENSITIVITY) > 1e-9:
        canonical += "|patoh=%.4f" % atoh_sensitivity
    # Likewise fold a NON-default shape_index (target perimeter P0=p0*sqrt(A0))
    # into the hash so a fit that adds it doesn't collide with the P0=0 runs.
    # Formatted at %.2f — the SAME resolution as the readable params above and as
    # find_mechanical_parameters' 2-decimal cache — so reuse_existing_run hits at
    # the same granularity for shape_index as for gSC/gHC/aHC (a finer %.4f made
    # equivalent 2-decimal points land in distinct folders and needlessly recompute).
    if abs(shape_index) > 1e-9:
        canonical += "|p0=%.2f" % shape_index
    # Likewise fold a NON-default mechanical steady-state cutoff into the hash so
    # a base run relaxed to 0.03 and an ablation run relaxed to 0.02 each get a
    # DISTINCT folder from the historical 0.01 runs (a different threshold stops
    # the relaxation at a different state → different roundness / area, so they
    # must NOT collide and be reused for one another).
    if abs(quasi_static_threshold - _DEFAULT_QUASI_STATIC_THRESHOLD) > 1e-9:
        canonical += "|qst=%.3f" % quasi_static_threshold
    # Fold a non-None line tension into the hash so a line-tension fit gets
    # DISTINCT folders from the no-line-tension runs (it changes the relaxed
    # morphology, so they must not collide / be reused for one another).
    if line_tension is not None:
        canonical += "|lt=%.4f" % line_tension
    # Type-dependent shape index: fold each per-type target perimeter into the
    # hash so an HC/SC-split fit never collides with (or reuses) a run made with
    # the single shared shape_index above. Only folded when actually supplied, so
    # shared-shape_index runs keep their historical folder names.
    if hc_shape_index is not None:
        canonical += "|p0hc=%.2f" % hc_shape_index
    if sc_shape_index is not None:
        canonical += "|p0sc=%.2f" % sc_shape_index
    if bending is not None:
        canonical += "|bend=%.4f" % bending
    # Fold an explicit PREFERRED AREA into the hash. This is mandatory for
    # correctness: the target area sets the whole force balance (and, via
    # P0 = p0*sqrt(A0), the target perimeter), so runs at different preferred
    # areas must never collide or be reused for one another. NOTE runs made
    # BEFORE the L0-normalization fix used an effective preferred area of
    # pi/4*L0^2 (~12x larger) while passing nothing here — those folders are
    # stale and must not be reused; always supply preferred_area post-fix.
    if preferred_area is not None:
        canonical += "|pa=%.4f" % preferred_area
    # Mechanosensitivity SHIFT K: changes the dynamics, so a non-zero value must
    # get its own folder (K=0 keeps the historical names).
    if stress_shift:
        canonical += "|ks=%.4f" % stress_shift
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    name = "fit_gSC%.4f_gHC%.2f_aHC%.2f_ps%.2f" % (
        gammaSC, gammaHC_ratio, alphaHC_ratio, psigma,
    )
    # Surface a NON-default shape_index in the readable prefix too (not only the
    # hash), so p0 runs are self-describing on disk; p0=0 keeps the historical name.
    if abs(shape_index) > 1e-9:
        name += "_p0%.2f" % shape_index
    # Surface a NON-default steady-state cutoff in the readable prefix too, so
    # 0.03 (base) / 0.02 (ablation) runs are self-describing on disk.
    if abs(quasi_static_threshold - _DEFAULT_QUASI_STATIC_THRESHOLD) > 1e-9:
        name += "_qst%.3f" % quasi_static_threshold
    if line_tension is not None:
        name += "_lt%.3f" % line_tension
    if bending is not None:
        name += "_bend%.3f" % bending
    if preferred_area is not None:
        name += "_pa%.3f" % preferred_area
    if stress_shift:
        name += "_ks%.3f" % stress_shift
    # Self-describing tag for the type-dependent shape index (HC/SC split).
    if hc_shape_index is not None or sc_shape_index is not None:
        name += "_p0hc%.2f_p0sc%.2f" % (
            hc_shape_index if hc_shape_index is not None else shape_index,
            sc_shape_index if sc_shape_index is not None else shape_index)
    name += "_%s" % digest
    if ablated_cells:
        # Keep a readable marker that this is the ablation variant (the
        # actual cell ids live in the hash above).
        name += "_abl"
    return name


def run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial_sheet_name=None, ablated_cells=None,
        only_differentiation=False, no_differentiation=False, end_on_steady_state=True, t_end=25, dt=0.01,
        random_forces=False, name=None, continue_existing_run=False, divisions=False,
        continue_from_time=None, randomize_notch_delta_levels=False, stress_dependent=False,
        notch_levels=None, delta_levels=None, repressor_levels=None, verbose_log=True,
        max_wall_seconds=None, min_progress_rate=None, progress_window_seconds=30.0,
        reuse_existing_run=False, rerun_stalled_runs=False,
        atoh_sensitivity=_DEFAULT_ATOH_SENSITIVITY, shape_index=0.0,
        sharp_angle_threshold=0.1,
        quasi_static_threshold=_DEFAULT_QUASI_STATIC_THRESHOLD,
        line_tension=0.2,
        hc_shape_index=None, sc_shape_index=None, bending=None,
        notch_sensitivity=0.2, repressor_sensitivity=0.45, stress_hill_exponent=None,
        initial_notch_delta_level=None, preferred_area_override=None,
        save_interval=None, stress_shift=0.0, notch_inhibition=False):
    """Run a simulation; optionally resume an interrupted one OR
    fork a fresh one from a chosen snapshot of an existing archive.

    Parameters
    ----------
    continue_existing_run : bool, default False
        If True, the run named ``name`` is treated as a partially
        completed archive sitting in ``results/<name>/<name>.hf5``.
        The sheet is loaded from that archive, every snapshot after
        ``continue_from_time`` is dropped, and the simulation
        continues writing into the SAME archive — no separate
        ``part2`` file. ``t_end`` is interpreted as the CUMULATIVE
        end time, so a run originally launched with ``t_end=100``
        that died at ``t=17.6`` can be resumed with the same
        ``t_end=100`` and it will only have to cover the remaining
        ``100 - continue_from_time`` units.

        If False, the standard "new run" behaviour applies:
        ``results/<name>/`` is created from scratch and the
        simulation starts at t=0.
    continue_from_time : float, optional
        When ``continue_existing_run`` is True, the time stamp to
        rewind to. ``None`` (the default) uses the LAST recorded
        time in the archive — handy when the previous run died with
        a clean tail and you just want to pick up where it stopped.
        Pass an explicit value to rewind further (e.g. to skip past
        a known-bad event).

        When ``continue_existing_run`` is False, ``continue_from_time``
        instead picks the SNAPSHOT of ``initial_sheet_name``'s
        archive to use as the initial condition for a brand-new run
        — a "fork" off a chosen moment. The source archive is NOT
        modified; results land in the fresh ``results/<name>/``
        folder; the new run's clock still starts at t=0. This is
        the same path the simulator has always taken when given an
        ``initial_sheet_name``, just with the snapshot chosen by
        time rather than always being the last one. ``None`` (the
        default) keeps the historical "load last snapshot"
        behaviour.
    reuse_existing_run : bool, default False
        Smart restart for re-launched parameter fits. When True and a
        results folder for ``name`` already exists, the previous run's
        ``debug.log`` is classified (via :func:`_classify_existing_run`)
        and acted on instead of the blind "directory already exists ->
        return" cache hit:

        - ``completed`` (log has ``run() finished successfully``): keep
          the archive and return — the caller computes p-values from it.
        - ``stalled`` (the solver hit its dt floor via non-convergence /
          self-intersecting fold / negative-area cell, OR the slow-
          progress / wall-clock guard fired): by default the parameter
          point is treated as genuinely bad, so ``run`` raises
          ``RuntimeError`` — the fit worker catches it and scores
          worst-case WITHOUT re-simulating. Set ``rerun_stalled_runs=True``
          to instead discard the stalled folder and re-run the point from
          scratch (e.g. after a code fix that might no longer stall on it,
          or when you want a fresh attempt rather than an automatic
          worst-case score).
        - ``interrupted`` (crashed for an EXTERNAL reason — out of disk,
          process killed, machine reboot — or the log just ends): resume
          the run's OWN archive from its latest snapshot. If no snapshot
          was ever written (``history.hf5`` missing/empty) the stub
          folder is deleted and the point is re-run from scratch.

        Has no effect unless the folder already exists, and is ignored
        when ``continue_existing_run`` is True (an explicit resume).
    notch_inhibition : bool, default False
        Zero the repressor production term, reproducing the experimental
        Notch/repressor block: every cell differentiates. Used to calibrate
        the simulation time unit (all-HC <-> 48 h).
    rerun_stalled_runs : bool, default False
        Only relevant when ``reuse_existing_run`` is True and the existing
        run classifies as ``stalled``. False (the default) keeps the
        historical behaviour: raise ``RuntimeError`` and let the caller
        score the point worst-case without re-running. True instead deletes
        the stalled folder and re-runs the point from scratch, the same way
        an ``interrupted`` run with no snapshot is handled.
    atoh_sensitivity : float, default 0.377
        The delta half-max (``patoh``) of the Atoh1 Hill function
        ``delta^m / (atoh_sensitivity^m + delta^m)``: a cell's ``atoh_level``
        crosses the 0.5 ``differentiation_threshold`` exactly when
        ``delta == atoh_sensitivity``, so this IS the delta threshold that
        splits HC (delta above) from SC (delta below), driving both the cell
        types and the interpolated HC/SC mechanics. Exposed so a fit can set it
        from a per-sheet loaded delta threshold. Only a NON-default value is
        folded into the results-folder hash, so default-atoh runs keep their
        existing folder names / cache.
    randomize_notch_delta_levels : bool, default False
        When True, every LI level (``notch_level``, ``delta_level``,
        ``repressor_level``) is freshly randomised at the start of
        the run — overriding the default "preserve loaded values"
        behaviour. Useful for parameter sweeps that re-use a saved
        geometry (resume OR fork mode) but want a different LI
        initial condition per sweep point. On a fresh sheet (no
        ``initial_sheet_name``) this flag is a no-op: the LI
        columns are randomised either way.
    notch_levels, delta_levels, repressor_levels : str or np.ndarray, optional
        Per-cell lateral-inhibition initial values. Each may be a path to
        a ``.npy`` file or a 1-D array; entry ``i`` holds the value for
        the cell whose ``unique_id == i`` (entry ``i`` == cell ``i`` on a
        fresh sheet). When supplied these take precedence over both the
        preserve-loaded and random-seed initialisation (and over
        ``randomize_notch_delta_levels``); any channel left ``None`` falls
        back to the default behaviour. This is the clean replacement for
        the old pickled-DataFrame ``saved_notch_delta_levels_file``.
    """
    # Sheet Parameters


    if name is None:
        # Short, bounded-length, unique folder name. The old long
        # ``periodic_from<initial>_gammaSC-...`` scheme compounded when runs
        # were forked from one another and overran the OS path limit — see
        # _short_run_folder_name. The ablation list is folded into that name's
        # hash, so it is NOT appended again below for the generated case.
        name = _short_run_folder_name(
            initial_sheet_name, gammaSC, gammaHC_ratio, alphaHC_ratio,
            psigma, ablated_cells, atoh_sensitivity, shape_index,
            quasi_static_threshold, line_tension,
            hc_shape_index, sc_shape_index, bending,
            preferred_area=preferred_area_override, stress_shift=stress_shift,
        )
    elif ablated_cells is not None and len(ablated_cells) > 0:
        # Caller supplied an explicit ``name`` AND an ablation list: keep the
        # historical behaviour of marking the ablation variant by suffix.
        name += "ablated"
        for cell in ablated_cells:
            name += "_%d" % cell

    if continue_existing_run:
        # The archive we're resuming from is the SAME path as the
        # output we'll keep writing — both live at
        # ``results/<name>/<name>.hf5``. ``name`` is therefore NOT
        # renamed here (the old "+= 'part2'" branch created a
        # parallel archive, which is exactly what the user wants to
        # get rid of).
        initial_sheet_name = name
    elif initial_sheet_name is None:
        initial_sheet_name = ""

    max_bond_length = 0.2
    min_bond_length = 0.05

    # In case initial_sheet_name == "", creating a new sheet with the following parameters
    nx = 20
    ny = 20
    distx = 1
    disty = 1

    # Model version select
    random_sensitivity = False
    aging_sensitivity = False
    contact_dependent_differentiation = True
    # notch_inhibition is a run() arg now: True ZEROES repressor production
    # (lateral_inhibition_model: `repressor_production = 0 if self.inhibition`),
    # so repressor decays away, decreasing_hill(0) drives Delta to its maximum
    # and every cell turns HC. That is the Notch-inhibition experiment, and the
    # time to reach all-HC is what calibrates simulation time against the
    # measured 48 h.
    # ``stress_dependent`` now comes from the call argument (default False
    # keeps the historical behaviour). It must be True for ``psigma`` /
    # ``mechanosensitivity`` to have any effect — see the guard below where
    # ``mechanosensitivity`` is forced to 0 when stress dependence is off.
    intercalations = True
    delaminations = True
    if ablated_cells is None:
        ablated_cells = []
    quasi_static = True
    # quasi_static_threshold is now a run() parameter (default
    # _DEFAULT_QUASI_STATIC_THRESHOLD) so the mechanics fit can relax base and
    # ablation runs to different cutoffs; it is folded into the folder hash above.

    # Model Parameters
    # General parameters
    movie_frames = 100

    # 2D vertex related parameters. The perimeter energy uses
    # ContractilityPerimeterElasticity (1/2*contractility*(P - prefered_perimeter)**2)
    # instead of plain FaceContractility; with shape_index=0 (P0=0) the two are
    # identical, and shape_index>0 adds a target perimeter (the roundness knob).
    effectors = [ContractilityPerimeterElasticity, FaceAreaElasticity]
    tension = {('HC', 'HC'): 0.05,
               ('HC', 'SC'): 0.05,
               ('SC', 'SC'): 0.05
               }
    # Optional edge LINE TENSION. Off by default (the historical behaviour: the
    # tension dict above is passed to the model but the LineTension effector is
    # NOT active, so it has no effect). When a caller sets ``line_tension`` we add
    # the effector and use that value for every cell-type pair — a small line
    # tension penalises jagged, high-curvature bonds and smooths the boundaries.
    if line_tension is not None:
        effectors = effectors + [LineTension]
        tension = {k: float(line_tension) for k in tension}
    # Uniform boundary BENDING (curvature) stiffness. Penalises kinked bonds
    # WITHOUT shortening the perimeter, so unlike line tension it does not fight
    # the elongation needed to reach the experimental roundness.
    if bending is not None:
        effectors = effectors + [BoundaryBending]
    # Preferred cell area = area of a circle whose radius is HALF the lattice
    # unit (distx = disty = 1, so radius 0.5): pi * 0.5**2 = pi/4 ~ 0.785, which
    # matches the actual mean cell area of the saved arrays (~0.76). NOTE: this
    # was previously a typo `1/(4*pi)` ~ 0.0796 — ~10x too small, which drove
    # every cell to shrink hard, made the periodic tiling jagged, and triggered
    # the sharp-corner collapse cascade (see [[sharp-corner-collapse-prevents-folds]]).
    preferred_area = {'HC': np.pi / 4,
                      'SC': np.pi / 4}
    contractility = {'HC': gammaSC * gammaHC_ratio,
                     'SC': gammaSC}

    repulsion = {'HC': 0.001,
                 'SC': 0.}
    repulsion_distance = {'HC': 2.0,
                          'SC': 0.}
    repulsion_exponent = 7.
    elasticity = {'HC': alphaHC_ratio,
                  'SC': 1.}

    # Topological events related parameters
    division_area = 1.2
    intercalation_length = 0.04
    # Interior-angle threshold (radians) for collapsing an incipient fold: a
    # face pinched by two non-adjacent vertices drifting together has a small
    # angle at their shared vertex long before any edge is short enough for
    # intercalation. None (the default) lets ``simulate`` derive it from the
    # sheet's length scales (the head angle at which a corner's two neighbours,
    # on max_bond_length-long edges, sit max(min_bond_length,
    # intercalation_length) apart — ~0.25 rad / ~14 deg here — see
    # ``VirtualSheet.default_sharp_angle_threshold``); the ``sharp_angle_threshold``
    # run() parameter (default 0.1 rad = 5.73 deg) overrides it — LOWER it to stop
    # the collapse machinery from thrashing on stable near-threshold corners that
    # reform every step (the dominant cost in stalled stiff/fluid runs).
    delamination_area = 0.1
    delamination_rate = 1.1
    viscosity = 1

    # Lateral Inhibition parameters
    differentiation_threshold = 0.5
    l = 3  # decreasing Hill exponent
    m = 3  # increasing Hill exponent
    betaN = 1  # maximum production rate Notch for classical model
    betaD = 1  # maximum production rate Delta for classical model
    notch_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    # repressor_sensitivity (PR) is a run() arg now (default 0.45): how much Delta
    # production is sensitive to repressor level (sens^l / (sens^l + repressor^l)).
    # atoh_sensitivity (patoh) is the DELTA half-max of the Atoh1 Hill function
    # (delta^m / (atoh_sensitivity^m + delta^m)): a cell's atoh_level crosses the
    # 0.5 differentiation_threshold exactly when delta == atoh_sensitivity, so it
    # IS the delta threshold separating HC (delta above) from SC (delta below).
    # It's a run() parameter now (default 0.377) so the fit can drive it from the
    # per-sheet loaded delta threshold. See the signature.
    atoh_by_repressor = False  # if True, Atoh1 production will be set by repressor level instead of delta (sensitivity^l / (sensitivity^l + repressor^l)
    # notch_sensitivity (PS) is a run() arg now (default 0.2): how much Repressor
    # production is sensitive to signaling level (sig^m / (sens^m + sig^m)).
    delta_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    notch_delta_production_ratio = 1  # beta
    sensitivity_aging_rate = 10  # Notch sensitivity change rate (for aging sensitivity version)
    mechanosensitivity = psigma  # Sensitivity to mechanical stress (for stress dependent version)
    stress_effectors = [ContractilityPerimeterElasticity]  # effectors to calculate stress (for stress dependent version)
    li_steady_state_threshold=0.001

    if not stress_dependent:
        mechanosensitivity = 0

    results_dir = os.path.join(RESULTS_DIR, name)
    if continue_existing_run:
        if not os.path.exists(results_dir):
            print("Directory %s doesn't exist. Unable to continue from existing run." % results_dir)
            return name
    elif os.path.exists(results_dir):
        if not reuse_existing_run:
            print("Directory %s already exists" % results_dir)
            return name
        # Smart restart: classify the existing run from its log and act
        # (see the ``reuse_existing_run`` docstring for the three cases).
        status = _classify_existing_run(results_dir)
        latest = _latest_archive_time(results_dir)
        print("Reusing existing run %s: status=%s, latest archive t=%s"
              % (results_dir, status, latest))
        if status == "completed":
            # "Completed" only means the process exited cleanly. A run that ran
            # out of ``t_end`` WITHOUT settling is not finished in the sense
            # that matters — and psigma>0 runs hit that far more often, because
            # gating delta production slows the approach to steady state. Such a
            # run is EXTENDED in place from its last frame: ``t_end`` is
            # CUMULATIVE on a resume, so only the missing span is simulated.
            # A run that did reach steady state is left alone whatever t_end is.
            if (not end_on_steady_state or latest is None
                    or _reached_steady_state(results_dir)
                    or float(t_end) <= float(latest) + float(dt)):
                return name
            print("Existing run %s completed at t=%s WITHOUT reaching steady "
                  "state; extending to t_end=%s." % (results_dir, latest, t_end))
            continue_existing_run = True
            initial_sheet_name = name
            continue_from_time = latest
        elif status == "stalled" and not rerun_stalled_runs:
            raise RuntimeError(
                "Existing run %s previously stalled (dt floor / fold / "
                "negative area / non-progress guard); scoring worst-case "
                "instead of re-running." % results_dir)
        elif status == "stalled":
            # rerun_stalled_runs=True: a stalled sheet is degenerate, so
            # resuming it (like an "interrupted" run) would only
            # re-degenerate — discard it and re-run from scratch instead.
            print("Existing run %s previously stalled; rerun_stalled_runs=True, "
                  "re-running from scratch." % results_dir)
            shutil.rmtree(results_dir)
            os.mkdir(results_dir)
        # status == "interrupted": resume the run's OWN archive from its
        # latest snapshot. With nothing to resume from, start over.
        elif latest is None:
            print("No usable snapshot in %s; re-running from scratch." % results_dir)
            shutil.rmtree(results_dir)
            os.mkdir(results_dir)
        else:
            continue_existing_run = True
            initial_sheet_name = name
            continue_from_time = latest
    else:
        os.mkdir(results_dir)

    #  Saving model  parameters
    params_file = os.path.join(results_dir, "parameters.txt")
    if continue_existing_run and os.path.exists(params_file):
        # The original run already wrote ``<name>_parameters.txt``.
        # Don't overwrite it (it documents the originating run);
        # write the resume's parameters to a numbered side-file so
        # the archive carries an audit trail of every restart.
        # ``name[-1] = ...`` USED to live here, but strings aren't
        # mutable in Python and that path crashed before it could
        # finish renaming anything.
        n = 1
        while True:
            candidate = os.path.join(
                results_dir,
                "parameters_continue%d.txt" % n,
            )
            if not os.path.exists(candidate):
                params_file = candidate
                break
            n += 1
    variables = locals().copy().items()


    with open(params_file, "w") as f:
        for var_name, var_value in variables:
            # Exclude built-in and special variables (e.g., those starting with '__')
            if not var_name.startswith("__") and not callable(var_value) and not isinstance(var_value, type(os)):
                f.write(f"{var_name}: {repr(var_value)}\n")

    # Enable debug logging IMMEDIATELY after the parameters dump
    # (so the dump above isn't polluted with the log-handler local)
    # but BEFORE any heavy work — that way every sheet build /
    # simulator / event log line from this run lands in the file.
    # The handler flushes on every record so the file is current
    # even when the process dies mid-step.
    #
    # On a resume we APPEND to the existing log so the original
    # failed-run transcript is preserved alongside the resumed-run
    # one. Otherwise we open ``w`` (default) and start clean.
    debug_log_path = os.path.join(results_dir, "debug.log")
    log_handler = _enable_debug_log(
        debug_log_path, append=bool(continue_existing_run),
    )
    run_log = logging.getLogger("periodic_tests")
    if continue_existing_run:
        run_log.info(
            "Resuming run from time %s into %s",
            continue_from_time, debug_log_path,
        )
    else:
        run_log.info("Debug log started at %s", debug_log_path)

    initial_sheet_name = os.path.join(RESULTS_DIR, initial_sheet_name)
    name = os.path.join(RESULTS_DIR, name)
    try:
        # Load or initialize sheet
        if continue_existing_run:
            # Resume: load the sheet from the existing archive at
            # the chosen time (defaulting to the last recorded
            # snapshot when ``continue_from_time`` is None).
            # ``force_periodic_box`` recovers legacy archives that
            # didn't stash their periodicity (the model is always a
            # periodic nx*distx by ny*disty box).
            sheet = load_sheet_from_file(
                initial_sheet_name,
                time_point=continue_from_time,
                force_periodic_box=(nx * distx, ny * disty),
            )
            # If the caller didn't pin a time, pin it now to the
            # last recorded snapshot so the truncate + solver-clock
            # plumbing below works on the same time stamp.
            if continue_from_time is None:
                _resume_hist = HistoryHdf5.from_archive(
                    os.path.join(initial_sheet_name, "history.hf5"), eptm_class=VirtualSheet,
                )
                continue_from_time = float(np.max(_resume_hist.time_stamps))
                run_log.info(
                    "continue_from_time defaulted to last snapshot t=%g",
                    continue_from_time,
                )
        elif os.path.isfile(os.path.join(initial_sheet_name, "history.hf5")):
            # Fork mode (when continue_from_time is set) OR the old
            # "load the last snapshot from the source archive"
            # behaviour (when it's None). Either way the source
            # archive is read-only here — the new simulation will
            # write into ``results/<name>/<name>.hf5`` (a different
            # path) starting at t=0. ``simulate(continue_from_time=
            # None)`` already does the right thing for that.
            sheet = load_sheet_from_file(
                initial_sheet_name, time_point=continue_from_time,
                force_periodic_box=(nx * distx, ny * disty),
            )
            if continue_from_time is not None:
                run_log.info(
                    "Forking new run from snapshot t=%g of %s",
                    continue_from_time, initial_sheet_name,
                )
        else:
            if continue_from_time is not None:
                # ``continue_from_time`` only makes sense when there
                # IS a source archive to read from. Surface the
                # disagreement loudly rather than silently making a
                # fresh sheet that has nothing to do with the
                # caller's intent.
                run_log.warning(
                    "continue_from_time=%g specified but no source "
                    "archive at %s.hf5; falling back to a fresh sheet.",
                    continue_from_time, initial_sheet_name,
                )
            sheet = initialize_sheet(nx, ny, distx, disty, max_bond_length, min_bond_length)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0

        # NOTE: the LI levels (notch / delta / repressor) now travel
        # with the HDF5 history. The old pickle side-channel
        # ``<name>_notch_delta_levels.pkl`` is gone — when ``sheet``
        # was loaded from an archive its LI columns are already
        # populated and InnerEarModel preserves them; on a fresh
        # sheet they're randomly seeded by ``initialize_notch_delta``.

        # Initialize model
        # A RESUME must never re-seed notch/delta/repressor: the archive's
        # evolved LI state is the whole point of resuming, and force_random
        # would overwrite it (mid-run HC/SC flip). run() may switch to a
        # resume internally via reuse_existing_run, so clamp it HERE rather
        # than trusting the caller's flag.
        if continue_existing_run:
            randomize_notch_delta_levels = False
        inner = InnerEarModel(sheet, tension=tension, repulsion=repulsion, repulsion_distance=repulsion_distance,
                              repulsion_exp=repulsion_exponent, preferred_area=preferred_area,
                              preferred_area_override=preferred_area_override, contractility=contractility,
                              elasticity=elasticity, shape_index=shape_index,
                              hc_shape_index=hc_shape_index, sc_shape_index=sc_shape_index,
                              bending=(0.0 if bending is None else bending),
                              differentiation_threshold=differentiation_threshold,
                              random_sensitivity=random_sensitivity,
                              l=l, m=m, betaN=betaN, betaD=betaD, inhibition=notch_inhibition,
                              notch_repressor_degradation_ratio=notch_repressor_degradation_ratio,
                              repressor_sensitivity=repressor_sensitivity, atoh_sensitivity=atoh_sensitivity,
                              delta_repressor_degradation_ratio=delta_repressor_degradation_ratio,
                              notch_delta_production_ratio=notch_delta_production_ratio,
                              stress_effectors=stress_effectors, mechanosensitivity=mechanosensitivity,
                              stress_shift=stress_shift,
                              stress_hill_exponent=stress_hill_exponent,
                              notch_sensitivity=notch_sensitivity, atoh_by_repressor=atoh_by_repressor,
                              randomize_notch_delta_levels=randomize_notch_delta_levels,
                              # LI seed arrays initialise the LI state at t=0 ONLY.
                              # A resume continues an existing trajectory whose
                              # notch/delta/repressor levels already live in the
                              # loaded archive — re-seeding them from the initial-
                              # sheet arrays there OVERWRITES the evolved state and
                              # (since the archive's LI can differ from the arrays)
                              # scrambles delta -> atoh -> HC/SC mid-run. So pass
                              # them through only for a FRESH run; on resume force
                              # None to preserve the archive's LI state.
                              notch_levels=(None if continue_existing_run else notch_levels),
                              delta_levels=(None if continue_existing_run else delta_levels),
                              repressor_levels=(None if continue_existing_run else repressor_levels),
                              initial_notch_delta_level=initial_notch_delta_level)
        draw_func = inner.get_draw_sheet_method(number_faces=True, number_edges=False, number_vertices=False,
                                                color_by="atoh")
        fig1, ax1 = draw_func(inner.sheet)
        plt.savefig(os.path.join(name, "initial.png"))
        # Pass the archive path so HistoryHdf5 writes each snapshot on the
        # fly; the partial file can be opened in another process while the
        # simulation is still running (useful for diagnosing hangs).
        history_file = os.path.join(name, "history.hf5")
        # Announce the run by its results-folder name on the console (workers'
        # stdout is visible), so a crash mid-fit can be tied to a specific run.
        print("[run] START    %s  (t_end=%g, dt=%g%s)"
              % (os.path.basename(name), t_end, dt,
                 (", resuming from t=%g" % continue_from_time)
                 if continue_existing_run else ""),
              flush=True)
        history = inner.simulate(t_end=t_end, dt=dt, until_steady_state=end_on_steady_state,
                                 lateral_inhibition_threshold=li_steady_state_threshold,
                                 only_differentiation=only_differentiation,
                                 random_forces=random_forces, aging_sensitivity=aging_sensitivity,
                                 no_differentiation=no_differentiation,
                                 contact_dependent_differentiation=contact_dependent_differentiation, divisions=divisions,
                                 intercalations=intercalations, delaminations=delaminations, ablated_cells=ablated_cells,
                                 sensitivity_aging_rate=sensitivity_aging_rate,
                                 division_area=division_area, intercalation_length=intercalation_length,
                                 sharp_angle_threshold=sharp_angle_threshold, verbose_log=verbose_log,
                                 delamination_area=delamination_area, delamination_rate=delamination_rate,
                                 viscosity=viscosity, effectors=effectors, quasi_static=quasi_static,
                                 quasi_static_threshold=quasi_static_threshold, atoh_by_repressor=atoh_by_repressor,
                                 history_file=history_file, save_interval=save_interval,
                                 max_wall_seconds=max_wall_seconds, min_progress_rate=min_progress_rate,
                                 progress_window_seconds=progress_window_seconds,
                                 continue_from_time=(continue_from_time if continue_existing_run else None))
        # ``inner.save_notch_delta(...)`` USED to write a pickle
        # side-file with the final LI levels for the next run to
        # consume. The HDF5 history now carries the same data on
        # every saved snapshot, so the pickle is redundant. The
        # method itself is kept on InnerEarModel as a stand-alone
        # utility for ad-hoc exports — nothing in the standard
        # run pipeline calls it any more.
        fig2, ax2 = draw_func(inner.sheet)
        plt.savefig(os.path.join(name, "finale.png"))
        inner.save_sheet_labels_to_numpy(inner.sheet, path=os.path.join(name, "labels.npy"))
        inner.save_contact_matrix_to_numpy(inner.sheet, path=os.path.join(name, "contact_matrix.npy"))
        inner.save_face_data_to_df(inner.sheet, path=os.path.join(name, "cell_info.pkl"))
        gif_func = inner.get_draw_sheet_method(number_faces=False, number_edges=False, number_vertices=False,
                                               color_by="atoh",
                                               arrange_sheet=True)
        create_gif_safe(history, os.path.join(name, "movie.gif"), num_frames=movie_frames, draw_func=gif_func)
        run_log.info(_RUN_SUCCESS_MARKER)
        print("[run] FINISHED %s" % os.path.basename(name), flush=True)
        return name
    except BaseException as exc:
        # ``BaseException`` (not ``Exception``) covers Ctrl+C
        # (KeyboardInterrupt) too — the user may stop a stuck run
        # explicitly and still want the partial log to land on disk.
        # ``run_log.exception`` automatically formats the active
        # traceback and writes it through the same handler, so the
        # entire chain is in the file before the raise propagates.
        run_log.exception("%s; traceback follows" % _RUN_CRASH_MARKER)
        print("[run] CRASHED  %s  (%s: %s)"
              % (os.path.basename(name), type(exc).__name__, exc), flush=True)
        raise
    finally:
        # Always flush + detach so that
        #  (a) every record written above is committed to disk, and
        #  (b) a re-run inside the same Python session doesn't keep
        #      writing to the previous run's file.
        _disable_debug_log(log_handler)
    # Return ONLY on the success path. A ``return`` inside the ``finally`` above
    # would silently SWALLOW the exception re-raised in ``except`` (Python
    # gotcha): run() would then hand back its folder name even on a crash, so
    # find_mechanical_parameters' worker never saw the RuntimeError, scored the
    # crashed/degenerate sheet instead of worst-case, and launched the ablation
    # run from a corrupt state. With the return out here, a crash propagates.
    return name

def _create_one_random_array(index):
    """Build the random array with the given integer ``index`` (independent
    unit of ``create_random_arrays``). Module-level + single-argument so it is
    picklable for ``ProcessPoolExecutor`` — and so a single Azure Batch task
    can build exactly one array."""
    print("Running random array %d" % index)
    gammaSC = 0.01
    gammaHC_ratio = 10
    alphaHC_ratio = 1
    psigma = 1
    name = run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, name=random_array_name(index),
               random_forces=True, end_on_steady_state=False, t_end=25, dt=0.01, divisions=True,
               no_differentiation=True)
    print("Finished %s" % name)
    return name


def restore_array_run(index, continue_from_time, t_end=None):
    """Resume a ``random_periodic_array{index}`` run from a fold-free time point
    and re-run the tail with the SAME parameters as ``_create_one_random_array``.

    Because the solver now carries the folded-face safety net, the continuation
    will not introduce new self-intersections (cells "growing into each other"),
    so re-extracting the best E17.5 / P0 frames afterwards yields fold-free
    sheets.

    IMPORTANT:
      * This TRUNCATES ``results/random_periodic_array{index}/history.hf5`` at
        ``continue_from_time`` and overwrites everything after it. **Back up the
        archive first** if you want to keep the original (folded) tail.
      * ``continue_from_time`` must be an EXISTING time stamp in the archive
        that is verified fold-free (see the values reported for arrays
        1/2/3/5/7/8).
      * The run uses ``random_forces=True``, so the continuation is stochastic —
        it does NOT reproduce the original trajectory, it produces a fresh,
        fold-free one. Re-run ``store_best_time_point_for_random_arrays`` after.
      * ``t_end`` defaults to the archive's current last time (re-run to the
        same end); the cumulative end-time semantics of ``run`` apply.
    """
    name = random_array_name(index)
    if t_end is None:
        hist = HistoryHdf5.from_archive(
            os.path.join(RESULTS_DIR, name, "history.hf5"), eptm_class=VirtualSheet)
        t_end = float(np.max(hist.time_stamps))
    return run(0.01, 10, 1, 1, name=name, initial_sheet_name=name,
               continue_existing_run=True, continue_from_time=continue_from_time,
               random_forces=True, end_on_steady_state=False, t_end=t_end, dt=0.01,
               divisions=True, no_differentiation=True)


def create_random_arrays(n=None, indices=None, n_workers=None):
    """Build random arrays by integer index. Pass ``indices`` (e.g. ``[7]`` for
    a single Batch task, or ``range(10)``) or a count ``n`` (shorthand for
    ``range(n)``). The arrays are independent, so they run in parallel across
    ``n_workers`` processes (default ``min(#arrays, cpu_count)``); pass
    ``n_workers=1`` to force serial."""
    if indices is None:
        if n is None:
            raise ValueError("Provide either n or indices")
        indices = range(n)
    indices = list(indices)
    if n_workers is None:
        n_workers = min(len(indices), os.cpu_count() or 1)
    if n_workers > 1 and len(indices) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            names = list(executor.map(_create_one_random_array, indices))
    else:
        names = [_create_one_random_array(i) for i in indices]
    return names


def _initialize_one_differentiated_array(args):
    """Differentiate one initial sheet (independent unit of
    ``initialize_differentiated_arrays``). Module-level + single-argument so
    it is picklable for ``ProcessPoolExecutor``."""
    gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial, end_time, dt, continue_existing_run = args
    print("Running on initial sheet:\n%s\nwith initial parameters:\n" % initial,
          "gammaSC=%.2f ,gammaHC_ratio=%.2f ,alphaHC_ratio=%.2f, psigma=%.2f" % (gammaSC, gammaHC_ratio, alphaHC_ratio, psigma))
    name = run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial_sheet_name=initial,
               continue_existing_run=continue_existing_run, end_on_steady_state=True, t_end=end_time, dt=dt, divisions=False)
    print("Finished running: %s" % name)
    return name


def initialize_differentiated_arrays(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma,
                                     stage="E17.5", indices=None, n_arrays=10, n_workers=None, end_time=10,
                                     dt=0.01, continue_existing_run=False):
    """Differentiate the ``<stage>`` fitted-initial-morphology arrays. Select
    which arrays by integer ``indices`` (e.g. ``[7]`` for a single Batch task,
    or ``range(n_arrays)`` by default). The sheets are independent, so they run
    in parallel across ``n_workers`` processes (default
    ``min(#sheets, cpu_count)``); pass ``n_workers=1`` to force serial."""
    if indices is None:
        indices = range(n_arrays)
    initial_sheets = [initial_morphology_name(i, stage) for i in indices]
    tasks = [(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial, end_time, dt, continue_existing_run) for initial in initial_sheets]
    if n_workers is None:
        n_workers = min(len(tasks), os.cpu_count() or 1)
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            names = list(executor.map(_initialize_one_differentiated_array, tasks))
    else:
        names = [_initialize_one_differentiated_array(task) for task in tasks]
    return names


def _psigma_tag(psigma):
    """Folder tag for psigma. 3 decimals when that is EXACT (so the runs
    already computed at 0.015 / 0.030 / 0.045 keep their names and stay
    reusable), otherwise 5 - the K=-0.06 regime needs psigma ~0.002-0.005,
    where 3 decimals would collide."""
    three = "%.3f" % psigma
    return three if abs(float(three) - psigma) < 1e-9 else "%.5f" % psigma


def _run_full_model_one(args):
    """One FULL-model run (lateral-inhibition differentiation + quasi-static
    mechanics, coupled) started from the array's UNDIFFERENTIATED morphology.
    Module-level + single-argument so it is picklable for ProcessPoolExecutor."""
    (initial, gammaSC, gammaHC_ratio, alphaHC_ratio, hc_shape_index, sc_shape_index,
     atoh_sensitivity, notch_sensitivity, repressor_sensitivity, bending,
     line_tension, quasi_static_threshold, t_end, dt, initial_notch_delta_level,
     psigma, preferred_area, save_interval, stress_shift,
     stress_hill_exponent, reuse_existing_run, continue_existing_run,
     name_prefix) = args
    # psigma (mechanosensitivity) gates delta production by mechanical stress and
    # only matters with stress_dependent=True. psigma=0 -> identical to the
    # no-stress baseline, so it KEEPS the plain ``fullmodel_<array>`` folder name
    # (and reuses the baseline run); psigma>0 gets its own ``_ps<psigma>`` folder.
    stress_dependent = float(psigma) != 0.0
    # %.3f, not %.2f: psigma values differing in the third decimal (0.035 vs
    # 0.045) would otherwise collapse onto the SAME folder. K is in the name
    # too, since it changes the gate as much as psigma does.
    _mtag = "" if stress_hill_exponent is None else "_m%d" % stress_hill_exponent
    # name_prefix exists because the psigma=0 folder name carries NO mechanics,
    # so a run with different mechanics would silently overwrite (or, worse, be
    # reused as) an earlier one. The v2 fit changed the model - contractility
    # instead of perimeter elasticity, no bending, a new A0 - so it must not
    # share "fullmodel_<array>" with the pre-v2 runs already on disk.
    name = ("%s_ps%s_ks%.3f%s_%s"
            % (name_prefix, _psigma_tag(psigma), stress_shift, _mtag, initial)
            if stress_dependent else "%s_%s" % (name_prefix, initial))
    print("[full model] %s  (gammaSC=%.4g alphaHC=%.4g hc_p0=%.4g sc_p0=%.4g | "
          "pS=%.4g pR=%.4g atoh=%.6g psigma=%.4g)"
          % (name, gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index,
             notch_sensitivity, repressor_sensitivity, atoh_sensitivity, psigma), flush=True)
    # no_differentiation=False -> differentiation runs; quasi_static is always on
    # inside run(). shape_index=0 because the per-type hc/sc_shape_index carry the
    # target perimeter.
    return run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial_sheet_name=initial,
               name=name, no_differentiation=False, stress_dependent=stress_dependent,
               end_on_steady_state=True, t_end=t_end, dt=dt, divisions=False,
               shape_index=0.0, hc_shape_index=hc_shape_index,
               sc_shape_index=sc_shape_index, bending=bending, line_tension=line_tension,
               quasi_static_threshold=quasi_static_threshold,
               preferred_area_override=preferred_area,
               save_interval=save_interval, stress_shift=stress_shift,
               stress_hill_exponent=stress_hill_exponent,
               atoh_sensitivity=atoh_sensitivity, notch_sensitivity=notch_sensitivity,
               repressor_sensitivity=repressor_sensitivity,
               # Fresh run: force an undifferentiated U(0, initial_notch_delta_level)
               # seed (ignore any LI columns the loaded morphology carries). Resume:
               # do NOT re-seed — preserve the archive's evolved LI state.
               randomize_notch_delta_levels=(not continue_existing_run),
               initial_notch_delta_level=initial_notch_delta_level,
               reuse_existing_run=(reuse_existing_run or continue_existing_run),
               continue_existing_run=continue_existing_run)


def run_full_model_arrays(stage, gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index,
                          atoh_sensitivity, notch_sensitivity=0.1, repressor_sensitivity=0.3,
                          gammaHC_ratio=1.0, bending=0.02, line_tension=None,
                          quasi_static_threshold=0.03, t_end=25, dt=0.01,
                          initial_notch_delta_level=0.01, psigma=0.0,
                          preferred_area=None,
                          # Dense recording is what filled the VM disk: every solver
                          # step is ~1 GB per run at t_end=50. The differentiation
                          # analysis traces cells across frames so it needs many, but
                          # 0.1 gives 500 frames - ~10x smaller and still finer than
                          # the 0.25 used for stress profiling.
                          save_interval=0.1, stress_shift=0.0,
                          stress_hill_exponent=None,
                          indices=None, n_arrays=10, n_workers=None,
                          reuse_existing_run=False, continue_existing_run=False,
                          name_prefix="fullmodel"):
    """Run the FULL model (differentiation + quasi-static mechanics) on ``stage``'s
    fitted-initial-morphology arrays, from an UNDIFFERENTIATED start, using that
    stage's best-fit MECHANICAL parameters (``gammaSC``, ``alphaHC_ratio``,
    ``hc_shape_index``, ``sc_shape_index`` — with ``gammaHC_ratio`` fixed at 1.0
    and ``bending`` as in the fit). The LATERAL-INHIBITION parameters
    (``notch_sensitivity`` = pS, ``repressor_sensitivity`` = pR,
    ``atoh_sensitivity`` = the delta threshold) are the SAME for both stages;
    only the mechanics differ. The arrays are independent, so they run in
    parallel across ``n_workers`` processes (default ``min(#arrays, cpu_count)``;
    pass ``n_workers=1`` for serial). Each result lands in
    ``fullmodel_<array-name>``. ``continue_existing_run=True`` resumes an
    interrupted set. Returns the run names."""
    if indices is None:
        indices = range(n_arrays)
    initial_sheets = [initial_morphology_name(i, stage) for i in indices]
    tasks = [(initial, gammaSC, gammaHC_ratio, alphaHC_ratio, hc_shape_index,
              sc_shape_index, atoh_sensitivity, notch_sensitivity, repressor_sensitivity,
              bending, line_tension, quasi_static_threshold, t_end, dt,
              initial_notch_delta_level, psigma, preferred_area,
              save_interval, stress_shift, stress_hill_exponent,
              reuse_existing_run, continue_existing_run, name_prefix)
             for initial in initial_sheets]
    if n_workers is None:
        n_workers = min(len(tasks), os.cpu_count() or 1)
    print("[full model] %s: %d array(s) on %d worker(s)  |  mechanics "
          "gammaSC=%.4g alphaHC=%.4g hc_p0=%.4g sc_p0=%.4g  |  LI pS=%.4g pR=%.4g "
          "atoh=%.6g init~U(0,%.3g)  |  bending=%s qst=%.4g t_end=%g"
          % (stage, len(tasks), n_workers, gammaSC, alphaHC_ratio, hc_shape_index,
             sc_shape_index, notch_sensitivity, repressor_sensitivity, atoh_sensitivity,
             initial_notch_delta_level, bending, quasi_static_threshold, t_end), flush=True)
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            names = list(executor.map(_run_full_model_one, tasks))
    else:
        names = [_run_full_model_one(task) for task in tasks]
    return names


# Markers run() writes to ``<results_dir>/debug.log``; a re-launched fit reads
# them back to classify an existing run without re-simulating it (see run()'s
# ``reuse_existing_run``). Kept as module constants so the writer (run()) and
# the reader (_classify_existing_run) can't drift apart.
# ---------------------------------------------------------------------------
# Differentiation-score cache.
#
# compare_full_model_differentiation_to_experiments is EXPENSIVE: for each run
# it rescans every recorded frame to find the best-matching initial frame, then
# traces every final HC backwards to its differentiation time. At save_interval
# 0.1 and t_end 100 that is ~1000 frames x 10 runs per psigma point, and it is
# pure re-derivation - the answer only changes when a RUN changes.
#
# So the score is cached against a fingerprint of the runs it was computed from:
# (name, size, mtime) of each history.hf5. Resuming/extending a run rewrites
# that file (see _rewrite_history_for_resume), so a changed fingerprint is
# exactly the signal to recompute. A point whose runs have all settled is never
# extended again, so it always hits the cache on a re-run with a larger t_end -
# which is the case this exists for. A point still short of steady state gets
# extended, its fingerprint moves, and it is correctly recomputed.
_SCORE_CACHE_FILE = "differentiation_score_cache.json"


def _history_fingerprint(model_names):
    """Stat-only identity of a point's runs. Cheap: no HDF5 or log is opened."""
    out = []
    for name in model_names:
        path = os.path.join(RESULTS_DIR, name, "history.hf5")
        try:
            st = os.stat(path)
            out.append([name, int(st.st_size), round(float(st.st_mtime), 3)])
        except OSError:
            out.append([name, None, None])
    return out


def _score_cache_key(stage, psigma, stress_shift, stress_hill_exponent,
                     type_by, threshold, max_number_of_neighbors):
    """Everything that changes the score but is NOT captured by the fingerprint."""
    return ("%s|ps=%.5f|ks=%.4f|m=%s|by=%s|thr=%.6f|nb=%d"
            % (stage, float(psigma), float(stress_shift),
               "default" if stress_hill_exponent is None else stress_hill_exponent,
               type_by, float(threshold), int(max_number_of_neighbors)))


def _load_score_cache():
    import json
    try:
        with open(os.path.join(RESULTS_DIR, _SCORE_CACHE_FILE)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}          # missing or corrupt -> just recompute everything


def _store_score_cache(cache):
    import json
    path = os.path.join(RESULTS_DIR, _SCORE_CACHE_FILE)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cache, fh, indent=1)
        os.replace(tmp, path)      # atomic: a killed sweep can't leave it torn
    except OSError:
        pass                       # caching is an optimisation, never fatal


_RUN_SUCCESS_MARKER = "run() finished successfully"
_RUN_CRASH_MARKER = "run() crashed"
# Substrings that appear in a crashing traceback when the failure is INTRINSIC
# to the dynamics (the parameter point is genuinely bad -> worst-case), as
# opposed to an external interruption (disk full, killed process -> resume):
#   - "dt fell below"      : the solver's dt-floor RuntimeError, raised for
#                            non-convergence, a self-intersecting fold, or a
#                            negative-area cell (solvers.py).
#   - "worst-case scoring" : the slow-progress / wall-clock non-progress guard
#                            RuntimeError (solvers.py).
_STALL_SIGNATURES = ("dt fell below", "worst-case scoring")
# Logged by the solver when the steady-state criterion is met (solvers.py), which
# is how a run that SETTLED is told apart from one that merely ran out of t_end.
_STEADY_STATE_MARKER = "steady state reached at t="


def _classify_existing_run(results_dir):
    """Classify a pre-existing run folder as ``"completed"``, ``"stalled"`` or
    ``"interrupted"`` by reading its ``debug.log`` (see run()'s
    ``reuse_existing_run``). Robust to resume-appended logs: the LAST status
    marker wins, so a run that crashed, was resumed and then finished reads as
    ``"completed"``.

    A missing/unreadable log, or one that ends with neither a success nor a
    recognized stall (e.g. the process was killed mid-step), is treated as
    ``"interrupted"`` — the safe default, since the caller falls back to a
    fresh run when there is also no snapshot to resume from."""
    log_path = os.path.join(results_dir, "debug.log")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return "interrupted"

    last_success = text.rfind(_RUN_SUCCESS_MARKER)
    last_crash = text.rfind(_RUN_CRASH_MARKER)
    if last_success > last_crash:
        # Covers the "no crash at all" case too (last_crash == -1).
        return "completed"
    if last_crash == -1:
        # Neither a success nor a crash marker: the run died without logging an
        # outcome (killed / power loss). Resume-eligible.
        return "interrupted"
    # The most recent outcome is a crash; inspect ITS traceback (everything
    # after the crash marker) for an intrinsic-failure signature.
    tail = text[last_crash:]
    if any(sig in tail for sig in _STALL_SIGNATURES):
        return "stalled"
    return "interrupted"


def _reached_steady_state(results_dir):
    """True when the run's FINAL segment ended by satisfying the steady-state
    criterion rather than simply running out of ``t_end``.

    Only the text after the last crash marker is inspected, so a run that
    settled, was later resumed and then died does not read as settled. A
    missing/unreadable log reads as False — the conservative answer, since the
    caller then extends a run it cannot prove is finished."""
    try:
        with open(os.path.join(results_dir, "debug.log"), "r",
                  encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return False
    # rfind == -1 (no crash) -> text[0:], i.e. the whole log.
    return _STEADY_STATE_MARKER in text[text.rfind(_RUN_CRASH_MARKER) + 1:]


def _latest_archive_time(results_dir):
    """Latest snapshot time in ``<results_dir>/history.hf5``, or ``None`` when
    the archive is missing, empty or unreadable (nothing to resume from)."""
    path = os.path.join(results_dir, "history.hf5")
    if not os.path.isfile(path):
        return None
    try:
        hist = HistoryHdf5.from_archive(path, eptm_class=VirtualSheet)
        stamps = hist.time_stamps
        if len(stamps) == 0:
            return None
        return float(np.max(stamps))
    except Exception:
        return None


def _strip_results_prefix(name):
    """``run`` returns ``<RESULTS_DIR>\\<name>`` on a fresh run but ``<name>`` on
    a cache hit (the directory-already-exists early return). ``load_history_file``
    expects the bare name, so normalize both forms to a prefix-less name. Also
    tolerates a legacy literal ``results`` prefix (e.g. test doubles that return
    ``results/fake``)."""
    norm = os.path.normpath(name)
    for base in (os.path.normpath(RESULTS_DIR), "results"):
        if norm == base:
            return ""
        prefix = base + os.sep
        if norm.startswith(prefix):
            return norm[len(prefix):]
    return name


def _li_levels_kwargs_for_initial_sheet(initial_sheet_name):
    """Resolve the per-cell lateral-inhibition initial-value files that sit
    next to an initial sheet's history archive.

    Looks for ``notch_levels.npy`` / ``delta_levels.npy`` /
    ``repressor_levels.npy`` in ``results/<initial_sheet_name>/`` (the same
    folder as ``history.hf5``). Each array is keyed by ``unique_id`` (entry
    ``i`` -> cell with ``unique_id == i``) and is handed to ``run`` so the
    initial sheet starts from those notch / delta / repressor levels.

    Returns a kwargs dict for ``run``:

    - all three files present  -> the three paths,
    - none present             -> ``{}`` (keep the previous behaviour: use
      whatever LI levels the loaded history carries, else a random seed),
    - some but not all present -> ``FileNotFoundError`` (a partial drop-in
      is almost certainly a mistake, and silently mixing loaded + random
      channels would corrupt the fit's cell-type assignment).
    """
    folder = os.path.join(RESULTS_DIR, initial_sheet_name)
    files = {
        "notch_levels": os.path.join(folder, "notch_levels.npy"),
        "delta_levels": os.path.join(folder, "delta_levels.npy"),
        "repressor_levels": os.path.join(folder, "repressor_levels.npy"),
    }
    present = {kw: path for kw, path in files.items() if os.path.isfile(path)}
    if not present:
        return {}
    if len(present) != len(files):
        missing = [os.path.basename(path)
                   for kw, path in files.items() if kw not in present]
        raise FileNotFoundError(
            f"Incomplete lateral-inhibition initial-value files in {folder}: "
            f"missing {missing}. Provide all of notch_levels.npy, "
            f"delta_levels.npy and repressor_levels.npy, or none."
        )
    return present


def _load_saved_threshold(initial_sheet_name, results_dir=None):
    """Read the per-array HC/SC classification threshold written next to a
    model's history archive at ``<RESULTS_DIR>/<initial_sheet_name>/threshold.npy``
    (see :func:`post_processing.save_li_levels_from_best_pval_jsonl`, which
    fills it from the JSONL ``D_threshold_mean``). Returns the scalar float;
    raises ``FileNotFoundError`` if the file is absent.

    ``results_dir`` defaults to the module-level :data:`RESULTS_DIR`, resolved at
    CALL time (not bound as a default-arg value) so an env override / test
    monkeypatch of ``RESULTS_DIR`` is honoured."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    path = os.path.join(results_dir, initial_sheet_name, "threshold.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No saved threshold for initial sheet %r at %s. Generate it with "
            "post_processing.save_li_levels_from_best_pval_jsonl, or call "
            "find_mechanical_parameters with use_saved_threshold=False."
            % (initial_sheet_name, path))
    return float(np.load(path))


# The per-term model distributions the worker extracts and the fit's objective
# scores (sum of z**2, z = standardized model-vs-experiment mean discrepancy).
# Kept as one constant so the objective, the pooled comparison, and the JSONL
# trace all agree. The ablation terms are only active when cells are ablated
# (``ablated_cells`` non-empty); roundness is always active.
# v2 (contractility fit): the two absolute roundness terms became ONE HC/SC
# roundness ratio, and the two ablation terms ONE HC/SC ablation ratio. Three
# terms now. extract_model_mechanics still returns the absolute distributions
# for diagnostics; they are just not scored. See _MECHANICS_EXPERIMENTAL_TYPE.
MECHANICS_TERMS = ("roundness_ratio", "ablation_ratio", "shrinkage")
_MECHANICS_ROUNDNESS_TERMS = ("roundness_ratio",)

# History snapshots for a FIT run are recorded every this many units of SIMULATION
# time instead of every solver dt. The objective reads only two frames: the base
# run's LAST (steady-state) frame and the ablation run's FIRST + LAST — all of
# which the solver records unconditionally (the first at entry, the last right
# before the steady-state break), so a coarse interval cannot lose them. Dense
# recording made a single base history ~700 MB and an ablation history ~1.6 GB,
# which filled a 128 GB disk mid-fit; at this cadence a run is ~15-20 MB.
_FIT_SAVE_INTERVAL = 10.0
_MECHANICS_ABLATION_TERMS = ("ablation_ratio",)
# A degenerate ACTIVE term (no usable model data at a bad parameter point) is
# scored this many sigma so the point gets a large but FINITE penalty — the
# objective is a sum of z**2, so this keeps it comparable instead of NaN/inf.
_WORST_CASE_NSIGMA = 1e3


def _evaluate_mechanics_for_sheet(args):
    """Run one initial sheet to steady state (and, if cells are ablated, the
    ablation run too), then return its per-term model DISTRIBUTIONS (not p
    values): ``{"area", "hc_roundness", "sc_roundness", "ablation"}`` arrays (the
    ablation entry is ``None`` when no cells are ablated). Returns ``None`` if the
    run / extraction degenerates (see below) so the caller simply drops that sheet
    from the pool.

    Distributions from every initial sheet are POOLED by
    :func:`find_mechanical_parameters`' objective and compared to the experiments
    with ONE hierarchical call per term (10 model runs vs 3 experimental
    repeats), instead of scoring each sheet against the experiments separately.
    That is why this worker no longer returns p values — the comparison happens
    once, after pooling.

    Module-level and single-argument so it is picklable for
    ``ProcessPoolExecutor`` — this is the unit parallelized across initial
    sheets. Area/roundness come from the un-ablated steady state; only the
    ablation term needs the extra ablation simulation, which is skipped
    entirely when ``ablated_cells`` is empty.

    The initial sheet's lateral-inhibition starting state is seeded from the
    ``{notch,delta,repressor}_levels.npy`` files sitting next to its history
    archive (see :func:`_li_levels_kwargs_for_initial_sheet`). The ablation
    run reloads the un-ablated run's own archive, which already carries those
    levels, so it doesn't re-load the files.

    ``fix_threshold`` and ``type_by`` are passed straight to
    :func:`extract_model_mechanics` so the HC/SC classification is identical for
    every sheet; ``fix_threshold=None`` keeps the per-run calculated mid-range
    threshold.
    """
    (gammaSC, gammaHC_ratio, alphaHC_ratio, initial, experimental_stage,
     ablated_cells, post_ablation_frame, fix_threshold, type_by,
     max_wall_seconds, min_progress_rate, progress_window_seconds,
     rerun_stalled_runs, shape_index,
     base_quasi_static_threshold, ablation_quasi_static_threshold, line_tension,
     hc_shape_index, sc_shape_index, bending, preferred_area) = args
    # Optional non-progress safety net: a crawling parameter region bails with a
    # RuntimeError (caught below) so it scores worst-case instead of stalling.
    stall_kwargs = dict(max_wall_seconds=max_wall_seconds,
                        min_progress_rate=min_progress_rate,
                        progress_window_seconds=progress_window_seconds)

    # When the classification threshold is a DELTA threshold (type_by ==
    # 'delta_level' — e.g. the per-sheet values loaded from threshold.npy via
    # use_saved_threshold), drive the simulation's atoh_sensitivity from it too,
    # so the MODEL itself makes cells with delta ABOVE the threshold high-atoh
    # (HC) and below it low-atoh (SC). atoh_sensitivity is the delta half-max of
    # the Atoh1 Hill, so atoh_level crosses the 0.5 differentiation threshold
    # exactly at delta == threshold. This keeps the simulated cell types /
    # mechanics consistent with how the same cells are classified for the
    # comparison, instead of gating atoh at a fixed default while classifying at
    # a different (loaded) threshold.
    atoh_kwargs = {}
    if type_by == "delta_level" and fix_threshold is not None:
        atoh_kwargs["atoh_sensitivity"] = fix_threshold

    # The base run's results-folder name is DETERMINISTIC from (initial sheet,
    # parameters) — compute it up front so every START / DONE / FAILED line names
    # the exact folder on disk, INCLUDING a failure in the p-value comparison
    # that happens AFTER the runs finished (where the exception traceback alone
    # doesn't say which run it was for).
    model_name = _short_run_folder_name(
        initial, gammaSC, gammaHC_ratio, alphaHC_ratio, 0,
        atoh_sensitivity=atoh_kwargs.get("atoh_sensitivity", _DEFAULT_ATOH_SENSITIVITY),
        shape_index=shape_index,
        quasi_static_threshold=base_quasi_static_threshold,
        line_tension=line_tension, bending=bending,
        hc_shape_index=hc_shape_index, sc_shape_index=sc_shape_index,
        preferred_area=preferred_area)
    print("[mechanics] START  %s  (initial=%s, gammaSC=%.4g, gammaHC_ratio=%.4g, "
          "alphaHC_ratio=%.4g, shape_index=%.4g)" % (model_name, initial, gammaSC,
                                   gammaHC_ratio, alphaHC_ratio, shape_index), flush=True)

    # Resolved BEFORE the try so a genuine config error (e.g. partial LI files)
    # still fails loudly instead of being masked as a "bad parameter point".
    li_kwargs = _li_levels_kwargs_for_initial_sheet(initial)

    try:
        base_name = _strip_results_prefix(
            run(gammaSC, gammaHC_ratio, alphaHC_ratio, 0, initial,
                no_differentiation=True, reuse_existing_run=True,
                rerun_stalled_runs=rerun_stalled_runs, shape_index=shape_index,
                quasi_static_threshold=base_quasi_static_threshold,
                line_tension=line_tension, bending=bending,
                hc_shape_index=hc_shape_index, sc_shape_index=sc_shape_index,
                preferred_area_override=preferred_area,
                save_interval=_FIT_SAVE_INTERVAL,
                **li_kwargs, **stall_kwargs, **atoh_kwargs))

        ablation_kwargs = {}
        if ablated_cells:
            try:
                with_ablation_name = _strip_results_prefix(
                    run(gammaSC, gammaHC_ratio, alphaHC_ratio, 0, base_name,
                        no_differentiation=True, ablated_cells=ablated_cells,
                        reuse_existing_run=True, rerun_stalled_runs=rerun_stalled_runs,
                        shape_index=shape_index,
                        quasi_static_threshold=ablation_quasi_static_threshold,
                        line_tension=line_tension, bending=bending,
                        hc_shape_index=hc_shape_index, sc_shape_index=sc_shape_index,
                        preferred_area_override=preferred_area,
                        # Safe to coarsen ONLY when the analysis reads the LAST
                        # frame (-1), which is always recorded. A positive frame
                        # INDEX would point at a different simulation time under
                        # a coarse cadence, so keep those dense.
                        save_interval=(_FIT_SAVE_INTERVAL
                                       if post_ablation_frame == -1 else None),
                        **stall_kwargs, **atoh_kwargs))
                ablation_kwargs = dict(ablation_model_name=with_ablation_name,
                                       ablated_cells=ablated_cells,
                                       post_ablation_frame=post_ablation_frame)
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001
                # The ablation run is EXTRA (only the ablation term needs it). If
                # it degenerates — e.g. at stiff parameters a virtual-vertex
                # collapse cascade empties a face and tyssue's
                # drop_two_sided_faces indexes face_df with a face-less mask
                # (pandas IndexingError) — don't throw away the base run's
                # perfectly good area/roundness. Skip ONLY the ablation term for
                # this sheet; the pool tolerates sheets that contribute no
                # ablation ratio.
                print("[mechanics] ABLATION FAILED %s  (%s: %s); keeping base run, "
                      "skipping ablation term"
                      % (model_name, type(exc).__name__, exc), flush=True)
                ablation_kwargs = {}

        terms = extract_model_mechanics(
            base_name, type_by=type_by, threshold=fix_threshold, **ablation_kwargs)
        print("[mechanics] DONE   %s  %s"
              % (model_name, "  ".join(
                  "%s n=%d" % (t, 0 if terms.get(t) is None else len(terms[t]))
                  for t in MECHANICS_TERMS)), flush=True)
        return terms
    except FileNotFoundError:
        # A genuine config error (e.g. missing LI-level .npy files) is NOT a
        # degenerate parameter point — let it propagate loudly.
        raise
    except Exception as exc:  # noqa: BLE001
        # A single initial sheet can legitimately fail at extreme parameters: the
        # stiff steady-state geometry degenerates and the simulation / extraction
        # blows up in OPEN-ENDED ways — the solver hits its dt floor
        # (RuntimeError), a KDE inverts a singular covariance (LinAlgError), a
        # ratio divides by ~zero (ZeroDivisionError / FloatingPointError), the
        # hierarchical fit drops the stage coefficient (KeyError), or a topology
        # collapse cascade empties a face so tyssue's drop_two_sided_faces indexes
        # with a misaligned mask (pandas IndexingError). Enumerating every such
        # type proved brittle (each extreme corner finds a new one and, uncaught
        # inside a ProcessPoolExecutor worker, tears down the WHOLE optimization),
        # so ANY failure of the simulation / extraction is treated as "this sheet
        # degenerated": return None so the objective DROPS it from the pool and
        # compares the survivors; only if EVERY sheet degenerates is the point
        # scored worst-case. Genuine config errors (missing LI files ->
        # FileNotFoundError) are re-raised above so they still surface.
        print("[mechanics] FAILED %s  (%s: %s); dropping this sheet from the pool"
              % (model_name, type(exc).__name__, exc), flush=True)
        logging.getLogger("run_model").warning(
            "mechanics evaluation failed for %s (initial=%r) at gammaSC=%.4g "
            "gammaHC_ratio=%.4g alphaHC_ratio=%.4g (%s: %s); dropping this sheet "
            "from the pool",
            model_name, initial, gammaSC, gammaHC_ratio, alphaHC_ratio,
            type(exc).__name__, exc)
        return None


def find_mechanical_parameters(experimental_stage, initial_sheets=None,
                               indices=None, n_sheets=10,
                               gammaSC_bounds=(0.001, 0.1),
                               gammaHC_ratio_bounds=(1.0, 20.0),
                               alphaHC_ratio_bounds=(0.1, 5.0),
                               shape_index_bounds=None,
                               hc_shape_index_bounds=None,
                               sc_shape_index_bounds=None,
                               gammaHC_ratio_fixed=1.0,
                               alphaHC_ratio_fixed=1.0,
                               ablated_cells=(), post_ablation_frame=-1,
                               n_calls=40, n_initial_points=10,
                               n_workers=None, random_state=0,
                               pval_floor=1e-300, x0=None,
                               fix_threshold=None, type_by='atoh_level',
                               use_saved_threshold=False,
                               max_wall_seconds=None, min_progress_rate=None,
                               progress_window_seconds=30.0,
                               rerun_stalled_runs=False,
                               landscape_resolution=20,
                               base_quasi_static_threshold=_DEFAULT_QUASI_STATIC_THRESHOLD,
                               ablation_quasi_static_threshold=_DEFAULT_QUASI_STATIC_THRESHOLD,
                               line_tension=0.2, bending=None,
                               preferred_area=None):
    """Find the mechanical parameters (gammaSC, gammaHC_ratio, alphaHC_ratio)
    that best fit the experimental measurements, via Gaussian-process
    Bayesian optimization.

    For each candidate parameter point every initial sheet in ``initial_sheets``
    is simulated (:func:`_evaluate_mechanics_for_sheet`) and its per-term model
    distributions are extracted; the runs are then POOLED and each of the four
    objectives — HC roundness, SC roundness, HC area-change-after-ablation, SC
    area-change-after-ablation — is scored by a STANDARDIZED MEAN DISCREPANCY
    (:func:`compare_pooled_model_mechanics_to_experiments`):
    ``z = (mean_model - mean_exp) / SEM_exp``, where ``SEM_exp`` is the standard
    error of the experimental biological-repeat means. We MINIMIZE the sum of
    ``z**2`` over the active terms — the model means landing within the
    experimental replicate uncertainty gives ~0. This replaced a per-term p-value
    objective, which saturated at ~0 almost everywhere (a flat landscape that
    pinned the optimizer at the bounds). The ablation terms are active only when
    ``ablated_cells`` is non-empty. A run that degenerates is dropped from the
    pool; an active term left with no usable data is penalized worst-case.

    Lateral-inhibition initial state
    --------------------------------
    Each initial sheet is started from the per-cell notch / delta / repressor
    levels stored alongside its history archive, in
    ``results/<initial_sheet>/{notch,delta,repressor}_levels.npy`` (arrays
    keyed by ``unique_id`` — entry ``i`` is the value for the cell with
    ``unique_id == i``). All three files must be present together; if none are
    present the previous behaviour is kept (use the levels carried by the
    loaded history, else a random seed). This fixes the cell-type assignment
    per sheet so the mechanics comparison is reproducible across candidates.

    Speedups baked in:

    * **Bayesian optimization** instead of finite-difference gradient descent
      — global, derivative-free, and noise-aware, so it needs far fewer
      (expensive) simulations and won't stall on the flat product landscape.
    * **Parallel evaluation** across initial sheets (``ProcessPoolExecutor``).
    * **No ablation runs for the area/roundness terms** — those read the
      un-ablated steady state; the ablation simulation runs only when
      ``ablated_cells`` is given.
    * **Caching / common random numbers** — the same ``initial_sheets`` are
      reused for every candidate (CRN, lower-variance comparisons), an
      in-process cache skips re-evaluating parameter points that round to an
      already-seen value, and ``run`` itself reuses an existing results folder.
    * **Box-constrained parameters** via the ``*_bounds`` arguments.

    Parameters
    ----------
    experimental_stage : str
        ``"E17.5"`` or ``"P0"``.
    initial_sheets : sequence of str, optional
        Explicit initial-sheet result names to average over. When omitted they
        are built from ``indices`` (or ``range(n_sheets)``) as the stage's
        fitted-initial-morphology arrays via ``initial_morphology_name``.
    indices : sequence of int, optional
        Which array indices to use when ``initial_sheets`` is not given.
    n_sheets : int
        Number of arrays (``range(n_sheets)``) when neither ``initial_sheets``
        nor ``indices`` is given.
    gammaSC_bounds, gammaHC_ratio_bounds, alphaHC_ratio_bounds : (low, high)
        Search box for each parameter.
    ablated_cells : sequence of int
        Cells to ablate for the ablation term; empty disables that term.
    post_ablation_frame : int
        Frame passed to the ablation comparison.
    base_quasi_static_threshold, ablation_quasi_static_threshold : float
        Mechanical steady-state cutoffs (max per-step vertex speed) for the base
        (un-ablated) and ablation relaxations. BOTH default to the historical
        0.01 so a new optimization REUSES the existing 0.01 run folders (a
        non-0.01 value is folded into the run hash, minting a distinct folder that
        never reuses a differently-relaxed archive). For a FRESH fit where
        re-running is acceptable, pass 0.03 / 0.02: analysis showed the measured
        outputs converge well before the velocity reaches 0.01 (the tail is wasted
        compute), while the ablation area-change stays more threshold-sensitive so
        it wants the tighter cutoff — this trims ~20-30% off run time but cannot
        reuse the 0.01 archives.
    hc_shape_index_bounds, sc_shape_index_bounds : (low, high), optional
        Search box for the TYPE-DEPENDENT shape index (separate target perimeter
        P0 = p0*sqrt(A0) for HC and SC). Give these INSTEAD of
        ``shape_index_bounds`` to let HC and SC roundness be matched
        independently — one shared shape_index drives them together, which is
        what caps the roundness fit. Pair with ``gammaHC_ratio_bounds=None`` to
        keep exactly 4 fitted parameters (one per comparison term).
    gammaHC_ratio_fixed : float, default 1.0
        Value ``gammaHC_ratio`` is held at when ``gammaHC_ratio_bounds`` is None
        (1.0 = HC and SC share one contractility). More generally, ANY parameter
        whose ``*_bounds`` is None is fixed rather than fitted, so the
        parameterisation is chosen at the call site.
    line_tension : float, optional
        Edge line tension applied to every cell-type pair for both the base and
        ablation runs (default ``None`` = off, the historical behaviour). A small
        value (~0.05) penalises jagged, high-curvature bonds and smooths the cell
        boundaries. Folded into the run hash, so a line-tension fit gets its own
        folders and never reuses the no-line-tension archives.
    n_calls, n_initial_points : int
        Total simulations-backed evaluations and the size of the space-filling
        initial design.
    n_workers : int, optional
        Parallel worker processes (default: ``min(#sheets, cpu_count)``).
    random_state : int
        Seed for the optimizer.
    pval_floor : float
        p values are clipped up to this floor before taking logs, so a zero /
        underflowed p value gives a large-but-finite penalty instead of -inf.
    x0 : array-like, optional
        Optional initial guess ``[gammaSC, gammaHC_ratio, alphaHC_ratio, P0]``
        evaluated before the design.
    fix_threshold : float, optional
        Fixed HC/SC classification threshold on ``type_by`` to use for EVERY
        sheet (passed straight to :func:`extract_model_mechanics`). When ``None``
        (the default) each run computes its own threshold as the mid-range
        ``(max + min) / 2`` of ``type_by`` — supplying a value here pins it so
        the HC/SC split is identical across sheets and candidates.
    type_by : str, default ``'atoh_level'``
        ``face_df`` column used to classify cells as HC vs SC in every
        comparison. Cells with ``type_by`` above the threshold are HCs.
    use_saved_threshold : bool, default False
        When True, each initial sheet's HC/SC classification threshold is read
        from ``results/<initial_sheet>/threshold.npy`` (written by
        :func:`post_processing.save_li_levels_from_best_pval_jsonl` from the
        JSONL ``D_threshold_mean``) and used as the threshold for THAT sheet's
        comparisons — and its derivatives (the ablation run forks from it, and
        all three p-value terms share the same threshold). This overrides
        ``fix_threshold`` (which is a single value for all sheets). Because the
        saved value is a DELTA threshold, pair it with ``type_by='delta_level'``.
        Raises if a sheet has no ``threshold.npy``.
    rerun_stalled_runs : bool, default False
        Controls what happens when a re-launched fit hits a results folder
        left over from a previous stalled evaluation (solver dt floor / fold
        / negative-area cell / non-progress guard). False (the default,
        historical behaviour) scores that point worst-case immediately
        without re-running it. True re-runs it from scratch instead — useful
        after a code change that might no longer stall on that point, or
        when you'd rather pay for a fresh simulation than accept an
        automatic worst-case score. Passed straight through to :func:`run`
        as its own ``rerun_stalled_runs`` argument for every evaluation
        (base run and, if ``ablated_cells`` is given, the ablation run).

    Returns
    -------
    best_params : numpy.ndarray
        ``[gammaSC, gammaHC_ratio, alphaHC_ratio]`` of the best fit.
    result : dict
        Full optimizer trace (``x``, ``fun``, ``X``, ``y``) from
        :func:`bayesian_optimization.minimize`.
    """
    import bayesian_optimization as bo
    import json
    from concurrent.futures import ProcessPoolExecutor

    if initial_sheets is None:
        # Build the stage's fitted-initial-morphology sheets by index
        # (``indices``, e.g. a subset, or ``range(n_sheets)`` by default).
        if indices is None:
            indices = range(n_sheets)
        initial_sheets = [initial_morphology_name(i, experimental_stage) for i in indices]
    initial_sheets = list(initial_sheets)
    ablated_cells = list(ablated_cells)
    # Per-sheet HC/SC threshold, loaded once (it is static per sheet). When
    # use_saved_threshold is on, each sheet uses its own threshold.npy value
    # (overriding the single global fix_threshold); otherwise every sheet uses
    # fix_threshold (None -> each comparison computes its own mid-range).
    if use_saved_threshold:
        per_sheet_threshold = {init: _load_saved_threshold(init) for init in initial_sheets}
    else:
        per_sheet_threshold = None
    # --- which mechanical parameters the optimizer actually fits -------------
    # A parameter is FITTED when its ``*_bounds`` is given, and FIXED at the
    # companion default otherwise. The parameterisation is therefore chosen
    # entirely at the CALL SITE, so switching between them (or reverting) is an
    # argument change, not a code change. Two useful boxes, both 4-dimensional
    # (one parameter per comparison term):
    #
    #   historical : gammaSC, gammaHC_ratio, alphaHC_ratio, shape_index
    #                -> pass gammaHC_ratio_bounds + shape_index_bounds
    #   per-type P0: gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index
    #                -> pass hc_/sc_shape_index_bounds and gammaHC_ratio_bounds=None
    #                   (gamma then fixed at gammaHC_ratio_fixed, i.e. HC and SC
    #                   share one contractility). A SPLIT target perimeter is what
    #                   lets HC and SC roundness be matched independently; one
    #                   shared shape_index drives them together.
    #
    # shape_index (target perimeter P0 = p0*sqrt(A0)): a good fluid-regime range
    # is roughly (0, 4.5) — the vertex-model rigidity transition is ~3.81. Left
    # off, it stays 0 and the perimeter effector is plain contractility.
    _param_spec = (
        ("gammaSC",        gammaSC_bounds,        None),
        ("gammaHC_ratio",  gammaHC_ratio_bounds,  gammaHC_ratio_fixed),
        ("alphaHC_ratio",  alphaHC_ratio_bounds,  alphaHC_ratio_fixed),
        ("shape_index",    shape_index_bounds,    0.0),
        ("hc_shape_index", hc_shape_index_bounds, None),
        ("sc_shape_index", sc_shape_index_bounds, None),
    )
    bounds, param_names, fixed_params = [], [], {}
    for _pname, _pbounds, _pfixed in _param_spec:
        if _pbounds is not None:
            bounds.append(tuple(_pbounds))
            param_names.append(_pname)
        else:
            fixed_params[_pname] = _pfixed
    param_names = tuple(param_names)
    if not bounds:
        raise ValueError("no fitted parameters: give at least one *_bounds")
    print("[mechanics] fitting %d parameter(s): %s | fixed: %s"
          % (len(param_names), ", ".join(param_names),
             ", ".join("%s=%s" % (k, v) for k, v in fixed_params.items() if v is not None)
             or "(none)"), flush=True)
    if n_workers is None:
        n_workers = min(len(initial_sheets), os.cpu_count() or 1)
    cache = {}

    # Per-evaluation trace. Written INCREMENTALLY (one JSON line appended per
    # evaluated point, flushed) so a killed / crashed fit still leaves a full,
    # diagnosable record — unlike the params/objective .npy arrays, which are
    # only saved once the whole optimization finishes. Each line carries the
    # parameters, the objective (sum of z**2), the signed per-term discrepancy
    # (nsigma_<term>) and its z**2 contribution (obj_<term>), the pooled model
    # sample size per term, and which sheets contributed. Truncated at the start
    # so each fit invocation gets a clean trace.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trace_path = os.path.join(RESULTS_DIR, "%s_optimization_trace.jsonl" % experimental_stage)
    open(trace_path, "w").close()
    eval_counter = [0]

    def objective(params):
        params = [float(p) for p in params]
        # Map the optimizer vector onto NAMED parameters via param_names (the
        # order the spec above produced), then fill in whatever is held fixed.
        vals = dict(fixed_params)
        vals.update(dict(zip(param_names, params)))
        gammaSC = vals["gammaSC"]
        gammaHC_ratio = vals["gammaHC_ratio"]
        alphaHC_ratio = vals["alphaHC_ratio"]
        shape_index = vals["shape_index"]
        hc_shape_index = vals["hc_shape_index"]
        sc_shape_index = vals["sc_shape_index"]
        # ``run`` quantizes parameters to 2 decimals in the folder name, so
        # cache at that resolution to avoid recomputing equivalent points. Keyed
        # on the FITTED parameters only (the fixed ones are constant).
        key = tuple(round(vals[n], 2) for n in param_names)
        if key in cache:
            return cache[key]

        tasks = [(gammaSC, gammaHC_ratio, alphaHC_ratio, initial, experimental_stage,
                  ablated_cells, post_ablation_frame,
                  (per_sheet_threshold[initial] if per_sheet_threshold is not None
                   else fix_threshold),
                  type_by,
                  max_wall_seconds, min_progress_rate, progress_window_seconds,
                  rerun_stalled_runs, shape_index,
                  base_quasi_static_threshold, ablation_quasi_static_threshold,
                  line_tension, hc_shape_index, sc_shape_index, bending,
                  preferred_area)
                 for initial in initial_sheets]
        if n_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                details_list = list(executor.map(_evaluate_mechanics_for_sheet, tasks))
        else:
            details_list = [_evaluate_mechanics_for_sheet(task) for task in tasks]

        # Pool every SUCCEEDING sheet's per-term distributions (a sheet that
        # degenerated returns None and is dropped), then score each term by the
        # standardized model-vs-experiment mean discrepancy (n-sigma) over the
        # WHOLE pool — see compare_pooled_model_mechanics_to_experiments.
        model_terms = {term: [] for term in MECHANICS_TERMS}
        n_ok = 0
        for details in details_list:
            if details is None:
                continue
            n_ok += 1
            for term in MECHANICS_TERMS:
                # .get, not [...]: a details dict produced before the ratio
                # terms existed (or a test double) simply contributes nothing
                # rather than raising KeyError mid-fit.
                if details.get(term) is not None:
                    model_terms[term].append(details[term])

        # Active terms: roundness always; the ablation terms only when cells were
        # ablated (otherwise there is no ablation run to compare against).
        active = list(_MECHANICS_ROUNDNESS_TERMS) + ["shrinkage"]
        if ablated_cells:
            active += list(_MECHANICS_ABLATION_TERMS)

        # Objective = sum of z**2 over the ACTIVE terms (lower is better; 0 = the
        # model means land exactly on the experimental means). An active term with
        # no usable data (every sheet degenerated, or the ablation term produced no
        # cells) is a degenerate miss -> penalized _WORST_CASE_NSIGMA; an inactive
        # term contributes nothing.
        zscores = compare_pooled_model_mechanics_to_experiments(
            model_terms, experimental_stage)
        obj_terms = {}
        for term in MECHANICS_TERMS:
            if term not in active:
                obj_terms[term] = 0.0
                continue
            z = zscores[term]
            if not np.isfinite(z):
                z = _WORST_CASE_NSIGMA
            obj_terms[term] = float(z * z)
        value = float(sum(obj_terms[t] for t in active))
        cache[key] = value

        # Record this evaluation. objective == sum over ACTIVE terms of z**2, so
        # each term's signed z (nsigma_<term>, the standardized mean gap) and its
        # z**2 contribution (obj_<term>) tell you WHICH metric is off and in which
        # direction — plus the pooled model sample size per term and which sheets
        # contributed. Appended + flushed immediately so an interrupted run keeps
        # every completed step.
        eval_counter[0] += 1
        record = {
            "eval": eval_counter[0],
            "gammaSC": gammaSC, "gammaHC_ratio": gammaHC_ratio,
            "alphaHC_ratio": alphaHC_ratio, "shape_index": shape_index,
            "hc_shape_index": hc_shape_index, "sc_shape_index": sc_shape_index,
            "bending": bending,
            "fitted_params": list(param_names),
            "objective": value,
            "n_sheets": len(initial_sheets), "n_contributing": n_ok,
        }
        for term in MECHANICS_TERMS:
            record["nsigma_" + term] = float(zscores[term])   # may be nan
            record["obj_" + term] = obj_terms[term]
            record["n_" + term] = int(sum(len(a) for a in model_terms[term]))
        record["sheets"] = [
            {"initial": init, "ok": d is not None}
            for init, d in zip(initial_sheets, details_list)]
        with open(trace_path, "a") as _tf:
            _tf.write(json.dumps(record) + "\n")
        print("params gammaSC=%.4f gammaHC_ratio=%.4f alphaHC_ratio=%.4f shape_index=%.4f "
              "-> sum z^2 = %.6g  (%s; %d/%d sheets)"
              % (gammaSC, gammaHC_ratio, alphaHC_ratio, shape_index, value,
                 "  ".join("%s=%+.2f" % (t, zscores[t]) for t in active),
                 n_ok, len(initial_sheets)))
        return value

    result = bo.minimize(objective, bounds, n_calls=n_calls,
                         n_initial_points=n_initial_points,
                         random_state=random_state, x0=x0,
                         return_surrogate=True)
    best_params = result["x"]
    # param_names grows a 4th entry ("shape_index") only when it's being fitted,
    # so format against it rather than assuming 3 params (a hardcoded 3-way
    # format string crashes once shape_index is added).
    print("Best params: "
          + ", ".join("%s=%.4f" % (n, v) for n, v in zip(param_names, best_params)))
    print("Best mean -log p = %.6g (product p ~ %.3g)" % (result["fun"], np.exp(-result["fun"])))
    np.save(os.path.join(RESULTS_DIR, "%s_optimization_params.npy"%experimental_stage),result["X"])
    np.save(os.path.join(RESULTS_DIR, "%s_optimization_objective.npy" % experimental_stage), result["y"])
    print("Saved per-step trace (%d evaluations) to %s"
          % (eval_counter[0], os.path.basename(trace_path)))

    # Estimated objective landscape from the fitted GP surrogate: predict the
    # posterior mean + std on a dense grid over the search box and save the
    # bundle (grid axes + mean + std + the training points), so it can be
    # sliced / plotted later without re-running or re-fitting the GP.
    surrogate = result.get("surrogate")
    if surrogate is not None:
        axes = [np.linspace(lo, hi, landscape_resolution) for (lo, hi) in bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.column_stack([m.ravel() for m in mesh])
        mu, std = surrogate(grid)
        shape = tuple(len(a) for a in axes)
        np.savez(
            os.path.join(RESULTS_DIR, "%s_optimization_landscape.npz" % experimental_stage),
            param_names=np.asarray(param_names),
            bounds=np.asarray(bounds, float),
            axes=np.asarray(axes),
            mean=mu.reshape(shape),
            std=std.reshape(shape),
            X=result["X"], y=result["y"],
        )
        print("Saved GP landscape (%s grid) to %s_optimization_landscape.npz"
              % ("x".join(str(s) for s in shape), experimental_stage))
    else:
        print("No GP surrogate available (final GP fit failed); skipping landscape.")
    return best_params, result


def find_psigma(mechanical_params, psigma_bounds=(0.0, 2.0), n_grid=11, n_refine=2,
                n_arrays=10, indices=None, n_workers=None,
                notch_sensitivity=0.1, repressor_sensitivity=0.3,
                atoh_sensitivity=0.355079, bending=0.02,
                quasi_static_threshold=0.03, initial_notch_delta_level=0.01,
                preferred_area=None, stress_shift=0.0, save_interval=0.1,
                stress_hill_exponent=None,
                type_by='delta_level', threshold=0.355079,
                max_number_of_neighbors=2, t_end=50, dt=0.01, plot=True,
                save_json=True, use_score_cache=True):
    """Search the single mechanosensitivity ``psigma`` (SHARED across stages)
    that MINIMIZES the FULL-model differentiation score of
    :func:`compare_full_model_differentiation_to_experiments` — the sum of its
    three chi^2 scores — summed over the fitted stages.

    ``psigma`` (mechanosensitivity) gates delta production by mechanical stress,
    so it only acts when > 0 (``stress_dependent`` is turned on automatically).
    For each candidate ``psigma`` and each stage the full model (differentiation
    + quasi-static mechanics) is run on all ``n_arrays`` initial arrays via
    :func:`run_full_model_arrays` (arrays parallel across ``n_workers``), and the
    runs are POOLED and scored against experiment. ``psigma=0`` reuses the
    existing ``fullmodel_*`` baseline, so the sweep tells you directly whether
    mechano-sensitivity improves the score over psigma=0.

    A coarse-to-fine line search (``run`` quantizes ``psigma`` to 2 decimals, so
    refinement stops at a 0.01 grid). LOWER score is better (unlike the previous
    p-value version, which maximized).

    Parameters
    ----------
    mechanical_params : dict
        ``{stage: (gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index)}`` —
        each stage's best-fit mechanics (``gammaHC_ratio`` fixed at 1.0). Keys
        pick which stages are fitted.
    notch_sensitivity, repressor_sensitivity, atoh_sensitivity, bending,
    quasi_static_threshold, initial_notch_delta_level
        The SHARED lateral-inhibition / mechanics settings of the full-model
        runs (pS, pR, delta threshold, bending, qst, LI seed range).
    type_by, threshold, max_number_of_neighbors
        Forwarded to the comparison (HC/SC identity + neighbor-count cap).
    use_score_cache : bool, default True
        Reuse a previously computed differentiation score when NONE of the
        point's runs has changed since (fingerprint = size + mtime of each
        ``history.hf5``). Scoring rescans every frame of every run, so this is
        the dominant cost of re-running a sweep whose points are already
        simulated. A point whose runs have all settled is never extended again
        and therefore always hits; a point still short of steady state gets
        extended, its fingerprint moves, and it is recomputed. Set False to
        force a full recomputation.

    Returns
    -------
    best_psigma : float
        The ``psigma`` minimizing the summed differentiation score.
    scores : dict
        ``{psigma: {stage: comparison-dict, "objective": summed total}}`` for
        every evaluated ``psigma`` (the full landscape).
    """
    import json
    stages = list(mechanical_params.keys())
    # Tag outputs by stage set: a per-stage sweep must not clobber the
    # combined one (or the other stage's).
    # ...and by the GATE SHAPE. K (and the Hill exponent) change what a given
    # psigma MEANS, so a re-sweep at a different K is a different experiment and
    # must not overwrite the previous one's scores. The K=-0.060 combined sweep
    # silently clobbered the K=-0.080 E17.5 numbers, which survived only because
    # they had been copied into a docstring; hence _ks / _m in the name, matching
    # the run-folder convention.
    tag = "_".join(stages) + "_ks%.3f" % stress_shift
    if stress_hill_exponent is not None:
        tag += "_m%d" % stress_hill_exponent
    json_path = (os.path.join(RESULTS_DIR, "psigma_scores_%s.json" % tag)
                 if save_json else None)
    if json_path:
        print("[find_psigma] scores -> %s" % json_path, flush=True)
    scores = {}   # rounded psigma -> {stage: comparison-dict, "objective": total sum}
    # Scores already derived from runs that have not changed since; see
    # _SCORE_CACHE_FILE. Loaded once, written through on every miss.
    score_cache = _load_score_cache() if use_score_cache else {}

    def evaluate(psigmas):
        # De-duplicate at 3 decimals. NOT 2: psigma is now a junction-stress
        # threshold of order 0.01-0.05, so rounding to 2 decimals collapsed
        # distinct sweep points onto each other (0.015 and 0.045 became
        # 0.01 and 0.04).
        wanted = sorted({round(float(p), 5) for p in psigmas} - set(scores))
        for psigma in wanted:
            entry, objective = {}, 0.0
            for stage in stages:
                gammaSC, alphaHC_ratio, hc_p0, sc_p0 = mechanical_params[stage]
                # Run the full model on every array for this (psigma, stage);
                # psigma=0 reuses the existing fullmodel_* baseline.
                model_names = run_full_model_arrays(
                    stage, gammaSC, alphaHC_ratio, hc_p0, sc_p0,
                    atoh_sensitivity=atoh_sensitivity,
                    notch_sensitivity=notch_sensitivity,
                    repressor_sensitivity=repressor_sensitivity,
                    bending=bending, quasi_static_threshold=quasi_static_threshold,
                    initial_notch_delta_level=initial_notch_delta_level,
                    preferred_area=preferred_area, stress_shift=stress_shift,
                    stress_hill_exponent=stress_hill_exponent,
                    save_interval=save_interval,
                    psigma=psigma, t_end=t_end, dt=dt,
                    n_arrays=n_arrays, indices=indices, n_workers=n_workers,
                    reuse_existing_run=True)
                _ck = _score_cache_key(stage, psigma, stress_shift,
                                       stress_hill_exponent, type_by, threshold,
                                       max_number_of_neighbors)
                _fp = _history_fingerprint(model_names)
                _hit = score_cache.get(_ck) if use_score_cache else None
                if _hit is not None and _hit.get("fingerprint") == _fp:
                    # Not one history.hf5 has changed since this was scored, so
                    # the comparison would re-derive the identical numbers.
                    result = _hit["result"]
                    print("[find_psigma] %s psigma=%.5g: cached score reused "
                          "(%s/%s runs settled; no run changed) -> total %.3f"
                          % (stage, psigma, _hit.get("n_settled", "?"),
                             _hit.get("n_runs", len(model_names)),
                             float(result["total"])), flush=True)
                else:
                    result = compare_full_model_differentiation_to_experiments(
                        stage, model_names=model_names, type_by=type_by,
                        threshold=threshold, max_number_of_neighbors=max_number_of_neighbors)
                    if use_score_cache:
                        score_cache[_ck] = {
                            "fingerprint": _fp, "result": result,
                            "n_runs": len(model_names),
                            "n_settled": sum(
                                _reached_steady_state(os.path.join(RESULTS_DIR, n))
                                for n in model_names)}
                        _store_score_cache(score_cache)
                entry[stage] = result
                objective += float(result["total"])
            entry["objective"] = objective
            scores[psigma] = entry
            print("[find_psigma] psigma=%.5g -> total score (sum over stages) = %.3f  [%s]"
                  % (psigma, objective,
                     ", ".join("%s=%.1f" % (s, entry[s]["total"]) for s in stages)), flush=True)
            # Persist after EVERY point, not at the end: a sweep that dies
            # (disk, OOM) otherwise leaves hours of scoring only on screen.
            if json_path:
                with open(json_path, "w") as _jf:
                    json.dump({"stages": stages, "stress_shift": stress_shift,
                               "stress_hill_exponent": stress_hill_exponent,
                               "preferred_area": preferred_area,
                               "mechanical_params": {k: list(v) for k, v in
                                                     mechanical_params.items()},
                               "scores": {("%.4f" % k): v
                                          for k, v in scores.items()}},
                              _jf, indent=1)

    low, high = psigma_bounds
    current_step = (high - low) / (n_grid - 1) if n_grid > 1 else 0.0
    evaluate(np.linspace(low, high, n_grid) if n_grid > 1 else [low])
    best = min(scores, key=lambda k: scores[k]["objective"])
    for _ in range(n_refine):
        new_low = max(psigma_bounds[0], best - current_step)
        new_high = min(psigma_bounds[1], best + current_step)
        next_step = (new_high - new_low) / (n_grid - 1)
        if next_step < 0.00001:  # finer than the 5-decimal de-duplication above
            break
        evaluate(np.linspace(new_low, new_high, n_grid))
        best = min(scores, key=lambda k: scores[k]["objective"])
        current_step = next_step

    print("\n[find_psigma] BEST psigma = %.2f  (total score = %.3f; lower is better)"
          % (best, scores[best]["objective"]))
    baseline = scores.get(0.0, {}).get("objective")
    if baseline is not None and abs(best) > 1e-9:
        verdict = "IMPROVES" if scores[best]["objective"] < baseline else "does NOT improve"
        print("[find_psigma] mechano-sensitivity %s the fit: best psigma=%.5g score %.3f "
              "vs psigma=0 baseline %.3f  (delta %.3f)"
              % (verdict, best, scores[best]["objective"], baseline,
                 scores[best]["objective"] - baseline))
    elif baseline is not None:
        print("[find_psigma] best psigma is 0 -> mechano-sensitivity does NOT improve the "
              "fit (baseline score %.3f is the grid minimum)" % baseline)

    if plot:
        psigmas = sorted(scores)
        fig, ax = plt.subplots()
        ax.plot(psigmas, [scores[p]["objective"] for p in psigmas], "k-o",
                label="total (sum over stages)")
        for stage in stages:
            ax.plot(psigmas, [scores[p][stage]["total"] for p in psigmas], "--o", label=stage)
        ax.axvline(best, color="grey", ls=":")
        ax.set_xlabel("psigma (mechanosensitivity)")
        ax.set_ylabel("differentiation score (chi^2 total; lower = better)")
        ax.set_title("psigma sweep: full-model differentiation vs experiment")
        ax.legend()
        out_path = os.path.join(RESULTS_DIR, "psigma_fit_%s.png" % tag)
        plt.savefig(out_path)
        plt.close(fig)
        print("Saved psigma landscape to %s" % out_path)

    if json_path:
        print("Saved per-stage scores to %s" % json_path)
    return best, scores


if __name__ == "__main__":
    # create_random_arrays(10)
    # for stage in ["E17.5", "P0"]:
    #     initialize_differentiated_arrays(0.01,10,1,0, n_workers=5,
    #                                  continue_existing_run=True, end_time=50,dt=0.01, stage=stage)
    # ================= FINAL mechanical fit =================================
    # BENDING replaces line tension. Line tension penalises perimeter LENGTH, so it
    # fought the elongation needed for the experimental roundness (measured: at
    # matched roundness it still left ~10-14 deg bond bend). BoundaryBending
    # penalises CURVATURE only, leaving the perimeter free, which decouples the two.
    # Measured at gammaSC=0.05, p0=1.26 (line tension off):
    #   bending 0    -> HC 0.737 / SC 0.622, bond bend 14.1 deg
    #   bending 0.02 -> HC 0.786 / SC 0.651, bond bend  1.3 deg
    #   bending 0.10 -> HC 0.767 / SC 0.661, bond bend  0.3 deg
    # i.e. 10-100x smoother at essentially unchanged (and near-experimental)
    # roundness. So p0 stays near its original ~1.1-1.5 range, NOT the 1.35-2.0
    # that a strong line tension required. Higher bending also costs run time
    # (stiff term -> smaller dt), so we use the SMALLEST value that is already
    # smooth: kappa = 0.02 (1.3 deg bond bend, and the closest roundness match
    # of the sweep: SC 0.651 vs 0.649 experimental).
    # Flip STAGE to run the other stage.
    STAGE = "E17.5"                       # "E17.5" or "P0"
    _CFG = {
        # x0 order = the fitted params: gammaSC, alphaHC_ratio, hc_p0, sc_p0
        "E17.5": dict(suffix="E17", x0=[0.05, 1.03, 1.20, 1.26]),
        "P0":    dict(suffix="P0",  x0=[0.10, 1.04, 1.28, 1.32]),
    }[STAGE]
    initial_sheets = ["random_periodic_array%d_for_%s" % (i, _CFG["suffix"])
                      for i in range(10)]
    # ---- TYPE-DEPENDENT SHAPE INDEX parameterisation -----------------------
    # Still exactly 4 fitted parameters (one per comparison term), but the
    # type-dependence moved from gamma to the shape index:
    #     gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index
    # gammaHC_ratio is FIXED at gammaHC_ratio_fixed (1.0 -> HC and SC share one
    # contractility); it was inert in the previous fits (|rank r| <= 0.10 on all
    # four terms, and the GP lengthscale saturated). Splitting the target
    # perimeter is what lets HC and SC roundness be matched independently.
    #
    # TO REVERT to the historical box, swap the four bound lines below for:
    #     gammaHC_ratio_bounds=(1.0, 1.4),
    #     shape_index_bounds=(1.1, 1.4),
    #     (drop hc_shape_index_bounds / sc_shape_index_bounds)
    # and use x0=[gammaSC, gammaHC_ratio, alphaHC_ratio, shape_index].
    # No other code change is needed - the parameterisation is chosen here.
    find_mechanical_parameters(STAGE, initial_sheets=initial_sheets,
                               indices=None, n_sheets=10,
                               gammaSC_bounds=(0.02, 0.15),
                               gammaHC_ratio_bounds=None,      # -> fixed at 1.0
                               alphaHC_ratio_bounds=(1.00, 1.15),
                               hc_shape_index_bounds=(1.05, 1.55),
                               sc_shape_index_bounds=(1.05, 1.55),
                               ablated_cells=(337, 304, 65, 114), post_ablation_frame=-1,
                               n_calls=60, n_initial_points=25, random_state=0,
                               pval_floor=1e-300,
                               x0=_CFG["x0"], use_saved_threshold=True,
                               type_by="delta_level",
                               min_progress_rate=1e-4, max_wall_seconds=10000, rerun_stalled_runs=False,
                               base_quasi_static_threshold=0.03,
                               ablation_quasi_static_threshold=0.02,
                               line_tension=None,   # replaced by bending
                               bending=0.02,
                               )
