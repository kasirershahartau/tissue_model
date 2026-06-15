import numpy as np
import atexit
import logging
import os, shutil, sys
from tyssue import HistoryHdf5
from matplotlib import pyplot as plt
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility
from post_processing import (compare_model_mechanics_to_experiments,
                             compare_differentiation_to_experiments, create_gif_safe,
                             random_array_name, initial_morphology_name, _STAGE_SHEET_SUFFIX)

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

    for logger_name in _DEBUG_LOG_TARGETS:
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.DEBUG)
        if handler not in lg.handlers:
            lg.addHandler(handler)

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


def run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial_sheet_name=None, ablated_cells=None,
        only_differentiation=False, no_differentiation=False, end_on_steady_state=True, t_end=10, dt=0.01,
        random_forces=False, name=None, continue_existing_run=False, divisions=False,
        continue_from_time=None, randomize_notch_delta_levels=False, stress_dependent=False):
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
    randomize_notch_delta_levels : bool, default False
        When True, every LI level (``notch_level``, ``delta_level``,
        ``repressor_level``) is freshly randomised at the start of
        the run — overriding the default "preserve loaded values"
        behaviour. Useful for parameter sweeps that re-use a saved
        geometry (resume OR fork mode) but want a different LI
        initial condition per sweep point. On a fresh sheet (no
        ``initial_sheet_name``) this flag is a no-op: the LI
        columns are randomised either way.
    """
    # Sheet Parameters


    if name is None:
        name = "periodic_from%s_gammaSC-%.2f_gammaHC_ratio-%.2f_alphaHC_ratio-%.2f_psigma-%.2f"%(initial_sheet_name, gammaSC, gammaHC_ratio, alphaHC_ratio, psigma)
    if ablated_cells is not None:
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
    notch_inhibition = False
    # ``stress_dependent`` now comes from the call argument (default False
    # keeps the historical behaviour). It must be True for ``psigma`` /
    # ``mechanosensitivity`` to have any effect — see the guard below where
    # ``mechanosensitivity`` is forced to 0 when stress dependence is off.
    intercalations = True
    delaminations = True
    if ablated_cells is None:
        ablated_cells = []
    quasi_static = True
    quasi_static_threshold = 0.01

    # Model Parameters
    # General parameters
    movie_frames = 100

    # 2D vertex related parameters
    effectors = [FaceContractility, FaceAreaElasticity]
    tension = {('HC', 'HC'): 0.05,
               ('HC', 'SC'): 0.05,
               ('SC', 'SC'): 0.05
               }
    preferred_area = {'HC': 1 / (4 * np.pi),
                      'SC': 1 / (4 * np.pi)}
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
    repressor_sensitivity = 0.45 # PR - how much Delta production is sensitive to repressor level (sensitivity^l / (sensitivity^l + repressor^l)
    atoh_sensitivity = 0.377  # how much Atoh1 production is sensitive to delta level (delta^l / (sensitivity^l + delta^l)
    atoh_by_repressor = False  # if True, Atoh1 production will be set by repressor level instead of delta (sensitivity^l / (sensitivity^l + repressor^l)
    notch_sensitivity = 0.2  # PS - how much Repressor production is sensitive to signaling level (signaling^m / (sensitivity^m + signaling^m))
    delta_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    notch_delta_production_ratio = 1  # beta
    sensitivity_aging_rate = 10  # Notch sensitivity change rate (for aging sensitivity version)
    mechanosensitivity = psigma  # Sensitivity to mechanical stress (for stress dependent version)
    stress_effectors = [FaceContractility]  # effectors to calculate stress (for stress dependent version)
    li_steady_state_threshold=0.001

    if not stress_dependent:
        mechanosensitivity = 0

    results_dir = os.path.join("results", name)
    if continue_existing_run:
        if not os.path.exists(results_dir):
            print("Directory %s doesn't exist. Unable to continue from existing run." % results_dir)
            return name
    else:
        if os.path.exists(results_dir):
            print("Directory %s already exists" % results_dir)
            return name
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

    initial_sheet_name = os.path.join("results", initial_sheet_name)
    name = os.path.join("results", name)
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
        inner = InnerEarModel(sheet, tension=tension, repulsion=repulsion, repulsion_distance=repulsion_distance,
                              repulsion_exp=repulsion_exponent, preferred_area=preferred_area, contractility=contractility,
                              elasticity=elasticity, differentiation_threshold=differentiation_threshold,
                              random_sensitivity=random_sensitivity,
                              l=l, m=m, betaN=betaN, betaD=betaD, inhibition=notch_inhibition,
                              notch_repressor_degradation_ratio=notch_repressor_degradation_ratio,
                              repressor_sensitivity=repressor_sensitivity, atoh_sensitivity=atoh_sensitivity,
                              delta_repressor_degradation_ratio=delta_repressor_degradation_ratio,
                              notch_delta_production_ratio=notch_delta_production_ratio,
                              stress_effectors=stress_effectors, mechanosensitivity=mechanosensitivity,
                              notch_sensitivity=notch_sensitivity, atoh_by_repressor=atoh_by_repressor,
                              randomize_notch_delta_levels=randomize_notch_delta_levels)
        draw_func = inner.get_draw_sheet_method(number_faces=True, number_edges=False, number_vertices=False,
                                                color_by="atoh")
        fig1, ax1 = draw_func(inner.sheet)
        plt.savefig(os.path.join(name, "initial.png"))
        # Pass the archive path so HistoryHdf5 writes each snapshot on the
        # fly; the partial file can be opened in another process while the
        # simulation is still running (useful for diagnosing hangs).
        history_file = os.path.join(name, "history.hf5")
        history = inner.simulate(t_end=t_end, dt=dt, until_steady_state=end_on_steady_state,
                                 lateral_inhibition_threshold=li_steady_state_threshold,
                                 only_differentiation=only_differentiation,
                                 random_forces=random_forces, aging_sensitivity=aging_sensitivity,
                                 no_differentiation=no_differentiation,
                                 contact_dependent_differentiation=contact_dependent_differentiation, divisions=divisions,
                                 intercalations=intercalations, delaminations=delaminations, ablated_cells=ablated_cells,
                                 sensitivity_aging_rate=sensitivity_aging_rate,
                                 division_area=division_area, intercalation_length=intercalation_length,
                                 delamination_area=delamination_area, delamination_rate=delamination_rate,
                                 viscosity=viscosity, effectors=effectors, quasi_static=quasi_static,
                                 quasi_static_threshold=quasi_static_threshold, atoh_by_repressor=atoh_by_repressor,
                                 history_file=history_file,
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
        run_log.info("run() finished successfully")
        return name
    except BaseException:
        # ``BaseException`` (not ``Exception``) covers Ctrl+C
        # (KeyboardInterrupt) too — the user may stop a stuck run
        # explicitly and still want the partial log to land on disk.
        # ``run_log.exception`` automatically formats the active
        # traceback and writes it through the same handler, so the
        # entire chain is in the file before the raise propagates.
        run_log.exception("run() crashed; traceback follows")
        raise
    finally:
        # Always flush + detach so that
        #  (a) every record written above is committed to disk, and
        #  (b) a re-run inside the same Python session doesn't keep
        #      writing to the previous run's file.
        _disable_debug_log(log_handler)
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

def _strip_results_prefix(name):
    """``run`` returns ``results\\<name>`` on a fresh run but ``<name>`` on a
    cache hit (the directory-already-exists early return). ``load_history_file``
    expects the bare name, so normalize both forms to a prefix-less name."""
    parts = os.path.normpath(name).split(os.sep)
    if parts and parts[0] == "results":
        parts = parts[1:]
    return os.path.join(*parts) if parts else ""


def _evaluate_mechanics_for_sheet(args):
    """Run one initial sheet to steady state (and, if cells are ablated, the
    ablation run too), then return the per-term mechanics p values dict.

    Module-level and single-argument so it is picklable for
    ``ProcessPoolExecutor`` — this is the unit parallelized across initial
    sheets. Area/roundness come from the un-ablated steady state; only the
    ablation term needs the extra ablation simulation, which is skipped
    entirely when ``ablated_cells`` is empty.
    """
    (gammaSC, gammaHC_ratio, alphaHC_ratio, initial, experimental_stage,
     ablated_cells, post_ablation_frame) = args

    base_name = _strip_results_prefix(
        run(gammaSC, gammaHC_ratio, alphaHC_ratio, 0, initial, no_differentiation=True))

    ablation_kwargs = {}
    if ablated_cells:
        with_ablation_name = _strip_results_prefix(
            run(gammaSC, gammaHC_ratio, alphaHC_ratio, 0, base_name,
                no_differentiation=True, ablated_cells=ablated_cells))
        ablation_kwargs = dict(ablation_model_name=with_ablation_name,
                               ablated_cells=ablated_cells,
                               post_ablation_frame=post_ablation_frame)

    _, details = compare_model_mechanics_to_experiments(
        base_name, experimental_stage, **ablation_kwargs)
    return details


def find_mechanical_parameters(experimental_stage, initial_sheets=None,
                               indices=None, n_sheets=10,
                               gammaSC_bounds=(0.001, 0.1),
                               gammaHC_ratio_bounds=(1.0, 20.0),
                               alphaHC_ratio_bounds=(0.1, 5.0),
                               ablated_cells=(), post_ablation_frame=-1,
                               n_calls=40, n_initial_points=10,
                               n_workers=None, random_state=0,
                               pval_floor=1e-300, x0=None):
    """Find the mechanical parameters (gammaSC, gammaHC_ratio, alphaHC_ratio)
    that best fit the experimental measurements, via Gaussian-process
    Bayesian optimization.

    The fit maximizes the agreement reported by
    :func:`compare_model_mechanics_to_experiments` (product of the area,
    roundness and ablation p values). To keep the optimization numerically
    well-behaved we maximize the **sum of log p values** rather than the raw
    product (which underflows toward zero), averaged across ``initial_sheets``
    — i.e. we MINIMIZE ``mean over sheets of (-sum_i log p_i)``.

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
        Optional initial guess ``[gammaSC, gammaHC_ratio, alphaHC_ratio]``
        evaluated before the design.

    Returns
    -------
    best_params : numpy.ndarray
        ``[gammaSC, gammaHC_ratio, alphaHC_ratio]`` of the best fit.
    result : dict
        Full optimizer trace (``x``, ``fun``, ``X``, ``y``) from
        :func:`bayesian_optimization.minimize`.
    """
    import bayesian_optimization as bo
    from concurrent.futures import ProcessPoolExecutor

    if initial_sheets is None:
        # Build the stage's fitted-initial-morphology sheets by index
        # (``indices``, e.g. a subset, or ``range(n_sheets)`` by default).
        if indices is None:
            indices = range(n_sheets)
        initial_sheets = [initial_morphology_name(i, experimental_stage) for i in indices]
    initial_sheets = list(initial_sheets)
    ablated_cells = list(ablated_cells)
    bounds = [tuple(gammaSC_bounds), tuple(gammaHC_ratio_bounds), tuple(alphaHC_ratio_bounds)]
    if n_workers is None:
        n_workers = min(len(initial_sheets), os.cpu_count() or 1)
    cache = {}

    def objective(params):
        gammaSC, gammaHC_ratio, alphaHC_ratio = (float(p) for p in params)
        # ``run`` quantizes parameters to 2 decimals in the folder name, so
        # cache at that resolution to avoid recomputing equivalent points.
        key = (round(gammaSC, 2), round(gammaHC_ratio, 2), round(alphaHC_ratio, 2))
        if key in cache:
            return cache[key]

        tasks = [(gammaSC, gammaHC_ratio, alphaHC_ratio, initial, experimental_stage,
                  ablated_cells, post_ablation_frame) for initial in initial_sheets]
        if n_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                details_list = list(executor.map(_evaluate_mechanics_for_sheet, tasks))
        else:
            details_list = [_evaluate_mechanics_for_sheet(task) for task in tasks]

        # Sum of log p over the three terms, averaged across sheets; negated
        # so the optimizer minimizes.
        per_sheet_sum_log = []
        for details in details_list:
            per_sheet_sum_log.append(sum(
                np.log(max(details[term], pval_floor)) for term in ("area", "roundness", "ablation")))
        value = -float(np.mean(per_sheet_sum_log))
        cache[key] = value
        print("params gammaSC=%.4f gammaHC_ratio=%.4f alphaHC_ratio=%.4f "
              "-> mean -log p = %.6g (product p ~ %.3g)"
              % (gammaSC, gammaHC_ratio, alphaHC_ratio, value, np.exp(-value)))
        return value

    result = bo.minimize(objective, bounds, n_calls=n_calls,
                         n_initial_points=n_initial_points,
                         random_state=random_state, x0=x0)
    best_params = result["x"]
    print("Best params: gammaSC=%.4f, gammaHC_ratio=%.4f, alphaHC_ratio=%.4f" % tuple(best_params))
    print("Best mean -log p = %.6g (product p ~ %.3g)" % (result["fun"], np.exp(-result["fun"])))
    return best_params, result


def _evaluate_psigma_for_model(args):
    """Run one differentiation simulation (a single stage / initial-sheet /
    psigma combination) and return its differentiating-cell HC-neighbor fit
    p value. Module-level + single-argument so it is picklable for
    ``ProcessPoolExecutor`` — the unit parallelized in :func:`find_psigma`.

    ``stress_dependent=True`` is essential here: it is what makes ``psigma``
    (the mechanosensitivity) actually influence differentiation.
    """
    (psigma, stage, initial, gammaSC, gammaHC_ratio, alphaHC_ratio,
     type_by, threshold, max_number_of_neighbors) = args
    model_name = _strip_results_prefix(
        run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, initial_sheet_name=initial,
            stress_dependent=True, end_on_steady_state=True, t_end=25, dt=0.01))
    pval, _, _ = compare_differentiation_to_experiments(
        model_name, stage, type_by=type_by, threshold=threshold,
        max_number_of_neighbors=max_number_of_neighbors)
    return (round(float(psigma), 2), stage, float(pval))


def find_psigma(mechanical_params, initial_sheets=None, indices=None,
                psigma_bounds=(0.0, 2.0), n_grid=11, n_refine=2,
                n_initial_sheets=10, n_workers=None,
                type_by='atoh_level', threshold=None, max_number_of_neighbors=2,
                pval_floor=1e-300, plot=True):
    """Fit the single mechanosensitivity parameter ``psigma`` SHARED by the
    E17.5 and P0 models, by matching the differentiating-cell HC-neighbor
    distribution (via :func:`compare_differentiation_to_experiments`).

    The two developmental stages are modelled identically except for (a) their
    fitted initial morphology and (b) their fitted mechanical parameters; only
    ``psigma`` is common and unknown. For each candidate ``psigma`` we run a
    differentiation simulation per stage and per initial sheet, score each with
    its experimental differentiation p value, and maximize the COMBINED fit

        objective(psigma) = mean_sheets log p_{E17.5} + mean_sheets log p_{P0}

    (sum of per-stage mean log p values — i.e. the product of the two stages'
    geometric-mean p values). Maximizing the sum of logs both keeps the score
    numerically well-scaled and forces a single ``psigma`` to fit BOTH stages,
    rather than letting one stage dominate.

    Optimization method — a **coarse-to-fine parallel line search**. For a
    single bounded, noisy, expensive parameter this is the most suitable
    choice: it makes no smoothness/unimodality assumption (unlike
    golden-section/Brent, which noise breaks), it returns the full
    fit-vs-``psigma`` landscape for inspection, and every (psigma x stage x
    sheet) simulation is independent so the entire grid runs at once across
    ``n_workers`` processes — maximal parallelism. ``run`` quantizes ``psigma``
    to 2 decimals, so refinement stops once the grid spacing reaches 0.01 (the
    finest resolution the simulator distinguishes).

    Parameters
    ----------
    mechanical_params : dict
        ``{stage: (gammaSC, gammaHC_ratio, alphaHC_ratio)}`` — the per-stage
        mechanical parameters from :func:`find_mechanical_parameters`. Its keys
        define which stages are fitted (typically ``"E17.5"`` and ``"P0"``).
    initial_sheets : dict, optional
        ``{stage: [initial-sheet result names]}``. When omitted, built per stage
        from ``indices`` (or ``range(n_initial_sheets)``) via
        ``initial_morphology_name``.
    indices : sequence of int, optional
        Which array indices to use when ``initial_sheets`` is not given (shared
        across stages).
    psigma_bounds : (low, high)
        Search interval for ``psigma``.
    n_grid : int
        Points per line-search pass.
    n_refine : int
        Coarse-to-fine refinement passes around the running best.
    n_workers : int, optional
        Worker processes (default ``cpu_count``).
    pval_floor : float
        p values are clipped up to this floor before ``log`` to avoid ``-inf``.
    plot : bool
        Save an ``objective``/per-stage vs ``psigma`` plot to ``results``.

    Returns
    -------
    best_psigma : float
        The ``psigma`` maximizing the combined fit.
    scores : dict
        ``{psigma: {stage: mean_log_p, ..., "objective": sum}}`` for every
        evaluated ``psigma`` (the full landscape).
    """
    from concurrent.futures import ProcessPoolExecutor
    from collections import defaultdict

    stages = list(mechanical_params.keys())
    if initial_sheets is None:
        idx = list(indices) if indices is not None else range(n_initial_sheets)
        initial_sheets = {stage: [initial_morphology_name(i, stage) for i in idx]
                          for stage in stages}
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    scores = {}  # rounded psigma -> {stage: mean_log_p, "objective": sum}

    def evaluate(psigmas):
        # De-duplicate at run()'s 2-decimal resolution and skip cached points.
        wanted = sorted({round(float(p), 2) for p in psigmas} - set(scores))
        tasks = []
        for psigma in wanted:
            for stage in stages:
                gammaSC, gammaHC_ratio, alphaHC_ratio = mechanical_params[stage]
                for initial in initial_sheets[stage]:
                    tasks.append((psigma, stage, initial, gammaSC, gammaHC_ratio, alphaHC_ratio,
                                  type_by, threshold, max_number_of_neighbors))
        if not tasks:
            return
        if n_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_evaluate_psigma_for_model, tasks))
        else:
            results = [_evaluate_psigma_for_model(task) for task in tasks]

        collected = defaultdict(lambda: defaultdict(list))
        for psigma, stage, pval in results:
            collected[psigma][stage].append(pval)
        for psigma, stage_pvals in collected.items():
            entry, objective = {}, 0.0
            for stage in stages:
                mean_log = float(np.mean([np.log(max(p, pval_floor)) for p in stage_pvals[stage]]))
                entry[stage] = mean_log
                objective += mean_log
            entry["objective"] = objective
            scores[psigma] = entry
            print("psigma=%.2f -> objective(sum log p)=%.4g  [%s]"
                  % (psigma, objective,
                     ", ".join("%s p~%.3g" % (s, np.exp(entry[s])) for s in stages)))

    low, high = psigma_bounds
    current_step = (high - low) / (n_grid - 1)
    evaluate(np.linspace(low, high, n_grid))
    best = max(scores, key=lambda k: scores[k]["objective"])
    for _ in range(n_refine):
        new_low = max(psigma_bounds[0], best - current_step)
        new_high = min(psigma_bounds[1], best + current_step)
        next_step = (new_high - new_low) / (n_grid - 1)
        if next_step < 0.01:  # run() can't resolve a finer psigma
            break
        evaluate(np.linspace(new_low, new_high, n_grid))
        best = max(scores, key=lambda k: scores[k]["objective"])
        current_step = next_step

    print("Best psigma = %.2f (combined sum log p = %.4g, combined p ~ %.3g)"
          % (best, scores[best]["objective"], np.exp(scores[best]["objective"])))

    if plot:
        psigmas = sorted(scores)
        fig, ax = plt.subplots()
        ax.plot(psigmas, [scores[p]["objective"] for p in psigmas], "k-o", label="combined (sum log p)")
        for stage in stages:
            ax.plot(psigmas, [scores[p][stage] for p in psigmas], "--o", label="%s (mean log p)" % stage)
        ax.axvline(best, color="grey", ls=":")
        ax.set_xlabel("psigma")
        ax.set_ylabel("log p (higher = better fit)")
        ax.set_title("psigma fit: differentiating-cell HC neighbors")
        ax.legend()
        out_path = os.path.join("results", "psigma_fit.png")
        plt.savefig(out_path)
        plt.close(fig)
        print("Saved fit landscape to %s" % out_path)

    return best, scores


if __name__ == "__main__":
    # create_random_arrays(1)
    initialize_differentiated_arrays(0.01,10,1,0, n_workers=5,
                                     continue_existing_run=False, end_time=35,dt=0.01, stage="P0")