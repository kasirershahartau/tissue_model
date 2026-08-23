import atexit
import logging
import os, shutil, sys
import numpy as np
from tyssue import HistoryHdf5
from matplotlib import pyplot as plt
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from post_processing import create_gif_safe, RESULTS_DIR
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility


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


def _enable_debug_log(log_path):
    """Attach a DEBUG-level ``_FlushFileHandler`` to all loggers in
    :data:`_DEBUG_LOG_TARGETS`, writing to ``log_path``.

    Returns the handler so the caller can flush / close / detach it in
    a ``finally`` block. Also registers an ``atexit`` callback that
    flushes the file on interpreter exit, so a partial run still
    leaves a usable log even if the caller forgot to clean up.
    """
    handler = _FlushFileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
        "%(filename)s:%(lineno)d: %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Set the level on every target logger so their DEBUG/INFO records are
    # created, but attach the handler to the ROOT logger ONLY. Every record
    # propagates up to root, so a single handler there captures each one
    # exactly once. Attaching the SAME handler to several loggers in one
    # propagation chain (e.g. "tyssue.solvers.viscous" -> "tyssue" -> root, all
    # in _DEBUG_LOG_TARGETS) emitted each record once per handler on the chain
    # — 3 duplicate lines for every solver warning.
    for logger_name in _DEBUG_LOG_TARGETS:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    # Keep matplotlib's chatty DEBUG/INFO out of the file (root is at DEBUG and
    # matplotlib propagates to it). WARNING+ from matplotlib still gets through.
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


def load_sheet_from_file(initial_sheet_name, two_dim=True):
    history = HistoryHdf5.from_archive("%s.hf5" % initial_sheet_name, eptm_class=VirtualSheet)
    last_time_point = np.max(history.time_stamps)
    sheet = history.retrieve(last_time_point)
    sheet.arrange_sheet_from_history(two_dim)
    sheet.initiate_edge_order()
    return sheet


def run():
    # Sheet Parameters
    initial_sheet_name = ""
    # ``load_lateral_inhibition_data_from_file`` USED to live here
    # as the trigger for the pickle side-channel. The LI columns
    # now travel with the HDF5 history file.
    name = "random_periodic_array_test3"
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
    only_differentiation = False
    no_differentiation = True
    contact_dependent_differentiation = True
    notch_inhibition = False
    stress_dependent = False
    divisions = True
    intercalations = True
    delaminations = True
    ablated_cells = []
    random_forces = True
    quasi_static = False
    quasi_static_threshold=0.01

    # Model Parameters
    # General parameters
    t_end = 10
    dt = 0.001
    movie_frames = 100

    # 2D vertex related parameters
    effectors = [FaceContractility, FaceAreaElasticity]
    tension = {('HC', 'HC'): 0.05,
               ('HC', 'SC'): 0.05,
               ('SC', 'SC'): 0.05
               }
    # Preferred cell area = area of a circle whose radius is HALF the lattice
    # unit (distx = disty = 1, so radius 0.5): pi * 0.5**2 = pi/4 ~ 0.785, which
    # matches the actual mean cell area of the saved arrays (~0.76). Was a typo
    # `1/(4*pi)` ~ 0.0796 (~10x too small) — that drove cells to shrink hard,
    # jagged the periodic tiling, and triggered the sharp-corner collapse
    # cascade (see [[sharp-corner-collapse-prevents-folds]]); fixed to match run_model.
    preferred_area = {'HC': np.pi / 4,
                      'SC': np.pi / 4}
    contractility = {'HC': 0.1,
                     'SC': 0.01}

    repulsion = {'HC': 0.001,
                 'SC': 0.}
    repulsion_distance = {'HC': 2.0,
                          'SC': 0.}
    repulsion_exponent = 7.
    elasticity = {'HC': 1.,
                  'SC': 1.}

    # Topological events related parameters
    division_area = 1.2
    intercalation_length = 0.04
    delamination_area = 0.1
    delamination_rate = 1.2
    viscosity = 1

    # Lateral Inhibition parameters
    differentiation_threshold = 0.5
    l = 3  # decreasing Hill exponent
    m = 3  # increasing Hill exponent
    betaN = 1  # maximum production rate Notch for classical model
    betaD = 1  # maximum production rate Delta for classical model
    notch_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    repressor_sensitivity = 0.35  # PR - how much Delta production is sensitive to repressor level (sensitivity^l / (sensitivity^l + repressor^l)
    atoh_sensitivity = 0.377  # how much Atoh1 production is sensitive to delta level (delta^l / (sensitivity^l + delta^l)
    atoh_by_repressor = False  # if True, Atoh1 production will be set by repressor level instead of delta (sensitivity^l / (sensitivity^l + repressor^l)
    notch_sensitivity = 0.2  # PS - how much Repressor production is sensitive to signaling level (signaling^m / (sensitivity^m + signaling^m))
    delta_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    notch_delta_production_ratio = 1 # beta
    sensitivity_aging_rate = 10  # Notch sensitivity change rate (for aging sensitivity version)
    mechanosensitivity = 0 # Sensitivity to mechanical stress (for stress dependent version)
    stress_effectors = [FaceContractility]  # effectors to calculate stress (for stress dependent version)

    if not stress_dependent:
        mechanosensitivity = 0


    results_dir = os.path.join(RESULTS_DIR, name)
    if os.path.exists(results_dir):
        # Pass --force / -f to overwrite without prompting (handy for CI
        # and headless runs); otherwise ask.
        force_flag = any(arg in ("--force", "-f") for arg in sys.argv[1:])
        if not force_flag and sys.stdin.isatty():
            overwrite = input("overwriting existing results, are you sure? (y/n)")
            if overwrite not in ["y", "Y", "yes", "Yes"]:
                exit(0)
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)


    #  Saving model  parameters
    params_file = os.path.join(os.path.join(RESULTS_DIR, name, name + "_parameters.txt"))
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
    debug_log_path = os.path.join(results_dir, name + "_debug.log")
    log_handler = _enable_debug_log(debug_log_path)
    run_log = logging.getLogger("periodic_tests")
    run_log.info("Debug log started at %s", debug_log_path)

    initial_sheet_name = os.path.join(RESULTS_DIR, initial_sheet_name, initial_sheet_name)
    name = os.path.join(RESULTS_DIR, name, name)

    try:
        # Load or initialize sheet
        if os.path.isfile("%s.hf5" % initial_sheet_name):
            sheet = load_sheet_from_file(initial_sheet_name)
        else:
            sheet = initialize_sheet(nx, ny, distx, disty, max_bond_length, min_bond_length)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0

        # The LI levels (notch / delta / repressor) travel with the
        # HDF5 history now — see InnerEarModel.initialize_notch_delta.
        # No more ``<name>_notch_delta_levels.pkl`` side-channel.

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
                              notch_sensitivity=notch_sensitivity, atoh_by_repressor=atoh_by_repressor)
        draw_func = inner.get_draw_sheet_method(number_faces=True, number_edges=False, number_vertices=False, color_by="atoh")
        fig1, ax1 = draw_func(inner.sheet)
        plt.savefig("%s_initial.png" % name)
        # Pass the archive path so HistoryHdf5 writes each snapshot on the
        # fly; the partial file can be opened in another process while the
        # simulation is still running (useful for diagnosing hangs).
        history_file = "%s.hf5" % name
        history = inner.simulate(t_end=t_end, dt=dt, only_differentiation=only_differentiation,
                                 random_forces=random_forces, aging_sensitivity=aging_sensitivity,
                                 no_differentiation=no_differentiation,
                                 contact_dependent_differentiation=contact_dependent_differentiation, divisions=divisions,
                                 intercalations=intercalations, delaminations=delaminations, ablated_cells=ablated_cells,
                                 sensitivity_aging_rate=sensitivity_aging_rate,
                                 division_area=division_area, intercalation_length=intercalation_length,
                                 delamination_area=delamination_area, delamination_rate=delamination_rate,
                                 viscosity=viscosity, effectors=effectors, quasi_static=quasi_static,
                                 quasi_static_threshold=quasi_static_threshold, atoh_by_repressor=atoh_by_repressor,
                                 history_file=history_file)
        # ``inner.save_notch_delta(...)`` USED to be called here. The
        # HDF5 history now carries the LI levels on every snapshot,
        # so the dedicated pickle export is no longer part of the
        # standard run pipeline. (The ``save_notch_delta`` method
        # itself is kept on InnerEarModel as a stand-alone utility
        # for ad-hoc exports.)
        fig2, ax2 = draw_func(inner.sheet)
        plt.savefig("%s_finale.png" % name)
        inner.save_sheet_labels_to_numpy(inner.sheet, path="%s_labels.npy" % name)
        inner.save_contact_matrix_to_numpy(inner.sheet, path="%s_contact_matrix.npy" % name)
        inner.save_face_data_to_df(inner.sheet, path="%s_cells_info.pkl" % name)
        gif_func = inner.get_draw_sheet_method(number_faces=True, number_edges=False, number_vertices=False, color_by="atoh",
                                               arrange_sheet=True)
        create_gif_safe(history, os.path.join(os.getcwd(), "%s.gif" % name), num_frames=movie_frames, draw_func=gif_func)
        run_log.info("run() finished successfully")
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

if __name__ == "__main__":
    run()