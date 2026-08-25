"""The v1 mechanical fit: Bayesian optimization over the parameter space.

RETIRED. Superseded by the v2 fit — a grid search plus a self-consistent solve
for the preferred area (``grid_fit_mechanics_v2.py``, ``selfconsistent_scan.py``)
— which is what the paper describes and what produced the published parameters.
Kept here because it ran for a long time and its behaviour is worth being able to
consult, not because it should be used again.

Why it was dropped: the objective it optimises is a p-value, and that saturated
at ~0 across almost the whole parameter range, giving a flat landscape with no
gradient that pinned the optimizer at the bounds. The v2 fit scores an n-sigma
distance instead, which varies smoothly and is zero exactly when the model mean
lands on the experimental mean.

Nothing on the core or manuscript branches imports this module.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bayesian_optimization as bo
import pandas as pd
from matplotlib import pyplot as plt

from run_model import (run, MECHANICS_TERMS, _MECHANICS_ROUNDNESS_TERMS,
                       _MECHANICS_ABLATION_TERMS, _DEFAULT_ATOH_SENSITIVITY,
                       _DEFAULT_QUASI_STATIC_THRESHOLD, _FIT_SAVE_INTERVAL,
                       _li_levels_kwargs_for_initial_sheet, _short_run_folder_name,
                       _strip_results_prefix, _load_saved_threshold, RESULTS_DIR)
from mechanics_eval import _evaluate_mechanics_for_sheet
from post_processing import (initial_morphology_name, extract_model_mechanics,
                             compare_pooled_model_mechanics_to_experiments,
                             _WORST_CASE_NSIGMA)



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


# --- reading what the fit wrote -------------------------------------------


def load_mechanical_optimization(stage, results_dir=RESULTS_DIR,
                                 n_params=3):
    """Load the optimizer trace ``find_mechanical_parameters`` saved for
    ``stage`` — the ``<stage>_optimization_params.npy`` (evaluated points,
    shape ``(n_calls, n_params)``) and ``<stage>_optimization_objective.npy``
    (their objective values, shape ``(n_calls,)``).

    The objective is ``sum_terms z**2`` (``z`` = standardized model-vs-experiment
    mean discrepancy per term: HC/SC roundness and HC/SC area-change-after-
    ablation) — LOWER is a better fit, ~0 meaning the model means land within the
    experimental replicate uncertainty.

    Raises a clear error if the files are absent or EMPTY (a completed fit
    writes ``n_calls`` rows; a 0-row file means the trace never reached disk)."""
    px = os.path.join(results_dir, "%s_optimization_params.npy" % stage)
    py = os.path.join(results_dir, "%s_optimization_objective.npy" % stage)
    for p in (px, py):
        if not os.path.isfile(p):
            raise FileNotFoundError("No optimization result at %s" % p)
    X = np.load(px)
    y = np.asarray(np.load(py), float).ravel()
    if X.size == 0 or y.size == 0:
        raise ValueError(
            "Optimization result files for %r are EMPTY (params shape %s, "
            "objective shape %s). A completed find_mechanical_parameters writes "
            "an (n_calls x %d) params array and an (n_calls,) objective array — "
            "0 rows means the trace was never saved (the run didn't finish, or "
            "result['X']/result['y'] were empty). Re-run the fit, or re-save "
            "result['X']/result['y'], before plotting."
            % (stage, X.shape, y.shape, n_params))
    X = np.atleast_2d(X).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("params/objective length mismatch: %d vs %d"
                         % (X.shape[0], y.shape[0]))
    return X, y


def load_mechanical_optimization_trace(stage, results_dir=RESULTS_DIR):
    """Read the per-evaluation JSONL trace ``find_mechanical_parameters`` writes
    incrementally (``<stage>_optimization_trace.jsonl``) into a DataFrame.

    One row per evaluated point, with the parameters, the ``objective``
    (``sum_terms z**2`` over the active terms), the signed per-term discrepancy
    (``nsigma_hc_roundness`` / ``nsigma_sc_roundness`` / ``nsigma_hc_ablation`` /
    ``nsigma_sc_ablation`` — standardized model-vs-experiment mean gap; may be
    NaN when a term has no data), its ``z**2`` contribution (``obj_<term>``, which
    sum to ``objective``), the pooled model sample size per term (``n_<term>``),
    how many sheets contributed (``n_contributing`` of ``n_sheets``) and the
    per-sheet ok/dropped status (``sheets``). This is the crash-resistant record —
    it exists even if the run was killed before the final ``.npy``/landscape
    save."""
    import json
    path = os.path.join(results_dir, "%s_optimization_trace.jsonl" % stage)
    if not os.path.isfile(path):
        raise FileNotFoundError("No optimization trace at %s" % path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Optimization trace %s is empty (0 evaluations)." % path)
    return pd.DataFrame(rows)


def load_mechanical_optimization_landscape(stage, results_dir=RESULTS_DIR):
    """Load the GP-surrogate landscape bundle
    (``<stage>_optimization_landscape.npz``) saved at the end of
    ``find_mechanical_parameters``.

    Returns a dict with: ``param_names`` (d,), ``bounds`` (d, 2), ``axes``
    (d, res) the per-parameter grid coordinates, ``mean`` / ``std`` the GP
    posterior objective (shape ``res**d``, indexed ``[i_gammaSC, i_gammaHC, ...]``,
    ``indexing='ij'``), and ``X`` / ``y`` the evaluated training points. The
    objective is ``sum of z**2`` (per-term standardized mean discrepancy)."""
    path = os.path.join(results_dir, "%s_optimization_landscape.npz" % stage)
    if not os.path.isfile(path):
        raise FileNotFoundError("No optimization landscape at %s" % path)
    return dict(np.load(path, allow_pickle=True))


def plot_mechanical_optimization(
        stage, results_dir=RESULTS_DIR,
        param_names=("gammaSC", "gammaHC_ratio", "alphaHC_ratio"),
        bounds=None, pval_floor=1e-300, save_path=None, show=True):
    """Visualize the mechanical-parameter Bayesian-optimization trace saved by
    :func:`run_model.find_mechanical_parameters` for ``stage`` (e.g. ``"E17.5"``).

    Produces a 2x2 figure:

    1. **Convergence** — every evaluation's objective (``sum of z**2`` over the
       active terms; lower is better) plus the running best, with the overall
       best starred and any degenerate evaluations (a term worst-cased at ~1e6)
       flagged off-scale, so you can see how much of the search actually improved.
    2-4. **Per-parameter marginals** — objective vs each parameter, coloured by
       evaluation order (to see where the optimizer concentrated), best point
       starred, and the bound edges drawn (``bounds`` optional; falls back to
       the explored min/max) so a fit pinned AT a bound is obvious.

    Also prints a summary (best params, best objective, per-parameter value,
    boundary-pinning, and the fraction of evaluations that degenerated). Returns
    ``(fig, (X, y))``.

    ``bounds`` : optional list of ``(lo, hi)`` per parameter — pass the same
    ``gammaSC_bounds`` / ``gammaHC_ratio_bounds`` / ``alphaHC_ratio_bounds``
    used for the fit to mark the true search box.
    """
    X, y = load_mechanical_optimization(stage, results_dir, n_params=len(param_names))
    n, d = X.shape
    evals = np.arange(1, n + 1)
    running_best = np.minimum.accumulate(y)
    best_idx = int(np.argmin(y))
    # A point with any worst-cased (degenerate) term has objective >= 1e6
    # (_WORST_CASE_NSIGMA**2); a real fit sits at O(1-100). So this cleanly flags
    # the degenerate evaluations that starve the optimizer.
    degenerate = 1e5
    at_ceiling = int(np.sum(y >= degenerate))

    fig = plt.figure(figsize=(14, 9))

    # (1) Convergence.
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(evals, y, s=18, c="0.65", label="evaluations")
    ax.plot(evals, running_best, "b-", lw=2, label="running best")
    ax.scatter([best_idx + 1], [y[best_idx]], marker="*", s=220,
               c="crimson", zorder=5, label="best")
    if at_ceiling:
        ax.text(0.02, 0.96, "%d degenerate eval(s) off-scale (a term worst-cased)"
                % at_ceiling, transform=ax.transAxes, va="top", fontsize=8,
                color="0.3")
        # keep the informative range visible despite the 1e6 outliers.
        good = y[y < degenerate]
        if good.size:
            ax.set_ylim(0, float(good.max()) * 1.1)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel(r"objective = $\sum z^2$  (lower is better)")
    ax.set_title(r"Convergence   best $\sum z^2$ = %.3g   (%d/%d degenerate)"
                 % (y[best_idx], at_ceiling, n))
    ax.legend(loc="upper right", fontsize=8)

    # (2-4) Per-parameter marginals.
    for j in range(d):
        ax = fig.add_subplot(2, 2, 2 + j)
        sc = ax.scatter(X[:, j], y, c=evals, cmap="viridis", s=28,
                        edgecolors="none")
        ax.scatter([X[best_idx, j]], [y[best_idx]], marker="*", s=220,
                   c="crimson", zorder=5)
        lo, hi = (bounds[j] if bounds is not None
                  else (float(X[:, j].min()), float(X[:, j].max())))
        for edge in (lo, hi):
            ax.axvline(edge, color="0.6", ls=":", lw=1)
        # Flag a best value sitting on a bound (fit wants to escape the box).
        span = (hi - lo) or 1.0
        pinned = ""
        if bounds is not None and min(abs(X[best_idx, j] - lo),
                                      abs(X[best_idx, j] - hi)) <= 0.02 * span:
            pinned = "  [PINNED at bound]"
        ax.set_xlabel(param_names[j])
        ax.set_ylabel(r"objective ($\sum z^2$)")
        ax.set_title("%s   best = %.4g%s" % (param_names[j], X[best_idx, j], pinned))
    cbar = fig.colorbar(sc, ax=fig.axes[1:], fraction=0.025, pad=0.02)
    cbar.set_label("evaluation order")

    fig.suptitle("Mechanical-parameter fit — %s" % stage, fontsize=13)

    # Console summary — the actionable bits.
    print("=== Mechanical optimization summary (%s) ===" % stage)
    print("evaluations: %d   best objective (sum z^2): %.4g   "
          "(~%.2f sigma per active term)"
          % (n, y[best_idx], (y[best_idx] / 4.0) ** 0.5))
    for j in range(d):
        lo, hi = (bounds[j] if bounds is not None
                  else (float(X[:, j].min()), float(X[:, j].max())))
        tag = ""
        if bounds is not None and min(abs(X[best_idx, j] - lo),
                                      abs(X[best_idx, j] - hi)) <= 0.02 * ((hi - lo) or 1.0):
            tag = "  <-- at bound (consider widening)"
        print("  %-14s best=%.4g   explored [%.4g, %.4g]%s"
              % (param_names[j], X[best_idx, j], X[:, j].min(), X[:, j].max(), tag))
    print("  degenerate (a term worst-cased) evaluations: %d / %d (%.0f%%)"
          % (at_ceiling, n, 100.0 * at_ceiling / n))
    if at_ceiling > 0.3 * n:
        print("  NOTE: a large fraction of the search degenerated (a term had no "
              "usable data — dropped sheets or empty ablation), which starves the "
              "optimizer. Check the debug logs (stall guard) and narrow the bounds "
              "to the solver-tractable region.")

    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (X, y)
