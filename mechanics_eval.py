"""Evaluating one parameter point of the mechanical fit on one initial sheet.

Runs the sheet (and, when asked, its ablation partner), measures the mechanical
observables the fit is scored on, and returns them as distributions rather than
summary numbers so the caller can pool across sheets before comparing with
experiment.

A run that fails is reported as such rather than substituted: a parameter point
whose simulation dies is a fact about that point, and scoring it from a partial
archive would hide it.
"""

import logging

from run_model import (run, MECHANICS_TERMS, _DEFAULT_ATOH_SENSITIVITY,
                       _FIT_SAVE_INTERVAL, _li_levels_kwargs_for_initial_sheet,
                       _short_run_folder_name, _strip_results_prefix)
from post_processing import extract_model_mechanics

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
