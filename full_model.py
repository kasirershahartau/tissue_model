"""Running the full model — differentiation plus mechanics — over a stage's arrays.

The reusable ``run()`` in run_model drives ONE simulation. These launch the
project's configurations: a stage's ten fitted initial morphologies, in parallel,
at a given set of mechanical and lateral-inhibition parameters, reusing any run
that already completed with the same parameters.

``find_psigma`` sweeps the mechanosensitivity threshold and scores each value
against the experimental differentiation statistics.
"""

from concurrent.futures import ProcessPoolExecutor

from run_model import (run, _run_full_model_one, _initialize_one_differentiated_array,
                       _psigma_tag, RESULTS_DIR)
from post_processing import (initial_morphology_name, random_array_name,
                             compare_full_model_differentiation_to_experiments)


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
