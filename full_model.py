"""Running the full model — differentiation plus mechanics — over a stage's arrays.

The reusable ``run()`` in run_model drives ONE simulation. These launch the
project's configurations: a stage's ten fitted initial morphologies, in parallel,
at a given set of mechanical and lateral-inhibition parameters, reusing any run
that already completed with the same parameters.

``find_psigma`` sweeps the mechanosensitivity threshold and scores each value
against the experimental differentiation statistics.
"""

from concurrent.futures import ProcessPoolExecutor

# find_psigma builds its grids with np.linspace, but numpy was never imported
# here, so ANY call with n_grid > 1 died on NameError before reaching the first
# simulation. Pre-existing (present at HEAD): the archived psigma sweeps were
# driven by run_psigma_repeats / run_psigma_sweep_v2 instead, which is why it
# went unnoticed.
import numpy as np

# Same story for os, matplotlib and the score-cache helpers: every one of them
# is used by find_psigma and none was imported, so the function could not run
# at all. Fixed together rather than one NameError at a time.
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from run_model import (run, _run_full_model_one, _initialize_one_differentiated_array,
                       _psigma_tag, RESULTS_DIR,
                       _score_cache_key, _history_fingerprint,
                       _load_score_cache, _store_score_cache,
                       _reached_steady_state)
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


#: Folder prefix per REPEAT of the full model. A repeat is the same (array,
#: stage, psigma) simulated again from the same initial sheet: the run is
#: stochastic, so the spread across repeats is the run-to-run scatter a score
#: difference has to beat. The name must differ or the second repeat would be
#: handed back the first one's folder by ``reuse_existing_run``.
#:
#: Defined here rather than in run_psigma_repeats.py (which now imports it) so
#: the fit and the repeat driver cannot drift onto different names -- and these
#: ARE the archived names, so the existing pT = 0 baseline is reused rather
#: than re-simulated.
REPEAT_PREFIX = {1: "fullmodel_v2", 2: "fullmodel_v2r2", 3: "fullmodel_v2r3",
                 4: "fullmodel_v2r4", 5: "fullmodel_v2r5",
                 6: "fullmodel_v2r6", 7: "fullmodel_v2r7",
                 8: "fullmodel_v2r8", 9: "fullmodel_v2r9",
                 10: "fullmodel_v2r10"}


def full_model_tasks(stage, gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index,
                     atoh_sensitivity, notch_sensitivity=0.1,
                     repressor_sensitivity=0.3, gammaHC_ratio=1.0, bending=0.02,
                     line_tension=None, quasi_static_threshold=0.03, t_end=25,
                     dt=0.01, initial_notch_delta_level=0.01, psigma=0.0,
                     preferred_area=None, save_interval=0.1, stress_shift=0.0,
                     stress_hill_exponent=None, indices=None, n_arrays=10,
                     reuse_existing_run=False, continue_existing_run=False,
                     name_prefix="fullmodel", l0_ratio=1.0):
    """The ``_run_full_model_one`` task tuples for one (stage, psigma) point,
    WITHOUT running them.

    Split out of :func:`run_full_model_arrays` so a caller sweeping several
    psigma values can pool every (psigma, stage, array) task together instead
    of one point at a time. One point offers only ``n_arrays`` tasks, which
    caps a sweep's concurrency at 10 however many cores are rented; the whole
    pass offers ``n_grid * n_stages * n_arrays``.
    """
    if indices is None:
        indices = range(n_arrays)
    initial_sheets = [initial_morphology_name(i, stage) for i in indices]
    return [(initial, gammaSC, gammaHC_ratio, alphaHC_ratio, hc_shape_index,
             sc_shape_index, atoh_sensitivity, notch_sensitivity,
             repressor_sensitivity, bending, line_tension,
             quasi_static_threshold, t_end, dt, initial_notch_delta_level,
             psigma, preferred_area, save_interval, stress_shift,
             stress_hill_exponent, reuse_existing_run, continue_existing_run,
             name_prefix,
             # keep l0_ratio LAST: this tuple is positional and unpacked by
             # position in _run_full_model_one
             l0_ratio)
            for initial in initial_sheets]


def _stage_mechanics(entry, default_preferred_area=None, default_shape_index=0.0):
    """Normalise one stage's mechanical parameters.

    Two accepted forms:

    * the legacy 4-sequence ``(gammaSC, alphaHC_ratio, hc_p0, sc_p0)``, which
      leaves ``gammaHC_ratio`` at 1.0 and takes the sweep-wide
      ``preferred_area``;
    * a dict, which is what the v2 fit needs -- ``R_gamma`` is 6.746 at E17.5
      and 3.851 at P0, and ``A0`` differs per stage too, neither of which the
      4-sequence can carry. Keys: ``gammaSC``/``gamma_sc``,
      ``gammaHC_ratio``/``R_gamma``, ``alphaHC_ratio``/``R_alpha``,
      ``preferred_area``/``A0``, ``hc_shape_index``, ``sc_shape_index``.
    """
    if isinstance(entry, dict):
        def pick(*names, **kw):
            for n in names:
                if n in entry and entry[n] is not None:
                    return entry[n]
            return kw.get("default")
        gammaSC = pick("gammaSC", "gamma_sc")
        if gammaSC is None:
            raise ValueError("stage mechanics need gammaSC (or gamma_sc): %r" % (entry,))
        alpha = pick("alphaHC_ratio", "R_alpha", default=1.0)
        return {"gammaSC": float(gammaSC),
                "gammaHC_ratio": float(pick("gammaHC_ratio", "R_gamma", default=1.0)),
                "alphaHC_ratio": float(alpha),
                "hc_shape_index": float(pick("hc_shape_index", default=default_shape_index)),
                "sc_shape_index": float(pick("sc_shape_index", default=default_shape_index)),
                "preferred_area": pick("preferred_area", "A0",
                                       default=default_preferred_area)}
    vals = list(entry)
    if len(vals) != 4:
        raise ValueError("stage mechanics must be a dict or a 4-sequence "
                         "(gammaSC, alphaHC_ratio, hc_p0, sc_p0), got %r" % (entry,))
    gammaSC, alpha, hc_p0, sc_p0 = vals
    return {"gammaSC": float(gammaSC), "gammaHC_ratio": 1.0,
            "alphaHC_ratio": float(alpha), "hc_shape_index": float(hc_p0),
            "sc_shape_index": float(sc_p0),
            "preferred_area": default_preferred_area}


def _score_one_point(args):
    """Score one (stage, model_names) point. Module-level and picklable so the
    scoring pass can run in the same pool as the simulations -- it rescans
    every frame of every run, so with the points themselves parallel it would
    otherwise become the serial bottleneck."""
    (stage, model_names, type_by, threshold, max_number_of_neighbors) = args
    return compare_full_model_differentiation_to_experiments(
        stage, model_names=model_names, type_by=type_by, threshold=threshold,
        max_number_of_neighbors=max_number_of_neighbors)


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
                          name_prefix="fullmodel", l0_ratio=1.0):
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
    tasks = full_model_tasks(
        stage, gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index,
        atoh_sensitivity, notch_sensitivity=notch_sensitivity,
        repressor_sensitivity=repressor_sensitivity, gammaHC_ratio=gammaHC_ratio,
        bending=bending, line_tension=line_tension,
        quasi_static_threshold=quasi_static_threshold, t_end=t_end, dt=dt,
        initial_notch_delta_level=initial_notch_delta_level, psigma=psigma,
        preferred_area=preferred_area, save_interval=save_interval,
        stress_shift=stress_shift, stress_hill_exponent=stress_hill_exponent,
        indices=indices, n_arrays=n_arrays,
        reuse_existing_run=reuse_existing_run,
        continue_existing_run=continue_existing_run, name_prefix=name_prefix,
        l0_ratio=l0_ratio)
    if n_workers is None:
        n_workers = min(len(tasks), os.cpu_count() or 1)
    print("[full model] %s: %d array(s) on %d worker(s)  |  mechanics "
          "gammaSC=%.4g alphaHC=%.4g hc_p0=%.4g sc_p0=%.4g  |  LI pS=%.4g pR=%.4g "
          "atoh=%.6g init~U(0,%.3g) l0_ratio=%.4g  |  bending=%s qst=%.4g t_end=%g"
          % (stage, len(tasks), n_workers, gammaSC, alphaHC_ratio, hc_shape_index,
             sc_shape_index, notch_sensitivity, repressor_sensitivity, atoh_sensitivity,
             initial_notch_delta_level, l0_ratio, bending, quasi_static_threshold,
             t_end), flush=True)
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            names = list(executor.map(_run_full_model_one, tasks))
    else:
        names = [_run_full_model_one(task) for task in tasks]
    return names


def find_psigma(mechanical_params, psigma_bounds=(0.0, 2.0), n_grid=11, n_refine=2,
                n_arrays=10, n_repeats=3, indices=None, n_workers=None,
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
    + quasi-static mechanics) is run on all ``n_arrays`` initial arrays, and the
    runs are POOLED and scored against experiment.

    REPEATS. ``n_repeats`` (default 3) independent simulations of every
    (array, stage, psigma), pooled into that point's single score. The runs are
    stochastic, so one repeat per array cannot separate a psigma dependence
    from run-to-run scatter -- which matters most exactly where this fit is
    decided, among neighbouring points a few thousandths apart. Each repeat
    gets its own folder prefix from :data:`REPEAT_PREFIX`; those are the
    archived names, so the existing pT = 0 baseline is reused, not re-run.

    Every point in a sweep MUST be scored on the same number of repeats: more
    repeats shrink the simulated SEM, so points measured at different repeat
    counts are not comparable and the one with fewer repeats will look better.

    PARALLELISM. A whole PASS of the line search goes into one pool: every
    (psigma, stage, array, repeat) run of that pass is independent, so
    ``n_workers`` (default: all cores) can be filled with up to ``n_grid *
    n_stages * n_arrays * n_repeats`` runs, and the per-point scoring -- which
    rescans every frame of every run -- is pooled the same way. Only the PASSES
    are sequential, since each refinement needs the previous pass's minimum.
    Before this, points and stages ran one at a time and only the ``n_arrays``
    arrays were parallel, so a sweep could never use more than 10 cores no
    matter what was rented. ``psigma=0`` reuses the
    existing ``fullmodel_*`` baseline, so the sweep tells you directly whether
    mechano-sensitivity improves the score over psigma=0.

    A coarse-to-fine line search (``run`` quantizes ``psigma`` to 2 decimals, so
    refinement stops at a 0.01 grid). LOWER score is better (unlike the previous
    p-value version, which maximized).

    Parameters
    ----------
    mechanical_params : dict
        ``{stage: mechanics}``; the keys pick which stages are fitted. Each
        value is either a dict (preferred — it can carry the per-stage
        ``gammaHC_ratio``/``R_gamma`` and ``preferred_area``/``A0`` the v2 fit
        needs) or the legacy 4-sequence ``(gammaSC, alphaHC_ratio,
        hc_shape_index, sc_shape_index)``, which leaves ``gammaHC_ratio`` at
        1.0 and uses the sweep-wide ``preferred_area``. See
        :func:`_stage_mechanics`.
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
    if not 1 <= int(n_repeats) <= len(REPEAT_PREFIX):
        raise ValueError("n_repeats must be 1..%d (one folder prefix each), got %r"
                         % (len(REPEAT_PREFIX), n_repeats))
    n_repeats = int(n_repeats)
    # Resolved ONCE, up front, so a malformed entry fails before any simulation
    # rather than hours in. Carries per-stage gammaHC_ratio and A0, which the
    # legacy 4-tuple cannot express (see _stage_mechanics).
    mechanics = {s: _stage_mechanics(mechanical_params[s], preferred_area)
                 for s in stages}
    for s in stages:
        print("[find_psigma] %-6s gammaSC=%.5g gammaHC_ratio=%.5g "
              "alphaHC_ratio=%.5g A0=%s p0=(%.3g, %.3g)"
              % (s, mechanics[s]["gammaSC"], mechanics[s]["gammaHC_ratio"],
                 mechanics[s]["alphaHC_ratio"], mechanics[s]["preferred_area"],
                 mechanics[s]["hc_shape_index"], mechanics[s]["sc_shape_index"]),
              flush=True)

    def evaluate(psigmas):
        # De-duplicate at 3 decimals. NOT 2: psigma is now a junction-stress
        # threshold of order 0.01-0.05, so rounding to 2 decimals collapsed
        # distinct sweep points onto each other (0.015 and 0.045 became
        # 0.01 and 0.04).
        wanted = sorted({round(float(p), 5) for p in psigmas} - set(scores))
        if not wanted:
            return

        # ---- 1. every (psigma, stage, array) run of this pass, in ONE pool ----
        # The points in a pass are independent, and so are the two stages, so
        # the whole pass is one flat batch. Doing a point at a time offers only
        # n_arrays (=10) tasks and leaves a bigger VM idle; this offers
        # len(wanted) * n_stages * n_arrays.
        flat, span = [], {}
        for psigma in wanted:
            for stage in stages:
                mech = mechanics[stage]
                start = len(flat)
                # All repeats of a point are contiguous, so the point's pooled
                # model names are one slice of the results.
                for rep in range(1, n_repeats + 1):
                    flat.extend(full_model_tasks(
                        stage, mech["gammaSC"], mech["alphaHC_ratio"],
                        mech["hc_shape_index"], mech["sc_shape_index"],
                        gammaHC_ratio=mech["gammaHC_ratio"],
                        atoh_sensitivity=atoh_sensitivity,
                        notch_sensitivity=notch_sensitivity,
                        repressor_sensitivity=repressor_sensitivity,
                        bending=bending, quasi_static_threshold=quasi_static_threshold,
                        initial_notch_delta_level=initial_notch_delta_level,
                        preferred_area=mech["preferred_area"],
                        stress_shift=stress_shift,
                        stress_hill_exponent=stress_hill_exponent,
                        save_interval=save_interval,
                        psigma=psigma, t_end=t_end, dt=dt,
                        n_arrays=n_arrays, indices=indices,
                        reuse_existing_run=True,
                        name_prefix=REPEAT_PREFIX[rep]))
                span[(psigma, stage)] = (start, len(flat))
        pool_size = n_workers or (os.cpu_count() or 1)
        pool_size = max(1, min(pool_size, len(flat)))
        print("[find_psigma] pass: %d psigma x %d stage(s) x %d array(s) x %d "
              "repeat(s) = %d run(s) on %d worker(s)"
              % (len(wanted), len(stages), n_arrays, n_repeats, len(flat),
                 pool_size), flush=True)
        if pool_size > 1 and len(flat) > 1:
            with ProcessPoolExecutor(max_workers=pool_size) as pool:
                all_names = list(pool.map(_run_full_model_one, flat))
        else:
            all_names = [_run_full_model_one(t) for t in flat]
        names_of = {k: all_names[a:b] for k, (a, b) in span.items()}

        # ---- 2. score each point, cache hits first, misses in one pool -------
        # Scoring rescans every frame of every run, so once the simulations are
        # parallel it is the next bottleneck; the points are independent here
        # too. The cache is read and written by the PARENT only.
        results, to_score = {}, []
        for psigma in wanted:
            for stage in stages:
                model_names = names_of[(psigma, stage)]
                _ck = _score_cache_key(stage, psigma, stress_shift,
                                       stress_hill_exponent, type_by, threshold,
                                       max_number_of_neighbors)
                _fp = _history_fingerprint(model_names)
                _hit = score_cache.get(_ck) if use_score_cache else None
                if _hit is not None and _hit.get("fingerprint") == _fp:
                    # Not one history.hf5 has changed since this was scored, so
                    # the comparison would re-derive the identical numbers.
                    results[(psigma, stage)] = _hit["result"]
                    print("[find_psigma] %s psigma=%.5g: cached score reused "
                          "(%s/%s runs settled; no run changed) -> total %.3f"
                          % (stage, psigma, _hit.get("n_settled", "?"),
                             _hit.get("n_runs", len(model_names)),
                             float(_hit["result"]["total"])), flush=True)
                else:
                    to_score.append((psigma, stage, _ck, _fp, model_names))
        if to_score:
            args = [(stage, names, type_by, threshold, max_number_of_neighbors)
                    for _p, stage, _c, _f, names in to_score]
            score_pool = max(1, min(pool_size, len(args)))
            print("[find_psigma] scoring %d point(s) on %d worker(s)"
                  % (len(args), score_pool), flush=True)
            if score_pool > 1 and len(args) > 1:
                with ProcessPoolExecutor(max_workers=score_pool) as pool:
                    scored = list(pool.map(_score_one_point, args))
            else:
                scored = [_score_one_point(a) for a in args]
            for (psigma, stage, _ck, _fp, model_names), result in zip(to_score, scored):
                results[(psigma, stage)] = result
                if use_score_cache:
                    score_cache[_ck] = {
                        "fingerprint": _fp, "result": result,
                        "n_runs": len(model_names),
                        "n_settled": sum(
                            _reached_steady_state(os.path.join(RESULTS_DIR, n))
                            for n in model_names)}
            if use_score_cache:
                _store_score_cache(score_cache)

        # ---- 3. assemble, in psigma order, persisting as we go --------------
        for psigma in wanted:
            entry, objective = {}, 0.0
            for stage in stages:
                result = results[(psigma, stage)]
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
                               # the RESOLVED mechanics, so the json says what
                               # was actually run rather than what was passed
                               "mechanical_params": mechanics,
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
