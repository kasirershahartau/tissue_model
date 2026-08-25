"""Shared runner for the two morphology tests (line tension / sharp-angle
threshold). Re-simulates the BEST-FIT E17.5 point (10 base + 10 ablation runs)
with a single modification, into DISTINCT results folders (a ``tag`` prefix, so
nothing collides with or overwrites the real best-fit runs), then pools and
scores the four n-sigma terms and prints them next to the original best fit.

Used by test_line_tension.py and test_sharp_angle.py.
"""
import os
import re
import ast
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import run_model
from run_model import (run, _li_levels_kwargs_for_initial_sheet, _load_saved_threshold,
                       _strip_results_prefix, MECHANICS_TERMS)
from post_processing import (extract_model_mechanics,
                             compare_pooled_model_mechanics_to_experiments, RESULTS_DIR)

STAGE = "E17.5"
N_SHEETS = 10
ABLATED_CELLS = [337, 304, 65, 114]     # E17.5 ablation targets (as the fit used)
POST_ABLATION_FRAME = -1
QST = 0.01                              # match the original best-fit runs (base & ablation at 0.01)
INITuple = tuple("random_periodic_array%d_for_E17" % i for i in range(N_SHEETS))
_BASE_RX = re.compile(r"^fit_gSC[\d.]+_gHC[\d.]+_aHC[\d.]+_ps[\d.]+(?:_p0[\d.]+)?_[0-9a-f]{10}$")
_BEST = dict(gammaSC=0.0455, gammaHC_ratio=1.2163, alphaHC_ratio=1.0295, shape_index=1.2585)
_TOL = dict(gammaSC=0.002, gammaHC_ratio=0.01, alphaHC_ratio=0.01, shape_index=0.01)


def _parse(folder):
    p = {}
    with open(os.path.join(RESULTS_DIR, folder, "parameters.txt"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            k, sep, v = line.partition(":")
            if sep:
                try:
                    p[k.strip()] = ast.literal_eval(v.strip())
                except (ValueError, SyntaxError):
                    pass
    return p


def best_params():
    """Exact (gSC, gHC, aHC, p0) the best-fit runs used, read from disk so the
    test is identical to the original except for the one modification."""
    for name in sorted(os.listdir(RESULTS_DIR)):
        if not _BASE_RX.match(name) or not os.path.isfile(os.path.join(RESULTS_DIR, name, "parameters.txt")):
            continue
        try:
            p = _parse(name)
        except FileNotFoundError:
            continue
        if all(abs(float(p.get(k, 1e9)) - _BEST[k]) <= _TOL[k] for k in _BEST):
            return (float(p["gammaSC"]), float(p["gammaHC_ratio"]),
                    float(p["alphaHC_ratio"]), float(p.get("shape_index", 0.0)))
    raise RuntimeError("no best-fit base run folder found to read exact params from")


def original_scores():
    """The best fit's own per-term n-sigma, from the optimization trace."""
    path = os.path.join(RESULTS_DIR, "%s_optimization_trace.jsonl" % STAGE)
    L = [json.loads(l) for l in open(path)]
    best = min(L, key=lambda d: d["objective"])
    return {t: best["nsigma_%s" % t] for t in MECHANICS_TERMS}, best["objective"]


def _run_one_sheet(args):
    """Base + ablation run for ONE initial sheet, into tag-prefixed folders, then
    extract its per-term model distributions. Returns (i, terms|None, err)."""
    i, initial, gSC, gHC, aHC, p0, thr, tag, sharp_angle_threshold, line_tension = args
    li_kwargs = _li_levels_kwargs_for_initial_sheet(initial)
    atoh_kwargs = {"atoh_sensitivity": thr}
    common = dict(no_differentiation=True, reuse_existing_run=False, shape_index=p0,
                  quasi_static_threshold=QST, sharp_angle_threshold=sharp_angle_threshold,
                  line_tension=line_tension)
    base_name = "%s_a%d" % (tag, i)
    try:
        base = _strip_results_prefix(
            run(gSC, gHC, aHC, 0, initial, name=base_name, **common, **li_kwargs, **atoh_kwargs))
        abl = _strip_results_prefix(
            run(gSC, gHC, aHC, 0, base, name=base_name, ablated_cells=ABLATED_CELLS,
                **common, **atoh_kwargs))
        terms = extract_model_mechanics(
            base, type_by="delta_level", threshold=thr,
            ablation_model_name=abl, ablated_cells=ABLATED_CELLS,
            post_ablation_frame=POST_ABLATION_FRAME)
        return i, terms, None
    except Exception as exc:  # noqa: BLE001 - a degenerate sheet must not kill the test
        return i, None, "%s: %s" % (type(exc).__name__, exc)


def run_modification_test(tag, sharp_angle_threshold=0.1, line_tension=None, n_workers=5):
    gSC, gHC, aHC, p0 = best_params()
    print("=" * 74, flush=True)
    print("MODIFICATION TEST  tag=%s" % tag, flush=True)
    print("  best-fit params: gammaSC=%.4f gammaHC=%.4f alphaHC=%.4f shape_index=%.4f"
          % (gSC, gHC, aHC, p0), flush=True)
    print("  sharp_angle_threshold=%s rad   line_tension=%s   (base+ablation qst=%.2f)"
          % (sharp_angle_threshold, line_tension, QST), flush=True)
    print("  running %d base + %d ablation simulations into '%s_a*' folders ..."
          % (N_SHEETS, N_SHEETS, tag), flush=True)

    per_sheet_thr = {init: _load_saved_threshold(init) for init in INITuple}
    tasks = [(i, init, gSC, gHC, aHC, p0, per_sheet_thr[init], tag,
              sharp_angle_threshold, line_tension)
             for i, init in enumerate(INITuple)]

    results = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, terms, err in ex.map(_run_one_sheet, tasks):
                results.append((i, terms, err))
                print("  sheet %d: %s" % (i, "ok" if terms else "DROPPED (%s)" % err), flush=True)
    else:
        for task in tasks:
            i, terms, err = _run_one_sheet(task)
            results.append((i, terms, err))
            print("  sheet %d: %s" % (i, "ok" if terms else "DROPPED (%s)" % err), flush=True)

    model_terms = {t: [] for t in MECHANICS_TERMS}
    for _, terms, _ in results:
        if terms is None:
            continue
        for t in MECHANICS_TERMS:
            arr = terms.get(t)
            if arr is not None and len(np.atleast_1d(arr)):
                model_terms[t].append(np.asarray(arr, float))

    z = compare_pooled_model_mechanics_to_experiments(model_terms, STAGE)
    obj = float(np.nansum([v ** 2 for v in z.values()]))
    orig_z, orig_obj = original_scores()

    print("-" * 74, flush=True)
    print("%-14s %12s %14s %10s" % ("term", "ORIGINAL z", "MODIFIED z", "delta"), flush=True)
    for t in MECHANICS_TERMS:
        oz, mz = orig_z[t], z[t]
        d = (mz - oz) if (np.isfinite(oz) and np.isfinite(mz)) else float("nan")
        print("%-14s %12.2f %14.2f %+10.2f" % (t, oz, mz, d), flush=True)
    print("-" * 74, flush=True)
    print("objective (sum z^2):   ORIGINAL %.2f    MODIFIED %.2f" % (orig_obj, obj), flush=True)
    print("morphology images: %s\\%s_a*\\finale.png" % (RESULTS_DIR, tag), flush=True)
    print("=" * 74, flush=True)
    return z, obj
