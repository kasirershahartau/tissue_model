"""Score ONE mechanical parameter point under the current objective, without
running a fit and without touching the optimization trace.

    python score_point.py E17.5                       # that stage's stored best
    python score_point.py E17.5 --gammaSC 0.2461 --alphaHC 1.06 --hc 4.86 --sc 5.72

Runs all 10 sheets (base + ablation), pools them exactly as
``find_mechanical_parameters`` does, and reports every term's signed n-sigma and
its z**2 contribution plus the total. Completed runs are reused, so re-scoring a
point that was already simulated is fast.

Use it to check whether an EXISTING best point is still acceptable after the
objective changed (e.g. once ``shrinkage`` was added as a fifth term), before
deciding whether a full re-fit is worth the hours.
"""
import argparse
import numpy as np

import run_model as rm
from post_processing import (extract_model_mechanics,
                             compare_pooled_model_mechanics_to_experiments)

# stored bests, so `score_point.py <stage>` just works
BEST = {
    "E17.5": dict(gammaSC=0.2461, alphaHC=1.00, hc=4.86, sc=5.72),   # 5-term score 1.076
    "P0":    dict(gammaSC=0.2298, alphaHC=1.00, hc=5.1487, sc=5.6706),
}
PREFERRED_AREA = 0.593 * np.pi / 4
ABLATED_CELLS = [337, 304, 65, 114]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["E17.5", "P0"])
    ap.add_argument("--gammaSC", type=float, default=None)
    ap.add_argument("--alphaHC", type=float, default=None)
    ap.add_argument("--hc", type=float, default=None, help="hc_shape_index")
    ap.add_argument("--sc", type=float, default=None, help="sc_shape_index")
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    d = BEST[a.stage]
    gammaSC = a.gammaSC if a.gammaSC is not None else d["gammaSC"]
    alphaHC = a.alphaHC if a.alphaHC is not None else d["alphaHC"]
    hc = a.hc if a.hc is not None else d["hc"]
    sc = a.sc if a.sc is not None else d["sc"]

    suffix = "E17" if a.stage == "E17.5" else "P0"
    sheets = ["random_periodic_array%d_for_%s" % (i, suffix) for i in range(a.n_sheets)]
    print("scoring %s | gammaSC=%.4f alphaHC=%.4f hc_p0=%.4f sc_p0=%.4f | A0=%.4f"
          % (a.stage, gammaSC, alphaHC, hc, sc, PREFERRED_AREA), flush=True)

    # Same task tuple the fit worker consumes (see _evaluate_mechanics_for_sheet).
    tasks = [(gammaSC, 1.0, alphaHC, s, a.stage, ABLATED_CELLS, -1,
              float(rm._load_saved_threshold(s)), "delta_level",
              10000, 1e-4, 30.0, False, 0.0, 0.03, 0.02, None,
              hc, sc, 0.02, PREFERRED_AREA)
             for s in sheets]
    n_workers = a.workers or min(len(tasks), __import__("os").cpu_count() or 1)
    if n_workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            details = list(ex.map(rm._evaluate_mechanics_for_sheet, tasks))
    else:
        details = [rm._evaluate_mechanics_for_sheet(t) for t in tasks]

    model_terms = {term: [] for term in rm.MECHANICS_TERMS}
    n_ok = 0
    for d_ in details:
        if d_ is None:
            continue
        n_ok += 1
        for term in rm.MECHANICS_TERMS:
            if d_.get(term) is not None:
                model_terms[term].append(d_[term])

    z = compare_pooled_model_mechanics_to_experiments(model_terms, a.stage)
    active = list(rm._MECHANICS_ROUNDNESS_TERMS) + ["shrinkage"] + list(rm._MECHANICS_ABLATION_TERMS)
    print("\n%-14s %8s %9s %7s" % ("term", "z", "z^2", "n"))
    total = 0.0
    for term in active:
        zz = z[term]
        if not np.isfinite(zz):
            zz = rm._WORST_CASE_NSIGMA
        total += zz * zz
        print("%-14s %+8.2f %9.3f %7d"
              % (term, z[term], zz * zz,
                 sum(len(x) for x in model_terms[term])))
    print("%-14s %8s %9.3f   (%d/%d sheets contributed)"
          % ("TOTAL", "", total, n_ok, len(sheets)))


if __name__ == "__main__":
    main()
