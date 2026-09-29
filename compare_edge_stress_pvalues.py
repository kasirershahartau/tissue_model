"""Hierarchical p-values between interface classes, per stage and pT.

    python compare_edge_stress_pvalues.py
    python compare_edge_stress_pvalues.py --stub-missing-deps
    python compare_edge_stress_pvalues.py --fake        # plumbing check, no stats dep

Compares the interface-stress distributions of HC:HC, HC:SC and SC:SC against
each other, within each (stage, pT), using HierarchicalTwoSamplesCompare.

THE HIERARCHY. One data point is one INITIAL ARRAY: each of its repeats is
reduced to its mean pair stress, and those are averaged. Ten arrays give ten
values per class. This is the unit every score in this project treats as
independent — repeats of one array share an initial condition — and it keeps the
test from counting ~20 000 pairs as 20 000 observations. Stress is continuous,
so the comparer is called with continues=True.

--group-mode pairs restores the other layout, every pair as a measurement within
its array's replicate. It is the more literally hierarchical arrangement but
gives the test far more observations, and correspondingly smaller p-values.

NOTE ON THE REPEAT COUNT. These points have THREE repeats per array, not ten
(10 arrays x 3 repeats = 30 runs at each stage and pT); the averaging is over
those three.

WHAT IS COMPARED. All three pairings of the classes, so HC:HC appears here even
though the figure omits it. HC:HC rests on very few pairs (62-470 across ten
arrays, against ~19 000 for the others), and an array with none contributes no
replicate — the n_arrays columns say how many actually did.

DEPENDENCIES. statistical_analysis lives in the sibling tissue_analyzing_tool
package and imports statsmodels and scikit_posthocs at module level. Like
plot_pvalue_vs_psigma, this script imports NOTHING from post_processing, so it
needs only pandas, numpy and that stats stack — no tyssue.
--stub-missing-deps injects a dummy scikit_posthocs (unused by the comparer);
--fake swaps in Mann-Whitney so the data handling can be checked without the
dependency at all, and is NOT a substitute for the hierarchical test: it treats
every pair as independent and will report far smaller p-values.
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("TISSUE_RESULTS_DIR", r"D:\Kasirer\results")
ANALYZER_PATH = os.environ.get(
    "TISSUE_ANALYZER_PATH",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_image_processing\tissue_analyzing_tool")

DETAIL = "fullmodel_edge_stress.pkl"
RUNS = "fullmodel_runs.pkl"
OUT = "edge_stress_pvalues"
XLSX = "fullmodel_tables.xlsx"
CLASSES = ("HC:HC", "HC:SC", "SC:SC")


def get_comparer(stub):
    sys.path.insert(0, ANALYZER_PATH)
    if stub and "scikit_posthocs" not in sys.modules:
        import types
        sys.modules["scikit_posthocs"] = types.ModuleType("scikit_posthocs")
        print("  NOTE: scikit_posthocs stubbed out (unused by the comparer)")
    from statistical_analysis import HierarchicalTwoSamplesCompare
    return HierarchicalTwoSamplesCompare


def _read(path):
    """A saved table, with the parameter under the name this code uses.

    Inlined rather than imported from build_experimental_tables, which pulls in
    post_processing and so tyssue; this script must run in the stats
    environment.
    """
    f = pd.read_pickle(path)
    return f.rename(columns={c: c.replace("pT", "psigma")
                             for c in f.columns if "pT" in c})


def load(results_dir, frame, effectors):
    detail = _read(os.path.join(results_dir, DETAIL))
    runs = _read(os.path.join(results_dir, RUNS))
    if "effectors" in detail.columns:
        detail = detail[detail["effectors"] == effectors]
    detail = detail[detail["frame"] == frame]
    if not len(detail):
        raise SystemExit("no rows for frame=%s effectors=%s" % (frame, effectors))
    return detail.merge(runs[["model_name", "stage", "psigma", "initial_array"]],
                        on="model_name", how="left")


def replicates(g, edge_class, mode="array"):
    """The replicates handed to the comparer; arrays with no pairs are dropped.

    ``array`` (the default): one data point per initial array — the mean over
    that array's repeats, each repeat first reduced to its own mean pair stress.
    Ten arrays therefore give ten single-valued replicates. Averaging the
    repeats first is what makes an array the independent unit: repeats of one
    array share an initial condition and are not independent draws.

    ``pairs``: every neighbour pair of the array as a measurement WITHIN that
    array's replicate. Genuinely hierarchical, but it hands the test ~20 000
    observations and will report far smaller p-values.
    """
    sub = g[g["edge_class"] == edge_class]
    out = []
    for _arr, h in sub.groupby("initial_array"):
        if mode == "array":
            per_run = h.groupby("model_name")["stress"].mean().to_numpy(float)
            per_run = per_run[np.isfinite(per_run)]
            if per_run.size:
                out.append(np.array([per_run.mean()], float))
        else:
            v = h["stress"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="t0", choices=("t0", "initial"))
    ap.add_argument("--effectors", default="contractility")
    ap.add_argument("--group-mode", dest="group_mode", default="array",
                    choices=("array", "pairs"),
                    help="array (default): one point per initial array, the mean "
                         "over its repeats. pairs: every neighbour pair as a "
                         "measurement within its array's replicate")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--stub-missing-deps", dest="stub", action="store_true",
                    help="inject a dummy scikit_posthocs so the module imports")
    ap.add_argument("--fake", action="store_true",
                    help="Mann-Whitney instead of the hierarchical comparer, to "
                         "check the data handling without the stats dependency")
    ap.add_argument("--no-xlsx", dest="xlsx", action="store_false")
    a = ap.parse_args()

    compare = None if a.fake else get_comparer(a.stub)
    detail = load(a.results_dir, a.frame, a.effectors)

    rows = []
    for (stage, ps), g in detail.groupby(["stage", "psigma"]):
        for c1, c2 in itertools.combinations(CLASSES, 2):
            d1 = replicates(g, c1, a.group_mode)
            d2 = replicates(g, c2, a.group_mode)
            rec = dict(stage=stage, psigma=float(ps), frame=a.frame,
                       effectors=a.effectors, class_1=c1, class_2=c2,
                       group_mode=a.group_mode,
                       n_arrays_1=len(d1), n_arrays_2=len(d2),
                       n_values_1=int(sum(x.size for x in d1)),
                       n_values_2=int(sum(x.size for x in d2)),
                       mean_1=float(np.mean([x.mean() for x in d1])) if d1 else np.nan,
                       mean_2=float(np.mean([x.mean() for x in d2])) if d2 else np.nan,
                       pvalue=np.nan, test="mannwhitney" if a.fake else "hierarchical",
                       note="")
            if len(d1) < 2 or len(d2) < 2:
                rec["note"] = "too few replicates"
                rows.append(rec)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if a.fake:
                        from scipy import stats as st
                        rec["pvalue"] = float(st.mannwhitneyu(
                            np.concatenate(d1), np.concatenate(d2),
                            alternative="two-sided")[1])
                    else:
                        rec["pvalue"] = float(compare(
                            [np.asarray(x, float) for x in d1],
                            [np.asarray(x, float) for x in d2],
                            continues=True).compare_samples(verbose=True))
            except Exception as exc:                        # noqa: BLE001
                rec["note"] = "%s: %s" % (type(exc).__name__, str(exc)[:70])
            rows.append(rec)
        print("  %-6s pT %.3f done" % (stage, ps), flush=True)

    res = pd.DataFrame(rows)
    out = res.rename(columns={"psigma": "pT"})
    out.to_pickle(os.path.join(a.results_dir, OUT + ".pkl"))
    out.to_csv(os.path.join(a.results_dir, OUT + ".csv"), index=False)
    if a.xlsx:
        path = os.path.join(a.results_dir, XLSX)
        try:
            mode = "a" if os.path.isfile(path) else "w"
            kw = dict(if_sheet_exists="replace") if mode == "a" else {}
            with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kw) as w:
                out.to_excel(w, sheet_name=OUT[:31], index=False)
            print("  added sheet '%s' to %s" % (OUT, XLSX))
        except Exception as exc:                            # noqa: BLE001
            print("  xlsx failed (%s: %s); the pkl and csv are written"
                  % (type(exc).__name__, exc))

    print("\n  %-6s %-7s %-7s %-7s %10s %10s %12s %s"
          % ("stage", "pT", "class 1", "class 2", "mean 1", "mean 2",
             "p", "note"))
    for _i, r in out.iterrows():
        print("  %-6s %-7.3f %-7s %-7s %10.4f %10.4f %12.3g %s"
              % (r["stage"], r["pT"], r["class_1"], r["class_2"],
                 r["mean_1"], r["mean_2"], r["pvalue"], r["note"]))
    print("\nwrote %s.pkl / .csv" % os.path.join(a.results_dir, OUT))


if __name__ == "__main__":
    main()
