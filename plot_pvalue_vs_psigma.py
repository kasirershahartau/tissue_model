"""Score 1 and score 2 against psigma as HIERARCHICAL p-VALUES, both stages.

    python plot_pvalue_vs_psigma.py                 # run this in an env that has
                                                    # statsmodels + scikit_posthocs
    python plot_pvalue_vs_psigma.py --group-mode repeats
    python plot_pvalue_vs_psigma.py --stub-missing-deps    # see the note below

Companion to plot_scores_vs_psigma.py, which plots the same four buckets as
chi^2. Writes <results>/pvalue_vs_psigma.png and pvalue_vs_psigma.csv.

READ THE AXIS THE OTHER WAY ROUND. chi^2 is a distance, so low is good; a p-value
is evidence AGAINST the model matching, so HIGH is good and points below 0.05 are
where the model is distinguishable from the experiment. The two figures should
look roughly like mirror images, and where they do not is the interesting part.

WHAT IS COMPARED. Each initial array contributes ONE data point per bucket, the
mean over its repeats (--group-mode averages, the default), against the three
experimental values. So dist1 has 10 replicates and dist2 has 3, and the
comparison is between arrays and experiments — the repeat-to-repeat spread is
averaged away rather than being counted as independent evidence, exactly as in
the chi^2 version. --group-mode repeats instead hands the comparer each array's
repeats as measurements WITHIN that array's replicate, which is the more
genuinely hierarchical layout; it changes nothing at psigma values that only have
one repeat.

BUCKETS ARE PLOTTED SEPARATELY, NOT COMBINED. chi^2 could be summed because the
terms add; p-values cannot, and Fisher's method would be wrong here because the
two buckets of each score are compositional (%0 and %1 of the same total, HC:HC
and HC:SC of the same pair count) and therefore dependent. Each subplot shows its
two buckets as separate lines.

EXPECT SATURATION. The p-value objective was already tried for the MECHANICAL fit
and abandoned: it sat at ~0 across almost the whole parameter range, giving a
flat landscape with no gradient (see the docstring of
compare_pooled_model_mechanics_to_experiments). With 10 model replicates against
3 experimental ones a small real offset is easily significant, so p may pin to 0
wherever the model is systematically off — that is a property of the test, not a
new finding. The chi^2 figure remains the one with usable gradient.

DEPENDENCIES. statistical_analysis lives in the sibling tissue_analyzing_tool
package and imports statsmodels and scikit_posthocs at module level. This script
deliberately does NOT import post_processing (and so not tyssue), so it runs in
any environment with pandas, numpy, matplotlib and that stats stack.
--stub-missing-deps injects a dummy scikit_posthocs so the module can import
without it; HierarchicalTwoSamplesCompare does not use that package, but the flag
is opt-in because stubbing a stats dependency is not something to do silently.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get("TISSUE_RESULTS_DIR", r"D:\Kasirer\results")
EXP_DIR = os.environ.get(                       # same default as post_processing
    "EXPERIMENTAL_DATA_DIR",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\papers"
    r"\Dynamic lateral inhibition in the utricle\Experimental Data")
ANALYZER_PATH = os.environ.get(
    "TISSUE_ANALYZER_PATH",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_image_processing\tissue_analyzing_tool")

OUT_PNG = "pvalue_vs_psigma.png"
OUT_CSV = "pvalue_vs_psigma.csv"
COLOUR = {"E17.5": "tab:blue", "P0": "tab:red"}
ZOOM = (0.1585, 0.1715)

# (score, bucket label, model column, how to get the experimental values)
BUCKETS = (
    (1, "HC:HC %", "pct_HCHC_contacts_t0", "pairs_hchc"),
    (1, "HC:SC %", "pct_HCSC_contacts_t0", "pairs_hcsc"),
    (2, "% events, 0 HC nb", "pct_events_0_HC_neighbours", "events_0"),
    (2, "% events, 1 HC nb", "pct_events_1_HC_neighbours", "events_1"),
)


def get_comparer(stub):
    sys.path.insert(0, ANALYZER_PATH)
    if stub and "scikit_posthocs" not in sys.modules:
        import types
        sys.modules["scikit_posthocs"] = types.ModuleType("scikit_posthocs")
        print("  NOTE: scikit_posthocs stubbed out (unused by the comparer)")
    from statistical_analysis import HierarchicalTwoSamplesCompare
    return HierarchicalTwoSamplesCompare


# ---------------------------------------------------------------- experiment --
def _pair_percentages(is_HC, adjacency):
    A = (np.asarray(adjacency) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    hc = np.asarray(is_HC, bool); sc = ~hc
    total = A.sum() / 2.0
    if total <= 0:
        return np.nan, np.nan
    return (100.0 * (A[np.ix_(hc, hc)].sum() / 2.0) / total,
            100.0 * A[np.ix_(hc, sc)].sum() / total)


_EXP = {}


def exp_values(stage, key):
    """The three experimental values for a bucket, one per experiment."""
    if stage not in _EXP:
        prefix = "E17" if stage == "E17.5" else "P0"
        hchc, hcsc = [], []
        for e in (1, 2, 3):
            ci = pd.read_pickle(os.path.join(EXP_DIR, stage,
                                "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
            cm = np.load(os.path.join(EXP_DIR, stage,
                         "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
            valid = ci.valid.values.astype(bool)
            a, b = _pair_percentages((ci.type.values == 1)[valid],
                                     np.asarray(cm)[np.ix_(valid, valid)])
            hchc.append(a); hcsc.append(b)
        p0, p1 = [], []
        for i in range(3):
            counts = np.asarray(np.load(os.path.join(
                EXP_DIR, "%s differentiating cells_experiment%d.npy" % (stage, i))),
                float)
            if counts.size == 0:
                continue
            p0.append(100.0 * np.mean(counts == 0))
            p1.append(100.0 * np.mean(counts == 1))
        _EXP[stage] = dict(pairs_hchc=hchc, pairs_hcsc=hcsc,
                           events_0=p0, events_1=p1)
    return list(_EXP[stage][key])


# --------------------------------------------------------------------- model --
def model_groups(g, col, mode):
    """[[measurement, ...], ...] — one inner list per initial array."""
    out = []
    for _arr, gg in g.groupby("initial_array"):
        v = gg[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        out.append([float(v.mean())] if mode == "averages" else [float(x) for x in v])
    return out


def combine(p):
    """One p per score from its two buckets: the smaller, Bonferroni-doubled.

    Fisher's method is not available here — the two buckets of each score are
    compositional (%0 and %1 of the same total; HC:HC and HC:SC of the same pair
    count), so their p-values are dependent and Fisher would overstate the
    combined evidence. min-p with a factor 2 is valid under ANY dependence; it is
    conservative, which is the right direction to err when the conclusion is
    'the model is distinguishable from the data here'."""
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return min(1.0, 2.0 * float(p.min())) if p.size else np.nan


def draw(res, a):
    """The chi^2 figure's layout, with p on the y axis: one line per stage."""
    bad = res[res["note"].fillna("") != ""]
    if len(bad):
        print("\n  %d bucket(s) produced no p-value:" % len(bad))
        for _i, r in bad.head(8).iterrows():
            print("    %-6s %.3f %-20s %s" % (r["stage"], r["psigma"],
                                              r["bucket"], r["note"]))

    comb = (res.dropna(subset=["pvalue"]).groupby(["stage", "score", "psigma"])
            ["pvalue"].apply(lambda s: combine(s.to_numpy())).reset_index())
    comb = comb.rename(columns={"pvalue": "p_combined"})
    comb.to_csv(os.path.join(RESULTS_DIR, OUT_CSV.replace(".csv", "_combined.csv")),
                index=False)

    # The p-values span 1 down to ~1e-186, so an honest log axis compresses the
    # only region anyone reads (around 0.05) into a sliver. Everything below the
    # floor means the same thing — the model is decisively distinguishable — and
    # the exact exponent carries no information, so clip there and mark it.
    floor = float(a.floor)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6),
                             gridspec_kw=dict(height_ratios=[1.35, 1.0]))
    for col_i, score in ((0, 1), (1, 2)):
        labels = [b[1] for b in BUCKETS if b[0] == score]
        for row, zoom in ((0, False), (1, True)):
            ax = axes[row][col_i]
            for stage in ("E17.5", "P0"):
                c = COLOUR[stage]
                # the two buckets, faint, so the combination is auditable
                for k, lab in enumerate(labels):
                    s = res[(res["stage"] == stage) & (res["score"] == score)
                            & (res["bucket"] == lab)].sort_values("psigma")
                    s = s[np.isfinite(s["pvalue"])]
                    if zoom:
                        s = s[s["psigma"].between(*ZOOM)]
                    if len(s):
                        ax.plot(s["psigma"], np.maximum(s["pvalue"], floor),
                                "-" if k == 0 else "--", color=c, lw=0.9,
                                alpha=0.32, zorder=2,
                                label="%s %s" % (stage, lab) if not zoom else None)
                s = comb[(comb["stage"] == stage) & (comb["score"] == score)]
                s = s.sort_values("psigma")
                if zoom:
                    s = s[s["psigma"].between(*ZOOM)]
                if not len(s):
                    continue
                y = np.maximum(s["p_combined"].to_numpy(float), floor)
                ax.plot(s["psigma"], y, "-o", color=c, ms=4.5, lw=1.8, zorder=3,
                        label="%s (combined)" % stage if not zoom else None)
                at_floor = s[s["p_combined"] <= floor]
                if len(at_floor):
                    ax.plot(at_floor["psigma"], np.full(len(at_floor), floor),
                            "v", color=c, ms=8, mfc="none", mew=1.5, zorder=4)
            ax.axhline(0.05, color="k", lw=1.0, ls=":", alpha=0.7)
            ax.set_yscale("log")
            ax.set_ylim(floor / 3.0, 3.0)
            ax.set_xlabel("$p_\\sigma$")
            ax.set_ylabel("hierarchical p-value  (HIGHER = better agreement)")
            ax.grid(alpha=0.25, which="both")
            if zoom:
                ax.set_xlim(*ZOOM)
                ax.set_title("zoom: $p_\\sigma$ 0.159 – 0.171", fontsize=9)
            else:
                ax.annotate("p = 0.05", (ax.get_xlim()[0], 0.058), fontsize=7.5,
                            alpha=0.8)
                ax.set_title("Score %d — %s" % (score, " and ".join(labels)),
                             fontsize=11)
                ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Hierarchical p-value against $p_\\sigma$ — bold: the score's two "
                 "buckets combined (min-p x2); faint: each bucket. Triangles = "
                 "clipped at p $\\leq$ %g%s"
                 % (floor, "   [FAKE p-values]" if a.fake else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS_DIR, OUT_PNG), dpi=170, bbox_inches="tight")
    print("\nwrote %s\n      %s" % (os.path.join(RESULTS_DIR, OUT_PNG),
                                    os.path.join(RESULTS_DIR, OUT_CSV)))

    print("\n  combined p per score (min of the two buckets x 2)")
    print("    %-6s %-7s %10s %10s" % ("stage", "psigma", "score 1", "score 2"))
    for (stage, ps), g in comb.groupby(["stage", "psigma"]):
        v = {int(r["score"]): r["p_combined"] for _i, r in g.iterrows()}
        print("    %-6s %-7.3f %10.3g %10.3g"
              % (stage, ps, v.get(1, np.nan), v.get(2, np.nan)))
    print("\n  psigma values where the model is NOT distinguishable (both p > 0.05)")
    for stage in ("E17.5", "P0"):
        g = comb[comb["stage"] == stage].pivot_table(index="psigma", columns="score",
                                                     values="p_combined")
        if 1 in g and 2 in g:
            ok = g[(g[1] > 0.05) & (g[2] > 0.05)]
            print("    %-6s %s" % (stage, ", ".join("%.3f" % x for x in ok.index)
                                   if len(ok) else "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-mode", dest="group_mode", default="averages",
                    choices=["averages", "repeats"])
    ap.add_argument("--keep-collapsed", dest="keep_collapsed", action="store_true",
                    help="keep runs whose pattern collapsed (HC fraction < 0.10)")
    ap.add_argument("--floor", type=float, default=1e-8,
                    help="display floor for the log y axis; anything at or below "
                         "it is drawn as a triangle on the floor (default 1e-8)")
    ap.add_argument("--from-csv", dest="from_csv", action="store_true",
                    help="re-draw from an existing pvalue_vs_psigma.csv without "
                         "recomputing anything (needs no stats stack)")
    ap.add_argument("--stub-missing-deps", dest="stub", action="store_true")
    ap.add_argument("--fake-pvalues", dest="fake", action="store_true",
                    help="Mann-Whitney instead of the hierarchical comparer. For "
                         "checking the plumbing where the stats stack is missing "
                         "- NOT a substitute for the real test.")
    a = ap.parse_args()

    compare = None
    if not a.fake and not a.from_csv:
        compare = get_comparer(a.stub)

    if a.from_csv:
        res = pd.read_csv(os.path.join(RESULTS_DIR, OUT_CSV))
        res["note"] = res["note"].fillna("")
        print("re-drawing from %s: %d row(s)" % (OUT_CSV, len(res)))
        draw(res, a)
        return

    df = pd.read_pickle(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    df = df[df["error"].fillna("") == ""]
    if not a.keep_collapsed:
        df = df[~df["collapsed"].astype(bool)]
    print("%d run(s), %d psigma value(s), group mode %r"
          % (len(df), df["psigma"].nunique(), a.group_mode))
    if a.group_mode == "averages":
        print("  NOTE: one measurement per replicate, so the random effect for "
              "replicate is\n        degenerate — replicate variance cannot be "
              "separated from residual\n        variance and the comparer's mixed "
              "model may fail or fall through to a\n        GLM. That is inherent "
              "to comparing summary percentages, not a bug. If\n        buckets "
              "come back with errors, try --group-mode repeats.")

    rows = []
    for (stage, ps), g in df.groupby(["stage", "psigma"]):
        for score, label, col, key in BUCKETS:
            dist1 = model_groups(g, col, a.group_mode)
            dist2 = [[v] for v in exp_values(stage, key)]
            rec = dict(stage=stage, psigma=float(ps), score=score, bucket=label,
                       n_arrays=len(dist1), n_runs=len(g),
                       model_mean=float(np.mean([np.mean(d) for d in dist1]))
                       if dist1 else np.nan,
                       exp_mean=float(np.mean([np.mean(d) for d in dist2]))
                       if dist2 else np.nan,
                       pvalue=np.nan, note="")
            if len(dist1) < 2 or len(dist2) < 2:
                rec["note"] = "too few replicates"
                rows.append(rec); continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if a.fake:
                        from scipy import stats as st
                        flat1 = [x for d in dist1 for x in d]
                        flat2 = [x for d in dist2 for x in d]
                        rec["pvalue"] = float(st.mannwhitneyu(
                            flat1, flat2, alternative="two-sided")[1])
                    else:
                        # percentages are continuous -> continues=True
                        rec["pvalue"] = float(compare(
                            [np.asarray(d, float) for d in dist1],
                            [np.asarray(d, float) for d in dist2],
                            continues=True).compare_samples())
            except Exception as exc:                    # noqa: BLE001
                rec["note"] = "%s: %s" % (type(exc).__name__, str(exc)[:70])
            rows.append(rec)
        print("  %-6s psigma %.3f done" % (stage, ps), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, OUT_CSV), index=False)
    draw(res, a)


if __name__ == "__main__":
    main()
