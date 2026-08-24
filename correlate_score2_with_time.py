"""Does a P0 run's score 2 depend on how long it ran?

    python correlate_score2_with_time.py
    python correlate_score2_with_time.py --stage E17.5

Reads fullmodel_runs.pkl and scores EACH RUN on its own:

    z_bucket = (run % - mean_exp %) / SEM_exp        score2_run = z0^2 + z1^2

with no SEM_sim term, because a single run has no spread to put there. That
makes these numbers larger than the pooled scores and NOT comparable to them —
they exist only to be correlated against t_final, and any monotone rescaling of
score 2 leaves a rank correlation unchanged.

WHY THE WITHIN-PSIGMA NUMBER IS THE ONE THAT ANSWERS THE QUESTION. psigma moves
both quantities on its own: it changes how fast the pattern develops (so
t_final) and how well the neighbour statistics match (so score 2). Correlating
across the whole sweep would therefore mostly measure psigma, not run length. The
within-psigma correlation removes each psigma group's mean from both variables
first, so it asks the intended question: among runs that share a parameter point,
do the ones that ran longer score differently?

Runs with no differentiation events have no score 2 at all and are dropped, as
are collapsed runs (HC fraction < 0.10), where the statistic is degenerate.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from post_processing import RESULTS_DIR
from build_fullmodel_table import exp_targets


def per_run_score2(df, stage):
    """z^2 sum over the two buckets, experimental SEM only."""
    _hh, _hs, e_p0, e_p1 = exp_targets(stage)
    out = df.copy()
    for lab, col, exp in (("ev0", "pct_events_0_HC_neighbours", e_p0),
                          ("ev1", "pct_events_1_HC_neighbours", e_p1)):
        e = np.asarray(exp, float)
        me = float(e.mean())
        se = float(e.std(ddof=1) / np.sqrt(e.size)) if e.size > 1 else np.nan
        out["%s_exp_mean" % lab] = me
        out["%s_exp_sem" % lab] = se
        out["%s_z" % lab] = (out[col] - me) / se
    out["score2_run"] = out["ev0_z"] ** 2 + out["ev1_z"] ** 2
    return out


def report(x, y, label):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    if x.size < 4 or np.allclose(x, x[0]):
        print("    %-26s n=%-4d (not enough spread)" % (label, x.size))
        return
    r, pr = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    print("    %-26s n=%-4d  Pearson r=%+.3f (p=%.2g)   Spearman rho=%+.3f (p=%.2g)"
          % (label, x.size, r, pr, rho, ps))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="P0", choices=["P0", "E17.5"])
    ap.add_argument("--keep-collapsed", dest="keep_collapsed", action="store_true")
    ap.add_argument("--min-runs", dest="min_runs", type=int, default=8,
                    help="psigma groups smaller than this are not reported alone")
    a = ap.parse_args()

    df = pd.read_pickle(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    df = df[(df["stage"] == a.stage) & (df["error"].fillna("") == "")]
    n_all = len(df)
    if not a.keep_collapsed:
        df = df[~df["collapsed"].astype(bool)]
    df = df[df["n_differentiation_events"] > 0]
    df = per_run_score2(df, a.stage)
    df["dt"] = df["t_final"] - df["t0"]

    print("=" * 78)
    print("SCORE 2 vs RUN LENGTH  |  %s  |  %d run(s) of %d usable"
          % (a.stage, len(df), n_all))
    print("=" * 78)
    print("  per-run score 2 uses the experimental SEM only (no SEM_sim), so it is")
    print("  not on the same scale as the pooled scores.")
    print("  score2_run: median %.1f   IQR %.1f-%.1f"
          % (df["score2_run"].median(), df["score2_run"].quantile(.25),
             df["score2_run"].quantile(.75)))
    print("  t_final:    median %.2f   range %.2f-%.2f"
          % (df["t_final"].median(), df["t_final"].min(), df["t_final"].max()))
    print("  t_final-t0: median %.2f   range %.2f-%.2f"
          % (df["dt"].median(), df["dt"].min(), df["dt"].max()))

    print("\n  ACROSS ALL psigma (confounded by psigma -- see the module docstring)")
    report(df["t_final"], df["score2_run"], "t_final vs score2")
    report(df["dt"], df["score2_run"], "t_final - t0 vs score2")
    report(df["psigma"], df["score2_run"], "psigma vs score2")
    report(df["psigma"], df["t_final"], "psigma vs t_final")

    print("\n  WITHIN psigma (each group centred on its own mean) <- the answer")
    for col, lab in (("t_final", "t_final vs score2"), ("dt", "t_final - t0 vs score2")):
        cx = df[col] - df.groupby("psigma")[col].transform("mean")
        cy = df["score2_run"] - df.groupby("psigma")["score2_run"].transform("mean")
        report(cx, cy, lab)

    print("\n  PER psigma group (>= %d runs)" % a.min_runs)
    print("    %-8s %5s %10s %10s %12s %12s"
          % ("psigma", "n", "med t_end", "med score2", "r(t_final)", "r(t_final-t0)"))
    for ps, g in df.groupby("psigma"):
        if len(g) < a.min_runs:
            continue
        def rr(col):
            ok = np.isfinite(g[col]) & np.isfinite(g["score2_run"])
            if ok.sum() < 4 or np.allclose(g[col][ok], g[col][ok].iloc[0]):
                return np.nan
            return stats.pearsonr(g[col][ok], g["score2_run"][ok])[0]
        print("    %-8.3f %5d %10.2f %10.1f %12s %12s"
              % (ps, len(g), g["t_final"].median(), g["score2_run"].median(),
                 "%+.3f" % rr("t_final") if np.isfinite(rr("t_final")) else "-",
                 "%+.3f" % rr("dt") if np.isfinite(rr("dt")) else "-"))

    out = os.path.join(RESULTS_DIR, "score2_vs_time_%s.csv"
                       % a.stage.replace(".", ""))
    df[["model_name", "psigma", "initial_array", "repeat", "t0", "t_final", "dt",
        "n_differentiation_events", "pct_events_0_HC_neighbours",
        "pct_events_1_HC_neighbours", "ev0_z", "ev1_z", "score2_run",
        "steady_state", "collapsed"]].to_csv(out, index=False)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
