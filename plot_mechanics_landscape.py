"""How each mechanical-fit term scores across the parameter space.

    python plot_mechanics_landscape.py
    python plot_mechanics_landscape.py --metric nsigma
    python plot_mechanics_landscape.py --only heat_roundness

Five figures, from mechanics_points.pkl (the SCORES live at the parameter point,
not the run — an n-sigma is one number over the ~10 sheets of a point):

    heat_roundness      chi^2 of the HC:SC roundness ratio, alphaHC by gammaHC/gammaSC
    heat_ablation       chi^2 of the area change near an ablation, same axes
    roundness_vs_rgamma chi^2 of roundness against gammaHC/gammaSC
    roundness_vs_rgamma_modal  the same, on one slice of the scan (modal_slice)
    ablation_vs_alpha   chi^2 of the ablation term against alphaHC
    shrinkage_vs_A0     chi^2 of the linear shrinkage against A0

alphaSC is 1 throughout the v2 fit, so alphaHC IS R_alpha; likewise
gammaHC/gammaSC is R_gamma. Both are read straight from the points table.

THE SCAN IS NOT A FILLED GRID, and the figures show that rather than hide it.
The E17.5 coupled grid moved alphaHC and gammaHC/gammaSC TOGETHER (5 x 5 points
on the diagonal alphaHC = R_gamma); the self-consistent scan then held alphaHC at
3.5 and moved R_gamma alone. So the heatmap is a diagonal plus one row, and the
blank cells are parameter pairs nobody ran, not zeros. At P0 alphaHC took only
two values (1.166 once, 1.757 everywhere else), which makes that heatmap
effectively one row wide.

WHAT "AVERAGE OVER THE OTHER PARAMETERS" MEANS HERE. Each 1-D point is the mean
over the measured points sharing that x — gammaSC, A0 and whatever else varied.
That is a mean over an UNBALANCED set, not a controlled marginal: at a given
R_gamma the E17.5 average may mix a coupled-grid point (alphaHC = R_gamma) with
self-consistent points (alphaHC = 3.5). The grey dots behind every mean show the
spread being averaged, and the whisker is its SEM.

THE STAGES ARE DRAWN SEPARATELY, always. The P0 roundness chi^2 reaches 676
where E17.5 never leaves 13, so one shared colour scale would flatten E17.5 to a
single colour. Each panel carries its own scale, and the colourbar says so.

A0 IS RECONSTRUCTED for the points whose scan did not record it. The value comes
from the point's own A0 when the scan wrote one; otherwise from the runs table,
when every run at that (stage, alphaHC, gammaSC, gammaHC) shares one A0; and for
the boundary scan, which the builder documents as running at pi/4, from the
candidate nearest pi/4. 47 of the 49 scored points resolve by the first two
rules; the two that need the third are boundary_P0 points whose gamma values the
self-consistent scan also visited at a different A0.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from post_processing import RESULTS_DIR
from plot_neighbor_pairs import STYLE, POINT_COLOUR

STAGES = ("E17.5", "P0")
TERM_LABEL = {"roundness_ratio": "HC:SC roundness ratio",
              "ablation_ratio": "HC:SC area change near ablation",
              "shrinkage": "linear shrinkage"}
# chi^2 spans six decades, so the colour scale is logarithmic and floored here;
# anything below is already a better-than-perfect match and gets the end colour
CHI2_FLOOR = 1e-2
POINT_S = 18            # the grey cloud behind each 1-D mean
A0_PI4 = np.pi / 4      # the boundary scan's documented A0


def resolve_a0(points, runs):
    """A0 per point: the scan's own value, else the runs', else the scan's default.

    The scans disagree about what A0 was: the coupled grid used the closed-form
    step-1 value and never wrote it down, the boundary scan used pi/4, and the
    self-consistent scans solved for it and did record it. Rule 2 is safe only
    when the matched runs agree on one A0 — where two scans visited the same
    gamma point at different A0 they do not, and rule 3 breaks the tie using the
    provenance build_mechanics_table documents.
    """
    key = runs.groupby(["stage", "alphaHC", "gammaSC", "gammaHC"])["A0"].agg(
        values=lambda v: sorted(set(np.round(v.dropna().to_numpy(float), 6)))
    ).reset_index()
    out, how = [], []
    for _i, row in points.iterrows():
        if np.isfinite(row.get("A0", np.nan)):
            out.append(float(row["A0"])); how.append("scan")
            continue
        gh = float(row["gammaSC"]) * float(row["R_gamma"])
        m = key[(key["stage"] == row["stage"])
                & np.isclose(key["alphaHC"], row["R_alpha"], rtol=1e-3)
                & np.isclose(key["gammaSC"], row["gammaSC"], rtol=1e-6)
                & np.isclose(key["gammaHC"], gh, rtol=2e-3)]
        cands = list(m["values"].iloc[0]) if len(m) == 1 else []
        if len(cands) == 1:
            out.append(float(cands[0])); how.append("runs")
        elif cands and str(row["scan"]).startswith("boundary"):
            best = min(cands, key=lambda v: abs(v - A0_PI4))
            out.append(float(best)); how.append("pi/4")
        else:
            out.append(np.nan); how.append("unresolved")
    return np.asarray(out, float), how


def load(results_dir=RESULTS_DIR):
    """The scored parameter points, with alphaHC, gammaHC/gammaSC and A0 named."""
    points = pd.read_pickle(os.path.join(results_dir, "mechanics_points.pkl"))
    runs = pd.read_pickle(os.path.join(results_dir, "mechanics_runs.pkl"))
    # n_sheets 0 means the point produced no usable sheet at all: its scores are
    # nan and its total_chi2 a meaningless 0. Those are not measurements.
    points = points[points["n_sheets"].fillna(0) > 0].copy()
    points["alphaHC"] = points["R_alpha"].astype(float)
    points["gamma_ratio"] = points["R_gamma"].astype(float)
    a0, how = resolve_a0(points, runs)
    points["A0_used"] = a0
    points["A0_source"] = how
    # Two scans sometimes visited the SAME parameter point and the table then
    # carries it twice with identical scores (stress_P0 and rgamma_P0 at
    # R_gamma 1.757; rgamma_P0 and boundary_P0 at 1.940). Averaging a value with
    # itself leaves the mean alone but drives the SEM to zero, so the duplicate
    # goes. Points that share gamma but differ in A0 are NOT duplicates and stay.
    before = len(points)
    # rounded: one scan wrote A0 itself and the other's came back through the
    # runs table, so the same A0 can differ in the last few digits
    points = points.assign(_a0=points["A0_used"].round(5)).drop_duplicates(
        subset=["stage", "alphaHC", "gamma_ratio", "gammaSC", "_a0"],
        keep="first").drop(columns="_a0")
    if before != len(points):
        print("  dropped %d point(s) a second scan had already recorded"
              % (before - len(points)))
    return points


def best_a0(points, term="shrinkage", metric="chi2"):
    """One A0 per (stage, alphaHC, gammaSC, gamma ratio): the best-scoring one.

    A0 was set self-consistently for each combination of the other parameters,
    so it is not a free axis to be pinned — it is a value the scan SOLVED FOR,
    and where two scans disagree the right one is the one that actually fits.
    Shrinkage is the term A0 controls, so the tie is broken on its score.

    Only two combinations are affected, both P0, and in both the boundary scan's
    pi/4 loses to the self-consistent value by two orders of magnitude
    (shrinkage chi^2 6.48 vs 0.05 at gamma ratio 5.758, 3.41 vs 0.03 at 4.645).
    Everywhere else the combination already had exactly one A0.
    """
    col = "%s_%s" % (term, metric)
    score = points[col].abs() if metric == "nsigma" else points[col]
    order = points.assign(_s=score.fillna(np.inf),
                          _a=points["alphaHC"].round(6),
                          _g=points["gammaSC"].round(6),
                          _r=points["gamma_ratio"].round(6)).sort_values("_s")
    keep = order.drop_duplicates(subset=["stage", "_a", "_g", "_r"], keep="first")
    return points.loc[points.index.isin(keep.index)]


PIN_ORDER = ("alphaHC", "gammaSC")
PIN_LABEL = {"alphaHC": r"$\alpha_{HC}$", "gammaSC": r"$\gamma_{SC}$",
             "A0_used": "$A_0$"}
MIN_SWEPT = 3          # a slice with fewer x values is not a curve


def modal_slice(sub, xcol="gamma_ratio", order=PIN_ORDER, min_swept=MIN_SWEPT):
    """Hold the nuisance parameters at their most common value, where that leaves
    a sweep. Returns (subset, {pinned: value}, [not pinned]).

    A0 IS NOT IN ``order`` and must not be: it was set self-consistently for each
    combination of the other parameters, so it is determined rather than free,
    and best_a0 resolves it before any pinning. Pinning it as well empties the
    selection outright — no (A0, alphaHC, gammaSC) triple holds more than one
    point, so the three modal values never co-occur.

    Of what remains, pin as many as keep at least ``min_swept`` distinct x
    values, most-constraining first. In practice that is alphaHC alone at E17.5
    (6 gamma ratios; pinning gammaSC too would leave 2) and alphaHC with gammaSC
    at P0 (7 gamma ratios).
    """
    cur = sub
    pinned, free = {}, []
    for col in order:
        key = cur[col].round(6)
        counts = key.value_counts()
        if counts.empty:
            free.append(col)
            continue
        cand = cur[key == counts.index[0]]
        if cand[xcol].round(6).nunique() >= min_swept:
            cur, pinned[col] = cand, float(counts.index[0])
        else:
            free.append(col)
    return cur, pinned, free


def _pin_note(pinned, free):
    held = ", ".join("%s = %g" % (PIN_LABEL[c], v) for c, v in pinned.items())
    loose = ", ".join(PIN_LABEL[c] for c in free)
    return ("held at the most common value: %s" % held if held else "nothing pinned") \
        + ("; %s still %s" % (loose, "varies" if len(free) == 1 else "vary")
           if loose else "")


def _metric(term, metric):
    return "%s_%s" % (term, metric)


def _values(g, col):
    v = g[col].to_numpy(float)
    return v[np.isfinite(v)]


def _cells(sub, col):
    """(alphaHC values, R_gamma values, mean grid, count grid) for one stage."""
    ys = np.array(sorted(sub["alphaHC"].unique()), float)
    xs = np.array(sorted(sub["gamma_ratio"].unique()), float)
    yi = {v: i for i, v in enumerate(ys)}
    xi = {v: i for i, v in enumerate(xs)}
    grid = np.full((len(ys), len(xs)), np.nan)
    count = np.zeros_like(grid, int)
    for (y, x), g in sub.groupby(["alphaHC", "gamma_ratio"]):
        v = _values(g, col)
        if v.size:
            grid[yi[y], xi[x]] = v.mean()
            count[yi[y], xi[x]] = v.size
    return ys, xs, grid, count


def heatmap(points, term, metric, out_stem, annotate=True):
    """One panel per stage: the term's score over alphaHC and gammaHC/gammaSC."""
    col = _metric(term, metric)
    fig, axes = plt.subplots(1, len(STAGES), figsize=(13.5, 5.0))
    for ax, stage in zip(np.atleast_1d(axes), STAGES):
        sub = points[points["stage"] == stage]
        ys, xs, grid, count = _cells(sub, col)
        finite = grid[np.isfinite(grid)]
        if metric == "chi2":
            pos = finite[finite > 0]
            # the floor only bites when a point fits so well its chi^2 is
            # essentially zero; on a term whose scores all sit within one decade
            # (the ablation ratio) forcing it would flatten the panel to one
            # colour, so vmin is the data's own minimum otherwise
            lo = float(pos.min()) if pos.size else CHI2_FLOOR
            hi = float(finite.max()) if finite.size else 1.0
            lo, hi = max(CHI2_FLOOR, lo), max(hi, max(CHI2_FLOOR, lo) * 1.01)
            # a log scale on a range narrower than about a decade prints ticks
            # like "1.3 x 10^0"; below that a linear scale reads better and
            # loses nothing
            wide = hi / lo > 20.0
            norm = LogNorm(vmin=lo, vmax=hi) if wide else Normalize(vmin=lo, vmax=hi)
            cmap = "viridis_r"
            extend = "min" if pos.size and pos.min() < CHI2_FLOOR else "neither"
        else:
            top = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            norm = TwoSlopeNorm(vmin=-top, vcenter=0.0, vmax=top)
            cmap, extend = "coolwarm", "neither"
        # a categorical mesh: every measured point is one cell whatever the
        # spacing of the values, and an unrun pair stays blank
        mesh = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1),
                             np.ma.masked_invalid(grid), norm=norm, cmap=cmap,
                             edgecolors="white", linewidth=0.8)
        ax.set_facecolor("0.92")
        if annotate:
            for i in range(len(ys)):
                for j in range(len(xs)):
                    if not np.isfinite(grid[i, j]):
                        continue
                    ax.text(j + 0.5, i + 0.5, "%.3g" % grid[i, j], ha="center",
                            va="center", fontsize=7.5, color="white",
                            path_effects=[pe.withStroke(linewidth=1.8,
                                                        foreground="0.15")])
        ax.set_xticks(np.arange(len(xs)) + 0.5)
        ax.set_xticklabels(["%g" % v for v in xs], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(ys)) + 0.5)
        ax.set_yticklabels(["%g" % v for v in ys], fontsize=9)
        ax.set_xlabel(r"$\gamma_{HC}/\gamma_{SC}$", fontsize=11)
        ax.set_ylabel(r"$\alpha_{HC}$", fontsize=11)
        ax.set_title("%s   (%d point(s), %d cell(s) filled of %d)"
                     % (stage, len(sub), int(np.isfinite(grid).sum()),
                        grid.size), fontsize=10)
        cb = fig.colorbar(mesh, ax=ax, extend=extend, pad=0.02)
        cb.set_label(r"$\chi^2$ (lower is better)" if metric == "chi2"
                     else "n-sigma", fontsize=10)
    fig.suptitle("%s: fit score over the mechanical parameters\n"
                 "each cell is the mean over the points measured there "
                 "(gammaSC, A0 vary within a cell); grey = never run"
                 % TERM_LABEL[term], fontsize=11.5, y=0.985, va="top",
                 linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.155,
                        wspace=0.28)
    return save(fig, out_stem)


def one_d(points, term, metric, xcol, xlabel, out_stem, logx=False, connect=True,
          modal=False):
    """One panel per stage: the term's score against one parameter.

    ``connect`` joins the means. It belongs on a genuine sweep — alphaHC and
    gammaHC/gammaSC were each moved along a line — but NOT on A0, where the
    values on the axis come from different scans that each picked their own A0
    alongside their own gammas. A line there would draw a trajectory through
    points that lie on no common path, so that figure shows markers only.

    ``modal`` first restricts each stage to its modal-parameter slice — see
    modal_slice, which also explains why all three cannot be pinned at once.
    """
    col = _metric(term, metric)
    fig, axes = plt.subplots(1, len(STAGES), figsize=(12.0, 4.6))
    for ax, stage in zip(np.atleast_1d(axes), STAGES):
        sub = points[(points["stage"] == stage) & points[xcol].notna()]
        note = ""
        if modal:
            n0 = len(sub)
            sub = best_a0(sub)
            if len(sub) != n0:
                print("    %-6s dropped %d duplicate A0 on shrinkage score"
                      % (stage, n0 - len(sub)))
            sub, pinned, free = modal_slice(sub, xcol)
            note = _pin_note(pinned, free)
            print("    %-6s %s -> %d point(s)" % (stage, note, len(sub)))
        face, edge = STYLE[(True, stage)]
        xs, ms, ss, ns = [], [], [], []
        for x, g in sub.groupby(xcol):
            v = _values(g, col)
            if not v.size:
                continue
            ax.scatter(np.full(v.size, float(x)), v, s=POINT_S, marker=".",
                       color=POINT_COLOUR, alpha=0.7, linewidths=0, zorder=2)
            xs.append(float(x)); ms.append(v.mean()); ns.append(v.size)
            ss.append(v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan)
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]; ms = np.asarray(ms)[order]
        ss = np.asarray(ss)[order]
        ax.errorbar(xs, ms, yerr=ss, marker="o", markersize=7,
                    markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.6,
                    ecolor=edge, color=edge, linewidth=1.4 if connect else 0.0,
                    linestyle="-" if connect else "none",
                    # elinewidth follows the line width unless it is given, so
                    # without this the whiskers vanish when the line is off
                    elinewidth=1.4, capsize=4, zorder=3)
        if metric == "chi2":
            # same rule as the heatmaps: log only when the scores span more
            # than about a decade, which they do for roundness and not for the
            # ablation term
            allv = _values(sub, col)
            pos = allv[allv > 0]
            if pos.size and pos.max() / max(pos.min(), CHI2_FLOOR) > 20.0:
                ax.set_yscale("log")
            ax.set_ylabel(r"$\chi^2$ (lower is better)", fontsize=10.5)
        else:
            ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
            ax.set_ylabel("n-sigma", fontsize=10.5)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title("%s   (%d point(s) at %d value(s))%s"
                     % (stage, len(sub), len(xs),
                        "\n" + note if note else ""), fontsize=10)
    second = ("one slice of the scan: $A_0$ taken at its best shrinkage score, "
              "the rest held at their most common value where that leaves a "
              "sweep (see each panel)"
              if modal else
              "each marker is the mean over the points at that value "
              "(grey dots), whisker = SEM")
    fig.suptitle("%s: fit score against %s\n%s"
                 % (TERM_LABEL[term], _plain(xlabel), second), fontsize=11.5,
                 y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.745 if modal else 0.775,
                        bottom=0.135, wspace=0.22)
    return save(fig, out_stem)


def _plain(label):
    """The mathtext axis label as something the console title can hold."""
    return (label.replace(r"$\gamma_{HC}/\gamma_{SC}$", "gammaHC/gammaSC")
            .replace(r"$\alpha_{HC}$", "alphaHC").replace("$A_0$", "A0")
            .replace("$", ""))


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=("chi2", "nsigma"), default="chi2",
                    help="chi2 (default, the score) or the signed n-sigma")
    ap.add_argument("--only", default=None,
                    help="draw one figure: heat_roundness, heat_ablation, "
                         "roundness_vs_rgamma, roundness_vs_rgamma_modal, "
                         "ablation_vs_alpha, shrinkage_vs_A0")
    ap.add_argument("--no-annotate", dest="annotate", action="store_false",
                    help="heatmaps: leave the numbers off the cells")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    a = ap.parse_args()

    points = load(a.results_dir)
    print("  %d scored parameter point(s); A0 resolved for %d"
          % (len(points), int(points["A0_used"].notna().sum())))
    print("  A0 source: %s"
          % ", ".join("%s %d" % (k, v) for k, v in
                      points["A0_source"].value_counts().items()))
    print()
    print("  %-6s %8s %8s %10s %12s %12s %12s"
          % ("stage", "alphaHC", "Rgamma", "A0", "roundness", "ablation",
             "shrinkage"))
    for _i, r in points.sort_values(["stage", "alphaHC", "gamma_ratio"]).iterrows():
        print("  %-6s %8.3f %8.4f %10s %12.4g %12.4g %12.4g"
              % (r["stage"], r["alphaHC"], r["gamma_ratio"],
                 "%.4f" % r["A0_used"] if np.isfinite(r["A0_used"]) else "-",
                 r[_metric("roundness_ratio", a.metric)],
                 r[_metric("ablation_ratio", a.metric)],
                 r[_metric("shrinkage", a.metric)]))

    tag = "" if a.metric == "chi2" else "_nsigma"
    jobs = {
        "heat_roundness": lambda: heatmap(
            points, "roundness_ratio", a.metric,
            os.path.join(a.out_dir, "mech_heat_roundness" + tag), a.annotate),
        "heat_ablation": lambda: heatmap(
            points, "ablation_ratio", a.metric,
            os.path.join(a.out_dir, "mech_heat_ablation" + tag), a.annotate),
        "roundness_vs_rgamma": lambda: one_d(
            points, "roundness_ratio", a.metric, "gamma_ratio",
            r"$\gamma_{HC}/\gamma_{SC}$",
            os.path.join(a.out_dir, "mech_roundness_vs_rgamma" + tag)),
        "roundness_vs_rgamma_modal": lambda: one_d(
            points, "roundness_ratio", a.metric, "gamma_ratio",
            r"$\gamma_{HC}/\gamma_{SC}$",
            os.path.join(a.out_dir, "mech_roundness_vs_rgamma_modal" + tag),
            modal=True),
        "ablation_vs_alpha": lambda: one_d(
            points, "ablation_ratio", a.metric, "alphaHC", r"$\alpha_{HC}$",
            os.path.join(a.out_dir, "mech_ablation_vs_alphaHC" + tag)),
        "shrinkage_vs_A0": lambda: one_d(
            points, "shrinkage", a.metric, "A0_used", "$A_0$",
            os.path.join(a.out_dir, "mech_shrinkage_vs_A0" + tag), connect=False),
    }
    if a.only:
        if a.only not in jobs:
            raise SystemExit("--only must be one of: %s" % ", ".join(jobs))
        jobs = {a.only: jobs[a.only]}
    print()
    for name, fn in jobs.items():
        print("  wrote %s.png / .svg" % fn())


if __name__ == "__main__":
    main()
