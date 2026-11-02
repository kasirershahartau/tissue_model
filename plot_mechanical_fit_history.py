"""The six figures that accompany the mechanical-fit history document.

    python plot_mechanical_fit_history.py
    python plot_mechanical_fit_history.py --only step4_grid

Everything is read from the scan JSONs the fit itself wrote — nothing is
re-simulated and nothing is retyped:

    grid_fit_mechanics_v2_E17.5.json   step 4, the E17.5 coupled 5x5 grid
    p0_from_e17_stiffness.json         step 5, P0 with alphaHC derived
    p0_rgamma_scan.json                step 5c, the decoupling diagnostic
    p0_boundary_scan.json              step 5d, the A0 = pi/4 ceiling walk
    p0_selfconsistent_scan.json        step 5e, P0 with A0 solved
    e17_selfconsistent_scan.json       step 6, E17.5 with A0 solved

Figures, written to <results>/ as png and svg:

    fit_step4_e17_grid          objective and roundness z over (R, gammaSC)
    fit_step5c_rgamma           P0: the objective against R_gamma at fixed alpha
    fit_step5d_boundary         P0: roundness found, shrinkage lost, runs dying
    fit_selfconsistent          both stages: the three z terms against gammaSC
    fit_a0_convergence          A0 -> run -> measure -> A0, per point
    fit_progression             the best objective at each step

A NOTE ON WHAT THE POINTS MEAN. A scan point is not a run: it is 10 initial
sheets, each simulated twice (a base run and an ablation run), pooled into one
z per term. ``n_sheets_ok`` is how many of the 10 survived, and a point with 0
has an infinite objective rather than a good one — step 5d has five of those and
they are drawn as failures, not omitted.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from post_processing import RESULTS_DIR
from plot_neighbor_pairs import STYLE

SCANS = {"step4": "grid_fit_mechanics_v2_E17.5.json",
         "step5": "p0_from_e17_stiffness.json",
         "step5c": "p0_rgamma_scan.json",
         "step5d": "p0_boundary_scan.json",
         "step5e": "p0_selfconsistent_scan.json",
         "step6": "e17_selfconsistent_scan.json"}

TERMS = ("roundness_ratio", "ablation_ratio", "shrinkage")
TERM_LABEL = {"roundness_ratio": "HC:SC roundness ratio",
              "ablation_ratio": "area change near ablation",
              "shrinkage": "linear shrinkage"}
TERM_COLOUR = {"roundness_ratio": "#1f77b4",      # blue
               "ablation_ratio": "#9467bd",       # purple
               "shrinkage": "#8c564b"}            # brown
# the two points the full model actually runs on
CHOSEN = {"E17.5": dict(gamma_sc=0.0105, R_gamma=6.746125, R_alpha=3.5,
                        A0=0.741812),
          "P0": dict(gamma_sc=0.0105, R_gamma=3.850522, R_alpha=1.757,
                     A0=0.758542)}

KEY_RE = (re.compile(r"^Ra=(?P<ra>[\d.]+)\|Rg=(?P<rg>[\d.]+)\|gSC=(?P<g>[\d.]+)$"),
          re.compile(r"^R=(?P<r>[\d.]+)\|gSC=(?P<g>[\d.]+)$"),
          re.compile(r"^R=(?P<r>[\d.]+)$"))


def load_scan(name, results_dir=RESULTS_DIR):
    """One row per parameter point, with the key parsed and the z terms flat."""
    path = os.path.join(results_dir, SCANS[name])
    blob = json.load(open(path))
    rows = []
    for key, v in (blob.get("points") or {}).items():
        if not v:
            continue
        ra = rg = gsc = np.nan
        for rx in KEY_RE:
            m = rx.match(key)
            if m is None:
                continue
            d = m.groupdict()
            gsc = float(d["g"]) if "g" in d else np.nan
            if "ra" in d:
                ra, rg = float(d["ra"]), float(d["rg"])
            else:                       # the coupled forms: one R is both
                ra = rg = float(d["r"])
            break
        z = v.get("z") or {}
        row = dict(key=key, R_alpha=ra, R_gamma=rg,
                   # step 5 carries gammaSC in the value, not the key
                   gamma_sc=float(v["gamma_sc"]) if v.get("gamma_sc") else gsc,
                   objective=float(v.get("objective", np.inf)),
                   n_sheets_ok=int(v.get("n_sheets_ok") or 0),
                   A0=v.get("A0"), A0_trail=v.get("A0_trail"))
        # a point that lost every sheet has no z at all; keep the row so the
        # figure can show the failure rather than a gap
        for t in TERMS:
            row["z_" + t] = float(z.get(t)) if z.get(t) is not None else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs.update(stage=blob.get("stage"), lam=blob.get("lambda"),
                       shrinkage_pct=blob.get("shrinkage_pct"))
    return frame


def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200)
    plt.close(fig)
    return stem


def _annot(ax, x, y, text, size=7.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="white",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="0.15")])


def step4_grid(out_dir):
    """The E17.5 coupled grid: the objective, and why R is what roundness saw."""
    g = load_scan("step4")
    ys = np.array(sorted(g["R_alpha"].unique()), float)
    xs = np.array(sorted(g["gamma_sc"].unique()), float)
    yi = {v: i for i, v in enumerate(ys)}
    xi = {v: i for i, v in enumerate(xs)}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    panels = [("objective", r"objective = $\sum z^2$", "viridis_r", True),
              ("z_roundness_ratio", "roundness n-sigma", "coolwarm", False)]
    for ax, (col, label, cmap, logc) in zip(axes, panels):
        grid = np.full((len(ys), len(xs)), np.nan)
        for _i, r in g.iterrows():
            grid[yi[r["R_alpha"]], xi[r["gamma_sc"]]] = r[col]
        finite = grid[np.isfinite(grid)]
        if logc:
            norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max()))
        else:
            top = float(np.abs(finite).max())
            norm = plt.Normalize(vmin=-top, vmax=top)
        mesh = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1),
                             np.ma.masked_invalid(grid), norm=norm, cmap=cmap,
                             edgecolors="white", linewidth=0.8)
        for i in range(len(ys)):
            for j in range(len(xs)):
                if np.isfinite(grid[i, j]):
                    _annot(ax, j + 0.5, i + 0.5, "%.3g" % grid[i, j])
        ax.set_xticks(np.arange(len(xs)) + 0.5)
        ax.set_xticklabels(["%g" % v for v in xs], fontsize=9)
        ax.set_yticks(np.arange(len(ys)) + 0.5)
        ax.set_yticklabels(["%g" % v for v in ys], fontsize=9)
        ax.set_xlabel(r"$\gamma_{SC}$", fontsize=11)
        ax.set_ylabel(r"$R = \alpha_{HC} = \gamma_{HC}/\gamma_{SC}$", fontsize=11)
        ax.set_title(label, fontsize=10.5)
        fig.colorbar(mesh, ax=ax, pad=0.02)
        # the winner, and the ceiling it sits against
        best = g.loc[g["objective"].idxmin()]
        ax.add_patch(plt.Rectangle((xi[best["gamma_sc"]], yi[best["R_alpha"]]),
                                   1, 1, fill=False, edgecolor="black",
                                   linewidth=2.6, zorder=5))
    fig.suptitle("Step 4 — E17.5 coupled 5x5 grid: $\\alpha$ and $\\gamma$ moved "
                 "together\nbest R = 3.5, $\\gamma_{SC}$ = 0.0175 (boxed), "
                 "objective 4.103 — and it sits ON the $\\gamma_{SC}$ boundary",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.12,
                        wspace=0.26)
    return save(fig, os.path.join(out_dir, "fit_step4_e17_grid"))


def step5c_rgamma(out_dir):
    """P0 at fixed alpha: the objective and roundness against R_gamma."""
    g = load_scan("step5c").sort_values("R_gamma")
    face, edge = STYLE[(True, "P0")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    # both panels stay linear: the objective spans only a factor 3.4 here, so a
    # log axis would print nothing but "6 x 10^2"-style ticks
    for ax, (col, label, target) in zip(
            axes, [("objective", r"objective = $\sum z^2$", False),
                   ("z_roundness_ratio", "roundness n-sigma", True)]):
        ax.plot(g["R_gamma"], g[col], marker="o", markersize=7,
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.6,
                color=edge, linewidth=1.4, zorder=3)
        if target:
            # labelled inside the axes so the text cannot be clipped away
            ax.axhline(0.0, color="0.45", lw=1.0, linestyle="--", zorder=1)
            ax.annotate("experiment", xy=(0.02, 0.0),
                        xycoords=("axes fraction", "data"), xytext=(0, 6),
                        textcoords="offset points", fontsize=8.5, color="0.35")
        # the A0 < pi/4 ceiling, and the point step 5 had derived
        ax.axvline(1.95, color="0.3", lw=1.2, linestyle=":", zorder=1)
        ax.annotate("$A_0 < \\pi/4$ ceiling\nat $\\gamma_{SC}$ = 0.0175",
                    xy=(1.95, 0.5), xycoords=("data", "axes fraction"),
                    xytext=(-6, 0), textcoords="offset points", rotation=90,
                    ha="right", va="center", fontsize=8, color="0.3")
        ax.axvline(1.757, color=edge, lw=1.0, linestyle="-.", alpha=0.7, zorder=1)
        ax.annotate("step 5: $R_\\gamma = R_\\alpha$ = 1.757",
                    xy=(1.757, 0.06), xycoords=("data", "axes fraction"),
                    xytext=(-6, 0), textcoords="offset points", rotation=90,
                    ha="right", va="bottom", fontsize=8, color=edge)
        ax.set_xlabel(r"$\gamma_{HC}/\gamma_{SC}$", fontsize=11)
        ax.set_ylabel(label, fontsize=10.5)
    fig.suptitle("Step 5c — P0 with $\\alpha_{HC}$ pinned at 1.757 and "
                 "$\\gamma_{SC}$ at 0.0175\nroundness tracks $R_\\gamma$, so "
                 "decoupling is the fix — but the ceiling stops the sweep at "
                 "1.95, far short of the ~4 needed",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.135,
                        wspace=0.24)
    return save(fig, os.path.join(out_dir, "fit_step5c_rgamma"))


def step5d_boundary(out_dir):
    """The ceiling walk: roundness crosses, shrinkage breaks, runs die."""
    g = load_scan("step5d").sort_values("R_gamma")
    ok = g[g["n_sheets_ok"] > 0]
    dead = g[g["n_sheets_ok"] == 0]
    face, edge = STYLE[(True, "P0")]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for ax, term in zip(axes, ("roundness_ratio", "shrinkage")):
        col = "z_" + term
        ax.plot(ok["R_gamma"], ok[col], marker="o", markersize=7,
                markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.6,
                color=edge, linewidth=1.4, zorder=3)
        ax.axhline(0.0, color="0.45", lw=1.0, linestyle="--", zorder=1)
        for _i, r in dead.iterrows():
            ax.axvline(r["R_gamma"], color="0.75", lw=6, alpha=0.55, zorder=0)
        # the point with a partial sheet count is not the same kind of number
        part = ok[ok["n_sheets_ok"] < 10]
        ax.scatter(part["R_gamma"], part[col], s=150, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma_{HC}/\gamma_{SC}$  (log)", fontsize=11)
        ax.set_ylabel("%s n-sigma" % TERM_LABEL[term], fontsize=10.5)
        ax.set_title(TERM_LABEL[term], fontsize=10.5, color=TERM_COLOUR[term])
    axes[0].annotate("crosses the target\nnear $R_\\gamma \\approx 4$",
                     xy=(4.6453, 2.24), xytext=(18, -34),
                     textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    axes[1].annotate("but shrinkage collapses:\nthe step-1 'all cells are\n"
                     "circles' idealisation fails",
                     xy=(7.427, -2.98), xytext=(-165, 78),
                     textcoords="offset points", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    handles = [plt.Line2D([], [], color="0.75", lw=6, alpha=0.55),
               plt.Line2D([], [], marker="o", linestyle="none",
                          markerfacecolor="none", markeredgecolor="black",
                          markersize=11, markeredgewidth=1.6)]
    fig.legend(handles, ["all 10 sheets failed (objective infinite)",
                         "fewer than 10 sheets survived"],
               frameon=False, fontsize=9, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Step 5d — walking the $A_0 = \\pi/4$ ceiling at P0\n"
                 "the roundness crossing is found where step 5c predicted, and "
                 "everything else breaks getting there",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.775, bottom=0.215,
                        wspace=0.22)
    return save(fig, os.path.join(out_dir, "fit_step5d_boundary"))


def selfconsistent(out_dir):
    """Steps 5e and 6: the three terms against gammaSC, with the chosen point."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for ax, (stage, name) in zip(axes, (("E17.5", "step6"), ("P0", "step5e"))):
        g = load_scan(name).sort_values("gamma_sc")
        for term in TERMS:
            ax.plot(g["gamma_sc"], g["z_" + term], marker="o", markersize=6.5,
                    color=TERM_COLOUR[term], linewidth=1.5,
                    label=TERM_LABEL[term], zorder=3)
        ax.axhline(0.0, color="0.45", lw=1.0, linestyle="--", zorder=1)
        pick = CHOSEN[stage]["gamma_sc"]
        ax.axvline(pick, color=STYLE[(True, stage)][1], lw=1.6, alpha=0.8,
                   zorder=2)
        ax.annotate("chosen: $\\gamma_{SC}$ = %g\n$R_\\gamma$ = %.3f, "
                    "$A_0$ = %.5f" % (pick, CHOSEN[stage]["R_gamma"],
                                      CHOSEN[stage]["A0"]),
                    xy=(pick, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(7, 0), textcoords="offset points", va="top",
                    fontsize=8.5, color=STYLE[(True, stage)][1])
        ax.set_xlabel(r"$\gamma_{SC}$", fontsize=11)
        ax.set_title("%s   ($\\alpha_{HC}$ = %g, $A_0$ solved at every point)"
                     % (stage, CHOSEN[stage]["R_alpha"]), fontsize=10.5)
    axes[0].set_ylabel("n-sigma", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Steps 5e and 6 — $A_0$ solved self-consistently at each point\n"
                 "shrinkage is flat near zero because $A_0$ absorbs it; "
                 "$R_\\gamma$ sets roundness; nothing moves the ablation term",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.775, bottom=0.135,
                        wspace=0.12)
    return save(fig, os.path.join(out_dir, "fit_selfconsistent"))


def a0_convergence(out_dir):
    """A0 -> run -> measure -> A0, one line per point."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
    for ax, (stage, name) in zip(axes, (("E17.5", "step6"), ("P0", "step5e"))):
        g = load_scan(name).sort_values("gamma_sc")
        cmap = plt.get_cmap("viridis")
        longest = 0
        for k, (_i, r) in enumerate(g.iterrows()):
            trail = list(r["A0_trail"] or [])
            if not trail:
                continue
            longest = max(longest, len(trail))
            colour = cmap(k / max(len(g) - 1, 1))
            ax.plot(range(len(trail)), trail, marker="o", markersize=5,
                    linewidth=1.5, color=colour, zorder=3)
            # labelled at the end of its own curve: a legend here would have to
            # sit on top of the lines, and the curves are well separated
            ax.annotate(r"$\gamma_{SC}$ = %g" % r["gamma_sc"],
                        xy=(len(trail) - 1, trail[-1]), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=colour)
        ax.set_xlim(-0.15, longest - 1 + 0.95)
        ax.axhline(np.pi / 4, color="0.45", lw=1.0, linestyle="--", zorder=1)
        ax.annotate(r"$\pi/4$ = 0.7854, the seed", xy=(0.02, np.pi / 4),
                    xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.5, color="0.35")
        ax.set_xlabel("iteration", fontsize=11)
        ax.set_xticks(range(longest))
        ax.set_title("%s   (converged in %d pass(es))" % (stage, longest - 1),
                     fontsize=10.5)
    axes[0].set_ylabel("$A_0$", fontsize=11)
    fig.suptitle("The self-consistent solve for $A_0$: seed at $\\pi/4$, run, "
                 "measure the real cell areas and perimeters, repeat",
                 fontsize=11.5, y=0.97, va="top")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.135,
                        wspace=0.08)
    return save(fig, os.path.join(out_dir, "fit_a0_convergence"))


def progression(out_dir):
    """The best objective reached at each step, per stage."""
    steps = [("step4", "E17.5", "4\ncoupled grid"),
             ("step6", "E17.5", "6\n$A_0$ solved"),
             ("step5", "P0", "5\n$\\alpha$ from stress"),
             ("step5c", "P0", "5c\n$R_\\gamma$ swept"),
             ("step5d", "P0", "5d\nceiling walk"),
             ("step5e", "P0", "5e\n$A_0$ solved")]
    rows = []
    for name, stage, label in steps:
        g = load_scan(name)
        g = g[np.isfinite(g["objective"]) & (g["n_sheets_ok"] >= 10)]
        rows.append(dict(stage=stage, label=label,
                         best=float(g["objective"].min()) if len(g) else np.nan))
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    x = np.arange(len(frame))
    for stage in ("E17.5", "P0"):
        m = frame["stage"] == stage
        face, edge = STYLE[(True, stage)]
        ax.bar(x[m.to_numpy()], frame.loc[m, "best"], width=0.62, color=face,
               edgecolor=edge, linewidth=1.8, label=stage, zorder=3)
    for xi, v in zip(x, frame["best"]):
        ax.annotate("%.3g" % v, xy=(xi, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], fontsize=9.5)
    ax.set_ylabel(r"best objective at that step  ($\sum z^2$)", fontsize=10.5)
    ax.set_xlabel("step", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("How the fit came down\nonly points where all 10 sheets "
                 "survived are counted, so step 5d shows its best COMPLETE "
                 "point, not its best number",
                 fontsize=11.5, y=0.985, va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.145)
    return save(fig, os.path.join(out_dir, "fit_progression"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    jobs = {"step4_grid": step4_grid, "step5c_rgamma": step5c_rgamma,
            "step5d_boundary": step5d_boundary,
            "selfconsistent": selfconsistent, "a0_convergence": a0_convergence,
            "progression": progression}
    if a.only:
        if a.only not in jobs:
            raise SystemExit("--only must be one of: %s" % ", ".join(jobs))
        jobs = {a.only: jobs[a.only]}
    for name, fn in jobs.items():
        print("  wrote %s.png / .svg" % fn(a.out_dir))


if __name__ == "__main__":
    main()
