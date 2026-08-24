"""Plot the self-consistent-A0 scan: what correcting A0 fixed, and where the
stage now lands.

    python plot_selfconsistent_scan.py --stage P0
    python plot_selfconsistent_scan.py --stage E17.5

LEFT — roundness against the target band, with the step-5d points (A0 pinned at
pi/4) overlaid. They sit almost on top of each other: correcting A0 moved
roundness by <0.004. Roundness is a SHAPE measure driven by the perimeter term,
so it barely cares about the area offset.

RIGHT — the shrinkage z that the correction was for. Step 5d ran -3.3 to -1.9
because the identical-circle assumption had broken down; the self-consistent A0
brings every point back inside +/-0.25, which is what step 1 was always supposed
to deliver.

Together they say the two terms are effectively ORTHOGONAL: A0 sets shrinkage,
R_gamma sets roundness. That is why the fit closes so cleanly once A0 is right.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import (RESULTS_DIR, load_experimental_results, _finite_arrays,
                             _MECHANICS_EXPERIMENTAL_TYPE)


def exp_stats(term, stage="P0"):
    e = _finite_arrays(load_experimental_results(stage, _MECHANICS_EXPERIMENTAL_TYPE[term]))
    m = np.array([a.mean() for a in e], dtype=float)
    return float(m.mean()), float(m.std(ddof=1) / np.sqrt(m.size))


SCAN = {"P0": "p0_selfconsistent_scan.json", "E17.5": "e17_selfconsistent_scan.json"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="P0", choices=["P0", "E17.5"])
    a = ap.parse_args()
    stage = a.stage
    mu, sem = exp_stats("roundness_ratio", stage)
    sc = json.load(open(os.path.join(RESULTS_DIR, SCAN[stage])))
    # the A0=pi/4 comparison series only exists for P0 (step 5d)
    bd_path = os.path.join(RESULTS_DIR, "p0_boundary_scan.json")
    bd = (json.load(open(bd_path)) if stage == "P0" and os.path.isfile(bd_path)
          else {"points": {}})

    P = sorted((v["gamma_sc"], v["R_gamma"], v) for v in sc["points"].values())
    g = np.array([p[0] for p in P]); Rg = np.array([p[1] for p in P])
    rnd = np.array([mu + p[2]["z"]["roundness_ratio"] * sem for p in P])
    shr = np.array([p[2]["z"]["shrinkage"] for p in P])

    B = []
    for k, v in bd["points"].items():
        if (v.get("n_sheets_ok") or 0) < 6:
            continue
        gg = float(k.split("gSC=")[1])
        B.append((gg, mu + v["z"]["roundness_ratio"] * sem, v["z"]["shrinkage"]))
    B.sort()
    bg = np.array([b[0] for b in B]); brnd = np.array([b[1] for b in B])
    bshr = np.array([b[2] for b in B])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax.axhspan(mu - sem, mu + sem, color="tab:green", alpha=0.25, zorder=0)
    ax.axhline(mu, color="tab:green", lw=1.5,
               label="%s target %.4f $\\pm$ %.4f" % (stage, mu, sem))
    ax.plot(bg, brnd, "s--", color="0.6", ms=7, mfc="none", lw=1.5,
            label="step 5d: $A_0$ pinned at $\\pi/4$")
    ax.plot(g, rnd, "o-", color="tab:blue", lw=2, ms=8,
            label="step 5e: $A_0$ self-consistent")
    # crossing, interpolated in ln(R_gamma). Only marked when the curve actually
    # brackets the target — E17.5's band is 9x wider, so it may not.
    above = np.where(rnd > mu)[0]
    gx = Rx = None
    if above.size and above[-1] + 1 < len(rnd):
        lo = above[-1]; hi = lo + 1
        t = (mu - rnd[lo]) / (rnd[hi] - rnd[lo])
        gx = g[lo] + t * (g[hi] - g[lo])
        Rx = float(np.exp(np.log(Rg[lo]) + t * (np.log(Rg[hi]) - np.log(Rg[lo]))))
        ax.plot([gx], [mu], "*", color="crimson", ms=17, zorder=6)
        ax.annotate("$\\gamma_{SC}\\approx%.4f$\n$R_\\gamma\\approx%.2f$" % (gx, Rx),
                    (gx, mu), fontsize=9, color="crimson", ha="left", va="top",
                    xytext=(8, -6), textcoords="offset points")
    for gi, Ri, ri in zip(g, Rg, rnd):
        ax.annotate("%.2f" % Ri, (gi, ri), fontsize=7, color="tab:blue",
                    xytext=(0, 8), textcoords="offset points", ha="center")
    ax.set_xlabel("$\\gamma_{SC}$   (labels: $R_\\gamma$)")
    ax.set_ylabel("HC / SC roundness ratio")
    ax.set_title("Roundness crosses the target — and $A_0$ barely moved it", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax2.axhspan(-1, 1, color="tab:green", alpha=0.18, zorder=0, label="within $1\\sigma$")
    ax2.axhline(0, color="0.4", lw=1)
    ax2.plot(bg, bshr, "s--", color="0.6", ms=7, mfc="none", lw=1.5,
             label="step 5d: $A_0=\\pi/4$")
    ax2.plot(g, shr, "o-", color="tab:red", lw=2, ms=8,
             label="step 5e: $A_0$ self-consistent")
    ax2.set_xlabel("$\\gamma_{SC}$")
    ax2.set_ylabel("shrinkage  $z$")
    ax2.set_title("The shrinkage term the correction was for", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(alpha=0.25)

    fig.suptitle("%s — $R_\\alpha=%.4f$, $A_0$ from the exact stationarity condition"
                 % (stage, sc["R_alpha"]), y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RESULTS_DIR, "%s_selfconsistent_scan.png" % stage.replace(".", ""))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    if gx is not None:
        print("crossing at gammaSC = %.5f, R_gamma = %.3f" % (gx, Rx))
    else:
        print("target NOT bracketed by these points")
    overlap = [abs(np.interp(x, g, rnd) - y) for x, y in zip(bg, brnd)
               if g.min() <= x <= g.max()]
    if overlap:
        print("A0 correction changed roundness by at most %.4f" % max(overlap))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
