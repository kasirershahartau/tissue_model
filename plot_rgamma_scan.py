"""Plot the step-5c R_gamma diagnostic: is HC/SC roundness gamma- or alpha-driven?

    python plot_rgamma_scan.py

Left panel is the whole argument. Two P0 sweeps at the SAME gammaSC over an
OVERLAPPING R range:

  * decoupled — R_alpha pinned at 1.757 by the stress ratio, R_gamma swept
  * coupled   — R_alpha = R_gamma (step 5)

If roundness were alpha-driven the decoupled curve would be much flatter than
the coupled one. They lie on top of each other instead, so the response belongs
to gamma almost entirely. Comparing against the E17.5 coupled grid would be
misleading here: its often-quoted slope was measured over R = 1.75..3.5 where
the response has already saturated, and against a low-R slope that comparison
would wrongly suggest alpha OPPOSES roundness.

Points that lost sheets (MemoryError, not physics) are drawn hollow and excluded
from the fits.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import (RESULTS_DIR, load_experimental_results, _finite_arrays,
                             _MECHANICS_EXPERIMENTAL_TYPE)

SCAN = "p0_rgamma_scan.json"
STEP5 = "p0_from_e17_stiffness.json"


def exp_stats(term, stage):
    e = _finite_arrays(load_experimental_results(stage, _MECHANICS_EXPERIMENTAL_TYPE[term]))
    m = np.array([a.mean() for a in e], dtype=float)
    return float(m.mean()), float(m.std(ddof=1) / np.sqrt(m.size))


def main():
    mu, sem = exp_stats("roundness_ratio", "P0")
    d = json.load(open(os.path.join(RESULTS_DIR, SCAN)))
    n_sheets = max((v.get("n_sheets_ok") or 0) for v in d["points"].values())

    full, partial = {}, {}
    for k, v in d["points"].items():
        Rg = float(k.split("Rg=")[1].split("|")[0])
        z = (v.get("z") or {}).get("roundness_ratio")
        rec = (mu + z * sem if z is not None and np.isfinite(z) else np.nan,
               v["objective"], v.get("n_sheets_ok") or 0)
        (full if rec[2] >= max(1, int(0.6 * n_sheets)) else partial)[Rg] = rec

    s5 = json.load(open(os.path.join(RESULTS_DIR, STEP5)))["points"]
    coupled = {float(k.split("=")[1]): mu + v["z"]["roundness_ratio"] * sem
               for k, v in s5.items()}

    lam, f, Ra, g = d["lambda"], d["f_HC"], d["R_alpha"], d["gamma_sc"]
    ceiling = (((1 - lam ** 2) / 8 / g) * (Ra * f + 1 - f) - (1 - f)) / f

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    xs = np.array(sorted(full)); ys = np.array([full[x][0] for x in xs])
    ax.plot(xs, ys, "o-", color="tab:blue", lw=2, ms=7,
            label="decoupled: $R_\\alpha$ pinned 1.757, $R_\\gamma$ swept")
    if partial:
        px = np.array(sorted(partial))
        ax.plot(px, [partial[x][0] for x in px], "o", mfc="none", ms=7,
                color="tab:blue", label="lost most sheets")
    cx = np.array(sorted(coupled))
    ax.plot(cx, [coupled[x] for x in cx], "s--", color="tab:orange", ms=8, mfc="none",
            lw=2, label="coupled: $R_\\alpha=R_\\gamma$ (step 5)")

    ax.axhspan(mu - sem, mu + sem, color="tab:green", alpha=0.25, zorder=0)
    ax.axhline(mu, color="tab:green", lw=1.5, zorder=1,
               label="P0 target %.4f $\\pm$ %.4f" % (mu, sem))
    ax.axvline(ceiling, color="0.4", ls=":", lw=1.5)
    ax.annotate("$A_0<\\pi/4$ ceiling\n$R_\\gamma=%.2f$" % ceiling, (ceiling, 1.055),
                ha="right", va="center", fontsize=8, color="0.3",
                xytext=(-6, 0), textcoords="offset points")

    # The response is LOGARITHMIC in R, not saturating: roundness = a + b*ln(R)
    # fits to <0.0012 over R = 1.1..1.94 here and over 1.25..5.0 on the E17.5
    # coupled grid, with the SAME b (0.127) in both. Extrapolating on a local
    # linear slope instead would badly understate the reachable roundness.
    b, c = np.polyfit(np.log(xs), ys, 1)
    need = float(np.exp((mu - c) / b))
    xf = np.linspace(xs[0], need * 1.05, 200)
    ax.plot(xf, c + b * np.log(xf), ":", color="tab:blue", lw=1.5,
            label="$a+b\\ln R_\\gamma$,  $b=%.4f$" % b)
    ax.plot([need], [mu], "*", color="tab:blue", ms=15, zorder=5)
    ax.annotate("$R_\\gamma\\approx%.1f$ at this $\\gamma_{SC}$\n"
                "(needs $\\gamma_{SC}\\!\\approx\\!0.0102$ to clear the ceiling)" % need,
                (need, mu), fontsize=8, color="tab:blue", ha="right", va="top",
                xytext=(-8, -6), textcoords="offset points")
    ax.set_xlabel("$R_\\gamma=\\gamma_{HC}/\\gamma_{SC}$")
    ax.set_ylabel("HC / SC roundness ratio")
    ax.set_title("Roundness tracks $R_\\gamma$, not $R_\\alpha$", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax2.plot(xs, [full[x][1] for x in xs], "o-", color="tab:red", lw=2, ms=7)
    ax2.axvline(ceiling, color="0.4", ls=":", lw=1.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("$R_\\gamma$")
    ax2.set_ylabel("objective  ($\\sum z^2$)")
    ax2.set_title("Objective falls monotonically towards the ceiling", fontsize=10)
    ax2.grid(alpha=0.25, which="both")
    for x in xs:
        ax2.annotate("%.0f" % full[x][1], (x, full[x][1]), fontsize=7,
                     xytext=(0, 7), textcoords="offset points", ha="center")

    fig.suptitle("Step 5c — P0, $\\gamma_{SC}=%.4g$, $R_\\alpha=%.4f$ pinned by the "
                 "measured stress ratio" % (g, Ra), y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(RESULTS_DIR, "p0_rgamma_scan.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")

    print("decoupled slope %.4f /unit R_gamma  (%.3g..%.3g)"
          % ((ys[-1] - ys[0]) / (xs[-1] - xs[0]), xs[0], xs[-1]))
    if len(cx) >= 2:
        print("coupled   slope %.4f /unit R        (%.3g..%.3g)"
              % ((coupled[cx[-1]] - coupled[cx[0]]) / (cx[-1] - cx[0]), cx[0], cx[-1]))
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
