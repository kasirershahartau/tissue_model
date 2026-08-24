"""Violin plot: distance of ablation-induced differentiation events, psigma 0 vs 0.162.

    python plot_ablation_distances.py

Each ablation run is matched against its OWN no-ablation control, forked from the
same source at the same time — so the two differ by exactly one thing, the
ablated cell, and any event present in one but not the other is caused by the
ablation.

BOTH DIRECTIONS ARE SHOWN. "ablation-only" events are ones the ablation produced
and the control did not; "control-only" are the reverse. That distinction is the
whole interpretation:

  * ablation-only >> control-only  ->  the ablation ADDS differentiation
  * ablation-only ~= control-only  ->  the ablation REDISTRIBUTES it; the same
    number of cells differentiate, in different places

At psigma = 0 the two runs are nearly identical apart from a handful of events
right beside the wound. At psigma = 0.162 they diverge across the whole sheet,
which is why the counts must be read alongside each other rather than alone.

Distances are periodic min-image, measured on the pre-ablation frame, and the
box is 20x20 — so ~14 is the largest separation possible and a median near 8 is
essentially "anywhere".
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import RESULTS_DIR

IN_JSON = "ablate_psigma_repeats.json"


def diff_events(recs, psigma):
    """(ablation-only distances, control-only distances) pooled over array+repeat."""
    abl_only, ctrl_only, n_sets = [], [], 0
    keys = {(r["array"], r["repeat"]) for r in recs
            if r.get("psigma") == psigma and not r.get("error")}
    for (i, rep) in sorted(keys):
        A = [r for r in recs if r.get("psigma") == psigma and r.get("array") == i
             and r.get("repeat") == rep and not r.get("control") and not r.get("error")]
        C = [r for r in recs if r.get("psigma") == psigma and r.get("array") == i
             and r.get("repeat") == rep and r.get("control") and not r.get("error")]
        if not A or not C:
            continue
        n_sets += 1
        da = sorted(round(e["distance"], 3) for e in (A[0].get("events") or []))
        dc = sorted(round(e["distance"], 3) for e in (C[0].get("events") or []))
        rem = list(dc)
        for x in da:
            if x in rem:
                rem.remove(x)
            else:
                abl_only.append(x)
        ctrl_only.extend(rem)          # whatever the control had and the ablation did not
    return np.array(abl_only), np.array(ctrl_only), n_sets


def all_events(recs, psigma, control=False):
    """Every differentiation event in the ablation (or control) runs, unsubtracted."""
    vals, n = [], 0
    for r in recs:
        if r.get("psigma") != psigma or r.get("error"):
            continue
        if bool(r.get("control")) != control:
            continue
        n += 1
        vals.extend(e["distance"] for e in (r.get("events") or []))
    return np.array(vals, float), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psigma", type=float, nargs="+", default=[0.0, 0.162])
    ap.add_argument("--mode", default="diff", choices=["diff", "all", "control"],
                    help="diff: events the ablation caused (control subtracted). "
                         "all: every event in the ablation runs, unsubtracted — "
                         "the raw spatial distribution of differentiation. "
                         "control: every event in the CONTROL runs, i.e. the "
                         "background with nothing ablated. Distances are still "
                         "measured from the cell that WOULD have been ablated, so "
                         "the axis means the same thing in all three modes.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(
        RESULTS_DIR, "ablation_distances%s.png"
        % {"diff": "", "all": "_all", "control": "_control"}[a.mode])

    recs = json.load(open(os.path.join(RESULTS_DIR, IN_JSON)))
    data, labels, colors, counts = [], [], [], []
    print("  %-9s %-14s %7s %8s %8s %8s %8s"
          % ("psigma", "set", "events", "per abl", "median", "<=2.5", "max"))
    for ps in a.psigma:
        if a.mode in ("all", "control"):
            ctrl = a.mode == "control"
            vv, n = all_events(recs, ps, control=ctrl)
            pairs = ((vv, "control events" if ctrl else "all events",
                      "0.45" if ctrl else "tab:red"),)
        else:
            ao, co, n = diff_events(recs, ps)
            pairs = ((ao, "ablation-only", "tab:red"), (co, "control-only", "0.6"))
        for vals, tag, col in pairs:
            data.append(vals if vals.size else np.array([np.nan]))
            labels.append("$p_\\sigma$=%.3f\n%s" % (ps, tag))
            colors.append(col)
            counts.append(vals.size)
            print("  %-9.3f %-14s %7d %8.2f %8s %8s %8s"
                  % (ps, tag, vals.size, vals.size / max(n, 1),
                     "%.2f" % np.median(vals) if vals.size else "-",
                     "%d (%.0f%%)" % ((vals <= 2.5).sum(), 100 * (vals <= 2.5).mean())
                     if vals.size else "-",
                     "%.2f" % vals.max() if vals.size else "-"))

    fig, ax = plt.subplots(figsize=(9, 5.4))
    pos = np.arange(len(data))
    ok = [i for i, d in enumerate(data) if np.isfinite(d).sum() > 1]
    if ok:
        parts = ax.violinplot([data[i] for i in ok], positions=pos[ok],
                              widths=0.75, showextrema=False, showmedians=True)
        for b, i in zip(parts["bodies"], ok):
            b.set_facecolor(colors[i]); b.set_alpha(0.45); b.set_edgecolor("k")
        parts["cmedians"].set_color("k")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        # jittered points: with n as low as 12 a violin alone is misleading
        ax.plot(i + rng.normal(0, 0.055, d.size), d, ".", ms=4, alpha=0.45,
                color=colors[i], zorder=3)
        ax.annotate("n=%d" % d.size, (i, ax.get_ylim()[1]), fontsize=9,
                    ha="center", va="top", xytext=(0, -4), textcoords="offset points")
    ax.axhline(2.5, color="tab:blue", ls="--", lw=1.3)
    ax.annotate("2.5 — first ring", (len(data) - 0.45, 2.5), fontsize=8,
                color="tab:blue", va="bottom", ha="right")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("distance from the ablated cell (pre-ablation frame)")
    ax.set_title({"diff": "Differentiation events caused by a single-HC ablation",
                  "all": "ALL differentiation events in the ablation runs",
                  "control": "Differentiation events in the CONTROL runs "
                             "(nothing ablated)"}[a.mode]
                 + " — P0, 3 repeats x 10 arrays", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
