"""How many full-model runs ended in a DEAD steady state (all SC or all HC)?

    python count_dead_states.py
    python count_dead_states.py --psigma 0 0.15 0.25

A run's steady-state flag says the mechanics and the lateral-inhibition system
stopped changing. It does NOT say a pattern formed: when mechanosensitivity
suppresses delta production hard enough, differentiation never starts, the LI
system flatlines, and the run "settles" at t ~ 6 with every cell still SC. That
is indistinguishable from a healthy convergence by the flag alone, so count the
final-frame HC fraction directly.

  dead-SC   HC fraction < 1%    nothing differentiated
  dead-HC   HC fraction > 99%   everything differentiated (runaway, not seen so far)
  patterned anything in between
"""
import argparse
import os

import numpy as np

from post_processing import (RESULTS_DIR, initial_morphology_name,
                             load_history_file, get_time_points)

TH = 0.355079
def folder(psigma, stage, i, prefix="fullmodel_v2"):
    """Run folder for (psigma, stage, array) under a given repeat prefix.

    The prefix is what separates repeats of the same parameter point: the folder
    name encodes psigma, K and the array but nothing about which realisation it
    is, so repeats live under fullmodel_v2 / _v2r2 / _v2r3."""
    init = initial_morphology_name(i, stage)
    if psigma == 0:
        return "%s_%s" % (prefix, init)
    return "%s_ps%.3f_ks0.000_%s" % (prefix, psigma, init)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psigma", type=float, nargs="+",
                    default=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35])
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--run-prefix", dest="prefixes", nargs="+",
                    default=["fullmodel_v2"],
                    help="repeat prefixes to scan (missing folders are skipped)")
    a = ap.parse_args()

    print("FINAL-FRAME HC FRACTION  (HC = delta_level > %.6f)" % TH)
    print("  dead-SC: <1%   dead-HC: >99%   patterned: in between\n")
    print("  %-14s %-8s %-6s %5s %8s %8s %10s %9s   %s"
          % ("prefix", "psigma", "stage", "runs", "dead-SC", "dead-HC", "patterned",
             "t_end med", "HC fraction per run"), flush=True)
    for prefix in a.prefixes:
      for psigma in a.psigma:
        for stage in ("E17.5", "P0"):
            fr, ts = [], []
            for i in range(a.n_arrays):
                name = folder(psigma, stage, i, prefix)
                if not os.path.isdir(os.path.join(RESULTS_DIR, name)):
                    continue
                try:
                    h = load_history_file(name)
                    t = np.asarray(get_time_points(h), float)
                    s = h.retrieve(float(t[-1]))
                    s.arrange_sheet_from_history()
                    d = s.face_df["delta_level"].to_numpy(float)
                    fr.append(float((d > TH).mean()))
                    ts.append(float(t[-1]))
                except Exception as exc:                  # noqa: BLE001
                    print("     %-34s read failed: %s"
                          % (name[-34:], type(exc).__name__), flush=True)
            if not fr:
                continue
            fr = np.array(fr)
            # Judged against the ~0.31 baseline, not against zero: a run at 2-3%
            # HC is collapsed even though a "<1%" test would call it patterned.
            dsc = int((fr < 0.01).sum())
            dhc = int((fr > 0.99).sum())
            weak = int(((fr >= 0.01) & (fr < 0.10)).sum())
            print("  %-14s %-8s %-6s %5d %8d %8d %10d %9.2f   %s%s"
                  % (prefix, "%.3f" % psigma, stage, fr.size, dsc, dhc,
                     fr.size - dsc - dhc, np.median(ts),
                     " ".join("%.3f" % x for x in fr),
                     "   [%d more below 0.10]" % weak if weak else ""), flush=True)


if __name__ == "__main__":
    main()
