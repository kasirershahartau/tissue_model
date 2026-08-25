"""Probe: how much t_end does psigma=0.060 actually need to settle?

    python probe_psigma_settling.py --dry-run
    python probe_psigma_settling.py --t-end 300 --workers 4

WHY A PROBE. At t_end=100, psigma=0.060 - the best-scoring point - had 19 of 20
runs still short of steady state, and 0.075/0.090 had 20 of 20. Settle-time
medians grow faster than linearly in psigma (~27, 31, 48, 65 for psigma 0 ..
0.045), so the t_end the full sweep needs is a guess, and guessing wrong across
all 7 psigma points x 2 stages costs ~140 runs.

This extends ONLY psigma=0.060 (20 runs) and reports where they settle. That
answers two things for the price of a seventh of the sweep:

  * does psigma=0.060 settle AT ALL, or is the approach asymptotic? Strong
    gating slows differentiation, and it is entirely possible that high psigma
    never reaches the criterion - which no amount of t_end fixes, and which is
    worth discovering on 20 runs rather than 140.
  * if it does settle, at what t - which sets t_end for the real sweep.

SAVE INTERVAL. The extension is recorded at --save-interval (default 0.5, vs the
0.1 the first 100 units used). This probe only needs the steady-state marker
from debug.log, not scoring resolution, and 0.1 would add ~10 GB per run. The
archive ends up mixed-resolution (0.1 up to t=100, coarser after); scoring
handles that fine - get_time_points returns whatever was recorded, and the
best-matching frame sits at t~10, well inside the dense region.

MEMORY. Resuming reads the whole kept archive into DataFrames per worker
(_rewrite_history_for_resume), so 5 GB archives x N workers is the binding
constraint - not CPU. Keep --workers low; 4 is deliberate.
"""
import argparse
import os
import re

import numpy as np

from post_processing import RESULTS_DIR
from run_model import _psigma_tag, _classify_existing_run, _reached_steady_state
from full_model import run_full_model_arrays

MECH = {
    "E17.5": (0.2461, 1.00, 4.86, 5.72),
    "P0":    (0.2298, 1.00, 5.1487, 5.6706),
}
PREFERRED_AREA = 0.593 * np.pi / 4
STRESS_SHIFT = -0.080
PSIGMA = 0.060
_STEADY_RE = re.compile(r"steady state reached at t=([0-9.eE+-]+)")


def folder(stage, i):
    suffix = "E17" if stage == "E17.5" else "P0"
    return "fullmodel_ps%s_ks%.3f_random_periodic_array%d_for_%s" % (
        _psigma_tag(PSIGMA), STRESS_SHIFT, i, suffix)


def settled_at(stage, i):
    """Time of the LAST steady-state marker, or None if never settled."""
    d = os.path.join(RESULTS_DIR, folder(stage, i))
    if not _reached_steady_state(d):
        return None
    try:
        with open(os.path.join(d, "debug.log"), encoding="utf-8", errors="replace") as fh:
            hits = _STEADY_RE.findall(fh.read())
        return float(hits[-1]) if hits else None
    except OSError:
        return None


def report(title):
    print("\n%s\n%s" % (title, "=" * len(title)))
    print("  %-6s %-6s %-12s %-10s" % ("stage", "array", "status", "settled at"))
    times = []
    for stage in MECH:
        for i in range(10):
            d = os.path.join(RESULTS_DIR, folder(stage, i))
            if not os.path.isdir(d):
                print("  %-6s %-6d %-12s %-10s" % (stage, i, "MISSING", "-"))
                continue
            t = settled_at(stage, i)
            if t is not None:
                times.append(t)
            print("  %-6s %-6d %-12s %-10s"
                  % (stage, i, _classify_existing_run(d), "-" if t is None else "%.1f" % t))
    n = len(times)
    print("\n  settled %d/20" % n)
    if n:
        a = np.array(times)
        print("  settle times: min=%.1f median=%.1f max=%.1f" % (a.min(), np.median(a), a.max()))
    return times


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-end", dest="t_end", type=float, default=300)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    before = report("BEFORE (current state at t_end=100)")
    print("\nplan: extend the 20 psigma=%.3f runs to t_end=%g, save_interval=%g, "
          "%d workers" % (PSIGMA, a.t_end, a.save_interval, a.workers))
    print("      already-settled runs are left alone; the rest resume from their "
          "last frame")
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was run.")

    for stage, (gammaSC, alphaHC, hc_p0, sc_p0) in MECH.items():
        print("\n=== extending %s ===" % stage, flush=True)
        run_full_model_arrays(
            stage, gammaSC, alphaHC, hc_p0, sc_p0,
            atoh_sensitivity=0.355079, notch_sensitivity=0.1,
            repressor_sensitivity=0.3, bending=0.02,
            quasi_static_threshold=0.03, initial_notch_delta_level=0.01,
            preferred_area=PREFERRED_AREA, stress_shift=STRESS_SHIFT,
            save_interval=a.save_interval, psigma=PSIGMA,
            t_end=a.t_end, dt=0.01, n_arrays=10, n_workers=a.workers,
            reuse_existing_run=True)

    after = report("AFTER (t_end=%g)" % a.t_end)
    print("\n  gained %d newly-settled run(s)" % (len(after) - len(before)))
    if len(after) < 20:
        print("  %d run(s) STILL unsettled at t=%g -> the approach may be "
              "asymptotic; a larger t_end may not help." % (20 - len(after), a.t_end))
