"""Is the declared steady state actually steady? Continue settled runs and watch.

    python check_steady_state.py --dry-run
    python check_steady_state.py --workers 4

THE EVIDENCE THAT PROMPTED THIS. The single-HC ablation CONTROLS fork a run the
solver had already declared steady on both criteria, change nothing, and run 5
more time units. If the lateral-inhibition state were at a fixed point they
should produce almost no new differentiation. They produce 2.43 events per run at
psigma = 0 and 10.57 at psigma = 0.162, spread uniformly across the sheet.

Corroborating: E17.5 array 1, the one run that never tripped the flag, was moving
at |d delta|/dt = 9.7e-05 at t = 100, while array 0 AT THE MOMENT IT WAS DECLARED
STEADY was at 2.0e-03 — twenty times larger. The threshold passes runs that are
still visibly evolving.

WHY IT MATTERS. Scores 2 and 3 measure differentiation from t0 (~8.4 E17.5, ~9.2
P0) to the end of the run, and runs stop around t = 15. If differentiation is
still in progress when a run stops, those scores depend on WHERE IT HAPPENED TO
STOP — a per-run quantity with no fixed endpoint. That is systematic, not noise
that repeats average away. And because the background rate is 4.3x higher at
psigma = 0.162 than at 0, the amount of unconverged drift baked into a score
depends on psigma, so comparisons across psigma are partly comparisons of how far
each run got.

WHAT THIS DOES. Fork each source at its last frame, ablate NOTHING, run --t-end
further, and record the HC fraction and the mean |d delta|/dt over time. Read the
result as:

  * HC fraction PLATEAUS quickly      -> the criterion is roughly right and the
                                         control events are a slow tail; the
                                         effect on the scores is bounded.
  * HC fraction KEEPS CLIMBING        -> the criterion is too weak; it needs a
                                         tighter tolerance and probably a longer
                                         steady_state_min_steps, and the existing
                                         scores are endpoint-dependent.

Deliberately does NOT change the criterion — that would mean re-running
everything, and the curve should decide it first.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, load_history_file, get_time_points)
from run_model import run
from ablate_single_hc import MECH, SHAPE_INDEX, BENDING, ATOH_SENSITIVITY, TYPE_BY, THRESHOLD
from run_psigma_repeats import REPEAT_PREFIX
from score_psigma_pooled import run_name

OUT_JSON = "steady_state_check.json"
OUT_PNG = "steady_state_check.png"


def fork_name(stage, psigma, i):
    suffix = "E17" if stage == "E17.5" else "P0"
    return "sscheck_ps%.3f_array%d_for_%s" % (psigma, i, suffix)


def one_run(args):
    stage, i, psigma, t_end, save_interval, dry = args
    src = run_name(stage, psigma, REPEAT_PREFIX[1], i)
    name = fork_name(stage, psigma, i)
    rec = dict(stage=stage, array=i, psigma=psigma, source=src, folder=name, error=None)
    try:
        history = load_history_file(src)
        t_last = float(np.max(get_time_points(history)))
        rec["t_last"] = t_last
        if dry:
            return rec
        gammaSC, R_gamma, R_alpha, A0 = MECH[stage]
        run(gammaSC, R_gamma, R_alpha, psigma,
            initial_sheet_name=src, continue_from_time=t_last,
            continue_existing_run=False,           # FORK, like the ablation controls
            randomize_notch_delta_levels=False,    # keep the evolved LI state
            stress_dependent=(float(psigma) != 0.0),
            ablated_cells=[], name=name,
            t_end=t_end, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,             # run the FULL span, no early exit
            max_wall_seconds=7200, min_progress_rate=1e-4,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX, bending=BENDING,
            quasi_static_threshold=0.03, preferred_area_override=A0,
            reuse_existing_run=True)
        return rec
    except Exception as exc:                       # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        return rec


def trace(rec, every=1.0):
    """HC fraction and mean |d delta|/dt against time since the fork."""
    history = load_history_file(rec["folder"])
    stamps = np.asarray(get_time_points(history), float)
    want = np.arange(0.0, stamps[-1] + 1e-9, every)
    ts, hc, rate, prev = [], [], [], None
    for t in want:
        s = history.retrieve(float(t))
        s.arrange_sheet_from_history()
        d = s.face_df["delta_level"].to_numpy(float)
        ts.append(float(t)); hc.append(float((d > THRESHOLD).mean()))
        if prev is not None and len(d) == len(prev[1]) and t > prev[0]:
            rate.append(float(np.abs(d - prev[1]).mean() / (t - prev[0])))
        else:
            rate.append(float("nan"))
        prev = (t, d)
    return ts, hc, rate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["P0"], choices=["E17.5", "P0"])
    ap.add_argument("--psigma", type=float, nargs="+", default=[0.0, 0.162])
    ap.add_argument("--arrays", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t-end", dest="t_end", type=float, default=30.0)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", dest="analyse_only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [(st, i, ps, a.t_end, a.save_interval, a.dry_run)
             for st in a.stage for ps in a.psigma for i in a.arrays]
    print("=" * 78)
    print("STEADY-STATE CHECK | %s | psigma %s | %d fork(s), %g time units each"
          % (", ".join(a.stage), a.psigma, len(tasks), a.t_end))
    print("=" * 78)
    if a.dry_run:
        for t in tasks[:6]:
            r = one_run(t)
            print("  %-6s ps=%-6.3f array%-2d  fork %s at t=%.2f"
                  % (r["stage"], r["psigma"], r["array"], r["folder"],
                     r.get("t_last", float("nan"))))
        print("  ... %d total" % len(tasks))
        raise SystemExit("\n--dry-run: nothing was run.")

    if a.analyse_only:
        recs = [one_run(t[:5] + (True,)) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        recs, per_pool = [], max(1, 3 * a.workers)
        for s in range(0, len(tasks), per_pool):       # recycle: see run_task_pool
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                recs.extend(ex.map(one_run, tasks[s:s + per_pool]))

    out = []
    for rec in recs:
        if rec.get("error"):
            print("  %s: %s" % (rec["folder"], rec["error"])); out.append(rec); continue
        try:
            ts, hc, rate = trace(rec)
            rec = dict(rec, t=ts, hc_fraction=hc, delta_rate=rate)
        except Exception as exc:                   # noqa: BLE001
            rec = dict(rec, error="trace: %s: %s" % (type(exc).__name__, exc))
        out.append(rec)
    with open(os.path.join(RESULTS_DIR, OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("HC FRACTION DRIFT AFTER THE DECLARED STEADY STATE")
    print("=" * 78)
    print("  %-6s %-8s %6s %10s %10s %10s %12s"
          % ("stage", "psigma", "array", "HC at t=0", "at t=end", "drift", "|dd|/dt end"))
    for ps in a.psigma:
        drifts = []
        for r in out:
            if r.get("psigma") != ps or r.get("error") or "hc_fraction" not in r:
                continue
            h = r["hc_fraction"]
            drifts.append(h[-1] - h[0])
            print("  %-6s %-8.3f %6d %10.4f %10.4f %+10.4f %12.2e"
                  % (r["stage"], ps, r["array"], h[0], h[-1], h[-1] - h[0],
                     r["delta_rate"][-1]))
        if drifts:
            print("    -> psigma %.3f mean drift %+.4f over %g time units\n"
                  % (ps, float(np.mean(drifts)), a.t_end))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cols = {a.psigma[0]: "tab:blue"}
    for k, ps in enumerate(a.psigma):
        cols[ps] = ["tab:blue", "tab:red", "tab:green"][k % 3]
    for r in out:
        if r.get("error") or "hc_fraction" not in r:
            continue
        c = cols[r["psigma"]]
        ax.plot(r["t"], r["hc_fraction"], "-", color=c, alpha=0.7, lw=1.3)
        ax2.plot(r["t"], r["delta_rate"], "-", color=c, alpha=0.7, lw=1.3)
    for ps in a.psigma:
        ax.plot([], [], "-", color=cols[ps], label="$p_\\sigma$=%.3f" % ps)
    ax.set_xlabel("time since the declared steady state")
    ax.set_ylabel("HC fraction")
    ax.set_title("Does the pattern keep growing after 'steady'?", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax2.set_yscale("log")
    ax2.set_xlabel("time since the declared steady state")
    ax2.set_ylabel("mean $|\\Delta\\delta|/\\Delta t$")
    ax2.set_title("Lateral-inhibition drift rate", fontsize=10)
    ax2.grid(alpha=0.25, which="both")
    fig.suptitle("Steady-state check — forks continued with nothing ablated", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS_DIR, OUT_PNG), dpi=160, bbox_inches="tight")
    print("wrote %s and %s" % (os.path.join(RESULTS_DIR, OUT_JSON),
                               os.path.join(RESULTS_DIR, OUT_PNG)))


if __name__ == "__main__":
    main()
