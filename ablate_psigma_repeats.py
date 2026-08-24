"""Single-HC ablation at two psigma values, 3 repeats per array, P0 only.

    python ablate_psigma_repeats.py --psigma 0 0.162 --dry-run
    python ablate_psigma_repeats.py --psigma 0 0.162 --workers 5

Two measurements per run set:

  1. DISTANCE of each differentiation event from the ablated cell, measured on
     the pre-ablation frame (periodic min-image), matched against that array's
     own no-ablation control.
  2. AREA-CHANGE SCORE — the HC/SC area-change ratio of the ablated cell's
     neighbours, scored with the SAME machinery as the mechanical-parameter fit
     (compare_pooled_model_mechanics_to_experiments -> "ablation_ratio").

WHY THREE SOURCES AND NOT THREE FORKS. A fork keeps the source's
lateral-inhibition state (randomize_notch_delta_levels=False) and picks the
ablated cell from a seeded RNG, so the run is DETERMINISTIC given its source:
three forks of one source would be bit-identical and would not be repeats at
all. Repeat k therefore forks the k-th independent tissue realisation
(fullmodel_v2 / _v2r2 / _v2r3, from run_psigma_repeats.py) AND offsets the
HC-choice seed, so each repeat samples a different tissue and a different
ablated cell.

CONTROLS ARE NOT OPTIONAL. Forking recomputes the lateral-inhibition length
normalisation from the loaded frame, which nudges every cell at once, so the
background differentiation rate is not zero. In the psigma=0 round the control
events were bit-identical to the ablation's and the entire signal was 2-4 extra
events at distance < 1.2 — invisible without the matched control.

The area-change protocol differs from the mechanical fit's: ONE HC on a
differentiating tissue here, versus FOUR cells on a no-differentiation run
there. The metric and the experimental target are the same; the setup is not.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, load_history_file,
                             calc_area_change_after_ablation, _hc_over_mean_sc,
                             compare_pooled_model_mechanics_to_experiments)
from run_model import run, load_sheet_from_file
from ablate_single_hc import (MECH, SHAPE_INDEX, BENDING, ATOH_SENSITIVITY,
                              TYPE_BY, THRESHOLD, BOX, pick_hc, analyse)
from run_psigma_repeats import REPEAT_PREFIX
from score_psigma_pooled import run_name

OUT_JSON = "ablate_psigma_repeats.json"
STAGE = "P0"


def source_state(stage, i, psigma, repeat):
    """(name, last time, sheet) of the run this ablation forks."""
    src = run_name(stage, psigma, REPEAT_PREFIX[repeat], i)
    history = load_history_file(src)
    from post_processing import get_time_points
    t_last = float(np.max(get_time_points(history)))
    sheet = load_sheet_from_file(os.path.join(RESULTS_DIR, src),
                                 time_point=t_last, force_periodic_box=BOX)
    sheet.geom.update_all(sheet)
    return src, t_last, sheet


def names_for(stage, i, psigma, repeat, label, control):
    """(name passed to run(), resulting folder) — psigma and repeat both tagged
    so nothing collides with the psigma=0 single-repeat set already on disk."""
    tag = "ps%.3f_r%d" % (psigma, repeat)
    suffix = "E17" if stage == "E17.5" else "P0"
    if control:
        base = "ctrlhc_%s_array%d_for_%s" % (tag, i, suffix)
        return base, base
    base = "ablhc_%s_array%d_for_%s" % (tag, i, suffix)
    return base, "%sablated_%d" % (base, label)


def one_run(args):
    stage, i, psigma, repeat, t_end, save_interval, seed, dry, control = args
    try:
        src, t_last, sheet = source_state(stage, i, psigma, repeat)
        # seed offset by repeat so each repeat ablates a DIFFERENT cell
        label, _c = pick_hc(sheet, seed + 1000 * repeat + i)
        base, folder = names_for(stage, i, psigma, repeat, label, control)
        rec = dict(stage=stage, array=i, psigma=psigma, repeat=repeat, source=src,
                   t_last=t_last, ablated_label=label, folder=folder,
                   control=control, error=None)
        if dry:
            return rec
        gammaSC, R_gamma, R_alpha, A0 = MECH[stage]
        # ORDER: run(gammaSC, gammaHC_ratio, alphaHC_ratio, psigma, ...)
        run(gammaSC, R_gamma, R_alpha, psigma,
            initial_sheet_name=src, continue_from_time=t_last,
            continue_existing_run=False,            # FORK from the steady state
            randomize_notch_delta_levels=False,     # keep its LI state
            stress_dependent=(float(psigma) != 0.0),
            ablated_cells=([] if control else [label]), name=base,
            t_end=t_end, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,
            max_wall_seconds=3600, min_progress_rate=1e-4,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX, bending=BENDING,
            quasi_static_threshold=0.03, preferred_area_override=A0,
            reuse_existing_run=True)
        return rec
    except Exception as exc:                        # noqa: BLE001
        return dict(stage=stage, array=i, psigma=psigma, repeat=repeat, folder=None,
                    control=control, error="%s: %s" % (type(exc).__name__, exc))


def area_change_ratios(rec):
    """Per-HC area-change ratio over the mean SC ratio, for one ablation run."""
    if rec.get("control") or rec.get("error") or not rec.get("folder"):
        return None
    try:
        history = load_history_file(rec["folder"])
        hc, sc = calc_area_change_after_ablation(
            history, rec["folder"], ablated_cells=[rec["ablated_label"]],
            end_time=-1, type_by=TYPE_BY, threshold=THRESHOLD)
        return _hc_over_mean_sc(hc, sc)
    except Exception as exc:                        # noqa: BLE001
        print("    area-change failed for %s: %s" % (rec["folder"][-34:],
                                                     type(exc).__name__), flush=True)
        return None


def _map_recycled(fn, tasks, workers, tasks_per_pool=None):
    """ex.map over tasks, REBUILDING the pool every tasks_per_pool tasks.

    A single long-lived ProcessPoolExecutor accumulates memory across tasks and
    then dies on whatever is scheduled last — the first attempt at this sweep
    lost 48 of 120 runs to MemoryError, every one of them in the second psigma
    block, while the first block completed untouched. Python 3.10 has no
    max_tasks_per_child, so the pool is torn down and rebuilt instead; workers
    exit and hand their memory back. Rebuild costs seconds against runs that
    take minutes.
    """
    from concurrent.futures import ProcessPoolExecutor
    if tasks_per_pool is None:
        tasks_per_pool = max(1, 3 * workers)
    out, total = [], len(tasks)
    for start in range(0, total, tasks_per_pool):
        batch = tasks[start:start + tasks_per_pool]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out.extend(ex.map(fn, batch))
        print("  -- pool %d/%d done (%d/%d tasks) --"
              % (start // tasks_per_pool + 1,
                 (total + tasks_per_pool - 1) // tasks_per_pool,
                 min(start + len(batch), total), total), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psigma", type=float, nargs="+", required=True)
    ap.add_argument("--stage", default=STAGE, choices=["E17.5", "P0"])
    ap.add_argument("--arrays", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--t-end", dest="t_end", type=float, default=5.0)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--analyse-only", dest="analyse_only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = [(a.stage, i, ps, r, a.t_end, a.save_interval, a.seed, a.dry_run, ctrl)
             for ps in a.psigma for r in a.repeats for i in a.arrays
             for ctrl in (False, True)]
    print("=" * 78)
    print("SINGLE-HC ABLATION | %s | psigma %s | repeats %s | %d run(s)"
          % (a.stage, a.psigma, a.repeats, len(tasks)))
    print("=" * 78)
    if a.dry_run:
        for t in tasks[:8]:
            r = one_run(t)
            print("  %-6s ps=%-6.3f r%d array%-2d %-8s -> %s"
                  % (r["stage"], r["psigma"], r["repeat"], r["array"],
                     "control" if r["control"] else "ablate", r.get("folder")))
        print("  ... %d total" % len(tasks))
        raise SystemExit("\n--dry-run: nothing was run.")

    if a.analyse_only:
        recs = [one_run(t[:7] + (True, t[8])) for t in tasks]
    else:
        recs = _map_recycled(one_run, tasks, a.workers)

    # ----- analysis -------------------------------------------------------
    out = []
    for rec in recs:
        if rec.get("error") or not rec.get("folder"):
            out.append(rec); continue
        try:
            pre = source_state(rec["stage"], rec["array"], rec["psigma"], rec["repeat"])
            rec = analyse(rec, pre_state=pre)
        except Exception as exc:                    # noqa: BLE001
            rec = dict(rec); rec["error"] = "analyse: %s: %s" % (type(exc).__name__, exc)
        out.append(rec)
    with open(os.path.join(RESULTS_DIR, OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("1) DIFFERENTIATION EVENTS (matched per array+repeat against its control)")
    print("=" * 78)
    for ps in a.psigma:
        new_d = []
        for r in a.repeats:
            for i in a.arrays:
                A = [x for x in out if x.get("psigma") == ps and x.get("repeat") == r
                     and x.get("array") == i and not x.get("control") and not x.get("error")]
                C = [x for x in out if x.get("psigma") == ps and x.get("repeat") == r
                     and x.get("array") == i and x.get("control") and not x.get("error")]
                if not A or not C:
                    continue
                da = sorted(round(e["distance"], 2) for e in (A[0].get("events") or []))
                dc = sorted(round(e["distance"], 2) for e in (C[0].get("events") or []))
                rem = list(dc)
                for x in da:
                    if x in rem:
                        rem.remove(x)
                    else:
                        new_d.append(x)
        new_d = np.array(new_d, float)
        n_sets = len([1 for r in a.repeats for i in a.arrays])
        print("  psigma %-7.3f  %d ablation-only event(s) over %d ablation(s) = %.2f each"
              % (ps, new_d.size, n_sets, new_d.size / max(n_sets, 1)))
        if new_d.size:
            print("      distances: min %.2f  median %.2f  max %.2f   | <=2.5: %d (%.0f%%)"
                  % (new_d.min(), np.median(new_d), new_d.max(),
                     int((new_d <= 2.5).sum()), 100 * (new_d <= 2.5).mean()))

    print("\n" + "=" * 78)
    print("2) AREA-CHANGE SCORE (same term as the mechanical fit)")
    print("=" * 78)
    for ps in a.psigma:
        vals = [v for v in (area_change_ratios(x) for x in out
                            if x.get("psigma") == ps) if v is not None and len(v)]
        if not vals:
            print("  psigma %-7.3f  no usable ablation runs" % ps); continue
        z = compare_pooled_model_mechanics_to_experiments({"ablation_ratio": vals},
                                                          a.stage)
        pooled = np.concatenate([np.asarray(v, float) for v in vals])
        print("  psigma %-7.3f  %d run(s), %d HC cell(s)  model mean %.4f   z = %+.3f"
              % (ps, len(vals), pooled.size, float(np.nanmean(pooled)),
                 z.get("ablation_ratio", float("nan"))))
    print("\nwrote %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
