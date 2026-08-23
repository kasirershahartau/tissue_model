"""Ablate one HC from a STRICT steady state — no control run needed.

    python ablate_from_strict.py --dry-run
    python ablate_from_strict.py --workers 4

WHY NO CONTROL. strict_steady_state.py identifies, per array, a time t* after
which the UNABLATED tissue produces zero differentiation events for --window
units. That event-free stretch IS the counterfactual: forking the same state and
ablating, every event inside the window is caused by the ablation. Earlier rounds
needed a matched control because the source was only loosely converged and
produced 2.43 (psigma 0) to 10.57 (psigma 0.162) background events per run.

Arrays that never reached a strict steady state are SKIPPED, not ablated from
their last frame — an ablation measured against a drifting baseline is the thing
this whole exercise exists to avoid.

Distances use the periodic minimum image and are measured on the pre-ablation
frame; cells are tracked by ``id`` because ``unique_id`` is recompacted whenever
a face is removed — which is exactly what an ablation does.

RETRYING A DEAD RUN. --only picks single points and --seed re-draws which HC is
ablated; the results JSON is merged, not overwritten, so the surviving points
stay. A new seed is not optional: the solver is deterministic, so re-running the
same draw reproduces the same dt-floor death. It does mean the retried point is
conditioned on the solver surviving that geometry, so note it if the retried
arrays end up looking different from the rest.
"""
import argparse
import json
import os

import numpy as np

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type)
from run_model import run, load_sheet_from_file
from ablate_single_hc import (MECH, SHAPE_INDEX, BENDING, ATOH_SENSITIVITY,
                              TYPE_BY, THRESHOLD, BOX, analyse)
from strict_steady_state import OUT_JSON as STRICT_JSON

OUT_JSON = "ablate_from_strict.json"


def strict_states(path=None):
    """{(stage, psigma, array): (folder, t_strict)} for arrays that converged."""
    p = path or os.path.join(RESULTS_DIR, STRICT_JSON)
    if not os.path.isfile(p):
        raise SystemExit("no strict steady states at %s — run strict_steady_state.py" % p)
    out = {}
    for r in json.load(open(p)):
        if r.get("error") or not r.get("converged"):
            continue
        out[(r["stage"], r["psigma"], r["array"])] = (r["folder"], float(r["t_strict"]))
    return out


def base_folder(stage, psigma, i):
    return "ablstrict_ps%.3f_array%d_for_%s" % (
        psigma, i, "E17" if stage == "E17.5" else "P0")


def used_labels(stage, psigma, i, records):
    """(all labels already tried, labels that produced a usable ablation).

    The two differ and both matter. Every tried label is BLOCKED from a new draw
    — including one whose run died, since re-picking it would just reproduce the
    death on the same deterministic geometry. But only the successful ones COUNT
    towards the requested number of repeats, otherwise an array that lost a run
    would quietly settle for one ablation fewer than the rest.

    Tried labels come from the folders on disk as well as the JSON, because a
    superseded attempt survives on disk after its record has been replaced."""
    mine = [r for r in records
            if r.get("stage") == stage and r.get("array") == i
            and abs(float(r.get("psigma", -1)) - psigma) < 1e-9
            and r.get("ablated_label") is not None]
    good = {int(r["ablated_label"]) for r in mine if not r.get("error")}
    tried = {int(r["ablated_label"]) for r in mine}
    prefix = base_folder(stage, psigma, i) + "ablated_"
    for d in os.listdir(RESULTS_DIR):
        if d.startswith(prefix):
            tail = d[len(prefix):].split("__")[0]
            if tail.isdigit():
                tried.add(int(tail))
    return tried, good


def pick_distinct_hcs(sheet, seed, n, exclude=()):
    """n distinct non-boundary HC labels, none of them in ``exclude``.

    Sampling WITHOUT replacement from the eligible pool is what guarantees two
    repeats of the same array never ablate the same cell — drawing independently
    and hoping they differ would collide at a rate of roughly n^2/(2*|pool|)."""
    hc_pos, _ = get_non_boundary_cell_ids_from_type(
        sheet, "HC", type_by=TYPE_BY, threshold=THRESHOLD)
    labels = [int(sheet.face_df.index.values[p]) for p in np.asarray(hc_pos)]
    pool = [l for l in labels if l not in set(exclude)]
    if len(pool) < n:
        raise RuntimeError("only %d eligible HC left, need %d" % (len(pool), n))
    chosen = np.random.RandomState(seed).choice(pool, size=n, replace=False)
    return [int(c) for c in chosen]


def one_run(args):
    stage, i, psigma, src, t_star, t_end, save_interval, seed, label, rep, dry = args
    base = base_folder(stage, psigma, i)
    rec = dict(stage=stage, array=i, psigma=psigma, source=src, t_strict=t_star,
               control=False, seed=seed, repeat=rep, error=None)
    try:
        rec.update(ablated_label=label, folder="%sablated_%d" % (base, label))
        if dry:
            return rec
        gammaSC, R_gamma, R_alpha, A0 = MECH[stage]
        run(gammaSC, R_gamma, R_alpha, psigma,
            initial_sheet_name=src, continue_from_time=t_star,
            continue_existing_run=False,           # fork AT the strict state
            randomize_notch_delta_levels=False,
            stress_dependent=(float(psigma) != 0.0),
            ablated_cells=[label], name=base,
            t_end=t_end, dt=0.01, save_interval=save_interval,
            end_on_steady_state=False,
            max_wall_seconds=3600, min_progress_rate=1e-4,
            atoh_sensitivity=ATOH_SENSITIVITY,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX, bending=BENDING,
            quasi_static_threshold=0.03, preferred_area_override=A0,
            reuse_existing_run=True)
        return rec
    except Exception as exc:                       # noqa: BLE001
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t-end", dest="t_end", type=float, default=5.0)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=24680,
                    help="seeds the draw of which HC gets ablated. Cells already "
                         "ablated for an array are excluded, so a rerun extends "
                         "the set rather than repeating it; a different seed just "
                         "reshuffles which of the remaining HCs come next.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="ablations per array. Each repeat forks the SAME strict "
                         "state and removes a DIFFERENT randomly chosen HC, so it "
                         "measures how much the response depends on which cell is "
                         "removed. Existing ablations count towards the target.")
    ap.add_argument("--only", nargs="+", default=None, metavar="PSIGMA:ARRAY",
                    help="retry just these points, e.g. --only 0.0:5 0.162:6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    strict = strict_states()
    existing = []
    out_path = os.path.join(RESULTS_DIR, OUT_JSON)
    if os.path.isfile(out_path):
        try:
            existing = json.load(open(out_path))
        except (OSError, ValueError):
            existing = []

    # Applied HERE rather than after the loop: building a task loads the strict
    # state's sheet, and loading twenty 250 MB histories to then discard
    # nineteen of them turns a two-minute rerun into a long one.
    want = None
    if a.only:
        want = set()
        for spec in a.only:
            ps_s, _sep, arr_s = spec.partition(":")
            want.add((round(float(ps_s), 6), int(arr_s)))

    tasks = []
    for (st, ps, i), (folder, t_star) in sorted(strict.items()):
        if want is not None and (round(ps, 6), i) not in want:
            continue
        tried, good = used_labels(st, ps, i, existing)
        need = max(a.repeats - len(good), 0)
        if not need:
            continue
        sheet = load_sheet_from_file(os.path.join(RESULTS_DIR, folder),
                                     time_point=t_star, force_periodic_box=BOX)
        sheet.geom.update_all(sheet)
        # one stream per (psigma, array) so a rerun reproduces the same draws
        labels = pick_distinct_hcs(sheet, a.seed + i + int(round(ps * 1e6)),
                                   need, exclude=tried)
        for k, label in enumerate(labels):
            tasks.append((st, i, ps, folder, t_star, a.t_end, a.save_interval,
                          a.seed, label, len(good) + k + 1, a.dry_run))
        print("  %-6s ps=%-6.3f array%-2d  usable %d %s%s, adding %s"
              % (st, ps, i, len(good), sorted(good),
                 "  (also blocked: %s)" % sorted(tried - good) if tried - good else "",
                 labels))
    if want is not None:
        asked = want - {(round(ps, 6), i) for (_st, ps, i) in strict}
        if asked:
            raise SystemExit("no strict steady state for %s" % sorted(asked))
    print("=" * 78)
    print("ABLATION FROM STRICT STEADY STATE | %d ablation(s) to run" % len(tasks))
    print("=" * 78)
    if not tasks:
        raise SystemExit("no array reached a strict steady state — nothing to ablate")
    if a.dry_run:
        for t in tasks[:8]:
            r = one_run(t)
            print("  %-6s ps=%-6.3f array%-2d  t*=%.1f  ablate %s"
                  % (r["stage"], r["psigma"], r["array"], r["t_strict"],
                     r.get("ablated_label")))
        print("  ... %d total" % len(tasks))
        raise SystemExit("\n--dry-run: nothing was run.")

    from concurrent.futures import ProcessPoolExecutor
    recs, per_pool = [], max(1, 3 * a.workers)
    for s in range(0, len(tasks), per_pool):        # recycled pool, see run_task_pool
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            recs.extend(ex.map(one_run, tasks[s:s + per_pool]))

    out = []
    for rec in recs:
        if rec.get("error") or not rec.get("folder"):
            out.append(rec); continue
        try:
            pre = load_sheet_from_file(os.path.join(RESULTS_DIR, rec["source"]),
                                       time_point=rec["t_strict"], force_periodic_box=BOX)
            pre.geom.update_all(pre)
            rec = analyse(rec, pre_state=(rec["source"], rec["t_strict"], pre))
        except Exception as exc:                    # noqa: BLE001
            rec = dict(rec, error="analyse: %s: %s" % (type(exc).__name__, exc))
        out.append(rec)

    # Merge into whatever is already there, keyed by (stage, psigma, array), so a
    # --only retry replaces just those points instead of throwing away the rest.
    # Keyed by the ABLATED CELL as well as the array: with several repeats per
    # array the old (stage, psigma, array) key silently kept only the last one.
    path = os.path.join(RESULTS_DIR, OUT_JSON)

    def key(r):
        return (r.get("stage"), round(float(r.get("psigma", 0)), 6), r.get("array"),
                r.get("ablated_label"))

    merged = {}
    if os.path.isfile(path):
        try:
            for r in json.load(open(path)):
                merged[key(r)] = r
        except (OSError, ValueError):
            merged = {}
    for rec in out:
        merged[key(rec)] = rec
    out = [merged[k] for k in sorted(merged, key=lambda k: (str(k[0]), k[1], k[2],
                                                            k[3] if k[3] is not None else -1))]
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print("\n" + "=" * 78)
    print("DIFFERENTIATION AFTER ABLATION FROM A STRICT STEADY STATE")
    print("  (no control needed: the unablated tissue produced zero events here)")
    print("=" * 78)
    for ps in sorted({r["psigma"] for r in out}):
        d = np.array([e["distance"] for r in out if r.get("psigma") == ps
                      and not r.get("error") for e in (r.get("events") or [])], float)
        n = len([r for r in out if r.get("psigma") == ps and not r.get("error")])
        print("  psigma %-7.3f %2d ablation(s), %3d event(s) = %.2f each"
              % (ps, n, d.size, d.size / max(n, 1)))
        if d.size:
            print("      distance: min %.2f  median %.2f  max %.2f   <=2.5: %d (%.0f%%)"
                  % (d.min(), np.median(d), d.max(),
                     int((d <= 2.5).sum()), 100 * (d <= 2.5).mean()))
    print("\nwrote %s" % os.path.join(RESULTS_DIR, OUT_JSON))


if __name__ == "__main__":
    main()
