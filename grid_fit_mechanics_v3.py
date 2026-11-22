"""Grid fit of the mechanical parameters, scored as in SI section 7.

    python grid_fit_mechanics_v3.py --dry-run
    python grid_fit_mechanics_v3.py --stage E17.5 --workers 6
    python grid_fit_mechanics_v3.py --stage P0 --workers 6
    python grid_fit_mechanics_v3.py --score-only

THE MODEL (SI 3.4). E = sum alpha_n/2 (A_n - A0_n)^2 + Gamma_n/2 L_n^2, with
alpha_SC = 1, A0_n each cell's area in the initial array, and alpha_n, Gamma_n
interpolated between the HC and SC values by a Hill function of the cell's
delta at threshold p_D (Eqs. 16-17, exponent 5). Three free parameters per
stage: alpha~_HC, Gamma~_SC, Gamma~_HC. The grid runs on Gamma~_SC and the
ratio R_gamma = Gamma~_HC / Gamma~_SC (Gamma~_HC is reported): R_gamma < 1
would make HCs LESS round than SCs, against both stages' data, and roundness
is ~linear in ln R_gamma, so a geometric R_gamma grid covers it evenly.

PER SIMULATION (one pattern x one grid point):
  1. the pattern's mechanics-only run to steady state (median vertex speed
     < 1e-3 for 4 steps; no differentiation);
  2. a fork of its steady state in which 4 well-separated HCs -- the same 4
     at every grid point, chosen once per pattern (ablation_targets) -- are
     ablated, run until all 4 have been removed.
One record per simulation goes to ``<RESULTS_DIR>/grid_fit_mechanics_v3_<stage><tag>.jsonl``
as it finishes, so an interrupted fit keeps what it did and a re-run skips it.

SCORING (mechanics_scoring) happens only once both stages are simulated:
every (E17.5 point, P0 point) pair gets J_mech = J_E17.5 + J_P0 + chi^2_stress,
and the ranking goes to ``grid_fit_mechanics_v3_scores.json``.

THE PATTERNS. Each initial array must carry the lateral-inhibition sidecars
(notch/delta/repressor_levels.npy) and its threshold p_D (threshold.npy) --
the steady-state pattern and threshold chosen by the LI-only fit. Their
selection criteria are still to be fixed (SI 7.5, step 1), so this script
reads whatever is beside the arrays and REFUSES to run without them, rather
than falling back to a random delta.
"""
import argparse
import json
import os

import numpy as np

from history_io import RESULTS_DIR
from post_processing import initial_morphology_name
from run_model import _load_saved_threshold, _li_levels_kwargs_for_initial_sheet
import mechanics_scoring as ms

STAGES = ("E17.5", "P0")

# Coarse grid, 5 x 5 x 5 (the deferred plan's values). gammaSC stops at 0.0175:
# above ~0.018 the earlier self-consistent A0 exceeded pi/4; with per-cell A0
# the bound that matters is lambda^2 > 0, which Eq. 31 checks per run.
GRID_ALPHA_HC = [1.25, 1.75, 2.5, 3.5, 5.0]
GRID_GAMMA_SC = [0.005, 0.0075, 0.0105, 0.0145, 0.0175]
GRID_R_GAMMA = [1.5, 2.5, 4.0, 6.5, 10.0]

N_ABLATED = 4
MIN_SEPARATION_HOPS = 5
BASE_STATISTIC = "median"
BASE_THRESHOLD = 1e-3            # median vertex speed, both stages
ABLATION_STOP = "removed"
T_END_CAP = 100.0                # cap only; each run stops on its own rule
MIN_PROGRESS_RATE = 1e-4
MAX_WALL_SECONDS = 10000
PROGRESS_WINDOW_SECONDS = 30.0
RERUN_STALLED = False

RECORDS = "grid_fit_mechanics_v3_%s%s.jsonl"
TARGETS = "grid_fit_mechanics_v3_ablation_targets_%s%s.json"
SCORES = "grid_fit_mechanics_v3_scores%s.json"
# Suffix of every output file (--tag), so a verification run on a few arrays
# keeps its records apart from the production fit's.
TAG = ""


def point_key(alpha_hc, gamma_sc, r_gamma):
    return "aHC=%.4f|gSC=%.5f|Rg=%.4f" % (alpha_hc, gamma_sc, r_gamma)


def ablation_targets(stage, initials, refresh=False):
    """The ablated HCs of each pattern, chosen once and kept on disk, so every
    grid point -- in this run or a later one -- ablates the same cells."""
    from ablation_targets import ablation_hcs_for_initial, pattern_seed
    path = os.path.join(RESULTS_DIR, TARGETS % (stage, TAG))
    known = {}
    if os.path.isfile(path) and not refresh:
        with open(path) as fh:
            known = json.load(fh)
    changed = False
    for initial in initials:
        threshold = _load_saved_threshold(initial)
        rec = known.get(initial)
        # Re-choose when anything the choice depends on has changed: the
        # threshold (which cells are HCs), the seed or the separation rule.
        if (rec is not None and abs(rec["threshold"] - threshold) < 1e-12
                and rec.get("seed") == pattern_seed(initial)
                and rec.get("min_separation") == MIN_SEPARATION_HOPS):
            continue
        ids, info = ablation_hcs_for_initial(initial, threshold, n=N_ABLATED,
                                             min_separation=MIN_SEPARATION_HOPS)
        info["threshold"] = threshold
        known[initial] = info
        changed = True
    if changed:
        with open(path, "w") as fh:
            json.dump(known, fh, indent=1)
    return known


def build_task(stage, initial, alpha_hc, gamma_sc, r_gamma, ids, threshold):
    """The 25-field task _evaluate_mechanics_for_sheet expects."""
    return (float(gamma_sc), float(r_gamma), float(alpha_hc), initial, stage,
            {"ids": list(ids)}, -1, float(threshold), "delta_level",
            MAX_WALL_SECONDS, MIN_PROGRESS_RATE, PROGRESS_WINDOW_SECONDS,
            RERUN_STALLED, 0.0,
            BASE_THRESHOLD, BASE_THRESHOLD,   # the second is unused: removal stop
            None,                             # no line tension
            None, None, None,                 # no per-type p0, no bending
            None, True,                       # A0 per cell
            BASE_STATISTIC, ABLATION_STOP, T_END_CAP)


def _mean(a):
    if a is None:
        return None
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else None


def _recorded_seed(run_name):
    """The seed run() drew and recorded for ``run_name`` (None if no record)."""
    from run_model import _recorded_parameter
    if not run_name:
        return None
    return _recorded_parameter(os.path.join(RESULTS_DIR, run_name), "seed")


def record_from_terms(stage, key, params, initial, threshold, terms, task=None):
    """One JSON-able line per simulation. The per-cell arrays are reduced to
    the per-simulation value the objective uses: mean(HC / mean SC) is the
    HC/SC ratio of the means.

    Every record carries the SEEDS of its runs (read from their
    parameters.txt), so any simulation of the fit can be replayed exactly --
    including a failed one, whose base run is found by the task's own name
    (mechanics_eval.base_run_name)."""
    from mechanics_eval import base_run_name
    rec = {"stage": stage, "point": key, "initial": initial,
           "threshold": threshold, **params, "failed": terms is None}
    if terms is None:
        name = base_run_name(task) if task is not None else None
        rec.update({"base_run": name, "base_seed": _recorded_seed(name)})
        return rec
    rec.update({
        "roundness_ratio": _mean(terms.get("roundness_ratio")),
        "ablation_ratio": _mean(terms.get("ablation_ratio")),
        "n_hc_neighbours": (None if terms.get("hc_ablation") is None
                            else int(len(terms["hc_ablation"]))),
        "n_sc_neighbours": (None if terms.get("sc_ablation") is None
                            else int(len(terms["sc_ablation"]))),
        "lambda_squared": terms.get("lambda_squared"),
        "stress": terms.get("stress"),
        "ablated_removed": terms.get("ablated_removed"),
        "base_run": terms.get("base_run"),
        "ablation_run": terms.get("ablation_run"),
        "ablated_ids": terms.get("ablated_ids"),
        "ablated_labels": terms.get("ablated_labels"),
        "base_seed": _recorded_seed(terms.get("base_run")),
        "ablation_seed": _recorded_seed(terms.get("ablation_run")),
        # Everything the scores are computed FROM, so any score -- or a
        # variant of one -- can be recomputed from the records alone:
        # per-cell roundness and area change by type (with their means and
        # counts) and the sums of Eqs. 40 and 42.
        "raw": _raw_data(terms),
    })
    return rec


def _finite_list(a):
    if a is None:
        return None
    a = np.asarray(a, float)
    return [float(x) for x in a[np.isfinite(a)]]


def _raw_data(terms):
    raw = {}
    for kind, hc_key, sc_key in (("roundness", "hc_roundness", "sc_roundness"),
                                 ("area_change", "hc_ablation", "sc_ablation")):
        for cell, key in (("hc", hc_key), ("sc", sc_key)):
            vals = _finite_list(terms.get(key))
            raw["%s_%s" % (cell, kind)] = vals
            raw["n_%s_%s" % (cell, kind)] = None if vals is None else len(vals)
            raw["mean_%s_%s" % (cell, kind)] = (float(np.mean(vals))
                                                if vals else None)
    raw["energy_sums"] = terms.get("energy_sums")
    return raw


def load_records(stage):
    path = os.path.join(RESULTS_DIR, RECORDS % (stage, TAG))
    out = []
    if os.path.isfile(path):
        with open(path) as fh:
            out = [json.loads(line) for line in fh if line.strip()]
    return out


def simulate(stage, initials, grid, workers, dry_run=False):
    from grid_fit_mechanics_v2 import run_task_pool
    targets = ablation_targets(stage, initials)
    done = {(r["point"], r["initial"]) for r in load_records(stage)}
    tasks, meta = [], {}
    for alpha_hc, gamma_sc, r_gamma in grid:
        key = point_key(alpha_hc, gamma_sc, r_gamma)
        for initial in initials:
            if (key, initial) in done:
                continue
            t = targets[initial]
            task_key = "%s @ %s" % (key, initial)
            tasks.append((task_key, build_task(stage, initial, alpha_hc,
                                               gamma_sc, r_gamma, t["ids"],
                                               t["threshold"])))
            meta[task_key] = (key, initial, t["threshold"],
                              {"alphaHC": alpha_hc, "gammaSC": gamma_sc,
                               "R_gamma": r_gamma,
                               "gammaHC": gamma_sc * r_gamma},
                              tasks[-1][1])
    print("%s: %d grid point(s) x %d pattern(s); %d simulation(s) to run, "
          "%d already recorded" % (stage, len(grid), len(initials), len(tasks),
                                   len(done)))
    for initial in initials:
        t = targets[initial]
        print("   %-32s p_D=%.4f  ablate ids %s  (%d hops apart; HC/SC "
              "neighbours %s)" % (initial, t["threshold"], t["ids"],
                                  t["min_separation_hops"],
                                  t["neighbours_hc_sc"]))
    if dry_run or not tasks:
        return
    path = os.path.join(RESULTS_DIR, RECORDS % (stage, TAG))

    def handle(task_key, details, n, total):
        key, initial, threshold, params, task = meta[task_key]
        rec = record_from_terms(stage, key, params, initial, threshold, details,
                                task=task)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("  [%4d/%4d] %s %s" % (n, total, task_key,
                                     "FAILED" if rec["failed"] else "done"),
              flush=True)

    run_task_pool(tasks, workers, handle)


def summaries(stage):
    by_point = {}
    for rec in load_records(stage):
        by_point.setdefault(rec["point"], []).append(
            None if rec.get("failed") else rec)
    return {k: ms.stage_summary(v) for k, v in by_point.items()}


def score(top=10):
    targets = {s: ms.experimental_targets(s) for s in STAGES}
    points = {s: summaries(s) for s in STAGES}
    for s in STAGES:
        print("%s: %d grid point(s) recorded" % (s, len(points[s])))
    if not all(points.values()):
        raise SystemExit("both stages must be simulated before J_mech can be "
                         "formed (the stress term couples them)")
    ranked = ms.rank_pairs(points["E17.5"], points["P0"],
                           targets["E17.5"], targets["P0"])
    per_stage = {s: {k: dict(zip(("terms", "J"),
                                 ms.stage_chi2(v, targets[s])))
                     for k, v in points[s].items()} for s in STAGES}
    # Every simulation of every grid point, with its seeds, its observables
    # and its OWN score (single value, no model SEM), so a point's J can be
    # traced to the patterns that drive it.
    simulations = {}
    for s in STAGES:
        for rec in load_records(s):
            entry = {k: rec.get(k) for k in (
                "initial", "failed", "base_run", "ablation_run", "base_seed",
                "ablation_seed", "roundness_ratio", "ablation_ratio",
                "n_hc_neighbours", "n_sc_neighbours", "lambda_squared",
                "stress", "ablated_removed", "ablated_ids")}
            # the means, counts and sums only; the per-cell lists stay in
            # the JSONL records, which are the archive
            raw = rec.get("raw") or {}
            entry["raw"] = {k: v for k, v in raw.items()
                            if not isinstance(v, list)}
            # records written before seeds were recorded: read them now
            for run_key, seed_key in (("base_run", "base_seed"),
                                      ("ablation_run", "ablation_seed")):
                if entry.get(seed_key) is None:
                    entry[seed_key] = _recorded_seed(entry.get(run_key))
            if not rec.get("failed"):
                entry["score"] = ms.simulation_chi2(rec, targets[s])
            simulations.setdefault(s, {}).setdefault(rec["point"], []).append(entry)
    out = {"targets": targets, "summaries": points, "per_stage": per_stage,
           "simulations": simulations,
           "ranked_pairs": ranked[:200], "n_pairs": len(ranked)}
    path = os.path.join(RESULTS_DIR, SCORES % TAG)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nbest %d of %d pairs:" % (min(top, len(ranked)), len(ranked)))
    for r in ranked[:top]:
        print("  J=%10.4g  E17.5 %s (J %.3g)  P0 %s (J %.3g)  stress %.3g"
              % (r["J_mech"], r["E17.5"], r["J_E17.5"], r["P0"], r["J_P0"],
                 r["chi2_stress"]))
    print("wrote %s" % path)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES + ("both",), default="both")
    ap.add_argument("--n-sheets", dest="n_sheets", type=int, default=10)
    ap.add_argument("--initials", nargs="+", default=None,
                    help="initial arrays to use instead of the stage's ten "
                         "(e.g. for a verification run); one stage only")
    ap.add_argument("--workers", type=int, default=6,
                    help="6 on this box: D: is an HDD")
    ap.add_argument("--alpha-hc", dest="alpha_hc", type=float, nargs="+",
                    default=GRID_ALPHA_HC)
    ap.add_argument("--gamma-sc", dest="gamma_sc", type=float, nargs="+",
                    default=GRID_GAMMA_SC)
    ap.add_argument("--r-gamma", dest="r_gamma", type=float, nargs="+",
                    default=GRID_R_GAMMA)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for every output file, e.g. _verify")
    a = ap.parse_args()
    global TAG
    TAG = a.tag

    if a.score_only:
        score()
        return
    if any(r <= 1 for r in a.r_gamma) or any(x <= 0 for x in a.alpha_hc + a.gamma_sc):
        raise SystemExit("need R_gamma > 1 and positive alphaHC, gammaSC")
    stages = STAGES if a.stage == "both" else (a.stage,)
    if a.initials and len(stages) != 1:
        raise SystemExit("--initials needs a single --stage")
    grid = [(x, g, r) for x in a.alpha_hc for g in a.gamma_sc for r in a.r_gamma]
    for stage in stages:
        initials = a.initials or [initial_morphology_name(i, stage)
                                  for i in range(a.n_sheets)]
        for initial in initials:
            # Refuse up front: without the sidecars run() would fall back to a
            # random delta and the fit would classify cells by noise.
            if not _li_levels_kwargs_for_initial_sheet(initial):
                raise SystemExit("%s has no lateral-inhibition sidecars "
                                 "(notch/delta/repressor_levels.npy)" % initial)
            _load_saved_threshold(initial)
        simulate(stage, initials, grid, a.workers, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
