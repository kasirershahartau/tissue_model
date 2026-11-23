"""Grid fit of the mechanical parameters, scored as in SI section 7.

    python grid_fit_mechanics_v3.py --dry-run
    python grid_fit_mechanics_v3.py --stage E17.5 --workers 6
    python grid_fit_mechanics_v3.py --stage P0 --workers 6
    python grid_fit_mechanics_v3.py --score-only
    python grid_fit_mechanics_v3.py --fine --workers 6   # needs BOTH coarse grids complete
    python grid_fit_mechanics_v3.py --score-only         # ranks coarse + fine

    # re-score from the run folders, running nothing; then score that set
    python grid_fit_mechanics_v3.py --re-extract
    python grid_fit_mechanics_v3.py --score-only --tag _reextract
    # re-run one recorded simulation with its own seeds, into a results dir
    # (TISSUE_RESULTS_DIR) that holds the initial array but not the runs
    python grid_fit_mechanics_v3.py --replay "aHC=...|gSC=...|Rg=..." \
        --stage E17.5 --initial initial_array_0_for_E17 --from <fit results dir>

THE FINE GRID (decided 2026-11-22): 5 x 5 x 5 centred on each stage's point
in the best-ranked (E17.5, P0) pair OF THE COARSE POINTS -- ranked from the
records, never read from the scores file, which after a --score-only also
ranks fine points; a resume that finds a different centre than the fine grid
it started refuses (check_fine_grid_unchanged). At a quarter of the coarse
spacing in every parameter (alpha_HC and R_gamma steps 0.5; gammaSC step
gammaSC_bar/12 of the centre's own pair). A centre at the lowest coarse value
gets its window shifted inward (e.g. alpha_HC 1.5-3.5 around 2), never past
a lower bound. Its derivation goes to
grid_fit_mechanics_v3_finegrid_<stage>.json; its records join the coarse ones,
so the final ranking covers both.

THE MODEL (SI 3.4). E = sum alpha_n/2 (A_n - A0_n)^2 + Gamma_n/2 L_n^2, with
alpha_SC = 1, A0_n each cell's area in the initial array, and alpha_n, Gamma_n
interpolated between the HC and SC values by a Hill function of the cell's
delta at threshold p_D (Eqs. 16-17, exponent 5). Three free parameters per
stage: alpha~_HC, Gamma~_SC, Gamma~_HC. The grid runs on Gamma~_SC and the
ratio R_gamma = Gamma~_HC / Gamma~_SC (Gamma~_HC is reported): R_gamma < 1
would make HCs LESS round than SCs, against both stages' data.

THE COARSE GRID (decided 2026-11-22): alpha_HC and R_gamma in {2, 4, 6, 8,
10}; gammaSC at five values evenly spaced between 0 and twice gammaSC_bar,
the value Eq. 40 gives when the experimental mean shrinkage is plugged in with
the initial patterns' data (see :func:`gamma_sc_estimate`).

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

# Coarse grid, 5 x 5 x 5. gammaSC is not listed: it is k/6 * 2 * gammaSC_bar
# for k = 1..5 -- five values evenly spaced strictly inside (0, 2 gammaSC_bar),
# centred on gammaSC_bar -- with gammaSC_bar from Eq. 40 (gamma_sc_estimate).
GRID_ALPHA_HC = [2.0, 4.0, 6.0, 8.0, 10.0]
GRID_R_GAMMA = [2.0, 4.0, 6.0, 8.0, 10.0]
GAMMA_SC_FRACTIONS = [1 / 3, 2 / 3, 1.0, 4 / 3, 5 / 3]
# gammaSC values are rounded to this many decimals before use, so the value
# simulated, the one in the record and the one in the folder name (%.4f, and
# %.4f in its hash) can never disagree about which point a run belongs to.
GAMMA_SC_DECIMALS = 5
GRID = "grid_fit_mechanics_v3_grid_%s%s.json"
FINE_GRID = "grid_fit_mechanics_v3_finegrid_%s%s.json"
# The FINE grid: 5 x 5 x 5 centred on the best coarse point (the top-ranked
# (E17.5, P0) pair, each stage's own point), at a QUARTER of the coarse
# spacing in every parameter: centre + {-2, -1, 0, 1, 2} fine steps, so it
# spans the coarse cell on either side of the centre.
FINE_OFFSETS = [-2, -1, 0, 1, 2]
FINE_REFINEMENT = 4

N_ABLATED = 4
MIN_SEPARATION_HOPS = 5
BASE_STATISTIC = "median"
BASE_THRESHOLD = 1e-3            # median vertex speed, both stages
ABLATION_STOP = "removed"
T_END_CAP = 100.0                # cap only; each run stops on its own rule
MIN_PROGRESS_RATE = 1e-4
# Per-run wall-clock cap. A stalling parameter corner once used 37% of a whole
# fit (6.8 h per run); a run at the phase-1 point takes ~6-7 min here, so a
# run still going after 25 min is scored as failed rather than waited for.
MAX_WALL_SECONDS = 1500
PROGRESS_WINDOW_SECONDS = 30.0
RERUN_STALLED = False

RECORDS = "grid_fit_mechanics_v3_%s%s.jsonl"
TARGETS = "grid_fit_mechanics_v3_ablation_targets_%s%s.json"
SCORES = "grid_fit_mechanics_v3_scores%s.json"
# Suffix of every output file (--tag), so a verification run on a few arrays
# keeps its records apart from the production fit's.
TAG = ""


def gamma_sc_estimate(stage, initials, alpha_hc, r_gamma, lam=None):
    """gammaSC from Eq. 40 at the experimental mean shrinkage, on the
    initial patterns.

    On an initial pattern every cell is at its own preferred area (A0_n is
    the cell's initial area), so Eq. 40 with lambda = lambda_exp solves to

        gammaSC = 2 (1 - lambda_exp^2) sum(alpha_n A_n^2) / sum(g_n L_n^2),

    alpha_n = 1 + (alpha_HC - 1) w_n,  g_n = 1 + (R_gamma - 1) w_n,  with w_n
    the Hill weight of Eqs. 24-25 (delta at the pattern's p_D, exponent
    HILL_EXPONENT). Returns ``(mean over patterns, per-pattern values)``.
    """
    from lateral_inhibition_model import HILL_EXPONENT
    from post_processing import load_history_file
    if lam is None:
        lam = ms.experimental_targets(stage)["shrinkage"][0]
    out = []
    for initial in initials:
        h = load_history_file(initial)
        sheet = h.retrieve(float(np.min(h.time_stamps)))
        sheet.arrange_sheet_from_history()
        sheet.geom.update_all(sheet)
        fd = sheet.face_df
        delta = np.load(os.path.join(RESULTS_DIR, initial, "delta_levels.npy"))
        d = delta[fd["unique_id"].to_numpy(int)]
        p_d = _load_saved_threshold(initial)
        w = d ** HILL_EXPONENT / (p_d ** HILL_EXPONENT + d ** HILL_EXPONENT)
        out.append(gamma_sc_for_shrinkage(
            fd["area"].to_numpy(float), fd["perimeter"].to_numpy(float), w,
            alpha_hc, r_gamma, lam))
    return float(np.mean(out)), out


def gamma_sc_for_shrinkage(A, L, w, alpha_hc, r_gamma, lam):
    """The gammaSC at which Eq. 40 gives ``lam`` on cells at their preferred
    areas (A0 = A), HC weights ``w``."""
    alpha = 1 + (alpha_hc - 1) * np.asarray(w, float)
    g = 1 + (r_gamma - 1) * np.asarray(w, float)
    A, L = np.asarray(A, float), np.asarray(L, float)
    return float(2 * (1 - lam ** 2) * np.sum(alpha * A ** 2) / np.sum(g * L ** 2))


def build_grid(stage, initials, alpha_values=None, r_values=None,
               per_point=True):
    """The coarse grid for ``stage`` and its provenance.

    ``per_point`` (default) centres the five gammaSC values on the
    gammaSC_bar of each (alpha_HC, R_gamma); otherwise on ONE gammaSC_bar,
    the mean over all 25 of them, giving a rectangular box.
    """
    alpha_values = alpha_values or GRID_ALPHA_HC
    r_values = r_values or GRID_R_GAMMA
    lam = ms.experimental_targets(stage)["shrinkage"][0]
    bars = {(a, r): gamma_sc_estimate(stage, initials, a, r, lam)
            for a in alpha_values for r in r_values}
    pooled = float(np.mean([m for m, _ in bars.values()]))
    grid, rows = [], []
    for (a, r), (bar, per) in sorted(bars.items()):
        centre = bar if per_point else pooled
        for f in GAMMA_SC_FRACTIONS:
            grid.append((a, round(f * centre, GAMMA_SC_DECIMALS), r))
        rows.append({"alphaHC": a, "R_gamma": r, "gammaSC_bar": bar,
                     "gammaSC_bar_per_pattern": per})
    info = {"stage": stage, "lambda_exp": lam, "per_point": per_point,
            "pooled_gammaSC_bar": pooled, "fractions": GAMMA_SC_FRACTIONS,
            "initials": list(initials), "bars": rows,
            "points": [{"alphaHC": a, "gammaSC": g, "R_gamma": r}
                       for a, g, r in grid]}
    return grid, info


def coarse_spacing(stage, centre):
    """The coarse grid's step in each parameter at ``centre``:
    (alpha_HC step, gammaSC step, R_gamma step). The gammaSC step is the one
    of the centre's own (alpha_HC, R_gamma) pair -- gammaSC_bar / 3 -- read
    from the coarse grid's record, since it differs from pair to pair."""
    a, g, r = centre
    with open(os.path.join(RESULTS_DIR, GRID % (stage, TAG))) as fh:
        info = json.load(fh)
    alphas = sorted({row["alphaHC"] for row in info["bars"]})
    rs = sorted({row["R_gamma"] for row in info["bars"]})
    bar = next(row["gammaSC_bar"] for row in info["bars"]
               if abs(row["alphaHC"] - a) < 1e-9 and abs(row["R_gamma"] - r) < 1e-9)
    centre_bar = info["pooled_gammaSC_bar"] if not info["per_point"] else bar
    f = info["fractions"]
    return (float(np.diff(alphas).min()), float((f[1] - f[0]) * centre_bar),
            float(np.diff(rs).min()))


def _fine_offsets(centre, step, lower):
    """FINE_OFFSETS, shifted up by the fewest whole steps that keep every
    value strictly above ``lower`` -- for a centre at the lowest coarse value
    (alpha_HC or R_gamma = 2, whose -2 step would land on 1). Same spacing,
    still 5 points, just off-centre."""
    shift = 0
    while centre + (FINE_OFFSETS[0] + shift) * step <= lower + 1e-12:
        shift += 1
        if shift > len(FINE_OFFSETS):
            raise SystemExit("cannot place a fine grid above %g around %g"
                             % (lower, centre))
    return [o + shift for o in FINE_OFFSETS]


def build_fine_grid(stage, centre):
    """5 x 5 x 5 around ``centre`` = (alpha_HC, gammaSC, R_gamma) at a
    quarter of the coarse spacing. Where the centred window would cross a
    lower bound (alpha_HC, R_gamma > 1; gammaSC > 0) it is shifted inward
    (_fine_offsets), parameter by parameter."""
    steps = coarse_spacing(stage, centre)
    fine = [s / FINE_REFINEMENT for s in steps]
    lowers = (1.0, 0.0, 1.0)
    offsets = [_fine_offsets(c, st, lo) for c, st, lo in zip(centre, fine, lowers)]
    grid = []
    for i in offsets[0]:
        for j in offsets[1]:
            for k in offsets[2]:
                grid.append((round(centre[0] + i * fine[0], 6),
                             round(centre[1] + j * fine[1], GAMMA_SC_DECIMALS),
                             round(centre[2] + k * fine[2], 6)))
    names = ("alphaHC", "gammaSC", "R_gamma")
    info = {"stage": stage, "centre": dict(zip(names, centre)),
            "coarse_steps": dict(zip(names, steps)),
            "fine_steps": dict(zip(names, fine)),
            "offsets": dict(zip(names, offsets)),
            "points": [{"alphaHC": a, "gammaSC": g_, "R_gamma": r}
                       for a, g_, r in grid]}
    return grid, info


def coarse_point_keys(stage):
    """The coarse grid's point keys and patterns, from its own record."""
    with open(os.path.join(RESULTS_DIR, GRID % (stage, TAG))) as fh:
        info = json.load(fh)
    keys = [point_key(p["alphaHC"], p["gammaSC"], p["R_gamma"])
            for p in info["points"]]
    return keys, list(info["initials"])


def best_coarse_point(stage):
    """This stage's point in the best (E17.5, P0) pair of the COARSE grid.

    Ranked here, from the records, over the coarse points only -- never read
    from the scores file. The fine records go into the same JSONL as the
    coarse ones, so a ranking that included them (a --score-only run during
    the fine phase, say) would move the centre and give a different fine
    grid on resume. And it refuses until BOTH stages' coarse grids are
    complete (every point x pattern recorded, failures included), since a
    centre chosen from a partial grid could change once the rest arrived.
    """
    targets = {s: ms.experimental_targets(s) for s in STAGES}
    points = {}
    for s in STAGES:
        keys, initials = coarse_point_keys(s)
        records = load_records(s)
        have = {(r["point"], r["initial"]) for r in records}
        missing = sum((k, i) not in have for k in keys for i in initials)
        if missing:
            raise SystemExit(
                "%s coarse grid incomplete: %d of %d simulations recorded; the "
                "fine grid is centred only on a complete coarse grid"
                % (s, len(keys) * len(initials) - missing,
                   len(keys) * len(initials)))
        points[s] = summaries(s, keys=set(keys))
    best = ms.rank_pairs(points["E17.5"], points["P0"],
                         targets["E17.5"], targets["P0"])[0]
    best = {k: best[k] for k in ("E17.5", "P0", "J_mech")}
    a, g, r = (float(x.split("=")[1]) for x in best[stage].split("|"))
    return (a, g, r), best


def check_fine_grid_unchanged(stage, info):
    """On a resume the fine grid must be the one already started. The coarse
    grid is complete and unchanged by then, so its centre cannot have moved;
    if it has, something rewrote the coarse records, and the fit stops rather
    than start a second fine grid beside the first."""
    path = os.path.join(RESULTS_DIR, FINE_GRID % (stage, TAG))
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        old = json.load(fh)
    if old["points"] != json.loads(json.dumps(info["points"])):
        raise SystemExit(
            "%s: the fine grid would now centre on %s, but %s was built around "
            "%s; the coarse records have changed. Move that file aside to "
            "build a new fine grid." % (stage, info["centre"], path,
                                        old["centre"]))


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


def build_task(stage, initial, alpha_hc, gamma_sc, r_gamma, ids, threshold,
               seeds=None):
    """The 25-field task _evaluate_mechanics_for_sheet expects; with
    ``seeds`` ({"base": int, "ablation": int}) a 26th field that replays a
    recorded simulation with its own seeds."""
    task = _task_fields(stage, initial, alpha_hc, gamma_sc, r_gamma, ids,
                        threshold)
    return task if seeds is None else task + (dict(seeds),)


def _task_fields(stage, initial, alpha_hc, gamma_sc, r_gamma, ids, threshold):
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
    # Two grid points must never share a run folder: the folder name holds
    # gammaSC at 4 decimals and alpha/R at 2, so a too-fine grid would make
    # reuse_existing_run hand one point's runs to another. Refuse it.
    from mechanics_eval import base_run_name
    names = {}
    for task_key, task in tasks:
        names.setdefault(base_run_name(task), []).append(task_key)
    clashes = {n: k for n, k in names.items() if len(k) > 1}
    if clashes:
        raise SystemExit("grid points share run folders: %s"
                         % list(clashes.values())[:3])
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


# What a re-extraction or a replay must reproduce: the observables, the cells,
# the seeds and the raw data the score is computed from.
COMPARED_FIELDS = ("failed", "roundness_ratio", "ablation_ratio",
                   "lambda_squared", "stress", "n_hc_neighbours",
                   "n_sc_neighbours", "ablated_removed", "ablated_ids",
                   "ablated_labels", "base_seed", "ablation_seed", "raw")
REEXTRACT_TAG = "_reextract"
REPLAY_TAG = "_replay"


def differing_fields(original, new):
    """The COMPARED_FIELDS on which two records of one simulation disagree,
    compared exactly (as JSON). A field the original does not carry (a
    record written before it existed) is not compared."""
    return [k for k in COMPARED_FIELDS if k in original
            and json.dumps(original.get(k), sort_keys=True)
            != json.dumps(new.get(k), sort_keys=True)]


def re_extract(stage):
    """Re-measure every recorded simulation of ``stage`` from its run
    folders, through the extraction code that scored it, WITHOUT running
    anything; write the records to the ``_reextract`` file and report any
    that differ from the originals. Failed records carry over unchanged:
    nothing was measured for them. Score the result with
    ``--score-only --tag <tag>_reextract``."""
    from mechanics_eval import terms_from_runs
    out = os.path.join(RESULTS_DIR, RECORDS % (stage, TAG + REEXTRACT_TAG))
    if os.path.isfile(out):
        raise SystemExit("%s exists; delete it to re-extract" % out)
    records = load_records(stage)
    for rec in records:
        for run in (rec.get("base_run"), rec.get("ablation_run")):
            if (not rec.get("failed") and run and not os.path.isfile(
                    os.path.join(RESULTS_DIR, run, "history.hf5"))):
                raise SystemExit("run %s (%s @ %s) is not in %s"
                                 % (run, rec["point"], rec["initial"],
                                    RESULTS_DIR))
    n_diff = 0
    with open(out, "w") as fh:
        for n, rec in enumerate(records, 1):
            if rec.get("failed"):
                new = dict(rec)
            else:
                terms = terms_from_runs(
                    rec["base_run"], rec.get("ablation_run"),
                    rec.get("ablated_labels"), rec.get("ablated_ids"),
                    type_by="delta_level", threshold=rec["threshold"],
                    post_ablation_frame=-1,
                    require_ablated_removed=ABLATION_STOP == "removed")
                params = {k: rec[k] for k in ("alphaHC", "gammaSC", "R_gamma",
                                              "gammaHC")}
                new = record_from_terms(stage, rec["point"], params,
                                        rec["initial"], rec["threshold"], terms)
            fh.write(json.dumps(new) + "\n")
            diff = differing_fields(rec, new)
            n_diff += bool(diff)
            print("  [%4d/%4d] %s @ %s %s" % (n, len(records), rec["point"],
                                             rec["initial"],
                                             "DIFFERS in %s" % diff if diff
                                             else "identical"), flush=True)
    print("%s: %d record(s) re-extracted, %d differ; wrote %s"
          % (stage, len(records), n_diff, out))
    return n_diff


def replay(stage, key, initial, source_dir):
    """Re-run ONE recorded simulation with its recorded seeds and compare.

    ``source_dir`` holds the fit's records (and its ablation targets); the
    replay runs into RESULTS_DIR, which must hold the initial array but NOT
    the original runs: the seed is not part of the folder name, so an
    existing folder would be reused rather than re-simulated."""
    from mechanics_eval import _evaluate_mechanics_for_sheet, base_run_name
    path = os.path.join(source_dir, RECORDS % (stage, TAG))
    with open(path) as fh:
        found = [r for r in (json.loads(l) for l in fh if l.strip())
                 if r["point"] == key and r["initial"] == initial]
    if len(found) != 1:
        raise SystemExit("%d record(s) of %s @ %s in %s"
                         % (len(found), key, initial, path))
    rec = found[0]
    seeds = {"base": rec.get("base_seed"), "ablation": rec.get("ablation_seed")}
    if seeds["base"] is None:
        raise SystemExit("the record has no base seed; it cannot be replayed")
    ids = rec.get("ablated_ids")
    if ids is None:
        # a failed record names no cells: they are the pattern's targets
        with open(os.path.join(source_dir, TARGETS % (stage, TAG))) as fh:
            ids = json.load(fh)[initial]["ids"]
    if not os.path.isdir(os.path.join(RESULTS_DIR, initial)):
        raise SystemExit("copy or link the initial array %s into %s first"
                         % (initial, RESULTS_DIR))
    task = build_task(stage, initial, rec["alphaHC"], rec["gammaSC"],
                      rec["R_gamma"], ids, rec["threshold"], seeds=seeds)
    name = base_run_name(task)
    if os.path.exists(os.path.join(RESULTS_DIR, name)):
        raise SystemExit("%s already holds %s: a replay needs a results dir "
                         "without the original runs" % (RESULTS_DIR, name))
    print("replaying %s @ %s (%s) with seeds %s" % (key, initial, stage, seeds),
          flush=True)
    terms = _evaluate_mechanics_for_sheet(task)
    params = {k: rec[k] for k in ("alphaHC", "gammaSC", "R_gamma", "gammaHC")}
    new = record_from_terms(stage, key, params, initial, rec["threshold"],
                            terms, task=task)
    with open(os.path.join(RESULTS_DIR, RECORDS % (stage, TAG + REPLAY_TAG)),
              "a") as fh:
        fh.write(json.dumps(new) + "\n")
    diff = differing_fields(rec, new)
    if diff:
        print("REPLAY DIFFERS in %s" % diff)
        for k in diff:
            if k != "raw":
                print("   %-18s recorded %r, replay %r" % (k, rec.get(k), new.get(k)))
    else:
        print("REPLAY IDENTICAL on %s" % ", ".join(
            k for k in COMPARED_FIELDS if k in rec))
    return diff


def summaries(stage, keys=None):
    """Model mean/SEM per grid point; ``keys`` limits it to those points."""
    by_point = {}
    for rec in load_records(stage):
        if keys is not None and rec["point"] not in keys:
            continue
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
                    default=None,
                    help="explicit gammaSC values (e.g. a verification point); "
                         "by default they come from Eq. 40 (build_grid)")
    ap.add_argument("--pooled-gamma-sc", dest="pooled", action="store_true",
                    help="one gammaSC_bar for all (alphaHC, R_gamma) -- a "
                         "rectangular box -- instead of one per pair")
    ap.add_argument("--r-gamma", dest="r_gamma", type=float, nargs="+",
                    default=GRID_R_GAMMA)
    ap.add_argument("--fine", action="store_true",
                    help="run the FINE grid around the best coarse point "
                         "(needs the coarse scores: run --score-only first)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for every output file, e.g. _verify")
    ap.add_argument("--re-extract", dest="re_extract", action="store_true",
                    help="re-measure every record from its run folders in "
                         "RESULTS_DIR, running nothing; writes the "
                         "<tag>_reextract records (score them with "
                         "--score-only --tag <tag>_reextract)")
    ap.add_argument("--replay", default=None, metavar="POINT_KEY",
                    help="re-run ONE recorded simulation with its recorded "
                         "seeds (needs --stage, --initial, --from) into "
                         "RESULTS_DIR, and compare it with the record")
    ap.add_argument("--initial", default=None,
                    help="--replay: the pattern of the simulation")
    ap.add_argument("--from", dest="source_dir", default=None,
                    help="--replay: the folder holding the fit's records "
                         "(not RESULTS_DIR)")
    a = ap.parse_args()
    global TAG
    TAG = a.tag

    if a.score_only:
        score()
        return
    if a.re_extract:
        stages = STAGES if a.stage == "both" else (a.stage,)
        n_diff = sum(re_extract(s) for s in stages)
        raise SystemExit(1 if n_diff else 0)
    if a.replay:
        if a.stage == "both" or not a.initial or not a.source_dir:
            raise SystemExit("--replay needs --stage, --initial and --from")
        raise SystemExit(1 if replay(a.stage, a.replay, a.initial,
                                     a.source_dir) else 0)
    if (any(r <= 1 for r in a.r_gamma) or any(x <= 1 for x in a.alpha_hc)
            or any(x <= 0 for x in (a.gamma_sc or []))):
        raise SystemExit("need R_gamma > 1, alphaHC > 1 and gammaSC > 0")
    stages = STAGES if a.stage == "both" else (a.stage,)
    if a.initials and len(stages) != 1:
        raise SystemExit("--initials needs a single --stage")
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
        if a.fine:
            centre, best = best_coarse_point(stage)
            grid, info = build_fine_grid(stage, centre)
            info["from_pair"] = best
            check_fine_grid_unchanged(stage, info)
            with open(os.path.join(RESULTS_DIR, FINE_GRID % (stage, TAG)),
                      "w") as fh:
                json.dump(info, fh, indent=1)
            print("%s: fine grid around %s (best pair J_mech %.4g), steps %s"
                  % (stage, info["centre"], best["J_mech"], info["fine_steps"]))
        elif a.gamma_sc:
            grid = [(x, g, r) for x in a.alpha_hc for g in a.gamma_sc
                    for r in a.r_gamma]
        else:
            grid, info = build_grid(stage, initials, a.alpha_hc, a.r_gamma,
                                    per_point=not a.pooled)
            # The grid's derivation travels with the fit's records.
            with open(os.path.join(RESULTS_DIR, GRID % (stage, TAG)), "w") as fh:
                json.dump(info, fh, indent=1)
            print("%s: lambda_exp %.4f; gammaSC_bar %s (pooled %.5f)"
                  % (stage, info["lambda_exp"],
                     "per (alphaHC, R_gamma)" if not a.pooled else "pooled",
                     info["pooled_gammaSC_bar"]))
            for row in info["bars"]:
                print("   alphaHC %4.1f  R_gamma %4.1f  gammaSC_bar %.5f  "
                      "(patterns %.5f-%.5f)"
                      % (row["alphaHC"], row["R_gamma"], row["gammaSC_bar"],
                         min(row["gammaSC_bar_per_pattern"]),
                         max(row["gammaSC_bar_per_pattern"])))
        simulate(stage, initials, grid, a.workers, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
