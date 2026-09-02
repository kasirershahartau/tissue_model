"""Full-model runs -> two tables. Score 3 is gone; isolated SCs replace it.

    python build_fullmodel_table.py --limit 6        # quick check
    python build_fullmodel_table.py --workers 6      # full pass (resumable)

Writes to <results>/:
    fullmodel_tables.xlsx      sheets: runs, psigma
    fullmodel_runs.pkl         per-simulation table  (pandas)
    fullmodel_psigma.pkl       per-parameter-point table (pandas)
    fullmodel_runs.csv         incremental; an interrupted pass resumes from it

SCORE 3 IS DROPPED. It asked what fraction of the SCs present at t0 later
differentiated, so its value depended on where each run happened to stop — and
the steady-state check showed runs are still turning over when they stop,
especially at psigma > 0. In its place: how many SCs have NO HC neighbour in the
FINAL frame, as a percentage of all cells and of all SCs. That is a property of
one frame, so it carries no endpoint dependence, and it measures the same
failure mode score 3 was aimed at — patches of tissue the pattern never reached.

WHY TWO TABLES. Everything in ``runs`` is measured on a single simulation. The
n-sigma scores are not: they compare a POOLED set of runs against the three
experiments, so they only exist per parameter point and live in ``psigma``.

REPEATS ARE AVERAGED PER ARRAY before any statistic, exactly as
score_psigma_pooled.py does. SEM_sim sits in the n-sigma denominator, so treating
repeats of one array as independent points would shrink it by ~sqrt(n_repeats)
and inflate every score without the model having changed. The array-to-array
spread is the variation that belongs there; the seed-to-seed spread is noise.
So the psigma rows reproduce the pooled scores, not per-run ones.

CONTACT COUNTS AND PERCENTAGES come from one adjacency and one non-boundary
selection, so they cannot disagree; each pair is counted once and the three
counts sum to the total.

Parameters are read from each run's own parameters.txt, never from a sweep
script's constants, so a row always describes the run that actually happened.
Retried runs: the ``__dead*`` folders are excluded and the surviving run carries
``had_dead_retry``.
"""
import argparse
import fnmatch
import os
import re

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, experimental_results_folder,
                             load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type,
                             calc_contact_with_neighbors_from_type,
                             _best_matching_frame_by_neighbor_pairs,
                             _exp_neighbor_pair_percentages, _nsigma_and_chi2,
                             calc_HC_neighbors_at_differentiation)
from build_experimental_tables import (carried_over_sheets, FULLMODEL_SHEETS,
                                       to_output_names)
from build_run_table import read_parameters, _num, stage_of, array_of
from run_model import _reached_steady_state
from run_psigma_repeats import REPEAT_PREFIX

TYPE_BY = "delta_level"
THRESHOLD = 0.355079
COLLAPSE_HC_FRACTION = 0.10       # below this the pattern collapsed (run_psigma_sweep_v2)
MIN_TRAJECTORY_T = 1.0            # shorter than this and the run never started
# Differentiating cells are counted per HC-neighbour number, one column each, in
# addition to the 0/1/2+ buckets the scores use. The range runs past anything
# observed (the max in the data is 2) so the split is genuinely exhaustive; the
# overflow column exists to prove it — if it is ever non-zero the range is short.
MAX_NB = 6


# ---------------------------------------------------------------- per run ----
def pair_counts(sheet, type_by, threshold):
    """(HC:HC, HC:SC, SC:SC, total) neighbour-pair COUNTS, each pair once.

    Same adjacency and same non-boundary selection as
    _sim_neighbor_pair_percentages, so the counts and the percentages derived
    from them describe one and the same frame."""
    all_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="all", type_by=type_by, threshold=threshold)
    hc_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="HC", type_by=type_by, threshold=threshold)
    A = (np.asarray(sheet.get_contact_matrix()[np.ix_(all_idx, all_idx)]) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    hc = np.isin(all_idx, hc_idx)
    sc = ~hc
    hchc = A[np.ix_(hc, hc)].sum() / 2.0
    scsc = A[np.ix_(sc, sc)].sum() / 2.0
    hcsc = A[np.ix_(hc, sc)].sum()          # each HC-SC pair appears once here
    return float(hchc), float(hcsc), float(scsc), float(hchc + hcsc + scsc)


def frame_composition(sheet, type_by, threshold):
    """(n SC with zero HC neighbours, n HC, n SC, n cells) among non-boundary cells."""
    all_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="all", type_by=type_by, threshold=threshold)
    hc_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="HC", type_by=type_by, threshold=threshold)
    n_hc_nb, _ = calc_contact_with_neighbors_from_type(
        sheet, "all", "HC", type_by=type_by, threshold=threshold)
    is_hc = np.isin(all_idx, hc_idx)
    n_hc_nb = np.asarray(n_hc_nb)
    n_iso = int(((~is_hc) & (n_hc_nb == 0)).sum())
    return n_iso, int(is_hc.sum()), int((~is_hc).sum()), int(np.size(all_idx))


def psigma_of_name(name):
    """psigma from the folder name; an untagged run is psigma = 0.

    Name-based so a run can be selected without opening its parameters.txt —
    ``_psigma_tag`` writes 3 decimals when that is exact and 5 otherwise, and
    both forms are accepted here."""
    m = re.search(r"_ps(\d+\.\d+)_ks", name)
    return float(m.group(1)) if m else 0.0


def repeat_of(name):
    """1..10 from the run prefix. Repeat 1 has no r-suffix and is a prefix of the
    others, so the longest matching prefix wins."""
    for r in sorted(REPEAT_PREFIX, key=lambda k: -len(REPEAT_PREFIX[k])):
        if name.startswith(REPEAT_PREFIX[r] + "_"):
            return r
    return np.nan


_TARGET_CACHE = {}


def exp_targets(stage):
    """Per-experiment experimental values: (HC:HC%, HC:SC%, %0-nb, %1-nb) lists.

    Loaded exactly as compare_full_model_differentiation_to_experiments loads
    them, so the n-sigma here and the n-sigma in the scoring report agree."""
    if stage in _TARGET_CACHE:
        return _TARGET_CACHE[stage]
    prefix = "E17" if stage == "E17.5" else "P0"
    hchc, hcsc = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        a, b, _c = _exp_neighbor_pair_percentages(ci, cm)
        hchc.append(a); hcsc.append(b)
    p0, p1 = [], []
    for i in range(3):
        counts = np.asarray(np.load(os.path.join(
            experimental_results_folder,
            "%s differentiating cells_experiment%d.npy" % (stage, i))), float)
        if counts.size == 0:
            continue
        p0.append(100.0 * np.mean(counts == 0))
        p1.append(100.0 * np.mean(counts == 1))
    _TARGET_CACHE[stage] = (hchc, hcsc, p0, p1)
    return _TARGET_CACHE[stage]


_ISO_CACHE = {}


def exp_isolated_sc(stage):
    """Per-experiment (%SCs, %cells) with no HC neighbour, frame 1.

    Same valid-cell mask and same type column as _exp_neighbor_pair_percentages,
    so this is the population score 1's target comes from — the model number is
    therefore comparable to it without further correction."""
    if stage in _ISO_CACHE:
        return _ISO_CACHE[stage]
    prefix = "E17" if stage == "E17.5" else "P0"
    of_sc, of_all = [], []
    for e in (1, 2, 3):
        ci = pd.read_pickle(os.path.join(experimental_results_folder, stage,
                            "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(experimental_results_folder, stage,
                     "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        valid = ci.valid.values.astype(bool)
        is_hc = (ci.type.values == 1)[valid]
        A = np.asarray(cm)[np.ix_(valid, valid)] > 0
        np.fill_diagonal(A, False)
        iso = int(((~is_hc) & (A[:, is_hc].sum(axis=1) == 0)).sum())
        of_sc.append(100.0 * iso / max(int((~is_hc).sum()), 1))
        of_all.append(100.0 * iso / max(int(valid.sum()), 1))
    _ISO_CACHE[stage] = (of_sc, of_all)
    return _ISO_CACHE[stage]


def differentiation_events(history, t0, threshold, type_by=TYPE_BY):
    """One record per differentiation event, with its neighbourhood at the moment
    it differentiated.

    The detection deliberately mirrors calc_HC_neighbors_at_differentiation: take
    every non-boundary HC of the FINAL frame, walk back through the uninterrupted
    run of HC frames, and call the first of them the differentiation frame; a cell
    that was already a HC at t0 has no crossing in the window and is skipped. The
    HC-neighbour counts here therefore reproduce that function's output exactly,
    which is asserted in the tests below — if the two ever diverge, score 2 and
    this sheet would be describing different sets of cells.

    Two passes: the first caches only delta per frame (cheap) to locate the
    crossings, the second visits just the frames that actually host an event and
    computes contacts there, since the contact matrix is the expensive part.
    """
    stamps = np.asarray(get_time_points(history), float)
    stamps = stamps[stamps >= t0]
    if stamps.size == 0:
        return []
    value_by_id = []
    for t in stamps:
        s = history.retrieve(float(t))
        s.arrange_sheet_from_history()
        value_by_id.append(s.face_df.set_index("id")[type_by])

    final = history.retrieve(float(stamps[-1]))
    final.arrange_sheet_from_history()
    _idx, final_hc_ids = get_non_boundary_cell_ids_from_type(
        final, cell_type="HC", type_by=type_by, threshold=threshold)

    def is_hc(v):
        return v is not None and not (v is np.nan or (isinstance(v, float) and np.isnan(v))) \
            and v > threshold

    by_frame = {}
    last = stamps.size - 1
    for cid in final_hc_ids:
        f = last
        while f > 0 and is_hc(value_by_id[f - 1].get(cid)):
            f -= 1
        if f == 0:                       # already a HC at t0, never crossed here
            continue
        by_frame.setdefault(f, []).append(cid)

    rows = []
    for f in sorted(by_frame):
        s = history.retrieve(float(stamps[f]))
        s.arrange_sheet_from_history()
        s.geom.update_all(s)
        _ai, all_ids = get_non_boundary_cell_ids_from_type(
            s, cell_type="all", type_by=type_by, threshold=threshold)
        n_hc, len_hc = calc_contact_with_neighbors_from_type(
            s, "all", "HC", type_by=type_by, threshold=threshold)
        n_sc, len_sc = calc_contact_with_neighbors_from_type(
            s, "all", "SC", type_by=type_by, threshold=threshold)
        n_all, len_all = calc_contact_with_neighbors_from_type(
            s, "all", "all", type_by=type_by, threshold=threshold)
        pos = {int(c): k for k, c in enumerate(np.asarray(all_ids))}
        area = s.face_df.set_index("id")["area"]
        delta = s.face_df.set_index("id")[type_by]
        for cid in by_frame[f]:
            k = pos.get(int(cid))
            if k is None:                # boundary cell in that frame: no counts
                continue
            rows.append(dict(
                cell_id=int(cid), t_differentiated=float(stamps[f]),
                dt_since_t0=float(stamps[f]) - float(t0),
                n_HC_neighbours=int(np.asarray(n_hc)[k]),
                n_SC_neighbours=int(np.asarray(n_sc)[k]),
                n_neighbours=int(np.asarray(n_all)[k]),
                contact_length_HC=float(np.asarray(len_hc)[k]),
                contact_length_SC=float(np.asarray(len_sc)[k]),
                contact_length_total=float(np.asarray(len_all)[k]),
                area_at_differentiation=float(area.get(cid, np.nan)),
                delta_at_differentiation=float(delta.get(cid, np.nan))))
    return rows


def one_run(args):
    name, threshold = args
    try:
        stage = stage_of(name)
        if stage is None:
            return dict(model_name=name, error="cannot infer stage from name")
        p = read_parameters(name)
        row = dict(
            model_name=name, stage=stage, initial_array=array_of(name),
            repeat=repeat_of(name), psigma=_num(p, "psigma"),
            K_stress_shift=(_num(p, "stress_shift")
                            if str(p.get("stress_dependent")) == "True" else np.nan),
            hill_exponent=_num(p, "hill_exponent"),
            gammaSC=_num(p, "gammaSC"), gammaHC_ratio=_num(p, "gammaHC_ratio"),
            alphaHC_ratio=_num(p, "alphaHC_ratio"),
            A0=_num(p, "preferred_area_override"),
            hc_shape_index=_num(p, "hc_shape_index"),
            sc_shape_index=_num(p, "sc_shape_index"), bending=_num(p, "bending"),
            pS=_num(p, "notch_sensitivity"), pR=_num(p, "repressor_sensitivity"),
            atoh_threshold=_num(p, "atoh_sensitivity"),
            delta_threshold=threshold,
            had_dead_retry=os.path.isdir(os.path.join(RESULTS_DIR, name + "__dead1")),
            error="")
        row["gammaHC"] = row["gammaSC"] * row["gammaHC_ratio"]

        history = load_history_file(name)
        stamps = np.asarray(get_time_points(history), float)
        # A run whose history never left its first frames did not COLLAPSE, it
        # never ran — a crashed batch leaves such folders behind. Scored as data
        # it reads as zero HCs and inflates the collapse rate, which is exactly
        # the quantity the high-psigma points exist to measure, so it is reported
        # as an error instead and dropped from every statistic.
        if stamps.size < 2 or float(stamps[-1]) < MIN_TRAJECTORY_T:
            return dict(model_name=name, stage=stage, psigma=row["psigma"],
                        repeat=row["repeat"], initial_array=row["initial_array"],
                        t_final=float(stamps[-1]) if stamps.size else np.nan,
                        error="incomplete: no trajectory (t_final < %g)"
                              % MIN_TRAJECTORY_T)
        row["t_final"] = float(stamps[-1])
        row["n_frames"] = int(stamps.size)
        row["steady_state"] = bool(_reached_steady_state(os.path.join(RESULTS_DIR, name)))

        # ---- chosen initial frame t0 and its contact composition ------------
        e_hchc, e_hcsc, _p0, _p1 = exp_targets(stage)
        t0, _h, _s = _best_matching_frame_by_neighbor_pairs(
            history, float(np.nanmean(e_hchc)), float(np.nanmean(e_hcsc)),
            TYPE_BY, threshold)
        sheet0 = history.retrieve(float(t0))
        sheet0.arrange_sheet_from_history()
        hchc, hcsc, scsc, tot = pair_counts(sheet0, TYPE_BY, threshold)
        row.update(t0=float(t0),
                   n_HCHC_contacts_t0=hchc, n_HCSC_contacts_t0=hcsc,
                   n_SCSC_contacts_t0=scsc, n_total_contacts_t0=tot,
                   pct_HCHC_contacts_t0=100.0 * hchc / tot if tot else np.nan,
                   pct_HCSC_contacts_t0=100.0 * hcsc / tot if tot else np.nan,
                   pct_SCSC_contacts_t0=100.0 * scsc / tot if tot else np.nan)
        n_iso0, n_hc0, n_sc0, n_all0 = frame_composition(sheet0, TYPE_BY, threshold)
        row.update(n_cells_t0=n_all0, n_HC_t0=n_hc0, n_SC_t0=n_sc0,
                   hc_fraction_t0=n_hc0 / n_all0 if n_all0 else np.nan,
                   n_SC_no_HC_neighbour_t0=n_iso0)

        # ---- differentiation events after t0 --------------------------------
        counts = np.asarray(calc_HC_neighbors_at_differentiation(
            history, initial_time_point=t0, type_by=TYPE_BY,
            threshold=threshold), dtype=float)
        counts = counts[~np.isnan(counts)]
        n = counts.size
        row["n_differentiation_events"] = int(n)
        for nb, lab in ((0, "0"), (1, "1"), (2, "2plus")):
            k = int((counts == nb).sum()) if nb < 2 else int((counts >= 2).sum())
            row["n_events_%s_HC_neighbours" % lab] = k
            row["pct_events_%s_HC_neighbours" % lab] = 100.0 * k / n if n else np.nan
        # the 2+ bucket resolved: one column per exact neighbour count
        for nb in range(MAX_NB + 1):
            k = int((counts == nb).sum())
            row["n_events_exactly_%d_HC_neighbours" % nb] = k
            row["pct_events_exactly_%d_HC_neighbours" % nb] = (100.0 * k / n if n
                                                               else np.nan)
        k = int((counts > MAX_NB).sum())
        row["n_events_more_than_%d_HC_neighbours" % MAX_NB] = k
        row["pct_events_more_than_%d_HC_neighbours" % MAX_NB] = (100.0 * k / n if n
                                                                 else np.nan)

        # ---- final frame: isolated SCs (score 3's replacement) ---------------
        sheetF = history.retrieve(float(stamps[-1]))
        sheetF.arrange_sheet_from_history()
        n_iso, n_hc, n_sc, n_all = frame_composition(sheetF, TYPE_BY, threshold)
        row.update(n_cells_final=n_all, n_HC_final=n_hc, n_SC_final=n_sc,
                   hc_fraction_final=n_hc / n_all if n_all else np.nan,
                   n_SC_no_HC_neighbour_final=n_iso,
                   pct_SC_no_HC_neighbour_of_all_cells=(100.0 * n_iso / n_all
                                                        if n_all else np.nan),
                   pct_SC_no_HC_neighbour_of_SC=100.0 * n_iso / n_sc if n_sc else np.nan)
        # a collapsed run is a different outcome, not a noisy sample of the same one
        row["collapsed"] = bool(n_all and (n_hc / n_all) < COLLAPSE_HC_FRACTION)

        # per-event detail, carried out of band so the runs CSV keeps one row
        # per run; main() peels it off into its own table
        ev = differentiation_events(history, t0, threshold)
        for e in ev:
            e.update(model_name=name, stage=stage, psigma=row["psigma"],
                     initial_array=row["initial_array"], repeat=row["repeat"],
                     t0=float(t0))
        row["_events"] = ev
        row["n_events_recorded"] = len(ev)
        return row
    except Exception as exc:                            # noqa: BLE001
        return dict(model_name=name, error="%s: %s" % (type(exc).__name__, exc))


# ------------------------------------------------------------- per psigma ----
def _per_array(g, col):
    """One value per initial array: the mean over that array's repeats."""
    v = g.groupby("initial_array")[col].mean().to_numpy(float)
    return v[np.isfinite(v)]


def _mean_sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(v.mean()) if v.size else np.nan,
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)


def psigma_table(df, drop_collapsed=False):
    rows = []
    # The saved runs table has no ``error`` column: incomplete runs are dropped
    # as rows before saving, so there is nothing left to filter. The in-memory
    # frame built during a pass still has it.
    ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    if drop_collapsed:
        ok = ok[~ok["collapsed"].astype(bool)]
    for (stage, ps), g in ok.groupby(["stage", "psigma"]):
        e_hchc, e_hcsc, e_p0, e_p1 = exp_targets(stage)
        row = dict(stage=stage, psigma=float(ps), n_runs=len(g),
                   n_arrays=int(g["initial_array"].nunique()),
                   n_repeats=int(g["repeat"].nunique()),
                   n_collapsed=int(g["collapsed"].astype(bool).sum()),
                   n_steady_state=int(g["steady_state"].astype(bool).sum()),
                   mean_t0=float(g["t0"].mean()),
                   mean_t_final=float(g["t_final"].mean()))
        for score, buckets in (
                (1, (("HCHC", "pct_HCHC_contacts_t0", e_hchc),
                     ("HCSC", "pct_HCSC_contacts_t0", e_hcsc))),
                (2, (("ev0", "pct_events_0_HC_neighbours", e_p0),
                     ("ev1", "pct_events_1_HC_neighbours", e_p1)))):
            tot, undef = 0.0, 0
            for lab, col, exp in buckets:
                sim = _per_array(g, col)
                z, chi2, ms, me = _nsigma_and_chi2(sim, exp)
                _m, sem_s = _mean_sem(sim)
                _m, sem_e = _mean_sem(exp)
                row.update({"%s_sim_mean" % lab: ms, "%s_sim_sem" % lab: sem_s,
                            "%s_exp_mean" % lab: me, "%s_exp_sem" % lab: sem_e,
                            "%s_nsigma" % lab: z, "%s_chi2" % lab: chi2})
                tot += 0.0 if not np.isfinite(chi2) else chi2
                undef += 0 if np.isfinite(chi2) else 1
            row["score%d" % score] = tot
            # A term is undefined when the runs produced nothing to measure — at
            # high psigma the pattern collapses and there are no differentiation
            # events at all. Summing it as zero (the convention in
            # compare_full_model_differentiation_to_experiments) then makes a dead
            # point look like a PERFECT fit, so count the undefined terms and let
            # callers drop those points instead of ranking them best.
            row["score%d_undefined_terms" % score] = undef
        row["score1_plus_score2"] = row["score1"] + row["score2"]
        for lab, col in (("iso_SC_of_all_cells", "pct_SC_no_HC_neighbour_of_all_cells"),
                         ("iso_SC_of_SC", "pct_SC_no_HC_neighbour_of_SC"),
                         ("n_events", "n_differentiation_events"),
                         ("hc_fraction_final", "hc_fraction_final")):
            m, s = _mean_sem(_per_array(g, col))
            row["%s_mean" % lab] = m
            row["%s_sem" % lab] = s
        # The isolated-SC statistic replaces score 3, so it needs its experimental
        # target in the same row — otherwise the model number is uninterpretable.
        e_sc, e_all = exp_isolated_sc(stage)
        for lab, col, exp in (("iso_SC_of_SC", "pct_SC_no_HC_neighbour_of_SC", e_sc),
                              ("iso_SC_of_all_cells",
                               "pct_SC_no_HC_neighbour_of_all_cells", e_all)):
            z, chi2, _ms, me = _nsigma_and_chi2(_per_array(g, col), exp)
            _m, sem_e = _mean_sem(exp)
            row["%s_exp_mean" % lab] = me
            row["%s_exp_sem" % lab] = sem_e
            row["%s_nsigma" % lab] = z
            row["%s_chi2" % lab] = chi2
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["stage", "psigma"]).reset_index(drop=True)


def _prune_for_output(df, pdf):
    """Trim the SAVED tables to the columns worth reading.

    Applied at save time only: psigma_table() keeps returning everything, because
    the figures compute their own table from the runs and rely on the
    ``*_undefined_terms`` flags to tell a dead parameter point from a good fit.
    The CSV cache also keeps every column, so nothing is lost — this shapes the
    deliverable, not the data.

    Incomplete runs are dropped as ROWS, not just flagged. They are empty run
    folders left by a crashed batch, and with them gone the ``error`` column has
    nothing left to say.
    """
    n_before = len(df)
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""].copy()
    dropped = n_before - len(df)

    # The 0/1 buckets and the "exactly" columns are the same numbers under two
    # names; keep the plain form. Counts above 3 are zero across all 30556
    # events, so those columns carry nothing.
    ren = {}
    for k in (2, 3):
        for pre in ("n", "pct"):
            src = "%s_events_exactly_%d_HC_neighbours" % (pre, k)
            if src in df.columns:
                ren[src] = "%s_events_%d_HC_neighbours" % (pre, k)
    df = df.rename(columns=ren)

    drop = ["K_stress_shift", "hill_exponent", "gammaHC_ratio", "alphaHC_ratio",
            "hc_shape_index", "sc_shape_index", "bending", "atoh_threshold",
            "error", "n_events_recorded"]
    for pre in ("n", "pct"):
        drop += ["%s_events_2plus_HC_neighbours" % pre,
                 "%s_events_more_than_%d_HC_neighbours" % (pre, MAX_NB)]
        drop += ["%s_events_exactly_%d_HC_neighbours" % (pre, k)
                 for k in (0, 1) + tuple(range(4, MAX_NB + 1))]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    pdf = pdf.drop(columns=[c for c in pdf.columns if c.endswith("_undefined_terms")])
    print("\n  pruned for output: %d incomplete run(s) dropped, "
          "runs -> %d cols, psigma -> %d cols" % (dropped, df.shape[1], pdf.shape[1]))
    return df, pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="fullmodel_v2*",
                    help="run folders to include (default: every v2 full-model run)")
    ap.add_argument("--psigma", type=float, nargs="+", default=None,
                    help="only runs at these psigma values (0 = the untagged "
                         "runs). The CSV is shared, so a filtered pass is just a "
                         "reordering: a later unfiltered pass skips what it did.")
    ap.add_argument("--repeats", type=int, nargs="+", default=None,
                    help="only MEASURE these repeat indices (1 = the unsuffixed "
                         "prefix); the saved tables still describe everything")
    ap.add_argument("--max-repeat", dest="max_repeat", type=int, default=None,
                    help="restrict the SAVED tables to repeats 1..N as well, so "
                         "every point rests on the same number of repeats. Unlike "
                         "--repeats this narrows the output, which is the point.")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--drop-collapsed", dest="drop_collapsed", action="store_true",
                    help="exclude collapsed runs from the psigma statistics "
                         "(they are always kept in the runs table)")
    ap.add_argument("--refresh-json", dest="refresh_json", default=None,
                    help="recompute the runs named in this JSON report (e.g. "
                         "revive_dead_runs.json). A revived run REUSES its folder "
                         "name, so without this the cached row from the dead "
                         "attempt would be kept and the revival silently ignored.")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    # ``all_names`` is every run the tables should DESCRIBE; ``names`` is the
    # subset this pass will MEASURE. The filters must not touch the former —
    # writing only the filtered subset would shrink the saved tables to whatever
    # the last partial pass happened to look at.
    all_names = sorted(d for d in os.listdir(RESULTS_DIR)
                       if fnmatch.fnmatch(d, a.pattern)
                       and os.path.isdir(os.path.join(RESULTS_DIR, d))
                       and not d.endswith("_abl")
                       and not re.search(r"__dead\d*$", d))
    if a.max_repeat is not None:
        all_names = [n for n in all_names
                     if not np.isnan(repeat_of(n)) and repeat_of(n) <= a.max_repeat]
    names = list(all_names)
    if a.psigma is not None:
        want = {round(float(p), 6) for p in a.psigma}
        names = [n for n in names if round(psigma_of_name(n), 6) in want]
    if a.repeats is not None:
        want_r = set(a.repeats)
        names = [n for n in names if repeat_of(n) in want_r]
    if a.limit:
        names = names[:a.limit]
    csv_path = os.path.join(RESULTS_DIR, "fullmodel_runs.csv")
    if a.no_resume and os.path.isfile(csv_path):
        os.remove(csv_path)
    done = set()
    if os.path.isfile(csv_path):
        try:
            done = set(pd.read_csv(csv_path)["model_name"])
        except Exception:                               # noqa: BLE001
            done = set()
    stale = set()
    if a.refresh_json:
        import json
        p = (a.refresh_json if os.path.isabs(a.refresh_json)
             else os.path.join(RESULTS_DIR, a.refresh_json))
        try:
            for r in json.load(open(p)):
                nm = r.get("name")
                if nm:
                    stale.add(nm)
        except (OSError, ValueError) as exc:
            print("  could not read %s (%s); nothing refreshed" % (p, exc))
        done -= stale
        print("  refreshing %d run(s) named in %s"
              % (len(stale & set(names)), os.path.basename(p)))
    todo = [(n, a.threshold) for n in names if n not in done]
    print("full-model runs matching %r: %d;  %d already done,  %d to do"
          % (a.pattern, len(names), len(names) - len(todo), len(todo)), flush=True)

    # An error row carries only a few keys while a good row carries ~50. Appending
    # it as-is writes that many fields into a wider CSV, and read_csv then maps
    # them POSITIONALLY onto the first columns — an incomplete run's array index
    # silently becomes its psigma. Every appended row is therefore reindexed onto
    # the file's own column order first.
    cols = None
    if os.path.isfile(csv_path):
        try:
            cols = list(pd.read_csv(csv_path, nrows=0).columns)
        except Exception:                               # noqa: BLE001
            cols = None

    ev_path = os.path.join(RESULTS_DIR, "fullmodel_events.csv")
    if a.no_resume and os.path.isfile(ev_path):
        os.remove(ev_path)
    ev_cols = None
    if os.path.isfile(ev_path):
        try:
            ev_cols = list(pd.read_csv(ev_path, nrows=0).columns)
        except Exception:                               # noqa: BLE001
            ev_cols = None

    if todo:
        from concurrent.futures import ProcessPoolExecutor
        per_pool = max(1, 3 * a.workers)                # recycle: see run_task_pool
        for s in range(0, len(todo), per_pool):
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for r in ex.map(one_run, todo[s:s + per_pool]):
                    events = r.pop("_events", None) or []
                    if events:
                        ef = pd.DataFrame(events)
                        if ev_cols is None:
                            ev_cols = list(ef.columns)
                        ef = ef.reindex(columns=ev_cols)
                        ef.to_csv(ev_path, mode="a", index=False,
                                  header=not os.path.isfile(ev_path))
                    frame = pd.DataFrame([r])
                    if cols is None:                    # first row defines the order
                        cols = list(frame.columns)
                    elif set(frame.columns) - set(cols):
                        cols = cols + [c for c in frame.columns if c not in cols]
                    frame = frame.reindex(columns=cols)
                    frame.to_csv(csv_path, mode="a", index=False,
                                 header=not os.path.isfile(csv_path))
            print("  %d/%d" % (min(s + per_pool, len(todo)), len(todo)), flush=True)

    df = pd.read_csv(csv_path)
    df = df[df["model_name"].isin(all_names)].drop_duplicates("model_name", keep="last")
    bad = df[df["error"].fillna("") != ""]
    if len(bad):
        print("\n  %d run(s) failed and carry only an error string:" % len(bad))
        for _i, r in bad.head(5).iterrows():
            print("    %-52s %s" % (str(r["model_name"])[:52], str(r["error"])[:60]))
    pdf = psigma_table(df, drop_collapsed=a.drop_collapsed)

    edf = pd.DataFrame()
    if os.path.isfile(ev_path):
        edf = pd.read_csv(ev_path)
        edf = edf[edf["model_name"].isin(all_names)].drop_duplicates(
            ["model_name", "cell_id", "t_differentiated"], keep="last")
        edf.to_pickle(os.path.join(RESULTS_DIR, "fullmodel_events.pkl"))

    df, pdf = _prune_for_output(df, pdf)
    # published under the manuscript's name for the parameter; readers map it
    # back with build_experimental_tables.read_table
    df, pdf, edf = (to_output_names(f) for f in (df, pdf, edf))

    df.to_pickle(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    pdf.to_pickle(os.path.join(RESULTS_DIR, "fullmodel_psigma.pkl"))
    xlsx = os.path.join(RESULTS_DIR, "fullmodel_tables.xlsx")
    try:
        with pd.ExcelWriter(xlsx) as writer:
            df.to_excel(writer, sheet_name="runs", index=False)
            pdf.to_excel(writer, sheet_name="pT", index=False)
            if len(edf):
                # one row per differentiation event; ~30k rows, well inside
                # Excel's sheet limit
                edf.to_excel(writer, sheet_name="events", index=False)
            # the experimental targets the scores compare against, written by
            # build_experimental_tables.py; carried over so rewriting this
            # workbook does not drop them
            for name, frame in carried_over_sheets(FULLMODEL_SHEETS):
                frame.to_excel(writer, sheet_name=name, index=False)
                print("  carried over sheet %s (%d rows)" % (name, len(frame)))
    except Exception as exc:                            # noqa: BLE001
        print("  xlsx failed (%s: %s); the pickles and the CSV are written"
              % (type(exc).__name__, exc))

    print("\n  runs   %6d rows x %3d cols" % df.shape)
    print("  psigma %6d rows x %3d cols" % pdf.shape)
    if len(pdf):
        print("\n  %-6s %-7s %5s %8s %8s %8s %10s %9s"
              % ("stage", "psigma", "runs", "score1", "score2", "s1+s2",
                 "iso%/cells", "iso%/SC"))
        for _i, r in pdf.iterrows():
            print("  %-6s %-7.3f %5d %8.3f %8.3f %8.3f %10.2f %9.2f"
                  % (r["stage"], r["psigma"], r["n_runs"], r["score1"], r["score2"],
                     r["score1_plus_score2"], r["iso_SC_of_all_cells_mean"],
                     r["iso_SC_of_SC_mean"]))
    print("\nwrote %s" % xlsx)
    print("      fullmodel_runs.pkl, fullmodel_psigma.pkl, fullmodel_runs.csv")


if __name__ == "__main__":
    main()
