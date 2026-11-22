"""The mechanical-fit objective of SI section 7 (Eqs. 28-38).

Per simulation (one pattern, one grid point) four observables are measured on
the base run's steady state and its ablation run:

* the HC/SC roundness ratio                                     (7.1, Eq. 28)
* the HC/SC area-change ratio of the ablated cells' neighbours  (7.2, Eq. 29)
* the linear shrinkage lambda_model of a free cut               (7.3, Eqs. 30-32)
* the isotropic stress sigma~                                   (7.4, Eq. 36)

Each per-stage term compares the MEAN over a stage's simulations with the
mean over the experimental repeats, scaled by BOTH standard errors:

    chi^2 = (<model> - <exp>)^2 / (SEM_exp^2 + SEM_model^2)

(the SI's Eqs. 28, 29 and 32 print only SEM_exp; the model SEM is included as
in Eqs. 25-26 -- decided 2026-11-22). The stress has no dimensionless
experimental counterpart, so only the E17.5/P0 RATIO of the stage means is
compared, with each side's SEM propagated:

    SEM(r) = r * sqrt((SEM_E / mean_E)^2 + (SEM_P / mean_P)^2),   r = mean_E / mean_P

which is the SI's Eq. 36-bis denominator written per side. Why the ratio is
the right comparison: with the SI's scalings, sigma = alpha_SC Lbar^2 sigma~
and mu~ = gamma_R mu / (alpha_SC Lbar^2), so sigma/mu = gamma_R sigma~/mu~.
The measured stress-over-viscosity ratio of the two stages is therefore
exactly sigma~_E / sigma~_P whenever both stages share mu~ (the code uses 1)
and gamma_R -- no assumption about alpha_SC or Lbar across stages is needed.

    J_stage = chi^2_roundness + chi^2_abl-area + chi^2_shrinkage
    J_mech  = J_E17.5 + J_P0 + chi^2_stress                       (Eq. 38)

Every grid point of both stages is simulated first; J_mech is then evaluated
for every (E17.5 point, P0 point) PAIR, because the stress term couples them.

DEGENERATE CASES. A term with fewer than two usable simulations, or no usable
experimental SEM, scores ``WORST_CHI2``. Shrinkage also scores ``WORST_CHI2``
when ANY simulation of the stage has lambda_model^2 <= 0 (the energy then has
no shrinkage minimum -- Eq. 31's only stationary point is lambda = 0); decided
2026-11-22. The ablation ratio of a simulation whose neighbours include no HC
is undefined and that simulation is dropped from the ablation term, as the
experimental repeat with no HC neighbour is.
"""
import numpy as np

# Score of an undefined or rejected term: large but finite, so J stays a sum
# that can be ranked. 1e6 is what the SI fixes for lambda^2 <= 0, and what the
# previous fit's _WORST_CASE_NSIGMA = 1e3 squared to.
WORST_CHI2 = 1e6

STAGE_TERMS = ("roundness_ratio", "ablation_ratio", "shrinkage")


# --------------------------------------------------------------------------- #
# Per-simulation closed forms                                                  #
# --------------------------------------------------------------------------- #
def _mechanical_arrays(face_df):
    """(A, L, A0, alpha, Gamma) of the cells the energy sums over.

    Cells tagged for removal (``type == -1``) are left out: their parameters
    were overwritten to drive them to zero area, so they are not part of the
    tissue whose shrinkage or stress is measured. Eq. 30 has no target
    perimeter, so a sheet that has one is refused rather than scored with a
    formula that does not describe it.
    """
    need = ("area", "perimeter", "prefered_area", "area_elasticity",
            "contractility")
    missing = [c for c in need if c not in face_df.columns]
    if missing:
        raise ValueError("face_df lacks %s" % missing)
    keep = np.ones(len(face_df), bool)
    if "type" in face_df.columns:
        keep &= face_df["type"].to_numpy() != -1
    if "prefered_perimeter" in face_df.columns:
        p0 = face_df["prefered_perimeter"].to_numpy(float)[keep]
        if np.any(np.abs(p0) > 1e-12):
            raise ValueError("Eqs. 30-36 assume no target perimeter, but "
                             "prefered_perimeter is non-zero")
    cols = [face_df[c].to_numpy(float)[keep] for c in need]
    if not all(np.isfinite(c).all() for c in cols):
        raise ValueError("non-finite mechanical columns")
    return cols


def shrinkage_lambda_squared(face_df):
    """Eq. 31: the scale^2 minimising E(lambda) of a freely relaxing cut.

        lambda^2 = sum(2 A0 A alpha - Gamma L^2) / (2 sum(A^2 alpha))

    It can be <= 0 (contractility beats the area term): the energy then has
    no shrinkage minimum, and the caller scores the stage worst-case.
    """
    A, L, A0, alpha, gamma = _mechanical_arrays(face_df)
    return float(np.sum(2 * A0 * A * alpha - gamma * L ** 2)
                 / (2 * np.sum(A ** 2 * alpha)))


def isotropic_stress(face_df):
    """Eq. 36: sigma~ = dE/dA_tot at lambda = 1,

        sum(2 alpha A (A - A0) + Gamma L^2) / (2 sum(A)).
    """
    A, L, A0, alpha, gamma = _mechanical_arrays(face_df)
    return float(np.sum(2 * alpha * A * (A - A0) + gamma * L ** 2)
                 / (2 * np.sum(A)))


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def mean_sem(values):
    """(mean, SEM, n) of the finite values; SEM is nan below n = 2."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    sem = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 \
        else float("nan")
    return float(v.mean()), sem, int(v.size)


def chi2(model_mean, model_sem, exp_mean, exp_sem):
    """(model - exp)^2 / (SEM_exp^2 + SEM_model^2); WORST_CHI2 if undefined."""
    vals = (model_mean, model_sem, exp_mean, exp_sem)
    if not all(np.isfinite(v) for v in vals):
        return WORST_CHI2
    denom = exp_sem ** 2 + model_sem ** 2
    if denom <= 0:
        return WORST_CHI2
    return float((model_mean - exp_mean) ** 2 / denom)


def ratio_sem(mean_a, sem_a, mean_b, sem_b):
    """(a/b, propagated SEM) for independent a and b."""
    r = mean_a / mean_b
    return r, abs(r) * float(np.hypot(sem_a / mean_a, sem_b / mean_b))


# --------------------------------------------------------------------------- #
# Experimental targets                                                         #
# --------------------------------------------------------------------------- #
_EXPERIMENTAL_TYPE = {
    "roundness_ratio": "HC to SC roundness ratio",
    "ablation_ratio": "HC to SC area change ratio after ablation",
}
STRESS_COLUMN = "Stress over viscosity (1/min)"


def experimental_targets(stage):
    """``{term: (mean, SEM, n)}`` over the experimental repeats of ``stage``,
    plus ``"stress"``: the stress over viscosity, which only enters as a
    ratio between stages.

    The ratio terms average each repeat first (each HC over its own repeat's
    mean SC); shrinkage is lambda = R_final / R_initial per circular ablation
    (the data stores 100 * (1 - lambda)).
    """
    import pandas as pd
    from post_processing import (load_experimental_results,
                                 resolve_circular_ablation_file)
    out = {}
    for term, exp_type in _EXPERIMENTAL_TYPE.items():
        arrays = [np.asarray(a, float) for a in
                  load_experimental_results(stage, exp_type)]
        out[term] = mean_sem([a[np.isfinite(a)].mean() for a in arrays
                              if np.isfinite(a).any()])
    pct = [float(np.asarray(a, float).mean())
           for a in load_experimental_results(stage, "cut shrinkage")]
    out["shrinkage"] = mean_sem([1.0 - p / 100.0 for p in pct])
    table = pd.read_excel(resolve_circular_ablation_file(),
                          sheet_name="Overall data")
    stress = pd.to_numeric(table[table["Stage"] == stage][STRESS_COLUMN],
                           errors="coerce").dropna().to_numpy(float)
    out["stress"] = mean_sem(stress)
    return out


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def stage_summary(records):
    """Model mean/SEM per observable over one grid point's simulations.

    ``records`` are per-simulation dicts with ``roundness_ratio``,
    ``ablation_ratio``, ``lambda_squared`` and ``stress`` (any may be None).
    Shrinkage is summarised as lambda_model = sqrt(lambda^2) and flagged
    ``rejected`` when any simulation has lambda^2 <= 0.
    """
    recs = [r for r in records if r is not None]
    lam_sq = [r.get("lambda_squared") for r in recs]
    lam_sq = [x for x in lam_sq if x is not None and np.isfinite(x)]
    rejected = any(x <= 0 for x in lam_sq)
    lam = [np.sqrt(x) for x in lam_sq if x > 0]
    return {
        "roundness_ratio": mean_sem([r.get("roundness_ratio") for r in recs]),
        "ablation_ratio": mean_sem([r.get("ablation_ratio") for r in recs]),
        "shrinkage": mean_sem(lam),
        "shrinkage_rejected": bool(rejected),
        "n_lambda_sq_nonpositive": int(sum(x <= 0 for x in lam_sq)),
        "stress": mean_sem([r.get("stress") for r in recs]),
        "n_simulations": len(recs),
    }


def stage_chi2(summary, targets):
    """``({term: chi^2}, J_stage)`` for one grid point of one stage."""
    terms = {}
    for term in STAGE_TERMS:
        m, s, n = summary[term]
        e_m, e_s, _ = targets[term]
        if n < 2:
            terms[term] = WORST_CHI2
        else:
            terms[term] = chi2(m, s, e_m, e_s)
    if summary["shrinkage_rejected"]:
        terms["shrinkage"] = WORST_CHI2
    return terms, float(sum(terms.values()))


def stress_chi2(summary_e17, summary_p0, targets_e17, targets_p0):
    """chi^2 of the E17.5/P0 stress ratio (ratio of the stage means)."""
    (mE, sE, nE), (mP, sP, nP) = summary_e17["stress"], summary_p0["stress"]
    if nE < 2 or nP < 2 or not (mE > 0 and mP > 0):
        return WORST_CHI2
    r_model, s_model = ratio_sem(mE, sE, mP, sP)
    (eE, esE, _), (eP, esP, _) = targets_e17["stress"], targets_p0["stress"]
    r_exp, s_exp = ratio_sem(eE, esE, eP, esP)
    return chi2(r_model, s_model, r_exp, s_exp)


def rank_pairs(points_e17, points_p0, targets_e17, targets_p0):
    """Every (E17.5, P0) pair of grid points, best J_mech first.

    ``points_*`` map a grid-point key to its :func:`stage_summary`. Returns a
    list of dicts with both keys, J_mech and its parts.
    """
    scored_e = {k: stage_chi2(v, targets_e17) for k, v in points_e17.items()}
    scored_p = {k: stage_chi2(v, targets_p0) for k, v in points_p0.items()}
    rows = []
    for ke, (te, je) in scored_e.items():
        for kp, (tp, jp) in scored_p.items():
            cs = stress_chi2(points_e17[ke], points_p0[kp],
                             targets_e17, targets_p0)
            rows.append({"E17.5": ke, "P0": kp, "J_mech": je + jp + cs,
                         "J_E17.5": je, "J_P0": jp, "chi2_stress": cs,
                         "terms_E17.5": te, "terms_P0": tp})
    rows.sort(key=lambda r: r["J_mech"])
    return rows
