"""Run the FULL model (differentiation + mechanics) at each stage's v2 best fit, psigma = 0.

    python run_fitted_full_model.py --dry-run
    python run_fitted_full_model.py --stage E17.5 --workers 6
    python run_fitted_full_model.py --workers 6            # both stages

Mechanics are READ FROM THE FIT, not retyped: the best-scoring point of each
stage's self-consistent scan, chosen on roundness + shrinkage only (the ablation
term is excluded from the objective — the model fails it structurally at every
parameter setting, so including it would just add a constant offset). A0 is that
point's CONVERGED self-consistent value, so the run inherits the shrinkage match.

    E17.5  gammaSC 0.0105  R_gamma 6.746  R_alpha 3.5    A0 0.74181
    P0     gammaSC 0.0105  R_gamma 3.851  R_alpha 1.757  A0 0.75854

psigma = 0 means NO mechanosensitivity: delta production is not gated by stress,
so this is the baseline the psigma sweeps are measured against. It also keeps
stress_dependent off inside run().

THE FOLDER NAME IS PREFIXED "fullmodel_v2". At psigma = 0 the usual name is just
"fullmodel_<array>", which encodes NO mechanics — and 20 such folders from the
PRE-v2 model (perimeter elasticity, bending 0.02, A0 0.4657) are already on disk.
Without the prefix this run would overwrite them, or silently be skipped in
favour of them under --resume. The prefix keeps the two sets apart.

LATERAL INHIBITION IS UNCHANGED and shared by both stages — pS = 0.1, pR = 0.3,
atoh threshold 0.355079, levels seeded U(0, 0.01). Only the mechanics differ
between stages, which is the point of the comparison.

t_end is a CAP, not a duration: end_on_steady_state stops each run once it
settles. The default 100 comes from the psigma work, where t_end = 50 left many
runs unsettled and they had to be extended afterwards.
"""
import argparse
import json
import os

import numpy as np

from post_processing import RESULTS_DIR
from run_model import run_full_model_arrays
import grid_fit_mechanics_v2 as g2

SCAN = {"E17.5": "e17_selfconsistent_scan.json",
        "P0": "p0_selfconsistent_scan.json"}
NAME_PREFIX = "fullmodel_v2"

# Shared lateral-inhibition parameters (identical to run_full_model.py).
ATOH_SENSITIVITY = 0.355079
NOTCH_SENSITIVITY = 0.1
REPRESSOR_SENSITIVITY = 0.3
INITIAL_LI_LEVEL = 0.01

# v2 mechanics that are the same at both stages.
SHAPE_INDEX = 0.0       # pure contractility (p0 = 0)
BENDING = 0.0           # no bending elasticity
LINE_TENSION = None


def best_point(stage, terms=("roundness_ratio", "shrinkage")):
    """The scan's best point on `terms` only, with its converged A0.

    Scored WITHOUT the ablation term on purpose: it is flat in every parameter
    (the model cannot make HC shrink while SC expand), so including it would
    pick the same point but with a misleading objective.
    """
    path = os.path.join(RESULTS_DIR, SCAN[stage])
    if not os.path.isfile(path):
        raise SystemExit("no fit result for %s at %s" % (stage, path))
    pts = json.load(open(path))["points"]
    scored = []
    for v in pts.values():
        z = v.get("z") or {}
        vals = [z.get(t) for t in terms]
        if any(x is None or not np.isfinite(x) for x in vals):
            continue
        scored.append((sum(x * x for x in vals), v))
    if not scored:
        raise SystemExit("no scorable points in %s" % path)
    obj, v = min(scored, key=lambda s: s[0])
    return dict(gamma_sc=v["gamma_sc"], R_gamma=v["R_gamma"],
                R_alpha=v.get("R_alpha", json.load(open(path))["R_alpha"]),
                A0=v["A0"], objective=obj, z=v.get("z") or {},
                n_sheets_ok=v.get("n_sheets_ok"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=list(SCAN), choices=list(SCAN))
    ap.add_argument("--n-arrays", dest="n_arrays", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None,
                    help="default: min(#arrays, cpu_count). ~1.3 GB per run.")
    ap.add_argument("--t-end", dest="t_end", type=float, default=100,
                    help="CAP; runs stop early on steady state (default 100)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--save-interval", dest="save_interval", type=float, default=0.1)
    ap.add_argument("--psigma", type=float, default=0.0,
                    help="0 = no mechanosensitivity (the baseline)")
    ap.add_argument("--atoh", type=float, default=ATOH_SENSITIVITY)
    ap.add_argument("--name-prefix", dest="name_prefix", default=NAME_PREFIX)
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed runs and continue interrupted ones")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("FULL MODEL at the v2 best fit  |  psigma = %.4g  |  prefix %s_"
          % (a.psigma, a.name_prefix))
    print("=" * 78)
    print("  LI (shared): pS=%.4g  pR=%.4g  atoh=%.6g  levels ~ U(0, %.3g)"
          % (NOTCH_SENSITIVITY, REPRESSOR_SENSITIVITY, a.atoh, INITIAL_LI_LEVEL))
    print("  mechanics:   shape_index=0 (pure contractility), bending=0,"
          " line_tension=None, qst=%.4g" % g2.BASE_QST)
    print("  t_end<=%g (steady state ends it earlier), dt=%g, save_interval=%g"
          % (a.t_end, a.dt, a.save_interval))

    fits = {}
    print("\n  %-7s %9s %9s %9s %10s %10s %9s %s"
          % ("stage", "gammaSC", "R_gamma", "R_alpha", "A0", "A0/(pi/4)",
             "obj(fit)", "z(rnd, abl, shr)"))
    for stage in a.stage:
        f = best_point(stage)
        fits[stage] = f
        z = f["z"]
        print("  %-7s %9.4g %9.3f %9.3f %10.5f %10.4f %9.3f  %+.2f, %+.2f, %+.2f"
              % (stage, f["gamma_sc"], f["R_gamma"], f["R_alpha"], f["A0"],
                 f["A0"] / (np.pi / 4), f["objective"],
                 z.get("roundness_ratio", float("nan")),
                 z.get("ablation_ratio", float("nan")),
                 z.get("shrinkage", float("nan"))))
    print("\n  (obj(fit) is roundness+shrinkage only; the ablation z is shown but"
          " NOT fitted — see the module docstring)")

    existing = [d for d in os.listdir(RESULTS_DIR)
                if d.startswith(a.name_prefix + "_")] if os.path.isdir(RESULTS_DIR) else []
    print("\n  %d folder(s) with this prefix already exist%s"
          % (len(existing), " (--resume will reuse them)" if a.resume
             else " — WITHOUT --resume they will be re-run from scratch"))
    print("  %d run(s) planned: %d stage(s) x %d array(s)"
          % (len(a.stage) * a.n_arrays, len(a.stage), a.n_arrays))
    if a.dry_run:
        raise SystemExit("\n--dry-run: nothing was simulated.")

    for stage in a.stage:
        f = fits[stage]
        print("\n" + "-" * 78)
        names = run_full_model_arrays(
            stage,
            gammaSC=f["gamma_sc"],
            gammaHC_ratio=f["R_gamma"],
            alphaHC_ratio=f["R_alpha"],
            hc_shape_index=SHAPE_INDEX, sc_shape_index=SHAPE_INDEX,
            atoh_sensitivity=a.atoh,
            notch_sensitivity=NOTCH_SENSITIVITY,
            repressor_sensitivity=REPRESSOR_SENSITIVITY,
            bending=BENDING, line_tension=LINE_TENSION,
            quasi_static_threshold=g2.BASE_QST,
            preferred_area=f["A0"],
            psigma=a.psigma,
            initial_notch_delta_level=INITIAL_LI_LEVEL,
            t_end=a.t_end, dt=a.dt, save_interval=a.save_interval,
            n_arrays=a.n_arrays, n_workers=a.workers,
            reuse_existing_run=a.resume, continue_existing_run=a.resume,
            name_prefix=a.name_prefix)
        print("\n%s full model done: %d run(s)" % (stage, len(names)))
        for n in names:
            print("   ", n)


if __name__ == "__main__":
    main()
