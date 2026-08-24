"""Launch the FULL model (lateral-inhibition differentiation + quasi-static
mechanics, coupled) for one stage's arrays, in parallel, from an
UNDIFFERENTIATED start.

    python run_full_model.py E17.5
    python run_full_model.py P0        # once P0's mechanical fit is chosen

Shared LI parameters (same for both stages): pS=0.1, pR=0.3, atoh threshold
0.355079 (a delta-level threshold), and notch/delta/repressor each seeded from
U(0, 0.01) — the random, near-zero start whose tiny spread lateral inhibition
amplifies into the salt-and-pepper pattern.

Per-stage MECHANICS come from that stage's best fit. Everything else matches the
mechanical fit: gammaHC_ratio=1, bending=0.02, no line tension, quasi-static
threshold 0.03, dt=0.01, no stress dependence, and the preferred area set by the
circular-ablation data (0.593*pi/4 — see SHRINKAGE_ESTIMATE_METHOD.md).

t_end=50: at t_end=25 most runs had NOT reached steady state (9 of 10 hit the
cap), so differentiation was still in progress at cutoff — which inflates the
"% differentiating" comparison.
"""
import os
import sys
import numpy as np

from full_model import run_full_model_arrays

# best-fit mechanics per stage: (gammaSC, alphaHC_ratio, hc_shape_index, sc_shape_index)
MECH = {
    # 5-term score 1.076 at alphaHC=1.00 (better than the 4-term fit's 1.06).
    "E17.5": (0.2461, 1.00, 4.86, 5.72),
    # 5-term fit (shrinkage included), objective 10.723; alphaHC FIXED at 1.0.
    "P0": (0.2298, 1.00, 5.1487, 5.6706),
}

PREFERRED_AREA = 0.593 * np.pi / 4           # 0.4657
T_END = 50                                   # was 25 -> most runs hit the cap

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    # --resume: reuse COMPLETED runs and continue INTERRUPTED ones (e.g. after
    # the disk filled) instead of starting over. Off by default because the
    # folder name does not encode the mechanics, so reusing blindly after a
    # parameter change would silently keep the OLD results.
    resume = "--resume" in sys.argv
    stage = args[0] if args else "E17.5"
    if stage not in MECH:
        raise SystemExit("no mechanics set for %r yet (fill in MECH). Have: %s"
                         % (stage, ", ".join(MECH)))
    gammaSC, alphaHC, hc_p0, sc_p0 = MECH[stage]
    if resume:
        print("--resume: completed runs reused, interrupted runs continued", flush=True)
    # Worker cap, as for the fits: set TISSUE_FIT_WORKERS when sharing a VM
    # (~1.3 GB per simulation, so 64 GB tolerates ~45 concurrent).
    n_workers = os.environ.get("TISSUE_FIT_WORKERS")
    n_workers = int(n_workers) if n_workers else None

    names = run_full_model_arrays(
        stage, gammaSC=gammaSC, alphaHC_ratio=alphaHC,
        hc_shape_index=hc_p0, sc_shape_index=sc_p0,
        atoh_sensitivity=0.355079, notch_sensitivity=0.1, repressor_sensitivity=0.3,
        bending=0.02, quasi_static_threshold=0.03,
        preferred_area=PREFERRED_AREA,
        initial_notch_delta_level=0.01,      # random LI seed, U(0, 0.01)
        t_end=T_END, dt=0.01, n_arrays=10, n_workers=n_workers,
        reuse_existing_run=resume)
    print("\n%s full model done: %d runs" % (stage, len(names)))
    for n in names:
        print("  ", n)
