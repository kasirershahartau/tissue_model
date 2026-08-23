"""Measured circular-ablation shrinkage -> the model preferred area that reproduces it.

Experiment: initial radius 60 um, 'Final radius (um)' per ablation; shrinkage is
LINEAR in radius. E17.5 and P0 pooled (no significant difference).

Model: a freely relaxing cut piece rescales isotropically by lambda minimizing
    E(l) = sum_faces [ K/2 (l^2 A - A0)^2 + G/2 (l P - P0)^2 ]
(bending depends only on angles -> scale invariant -> drops out).
Lowering A0 raises tension BOTH ways: directly (area term) and by lowering the
target perimeter P0 = shape_index*sqrt(A0), which is what currently RESISTS the
contraction. Solve for the A0 giving lambda* = 1 - measured_shrinkage.

The in-tissue mean cell AREA is pinned by the periodic box, so it is unchanged by
A0; the perimeters would shift somewhat, hence "estimate" (verify with a run).
"""
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:/Users/Kasirer/Phd/mouse_ear_project/tissue_model")
sys.path.insert(0, r"C:/Users/Kasirer/Phd/mouse_ear_project/tissue_model/tyssue/src")
os.environ.setdefault("TISSUE_RESULTS_DIR", r"D:/Kasirer/results")
from post_processing import load_history_file, get_time_points

XL = (r"C:/Users/Kasirer/Phd/mouse_ear_project/papers/Dynamic lateral inhibition"
      r" in the utricle/Raw Data/circular_ablation_raw_data(figure 3 +S4).xlsx")
R0 = 60.0
RD = os.environ["TISSUE_RESULTS_DIR"]
PAT = "fit_gSC0.25_gHC1.00_aHC1.06_ps0.00_qst0.030_bend0.020_pa0.707_p0hc3.95_p0sc4.64_*"

# ---------- 1. experimental shrinkage ---------------------------------------
d = pd.read_excel(XL, sheet_name="Overall data")
rf = d["Final radius (um)"].astype(float)
shr = 100.0 * (R0 - rf) / R0
print("=" * 68)
print("EXPERIMENT  (initial radius %.0f um, %d ablations)" % (R0, len(rf)))
print("=" * 68)
for st in ("E17.5", "P0"):
    m = d["Stage"] == st
    print("  %-6s n=%2d   final radius %.2f +- %.2f um   shrinkage %.2f%% +- %.2f%%"
          % (st, m.sum(), rf[m].mean(), rf[m].std(ddof=1),
             shr[m].mean(), shr[m].std(ddof=1)))
SHRINK = float(shr.mean())
SEM = float(shr.std(ddof=1) / np.sqrt(len(shr)))
print("  POOLED n=%d  final radius %.2f um   LINEAR shrinkage %.2f%% +- %.2f%% (SEM)"
      % (len(rf), rf.mean(), SHRINK, SEM))
LAMBDA_TARGET = 1.0 - SHRINK / 100.0
print("  -> target relaxed scale lambda* = %.4f" % LAMBDA_TARGET)

# ---------- 2. model: lambda*(A0) -------------------------------------------
folders = sorted(f for f in glob.glob(os.path.join(RD, PAT)) if not f.endswith("_abl"))


def face_data(folder):
    h = load_history_file(os.path.basename(folder))
    t = get_time_points(h)
    s = h.retrieve(float(t[-1])); s.arrange_sheet_from_history(); s.geom.update_all(s)
    fd = s.face_df
    A = fd["area"].to_numpy(float); P = fd["perimeter"].to_numpy(float)
    A0 = fd["prefered_area"].to_numpy(float); P0 = fd["prefered_perimeter"].to_numpy(float)
    K = fd["area_elasticity"].to_numpy(float); G = fd["contractility"].to_numpy(float)
    si = P0 / np.sqrt(A0)                 # per-cell shape index (HC/SC), preserved
    return A, P, K, G, si, float(A0[0])


sheets = [face_data(f) for f in folders]
A0_CUR = sheets[0][5]


def lam_for(A0_new, sheet):
    """Relaxed scale for one sheet at a candidate preferred area."""
    A, P, K, G, si, _ = sheet
    P0 = si * np.sqrt(A0_new)             # target perimeter follows the area
    def dE(l):
        return np.sum(K * (l * l * A - A0_new) * 2 * l * A + G * (l * P - P0) * P)
    lo, hi = 0.3, 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dE(lo) * dE(mid) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def lam_mean(A0_new):
    return float(np.mean([lam_for(A0_new, s) for s in sheets]))


print("\n" + "=" * 68)
print("MODEL  (best E17.5 re-fit so far; %d sheets; current A0=%.4f = %.3f*pi/4)"
      % (len(sheets), A0_CUR, A0_CUR / (np.pi / 4)))
print("=" * 68)
print("  %-12s %-12s %-10s %s" % ("A0", "A0/(pi/4)", "lambda*", "linear shrinkage"))
for frac in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60):
    a0 = frac * np.pi / 4
    l = lam_mean(a0)
    print("  %-12.4f %-12.2f %-10.4f %.2f%%" % (a0, frac, l, 100 * (1 - l)))

# solve for the A0 hitting the measured shrinkage
lo, hi = 0.10, np.pi / 4
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if lam_mean(mid) < LAMBDA_TARGET: lo = mid       # too much shrinkage -> raise A0
    else: hi = mid
A0_STAR = 0.5 * (lo + hi)
print("\n" + "=" * 68)
print("ANSWER")
print("=" * 68)
print("  measured linear shrinkage %.2f%%  ->  preferred area A0 = %.4f"
      % (SHRINK, A0_STAR))
print("             = %.4f * pi/4      (current setting is %.2f * pi/4)"
      % (A0_STAR / (np.pi / 4), A0_CUR / (np.pi / 4)))
print("             = %.3f x the mean cell area (%.4f)"
      % (A0_STAR / np.mean(sheets[0][0]), np.mean(sheets[0][0])))
lo_t, hi_t = 1 - (SHRINK + SEM) / 100, 1 - (SHRINK - SEM) / 100
def solve(target):
    a, b = 0.10, np.pi / 4
    for _ in range(80):
        m = 0.5 * (a + b)
        if lam_mean(m) < target: a = m
        else: b = m
    return 0.5 * (a + b)
print("  +-1 SEM on the measurement -> A0 in [%.4f, %.4f]  (%.2f - %.2f * pi/4)"
      % (solve(lo_t), solve(hi_t), solve(lo_t) / (np.pi / 4), solve(hi_t) / (np.pi / 4)))
