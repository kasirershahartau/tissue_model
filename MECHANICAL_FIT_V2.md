# Mechanical fit v2 — method and results

Fits the HC/SC mechanical parameters of the periodic vertex model to the
E17.5 and P0 utricle, then hands them to the full (differentiation) model.
Companion to `SHRINKAGE_ESTIMATE_METHOD.md`.

## 1. The model

Per cell, area elasticity plus **face contractility** (shape index p0 = 0), and
**no bending elasticity**:

    E_i = (alpha_i / 2) (A_i - A0)^2  +  (gamma_i / 2) P_i^2

with `alpha_SC = 1` fixing the energy scale. Free parameters:

| symbol | meaning |
|---|---|
| `gamma_SC` | SC contractility |
| `R_gamma = gamma_HC / gamma_SC` | contractility contrast |
| `R_alpha = alpha_HC / alpha_SC` | area-stiffness contrast |
| `A0` | preferred area (not fitted — see §3) |

Runs use `no_differentiation=True`, so the HC fraction is fixed by each initial
array and its saved delta threshold (f_HC = 0.4932 at E17.5, 0.5213 at P0).

## 2. The score

Three terms, each `z = (mean_model - mean_exp) / SEM_exp`, with the experimental
mean and SEM taken over **per-experiment means**. Objective = sum of z².

| term | E17.5 | P0 |
|---|---|---|
| HC/SC roundness ratio | 1.2453 ± 0.0541 (4.3%) | 1.1955 ± 0.0059 (0.5%) |
| HC/SC area change after ablation | 0.8142 ± 0.1494 (18.4%) | 0.8659 ± 0.0358 (4.1%) |
| cut shrinkage (%) | 7.5076 ± 0.5032 | 7.8065 ± 1.1382 |

The SEM columns matter as much as the means: P0's roundness SEM is 9x tighter
than E17.5's, which is why the two stages constrain the fit so unequally.

## 3. Step 1 — A0 follows from the shrinkage

Minimising the affine-relaxation energy `E(l) = sum_i [alpha_i/2 (l^2 A_i - A0)^2
+ gamma_i/2 (l P_i)^2]` over the linear factor `l` and setting `l = lambda` (the
measured shrinkage) gives the stationarity condition

    A0 = lambda^2 * sum(a_i A_i^2)/sum(a_i A_i)  +  sum(g_i P_i^2)/(2 sum(a_i A_i))

If every cell is idealised as a circle of **diameter 1** (A = pi/4, P = pi) this
collapses to the closed form used through steps 2–5d:

    A0 = (pi/4) (lambda^2 + 8 * avg_gamma / avg_alpha)

**The idealisation fails once alpha and gamma are decoupled.** At gamma_SC =
0.005, R_gamma = 9.1 the real geometry is A_HC 0.672 / A_SC 0.805 and P_HC 2.970
/ P_SC 3.687, and the closed form overstates A0 by 6.2% — leaving cells
under-stretched and the shrinkage term at z = -3.3. Using the exact condition
instead (A_i, P_i measured from the run, so A0 is solved **self-consistently**:
A0 -> run -> measure -> A0) converges in 2–3 passes and returns every point to
|z| < 0.7. See `selfconsistent_scan.py`.

## 4. Step 2 — the coupling assumption, and dropping it

The original scheme assumed one contrast, `R = R_alpha = R_gamma`, which made
`avg_gamma/avg_alpha` collapse to `gamma_SC` exactly (the cell counts cancel) and
removed the self-consistency. It was dropped because it forced HC and SC to share
a preferred area and could not satisfy P0.

## 5. What each observable actually constrains

This is the central result of the exercise, and it was not obvious in advance.

| observable | constrains | evidence |
|---|---|---|
| shrinkage | `A0` | z < 0.7 at every point once A0 is self-consistent |
| roundness | `R_gamma` | at P0, gamma-only slope 0.0907/unit vs coupled 0.0948 — **alpha contributes 4%** |
| ablation | `R_alpha` (weakly) | the only alpha-sensitive term, but saturating |
| stress/viscosity | `avg_alpha` **ratio between stages** | see §6 |

Two quantitative regularities fell out:

* **Roundness is logarithmic in the contrast**: `roundness = a + b ln(R_gamma)`,
  with b = 0.1272 (P0) and 0.1275 (E17.5 coupled grid, out to R = 5), residuals
  < 0.0012. Do not extrapolate on a local linear slope — the falling slope looks
  like saturation but there is no asymptote.
* **A0 and R_gamma are orthogonal in effect**: correcting A0 by 5% moved
  roundness by at most 0.0034 while fixing shrinkage entirely.

## 6. The stress ratio (used as an input)

Substituting the step-1 A0 into `sigma = alpha (A - A0) + 2 pi gamma` makes the
gamma terms cancel exactly, leaving

    sigma = (pi/4) (1 - lambda^2) avg_alpha

so stress is *stiffness x strain*, and the strain is the measured shrinkage
itself. Comparing stages therefore needs the shrinkage correction:

    k = avg_alpha_P0 / avg_alpha_E17.5
      = (stress_P0 / stress_E17.5) * (1 - lam_E^2)/(1 - lam_P^2)
      = 0.64847 * 0.96321 = 0.62461

    R_alpha(P0) = 1 + [k (1 + (R_alpha(E17.5) - 1) f_E) - 1] / f_P0

Without the correction R_alpha(P0) comes out 5.8% too stiff. The areal modulus
`K = alpha A` carries no such factor — only the residual stress does.

## 7. Results

Best point per stage (`selfconsistent_scan.py`, scored on roundness + shrinkage):

| stage | gamma_SC | R_gamma | R_alpha | A0 | A0/(pi/4) | z rnd | z abl | z shr |
|---|---|---|---|---|---|---|---|---|
| E17.5 | 0.0105 | 6.746 | 3.5 | 0.74181 | 0.9445 | −0.00 | +1.08 | +0.32 |
| P0 | 0.0105 | 3.851 | 1.757 | 0.75854 | 0.9658 | −0.94 | +2.99 | −0.15 |

Roundness crossings and their 1-sigma ranges, from `roundness = a + b ln R_gamma`:

| stage | R_gamma* | 1 SEM | 1-sigma range |
|---|---|---|---|
| E17.5 | 6.73 | x/ 1.74 | [3.88, 11.70] |
| P0 | 4.11 | x/ 1.06 | [3.88, 4.35] |

**The two stages' R_gamma are not significantly different.** P0's 4.11 scores
z = −0.89 against E17.5 (compatible); E17.5's 6.73 scores z = +8.59 against P0
(excluded). A single R_gamma ≈ 4.1 fits both stages within 1 sigma. The apparent
6.8 -> 4.1 drop is an artefact of E17.5's 9x wider roundness SEM. A stage
difference in mechanics *is* measured — `avg_alpha` falls by 0.625 — but it is in
alpha, not in the contractility contrast.

## 8. What the fit does not determine

* **gamma_SC.** Roundness fixes R_gamma; shrinkage is absorbed into A0 at any
  point. So an iso-roundness curve of equally good fits runs through
  (gamma_SC, R_gamma) space. The quoted 0.0105 is where the *scanned family*
  crosses, not a measurement — and the two stages agreeing on it is an artefact
  of both families being constructed the same way. Do not report it as a finding.
* **R_alpha, above ~3.5.** The ablation term is the only alpha-sensitive one and
  it saturates: on the E17.5 grid the model value moves −0.060 per unit R at
  R ≈ 1.5 but −0.0008 by R = 5. So R_alpha is bounded *below*, not measured, and
  R_alpha(P0) inherits that through §6 (R_E 3.5 -> R_P0 1.76; R_E 5 -> R_P0 2.64).
  The choice R_alpha(E17.5) = 3.5 was checked to sit at the knee in the decoupled
  configuration (ablation z 1.02, matching the coupled grid's 1.028).

## 9. The failed prediction

The ablation term is not fitted, and cannot be: **the model predicts HC
*expansion* where the data show HC *contraction*.**

| | HC | SC |
|---|---|---|
| model (P0) | +3.5% | +6.6% |
| experiment P0 | −1.8% | +13.3% |
| experiment E17.5 | −4.7% | +4.7% |

The stiffness contrast does act — HC change area half as much as SC, as intended
— but in a fixed-area periodic box the freed area must be absorbed by the
neighbours, and stiffness only sets each cell's *share* of it. A share can be
made small; it cannot be made negative. At equilibrium every cell shares a
pressure, `A_i = A0 + (Pi - 2 pi gamma_i)/alpha_i`, so `dA_i = dPi / alpha_i`:
one sign for all cells, magnitude set by alpha alone.

The metric compounds this. It is a ratio of numbers near 1,
`(1 + d_HC)/(1 + d_SC) ≈ 1 − d_SC (1 − c)` with `c = d_HC/d_SC`, so its distance
from 1 is the *product* of contrast and response magnitude. Raising R_alpha
improves c but stiffens the tissue so d_SC falls, and the two nearly cancel:
R = 2.5 -> 3.5 improved c by 33% and moved the score 0.4%. Reaching the P0 target
would need d_SC ≈ 15.5% against the model's 5–7%.

Both stages miss in the same direction (model ≈ 0.967 / 0.973 vs measured 0.814 /
0.866), so this is one qualitative prediction the model gets wrong at both
stages, not a stage-specific failure. Reproducing it would need a modelling
change — total area not conserved (free boundary or tissue-level contraction),
or an active HC response to wounding — not a different parameter.

## 10. Scripts

| script | role |
|---|---|
| `grid_fit_mechanics_v2.py` | the original coupled 5x5 grid (E17.5); also hosts `run_task_pool` |
| `p0_from_e17_stiffness.py` | the stress-ratio derivation of R_alpha(P0) |
| `p0_rgamma_scan.py` | the diagnostic that showed roundness is gamma-driven |
| `p0_boundary_scan.py` | the A0 = pi/4 family (superseded by the self-consistent A0) |
| `selfconsistent_scan.py` | **the fit** — either stage, exact A0, optional R_alpha scan |
| `plot_selfconsistent_scan.py` | the result figure, per stage |
| `run_fitted_full_model.py` | hands the fitted mechanics to the full model at psigma = 0 |

Worker pools are **recycled** every `3 x workers` tasks (`run_task_pool`). A
single long-lived `ProcessPoolExecutor` leaks memory across tasks and dies on
whatever is scheduled last, which looks exactly like a parameter-dependent
failure — it cost two runs before being diagnosed. `max_tasks_per_child` would be
the clean fix but needs Python 3.11; this runs on 3.10.
