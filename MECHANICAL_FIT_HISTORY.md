# How the mechanical parameters were fitted

A step-by-step reconstruction of the v2 mechanical fit, from the scan scripts,
the six scan JSONs they wrote, and those files' timestamps.

Companion to `MECHANICAL_FIT_V2.md`, which is the authoritative statement of the
**method and results**. This document covers the **chronology** instead: the
order the steps were taken, the numbers each produced, why each one failed, and
the figures that show it. Where the two overlap, `MECHANICAL_FIT_V2.md` wins.

**On sourcing.** The per-step commits were squashed in the three-branch cleanup
on 23 Aug 2026 (`3469754 Baseline before the three-branch cleanup`), so git holds
no step-by-step history. Everything below is reconstructed from the scripts'
own docstrings and from the JSON each scan wrote, which do carry the numbers.

Figures are produced by `plot_mechanical_fit_history.py`, which reads only those
JSONs — nothing here is re-simulated or retyped.

---

## Chronology

| when | script | output |
|---|---|---|
| 12 Aug 14:04 | `grid_fit_mechanics_v2.py` | `grid_fit_mechanics_v2_E17.5.json` |
| 12 Aug 17:14 | `p0_from_e17_stiffness.py` | `p0_from_e17_stiffness.json` |
| — | `p0_gamma_scan.py` (step 5b) | **never ran** — no JSON; superseded by 5c |
| 13 Aug 07:20 | `p0_rgamma_scan.py` | `p0_rgamma_scan.json` |
| 13 Aug 15:50 | `p0_boundary_scan.py` | `p0_boundary_scan.json` |
| 13 Aug 19:03 | `selfconsistent_scan.py --stage P0` | `p0_selfconsistent_scan.json` |
| 14 Aug 01:54 | `selfconsistent_scan.py --stage E17.5` | `e17_selfconsistent_scan.json` |

The whole fit took about 36 hours of wall-clock time on a 32-core Azure VM.

*Figure: `fit_progression`* — the best objective reached at each step, per stage.

---

## The model, and what was held fixed

v2 is **pure contractility**: `FaceContractility` Γ/2·P², shape index p0 = 0,
**no bending**, no line tension. α_SC ≡ 1 by definition, so α_HC *is* R_alpha and
γ_HC/γ_SC *is* R_gamma.

Fixed in every run of every step (`grid_fit_mechanics_v2.py:70-81`):

| what | value |
|---|---|
| ablated cells | 337, 304, 65, 114 |
| post-ablation frame | −1 (the last) |
| type_by | `delta_level` |
| base `quasi_static_threshold` | 0.03 |
| ablation `quasi_static_threshold` | 0.02 |
| line tension | `None` |
| bending | 0.0 |
| shape index (global and per type) | 0.0 |
| initial sheets per stage | 10 |
| `no_differentiation` | True — cell types frozen during a mechanical run |

A scan *point* is not a run: it is 10 initial sheets, each simulated **twice** (a
base run plus an ablation run), pooled into one z per term. `n_sheets_ok` records
how many of the 10 survived.

### The targets

| term | E17.5 | P0 |
|---|---|---|
| HC:SC roundness ratio | 1.2453 ± 0.0541 (4.3%) | 1.1955 ± 0.0059 (0.5%) |
| HC:SC area change near ablation | 0.8142 ± 0.1494 | 0.8659 ± 0.0358 |
| cut shrinkage % | 7.5076 ± 0.5032 | 7.8065 ± 1.1382 |

z = (mean_model − mean_exp) / SEM_exp over per-experiment means; the objective is
Σz² over the three terms.

---

## Step 1 — A0 in closed form, never fitted

Minimising E = Σ[α_i/2 (λ²A − A0)² + γ_i/2 (λP)²] over the affine factor λ,
idealising every cell as a circle of diameter 1:

    A0 = (π/4) (λ² + 8 · avg_γ/avg_α),    λ = 1 − shrinkage% / 100

λ = **0.924924** at E17.5 (shrinkage 7.5076%), **0.921935** at P0 (7.8065%), both
read from the experimental data at run time rather than hardcoded.

The idealisation's sanity check: π/4 = 0.7854 against the actual packed cell area
400/508 = 0.7874 — a 0.3% match.

## Step 2 — the single-ratio assumption

α_HC/α_SC = γ_HC/γ_SC = R. Then avg_γ/avg_α collapses to γ_SC *exactly*, the
HC/SC counts cancelling, so **A0 depends on γ_SC alone**. This is what made the
fit two-dimensional and removed any need to iterate on A0.

It also imposes a hard ceiling. Cells must be stretched relative to their
preferred area — that is what puts the tissue under tension — so A0 < π/4, i.e.

    γ_SC < (1 − λ²)/8  =  0.018064 (E17.5),  0.018755 (P0)

confining A0 to [0.6719, 0.7854). The model therefore **cannot** reach the pre-v2
fit's A0 = 0.4657.

## Step 3 — the score

The three terms above, each as a z against the experimental mean and SEM;
objective = Σz². A degenerate term (no model data, or no usable experimental SEM)
is charged a large but finite worst-case n-sigma so the objective stays
comparable.

---

## Step 4 — the E17.5 coupled 5×5 grid

R ∈ {1.25, 1.75, 2.5, 3.5, 5.0} — geometric, ratio ≈ √2.
γ_SC ∈ {0.002, 0.006, 0.010, 0.014, 0.0175} — **linear**, because A0 is linear in
γ_SC, so this sweeps A0 evenly across its whole admissible band.

25 points × 10 sheets = 250 tasks, each a base run plus an ablation run.

| rank | point | objective | round z | abl z | shrink z |
|---|---|---|---|---|---|
| 1 | R = 3.5, γ_SC = 0.0175 | 4.103 | −1.15 | +1.03 | −1.31 |
| 2 | R = 2.5, γ_SC = 0.0175 | 4.944 | −1.95 | +1.08 | −0.01 |
| 3 | R = 2.5, γ_SC = 0.0140 | 5.318 | −1.96 | +1.07 | −0.56 |

Two results that shaped everything after:

- Roundness responds to **R** (spanning 0.10–0.18 across the R range) and
  essentially not at all to γ_SC (0.0065 across its whole range).
- The best point sits **on the upper γ_SC boundary** — the score wanted to leave
  the admissible band.

*Figure: `fit_step4_e17_grid`* — objective and roundness n-sigma over the grid,
with the winner boxed.

---

## Step 5 — P0's α_HC derived from stress, not fitted

Rather than refit P0 from scratch, α_HC was *derived*. Substituting the step-1 A0
into σ = α(A − A0) + 2πγ makes the γ terms cancel exactly:

    σ = (π/4) (1 − λ²) avg_α

so the measured stress/viscosity ratio pins avg_α, and contains no γ_SC at all.
The (1 − λ²) factor is what distinguishes stress from Young's modulus: the areal
modulus K = αA carries no such factor.

From `circular_ablation_raw_data(figure 3 +S4).xlsx`, column
"Stress over viscosity (1/min)":

| | value | n |
|---|---|---|
| E17.5 | 0.30896 ± 0.04566 | 14 |
| P0 | 0.20035 ± 0.01905 | 14 |
| ratio | 0.6485 ± 0.1140 | |
| × shrinkage correction (1−λ_E²)/(1−λ_P²) = 0.9632 | **k = 0.6246 ± 0.1098** | |

The stress column was used rather than the modulus column because it is the more
reliable measurement (SEM 18% vs 23%), at the cost of needing that correction.

f_HC measured over the 10 arrays — fixed, because `no_differentiation=True`:
**0.4932** (E17.5), **0.5213** (P0). Then
R_P0 = 1 + [k(1 + (R_E − 1) f_E) − 1] / f_P0:

| carried over from | avg_α E17.5 | avg_α P0 | R_P0 |
|---|---|---|---|
| R_E = 3.5 | 2.2331 | 1.3948 | **1.757 ± 0.470** |
| R_E = 2.5 | 1.7398 | 1.0867 | 1.166 ± 0.366 |

**Feasibility.** P0 is measured *softer*, so avg_α must fall; since
avg_α = 1 + (R−1)f_HC bottoms out at 1, R_P0 > 1 requires R_E > 2.218. That alone
eliminated the three lowest E17.5 R values.

γ_SC was held at E17.5's 0.0175, giving A0(P0) = 0.77752 (E17.5's was 0.78185 —
they differ only through each stage's own λ; forcing E17.5's A0 onto P0 would
break P0's shrinkage term).

**This failed badly.** R = 1.757 → objective **263.9**, roundness z = **−15.98**.
R = 1.166 → 658.8.

---

## Step 5c — the diagnosis: roundness is γ-driven

The conflict was explicit: hitting P0's roundness target needs R ≈ 3.9, but the
stress ratio derives 1.757, and under step 2's single-ratio assumption both
cannot hold.

The way out is that the stress constrains **avg_α alone**. So pin R_alpha = 1.757
— which keeps the stress match exact — and let R_gamma carry the roundness
contrast. The existing grid could not say which of α or γ roundness responds to,
because it moved them together. This scan asked that one question.

R_alpha = 1.757, γ_SC = 0.0175, R_gamma swept, with A0 recomputed at every point
from the decoupled form:

    avg_γ/avg_α = γ_SC (R_γ f + 1 − f) / (R_α f + 1 − f)
    A0 = (π/4)(λ² + 8 avg_γ/avg_α)

A0 co-varying with R_gamma is not a confound — it is what holds shrinkage matched
while the contrast moves.

| R_gamma | 1.10 | 1.25 | 1.40 | 1.55 | 1.757 | 1.85 | 1.94 |
|---|---|---|---|---|---|---|---|
| objective | 685.2 | 549.6 | 440.4 | 346.0 | 263.9 | 228.2 | 201.2 |
| roundness z | −26.0 | −23.2 | −20.8 | −18.4 | −16.0 | −14.8 | −13.9 |

Monotone: roundness **does** track R_gamma, so decoupling is the fix. But the
A0 < π/4 ceiling caps R_gamma at about 1.95 at γ_SC = 0.0175, far short of the
~4 required.

Fitting roundness = a + b·ln(R_gamma) gave **b = 0.1272** (the E17.5 grid
independently gives 0.1275 out to R = 5), predicting the target 1.1955 at
**γ_SC ≈ 0.0105, R_gamma ≈ 4**. That prediction is what the next two steps
chased — and it is where the answer landed.

**Also established here:** R_gamma > 1 is required, not conventional.
Contractility penalises perimeter, so a cell with higher Γ shrinks its perimeter
towards a circle; γ_HC > γ_SC is exactly what makes HC *rounder* than SC.
R_gamma < 1 predicts a roundness ratio below 1, when both stages measure ≈1.20.

*Figure: `fit_step5c_rgamma`* — objective and roundness against R_gamma, with the
ceiling and the step-5 point marked.

---

## Step 5d — walking the A0 = π/4 ceiling

Since the objective fell monotonically towards the ceiling at every γ_SC, the
optimum always sits *on* it. Rather than pay for a full 2-D (γ_SC, R_gamma) grid
whose interior was known to be worse, this walked the ceiling: for each γ_SC,
R_gamma set to the largest value still satisfying A0 < π/4 (margin 0.002, giving
A0 = 0.785162).

γ_SC ∈ {0.005, 0.006, 0.0075, 0.009, 0.0105, 0.0125, 0.0145, 0.0175, 0.021,
0.026}, geometric — because R_gamma ~ 1/γ_SC and roundness ~ ln(R_gamma), so log
spacing gives even coverage in the fitted quantity.

On the ceiling both A0 and avg_γ are constant, so this is a 1-D family at fixed
preferred area and fixed mean contractility along which only the HC/SC
contractility *split* changes. Every point is equally consistent with the
measured shrinkage and with the measured stress ratio.

**The prediction held, and everything else broke.** Roundness z went from −13.87
(R_gamma 1.94) through +2.24 (4.645) to +13.89 (9.096) — the crossing is near
R_gamma ≈ 4, where step 5c said it would be. But:

- shrinkage collapsed, z = −2.55 to −3.27 at low γ_SC;
- **5 of 10 points returned zero usable sheets** (objective infinite), and a
  sixth yielded only 4 sheets and no ablation term;
- best usable objective: 55.29.

The shrinkage collapse is the diagnostic one. Step 1's "every cell is a circle"
idealisation fails once α and γ decouple: at γ_SC = 0.005, R_gamma = 9.1 the
measured geometry was A_HC 0.672 / A_SC 0.805 and P_HC 2.970 / P_SC 3.687 —
nothing like identical cells.

*Figure: `fit_step5d_boundary`* — roundness found, shrinkage lost, dead points
marked.

---

## Steps 5e and 6 — A0 solved self-consistently

Minimising the same energy **without** assuming identical cells:

    A0 = λ² · Σ(α_i A_i²)/Σ(α_i A_i)  +  Σ(γ_i P_i²)/(2 Σ(α_i A_i))

which reduces to (π/4)(λ² + 8 avg_γ/avg_α) when every A_i, P_i is equal — it
*generalises* step 1 rather than replacing it. A_i and P_i come from the run, so
it is solved by iteration: A0 → run → measure → A0, seeded at π/4.

**Convergence.** P0 converged in **two** passes at every point; E17.5 took
**three**. (The script's docstring quotes the two-pass figure, which is the P0
result.) Shrinkage z was pulled from −3.3…−1.9 back to −0.23…−0.03 while
roundness moved by at most 0.0034 — A0 sets shrinkage, R_gamma sets roundness,
and the two are effectively orthogonal.

*Figure: `fit_a0_convergence`* — the A0 trail per point, per stage.

### Step 5e — P0 (R_alpha = 1.757)

| γ_SC | R_gamma | A0 | objective | round z | abl z | shrink z |
|---|---|---|---|---|---|---|
| **0.0105** | **3.8505** | **0.758542** | **9.837** | −0.94 | +2.99 | −0.15 |
| 0.0090 | 4.6453 | 0.752929 | 13.554 | +1.99 | +3.09 | −0.18 |
| 0.0125 | 3.0875 | 0.765816 | 32.184 | −4.82 | +2.99 | −0.10 |
| 0.0075 | 5.7580 | 0.746614 | 42.508 | +5.82 | +2.93 | −0.23 |
| 0.0145 | 2.5350 | 0.772913 | 83.059 | −8.62 | +2.97 | −0.03 |

### Step 6 — E17.5 (R_alpha = 3.5)

| γ_SC | R_gamma | A0 | objective | round z | abl z | shrink z |
|---|---|---|---|---|---|---|
| 0.0090 | 8.0417 | 0.735400 | 1.242 | +0.32 | +1.03 | +0.27 |
| **0.0105** | **6.7461** | **0.741812** | **1.276** | **−0.0025** | +1.08 | +0.32 |
| 0.0125 | 5.5024 | 0.749715 | 1.305 | −0.36 | +1.01 | +0.41 |
| 0.0145 | 4.6017 | 0.757245 | 1.731 | −0.68 | +1.02 | +0.48 |
| 0.0175 | 3.6367 | 0.768115 | 2.677 | −1.12 | +1.02 | +0.61 |

*Figure: `fit_selfconsistent`* — the three z terms against γ_SC for both stages,
with the chosen point marked.

---

## The final selection

`run_fitted_full_model.py:61` takes the best-scoring point of each stage's
self-consistent scan on **roundness + shrinkage only**, excluding the ablation
term — the model fails that term structurally at every parameter setting, so
including it adds a near-constant offset that can only add noise to the choice.

That is why E17.5's chosen point is **not** its total-objective argmin. On
roundness + shrinkage:

- γ_SC = 0.0105 → 0.000006 + 0.104185 = **0.104**
- γ_SC = 0.0090 → 0.101992 + 0.072323 = 0.174

The 0.0105 point nails the roundness ratio exactly (z = −0.0025). P0's pick is
unchanged either way (0.910 vs 3.985).

### The values the full model runs on

| | E17.5 | P0 |
|---|---|---|
| α_SC | 1 | 1 |
| α_HC = R_alpha | 3.5 | 1.757 |
| γ_SC | 0.0105 | 0.0105 |
| R_gamma = γ_HC/γ_SC | 6.746125 | 3.850522 |
| γ_HC | 0.070834 | 0.040430 |
| A0 (converged) | 0.741812 | 0.758542 |
| roundness n-σ | −0.0025 | −0.942 |
| ablation n-σ | +1.083 | +2.988 |
| shrinkage n-σ | +0.323 | −0.151 |
| **total χ²** | **1.276** | **9.837** |

Shared across both stages: shape index 0, bending 0, line tension `None`,
`atoh_sensitivity` 0.355079, `quasi_static_threshold` 0.03. Lateral inhibition is
identical at both stages: pS = 0.1, pR = 0.3, levels seeded U(0, 0.01).

Verified against the runs themselves: all 220 full-model folders per stage carry
exactly one mechanical tuple, matching the table above.

---

## Two caveats to carry into the write-up

**E17.5's α_HC = 3.5 is an assumption, not a fitted result.** It was carried over
from the coupled grid, where R was fitting roundness and was therefore really
acting as an R_gamma. `selfconsistent_scan.py` says so explicitly: at E17.5
roundness is γ-driven (α contributes ~4%), shrinkage is absorbed into A0, and the
only α-sensitive term is the ablation ratio, which no parameter moves. Since
R_alpha(E17.5) feeds R_alpha(P0) through the stress ratio, P0's 1.757 inherits
that assumption. `--r-alpha` exists to test what it changes.

**The ablation term never fits at P0.** It sits at roughly +3 sigma at every
parameter point in every scan (χ² 7.3–13.3 across the whole landscape), and is
the dominant contribution to P0's final total of 9.84.

A third, smaller point worth knowing: γ_SC = 0.0105 was never a grid value. It
entered only at steps 5e/6, where each γ_SC was visited exactly once with
R_gamma and A0 both determined by it. So restricting the tables to the
best-fitting γ_SC leaves exactly one point per stage — the chosen one.

---

## Reproducing the figures

    python plot_mechanical_fit_history.py
    python plot_mechanical_fit_history.py --only step4_grid

Writes, as png and svg, into the results directory:

| figure | shows |
|---|---|
| `fit_step4_e17_grid` | step 4: objective and roundness over (R, γ_SC) |
| `fit_step5c_rgamma` | step 5c: objective and roundness against R_gamma |
| `fit_step5d_boundary` | step 5d: roundness found, shrinkage lost, runs dying |
| `fit_selfconsistent` | steps 5e/6: the three z terms against γ_SC |
| `fit_a0_convergence` | the A0 → run → measure → A0 iteration |
| `fit_progression` | the best objective at each step |
