# Estimating circular-ablation shrinkage in the model

How the model's predicted shrinkage of a freely-relaxing cut disc is computed,
and how it is used to choose the cell **preferred area** `A0`.

*Written 2026-07-30. Reference implementation:*
`scratchpad/solve_preferred_area.py` (experiment + first-order solve) and
`scratchpad/sweep_preferred_area.py` (simulated sweep over `A0`).

---

## 1. What is being modelled

In the experiment a circular region of radius **R₀ = 60 µm** is cut free and
relaxes to a smaller radius. Cutting releases the mechanical constraint imposed
by the surrounding tissue, so the isolated piece relaxes to the size that
minimises **its own** energy. The measured shrinkage is therefore a readout of
the residual tension the tissue was under while intact.

The simulated tissue is periodic, so the box pins the mean cell area and the
tissue *cannot* relax. What follows estimates what it *would* do if released.

**Experimental value** (`Raw Data/circular_ablation_raw_data(figure 3 +S4).xlsx`,
sheet `Overall data`, column `Final radius (um)`, 28 ablations):

| stage | n | final radius | linear shrinkage |
|---|---|---|---|
| E17.5 | 14 | 55.50 µm | 7.51% ± 1.88% |
| P0    | 14 | 55.32 µm | 7.81% ± 4.26% |
| **pooled** | **28** | **55.41 µm** | **7.66% ± 0.61% (SEM)** |

The two stages are statistically indistinguishable, so they are pooled and a
single `A0` is used for both. Shrinkage is **linear in the radius**
(target relaxed scale `lambda* = 0.9234`).

---

## 2. Assumption: affine (uniform isotropic) relaxation

The released piece is assumed to rescale uniformly by a factor `lambda`.
Under `r -> lambda*r`:

* areas scale as `A -> lambda^2 * A`
* perimeters as `P -> lambda * P`
* **angles are unchanged**

`lambda` is then the single degree of freedom, fixed by minimising the energy.

---

## 3. The energy

Using the model's own functional, summed over cells:

```
E(lambda) = sum_i [ (K_i/2)(lambda^2 A_i - A0_i)^2
                  + (G_i/2)(lambda P_i - P0_i)^2 ]  +  E_bend
```

with per-cell area elasticity `K_i` (`area_elasticity`), contractility `G_i`
(`contractility`), target area `A0_i` (`prefered_area`) and target perimeter
`P0_i = s_i * sqrt(A0_i)` (`prefered_perimeter`, `s_i` = shape index). All are
read directly from the simulation's `face_df`, so nothing is re-derived by hand.

**The bending term drops out exactly.** `BoundaryBending` is
`sum kappa*(1 - cos theta)` over boundary-vertex angles, and angles are
invariant under uniform scaling, so `dE_bend/dlambda = 0`. It cannot influence
`lambda*`.

---

## 4. Solving

Setting `dE/dlambda = 0`:

```
sum_i [ K_i (lambda^2 A_i - A0_i)(2 lambda A_i)
      + G_i (lambda P_i - P0_i) P_i ] = 0
```

which collects into a depressed cubic (no `lambda^2` term):

```
2 (sum K A^2) lambda^3
  + (sum G P^2 - 2 sum K A A0) lambda
  - (sum G P P0)  =  0
```

Solved by bisection on `lambda` in [0.3, 1.5] (the derivative is monotonic
there, so the root is unique), per sheet, then averaged across sheets. Outputs:

* **linear** shrinkage `= (1 - lambda*) * 100%`  <- what the experiment measures
* areal shrinkage `= (1 - lambda*^2) * 100%`
* a cut circle of radius `R` becomes `R * lambda*`

---

## 5. The physics it captures

The two terms **compete**, which is the whole story:

* **Area term** — with `A0 < A` (cells compressed relative to target) it
  *drives* contraction.
* **Perimeter term** — with `P0 > P` (cells want longer junctions than they
  have) it *resists* contraction.

Worked example at `A0 = 0.9 * pi/4` (best E17.5 re-fit point: gammaSC 0.246,
alphaHC 1.063, hc_p0 3.946, sc_p0 4.644): the area term alone would give 4.0%
linear shrinkage, but SC cells (outnumbering HCs ~2.5:1) have `P0 ~ 3.91`
against an actual `P ~ 3.66`, and their resistance pulls the net down to
**1.44%** — far below the measured 7.66%.

This also explains why lowering `A0` is **doubly** effective: it increases area
tension *and*, through `P0 = s*sqrt(A0)`, removes perimeter resistance. Hence
the sensitivity is much steeper than an area-only argument suggests
(~3% more shrinkage per 0.05 drop in `A0/(pi/4)`).

---

## 6. Limitations

1. **Affine assumption.** Real relaxation is non-affine — cells rearrange and
   relax unequally. This is the largest approximation.
2. **No topology changes.** T1/T2 transitions during relaxation are ignored.
3. **Free-edge effects.** Cells at the cut boundary lose neighbours, so their
   tension differs. For a 60 µm disc holding many cells the boundary is a
   modest fraction, but not zero.
4. **Equilibrium only.** It predicts the final relaxed state, not the kinetics —
   appropriate here, since the comparison is to the *final* radius. (The
   spreadsheet's Young's-modulus/viscosity columns describe the transient, about
   which this calculation says nothing.)
5. **"Before" state is the periodic equilibrium**, itself a model output.
6. **Purely passive** — no active or biological response to wounding.

Because of (1)-(3) the absolute number should be treated as accurate to order
unity; it is used mainly for **relative** comparison across `A0`, which is what
selects the preferred area.

---

## 7. Two levels of the calculation

* **First-order** (`solve_preferred_area.py`): holds the cell geometry fixed at
  the values equilibrated for one particular `A0`, and solves for the `A0` that
  reproduces the measured shrinkage. Fast, but ignores that changing `A0`
  changes the equilibrium shapes (`P0` moves with `sqrt(A0)`).
* **Simulated sweep** (`sweep_preferred_area.py`): re-runs the tissue relaxation
  at each candidate `A0` and applies the same `lambda*` calculation to each
  **re-equilibrated** tissue, also reporting roundness so the cost to the
  roundness fit is visible. This removes the fixed-geometry limitation, but not
  limitations (1)-(3).

Making the number quantitative rather than comparative would require explicitly
simulating a cut disc with a free boundary — a substantially bigger piece of
work.

### Sweep result (2026-07-30, best E17.5 point, 2 arrays per value)

Shape indices held at `hc 3.946 / sc 4.644`:

| A0 / (pi/4) | lambda* | shrinkage | HC round | SC round |
|---|---|---|---|---|
| 0.70 | 0.918 | 8.24% | 0.864 | 0.803 |
| 0.75 | 0.943 | 5.72% | 0.862 | 0.782 |
| 0.79 | 0.958 | 4.22% | 0.861 | 0.744 |
| 0.83 | 0.970 | 2.98% | 0.841 | 0.712 |
| 0.87 | 0.979 | 2.14% | 0.818 | 0.669 |
| 0.90 | 0.985 | 1.50% | 0.799 | 0.642 |
| *experiment* | 0.923 | **7.66%** | **0.804** | **0.649** |

Two things to note.

**(a) The first-order estimate was optimistic.** It predicted `0.79*pi/4` would
give 7.66%; the re-equilibrated tissue delivers only 4.22% there. Letting cells
re-arrange relieves much of the tension, so a lower `A0` is required.

**(b) At FIXED shape indices, shrinkage and roundness cannot both be matched.**
Roundness matches at `0.90*pi/4` (HC 0.799/0.804, SC 0.642/0.649) but gives only
1.5% shrinkage; shrinkage needs ~`0.71*pi/4`, where SC roundness has run away to
0.80. The two observables pull `A0` in opposite directions.

**Resolution — move `A0` and the shape index TOGETHER.** Lowering `A0` also
lowers `P0 = s*sqrt(A0)`, and it is that drop which over-rounds the cells.
Raising `s` to hold `P0` restores roundness while the smaller `A0` still supplies
the tension. Solving with `P0` held fixed gives `A0 = 0.593*pi/4` with the shape
indices scaled by 1.232 (hc 3.95 -> 4.86, sc 4.64 -> 5.72), **confirmed by direct
simulation** (2 arrays):

| | model | experiment |
|---|---|---|
| linear shrinkage | **7.65%** | 7.66% |
| HC roundness | 0.792 | 0.804 |
| SC roundness | 0.646 | 0.649 |

This is the configuration used by `run_refit.py` (folder tag `_pa0.466`).

**Consequence for interpreting the shape index:** because it is defined against
the PREFERRED area (`s = P0/sqrt(A0)`) and `A0` (0.466) now sits well below the
actual cell area (~0.766), the fitted `s` (~4.9-5.7) is NOT directly comparable
to the `P/sqrt(A)` measured on real cells (~4.0). The dimensionless quantity to
compare with experiment is the resulting ROUNDNESS, which does match.

---

## 8. Why this matters for the fit

`A0` is not otherwise strongly constrained by the roundness/ablation objectives,
but it sets the tissue's residual tension — and, before the L0-normalization fix,
an inflated `A0` (`pi/4 * L0^2`, ~12x the real cell area) left the tissue
*compressed*, so it would have **expanded** on a cut, opposite to experiment.
The circular-ablation data is what pins `A0` down. See also
`memory/preferred-area-is-pi-over-4.md`.
