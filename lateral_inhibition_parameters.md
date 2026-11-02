# Literature values for the lateral-inhibition length scale L0

Started 2026-10-26, after the collaborator review flagged that

    L0 = kappa * beta_D / gamma_R^2 ,    kappa = kappa_plus * kappa_s / (kappa_minus + kappa_s)

was *set* to the average cell perimeter rather than checked against measured
values.

**Result: it should not have been. With published rates L0 exceeds a cell
perimeter by 2-5 orders of magnitude, and the most conservative combination of
literature values still gives ~50x.** Derivation in §6, uncertainty budget in §7.

Only peer-reviewed journal articles are used. Each carries a link.


## 1. What the dimensions demand

Following Shaya et al. 2017, the source of our model:

| symbol | meaning | units |
|---|---|---|
| `n_ij`, `d_ij` | Notch / Delta on the boundary between cells i and j | molecules · length<sup>-1</sup> |
| `kappa_plus` | Notch-Delta association | length · molecule<sup>-1</sup> · time<sup>-1</sup> |
| `kappa_minus` | complex dissociation | time<sup>-1</sup> |
| `kappa_s` | conversion of complex into signal (S2/S3 cleavage) | time<sup>-1</sup> |
| `kappa` | `kappa_plus * kappa_s / (kappa_minus + kappa_s)` | length · molecule<sup>-1</sup> · time<sup>-1</sup> |
| `beta_D`, `beta_N` | maximal production, **whole cell** | molecules · time<sup>-1</sup> |
| `gamma_R` | repressor degradation | time<sup>-1</sup> |

Then `[L0] = (L/(M*T)) * (M/T) * T^2 = L`, a genuine length.

Where it comes from: scaling `t' = gamma_R*t`, `n' = n/nu`, `l' = l/L0` and
requiring the complex-formation term to have unit coefficient gives
`nu = gamma_R/kappa`, and the production term then carries the dimensionless group

    beta_hat = beta * kappa / (L0 * gamma_R^2)

so **L0 is exactly the length that makes `beta_hat = 1`.** That is the cleanest
statement of what the old derivation assumed: it asserted L0, when it should
have asserted `kappa` and let L0 follow.

A useful regrouping: `L0 = (kappa*beta_D/gamma_R)/gamma_R` — **the distance the
signalling front advances in one repressor lifetime.** Suggestive that it might
land near a cell perimeter; it does not.


## 2. The assumption is inherited, not ours

Shaya et al. 2017 non-dimensionalise with the same choice. From their STAR
Methods: *"Where L̂ is the average cell perimeter (averaged over all the cells in
the simulated lattice)."* Their Fig. 4B parameters (`betaN = 3.9`,
`betaD = 3.9`, `betaR = 194.1`) are dimensionless and not traced to measured
rates. Sprinzak et al. 2011 likewise write their lateral-inhibition equations
(Eqs. 7-9) directly in dimensionless form and **assume the Notch and DSL
degradation rates equal** *"for simplicity (no loss of generality for the steady
state solutions)"*.

**I found no published measurement or estimate of `kappa*beta_D/gamma_R^2`.**

- Shaya O, Binshtok U, Hersch M, et al., Sprinzak D. *Cell-cell contact area
  affects Notch signaling and Notch-dependent patterning.* Dev Cell 40:505-511.e6 (2017).
  [PMC5435110](https://pmc.ncbi.nlm.nih.gov/articles/PMC5435110/) · doi:10.1016/j.devcel.2017.02.009
- Sprinzak D, Lakhanpal A, LeBon L, Garcia-Ojalvo J, Elowitz MB. *Mutual
  inactivation of Notch receptors and ligands facilitates developmental
  patterning.* PLoS Comput Biol 7:e1002069 (2011).
  [open access](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002069)

Useful measurements in Shaya et al.: Dll1 diffusion length 1-2 µm (so signal ∝
contact area); signal scales over contact widths 1-40 µm; TEC decay half-life
~60 min; reporter delay 255 ± 29 min.


## 3. The rate constants — Khait et al. 2016

Same group, same model lineage. Their Fig. 3C caption publishes the rate set and
their supplement gives the full provenance of every number.

- Khait I, Orsher Y, Golan O, Binshtok U, Gordon-Bar N, Amir-Zilberstein L,
  Sprinzak D. *Quantitative analysis of Delta-like 1 membrane dynamics
  elucidates the role of contact geometry on Notch signaling.*
  Cell Rep 14(2):225-233 (2016).
  [doi:10.1016/j.celrep.2015.12.040](https://doi.org/10.1016/j.celrep.2015.12.040)

Their model is ours: `k+`/`k-` form the complex `[NDl]`, which is processed at
`k_s` to give signal `S = k_s * integral([NDl]) d2r`.

### Measured (FRAP-TIRF, CHO-Dll1-mCherry, n = 117 cells)

| quantity | value |
|---|---|
| Dll1 diffusion coefficient | **D = 0.038 ± 0.024 µm²s<sup>-1</sup>** (MDCK 0.066 ± 0.04) |
| Dll1 endocytosis rate | **K_endo = 0.018 ± 0.009 s<sup>-1</sup>** (MDCK 0.029 ± 0.019) |
| diffusion-length scale | lambda = sqrt(D/K_endo) = **1.56 ± 0.64 µm** |
| Dll1 lacking ICD | D = 0.0069 µm²s<sup>-1</sup> |
| Dll1 in filopodia | D = 0.060 ± 0.034 µm²s<sup>-1</sup> |

`K_endo` is effectively **`gamma_D` for surface Dll1: a 38 s membrane half-life.**

### Derived, with provenance (their "Parameters estimation", suppl. pp. 9-10)

**Production.** Transmembrane-protein surface densities span **1-1000 µm<sup>-2</sup>**
(McCloskey & Poo 1986; Waters et al. 1990; Shimizu et al. 1999). Within that
range they *chose* `N_ss = 100` and `Dl_ss = 10 µm^-2` (wanting `Dl0 < N0`),
then used the measured `k_endo` to get

    Dl0*k_exo = 10 * 0.02 = 0.2 µm^-2 s^-1      N0*k_exo = 2 µm^-2 s^-1

So the production rates are **a choice inside a 3-decade literature range**,
multiplied by a measured turnover rate.

**Binding.** Starting from the *measured* 3-D affinity of soluble Dll1 for
Notch1, `k_d3D = 3.4 µM` (Andrawes et al. 2013 — consistent with Luca et al.'s
7.5-12.7 µM for DLL4), and a *typical* protein on-rate
`k+_3D = 1e5 M^-1 s^-1` (Schlosshauer & Baker 2004):

    k_off  = k_d3D * k+_3D = 0.34 s^-1
    k_d2D  = lambda_d * k_d3D = 2.04 µm^-2      (lambda_d ~ 1 nm, membrane thickness; Wu et al. 2011, 2013)
    k+_2D  = k_off / k_d2D = 0.167 µm^2 s^-1
    k_s    = k_off = 0.34 s^-1                  (they assume k_s >> k-)
    k-     = k_s / 10 = 0.034 s^-1              (a choice, not a measurement)

**The key structural fact:** substituting, `k+_2D = k_d3D*k+_3D / (lambda_d*k_d3D)
= k+_3D / lambda_d`. **The measured affinity cancels.** And since `k_s >> k-`,

    kappa_2D = k+ * k_s / (k- + k_s) = 0.167 * 0.34 / 0.374 = 0.152 µm^2 s^-1 ~= k+_2D

so **`kappa` is essentially `k+_3D / lambda_d`** — it rests on a *generic*
protein association rate and a 1 nm membrane thickness, and **not at all on the
measured Notch-Delta affinity.** That, not the affinity, is the number to attack.

**Can the generic on-rate be replaced by a measured one? No — but it checks
out.** Luca et al.'s supplement settles this. Their Kd values come from a
*steady-state* 1:1 Langmuir fit, not a kinetic fit, so **no wild-type rate
constants were ever extracted.** The raw sensorgrams (their fig. S3 — the main
text's pointer to "fig. S2" is a mislabel; S2 is the luciferase assay) show why:
for wild-type DLL4(N-EGF2) against all three Notch1 constructs, association
saturates and dissociation returns to baseline essentially within the
instrument's response, whereas the affinity-matured variants show visible decay
over the 60 s dissociation phase. Unresolvable dissociation puts
`k_off` at roughly **>= 0.1-1 s^-1**, and the square-wave association shape at
46 nM - 33 µM is what `k+_3D ~ 1e5 M^-1 s^-1` with `k_off ~ 0.34 s^-1` predicts
(`k_obs = k+*C + k_off` = 0.34-3.6 s<sup>-1</sup> across that range).

So Khait's assumed `k_off = 0.34 s^-1` is **corroborated** by the measured
sensorgram shapes, and `k+_3D = 1e5` is consistent with them — but not pinned.
Pinning it needs a different technique (stopped-flow, or direct 2-D
measurements on apposed membranes), not this dataset.

- Andrawes MB, Xu X, Liu H, et al., Blacklow SC. *Intrinsic selectivity of Notch 1
  for Delta-like 4 over Delta-like 1.* J Biol Chem 288:25477-25489 (2013).
  [doi:10.1074/jbc.M113.454850](https://doi.org/10.1074/jbc.M113.454850)
- Schlosshauer M, Baker D. Protein Sci 13:1660-1669 (2004).
  [doi:10.1110/ps.03517304](https://doi.org/10.1110/ps.03517304)
- Wu Y, Vendome J, Shapiro L, Ben-Shaul A, Honig B. *Transforming binding
  affinities from three dimensions to two with application to cadherin
  clustering.* Nature 475:510-513 (2011). [doi:10.1038/nature10183](https://doi.org/10.1038/nature10183)
- Wu Y, Honig B, Ben-Shaul A. Biophys J 104:1221-1229 (2013).
  [doi:10.1016/j.bpj.2013.02.009](https://doi.org/10.1016/j.bpj.2013.02.009)
- McCloskey MA, Poo MM. *Rates of membrane-associated reactions: reduction of
  dimensionality revisited.* J Cell Biol 102:88-96 (1986).
  [doi:10.1083/jcb.102.1.88](https://doi.org/10.1083/jcb.102.1.88)


## 4. `gamma_R` — repressor degradation

| value | half-life | source |
|---|---|---|
| 0.0315 min<sup>-1</sup> | **22.0 min** | Hes1 protein, fibroblasts |
| 0.029 min<sup>-1</sup> | 23.9 min | *hes1* mRNA |
| — | ~16 min | Hes1 protein, embryonic stem cells |

- Hirata H, Yoshiura S, Ohtsuka T, et al. Science 298:840-843 (2002).
  [doi:10.1126/science.1074560](https://doi.org/10.1126/science.1074560)
- Kobayashi T, Mizuno H, Imayoshi I, et al. Genes Dev 23:1870-1875 (2009).
  [full text](https://genesdev.cshlp.org/content/23/16/1870.full)

**`gamma_R` = 0.0315 min<sup>-1</sup> = 5.25e-4 s<sup>-1</sup>.** L0 goes as
`1/gamma_R^2`, so the 15-28 min spread alone moves L0 by ~3x.


## 5. Other rates

- Agrawal S, Archer C, Schaffer DV. PLoS Comput Biol 5:e1000390 (2009).
  [PMC2680760](https://pmc.ncbi.nlm.nih.gov/articles/PMC2680760/)

| parameter | value | half-life | primary source |
|---|---|---|---|
| full-length Notch1 | 0.017 min<sup>-1</sup> | **41 min** | [Logeat et al. 1998](https://doi.org/10.1073/pnas.95.14.8108) |
| NICD | 0.0014 / 0.00385 min<sup>-1</sup> | 8.3 h / 3.0 h | [Foltz et al. 2002](https://doi.org/10.1016/s0960-9822(02)00888-6) |
| Notch1 mRNA | 0.0058 min<sup>-1</sup> | 120 min | [Kessler & Delany 2007](https://doi.org/10.1210/en.2006-0443) |
| maximal transcription, Notch1 | 5.5e-11 M·min<sup>-1</sup> | ≈66 molecules·min<sup>-1</sup> | — |

Affinity (SPR, wild-type DLL4(N-EGF2) vs Notch1): Kd = **7.5 µM** (EGF11-13),
8.6 µM (10-14), 12.7 µM (1-14) — all from a **steady-state** 1:1 Langmuir fit
(suppl. p. 5), so these are equilibrium constants only. The affinity-matured
variants (56-540 nM) are **engineered** and must not be used as wild-type.

- Luca VC, Jude KM, Pierce NW, Nachury MV, Fischer S, Garcia KC. Science
  347:847-853 (2015). [PMC4445638](https://pmc.ncbi.nlm.nih.gov/articles/PMC4445638/)


## 6. The estimate

Khait works in 2-D surface densities, our model in line densities along the
junction. Let `w` be the width of the reactive junctional band and `W = A/L` the
effective width over which surface density is defined. Then

    kappa_1D       = kappa_2D * w / W^2
    beta_D (total) = (surface production) * A = 0.2 * L * W

and

    L0 / L = kappa_2D * (Dl0*k_exo) * (w/W) / gamma_R^2

The geometric factor is only the **ratio** `w/W <= 1`, not either width
separately — which is what makes this defendable, since neither has been
measured. Taking `w = W` (all Delta in the reactive band) gives the upper case:

    kappa_2D * 0.2 = 0.152 * 0.2 = 3.04e-2 s^-2
    gamma_R^2      = (5.25e-4)^2 = 2.76e-7 s^-2

    L0 / L = 1.1e5

### Three equivalent statements

1. **L0 is ~10^5 cell perimeters** — of order 1 m for L ≈ 10 µm, not a cellular
   length at all.
2. **Setting L0 = perimeter implies a repressor half-life of ~4 s.** Requiring
   `L0 = L` forces `gamma_R = sqrt(0.0304) = 0.174 s^-1`, i.e. `ln2/0.174 = 4.0 s`.
   Hes1's is 22 min — **330x slower**, and L0 goes as `1/gamma_R^2`: hence 10^5.
3. **The dimensionless production should be ~10^5, not ~4.** Keeping the average
   perimeter as the length unit, `beta_hat_D = L0_true/L̂ ~ 1e5`, whereas Shaya
   et al. run at `beta_hat_D = 3.9` and we inherited that regime.


## 7. Uncertainty budget

Every soft input, pushed in the direction that *shrinks* L0:

| input | central | conservative | effect on L0 |
|---|---|---|---|
| `k+_3D` (generic on-rate) | 1e5 M<sup>-1</sup>s<sup>-1</sup> | 1e4 | ÷10 |
| `lambda_d` (2D/3D conversion) | 1 nm | 10 nm | ÷10 |
| `Dl_ss` (chosen in a 1-1000 µm<sup>-2</sup> range) | 10 µm<sup>-2</sup> | 1 | ÷10 |
| `gamma_R` half-life | 22 min | 15 min | ÷2.2 |
| `w/W` geometric factor | 1 | 0.1 | ÷10 |

All five at once:

    k+_2D = 1e4 M^-1 s^-1 / 10 nm = 1.66e-3 µm^2 s^-1
    L0/L  = 1.66e-3 * 0.02 * 0.1 / 5.9e-7 ~= 5.6

and without the `w/W` factor, ~56. **Even stacking every conservative choice,
L0 stays at least ~5-50 perimeters, and the honest central value is 10^4-10^5.**
Going the other way (`Dl_ss` = 1000 µm<sup>-2</sup>) pushes it to 10^7.

The conclusion is not sensitive to the measured Notch-Delta affinity, which
cancels out of `kappa` entirely (§3).

### A second, independent inconsistency

The models assume `gamma_N = gamma_D = gamma_R`. Measured:

| | rate | half-life |
|---|---|---|
| Dll1, surface (K_endo) | 0.018 s<sup>-1</sup> | 38 s |
| Notch1, full-length | 2.8e-4 s<sup>-1</sup> | 41 min |
| Hes1 (repressor) | 5.25e-4 s<sup>-1</sup> | 22 min |

`gamma_D/gamma_R ~= 34`. The quasi-steady-state treatment of Delta is probably
safe; the equal-rate constraint is not.


## 8. Still open

| quantity | status |
|---|---|
| `kappa_plus`, `kappa_minus` measured directly | **Closed, negatively.** Luca et al. fitted Kd at steady state; wild-type association and dissociation are both too fast for the SPR to resolve (their fig. S3), so no measured rate constants exist. The generic `k+_3D = 1e5` that `kappa` rests on is corroborated but not replaceable from this dataset |
| `Dl_ss` for utricle supporting cells | the single widest source of uncertainty (3 decades); a quantitative immunostain would close it |
| Hes5 half-life | not found; may matter more than Hes1 here, and L0 goes as `1/gamma_R^2` |
| Sprinzak 2011 Table S1 | separate supplementary PDF |


## 9. Where this leaves the model

Assert `kappa` (equivalently `beta_hat_D`) from measured rates and let L0
follow, instead of asserting L0 = perimeter and letting `kappa` be whatever it
must be. The practical question for the re-run is what `beta_hat_D` of 10^2-10^5
does to patterning, since the published models — and ours — were explored at
`beta_hat_D ~ 4`.
