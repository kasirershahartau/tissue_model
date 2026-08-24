# Code cleanup: three branches

Plan only — nothing is moved until the full-model recompute finishes, because on
Windows each worker re-imports the modules whenever the pool recycles.

## Shape

The branches **stack**, they do not diverge:

    core  ──▶  manuscript  ──▶  scratch
                  ▲
                  └── working tree / default branch

`post_processing` is imported by 59 of the 79 scripts and `run_model` by 37, so a
`manuscript` branch holding only manuscript code would not execute. Stacking
keeps every branch a working repo, and `core` stays the thing to cherry-pick from
when the upstreaming work starts after submission.

## A separate deliverable: patches to tyssue itself

`tyssue/` is a clone of `github.com/damcb/tyssue` at upstream `main`, and the
outer repo tracks none of it. Two local modifications are already PR-shaped and
belong on a branch **in that clone**, not here:

| file | change | why upstream cares |
|---|---|---|
| `src/tyssue/core/history.py` | `select_column` instead of `select(columns=[...])` | PyTables materialises whole rows before subsetting: 1769 MB → 82 MB on a 5 GB archive, on every archive open |
| `src/tyssue/core/objects.py` | explicit column selection in `groupby().apply()` | silences a pandas ≥ 2.1 DeprecationWarning with zero logic change |

`src/tyssue/_mesh_generation.py` is untracked in that clone and still needs
identifying before it is either committed or removed.

## Naming

Generic names (`signal` / `primary` / `secondary` in place of `atoh` / `HC` /
`SC`) are right for upstream, and get applied to class names, function names,
docstrings and new APIs.

**The stored column names stay as they are for now.** `atoh_level` and
`delta_level` are written into every history archive on disk; renaming them in
`core` — which `manuscript` stacks on — breaks reading every existing archive and
every script that reads one, for a benefit that is not collected until the
upstreaming phase. The column rename belongs with that work, behind an alias
shim.

## Allocation

### core — no project references, or generic once separated

| what | files |
|---|---|
| geometry / topology | `virtual_sheet.py`, `periodic_sheet.py`, `topological_events.py` |
| solving | `solvers.py` |
| effectors | `empty_effector.py`, `random_effector.py`, `face_repulsion_efffector.py` (rename: `efffector` → `effector`) |
| tests | `test_periodic.py` (6519 lines), `periodic_tests.py` |
| **split out of `inner_ear_model.py`** | `ContractilityPerimeterElasticity`, `BoundaryBending`, `_truncate_history_file`, `_rewrite_history_for_resume` |
| **split out of `run_model.py`** | `initialize_sheet`, `load_sheet_from_file`, `run`, resume/classify machinery, debug-log helpers |
| **split out of `post_processing.py`** | history I/O and geometry: `load_history_file`, `get_time_points`, `find_non_boundary_cells`, `get_non_boundary_cell_ids_from_type`, `calc_contact_with_neighbors_from_type`, `calc_roundness_for_type`, `calc_area_for_type`, `calc_roundness_for_time_point`, `calc_contacts_for_time_point`, `create_gif_safe`, `redraw`, `drop_corrupted_snapshots`, `extract_time_point_to_new_history`, `save_data_of_a_given_time_point` |
| differentiation | `lateral_inhibition_model.py` (domain-renamed at the API surface) |

The four modules verified clean of project terms (`utricle`, `E17`, `atoh`,
`RESULTS_DIR`) are `virtual_sheet`, `periodic_sheet`, `topological_events` and
`solvers` — they move as-is.

Ablation is a **capability** of the core model (`ablated_cells` in `run_model` /
`inner_ear_model`); the ablation *experiments* are manuscript.

### manuscript — needed for the paper, not for a general tyssue user

| what | files |
|---|---|
| model + driver remainders | `inner_ear_model.py` (the `InnerEarModel` class), `run_model.py` (`find_mechanical_parameters`, `find_psigma`, full-model array runners), `post_processing.py` (the other ~54 functions) |
| mechanical fit | `grid_fit_mechanics_v2.py`, `selfconsistent_scan.py`, `p0_from_e17_stiffness.py`, `p0_gamma_scan.py`, `p0_rgamma_scan.py`, `p0_boundary_scan.py`, `solve_preferred_area.py`, `sweep_preferred_area.py`, `run_refit.py`, `score_point.py`, `run_best_point.py`, `bayesian_optimization.py` |
| full model / psigma | `run_fitted_full_model.py`, `run_full_model.py`, `run_psigma_repeats.py`, `run_psigma_sweep_v2.py`, `score_psigma_pooled.py`, `compare_psigma_points.py`, `extend_psigma_repeats.py`, `revive_dead_runs.py` |
| ablation experiments | `ablate_single_hc.py`, `ablate_from_strict.py`, `ablate_psigma_repeats.py`, `strict_steady_state.py` |
| tables | `build_fullmodel_table.py`, `build_mechanics_table.py`, `build_ablation_table.py` |
| figures | `plot_scores_3rep.py`, `plot_isolated_sc_vs_psigma.py`, `plot_ablation_distances.py`, `plot_hill_gate.py`, `plot_experimental_scoring_data.py`, `plot_selfconsistent_scan.py`, `plot_rgamma_scan.py`, `plot_nsigma_vs_params.py`, `face_stress_over_time.py`, `face_stress_ps0_groups.py` |
| analyses | `correlate_score2_with_time.py`, `cross_stage_comparison.py`, `check_steady_state.py`, `calibrate_time_by_notch_inhibition.py` |
| infrastructure | `azure_run.py` |
| docs | `MECHANICAL_FIT_V2.md`, `SHRINKAGE_ESTIMATE_METHOD.md` |

### scratch — served its purpose

| why | files |
|---|---|
| superseded by v2 | `run_psigma_sweep.py`, `run_psigma_sweep_stage.py`, `plot_psigma_scores.py`, `plot_psigma_v2.py`, `score_v2_full_model.py` |
| one-off repairs | `fix_corrupted_resumed_runs.py`, `check_and_sweep.py`, `compact_history_files.py`, `count_dead_states.py`, `make_missing_gifs.py` |
| probes and abandoned tests | `probe_psigma_settling.py`, `check_bending_needed.py`, `test_line_tension.py`, `test_sharp_angle.py`, `test_save_interval_sensitivity.py`, `test_zero_shape_index_mechanics.py`, `mod_test_common.py`, `sweep_scoring_windows.py`, `plot_window_figures.py` |
| unused / unidentified | `model_factory_shachaf_new.py` (460 lines, imported by nothing), `results_analysis.py` (25 lines, imported by nothing) |

## Judgement calls — settled

| item | decision |
|---|---|
| `plot_pvalue_vs_psigma.py` | **manuscript** |
| `azure_run.py` | **manuscript** |
| `bayesian_optimization.py` | **scratch** — see the caveat below |
| `model_factory_shachaf_new.py`, `results_analysis.py` | **delete** |

`model_factory_shachaf_new.py` is untracked, so deleting it leaves no copy in git
history at all; `results_analysis.py` is tracked and stays recoverable. Flagged
once — the instruction stands unless changed.

### The v1 Bayesian fit is retired whole

`find_mechanical_parameters` is not a general fitter with a BO branch — it
imports `bayesian_optimization` unconditionally and returns the optimiser trace.
It *is* the v1 fit. No v2 script calls it: `grid_fit_mechanics_v2.py` and
`selfconsistent_scan.py` import only helpers from `run_model`, and the paper
describes the v2 fit only. So everything downstream of it goes to **scratch**
together:

| item | where it lives now |
|---|---|
| `bayesian_optimization.py` | own module |
| `find_mechanical_parameters` | `run_model.py` |
| `run_refit.py` | own module — exists only to call it |
| `plot_nsigma_vs_params.py` | own module — plots the BO trace |
| `load_mechanical_optimization`, `load_mechanical_optimization_trace`, `load_mechanical_optimization_landscape`, `plot_mechanical_optimization` | `post_processing.py` — read/plot the BO output files |
| the `find-mech` subcommand | `azure_run.py` (the file itself stays in manuscript) |
| 14 tests | `test_periodic.py` — 8 BO, 12 fit, overlapping |

That is 14 of 244 tests. Six of them cover behaviours worth keeping *if the
function stayed* (trace survives a crash, per-array thresholds, the shape-index
parameterisation, crash propagation) — they retire with it, and their coverage is
lost deliberately rather than by accident.

`score_point.py` and `test_zero_shape_index_mechanics.py` mention
`find_mechanical_parameters` only in prose, so they need no change beyond the
docstring wording.

## Corrections made while executing

Three assignments in the tables above were wrong, each found by a check rather
than by reading:

1. **`InnerEarModel` belongs in core**, not manuscript. It reads as
   project-specific but is a generic two-cell-type vertex model, and `run()`
   constructs it directly — core cannot work without it. Only its vocabulary is
   domain-bound.
2. **`build_run_table.py` belongs in manuscript**, not scratch. It is superseded
   as a *script*, but `build_fullmodel_table`, `build_mechanics_table` and
   `face_stress_ps0_groups` all import its parameter-parsing helpers.
3. **`plot_scores_vs_psigma.py` belongs in manuscript**, not scratch, for the
   same reason: `plot_isolated_sc_vs_psigma` imports `read_runs_csv`, `COLOUR`
   and `ZOOM` from it.

The lesson for the remaining branch: "superseded" describes whether a script is
still *run*, not whether its functions are still *imported*. Check both.

## Comment policy

Cut: conversational asides, "as we discussed", narration of what was tried.

Keep: anything that stops a bug being reintroduced — the `select_column` memory
note, the `prefered_area` misspelling, the "periodicity is safe by construction,
do not re-fix" note, the non-obvious ordering constraints. These read as
explanations but are load-bearing.

Anything ambiguous gets flagged rather than removed unilaterally.

## Sequencing

1. recompute finishes
2. branch `core` from `main`, move/split, fix imports, run `test_periodic.py`
3. branch `manuscript` from `core`, restore the rest, smoke-test one script per
   family (fit, sweep, table, figure)
4. branch `scratch` from `manuscript`, add the remainder
5. set `manuscript` as the working branch
6. separately: branch the two patches in the `tyssue/` clone
