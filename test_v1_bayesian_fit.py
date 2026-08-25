"""Tests for the retired v1 Bayesian fit.

Split out of test_periodic.py when the v1 fit left the core and manuscript
branches. They exercise find_mechanical_parameters and the files it wrote, so
they only make sense next to ``v1_bayesian_fit.py``.

Six of them cover behaviour that is worth having generally — a trace surviving a
crash midway, per-array thresholds, the per-type shape-index parameterisation,
failures not being swallowed — but they can only reach it through the retired
fit, so they retire with it. If an equivalent path appears in the v2 fit, port
these rather than writing them again.
"""

"""
Progressive unit tests for the periodic-boundary-condition support
in :mod:`periodic_sheet` and :mod:`virtual_sheet`.

The tests are deliberately layered, from "does the lattice get built
at all" up to "does a topology operation across the boundary leave a
self-consistent sheet". Failures at lower layers usually explain
failures at higher ones, so run them in order (``pytest -x`` is fine).

Run with::

    pytest -x test_periodic.py
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Use a non-interactive backend; some Windows installs crash on the
# default Qt backend when matplotlib touches a Collection.
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from periodic_sheet import PeriodicBoundarySheet, PeriodicPlanarGeometry
from virtual_sheet import VirtualSheet
from topological_events import (
    index_preserving_type1_transition,
    index_preserving_cell_division,
    index_preserving_remove,
)


class TestMechanicalParamsFixedThreshold:
    """``find_mechanical_parameters`` can pin a fixed HC/SC classification
    threshold (``fix_threshold``) and ``type_by`` that are used for EVERY
    comparison; otherwise each comparison computes its own mid-range
    threshold (the previous behaviour)."""

    @staticmethod
    def _stub_worker_deps(monkeypatch):
        """Stub out the heavy bits so ``_evaluate_mechanics_for_sheet`` runs
        without simulating, capturing what ``extract_model_mechanics`` is called
        with (the worker now extracts per-term distributions, not p-values)."""
        import run_model
        captured = {}
        monkeypatch.setattr(run_model, "run", lambda *a, **k: "results/fake")
        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet",
                            lambda initial: {})

        def fake_extract(model_name, type_by=None, threshold=None, **kw):
            captured["type_by"] = type_by
            captured["threshold"] = threshold
            return {"hc_roundness": np.array([0.5]),
                    "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

        monkeypatch.setattr(run_model, "extract_model_mechanics", fake_extract)
        return run_model, captured

    def test_fixed_threshold_and_type_by_used(self, monkeypatch):
        run_model, captured = self._stub_worker_deps(monkeypatch)
        task = (0.01, 10.0, 1.0, "sheet0", "E17.5", [], -1, 0.42, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert captured["threshold"] == 0.42
        assert captured["type_by"] == "delta_level"

    def test_defaults_keep_calculated_threshold(self, monkeypatch):
        run_model, captured = self._stub_worker_deps(monkeypatch)
        task = (0.01, 10.0, 1.0, "sheet0", "E17.5", [], -1, None, "atoh_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        # threshold None -> extraction computes its own mid-range threshold.
        assert captured["threshold"] is None
        assert captured["type_by"] == "atoh_level"

    def test_find_threads_params_into_tasks(self, monkeypatch, tmp_path):
        # The top-level method must put fix_threshold / type_by into every
        # per-sheet task tuple. Capture the tuples by stubbing the worker and
        # the optimizer (so nothing actually simulates).
        import run_model
        import bayesian_optimization as bo
        # find_mechanical_parameters WRITES its trace / params / objective /
        # landscape files into RESULTS_DIR at the end — redirect it to a tmp dir
        # so the test doesn't clobber real optimization results on disk.
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results").mkdir()
        seen = []

        def fake_worker(task):
            seen.append(task)
            return {"hc_roundness": np.array([0.5]),
                    "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet", fake_worker)
        # The objective now pools the workers' distributions and compares once;
        # stub that so the test doesn't need the real statistical backend.
        monkeypatch.setattr(
            run_model, "compare_pooled_model_mechanics_to_experiments",
            lambda model_terms, stage, **kw: {"roundness_ratio": 0.5,
                                              "shrinkage": 0.5,
                                              "ablation_ratio": float("nan")})

        def fake_minimize(obj, bounds, **kw):
            obj([0.01, 10.0, 1.0])  # evaluate once
            return {"x": np.array([0.01, 10.0, 1.0]), "fun": 0.0, "X": [], "y": []}

        monkeypatch.setattr(bo, "minimize", fake_minimize)

        run_model.find_mechanical_parameters(
            "E17.5", initial_sheets=["sheetA", "sheetB"], n_workers=1,
            fix_threshold=0.3, type_by="repressor_level")

        assert len(seen) == 2  # one task per initial sheet
        for task in seen:
            assert task[7] == 0.3                 # fix_threshold
            assert task[8] == "repressor_level"   # type_by
            # stall-guard params (max_wall_seconds, min_progress_rate,
            # progress_window_seconds) trail the tuple; default = off.
            assert task[9:12] == (None, None, 30.0)
            assert task[12] is False              # rerun_stalled_runs default
            assert task[13] == 0.0                 # shape_index (not optimized here)
            # base/ablation quasi_static_threshold default to 0.01 so a new
            # optimization reuses the existing 0.01 run folders (0.03 / 0.02 is
            # opt-in for a fresh fit).
            assert task[14] == 0.01                # base_quasi_static_threshold default
            assert task[15] == 0.01                # ablation_quasi_static_threshold default
            assert task[16] == 0.2                 # line_tension default (fixed 0.2)
            assert task[17] is None and task[18] is None  # hc/sc_shape_index off

    def test_per_type_shape_index_parameterisation(self, monkeypatch, tmp_path):
        # Choosing the parameterisation at the CALL SITE: dropping
        # gammaHC_ratio_bounds fixes gamma (HC == SC contractility) and the two
        # per-type shape indices become the fitted dimensions instead - still 4
        # parameters, one per comparison term.
        import run_model
        import bayesian_optimization as bo
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results").mkdir()
        seen, dims = [], {}

        def fake_worker(task):
            seen.append(task)
            return {"hc_roundness": np.array([0.5]), "sc_roundness": np.array([0.5]),
                    "hc_ablation": None, "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet", fake_worker)
        monkeypatch.setattr(
            run_model, "compare_pooled_model_mechanics_to_experiments",
            lambda model_terms, stage, **kw: {"roundness_ratio": 0.5,
                                              "shrinkage": 0.5,
                                              "ablation_ratio": float("nan")})

        def fake_minimize(obj, bounds, **kw):
            dims["n"] = len(bounds)
            obj([0.05, 1.05, 1.2, 1.4])      # gammaSC, alphaHC, hc_p0, sc_p0
            return {"x": np.array([0.05, 1.05, 1.2, 1.4]), "fun": 0.0, "X": [], "y": []}

        monkeypatch.setattr(bo, "minimize", fake_minimize)
        run_model.find_mechanical_parameters(
            "P0", initial_sheets=["sheetA"], n_workers=1,
            gammaSC_bounds=(0.01, 0.2), gammaHC_ratio_bounds=None,
            alphaHC_ratio_bounds=(1.0, 1.2),
            hc_shape_index_bounds=(1.1, 1.4), sc_shape_index_bounds=(1.1, 1.4))
        assert dims["n"] == 4                  # still exactly 4 fitted parameters
        t = seen[0]
        assert t[0] == 0.05                    # gammaSC (fitted)
        assert t[1] == 1.0                     # gammaHC_ratio FIXED -> HC == SC gamma
        assert t[2] == 1.05                    # alphaHC_ratio (fitted)
        assert t[17] == 1.2 and t[18] == 1.4   # hc/sc_shape_index (fitted)

    def test_find_threads_rerun_stalled_runs_into_tasks(self, monkeypatch, tmp_path):
        # rerun_stalled_runs=True on find_mechanical_parameters must reach
        # every per-sheet task tuple (last field), same plumbing as above.
        import run_model
        import bayesian_optimization as bo
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results").mkdir()
        seen = []

        def fake_worker(task):
            seen.append(task)
            return {"hc_roundness": np.array([0.5]),
                    "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet", fake_worker)
        monkeypatch.setattr(
            run_model, "compare_pooled_model_mechanics_to_experiments",
            lambda model_terms, stage, **kw: {"roundness_ratio": 0.5,
                                              "shrinkage": 0.5,
                                              "ablation_ratio": float("nan")})

        def fake_minimize(obj, bounds, **kw):
            obj([0.01, 10.0, 1.0])
            return {"x": np.array([0.01, 10.0, 1.0]), "fun": 0.0, "X": [], "y": []}

        monkeypatch.setattr(bo, "minimize", fake_minimize)

        run_model.find_mechanical_parameters(
            "E17.5", initial_sheets=["sheetA", "sheetB"], n_workers=1,
            rerun_stalled_runs=True)

        assert len(seen) == 2
        for task in seen:
            assert task[12] is True


class TestSaveLILevelsFromJsonl:
    """``post_processing.save_li_levels_from_best_pval_jsonl`` extracts the
    per-cell N_final/D_final/R_final lists (entry i -> cell with unique_id i)
    from a ``*_best_pval_per_array.jsonl`` file and writes them as
    notch/delta/repressor ``_levels.npy`` in each matching model folder."""

    @staticmethod
    def _make_model_folder(results_dir, n_cells, array_index=1, dev_stage="E17"):
        # A cells_info.pkl indexed by unique_id 0..n-1 is enough for the
        # length check (no need to fabricate a full history archive).
        folder = results_dir / ("random_periodic_array%d_for_%s"
                                % (array_index, dev_stage))
        folder.mkdir(parents=True)
        ci = pd.DataFrame({"x": np.arange(n_cells)},
                          index=pd.Index(range(n_cells), name="unique_id"))
        ci.to_pickle(folder / "cells_info.pkl")
        return folder

    def test_writes_three_npy_matching_json(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        folder = self._make_model_folder(results, n_cells=5)
        N = [0.1, 0.2, 0.3, 0.4, 0.5]
        D = [0.5, 0.4, 0.3, 0.2, 0.1]
        R = [0.9, 0.8, 0.7, 0.6, 0.5]
        jf = tmp_path / "E17_best_pval_per_array.jsonl"
        jf.write_text(json.dumps({"array_index": 1, "dev_stage": "E17",
                                  "N_final": N, "D_final": D, "R_final": R}) + "\n")
        written = save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results))
        assert len(written) == 1
        np.testing.assert_array_equal(np.load(folder / "notch_levels.npy"), N)
        np.testing.assert_array_equal(np.load(folder / "delta_levels.npy"), D)
        np.testing.assert_array_equal(np.load(folder / "repressor_levels.npy"), R)

    def test_dev_stage_maps_to_folder_suffix(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        folder = self._make_model_folder(results, n_cells=3, array_index=4,
                                         dev_stage="P0")
        jf = tmp_path / "P0_best_pval_per_array.jsonl"
        jf.write_text(json.dumps({"array_index": 4, "dev_stage": "P0",
                                  "N_final": [0.1, 0.2, 0.3],
                                  "D_final": [0.3, 0.2, 0.1],
                                  "R_final": [0.5, 0.5, 0.5]}) + "\n")
        written = save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results))
        assert written == [str(folder)]
        assert (folder / "notch_levels.npy").is_file()

    def test_length_mismatch_raises(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        self._make_model_folder(results, n_cells=5)  # model has 5 cells
        jf = tmp_path / "E17_best_pval_per_array.jsonl"
        jf.write_text(json.dumps({"array_index": 1, "dev_stage": "E17",
                                  "N_final": [0, 0, 0], "D_final": [0, 0, 0],
                                  "R_final": [0, 0, 0]}) + "\n")  # only 3
        with pytest.raises(ValueError, match="unique_id"):
            save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results))

    def test_missing_folder_is_skipped(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        results.mkdir()
        jf = tmp_path / "E17_best_pval_per_array.jsonl"
        jf.write_text(json.dumps({"array_index": 7, "dev_stage": "E17",
                                  "N_final": [0], "D_final": [0],
                                  "R_final": [0]}) + "\n")
        # No matching model folder -> nothing written, no error.
        assert save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results)) == []

    def test_writes_threshold_npy_from_D_threshold_mean(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        folder = self._make_model_folder(results, n_cells=3)
        jf = tmp_path / "E17_best_pval_per_array.jsonl"
        jf.write_text(json.dumps({"array_index": 1, "dev_stage": "E17",
                                  "N_final": [0.1, 0.2, 0.3], "D_final": [0.3, 0.2, 0.1],
                                  "R_final": [0.5, 0.5, 0.5],
                                  "D_threshold_mean": 0.345}) + "\n")
        save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results))
        thr = np.load(folder / "threshold.npy")
        assert float(thr) == 0.345

    def test_threshold_skipped_when_field_absent(self, tmp_path):
        import json
        from post_processing import save_li_levels_from_best_pval_jsonl
        results = tmp_path / "results"
        folder = self._make_model_folder(results, n_cells=3)
        jf = tmp_path / "E17_best_pval_per_array.jsonl"
        # No D_threshold_mean -> LI files written, no threshold.npy, no error.
        jf.write_text(json.dumps({"array_index": 1, "dev_stage": "E17",
                                  "N_final": [0.1, 0.2, 0.3], "D_final": [0.3, 0.2, 0.1],
                                  "R_final": [0.5, 0.5, 0.5]}) + "\n")
        save_li_levels_from_best_pval_jsonl(str(jf), results_dir=str(results))
        assert (folder / "notch_levels.npy").is_file()
        assert not (folder / "threshold.npy").exists()

    def test_writes_from_best_row_json(self, tmp_path):
        # The NEW layout: a JSON LIST of records with array_id / D_threshold
        # field names (vs array_index / D_threshold_mean) and extra ignored
        # fields. Maps to the same folders / files as the jsonl extractor.
        import json
        from post_processing import save_li_levels_from_best_row_json
        results = tmp_path / "results"
        f1 = self._make_model_folder(results, n_cells=3, array_index=1, dev_stage="E17")
        f2 = self._make_model_folder(results, n_cells=2, array_index=4, dev_stage="P0")
        jf = tmp_path / "best_row_per_morphology.json"
        jf.write_text(json.dumps([
            {"dev_stage": "E17", "array_id": 1, "pS": 0.1, "pR": 0.3,
             "chi2_sum_edge_+_newly_diff": 9.9, "D_threshold": 0.345,
             "N_final": [0.1, 0.2, 0.3], "D_final": [0.3, 0.2, 0.1], "R_final": [0.5, 0.5, 0.5]},
            {"dev_stage": "P0", "array_id": 4, "D_threshold": 0.28,
             "N_final": [0.7, 0.8], "D_final": [0.2, 0.1], "R_final": [0.4, 0.6]},
        ]))
        written = save_li_levels_from_best_row_json(str(jf), results_dir=str(results))
        assert set(written) == {str(f1), str(f2)}
        np.testing.assert_array_equal(np.load(f1 / "delta_levels.npy"), [0.3, 0.2, 0.1])
        np.testing.assert_array_equal(np.load(f2 / "notch_levels.npy"), [0.7, 0.8])
        assert float(np.load(f1 / "threshold.npy")) == 0.345
        assert float(np.load(f2 / "threshold.npy")) == 0.28


class TestMechanicsEvaluationRobustness:
    """A single initial sheet that degenerates (e.g. extreme gammaSC ->
    near-constant areas/roundness, or the solver hits its dt floor) must be
    DROPPED from the pool (worker returns None) instead of tearing down the whole
    optimization; the objective pools the surviving sheets. Genuine configuration
    errors must still propagate."""

    @staticmethod
    def _task(gammaSC=0.5):
        return (gammaSC, 10.0, 1.0, "sheet0", "E17.5", [], -1, None, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)

    def _stub_run(self, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "run", lambda *a, **k: "results/fake")
        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet",
                            lambda initial: {})
        return run_model

    def test_singular_matrix_drops_sheet(self, monkeypatch):
        run_model = self._stub_run(monkeypatch)

        def boom(*a, **k):
            raise np.linalg.LinAlgError("singular matrix")

        monkeypatch.setattr(run_model, "extract_model_mechanics", boom)
        assert run_model._evaluate_mechanics_for_sheet(self._task()) is None

    def test_solver_runtime_error_drops_sheet(self, monkeypatch):
        run_model = self._stub_run(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("dt fell below floor")

        monkeypatch.setattr(run_model, "extract_model_mechanics", boom)
        assert run_model._evaluate_mechanics_for_sheet(self._task()) is None

    def test_extraction_valueerror_drops_sheet(self, monkeypatch):
        # A degenerate steady state (empty / zero-variance HC or SC sample) makes
        # extraction raise (e.g. a ratio dividing by ~zero). That sheet must be
        # dropped, not crash the optimization.
        run_model = self._stub_run(monkeypatch)

        def boom(*a, **k):
            raise ValueError("zero-size array")

        monkeypatch.setattr(run_model, "extract_model_mechanics", boom)
        assert run_model._evaluate_mechanics_for_sheet(self._task()) is None

    def test_topology_indexing_error_drops_sheet(self, monkeypatch):
        # The actual crash seen at extreme parameters: a virtual-vertex collapse
        # cascade empties a face -> tyssue's drop_two_sided_faces indexes face_df
        # with a face-less boolean mask -> pandas IndexingError. It is in no
        # hand-listed set, so the worker must catch it BROADLY and drop the sheet
        # rather than let it tear down the whole ProcessPool optimization.
        import pandas as pd
        run_model = self._stub_run(monkeypatch)

        def boom(*a, **k):
            raise pd.errors.IndexingError("Unalignable boolean Series")

        monkeypatch.setattr(run_model, "extract_model_mechanics", boom)
        assert run_model._evaluate_mechanics_for_sheet(self._task()) is None

    def test_ablation_failure_keeps_base_run(self, monkeypatch):
        # The ablation run is EXTRA. If it degenerates (e.g. the topology
        # IndexingError) the base run's area/roundness must still be used — only
        # the ablation term is skipped for this sheet.
        import pandas as pd
        import run_model
        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet",
                            lambda initial: {})

        def run_stub(*a, **k):
            # base run has no ablated_cells; the ablation run does -> crash it.
            if k.get("ablated_cells"):
                raise pd.errors.IndexingError("Unalignable boolean Series")
            return "results/fake_base"

        monkeypatch.setattr(run_model, "run", run_stub)
        seen = {}

        def extract_stub(model_name, type_by=None, threshold=None, **kw):
            seen.update(kw)
            return {"hc_roundness": np.array([0.5]),
                    "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

        monkeypatch.setattr(run_model, "extract_model_mechanics", extract_stub)
        task = (0.5, 10.0, 1.0, "sheet0", "E17.5", [3, 7], -1, None, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        terms = run_model._evaluate_mechanics_for_sheet(task)
        assert terms is not None and list(terms["hc_roundness"]) == [0.5]  # base kept
        assert "ablation_model_name" not in seen  # ablation term skipped

    def test_filenotfound_inside_run_propagates(self, monkeypatch):
        # A FileNotFoundError from the base run (a genuine config / missing-file
        # error, NOT a degenerate parameter point) must PROPAGATE, not be masked
        # as a dropped sheet by the broad catch.
        import run_model
        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet",
                            lambda initial: {})

        def boom(*a, **k):
            raise FileNotFoundError("missing archive")

        monkeypatch.setattr(run_model, "run", boom)
        with pytest.raises(FileNotFoundError):
            run_model._evaluate_mechanics_for_sheet(self._task())

    def test_failure_prints_model_name(self, monkeypatch, capsys):
        """A dropped sheet must name the exact results folder on stdout so a
        failure can be tied back to a specific run on disk."""
        run_model = self._stub_run(monkeypatch)

        def boom(*a, **k):
            raise ValueError("degenerate")

        monkeypatch.setattr(run_model, "extract_model_mechanics", boom)
        run_model._evaluate_mechanics_for_sheet(self._task())
        out = capsys.readouterr().out
        # The deterministic base-run folder name for this task's params (the base
        # run relaxes to the task's base_quasi_static_threshold = 0.03).
        expected = run_model._short_run_folder_name("sheet0", 0.5, 10.0, 1.0, 0,
                                                    quasi_static_threshold=0.03)
        assert "[mechanics] START" in out and expected in out
        assert "[mechanics] FAILED" in out and expected in out

    def test_all_sheets_dropped_gives_large_finite_penalty(self):
        # When EVERY sheet degenerates, each active term has no model data so its
        # z is nan -> the objective substitutes _WORST_CASE_NSIGMA and sums z**2,
        # a large but FINITE penalty (not inf/nan) the optimizer can compare.
        import run_model
        zscores = {t: float("nan") for t in run_model.MECHANICS_TERMS}
        active = run_model.MECHANICS_TERMS  # all four (cells ablated)
        value = sum((run_model._WORST_CASE_NSIGMA if not np.isfinite(zscores[t])
                     else zscores[t]) ** 2 for t in active)
        assert np.isfinite(value) and value > 1e6

    def test_config_error_propagates(self, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "run", lambda *a, **k: "results/fake")

        def bad_li(initial):
            raise FileNotFoundError("missing notch_levels.npy")

        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet", bad_li)
        with pytest.raises(FileNotFoundError):
            run_model._evaluate_mechanics_for_sheet(self._task())

    def test_run_propagates_crash_not_swallowed_by_finally(self, monkeypatch, tmp_path):
        """A crash inside ``run()`` must PROPAGATE so the fit worker scores it
        worst-case. A ``return`` in the cleanup ``finally`` used to swallow the
        re-raised exception, so run() handed back its folder name even on a
        crash — the worker then ran on a crashed/degenerate sheet (and launched
        the ablation run from a corrupt state) instead of scoring worst-case."""
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results").mkdir()   # run() does os.mkdir("<RESULTS_DIR>/<name>")

        def boom(*a, **k):
            raise RuntimeError("boom: simulated solver crash")

        # The sheet build is the first thing inside run()'s try; crash it
        # (mock both entry points so any branch hits it immediately).
        monkeypatch.setattr(run_model, "load_sheet_from_file", boom)
        monkeypatch.setattr(run_model, "initialize_sheet", boom)
        with pytest.raises(RuntimeError, match="boom"):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="whatever",
                          name="crashtest", end_on_steady_state=False,
                          t_end=1, dt=0.01)


class TestPooledMechanicsComparison:
    """The fit POOLS every initial-array run at the same parameters and scores the
    whole pool with one standardized mean discrepancy (n-sigma) per term. A
    degenerate run is dropped from the pool; if every run degenerates the active
    terms have no data and the point is scored worst-case."""

    @staticmethod
    def _arrays(v, ablation=None):
        return {"roundness_ratio": np.array([v]),
                "ablation_ratio": None if ablation is None else np.array([ablation]),
                "shrinkage": np.array([7.5])}

    def _run_once(self, monkeypatch, tmp_path, worker, sheets):
        import run_model
        import bayesian_optimization as bo
        rd = str(tmp_path / "results")
        (tmp_path / "results").mkdir()
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet", worker)
        seen = {"compare_calls": 0}

        def fake_pooled(model_terms, stage, **kw):
            seen["compare_calls"] += 1
            seen["terms"] = {t: [list(a) for a in model_terms[t]]
                             for t in run_model.MECHANICS_TERMS}
            # z = 0.5 for a term with data, nan for an empty one (so all-empty ->
            # active terms nan -> objective substitutes the worst-case penalty).
            return {t: (0.5 if model_terms[t] else float("nan"))
                    for t in run_model.MECHANICS_TERMS}

        monkeypatch.setattr(run_model, "compare_pooled_model_mechanics_to_experiments",
                            fake_pooled)
        captured = {}

        def fake_minimize(obj, bounds, **kw):
            captured["value"] = obj([0.01, 10.0, 1.0])
            return {"x": np.array([0.01, 10.0, 1.0]), "fun": 0.0, "X": [], "y": []}

        monkeypatch.setattr(bo, "minimize", fake_minimize)
        run_model.find_mechanical_parameters(
            "E17.5", initial_sheets=sheets, n_workers=1)
        return run_model, rd, seen, captured

    def test_all_sheets_pooled_into_one_call(self, monkeypatch, tmp_path):
        # Each sheet contributes its own array; ONE pooled call sees all of them.
        def worker(task):
            return self._arrays(float(task[3][-1]))  # "s0","s1","s2" -> 0,1,2

        _, _, seen, _ = self._run_once(
            monkeypatch, tmp_path, worker, ["s0", "s1", "s2"])
        assert seen["compare_calls"] == 1
        assert seen["terms"]["roundness_ratio"] == [[0.0], [1.0], [2.0]]
        # no ablation simulated -> the ablation pools are empty.
        assert seen["terms"]["ablation_ratio"] == []

    def test_degenerate_sheet_dropped_from_pool(self, monkeypatch, tmp_path):
        import post_processing as pp

        def worker(task):
            return None if task[3] == "s1" else self._arrays(float(task[3][-1]))

        _, rd, seen, _ = self._run_once(
            monkeypatch, tmp_path, worker, ["s0", "s1", "s2"])
        # s1 dropped -> only two arrays pooled, and the trace records it.
        assert seen["terms"]["roundness_ratio"] == [[0.0], [2.0]]
        tr = pp.load_mechanical_optimization_trace("E17.5", results_dir=rd)
        assert tr["n_contributing"].iloc[0] == 2

    def test_every_sheet_degenerate_scores_worst_case(self, monkeypatch, tmp_path):
        # All runs fail -> the active (roundness) terms have no data -> z is nan ->
        # the objective substitutes _WORST_CASE_NSIGMA, a large finite penalty.
        _, _, _, captured = self._run_once(
            monkeypatch, tmp_path, lambda task: None, ["s0", "s1"])
        assert np.isfinite(captured["value"]) and captured["value"] > 1e6


class TestComparePooledMechanics:
    """``compare_pooled_model_mechanics_to_experiments`` scores each term by the
    standardized mean discrepancy ``z = (mean_model - mean_exp) / SEM_exp``, where
    ``SEM_exp`` is the standard error of the experimental biological-repeat means.
    ``z`` is nan when a term can't be scored (no model data, <2 repeats, zero SEM)."""

    @staticmethod
    def _stub_exp(monkeypatch, repeat_means, n_per=50):
        # experimental repeats whose per-repeat means are `repeat_means`.
        import post_processing as pp
        monkeypatch.setattr(pp, "load_experimental_results",
                            lambda stage, t: [np.full(n_per, m) for m in repeat_means])

    @staticmethod
    def _mt(**terms):
        base = {t: [] for t in ("roundness_ratio", "ablation_ratio", "shrinkage")}
        base.update(terms)
        return base

    def test_zero_when_model_matches_exp_mean(self, monkeypatch):
        import post_processing as pp
        self._stub_exp(monkeypatch, [0.6, 0.7, 0.8])  # exp grand mean = 0.7
        z = pp.compare_pooled_model_mechanics_to_experiments(
            self._mt(roundness_ratio=[np.full(100, 0.7)] * 5), "E17.5")
        assert abs(z["roundness_ratio"]) < 1e-9
        assert np.isnan(z["ablation_ratio"])  # no model data -> nan

    def test_z_scales_with_experimental_sem(self, monkeypatch):
        import post_processing as pp
        means = [0.60, 0.70, 0.80]
        self._stub_exp(monkeypatch, means)
        gm = float(np.mean(means))
        sem = float(np.std(means, ddof=1) / np.sqrt(3))
        z = pp.compare_pooled_model_mechanics_to_experiments(
            self._mt(roundness_ratio=[np.full(100, gm + 2 * sem)]), "E17.5")
        assert abs(z["roundness_ratio"] - 2.0) < 1e-6

    def test_fewer_than_two_repeats_is_nan(self, monkeypatch):
        import post_processing as pp
        self._stub_exp(monkeypatch, [0.7])  # a single repeat -> no SEM
        z = pp.compare_pooled_model_mechanics_to_experiments(
            self._mt(roundness_ratio=[np.full(10, 0.7)]), "E17.5")
        assert np.isnan(z["roundness_ratio"])

    def test_nonfinite_model_cells_dropped(self, monkeypatch):
        import post_processing as pp
        means = [0.60, 0.70, 0.80]
        self._stub_exp(monkeypatch, means)
        gm = float(np.mean(means))
        model = np.array([gm - 0.1, np.nan, gm + 0.1, np.inf])  # finite mean = gm
        z = pp.compare_pooled_model_mechanics_to_experiments(
            self._mt(roundness_ratio=[model]), "E17.5")
        assert abs(z["roundness_ratio"]) < 1e-9

    def test_exception_in_experimental_load_is_nan(self, monkeypatch):
        import post_processing as pp

        def boom(stage, t):
            raise RuntimeError("bad experimental read")

        monkeypatch.setattr(pp, "load_experimental_results", boom)
        z = pp.compare_pooled_model_mechanics_to_experiments(
            self._mt(roundness_ratio=[np.full(5, 0.7)]), "E17.5")
        assert np.isnan(z["roundness_ratio"])


class TestOptimizationTraceAndLandscape:
    """``find_mechanical_parameters`` must leave (a) a crash-resistant per-step
    JSONL trace carrying the INDIVIDUAL p-values, and (b) the GP-surrogate
    optimization landscape — so a re-run can be diagnosed."""

    @staticmethod
    def _fake_worker(task):
        # The worker returns per-term DISTRIBUTIONS. Encode a param-dependent value
        # into the roundness arrays so the (stubbed) n-sigma objective varies over
        # the search space -> a non-degenerate landscape. No ablation here.
        gSC, gHC, aHC = task[0], task[1], task[2]
        return {"hc_roundness": np.array([2.0 * np.exp(-((gSC - 0.05) / 0.03) ** 2)]),
                "sc_roundness": np.array([0.3 + aHC / 10.0]),
                "hc_ablation": None, "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])}

    @staticmethod
    def _fake_pooled_compare(model_terms, stage, **kw):
        # z per term == mean of the pooled per-sheet values (param-dependent), nan
        # for an empty term.
        import run_model
        return {t: (float(np.mean(np.concatenate(model_terms[t]))) if model_terms[t]
                    else float("nan"))
                for t in run_model.MECHANICS_TERMS}

    def test_minimize_returns_working_surrogate(self):
        import bayesian_optimization as bo
        res = bo.minimize(lambda x: float((x[0] - 2) ** 2 + (x[1] + 1) ** 2),
                          [(-5, 5), (-5, 5)], n_calls=8, n_initial_points=4,
                          verbose=False, return_surrogate=True)
        assert callable(res["surrogate"]) and res["gp"] is not None
        mu, std = res["surrogate"](np.array([[2.0, -1.0], [0.0, 0.0]]))
        assert mu.shape == (2,) and std.shape == (2,) and np.all(std >= 0)

    def test_minimize_without_surrogate_has_no_extra_keys(self):
        import bayesian_optimization as bo
        res = bo.minimize(lambda x: float(x[0] ** 2), [(-1, 1)],
                          n_calls=4, n_initial_points=2, verbose=False)
        assert "surrogate" not in res and "gp" not in res

    def test_find_writes_trace_and_landscape(self, tmp_path, monkeypatch):
        import run_model
        import post_processing as pp
        rd = str(tmp_path / "results")
        (tmp_path / "results").mkdir()
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet",
                            self._fake_worker)
        monkeypatch.setattr(run_model, "compare_pooled_model_mechanics_to_experiments",
                            self._fake_pooled_compare)
        run_model.find_mechanical_parameters(
            "E17.5", initial_sheets=["a", "b"], n_workers=1,
            n_calls=6, n_initial_points=3, landscape_resolution=6)

        # (a) per-step trace with the per-term n-sigma discrepancies.
        tr = pp.load_mechanical_optimization_trace("E17.5", results_dir=rd)
        assert len(tr) == 6
        for col in ("nsigma_roundness_ratio", "nsigma_ablation_ratio",
                    "nsigma_shrinkage",
                    "obj_roundness_ratio", "obj_ablation_ratio",
                    "obj_shrinkage", "n_contributing", "sheets"):
            assert col in tr.columns
        # objective decomposes additively into the per-term z**2 contributions
        # (the inactive ablation terms contribute 0).
        assert np.allclose(tr["objective"],
                           tr["obj_roundness_ratio"] + tr["obj_ablation_ratio"]
                           + tr["obj_shrinkage"])
        # per-sheet contribution status recorded (two initial sheets, both ok).
        assert (tr["n_contributing"] == 2).all()
        assert len(tr["sheets"].iloc[0]) == 2
        assert set(tr["sheets"].iloc[0][0]) == {"initial", "ok"}
        assert tr["sheets"].iloc[0][0]["ok"] is True

        # (b) GP-surrogate landscape bundle.
        ls = pp.load_mechanical_optimization_landscape("E17.5", results_dir=rd)
        assert ls["mean"].shape == (6, 6, 6) and ls["std"].shape == (6, 6, 6)
        assert np.isfinite(ls["mean"]).all() and (ls["std"] >= 0).all()
        assert [str(p) for p in ls["param_names"]] == \
            ["gammaSC", "gammaHC_ratio", "alphaHC_ratio"]
        assert ls["axes"].shape == (3, 6) and ls["X"].shape == (6, 3)

    def test_trace_survives_a_crash_midway(self, tmp_path, monkeypatch):
        """The trace is appended per step, so a fit that dies partway still
        leaves the completed evaluations on disk."""
        import run_model
        import post_processing as pp
        import bayesian_optimization as bo
        rd = str(tmp_path / "results")
        (tmp_path / "results").mkdir()
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet",
                            self._fake_worker)
        monkeypatch.setattr(run_model, "compare_pooled_model_mechanics_to_experiments",
                            self._fake_pooled_compare)

        def dying_minimize(obj, bounds, **kw):
            obj([0.01, 10.0, 1.0])
            obj([0.05, 5.0, 2.0])
            raise RuntimeError("optimizer killed mid-run")

        monkeypatch.setattr(bo, "minimize", dying_minimize)
        with pytest.raises(RuntimeError, match="killed"):
            run_model.find_mechanical_parameters(
                "E17.5", initial_sheets=["a"], n_workers=1)
        # Both completed evaluations are on disk despite the crash.
        tr = pp.load_mechanical_optimization_trace("E17.5", results_dir=rd)
        assert len(tr) == 2


class TestDeltaThresholdDrivesAtohSensitivity:
    """A per-sheet loaded DELTA threshold (type_by='delta_level') must be used
    as the simulation's ``atoh_sensitivity`` so the model itself makes cells
    with delta ABOVE the threshold high-atoh (HC) and below it low-atoh (SC).
    ``atoh_sensitivity`` is the delta half-max of the Atoh1 Hill, so atoh crosses
    0.5 exactly at delta == threshold."""

    def _stub(self, monkeypatch):
        import run_model
        captured = []
        monkeypatch.setattr(run_model, "run",
                            lambda *a, **k: captured.append(k) or "results/fake")
        monkeypatch.setattr(run_model, "_li_levels_kwargs_for_initial_sheet",
                            lambda initial: {})
        monkeypatch.setattr(
            run_model, "extract_model_mechanics",
            lambda *a, **k: {"hc_roundness": np.array([0.5]),
                             "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])})
        return run_model, captured

    def test_delta_threshold_passed_as_atoh_sensitivity(self, monkeypatch):
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [], -1, 0.42, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert captured, "run() was not called"
        assert captured[0].get("atoh_sensitivity") == 0.42

    def test_atoh_threshold_not_used_as_sensitivity(self, monkeypatch):
        # Classifying by atoh_level -> the threshold is an ATOH threshold, NOT a
        # delta one, so it must NOT be repurposed as atoh_sensitivity.
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [], -1, 0.5, "atoh_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert captured
        assert "atoh_sensitivity" not in captured[0]

    def test_no_threshold_leaves_default(self, monkeypatch):
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [], -1, None, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert captured
        assert "atoh_sensitivity" not in captured[0]

    def test_nondefault_atoh_gets_distinct_folder_default_preserved(self):
        import run_model
        f = run_model._short_run_folder_name
        default = f("sheetA", 0.55, 2.75, 2.59, 0)
        nondef = f("sheetA", 0.55, 2.75, 2.59, 0, atoh_sensitivity=0.42)
        # default keeps its historical name (no patoh in the hashed canonical),
        # a non-default atoh yields a DISTINCT folder so reuse can't collide.
        assert "patoh" not in default
        assert nondef != default
        # and the default equals the same call with the explicit default value.
        assert default == f("sheetA", 0.55, 2.75, 2.59, 0,
                            atoh_sensitivity=run_model._DEFAULT_ATOH_SENSITIVITY)

    def test_base_and_ablation_relaxed_to_their_own_thresholds(self, monkeypatch):
        # The base (un-ablated) run must relax to base_quasi_static_threshold and
        # the ablation run to the (tighter) ablation_quasi_static_threshold.
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [3, 7], -1, None, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, None, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert len(captured) == 2  # base run, then ablation run
        assert captured[0]["quasi_static_threshold"] == 0.03  # base
        assert captured[1]["quasi_static_threshold"] == 0.02  # ablation

    def test_nondefault_quasi_static_threshold_gets_distinct_folder(self):
        import run_model
        f = run_model._short_run_folder_name
        default = f("sheetA", 0.55, 2.75, 2.59, 0)
        base = f("sheetA", 0.55, 2.75, 2.59, 0, quasi_static_threshold=0.03)
        abl = f("sheetA", 0.55, 2.75, 2.59, 0, quasi_static_threshold=0.02)
        # Historical 0.01 name keeps no qst tag; 0.03 and 0.02 each get a
        # DISTINCT self-describing folder so reuse_existing_run can't collide.
        assert "qst" not in default
        assert "_qst0.030" in base and "_qst0.020" in abl
        assert len({default, base, abl}) == 3
        # explicit default equals the implicit default (no needless recompute).
        assert default == f("sheetA", 0.55, 2.75, 2.59, 0,
                            quasi_static_threshold=run_model._DEFAULT_QUASI_STATIC_THRESHOLD)

    def test_line_tension_reaches_base_and_ablation(self, monkeypatch):
        # A fit-level line_tension must reach BOTH the base and the ablation run.
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [3, 7], -1, None, "delta_level",
                None, None, 30.0, False, 0.0, 0.03, 0.02, 0.05, None, None, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert len(captured) == 2
        assert captured[0]["line_tension"] == 0.05  # base
        assert captured[1]["line_tension"] == 0.05  # ablation

    def test_per_type_shape_index_reaches_runs_and_folder(self, monkeypatch):
        # A type-dependent shape index must reach BOTH runs and mint a DISTINCT,
        # self-describing folder (it changes the relaxed morphology).
        run_model, captured = self._stub(monkeypatch)
        task = (0.55, 2.75, 2.59, "sheetA", "E17.5", [3, 7], -1, None, "delta_level",
                None, None, 30.0, False, 1.3, 0.03, 0.02, None, 1.2, 1.4, None, None)
        run_model._evaluate_mechanics_for_sheet(task)
        assert len(captured) == 2
        for kw in captured:                      # base and ablation
            assert kw["hc_shape_index"] == 1.2
            assert kw["sc_shape_index"] == 1.4
        f = run_model._short_run_folder_name
        shared = f("sheetA", 0.55, 2.75, 2.59, 0, shape_index=1.3)
        split = f("sheetA", 0.55, 2.75, 2.59, 0, shape_index=1.3,
                  hc_shape_index=1.2, sc_shape_index=1.4)
        assert "p0hc" not in shared
        assert "_p0hc1.20_p0sc1.40" in split and split != shared

    def test_line_tension_gets_distinct_folder(self):
        import run_model
        f = run_model._short_run_folder_name
        default = f("sheetA", 0.55, 2.75, 2.59, 0)
        lt = f("sheetA", 0.55, 2.75, 2.59, 0, line_tension=0.05)
        # No line tension -> historical name; a line-tension run gets a DISTINCT
        # self-describing folder so it never reuses a no-line-tension archive.
        assert "lt" not in default
        assert "_lt0.050" in lt and lt != default
        assert default == f("sheetA", 0.55, 2.75, 2.59, 0, line_tension=None)
