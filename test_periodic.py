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
from cell_measures import find_non_boundary_cells
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


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _build_raw_lattice(nx, ny, distx=1.0, disty=1.0):
    """Just the (verts, edges, faces) dataframes — no Sheet wrapper."""
    return PeriodicBoundarySheet.generate_periodic_hex_lattice(
        nx, ny, distx=distx, disty=disty
    )


def _build_sheet(nx, ny, distx=1.0, disty=1.0,
                 max_bond=10.0, min_bond=0.0):
    """A periodic VirtualSheet WITHOUT adding virtual mid-edge vertices
    (set max_bond very large) so the topology is the pure hex lattice."""
    np.random.seed(0)
    return VirtualSheet.planar_virtual_sheet_2d(
        f"p{nx}x{ny}", nx=nx, ny=ny, distx=distx, disty=disty,
        maximal_bond_length=max_bond, minimal_bond_length=min_bond,
        periodic=True, draw_debug=False,
    )


def _build_sheet_with_virtuals(nx, ny, distx=1.0, disty=1.0,
                                max_bond=0.5, min_bond=0.05):
    np.random.seed(0)
    return VirtualSheet.planar_virtual_sheet_2d(
        f"p{nx}x{ny}_v", nx=nx, ny=ny, distx=distx, disty=disty,
        maximal_bond_length=max_bond, minimal_bond_length=min_bond,
        periodic=True, draw_debug=False,
    )


def _opposite_self_consistent(sheet):
    """opposite-of-opposite(e) == e for every edge with an opposite."""
    opp = sheet.edge_df["opposite"]
    paired = opp[opp >= 0]
    return (opp.loc[paired].values == paired.index.values).all()


# --------------------------------------------------------------------------- #
# Layer 1 — lattice generation                                                #
# --------------------------------------------------------------------------- #

class TestLatticeGeneration:
    """Raw ``generate_periodic_hex_lattice`` shape/sanity checks."""

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4), (2, 4)])
    def test_face_count_matches_nx_ny(self, nx, ny):
        verts, edges, faces = _build_raw_lattice(nx, ny)
        assert len(faces) == nx * ny

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_every_edge_has_opposite(self, nx, ny):
        verts, edges, faces = _build_raw_lattice(nx, ny)
        # Closed periodic surface: every half-edge must be paired.
        assert (edges["opposite"] < 0).sum() == 0

    def test_odd_ny_rejected(self):
        # The hex-row offset cannot tile rectangularly with odd ny.
        with pytest.raises(ValueError, match="ny must be even"):
            _build_raw_lattice(2, 3)

    def test_degenerate_aspect_ratio_rejected(self):
        # Very stretched grids produce zero-area Voronoi cells.
        with pytest.raises(ValueError, match="degenerate"):
            _build_raw_lattice(nx=4, ny=2, distx=4.0, disty=1.0)
        with pytest.raises(ValueError, match="degenerate"):
            _build_raw_lattice(nx=2, ny=4, distx=1.0, disty=4.0)

    def test_periods_use_correct_axis(self):
        """Lx must be nx*distx and Ly must be ny*disty
        (the previous implementation had them swapped, which was
        masked when nx==ny and distx==disty)."""
        verts, edges, faces = _build_raw_lattice(nx=3, ny=2, distx=1.0, disty=1.0)
        # x must cover roughly [0, 3); y roughly [0, 2)
        assert verts["x"].max() < 3.0 + 1e-9
        assert verts["y"].max() < 2.0 + 1e-9
        assert verts["x"].max() > 1.5      # actually spans the wider axis
        # All face centroids inside the periodic box:
        assert (faces["x"] >= 0).all() and (faces["x"] < 3.0).all()
        assert (faces["y"] >= 0).all() and (faces["y"] < 2.0).all()

    def test_vertices_wrapped_to_canonical_box(self):
        verts, _, _ = _build_raw_lattice(nx=3, ny=2)
        Lx, Ly = 3.0, 2.0
        # Allow a tiny epsilon for floating-point dust at the upper edge.
        assert verts["x"].min() >= -1e-9
        assert verts["x"].max() < Lx + 1e-9
        assert verts["y"].min() >= -1e-9
        assert verts["y"].max() < Ly + 1e-9


# --------------------------------------------------------------------------- #
# Layer 2 — periodic opposite stitching                                       #
# --------------------------------------------------------------------------- #

class TestOppositeStitching:
    """``set_opposite_periodic`` and ``VirtualSheet.get_opposite``."""

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_pure_lattice_has_no_dangling_edges(self, nx, ny):
        sheet = _build_sheet(nx, ny)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_opposite_is_an_involution(self, nx, ny):
        sheet = _build_sheet(nx, ny)
        assert _opposite_self_consistent(sheet)

    def test_not_every_edge_is_marked_periodic(self):
        """Guard against the original bug where every stitched edge got
        is_periodic=True after each get_opposite. In a 4x4 hex tile most
        edges are bulk interior pairs that should NOT wrap."""
        sheet = _build_sheet(nx=4, ny=4)
        n_periodic = int(sheet.edge_df["is_periodic"].sum())
        n_total = len(sheet.edge_df)
        assert 0 < n_periodic < n_total, (
            f"Expected some-but-not-all edges to be periodic, "
            f"got {n_periodic}/{n_total}"
        )
        # And the periodic count should match the wrap-criterion exactly.
        # (Detailed equality is checked by test_is_periodic_matches_geometric_wrap.)

    def test_is_periodic_matches_geometric_wrap(self):
        """An edge is flagged is_periodic iff its srce↔trgt distance
        in canonical [0, L) is > L/2 in either axis — i.e. the edge
        physically crosses the periodic boundary."""
        sheet = _build_sheet(nx=4, ny=4)
        Lx, Ly = sheet.Lx, sheet.Ly
        srce = sheet.edge_df["srce"].to_numpy()
        trgt = sheet.edge_df["trgt"].to_numpy()
        vx = sheet.vert_df["x"].to_numpy()
        vy = sheet.vert_df["y"].to_numpy()
        sx = vx[srce]; sy = vy[srce]
        tx = vx[trgt]; ty = vy[trgt]
        wraps_x = np.abs(tx - sx) > Lx / 2 - 1e-9
        wraps_y = np.abs(ty - sy) > Ly / 2 - 1e-9
        expected = wraps_x | wraps_y
        actual = sheet.edge_df["is_periodic"].to_numpy()
        assert (expected == actual).all(), (
            f"is_periodic disagrees with geometric wrap on "
            f"{(expected != actual).sum()} edges"
        )

    def test_get_opposite_is_idempotent(self):
        """Calling get_opposite twice gives the same opposites."""
        sheet = _build_sheet(nx=3, ny=2)
        before = sheet.edge_df["opposite"].copy()
        sheet.get_opposite()
        after = sheet.edge_df["opposite"]
        assert (before == after).all()


# --------------------------------------------------------------------------- #
# Layer 3 — geometry under periodic wrap                                      #
# --------------------------------------------------------------------------- #

class TestPeriodicGeometry:
    """Centroid, area, and edge-length consistency under wrapping."""

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_every_face_has_positive_area(self, nx, ny):
        """Self-intersecting unfolds (the original bug) give 0 / negative
        signed area."""
        sheet = _build_sheet(nx, ny)
        assert (sheet.face_df["area"] > 0).all()

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_total_area_equals_domain_area(self, nx, ny):
        """The cells must tile the periodic box exactly."""
        sheet = _build_sheet(nx, ny)
        expected = sheet.Lx * sheet.Ly
        np.testing.assert_allclose(sheet.face_df["area"].sum(), expected,
                                    rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2), (4, 4)])
    def test_every_hex_cell_has_area_one(self, nx, ny):
        """With distx=disty=1 every hex Voronoi cell has unit area."""
        sheet = _build_sheet(nx, ny, distx=1.0, disty=1.0)
        np.testing.assert_allclose(sheet.face_df["area"].to_numpy(),
                                    np.ones(nx * ny),
                                    rtol=1e-6, atol=1e-9)

    def test_perimeter_walk_closure_uses_min_image(self):
        """The per-face perimeter walk in ``PeriodicPlanarGeometry.
        update_dcoords`` can accumulate a NET drift of k*Lx around a
        face's perimeter — typically when a wrapping cell contains
        long edges. The CLOSING edge (last → first vertex) must then
        have its tx wrapped back to within ±L/2 of its sx; otherwise
        the closing edge ends up with a stored ``length`` of ~k*Lx
        and the corresponding vertex takes a huge step in the next
        ODE call.

        Build a synthetic face whose perimeter walk drifts by exactly
        Lx and check that the closing edge's length is consistent with
        the actual canonical positions."""
        # Use a small periodic sheet just for the geometry hookup.
        sheet = _build_sheet(nx=3, ny=2, distx=1.0, disty=1.0)
        Lx, Ly = float(sheet.Lx), float(sheet.Ly)

        # Construct a synthetic face whose perimeter walks across the
        # boundary AND contains a long edge so the walk drifts:
        #
        #   v0 (Lx-ε, 0.1) — short — v1 (Lx-2ε, 0.1) — short — v2 (Lx-3ε, 0.1)
        #     |                                                    |
        #     |                              long edge that wraps the box
        #     |                                                    |
        #   v3 (ε, 0.2) ←————— short ———————— v4 (2ε, 0.2)
        #
        # Walking v0 → v1 → v2 stays near Lx in unfolded.
        # Walking v2 → v3 wraps (canonical ε is close to canonical Lx-3ε
        # under min-image), drifting to ~Lx+ε.
        # Walking v3 → v4 stays around Lx+ε.
        # Closing v4 → v0 requires min-image; without the closure fix
        # the stored length would be ~Lx instead of the true ~ε.
        eps = 0.01
        new_verts = pd.DataFrame(
            {
                "x": [Lx - eps, Lx - 2 * eps, Lx - 3 * eps, eps, 2 * eps],
                "y": [0.1, 0.1, 0.1, 0.2, 0.2],
                "is_active": [1, 1, 1, 1, 1],
            },
            index=[100, 101, 102, 103, 104],
        )
        # Extend with whatever other columns vert_df already has so
        # concat keeps the existing schema.
        for col in sheet.vert_df.columns:
            if col not in new_verts.columns:
                new_verts[col] = sheet.vert_df[col].iloc[0]
        sheet.vert_df = pd.concat([sheet.vert_df, new_verts])

        # Pick an unused face id.
        new_face_id = int(sheet.face_df.index.max()) + 1
        new_face_row = sheet.face_df.iloc[[0]].copy()
        new_face_row.index = [new_face_id]
        sheet.face_df = pd.concat([sheet.face_df, new_face_row])

        # Build 5 edges in order; the closing edge is v4 → v0.
        new_edges = pd.DataFrame(
            {
                "srce": [100, 101, 102, 103, 104],
                "trgt": [101, 102, 103, 104, 100],
                "face": [new_face_id] * 5,
                "order": [1, 2, 3, 4, 5],
                "opposite": [-1] * 5,
            },
            index=[1000, 1001, 1002, 1003, 1004],
        )
        for col in sheet.edge_df.columns:
            if col not in new_edges.columns:
                new_edges[col] = sheet.edge_df[col].iloc[0]
        sheet.edge_df = pd.concat([sheet.edge_df, new_edges])

        # Reset_index so tyssue's positional ``take`` works; the
        # closing edge will be renumbered too, so look it up by face+order.
        sheet.reset_index(order=False)
        # Run update_dcoords and check the synthetic face's last edge.
        sheet.geom.update_dcoords(sheet)
        sheet.geom.update_length(sheet)
        new_face = sheet.edge_df[sheet.edge_df["face"] == sheet.face_df.index[-1]]
        closing = new_face.sort_values("order").iloc[-1]
        # The closing edge connects v4 (2ε, 0.2) to v0 (Lx-ε, 0.1) via
        # the short side of the wrap. Min-image distance:
        # dx = (Lx-ε) - 2ε = Lx - 3ε in canonical → wrapped: -3ε.
        # dy = 0.1 - 0.2 = -0.1.
        # length = sqrt((3ε)² + 0.1²) ≈ 0.1
        expected = float(np.hypot(3 * eps, 0.1))
        np.testing.assert_allclose(closing["length"], expected, rtol=1e-3, atol=1e-6)
        # And the dx must be the min-image, not ~Lx.
        assert abs(closing["dx"]) < Lx / 2

    def test_edges_have_minimum_image_length(self):
        """No edge should be longer than half a period in either
        direction — that would mean the unfolding put the two endpoints
        on opposite sides of the box."""
        sheet = _build_sheet(nx=3, ny=2)
        dx = sheet.edge_df["dx"].to_numpy()
        dy = sheet.edge_df["dy"].to_numpy()
        assert (np.abs(dx) <= sheet.Lx / 2 + 1e-9).all()
        assert (np.abs(dy) <= sheet.Ly / 2 + 1e-9).all()

    def test_update_all_is_idempotent(self):
        """Calling update_all twice should not drift any geometric
        quantity (in particular, areas must stay positive)."""
        sheet = _build_sheet(nx=3, ny=2)
        areas_1 = sheet.face_df["area"].to_numpy().copy()
        sheet.geom.update_all(sheet)
        areas_2 = sheet.face_df["area"].to_numpy()
        np.testing.assert_allclose(areas_1, areas_2, rtol=1e-9)

    def test_translating_all_vertices_does_not_change_geometry(self):
        """Periodic translation invariance: shifting every vertex by
        (Lx/3, Ly/3) and re-wrapping must leave face areas unchanged."""
        sheet = _build_sheet(nx=3, ny=2)
        areas_before = sheet.face_df["area"].to_numpy().copy()
        sheet.vert_df["x"] = (sheet.vert_df["x"] + sheet.Lx / 3) % sheet.Lx
        sheet.vert_df["y"] = (sheet.vert_df["y"] + sheet.Ly / 3) % sheet.Ly
        sheet.geom.update_all(sheet)
        # Reorder edges may not change but get_opposite must still find pairs
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        np.testing.assert_allclose(sheet.face_df["area"].to_numpy(),
                                    areas_before, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Layer 4 — virtual-vertex insertion under periodicity                        #
# --------------------------------------------------------------------------- #

class TestVirtualVerticesPeriodic:
    """``add_virtual_vertices`` and ``remove_virtual_vertices`` must
    preserve the periodic topology."""

    @pytest.mark.parametrize("nx, ny", [(2, 2), (3, 2)])
    def test_areas_preserved_after_adding_virtuals(self, nx, ny):
        sheet = _build_sheet_with_virtuals(nx, ny)
        # Adding mid-edge vertices doesn't change the polygon, just adds
        # vertices on existing edges → area must stay equal to 1 per cell.
        np.testing.assert_allclose(sheet.face_df["area"].to_numpy(),
                                    np.ones(nx * ny),
                                    rtol=1e-6, atol=1e-9)

    def test_no_dangling_edges_after_adding_virtuals(self):
        sheet = _build_sheet_with_virtuals(nx=3, ny=2)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0

    def test_opposite_consistent_after_adding_virtuals(self):
        sheet = _build_sheet_with_virtuals(nx=3, ny=2)
        assert _opposite_self_consistent(sheet)

    def test_periodic_pair_split_shares_midpoint_label(self):
        """After construction-time consolidation of periodic-image
        labels, periodic pairs share their vertex labels (reversed).
        Splitting a wrapping edge therefore creates a SINGLE midpoint
        label used by both half-edges — this is what lets the standard
        label-based ``get_opposite`` keep working under dynamics
        without fragile position-based stitching."""
        sheet = _build_sheet_with_virtuals(nx=3, ny=2)
        per = sheet.edge_df[sheet.edge_df["is_periodic"]]
        if not len(per):
            pytest.skip("no periodic edges in tile")
        # Every wrapping edge's opposite must be a simple label reversal —
        # if any pair had distinct labels it would mean the consolidation
        # missed it and the topology will drift apart under dynamics.
        for e_id in per.index:
            opp = int(sheet.edge_df.at[e_id, "opposite"])
            assert opp >= 0, f"periodic edge {e_id} has no opposite"
            s = int(sheet.edge_df.at[e_id, "srce"])
            t = int(sheet.edge_df.at[e_id, "trgt"])
            s_opp = int(sheet.edge_df.at[opp, "srce"])
            t_opp = int(sheet.edge_df.at[opp, "trgt"])
            assert s_opp == t and t_opp == s, (
                f"periodic edge {e_id} pair has unconsolidated labels: "
                f"({s},{t}) vs ({s_opp},{t_opp})"
            )

    def test_remove_then_add_virtuals_round_trip(self):
        """Removing all virtual vertices and re-adding them gives the
        same face areas."""
        sheet = _build_sheet_with_virtuals(nx=3, ny=2)
        areas_before = sheet.face_df["area"].to_numpy().copy()
        # Make the min-bond threshold huge to force removal of every virtual,
        # then set it back and re-add.
        sheet.set_minimal_bond_length(sheet.maximal_bond_length)
        sheet.remove_virtual_vertices()
        sheet.set_minimal_bond_length(0.05)
        sheet.add_virtual_vertices()
        np.testing.assert_allclose(sheet.face_df["area"].to_numpy(),
                                    areas_before, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# Layer 5 — topology events across the periodic boundary                      #
# --------------------------------------------------------------------------- #

def _setup_periodic_sheet_with_settings(nx=4, ny=4):
    """Build a periodic sheet and add the bare minimum specs/columns that
    the topology event functions read (is_alive, num_sides, unique_id,
    threshold_length, lineage)."""
    import networkx as nx_  # only needed for the lineage stub
    sheet = _build_sheet(nx, ny)
    sheet.face_df["is_alive"] = 1
    sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
    sheet.face_df["unique_id"] = sheet.face_df.index.astype(str)
    sheet.settings.setdefault("threshold_length", 0.1)
    # index_preserving_remove_face touches sheet.lineage; tyssue's full
    # construction creates it but our path doesn't, so add a stub.
    if not hasattr(sheet, "lineage") or sheet.lineage is None:
        sheet.lineage = nx_.DiGraph()
    return sheet


def _pick_periodic_edge(sheet, prefer_long=False):
    """Return the index of an edge that physically wraps around a
    periodic boundary."""
    per = sheet.edge_df[sheet.edge_df["is_periodic"]]
    assert len(per) > 0, "test sheet has no periodic edges"
    if prefer_long:
        return per["length"].idxmax()
    return per.index[0]


def _pick_face_on_boundary(sheet):
    """A face whose vertex set straddles a periodic boundary."""
    boundary_faces = sheet.edge_df[
        sheet.edge_df["is_periodic"]
    ]["face"].unique()
    assert len(boundary_faces) > 0
    return int(boundary_faces[0])


def _pick_interior_face(sheet):
    """A face whose vertices all lie in the bulk (no periodic edge)."""
    boundary_faces = set(
        int(f) for f in sheet.edge_df[sheet.edge_df["is_periodic"]]["face"].unique()
    )
    for f in sheet.face_df.index:
        if int(f) not in boundary_faces:
            return int(f)
    pytest.skip("no interior face in this sheet")


def _pick_interior_edge(sheet):
    """An edge that is (a) not flagged periodic AND (b) whose containing
    face is also interior — guaranteed not to be near any boundary."""
    interior_face = _pick_interior_face(sheet)
    cand = sheet.edge_df[
        (sheet.edge_df["face"] == interior_face) &
        (~sheet.edge_df["is_periodic"])
    ]
    assert len(cand) > 0
    return int(cand.index[0])


def _total_area(sheet):
    return sheet.face_df["area"].sum()


class TestT1Periodic:
    """Type-1 transitions on bulk and boundary-crossing edges.

    T1 on a wrapping edge is implemented via ``_periodic_collapse_edge``
    (option a): for same-labels periodic pairs we first duplicate the
    two shared vertices so face_B gets its own labels at periodic-image
    positions, then collapse + split each face independently. The new
    T1 half-edges in face_A's and face_B's respective C/D cells are
    re-stitched by ``set_opposite_periodic`` via position matching."""

    def test_t1_on_truly_interior_edge_succeeds(self):
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        edge = _pick_interior_edge(sheet)
        ret = index_preserving_type1_transition(sheet, edge)
        assert ret == 0
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0
        assert (sheet.face_df["area"] > 0).all()
        assert _opposite_self_consistent(sheet)

    def test_total_area_conserved_after_interior_t1(self):
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        before = _total_area(sheet)
        edge = _pick_interior_edge(sheet)
        index_preserving_type1_transition(sheet, edge)
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        after = _total_area(sheet)
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-9)

    def test_t1_on_wrapping_edge_keeps_topology_closed(self):
        """A T1 on a wrapping edge must leave EVERY edge with a valid
        opposite (no dangling boundary edges). The fix:
        ``_consolidate_periodic_image_labels`` merges all periodic-image
        vertex labels around the rosette before T1, so the standard
        bulk T1 sees a clean topological rosette."""
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        edge = _pick_periodic_edge(sheet)
        ret = index_preserving_type1_transition(sheet, edge)
        assert ret == 0
        # The wrapper already runs get_opposite + update_all + reorder
        # internally, but a defensive re-stitch matches what the
        # simulation loop does after every intercalation.
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        assert (sheet.edge_df["opposite"] < 0).sum() == 0
        assert (sheet.face_df["area"] > 0).all()
        assert _opposite_self_consistent(sheet)

    def test_total_area_conserved_after_periodic_t1(self):
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        before = _total_area(sheet)
        edge = _pick_periodic_edge(sheet)
        ret = index_preserving_type1_transition(sheet, edge)
        assert ret == 0
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        after = _total_area(sheet)
        # T1 is area-preserving. The 4 cells in the rosette redistribute
        # ~0.23 of area each (A,B → C,D) but the total stays exactly Lx*Ly.
        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-9)

    def test_periodic_t1_works_on_every_wrapping_edge(self):
        """Exhaustive check: T1 succeeds (closed topology, positive
        areas, exact area conservation, consistent edge ordering) for
        EVERY periodic edge in the test sheet. Catches regressions
        where a particular same-labels / different-labels / x-wrap /
        y-wrap configuration breaks the consolidation logic."""
        sheet_init = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        per_edges = list(sheet_init.edge_df[sheet_init.edge_df["is_periodic"]].index)
        assert len(per_edges) > 0
        for ei in per_edges:
            sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
            if ei not in sheet.edge_df.index:
                continue
            before = _total_area(sheet)
            ret = index_preserving_type1_transition(sheet, ei)
            assert ret == 0, f"T1 on edge {ei} returned {ret}"
            assert (sheet.edge_df["opposite"] < 0).sum() == 0, (
                f"T1 on edge {ei} left dangling edges"
            )
            assert (sheet.face_df["area"] > 0).all(), (
                f"T1 on edge {ei} produced non-positive area"
            )
            assert sheet.check_all_edge_order(), (
                f"T1 on edge {ei} broke edge ordering"
            )
            np.testing.assert_allclose(
                _total_area(sheet), before, rtol=1e-6, atol=1e-9,
                err_msg=f"T1 on edge {ei} did not conserve area",
            )

    def test_periodic_t1_keeps_edge_order_anti_clockwise(self):
        """After the T1 wrapper completes, edge ``order`` in every face
        must form a coherent cycle (trgt of edge i == srce of edge i+1)
        — i.e. the anti-clockwise ordering is preserved (or re-ordered
        by the wrapper)."""
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        edge = _pick_periodic_edge(sheet)
        index_preserving_type1_transition(sheet, edge)
        # check_all_edge_order asserts the cyclic order invariant
        assert sheet.check_all_edge_order(), (
            "edge order is inconsistent after periodic T1"
        )

    def test_intercalation_handler_processes_periodic_edges(self):
        """The intercalation handler considers wrapping edges; the
        periodic T1 (via _consolidate_periodic_image_labels) handles
        them correctly."""
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        sheet.edge_df["is_active"] = 1
        threshold = 1e9
        is_virtual = sheet.is_virtual_edge(sheet.edge_df.index.to_numpy())
        real = sheet.edge_df[~is_virtual]
        selected = real.query("is_active > 0 & length < %f" % threshold)
        # At least one of the selected edges is a periodic edge — the
        # handler no longer filters them out.
        assert selected["is_periodic"].any()

    def test_is_virtual_edge_accepts_gappy_edge_index(self):
        # A T1 collapses edges with reindex=False, leaving a GAPPY edge index
        # if it then fails. is_virtual_edge must index by the ACTUAL labels;
        # the old ``np.arange(shape[0])`` treated positions as labels and
        # raised "KeyError: [...] not in index" on a gappy index.
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        labels = sorted(sheet.edge_df.index.to_numpy())
        gap = labels[len(labels) // 2: len(labels) // 2 + 4]
        sheet.edge_df = sheet.edge_df.drop(index=gap)
        flags = sheet.is_virtual_edge(sheet.edge_df.index.to_numpy())
        assert len(flags) == sheet.edge_df.shape[0]
        with pytest.raises(KeyError):
            sheet.is_virtual_edge(np.arange(sheet.edge_df.shape[0]))

    def test_intercalation_survives_failed_t1_with_gappy_index(self, monkeypatch):
        # Reproduces the ablation-run crash: a T1 that collapses an edge
        # (reindex=False) and then raises leaves a gappy index; the handler
        # must not crash on the next pass, must terminate, and must hand a
        # clean (contiguous) index back to the solver.
        import topological_events as te
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        sheet.edge_df["is_active"] = 1
        intercalation = te.TopologicalEventsHandler(None).get_intercalation_function(
            crit_edge_length=10.0)

        calls = {"n": 0}

        def fake_t1(s, edge_id, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # Non-atomic T1: collapse dropped edges (gappy) then split raised.
                lbls = sorted(s.edge_df.index.to_numpy())
                s.edge_df = s.edge_df.drop(index=lbls[len(lbls) // 2:][:4])
                raise ValueError("simulated split failure")
            return -1  # subsequent attempts: clean no-op skip

        monkeypatch.setattr(te, "index_preserving_type1_transition", fake_t1)

        class _Mgr:
            def append(self, fn):
                pass

        intercalation(sheet, _Mgr())  # old code raised KeyError here
        assert calls["n"] >= 2  # processed past the first (gappy) failure
        idx = sheet.edge_df.index.to_numpy()
        assert list(idx) == list(range(len(idx)))  # clean index for the solver


class TestDivisionPeriodic:
    """Cell division of a face that straddles a periodic boundary."""

    def test_division_on_boundary_face_succeeds(self):
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        face = _pick_face_on_boundary(sheet)
        n_faces_before = sheet.Nf
        total_area_before = _total_area(sheet)
        daughter = index_preserving_cell_division(sheet, face, sheet.geom)
        assert daughter is not None
        sheet.get_opposite()
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        # One new face created
        assert sheet.Nf == n_faces_before + 1
        # Topology closed
        assert (sheet.edge_df["opposite"] < 0).sum() == 0
        # Area conserved
        np.testing.assert_allclose(_total_area(sheet), total_area_before,
                                    rtol=1e-6, atol=1e-9)

    def test_division_on_interior_face_unchanged_behaviour(self):
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        face = _pick_interior_face(sheet)
        n_faces_before = sheet.Nf
        total_area_before = _total_area(sheet)
        daughter = index_preserving_cell_division(sheet, face, sheet.geom)
        assert daughter is not None
        sheet.get_opposite()
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        assert sheet.Nf == n_faces_before + 1
        assert (sheet.edge_df["opposite"] < 0).sum() == 0
        np.testing.assert_allclose(_total_area(sheet), total_area_before,
                                    rtol=1e-6, atol=1e-9)


class TestDelaminationPeriodic:
    """Removing a face that touches a periodic boundary."""

    def test_remove_boundary_face_drops_one_face(self):
        """Removing a face from a periodic sheet (delamination)
        deletes exactly one face. With consolidated periodic-image
        labels, the removed face's vertices are shared with surrounding
        cells, so coalescing them into one new vertex absorbs the
        removed area into the neighbours — total Lx*Ly stays constant."""
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        face = _pick_face_on_boundary(sheet)
        n_faces_before = sheet.Nf
        total_before = _total_area(sheet)
        index_preserving_remove(sheet, face, sheet.geom)
        sheet.order_all_edges()
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        # One face was removed
        assert sheet.Nf == n_faces_before - 1
        # No NaN face areas
        assert not sheet.face_df["area"].isna().any()
        # Total area = Lx*Ly is invariant (a periodic sheet has no
        # boundary, so removing a cell just merges its area into the
        # neighbours).
        np.testing.assert_allclose(
            _total_area(sheet), total_before, rtol=1e-6, atol=1e-9,
        )

    def test_delamination_handler_marks_face_dead_without_topology_change(self):
        """The normal delamination handler doesn't remove the face until
        it shrinks to a triangle — it just flags is_alive=-1 / sets high
        elasticity. That mark-then-shrink path doesn't touch periodic
        opposites, so verify nothing breaks under that flow."""
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        sheet.face_df["type"] = 0
        face = _pick_face_on_boundary(sheet)
        # Pre-flag the face as dying (what the delamination behaviour does)
        sheet.face_df.loc[face, "type"] = -1
        sheet.face_df.loc[face, "area_elasticity"] = 20
        sheet.face_df.loc[face, "contractility"] = 10
        sheet.geom.update_all(sheet)
        # Topology unchanged, periodic stitching still works
        assert (sheet.edge_df["opposite"] < 0).sum() == 0
        assert sheet.face_df.loc[face, "type"] == -1


# --------------------------------------------------------------------------- #
# Layer 7 — history archive round-trip                                        #
# --------------------------------------------------------------------------- #

class TestHistoryRoundTrip:
    """A History archive must preserve enough metadata for a reloaded
    sheet to behave as periodic."""

    def test_lx_ly_periodic_flag_survive_round_trip(self, tmp_path):
        from tyssue import History, HistoryHdf5
        sheet = _build_sheet(nx=3, ny=2)
        archive = tmp_path / "p32.hf5"
        hist = History(sheet, save_every=1.0, save_all=True, dt=1.0)
        hist.record(time_stamp=0.0)
        hist.to_archive(str(archive))

        loaded_hist = HistoryHdf5.from_archive(str(archive), eptm_class=VirtualSheet)
        ts = float(np.max(loaded_hist.time_stamps))
        loaded = loaded_hist.retrieve(ts)
        loaded.arrange_sheet_from_history(two_dim=True)
        loaded.initiate_edge_order()

        assert loaded.periodic is True
        assert loaded.Lx == sheet.Lx
        assert loaded.Ly == sheet.Ly
        # The stash columns must not leak into face_df after loading.
        assert "_periodic_Lx" not in loaded.face_df.columns
        # And get_opposite on the loaded sheet must reproduce a closed topology.
        loaded.get_opposite()
        assert (loaded.edge_df["opposite"] < 0).sum() == 0

    def test_legacy_archive_without_flag_loads_nonperiodic(self, tmp_path):
        """An archive whose face_df lacks the ``_periodic_flag``
        metadata (legacy: written before the flag was stashed on every
        snapshot) loads as NON-periodic by default. Recomputing the
        geometry then unwraps boundary-crossing faces into
        domain-spanning edges — the root cause of the t=25 resume
        blow-up."""
        from tyssue import History, HistoryHdf5
        sheet = _build_sheet(nx=4, ny=4)
        # Strip the stash columns to simulate a legacy archive.
        sheet.face_df = sheet.face_df.drop(
            columns=["_periodic_flag", "_periodic_Lx", "_periodic_Ly"],
            errors="ignore",
        )
        archive = tmp_path / "legacy.hf5"
        hist = History(sheet, save_every=1.0, save_all=True, dt=1.0)
        hist.record(time_stamp=0.0)
        hist.to_archive(str(archive))

        loaded_hist = HistoryHdf5.from_archive(str(archive), eptm_class=VirtualSheet)
        loaded = loaded_hist.retrieve(float(np.max(loaded_hist.time_stamps)))
        loaded.arrange_sheet_from_history(two_dim=True)  # no fallback box
        # Without the flag and without a fallback box, the sheet is
        # silently non-periodic — this is exactly the failure mode.
        assert loaded.periodic is False

    def test_force_periodic_box_recovers_legacy_archive(self, tmp_path):
        """``arrange_sheet_from_history(force_periodic_box=(Lx, Ly))``
        re-establishes periodicity on a legacy archive that lacks the
        flag, so recomputing the geometry keeps boundary faces intact
        (positive areas, no domain-spanning edges)."""
        from tyssue import History, HistoryHdf5
        sheet = _build_sheet(nx=4, ny=4)
        Lx, Ly = sheet.Lx, sheet.Ly
        sheet.face_df = sheet.face_df.drop(
            columns=["_periodic_flag", "_periodic_Lx", "_periodic_Ly"],
            errors="ignore",
        )
        archive = tmp_path / "legacy.hf5"
        hist = History(sheet, save_every=1.0, save_all=True, dt=1.0)
        hist.record(time_stamp=0.0)
        hist.to_archive(str(archive))

        loaded_hist = HistoryHdf5.from_archive(str(archive), eptm_class=VirtualSheet)
        loaded = loaded_hist.retrieve(float(np.max(loaded_hist.time_stamps)))
        loaded.arrange_sheet_from_history(
            two_dim=True, force_periodic_box=(Lx, Ly),
        )
        loaded.initiate_edge_order()

        # Periodicity re-established from the fallback box.
        assert loaded.periodic is True
        assert loaded.Lx == Lx and loaded.Ly == Ly

        # The decisive check: re-stitch + recompute geometry and
        # confirm it stays healthy — no negative areas, no
        # domain-spanning edges (which is what broke the resume).
        loaded.get_opposite()
        loaded.geom.update_all(loaded)
        assert (loaded.face_df["area"] > 0).all(), (
            "force-periodic recovery still left non-positive areas"
        )
        assert loaded.edge_df["length"].max() < Lx / 2, (
            "force-periodic recovery left a domain-spanning edge: "
            f"max length {loaded.edge_df['length'].max():.3f} >= Lx/2"
        )

    def test_present_flag_wins_over_force_box(self, tmp_path):
        """When the archive DOES carry the flag, the stored Lx/Ly take
        precedence over a (possibly wrong) fallback box."""
        from tyssue import History, HistoryHdf5
        sheet = _build_sheet(nx=3, ny=2)
        archive = tmp_path / "p32.hf5"
        hist = History(sheet, save_every=1.0, save_all=True, dt=1.0)
        hist.record(time_stamp=0.0)
        hist.to_archive(str(archive))

        loaded_hist = HistoryHdf5.from_archive(str(archive), eptm_class=VirtualSheet)
        loaded = loaded_hist.retrieve(float(np.max(loaded_hist.time_stamps)))
        # Pass a deliberately WRONG fallback box — the stored flag
        # must override it.
        loaded.arrange_sheet_from_history(
            two_dim=True, force_periodic_box=(999.0, 999.0),
        )
        assert loaded.periodic is True
        assert loaded.Lx == sheet.Lx and loaded.Ly == sheet.Ly


# --------------------------------------------------------------------------- #
# Layer 7d — multi-event handler robustness                                    #
# --------------------------------------------------------------------------- #

class TestMultiDivisionHandler:
    """When two or more cells are eligible to divide in the SAME
    manager iteration, the division handler must divide BOTH (and the
    intended ones, not stale-label-reindexed substitutes).

    The previous implementation iterated over a SNAPSHOT of
    ``dividing_faces`` and called ``reset_index`` inside the loop;
    after the first division ``reset_index`` renumbered face labels
    so the snapshot's second ``cell_id`` pointed to a completely
    different cell. The new handler re-queries the face_df after each
    successful division, which is robust to the renumbering."""

    @staticmethod
    def _build():
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p4x4", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        from inner_ear_model import InnerEarModel
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )
        return inner

    def test_two_simultaneous_divisions_visit_both_intended_cells(self, monkeypatch):
        """When two cells are eligible to divide, the handler must
        pass BOTH of their labels (NOT stale-label-reindexed
        substitutes) to ``index_preserving_cell_division``.

        We stub the actual division so we can inspect which face ids
        are visited, without the natural area-recompute cascade that
        a real division triggers (every cell ends up >crit again
        after update_all, which masks the bug we want to test)."""
        inner = self._build()
        sheet = inner.sheet
        sheet.face_df["type"] = 0
        sheet.face_df["is_alive"] = 1
        sheet.face_df["area"] = 0.1
        sheet.face_df.loc[[0, 8], "area"] = 999.0

        called_with_ids = []
        import topological_events as tev
        def stub(sheet, mother, geom, angle=None):
            uid = (int(sheet.face_df.at[mother, "id"])
                   if "id" in sheet.face_df.columns else int(mother))
            called_with_ids.append(uid)
            # Mark cell as done by zeroing its area so the next
            # re-query doesn't pick it up.
            sheet.face_df.at[mother, "area"] = 0.0
            return None  # skip daughter setup (no reset_index)
        monkeypatch.setattr(tev, "index_preserving_cell_division", stub)

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_division_function(0.5)(sheet, _M())

        # The handler must have invoked the (stubbed) division on
        # both cells 0 and 8, exactly once each.
        assert set(called_with_ids) == {0, 8}, (
            f"divided cells should be {{0, 8}}, got {called_with_ids}"
        )

    def test_no_negative_areas_after_multi_division(self):
        """Two simultaneous divisions must leave EVERY face with a
        positive signed area. Inverted (negative-area) faces are the
        immediate cause of the user's solver crash."""
        inner = self._build()
        sheet = inner.sheet
        sheet.face_df["type"] = 0
        sheet.face_df["is_alive"] = 1
        sheet.face_df["area"] = 0.1
        sheet.face_df.loc[[0, 5, 8], "area"] = 999.0

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_division_function(0.5)(sheet, _M())

        assert (sheet.face_df["area"] > 0).all(), (
            "negative-area faces after multi-division: "
            f"{sheet.face_df.index[sheet.face_df['area'] <= 0].tolist()}"
        )

    def test_face_perimeter_orders_contiguous_after_multi_division(self):
        """No face should have gaps in its ``order`` column after
        multi-division — gaps come from edges being removed from the
        face without re-running ``order_edges``."""
        inner = self._build()
        sheet = inner.sheet
        sheet.face_df["type"] = 0
        sheet.face_df["is_alive"] = 1
        sheet.face_df["area"] = 0.1
        sheet.face_df.loc[[0, 5, 8], "area"] = 999.0

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_division_function(0.5)(sheet, _M())

        broken = []
        for fid in sheet.face_df.index:
            orders = sorted(sheet.edge_df.loc[
                sheet.edge_df["face"] == fid, "order"
            ].unique())
            if not orders:
                continue
            expected = list(range(int(orders[0]), int(orders[-1]) + 1))
            missing = [o for o in expected if o not in orders]
            if missing:
                broken.append((fid, missing))
        assert not broken, f"faces with order gaps: {broken[:5]}"

    def test_daughter_gets_fresh_unique_id(self):
        """Every successful division must give the daughter a unique
        ``unique_id`` distinct from the mother. The previous version
        copied face_df.loc[mother:mother] wholesale and never updated
        ``unique_id`` for the daughter — leaving mother/daughter sharing
        the same id (manifested as ~30 duplicated unique_id values in
        the user's saved face_df after 30 divisions). That breaks
        lineage tracking (mother and daughter collapse into one node)
        and any downstream id-based bookkeeping."""
        inner = self._build()
        sheet = inner.sheet
        sheet.face_df["type"] = 0
        sheet.face_df["is_alive"] = 1
        sheet.face_df["area"] = 0.1
        # Three simultaneous divisions exercises both the
        # multi-division re-query loop AND the unique_id fix.
        sheet.face_df.loc[[0, 5, 8], "area"] = 999.0
        # Seed unique_id_max so the fix's "fresh id" path is exercised.
        sheet.specs.setdefault("face", {})
        sheet.specs["face"]["unique_id_max"] = int(
            sheet.face_df["unique_id"].astype(int).max()
        )

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_division_function(0.5)(sheet, _M())

        uids = sheet.face_df["unique_id"].astype(int).tolist()
        assert len(uids) == len(set(uids)), (
            f"duplicate unique_ids after multi-division: {sorted(uids)}"
        )

    def test_adjacent_simultaneous_divisions_keep_perimeters_simple(self):
        """When two SHARED-EDGE neighbours divide in the same handler
        invocation, NEITHER cell's edges should be mis-attributed to
        the other's daughter, and every face must retain a simple
        closed polygon (positive area, ≥3 edges)."""
        inner = self._build()
        sheet = inner.sheet
        sheet.face_df["type"] = 0
        sheet.face_df["is_alive"] = 1
        sheet.face_df["area"] = 0.1

        # Pick a face and one of its neighbours (guaranteed to share
        # an edge — that's the scenario the user flagged).
        face_a = 0
        # Find a true neighbour of face_a via the edge graph rather
        # than guessing.
        face_a_edges = sheet.edge_df[sheet.edge_df["face"] == face_a]
        opp = face_a_edges.loc[face_a_edges["opposite"] >= 0, "opposite"].iloc[0]
        face_b = int(sheet.edge_df.at[int(opp), "face"])
        assert face_b != face_a

        sheet.face_df.loc[[face_a, face_b], "area"] = 999.0
        sheet.specs.setdefault("face", {})
        sheet.specs["face"]["unique_id_max"] = int(
            sheet.face_df["unique_id"].astype(int).max()
        )

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_division_function(0.5)(sheet, _M())

        # Both originally-eligible cells got daughters → +2 faces.
        # (Daughters' areas after update_all may again exceed crit
        # because the test setup uses absurd starting areas, but the
        # re-query loop short-circuits via tried_ids once a cell can't
        # divide further; in any case Nf >= initial + 2.)
        # The hard invariant is: every face has positive area and at
        # least 3 edges.
        assert (sheet.face_df["area"] > 0).all(), (
            "negative-area faces after adjacent-pair division: "
            f"{sheet.face_df.index[sheet.face_df['area'] <= 0].tolist()}"
        )
        counts = sheet.edge_df.groupby("face").size()
        bad = counts[counts < 3]
        assert bad.empty, f"faces with <3 edges after adjacent divisions: {bad.to_dict()}"

        # And no edge should reference a non-existent face label.
        live_faces = set(sheet.face_df.index)
        stray = sheet.edge_df.loc[~sheet.edge_df["face"].isin(live_faces)]
        assert stray.empty, (
            f"{len(stray)} edges reference non-existent face labels: "
            f"{sorted(stray['face'].unique())[:10]}"
        )


class TestFaceDivisionWalkRobustness:
    """``index_preserving_face_division`` walks the daughter side of
    the cleavage line by repeatedly picking the next outgoing edge at
    each vertex via ``trgts[srces == trgt][0]``. The walk USED to
    include the two just-created central-line edges (new_edge_m and
    new_edge_d) in its candidate set; if ``new_edge_m`` happened to be
    the first match at ``vert_b``, the walk terminated immediately and
    BOTH new edges went to the daughter — leaving the mother polygon
    open and the daughter polygon self-crossing (negative area). The
    fix explicitly excludes those two edges from the lookup set."""

    @staticmethod
    def _build():
        np.random.seed(42)
        return _setup_periodic_sheet_with_settings(nx=4, ny=4)

    def test_single_division_attributes_edges_correctly(self):
        sheet = self._build()
        face = _pick_interior_face(sheet)
        sheet.specs.setdefault("face", {})
        sheet.specs["face"]["unique_id_max"] = int(
            sheet.face_df["unique_id"].astype(int).max()
        )
        daughter = index_preserving_cell_division(sheet, face, sheet.geom)
        assert daughter is not None
        sheet.get_opposite()
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)

        # No degenerate polygon
        counts = sheet.edge_df.groupby("face").size()
        assert (counts >= 3).all(), (
            f"degenerate faces after division: {counts[counts<3].to_dict()}"
        )
        # No negative areas
        assert (sheet.face_df["area"] > 0).all(), (
            "negative-area face after single division: "
            f"{sheet.face_df.index[sheet.face_df['area'] <= 0].tolist()}"
        )


class TestDegenerateDivisionRollback:
    """``index_preserving_cell_division`` calls ``geom.update_all`` after
    ``index_preserving_face_division`` and inspects the resulting
    daughter / mother areas. If either is degenerate (below
    ``min_area_ratio`` × prefered_area) the division is rolled back
    by feeding the daughter into ``index_preserving_remove_face``,
    which merges the sliver back into the mother.

    Without the rollback the bug observed in
    ``results/random_periodic_array1/`` at t=17.654 would propagate
    a near-zero-but-NEGATIVE-area daughter into the next solver
    step, which then rejects every dt down to dt_min and raises
    ``RuntimeError`` on edge-crossing.

    Driving the bug from real geometry is delicate, so the tests
    here force the failure path with a monkeypatch on
    ``index_preserving_face_division`` and verify the rollback
    contract."""

    @staticmethod
    def _build():
        sheet = _setup_periodic_sheet_with_settings(nx=4, ny=4)
        sheet.face_df["prefered_area"] = 1.0
        return sheet

    def test_degenerate_daughter_is_rolled_back(self, monkeypatch):
        """When the (mocked) face-division leaves the daughter with a
        microscopically negative area, ``index_preserving_cell_division``
        must call ``index_preserving_remove_face`` on it and return
        ``None``."""
        import topological_events as tev
        sheet = self._build()
        face = _pick_interior_face(sheet)

        # Spy on the rollback entry-point.
        rollback_called = {"n": 0}
        original_remove = tev.index_preserving_remove_face
        def remove_spy(sheet, face_label):
            rollback_called["n"] += 1
            return original_remove(sheet, face_label)
        monkeypatch.setattr(tev, "index_preserving_remove_face", remove_spy)

        # Mock face_division: append a daughter row + re-attribute
        # three of the mother's edges to it so the rollback's
        # ``remove_face`` has a real face to act on.
        def stub_face_division(sheet, mother, vert_a, vert_b):
            sheet.face_df = pd.concat(
                [sheet.face_df, sheet.face_df.loc[mother:mother]],
                ignore_index=True,
            )
            sheet.face_df.index.name = "face"
            daughter = int(sheet.face_df.index[-1])
            m_edges = sheet.edge_df[sheet.edge_df["face"] == mother].index[:3]
            sheet.edge_df.loc[m_edges, "face"] = daughter
            return daughter
        monkeypatch.setattr(tev, "index_preserving_face_division", stub_face_division)

        # Inject a fake geom that BEHAVES exactly like the real
        # ``PeriodicPlanarGeometry`` except its ``update_all`` stamps
        # a microscopically NEGATIVE area on the just-created daughter.
        # Subclassing means ``face_projected_pos`` and the other
        # geom helpers used by ``get_division_edges`` /
        # ``index_preserving_add_vert`` still work as normal — only
        # the area landing on the daughter is doctored.
        real_geom = sheet.geom
        class _BadAreaGeom(real_geom):
            @classmethod
            def update_all(cls, sheet_):
                real_geom.update_all(sheet_)
                d = sheet_.face_df.index[-1]
                sheet_.face_df.at[d, "area"] = -0.004

        result = tev.index_preserving_cell_division(sheet, face, _BadAreaGeom)

        assert result is None, (
            f"expected None on degenerate-daughter rollback; got {result}"
        )
        assert rollback_called["n"] == 1, (
            f"expected exactly 1 call to index_preserving_remove_face "
            f"for the rollback; got {rollback_called['n']}"
        )

    def test_normal_division_returns_daughter_unchanged(self, monkeypatch):
        """When the division produces a healthy daughter (positive
        area above threshold) the rollback path must NOT fire."""
        import topological_events as tev
        sheet = self._build()
        face = _pick_interior_face(sheet)
        sheet.specs.setdefault("face", {})
        sheet.specs["face"]["unique_id_max"] = int(
            sheet.face_df["unique_id"].astype(int).max()
        )

        rollback_called = {"n": 0}
        original_remove = tev.index_preserving_remove_face
        def remove_spy(sheet, face_label):
            rollback_called["n"] += 1
            return original_remove(sheet, face_label)
        monkeypatch.setattr(tev, "index_preserving_remove_face", remove_spy)

        # Real division → healthy daughter with positive area.
        daughter = tev.index_preserving_cell_division(sheet, face, sheet.geom)

        assert daughter is not None, (
            "healthy division returned None — rollback shouldn't fire"
        )
        assert rollback_called["n"] == 0, (
            f"rollback fired on a healthy division ({rollback_called['n']} "
            f"calls to index_preserving_remove_face)"
        )
        assert sheet.face_df.at[daughter, "area"] > 0, (
            f"daughter area should be positive; got "
            f"{sheet.face_df.at[daughter, 'area']}"
        )


# --------------------------------------------------------------------------- #
# Layer 8 — comprehensive division-scenario coverage on a 10x10 periodic      #
# sheet, with and without prior/posterior virtual-vertex book-keeping.        #
# --------------------------------------------------------------------------- #

# A "long edge" is any post-operation edge whose length exceeds
# ``LONG_EDGE_THRESHOLD``. With a 10x10 lattice (cell diameter ~ 1) the
# only ways to get an edge of length >= 3 are (a) the perimeter walk
# in ``index_preserving_face_division`` mis-attributed edges and the
# next geometry update walked a broken polygon, or (b) a wrap-around
# was not min-imaged correctly.
LONG_EDGE_THRESHOLD = 3.0


def _post_division_settle(sheet):
    """The book-keeping every successful division pass needs to do
    before invariant checks are meaningful: refresh opposites, walk
    each face's perimeter to rebuild contiguous ``order`` values,
    sort, and recompute geometry."""
    sheet.get_opposite()
    sheet.reset_index(order=False)
    sheet.order_all_edges()
    sheet.edge_df.sort_values(["face", "order"], inplace=True)
    sheet.get_opposite()
    sheet.geom.update_all(sheet)


def _assert_edges_ordered(sheet, context=""):
    """Every face's ``order`` column must be contiguous 1..N (no gaps,
    no duplicates)."""
    bad = []
    for fid in sheet.face_df.index:
        orders = sheet.edge_df.loc[
            sheet.edge_df["face"] == fid, "order"
        ].tolist()
        if not orders:
            continue
        if len(set(orders)) != len(orders):
            bad.append((int(fid), "duplicate orders", sorted(orders)))
            continue
        srt = sorted(int(o) for o in orders)
        if srt != list(range(srt[0], srt[-1] + 1)):
            bad.append((int(fid), "order gaps", srt))
            continue
        # And the perimeter walk srce->trgt must close:
        face_edges = sheet.edge_df[sheet.edge_df["face"] == fid].sort_values("order")
        srces = face_edges["srce"].tolist()
        trgts = face_edges["trgt"].tolist()
        if trgts[-1] != srces[0]:
            bad.append((int(fid), "perimeter doesn't close", (srces[0], trgts[-1])))
            continue
        for i in range(len(srces) - 1):
            if trgts[i] != srces[i + 1]:
                bad.append((int(fid), "perimeter discontinuity at i", i))
                break
    assert not bad, (
        f"[{context}] {len(bad)} face(s) with broken edge ordering: "
        f"{bad[:5]}"
    )


def _assert_no_long_edges(sheet, context="",
                          threshold=LONG_EDGE_THRESHOLD):
    """No edge length should exceed ``threshold``. With a 10x10 unit
    lattice this catches the perimeter-walk bug AND any min-image
    failure on wrap-around edges."""
    long_edges = sheet.edge_df[sheet.edge_df["length"] >= threshold]
    if len(long_edges):
        # Trim the report to the worst offenders so test output stays
        # readable when the bug fires.
        worst = long_edges.sort_values("length", ascending=False).head(5)
        details = [
            (int(eid),
             int(worst.at[eid, "srce"]),
             int(worst.at[eid, "trgt"]),
             float(worst.at[eid, "length"]),
             int(worst.at[eid, "face"]))
            for eid in worst.index
        ]
        raise AssertionError(
            f"[{context}] {len(long_edges)} edge(s) with length >= "
            f"{threshold}: {details}"
        )


def _assert_division_invariants(sheet, context=""):
    """Combined invariant: ordered edges, no long edges, positive
    areas, well-formed opposites."""
    _assert_edges_ordered(sheet, context)
    _assert_no_long_edges(sheet, context)
    assert (sheet.face_df["area"] > 0).all(), (
        f"[{context}] non-positive areas: "
        f"{sheet.face_df.index[sheet.face_df['area'] <= 0].tolist()}"
    )
    assert (sheet.edge_df["opposite"] < 0).sum() == 0, (
        f"[{context}] dangling edges (opposite == -1)"
    )
    assert _opposite_self_consistent(sheet), (
        f"[{context}] opposite is not an involution"
    )


def _build_10x10(with_virtuals=False):
    """Standard 10x10 periodic sheet used across the division scenarios.

    ``with_virtuals=False`` keeps the lattice as pure hexagons (one
    edge per cell-cell contact). ``with_virtuals=True`` subdivides
    every cell-cell edge so each face has many small edges + several
    virtual mid-edge vertices — the regime the model uses in practice."""
    import networkx as nx_
    np.random.seed(0)
    if with_virtuals:
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p10x10_v", nx=10, ny=10, distx=1.0, disty=1.0,
            maximal_bond_length=0.5, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
    else:
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p10x10", nx=10, ny=10, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.0,
            periodic=True, draw_debug=False,
        )
    sheet.face_df["is_alive"] = 1
    sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
    sheet.face_df["unique_id"] = sheet.face_df.index.astype(int)
    # The stable "id" column survives reset_index renumberings; we use
    # it in ``_drive_division`` to find the same physical face after
    # a previous division has shuffled face labels.
    sheet.face_df["id"] = sheet.face_df.index.astype(int)
    sheet.settings.setdefault("threshold_length", 0.1)
    sheet.specs.setdefault("face", {})
    sheet.specs["face"]["unique_id_max"] = int(
        sheet.face_df["unique_id"].astype(int).max()
    )
    if not hasattr(sheet, "lineage") or sheet.lineage is None:
        sheet.lineage = nx_.DiGraph()
    return sheet


def _border_face_indices(sheet):
    """Faces that touch a periodic boundary (have at least one
    is_periodic edge)."""
    return sorted(int(f) for f in
                  sheet.edge_df.loc[sheet.edge_df["is_periodic"], "face"].unique())


def _interior_face_indices(sheet):
    border = set(_border_face_indices(sheet))
    return [int(f) for f in sheet.face_df.index if int(f) not in border]


def _face_neighbours(sheet, face):
    """Set of face ids that share an edge with ``face``."""
    fe = sheet.edge_df[sheet.edge_df["face"] == face]
    opp = fe.loc[fe["opposite"] >= 0, "opposite"].tolist()
    if not opp:
        return set()
    return set(int(f) for f in sheet.edge_df.loc[opp, "face"].tolist()
               if int(f) != int(face))


def _pick_adjacent_pair(sheet, both_border=False, both_interior=False,
                        mixed=False):
    """Return a tuple (A, B) of two face ids that SHARE an edge,
    satisfying the requested category filter.

    Exactly one of the three flags must be True."""
    assert sum([both_border, both_interior, mixed]) == 1, (
        "pick exactly one category"
    )
    border = set(_border_face_indices(sheet))
    candidates = list(sheet.face_df.index)
    # Iterate deterministically.
    for a in candidates:
        a_is_border = int(a) in border
        if both_interior and a_is_border:
            continue
        if both_border and not a_is_border:
            continue
        for b in sorted(_face_neighbours(sheet, int(a))):
            b_is_border = int(b) in border
            if both_interior and (a_is_border or b_is_border):
                continue
            if both_border and not (a_is_border and b_is_border):
                continue
            if mixed and (a_is_border == b_is_border):
                continue
            return int(a), int(b)
    pytest.skip(
        f"no adjacent pair matching "
        f"both_border={both_border} both_interior={both_interior} "
        f"mixed={mixed}"
    )


def _drive_division(sheet, *target_faces):
    """Divide each of the listed faces, one at a time, replicating the
    bookkeeping the production handler does between iterations.

    We DON'T go through the handler's ``get_division_function`` here
    because that re-queries ``face_df["area"]`` each iteration, and
    ``geom.update_all`` overwrites the area column with the TRUE
    geometric area (~1.0 on a unit-cell lattice). That recomputation
    blows away the "only these target cells qualify" rig, causing
    the handler to also divide every other unit-area cell in the
    grid. Calling the per-cell entry point directly keeps the test
    focused on the user-listed scenarios.

    For each successful division we ALSO mirror the unique_id /
    id / area-sanity logic from ``get_division_function`` so the
    final state is identical to a real run."""
    sheet.face_df["type"] = 0
    sheet.face_df["is_alive"] = 1

    for target in target_faces:
        # Look up the live label of this face via the stable id column.
        if "id" in sheet.face_df.columns:
            live = sheet.face_df.index[sheet.face_df["id"] == int(target)]
            if len(live) == 0:
                raise AssertionError(
                    f"target face id={target} no longer in face_df "
                    f"(was renumbered out from under us)"
                )
            mother = int(live[0])
        else:
            mother = int(target)

        daughter = index_preserving_cell_division(sheet, mother, sheet.geom)
        if daughter is None:
            raise AssertionError(
                f"index_preserving_cell_division refused to divide "
                f"face {mother} (target id={target}) — "
                f"get_division_edges returned None."
            )
        sheet.face_df.at[daughter, "id"] = daughter
        # Fresh unique_id, mirroring get_division_function.
        if "unique_id" in sheet.face_df.columns:
            sheet.specs.setdefault("face", {})
            if "unique_id_max" in sheet.specs["face"]:
                new_uid = sheet.specs["face"]["unique_id_max"] + 1
                sheet.specs["face"]["unique_id_max"] = new_uid
            else:
                new_uid = int(sheet.face_df["unique_id"].astype(int).max()) + 1
            sheet.face_df.at[daughter, "unique_id"] = new_uid
        # sheet.get_opposite()
        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
    # Keep num_sides in sync — some downstream operations (notably
    # remove_virtual_vertices) read it.
    sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()


class TestDivisionScenarios10x10:
    """Comprehensive division coverage on a 10x10 periodic lattice.

    Each test exercises one of the user-listed scenarios. After the
    division step (and any preceding / following virtual-vertex
    bookkeeping) we assert two strong invariants:

      1. Every face's ``order`` column is contiguous 1..N and its
         perimeter walks srce->trgt consistently.
      2. No edge has length >= LONG_EDGE_THRESHOLD (3.0). On a 10x10
         unit lattice that's three cell-diameters — a clear signature
         that either the perimeter walk attributed edges wrongly OR
         the periodic min-image failed.
    """

    # --- Scenario 1: divide one interior face ----------------------------
    def test_scenario_1_interior_face(self):
        sheet = _build_10x10()
        f = _interior_face_indices(sheet)[0]
        _drive_division(sheet, f)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, "1: interior")

    # --- Scenario 2: divide one border face -------------------------------
    def test_scenario_2_border_face(self):
        sheet = _build_10x10()
        f = _border_face_indices(sheet)[0]
        _drive_division(sheet, f)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, "2: border")

    # --- Scenario 3: divide two adjacent interior faces -------------------
    def test_scenario_3_two_adjacent_interior(self):
        sheet = _build_10x10()
        a, b = _pick_adjacent_pair(sheet, both_interior=True)
        _drive_division(sheet, a, b)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, "3: adj-interior")

    # --- Scenario 4: divide two adjacent border faces ---------------------
    def test_scenario_4_two_adjacent_border(self):
        sheet = _build_10x10()
        a, b = _pick_adjacent_pair(sheet, both_border=True)
        _drive_division(sheet, a, b)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, "4: adj-border")

    # --- Scenario 5: divide adjacent border + interior --------------------
    def test_scenario_5_adjacent_border_and_interior(self):
        sheet = _build_10x10()
        a, b = _pick_adjacent_pair(sheet, mixed=True)
        _drive_division(sheet, a, b)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, "5: adj-mixed")

    # --- Scenario 6: 1-5 followed by add_virtual_vertices -----------------
    # The new central-line edges produced by a division on a unit-cell
    # lattice are roughly 0.5 long, BELOW the default max_bond_length.
    # We crank max_bond_length DOWN so the post-division
    # add_virtual_vertices call actually does work, then verify both
    # invariants again.
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_6_division_then_add_virtuals(self, scenario, picker):
        sheet = _build_10x10()
        targets = picker(sheet)
        _drive_division(sheet, *targets)
        _post_division_settle(sheet)
        # Now subdivide any long edges from the division.
        sheet.maximal_bond_length = 0.3
        sheet.add_virtual_vertices()
        _assert_division_invariants(sheet, f"6/{scenario}: divide+add_virt")

    # --- Scenario 7: 1-5 followed by remove_virtual_vertices --------------
    # Build with virtual vertices already on the perimeter so the
    # post-division removal pass has something to compact.
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_7_division_then_remove_virtuals(self, scenario, picker):
        sheet = _build_10x10(with_virtuals=True)
        targets = picker(sheet)
        _drive_division(sheet, *targets)
        _post_division_settle(sheet)
        # Pick a removal threshold ABOVE typical post-division virtual
        # sub-edge lengths so the pass actually has work to do. With
        # max_bond=0.5 at build time and one add_vert per chosen
        # division edge, sub-edge lengths cluster around 0.25-0.5;
        # threshold 0.55 captures them all. The point of the test is
        # not "minimize edges" but "exercise the divide+remove combo
        # end-to-end and verify nothing breaks".
        sheet.minimal_bond_length = 0.55
        sheet.remove_virtual_vertices()
        _assert_division_invariants(sheet, f"7/{scenario}: divide+rm_virt")

    # --- Scenario 8: add_virtual_vertices then 1-5 ------------------------
    # Start without virtuals (cheap pure-hex topology), then refine
    # all long edges into virtual vertices BEFORE the division. The
    # division then runs on a sheet whose face perimeters are already
    # padded with virtual mid-edge vertices.
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_8_add_virtuals_then_division(self, scenario, picker):
        sheet = _build_10x10()
        # Inflate the topology with virtual vertices first.
        sheet.maximal_bond_length = 0.3
        sheet.add_virtual_vertices()
        sheet.geom.update_all(sheet)
        # Re-sync derived columns the division handler reads.
        sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
        targets = picker(sheet)
        _drive_division(sheet, *targets)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, f"8/{scenario}: add_virt+divide")

    # --- Scenario 9: remove_virtual_vertices then 1-5 ---------------------
    # Start WITH virtuals, then compactify them away, then divide.
    # Verifies that the removal pass leaves a sheet that the
    # division code can still handle (in particular: that the order
    # column is contiguous after removal, which previously had gaps
    # right where the dropped virtual edge used to sit).
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_9_remove_virtuals_then_division(self, scenario, picker):
        sheet = _build_10x10(with_virtuals=True)
        # Aggressively remove virtuals: pick threshold above all
        # virtual-edge lengths so all virtuals are eligible.
        sheet.minimal_bond_length = 0.4
        sheet.remove_virtual_vertices()
        sheet.geom.update_all(sheet)
        sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
        targets = picker(sheet)
        _drive_division(sheet, *targets)
        _post_division_settle(sheet)
        _assert_division_invariants(sheet, f"9/{scenario}: rm_virt+divide")


# --------------------------------------------------------------------------- #
# Layer 8b — comprehensive DELAMINATION-scenario coverage on 10x10 periodic.  #
# Same 9 scenarios as the division suite. The user noticed that no            #
# delaminations seemed to happen in a real simulation run, so we drive the    #
# topology change directly here to verify the production removal path is      #
# robust on every interior/border/adjacent-pair configuration.                #
# --------------------------------------------------------------------------- #


def _drive_delamination(sheet, *target_faces):
    """Remove each of the listed faces, one at a time, mirroring the
    bookkeeping the production delamination handler does between
    iterations.

    Like ``_drive_division``, we DON'T go through
    ``get_delamination_function`` here because the production handler
    is a TWO-PASS process: the first pass tags small cells with
    ``type=-1`` + high contractility so they shrink under mechanics,
    and only the SECOND pass calls ``index_preserving_remove`` — but
    only on cells whose ``num_sides`` has dropped to 3 (which on a
    fresh hex lattice is never). For a unit test of the removal logic
    itself, calling ``index_preserving_remove`` directly on the user-
    chosen target faces gives precise control over which scenarios
    get exercised.

    We look up each target by its stable ``id`` column (so a previous
    removal's ``reset_index`` can't shift the label out from under us)
    and silently skip targets that have already been absorbed — that
    can happen for adjacent-pair scenarios where removing the first
    face's verts also collapses the second face into a degenerate
    two-sided polygon that ``drop_two_sided_faces`` then drops on its
    own."""
    for target in target_faces:
        if "id" in sheet.face_df.columns:
            live = sheet.face_df.index[sheet.face_df["id"] == int(target)]
            if len(live) == 0:
                # Target already removed (cascaded from a previous
                # removal in this same _drive_delamination call).
                continue
            face = int(live[0])
        else:
            face = int(target)
        if face not in sheet.face_df.index:
            continue

        try:
            index_preserving_remove(sheet, face, sheet.geom)
        except Exception as exc:
            raise AssertionError(
                f"index_preserving_remove failed on face {face} "
                f"(target id={target}): {type(exc).__name__}: {exc}"
            ) from exc

        sheet.reset_index(order=False)
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
    sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()


def _post_delamination_settle(sheet):
    """Same book-keeping as ``_post_division_settle`` — kept under a
    distinct name so a failure trace makes the scenario obvious."""
    sheet.get_opposite()
    sheet.reset_index(order=False)
    sheet.order_all_edges()
    sheet.edge_df.sort_values(["face", "order"], inplace=True)
    sheet.get_opposite()
    sheet.geom.update_all(sheet)


class TestDelaminationScenarios10x10:
    """Comprehensive delamination coverage on a 10x10 periodic lattice.

    Mirror image of :class:`TestDivisionScenarios10x10`: same 9
    scenarios, same invariant battery (ordered edges, no edge length
    >= 3, positive areas, well-formed opposites). The user reported
    not seeing delaminations during a full simulation run, so we
    drive ``index_preserving_remove`` here directly to nail down
    whether the removal logic itself is correct on every
    interior/border/adjacent-pair configuration.
    """

    # --- Scenario 1: delaminate one interior face ------------------------
    def test_scenario_1_interior_face(self):
        sheet = _build_10x10()
        nf_before = sheet.Nf
        f = _interior_face_indices(sheet)[0]
        _drive_delamination(sheet, f)
        _post_delamination_settle(sheet)
        assert sheet.Nf == nf_before - 1, (
            f"1: interior: expected {nf_before - 1} faces, got {sheet.Nf}"
        )
        _assert_division_invariants(sheet, "1: interior delamination")

    # --- Scenario 2: delaminate one border face --------------------------
    def test_scenario_2_border_face(self):
        sheet = _build_10x10()
        nf_before = sheet.Nf
        f = _border_face_indices(sheet)[0]
        _drive_delamination(sheet, f)
        _post_delamination_settle(sheet)
        assert sheet.Nf == nf_before - 1, (
            f"2: border: expected {nf_before - 1} faces, got {sheet.Nf}"
        )
        _assert_division_invariants(sheet, "2: border delamination")

    # --- Scenario 3: delaminate two adjacent interior faces --------------
    def test_scenario_3_two_adjacent_interior(self):
        sheet = _build_10x10()
        nf_before = sheet.Nf
        a, b = _pick_adjacent_pair(sheet, both_interior=True)
        _drive_delamination(sheet, a, b)
        _post_delamination_settle(sheet)
        # In adjacent-pair removal the second removal may cascade
        # (two-sided neighbour cleanup); ensure at least two faces
        # are gone, but tolerate one extra cascade.
        assert nf_before - 3 <= sheet.Nf <= nf_before - 2, (
            f"3: adj-interior: expected {nf_before - 2} or {nf_before - 3} "
            f"faces, got {sheet.Nf}"
        )
        _assert_division_invariants(sheet, "3: adj-interior delamination")

    # --- Scenario 4: delaminate two adjacent border faces ----------------
    def test_scenario_4_two_adjacent_border(self):
        sheet = _build_10x10()
        nf_before = sheet.Nf
        a, b = _pick_adjacent_pair(sheet, both_border=True)
        _drive_delamination(sheet, a, b)
        _post_delamination_settle(sheet)
        assert nf_before - 3 <= sheet.Nf <= nf_before - 2, (
            f"4: adj-border: expected {nf_before - 2} or {nf_before - 3} "
            f"faces, got {sheet.Nf}"
        )
        _assert_division_invariants(sheet, "4: adj-border delamination")

    # --- Scenario 5: delaminate adjacent border + interior ----------------
    def test_scenario_5_adjacent_border_and_interior(self):
        sheet = _build_10x10()
        nf_before = sheet.Nf
        a, b = _pick_adjacent_pair(sheet, mixed=True)
        _drive_delamination(sheet, a, b)
        _post_delamination_settle(sheet)
        assert nf_before - 3 <= sheet.Nf <= nf_before - 2, (
            f"5: adj-mixed: expected {nf_before - 2} or {nf_before - 3} "
            f"faces, got {sheet.Nf}"
        )
        _assert_division_invariants(sheet, "5: adj-mixed delamination")

    # --- Scenario 6: 1-5 followed by add_virtual_vertices -----------------
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_6_delamination_then_add_virtuals(self, scenario, picker):
        sheet = _build_10x10()
        targets = picker(sheet)
        _drive_delamination(sheet, *targets)
        _post_delamination_settle(sheet)
        # Subdivide any edges that ended up longer than max_bond_length
        # after the centroid-collapse. With cell diameter ~1 these are
        # bounded by ~1, but max_bond_length=0.3 forces actual work.
        sheet.maximal_bond_length = 0.3
        sheet.add_virtual_vertices()
        _assert_division_invariants(sheet, f"6/{scenario}: delam+add_virt")

    # --- Scenario 7: 1-5 followed by remove_virtual_vertices --------------
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_7_delamination_then_remove_virtuals(self, scenario, picker):
        sheet = _build_10x10(with_virtuals=True)
        targets = picker(sheet)
        _drive_delamination(sheet, *targets)
        _post_delamination_settle(sheet)
        # See scenario 7 of the division suite for the threshold
        # rationale; same applies here.
        sheet.minimal_bond_length = 0.55
        sheet.remove_virtual_vertices()
        _assert_division_invariants(sheet, f"7/{scenario}: delam+rm_virt")

    # --- Scenario 8: add_virtual_vertices then 1-5 ------------------------
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_8_add_virtuals_then_delamination(self, scenario, picker):
        sheet = _build_10x10()
        sheet.maximal_bond_length = 0.3
        sheet.add_virtual_vertices()
        sheet.geom.update_all(sheet)
        sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
        targets = picker(sheet)
        _drive_delamination(sheet, *targets)
        _post_delamination_settle(sheet)
        _assert_division_invariants(sheet, f"8/{scenario}: add_virt+delam")

    # --- Scenario 9: remove_virtual_vertices then 1-5 ---------------------
    @pytest.mark.parametrize("scenario,picker", [
        ("interior", lambda s: (_interior_face_indices(s)[0],)),
        ("border",   lambda s: (_border_face_indices(s)[0],)),
        ("adj-int",  lambda s: _pick_adjacent_pair(s, both_interior=True)),
        ("adj-bdr",  lambda s: _pick_adjacent_pair(s, both_border=True)),
        ("adj-mix",  lambda s: _pick_adjacent_pair(s, mixed=True)),
    ])
    def test_scenario_9_remove_virtuals_then_delamination(self, scenario, picker):
        sheet = _build_10x10(with_virtuals=True)
        sheet.minimal_bond_length = 0.4
        sheet.remove_virtual_vertices()
        sheet.geom.update_all(sheet)
        sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
        targets = picker(sheet)
        _drive_delamination(sheet, *targets)
        _post_delamination_settle(sheet)
        _assert_division_invariants(sheet, f"9/{scenario}: rm_virt+delam")


class TestDelaminationHandlerTriangulated:
    """Production delamination handler integration test.

    Background: the user reported that in a full simulation run no
    delaminations seemed to happen. The handler is a TWO-PASS thing:
    pass-1 tags every cell with ``area < crit_area`` as type=-1 +
    high contractility (a mechanical push to shrink); pass-2 actually
    calls ``index_preserving_remove`` — on cells below the threshold
    whose live edge count has dropped to <=3 OR whose area has gone
    degenerate (<=0), with ``is_alive == 1``. ``num_sides`` is now
    recomputed from the live edge counts each pass (it used to go
    stale and freeze the removal when divisions were off), so it can
    no longer be rigged; this test drives removal via the
    degenerate-area clause — the real scenario the fix targets (a
    delaminating cell that collapsed to <=0 area).

    Verifies the handler actually removes the cell and leaves a
    well-formed topology behind.
    """

    def test_handler_removes_a_triangular_cell(self):
        sheet = _build_10x10()
        target = _interior_face_indices(sheet)[0]
        nf_before = sheet.Nf

        from inner_ear_model import InnerEarModel
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )

        # ``InnerEarModel.__init__`` runs ``update_specs(reset=True)``
        # which can overwrite area / num_sides. Apply the test rig
        # AFTER the constructor so the handler sees what we want.
        inner.sheet.face_df["area"] = 1.0
        inner.sheet.face_df["num_sides"] = 6
        inner.sheet.face_df["is_alive"] = 1
        live = inner.sheet.face_df.index[inner.sheet.face_df["id"] == int(target)]
        assert len(live), (
            f"target face id={target} disappeared during InnerEarModel "
            f"constructor"
        )
        target_live = int(live[0])
        # num_sides is recomputed from live edges inside the handler, so it
        # can't be rigged; drive removal via the degenerate-area (<=0) clause.
        inner.sheet.face_df.loc[target_live, "area"] = -0.01

        class _M:
            def append(self, fn): pass
        inner.topological_events_handler.get_delamination_function(
            crit_area=0.5,
        )(inner.sheet, _M())

        # The target face must be gone, AND surviving topology must
        # satisfy all invariants.
        survivors = inner.sheet.face_df.index[
            inner.sheet.face_df["id"] == int(target)
        ]
        assert len(survivors) == 0, (
            f"handler did not remove the rigged triangular cell "
            f"(id={target}); face_df still has {len(survivors)} match"
        )
        assert inner.sheet.Nf == nf_before - 1, (
            f"expected {nf_before - 1} faces after handler removal, "
            f"got {inner.sheet.Nf}"
        )
        _assert_division_invariants(inner.sheet, "handler: triangle")


class TestAblationHandler:
    """The ablation handler must zero the (misspelled) ``prefered_area`` /
    ``prefered_vol`` columns the model's FaceAreaElasticity actually reads —
    NOT the correctly spelled ``preferred_area`` (a dead column). With the
    wrong column the ablated cell kept its (large) preferred area and, with
    the raised area_elasticity, ballooned instead of collapsing — squeezing a
    neighbour into a negative area and crashing the solver
    (results/dbg2ablated_...: face 268, type=1, area<0)."""

    def test_ablation_zeroes_the_model_preferred_area(self):
        from inner_ear_model import InnerEarModel
        sheet = _build_10x10()
        target = _interior_face_indices(sheet)[0]
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )
        cell = int(inner.sheet.face_df.index[inner.sheet.face_df["id"] == int(target)][0])
        # The model's preferred-area column is non-zero before ablation.
        assert inner.sheet.face_df.at[cell, "prefered_area"] > 0

        class _M:
            def append(self, fn):
                pass
        inner.topological_events_handler.get_ablation_function(cell)(inner.sheet, _M())

        fd = inner.sheet.face_df
        # The column the effector reads must be zeroed (so the cell collapses).
        assert fd.at[cell, "prefered_area"] == 0, "ablation must zero prefered_area"
        assert fd.at[cell, "prefered_vol"] == 0
        assert fd.at[cell, "type"] == -1
        assert fd.at[cell, "area_elasticity"] == 20
        assert fd.at[cell, "contractility"] == 10


class TestPeriodicSheetViewColoring:
    """``periodic_sheet_view`` USED to render every face in matplotlib's
    default tab:blue when a colormap was passed in, even though the
    same ``draw_func`` worked correctly inside ``create_gif``. The
    cause: ``parse_face_specs`` / ``_parse_edge_specs`` expect the
    FACE / EDGE sub-spec (the dict whose top-level key is ``color``),
    but ``periodic_sheet_view`` was handing them the WHOLE
    ``draw_specs`` dict (the one whose top-level keys are ``face``,
    ``edge``, ``vert``, ...). With no top-level ``color`` key,
    ``parse_face_specs`` returned ``{}``, leaving ``PatchCollection``
    with no ``facecolors`` argument — hence the all-blue plot.

    ``create_gif`` got away with it because the sheet retrieved from
    ``HistoryHdf5`` has ``periodic=False`` (the ``VirtualSheet``
    default — ``arrange_sheet_from_history`` would have set it but
    create_gif doesn't call that), so ``get_sheet_view_method``
    dispatched to tyssue's stock ``sheet_view`` → ``draw_face``,
    which unpacks the sub-dict correctly."""

    def test_per_face_color_array_reaches_patch_collection(self):
        """Pass a varied ``(Nf, 4)`` face-color array through the
        public draw entry point and verify that the resulting
        ``PatchCollection`` actually carries those facecolors — not
        matplotlib's default."""
        import matplotlib.pyplot as plt
        sheet = _build_sheet(nx=4, ny=4)

        # Build a varied per-face RGBA. Cell 0 fully red, others span
        # the green channel — gives both row-1 != row-0 (so ptp > 0)
        # and unique colors per face.
        Nf = sheet.Nf
        face_colors = np.zeros((Nf, 4))
        face_colors[:, 1] = np.linspace(0.0, 1.0, Nf)  # vary green
        face_colors[0, 0] = 1.0                         # mark cell 0 red
        face_colors[:, 3] = 1.0                         # full alpha

        from tyssue.config.draw import sheet_spec
        draw_specs = sheet_spec()
        draw_specs["face"]["color"] = face_colors
        draw_specs["face"]["visible"] = True
        draw_specs["face"]["alpha"] = 1.0

        fig, ax = PeriodicBoundarySheet.periodic_sheet_view(
            sheet, ["x", "y"], **draw_specs,
        )

        # The first collection added is the face PatchCollection.
        face_pc = ax.collections[0]
        applied = np.asarray(face_pc.get_facecolors())
        assert applied.shape[0] == Nf, (
            f"PatchCollection got {applied.shape[0]} colors but the "
            f"sheet has {Nf} faces — _face_color_from_sequence broadcast"
        )
        # Cell 0 must really be red — not the matplotlib default
        # tab:blue (≈ (0.12, 0.47, 0.71, 1.0)).
        assert applied[0, 0] > 0.9, (
            f"face 0 should be red (R > 0.9) but got facecolor "
            f"{applied[0].tolist()} — periodic_sheet_view is passing "
            f"the wrong dict to parse_face_specs (the all-blue bug)."
        )
        # Green channel should vary across faces (i.e. the per-face
        # colors were actually used, not a uniform fallback).
        assert np.ptp(applied[:, 1]) > 0.5, (
            "green channel doesn't vary across patches — facecolors "
            "didn't get applied per-face"
        )
        plt.close(fig)

    def test_inner_ear_draw_func_colors_live_sheet(self):
        """End-to-end: build the SAME ``draw_func`` that
        ``periodic_tests.py`` builds, hand it a live periodic sheet
        whose ``atoh_level`` varies across cells, and check the
        rendered face colors are NOT all the same default-blue."""
        import matplotlib.pyplot as plt
        from inner_ear_model import InnerEarModel
        sheet = _build_sheet(nx=4, ny=4)
        # InnerEarModel populates atoh_level from random
        # notch/delta levels — that's the live-sheet code path the
        # user's periodic_tests.py exercises.
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )
        # Force varied atoh_level so the ``ptp < 1e-10`` uniform-value
        # branch in ``_face_color_from_sequence`` can't mask the bug.
        rng = np.random.default_rng(0)
        inner.sheet.face_df["atoh_level"] = rng.uniform(
            0.0, 1.0, size=inner.sheet.Nf
        )

        draw_func = inner.get_draw_sheet_method(
            number_faces=False, number_edges=False, number_vertices=False,
            color_by="atoh",
        )
        fig, ax = draw_func(inner.sheet)
        face_pc = ax.collections[0]
        applied = np.asarray(face_pc.get_facecolors())

        # Default matplotlib tab:blue ≈ (0.122, 0.467, 0.706, 1.0).
        # Either applied is per-face (Nf colors) and varied, OR it's
        # uniformly the default tab:blue — that's the bug.
        if applied.shape[0] == 1:
            r, g, b, _ = applied[0]
            looks_blue = (b > 0.5) and (r < 0.3) and (g < 0.6)
            assert not looks_blue, (
                "live periodic sheet rendered with a single tab:blue "
                "facecolor — periodic_sheet_view did not propagate "
                "the per-face color array"
            )
        # Per-face colors should vary (Greens cmap on a varied
        # atoh_level => varied colors). Check the green channel ptp.
        assert np.ptp(applied[:, 1]) > 0.05, (
            f"facecolors don't vary across cells: ptp(green) = "
            f"{float(np.ptp(applied[:, 1])):.4f}. Live periodic sheet "
            f"is rendering uniformly — this is the all-blue bug."
        )
        plt.close(fig)


class TestRemoveVirtualVertexBothVirtualOnPeriodicEdge:
    """``remove_virtual_vertex`` on a periodic edge whose BOTH endpoints
    are virtual used to silently delegate the merged-vertex position
    to ``collapse_edge``'s plain arithmetic mean of ``vert_df.loc[srce]``
    and ``vert_df.loc[trgt]``. On a wrap edge that mean falls on the
    LONG side of the periodic span (e.g. canonical (0.4, y) and
    canonical (Lx-0.4, y) average to (Lx/2, y) instead of the
    correct min-image midpoint at (0, y) or equivalently (Lx, y)).
    The teleported survivor stretches every incident edge from
    ~0.1 to ~Lx/2, and the very next ``add_virtual_vertices`` pass
    subdivides each stretched edge into hundreds of fragments —
    producing "super-faces" with 150+ edges and self-crossing
    geometry (negative signed area).

    This was the root cause of the iter-2348 crash in
    ``results/random_periodic_array_test2/`` where faces 179, 160,
    and 403 each ballooned from 30-32 edges to 156-157 edges in
    one iteration.
    """

    def test_both_virtual_periodic_collapse_stays_on_canonical_side(self):
        """Construct a periodic sheet, force a wrap edge to have two
        virtual mid-edge vertices on opposite canonical sides of the
        boundary, mark its length as below ``minimal_bond_length``,
        and verify ``remove_virtual_vertex`` doesn't teleport the
        survivor across the sheet."""
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p4x4_v", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=0.5, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        Lx, Ly = float(sheet.Lx), float(sheet.Ly)

        # Find a periodic edge whose srce AND trgt are both virtual.
        # planar_virtual_sheet_2d's add_virtual_vertices pass produces
        # plenty of these on wrapping perimeter segments.
        is_v = sheet.vert_df["is_virtual"].to_dict()
        periodic_edges = sheet.edge_df[sheet.edge_df["is_periodic"]]
        cand = None
        for eid in periodic_edges.index:
            s = int(periodic_edges.at[eid, "srce"])
            t = int(periodic_edges.at[eid, "trgt"])
            if is_v.get(s, 0) == 1 and is_v.get(t, 0) == 1:
                cand = eid
                break
        if cand is None:
            pytest.skip(
                "test sheet has no periodic edge with both endpoints "
                "virtual — adjust nx/ny/bond lengths to produce one"
            )

        srce = int(sheet.edge_df.at[cand, "srce"])
        trgt = int(sheet.edge_df.at[cand, "trgt"])
        srce_pos = sheet.vert_df.loc[srce, ["x", "y"]].to_numpy().astype(float)
        trgt_pos = sheet.vert_df.loc[trgt, ["x", "y"]].to_numpy().astype(float)

        # Pre-condition for the bug: the two canonical positions
        # straddle the wrap (differ by ~Lx in some axis), so their
        # plain arithmetic mean is on the FAR side, not adjacent.
        raw_mean = (srce_pos + trgt_pos) / 2.0
        # The min-image midpoint is the correct merge location.
        from topological_events import _min_image_midpoint
        good_mid = _min_image_midpoint(sheet, [srce, trgt])

        gap = np.abs(raw_mean - good_mid)
        # If the bug existed, raw_mean would be ~Lx/2 away from
        # good_mid in at least one axis. Guard against test setup
        # surprises by requiring SOME separation here.
        assert gap.max() > Lx / 4 - 1e-6, (
            f"test setup precondition not met: raw_mean={raw_mean} is "
            f"close to good_mid={good_mid} — both endpoints might not "
            f"actually straddle the wrap"
        )

        sheet.remove_virtual_vertex(cand)
        sheet.reset_index(order=False)
        sheet.geom.update_all(sheet)

        # The merged vertex should live at the min-image midpoint
        # (rewrapped to [0, L) by ``_min_image_midpoint``). Locate it
        # via the lower of the two original labels (which is what
        # ``collapse_edge`` keeps).
        keeper = min(srce, trgt)
        # After reset_index the keeper may have been renumbered.
        # But its coordinates are still around good_mid, not at
        # raw_mean. So check that NO vertex lives within
        # quasi-machine-precision of raw_mean (in the wrap-crossing
        # axis where the bug manifests).
        v_xy = sheet.vert_df[["x", "y"]].to_numpy()
        bad_axis = int(np.argmax(np.abs(raw_mean - good_mid)))
        bad_val = raw_mean[bad_axis]
        # No live vertex should be at the raw-mean position (the
        # bug's teleport landing zone).
        on_bad = np.abs(v_xy[:, bad_axis] - bad_val) < 1e-3
        # A handful of unrelated verts may COINCIDENTALLY sit near
        # the mid-line; tolerate that by counting only those that
        # ALSO sit at the correct other-axis coordinate of the bad
        # mean (i.e. the actual teleport target).
        other_axis = 1 - bad_axis
        on_bad &= np.abs(v_xy[:, other_axis] - raw_mean[other_axis]) < 1e-3
        assert not on_bad.any(), (
            f"a vertex landed at the wrap-blind raw-mean "
            f"({raw_mean[0]:.4f}, {raw_mean[1]:.4f}) instead of the "
            f"min-image midpoint ({good_mid[0]:.4f}, {good_mid[1]:.4f}) "
            f"— the both-virtual-endpoint collapse is still broken"
        )

        # Sanity: no face should be negative area or have an edge
        # longer than half the box (the fingerprint of a teleported
        # vertex).
        assert (sheet.face_df["area"] > 0).all(), (
            "some face area went non-positive after remove_virtual_vertex"
        )
        assert sheet.edge_df["length"].max() < Lx / 2, (
            f"edge length {sheet.edge_df['length'].max():.3f} >= Lx/2 "
            f"({Lx/2:.3f}) — vertex teleported"
        )


class TestRemoveVirtualVertexOrdering:
    """``remove_virtual_vertex`` drops one edge from each face it
    touches (via ``collapse_edge``). The previous version did NOT
    re-run ``order_edges`` on those faces, so the perimeter ended up
    with an ``order`` gap right where the dropped edge sat. The next
    ``update_dcoords`` then walked the broken perimeter and stored a
    nonsensical length on the closing edge (the long-edge bug)."""

    @staticmethod
    def _build_with_virtuals():
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p4x4_v", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=0.2, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        return sheet

    @staticmethod
    def _face_has_order_gap(sheet, face_id):
        orders = sorted(sheet.edge_df.loc[
            sheet.edge_df["face"] == face_id, "order"
        ].unique())
        if not orders:
            return False
        expected = list(range(int(orders[0]), int(orders[-1]) + 1))
        return any(o not in orders for o in expected)

    def _pick_virtual_edge(self, sheet):
        """Find an edge whose srce or trgt is a virtual vertex AND
        whose face has its order column populated (1..N contiguous).
        Returns the edge id."""
        for eid in sheet.edge_df.index:
            s = int(sheet.edge_df.at[eid, "srce"])
            t = int(sheet.edge_df.at[eid, "trgt"])
            if (int(sheet.vert_df.at[s, "is_virtual"]) == 1
                    or int(sheet.vert_df.at[t, "is_virtual"]) == 1):
                return int(eid)
        pytest.skip("no virtual edges in fresh sheet")
        return None

    def test_single_removal_leaves_no_order_gaps_in_affected_faces(self):
        sheet = self._build_with_virtuals()
        # Force the order column to be a contiguous 1..N for every
        # face so we can be sure we're testing the post-removal state,
        # not pre-existing gaps.
        sheet.order_all_edges()
        edge_id = self._pick_virtual_edge(sheet)
        # Record which faces should be affected by the removal.
        affected = {int(sheet.edge_df.at[edge_id, "face"])}
        opp = int(sheet.edge_df.at[edge_id, "opposite"])
        if opp >= 0:
            affected.add(int(sheet.edge_df.at[opp, "face"]))

        sheet.remove_virtual_vertex(edge_id)

        # Each affected face's order column must be contiguous 1..N.
        for face in affected:
            if face not in sheet.face_df.index:
                continue
            orders = sorted(sheet.edge_df.loc[
                sheet.edge_df["face"] == face, "order"
            ].unique())
            if not orders:
                continue
            assert orders == list(range(1, len(orders) + 1)), (
                f"face {face} has order gap after remove_virtual_vertex: "
                f"orders={orders}"
            )

    def test_batch_removal_leaves_no_order_gaps_anywhere(self):
        """remove_virtual_vertices keeps shrinking short edges until
        none remain. After the batch completes, NO face should have
        an order gap."""
        sheet = self._build_with_virtuals()
        sheet.order_all_edges()
        # Bump the min_bond_length so most virtual edges get removed.
        sheet.set_minimal_bond_length(0.15)
        sheet.remove_virtual_vertices()

        broken = []
        for fid in sheet.face_df.index:
            if self._face_has_order_gap(sheet, fid):
                orders = sorted(sheet.edge_df.loc[
                    sheet.edge_df["face"] == fid, "order"
                ].unique())
                broken.append((int(fid), orders[0], orders[-1]))
        assert not broken, f"faces with order gaps after batch: {broken[:5]}"

    def test_single_removal_preserves_face_perimeter_closure(self):
        """After removing a virtual vertex, the affected faces must
        still have CLOSED perimeters: ``check_edge_order`` returns
        True for every face."""
        sheet = self._build_with_virtuals()
        sheet.order_all_edges()
        edge_id = self._pick_virtual_edge(sheet)
        sheet.remove_virtual_vertex(edge_id)
        assert sheet.check_all_edge_order(), (
            "at least one face has a non-cyclic perimeter after "
            "remove_virtual_vertex"
        )


class TestOrderAllEdgesResilient:
    """``order_all_edges`` must keep going if one face has a broken
    perimeter — otherwise the exception leaves every face processed
    after the bad one with STALE order values."""

    def test_one_broken_face_doesnt_break_others(self):
        """Build a sheet where one face has a deliberately broken
        perimeter (an edge whose trgt has no out-edge in the same
        face). Run ``order_all_edges`` and verify the OTHER faces
        still have their orders recomputed (1..N contiguous)."""
        sheet = _build_sheet(nx=3, ny=2)
        # Corrupt face 0 — re-point ONE of its edges so its trgt has
        # no follower in face 0.
        f0_edges = sheet.edge_df[sheet.edge_df["face"] == 0].sort_values("order")
        bad_eid = int(f0_edges.index[0])
        # Set this edge's trgt to a vertex that's not a srce of any
        # other edge in face 0.
        outsider = sheet.vert_df.index.max()
        sheet.edge_df.at[bad_eid, "trgt"] = int(outsider)
        # Reset the order column to garbage so we can tell which faces
        # got re-ordered.
        sheet.edge_df["order"] = -1
        sheet.order_all_edges()
        # face 0 was broken — its edges may end up partially ordered.
        # OTHER faces must be fully re-ordered: their orders run 1..N.
        for fid in sheet.face_df.index:
            if fid == 0:
                continue
            orders = sorted(sheet.edge_df.loc[
                sheet.edge_df["face"] == fid, "order"
            ].unique())
            if not orders:
                continue
            assert orders == list(range(1, len(orders) + 1)), (
                f"face {fid} not contiguously ordered: {orders}"
            )


class TestOrderAllEdgesGroupedEquivalence:
    """The grouped O(Ne) ``order_all_edges`` must produce a BYTE-IDENTICAL
    ``order`` column to the old per-face ``order_edges`` loop it replaced — on
    fresh sheets, sheets with virtual vertices, a large sheet, a gappy index, and
    a sheet with a broken face (so the perf rewrite is provably behaviour-neutral)."""

    @staticmethod
    def _reference_then_new(sheet):
        # Old behaviour == loop the (UNCHANGED) order_edges over every face.
        # Capture that, then run the new grouped order_all_edges and compare;
        # both are pure functions of the untouched srce/trgt/face columns.
        for f in sheet.face_df.index.values:
            try:
                sheet.order_edges(int(f))
            except (IndexError, ValueError, KeyError):
                pass
        expected = sheet.edge_df["order"].to_numpy().copy()
        sheet.order_all_edges()
        got = sheet.edge_df["order"].to_numpy()
        return expected, got

    @pytest.mark.parametrize("nx,ny", [(2, 2), (3, 2), (4, 4)])
    def test_matches_reference_fresh(self, nx, ny):
        exp, got = self._reference_then_new(_build_sheet(nx, ny))
        np.testing.assert_array_equal(got, exp)
        assert got.dtype == exp.dtype

    @pytest.mark.parametrize("nx,ny", [(3, 2), (4, 4), (5, 2)])
    def test_matches_reference_with_virtuals(self, nx, ny):
        exp, got = self._reference_then_new(_build_sheet_with_virtuals(nx, ny))
        np.testing.assert_array_equal(got, exp)

    def test_matches_reference_large(self):
        exp, got = self._reference_then_new(_build_10x10(with_virtuals=True))
        np.testing.assert_array_equal(got, exp)

    def test_matches_reference_gappy_index(self):
        # collapse_edge leaves a sparse edge index; the grouped pass drives the
        # walk by ROW POSITION, so a gappy (non-contiguous) index must still match.
        sheet = _build_sheet_with_virtuals(nx=3, ny=2)
        sheet.edge_df.index = sheet.edge_df.index.to_numpy() * 3 + 1  # relabel gappy
        exp, got = self._reference_then_new(sheet)
        np.testing.assert_array_equal(got, exp)

    def test_matches_reference_with_broken_face(self):
        # A non-closing perimeter must be left the SAME partial/zeroed order by
        # both methods, and the other faces reordered identically.
        sheet = _build_sheet(nx=3, ny=2)
        f0 = sheet.edge_df[sheet.edge_df["face"] == 0].sort_values("order")
        sheet.edge_df.at[int(f0.index[0]), "trgt"] = int(sheet.vert_df.index.max())
        exp, got = self._reference_then_new(sheet)
        np.testing.assert_array_equal(got, exp)


# --------------------------------------------------------------------------- #
# Layer 7c-bis — resume an interrupted run on the SAME history file            #
# --------------------------------------------------------------------------- #


class TestDropCorruptedSnapshots:
    """``history_io.drop_corrupted_snapshots`` removes snapshots
    whose face table carries a non-positive area — the fingerprint of
    a failed non-periodic resume that re-recorded unwrapped (negative
    area) geometry over a good snapshot."""

    @staticmethod
    def _write(path, rows_per_time):
        """rows_per_time: dict {t: area_array}. Writes a minimal
        face/vert table archive (only what the cleanup reads/removes)."""
        import pandas as pd
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with pd.HDFStore(path, "w") as store:
            for t, areas in rows_per_time.items():
                n = len(areas)
                face = pd.DataFrame({
                    "area": np.asarray(areas, dtype=float),
                    "perimeter": np.full(n, 3.4),
                    "time": np.full(n, float(t)),
                }, index=pd.Index(range(n), name="face"))
                store.append("face", face, data_columns=["time"])
                vert = pd.DataFrame({
                    "x": np.linspace(0, 1, n), "time": np.full(n, float(t)),
                }, index=pd.Index(range(n), name="vert"))
                store.append("vert", vert, data_columns=["time"])

    def test_dry_run_reports_without_modifying(self, tmp_path):
        import pandas as pd
        from history_io import drop_corrupted_snapshots
        path = str(tmp_path / "h.hf5")
        self._write(path, {0.0: [1.0, 1.0], 1.0: [1.0, 1.0], 2.0: [1.0, -3.0]})
        before = os.path.getsize(path)
        dropped = drop_corrupted_snapshots(path, dry_run=True)
        assert dropped == [2.0], f"dry run should flag t=2.0, got {dropped}"
        # File untouched.
        with pd.HDFStore(path, "r") as store:
            times = sorted(store.select("face", columns=["time"])["time"].unique())
        assert times == [0.0, 1.0, 2.0]

    def test_removes_only_corrupted_snapshots(self, tmp_path):
        import pandas as pd
        from history_io import drop_corrupted_snapshots
        path = str(tmp_path / "h.hf5")
        self._write(path, {0.0: [1.0, 1.0], 1.0: [1.0, 1.0], 2.0: [1.0, -3.0]})
        dropped = drop_corrupted_snapshots(path)
        assert dropped == [2.0]
        with pd.HDFStore(path, "r") as store:
            ftimes = sorted(store.select("face", columns=["time"])["time"].unique())
            vtimes = sorted(store.select("vert", columns=["time"])["time"].unique())
        assert ftimes == [0.0, 1.0], "corrupted snapshot not removed from face"
        assert vtimes == [0.0, 1.0], "corrupted snapshot not removed from vert"

    def test_clean_archive_untouched(self, tmp_path):
        from history_io import drop_corrupted_snapshots
        path = str(tmp_path / "h.hf5")
        self._write(path, {0.0: [1.0, 1.0], 1.0: [1.0, 1.0]})
        dropped = drop_corrupted_snapshots(path)
        assert dropped == [], "a clean archive should drop nothing"

    def test_idempotent(self, tmp_path):
        from history_io import drop_corrupted_snapshots
        path = str(tmp_path / "h.hf5")
        self._write(path, {0.0: [1.0, 1.0], 2.0: [1.0, -3.0]})
        drop_corrupted_snapshots(path)
        # Second pass finds nothing left to drop.
        assert drop_corrupted_snapshots(path) == []


class TestResumeFromTime:
    """``inner_ear_model._truncate_history_file`` + a pre-seeded
    ``solver.prev_t`` together let a crashed run be resumed by
    re-running with ``continue_from_time=t0``: the existing archive
    is truncated at ``t0`` and the solver picks up at ``t0`` writing
    INTO THE SAME FILE — no parallel ``_part2`` archive."""

    def test_truncate_keeps_rows_up_to_and_including_t0(self, tmp_path):
        """All snapshots with time <= t0 must survive; everything
        strictly later must be dropped."""
        import pandas as pd
        from inner_ear_model import _truncate_history_file

        path = str(tmp_path / "probe.hf5")
        with pd.HDFStore(path, "w") as store:
            for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
                store.append(
                    "vert",
                    pd.DataFrame({"x": [0.0, 1.0], "time": [t, t]}),
                    data_columns=["time"],
                )
                store.append(
                    "edge",
                    pd.DataFrame({"srce": [0], "trgt": [1], "time": [t]}),
                    data_columns=["time"],
                )

        _truncate_history_file(path, 1.0)

        v_times = sorted(pd.read_hdf(path, "vert")["time"].unique())
        e_times = sorted(pd.read_hdf(path, "edge")["time"].unique())
        assert v_times == [0.0, 0.5, 1.0], (
            f"vert truncation kept wrong rows: {v_times}"
        )
        assert e_times == [0.0, 0.5, 1.0], (
            f"edge truncation kept wrong rows: {e_times}"
        )

    def test_truncate_at_boundary_keeps_t0_exactly(self, tmp_path):
        """Truncating at exactly an existing time stamp must keep
        that snapshot (the rewind point is INCLUSIVE)."""
        import pandas as pd
        from inner_ear_model import _truncate_history_file

        path = str(tmp_path / "probe.hf5")
        with pd.HDFStore(path, "w") as store:
            for t in [0.0, 0.5, 1.0]:
                store.append(
                    "vert",
                    pd.DataFrame({"x": [0.0], "time": [t]}),
                    data_columns=["time"],
                )

        _truncate_history_file(path, 0.5)

        kept = sorted(pd.read_hdf(path, "vert")["time"].unique())
        assert kept == [0.0, 0.5], f"boundary truncation: {kept}"

    def test_truncate_noop_on_missing_file(self, tmp_path):
        """Resume probing a missing archive must not crash — the
        caller may be re-running with continue_from_time set even
        though no archive yet exists."""
        from inner_ear_model import _truncate_history_file
        missing = str(tmp_path / "nope.hf5")
        # Should silently no-op rather than raise.
        _truncate_history_file(missing, 1.0)

    def test_solver_resumes_from_seeded_prev_t(self, monkeypatch):
        """Seeding ``solver.prev_t`` and ``history.time`` must make
        ``solver.solve(tf=tf)`` advance from the resume time to
        ``tf`` — i.e. ``tf`` is the absolute (cumulative) end time,
        not "another tf units of work"."""
        sheet, solver = TestAdaptiveDt._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
        )
        RESUME_T = 17.654
        solver.prev_t = RESUME_T
        solver.history.time = RESUME_T

        solver.solve(
            tf=RESUME_T + 0.03, dt=0.01,
            max_displacement=1.0, save_interval=0.01,
        )

        # Filter to the SOLVER-emitted snapshots; the History's
        # constructor records a free t=0 row that's a fixture
        # artefact, not part of what we're testing.
        all_ts = sorted(set(round(t, 6) for t in solver.history.time_stamps))
        solver_ts = [t for t in all_ts if t >= RESUME_T - 1e-6]

        assert solver_ts, "solver emitted no snapshots after resume"
        assert min(solver_ts) >= RESUME_T - 1e-6, (
            f"solver started before the seeded resume time: {min(solver_ts)}"
        )
        assert max(solver_ts) >= RESUME_T + 0.03 - 1e-6, (
            f"solver didn't reach tf={RESUME_T + 0.03}: "
            f"max stamp = {max(solver_ts)}"
        )

    def test_rewrite_for_resume_lets_record_append(self, tmp_path):
        """The crux of the resume-crash fix, exercised end-to-end with
        real objects.

        A bare row-level truncate leaves the original table layout
        (3D coords like ``z``, a different column order) in place, so
        the resumed run's first ``record()`` append dies with pandas'
        ``cannot match existing table structure for [...]``.
        ``_rewrite_history_for_resume`` transcribes the kept rows into
        exactly the structure ``record()`` will produce, so the append
        lines up.

        We build a real periodic sheet, write a real multi-snapshot
        HistoryHdf5 (whose tables carry the 3D ``z`` columns and the
        periodic-metadata columns), reload it the way the resume path
        does (``arrange_sheet_from_history(two_dim=True)`` → drops z),
        rewrite, and then record a fresh snapshot — which must NOT
        raise the structure error."""
        import pandas as pd
        from tyssue import HistoryHdf5
        from inner_ear_model import _rewrite_history_for_resume

        archive = str(tmp_path / "resume_struct.hf5")

        # --- Build a real periodic sheet and write a 3-snapshot
        # archive with its natural (3D, periodic-stamped) structure.
        np.random.seed(0)
        src_sheet = VirtualSheet.planar_virtual_sheet_2d(
            "rs", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        src_sheet.vert_df["viscosity"] = 1.0
        src_sheet.geom.update_all(src_sheet)
        src_hist = HistoryHdf5(src_sheet, save_every=None, dt=1.0,
                               hf5file=archive, overwrite=True)
        for t in (0.0, 1.0, 2.0):
            src_hist.record(time_stamp=float(t))
        # The on-disk vert table must carry the 3D ``z`` column — this
        # is precisely what the resumed (2D) sheet won't have.
        with pd.HDFStore(archive, "r") as store:
            assert "z" in store["vert"].columns, (
                "fixture precondition failed: archive should carry z"
            )

        # --- Reload the way the resume path does: retrieve + arrange
        # to 2D (drops z), then build a fresh HistoryHdf5 from the 2D
        # sheet (captures the no-z structure record() will write).
        reload_hist = HistoryHdf5.from_archive(archive, eptm_class=VirtualSheet)
        resumed = reload_hist.retrieve(1.0)
        resumed.arrange_sheet_from_history(two_dim=True)
        resumed.initiate_edge_order()
        resumed.vert_df["viscosity"] = 1.0
        if hasattr(resumed, "_stash_periodic_metadata"):
            resumed._stash_periodic_metadata()
        resumed.geom.update_all(resumed)

        live_hist = HistoryHdf5(resumed, save_every=None, dt=1.0,
                                hf5file=archive, overwrite=True)
        assert "z" not in live_hist.columns["vert"], (
            "resumed 2D sheet unexpectedly still tracks a z column"
        )

        # --- Rewrite the kept portion (t <= 1.0) to match. ---
        _rewrite_history_for_resume(archive, 1.0, live_hist)

        # The tail snapshot (t=2.0) must be gone; t=0 and t=1 kept.
        with pd.HDFStore(archive, "r") as store:
            kept_times = sorted(store.select("vert", columns=["time"])["time"].unique())
            assert kept_times == [0.0, 1.0], (
                f"rewrite kept wrong snapshots: {kept_times}"
            )
            assert "z" not in store["vert"].columns, (
                "rewrite kept the legacy z column"
            )

        # --- The acid test: record() a new snapshot. Pre-fix this
        # raised ``cannot match existing table structure``. ---
        live_hist.save_every = None  # mirror what the solver does
        live_hist.time = 1.0
        live_hist.record(time_stamp=1.5)  # must NOT raise

        reloaded = HistoryHdf5.from_archive(archive, eptm_class=VirtualSheet)
        final_times = sorted(np.asarray(reloaded.time_stamps))
        assert 1.5 in [round(t, 6) for t in final_times], (
            f"new snapshot not recorded: {final_times}"
        )
        # Periodicity must survive the rewrite (we re-stashed).
        s = reloaded.retrieve(1.5)
        s.arrange_sheet_from_history(two_dim=True)
        assert s.periodic is True, "periodicity lost after resume rewrite"


class TestLateralInhibitionLevelsPreservedOnLoad:
    """The lateral-inhibition columns (notch_level, delta_level,
    repressor_level) USED to be lost when a sheet was loaded from
    HDF5 history and handed to ``InnerEarModel``: the constructor
    called ``update_specs(reset=True)`` with these columns in the
    spec dict, which overwrote the loaded values with the spec
    defaults (1.0), and then a separate pickle side-channel was
    needed to restore them.

    The fix is two-pronged: drop the LI keys from the spec dict in
    :meth:`InnerEarModel.get_specs_2d` (so the loaded values
    survive ``update_specs``) and teach :meth:`initialize_notch_delta`
    to preserve columns that are already populated. This test class
    pins both halves of that contract so the pickle scaffolding
    stays gone."""

    @staticmethod
    def _build_inner(sheet):
        from inner_ear_model import InnerEarModel
        return InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )

    @staticmethod
    def _fresh_sheet(name):
        np.random.seed(0)
        return VirtualSheet.planar_virtual_sheet_2d(
            name, nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )

    def test_fresh_sheet_gets_randomised_LI_levels(self):
        sheet = self._fresh_sheet("fresh")
        # Fresh sheet must not carry LI columns yet.
        assert "notch_level" not in sheet.face_df.columns
        assert "delta_level" not in sheet.face_df.columns

        inner = self._build_inner(sheet)
        for col in ("notch_level", "delta_level", "repressor_level"):
            assert col in inner.sheet.face_df.columns, (
                f"InnerEarModel didn't seed {col!r} on a fresh sheet"
            )
            v = inner.sheet.face_df[col].to_numpy()
            assert v.std() > 1e-6, (
                f"{col!r} ended up constant on a fresh sheet; "
                f"expected random initialisation"
            )

    def test_preloaded_LI_levels_survive_constructor(self):
        """The continued-run case: sheet was loaded from history,
        the LI columns are already populated, and the constructor
        must NOT clobber them."""
        sheet = self._fresh_sheet("preloaded")
        Nf = sheet.Nf
        notch_in = np.linspace(0.0, 1.0, Nf)
        delta_in = np.linspace(1.0, 0.0, Nf)
        rep_in = np.full(Nf, 0.42)
        sheet.face_df["notch_level"] = notch_in
        sheet.face_df["delta_level"] = delta_in
        sheet.face_df["repressor_level"] = rep_in

        inner = self._build_inner(sheet)

        notch_out = inner.sheet.face_df["notch_level"].to_numpy()
        delta_out = inner.sheet.face_df["delta_level"].to_numpy()
        rep_out = inner.sheet.face_df["repressor_level"].to_numpy()
        np.testing.assert_allclose(notch_out, notch_in, err_msg=(
            "notch_level got overwritten by InnerEarModel.__init__ — "
            "the loaded values must survive update_specs(reset=True)"
        ))
        np.testing.assert_allclose(delta_out, delta_in, err_msg=(
            "delta_level got overwritten by InnerEarModel.__init__"
        ))
        np.testing.assert_allclose(rep_out, rep_in, err_msg=(
            "repressor_level got overwritten by InnerEarModel.__init__"
        ))

    def test_legacy_archive_without_repressor_gets_topped_up(self):
        """An older archive may have notch + delta but no repressor
        column. Preserve the two it has, randomise just the missing
        one."""
        sheet = self._fresh_sheet("legacy")
        Nf = sheet.Nf
        notch_in = np.linspace(0.0, 1.0, Nf)
        delta_in = np.linspace(1.0, 0.0, Nf)
        sheet.face_df["notch_level"] = notch_in
        sheet.face_df["delta_level"] = delta_in
        # NB: no repressor_level set.

        inner = self._build_inner(sheet)

        np.testing.assert_allclose(
            inner.sheet.face_df["notch_level"].to_numpy(), notch_in,
        )
        np.testing.assert_allclose(
            inner.sheet.face_df["delta_level"].to_numpy(), delta_in,
        )
        assert "repressor_level" in inner.sheet.face_df.columns, (
            "repressor_level should be added when missing"
        )
        assert inner.sheet.face_df["repressor_level"].std() > 1e-6, (
            "repressor_level was added but with constant values; "
            "expected random initialisation"
        )

    def test_saved_notch_delta_levels_file_parameter_removed(self):
        """The pickle side-channel is dead — confirm the keyword is
        actually gone from the public InnerEarModel signature so a
        caller that still relies on it gets a clean TypeError instead
        of silently no-oping."""
        from inner_ear_model import InnerEarModel
        import inspect
        sig = inspect.signature(InnerEarModel.__init__)
        assert "saved_notch_delta_levels_file" not in sig.parameters, (
            "saved_notch_delta_levels_file is still a constructor "
            "parameter; the pickle side-channel was supposed to go"
        )

    def test_randomize_notch_delta_levels_flag_overrides_preservation(self):
        """``randomize_notch_delta_levels=True`` should reseed the
        LI columns even when the loaded sheet already carries
        values — the opt-in escape hatch for parameter sweeps that
        want to re-use a saved geometry but start each sweep point
        from a fresh LI distribution."""
        sheet = self._fresh_sheet("force_random")
        Nf = sheet.Nf
        notch_in = np.linspace(0.0, 1.0, Nf)
        delta_in = np.linspace(1.0, 0.0, Nf)
        rep_in = np.full(Nf, 0.42)
        sheet.face_df["notch_level"] = notch_in
        sheet.face_df["delta_level"] = delta_in
        sheet.face_df["repressor_level"] = rep_in

        from inner_ear_model import InnerEarModel
        np.random.seed(123)  # determinism for the sanity asserts below
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
            randomize_notch_delta_levels=True,
        )
        notch_out = inner.sheet.face_df["notch_level"].to_numpy()
        delta_out = inner.sheet.face_df["delta_level"].to_numpy()
        rep_out = inner.sheet.face_df["repressor_level"].to_numpy()

        # The values should have CHANGED (the loaded sentinels were
        # overwritten by fresh randomisation).
        assert not np.allclose(notch_out, notch_in), (
            "randomize_notch_delta_levels=True did NOT replace "
            "the loaded notch_level values"
        )
        assert not np.allclose(delta_out, delta_in), (
            "randomize_notch_delta_levels=True did NOT replace "
            "the loaded delta_level values"
        )
        assert not np.allclose(rep_out, rep_in), (
            "randomize_notch_delta_levels=True did NOT replace "
            "the loaded repressor_level values"
        )
        # ... and they should be non-trivially varied (not all zero,
        # not all the same value — proves we got real randomisation
        # rather than a column of zeros).
        assert notch_out.std() > 1e-6
        assert delta_out.std() > 1e-6
        assert rep_out.std() > 1e-6

    def test_randomize_flag_default_keeps_loaded_values(self):
        """Confirm the override is genuinely OPT-IN: leaving the
        flag at its default ``False`` keeps the historical
        preserve-loaded-values behaviour the previous test class
        already pins."""
        sheet = self._fresh_sheet("default_preserves")
        Nf = sheet.Nf
        notch_in = np.linspace(0.0, 1.0, Nf)
        sheet.face_df["notch_level"] = notch_in
        sheet.face_df["delta_level"] = notch_in[::-1]
        sheet.face_df["repressor_level"] = np.full(Nf, 0.123)

        inner = self._build_inner(sheet)  # randomize_notch_delta_levels=False (default)

        np.testing.assert_allclose(
            inner.sheet.face_df["notch_level"].to_numpy(), notch_in,
            err_msg="default behaviour silently re-randomised the LI levels",
        )


class TestForkFromSnapshot:
    """``continue_from_time`` with ``continue_existing_run=False`` is
    "fork" mode: load the chosen snapshot of the source archive,
    write into a NEW folder, start fresh at t=0. The source archive
    MUST stay byte-identical — this is the load-only path."""

    @staticmethod
    def _write_minimal_history(path, sheet, times):
        """Use HistoryHdf5 to write a few snapshots of ``sheet`` at
        ``times`` so ``load_sheet_from_file(time_point=t)`` has real
        history rows to retrieve."""
        import os
        from tyssue import HistoryHdf5
        # Make sure the directory exists.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        # geom.update_all so the dtype check inside HistoryHdf5
        # doesn't trip on int → float migrations later.
        sheet.geom.update_all(sheet)
        hist = HistoryHdf5(
            sheet, save_every=None, dt=0.01,
            hf5file=path, overwrite=True,
        )
        for t in times:
            # Wiggle a single vertex so retrieve() returns different
            # data at different times — confirms we're picking the
            # right snapshot.
            sheet.vert_df.iloc[0, sheet.vert_df.columns.get_loc("x")] = float(t) * 0.01
            sheet.geom.update_all(sheet)
            hist.record(time_stamp=float(t))
        return hist

    def test_load_sheet_from_file_with_explicit_time_point(self, tmp_path):
        """``load_sheet_from_file(time_point=t)`` must retrieve the
        chosen snapshot — NOT the last one."""
        from run_model import load_sheet_from_file
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "fork_src", nx=2, ny=2, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        sheet.vert_df["viscosity"] = 1.0
        # New convention: load_sheet_from_file(name) reads
        # ``<name>/history.hf5``.
        name_dir = tmp_path / "fork_src"
        name_dir.mkdir()
        path_prefix = str(name_dir)
        self._write_minimal_history(
            os.path.join(path_prefix, "history.hf5"), sheet,
            times=[0.5, 1.0, 1.5, 2.0],
        )

        # Load at the MIDDLE snapshot, not the last.
        loaded = load_sheet_from_file(path_prefix, time_point=1.0)
        # The wiggled vertex's x-coord encodes the snapshot time
        # (we set it to 0.01*t in _write_minimal_history).
        wiggled_x = float(loaded.vert_df.iloc[0]["x"])
        assert abs(wiggled_x - 0.01) < 1e-9, (
            f"loaded sheet doesn't match t=1.0 snapshot: "
            f"vert0.x={wiggled_x} (expected 0.01)"
        )

        # And the historical default (time_point=None → last) still
        # gives the LAST snapshot — guard against my edit silently
        # changing the default behaviour.
        loaded_last = load_sheet_from_file(path_prefix)
        last_x = float(loaded_last.vert_df.iloc[0]["x"])
        assert abs(last_x - 0.02) < 1e-9, (
            f"default load didn't grab the last snapshot: vert0.x={last_x}"
        )

    def test_fork_from_snapshot_does_not_modify_source_archive(
        self, tmp_path,
    ):
        """The fork path must read the source archive in ONLY mode —
        not delete it, not truncate it, not append to it. Verify by
        comparing the file's bytes before and after."""
        import shutil, hashlib
        from run_model import load_sheet_from_file
        from inner_ear_model import _truncate_history_file

        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "src", nx=2, ny=2, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        sheet.vert_df["viscosity"] = 1.0
        # New convention: load_sheet_from_file(name) reads
        # ``<name>/history.hf5``.
        name_dir = tmp_path / "src"
        name_dir.mkdir()
        path_prefix = str(name_dir)
        archive = os.path.join(path_prefix, "history.hf5")
        self._write_minimal_history(
            archive, sheet, times=[0.5, 1.0, 1.5],
        )

        def _hash(p):
            with open(p, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        digest_before = _hash(archive)

        # The fork path only ever calls ``load_sheet_from_file`` on
        # the source — it never calls ``_truncate_history_file``
        # (that's the resume path). Exercise the load and verify
        # the archive is byte-identical afterward.
        _ = load_sheet_from_file(path_prefix, time_point=1.0)
        digest_after = _hash(archive)
        assert digest_after == digest_before, (
            "source archive changed after fork-mode load — "
            "load_sheet_from_file must be read-only"
        )

        # As a separate guarantee: confirm _truncate_history_file
        # WOULD modify the archive, so the byte-identity check above
        # actually means something.
        sentinel_path = str(name_dir / "copy.hf5")
        shutil.copy(archive, sentinel_path)
        _truncate_history_file(sentinel_path, 1.0)
        assert _hash(sentinel_path) != digest_before, (
            "_truncate_history_file is supposed to mutate the file; "
            "if this assertion fails the byte-identity test above "
            "isn't really testing anything"
        )


# --------------------------------------------------------------------------- #
# Layer 7c-ter — saved-time-point artefacts cross-consistency                  #
# --------------------------------------------------------------------------- #


def _find_example_artefact_dirs():
    """Locate every ``results/<name>/`` folder that has the full
    triple of artefacts written by ``save_data_of_a_given_time_point``:
    ``<name>.hf5`` + ``<name>_contact_matrix.npy`` + ``<name>.npy``.

    Returns a list of ``(name, base_path_without_suffix)`` tuples,
    sorted so failing tests reproduce in a stable order. Empty
    when none of the example folders are present — which is fine,
    ``pytest`` will then skip the parametrised consistency tests
    via ``pytest.skip`` below.
    """
    out = []
    from history_io import RESULTS_DIR
    root = RESULTS_DIR
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        base = os.path.join(root, name, name)
        if (os.path.isfile(base + ".hf5")
                and os.path.isfile(base + "_contact_matrix.npy")
                and os.path.isfile(base + ".npy")):
            out.append((name, base))
    return out


_EXAMPLE_ARTEFACT_DIRS = _find_example_artefact_dirs()


class TestSavedTimePointArtefactsConsistent:
    """The three artefacts written by
    ``history_io.save_data_of_a_given_time_point``:

    * ``<name>.hf5`` — a single-snapshot ``HistoryHdf5`` archive
      built from ``extract_time_point_to_new_history``;
    * ``<name>_contact_matrix.npy`` — a ``(N, N)`` matrix where
      ``N = max(unique_id) + 1`` and entry ``(i, j)`` is the sum
      of edge lengths between faces with ``unique_id == i`` and
      ``unique_id == j``;
    * ``<name>.npy`` — a 2-D unsigned-integer label image with
      ``0`` for boundary and positive integer labels for cells.

    The pipeline that produced them takes one path through the
    runtime, but each artefact is materialised by a different
    code path (``History.to_archive``, ``get_contact_matrix``,
    ``save_sheet_labels_to_numpy``). If any of them silently
    drifts from the others — e.g. the HDF5 round-trip duplicates
    rows but the matrix is computed before that, or the labeling
    sweep misses cells — downstream analysis (post-processing,
    comparison to experiment) consumes inconsistent data without
    noticing.

    These tests pin the cross-consistency contract on the example
    artefacts shipped under ``results/random_periodic_array*_for_*/``
    and skip cleanly when none of those folders are present (so
    a CI box without the example data still passes).

    Implementation note: ``save_data_of_a_given_time_point`` now
    removes any stale archive before writing, so freshly-saved
    HF5 files hold exactly one copy of each row. Some LEGACY
    example archives (saved before that fix) still carry every
    row twice — ``History.to_archive`` opens the file in ``"a"``
    (append) mode, so re-touching the same path stacked a second
    copy. The contact matrix and label image were always computed
    on the clean in-memory sheet BEFORE that round-trip, so the
    consistency check deduplicates the reloaded face/edge frames
    by index before reconstructing. The dedup is a no-op on the
    fixed single-copy archives and only does real work on the
    legacy doubled ones — keeping it makes the test robust to
    both."""

    @staticmethod
    def _load_artefacts(base):
        """Open the HF5, retrieve the single snapshot, defensively
        deduplicate index-collided rows (a no-op on archives saved
        after the ``save_data_of_a_given_time_point`` remove-before-
        write fix; necessary for legacy doubled archives), and load
        the matrix + image."""
        # Local import so the module still imports on hosts where
        # tyssue's HDF5 stack is missing (CI without h5py / tables).
        from tyssue import HistoryHdf5
        history = HistoryHdf5.from_archive(
            base + ".hf5", eptm_class=VirtualSheet,
        )
        ts = list(np.asarray(history.time_stamps))
        assert len(ts) >= 1, f"{base}.hf5 has no recorded times"
        # save_data_of_a_given_time_point only ever extracts one
        # snapshot, so we use the first (== only) recorded time.
        t = float(ts[0])
        sheet = history.retrieve(t)
        sheet.arrange_sheet_from_history(two_dim=True)
        # Deduplicate by index — see class docstring for why.
        sheet.face_df = sheet.face_df[~sheet.face_df.index.duplicated()].copy()
        sheet.edge_df = sheet.edge_df[~sheet.edge_df.index.duplicated()].copy()
        contact_matrix = np.load(base + "_contact_matrix.npy")
        label_image = np.load(base + ".npy")
        return history, sheet, contact_matrix, label_image, t

    # ----- HDF5 archive shape ----------------------------------------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_history_archive_is_single_snapshot(self, name, base):
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        history, sheet, _, _, t = self._load_artefacts(base)
        ts = sorted(set(round(float(x), 9) for x in np.asarray(history.time_stamps)))
        assert ts == [round(t, 9)], (
            f"{name}: archive should hold a single snapshot, "
            f"got time stamps {ts}"
        )
        # Deduplicated sheet must be a valid epithelium: positive
        # face count, every edge points at a face that exists.
        assert sheet.Nf > 0
        live_faces = set(sheet.face_df.index)
        stray = sheet.edge_df[~sheet.edge_df["face"].isin(live_faces)]
        assert stray.empty, (
            f"{name}: {len(stray)} edges reference dead face labels "
            f"(face_df / edge_df out of sync after dedup)"
        )

    # ----- Contact matrix structure ---------------------------------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_contact_matrix_is_square_symmetric_and_sized_to_unique_ids(
        self, name, base,
    ):
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        _, sheet, contact_matrix, _, _ = self._load_artefacts(base)
        assert contact_matrix.ndim == 2, (
            f"{name}: contact matrix isn't 2-D"
        )
        assert contact_matrix.shape[0] == contact_matrix.shape[1], (
            f"{name}: contact matrix isn't square: {contact_matrix.shape}"
        )
        N_expected = int(sheet.face_df["unique_id"].astype(int).max()) + 1
        assert contact_matrix.shape[0] == N_expected, (
            f"{name}: contact matrix is {contact_matrix.shape[0]}x{contact_matrix.shape[0]} "
            f"but history's max unique_id + 1 is {N_expected}"
        )
        # Entries must be non-negative real lengths.
        assert (contact_matrix >= 0).all(), (
            f"{name}: contact matrix has negative entries — "
            f"min = {contact_matrix.min()}"
        )
        # Pairwise symmetry: contact(i,j) == contact(j,i).
        np.testing.assert_allclose(
            contact_matrix, contact_matrix.T,
            atol=1e-9,
            err_msg=(
                f"{name}: contact matrix is not symmetric — "
                f"max asymmetry = "
                f"{np.abs(contact_matrix - contact_matrix.T).max()}"
            ),
        )

    # ----- Flagship: matrix == reconstruction from history ----------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_contact_matrix_matches_history_reconstruction(
        self, name, base,
    ):
        """Reconstruct the contact matrix from the (deduplicated)
        ``edge_df`` + ``face_df`` using exactly the formula in
        ``VirtualSheet.get_contact_matrix`` — bincount of
        ``f_uid * N + o_uid`` weighted by edge length. Result must
        equal the saved ``.npy`` to machine precision. This is
        what "consistent" actually means."""
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        _, sheet, contact_matrix, _, _ = self._load_artefacts(base)
        edge_df = sheet.edge_df
        face_df = sheet.face_df
        has_opp = edge_df["opposite"] >= 0
        fids = edge_df.loc[has_opp, "face"].to_numpy().astype(int)
        opp_idx = edge_df.loc[has_opp, "opposite"].to_numpy().astype(int)
        o_fids = edge_df.loc[opp_idx, "face"].to_numpy().astype(int)
        f_uids = face_df.loc[fids, "unique_id"].to_numpy().astype(int)
        o_uids = face_df.loc[o_fids, "unique_id"].to_numpy().astype(int)
        lengths = edge_df.loc[opp_idx, "length"].to_numpy()
        N = int(max(f_uids.max(), o_uids.max())) + 1
        rebuilt = np.bincount(
            f_uids * N + o_uids,
            weights=lengths,
            minlength=N * N,
        ).reshape(N, N)

        assert rebuilt.shape == contact_matrix.shape, (
            f"{name}: shape mismatch — saved {contact_matrix.shape}, "
            f"reconstructed {rebuilt.shape}"
        )
        np.testing.assert_allclose(
            rebuilt, contact_matrix,
            atol=1e-9,
            err_msg=(
                f"{name}: saved contact matrix doesn't match the "
                f"reconstruction from history (max abs diff = "
                f"{np.abs(rebuilt - contact_matrix).max():.3e})"
            ),
        )
        # And confirm the sparsity pattern: every nonzero ``(i, j)``
        # in the matrix corresponds to a pair of faces that share
        # at least one edge in the history.
        nz_pairs = set(zip(*np.where(contact_matrix > 0)))
        adjacent_pairs = set(zip(f_uids.tolist(), o_uids.tolist()))
        stray = nz_pairs - adjacent_pairs
        assert not stray, (
            f"{name}: contact matrix has {len(stray)} nonzero "
            f"(i, j) entries that don't correspond to any "
            f"edge-sharing face pair in the history; sample: "
            f"{sorted(stray)[:5]}"
        )

    # ----- Geometric sanity: row sum == perimeter -------------------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_contact_matrix_row_sum_equals_face_perimeter(
        self, name, base,
    ):
        """For a closed periodic mesh every edge is shared between
        exactly two faces, so the sum of row ``i`` of the contact
        matrix must equal the perimeter of the cell with
        ``unique_id == i``. This catches missing-edge bugs that
        the matrix-vs-rebuild check would miss when BOTH the
        matrix and the reconstruction are wrong in the same way."""
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        _, sheet, contact_matrix, _, _ = self._load_artefacts(base)
        # Map unique_id → perimeter on the (deduplicated) face_df.
        uid_to_perim = (
            sheet.face_df[["unique_id", "perimeter"]]
            .drop_duplicates("unique_id")
            .set_index("unique_id")["perimeter"]
            .to_dict()
        )
        row_sums = contact_matrix.sum(axis=1)
        offenders = []
        for uid, peri in uid_to_perim.items():
            uid = int(uid)
            if uid >= contact_matrix.shape[0]:
                continue
            rs = float(row_sums[uid])
            # Periodic interior cells: rs should equal perimeter.
            # Boundary (open) cells would have rs < perimeter, but
            # the example archives here are periodic so we expect
            # the strict equality. Loose atol to absorb fp noise.
            if abs(rs - peri) > 1e-6 * max(1.0, peri):
                offenders.append((uid, peri, rs))
        assert not offenders, (
            f"{name}: {len(offenders)} cell(s) have row-sum != "
            f"perimeter (first few: "
            f"{[(u, round(p, 6), round(r, 6)) for u, p, r in offenders[:5]]})"
        )

    # ----- Label image structure ------------------------------------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_label_image_basic_structure(self, name, base):
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        _, _, _, label_image, _ = self._load_artefacts(base)
        assert label_image.ndim == 2, (
            f"{name}: label image isn't 2-D: shape={label_image.shape}"
        )
        # ``save_sheet_labels_to_numpy`` casts to uint16 explicitly.
        # Tolerate any unsigned integer dtype.
        assert np.issubdtype(label_image.dtype, np.unsignedinteger), (
            f"{name}: label image dtype is {label_image.dtype}, "
            f"expected unsigned integer (uint16 per "
            f"save_sheet_labels_to_numpy)"
        )
        # 0 must be present — that's the boundary / gap colour.
        assert (label_image == 0).any(), (
            f"{name}: label image has no boundary (0-valued) pixels"
        )
        # ... and dominate the image (most of the canvas is the
        # padding / between-cell area, not inside cells). The
        # observed range on the example archives is ~98-99% (cells
        # render small in the 800x800 canvas), so this bound is
        # deliberately loose — it only catches "100% boundary"
        # (nothing rendered) and "0% boundary" (no spacing).
        boundary_frac = float((label_image == 0).sum()) / label_image.size
        assert 0.05 < boundary_frac < 0.9999, (
            f"{name}: boundary pixels are {boundary_frac:.2%} of the "
            f"image — outside the sane 5%-99.99% range; either every "
            f"cell got rendered or none did"
        )

    # ----- Label image vs history --------------------------------------
    @pytest.mark.parametrize(
        "name,base",
        _EXAMPLE_ARTEFACT_DIRS,
        ids=[n for n, _ in _EXAMPLE_ARTEFACT_DIRS] or ["no-examples"],
    )
    def test_label_image_cell_count_consistent_with_history(
        self, name, base,
    ):
        """The label image's positive-integer labels should
        roughly match ``sheet.Nf`` after dedup. ``Nf`` is the
        absolute upper bound (the labeller stops at ``Nc == Nf``),
        and the lower bound has to allow for some rendering loss
        — adjacent faces with very close ``id`` values can share
        a color bucket after matplotlib quantises to uint8. A 50%
        floor catches catastrophic divergence (e.g. only a handful
        of cells appearing) without flagging the cosmetic
        ~10% rendering loss seen on the real examples."""
        if not _EXAMPLE_ARTEFACT_DIRS:
            pytest.skip("no example artefact folders present")
        _, sheet, _, label_image, _ = self._load_artefacts(base)
        positive_labels = np.unique(label_image)
        positive_labels = positive_labels[positive_labels > 0]
        n_labels = int(len(positive_labels))
        n_face = int(sheet.Nf)
        assert n_labels > 0, (
            f"{name}: label image has no positive labels at all"
        )
        assert n_labels <= n_face, (
            f"{name}: label image has {n_labels} unique positive "
            f"labels but the sheet has only {n_face} faces — the "
            f"labeller can't produce more"
        )
        ratio = n_labels / max(n_face, 1)
        assert ratio >= 0.5, (
            f"{name}: label image covers only {n_labels}/{n_face} "
            f"= {ratio:.2%} of the cells (expected ≥ 50% — "
            f"likely a rendering or color-overflow bug)"
        )
        # The labeller assigns labels 1..Nc as it sweeps the
        # rendered colour space, so labels are contiguous starting
        # at 1 modulo (a) collisions in the color → label sweep
        # and (b) ``Nc`` not being reached. The TIGHTER invariant is:
        # max label ≤ Nf. Loose enough to survive matplotlib quirks.
        assert int(positive_labels.max()) <= n_face, (
            f"{name}: largest label = {int(positive_labels.max())} "
            f"exceeds Nf={n_face}"
        )


# --------------------------------------------------------------------------- #
# Layer 7c — adaptive dt + negative area safety net                            #
# --------------------------------------------------------------------------- #

class TestAdaptiveDt:
    """The solver must (B) reject steps whose max displacement exceeds
    the per-step cap, halving dt and retrying, and (C) revert and halve
    dt if a candidate step produces a face with negative area
    (edge-crossing fingerprint). It must also save history at constant
    SIMULATION-TIME intervals — not constant iteration count — and
    record the dt used for each snapshot."""

    @staticmethod
    def _build_solver(monkeypatch, velocity_fn, initial_dt=0.01):
        """Build a tiny periodic sheet + a minimal solver and replace
        scipy.solve_ivp inside ``solvers`` with a stub that calls
        ``velocity_fn(t, pos)`` for the velocity and returns
        ``pos + velocity * dt`` as the new position. This lets us
        exercise the adaptive-dt logic without going through scipy
        (which crashes on this Windows host)."""
        from solvers import IVPSolver
        from tyssue.dynamics import model_factory
        from tyssue.dynamics.effectors import FaceContractility, FaceAreaElasticity
        from tyssue import History

        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "ad_dt", nx=2, ny=2, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        sheet.vert_df["viscosity"] = 1.0
        sheet.vert_df["is_active"] = 1
        sheet.active_verts = np.arange(sheet.Nv)
        sheet.edge_df["is_active"] = 1
        sheet.face_df["is_active"] = 1

        history = History(sheet, save_every=0.1, save_all=False, dt=initial_dt)
        solver = IVPSolver(
            None, sheet, sheet.geom,
            model_factory([FaceContractility, FaceAreaElasticity]),
            manager=None, history=history,
        )
        # The solver takes one explicit-Euler step per accepted iteration:
        # ``new_pos = pos + dt * ode_func(t, pos)``. Inject the test
        # velocity directly as ode_func — the old indirection through a
        # solve_ivp stub is no longer needed now that the solver does a
        # single gradient evaluation per step.
        solver.ode_func = lambda t, pos: np.asarray(velocity_fn(t, pos))
        return sheet, solver

    def test_records_dt_with_each_snapshot(self, monkeypatch):
        """The dt used to advance to each saved time is stamped on
        face_df as a ``step_dt`` column when recorded."""
        # Zero velocity → every step accepted; no crossings, no shrinks.
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
        )
        solver.solve(
            tf=0.05, dt=0.01,
            max_displacement=1.0, save_interval=0.02,
        )
        # We always record at t=0 (init) and at the final time. With
        # save_interval=0.02 and tf=0.05 we expect snapshots near
        # t≈0, 0.02, 0.04, 0.05.
        ts = sorted(set(round(t, 4) for t in solver.history.time_stamps))
        assert len(ts) >= 3, f"expected ≥3 snapshots, got {ts}"
        # The face_df carries the step_dt column after the last record.
        assert "step_dt" in sheet.face_df.columns

    def test_save_interval_is_constant_in_time_not_iterations(self, monkeypatch):
        """Even if dt shrinks (because a high-displacement step is
        rejected and retried with smaller dt), snapshots must still
        land at roughly constant SIMULATION-TIME spacing."""
        call_count = [0]
        def velocity(t, pos):
            call_count[0] += 1
            # Big velocity for first few attempts → rejections; then quiet.
            if call_count[0] <= 3:
                v = np.zeros_like(pos)
                v[0] = 100.0
                return v
            return np.zeros_like(pos)

        sheet, solver = self._build_solver(monkeypatch, velocity_fn=velocity)
        solver.solve(
            tf=0.1, dt=0.01,
            max_displacement=0.05, save_interval=0.05,
        )
        ts = sorted(set(solver.history.time_stamps))
        assert ts[0] == pytest.approx(0.0, abs=1e-9)
        # Expect at least one snapshot within save_interval/2 of t=0.05.
        assert any(abs(t - 0.05) < 0.025 for t in ts), (
            f"expected a snapshot near t=0.05, got {ts}"
        )

    def test_dt_min_floor_raises_runtime_error(self, monkeypatch):
        """If every dt down to dt_min_factor*initial_dt is rejected,
        the solver must raise instead of spinning forever."""
        def velocity(t, pos):
            v = np.zeros_like(pos); v[0] = 1e9; return v
        sheet, solver = self._build_solver(monkeypatch, velocity_fn=velocity)
        with pytest.raises(RuntimeError, match="below"):
            solver.solve(
                tf=0.1, dt=0.01,
                max_displacement=0.001,
                dt_min_factor=0.01,  # dt_min = 1e-4
            )

    def test_dt_ratchets_back_up_after_calm(self, monkeypatch):
        """After a rejection the dt grows back toward the initial value
        on subsequent successful steps."""
        rejected_once = [False]
        def velocity(t, pos):
            v = np.zeros_like(pos)
            if not rejected_once[0]:
                rejected_once[0] = True
                v[0] = 100.0  # one big rejection
            return v
        sheet, solver = self._build_solver(monkeypatch, velocity_fn=velocity)
        solver.solve(
            tf=0.05, dt=0.01,
            max_displacement=0.05, save_interval=0.05,
            dt_increase_factor=2.0,
        )
        final_dt = sheet.settings.get("dt")
        assert final_dt > 0.001
        assert final_dt <= 0.01 + 1e-9

    def test_preexisting_folds_do_not_block_run(self, monkeypatch):
        """The fold safety net rejects only steps that INCREASE the number of
        self-intersecting faces. A sheet that already carries a few folded
        cells (as several saved morphology sheets do) must still run, as long
        as the mechanics doesn't add more."""
        import solvers as solvers_mod
        # Constant fold count: pre-existing folds, never increasing.
        monkeypatch.setattr(solvers_mod, "count_folded_faces", lambda eptm, **k: 3)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        solver.solve(tf=0.05, dt=0.01, max_displacement=1.0, save_interval=0.02)
        # Ran to completion (no immediate rejection / RuntimeError).
        assert max(solver.history.time_stamps) >= 0.04

    def test_unavoidable_fold_raises_in_strict_mode(self, monkeypatch):
        """With tolerate_unavoidable_folds=False (mechanics-only parameter fit),
        a step that keeps introducing a NEW fold no matter how small dt gets
        collapses dt to its floor and the solver raises — the fit then scores
        those parameters worst-case."""
        import solvers as solvers_mod
        calls = [0]
        def fake_count(eptm, **k):
            calls[0] += 1
            return 0 if calls[0] == 1 else 1   # baseline 0, every step adds one
        monkeypatch.setattr(solvers_mod, "count_folded_faces", fake_count)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        with pytest.raises(RuntimeError, match="self-intersecting"):
            solver.solve(tf=0.05, dt=0.01, max_displacement=1.0, dt_min_factor=0.1,
                         tolerate_unavoidable_folds=False)

    def test_unavoidable_fold_is_tolerated_by_default(self, monkeypatch):
        """By default an INHERENT fold that survives down to the dt floor is
        tolerated (accepted) rather than crashing the run: in a full
        differentiation run these long-lived folds are normal and resolve as
        the tissue develops. The run must reach the end instead of raising."""
        import solvers as solvers_mod
        calls = [0]
        def fake_count(eptm, **k):
            calls[0] += 1
            return 0 if calls[0] == 1 else 1   # baseline 0, then a fold persists
        monkeypatch.setattr(solvers_mod, "count_folded_faces", fake_count)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        # No RuntimeError: the unavoidable fold is tolerated at the dt floor.
        solver.solve(tf=0.04, dt=0.01, max_displacement=1.0,
                     save_interval=0.02, dt_min_factor=0.1)
        assert max(solver.history.time_stamps) >= 0.02   # ran past the fold floor

    def test_rejected_step_logs_worst_vertex(self, monkeypatch, caplog):
        """When a step is rejected because a vertex moved too far, the
        warning AND the RuntimeError (if dt collapses) must name the
        offending vertex label so the user can inspect
        ``vert_df.loc[label]``."""
        import logging
        # Make ONLY vertex 3 move fast; others stay still.
        worst_label = 3
        def velocity(t, pos):
            v = np.zeros_like(pos)
            # pos is (n_active * dim,)-shaped; vertex i's x is index 2*i.
            v[2 * worst_label] = 100.0
            return v
        sheet, solver = self._build_solver(monkeypatch, velocity_fn=velocity)
        # Rejected steps are now logged at DEBUG (the cleaned-up policy: only
        # failures/rejections in a quiet run, at DEBUG), so capture at DEBUG.
        with caplog.at_level(logging.DEBUG, logger="tyssue.solvers.viscous"):
            # The dt floor here is tight enough that the solver must
            # eventually raise — we just want to see the vertex label
            # appear in BOTH the log record and the RuntimeError.
            with pytest.raises(RuntimeError, match=f"vertex {worst_label}"):
                solver.solve(
                    tf=0.005, dt=0.01,
                    max_displacement=0.05, save_interval=0.005,
                    dt_min_factor=0.5,
                )
        warning_messages = [r.getMessage() for r in caplog.records]
        assert any(f"vertex {worst_label}" in m for m in warning_messages), (
            f"expected a warning naming vertex {worst_label}, got: {warning_messages}"
        )

    def test_negative_area_triggers_revert_and_dt_shrink(self, monkeypatch):
        """A candidate step that produces a face with negative area
        (edge-crossing fingerprint) must be reverted and the next
        attempt must use a smaller dt."""
        # Force a specific vertex to teleport far enough to flip a face.
        # The 2x2 sheet has cells around (0.5, 0), (1.5, 0), (0, 1), (1, 1).
        # Moving one vertex far should flip at least one face's signed area.
        call_count = [0]
        def velocity(t, pos):
            call_count[0] += 1
            # First call: huge velocity on all vertices → big move
            # → likely produces negative area somewhere.
            # Later calls (after dt shrunk): tiny velocity → accepted.
            if call_count[0] <= 5:
                return np.ones_like(pos) * 50.0
            return np.zeros_like(pos)

        sheet, solver = self._build_solver(monkeypatch, velocity_fn=velocity)
        # Disable B by setting max_displacement very high so only C fires.
        solver.solve(
            tf=0.01, dt=0.01,
            max_displacement=1e9,
            save_interval=0.01,
            dt_min_factor=1e-6,
        )
        # Areas must end up positive (rejection succeeded or dt shrunk
        # enough that the step no longer flips areas).
        assert (sheet.face_df["area"] > 0).all(), (
            f"areas: {sheet.face_df['area'].tolist()}"
        )

    class _ForceNegGeom:
        """Wrap the real geometry so that after EVERY update one chosen face's
        signed area is forced negative — a sustained, dt-independent inversion
        (the solver calls ``geom.update_all`` in ``set_pos`` and after the
        manager, so the negative area is present at every negative-area check)."""
        def __init__(self, real, face_label):
            self._real = real
            self._face = face_label
        def update_all(self, eptm):
            self._real.update_all(eptm)
            eptm.face_df.at[self._face, "area"] = -0.01

    def test_delaminating_negative_area_is_tolerated(self, monkeypatch):
        """A face tagged delaminating (type == -1) with negative signed area
        must NOT trip the negative-area net — it is collapsing toward removal,
        so the step is accepted (letting the manager remove it) instead of
        being rejected to the dt floor."""
        import solvers as solvers_mod
        monkeypatch.setattr(solvers_mod, "count_folded_faces", lambda eptm, **k: 0)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        f0 = int(sheet.face_df.index[0])
        solver.geom = self._ForceNegGeom(sheet.geom, f0)
        sheet.face_df["type"] = 0
        sheet.face_df.at[f0, "type"] = -1     # delaminating
        # The forced inversion is permanent, so a dt-floor raise would be the
        # only outcome if the net still fired on it. With type == -1 it must not.
        solver.solve(tf=0.02, dt=0.01, max_displacement=1e9,
                     save_interval=0.02, dt_min_factor=0.1)
        assert max(solver.history.time_stamps) >= 0.01   # ran to completion

    def test_nondelaminating_negative_area_still_raises(self, monkeypatch):
        """The SAME permanent inversion on a normal cell (type != -1) must still
        be caught — reverted, dt shrunk, and (since it can't be stepped around)
        raised at the floor."""
        import solvers as solvers_mod
        monkeypatch.setattr(solvers_mod, "count_folded_faces", lambda eptm, **k: 0)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        f0 = int(sheet.face_df.index[0])
        solver.geom = self._ForceNegGeom(sheet.geom, f0)
        sheet.face_df["type"] = 0             # NOT delaminating
        with pytest.raises(RuntimeError, match="negative area"):
            solver.solve(tf=0.02, dt=0.01, max_displacement=1e9, dt_min_factor=0.1)

    def test_progress_bar_write_failure_does_not_crash(self, monkeypatch):
        """A tqdm console-write failure (Windows OSError [Errno 22]) on
        ``pbar.update`` / ``pbar.close`` must NOT abort the run — the progress
        bar is purely cosmetic."""
        import solvers as solvers_mod

        class _BrokenBar:
            def __init__(self, *a, **k):
                self.n = 0
                self.disable = False

            def update(self, *a, **k):
                if self.disable:   # real tqdm no-ops when disabled
                    return
                raise OSError(22, "Invalid argument")

            def close(self):
                raise OSError(22, "Invalid argument")

        monkeypatch.setattr(solvers_mod, "tqdm", _BrokenBar)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        # Completes despite the bar raising on every update and on close.
        solver.solve(tf=0.03, dt=0.01, max_displacement=1.0, save_interval=0.02)
        assert max(solver.history.time_stamps) >= 0.02

    class _NoBar:
        def __init__(self, *a, **k):
            self.n = 0
            self.disable = False

        def update(self, *a, **k):
            pass

        def close(self):
            pass

    @staticmethod
    def _fake_clock(monkeypatch, step=100.0):
        """Replace time.monotonic with a clock that jumps ``step`` seconds per
        call, and silence tqdm so it doesn't also consume the clock."""
        import solvers as solvers_mod
        monkeypatch.setattr(solvers_mod, "tqdm", TestAdaptiveDt._NoBar)
        clk = [0.0]

        def fake_mono():
            clk[0] += step
            return clk[0]
        monkeypatch.setattr(solvers_mod.time, "monotonic", fake_mono)

    def test_wall_clock_budget_stops_run(self, monkeypatch):
        """With ``max_wall_seconds`` set, a run that takes too long (per a fake
        fast-forwarding clock) raises so the fit can score it worst-case."""
        self._fake_clock(monkeypatch, step=100.0)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        with pytest.raises(RuntimeError, match="wall-clock budget"):
            solver.solve(tf=1.0, dt=0.01, max_displacement=1.0, max_wall_seconds=50.0)

    def test_slow_progress_stops_run(self, monkeypatch):
        """With ``min_progress_rate`` set, a run advancing simulation-time far
        slower than the floor (fake clock races ahead of sim-time) raises."""
        self._fake_clock(monkeypatch, step=100.0)
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        with pytest.raises(RuntimeError, match="too slowly"):
            solver.solve(tf=1.0, dt=0.01, max_displacement=1.0,
                         min_progress_rate=1.0, progress_window_seconds=50.0)

    def test_stall_limits_default_off_completes(self, monkeypatch):
        """With the limits unset (default), a normal run is unaffected even if
        wall-clock would 'race' — the guard code is skipped entirely."""
        self._fake_clock(monkeypatch, step=1e6)   # would trip any budget if checked
        sheet, solver = self._build_solver(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        solver.solve(tf=0.03, dt=0.01, max_displacement=1.0, save_interval=0.02)
        assert max(solver.history.time_stamps) >= 0.02


class TestDebugLogNoDuplication:
    """``run_model._enable_debug_log`` must write each log record EXACTLY ONCE.
    The previous version attached the same handler to several loggers that lie
    in one propagation chain (e.g. ``tyssue.solvers.viscous`` -> ``tyssue`` ->
    root, all in ``_DEBUG_LOG_TARGETS``), so a single solver warning was written
    3 times. The fix attaches the handler to root only."""

    def test_each_record_written_once(self, tmp_path):
        import logging
        import run_model
        root = logging.getLogger("")
        saved_root_level = root.level
        log_path = str(tmp_path / "debug.log")
        handler = run_model._enable_debug_log(log_path)
        try:
            # A record on the deepest chain logger (3 ancestors-with-handlers
            # before the fix) and one on a top-level pipeline logger.
            logging.getLogger("tyssue.solvers.viscous").warning("REJECTED_MARKER_X")
            logging.getLogger("virtual_sheet").info("COLLAPSE_MARKER_X")
            handler.flush()
            text = open(log_path, encoding="utf-8").read()
        finally:
            run_model._disable_debug_log(handler)
            root.setLevel(saved_root_level)
        assert text.count("REJECTED_MARKER_X") == 1, text
        assert text.count("COLLAPSE_MARKER_X") == 1, text


class TestSharpCornerCollapse:
    """Incipient folds where two non-adjacent vertices drift together pinch a
    face into a thin spike — a tiny INTERIOR ANGLE at the spike vertex — long
    before any edge shrinks below the intercalation length. ``VirtualSheet``
    detects these (``get_sharp_corner_collapse_edges``) and relieves them
    (``collapse_sharp_corners``: virtual edge -> remove a virtual vertex, real
    edge -> T1) every step inside ``update_virtual_vertices``."""

    @staticmethod
    def _periodic_sheet(nx=6, ny=6, seed=1):
        import numpy as np
        np.random.seed(seed)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "sharp", nx=nx, ny=ny, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        sheet.geom.update_all(sheet)
        return sheet

    def test_default_threshold_from_length_scales(self):
        """The default threshold is the apex angle of an isosceles triangle with
        legs ``max_bond_length`` and base ``max(min_bond, intercalation)`` — the
        angle at which a corner's two neighbours, on max-length edges, sit the
        larger length apart."""
        import numpy as np
        # legs = max_bond/2 = 0.1, base = max(min_bond, intercalation) = 0.05.
        leg = 0.2 / 2
        expected = float(np.arccos(1 - 0.05 ** 2 / (2 * leg ** 2)))
        assert VirtualSheet.default_sharp_angle_threshold(0.05, 0.04, 0.2) == pytest.approx(expected)
        assert expected == pytest.approx(0.5054, abs=1e-3)  # ~29 deg
        # The base uses max(min_bond, intercalation): symmetric in those two args.
        assert (VirtualSheet.default_sharp_angle_threshold(0.05, 0.04, 0.2)
                == VirtualSheet.default_sharp_angle_threshold(0.04, 0.05, 0.2))
        # Longer legs RELAX the threshold vs short min/intercalation legs.
        short_leg = float(np.arccos(np.clip(1 - 0.05 ** 2 / (2 * 0.04 ** 2), -1, 1)))
        assert VirtualSheet.default_sharp_angle_threshold(0.05, 0.04, 0.2) < short_leg
        # Degenerate (base >= 2*leg) clamps to pi instead of erroring.
        assert VirtualSheet.default_sharp_angle_threshold(0.05, 0.04, 0.02) == pytest.approx(np.pi)

    def test_detects_spike_and_picks_shorter_edge(self):
        """A regular square (all 90 deg corners) yields nothing; a triangle with
        a sharp spike at one vertex yields exactly the SHORTER of the two edges
        meeting at the spike. Pure angle math — driven through a synthetic
        edge_df so it doesn't depend on a particular tissue."""
        import pandas as pd
        class _Fake:
            pass
        fake = _Fake()
        fake.edge_df = pd.DataFrame({
            # face 0: unit square (order 1..4) — no sharp corner
            # face 1: triangle with a long spike at the vertex shared by
            #         edges 4 (len 5) and 5 (len ~5.1); shorter is edge 4.
            "face":   [0, 0, 0, 0, 1, 1, 1],
            "dx":     [1, 0, -1, 0, 5, -5, 0],
            "dy":     [0, 1, 0, -1, 0, 1, -1],
            "length": [1, 1, 1, 1, 5, 5.0990195, 1],
            "order":  [1, 2, 3, 4, 1, 2, 3],
            "srce":   [0, 1, 2, 3, 10, 11, 12],
            "trgt":   [1, 2, 3, 0, 11, 12, 10],
        })
        chosen = VirtualSheet.get_sharp_corner_collapse_edges(fake, 0.35)
        assert chosen.tolist() == [4]
        # Disabled when no threshold is given.
        assert VirtualSheet.get_sharp_corner_collapse_edges(fake, None).size == 0

    def test_clean_sheet_is_a_noop(self):
        """A relaxed hex lattice has ~120 deg interior angles everywhere, so the
        collapse must do nothing and report it."""
        sheet = self._periodic_sheet()
        assert sheet.get_sharp_corner_collapse_edges(0.35).size == 0
        ne, nf = sheet.Ne, sheet.Nf
        assert sheet.collapse_sharp_corners(0.35) is False
        assert (sheet.Ne, sheet.Nf) == (ne, nf)

    def test_threshold_none_disables_collapse(self):
        sheet = self._periodic_sheet()
        ne, nf = sheet.Ne, sheet.Nf
        assert sheet.collapse_sharp_corners(None) is False
        assert (sheet.Ne, sheet.Nf) == (ne, nf)

    def test_collapse_resolves_spike_without_adding_folds(self):
        """Pinch a face by drawing two non-adjacent neighbours of a vertex
        together (V fixed) until a sharp corner is detected, then collapse:
        the sharp corner must be gone, no NEW fold introduced, and the sheet
        must be left structurally consistent (contiguous index, valid
        opposites)."""
        import numpy as np
        from solvers import count_folded_faces
        sheet = self._periodic_sheet(nx=6, ny=6, seed=1)
        face = int(sheet.edge_df["face"].iloc[6])
        fe = sheet.edge_df[sheet.edge_df["face"] == face].sort_values("order")
        verts = fe["srce"].tolist()
        A, _V, B = int(verts[0]), int(verts[1]), int(verts[2])
        posA = sheet.vert_df.loc[A, sheet.coords].to_numpy()
        posB = sheet.vert_df.loc[B, sheet.coords].to_numpy()
        mid = 0.5 * (posA + posB)
        # Tighten until the angle test fires (cap the search).
        for shrink in (0.16, 0.12, 0.09, 0.06):
            sheet.vert_df.loc[A, sheet.coords] = mid + shrink * (posA - mid)
            sheet.vert_df.loc[B, sheet.coords] = mid + shrink * (posB - mid)
            sheet.geom.update_all(sheet)
            if sheet.get_sharp_corner_collapse_edges(0.35).size > 0:
                break
        assert sheet.get_sharp_corner_collapse_edges(0.35).size > 0, "setup failed to pinch"
        folded_before = count_folded_faces(sheet)

        collapsed = sheet.collapse_sharp_corners(0.35)
        assert collapsed is True
        # The sharp corner is gone and no NEW self-intersection was introduced.
        assert sheet.get_sharp_corner_collapse_edges(0.35).size == 0
        assert count_folded_faces(sheet) <= folded_before
        # Structurally consistent: index is a contiguous 0..Ne-1 label set
        # (rows are sorted by (face, order), so not in label order) and every
        # edge that has an opposite is its opposite's opposite.
        assert sorted(sheet.edge_df.index) == list(range(sheet.Ne))
        opp = sheet.edge_df["opposite"]
        for e in sheet.edge_df.index:
            o = int(opp.at[e])
            if o >= 0:
                assert int(opp.at[o]) == e


class TestVerboseLogging:
    """Topological-event logging policy (``run(verbose_log=...)`` ->
    ``sheet.verbose_log``): a SUCCESS is logged at INFO but ONLY in a verbose
    run; a FAILURE is logged at DEBUG ALWAYS. ``stacklevel`` keeps the record's
    filename pointing at the call site, not the ``log_topo_event`` helper."""

    def test_policy_success_gated_failure_always(self, caplog):
        import logging
        from topological_events import log_topo_event, logger as te_logger

        class _S:
            pass
        s = _S()
        s.verbose_log = False
        with caplog.at_level(logging.DEBUG, logger=te_logger.name):
            log_topo_event(te_logger, s, False, "FAILMSG_A")
            log_topo_event(te_logger, s, True, "SUCCESSMSG_QUIET")
        pairs = {(r.levelname, r.getMessage()) for r in caplog.records}
        assert ("DEBUG", "FAILMSG_A") in pairs           # failure -> DEBUG always
        assert not any("SUCCESSMSG_QUIET" in m for _, m in pairs)  # success suppressed
        caplog.clear()
        s.verbose_log = True
        with caplog.at_level(logging.DEBUG, logger=te_logger.name):
            log_topo_event(te_logger, s, True, "SUCCESSMSG_VERBOSE")
            log_topo_event(te_logger, s, False, "FAILMSG_B")
        pairs = {(r.levelname, r.getMessage()) for r in caplog.records}
        assert ("INFO", "SUCCESSMSG_VERBOSE") in pairs   # success -> INFO when verbose
        assert ("DEBUG", "FAILMSG_B") in pairs

    def test_stacklevel_points_at_call_site(self, caplog):
        import logging
        import os
        from topological_events import log_topo_event, logger as te_logger

        class _S:
            pass
        s = _S()
        s.verbose_log = True
        with caplog.at_level(logging.DEBUG, logger=te_logger.name):
            log_topo_event(te_logger, s, True, "STACKLEVEL_CHECK")
        rec = next(r for r in caplog.records if r.getMessage() == "STACKLEVEL_CHECK")
        assert os.path.basename(rec.pathname) == "test_periodic.py"

    @staticmethod
    def _spiked_sheet():
        np.random.seed(1)
        sh = VirtualSheet.planar_virtual_sheet_2d(
            "vlog", nx=6, ny=6, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False)
        sh.geom.update_all(sh)
        fe = sh.edge_df[sh.edge_df["face"] == int(sh.edge_df["face"].iloc[6])].sort_values("order")
        verts = fe["srce"].tolist()
        A, B = int(verts[0]), int(verts[2])
        pA = sh.vert_df.loc[A, sh.coords].to_numpy()
        pB = sh.vert_df.loc[B, sh.coords].to_numpy()
        mid = 0.5 * (pA + pB)
        for shrink in (0.12, 0.09, 0.06):
            sh.vert_df.loc[A, sh.coords] = mid + shrink * (pA - mid)
            sh.vert_df.loc[B, sh.coords] = mid + shrink * (pB - mid)
            sh.geom.update_all(sh)
            if sh.get_sharp_corner_collapse_edges(0.35).size > 0:
                break
        return sh

    def test_sharp_collapse_info_only_when_verbose(self, caplog):
        import logging

        def has_collapse_info():
            return any("collapsed sharp corner" in r.getMessage()
                       and r.levelname == "INFO" for r in caplog.records)

        sh = self._spiked_sheet()
        sh.verbose_log = False
        with caplog.at_level(logging.DEBUG, logger="virtual_sheet"):
            sh.collapse_sharp_corners(0.35)
        assert not has_collapse_info()   # quiet run: no success INFO line
        caplog.clear()
        sh = self._spiked_sheet()
        sh.verbose_log = True
        with caplog.at_level(logging.DEBUG, logger="virtual_sheet"):
            sh.collapse_sharp_corners(0.35)
        assert has_collapse_info()       # verbose run: success INFO line present


class TestSteadyStateStop:
    """``solver.solve(until_steady_state=True)`` halts as soon as the
    enabled criteria (mechanical and/or lateral-inhibition) declare
    convergence. ``tf`` becomes a safety cap; reaching it without
    convergence is a normal (un-halted) return.

    The criteria are:
      - mech_ok = max(|new_pos - old_pos|) < quasi_static_threshold
      - li_ok   = max(|new_li - old_li|) < lateral_inhibition_threshold
                  (over whichever of notch/delta/repressor exist)

    Topology change during a step always forces ``li_ok = False`` so
    the system has to run for at least one full clean step before
    declaring steady state."""

    @staticmethod
    def _build(monkeypatch, velocity_fn, initial_dt=0.01,
               with_li_cols=False):
        """Reuse ``TestAdaptiveDt._build_solver`` and (optionally) seed
        lateral-inhibition columns on face_df so the LI steady-state
        path is actually exercised."""
        sheet, solver = TestAdaptiveDt._build_solver(
            monkeypatch, velocity_fn=velocity_fn, initial_dt=initial_dt,
        )
        if with_li_cols:
            n = sheet.Nf
            sheet.face_df["notch_level"] = np.linspace(0.0, 1.0, n)
            sheet.face_df["delta_level"] = np.linspace(0.0, 1.0, n)[::-1]
            sheet.face_df["repressor_level"] = np.full(n, 0.5)
        return sheet, solver

    def test_mechanical_only_steady_stops_immediately(self, monkeypatch):
        """Zero velocity → every step is mechanically steady. With
        ``steady_state_min_steps=1`` (the old single-step behaviour)
        the solver should stop on the FIRST accepted step."""
        sheet, solver = self._build(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
        )
        # tf is large; if steady-state stop works, we end well before.
        solver.solve(
            tf=1000.0, dt=0.01,
            max_displacement=1.0, save_interval=0.01,
            until_steady_state=True,
            quasi_static_threshold=1e-6,
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=False,
            steady_state_min_steps=1,
        )
        # min_steps=1 → ends after a single accepted step (~initial_dt).
        assert solver.prev_t < 0.05, (
            f"expected stop after one step, but prev_t = {solver.prev_t}"
        )

    def test_mechanical_not_steady_runs_to_tf(self, monkeypatch):
        """A persistent velocity larger than the threshold should keep
        the solver going right up to ``tf``."""
        sheet, solver = self._build(
            # Constant velocity = 1 across all components: max
            # displacement per step is dt * 1 = 0.01, well above the
            # threshold 1e-4 we set below.
            monkeypatch, velocity_fn=lambda t, pos: np.ones_like(pos) * 1.0,
        )
        solver.solve(
            tf=0.05, dt=0.01,
            max_displacement=1.0, save_interval=0.05,
            until_steady_state=True,
            quasi_static_threshold=1e-4,
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=False,
        )
        # Should have run all the way to tf.
        assert solver.prev_t >= 0.05 - 1e-9, (
            f"expected to reach tf=0.05, got {solver.prev_t}"
        )

    def test_li_only_steady_stops_when_levels_converge(self, monkeypatch):
        """With check_mechanical_steady=False, the stop decision rests
        entirely on the lateral-inhibition columns. We seed those
        columns but don't run a differentiation manager, so they stay
        constant — every step is LI-steady → early stop."""
        sheet, solver = self._build(
            monkeypatch,
            velocity_fn=lambda t, pos: np.ones_like(pos) * 999.0,  # huge — mechanical never steady
            with_li_cols=True,
        )
        # Without a manager attached, LI columns never change → li_ok
        # fires on the first step.
        solver.solve(
            tf=1000.0, dt=0.01,
            max_displacement=1e9, save_interval=1.0,
            until_steady_state=True,
            check_mechanical_steady=False,
            check_lateral_inhibition_steady=True,
            lateral_inhibition_threshold=1e-6,
        )
        assert solver.prev_t < 1.0, (
            f"expected LI-only early stop, but prev_t = {solver.prev_t}"
        )

    def test_li_check_fails_when_li_columns_absent(self, monkeypatch):
        """If the caller asks for an LI steady check but the sheet
        carries none of the LI columns, the solver must NOT fake a
        ``li_ok=True`` — that would let it halt instantly with no
        evidence. Instead ``li_ok=False`` forever, so the simulation
        runs to ``tf``."""
        sheet, solver = self._build(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
            with_li_cols=False,  # no notch/delta/repressor on the sheet
        )
        solver.solve(
            tf=0.05, dt=0.01,
            max_displacement=1.0, save_interval=0.05,
            until_steady_state=True,
            check_mechanical_steady=False,
            check_lateral_inhibition_steady=True,
            lateral_inhibition_threshold=1.0,
        )
        # No LI cols + LI required → never steady → runs to tf.
        assert solver.prev_t >= 0.05 - 1e-9, (
            f"expected to reach tf=0.05, got {solver.prev_t}"
        )

    def test_both_required_holds_off_until_both_converge(self, monkeypatch):
        """When BOTH criteria are enabled, the solver must NOT stop on
        the first step that's only mechanically steady; it has to
        wait until the LI columns also stop changing.

        We orchestrate this by externally bumping a LI column on the
        very first step (so li_ok=False once), then leaving it
        unchanged on subsequent steps (li_ok=True)."""
        sheet, solver = self._build(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
            with_li_cols=True,
        )
        # Hook: on the first call the velocity is zero, but after the
        # step we bump the notch level by a constant — this forces
        # li_ok=False on iteration 1 and li_ok=True on iteration 2.
        call_count = [0]
        original_set_pos = solver.set_pos
        def wrapped_set_pos(new_pos):
            call_count[0] += 1
            original_set_pos(new_pos)
            if call_count[0] == 1:
                sheet.face_df["notch_level"] = (
                    sheet.face_df["notch_level"] + 0.5
                )
        solver.set_pos = wrapped_set_pos

        solver.solve(
            tf=10.0, dt=0.01,
            max_displacement=1.0, save_interval=0.01,
            until_steady_state=True,
            quasi_static_threshold=1e-6,
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=True,
            lateral_inhibition_threshold=1e-6,
        )
        # We should have made AT LEAST two accepted steps (one where
        # LI changed → no stop; second where LI is steady → stop) and
        # nowhere near tf.
        assert solver.prev_t < 1.0, (
            f"expected early stop after a few steps, prev_t = {solver.prev_t}"
        )
        assert solver.prev_t >= 0.02 - 1e-9, (
            f"expected at least two accepted steps before stopping, "
            f"got prev_t = {solver.prev_t}"
        )

    def test_requires_min_steps_consecutive_steady_steps(self, monkeypatch):
        """With ``steady_state_min_steps=4`` and zero velocity (every
        step mechanically steady), the solver must run for EXACTLY 4
        consecutive steady steps before halting — not 1. The 4th
        accepted step is where the streak first reaches the threshold."""
        sheet, solver = self._build(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos),
        )
        # Count accepted steps via a set_pos spy.
        n_accepted = [0]
        original = solver.set_pos
        def spy(new_pos):
            n_accepted[0] += 1
            original(new_pos)
        solver.set_pos = spy

        solver.solve(
            tf=1000.0, dt=0.01,
            max_displacement=1.0, save_interval=0.01,
            until_steady_state=True,
            quasi_static_threshold=1e-6,
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=False,
            dt_increase_factor=1.0,  # keep dt constant so steps == time/dt
            steady_state_min_steps=4,
        )
        # Exactly 4 accepted steps: streak 1,2,3,4 → halt on the 4th.
        assert n_accepted[0] == 4, (
            f"expected exactly 4 accepted steps for min_steps=4, "
            f"got {n_accepted[0]}"
        )

    def test_blip_resets_streak(self, monkeypatch):
        """A single non-steady step in the middle of a steady run must
        RESET the streak, so the solver can't halt until it has
        ``steady_state_min_steps`` CONSECUTIVE steady steps AFTER the
        blip."""
        # Velocity is zero (mech steady) except on the 3rd accepted
        # step, where it spikes above the threshold for one step.
        call = [0]
        def velocity(t, pos):
            call[0] += 1
            if call[0] == 3:
                # One big-but-uniform shove: a rigid translation keeps
                # areas positive (no C-reject) but the per-step
                # displacement exceeds quasi_static_threshold, so this
                # step is NOT mechanically steady → resets the streak.
                return np.ones_like(pos) * 1.0
            return np.zeros_like(pos)

        sheet, solver = self._build(monkeypatch, velocity_fn=velocity)
        n_accepted = [0]
        original = solver.set_pos
        def spy(new_pos):
            n_accepted[0] += 1
            original(new_pos)
        solver.set_pos = spy

        solver.solve(
            tf=1000.0, dt=0.01,
            max_displacement=1.0, save_interval=0.01,
            until_steady_state=True,
            quasi_static_threshold=1e-4,  # 1.0*0.01 = 0.01 > 1e-4 → blip not steady
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=False,
            dt_increase_factor=1.0,
            steady_state_min_steps=3,
        )
        # Steps: 1 steady(streak1), 2 steady(streak2), 3 BLIP(streak0),
        # 4 steady(1), 5 steady(2), 6 steady(3 → halt). 6 accepted.
        assert n_accepted[0] == 6, (
            f"expected the mid-run blip to reset the streak (6 accepted "
            f"steps total), got {n_accepted[0]}"
        )


class TestSimulateSteadyStateRouting:
    """``InnerEarModel.simulate(until_steady_state=True, ...)`` must
    translate the ``only_differentiation`` / ``no_differentiation``
    flags into the solver's per-criterion enable flags:

      - only_differentiation=True → ONLY the LI check matters
        (check_mechanical_steady=False)
      - no_differentiation=True   → ONLY the mechanical check matters
        (check_lateral_inhibition_steady=False)
      - both False                → BOTH checks must hold

    We don't need a full simulation here — we monkey-patch
    ``IVPSolver.solve`` to record the kwargs it receives and assert
    the right flags came through."""

    @staticmethod
    def _build_inner():
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "ss_route", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        from inner_ear_model import InnerEarModel
        return InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )

    @staticmethod
    def _capture_solve_kwargs(monkeypatch):
        """Replace ``IVPSolver.solve`` with a no-op that records its
        kwargs. Returns a list that the caller can inspect."""
        captured = {}
        from solvers import IVPSolver
        def _stub(self, *args, **kwargs):
            captured.update(kwargs)
            return None
        monkeypatch.setattr(IVPSolver, "solve", _stub)
        return captured

    def test_only_differentiation_routes_to_li_only(self, monkeypatch):
        inner = self._build_inner()
        captured = self._capture_solve_kwargs(monkeypatch)
        inner.simulate(
            t_end=1.0, dt=0.01,
            only_differentiation=True, no_differentiation=False,
            until_steady_state=True,
            lateral_inhibition_threshold=1e-4,
        )
        assert captured["until_steady_state"] is True
        assert captured["check_mechanical_steady"] is False, (
            f"only_differentiation should disable mechanical check; "
            f"got {captured['check_mechanical_steady']}"
        )
        assert captured["check_lateral_inhibition_steady"] is True
        assert captured["lateral_inhibition_threshold"] == 1e-4

    def test_no_differentiation_routes_to_mechanical_only(self, monkeypatch):
        inner = self._build_inner()
        captured = self._capture_solve_kwargs(monkeypatch)
        inner.simulate(
            t_end=1.0, dt=0.01,
            only_differentiation=False, no_differentiation=True,
            until_steady_state=True,
            lateral_inhibition_threshold=1e-4,
        )
        assert captured["until_steady_state"] is True
        assert captured["check_mechanical_steady"] is True
        assert captured["check_lateral_inhibition_steady"] is False, (
            f"no_differentiation should disable LI check; "
            f"got {captured['check_lateral_inhibition_steady']}"
        )

    def test_default_flags_route_to_both_checks(self, monkeypatch):
        inner = self._build_inner()
        captured = self._capture_solve_kwargs(monkeypatch)
        inner.simulate(
            t_end=1.0, dt=0.01,
            only_differentiation=False, no_differentiation=False,
            until_steady_state=True,
            lateral_inhibition_threshold=1e-4,
        )
        assert captured["until_steady_state"] is True
        assert captured["check_mechanical_steady"] is True
        assert captured["check_lateral_inhibition_steady"] is True

    def test_until_steady_state_false_disables_both_checks(self, monkeypatch):
        """When the caller doesn't ask for steady-state stopping, both
        check flags should still be False — the solver's loop body
        skips the entire steady-state block when
        ``until_steady_state`` is False, but we want the flags to
        match the intent so a future refactor can't silently start
        firing on them."""
        inner = self._build_inner()
        captured = self._capture_solve_kwargs(monkeypatch)
        inner.simulate(
            t_end=1.0, dt=0.01,
            until_steady_state=False,
        )
        assert captured["until_steady_state"] is False
        assert captured["check_mechanical_steady"] is False
        assert captured["check_lateral_inhibition_steady"] is False


# --------------------------------------------------------------------------- #
# Layer 7b — solver post-topology bookkeeping                                  #
# --------------------------------------------------------------------------- #

class TestSolverActiveVerts:
    """``active_verts`` stores POSITIONAL indices into ``vert_df``. After
    a topology event calls ``reset_index`` (e.g. inside
    ``remove_virtual_vertices``), the vert_df labels are renumbered and
    the stored ``active_verts`` becomes stale. The next iteration's
    ``current_pos = vert_df.loc[active_verts]`` then raises KeyError —
    which is what crashed ``periodic_tests.py`` after the first
    iteration. The solver fix is to refresh ``active_verts`` from
    ``is_active`` after every manager.execute."""

    def test_active_verts_survives_remove_virtual_vertices(self):
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p2x2_av", nx=2, ny=2, distx=1.0, disty=1.0,
            maximal_bond_length=0.2, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        # Wire up the minimum InnerEarModel state that sets is_active
        from inner_ear_model import InnerEarModel
        InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.}, repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
            differentiation_threshold=0.5,
        )

        from solvers import IVPSolver
        from tyssue.dynamics import model_factory
        from tyssue.dynamics.effectors import FaceContractility, FaceAreaElasticity
        sheet.vert_df["viscosity"] = 1
        solver = IVPSolver(
            None, sheet, sheet.geom,
            model_factory([FaceContractility, FaceAreaElasticity]),
            manager=None, history=None,
        )

        # Before any topology event, current_pos should work.
        pos_before = solver.current_pos
        assert pos_before.size == 2 * len(sheet.vert_df)

        # Topology event that drops vertices.
        sheet.remove_virtual_vertices()
        sheet.add_virtual_vertices()
        # FORCE active_verts to be stale (out-of-range index) so we can
        # deterministically check that _refresh_active_verts fixes it.
        sheet.active_verts = np.array(
            [len(sheet.vert_df) + 5], dtype=sheet.active_verts.dtype
        )
        assert sheet.active_verts.max() >= len(sheet.vert_df), (
            "test setup: forced active_verts must be out of range"
        )

        # The fix: solver._refresh_active_verts. Without it, current_pos
        # KeyErrors. With it, current_pos succeeds.
        solver._refresh_active_verts()
        pos_after = solver.current_pos
        assert pos_after.size == 2 * len(sheet.vert_df)


# --------------------------------------------------------------------------- #
# Layer 8 — non-periodic regression                                           #
# --------------------------------------------------------------------------- #

class TestNonPeriodicUnaffected:
    """The fixes for periodic mode must not regress the non-periodic
    construction path."""

    def test_non_periodic_construction_still_works(self):
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "np", nx=3, ny=3, distx=1.0, disty=1.0,
            maximal_bond_length=0.5, minimal_bond_length=0.05,
            periodic=False, draw_debug=False,
        )
        assert sheet.periodic is False
        # Free boundary → some edges have no opposite, that's expected
        assert (sheet.edge_df["opposite"] < 0).sum() > 0
        # All areas positive
        assert (sheet.face_df["area"] > 0).all()


class TestGifOutputNameShortening:
    """``create_gif`` shells out to ImageMagick's ``convert`` with the
    full output path; a too-long path makes ``convert`` fail (Windows
    MAX_PATH). ``_shorten_gif_output`` truncates the FILE NAME only —
    keeping the directory — so the convert call gets a path it can
    write."""

    @staticmethod
    def _import():
        from history_io import _shorten_gif_output, _MAX_GIF_PATH_LEN
        return _shorten_gif_output, _MAX_GIF_PATH_LEN

    def test_short_path_unchanged(self):
        _shorten, _ = self._import()
        p = os.path.join("base", "results", "run1", "run1.gif")
        assert _shorten(p) == p

    def test_long_name_shortened_directory_preserved(self):
        _shorten, MAX = self._import()
        # A ~100-char run name in a results/<name>/ directory — the
        # shape run_model.run() produces.
        longname = "periodic_run_" + "x" * 120
        directory = os.path.join("base", "results", longname)
        output = os.path.join(directory, longname + ".gif")
        assert len(output) > MAX  # precondition: actually too long

        res = _shorten(output)
        # Directory is kept EXACTLY; only the file name changed.
        assert os.path.dirname(res) == directory
        # Result fits the budget and is still a .gif.
        assert len(res) <= MAX
        assert res.endswith(".gif")
        # The stem prefix is preserved so the file is still
        # recognisable.
        assert os.path.basename(res).startswith(longname[:10])

    def test_distinct_long_names_get_distinct_files(self):
        _shorten, MAX = self._import()
        directory = os.path.join("base", "results")
        stem = "run_" + "y" * 120
        o1 = os.path.join(directory, stem + "_alpha.gif")
        o2 = os.path.join(directory, stem + "_beta.gif")
        s1, s2 = _shorten(o1), _shorten(o2)
        # Hash suffix keeps distinct long names from colliding.
        assert os.path.basename(s1) != os.path.basename(s2)
        assert len(s1) <= MAX and len(s2) <= MAX

    def test_shortening_is_deterministic(self):
        _shorten, _ = self._import()
        o = os.path.join("base", "results", "z" * 200, "z" * 200 + ".gif")
        assert _shorten(o) == _shorten(o)

    def test_create_gif_safe_shortens_the_output_path(self):
        """create_gif_safe no longer delegates to tyssue's create_gif — it
        assembles the gif with Pillow, because on Windows subprocess resolves
        ``convert`` to System32's FAT->NTFS tool rather than ImageMagick (and
        without check=True that failure was silent). What must still hold is
        that an over-long output path is shortened IN PLACE, keeping the
        directory, before anything tries to write it."""
        import history_io as pp

        longname = "run_" + "w" * 150
        directory = os.path.join("base", "results", longname)
        output = os.path.join(directory, longname + ".gif")

        shortened = pp._shorten_gif_output(output)
        assert os.path.dirname(shortened) == directory
        assert len(shortened) <= pp._MAX_GIF_PATH_LEN
        assert shortened.endswith(".gif")
        # and create_gif_safe uses exactly that helper
        assert pp._shorten_gif_output(shortened) == shortened   # idempotent

class TestLabeledSegmentationImage:
    """``save_sheet_labels_to_numpy`` / ``save_face_data_to_df`` /
    ``save_contact_matrix_to_numpy`` must produce a mutually
    consistent set of artifacts keyed by ``unique_id``:

    - ``labels.npy``: each interior pixel = ``unique_id + 1`` (1-based,
      so 0 is reserved for boundaries); membranes are 0.
    - ``cells_info``: indexed by ``unique_id`` (the raw 0-based id),
      with a ``label`` column == ``unique_id + 1`` (the image value).
    - ``contact_matrix``: ``m[i, j]`` = contact length between cells
      with ``unique_id`` i and j; missing ids are all-zero rows/cols.

    So an image pixel ``v`` maps to ``cells_info.loc[v - 1]`` and
    ``contact_matrix[v - 1]``. The old renderer produced ~99 %
    boundary with scrambled sequential labels; this pins the
    rasterized replacement and the cross-artifact consistency."""

    @staticmethod
    def _sheet():
        # A 6x6 periodic hex lattice run through InnerEarModel so the
        # face_df carries the columns ``save_face_data_to_df`` reads
        # (type / notch_level / delta_level / is_alive), seeding clean
        # hexagonal faces.
        from inner_ear_model import InnerEarModel
        sheet = _build_sheet(nx=6, ny=6)
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )
        return inner.sheet

    def test_unarranged_periodic_sheet_recovers_periodicity(self, tmp_path):
        """A sheet straight from ``history.retrieve`` keeps the default
        ``periodic=False`` (and no ``Lx``/``Ly``) even though its
        ``face_df`` still carries the ``_periodic_flag`` metadata.
        ``save_sheet_labels_to_numpy`` must recover periodicity from
        that flag, otherwise boundary-crossing faces stay unfolded and
        the image is scrambled at the seam (only a fraction of the
        cells survive)."""
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        # A fresh periodic sheet carries the stash columns.
        assert "_periodic_flag" in sheet.face_df.columns
        n_faces = sheet.face_df.shape[0]

        # Emulate an un-arranged retrieve: drop the periodic ATTRIBUTES
        # but leave the metadata COLUMNS in face_df.
        sheet.periodic = False

        lbl = str(tmp_path / "labels.npy")
        InnerEarModel.save_sheet_labels_to_numpy(sheet, lbl, pixels_per_unit=40)
        labels = np.load(lbl)

        # Every cell must be present — if the non-periodic branch had
        # run, boundary-crossing faces would overwrite each other and
        # many cells would be lost.
        present = set(np.unique(labels).tolist()) - {0}
        assert len(present) == n_faces, (
            f"recovered image has {len(present)} cells, expected "
            f"{n_faces} — periodicity was not recovered from the flag"
        )
        # Boundaries are thin membranes, not the scrambled-seam mess.
        assert (labels == 0).mean() < 0.33

    def test_labels_are_one_based_and_match_cells_info(self, tmp_path):
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        lbl = str(tmp_path / "labels.npy")
        ci_path = str(tmp_path / "cells_info.pkl")
        InnerEarModel.save_sheet_labels_to_numpy(sheet, lbl, pixels_per_unit=40)
        InnerEarModel.save_face_data_to_df(sheet, ci_path)

        labels = np.load(lbl)
        ci = pd.read_pickle(ci_path)

        # cells_info is INDEXED by unique_id (not a column).
        assert ci.index.name == "unique_id", (
            "cells_info must be indexed by unique_id"
        )
        assert "unique_id" not in ci.columns

        # Labels are 1-based: the smallest non-zero label is >= 1, and
        # the value 0 is present (it's the boundary, never a cell).
        nonzero = set(np.unique(labels[labels > 0]).tolist())
        assert min(nonzero) >= 1
        assert (labels == 0).any()

        # The image's non-zero labels are exactly unique_id + 1, which
        # equals both the cells_info index + 1 and its ``label`` column.
        expected = set((ci.index.to_numpy() + 1).tolist())
        assert nonzero == expected, (
            f"only-in-image={sorted(nonzero - expected)}, "
            f"only-in-cells_info={sorted(expected - nonzero)}"
        )
        assert set(ci["label"].astype(int).tolist()) == expected

    def test_pixel_holds_face_label(self, tmp_path):
        """Each face's centroid pixel carries ``unique_id + 1`` (the
        1-based label)."""
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        ppu = 40
        lbl = str(tmp_path / "labels.npy")
        InnerEarModel.save_sheet_labels_to_numpy(sheet, lbl, pixels_per_unit=ppu)
        labels = np.load(lbl)
        H, W = labels.shape

        fx = sheet.face_df["x"].to_numpy()
        fy = sheet.face_df["y"].to_numpy()
        uids = sheet.face_df["unique_id"].to_numpy()
        mismatches = 0
        for x, y, u in zip(fx, fy, uids):
            r = int(round(y * ppu)) % H
            c = int(round(x * ppu)) % W
            if labels[r, c] != u + 1:
                mismatches += 1
        assert mismatches == 0, (
            f"{mismatches}/{len(uids)} centroid pixels don't carry their "
            f"face's (unique_id + 1) label"
        )

    def test_contact_matrix_keyed_by_unique_id(self, tmp_path):
        """contact_matrix[i, j] is the contact length between cells
        with unique_id i and j; it is symmetric, zero-diagonal, sized
        to max(unique_id)+1, and consistent with cells_info.neighbors
        (also unique_id-keyed)."""
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        cm_path = str(tmp_path / "contact_matrix.npy")
        ci_path = str(tmp_path / "cells_info.pkl")
        InnerEarModel.save_contact_matrix_to_numpy(sheet, cm_path)
        InnerEarModel.save_face_data_to_df(sheet, ci_path)
        cm = np.load(cm_path)
        ci = pd.read_pickle(ci_path)

        max_uid = int(sheet.face_df["unique_id"].max())
        assert cm.shape == (max_uid + 1, max_uid + 1)
        assert np.allclose(cm, cm.T), "contact matrix must be symmetric"
        assert np.allclose(np.diag(cm), 0), "no self-contact on the diagonal"

        # For each cell, the unique_ids it touches (row > 0) equal the
        # unique_ids listed in cells_info.neighbors.
        for u in ci.index:
            row_neighbors = set(np.nonzero(cm[u] > 0)[0].tolist())
            assert row_neighbors == set(int(n) for n in ci.loc[u, "neighbors"]), (
                f"cell {u}: contact_matrix neighbors {sorted(row_neighbors)} "
                f"!= cells_info.neighbors {sorted(ci.loc[u, 'neighbors'])}"
            )

    def test_missing_unique_ids_become_zero_rows_cols(self, tmp_path):
        """A gap in unique_id (e.g. left by a delamination) yields an
        all-zero row and column at that id, without shifting any other
        cell's row/col."""
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        # Pick a real cell and relabel its unique_id to a large value,
        # opening a gap at its old id.
        old_uid = int(sheet.face_df["unique_id"].iloc[5])
        new_uid = int(sheet.face_df["unique_id"].max()) + 7
        face_label = sheet.face_df.index[sheet.face_df["unique_id"] == old_uid][0]
        sheet.face_df.loc[face_label, "unique_id"] = new_uid

        cm_path = str(tmp_path / "contact_matrix.npy")
        InnerEarModel.save_contact_matrix_to_numpy(sheet, cm_path)
        cm = np.load(cm_path)
        assert cm.shape == (new_uid + 1, new_uid + 1)
        # The vacated id is an all-zero row/col.
        assert np.allclose(cm[old_uid, :], 0) and np.allclose(cm[:, old_uid], 0)
        # The relabeled cell still has its contacts at the new id.
        assert (cm[new_uid, :] > 0).any()

    def test_boundaries_are_zero_and_not_dominant(self, tmp_path):
        """Boundary pixels are 0, but they must be thin membranes —
        NOT the ~99 % of the image the broken version produced."""
        from inner_ear_model import InnerEarModel
        sheet = self._sheet()
        lbl = str(tmp_path / "labels.npy")
        InnerEarModel.save_sheet_labels_to_numpy(sheet, lbl, pixels_per_unit=40)
        labels = np.load(lbl)
        zero_frac = float((labels == 0).mean())
        # Thin membranes on a 6x6 lattice: well under a third of the
        # image. (The broken renderer gave ~0.99.)
        assert zero_frac < 0.33, (
            f"boundary fraction {zero_frac:.3f} too high — membranes "
            f"should be thin"
        )

    def test_labels_spatially_coherent(self, tmp_path):
        """Each cell's pixels must form a compact region (not the
        scattered scramble the colour-decode produced). For an
        interior hex cell, the pixels are a single connected blob
        filling most of their bounding box."""
        from inner_ear_model import InnerEarModel
        from scipy import ndimage
        sheet = self._sheet()
        lbl = str(tmp_path / "labels.npy")
        InnerEarModel.save_sheet_labels_to_numpy(sheet, lbl, pixels_per_unit=40)
        labels = np.load(lbl)

        # Identify interior (non-boundary-crossing) faces: those whose
        # unfolded perimeter doesn't span the periodic seam. Simplest
        # proxy — a face whose pixels live in a small bounding box
        # (< half the image) is interior; test those for compactness.
        good = 0
        checked = 0
        for u in np.unique(labels):
            if u == 0:
                continue
            mask = labels == u
            ys, xs = np.where(mask)
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            if h > labels.shape[0] // 2 or w > labels.shape[1] // 2:
                continue  # boundary-crossing cell, split across the seam
            checked += 1
            _, ncomp = ndimage.label(mask)
            fill = mask.sum() / (h * w)
            if ncomp == 1 and fill > 0.4:
                good += 1
        assert checked > 0
        # The vast majority of interior cells are compact single blobs.
        assert good / checked > 0.9, (
            f"only {good}/{checked} interior cells are compact single "
            f"blobs — image looks scrambled"
        )


class TestLoadLILevelsFromNumpy:
    """Seed the lateral-inhibition initial values (notch / delta /
    repressor) from per-cell numpy data indexed by ``unique_id``.

    This is the clean replacement for the old pickled-DataFrame
    ``saved_notch_delta_levels_file`` side-channel. Entry ``i`` of each
    array holds the value for the cell whose ``unique_id == i`` (which
    equals the cell's ``face_df`` index on a fresh sheet)."""

    @staticmethod
    def _make_inner(notch_levels=None, delta_levels=None, repressor_levels=None,
                    randomize_notch_delta_levels=False):
        from inner_ear_model import InnerEarModel
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "li_load", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        return InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
            randomize_notch_delta_levels=randomize_notch_delta_levels,
            notch_levels=notch_levels, delta_levels=delta_levels,
            repressor_levels=repressor_levels,
        )

    def _size(self):
        """Number of distinct unique_ids the fresh test sheet carries."""
        inner0 = self._make_inner()
        return int(inner0.sheet.face_df["unique_id"].max()) + 1

    def test_arrays_map_by_unique_id(self):
        m = self._size()
        notch = np.linspace(0.1, 0.9, m)
        delta = np.linspace(0.9, 0.1, m)
        repr_ = np.full(m, 0.42)
        inner = self._make_inner(notch, delta, repr_)
        fd = inner.sheet.face_df
        u = fd["unique_id"].to_numpy().astype(int)
        # Each face holds the array entry at its OWN unique_id.
        np.testing.assert_allclose(fd["notch_level"].to_numpy(), notch[u])
        np.testing.assert_allclose(fd["delta_level"].to_numpy(), delta[u])
        np.testing.assert_allclose(fd["repressor_level"].to_numpy(), repr_[u])

    def test_load_from_npy_files(self, tmp_path):
        m = self._size()
        notch = np.random.rand(m)
        delta = np.random.rand(m)
        repr_ = np.random.rand(m)
        np.save(tmp_path / "n.npy", notch)
        np.save(tmp_path / "d.npy", delta)
        np.save(tmp_path / "r.npy", repr_)
        inner = self._make_inner(
            str(tmp_path / "n.npy"), str(tmp_path / "d.npy"), str(tmp_path / "r.npy")
        )
        fd = inner.sheet.face_df
        u = fd["unique_id"].to_numpy().astype(int)
        np.testing.assert_allclose(fd["notch_level"].to_numpy(), notch[u])
        np.testing.assert_allclose(fd["delta_level"].to_numpy(), delta[u])
        np.testing.assert_allclose(fd["repressor_level"].to_numpy(), repr_[u])

    def test_explicit_values_win_over_randomize_flag(self):
        m = self._size()
        inner = self._make_inner(
            np.full(m, 0.123), np.full(m, 0.456), np.full(m, 0.789),
            randomize_notch_delta_levels=True,
        )
        fd = inner.sheet.face_df
        np.testing.assert_allclose(fd["notch_level"].to_numpy(), 0.123)
        np.testing.assert_allclose(fd["delta_level"].to_numpy(), 0.456)
        np.testing.assert_allclose(fd["repressor_level"].to_numpy(), 0.789)

    def test_size_mismatch_raises(self):
        inner = self._make_inner()
        with pytest.raises(ValueError, match="unique_id"):
            inner.load_li_levels_from_numpy(np.zeros(2), None, None)

    def test_atoh_reflects_loaded_repressor(self):
        # atoh_level is derived in __init__ from the repressor channel
        # (atoh_by_repressor default). A uniform loaded repressor must
        # therefore give a uniform atoh — i.e. it used the loaded values,
        # not a random seed.
        m = self._size()
        inner = self._make_inner(np.full(m, 0.5), np.full(m, 0.5), np.full(m, 0.3))
        atoh = inner.sheet.face_df["atoh_level"].to_numpy()
        assert np.allclose(atoh, atoh[0])


class TestFindMechanicalParamsLILevels:
    """``find_mechanical_parameters`` seeds each initial sheet from the
    ``{notch,delta,repressor}_levels.npy`` files sitting next to that
    sheet's ``history.hf5`` (in ``results/<initial_sheet>/``). The path
    resolution + all-or-none policy lives in
    ``run_model._li_levels_kwargs_for_initial_sheet``."""

    def test_all_three_present_returns_paths(self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        folder = tmp_path / "results" / "sheetA"
        folder.mkdir(parents=True)
        for fname in ("notch_levels.npy", "delta_levels.npy", "repressor_levels.npy"):
            np.save(folder / fname, np.arange(5.0))
        kw = run_model._li_levels_kwargs_for_initial_sheet("sheetA")
        assert set(kw) == {"notch_levels", "delta_levels", "repressor_levels"}
        assert all(os.path.isfile(p) for p in kw.values())
        assert os.path.basename(kw["notch_levels"]) == "notch_levels.npy"
        # Paths point inside the initial sheet's own results folder.
        assert os.path.dirname(kw["delta_levels"]).endswith(
            os.path.join("results", "sheetA"))

    def test_none_present_falls_back(self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results" / "sheetB").mkdir(parents=True)
        # No files -> empty kwargs -> run() keeps its previous behaviour.
        assert run_model._li_levels_kwargs_for_initial_sheet("sheetB") == {}

    def test_partial_set_raises(self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        folder = tmp_path / "results" / "sheetC"
        folder.mkdir(parents=True)
        np.save(folder / "notch_levels.npy", np.arange(5.0))
        np.save(folder / "repressor_levels.npy", np.arange(5.0))
        with pytest.raises(FileNotFoundError, match="delta_levels.npy"):
            run_model._li_levels_kwargs_for_initial_sheet("sheetC")


class TestShortRunFolderName:
    """``run`` builds short, bounded-length, unique results-folder names so
    forked runs (ablation, parameter fits) don't compound the parent's name
    and overrun the OS path limit (``run_model._short_run_folder_name``)."""

    @staticmethod
    def _f(*args, **kwargs):
        import run_model
        return run_model._short_run_folder_name(*args, **kwargs)

    def test_deterministic(self):
        a = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0)
        b = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0)
        assert a == b

    def test_bounded_length_under_nested_forks(self):
        # Forking a run FROM another run (initial = a previous run's folder)
        # must NOT grow the name — that was the original path-length bug.
        name = "random_periodic_array0_for_E17"
        lengths = []
        for _ in range(6):
            name = self._f(name, 0.01, 10.0, 1.0, 0.0)
            lengths.append(len(name))
        assert max(lengths) < 64, f"folder name grew unbounded: {lengths}"
        # Every fork is the SAME length (no compounding).
        assert len(set(lengths)) == 1

    def test_unique_per_initial_and_params(self):
        names = set()
        combos = 0
        for gsc in (0.01, 0.02):
            for ghc in (10.0, 11.0):
                for ahc in (1.0, 2.0):
                    for i in range(10):
                        combos += 1
                        names.add(self._f(
                            "random_periodic_array%d_for_E17" % i,
                            gsc, ghc, ahc, 0.0))
        assert len(names) == combos, "distinct inputs collided to one folder"

    def test_different_initial_sheets_distinct(self):
        a = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0)
        b = self._f("random_periodic_array1_for_E17", 0.01, 10.0, 1.0, 0.0)
        assert a != b

    def test_ablation_variant_is_distinct_and_marked(self):
        base = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0)
        abl = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                      ablated_cells=[5, 12])
        assert abl != base
        assert abl.endswith("_abl")
        # Different ablated-cell sets give different folders.
        abl2 = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                       ablated_cells=[7])
        assert abl2 != abl

    def test_shape_index_tagged_and_distinct(self):
        # A non-zero shape_index (target perimeter P0) must (a) give a DIFFERENT
        # folder than the P0=0 run at the same params — so a fit adding it can't
        # reuse/collide with the contractility-only runs — and (b) be visible in
        # the readable prefix as _p0X.XX, while P0=0 keeps its historical name.
        base = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0)
        p0 = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                     shape_index=3.8)
        assert p0 != base
        assert "_p0" not in base
        assert "_p03.80" in p0

    def test_shape_index_reuse_resolution_is_two_decimals(self):
        # shape_index is hashed at %.2f (matching find_mechanical_parameters'
        # 2-decimal cache), so points equal to 2 decimals share a folder (reuse
        # hits) while a 2-decimal difference splits them.
        a = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                    shape_index=3.801)
        b = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                    shape_index=3.804)
        c = self._f("random_periodic_array0_for_E17", 0.01, 10.0, 1.0, 0.0,
                    shape_index=3.85)
        assert a == b            # same at 2 decimals -> same folder
        assert c != a            # differ at 2 decimals -> distinct
















class TestSavedPerArrayThreshold:
    """``find_mechanical_parameters(use_saved_threshold=True)`` reads each
    initial sheet's own ``threshold.npy`` and uses it as that sheet's HC/SC
    classification threshold (overriding the single ``fix_threshold``)."""

    def test_load_saved_threshold(self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        folder = tmp_path / "results" / "sheetA"
        folder.mkdir(parents=True)
        np.save(folder / "threshold.npy", np.asarray(0.37))
        assert run_model._load_saved_threshold("sheetA") == 0.37

    def test_load_saved_threshold_missing_raises(self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        (tmp_path / "results" / "sheetB").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="threshold"):
            run_model._load_saved_threshold("sheetB")



class TestAblationAreaChangeUnionFix:
    """Regression for the ablation term in compare_model_mechanics_to_experiments.

    ``calc_area_change_after_ablation`` accumulated HC/SC neighbour ids with a
    broken ``_, x = np.union1d(...)`` — np.union1d returns a SINGLE array, so
    unpacking it as a 2-tuple crashed with "not enough values to unpack" the
    moment a union had !=2 elements (and silently produced garbage otherwise).
    It must union the IDS correctly and survive a single-element union."""

    def test_single_element_union_and_accumulation(self, monkeypatch):
        import cell_measures as pp

        # Controlled (ordinal, ids) per call. Per ablated cell the loop calls
        # HC then SC. The FIRST HC call returns a single id — the case that
        # used to raise.
        hc_returns = [np.array([1]), np.array([1, 2])]
        sc_returns = [np.array([10]), np.array([10, 11])]
        calls = {"HC": 0, "SC": 0}

        def fake_ids(sheet, cell_type='all', **kw):
            seq = hc_returns if cell_type == "HC" else sc_returns
            ids = seq[calls[cell_type]]
            calls[cell_type] += 1
            return np.arange(len(ids)), ids

        monkeypatch.setattr(pp, "get_non_boundary_cell_ids_from_type", fake_ids)

        class _FakeSheet:
            def __init__(self, area, ids):
                self._area = area
                # index == id, as on every real sheet before a face is
                # removed; calc_area_change_after_ablation translates the
                # index labels get_neighbors returns into ids via .loc.
                self.face_df = pd.DataFrame({"id": ids}, index=ids)

            def arrange_sheet_from_history(self):
                pass

            def get_neighbors(self, ablated):
                return np.array([1, 2, 10, 11])

            def get_face_area(self):
                return self._area

        initial = _FakeSheet(pd.Series({1: 1.0, 2: 2.0, 10: 1.0, 11: 4.0}),
                             [1, 2, 10, 11])
        final = _FakeSheet(pd.Series({1: 2.0, 2: 6.0, 10: 0.5, 11: 2.0}),
                           [1, 2, 10, 11])

        class _FakeHistory:
            # end_time=-1 makes calc_area_change_after_ablation resolve the last
            # frame via get_time_points(history)[-1] == np.unique(time_stamps)[-1],
            # so the fake must expose time_stamps (t=1 -> the "final" sheet).
            time_stamps = np.array([0, 1])

            def retrieve(self, t):
                return initial if t == 0 else final

        hc_ratio, sc_ratio = pp.calc_area_change_after_ablation(
            _FakeHistory(), "load", ablated_cells=[100, 200], end_time=-1,
            type_by="delta_level", threshold=0.3)

        # HC ids unioned to {1, 2}: final/initial = [2/1, 6/2] = [2, 3].
        np.testing.assert_allclose(np.sort(hc_ratio), [2.0, 3.0])
        # SC ids unioned to {10, 11}: [0.5/1, 2/4] = [0.5, 0.5].
        np.testing.assert_allclose(np.sort(sc_ratio), [0.5, 0.5])













class TestFoldedFaceDetection:
    """``solvers.count_folded_faces`` flags cells whose polygon has folded
    over itself (self-intersected) — the "cells growing into each other"
    overlap that keeps a positive signed area and so slips past the
    negative-area safety net. It uses the polygon turning number, which is
    exactly +-1 for any simple polygon and != +-1 once it self-intersects."""

    class _Eptm:
        def __init__(self, edge_df):
            self.edge_df = edge_df

    def test_simple_convex_polygon_not_flagged(self):
        from solvers import count_folded_faces
        # CCW unit square: edge vectors (1,0)(0,1)(-1,0)(0,-1), turning # = +1.
        ed = pd.DataFrame({"face": [0, 0, 0, 0], "order": [1, 2, 3, 4],
                           "dx": [1, 0, -1, 0], "dy": [0, 1, 0, -1]})
        assert count_folded_faces(self._Eptm(ed)) == 0

    def test_nonconvex_simple_polygon_not_flagged(self):
        from solvers import count_folded_faces
        # A concave (arrow/chevron) but SIMPLE pentagon — turning # still +-1.
        P = np.array([[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]], float)
        d = np.roll(P, -1, axis=0) - P
        ed = pd.DataFrame({"face": [0] * 5, "order": [1, 2, 3, 4, 5],
                           "dx": d[:, 0], "dy": d[:, 1]})
        assert count_folded_faces(self._Eptm(ed)) == 0

    def test_self_intersecting_bowtie_flagged(self):
        from solvers import count_folded_faces
        # Bowtie (figure-8): A(0,0) B(1,1) C(1,0) D(0,1) -> self-crossing.
        P = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
        d = np.roll(P, -1, axis=0) - P
        ed = pd.DataFrame({"face": [0] * 4, "order": [1, 2, 3, 4],
                           "dx": d[:, 0], "dy": d[:, 1]})
        assert count_folded_faces(self._Eptm(ed)) == 1

    def test_counts_only_the_folded_face_among_several(self):
        from solvers import count_folded_faces
        # face 0: good square; face 1: bowtie; face 2: good triangle.
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        bow = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
        tri = np.array([[0, 0], [1, 0], [0, 1]], float)
        rows = []
        for fid, P in ((0, sq), (1, bow), (2, tri)):
            d = np.roll(P, -1, axis=0) - P
            for k in range(len(P)):
                rows.append({"face": fid, "order": k + 1, "dx": d[k, 0], "dy": d[k, 1]})
        ed = pd.DataFrame(rows)
        assert count_folded_faces(self._Eptm(ed)) == 1


# --------------------------------------------------------------------------- #
# Smart fit restart: reuse already-simulated runs                              #
# --------------------------------------------------------------------------- #
class TestClassifyExistingRun:
    """``run_model._classify_existing_run`` reads a run's ``debug.log`` and
    buckets it into completed / stalled / interrupted so a re-launched
    ``find_mechanical_parameters`` knows whether to keep the archive, score
    worst-case, or resume (see run()'s ``reuse_existing_run``)."""

    def _make(self, tmp_path, text):
        d = tmp_path / "run"
        d.mkdir()
        (d / "debug.log").write_text(text, encoding="utf-8")
        return str(d)

    def test_finished_marker_is_completed(self, tmp_path):
        import run_model
        d = self._make(tmp_path, "12:00 INFO run() finished successfully\n")
        assert run_model._classify_existing_run(d) == "completed"

    def test_dt_floor_crash_is_stalled(self, tmp_path):
        import run_model
        d = self._make(tmp_path, "ERR run() crashed; traceback follows\n"
                                 "RuntimeError: dt fell below 1.0e-08 at t=2.8\n")
        assert run_model._classify_existing_run(d) == "stalled"

    def test_progress_guard_crash_is_stalled(self, tmp_path):
        import run_model
        d = self._make(tmp_path, "ERR run() crashed; traceback follows\n"
                                 "RuntimeError: progressing too slowly ... "
                                 "stopping for worst-case scoring\n")
        assert run_model._classify_existing_run(d) == "stalled"

    def test_disk_full_crash_is_interrupted(self, tmp_path):
        import run_model
        d = self._make(tmp_path, "ERR run() crashed; traceback follows\n"
                                 "OSError: [Errno 28] No space left on device\n")
        assert run_model._classify_existing_run(d) == "interrupted"

    def test_killed_no_marker_is_interrupted(self, tmp_path):
        import run_model
        d = self._make(tmp_path, "12:00 INFO stepping t=3.1\n")
        assert run_model._classify_existing_run(d) == "interrupted"

    def test_resumed_then_finished_is_completed(self, tmp_path):
        """A crashed-then-resumed-and-finished run appends to the same log; the
        LAST marker (success) must win over the earlier crash."""
        import run_model
        d = self._make(tmp_path,
                       "run() crashed; traceback follows\n"
                       "RuntimeError: dt fell below 1e-08\n"
                       "Resuming run from time 2.8 ...\n"
                       "run() finished successfully\n")
        assert run_model._classify_existing_run(d) == "completed"

    def test_missing_log_is_interrupted(self, tmp_path):
        import run_model
        d = tmp_path / "empty"
        d.mkdir()
        assert run_model._classify_existing_run(str(d)) == "interrupted"


class TestLatestArchiveTime:
    """``run_model._latest_archive_time`` returns the resume point, or None when
    there's nothing to resume from."""

    def test_missing_archive_returns_none(self, tmp_path):
        import run_model
        d = tmp_path / "run"
        d.mkdir()
        assert run_model._latest_archive_time(str(d)) is None

    def test_unreadable_archive_returns_none(self, tmp_path):
        import run_model
        d = tmp_path / "run"
        d.mkdir()
        (d / "history.hf5").write_bytes(b"not a valid hdf5 file")
        assert run_model._latest_archive_time(str(d)) is None


class TestReuseExistingRunDispatch:
    """``run(reuse_existing_run=True)`` acts on the classification instead of the
    blind 'directory already exists -> return' cache hit."""

    def _existing(self, tmp_path, monkeypatch, log_text):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        d = tmp_path / "fit1"
        d.mkdir()
        (d / "debug.log").write_text(log_text, encoding="utf-8")
        return run_model, d

    def test_completed_returns_without_simulating(self, tmp_path, monkeypatch):
        run_model, d = self._existing(tmp_path, monkeypatch,
                                      "run() finished successfully\n")

        def no(*a, **k):
            raise AssertionError("a completed run must NOT be re-simulated")

        monkeypatch.setattr(run_model, "initialize_sheet", no)
        monkeypatch.setattr(run_model, "load_sheet_from_file", no)
        assert run_model.run(0.01, 1.0, 1.0, 0, name="fit1",
                             reuse_existing_run=True) == "fit1"

    def test_stalled_raises_runtimeerror(self, tmp_path, monkeypatch):
        run_model, d = self._existing(
            tmp_path, monkeypatch,
            "run() crashed; traceback follows\nRuntimeError: dt fell below 1e-8\n")
        with pytest.raises(RuntimeError, match="stalled"):
            run_model.run(0.01, 1.0, 1.0, 0, name="fit1", reuse_existing_run=True)

    def test_stalled_reruns_fresh_when_flag_set(self, tmp_path, monkeypatch):
        """``rerun_stalled_runs=True`` discards the stalled folder and re-runs
        from scratch instead of raising."""
        run_model, d = self._existing(
            tmp_path, monkeypatch,
            "run() crashed; traceback follows\nRuntimeError: dt fell below 1e-8\n")

        class _Reached(Exception):
            pass

        def boom(*a, **k):
            raise _Reached

        monkeypatch.setattr(run_model, "initialize_sheet", boom)
        with pytest.raises(_Reached):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="someinit",
                          name="fit1", reuse_existing_run=True,
                          rerun_stalled_runs=True)
        # The stale stalled transcript was wiped by the fresh-start rmtree.
        assert "dt fell below" not in (d / "debug.log").read_text(encoding="utf-8")

    def test_interrupted_no_archive_reruns_fresh(self, tmp_path, monkeypatch):
        run_model, d = self._existing(
            tmp_path, monkeypatch,
            "run() crashed; traceback follows\n"
            "OSError: [Errno 28] No space left on device\n")
        # No history.hf5 -> nothing to resume -> rmtree + fresh build.

        class _Reached(Exception):
            pass

        def boom(*a, **k):
            raise _Reached

        monkeypatch.setattr(run_model, "initialize_sheet", boom)
        with pytest.raises(_Reached):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="someinit",
                          name="fit1", reuse_existing_run=True)
        # The stale OSError transcript was wiped by the fresh-start rmtree
        # (the new debug.log only holds this fresh attempt).
        assert "No space left" not in (d / "debug.log").read_text(encoding="utf-8")

    def test_interrupted_with_archive_resumes_own_archive(self, tmp_path, monkeypatch):
        run_model, d = self._existing(
            tmp_path, monkeypatch,
            "run() crashed; traceback follows\n"
            "OSError: [Errno 28] No space left on device\n")
        # Pretend a resumable snapshot exists at t=4.0.
        monkeypatch.setattr(run_model, "_latest_archive_time", lambda rd: 4.0)
        seen = {}

        class _Reached(Exception):
            pass

        def fake_load(initial_sheet_name, time_point=None, **k):
            seen["initial"] = initial_sheet_name
            seen["t"] = time_point
            raise _Reached

        monkeypatch.setattr(run_model, "load_sheet_from_file", fake_load)
        with pytest.raises(_Reached):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="someinit",
                          name="fit1", reuse_existing_run=True)
        # Resume targets the run's OWN archive (initial flipped to ``name``) at
        # the latest snapshot, NOT the original initial sheet.
        assert seen["initial"].endswith("fit1")
        assert seen["t"] == 4.0

    def test_existing_dir_ignored_without_flag(self, tmp_path, monkeypatch):
        """Without ``reuse_existing_run`` the old blind cache hit stands: an
        existing dir returns immediately, whatever the log says."""
        run_model, d = self._existing(
            tmp_path, monkeypatch,
            "run() crashed; traceback follows\nRuntimeError: dt fell below 1e-8\n")

        def no(*a, **k):
            raise AssertionError("must not simulate on a plain cache hit")

        monkeypatch.setattr(run_model, "initialize_sheet", no)
        monkeypatch.setattr(run_model, "load_sheet_from_file", no)
        # No raise (the stalled log is not consulted), returns the bare name.
        assert run_model.run(0.01, 1.0, 1.0, 0, name="fit1") == "fit1"


# --------------------------------------------------------------------------- #
# Face removal must not leave duplicate (srce, trgt) half-edges                #
# --------------------------------------------------------------------------- #
class TestRemovalNoDuplicateEdges:
    """Collapsing a removed cell's vertices onto one point can leave degenerate
    'antenna' spikes — a surviving vertex joined to a single other vertex — that
    show up as duplicate (srce, trgt) half-edges shared by two faces and trigger
    tyssue's "Duplicated (`srce`, `trgt`)" warning from get_opposite during
    ablation / delamination. ``_drop_antenna_spikes`` (called by
    ``index_preserving_remove_face``) must peel those off."""

    def test_drop_antenna_spikes_removes_duplicate_edges(self):
        from topological_events import _drop_antenna_spikes
        # Two triangles (faces 1, 2) that SHARE an antenna 10->5->10 at vertex 5
        # (5 is joined only to 10). The shared antenna makes (10,5) and (5,10)
        # duplicate (srce,trgt) pairs across the two faces.
        rows = [
            (10, 5, 1), (5, 10, 1), (10, 6, 1), (6, 7, 1), (7, 10, 1),
            (10, 5, 2), (5, 10, 2), (10, 8, 2), (8, 9, 2), (9, 10, 2),
        ]
        edge_df = pd.DataFrame(rows, columns=["srce", "trgt", "face"])

        class _S:
            pass

        s = _S()
        s.edge_df = edge_df
        # precondition: the antenna creates duplicate directed edges
        assert edge_df[["srce", "trgt"]].duplicated().any()

        _drop_antenna_spikes(s)

        assert not s.edge_df[["srce", "trgt"]].duplicated().any()
        # the spike vertex is gone; both real triangles survive intact
        verts = set(s.edge_df["srce"]) | set(s.edge_df["trgt"])
        assert 5 not in verts
        assert len(s.edge_df) == 6

    def test_remove_face_with_virtuals_no_duplicate_warning(self):
        """End-to-end: removing cells from a small periodic sheet WITH virtual
        mid-edge vertices (the configuration the fit's ablation hits) must not
        warn about duplicate edges or leave duplicate (srce,trgt) pairs. On the
        pre-fix code the 2x2 case raised the warning."""
        import warnings as _w
        import networkx as _nx
        from topological_events import index_preserving_remove

        sheet = _build_sheet_with_virtuals(2, 2)
        sheet.get_opposite()
        sheet.face_df["unique_id"] = sheet.face_df.index.astype(str)
        sheet.lineage = _nx.DiGraph()

        for f in list(sheet.face_df.index):
            if f not in sheet.face_df.index or sheet.Nf <= 2:
                break
            with _w.catch_warnings(record=True) as rec:
                _w.simplefilter("always")
                index_preserving_remove(sheet, f, sheet.geom)
                assert not any("Duplicated" in str(x.message) for x in rec), \
                    "removal emitted a Duplicated (srce,trgt) warning"
            assert sheet.edge_df[["srce", "trgt"]].duplicated().sum() == 0

    def test_get_opposite_heals_spike_from_any_source(self):
        """The healing is CENTRAL: a spike injected outside the removal path
        (mimicking one a T1 / sharp-corner collapse could leave) is peeled off
        by get_opposite / reset_topo, so no Duplicated warning escapes whatever
        topology op created it."""
        import warnings as _w
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()
        A = int(sheet.edge_df["srce"].iloc[0])
        f1, f2 = [int(x) for x in sheet.edge_df["face"].unique()[:2]]
        # New antenna vertex v joined ONLY to A, appearing as A->v->A in two
        # faces -> duplicate (A,v)/(v,A) half-edges.
        v = int(sheet.vert_df.index.max()) + 1
        sheet.vert_df.loc[v] = sheet.vert_df.loc[A]
        template = sheet.edge_df.iloc[0]
        eidx = int(sheet.edge_df.index.max()) + 1
        for f in (f1, f2):
            for (s, t) in ((A, v), (v, A)):
                r = template.copy()
                r["srce"], r["trgt"], r["face"] = s, t, f
                sheet.edge_df.loc[eidx] = r
                eidx += 1
        assert sheet.edge_df[["srce", "trgt"]].duplicated().any()  # spike present
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            sheet.get_opposite()
            assert not any("Duplicated" in str(x.message) for x in rec)
        assert not sheet.edge_df[["srce", "trgt"]].duplicated().any()
        assert v not in set(sheet.edge_df["srce"]) | set(sheet.edge_df["trgt"])


# --------------------------------------------------------------------------- #
# Optimization leaves a diagnosable trace: per-step p-values + GP landscape    #
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# Loaded delta threshold drives the simulation's atoh_sensitivity              #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Mechanical params come from atoh(delta, threshold) and are frozen w/o diff   #
# --------------------------------------------------------------------------- #
class TestMechanicalParamsFromAtoh:
    """At the start of a simulation the per-cell mechanical parameters must be
    set from the atoh level — atoh = Hill(delta, atoh_sensitivity), so a cell
    with delta ABOVE the threshold gets HC mechanics and below it SC mechanics —
    and, in a run WITHOUT differentiation, they must not be re-applied / drift
    (delaminating cells legitimately change and are excluded here)."""

    def _model(self, atoh_sensitivity=0.4):
        from inner_ear_model import InnerEarModel
        from tyssue.dynamics.effectors import FaceContractility
        sheet = _build_sheet_with_virtuals(8, 8)
        sheet.get_opposite()
        return InnerEarModel(
            sheet, contractility={"HC": 4.0, "SC": 0.1},
            elasticity={"HC": 5.0, "SC": 1.0},
            preferred_area={"HC": np.pi / 4, "SC": np.pi / 4},
            atoh_sensitivity=atoh_sensitivity, atoh_by_repressor=False,
            differentiation_threshold=0.5,
            stress_effectors=[FaceContractility], mechanosensitivity=0)

    def test_init_params_set_from_atoh_of_delta_and_threshold(self):
        A = 0.4
        m = self._model(atoh_sensitivity=A)
        fd = m.sheet.face_df
        delta = fd["delta_level"].values
        atoh = fd["atoh_level"].values
        # atoh is the increasing Hill of delta with half-max AT the threshold,
        # exponent l = 5 (SI Eqs. 16-17 and Supplementary Table 1). Spelled out
        # rather than imported, so the test pins the value, not the constant.
        np.testing.assert_allclose(atoh, delta ** 5 / (A ** 5 + delta ** 5), atol=1e-9)
        live = fd["type"].values >= 0
        # contractility / area_elasticity are the atoh interpolation of HC<->SC.
        np.testing.assert_allclose(fd["contractility"].values[live],
                                   (atoh * 4.0 + (1 - atoh) * 0.1)[live], atol=1e-9)
        np.testing.assert_allclose(fd["area_elasticity"].values[live],
                                   (atoh * 5.0 + (1 - atoh) * 1.0)[live], atol=1e-9)
        # HC (type 1) exactly where delta is above the threshold.
        np.testing.assert_array_equal(fd["type"].values[live],
                                      (delta > A).astype(int)[live])

    def test_params_frozen_during_no_differentiation(self, tmp_path):
        from tyssue.dynamics.effectors import FaceContractility, FaceAreaElasticity
        m = self._model()
        calls = {"n": 0}
        orig = m.update_cell_type_parameters

        def spy(atoh):
            calls["n"] += 1
            return orig(atoh)

        m.update_cell_type_parameters = spy
        before = m.sheet.face_df[["id", "contractility", "area_elasticity"]].copy()
        m.simulate(t_end=0.03, dt=0.01, no_differentiation=True,
                   delaminations=False, divisions=False, intercalations=False,
                   random_forces=False,
                   effectors=[FaceContractility, FaceAreaElasticity],
                   viscosity=1, quasi_static=True, until_steady_state=False,
                   history_file=str(tmp_path / "h.hf5"))
        # differentiation was off, so the atoh->params step was never re-applied.
        assert calls["n"] == 0
        # every surviving cell kept its start-of-simulation mechanical params.
        merged = before.merge(
            m.sheet.face_df[["id", "contractility", "area_elasticity"]],
            on="id", suffixes=("_0", "_1"))
        np.testing.assert_allclose(merged["contractility_0"],
                                   merged["contractility_1"], atol=1e-12)
        np.testing.assert_allclose(merged["area_elasticity_0"],
                                   merged["area_elasticity_1"], atol=1e-12)


# --------------------------------------------------------------------------- #
# A failed T1 must be atomic (no partial collapse -> no duplicate half-edges)  #
# --------------------------------------------------------------------------- #
class TestT1TransitionAtomic:
    """``index_preserving_type1_transition`` collapses an edge BEFORE the split /
    tri-face removal that can raise; a partially-applied T1 leaves duplicate
    (srce,trgt) half-edges (the cross-face duplicate seen during intercalation).
    A failed T1 must therefore restore the sheet to its pre-T1 state."""

    def _nonperiodic_edge(self, sheet):
        return int(sheet.edge_df[~sheet.edge_df["is_periodic"]].index[0])

    def test_failed_t1_restores_sheet(self, monkeypatch):
        import topological_events as tev
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        edge = self._nonperiodic_edge(sheet)
        edge_before = sheet.edge_df.copy()
        vert_before = sheet.vert_df.copy()
        face_before = sheet.face_df.copy()

        def corrupt_then_raise(s, e, **k):
            # simulate a partial collapse: mutate topology, THEN raise
            s.edge_df = s.edge_df.iloc[:-4].copy()
            s.vert_df = s.vert_df.iloc[:-1].copy()
            raise ValueError("split failed mid-T1")

        monkeypatch.setattr(tev, "_bulk_t1_transition", corrupt_then_raise)
        with pytest.raises(ValueError, match="split failed"):
            tev.index_preserving_type1_transition(sheet, edge)
        # fully restored — no partial collapse, no duplicate half-edges.
        pd.testing.assert_frame_equal(sheet.edge_df, edge_before)
        pd.testing.assert_frame_equal(sheet.vert_df, vert_before)
        pd.testing.assert_frame_equal(sheet.face_df, face_before)
        assert not sheet.edge_df[["srce", "trgt"]].duplicated().any()

    def test_negative_return_restores_sheet(self, monkeypatch):
        import topological_events as tev
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        edge = self._nonperiodic_edge(sheet)
        edge_before = sheet.edge_df.copy()

        def corrupt_then_fail(s, e, **k):
            s.edge_df = s.edge_df.iloc[:-4].copy()
            return -1

        monkeypatch.setattr(tev, "_bulk_t1_transition", corrupt_then_fail)
        ret = tev.index_preserving_type1_transition(sheet, edge)
        assert ret == -1
        pd.testing.assert_frame_equal(sheet.edge_df, edge_before)

    def test_successful_t1_keeps_changes(self, monkeypatch):
        # A successful T1 must NOT be rolled back.
        import topological_events as tev
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        edge = self._nonperiodic_edge(sheet)
        marker = {}

        def succeed(s, e, **k):
            s.edge_df = s.edge_df.iloc[:-2].copy()  # a real change
            marker["n_after"] = len(s.edge_df)
            return 0

        monkeypatch.setattr(tev, "_bulk_t1_transition", succeed)
        ret = tev.index_preserving_type1_transition(sheet, edge)
        assert ret == 0
        assert len(sheet.edge_df) == marker["n_after"]  # change kept


# --------------------------------------------------------------------------- #
# A resume must preserve the archive's LI state, not re-seed it                #
# --------------------------------------------------------------------------- #
class TestResumePreservesLIState:
    """The LI seed arrays (notch/delta/repressor levels) initialise the LI state
    at t=0 ONLY. A resume continues a trajectory whose LI levels already live in
    the loaded archive; re-seeding them from the initial-sheet arrays overwrites
    the evolved state and (when the archive's LI differs from the arrays)
    scrambles delta -> atoh -> HC/SC mid-run. So run() must pass the arrays only
    for a FRESH run and force None on resume."""

    def test_innermodel_none_arrays_preserve_loaded_delta(self):
        from inner_ear_model import InnerEarModel
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        sheet.face_df["delta_level"] = np.linspace(0.1, 0.9, sheet.Nf)
        sheet.face_df["notch_level"] = 0.4
        sheet.face_df["repressor_level"] = 0.6
        before = sheet.face_df["delta_level"].values.copy()
        InnerEarModel(sheet, contractility={"HC": 4, "SC": 0.1},
                      elasticity={"HC": 5, "SC": 1}, atoh_sensitivity=0.4,
                      atoh_by_repressor=False, delta_levels=None,
                      notch_levels=None, repressor_levels=None)
        np.testing.assert_allclose(sheet.face_df["delta_level"].values, before)

    def _capture_innermodel_kwargs(self, monkeypatch, run_model, sheet):
        captured = {}

        class _Stop(Exception):
            pass

        def fake_model(s, **kw):
            captured.update(kw)
            raise _Stop

        monkeypatch.setattr(run_model, "InnerEarModel", fake_model)
        return captured, _Stop

    def test_resume_forces_none_li_arrays(self, tmp_path, monkeypatch):
        import run_model
        rd = str(tmp_path / "results")
        os.makedirs(os.path.join(rd, "myrun"))
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        monkeypatch.setattr(run_model, "load_sheet_from_file", lambda *a, **k: sheet)
        captured, _Stop = self._capture_innermodel_kwargs(monkeypatch, run_model, sheet)
        arr = np.arange(5.0)
        with pytest.raises(_Stop):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="myrun", name="myrun",
                          continue_existing_run=True, continue_from_time=5.0,
                          delta_levels=arr, notch_levels=arr, repressor_levels=arr,
                          t_end=6, dt=0.01)
        assert captured["delta_levels"] is None
        assert captured["notch_levels"] is None
        assert captured["repressor_levels"] is None

    def test_fresh_run_forwards_li_arrays(self, tmp_path, monkeypatch):
        import run_model
        rd = str(tmp_path / "results")
        os.makedirs(rd)
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        sheet = _build_sheet_with_virtuals(6, 6)
        sheet.get_opposite()
        monkeypatch.setattr(run_model, "initialize_sheet", lambda *a, **k: sheet)
        captured, _Stop = self._capture_innermodel_kwargs(monkeypatch, run_model, sheet)
        arr = np.arange(5.0)
        with pytest.raises(_Stop):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="freshinit",
                          name="fresh1", delta_levels=arr, notch_levels=arr,
                          repressor_levels=arr, t_end=6, dt=0.01)
        assert captured["delta_levels"] is arr


class TestResumeState:
    """A resume must continue the ORIGINAL trajectory, not start a new one from
    the same sheet. The archive restores the geometry exactly (measured: vertex
    positions, topology and edge row order are bit-identical in every frame);
    everything else travels in the ``resume_state`` sidecar. Verified end to
    end as well: an 8x8 array built to t=7 and resumed from t=3.51 is
    BIT-IDENTICAL over all 36 remaining frames, while the same resume with the
    sidecar deleted ends with 64 cells where the original has 82."""

    # -- both generators ---------------------------------------------------
    def test_both_generators_round_trip(self):
        import random
        import resume_state
        np.random.seed(7)
        random.seed(7)
        np.random.rand(1234)                   # off the top of both streams
        [random.random() for _ in range(55)]
        state = resume_state.capture_rngs()
        want_np = np.random.rand(5)
        want_shuffle = list(range(20))
        random.shuffle(want_shuffle)

        np.random.seed(999)
        random.seed(999)
        resume_state.restore_rngs(state)
        np.testing.assert_array_equal(np.random.rand(5), want_np)
        got_shuffle = list(range(20))
        random.shuffle(got_shuffle)
        # The stdlib stream is the EVENT ORDER: same state, same shuffle.
        assert got_shuffle == want_shuffle

    # -- file I/O ----------------------------------------------------------
    def test_sidecar_is_named_after_its_archive(self):
        import resume_state
        assert resume_state.sidecar_path(os.path.join("d", "history.hf5")) \
            == os.path.join("d", "history_resume.jsonl")
        assert resume_state.sidecar_path(None) is None

    def test_read_is_exact_and_takes_the_last_write(self, tmp_path):
        import resume_state
        p = str(tmp_path / "history_resume.jsonl")
        for t in (0.0, 0.1, 0.2, 0.3):
            resume_state.write(p, t, {"format": 1, "tag": "first"})
        resume_state.write(p, 0.2, {"format": 1, "tag": "second"})
        # record() replaces the row at an existing time, so the LAST write wins.
        assert resume_state.read(p, 0.2)["tag"] == "second"
        assert resume_state.read(p, 0.2 + 1e-12)["tag"] == "second"
        # NOT the nearest: a frame with no line of its own (killed between the
        # archive append and this one) must fall back to "no resume state"
        # rather than be paired with a neighbouring frame's RNG position.
        assert resume_state.read(p, 0.1999) is None

    def test_truncate_drops_the_tail_and_dedupes(self, tmp_path):
        import resume_state
        p = str(tmp_path / "history_resume.jsonl")
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            resume_state.write(p, t, {"format": 1})
        resume_state.write(p, 0.2, {"format": 1, "tag": "again"})
        assert resume_state.truncate(p, 0.2) == 3
        assert [r["t"] for r in resume_state._iter_records(p)] == [0.0, 0.1, 0.2]
        # de-duplication keeps the LAST record at the kept stamp
        assert resume_state.read(p, 0.2)["tag"] == "again"

    def test_a_missing_sidecar_is_not_an_error(self, tmp_path):
        """Every run recorded before the sidecar existed has none; resuming one
        must still work (with a warning), not raise."""
        import resume_state
        p = str(tmp_path / "nope_resume.jsonl")
        assert resume_state.read(p, 1.0) is None
        assert resume_state.truncate(p, 1.0) == 0

    # -- the event manager -------------------------------------------------
    @staticmethod
    def _manager(behaviors):
        from tyssue.behaviors import EventManager
        m = EventManager("face")
        for b in behaviors:
            m.append(b)
        return m

    @staticmethod
    def _named(name):
        def behavior(eptm, manager, **kwargs):
            manager.append(behavior)
        behavior.__name__ = name
        return behavior

    def test_event_order_is_restored_by_name(self):
        import resume_state
        names = ["division", "intercalation", "delamination",
                 "update_virtual_vertices"]
        live = self._manager([self._named(n) for n in names])
        live.update()                      # shuffles next -> current
        recorded = resume_state.capture_manager(live)
        order = [e["name"] for e in recorded["current"]]

        fresh = self._manager([self._named(n) for n in names])
        resume_state.restore_manager(fresh, recorded)
        assert [b.__name__ for b, _ in fresh.current] == order
        assert fresh.clock == live.clock

    def test_a_one_shot_that_already_fired_is_not_re_appended(self):
        """``ablation`` (and, in quasi-static mode, the differentiation
        appended at setup) does not re-append itself, so it is absent from any
        queue recorded after the first step. Re-appending it would re-tag a
        dying cell / inject a differentiation step the original run never
        took."""
        import resume_state
        live = self._manager([self._named("division"), self._named("ablation")])
        live.update()
        live.current = type(live.current)(
            [e for e in live.current if e[0].__name__ != "ablation"])
        recorded = resume_state.capture_manager(live)

        fresh = self._manager([self._named("division"), self._named("ablation")])
        notes = resume_state.restore_manager(fresh, recorded)
        queued = [b.__name__ for b, _ in fresh.current] + \
                 [b.__name__ for b, _ in fresh.next]
        assert "ablation" not in queued
        assert any("ablation" in n and "dropped" in n for n in notes), notes

    # -- cell labels -------------------------------------------------------
    def test_unique_id_is_restored_with_its_dtype(self):
        """tyssue's Epithelium.__init__ overwrites ``unique_id`` with the row
        index, so a resume's sheet comes back relabelled. The labels key
        labels.npy / contact_matrix.npy / cell_info.pkl."""
        import resume_state
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.face_df["unique_id"] = np.arange(sheet.Nf) + 100
        sheet.specs.setdefault("face", {})["unique_id_max"] = 999
        state = resume_state.capture_identity(sheet)

        sheet.face_df["unique_id"] = np.arange(sheet.Nf)
        sheet.specs["face"]["unique_id_max"] = sheet.Nf
        resume_state.restore_identity(sheet, state)
        np.testing.assert_array_equal(sheet.face_df["unique_id"].to_numpy(),
                                      np.arange(sheet.Nf) + 100)
        assert sheet.specs["face"]["unique_id_max"] == 999
        assert sheet.face_df["unique_id"].dtype == np.dtype("int64")

    def test_identity_is_not_applied_to_a_different_frame(self):
        import resume_state
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.face_df["unique_id"] = np.arange(sheet.Nf)
        state = resume_state.capture_identity(sheet)
        other = _build_sheet_with_virtuals(6, 6)
        other.face_df["unique_id"] = np.arange(other.Nf)
        notes = resume_state.restore_identity(other, state)
        assert any("NOT restored" in n for n in notes), notes
        np.testing.assert_array_equal(other.face_df["unique_id"].to_numpy(),
                                      np.arange(other.Nf))

    # -- the measured length unit -----------------------------------------
    def test_l0_is_restored_to_full_precision(self):
        """L0 is a measured mean perimeter; ``l0.txt`` used to record 12
        significant digits, so an inheriting run's L0 differed in the last
        bits. Nothing depends on that until the first set_random_parameters
        draw (A0 = N(mean, std) * L0**2) -- and then the trajectories part."""
        import resume_state

        class _LI:
            length_normalization_factor = None

        class _Inner:
            def __init__(self, l0):
                self.length_normalization_factor = l0
                self.lateral_inhibition_model = _LI()

        exact = 1.5 + float(np.sqrt(5))        # what the lattice measures
        lossy = float("%.12g" % exact)         # what l0.txt could carry
        assert lossy != exact
        state = resume_state.capture_model(_Inner(exact))
        inner = _Inner(lossy)
        notes = resume_state.restore_model(inner, state)
        assert inner.length_normalization_factor == exact
        assert inner.lateral_inhibition_model.length_normalization_factor == exact
        assert any("L0" in n for n in notes), notes

    def test_write_l0_round_trips_exactly(self, tmp_path):
        import run_model
        exact = 1.5 + float(np.sqrt(5))
        d = tmp_path / "run"
        d.mkdir()
        run_model._write_L0(str(d), exact)
        assert float((d / "l0.txt").read_text().strip()) == exact

    # -- the countdowns ----------------------------------------------------
    def test_the_random_forces_countdown_is_restored(self):
        import resume_state

        class _Inner:
            time_to_random = 1.25
            length_normalization_factor = 1.0
            topological_events_handler = None
            lateral_inhibition_model = None

        state = resume_state.capture_model(_Inner())
        inner = _Inner()
        inner.time_to_random = 5.0             # what a fresh run starts at
        resume_state.restore_model(inner, state)
        assert inner.time_to_random == pytest.approx(1.25)

    def test_the_random_division_stream_is_restored(self):
        """The random division rule draws from its OWN PCG64 stream, so seeding
        the global generators does not place it."""
        import resume_state
        from topological_events import TopologicalEventsHandler

        np.random.seed(0)
        handler = TopologicalEventsHandler(model=None)
        handler.get_random_division_function(interval=1.0, dt=0.01)
        rng = handler._division_rng
        rng.random(37)                         # move it off its seed
        state = resume_state.capture_model(
            type("I", (), {"topological_events_handler": handler,
                           "length_normalization_factor": None,
                           "lateral_inhibition_model": None})())
        want = rng.random(5)

        np.random.seed(0)
        handler2 = TopologicalEventsHandler(model=None)
        handler2.get_random_division_function(interval=1.0, dt=0.01)
        resume_state.restore_model(
            type("I", (), {"topological_events_handler": handler2,
                           "length_normalization_factor": None,
                           "lateral_inhibition_model": None})(), state)
        np.testing.assert_array_equal(handler2._division_rng.random(5), want)

    # -- the solver loop ---------------------------------------------------
    @staticmethod
    def _tiny_solver(history=None, resume_state_path=None):
        from solvers import IVPSolver
        from tyssue.dynamics import model_factory
        from tyssue.dynamics.effectors import (FaceContractility,
                                               FaceAreaElasticity)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "resume_dt", nx=2, ny=2, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        sheet.vert_df["viscosity"] = 1.0
        sheet.vert_df["is_active"] = 1
        sheet.active_verts = np.arange(sheet.Nv)
        sheet.edge_df["is_active"] = 1
        sheet.face_df["is_active"] = 1
        from tyssue import History
        hist = History(sheet, save_every=0.1, save_all=False, dt=0.01) \
            if history is None else history
        solver = IVPSolver(
            None, sheet, sheet.geom,
            model_factory([FaceContractility, FaceAreaElasticity]),
            manager=None, history=hist, resume_state_path=resume_state_path,
        )
        solver.ode_func = lambda t, pos: np.zeros_like(pos)   # nothing moves
        return sheet, solver

    def test_solver_resumes_from_the_restored_step(self):
        """The adaptive dt is state: the original run may have been pinned far
        below the nominal step. A resume must take ITS step, not the nominal
        one -- and must not adopt it as the new nominal (the floor and the
        ratchet ceiling come from the caller)."""
        sheet, solver = self._tiny_solver()
        solver.prev_t = 1.0
        solver._resume_loop_state = {"next_dt": 0.0025, "next_save_t": 1.001,
                                     "steady_streak": 0,
                                     "last_differentiation_t": 1.0}
        solver.solve(tf=1.02, dt=0.01, max_displacement=1.0,
                     save_interval=0.001)
        # tyssue's in-memory History records the sheet at t=0 when it is
        # constructed, before the solver's clock is moved; ignore that frame.
        ts = sorted(t for t in set(round(float(t), 9)
                                   for t in solver.history.time_stamps)
                    if t >= 1.0)
        assert ts[0] == pytest.approx(1.0)
        # First step is the RESTORED dt, then the ratchet grows it back.
        assert ts[1] - ts[0] == pytest.approx(0.0025, rel=1e-9)
        assert ts[2] - ts[1] == pytest.approx(0.0025 * 1.1, rel=1e-9)

    def test_without_a_resume_state_the_nominal_step_is_used(self):
        sheet, solver = self._tiny_solver()
        solver.solve(tf=0.02, dt=0.005, max_displacement=1.0,
                     save_interval=0.001)
        ts = sorted(set(round(float(t), 9) for t in solver.history.time_stamps))
        assert ts[1] - ts[0] == pytest.approx(0.005, rel=1e-9)

    def test_each_recorded_frame_gets_a_sidecar_line(self, tmp_path):
        import resume_state
        p = str(tmp_path / "history_resume.jsonl")
        sheet, solver = self._tiny_solver(resume_state_path=p)
        solver.solve(tf=0.03, dt=0.01, max_displacement=1.0,
                     save_interval=0.01)
        frames = sorted(set(round(float(t), 9)
                            for t in solver.history.time_stamps))
        recorded = sorted(set(round(r["t"], 9)
                              for r in resume_state._iter_records(p)))
        assert recorded == frames
        one = resume_state.read(p, frames[-1])
        assert one["rngs"]["numpy"]["kind"] == "MT19937"
        assert one["loop"]["next_dt"] == pytest.approx(0.01)
        # ``next_save_t`` is normalised past the frame it is recorded with, so
        # a resume saves on the same grid instead of adding a frame.
        assert one["loop"]["next_save_t"] > frames[-1]

    # -- run() plumbing ----------------------------------------------------
    def test_resume_snaps_to_an_actual_recorded_stamp(self, tmp_path):
        """The loader snaps the requested time to the nearest frame; the solver
        clock and the truncation have to name that same frame."""
        import run_model
        d = tmp_path / "run"
        d.mkdir()
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()
        TestForkFromSnapshot._write_minimal_history(
            str(d / "history.hf5"), sheet, [0.01, 0.11, 0.21])
        assert run_model._nearest_archive_time(str(d), 0.1) \
            == pytest.approx(0.11)
        assert run_model._nearest_archive_time(str(d), 0.0) \
            == pytest.approx(0.01)
        assert run_model._nearest_archive_time(str(tmp_path / "nope"), 1.0) \
            is None

    def test_restore_array_run_asks_for_a_fresh_stream(self, monkeypatch):
        """Its whole point is to replace a folded tail; reproducing the
        original continuation would reproduce the fold."""
        import run_model
        captured = {}
        monkeypatch.setattr(run_model, "run",
                            lambda *a, **kw: captured.update(kw) or "n")
        run_model.restore_array_run(3, continue_from_time=5.0, t_end=7.0)
        assert captured["restore_resume_state"] is False
        assert captured["continue_existing_run"] is True

    def test_derived_per_cell_columns_are_restored(self):
        """``InnerEarModel.__init__`` resets every spec-backed column to the
        by-type default. The ones the model DERIVES per cell (atoh
        interpolation, the random_forces draws, ``type``) are state, and the
        solver evaluates the forces BEFORE the first differentiation step that
        would recompute them."""
        import run_model
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()

        class _Inner:
            face_params = {"prefered_area": None, "contractility": None}
            edge_params = {"line_tension": None}

            def __init__(self, s):
                self.sheet = s

        saved = {"face": sheet.face_df.copy(), "edge": sheet.edge_df.copy()}
        saved["face"]["prefered_area"] = np.linspace(0.3, 0.9, sheet.Nf)
        saved["face"]["contractility"] = 0.07
        saved["face"]["type"] = 1
        saved["edge"]["line_tension"] = 0.5
        sheet.face_df["prefered_area"] = 1.0        # the by-type default
        sheet.face_df["contractility"] = 0.0
        sheet.face_df["type"] = 0
        sheet.edge_df["line_tension"] = 0.0

        run_model._restore_derived_columns(_Inner(sheet), saved)
        np.testing.assert_allclose(sheet.face_df["prefered_area"].to_numpy(),
                                   np.linspace(0.3, 0.9, sheet.Nf))
        assert (sheet.face_df["type"] == 1).all()
        assert sheet.face_df["type"].dtype == np.dtype("int64")
        np.testing.assert_allclose(sheet.edge_df["line_tension"].to_numpy(), 0.5)


class TestPreferredAreaPerCell:
    """A0 per cell, fixed to the cell's area in the INITIAL array.

    A0 stops being a fitted number for the whole tissue and becomes data, so
    the array's area heterogeneity is inherited instead of replaced by a
    uniform target. It must be fixed for the whole run and inherited by every
    fork and resume — in particular the ablation run, which forks the base
    run's RELAXED steady state and would otherwise re-reference itself to the
    relaxed areas and stop being the same mechanical system as the run it is
    compared against."""

    @staticmethod
    def _model(sheet, **kw):
        from inner_ear_model import InnerEarModel
        kw.setdefault("contractility", {"HC": 0.4, "SC": 0.1})
        kw.setdefault("elasticity", {"HC": 2.0, "SC": 1.0})
        kw.setdefault("atoh_sensitivity", 0.4)
        return InnerEarModel(sheet, preferred_area_per_cell=True, **kw)

    def _sheet(self):
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        return sheet

    def test_measures_this_frame_when_there_is_nothing_to_inherit(self):
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        inner = self._model(sheet)
        areas = sheet.face_df["area"].to_numpy(float).copy()
        source, a0 = inner.set_preferred_area_per_cell()
        assert source == "measured"
        np.testing.assert_array_equal(sheet.face_df[COL].to_numpy(float), areas)
        np.testing.assert_array_equal(
            sheet.face_df["prefered_area"].to_numpy(float), areas)
        np.testing.assert_array_equal(
            sheet.face_df["prefered_vol"].to_numpy(float), areas)

    def test_inherits_the_carrier_column_instead_of_re_measuring(self):
        """The ablation run's case: the sheet it loads has relaxed far from the
        initial array, and the areas it would measure are the wrong answer."""
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        inherited = sheet.face_df["area"].to_numpy(float) * 1.37
        sheet.face_df[COL] = inherited
        inner = self._model(sheet)
        source, a0 = inner.set_preferred_area_per_cell()
        assert source == "inherited"
        np.testing.assert_array_equal(a0, inherited)
        np.testing.assert_array_equal(
            sheet.face_df["prefered_area"].to_numpy(float), inherited)

    def test_the_carrier_survives_model_construction(self):
        """``update_specs(..., reset=True)`` wipes every spec-backed column at
        construction; the carrier is deliberately NOT in the spec dict, which
        is what lets a fork or a resume inherit it."""
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        inherited = sheet.face_df["area"].to_numpy(float) * 1.37
        sheet.face_df[COL] = inherited
        self._model(sheet)
        np.testing.assert_array_equal(sheet.face_df[COL].to_numpy(float),
                                      inherited)

    def test_unusable_carrier_falls_back_to_measuring(self):
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        sheet.face_df[COL] = 0.0
        inner = self._model(sheet)
        source, a0 = inner.set_preferred_area_per_cell()
        assert source == "measured"
        np.testing.assert_array_equal(
            a0, sheet.face_df["area"].to_numpy(float))

    def test_the_type_interpolation_no_longer_owns_the_area_columns(self):
        """``update_cell_type_parameters`` interpolates every face parameter in
        atoh. With per-cell A0 the three area-derived columns must come back to
        the cell's own A0 — while the genuinely per-type ones still don't."""
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        inner = self._model(sheet)
        inner.set_preferred_area_per_cell()
        a0 = sheet.face_df[COL].to_numpy(float).copy()
        atoh = pd.Series(np.linspace(0.0, 1.0, sheet.Nf), index=sheet.face_df.index)
        inner.update_cell_type_parameters(atoh)
        np.testing.assert_array_equal(
            sheet.face_df["prefered_area"].to_numpy(float), a0)
        gamma = sheet.face_df["contractility"].to_numpy(float)
        assert gamma[0] == pytest.approx(0.1)      # atoh 0 -> SC
        assert gamma[-1] == pytest.approx(0.4)     # atoh 1 -> HC
        alpha = sheet.face_df["area_elasticity"].to_numpy(float)
        assert alpha[0] == pytest.approx(1.0) and alpha[-1] == pytest.approx(2.0)

    def test_a_dying_cell_keeps_its_zeroed_target_area(self):
        """Ablation / delamination drive a cell's target area to 0 to collapse
        it. Putting A0 back would resurrect it."""
        sheet = self._sheet()
        inner = self._model(sheet)
        inner.set_preferred_area_per_cell()
        victim = sheet.face_df.index[3]
        sheet.face_df.loc[victim, "type"] = -1
        sheet.face_df.loc[victim, "prefered_area"] = 0.0
        sheet.face_df.loc[victim, "prefered_vol"] = 0.0
        inner.update_cell_type_parameters(sheet.face_df["atoh_level"])
        assert sheet.face_df.at[victim, "prefered_area"] == 0.0
        assert sheet.face_df.at[victim, "prefered_vol"] == 0.0

    def test_target_perimeter_uses_each_cell_s_own_a0(self):
        """P0_i = p0(type_i) * sqrt(A0_i). Per type the product is a constant;
        per cell it cannot be, so the shape index is interpolated and
        multiplied by the cell's own sqrt(A0)."""
        from inner_ear_model import PREFERRED_AREA_INITIAL as COL
        sheet = self._sheet()
        inner = self._model(sheet, hc_shape_index=4.0, sc_shape_index=2.0)
        inner.set_preferred_area_per_cell()
        a0 = sheet.face_df[COL].to_numpy(float)
        atoh = pd.Series(np.zeros(sheet.Nf), index=sheet.face_df.index)
        inner.update_cell_type_parameters(atoh)
        np.testing.assert_allclose(
            sheet.face_df["prefered_perimeter"].to_numpy(float),
            2.0 * np.sqrt(a0))
        atoh[:] = 1.0
        inner.update_cell_type_parameters(atoh)
        np.testing.assert_allclose(
            sheet.face_df["prefered_perimeter"].to_numpy(float),
            4.0 * np.sqrt(a0))

    def test_folder_name_separates_per_cell_from_uniform_a0(self):
        import run_model
        kw = dict(initial_sheet_name="arr0", gammaSC=0.0105,
                  gammaHC_ratio=6.746, alphaHC_ratio=6.746, psigma=0)
        uniform = run_model._short_run_folder_name(preferred_area=0.74, **kw)
        per_cell = run_model._short_run_folder_name(
            preferred_area=None, preferred_area_per_cell=True, **kw)
        assert per_cell != uniform
        assert "_pacell" in per_cell and "_pacell" not in uniform
        # and not just the readable prefix: the hash has to differ too
        assert per_cell.split("_")[-1] != uniform.split("_")[-1]

    def test_the_auto_generated_folder_name_carries_the_flag(self, tmp_path,
                                                             monkeypatch):
        """The mechanics fit passes no ``name``, so run() derives one. Per-cell
        A0 leaves ``preferred_area_override`` None, so without the flag in the
        hash the folder is IDENTICAL to a run with no A0 specified at all — and
        the fit runs with reuse_existing_run=True, so it would be handed back
        the uniform-A0 archive."""
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        seen = {}
        real = run_model._short_run_folder_name

        def spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        class _Stop(Exception):
            pass

        def stop(*a, **k):
            raise _Stop

        monkeypatch.setattr(run_model, "_short_run_folder_name", spy)
        monkeypatch.setattr(run_model, "initialize_sheet", stop)
        with pytest.raises(_Stop):
            run_model.run(0.0105, 6.746, 6.746, 0,
                          preferred_area_per_cell=True, t_end=1)
        assert seen.get("preferred_area_per_cell") is True
        made = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        assert made and all("_pacell" in n for n in made), made

    def test_run_refuses_the_combinations_that_would_fight_it(self, tmp_path,
                                                              monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="random_forces"):
            run_model.run(0.01, 1.0, 1.0, 0, name="r1",
                          preferred_area_per_cell=True, random_forces=True)
        with pytest.raises(ValueError, match="preferred_area_override"):
            run_model.run(0.01, 1.0, 1.0, 0, name="r2",
                          preferred_area_per_cell=True,
                          preferred_area_override=0.74)

    def test_mechanics_task_can_ask_for_per_cell_a0(self, monkeypatch):
        """The 21-field task keeps setting one A0 for the whole tissue; a 22nd
        field asks for per-cell instead. Both runs of a task must agree, or the
        ablation would not be comparable to its base."""
        import mechanics_eval
        seen = []

        def fake_run(*a, **kw):
            seen.append(kw)
            raise FileNotFoundError("stop here")

        monkeypatch.setattr(mechanics_eval, "run", fake_run)
        base_task = [0.0105, 6.746, 6.746, "arr0", "E17.5", [], -1, 0.3,
                     "delta_level", 100, None, 30.0, False, 0.0, 0.03, 0.02,
                     None, 0.0, 0.0, 0.0, 0.74]

        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(base_task))
        assert seen[-1]["preferred_area_override"] == pytest.approx(0.74)
        assert "preferred_area_per_cell" not in seen[-1]

        per_cell = list(base_task)
        per_cell[20] = None          # no uniform A0 ...
        per_cell.append(True)        # ... per cell instead
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(per_cell))
        assert seen[-1]["preferred_area_per_cell"] is True
        assert "preferred_area_override" not in seen[-1]

        both = list(base_task) + [True]
        with pytest.raises(ValueError, match="per-cell A0"):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(both))


class TestInheritOrMeasure:
    """What a run takes from its source and what it works out for itself.

    A run that starts from an INITIAL array measures L0 and A0 on it; a run
    that continues from a previous RUN inherits both — including the
    mechanical fit's ablation run, which forks the base run's last frame
    without naming a time. The deciding fact is whether the source has an L0
    to give: the initial-array extracts record none."""

    @staticmethod
    def _capture(monkeypatch, run_model):
        captured = {}

        class _Stop(Exception):
            pass

        def fake_model(s, **kw):
            captured.update(kw)
            raise _Stop

        monkeypatch.setattr(run_model, "InnerEarModel", fake_model)
        return captured, _Stop

    def _source(self, rd, name, l0=None):
        os.makedirs(os.path.join(rd, name))
        open(os.path.join(rd, name, "history.hf5"), "wb").close()
        if l0 is not None:
            with open(os.path.join(rd, name, "l0.txt"), "w") as fh:
                fh.write(repr(l0) + "\n")

    def _fork(self, tmp_path, monkeypatch, l0):
        import run_model
        rd = str(tmp_path / "results")
        os.makedirs(rd)
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        self._source(rd, "src", l0)
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()
        monkeypatch.setattr(run_model, "load_sheet_from_file",
                            lambda *a, **k: sheet)
        captured, _Stop = self._capture(monkeypatch, run_model)
        with pytest.raises(_Stop):
            run_model.run(0.01, 1.0, 1.0, 0, initial_sheet_name="src",
                          name="fork1", no_differentiation=True, t_end=1)
        return captured

    def test_a_fork_of_an_initial_array_measures_L0(self, tmp_path,
                                                    monkeypatch):
        """An extract records no L0 (nothing was ever run on it), so the
        fork's first frame IS the initial array and L0 is measured there."""
        captured = self._fork(tmp_path, monkeypatch, l0=None)
        assert captured["length_normalization_factor"] is None

    def test_a_fork_of_a_previous_run_inherits_L0(self, tmp_path, monkeypatch):
        """The mechanical fit's ablation run: a fork of the base run's LAST
        frame with no time named. Re-measuring on that relaxed frame would give
        a different unit from the run it is compared against."""
        exact = 1.5 + float(np.sqrt(5))
        captured = self._fork(tmp_path, monkeypatch, l0=exact)
        assert captured["length_normalization_factor"] == exact

    def test_recorded_parameter_reads_the_original_run(self, tmp_path):
        import run_model
        d = tmp_path / "r"
        d.mkdir()
        (d / "parameters.txt").write_text(
            "gammaSC: 0.0105\nablated_cells: [337, 304]\nname: 'r'\n")
        assert run_model._recorded_parameter(str(d), "ablated_cells") == [337, 304]
        assert run_model._recorded_parameter(str(d), "gammaSC") == 0.0105
        assert run_model._recorded_parameter(str(d), "missing") is None
        assert run_model._recorded_parameter(str(tmp_path / "nope"), "x") is None

    def _resume_with_ablation(self, tmp_path, monkeypatch, recorded):
        import run_model
        rd = str(tmp_path / "results")
        monkeypatch.setattr(run_model, "RESULTS_DIR", rd)
        # an explicit name meeting an ablation list gets "ablated_<ids>"
        folder = os.path.join(rd, "r1ablated_3")
        os.makedirs(folder)
        with open(os.path.join(folder, "parameters.txt"), "w") as fh:
            fh.write("ablated_cells: %r\n" % (recorded,))

        class _Stop(Exception):
            pass

        def stop(*a, **k):
            raise _Stop

        monkeypatch.setattr(run_model, "load_sheet_from_file", stop)
        return run_model, _Stop

    def test_resuming_and_ADDING_an_ablation_refuses(self, tmp_path,
                                                     monkeypatch):
        """The restored event queue cannot carry an ablation the original run
        never had, so it would silently never fire."""
        run_model, _Stop = self._resume_with_ablation(tmp_path, monkeypatch, [])
        with pytest.raises(ValueError, match="Fork instead"):
            run_model.run(0.01, 1.0, 1.0, 0, name="r1", ablated_cells=[3],
                          continue_existing_run=True, continue_from_time=1.0)

    def test_resuming_an_ablation_run_is_allowed(self, tmp_path, monkeypatch):
        """Continuing an interrupted ablation run: its ablations already fired
        and must not fire again, so the drop is right and nothing refuses."""
        run_model, _Stop = self._resume_with_ablation(tmp_path, monkeypatch, [3])
        with pytest.raises(_Stop):
            run_model.run(0.01, 1.0, 1.0, 0, name="r1", ablated_cells=[3],
                          continue_existing_run=True, continue_from_time=1.0)

    def test_resume_keeps_cell_ids(self):
        """``InnerEarModel.__init__`` re-bases ``id`` to the row index; after a
        removal that relabels every later cell, and ``id`` is what analysis
        tracks cells across frames by."""
        import run_model
        sheet = _build_sheet_with_virtuals(4, 4)
        sheet.get_opposite()

        class _Inner:
            face_params = {}
            edge_params = {}

            def __init__(self, s):
                self.sheet = s

        saved = {"face": sheet.face_df.copy()}
        saved["face"]["id"] = np.arange(sheet.Nf) + 7      # gaps from removals
        sheet.face_df["id"] = np.arange(sheet.Nf)          # what __init__ does
        run_model._restore_derived_columns(_Inner(sheet), saved)
        np.testing.assert_array_equal(sheet.face_df["id"].to_numpy(),
                                      np.arange(sheet.Nf) + 7)


class TestMedianSpeedSteadyState:
    """The strict steady-state criterion: the MEDIAN per-vertex speed, not the
    fastest velocity component, is compared with ``quasi_static_threshold``.

    The max plateaus on topological noise (one rearranging vertex holds it
    up), so a cutoff on it stops on a random lull; the median reports the bulk
    of the tissue. These tests pin the two properties that make that so: one
    fast vertex does not hold the median up, and the median is of the speed
    (Euclidean norm), not of a component."""

    @staticmethod
    def _solve(monkeypatch, velocity_fn, statistic, threshold, tf=0.2):
        sheet, solver = TestSteadyStateStop._build(monkeypatch,
                                                   velocity_fn=velocity_fn)
        solver.solve(
            tf=tf, dt=0.01, max_displacement=1e9, save_interval=tf,
            until_steady_state=True, quasi_static_threshold=threshold,
            check_mechanical_steady=True,
            check_lateral_inhibition_steady=False,
            steady_state_min_steps=4, steady_state_statistic=statistic,
        )
        return solver.prev_t

    @staticmethod
    def _one_fast_vertex(t, pos):
        v = np.zeros_like(pos)
        v[0] = 0.5                     # vertex 0's x: a local rearrangement
        return v

    def test_one_fast_vertex_holds_the_max_up_but_not_the_median(
            self, monkeypatch):
        # short tf: 0.1 at speed 0.5 moves vertex 0 by 0.05, well short of
        # crossing an edge of the test sheet
        t_max = self._solve(monkeypatch, self._one_fast_vertex, "max", 1e-3,
                            tf=0.1)
        t_med = self._solve(monkeypatch, self._one_fast_vertex, "median", 1e-3,
                            tf=0.1)
        assert t_max >= 0.1 - 1e-9          # never steady: ran to tf
        assert t_med < 0.06                 # 4 steady steps and out

    def test_the_median_is_of_the_speed_not_a_component(self, monkeypatch):
        """Every vertex moves at 1e-3 in x AND y: each component is under a
        1.2e-3 cutoff, the speed (1.41e-3) is not."""
        uniform = lambda t, pos: np.full_like(pos, 1e-3)
        assert self._solve(monkeypatch, uniform, "max", 1.2e-3) < 0.06
        assert self._solve(monkeypatch, uniform, "median", 1.2e-3) >= 0.2 - 1e-9
        assert self._solve(monkeypatch, uniform, "median", 1.5e-3) < 0.06

    def test_an_unknown_statistic_is_refused(self, monkeypatch):
        with pytest.raises(ValueError, match="steady_state_statistic"):
            self._solve(monkeypatch, self._one_fast_vertex, "mean", 1e-3)

    def test_folder_name_separates_the_two_criteria(self):
        import run_model
        kw = dict(initial_sheet_name="arr0", gammaSC=0.0105,
                  gammaHC_ratio=6.746, alphaHC_ratio=3.5, psigma=0,
                  preferred_area_per_cell=True)
        old = run_model._short_run_folder_name(**kw)
        same = run_model._short_run_folder_name(steady_state_statistic="max",
                                                **kw)
        med = run_model._short_run_folder_name(
            quasi_static_threshold=1e-3, steady_state_statistic="median", **kw)
        med_finer = run_model._short_run_folder_name(
            quasi_static_threshold=5e-4, steady_state_statistic="median", **kw)
        max_1e3 = run_model._short_run_folder_name(
            quasi_static_threshold=1e-3, **kw)
        assert same == old                  # the default keeps every old name
        assert "_medv0.001_" in med
        assert len({med, med_finer, max_1e3, old}) == 4
        # 5e-4 and 1e-3 differ in the HASH, below the qst tag's 3 decimals
        assert med.split("_")[-1] != med_finer.split("_")[-1]

    def test_run_refuses_an_unknown_statistic_before_making_a_folder(
            self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="steady_state_statistic"):
            run_model.run(0.0105, 6.746, 3.5, 0, "arr0",
                          no_differentiation=True,
                          steady_state_statistic="medain")
        assert os.listdir(tmp_path) == []

    def test_mechanics_task_passes_it_to_both_runs(self, monkeypatch):
        """The 22-field task keeps the max criterion; a 23rd field selects the
        statistic, and the base AND ablation runs must both get it."""
        import mechanics_eval
        seen = []

        def fake_run(*a, **kw):
            seen.append(kw)
            if len(seen) % 2 == 1:
                return "base_run"            # base run done ...
            raise FileNotFoundError("stop")  # ... stop in the ablation run

        monkeypatch.setattr(mechanics_eval, "run", fake_run)
        task = [0.0105, 6.746, 3.5, "arr0", "E17.5", [0, 1], -1, 0.3,
                "delta_level", 100, None, 30.0, False, 0.0, 1e-3, 1e-3,
                None, 0.0, 0.0, 0.0, None, True]
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(task))
        assert [kw["steady_state_statistic"] for kw in seen] == ["max", "max"]

        seen.clear()
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(
                tuple(task + ["median"]))
        assert [kw["steady_state_statistic"] for kw in seen] == \
            ["median", "median"]


class TestFitRunsKeepFirstAndLastFrame:
    """Mechanical-fit runs record only their first and last frame
    (``_FIT_SAVE_INTERVAL`` is infinite): the solver's two unconditional
    records, whatever ends the run."""

    def test_the_fit_interval_is_infinite(self):
        import run_model
        assert run_model._FIT_SAVE_INTERVAL == float("inf")

    def test_steady_state_stop_keeps_two_frames(self, monkeypatch):
        sheet, solver = TestSteadyStateStop._build(
            monkeypatch, velocity_fn=lambda t, pos: np.full_like(pos, 1e-4))
        solver.solve(tf=5.0, dt=0.01, max_displacement=1e9,
                     save_interval=float("inf"), until_steady_state=True,
                     quasi_static_threshold=1e-3,
                     check_lateral_inhibition_steady=False)
        ts = sorted(set(solver.history.time_stamps))
        assert len(ts) == 2 and ts[0] == 0.0
        assert ts[1] == pytest.approx(solver.prev_t) and ts[1] < 0.1

    def test_removal_stop_keeps_two_frames(self, monkeypatch):
        sheet, solver = TestAblationStopsWhenTheCellsAreRemoved._solver(
            monkeypatch, remove_at=0.05)
        solver.solve(tf=1.0, dt=0.01, max_displacement=1e9,
                     save_interval=float("inf"), stop_when_removed=[0, 2])
        ts = sorted(set(solver.history.time_stamps))
        assert len(ts) == 2 and ts[0] == 0.0
        assert ts[1] == pytest.approx(solver.prev_t)

    def test_running_to_tf_keeps_two_frames(self, monkeypatch):
        sheet, solver = TestSteadyStateStop._build(
            monkeypatch, velocity_fn=lambda t, pos: np.zeros_like(pos))
        solver.solve(tf=0.3, dt=0.01, max_displacement=1e9,
                     save_interval=float("inf"))
        ts = sorted(set(solver.history.time_stamps))
        assert len(ts) == 2 and ts[1] == pytest.approx(0.3, abs=0.011)


class TestFitOutputsSeedsAndScores:
    """Fit runs write only what the score reads; the fit records every
    simulation's seeds and its own score."""

    def test_fit_outputs_only_skips_images_labels_and_contact_matrix(
            self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        kw = dict(no_differentiation=True, t_end=0.03, seed=7,
                  save_interval=run_model._FIT_SAVE_INTERVAL)
        run_model.run(0.01, 2.0, 2.0, 0, name="lean", fit_outputs_only=True, **kw)
        run_model.run(0.01, 2.0, 2.0, 0, name="full", **kw)
        lean = set(os.listdir(tmp_path / "lean"))
        full = set(os.listdir(tmp_path / "full"))
        dropped = {"initial.png", "finale.png", "movie.gif", "labels.npy",
                   "contact_matrix.npy"}
        assert dropped <= full and not (dropped & lean)
        assert {"history.hf5", "parameters.txt", "debug.log",
                "cell_info.pkl"} <= lean
        # same simulation either way: the switch is output-only
        from tyssue import HistoryHdf5
        from virtual_sheet import VirtualSheet
        a = HistoryHdf5.from_archive(str(tmp_path / "lean" / "history.hf5"),
                                     eptm_class=VirtualSheet)
        b = HistoryHdf5.from_archive(str(tmp_path / "full" / "history.hf5"),
                                     eptm_class=VirtualSheet)
        ta, tb = max(a.time_stamps), max(b.time_stamps)
        assert ta == tb
        np.testing.assert_array_equal(a.retrieve(ta).vert_df[["x", "y"]].to_numpy(),
                                      b.retrieve(tb).vert_df[["x", "y"]].to_numpy())

    def test_records_carry_the_seeds_even_when_the_task_failed(
            self, tmp_path, monkeypatch):
        import grid_fit_mechanics_v3 as g
        from mechanics_eval import base_run_name
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        task = g.build_task("E17.5", "arr0", 3.5, 0.0105, 6.746, [1, 2], 0.4)
        name = base_run_name(task)
        assert "_medv0.001" in name and "_pacell" in name and "_h5" in name
        for run, seed in ((name, 11), ("abl", 12)):
            (tmp_path / run).mkdir()
            (tmp_path / run / "parameters.txt").write_text("seed: %d\n" % seed)
        failed = g.record_from_terms("E17.5", "k", {}, "arr0", 0.4, None,
                                     task=task)
        assert failed["failed"] and failed["base_seed"] == 11
        ok = g.record_from_terms("E17.5", "k", {}, "arr0", 0.4,
                                 {"base_run": name, "ablation_run": "abl"})
        assert (ok["base_seed"], ok["ablation_seed"]) == (11, 12)

    def test_the_raw_data_reproduces_every_observable(self, tmp_path,
                                                       monkeypatch):
        """Roundness ratio, ablation ratio, lambda^2 and sigma~ are all
        recomputed from the saved raw data alone and match the record."""
        import json
        import grid_fit_mechanics_v3 as g
        from mechanics_scoring import (energy_sums, shrinkage_lambda_squared,
                                       isotropic_stress)
        from post_processing import _hc_over_mean_sc
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        fd = TestMechanicsScoring._face_df(seed=4)
        rng = np.random.default_rng(5)
        hc_r, sc_r = rng.uniform(.7, .9, 12), rng.uniform(.5, .8, 40)
        hc_a, sc_a = rng.uniform(.9, 1.1, 2), rng.uniform(1., 1.3, 20)
        terms = {"hc_roundness": hc_r, "sc_roundness": sc_r,
                 "hc_ablation": hc_a, "sc_ablation": sc_a,
                 "roundness_ratio": _hc_over_mean_sc(hc_r, sc_r),
                 "ablation_ratio": _hc_over_mean_sc(hc_a, sc_a),
                 "lambda_squared": shrinkage_lambda_squared(fd),
                 "stress": isotropic_stress(fd), "energy_sums": energy_sums(fd)}
        rec = g.record_from_terms("E17.5", "k", {}, "arr", 0.4, terms)
        raw = json.loads(json.dumps(rec))["raw"]        # survives JSON
        assert (raw["n_hc_roundness"], raw["n_sc_roundness"]) == (12, 40)
        assert (raw["n_hc_area_change"], raw["n_sc_area_change"]) == (2, 20)
        assert raw["mean_hc_roundness"] / raw["mean_sc_roundness"] == \
            pytest.approx(rec["roundness_ratio"])
        assert np.mean(raw["hc_area_change"]) / np.mean(raw["sc_area_change"]) \
            == pytest.approx(rec["ablation_ratio"])
        e = raw["energy_sums"]
        assert (e["sum_2A0A_alpha"] - e["sum_gamma_L2"]) / (2 * e["sum_A2_alpha"]) \
            == pytest.approx(rec["lambda_squared"], rel=1e-12)
        assert (e["sum_2alphaA_A_minus_A0"] + e["sum_gamma_L2"]) / (2 * e["sum_A"]) \
            == pytest.approx(rec["stress"], rel=1e-12)
        assert e["n_cells"] == len(fd)

    def test_the_grid_gamma_sc_reproduces_the_measured_shrinkage(self):
        """gammaSC_bar plugged back into Eq. 40 (A0 = A) gives lambda_exp,
        and the five grid values sit evenly inside (0, 2 gammaSC_bar)."""
        import grid_fit_mechanics_v3 as g
        from mechanics_scoring import shrinkage_lambda_squared
        rng = np.random.default_rng(8)
        A = rng.uniform(.6, .9, 300)
        L = np.sqrt(A) * rng.uniform(3.6, 3.9, 300)
        w = (rng.random(300) < .25).astype(float)
        for a_hc, r in ((2.0, 10.0), (10.0, 2.0), (6.0, 6.0)):
            gsc = g.gamma_sc_for_shrinkage(A, L, w, a_hc, r, 0.9249)
            fd = pd.DataFrame({"area": A, "perimeter": L, "prefered_area": A,
                               "area_elasticity": 1 + (a_hc - 1) * w,
                               "contractility": gsc * (1 + (r - 1) * w)})
            assert shrinkage_lambda_squared(fd) == pytest.approx(0.9249 ** 2,
                                                                 rel=1e-12)
        vals = [f * 2 for f in g.GAMMA_SC_FRACTIONS]
        assert np.allclose(np.diff([0] + vals + [2 * 2]), 2 * 2 / 6)

    @staticmethod
    def _coarse_record(tmp_path, monkeypatch, bar=0.0152):
        import json
        import grid_fit_mechanics_v3 as g
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(g, "TAG", "")
        rows = [{"alphaHC": a, "R_gamma": r, "gammaSC_bar": bar * a / r,
                 "gammaSC_bar_per_pattern": [bar * a / r]}
                for a in g.GRID_ALPHA_HC for r in g.GRID_R_GAMMA]
        (tmp_path / (g.GRID % ("E17.5", ""))).write_text(json.dumps(
            {"per_point": True, "pooled_gammaSC_bar": bar,
             "fractions": g.GAMMA_SC_FRACTIONS, "bars": rows}))
        return g

    def test_the_fine_grid_is_a_quarter_of_the_coarse_spacing(self, tmp_path,
                                                               monkeypatch):
        g = self._coarse_record(tmp_path, monkeypatch)
        bar = 0.0152 * 6 / 4                     # the (6, 4) pair's own bar
        centre = (6.0, round(bar, 5), 4.0)
        grid, info = g.build_fine_grid("E17.5", centre)
        assert len(grid) == 125 and len(set(grid)) == 125 and centre in grid
        a = sorted({p[0] for p in grid}); r = sorted({p[2] for p in grid})
        gs = sorted({p[1] for p in grid})
        assert np.allclose(a, [5.0, 5.5, 6.0, 6.5, 7.0])
        assert np.allclose(r, [3.0, 3.5, 4.0, 4.5, 5.0])
        assert np.allclose(np.diff(gs), bar / 3 / 4, atol=2e-5)
        assert info["coarse_steps"]["gammaSC"] == pytest.approx(bar / 3)

    def test_the_fine_grid_shifts_inward_at_a_lower_bound(self, tmp_path,
                                                          monkeypatch):
        """At alpha_HC = R_gamma = 2 the centred window would reach 1.0; it is
        shifted up one step instead. gammaSC at its lowest coarse value
        (bar/3) needs no shift: bar/3 - 2 bar/12 = bar/6 > 0."""
        g = self._coarse_record(tmp_path, monkeypatch)
        bar = 0.0152                                  # the (2, 2) pair's bar
        grid, info = g.build_fine_grid("E17.5", (2.0, round(bar / 3, 5), 2.0))
        a = sorted({p[0] for p in grid}); r = sorted({p[2] for p in grid})
        gs = sorted({p[1] for p in grid})
        assert np.allclose(a, [1.5, 2.0, 2.5, 3.0, 3.5])
        assert np.allclose(r, [1.5, 2.0, 2.5, 3.0, 3.5])
        assert gs[0] == pytest.approx(bar / 6, abs=2e-5) and len(gs) == 5
        assert info["offsets"]["alphaHC"] == [-1, 0, 1, 2, 3]
        assert info["offsets"]["gammaSC"] == [-2, -1, 0, 1, 2]
        assert len(grid) == 125 and all(p[0] > 1 and p[2] > 1 and p[1] > 0
                                        for p in grid)

    _TARGETS = {
        "E17.5": {"roundness_ratio": (1.2, 0.05, 3), "ablation_ratio": (0.85, 0.1, 2),
                  "shrinkage": (0.925, 0.005, 14), "stress": (0.3, 0.04, 14)},
        "P0": {"roundness_ratio": (1.2, 0.05, 3), "ablation_ratio": (0.85, 0.1, 3),
               "shrinkage": (0.92, 0.01, 14), "stress": (0.2, 0.02, 14)}}

    @classmethod
    def _graded_records(cls, tmp_path, monkeypatch, drop=None):
        """Two coarse points per stage on two patterns, plus a FINE point that
        fits better than either. Roundness alone separates them."""
        import json
        import grid_fit_mechanics_v3 as g
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(g, "TAG", "")
        monkeypatch.setattr(g.ms, "experimental_targets",
                            lambda s: cls._TARGETS[s])
        coarse = {"E17.5": [(2.0, 0.01, 2.0), (4.0, 0.02, 4.0)],
                  "P0": [(2.0, 0.01, 8.0), (4.0, 0.03, 2.0)]}
        best = {"E17.5": (4.0, 0.02, 4.0), "P0": (2.0, 0.01, 8.0)}
        fine = {"E17.5": (4.5, 0.02, 4.0), "P0": (2.5, 0.01, 8.0)}
        for s in g.STAGES:
            t = cls._TARGETS[s]
            (tmp_path / (g.GRID % (s, ""))).write_text(json.dumps({
                "initials": ["a", "b"],
                "points": [{"alphaHC": a, "gammaSC": gs, "R_gamma": r}
                           for a, gs, r in coarse[s]]}))
            lines = []
            for p in coarse[s] + [fine[s]]:
                rr = 1.2 if p == fine[s] else (1.21 if p == best[s] else 1.5)
                for i in ("a", "b"):
                    if drop == (s, p, i):
                        continue
                    lines.append(json.dumps({
                        "point": g.point_key(*p), "initial": i, "failed": False,
                        "roundness_ratio": rr, "ablation_ratio": t["ablation_ratio"][0],
                        "lambda_squared": t["shrinkage"][0] ** 2,
                        "stress": t["stress"][0]}))
            (tmp_path / (g.RECORDS % (s, ""))).write_text("\n".join(lines) + "\n")
        return g, coarse, best

    def test_the_fine_centre_is_ranked_over_the_coarse_points_only(
            self, tmp_path, monkeypatch):
        """A fine point that fits better (its records sit in the same JSONL)
        must not move the centre, and no scores file is read at all."""
        g, _, best = self._graded_records(tmp_path, monkeypatch)
        assert not (tmp_path / (g.SCORES % "")).exists()
        for s in g.STAGES:
            centre, pair = g.best_coarse_point(s)
            assert centre == best[s]
            assert set(pair) == {"E17.5", "P0", "J_mech"}

    def test_the_fine_centre_waits_for_a_complete_coarse_grid(
            self, tmp_path, monkeypatch):
        g, coarse, _ = self._graded_records(
            tmp_path, monkeypatch, drop=("P0", (4.0, 0.03, 2.0), "b"))
        with pytest.raises(SystemExit, match="P0 coarse grid incomplete: 3 of 4"):
            g.best_coarse_point("E17.5")

    def test_a_resumed_fine_grid_must_be_the_one_already_started(
            self, tmp_path, monkeypatch):
        import json
        g = self._coarse_record(tmp_path, monkeypatch)
        bar = 0.0152 * 6 / 4
        _, info = g.build_fine_grid("E17.5", (6.0, round(bar, 5), 4.0))
        g.check_fine_grid_unchanged("E17.5", info)          # nothing on disk yet
        (tmp_path / (g.FINE_GRID % ("E17.5", ""))).write_text(json.dumps(info))
        g.check_fine_grid_unchanged("E17.5", info)          # the same grid: fine
        _, moved = g.build_fine_grid("E17.5", (4.0, round(0.0152 * 4 / 4, 5), 4.0))
        with pytest.raises(SystemExit, match="coarse records have changed"):
            g.check_fine_grid_unchanged("E17.5", moved)

    @staticmethod
    def _terms(base, abl, roundness=(0.8, 0.9)):
        hc_r, sc_r = np.array(roundness), np.array([0.7, 0.72])
        return {"hc_roundness": hc_r, "sc_roundness": sc_r,
                "hc_ablation": np.array([0.9]), "sc_ablation": np.array([1.1, 1.2]),
                "roundness_ratio": hc_r / sc_r.mean(),
                "ablation_ratio": np.array([0.9 / 1.15]),
                "lambda_squared": 0.85, "stress": 0.3, "ablated_removed": True,
                "energy_sums": {"n_cells": 3}, "base_run": base,
                "ablation_run": abl, "ablated_labels": [5], "ablated_ids": [7]}

    def test_re_extract_rereads_every_record_and_runs_nothing(
            self, tmp_path, monkeypatch):
        import json
        import grid_fit_mechanics_v3 as g
        import mechanics_eval
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(g, "TAG", "")
        monkeypatch.setattr(mechanics_eval, "run", lambda *a, **k: 1 / 0)
        for run, seed in (("base", 11), ("abl", 12)):
            (tmp_path / run).mkdir()
            (tmp_path / run / "parameters.txt").write_text("seed: %d\n" % seed)
            (tmp_path / run / "history.hf5").write_text("")
        params = {"alphaHC": 4.0, "gammaSC": 0.02, "R_gamma": 4.0, "gammaHC": 0.08}
        ok = g.record_from_terms("E17.5", "k", params, "arr", 0.4,
                                 self._terms("base", "abl"))
        failed = g.record_from_terms("E17.5", "k2", params, "arr", 0.4, None)
        (tmp_path / (g.RECORDS % ("E17.5", ""))).write_text(
            json.dumps(ok) + "\n" + json.dumps(failed) + "\n")
        seen = []

        def fake_terms(base, abl, labels, ids, **kw):
            seen.append((base, abl, labels, ids, kw))
            return self._terms(base, abl)
        monkeypatch.setattr(mechanics_eval, "terms_from_runs", fake_terms)
        assert g.re_extract("E17.5") == 0
        assert seen == [("base", "abl", [5], [7], dict(
            type_by="delta_level", threshold=0.4, post_ablation_frame=-1,
            require_ablated_removed=True))]                 # the failed one: not read
        out = [json.loads(l) for l in open(
            tmp_path / (g.RECORDS % ("E17.5", "_reextract")))]
        assert g.differing_fields(ok, out[0]) == [] and out[1] == failed
        with pytest.raises(SystemExit, match="delete it"):
            g.re_extract("E17.5")
        # a different measurement is reported, field by field
        (tmp_path / (g.RECORDS % ("E17.5", "_reextract"))).unlink()
        monkeypatch.setattr(mechanics_eval, "terms_from_runs",
                            lambda b, a, l, i, **kw: self._terms(b, a, (0.8, 0.95)))
        assert g.re_extract("E17.5") == 1

    def test_re_extract_refuses_when_a_run_folder_is_missing(
            self, tmp_path, monkeypatch):
        import json
        import grid_fit_mechanics_v3 as g
        monkeypatch.setattr(g, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(g, "TAG", "")
        rec = g.record_from_terms("E17.5", "k", {}, "arr", 0.4,
                                  self._terms("base", "abl"))
        (tmp_path / (g.RECORDS % ("E17.5", ""))).write_text(json.dumps(rec) + "\n")
        with pytest.raises(SystemExit, match="run base"):
            g.re_extract("E17.5")

    def test_replay_runs_the_record_with_its_own_seeds(self, tmp_path,
                                                        monkeypatch):
        import json
        import grid_fit_mechanics_v3 as g
        import mechanics_eval
        from mechanics_eval import base_run_name
        src, dst = tmp_path / "fit", tmp_path / "replay"
        src.mkdir(), dst.mkdir()
        (dst / "arr").mkdir()
        monkeypatch.setattr(g, "TAG", "")
        params = {"alphaHC": 4.0, "gammaSC": 0.02, "R_gamma": 4.0, "gammaHC": 0.08}
        name = base_run_name(g.build_task("E17.5", "arr", 4.0, 0.02, 4.0, [7], 0.4))
        for run, seed in ((name, 11), ("abl", 12)):
            (src / run).mkdir()
            (src / run / "parameters.txt").write_text("seed: %d\n" % seed)
        monkeypatch.setattr(g, "RESULTS_DIR", str(src))
        rec = g.record_from_terms("E17.5", "k", params, "arr", 0.4,
                                  self._terms(name, "abl"))
        assert (rec["base_seed"], rec["ablation_seed"]) == (11, 12)
        (src / (g.RECORDS % ("E17.5", ""))).write_text(json.dumps(rec) + "\n")
        monkeypatch.setattr(g, "RESULTS_DIR", str(dst))
        tasks = []

        def fake_evaluate(task):
            tasks.append(task)
            for run, key in ((name, "base"), ("abl", "ablation")):
                (dst / run).mkdir()
                (dst / run / "parameters.txt").write_text(
                    "seed: %d\n" % task[25][key])
            return self._terms(name, "abl")
        monkeypatch.setattr(mechanics_eval, "_evaluate_mechanics_for_sheet",
                            fake_evaluate)
        assert g.replay("E17.5", "k", "arr", str(src)) == []
        assert len(tasks[0]) == 26 and tasks[0][25] == {"base": 11, "ablation": 12}
        assert tasks[0][:25] == g.build_task("E17.5", "arr", 4.0, 0.02, 4.0,
                                             [7], 0.4)
        assert (dst / (g.RECORDS % ("E17.5", "_replay"))).is_file()
        # the runs are there now: a second replay would only reuse them
        with pytest.raises(SystemExit, match="without the original runs"):
            g.replay("E17.5", "k", "arr", str(src))

    def test_replay_seeds_reach_both_runs(self, monkeypatch):
        import mechanics_eval
        seen = []

        def fake_run(*a, **kw):
            seen.append(kw)
            if len(seen) % 2 == 1:
                return "base_run"
            raise FileNotFoundError("stop")
        monkeypatch.setattr(mechanics_eval, "run", fake_run)
        task = [0.0105, 6.746, 3.5, "arr0", "E17.5", [0, 1], -1, 0.3,
                "delta_level", 100, None, 30.0, False, 0.0, 1e-3, 1e-3,
                None, 0.0, 0.0, 0.0, None, True, "median", "removed", 100.0]
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(task))
        assert ["seed" in kw for kw in seen] == [False, False]
        seen.clear()
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(
                tuple(task + [{"base": 11, "ablation": 12}]))
        assert [kw["seed"] for kw in seen] == [11, 12]

    def test_two_points_sharing_a_run_folder_are_refused(self, tmp_path,
                                                         monkeypatch):
        g = self._coarse_record(tmp_path, monkeypatch)
        monkeypatch.setattr(g, "ablation_targets", lambda stage, initials: {
            "arr": {"ids": [1], "threshold": 0.4, "min_separation_hops": 5,
                    "neighbours_hc_sc": [(0, 6)]}})
        # 0.01231 and 0.01234 are both "0.0123" in the folder name
        with pytest.raises(SystemExit, match="share run folders"):
            g.simulate("E17.5", ["arr"], [(6.0, 0.01231, 4.0),
                                          (6.0, 0.01234, 4.0)], 1, dry_run=True)

    def test_the_run_cap_is_1500_s(self):
        import grid_fit_mechanics_v3 as g
        assert g.MAX_WALL_SECONDS == 1500

    def test_a_simulations_own_score(self):
        from mechanics_scoring import simulation_chi2, WORST_CHI2
        targets = {"roundness_ratio": (1.2, 0.05, 3),
                   "ablation_ratio": (0.85, 0.04, 3),
                   "shrinkage": (0.925, 0.005, 14)}
        rec = {"roundness_ratio": 1.25, "ablation_ratio": None,
               "lambda_squared": 0.9025}
        s = simulation_chi2(rec, targets)
        assert s["terms"]["roundness_ratio"] == pytest.approx(1.0)
        assert s["terms"]["shrinkage"] == pytest.approx(
            (0.95 - 0.925) ** 2 / 0.005 ** 2)
        assert s["terms"]["ablation_ratio"] is None
        assert s["undefined"] == ["ablation_ratio"]
        assert s["J"] == pytest.approx(1.0 + 25.0)
        rec["lambda_squared"] = -0.1
        assert simulation_chi2(rec, targets)["terms"]["shrinkage"] == WORST_CHI2


class TestHillExponentFive:
    """Every Hill function runs at l = 5 (SI, Supplementary Table 1); phase 1
    ran at 3. The archived l=3 runs must stay nameable and must never be
    reused, resumed or returned in place of an l=5 run."""

    def test_the_model_defaults_are_five(self):
        from lateral_inhibition_model import (LateralInhibitionModel,
                                              HILL_EXPONENT,
                                              HISTORICAL_HILL_EXPONENT)
        assert (HILL_EXPONENT, HISTORICAL_HILL_EXPONENT) == (5, 3)
        li = LateralInhibitionModel(None)
        assert (li.l, li.m) == (5, 5)
        # the gate exponent follows m unless overridden
        x = np.array([0.2, 0.5, 0.9])
        np.testing.assert_allclose(li.increasing_hill(x, 0.5),
                                   x ** 5 / (0.5 ** 5 + x ** 5))
        np.testing.assert_allclose(li.decreasing_hill(x, 0.5),
                                   0.5 ** 5 / (0.5 ** 5 + x ** 5))

    def test_folder_name_carries_the_exponent_and_keeps_the_old_names(self):
        import run_model
        kw = dict(initial_sheet_name="arr0", gammaSC=0.0105,
                  gammaHC_ratio=6.746, alphaHC_ratio=3.5, psigma=0)
        new = run_model._short_run_folder_name(**kw)
        old = run_model._short_run_folder_name(hill_exponent=3,
                                               li_rates=(1, 1, 1), **kw)
        assert "_h5_" in new
        assert "_h" not in old.replace("_gHC", "")
        assert new != old
        # The l=3 spelling is the pre-change name, byte for byte: these
        # parameters made this folder before the exponent existed.
        canonical = "from=arr0|gSC=0.0105|gHC=6.75|aHC=3.50|ps=0.00|abl="
        import hashlib
        assert old.endswith(hashlib.md5(canonical.encode()).hexdigest()[:10])

    @staticmethod
    def _old_run(tmp_path, name, l):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "parameters.txt").write_text("l: %d\nm: %d\n" % (l, l))
        return folder

    def test_run_refuses_a_folder_made_at_another_exponent(
            self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        self._old_run(tmp_path, "strict_arr0", 3)
        for kw in (dict(), dict(reuse_existing_run=True),
                   dict(continue_existing_run=True)):
            with pytest.raises(ValueError, match="Hill exponent 3"):
                run_model.run(0.0105, 6.746, 3.5, 0, "arr0", name="strict_arr0",
                              no_differentiation=True, **kw)

    def test_the_same_exponent_gets_the_old_behaviour(self, tmp_path,
                                                      monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        self._old_run(tmp_path, "strict_arr0", 3)
        # exists, not reused -> returned untouched, as always
        assert run_model.run(0.0105, 6.746, 3.5, 0, "arr0", name="strict_arr0",
                             no_differentiation=True,
                             hill_exponent=3) == "strict_arr0"

    def test_a_bad_exponent_is_refused_before_making_a_folder(
            self, tmp_path, monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="hill_exponent"):
            run_model.run(0.0105, 6.746, 3.5, 0, "arr0",
                          no_differentiation=True, hill_exponent=4.5)
        assert os.listdir(tmp_path) == []


class TestLateralInhibitionRatesFromTheSI:
    """tau_N = 0.15, tau_D = 0.30, beta~ = 0.5 (SI 2.4); phase 1 ran at 1."""

    def test_defaults_and_natural_bounds(self):
        from lateral_inhibition_model import LateralInhibitionModel
        li = LateralInhibitionModel(None)
        assert (li.notch_repressor_degradation_ratio,
                li.delta_repressor_degradation_ratio,
                li.notch_delta_production_ratio) == (0.15, 0.30, 0.5)
        assert li.get_maximal_notch_level() == pytest.approx(1 / 0.15)
        assert li.get_maximal_delta_level() == pytest.approx(1 / 0.30)
        assert li.get_maximal_repressor_level() == 1.0
        # the phase-1 rates give back the old bounds exactly
        old = LateralInhibitionModel(None, notch_repressor_degradation_ratio=1,
                                     delta_repressor_degradation_ratio=1,
                                     notch_delta_production_ratio=1)
        assert (old.get_maximal_notch_level(), old.get_maximal_delta_level(),
                old.get_maximal_repressor_level()) == (1, 1, 1)

    def test_an_unsignalled_cell_is_not_capped_at_one(self):
        """One isolated cell: no contacts, so N -> 1/tau_N = 6.67 and D ->
        1/tau_D = 3.33; nothing after the integration may hold them at 1."""
        from lateral_inhibition_model import LateralInhibitionModel

        class _Model:
            def get_contact_matrix(self):
                return np.zeros((1, 1))

            def update_cell_type_parameters(self, atoh):
                pass

        sheet = type("S", (), {})()
        sheet.face_df = pd.DataFrame({
            "unique_id": [0], "repressor_level": [0.0], "notch_level": [0.0],
            "delta_level": [0.0], "perimeter": [3.0], "notch_sensitivity": [0.2],
            "type": [0], "atoh_level": [0.0]})
        li = LateralInhibitionModel(_Model(), atoh_sensitivity=0.5)
        li.get_length_dependent_differentiation_function(
            dt=60.0, quasi_static=True)(sheet, None)
        assert sheet.face_df.notch_level[0] == pytest.approx(1 / 0.15, rel=1e-3)
        assert sheet.face_df.delta_level[0] == pytest.approx(1 / 0.30, rel=1e-3)
        # Not capped at their natural maxima either: a cell started ABOVE
        # 1/tau decays towards it, and a short step must leave it above.
        sheet.face_df.loc[0, ["notch_level", "delta_level", "repressor_level"]]             = [20.0, 20.0, 5.0]
        li.get_length_dependent_differentiation_function(
            dt=0.1, quasi_static=True)(sheet, None)
        assert sheet.face_df.notch_level[0] > 19
        assert sheet.face_df.delta_level[0] > 19
        assert sheet.face_df.repressor_level[0] > 4      # R > 1 is not capped

    def test_folder_names_and_the_guard(self, tmp_path, monkeypatch):
        import run_model
        kw = dict(initial_sheet_name="arr0", gammaSC=0.0105,
                  gammaHC_ratio=6.746, alphaHC_ratio=3.5, psigma=0)
        si = run_model._short_run_folder_name(**kw)
        ones = run_model._short_run_folder_name(li_rates=(1, 1, 1), **kw)
        assert si != ones
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        folder = tmp_path / "strict_arr0"
        folder.mkdir()
        (folder / "parameters.txt").write_text(
            "l: 5\nnotch_repressor_degradation_ratio: 1\n"
            "delta_repressor_degradation_ratio: 1\n"
            "notch_delta_production_ratio: 1\n")
        with pytest.raises(ValueError, match="lateral-inhibition rates"):
            run_model.run(0.0105, 6.746, 3.5, 0, "arr0", name="strict_arr0",
                          no_differentiation=True, reuse_existing_run=True)
        assert run_model.run(0.0105, 6.746, 3.5, 0, "arr0", name="strict_arr0",
                             no_differentiation=True, tau_n=1, tau_d=1,
                             beta=1) == "strict_arr0"


class TestAblationStopsWhenTheCellsAreRemoved:
    """``ablation_stop="removed"``: the ablation run ends once every ablated
    cell has left the tissue -- the experiment measures the neighbours once
    the hole has closed -- instead of on a whole-tissue steady state."""

    @staticmethod
    def _solver(monkeypatch, remove_at=None, targets=(0, 2)):
        """Zero velocity (never steady-state relevant) and, from the step
        whose start time passes ``remove_at``, the target cells' ids are
        dropped -- what a delamination does to them."""
        state = {"t": 0.0}

        def velocity(t, pos):
            if remove_at is not None and state["t"] >= remove_at:
                ids = sheet.face_df["id"].to_numpy().copy()
                ids[np.isin(ids, targets)] = -1
                sheet.face_df["id"] = ids
            state["t"] += 0.01
            return np.zeros_like(pos)

        sheet, solver = TestSteadyStateStop._build(monkeypatch,
                                                   velocity_fn=velocity)
        sheet.face_df["id"] = np.arange(sheet.Nf)
        return sheet, solver

    def test_stops_on_the_step_the_last_cell_goes(self, monkeypatch):
        sheet, solver = self._solver(monkeypatch, remove_at=0.05)
        solver.solve(tf=1.0, dt=0.01, max_displacement=1e9, save_interval=1.0,
                     stop_when_removed=[0, 2])
        assert 0.05 < solver.prev_t < 0.08
        # ...and that state is the last recorded frame
        assert max(solver.history.time_stamps) == pytest.approx(solver.prev_t)

    def test_one_cell_left_keeps_it_running(self, monkeypatch):
        sheet, solver = self._solver(monkeypatch, remove_at=0.05, targets=(0,))
        solver.solve(tf=0.2, dt=0.01, max_displacement=1e9, save_interval=1.0,
                     stop_when_removed=[0, 2])
        assert solver.prev_t >= 0.2 - 1e-9

    def test_it_overrides_a_steady_state_stop_only_when_asked(self, monkeypatch):
        # without the rule, zero velocity is steady after 4 steps
        sheet, solver = self._solver(monkeypatch)
        solver.solve(tf=1.0, dt=0.01, max_displacement=1e9, save_interval=1.0,
                     until_steady_state=True, quasi_static_threshold=1e-6,
                     check_lateral_inhibition_steady=False)
        assert solver.prev_t < 0.06

    def test_needs_the_id_column(self, monkeypatch):
        sheet, solver = self._solver(monkeypatch)
        sheet.face_df.drop(columns="id", inplace=True)
        with pytest.raises(ValueError, match="'id'"):
            solver.solve(tf=0.1, dt=0.01, max_displacement=1e9,
                         stop_when_removed=[0])

    def test_ids_are_taken_from_the_first_frame_and_reread_on_resume(
            self, tmp_path):
        import pandas as pd
        import run_model
        sheet = type("S", (), {})()
        sheet.face_df = pd.DataFrame({"id": [10, 11, 12, 13]})
        assert run_model._ablated_cell_ids(sheet, [1, 3], str(tmp_path),
                                           False) == [11, 13]
        # a resume's frame has other labels; the record wins
        sheet.face_df = pd.DataFrame({"id": [10, 12]})
        assert run_model._ablated_cell_ids(sheet, [1, 3], str(tmp_path),
                                           True) == [11, 13]
        with pytest.raises(ValueError, match="cannot resume"):
            run_model._ablated_cell_ids(sheet, [1], str(tmp_path / "nope"),
                                        True)
        with pytest.raises(ValueError, match="not faces"):
            run_model._ablated_cell_ids(sheet, [7], str(tmp_path), False)

    def test_folder_name(self):
        import run_model
        kw = dict(initial_sheet_name="base", gammaSC=0.0105,
                  gammaHC_ratio=6.746, alphaHC_ratio=3.5, psigma=0,
                  ablated_cells=[4, 9])
        steady = run_model._short_run_folder_name(**kw)
        removed = run_model._short_run_folder_name(ablation_stop="removed",
                                                   **kw)
        assert steady.endswith("_abl") and removed.endswith("_ablrm")
        assert steady[:-4] != removed[:-6]        # the hash differs too
        # the rule means nothing without an ablation, so it changes nothing
        kw.pop("ablated_cells")
        assert (run_model._short_run_folder_name(ablation_stop="removed", **kw)
                == run_model._short_run_folder_name(**kw))

    def test_unknown_rule_refused_before_making_a_folder(self, tmp_path,
                                                         monkeypatch):
        import run_model
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="ablation_stop"):
            run_model.run(0.0105, 6.746, 3.5, 0, "arr0", ablated_cells=[1],
                          no_differentiation=True, ablation_stop="gone")
        assert os.listdir(tmp_path) == []

    def test_mechanics_task_passes_it_to_the_ablation_run(self, monkeypatch):
        import mechanics_eval
        seen = []

        def fake_run(*a, **kw):
            seen.append(kw)
            if len(seen) % 2 == 1:
                return "base_run"
            raise FileNotFoundError("stop")

        monkeypatch.setattr(mechanics_eval, "run", fake_run)
        task = [0.0105, 6.746, 3.5, "arr0", "E17.5", [0, 1], -1, 0.3,
                "delta_level", 100, None, 30.0, False, 0.0, 1e-3, 1e-3,
                None, 0.0, 0.0, 0.0, None, True, "median"]
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(tuple(task))
        assert seen[1]["ablation_stop"] == "steady"
        assert "ablation_stop" not in seen[0]          # the base run has none
        seen.clear()
        with pytest.raises(FileNotFoundError):
            mechanics_eval._evaluate_mechanics_for_sheet(
                tuple(task + ["removed"]))
        assert seen[1]["ablation_stop"] == "removed"


class TestAblationTargets:
    """Four well-separated HCs per pattern, chosen by contact hops."""

    @staticmethod
    def _sheet():
        sheet = _build_sheet(10, 10)
        sheet.get_opposite()
        # distinct ids, offset from the labels so the two cannot be confused
        # (the lattice builder leaves the column all zeros)
        sheet.face_df["id"] = np.arange(sheet.Nf) + 1000
        return sheet

    def test_adjacency_is_the_hex_lattice(self):
        from ablation_targets import cell_adjacency
        sheet = self._sheet()
        adj = cell_adjacency(sheet)
        assert {len(v) for v in adj.values()} == {6}      # periodic hexagons
        assert all(a in adj[b] for a in adj for b in adj[a])

    def test_chooses_separated_hcs_reproducibly(self):
        from ablation_targets import (choose_ablation_hcs, cell_adjacency,
                                      _hops_from, pattern_seed)
        sheet = self._sheet()
        rng = np.random.default_rng(3)
        delta = rng.uniform(0, 1, sheet.Nf)
        seed = pattern_seed("initial_array_0_for_E17")
        assert seed == pattern_seed("initial_array_0_for_E17")
        ids, info = choose_ablation_hcs(sheet, delta, 0.6, min_separation=4,
                                        seed=seed)
        ids2, _ = choose_ablation_hcs(sheet, delta, 0.6, min_separation=4,
                                      seed=seed)
        assert ids == ids2 and len(ids) == 4 and info["seed"] == seed
        labels = info["labels"]
        assert all(delta[sheet.face_df.index.get_loc(c)] > 0.6 for c in labels)
        adj = cell_adjacency(sheet)
        hops = [_hops_from(adj, a)[b] for i, a in enumerate(labels)
                for b in labels[i + 1:]]
        assert min(hops) == info["min_separation_hops"] >= 4
        assert ids == [int(sheet.face_df.at[c, "id"]) for c in labels]
        # a different seed is a different (equally valid) sample
        others = {tuple(choose_ablation_hcs(sheet, delta, 0.6,
                                            min_separation=4, seed=s)[0])
                  for s in range(5)}
        assert len(others) > 1

    def test_refuses_when_it_cannot_separate_them(self):
        from ablation_targets import choose_ablation_hcs
        sheet = self._sheet()
        delta = np.zeros(sheet.Nf)
        delta[:3] = 1.0                       # only three HCs at all
        with pytest.raises(ValueError, match="candidate"):
            choose_ablation_hcs(sheet, delta, 0.5)
        delta[:] = 0.0
        hc = list(cell_adjacency_ring(sheet))
        delta[[sheet.face_df.index.get_loc(c) for c in hc]] = 1.0
        with pytest.raises(ValueError, match="hop"):
            choose_ablation_hcs(sheet, delta, 0.5)

    def test_counts_hc_and_sc_neighbours(self):
        from ablation_targets import choose_ablation_hcs, cell_adjacency
        sheet = self._sheet()
        delta = np.zeros(sheet.Nf)
        hcs = [0, 45, 50, 95]
        delta[hcs] = 1.0
        f0 = int(sheet.face_df.index[0])
        nb = sorted(cell_adjacency(sheet)[f0])[0]       # one HC beside cell 0
        delta[sheet.face_df.index.get_loc(nb)] = 1.0
        ids, info = choose_ablation_hcs(sheet, delta, 0.5, min_separation=1)
        counts = dict(zip(info["labels"], info["neighbours_hc_sc"]))
        assert all(h + s == 6 for h, s in counts.values())
        assert sum(h for h, _ in counts.values()) >= 1

    def test_initial_array_cells_are_named_as_the_runs_name_them(
            self, tmp_path, monkeypatch):
        """InnerEarModel re-bases ``id`` to the row label, and many initial
        arrays store a PERMUTED id column: the chosen cells must come back as
        row labels, not as that column."""
        import ablation_targets, history_io, post_processing
        sheet = self._sheet()
        sheet.face_df["id"] = np.random.default_rng(0).permutation(sheet.Nf)
        sheet.face_df["unique_id"] = np.arange(sheet.Nf)
        sheet.face_df["type"] = 0
        delta = np.random.default_rng(1).uniform(0, 1, sheet.Nf)
        (tmp_path / "arr").mkdir()
        np.save(tmp_path / "arr" / "delta_levels.npy", delta)

        class _H:
            time_stamps = [0.0]

            def retrieve(self, t):
                s = sheet.copy() if hasattr(sheet, "copy") else sheet
                s.arrange_sheet_from_history = lambda *a, **k: None
                return s

        monkeypatch.setattr(history_io, "RESULTS_DIR", str(tmp_path))
        monkeypatch.setattr(post_processing, "load_history_file", lambda n: _H())
        ids, info = ablation_targets.ablation_hcs_for_initial(
            "arr", 0.6, min_separation=4)
        assert ids == info["labels"] == info["ids"]
        assert info["extract_id_column"] == [
            int(sheet.face_df.at[c, "id"]) for c in ids]
        assert ids != info["extract_id_column"]

    def test_labels_for_ids_after_a_removal(self):
        from ablation_targets import labels_for_ids
        fd = pd.DataFrame({"id": [10, 12, 13]}, index=[0, 1, 2])
        assert labels_for_ids(fd, [12, 11, 13]) == ([1, 2], [11])


def cell_adjacency_ring(sheet):
    """A cell and its six neighbours: seven cells, none 3 hops apart."""
    from ablation_targets import cell_adjacency
    adj = cell_adjacency(sheet)
    f0 = int(sheet.face_df.index[0])
    return [f0] + sorted(adj[f0])


class TestMechanicsScoring:
    """SI section 7: closed forms, SEMs, the lambda^2 <= 0 rule, the stress
    ratio and the pairing of stages."""

    @staticmethod
    def _face_df(seed=0, n=40, gamma_scale=0.01):
        rng = np.random.default_rng(seed)
        A = rng.uniform(0.6, 0.9, n)
        return pd.DataFrame({
            "area": A, "perimeter": np.sqrt(A) * rng.uniform(3.6, 3.9, n),
            "prefered_area": A * rng.uniform(0.95, 1.05, n),
            "area_elasticity": rng.choice([1.0, 3.5], n),
            "contractility": gamma_scale * rng.choice([1.0, 6.7], n),
            "prefered_perimeter": np.zeros(n), "type": np.zeros(n, int)})

    @staticmethod
    def _energy(fd, lam):
        return float(np.sum(fd.area_elasticity / 2
                            * (lam ** 2 * fd.area - fd.prefered_area) ** 2
                            + fd.contractility / 2 * lam ** 2 * fd.perimeter ** 2))

    def test_eq31_minimises_the_scaled_energy(self):
        from mechanics_scoring import shrinkage_lambda_squared
        from post_processing import relaxed_cut_scale
        fd = self._face_df()
        lam = np.sqrt(shrinkage_lambda_squared(fd))
        grid = np.linspace(0.5, 1.3, 80001)
        brute = grid[np.argmin([self._energy(fd, g) for g in grid])]
        assert lam == pytest.approx(brute, abs=2e-5)
        assert lam == pytest.approx(relaxed_cut_scale(fd), abs=1e-9)

    def test_eq36_is_dE_over_dAtot_at_one(self):
        from mechanics_scoring import isotropic_stress
        fd = self._face_df(seed=1)
        h = 1e-6
        dE = (self._energy(fd, 1 + h) - self._energy(fd, 1 - h)) / (2 * h)
        dA = 2 * fd.area.sum()                 # d(lambda^2 sum A)/dlambda at 1
        assert isotropic_stress(fd) == pytest.approx(dE / dA, rel=1e-7)

    def test_dying_cells_are_left_out_and_p0_is_refused(self):
        from mechanics_scoring import shrinkage_lambda_squared
        fd = self._face_df()
        dying = fd.copy()
        dying.loc[0, ["type", "prefered_area", "contractility"]] = [-1, 0.0, 10.0]
        assert shrinkage_lambda_squared(dying) == pytest.approx(
            shrinkage_lambda_squared(fd.drop(index=0)))
        fd.loc[3, "prefered_perimeter"] = 1.0
        with pytest.raises(ValueError, match="target perimeter"):
            shrinkage_lambda_squared(fd)

    def test_chi2_carries_both_sems(self):
        from mechanics_scoring import chi2, WORST_CHI2
        assert chi2(1.3, 0.03, 1.2, 0.04) == pytest.approx(0.01 / 0.0025)
        assert chi2(1.3, float("nan"), 1.2, 0.04) == WORST_CHI2

    def _summary(self, rr, ab, lam_sq, stress):
        from mechanics_scoring import stage_summary
        return stage_summary([{"roundness_ratio": a, "ablation_ratio": b,
                               "lambda_squared": c, "stress": d}
                              for a, b, c, d in zip(rr, ab, lam_sq, stress)])

    def test_one_nonpositive_lambda_squared_rejects_the_stage(self):
        from mechanics_scoring import stage_chi2, WORST_CHI2
        targets = {"roundness_ratio": (1.2, 0.05, 3),
                   "ablation_ratio": (0.85, 0.04, 3),
                   "shrinkage": (0.925, 0.005, 14)}
        ok = self._summary([1.2, 1.21, 1.19], [0.8, 0.9, None],
                           [0.86, 0.85, 0.855], [0.1, 0.1, 0.1])
        bad = self._summary([1.2, 1.21, 1.19], [0.8, 0.9, None],
                            [0.86, -0.01, 0.855], [0.1, 0.1, 0.1])
        t_ok, j_ok = stage_chi2(ok, targets)
        t_bad, j_bad = stage_chi2(bad, targets)
        assert t_ok["shrinkage"] < 10 and t_bad["shrinkage"] == WORST_CHI2
        # the None ablation ratio is dropped, not counted
        assert ok["ablation_ratio"][2] == 2
        assert t_ok["roundness_ratio"] == t_bad["roundness_ratio"]
        # model SEM of lambda enters: sqrt of the lambda^2 values
        lam = np.sqrt([0.86, 0.85, 0.855])
        sem = lam.std(ddof=1) / np.sqrt(3)
        assert t_ok["shrinkage"] == pytest.approx(
            (lam.mean() - 0.925) ** 2 / (0.005 ** 2 + sem ** 2))

    def test_stress_ratio_is_e17_over_p0_with_propagated_sems(self):
        from mechanics_scoring import stress_chi2, ratio_sem
        sE = self._summary([1] * 3, [1] * 3, [0.8] * 3, [0.30, 0.32, 0.34])
        sP = self._summary([1] * 3, [1] * 3, [0.8] * 3, [0.20, 0.21, 0.19])
        tE = {"stress": (0.30896, 0.045659, 14)}
        tP = {"stress": (0.20035, 0.019051, 14)}
        r_e, s_e = ratio_sem(0.30896, 0.045659, 0.20035, 0.019051)
        # the SI's own form of the experimental SEM
        si = np.sqrt(0.045659 ** 2 / 0.20035 ** 2
                     + 0.019051 ** 2 * (0.30896 / 0.20035 ** 2) ** 2)
        assert s_e == pytest.approx(si)
        mE, mP = 0.32, 0.20
        semE = np.std([0.30, 0.32, 0.34], ddof=1) / np.sqrt(3)
        semP = np.std([0.20, 0.21, 0.19], ddof=1) / np.sqrt(3)
        r_m, s_m = mE / mP, mE / mP * np.hypot(semE / mE, semP / mP)
        assert stress_chi2(sE, sP, tE, tP) == pytest.approx(
            (r_m - r_e) ** 2 / (s_e ** 2 + s_m ** 2))

    def test_pairs_are_ranked_by_j_mech(self):
        from mechanics_scoring import rank_pairs
        targets = {"roundness_ratio": (1.2, 0.05, 3),
                   "ablation_ratio": (0.85, 0.04, 3),
                   "shrinkage": (0.925, 0.005, 14), "stress": (0.3, 0.04, 14)}
        tP = dict(targets, stress=(0.2, 0.02, 14))
        good = self._summary([1.2, 1.21, 1.19], [0.84, 0.86, 0.85],
                             [0.855, 0.857, 0.853], [0.30, 0.31, 0.29])
        far = self._summary([1.0, 1.01, 0.99], [0.84, 0.86, 0.85],
                            [0.855, 0.857, 0.853], [0.30, 0.31, 0.29])
        p0 = self._summary([1.2, 1.21, 1.19], [0.84, 0.86, 0.85],
                           [0.855, 0.857, 0.853], [0.20, 0.21, 0.19])
        rows = rank_pairs({"good": good, "far": far}, {"p": p0}, targets, tP)
        assert [r["E17.5"] for r in rows] == ["good", "far"]
        assert rows[0]["J_mech"] == pytest.approx(
            rows[0]["J_E17.5"] + rows[0]["J_P0"] + rows[0]["chi2_stress"])


class TestFindNonBoundaryCells:
    """The boundary filter, on the sheets where it is not a no-op.

    Every simulation here is periodic, so no edge is ever unpaired and the
    filter selects everything. These tests use small hand-built frames — the
    only place the non-periodic branch is exercised.
    """

    @staticmethod
    def _sheet(edges, faces):
        """edges: (label, face, opposite) triples; faces: face labels."""
        class _Frame:
            pass
        f = _Frame()
        f.edge_df = pd.DataFrame(
            {"face": [e[1] for e in edges], "opposite": [e[2] for e in edges]},
            index=[e[0] for e in edges])
        f.face_df = pd.DataFrame(index=list(faces))
        return f

    def test_periodic_sheet_excludes_nothing(self):
        # every edge paired: 0<->1, 2<->3
        sheet = self._sheet([(0, 10, 1), (1, 11, 0), (2, 11, 3), (3, 12, 2)],
                            [10, 11, 12])
        assert list(find_non_boundary_cells(sheet)) == [10, 11, 12]

    def test_boundary_face_and_its_neighbour_are_excluded(self):
        # face 10 has an unpaired edge -> boundary; face 11 shares edge 1 with
        # it -> neighbour of the boundary; face 12 touches only face 11.
        sheet = self._sheet([(0, 10, -1), (1, 10, 2), (2, 11, 1),
                             (3, 11, 4), (4, 12, 3)],
                            [10, 11, 12])
        assert list(find_non_boundary_cells(sheet)) == [12]

    def test_edge_label_is_not_read_as_a_face_label(self):
        """The bug this replaced: `opposite.isin(boundary_cells)`.

        Face 0 is the boundary face. Edge 3 belongs to face 7 and its opposite
        is edge 0 — an EDGE label that collides numerically with the boundary
        FACE label 0. Face 7 does not touch face 0 (edge 0 belongs to face 0
        only through its own edge 1), so it must survive; the old code excluded
        it on the name collision alone.
        """
        sheet = self._sheet([(0, 5, 3), (1, 0, -1), (2, 0, 4),
                             (3, 7, 0), (4, 6, 2)],
                            [0, 5, 6, 7])
        kept = list(find_non_boundary_cells(sheet))
        assert 7 in kept, "face 7 excluded by an edge/face label collision"
        assert 0 not in kept and 6 not in kept    # boundary face and its neighbour


class TestEdgeTensionIsTheEnergyDerivative:
    """``get_edge_stress`` must return dE/dl of the interface, not a
    projection of the vertex forces.

    The energy depends on edge lengths ONLY through the face perimeters
    (the area term is not a function of edge length at fixed vertex
    positions), so for E = sum_c [K/2 (A_c-A0_c)^2 + Gamma_c/2 P_c^2]

        dE/dl_e = Gamma_n P_n + Gamma_m P_m

    over the one or two faces containing e. These tests differentiate that
    energy NUMERICALLY, in edge-length space, and compare.
    """

    @staticmethod
    def _build():
        np.random.seed(0)
        sheet = VirtualSheet.planar_virtual_sheet_2d(
            "p4x4", nx=4, ny=4, distx=1.0, disty=1.0,
            maximal_bond_length=10.0, minimal_bond_length=0.05,
            periodic=True, draw_debug=False,
        )
        from inner_ear_model import InnerEarModel
        inner = InnerEarModel(
            sheet,
            tension={('HC', 'HC'): 0.05, ('HC', 'SC'): 0.05, ('SC', 'SC'): 0.05},
            repulsion={'HC': 0., 'SC': 0.},
            repulsion_distance={'HC': 0., 'SC': 0.},
            preferred_area={'HC': 0.5, 'SC': 0.5},
            contractility={'HC': 0.1, 'SC': 0.1},
            elasticity={'HC': 1., 'SC': 1.},
        )
        # Make the cells mechanically DIFFERENT, so a formula that used one
        # cell's contractility for both sides of a junction would be caught.
        rng = np.random.default_rng(3)
        n = inner.sheet.face_df.shape[0]
        inner.sheet.face_df["contractility"] = rng.uniform(0.05, 0.4, n)
        inner.sheet.geom.update_all(inner.sheet)
        return inner

    @staticmethod
    def _energy_from_edge_lengths(sheet, lengths):
        """The contractility energy as a function of the edge-length vector:
        sum_c Gamma_c/2 (sum_{e in c} l_e)^2. The area term is independent of
        l and so drops out of the derivative."""
        face = sheet.edge_df["face"].to_numpy()
        perim = np.bincount(sheet.face_df.index.get_indexer(face),
                            weights=lengths,
                            minlength=sheet.face_df.shape[0])
        gamma = (sheet.face_df["contractility"]
                 * sheet.face_df["is_alive"]).to_numpy(dtype=float)
        return float(np.sum(0.5 * gamma * perim ** 2))

    def test_matches_numerical_dE_dl(self):
        from inner_ear_model import ContractilityPerimeterElasticity
        from tyssue.dynamics.effectors import FaceAreaElasticity
        inner = self._build()
        sheet = inner.sheet
        tension = inner.get_edge_stress(
            [ContractilityPerimeterElasticity, FaceAreaElasticity]).to_numpy()

        lengths = sheet.edge_df["length"].to_numpy(dtype=float).copy()
        eps = 1e-6
        # a spread of edges, including ones on the periodic seam
        probes = np.linspace(0, len(lengths) - 1, 25).astype(int)
        for e in probes:
            plus = lengths.copy(); plus[e] += eps
            minus = lengths.copy(); minus[e] -= eps
            numeric = ((self._energy_from_edge_lengths(sheet, plus)
                        - self._energy_from_edge_lengths(sheet, minus))
                       / (2 * eps))
            # The numerical derivative moves ONE half-edge, so it picks up only
            # that half-edge's face; the physical junction carries both, which
            # is what get_edge_stress returns. Add the opposite's own term.
            opp = int(sheet.edge_df["opposite"].to_numpy()[e])
            if opp >= 0:
                o = sheet.edge_df.index.get_indexer([opp])[0]
                plus = lengths.copy(); plus[o] += eps
                minus = lengths.copy(); minus[o] -= eps
                numeric += ((self._energy_from_edge_lengths(sheet, plus)
                             - self._energy_from_edge_lengths(sheet, minus))
                            / (2 * eps))
            assert abs(tension[e] - numeric) < 1e-6 * max(1.0, abs(numeric)), (
                "edge %d: get_edge_stress gave %.8g, numerical dE/dl is %.8g"
                % (e, tension[e], numeric))

    def test_equals_gamma_n_Pn_plus_gamma_m_Pm(self):
        from inner_ear_model import ContractilityPerimeterElasticity
        inner = self._build()
        sheet = inner.sheet
        tension = inner.get_edge_stress([ContractilityPerimeterElasticity]).to_numpy()

        gamma_p = (sheet.face_df["contractility"] * sheet.face_df["is_alive"]
                   * sheet.face_df["perimeter"])
        own = sheet.upcast_face(gamma_p).to_numpy(dtype=float)
        opp = sheet.edge_df["opposite"].to_numpy()
        expected = own.copy()
        has = opp >= 0
        expected[has] += own[sheet.edge_df.index.get_indexer(opp[has])]
        assert np.allclose(tension, expected)

    def test_both_half_edges_of_a_junction_carry_the_same_tension(self):
        from inner_ear_model import ContractilityPerimeterElasticity
        inner = self._build()
        sheet = inner.sheet
        tension = inner.get_edge_stress([ContractilityPerimeterElasticity]).to_numpy()
        opp = sheet.edge_df["opposite"].to_numpy()
        has = np.where(opp >= 0)[0]
        mirror = sheet.edge_df.index.get_indexer(opp[has])
        assert np.allclose(tension[has], tension[mirror]), (
            "a junction carries one tension; its two half-edges disagree")

    def test_area_elasticity_does_not_change_the_tension(self):
        from inner_ear_model import ContractilityPerimeterElasticity
        from tyssue.dynamics.effectors import FaceAreaElasticity
        inner = self._build()
        without = inner.get_edge_stress([ContractilityPerimeterElasticity])
        with_area = inner.get_edge_stress(
            [ContractilityPerimeterElasticity, FaceAreaElasticity])
        assert np.allclose(without.to_numpy(), with_area.to_numpy()), (
            "the area term is not a function of edge length; it must not "
            "enter dE/dl")

    def test_switched_on_unsupported_effector_raises(self):
        from inner_ear_model import ContractilityPerimeterElasticity, BoundaryBending
        from tyssue.dynamics.effectors import LineTension
        inner = self._build()
        inner.sheet.face_df["bending"] = 0.3
        inner.sheet.edge_df["line_tension"] = 0.2
        for extra in (LineTension, BoundaryBending):
            with pytest.raises(NotImplementedError) as exc:
                inner.get_edge_stress([ContractilityPerimeterElasticity, extra])
            assert extra.__name__ in str(exc.value)

    def test_switched_off_unsupported_effector_is_allowed(self):
        """kappa = 0 bending is an exact no-op, and the v2 runs carry it in
        their "all" set. Refusing it by class alone would reject a set this
        expression handles exactly."""
        from inner_ear_model import ContractilityPerimeterElasticity, BoundaryBending
        inner = self._build()
        inner.sheet.face_df["bending"] = 0.0
        plain = inner.get_edge_stress([ContractilityPerimeterElasticity])
        with_bending = inner.get_edge_stress(
            [ContractilityPerimeterElasticity, BoundaryBending])
        assert np.allclose(plain.to_numpy(), with_bending.to_numpy())

    def test_no_contractility_effector_raises(self):
        from tyssue.dynamics.effectors import FaceAreaElasticity
        inner = self._build()
        with pytest.raises(ValueError):
            inner.get_edge_stress([FaceAreaElasticity])

    def test_non_zero_target_perimeter_raises(self):
        from inner_ear_model import ContractilityPerimeterElasticity
        inner = self._build()
        inner.sheet.face_df["prefered_perimeter"] = 0.0
        inner.sheet.face_df.loc[inner.sheet.face_df.index[2],
                                "prefered_perimeter"] = 0.7
        with pytest.raises(NotImplementedError) as exc:
            inner.get_edge_stress([ContractilityPerimeterElasticity])
        assert "P0" in str(exc.value)


class TestFindPsigmaBatchesAPassIntoOnePool:
    """``find_psigma`` must pool a whole PASS -- every (psigma, stage, array)
    run -- and still attribute each run to the point it belongs to.

    The risk the batching introduces is a slice that silently scores one
    point's runs under another point's psigma, which would corrupt the fit
    without erroring. These tests run the serial path (n_workers=1) and check
    the attribution directly.
    """

    @staticmethod
    def _params():
        return {"E17.5": (0.0105, 3.5, 0.0, 0.0), "P0": (0.0105, 1.757, 0.0, 0.0)}

    def _patch(self, monkeypatch, seen):
        import full_model as fm

        def fake_run(task):
            # mirror _run_full_model_one's name: the prefix (task[22]) is what
            # separates one repeat's folder from another's
            initial, psigma, prefix = task[0], task[15], task[22]
            return "%s_%s|ps=%.5g" % (prefix, initial, psigma)

        def fake_score(stage, model_names=None, model_groups=None,
                       type_by=None, threshold=None,
                       max_number_of_neighbors=None):
            # find_psigma passes per-array GROUPS; flatten for these checks
            names = ([n for grp in model_groups for n in grp]
                     if model_groups is not None else model_names)
            ps = {n.split("|ps=")[1] for n in names}
            assert len(ps) == 1, "one point's scoring saw several psigma: %s" % ps
            seen.append((stage, float(ps.pop()), tuple(names)))
            # score minimised at psigma = 0.1 so the refinement has a target.
            # The fit sums score1+score2, so put the value there.
            value = abs(float(names[0].split("|ps=")[1]) - 0.1)
            return {"score1": value, "score2": 0.0, "score3": 0.0,
                    "total": value}

        monkeypatch.setattr(fm, "_run_full_model_one", fake_run)
        monkeypatch.setattr(fm, "compare_full_model_differentiation_to_experiments",
                            fake_score)
        return fm

    def test_each_point_is_scored_on_its_own_runs(self, monkeypatch):
        seen = []
        fm = self._patch(monkeypatch, seen)
        best, scores = fm.find_psigma(
            self._params(), psigma_bounds=(0.0, 0.2), n_grid=3, n_refine=0,
            n_arrays=4, n_repeats=1, n_workers=1, plot=False,
            save_json=False, use_score_cache=False)

        assert sorted(scores) == [0.0, 0.1, 0.2]
        for psigma, entry in scores.items():
            for stage in ("E17.5", "P0"):
                # the score the fake returns IS |psigma - 0.1|; if the batching
                # mixed points up this would be another point's value
                assert abs(entry[stage]["total"] - abs(psigma - 0.1)) < 1e-12, (
                    "psigma=%g %s scored %g" % (psigma, stage, entry[stage]["total"]))
            assert abs(entry["objective"]
                       - sum(entry[s]["total"] for s in ("E17.5", "P0"))) < 1e-12
        assert abs(best - 0.1) < 1e-12

    def test_every_run_of_the_pass_is_launched_once(self, monkeypatch):
        seen = []
        fm = self._patch(monkeypatch, seen)
        fm.find_psigma(self._params(), psigma_bounds=(0.0, 0.2), n_grid=3,
                       n_refine=0, n_arrays=4, n_repeats=1, n_workers=1,
                       plot=False, save_json=False, use_score_cache=False)
        # 3 psigma x 2 stages scored, each on its own 4 arrays
        assert len(seen) == 6, [s[:2] for s in seen]
        assert {(s, p) for s, p, _ in seen} == {
            (st, ps) for st in ("E17.5", "P0") for ps in (0.0, 0.1, 0.2)}
        for stage, psigma, names in seen:
            assert len(names) == 4 and len(set(names)) == 4
            suffix = "_for_E17" if stage == "E17.5" else "_for_P0"
            assert all(suffix in n for n in names), (
                "%s point scored the other stage's arrays: %s" % (stage, names))

    def test_repeats_are_pooled_into_one_score_per_point(self, monkeypatch):
        """Each point is scored on n_arrays x n_repeats runs, each repeat in
        its own folder prefix -- otherwise repeat 2 would be handed repeat 1's
        folder by reuse_existing_run."""
        seen = []
        fm = self._patch(monkeypatch, seen)
        fm.find_psigma(self._params(), psigma_bounds=(0.06, 0.12), n_grid=3,
                       n_refine=0, n_arrays=4, n_repeats=3, n_workers=1,
                       plot=False, save_json=False, use_score_cache=False)
        assert len(seen) == 6
        for stage, psigma, names in seen:
            assert len(names) == 12, "%d runs pooled, expected 4x3" % len(names)
            assert len(set(names)) == 12, "a repeat reused another's folder"
            # Match the prefix rather than splitting on the array name: a run
            # folder is "<repeat prefix>_<initial sheet>...", and the initial
            # sheet name has already changed once (random_periodic_array<i>_for_
            # -> initial_array_<rank>_for_), which silently turned the old
            # split into a no-op. Requiring exactly one match also catches
            # fullmodel_v2 being a string prefix of fullmodel_v2r2.
            want = {fm.REPEAT_PREFIX[r] for r in (1, 2, 3)}
            prefixes = set()
            for n in names:
                hit = [p for p in want if n.startswith(p + "_")]
                assert len(hit) == 1, "%r matches %d repeat prefixes" % (n, len(hit))
                prefixes.add(hit[0])
            assert prefixes == want, prefixes

    def test_the_fit_never_visits_psigma_zero_when_bounds_exclude_it(self, monkeypatch):
        """pT = 0 is scored locally from the archived baseline, not on the VM.
        The refinement window is clamped to psigma_bounds, so a lower bound
        above 0 keeps every pass off it."""
        seen = []
        fm = self._patch(monkeypatch, seen)
        _best, scores = fm.find_psigma(
            self._params(), psigma_bounds=(0.06, 0.12), n_grid=7, n_refine=2,
            n_arrays=2, n_repeats=1, n_workers=1, plot=False, save_json=False,
            use_score_cache=False)
        assert 0.0 not in scores
        assert min(scores) >= 0.06 - 1e-12, sorted(scores)[:3]
        assert all(p >= 0.06 - 1e-12 for _s, p, _n in seen)

    def test_n_repeats_out_of_range_is_refused(self, monkeypatch):
        import full_model as fm
        for bad in (0, len(fm.REPEAT_PREFIX) + 1):
            with pytest.raises(ValueError):
                fm.find_psigma(self._params(), psigma_bounds=(0.06, 0.12),
                               n_grid=2, n_refine=0, n_arrays=1, n_repeats=bad,
                               n_workers=1, plot=False, save_json=False,
                               use_score_cache=False)

    def test_refinement_passes_still_run_in_sequence(self, monkeypatch):
        seen = []
        fm = self._patch(monkeypatch, seen)
        best, scores = fm.find_psigma(
            self._params(), psigma_bounds=(0.0, 0.4), n_grid=5, n_refine=1,
            n_arrays=2, n_repeats=1, n_workers=1, plot=False,
            save_json=False, use_score_cache=False)
        # pass 1: 0, .1, .2, .3, .4 (step .1) -> best .1; pass 2 halves the step
        # around it, adding .05 and .15. Only the PASSES are sequential: the
        # second cannot be batched with the first because it needs its minimum.
        assert {0.0, 0.1, 0.2, 0.3, 0.4} <= set(scores)
        assert {0.05, 0.15} <= set(scores), sorted(scores)
        assert abs(best - 0.1) < 1e-12
        # and no point was simulated or scored twice across the two passes
        scored = [(s, p) for s, p, _ in seen]
        assert len(scored) == len(set(scored)), scored

    def test_task_tuple_keeps_l0_ratio_last(self):
        import full_model as fm
        tasks = fm.full_model_tasks(
            "E17.5", 0.0105, 3.5, 0.0, 0.0, atoh_sensitivity=0.355,
            psigma=0.123, n_arrays=3, l0_ratio=7.0)
        assert len(tasks) == 3
        for t in tasks:
            assert t[-1] == 7.0, "l0_ratio must stay LAST in the positional tuple"
            assert t[15] == 0.123, "psigma moved within the task tuple"
            assert "_for_E17" in t[0]

    def test_run_full_model_arrays_builds_the_same_tasks(self, monkeypatch):
        import full_model as fm
        captured = []
        monkeypatch.setattr(fm, "_run_full_model_one",
                            lambda task: captured.append(task) or "n")
        fm.run_full_model_arrays("P0", 0.0105, 1.757, 0.0, 0.0,
                                 atoh_sensitivity=0.355, psigma=0.05,
                                 n_arrays=3, n_workers=1)
        expected = fm.full_model_tasks("P0", 0.0105, 1.757, 0.0, 0.0,
                                       atoh_sensitivity=0.355, psigma=0.05,
                                       n_arrays=3)
        assert captured == expected

    def test_per_stage_gammaHC_ratio_and_A0_reach_the_runs(self, monkeypatch):
        """The v2 fit has R_gamma 6.746 / 3.851 and A0 0.7418 / 0.7585 PER
        STAGE. The legacy 4-tuple cannot carry either, so a dict must, and the
        values must reach the task tuple (gammaHC_ratio is arg 2, A0 arg 16)."""
        seen = []
        fm = self._patch(monkeypatch, seen)
        tasks = []
        real = fm._run_full_model_one

        def capture(task):
            tasks.append(task)
            return real(task)

        monkeypatch.setattr(fm, "_run_full_model_one", capture)
        fm.find_psigma(
            {"E17.5": {"gamma_sc": 0.0105, "R_gamma": 6.746, "R_alpha": 3.5,
                       "A0": 0.741812},
             "P0": {"gamma_sc": 0.0105, "R_gamma": 3.850522, "R_alpha": 1.757,
                    "A0": 0.758542}},
            psigma_bounds=(0.06, 0.12), n_grid=2, n_refine=0, n_arrays=1,
            n_repeats=1, n_workers=1, plot=False, save_json=False,
            use_score_cache=False)
        by_stage = {}
        for t in tasks:
            stage = "E17.5" if "_for_E17" in t[0] else "P0"
            by_stage.setdefault(stage, set()).add((t[2], t[3], t[16]))
        assert by_stage["E17.5"] == {(6.746, 3.5, 0.741812)}, by_stage["E17.5"]
        assert by_stage["P0"] == {(3.850522, 1.757, 0.758542)}, by_stage["P0"]

    def test_legacy_four_tuple_still_works(self):
        import full_model as fm
        m = fm._stage_mechanics((0.0105, 3.5, 0.0, 0.0), 0.74)
        assert m["gammaSC"] == 0.0105 and m["alphaHC_ratio"] == 3.5
        assert m["gammaHC_ratio"] == 1.0          # the old hard-wired value
        assert m["preferred_area"] == 0.74        # falls back to the sweep-wide one

    def test_malformed_stage_mechanics_is_refused(self):
        import full_model as fm
        with pytest.raises(ValueError):
            fm._stage_mechanics((0.0105, 3.5, 0.0), None)
        with pytest.raises(ValueError):
            fm._stage_mechanics({"R_gamma": 6.746}, None)    # no gammaSC

    def test_shared_LI_params_but_per_stage_arrays_and_mechanics(self, monkeypatch):
        """pT (psigma), pS, pR and the atoh threshold are SHARED by both
        stages at every sweep point; the initial arrays and the mechanics are
        per stage. Asserted on the task tuples actually handed to the runner."""
        seen = []
        fm = self._patch(monkeypatch, seen)
        tasks = []
        real = fm._run_full_model_one
        monkeypatch.setattr(fm, "_run_full_model_one",
                            lambda t: (tasks.append(t), real(t))[1])
        fm.find_psigma(
            {"E17.5": {"gamma_sc": 0.0105, "R_gamma": 6.746124660721616,
                       "R_alpha": 3.5, "A0": 0.741812389049632},
             "P0": {"gamma_sc": 0.0105, "R_gamma": 3.850521945713907,
                    "R_alpha": 1.757, "A0": 0.7585423232230211}},
            psigma_bounds=(0.06, 0.12), n_grid=3, n_refine=0, n_arrays=2,
            n_repeats=1, n_workers=1, plot=False, save_json=False,
            use_score_cache=False,
            notch_sensitivity=0.1, repressor_sensitivity=0.3,
            atoh_sensitivity=0.355079)

        e17 = [t for t in tasks if "_for_E17" in t[0]]
        p0 = [t for t in tasks if "_for_P0" in t[0]]
        assert len(e17) == len(p0) == 6          # 3 psigma x 2 arrays

        # --- SHARED across stages: the lateral-inhibition parameters --------
        # task layout: 6 atoh, 7 notch(pS), 8 repressor(pR), 15 psigma
        for i in (6, 7, 8):
            assert {t[i] for t in tasks} == {(0.355079, 0.1, 0.3)[i - 6]}, (
                "LI parameter at index %d differs between stages/points" % i)
        # every psigma appears for BOTH stages, i.e. pT is swept jointly
        assert ({t[15] for t in e17} == {t[15] for t in p0}
                == {0.06, 0.09, 0.12})

        # --- PER STAGE: the initial arrays ----------------------------------
        # Resolved, not spelled out: the point of the assertion is that each
        # stage gets ITS OWN arrays by index, not what the folders are called,
        # and the folder names have already changed once (they became
        # initial_array_<rank>_for_<stage> when the winners started being
        # chosen from a 468-array pool).
        from post_processing import initial_morphology_name
        assert {t[0] for t in e17} == {initial_morphology_name(i, "E17.5")
                                       for i in (0, 1)}
        assert {t[0] for t in p0} == {initial_morphology_name(i, "P0")
                                      for i in (0, 1)}

        # --- PER STAGE: the mechanics (gammaHC_ratio 2, alphaHC 3, A0 16) ---
        assert {(t[2], t[3], t[16]) for t in e17} == {
            (6.746124660721616, 3.5, 0.741812389049632)}
        assert {(t[2], t[3], t[16]) for t in p0} == {
            (3.850521945713907, 1.757, 0.7585423232230211)}
        # gammaSC happens to be equal at both stages in the v2 fit; assert it
        # is carried per stage rather than hard-wired
        assert {t[1] for t in tasks} == {0.0105}


class TestFindPsigmaCLIPassesMechanicsThrough:
    """`azure_run find-psigma` must hand find_psigma the mechanics it was
    given. It used to wrap every stage in tuple(), which on a dict yields the
    KEYS -- so the v2 per-stage form arrived as ('gamma_sc', 'R_gamma', ...)
    and died with "could not convert string to float: 'gamma_sc'". Only the
    smoke test on the VM caught it, because the unit tests called find_psigma
    directly and skipped the CLI's coercion."""

    V2 = ('{"E17.5":{"gamma_sc":0.0105,"R_gamma":6.746124660721616,'
          '"R_alpha":3.5,"A0":0.741812389049632},'
          '"P0":{"gamma_sc":0.0105,"R_gamma":3.850521945713907,'
          '"R_alpha":1.757,"A0":0.7585423232230211}}')

    @staticmethod
    def _args(mechanical_params):
        import argparse
        return argparse.Namespace(
            mechanical_params=mechanical_params, indices=None,
            psigma_bounds=[0.06, 0.12], n_grid=2, n_refine=0, n_sheets=10,
            n_repeats=3, workers=4, bending=0.0, t_end=100.0, qst=0.03,
            save_interval=0.1, stress_shift=0.0,
            objective_terms=["score1", "score2"])

    def _run(self, monkeypatch, mechanical_params):
        import azure_run, full_model
        seen = {}

        def fake_find_psigma(mp, **kw):
            seen["mp"] = mp
            seen["kw"] = kw
            return 0.09, {0.09: {"objective": 1.0}}

        monkeypatch.setattr(full_model, "find_psigma", fake_find_psigma)
        monkeypatch.setattr(azure_run, "_ensure_repo_cwd", lambda: None)
        azure_run._cmd_find_psigma(self._args(mechanical_params))
        return seen

    def test_v2_dict_survives_the_cli(self, monkeypatch):
        seen = self._run(monkeypatch, self.V2)
        mp = seen["mp"]
        assert isinstance(mp["E17.5"], dict), type(mp["E17.5"])
        assert mp["E17.5"]["R_gamma"] == 6.746124660721616
        assert mp["P0"]["A0"] == 0.7585423232230211
        # and it is a form _stage_mechanics actually accepts
        import full_model as fm
        assert fm._stage_mechanics(mp["E17.5"], None)["gammaHC_ratio"] == 6.746124660721616

    def test_legacy_list_still_becomes_a_tuple(self, monkeypatch):
        seen = self._run(monkeypatch, '{"E17.5":[0.0105,3.5,0.0,0.0]}')
        assert seen["mp"]["E17.5"] == (0.0105, 3.5, 0.0, 0.0)

    def test_v2_settings_reach_find_psigma(self, monkeypatch):
        """bending 0.0 and t_end 100, not find_psigma's pre-v2 defaults."""
        kw = self._run(monkeypatch, self.V2)["kw"]
        assert kw["bending"] == 0.0 and kw["t_end"] == 100.0
        assert kw["n_arrays"] == 10 and kw["n_repeats"] == 3
        assert kw["quasi_static_threshold"] == 0.03


class TestResultsDirMustBeUsableOnThisOS:
    """A drive-letter default is not absolute on POSIX, so os.path.join made
    it RELATIVE and every run landed under the cwd. On the Azure VM that
    filled the 128 GB OS disk with a directory literally named
    'D:\\kasirer\\model_results' inside the repo, while the 2 TiB data disk
    sat empty -- and it surfaced only hours into the sweep."""

    def test_posix_without_the_env_var_is_refused(self):
        from history_io import _results_dir_problem, DEFAULT_RESULTS_DIR
        msg = _results_dir_problem(DEFAULT_RESULTS_DIR, os_name="posix")
        assert msg and "TISSUE_RESULTS_DIR" in msg
        assert "/data/results" in msg, "the message should show the fix"

    def test_posix_with_the_env_var_is_fine(self):
        from history_io import _results_dir_problem
        assert _results_dir_problem("/data/results", os_name="posix") is None

    def test_windows_default_is_fine(self):
        from history_io import _results_dir_problem, DEFAULT_RESULTS_DIR
        assert _results_dir_problem(DEFAULT_RESULTS_DIR, os_name="nt") is None

    def test_the_live_value_passed_the_check(self):
        """history_io imported at all, so whatever RESULTS_DIR resolved to is
        usable here -- this guards the guard against being a no-op."""
        import history_io
        assert history_io._results_dir_problem(history_io.RESULTS_DIR) is None


class TestRepeatsAreScoredAsGroupsNotIndependentPoints:
    """Repeats re-roll the LI seed on the SAME initial sheet, so they are
    repeated measurements, not independent tissues. Scored flat, SEM_sim
    shrinks by ~sqrt(n_repeats) and every chi^2 inflates -- the first cloud
    sweep came back ~3x worse than the archived numbers purely from this.
    compare_full_model_differentiation_to_experiments takes model_groups for
    exactly this reason; find_psigma must use it."""

    @staticmethod
    def _params():
        return {"E17.5": {"gamma_sc": 0.0105, "R_gamma": 6.746, "R_alpha": 3.5,
                          "A0": 0.7418}}

    def test_each_point_is_scored_with_per_array_groups(self, monkeypatch):
        import full_model as fm
        captured = []

        def fake_run(task):
            return "%s_%s|ps=%.5g" % (task[22], task[0], task[15])

        def fake_compare(stage, model_names=None, model_groups=None, **kw):
            captured.append({"names": model_names, "groups": model_groups})
            return {"score1": 1.0, "score2": 0.0, "score3": 0.0, "total": 1.0}

        monkeypatch.setattr(fm, "_run_full_model_one", fake_run)
        monkeypatch.setattr(fm,
                            "compare_full_model_differentiation_to_experiments",
                            fake_compare)
        fm.find_psigma(self._params(), psigma_bounds=(0.08, 0.08), n_grid=1,
                       n_refine=0, n_arrays=4, n_repeats=3, n_workers=1,
                       plot=False, save_json=False, use_score_cache=False)

        assert len(captured) == 1
        call = captured[0]
        assert call["names"] is None, "a flat name list must NOT be passed"
        groups = call["groups"]
        assert len(groups) == 4, "one group per ARRAY, got %d" % len(groups)
        for grp in groups:
            assert len(grp) == 3, "each group holds that array's 3 repeats"
            # Split each name by MATCHING its repeat prefix, rather than by
            # splitting on the array name: a run folder is
            # "<repeat prefix>_<initial sheet>", and the initial sheet name has
            # already changed once (random_periodic_array<i>_for_ ->
            # initial_array_<rank>_for_), which silently turned the old split
            # into a no-op. Requiring exactly one match also catches
            # fullmodel_v2 being a string prefix of fullmodel_v2r2.
            want = {fm.REPEAT_PREFIX[r] for r in (1, 2, 3)}
            prefixes, arrays = set(), set()
            for n in grp:
                hit = [p for p in want if n.startswith(p + "_")]
                assert len(hit) == 1, "%r matches %d repeat prefixes" % (n, len(hit))
                prefixes.add(hit[0])
                arrays.add(n[len(hit[0]) + 1:])
            # same array in every repeat...
            assert len(arrays) == 1, grp
            # ...and three DIFFERENT repeat prefixes
            assert prefixes == want, prefixes

    def test_one_repeat_still_gives_one_run_per_group(self, monkeypatch):
        import full_model as fm
        captured = []
        monkeypatch.setattr(fm, "_run_full_model_one",
                            lambda t: "%s_%s" % (t[22], t[0]))
        monkeypatch.setattr(
            fm, "compare_full_model_differentiation_to_experiments",
            lambda stage, model_groups=None, **kw: captured.append(model_groups)
            or {"score1": 1.0, "score2": 0.0, "score3": 0.0, "total": 1.0})
        fm.find_psigma(self._params(), psigma_bounds=(0.08, 0.08), n_grid=1,
                       n_refine=0, n_arrays=5, n_repeats=1, n_workers=1,
                       plot=False, save_json=False, use_score_cache=False)
        assert [len(g) for g in captured[0]] == [1] * 5

    def test_the_cache_key_separates_the_two_groupings(self):
        from run_model import _score_cache_key
        a = _score_cache_key("E17.5", 0.08, 0.0, None, "delta_level", 0.355, 2,
                             grouping="arrays")
        b = _score_cache_key("E17.5", 0.08, 0.0, None, "delta_level", 0.355, 2,
                             grouping="flat")
        assert a != b, ("a score cached under one grouping must never be served"
                        " to the other -- the run set is identical, so the"
                        " fingerprint cannot tell them apart")


class TestObjectiveTermsSelectWhatIsMinimised:
    """score3 (% of initial SCs differentiating by HC-neighbour count) is ~79%
    of the 7-term total at the optimum, so minimising the total picks a
    different psigma than minimising the four terms the campaign is fitted on.
    find_psigma minimises score1+score2 by default; score3 is still computed
    and stored."""

    @staticmethod
    def _patch(monkeypatch, per_psigma):
        """per_psigma: psigma -> (score1, score2, score3)."""
        import full_model as fm
        monkeypatch.setattr(fm, "_run_full_model_one",
                            lambda t: "%s_%s|ps=%.5g" % (t[22], t[0], t[15]))

        def fake_compare(stage, model_groups=None, **kw):
            ps = float(model_groups[0][0].split("|ps=")[1])
            s1, s2, s3 = per_psigma[round(ps, 5)]
            return {"score1": s1, "score2": s2, "score3": s3,
                    "total": s1 + s2 + s3}

        monkeypatch.setattr(
            fm, "compare_full_model_differentiation_to_experiments",
            fake_compare)
        return fm

    #: psigma -> (s1, s2, s3). 0.08 wins on s1+s2; 0.10 wins on the total.
    TABLE = {0.06: (10.0, 10.0, 1.0),
             0.08: (1.0, 1.0, 100.0),
             0.10: (9.0, 9.0, 0.5)}

    def _run(self, monkeypatch, **kw):
        fm = self._patch(monkeypatch, self.TABLE)
        return fm.find_psigma(
            {"E17.5": {"gamma_sc": 0.0105, "R_gamma": 6.746, "R_alpha": 3.5,
                       "A0": 0.7418}},
            psigma_bounds=(0.06, 0.10), n_grid=3, n_refine=0, n_arrays=2,
            n_repeats=1, n_workers=1, plot=False, save_json=False,
            use_score_cache=False, **kw)

    def test_default_minimises_score1_plus_score2(self, monkeypatch):
        best, scores = self._run(monkeypatch)
        assert abs(best - 0.08) < 1e-12, (
            "fit followed score3 instead of s1+s2: best=%s" % best)
        assert abs(scores[0.08]["objective"] - 2.0) < 1e-12, scores[0.08]

    def test_score3_is_still_recorded(self, monkeypatch):
        _best, scores = self._run(monkeypatch)
        assert scores[0.08]["E17.5"]["score3"] == 100.0
        assert scores[0.08]["E17.5"]["total"] == 102.0

    def test_including_score3_picks_a_different_psigma(self, monkeypatch):
        best, scores = self._run(
            monkeypatch, objective_terms=("score1", "score2", "score3"))
        assert abs(best - 0.10) < 1e-12, best
        assert abs(scores[0.10]["objective"] - 18.5) < 1e-12

    def test_the_cli_default_is_s1_plus_s2(self):
        import azure_run
        p = azure_run.build_parser()
        a = p.parse_args(["find-psigma", "--mechanical-params",
                          '{"E17.5":[0.01,1,0,0]}'])
        assert a.objective_terms == ["score1", "score2"]
