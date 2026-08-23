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
    """``post_processing.drop_corrupted_snapshots`` removes snapshots
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
        from post_processing import drop_corrupted_snapshots
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
        from post_processing import drop_corrupted_snapshots
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
        from post_processing import drop_corrupted_snapshots
        path = str(tmp_path / "h.hf5")
        self._write(path, {0.0: [1.0, 1.0], 1.0: [1.0, 1.0]})
        dropped = drop_corrupted_snapshots(path)
        assert dropped == [], "a clean archive should drop nothing"

    def test_idempotent(self, tmp_path):
        from post_processing import drop_corrupted_snapshots
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
    from post_processing import RESULTS_DIR
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
    ``post_processing.save_data_of_a_given_time_point``:

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
        from post_processing import _shorten_gif_output, _MAX_GIF_PATH_LEN
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
        import post_processing as pp

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

    def test_find_uses_per_array_threshold(self, tmp_path, monkeypatch):
        import run_model
        import bayesian_optimization as bo
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(run_model, "RESULTS_DIR", str(tmp_path / "results"))
        # Two initial sheets, each with its OWN saved threshold.
        for name, thr in (("sheetA", 0.30), ("sheetB", 0.45)):
            folder = tmp_path / "results" / name
            folder.mkdir(parents=True)
            np.save(folder / "threshold.npy", np.asarray(thr))

        seen = []
        monkeypatch.setattr(run_model, "_evaluate_mechanics_for_sheet",
                            lambda task: seen.append(task) or
                            {"hc_roundness": np.array([0.5]),
                             "sc_roundness": np.array([0.5]), "hc_ablation": None,
                    "sc_ablation": None,
                    # v2 scores the HC/SC RATIOS; a fake returning only
                    # the absolute terms leaves every scored term empty.
                    "roundness_ratio": np.array([1.0]),
                    "ablation_ratio": None,
                    "shrinkage": np.array([7.5])})
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
            use_saved_threshold=True, type_by="delta_level")

        # task = (..., initial(3), ..., fix_threshold(7), type_by(8), stall...)
        thr_by_sheet = {t[3]: t[7] for t in seen}
        assert thr_by_sheet == {"sheetA": 0.30, "sheetB": 0.45}
        assert all(t[8] == "delta_level" for t in seen)


class TestAblationAreaChangeUnionFix:
    """Regression for the ablation term in compare_model_mechanics_to_experiments.

    ``calc_area_change_after_ablation`` accumulated HC/SC neighbour ids with a
    broken ``_, x = np.union1d(...)`` — np.union1d returns a SINGLE array, so
    unpacking it as a 2-tuple crashed with "not enough values to unpack" the
    moment a union had !=2 elements (and silently produced garbage otherwise).
    It must union the IDS correctly and survive a single-element union."""

    def test_single_element_union_and_accumulation(self, monkeypatch):
        import post_processing as pp

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


# --------------------------------------------------------------------------- #
# Loaded delta threshold drives the simulation's atoh_sensitivity              #
# --------------------------------------------------------------------------- #
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
        # atoh is the increasing Hill of delta with half-max AT the threshold.
        np.testing.assert_allclose(atoh, delta ** 3 / (A ** 3 + delta ** 3), atol=1e-9)
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
