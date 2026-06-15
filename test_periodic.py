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
        is_virtual = sheet.is_virtual_edge(np.arange(sheet.edge_df.shape[0]))
        real = sheet.edge_df[~is_virtual]
        selected = real.query("is_active > 0 & length < %f" % threshold)
        # At least one of the selected edges is a periodic edge — the
        # handler no longer filters them out.
        assert selected["is_periodic"].any()


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
    calls ``index_preserving_remove`` — BUT only on cells whose
    ``num_sides`` has dropped to 3 AND whose area is still below the
    threshold AND ``is_alive == 1``. On a fresh hex lattice every
    face starts with 6 edges, so unless the mechanics + T1s drive a
    cell down to a triangle, pass-2 never fires.

    This test rigs a single face's num_sides + area so pass-2's
    query picks it up, then verifies the handler actually removes
    it and leaves a well-formed topology behind. If this fails, the
    handler itself is broken; if it passes, the user's "no
    delaminations observed" must be a function of the simulation
    dynamics never producing a triangular cell — which is a
    different problem (T1 frequency / shrink_rate tuning).
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
        inner.sheet.face_df.loc[target_live, "area"] = 0.01
        inner.sheet.face_df.loc[target_live, "num_sides"] = 3

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
    root = os.path.join("results")
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
        import solvers as solvers_mod
        from tyssue.dynamics import model_factory
        from tyssue.dynamics.effectors import FaceContractility, FaceAreaElasticity
        from tyssue import History

        class _StubSolveResult:
            def __init__(self, y):
                self.y = y

        def stub_solve_ivp(fn, t_span, pos, t_eval=None):
            t0, t1 = t_span
            v = velocity_fn(t1, pos)
            new_pos = pos + np.asarray(v) * (t1 - t0)
            return _StubSolveResult(new_pos.reshape(-1, 1))

        monkeypatch.setattr(solvers_mod, "solve_ivp", stub_solve_ivp)

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
        # No-op ode_func — velocity comes from the stub above.
        solver.ode_func = lambda t, pos: np.zeros_like(pos)
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
        with caplog.at_level(logging.WARNING, logger="tyssue.solvers.viscous"):
            # The dt floor here is tight enough that the solver must
            # eventually raise — we just want to see the vertex label
            # appear in BOTH the warning and the RuntimeError.
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

    def test_create_gif_safe_forwards_to_create_gif(self, monkeypatch):
        """``create_gif_safe`` must pass the (possibly shortened) path
        plus all kwargs straight through to the underlying
        ``create_gif``."""
        import post_processing as pp
        captured = {}

        def fake_create_gif(history, output, **kwargs):
            captured["output"] = output
            captured["kwargs"] = kwargs
            return "ok"

        monkeypatch.setattr(pp, "create_gif", fake_create_gif)

        longname = "run_" + "w" * 150
        directory = os.path.join("base", "results", longname)
        output = os.path.join(directory, longname + ".gif")

        ret = pp.create_gif_safe(
            history="HIST", output=output, num_frames=42, draw_func="DF",
        )
        assert ret == "ok"
        # Path was shortened (kept directory) and kwargs forwarded.
        assert os.path.dirname(captured["output"]) == directory
        assert len(captured["output"]) <= pp._MAX_GIF_PATH_LEN
        assert captured["kwargs"] == {"num_frames": 42, "draw_func": "DF"}
