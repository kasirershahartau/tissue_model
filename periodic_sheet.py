import logging
import numpy as np
import pandas as pd
from tyssue import Sheet
from tyssue.core.sheet import get_opposite as _tyssue_get_opposite
from tyssue.geometry.planar_geometry import PlanarGeometry
from tyssue.config.draw import sheet_spec
from tyssue.utils.utils import spec_updater
from tyssue.generation import hexa_grid2d, from_2d_voronoi
from tyssue.config.geometry import planar_spec
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)

class PeriodicBoundarySheet(Sheet):
    """
    A Sheet subclass that supports 2D periodic boundary conditions.
    """

    def __init__(self, identifier, datasets, specs=None, coords=None):
        super().__init__(identifier, datasets, specs, coords)

        # Mark periodic edges if present.
        # NOTE: tyssue already provides dx, dy on edges (= trgt - srce displacement),
        # so we don't (re)initialize them here.
        if "is_periodic" not in self.edge_df.columns:
            self.edge_df["is_periodic"] = False

    @staticmethod
    def refresh_is_periodic(Lx, Ly, vert_df, edge_df):
        """Recompute the ``is_periodic`` / ``at_x_boundary`` / ``at_y_boundary``
        edge columns from canonical (wrapped into [0, L)) vertex coordinates.

        An edge is flagged periodic iff its srce↔trgt distance in canonical
        coords exceeds half a period along x or y — i.e. the shortest path
        between the two vertices goes through a periodic boundary.

        Idempotent. Safe to call any time after vert_df is in (or near) the
        canonical box."""
        tol = 1e-9
        vx = vert_df["x"]
        vy = vert_df["y"]
        idx_dtype = vx.index.dtype
        srce_lbl = edge_df["srce"].to_numpy()
        trgt_lbl = edge_df["trgt"].to_numpy()
        # Align label dtype so reindex matches (see set_opposite_periodic).
        if srce_lbl.dtype != idx_dtype:
            srce_lbl = srce_lbl.astype(idx_dtype)
        if trgt_lbl.dtype != idx_dtype:
            trgt_lbl = trgt_lbl.astype(idx_dtype)
        # Vectorized label-based gather instead of a per-edge Python loop.
        sx = vx.reindex(srce_lbl).to_numpy()
        sy = vy.reindex(srce_lbl).to_numpy()
        tx = vx.reindex(trgt_lbl).to_numpy()
        ty = vy.reindex(trgt_lbl).to_numpy()
        wraps_x = np.abs(tx - sx) > Lx / 2 - tol
        wraps_y = np.abs(ty - sy) > Ly / 2 - tol
        edge_df["at_x_boundary"] = wraps_x
        edge_df["at_y_boundary"] = wraps_y
        edge_df["is_periodic"] = wraps_x | wraps_y

    @staticmethod
    def set_opposite_periodic(Lx, Ly, vert_df, edge_df):
        """
        Stitch periodic-boundary opposites for edges where ``opposite < 0``.

        Interior edges (whose opposite was already matched by vertex labels)
        are LEFT ALONE.

        Note: the ``is_periodic`` flag is intentionally NOT set here. It is
        recomputed from minimum-image criteria by
        :meth:`PeriodicPlanarGeometry.update_dcoords` on the next geometry
        update — the single source of truth.

        Uses label-based lookups so it is safe even when ``vert_df`` is not
        contiguously indexed.
        """
        PREC = 6  # rounding precision for key matching

        # Early-out: this routine only fills in opposites for edges whose
        # opposite is still unset (< 0). Once the sheet's opposites are all
        # matched (the steady state during pure-motion steps), there is
        # nothing to stitch and the rest of this function — notably the
        # O(E) coordinate-key lookup dict — would be built for nothing.
        # Skipping it then is bit-for-bit identical (the unstitched loop
        # below would simply not execute). New unset opposites only appear
        # after a topology change (e.g. a virtual vertex added on a
        # boundary edge), and those calls fall through and do the work.
        if not (edge_df["opposite"].to_numpy() < 0).any():
            return

        # Wrap into [0, L). Series keep label index, so .loc lookups are safe
        # regardless of whether the dataframe was reset_index'd.
        vx_w = (vert_df["x"] % Lx)
        vy_w = (vert_df["y"] % Ly)

        # Snap near-boundary floating-point residuals to 0
        vx_w = vx_w.where(vx_w <= Lx * (1.0 - 1e-10), 0.0)
        vy_w = vy_w.where(vy_w <= Ly * (1.0 - 1e-10), 0.0)
        vx_w = vx_w.where(np.abs(vx_w) >= 1e-10, 0.0)
        vy_w = vy_w.where(np.abs(vy_w) >= 1e-10, 0.0)

        # Per-edge wrapped endpoint coordinates, rounded to the matching
        # precision. Built vectorially (one np.round over the whole edge
        # table) rather than with a per-edge Python round() loop — same
        # keys, far less overhead since this runs on every time step.
        eids = edge_df.index.to_numpy()
        srce_lbl = edge_df["srce"].to_numpy()
        trgt_lbl = edge_df["trgt"].to_numpy()
        # ``reindex`` matches labels by exact dtype — if srce/trgt drifted
        # to float (a topology op can upcast the column) while the vertex
        # index is int, the lookup would silently return all-NaN. Vertex
        # labels are whole numbers, so align the dtypes first.
        idx_dtype = vx_w.index.dtype
        if srce_lbl.dtype != idx_dtype:
            srce_lbl = srce_lbl.astype(idx_dtype)
        if trgt_lbl.dtype != idx_dtype:
            trgt_lbl = trgt_lbl.astype(idx_dtype)
        sxr = np.round(vx_w.reindex(srce_lbl).to_numpy(), PREC).tolist()
        syr = np.round(vy_w.reindex(srce_lbl).to_numpy(), PREC).tolist()
        txr = np.round(vx_w.reindex(trgt_lbl).to_numpy(), PREC).tolist()
        tyr = np.round(vy_w.reindex(trgt_lbl).to_numpy(), PREC).tolist()

        # Build lookup: (wx_srce, wy_srce, wx_trgt, wy_trgt) → edge_id
        lookup = {}
        for i in range(len(eids)):
            lookup[(sxr[i], syr[i], txr[i], tyr[i])] = eids[i]

        opp_arr = edge_df["opposite"].to_numpy()
        still_missing = []

        for i in np.where(opp_arr < 0)[0]:
            eid = eids[i]
            if edge_df.at[eid, "opposite"] >= 0:
                continue  # already stitched by its partner this pass
            # Opposite direction: t_wrapped → s_wrapped
            key = (txr[i], tyr[i], sxr[i], syr[i])
            if key in lookup:
                opp = lookup[key]
                edge_df.at[eid, "opposite"] = opp
                edge_df.at[opp, "opposite"] = eid
            else:
                still_missing.append(eid)

        if still_missing:
            # Diagnostic: report unstitched edges so you can tune PREC.
            # (Reduce PREC to use FEWER decimals → more tolerant matching.)
            logger.warning(
                "%d edges still unstitched after periodic-pair matching. "
                "Float-point precision is borderline; consider lowering PREC.",
                len(still_missing),
            )
            pos_of_eid = {eids[i]: i for i in range(len(eids))}
            for eid in still_missing:
                i = pos_of_eid[eid]
                logger.debug(
                    "  e%s: wrap_src=(%.8f,%.8f) wrap_tgt=(%.8f,%.8f)",
                    eid, sxr[i], syr[i], txr[i], tyr[i],
                )

    # NOTE: VirtualSheet.planar_periodic_sheet_2d is the only construction
    # entry point. Use it instead of constructing a PeriodicBoundarySheet
    # directly — VirtualSheet wires up Lx/Ly, the periodic-aware geometry,
    # and the periodic get_opposite override.

    # @staticmethod
    # def generate_periodic_hex_lattice(base_sheet):
    #     vert_df = base_sheet.vert_df.copy()
    #     edge_df = base_sheet.edge_df.copy()
    #     face_df = base_sheet.face_df.copy()
    #
    #     # Add periodic columns
    #     for col, default in [("is_periodic", False), ("dx", 0.0), ("dy", 0.0)]:
    #         if col not in edge_df.columns:
    #             edge_df[col] = default
    #
    #     Lx = vert_df["x"].max() - vert_df["x"].min()
    #     Ly = vert_df["y"].max() - vert_df["y"].min()
    #     xmin, xmax = vert_df["x"].min(), vert_df["x"].max()
    #     ymin, ymax = vert_df["y"].min(), vert_df["y"].max()
    #
    #     # Use a generous tolerance based on the smallest edge length
    #     tol = base_sheet.edge_df["length"].min() * 0.3
    #
    #     # --- Identify boundary edges and assign dx/dy ---
    #     for eid, row in edge_df.iterrows():
    #         sv = vert_df.loc[row["srce"]]
    #         tv = vert_df.loc[row["trgt"]]
    #         on_left = (sv["x"] < xmin + tol) and (tv["x"] < xmin + tol)
    #         on_right = (sv["x"] > xmax - tol) and (tv["x"] > xmax - tol)
    #         on_bot = (sv["y"] < ymin + tol) and (tv["y"] < ymin + tol)
    #         on_top = (sv["y"] > ymax - tol) and (tv["y"] > ymax - tol)
    #         if on_left:
    #             edge_df.at[eid, "is_periodic"] = True
    #             edge_df.at[eid, "dx"] = +Lx
    #         elif on_right:
    #             edge_df.at[eid, "is_periodic"] = True
    #             edge_df.at[eid, "dx"] = -Lx
    #         if on_bot:
    #             edge_df.at[eid, "is_periodic"] = True
    #             edge_df.at[eid, "dy"] = +Ly
    #         elif on_top:
    #             edge_df.at[eid, "is_periodic"] = True
    #             edge_df.at[eid, "dy"] = -Ly
    #
    #     # --- Stitch opposite entries for periodic edges ---
    #     # Build a lookup: normalized (srce_pos, trgt_pos) → edge index
    #     # using minimum-image-wrapped coordinates
    #     def wrap(x, y, Lx, Ly, xmin, ymin):
    #         return ((x - xmin) % Lx + xmin,
    #                 (y - ymin) % Ly + ymin)
    #
    #     lookup = {}
    #     for eid, row in edge_df.iterrows():
    #         sx = vert_df.loc[row["srce"], "x"]
    #         sy = vert_df.loc[row["srce"], "y"]
    #         tx = vert_df.loc[row["trgt"], "x"]
    #         ty = vert_df.loc[row["trgt"], "y"]
    #         ws = wrap(sx, sy, Lx, Ly, xmin, ymin)
    #         wt = wrap(tx, ty, Lx, Ly, xmin, ymin)
    #         key = (round(ws[0], 5), round(ws[1], 5),
    #                round(wt[0], 5), round(wt[1], 5))
    #         lookup[key] = eid
    #
    #     for eid, row in edge_df.iterrows():
    #         if edge_df.at[eid, "opposite"] >= 0:
    #             continue  # already stitched internally
    #         sx = vert_df.loc[row["srce"], "x"]
    #         sy = vert_df.loc[row["srce"], "y"]
    #         tx = vert_df.loc[row["trgt"], "x"]
    #         ty = vert_df.loc[row["trgt"], "y"]
    #         # opposite direction, wrapped
    #         wt = wrap(tx, ty, Lx, Ly, xmin, ymin)
    #         ws = wrap(sx, sy, Lx, Ly, xmin, ymin)
    #         key = (round(wt[0], 5), round(wt[1], 5),
    #                round(ws[0], 5), round(ws[1], 5))
    #         if key in lookup:
    #             edge_df.at[eid, "opposite"] = lookup[key]
    #
    #     # Wrap all vertices into the canonical domain.
    #     # xmin = ymin = 0 by construction (hexa_grid2d starts cy=0, cx=0).
    #     vert_df["x"] = vert_df["x"] % Lx
    #     vert_df["y"] = vert_df["y"] % Ly
    #
    #     return vert_df, edge_df, face_df

    @staticmethod
    def generate_periodic_hex_lattice(nx, ny, distx=1.0, disty=1.0):
        """
        Build a periodic hexagonal lattice via periodic Voronoi tessellation.

        Tiles the nx×ny seeds in a 3×3 supercell with EXACT periodic offsets,
        runs Voronoi on all 9N seeds, then extracts the N central cells.
        This guarantees boundary vertices on opposite sides are exact periodic
        copies → coordinate-wrap stitching is exact.

        Requires ny even for the hex row-offset pattern to tile rectangularly.
        """
        if ny % 2 != 0:
            raise ValueError(
                f"ny must be even for rectangular periodic tiling, got ny={ny}"
            )
        # The hex Voronoi degenerates when one spacing is much larger than
        # the other: cells stretch into collinear strips and produce zero-
        # area Voronoi regions. Empirically anything outside ~[0.5, 2] of
        # the regular-hex ratio (disty ≈ distx * sqrt(3)/2 ≈ 0.866 * distx)
        # gives bad cells.
        ratio = disty / distx
        if not (0.5 <= ratio <= 2.0):
            raise ValueError(
                f"disty/distx={ratio:.3f} is outside the safe range [0.5, 2.0]; "
                "hex Voronoi cells become degenerate. Use distx ≈ disty "
                "(or disty ≈ distx * sqrt(3)/2 for regular hexagons)."
            )

        # --- 1. Central seeds ---
        centers = hexa_grid2d(nx, ny, distx, disty)
        N = len(centers)  # == nx * ny

        # hexa_grid2d(nx, ny, distx, disty) produces, with cy,cx = mgrid[0:ny, 0:nx]:
        #   centers[:, 0] = cx * distx  (tyssue "x", with +0.5 hex offset on even cy rows)
        #   centers[:, 1] = cy * disty  (tyssue "y")
        # Exact rectangular periods:
        Lx = nx * distx  # period in x (column direction in the cx grid)
        Ly = ny * disty  # period in y (row direction in the cy grid)

        # --- 2. 3×3 tile with exact periodic shifts ---
        copies = []
        for dix in (-1, 0, 1):
            for diy in (-1, 0, 1):
                copies.append(centers + np.array([dix * Lx, diy * Ly]))
        # Loop order: (dix,diy) = (-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)
        # Central (0,0) copy is index 4
        all_centers = np.vstack(copies)  # 9N × 2
        central_start = 4 * N

        # --- 3. Voronoi + full sheet ---
        vor = Voronoi(all_centers)
        datasets = from_2d_voronoi(vor)
        full_sheet = Sheet("_periodic_tmp", datasets, specs=planar_spec(), coords=["x", "y"])
        PlanarGeometry.update_all(full_sheet)  # populates face centroids
        full_sheet.get_opposite()
        # --- 4. Select the N central faces by centroid position ---
        cx_lo, cx_hi = centers[:, 0].min(), centers[:, 0].max()
        cy_lo, cy_hi = centers[:, 1].min(), centers[:, 1].max()
        eps = 0.1 * min(distx, disty)

        keep_mask = (
                full_sheet.face_df["x"].between(cx_lo - eps, cx_hi + eps) &
                full_sheet.face_df["y"].between(cy_lo - eps, cy_hi + eps)
        )
        assert keep_mask.sum() == N, (
            f"Expected {N} central faces, found {keep_mask.sum()}. "
            "Check that ny is even and distx/disty are consistent."
        )

        # --- 5. Extract: keep only central faces, edges, vertices ---
        face_df = full_sheet.face_df[keep_mask].copy()
        edge_df = full_sheet.edge_df[
            full_sheet.edge_df["face"].isin(face_df.index)
        ].copy()
        used_verts = pd.concat([edge_df["srce"], edge_df["trgt"]]).unique()
        vert_df = full_sheet.vert_df.loc[used_verts].copy()

        # Reindex everything to 0-based contiguous integers
        vert_map = {old: new for new, old in enumerate(vert_df.index)}
        edge_map = {old: new for new, old in enumerate(edge_df.index)}
        face_map = {old: new for new, old in enumerate(face_df.index)}

        vert_df.index = pd.RangeIndex(len(vert_df));
        vert_df.index.name = "vert"
        edge_df.index = pd.RangeIndex(len(edge_df));
        edge_df.index.name = "edge"
        face_df.index = pd.RangeIndex(len(face_df));
        face_df.index.name = "face"

        edge_df["srce"] = edge_df["srce"].map(vert_map)
        edge_df["trgt"] = edge_df["trgt"].map(vert_map)
        edge_df["face"] = edge_df["face"].map(face_map)

        # Remap internal opposites; boundary ones (not in edge_map) → -1
        edge_df["opposite"] = edge_df["opposite"].apply(
            lambda v: edge_map[v] if (v >= 0 and v in edge_map) else -1
        )

        # --- 6. Add periodic metadata columns ---
        edge_df["is_periodic"] = False
        edge_df["dx"] = 0.0
        edge_df["dy"] = 0.0

        # --- 7. Stitch periodic boundaries
        PeriodicBoundarySheet.set_opposite_periodic(Lx, Ly, vert_df, edge_df)

        # --- 7. Wrap all vertices into the canonical domain ---
        # xmin = ymin = 0 by construction (hexa_grid2d starts cy=0, cx=0).
        vert_df["x"] = vert_df["x"] % Lx
        vert_df["y"] = vert_df["y"] % Ly

        return vert_df, edge_df, face_df

    @staticmethod
    def periodic_sheet_view(sheet, coords=["x", "y"], ax=None, mode="2D", **draw_specs_kw):
        """
        Draw a periodic sheet correctly by unfolding each face polygon
        so no edge spans more than half the domain.

        Notes
        -----
        ``parse_face_specs`` and ``_parse_edge_specs`` expect the
        FACE-level and EDGE-level sub-spec dicts (the ones with a
        top-level ``"color"`` key) — NOT the full sheet_spec dict (the
        one whose keys are ``"face"``, ``"edge"``, ``"vert"``, ...).
        The previous version of this function passed the whole
        ``draw_specs`` dict in, so ``parse_face_specs`` looked for a
        ``"color"`` key at the top level, found nothing, returned
        ``{}``, and the resulting ``PatchCollection`` had no
        ``facecolors`` set — matplotlib then rendered every cell in
        its default tab:blue. Inside ``create_gif`` the sheet that
        came out of ``HistoryHdf5.retrieve`` had ``periodic=False``
        (the VirtualSheet default), so ``get_sheet_view_method``
        dispatched to tyssue's stock ``sheet_view`` → ``draw_face``,
        which correctly hands the FACE sub-spec to
        ``parse_face_specs``. That's why the gif looked great while
        the live-sheet plots in ``periodic_tests.py`` were all blue.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection, LineCollection
        from tyssue.draw.plt_draw import parse_face_specs, _parse_edge_specs

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        draw_specs = sheet_spec()
        spec_updater(draw_specs, draw_specs_kw)

        # Pass the FACE / EDGE sub-spec dicts, not the whole
        # ``draw_specs``. See the docstring above for why this matters.
        face_spec = parse_face_specs(draw_specs["face"], sheet)
        _, edge_spec = _parse_edge_specs(draw_specs["edge"], sheet)

        Lx = sheet.Lx
        Ly = sheet.Ly

        # Label-indexed series so .loc lookups work even when vert_df
        # is not contiguously indexed.
        vx = sheet.vert_df["x"]
        vy = sheet.vert_df["y"]

        patches = []
        edge_segs = []  # line segments for edges
        for face_id in sheet.face_df.index:
            edges = sheet.edge_df[sheet.edge_df["face"] == face_id]
            if "order" in edges.columns:
                edges = edges.sort_values("order")
            srce_ids = edges["srce"].to_numpy()

            xs = vx.loc[srce_ids].to_numpy().copy()
            ys = vy.loc[srce_ids].to_numpy().copy()

            # Unfold: walk around the polygon, keeping each vertex within
            # half a period of the previous one.
            for i in range(1, len(xs)):
                while xs[i] - xs[i - 1] > Lx / 2: xs[i] -= Lx
                while xs[i] - xs[i - 1] < -Lx / 2: xs[i] += Lx
                while ys[i] - ys[i - 1] > Ly / 2: ys[i] -= Ly
                while ys[i] - ys[i - 1] < -Ly / 2: ys[i] += Ly

            patches.append(Polygon(np.column_stack([xs, ys]), closed=True))

            n = len(xs)
            for i in range(n):
                j = (i + 1) % n
                edge_segs.append([(xs[i], ys[i]), (xs[j], ys[j])])

        ax.add_collection(PatchCollection(patches, match_original=False,
                                          **face_spec))
        lc = LineCollection(edge_segs, **edge_spec)
        ax.add_collection(lc)
        ax.set_xlim(-0.05 * Lx, 1.05 * Lx)
        ax.set_ylim(-0.05 * Ly, 1.05 * Ly)
        ax.set_aspect("equal")
        return fig, ax

class PeriodicPlanarGeometry(PlanarGeometry):
    """Planar geometry that supports 2D periodic boundary conditions.

    For periodic sheets we:
      1) wrap every vertex into [0, Lx) x [0, Ly) (canonical domain),
      2) walk around each face in edge ``order`` and "unfold" successive
         vertices so no consecutive pair is separated by more than half
         a period in either direction,
      3) write the unfolded positions back to sx, sy, tx, ty so the
         standard tyssue centroid / area / normal updates Just Work,
      4) mark edges whose canonical srce and trgt differ by more than
         half a period as ``is_periodic``.

    The per-face walk is the same construction used by
    :meth:`PeriodicBoundarySheet.periodic_sheet_view` for drawing, so
    the visual and geometric representations stay consistent.
    """

    @classmethod
    def update_dcoords(cls, sheet):
        if not getattr(sheet, "periodic", False):
            super().update_dcoords(sheet)
            return

        Lx = float(sheet.Lx)
        Ly = float(sheet.Ly)

        # 1) Re-wrap every vertex into the canonical [0, L) box.
        sheet.vert_df["x"] = sheet.vert_df["x"] % Lx
        sheet.vert_df["y"] = sheet.vert_df["y"] % Ly

        # NOTE: the per-face unfold below recomputes sx/sy/tx/ty for EVERY
        # edge and writes them all back (idx_arr spans the whole edge table),
        # and nothing reads those columns in between. So the standard
        # upcast_srce/upcast_trgt that used to populate them here was pure
        # overhead — dropped. We only ensure the columns exist so the
        # .loc write-back works on a freshly built sheet.
        for _c in ("sx", "sy", "tx", "ty"):
            if _c not in sheet.edge_df.columns:
                sheet.edge_df[_c] = 0.0

        # 2) Per-face unfold (uses edge `order` if present, falls back to
        #    edge_df row order otherwise).
        if "order" not in sheet.edge_df.columns:
            sheet.edge_df["order"] = sheet.edge_df.groupby("face").cumcount() + 1

        # Stable per-face walk via groupby + apply would be O(Ne) but slow
        # in pure python; do it inline once.
        ordered = sheet.edge_df[["face", "order", "srce", "trgt"]].sort_values(
            ["face", "order"]
        )
        face_arr = ordered["face"].to_numpy()
        srce_arr = ordered["srce"].to_numpy()
        idx_arr = ordered.index.to_numpy()
        vx = sheet.vert_df["x"]
        vy = sheet.vert_df["y"]
        xs_all = vx.loc[srce_arr].to_numpy().astype(float)
        ys_all = vy.loc[srce_arr].to_numpy().astype(float)

        # Walk: for every contiguous run with the same face id, unwrap each
        # subsequent vertex to within ±L/2 of the previous.
        #
        # Fast path: the inner per-edge walk only ever changes a coordinate
        # when some consecutive srce pair within the face is separated by
        # more than half a period — i.e. the cell straddles a periodic
        # boundary. For every interior face the walk is a guaranteed no-op
        # (if no RAW consecutive delta exceeds L/2, then by induction the
        # live-recomputed delta never does either, since nothing gets
        # shifted). So we precompute, vectorially, which positions have a
        # large intra-face step and skip the Python walk entirely for runs
        # that have none. This is bit-for-bit identical to walking every
        # face — it just avoids the per-edge Python cost for the majority
        # of (non-wrapping) faces.
        n = len(face_arr)
        if n:
            same_face = np.zeros(n, dtype=bool)
            same_face[1:] = face_arr[1:] == face_arr[:-1]
            needs_walk = same_face.copy()
            needs_walk[1:] &= (
                (np.abs(xs_all[1:] - xs_all[:-1]) > Lx / 2)
                | (np.abs(ys_all[1:] - ys_all[:-1]) > Ly / 2)
            )
            start = 0
            for i in range(1, n + 1):
                if i == n or face_arr[i] != face_arr[start]:
                    if needs_walk[start:i].any():
                        for j in range(start + 1, i):
                            dx = xs_all[j] - xs_all[j - 1]
                            if dx > Lx / 2:
                                xs_all[j] -= Lx
                            elif dx < -Lx / 2:
                                xs_all[j] += Lx
                            dy = ys_all[j] - ys_all[j - 1]
                            if dy > Ly / 2:
                                ys_all[j] -= Ly
                            elif dy < -Ly / 2:
                                ys_all[j] += Ly
                    start = i

        # tx_face_i = sx_face_(i+1) (cyclic) for each face's contiguous run.
        # CRITICAL: the per-face walk above can accumulate a NET drift of
        # k*Lx (k integer) across an entire perimeter — typically when a
        # cell crosses the periodic boundary and contains long edges
        # whose endpoints happen to lie on opposite sides of the wrap.
        # The closing edge therefore needs its tx wrapped back to be
        # within ±L/2 of its sx, using the same min-image rule we apply
        # to all other consecutive srces. Without this fix the closing
        # edge's stored ``length`` ends up as ~k*Lx instead of its real
        # value (≈the opposite edge's length), which then drives the
        # corresponding vertex to take a huge step in the next ODE call
        # and trips the adaptive-dt safety net.
        tx_all = xs_all.copy()
        ty_all = ys_all.copy()
        if len(face_arr):
            start = 0
            for i in range(1, len(face_arr) + 1):
                if i == len(face_arr) or face_arr[i] != face_arr[start]:
                    # tx[j] = xs[j+1] for j in [start, i-1)
                    tx_all[start : i - 1] = xs_all[start + 1 : i]
                    ty_all[start : i - 1] = ys_all[start + 1 : i]
                    # tx[i-1] closes the loop back to xs[start], but
                    # xs[start] is in the walk's INITIAL frame while
                    # xs[i-1] is in the (possibly drifted) FINAL frame.
                    # Apply min-image to the closure so the last edge's
                    # length stays consistent with its actual geometry.
                    closure_dx = xs_all[start] - xs_all[i - 1]
                    closure_dy = ys_all[start] - ys_all[i - 1]
                    closure_dx -= Lx * round(closure_dx / Lx)
                    closure_dy -= Ly * round(closure_dy / Ly)
                    tx_all[i - 1] = xs_all[i - 1] + closure_dx
                    ty_all[i - 1] = ys_all[i - 1] + closure_dy
                    start = i

        # Write back, preserving the original edge_df row order. Assign the
        # stacked numpy array directly (positional alignment with idx_arr) —
        # building an intermediate DataFrame just to call ``.to_numpy()`` on
        # it was pure overhead.
        sheet.edge_df.loc[idx_arr, ["sx", "sy", "tx", "ty"]] = np.column_stack(
            (xs_all, ys_all, tx_all, ty_all)
        )

        # 3) Recompute dx, dy from the unfolded positions
        sheet.edge_df["dx"] = sheet.edge_df["tx"] - sheet.edge_df["sx"]
        sheet.edge_df["dy"] = sheet.edge_df["ty"] - sheet.edge_df["sy"]

        # 4) Mark periodic edges (single source of truth for the flag).
        PeriodicBoundarySheet.refresh_is_periodic(
            Lx, Ly, sheet.vert_df, sheet.edge_df
        )

    @classmethod
    def face_projected_pos(cls, sheet, face, psi):
        """Override so that for a periodic mother face the projection
        uses the per-face *unfolded* vertex positions (read from edge_df
        sx, sy), not the canonical vert_df positions which would put
        opposite sides of a boundary-crossing cell on different sides of
        the projected centroid and break ``get_division_edges``."""
        if not getattr(sheet, "periodic", False):
            return super().face_projected_pos(sheet, face, psi)

        m_edges = sheet.edge_df[sheet.edge_df["face"] == face]
        # Each vertex of the mother face appears once as srce of one of
        # its edges. The (sx, sy) of that edge is the unfolded position.
        unfolded_x = pd.Series(
            m_edges["sx"].to_numpy(), index=m_edges["srce"].to_numpy()
        )
        unfolded_y = pd.Series(
            m_edges["sy"].to_numpy(), index=m_edges["srce"].to_numpy()
        )
        face_x = float(unfolded_x.mean())
        face_y = float(unfolded_y.mean())

        rot_pos = sheet.vert_df[sheet.coords].copy().astype(float)
        rot_pos.loc[unfolded_x.index, "x"] = unfolded_x.to_numpy()
        rot_pos.loc[unfolded_y.index, "y"] = unfolded_y.to_numpy()
        cos_p, sin_p = np.cos(psi), np.sin(psi)
        rx = (rot_pos["x"] - face_x) * cos_p - (rot_pos["y"] - face_y) * sin_p
        ry = (rot_pos["x"] - face_x) * sin_p + (rot_pos["y"] - face_y) * cos_p
        rot_pos["x"] = rx
        rot_pos["y"] = ry
        return rot_pos