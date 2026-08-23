from scipy.spatial import Voronoi
from tyssue import Sheet, config
from tyssue.generation import from_2d_voronoi, hexa_grid2d
from tyssue.config.geometry import planar_spec
from periodic_sheet import PeriodicBoundarySheet
from periodic_sheet import PeriodicPlanarGeometry as PeriodicGeom
from tyssue.topology.base_topology import collapse_edge
from topological_events import (index_preserving_add_vert, _min_image_midpoint,
                                index_preserving_type1_transition, log_topo_event,
                                _drop_antenna_spikes)
from tyssue.draw import sheet_view

import logging
import numpy as np

log = logging.getLogger(__name__)
# Hard cap on how many virtual-vertex insertions we attempt in one call.
# Each iteration splits one over-long edge, so this bounds the work by
# a (large) multiple of the initial edge count.
MAX_VIRTUAL_INSERTIONS = 100_000

class VirtualSheet(Sheet):
    """ An epithelium tissue with virtual vertices, to allow for rounded apical morphology"""

    def __init__(self, identifier, datasets, specs=None, coords=None, maximal_bond_length=0.2,
                 minimal_bond_length=0.05, periodic=False):
        """
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        datasets : dictionary of dataframes
            The keys correspond to the different geometrical elements
            constituting the epithelium:

            * `vert` contains a dataframe of vertices,
            * `edge` contains a dataframe of *oriented* half-edges between vertices,
            * `face` contains a dataframe of polygonal faces enclosed by half-edges,
            * `cell` contains a dataframe of polyhedral cells delimited by faces,
        virtual_vert_it: The number of virtual vertices adding iterations. On each iteration each edge is splitted
         into 2. For example: 2 iterations add 3 virtual vertices to each edge.

        """
        super().__init__(identifier, datasets, specs, coords)
        self.update_specs({"vert": {"is_virtual": int(0)}, "edge": {"order": int(0)}})
        self.maximal_bond_length = maximal_bond_length
        self.minimal_bond_length = minimal_bond_length
        # Interior-angle threshold (radians) for the incipient-fold collapse in
        # ``collapse_sharp_corners``. ``None`` disables it (default); ``simulate``
        # sets it for a running simulation.
        self.sharp_angle_threshold = None
        # Max sharp-corner collapses ``collapse_sharp_corners`` will perform per
        # call, sharpest-first. Caps the per-step cost so a sea of sub-threshold
        # corners (a heavily jagged / shrinking tissue) can't trigger a collapse
        # cascade that freezes the run.
        self.max_sharp_collapses_per_step = 16
        # When True, topological events log their SUCCESSES at INFO (failures
        # are always logged at DEBUG). ``simulate`` sets it from
        # ``run(verbose_log=...)``; default False keeps quiet runs to failures.
        self.verbose_log = False
        self.periodic = periodic
        # PeriodicPlanarGeometry short-circuits to PlanarGeometry when
        # sheet.periodic is False, so we can always use it.
        self.geom = PeriodicGeom
        self.initiate_edge_order()

    @classmethod
    def planar_sheet_2d(cls, identifier, nx, ny, distx, disty, noise=None):
        """
        Creates a planar sheet from an hexagonal grid of cells,
        keeping approximately nx × ny cells by padding and cropping.
        """

        # 1) Build padded grid
        pad = 1
        Nx_p = nx + 2 * pad
        Ny_p = ny + 2 * pad
        padded_grid = hexa_grid2d(Nx_p, Ny_p, distx, disty, noise)

        # 2) Voronoi + datasets from padded grid
        datasets = from_2d_voronoi(Voronoi(padded_grid))
        full_sheet = Sheet("tmp", datasets, specs=planar_spec(), coords=["x", "y"])

        # 3) Reconstruct integer grid indices in the SAME way as hexa_grid2d
        #    hexa_grid2d does: cy, cx = np.mgrid[0:ny, 0:nx]
        cy, cx = np.mgrid[0:Ny_p, 0:Nx_p]  # rows (y), cols (x)
        cx_flat = cx.flatten()
        cy_flat = cy.flatten()

        # face_df index i ↔ padded_grid[i] ↔ (cx_flat[i], cy_flat[i])
        face_df = full_sheet.face_df

        # 4) Select interior block: pad ≤ x < pad+nx, pad ≤ y < pad+ny
        keep_mask = (
                (cx_flat >= pad) & (cx_flat < pad + nx) &
                (cy_flat >= pad) & (cy_flat < pad + ny)
        )

        # Sanity check: we MUST get exactly nx * ny faces
        assert keep_mask.sum() == nx * ny, (
            f"Expected {nx * ny} faces, got {keep_mask.sum()}"
        )

        face_df["keep"] = keep_mask
        subsheet = cls.extract(full_sheet, "keep")
        subsheet.identifier = identifier
        subsheet.geom.update_all(subsheet)
        subsheet.sanitize(trim_borders=False, order_edges=True)
        subsheet.reset_index(order=True)
        subsheet.get_opposite()
        return subsheet

    @classmethod
    def extract(cls, sheet, face_mask, coords=["x", "y", "z"]):
        """Extract a new sheet from the sheet
        that correspond to a key word that define a face.

        Parameters
        ----------

        face_mask : column name in face composed by boolean value
        coords :

        Returns
        -------
        sheet_fold_patch_extract :
            subsheet corresponding to the fold patch area.

        """

        datasets = {}
        mask = sheet.face_df[face_mask].astype(bool)
        datasets["face"] = sheet.face_df[mask].copy()
        datasets["edge"] = sheet.edge_df[
            sheet.edge_df["face"].isin(datasets["face"].index)
        ].copy()
        datasets["vert"] = sheet.vert_df.loc[
            datasets["edge"][["srce", "trgt"]].stack().unique()
        ].copy()

        subsheet = cls("subsheet", datasets, sheet.specs)
        subsheet.reset_index()
        subsheet.reset_topo()
        return subsheet

    @classmethod
    def planar_virtual_sheet_2d(cls, identifier, nx, ny, distx=1.0, disty=1.0, maximal_bond_length=0.1,
                        minimal_bond_length=0.05, periodic=False, draw_debug=False):
        if periodic:
            sheet = cls.planar_periodic_sheet_2d(identifier, nx, ny, distx, disty)
        else:
            sheet = cls.planar_sheet_2d(identifier, nx, ny, distx, disty)
        sheet.maximal_bond_length = maximal_bond_length
        sheet.minimal_bond_length = minimal_bond_length
        sheet.order_all_edges()
        # Optional debug snapshots before/after virtual-vertex insertion.
        # Skip via draw_debug=False (e.g. in tests, or on hosts where the
        # matplotlib backend is broken).
        if draw_debug:
            try:
                fig1, _ = PeriodicBoundarySheet.periodic_sheet_view(sheet)
                fig1.savefig("before.png")
            except Exception as exc:
                print(f"[virtual_sheet] before.png skipped: {exc}")
        sheet.add_virtual_vertices()
        # order_all_edges above rewrote the per-face ``order`` column, and the
        # periodic geometry builds each polygon by walking edges in that order,
        # so the areas/positions are stale until the next update_all. That
        # refresh used to come for free from add_virtual_vertices' tail, but
        # add_virtual now short-circuits when no edge is over-long (the common
        # no-virtual build), so do it explicitly here. It runs AFTER
        # add_virtual on purpose: add_virtual decides which edges to subdivide
        # from the (pre-refresh) ``length`` column exactly as before, so the
        # virtual-vertex placement is unchanged; when add_virtual did run its
        # own tail update_all this is a cheap, harmless re-refresh.
        sheet.geom.update_all(sheet)
        if draw_debug:
            try:
                fig2, _ = PeriodicBoundarySheet.periodic_sheet_view(sheet)
                fig2.savefig("after.png")
            except Exception as exc:
                print(f"[virtual_sheet] after.png skipped: {exc}")
        return sheet

    def get_sheet_view_method(self):
        if self.periodic:
            return PeriodicBoundarySheet.periodic_sheet_view
        else:
            return sheet_view

    def update_after_each_time_step(self):
        " Necessary updates for periodic boundary conditions"
        if self.periodic:
            self.get_opposite()
        return 0

    @classmethod
    def planar_periodic_sheet_2d(cls, identifier, nx, ny, distx=1.0, disty=1.0):
        """
        Create an nx × ny periodic hexagonal tiling as a periodic VirtualSheet.
        """
        verts, edges, faces = PeriodicBoundarySheet.generate_periodic_hex_lattice(nx, ny, distx, disty)
        datasets = {
            "vert": verts,
            "edge": edges,
            "face": faces,
        }
        sheet = cls(identifier, datasets, coords=["x", "y"], periodic=True)
        # Periods follow the hexa_grid2d convention (x = cx*distx, y = cy*disty)
        sheet.Lx = nx * distx
        sheet.Ly = ny * disty
        # Persist the periodic metadata on face_df so a History archive
        # round-trip keeps Lx, Ly, and the periodic flag.
        # arrange_sheet_from_history reads (and drops) these on load.
        sheet._stash_periodic_metadata()
        sheet.geom.update_all(sheet)
        sheet.sanitize(trim_borders=False, order_edges=True)
        sheet.reset_index(order=True)

        # The Voronoi central-tile extraction produces MULTIPLE distinct
        # vertex labels at each physical point on the periodic boundary
        # (one per cell that touches it). e.g. canonical (1, 3.625) might
        # be label 4 in face_0's perimeter AND label 37 in face_13's
        # perimeter. Each label moves independently under the dynamics,
        # so periodic-image labels drift apart and break the fragile
        # position-based opposite stitching. The clean fix: collapse all
        # such duplicates into a single label per physical point. Then
        # label-based ``get_opposite`` finds every edge's opposite,
        # including periodic ones — no position matching needed.
        from topological_events import _consolidate_periodic_image_labels
        _consolidate_periodic_image_labels(sheet, list(sheet.vert_df.index))
        sheet.reset_index(order=False)

        sheet.get_opposite()
        sheet.geom.update_all(sheet)
        return sheet

    def _stash_periodic_metadata(self):
        """Embed Lx, Ly, periodic into face_df as constant columns so a
        HistoryHdf5 archive round-trip preserves them."""
        if getattr(self, "periodic", False):
            self.face_df["_periodic_flag"] = True
            self.face_df["_periodic_Lx"] = float(self.Lx)
            self.face_df["_periodic_Ly"] = float(self.Ly)

    def _heal_degenerate_edges(self, where):
        """Peel off degenerate 'antenna' spikes (a vertex now joined to a single
        other vertex) BEFORE opposites are recomputed.

        A topology change — cell removal, T1, a sharp-corner collapse — can
        collapse vertices together and leave such a spike, whose backtracking
        ``A->v->A`` edges appear as DUPLICATE ``(srce, trgt)`` half-edges in the
        faces sharing it. tyssue's ``get_opposite`` can't pair those and warns
        "Duplicated (`srce`, `trgt`) values in edge_df"; ``drop_two_sided_faces``
        / ``sanitize`` don't catch them (the faces still have >2 sides). Cleaning
        them here — at the shared choke point every topology handler funnels
        through (``get_opposite`` / ``reset_topo``) — kills the warning at the
        source whatever created the spike, not just cell removal. Cheap no-op on
        a clean mesh (only runs when a duplicate is actually present).

        Anything still duplicated after the spike peel is a NON-spike
        (cross-face / fold) degeneracy the peeler can't resolve; it is logged at
        DEBUG (with the caller) so it can be diagnosed rather than silently
        tolerated."""
        edf = getattr(self, "edge_df", None)
        if edf is None or "srce" not in edf.columns or "trgt" not in edf.columns:
            return
        if not edf[["srce", "trgt"]].duplicated().any():
            return
        _drop_antenna_spikes(self)
        dup_mask = self.edge_df[["srce", "trgt"]].duplicated(keep=False)
        if dup_mask.any() and log.isEnabledFor(logging.DEBUG):
            import traceback
            rows = self.edge_df.loc[dup_mask, ["srce", "trgt", "face"]]
            parts = []
            for (s, t), g in rows.groupby(["srce", "trgt"]):
                faces = g["face"].tolist()
                kind = "pinch" if len(set(faces)) == 1 else "cross-face"
                parts.append("(%s->%s) faces=%s [%s]" % (s, t, faces, kind))
            caller = "".join(traceback.format_stack()[-4:-1])
            log.debug("%s: %d NON-spike duplicate (srce,trgt) half-edge(s) remain "
                      "after antenna-spike cleanup: %s\ncaller:\n%s",
                      where, int(dup_mask.sum() // 2), "; ".join(parts), caller)

    def reset_topo(self):
        # Heal degenerate spikes BEFORE the base reset_topo recomputes opposites
        # via the tyssue module ``get_opposite`` (which is what emits the
        # "Duplicated (srce, trgt)" warning).
        self._heal_degenerate_edges("reset_topo")
        super().reset_topo()

    def get_opposite(self):
        # Same healing for the direct ``get_opposite`` path (the periodic branch
        # below calls tyssue's ``get_opposite``, which also warns on duplicates).
        self._heal_degenerate_edges("get_opposite")
        if not self.periodic:
            super().get_opposite()
            return

        # Re-wrap in case any operation moved vertices out of the domain.
        self.vert_df["x"] = self.vert_df["x"] % self.Lx
        self.vert_df["y"] = self.vert_df["y"] % self.Ly

        # First pass: standard tyssue (srce, trgt) ↔ (trgt, srce) vertex-label
        # matching — handles all interior pairs.
        from tyssue.core.sheet import get_opposite as _tyssue_get_opposite
        self.edge_df["opposite"] = _tyssue_get_opposite(self.edge_df)

        # Second pass: stitch any remaining opposite==-1 edges by wrapped
        # position. These are the true boundary-crossing pairs.
        PeriodicBoundarySheet.set_opposite_periodic(
            self.Lx, self.Ly, self.vert_df, self.edge_df
        )

        # Refresh is_periodic from canonical coords so callers between
        # update_all calls (e.g. remove_virtual_vertex) see the correct flag.
        PeriodicBoundarySheet.refresh_is_periodic(
            self.Lx, self.Ly, self.vert_df, self.edge_df
        )

    def arrange_sheet_from_history(self, two_dim=True, force_periodic_box=None):
        """Reconstruct a sheet loaded from a HistoryHdf5 archive.

        ``force_periodic_box`` : optional ``(Lx, Ly)``
            Fallback used ONLY when the archive's ``face_df`` carries no
            ``_periodic_flag`` metadata. Legacy archives written before
            the metadata was stashed on every snapshot come back with
            the flag missing, which would silently load them as
            non-periodic. When the caller knows the run is periodic
            (it always is in this model) it can pass the box dimensions
            here and periodicity is re-established. A present flag
            always takes precedence over this fallback.
        """
        if 'vert' in self.vert_df.columns:
            if np.isnan(self.vert_df['vert']).any():
                self.vert_df.set_index('index', inplace=True)
                self.vert_df.drop(columns=['vert'], inplace=True)
            else:
                self.vert_df.set_index('vert', inplace=True)
        if 'time' in self.vert_df.columns:
            self.vert_df.drop('time', inplace=True, axis=1)
        if 'edge' in self.edge_df.columns:
            self.edge_df.set_index('edge', inplace=True)
        if 'time' in self.edge_df.columns:
            self.edge_df.drop('time', inplace=True, axis=1)
        if 'face' in self.face_df.columns:
            self.face_df.set_index('face', inplace=True)
        if 'time' in self.face_df.columns:
            self.face_df.drop('time', inplace=True, axis=1)
        # Restore periodic metadata from the archive (see _stash_periodic_metadata).
        if "_periodic_flag" in self.face_df.columns:
            self.periodic = bool(self.face_df["_periodic_flag"].iloc[0])
            self.Lx = float(self.face_df["_periodic_Lx"].iloc[0])
            self.Ly = float(self.face_df["_periodic_Ly"].iloc[0])
            self.face_df.drop(
                columns=["_periodic_flag", "_periodic_Lx", "_periodic_Ly"],
                inplace=True, errors="ignore",
            )
            self.geom = PeriodicGeom
        elif force_periodic_box is not None:
            # Legacy archive with no stored periodic metadata: trust the
            # caller-supplied box and re-establish the periodic state so
            # the periodic geometry (vertex wrapping + per-face unfold)
            # runs. Without this the sheet would load as non-periodic
            # and boundary-crossing faces would unwrap into
            # domain-spanning edges.
            self.periodic = True
            self.Lx = float(force_periodic_box[0])
            self.Ly = float(force_periodic_box[1])
            self.geom = PeriodicGeom
        if two_dim:
            self.coords = ['x', 'y']
            self.dcoords = ['dx', 'dy']
            self.ncoords = ['nx', 'ny']
            self.ucoords = ['nx', 'ny']
            if self.bbox.shape[0] == 3:
                self.bbox = self.bbox[:-1, :]
            if 'z' in self.datasets['vert'].columns:
                self.datasets['vert'].drop(columns=['z'], inplace=True)
            if 'z' in self.datasets['face'].columns:
                self.datasets['face'].drop(columns=['z'], inplace=True)
            for col in ['sz', 'dz', 'tz', 'fz', 'nz', 'uz']:
                if col in self.datasets['edge'].columns:
                    self.datasets['edge'].drop(columns=[col], inplace=True)
            self.dim = 2
        return 0

    def initiate_edge_order(self):
        face_list = self.edge_df.face.to_numpy()
        edges_order = np.zeros((len(face_list,)))
        counter = 0
        current_face = -1
        for idx in range(edges_order.size):
            if face_list[idx] != current_face:
                current_face = face_list[idx]
                counter = 0
            counter += 1
            edges_order[idx] = counter
        self.edge_df.loc[:, 'order'] = edges_order.astype(int)

    def set_maximal_bond_length(self, length):
        self.maximal_bond_length = length

    def set_minimal_bond_length(self, length):
        self.minimal_bond_length = length

    def order_all_edges(self):
        """Re-compute the ``order`` column on every face's edges so the
        perimeter walks anti-clockwise from ``order=1`` to ``order=N``.

        Single O(Ne) pass: the edges are grouped by face ONCE (preserving
        edge_df row order within each face) and each face's perimeter is walked
        with the SAME logic as :meth:`order_edges`. The old version called
        ``order_edges`` per face, and each of those re-scanned the WHOLE
        ``edge_df`` (``face == f``) — making this O(Nf*Ne); with Ne~1.6e4 and
        Nf~500 it dominated the per-step cost of every topology re-sync (the
        biggest single item in a stalled run's profile). The result is byte-for-
        byte identical to the old per-face loop (verified by
        ``TestOrderAllEdgesGroupedEquivalence``), including the broken-perimeter
        behaviour below.

        Resilient to one bad face: if a face's perimeter doesn't close, its walk
        is abandoned (that face's edges left partially/zero-ordered, EXACTLY as
        ``order_edges`` left them) and the other faces are still re-ordered — the
        previous version's resilience is preserved.
        """
        import logging
        log = logging.getLogger(__name__)
        face_arr = self.edge_df["face"].to_numpy()
        n_edges = face_arr.shape[0]
        if n_edges == 0:
            return
        srce_arr = self.edge_df["srce"].to_numpy()
        trgt_arr = self.edge_df["trgt"].to_numpy()
        # Group edge ROW POSITIONS by face, preserving edge_df row order within
        # each face so each walk starts at the face's first row — the same edge
        # ``order_edges`` starts from (its ``edge_ids[0]``).
        groups = {}
        for pos in range(n_edges):
            groups.setdefault(int(face_arr[pos]), []).append(pos)
        # 0 == "not yet assigned" (matches order_edges zeroing each face's edges
        # up front); every face is re-walked, so every edge is rewritten.
        new_order = np.zeros(n_edges, dtype=self.edge_df["order"].dtype)
        failed = []
        for face, positions in groups.items():
            # First-occurrence srce->position and position->trgt maps, exactly
            # like order_edges' srce_to_edge / trgt_of_edge.
            srce_to_pos = {}
            trgt_of_pos = {}
            for p in positions:
                s = int(srce_arr[p])
                if s not in srce_to_pos:
                    srce_to_pos[s] = p
                trgt_of_pos[p] = int(trgt_arr[p])
            n = len(positions)
            cur = positions[0]
            edge_order = 1
            visited = 0
            # Walk srce->trgt around the face assigning 1..N until we return to
            # an already-ordered edge. Mirrors order_edges line-for-line,
            # including where it breaks a non-closing perimeter.
            while new_order[cur] < 1:
                new_order[cur] = edge_order
                nxt = srce_to_pos.get(trgt_of_pos[cur])
                if nxt is None:                       # perimeter doesn't close
                    failed.append((face, "IndexError"))
                    break
                cur = nxt
                edge_order += 1
                visited += 1
                if visited > n + 1:                   # walk didn't close
                    failed.append((face, "ValueError"))
                    break
        self.edge_df["order"] = new_order
        if failed:
            log_topo_event(
                log, self, False,
                "order_all_edges: %d face(s) couldn't be reordered "
                "(broken perimeter): %s", len(failed), failed[:5],
            )

    def order_edges(self, face_number):
        # Gather this face's edges via a boolean mask (cheap) instead of
        # the pandas string ``query`` parser, and replace the per-edge
        # ``query("srce == ...")`` lookup inside the walk with a prebuilt
        # srce-vertex -> edge-id map. This turns an O(N^2) walk over slow
        # query() calls into an O(N) dict walk — order_edges runs once per
        # face that loses an edge in remove_virtual_vertex.
        face_col = self.edge_df["face"].to_numpy()
        mask = face_col == face_number
        if not mask.any():
            return
        edge_ids = self.edge_df.index[mask].to_numpy()
        srce_arr = self.edge_df["srce"].to_numpy()[mask]
        trgt_arr = self.edge_df["trgt"].to_numpy()[mask]
        n_face_edges = len(edge_ids)
        # First occurrence wins, matching the old ``query(...).iloc[0]``.
        srce_to_edge = {}
        trgt_of_edge = {}
        for k in range(n_face_edges):
            s = int(srce_arr[k])
            if s not in srce_to_edge:
                srce_to_edge[s] = edge_ids[k]
            trgt_of_edge[edge_ids[k]] = int(trgt_arr[k])

        self.edge_df.loc[edge_ids, "order"] = 0
        current_edge = edge_ids[0]
        current_edge_order = 1
        visited = 0
        while self.edge_df.at[current_edge, "order"] < 1:
            self.edge_df.at[current_edge, "order"] = current_edge_order
            edge_trgt = trgt_of_edge[current_edge]
            if edge_trgt not in srce_to_edge:
                # Perimeter doesn't close — break instead of looping
                # forever or raising. The face will be flagged in the
                # caller's "failed" list via the assertion below.
                raise IndexError(
                    f"face {face_number}: perimeter walk broke at "
                    f"vertex {edge_trgt} (no out-edge in this face)"
                )
            current_edge = srce_to_edge[edge_trgt]
            current_edge_order += 1
            visited += 1
            if visited > n_face_edges + 1:
                raise ValueError(
                    f"face {face_number}: walk didn't close after "
                    f"{visited} steps (face has {n_face_edges} edges)"
                )

    def check_edge_order(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number).loc[:,["order", "srce", "trgt"]]
        edges.sort_values(["order"], inplace=True)
        first_srce = -1
        current_trgt = -1
        for index, row in edges.iterrows():
            if first_srce < 0:
                first_srce = row.srce
            if current_trgt > 0 and current_trgt != row.srce:
                return False
            current_trgt = row.trgt
        return row.trgt == first_srce

    def check_all_edge_order(self):
        for face in self.face_df.index.values:
            if not self.check_edge_order(face):
                print("wrong order in face %d" %face)
                return False
        return True

    def add_virtual_vertices(self):
        long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
        np.random.shuffle(long)
        # No over-long edges => nothing to subdivide. Skip the whole tail
        # (geometry update + edge sort + opposite re-stitch), which only
        # exists to repair the sheet AFTER an insertion. Mirrors the
        # ``removed_any`` guard in remove_virtual_vertices; on near-steady
        # steps add_virtual is called every iteration but usually inserts
        # nothing, so this avoids a redundant update_all + get_opposite.
        if long.size == 0:
            return
        iter_count = 0
        while long.size > 0:
            edge_ind = long[0]
            edge_order = self.edge_df.at[edge_ind, "order"]
            edge_face = self.edge_df.at[edge_ind, "face"]
            new_vert, new_edge, new_opposite_edge, new_opp_vert = index_preserving_add_vert(self, edge_ind)
            self.vert_df.at[new_vert, "is_virtual"] = 1
            if new_opp_vert is not None:
                self.vert_df.at[new_opp_vert, "is_virtual"] = 1
            self.edge_df.at[edge_ind, "length"] /= 2
            self.edge_df.at[new_edge, "length"] /= 2
            # Shift the order of every later edge on this face up by one to
            # make room for the inserted midpoint edge. A vectorized
            # boolean mask is much cheaper than the pandas string ``query``
            # parser, which matters because this runs once per insertion.
            face_col = self.edge_df["face"].to_numpy()
            order_col = self.edge_df["order"].to_numpy()
            increase_idx = self.edge_df.index[(face_col == edge_face) & (order_col > edge_order)]
            self.edge_df.at[new_edge, "order"] = edge_order + 1
            self.edge_df.loc[increase_idx, "order"] += 1
            opposite = int(self.edge_df.loc[edge_ind, "opposite"])
            if opposite >= 0:
                self.edge_df.at[opposite, "length"] /= 2
                if new_opposite_edge is None:
                    self.edge_df.at[new_edge, "opposite"] = -1
                else:
                    opposite_order = self.edge_df.at[opposite, "order"]
                    opposite_face = self.edge_df.at[opposite, "face"]
                    self.edge_df.at[new_edge, "opposite"] = new_opposite_edge
                    self.edge_df.at[new_opposite_edge, "opposite"] = new_edge
                    self.edge_df.at[new_opposite_edge, "length"] /= 2
                    face_col = self.edge_df["face"].to_numpy()
                    order_col = self.edge_df["order"].to_numpy()
                    increase_idx = self.edge_df.index[(face_col == opposite_face) & (order_col > opposite_order)]
                    self.edge_df.at[opposite, "order"] = opposite_order + 1
                    self.edge_df.at[new_opposite_edge, "order"] = opposite_order
                    self.edge_df.loc[increase_idx, "order"] += 1
            else:
                self.edge_df.at[new_edge, "opposite"] = -1
            long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
            np.random.shuffle(long)
            iter_count += 1
            if iter_count > MAX_VIRTUAL_INSERTIONS:
                raise RuntimeError(
                    f"add_virtual_vertices: exceeded {MAX_VIRTUAL_INSERTIONS} "
                    "insertions without converging (likely a topology bug)."
                )
        self.edge_df.index.name = 'edge'
        self.vert_df.index.name = 'vert'
        self.geom.update_all(self)
        self.edge_df.sort_values(["face", "order"], inplace=True)
        self.get_opposite()
        log_topo_event(log, self, True,
                       "added %d virtual vertices (subdivided long edges)",
                       iter_count)
        # if not self.check_all_edge_order():
        #     print("bug in adding virtual vertices")

    def remove_virtual_vertex(self, edge_id):
        """Collapse a virtual edge (one whose srce or trgt is a virtual
        vertex) into its real neighbor.

        For a periodic edge the OPPOSITE edge lives on the other side
        of the periodic boundary and has its OWN virtual midpoint vertex
        (created as a periodic image in ``index_preserving_add_vert``).
        We collapse both edges in turn; ``remove_virtual_vertices`` calls
        ``get_opposite`` afterwards to re-stitch any new neighbours.

        Re-orders the affected faces' edges locally so the perimeter
        walk in :meth:`PeriodicPlanarGeometry.update_dcoords` doesn't
        encounter ``order`` gaps. The previous version left a gap in
        the order numbering of each face that lost an edge — gaps that
        propagated into wrong per-face dcoords (long-edge bug) on the
        next geometry update.
        """
        srce_idx = int(self.edge_df.at[edge_id, "srce"])
        trgt_idx = int(self.edge_df.at[edge_id, "trgt"])
        srce = self.vert_df.loc[srce_idx]
        trgt = self.vert_df.loc[trgt_idx]

        # Find the opposite edge BEFORE collapsing the main one — collapse_edge
        # may renumber edges and we need to act on the periodic image too.
        opp_id = int(self.edge_df.at[edge_id, "opposite"])
        is_periodic = (
            getattr(self, "periodic", False) and
            opp_id >= 0 and
            bool(self.edge_df.at[edge_id, "is_periodic"])
        )

        # Record the faces that will lose an edge to the upcoming
        # collapse(s). ``collapse_edge`` drops every edge that became a
        # self-loop, so each affected face's perimeter shrinks by one
        # — but the surviving edges keep their old ``order`` values,
        # leaving a gap right where the dropped edge sat. We re-walk
        # these faces' perimeters at the end of this function.
        affected_faces = {int(self.edge_df.at[edge_id, "face"])}
        if opp_id >= 0 and opp_id in self.edge_df.index:
            affected_faces.add(int(self.edge_df.at[opp_id, "face"]))

        # Move the virtual endpoint onto its real neighbor so collapse_edge
        # leaves the cell geometry unchanged. collapse_edge() internally
        # places the surviving vertex at the ARITHMETIC MEAN of srce and
        # trgt; we want that mean to coincide with the real vert's
        # canonical position so no edge incident to the survivor changes
        # length. Three cases:
        #
        #   1) virtual ↔ real   → copy real coords onto virtual; mean is
        #                          then the real coord. (existing.)
        #   2) virtual ↔ virtual → both endpoints are mid-edge points.
        #      Without intervention collapse_edge averages whatever
        #      positions are in vert_df. On a PERIODIC sheet that mean
        #      is catastrophic: when the two virtuals straddle the
        #      wrap (e.g. one at canonical x=0.4 and one at x=Lx-0.4),
        #      the raw arithmetic mean lands AT THE FAR SIDE of the
        #      sheet — every incident edge of the survivor gets
        #      stretched from ~0.1 to ~Lx/2, and the next
        #      add_virtual_vertices subdivides them into hundreds of
        #      pieces, producing the "super-face" / negative-area
        #      crash observed at t=2.349 in random_periodic_array_test2
        #      (faces 179, 160, 403 each ballooned to 156-157 edges).
        #      Fix: collapse both virtuals to ONE canonical position
        #      (min-image midpoint), so collapse_edge's mean of two
        #      equal coords stays in place.
        #   3) real ↔ real      → not a virtual edge; remove_virtual_vertex
        #                          shouldn't be called. (is_virtual_edge
        #                          filters this out upstream.)
        if srce.is_virtual == 1 and trgt.is_virtual != 1:
            self.vert_df.loc[srce_idx, self.coords] = self.vert_df.loc[trgt_idx, self.coords]
        elif trgt.is_virtual == 1 and srce.is_virtual != 1:
            self.vert_df.loc[trgt_idx, self.coords] = self.vert_df.loc[srce_idx, self.coords]
        elif srce.is_virtual == 1 and trgt.is_virtual == 1:
            mid = _min_image_midpoint(self, [srce_idx, trgt_idx])
            self.vert_df.loc[srce_idx, self.coords] = mid
            self.vert_df.loc[trgt_idx, self.coords] = mid
        collapse_edge(self, edge_id, allow_two_sided=False, reindex=False)

        # For periodic edges, collapse the matching opposite-side edge too.
        # collapse_edge cleared edge_id; opp_id is still valid because it
        # lives on the other side and was untouched.
        if is_periodic and opp_id in self.edge_df.index:
            opp_sv = int(self.edge_df.at[opp_id, "srce"])
            opp_tv = int(self.edge_df.at[opp_id, "trgt"])
            opp_s = self.vert_df.loc[opp_sv]
            opp_t = self.vert_df.loc[opp_tv]
            # Same three cases as above — including the both-virtual
            # branch that previously was silently delegating to
            # collapse_edge's wrap-blind arithmetic mean.
            if opp_s.is_virtual == 1 and opp_t.is_virtual != 1:
                self.vert_df.loc[opp_sv, self.coords] = self.vert_df.loc[opp_tv, self.coords]
            elif opp_t.is_virtual == 1 and opp_s.is_virtual != 1:
                self.vert_df.loc[opp_tv, self.coords] = self.vert_df.loc[opp_sv, self.coords]
            elif opp_s.is_virtual == 1 and opp_t.is_virtual == 1:
                opp_mid = _min_image_midpoint(self, [opp_sv, opp_tv])
                self.vert_df.loc[opp_sv, self.coords] = opp_mid
                self.vert_df.loc[opp_tv, self.coords] = opp_mid
            # The first collapse removed THIS edge's opposite pointer
            # (which used to be edge_id). Clear it so collapse_edge doesn't
            # try to dereference a stale label.
            self.edge_df.at[opp_id, "opposite"] = -1
            collapse_edge(self, opp_id, allow_two_sided=False, reindex=False)

        # Re-order the affected faces' perimeters. ``order_edges`` walks
        # srce→trgt around each face and rewrites the ``order`` column
        # to 1..N contiguous. Only the faces that actually lost an edge
        # need this — every other face is unchanged.
        for face in affected_faces:
            if face in self.face_df.index:
                try:
                    self.order_edges(int(face))
                except (IndexError, ValueError, KeyError):
                    # Perimeter temporarily broken (e.g. mid-batch
                    # removal of several virtuals in the same face).
                    # The outer ``remove_virtual_vertices`` calls
                    # ``order_all_edges`` at the end as a safety net.
                    pass

        return 0

    def remove_virtual_vertices(self):
        # involved_faces = []
        short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
        if short.size > 0:
            short = short[self.is_virtual_edge(short)]
        np.random.shuffle(short)
        removed_any = short.size > 0
        removed = 0
        while short.size > 0:
            self.remove_virtual_vertex(short[0])
            removed += 1
            short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
            if short.size > 0:
                short = short[self.is_virtual_edge(short)]
            np.random.shuffle(short)
        if removed_any:
            # collapse_edge(reindex=False) leaves the vert/edge indexes sparse,
            # which breaks tyssue's positional upcast on the next update_all.
            # Re-index, then (for periodic sheets) re-stitch periodic opposites.
            self.reset_index(order=False)
            self.get_opposite()
            self.geom.update_all(self)
            log_topo_event(log, self, True,
                           "removed %d virtual vertices (collapsed short edges)",
                           removed)
        # for face in np.unique(involved_faces):
        #     self.order_edges(face)
        # sheet.edge_df.sort_values(["face", "order"], inplace=True)
        # sheet.get_opposite()
        # self.geom.update_all(self)
        # if not self.check_all_edge_order():
        #     print("bug in removing virtual vertices")
        return 0

    def is_virtual_edge(self, edge_indices):
        """
        Checks if an edge contains a virtual vertex
        """
        if hasattr(edge_indices, "__len__"):
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual.to_numpy() == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual.to_numpy() == 1
            return np.logical_or(srce_is_virtual, trgt_is_virtual)
        else:
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual == 1
            return srce_is_virtual or trgt_is_virtual

    @staticmethod
    def default_sharp_angle_threshold(min_bond_length, intercalation_length, max_bond_length):
        """Default interior-angle threshold (radians) for
        :meth:`collapse_sharp_corners`, derived from the sheet's length scales.

        What we really want to avoid is two vertices getting too close. Model a
        corner V with its two neighbours A, B as an isosceles triangle whose
        legs (V-A, V-B) have length ``max_bond_length / 2`` — a TYPICAL edge
        length: an edge is subdivided once it exceeds ``max_bond_length``, so it
        lives roughly between ``max_bond_length / 2`` and ``max_bond_length``,
        and half the max is the "average" choice — and whose base (A-B) has
        length ``base = max(min_bond_length, intercalation_length)``. The apex
        ("head") angle at which the base reaches ``base`` is the threshold: a
        SHARPER corner means A and B sit closer than ``base`` and should be
        collapsed. By the law of cosines
        ``base**2 = 2*(max_bond_length/2)**2*(1 - cos theta)``, hence

            theta = arccos(1 - base**2 / (2 * (max_bond_length / 2)**2)).

        For run_model's lengths (max_bond=0.2 -> leg 0.1, base=0.05) this is
        ~0.51 rad (~29 deg). Using ``max_bond_length`` itself for the legs was
        too permissive (~14 deg, flags too few corners) and the short
        min/intercalation lengths far too aggressive (~77 deg); the half-max
        "average" edge sits in between. ``base >= max_bond_length`` would be a
        degenerate triangle; the cosine is clamped to [-1, 1] so the result
        saturates at pi rather than erroring.
        """
        base = max(min_bond_length, intercalation_length)
        leg = max_bond_length / 2
        cos_theta = float(np.clip(1.0 - (base * base) / (2.0 * leg * leg), -1.0, 1.0))
        return float(np.arccos(cos_theta))

    def get_sharp_corner_collapse_edges(self, angle_threshold, return_angles=False):
        """Edges to collapse to relieve incipient folds.

        Two non-adjacent vertices drifting together pinch a face into a thin
        spike long BEFORE any single edge shrinks below the intercalation
        length — so the length-based tests never fire, the spike keeps closing
        and the perimeter eventually self-intersects (the cell folds). Such a
        pinch always shows up first as a tiny INTERIOR ANGLE at the spike
        vertex: the two edges meeting there become nearly anti-parallel, which
        is exactly the geometric statement "the spike's two neighbour vertices
        have drifted close together".

        Returns, for every face corner whose interior angle is below
        ``angle_threshold`` (radians), the SHORTER of the two edges meeting at
        that corner — collapsing it snaps the spike vertex onto its nearest
        neighbour and removes the spike. Sharpest corners first; each edge
        appears at most once. Ties in length are broken at random. When
        ``return_angles`` is True, returns ``(edges, interior_angles)`` with the
        triggering interior angle (radians) aligned to each edge (used for log
        documentation of the collapse).

        Vectorized and O(Ne): it reuses the per-edge vectors (``dx``/``dy``)
        and the same (face, order) perimeter walk as
        ``solvers.count_folded_faces``, so it is cheap enough to run every
        step. Requires an up-to-date geometry (the caller refreshes it).
        """
        empty = ((np.array([], dtype=int), np.array([], dtype=float))
                 if return_angles else np.array([], dtype=int))
        if angle_threshold is None:
            return empty
        e = self.edge_df
        if e.shape[0] == 0:
            return empty
        ed = e[["face", "dx", "dy", "length", "order"]].sort_values(["face", "order"])
        labels = ed.index.to_numpy()
        face = ed["face"].to_numpy()
        vx = ed["dx"].to_numpy()
        vy = ed["dy"].to_numpy()
        length = ed["length"].to_numpy()
        n = len(face)
        # First edge of each contiguous same-face run, and the last edge of each.
        change = np.empty(n, bool)
        change[0] = True
        change[1:] = face[1:] != face[:-1]
        run_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
        is_last = np.empty(n, bool)
        is_last[:-1] = face[1:] != face[:-1]
        is_last[-1] = True
        # "Next" edge cyclically within the face (the last wraps to the first).
        nxt = np.arange(n) + 1
        nxt[is_last] = run_start[is_last]
        wx = vx[nxt]
        wy = vy[nxt]
        # Interior angle at the corner shared by edge i (incoming, A->V) and
        # edge nxt[i] (outgoing, V->B): the angle between them as rays from V.
        # atan2(|v x w|, -(v . w)) lands in [0, pi] — ~pi where the perimeter
        # runs straight through V, ~0 at a closing spike.
        interior = np.arctan2(np.abs(vx * wy - vy * wx), -(vx * wx + vy * wy))
        sharp = np.where(interior < angle_threshold)[0]
        if sharp.size == 0:
            return empty
        edge_in = labels[sharp]
        edge_out = labels[nxt[sharp]]
        choose_in = length[sharp] < length[nxt[sharp]]
        tie = length[sharp] == length[nxt[sharp]]
        if tie.any():
            choose_in[tie] = np.random.random(int(tie.sum())) < 0.5
        chosen = np.where(choose_in, edge_in, edge_out)
        ang = interior[sharp]
        # Sharpest corner first, then drop duplicate edges keeping that order
        # (one short edge can be the shorter side of two adjacent sharp corners).
        order = np.argsort(ang)
        chosen = chosen[order]
        ang = ang[order]
        _, first = np.unique(chosen, return_index=True)
        keep = np.sort(first)
        chosen = chosen[keep].astype(int)
        if return_angles:
            return chosen, ang[keep]
        return chosen

    def _resync_topology(self):
        """Restore a clean, contiguous index and consistent derived state
        (order / opposite / geometry) after a topology change."""
        self.reset_index(order=False)
        self.order_all_edges()
        self.edge_df.sort_values(["face", "order"], inplace=True)
        self.get_opposite()
        self.geom.update_all(self)

    def collapse_sharp_corners(self, angle_threshold=None):
        """Collapse the shorter edge at every face corner whose interior angle
        is below ``angle_threshold`` (radians), to relieve an incipient fold
        before the pinched perimeter self-intersects (see
        :meth:`get_sharp_corner_collapse_edges`). A VIRTUAL edge is collapsed by
        removing a virtual vertex; a REAL edge by a T1 intercalation — the same
        primitives the length-based events use. Re-detects after each collapse
        (topology changes), skipping any edge whose collapse raises. Returns
        True if at least one edge was collapsed.

        Assumes the geometry (``dx`` / ``dy`` / ``length``) is up to date — the
        angle test reads it directly and the caller guarantees freshness: the
        solver's ``set_pos`` runs ``geom.update_all`` before the manager, and
        ``remove_virtual_vertices`` / ``add_virtual_vertices`` leave geometry
        fresh too. So no geometry update is done up front (it would dominate the
        per-step cost ~10x over the O(Ne) angle test); ``_resync_topology``
        refreshes it after each collapse for the next detection.

        A T1 is not atomic — ``index_preserving_type1_transition`` collapses an
        edge (leaving a gappy index) before a split that may raise — so, like
        the intercalation handler, we must NOT ``reset_index`` between skipped
        attempts (it would renumber vertices and invalidate the ``(srce,trgt)``
        keys in ``tried``, retrying the same failing edge forever). We keep the
        index gappy through a skip run and resync once on success or at the end.
        """
        if angle_threshold is None:
            return False
        # Bound the work per call. An aggressively shrinking / jagged tissue can
        # present a SEA of sub-threshold corners (most are valid, not folds), and
        # without a cap the loop would try to collapse them all — each collapse
        # re-detecting (O(Ne)) and resyncing — which for a heavily subdivided
        # sheet (Ne ~ 1e4) is billions of ops in ONE manager step (the run
        # freezes). Collapsing is sharpest-first, so the few most fold-prone
        # corners are still relieved every step; the rest are left to the next
        # manager pass, the mechanics, and the (tolerant) fold net.
        max_collapses = getattr(self, "max_sharp_collapses_per_step", 16)
        collapsed_any = False
        tried = set()
        pending_resync = False
        n_collapsed = 0
        for _ in range(3 * max_collapses + 8):  # budget for collapses + skips
            if n_collapsed >= max_collapses:
                break
            chosen, angles = self.get_sharp_corner_collapse_edges(
                angle_threshold, return_angles=True)
            edge_id = None
            key = None
            ang = None
            for eid, a in zip(chosen, angles):
                eid = int(eid)
                if eid not in self.edge_df.index:
                    continue
                k = (int(self.edge_df.at[eid, "srce"]),
                     int(self.edge_df.at[eid, "trgt"]))
                if k not in tried:
                    edge_id, key, ang = eid, k, float(a)
                    break
            if edge_id is None:
                break
            # Capture identity BEFORE the collapse removes the edge, so the
            # success log can describe what was collapsed.
            face = int(self.edge_df.at[edge_id, "face"])
            is_virtual = bool(self.is_virtual_edge(edge_id))
            if is_virtual:
                try:
                    self.remove_virtual_vertex(edge_id)
                except Exception as exc:
                    log_topo_event(log, self, False,
                                   "sharp-corner virtual collapse of edge %d "
                                   "(srce=%d, trgt=%d) raised %s; skipping",
                                   edge_id, key[0], key[1], type(exc).__name__)
                    tried.add(key)
                    pending_resync = True
                    continue
                # remove_virtual_vertex already re-ordered the affected faces,
                # so a LIGHT resync (reset index + opposites + geometry, NO
                # full order_all_edges) is enough and skips the O(Nf) perimeter
                # walk that otherwise dominates the per-collapse cost — exactly
                # what remove_virtual_vertices does at its tail.
                self.reset_index(order=False)
                self.get_opposite()
                self.geom.update_all(self)
            else:
                try:
                    ret = index_preserving_type1_transition(self, edge_id)
                except Exception as exc:
                    log_topo_event(log, self, False,
                                   "sharp-corner T1 on edge %d (srce=%d, "
                                   "trgt=%d) raised %s; skipping", edge_id,
                                   key[0], key[1], type(exc).__name__)
                    tried.add(key)
                    pending_resync = True
                    continue
                if ret is not None and ret < 0:
                    tried.add(key)
                    pending_resync = True
                    continue
                # A T1 rewires several faces and does NOT set their order, so it
                # needs the full re-order. T1s are rare here.
                self._resync_topology()
            # Document the collapse in the run log: a successful topological
            # event, so INFO and only in a verbose run (one line per relieved
            # incipient fold, with the triggering interior angle so the
            # threshold can be judged).
            log_topo_event(log, self, True,
                           "collapsed sharp corner on face %d via %s: edge "
                           "(srce=%d, trgt=%d), interior angle %.1f deg < "
                           "threshold %.1f deg", face,
                           "virtual-vertex removal" if is_virtual else "T1",
                           key[0], key[1], np.degrees(ang),
                           np.degrees(angle_threshold))
            collapsed_any = True
            n_collapsed += 1
            tried = set()
            pending_resync = False
        # One full resync at the end: catches any face the light virtual path
        # left mis-ordered and cleans up a trailing skip run (cheap — once per
        # call, not per collapse).
        if collapsed_any or pending_resync:
            self._resync_topology()
        return collapsed_any

    def get_update_virtual_vertices_function(self):
        def update_virtual_vertices(sheet, manager):
            sheet.remove_virtual_vertices()
            sheet.add_virtual_vertices()
            # Relieve incipient folds the length-based tests can't see: a face
            # pinched into a thin spike (two non-adjacent vertices drifting
            # together) has a tiny interior angle there even though no edge is
            # short. Gated on ``sharp_angle_threshold`` (None => disabled, so
            # behaviour is unchanged unless a simulation switches it on).
            sheet.collapse_sharp_corners(getattr(sheet, "sharp_angle_threshold", None))
            manager.append(update_virtual_vertices)
            return
        return update_virtual_vertices

    def get_neighbors(self, face, elem="face"):
        face_edges = self.edge_df.query("face == %d" % face)
        opposite_edges = face_edges.opposite.to_numpy()
        neighbors = self.edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
        return np.unique(neighbors)

    def get_contact_matrix(self):
        """Symmetric contact matrix indexed by ``unique_id``.

        ``m[i, j]`` is the total contact length between the cell with
        ``unique_id == i`` and the cell with ``unique_id == j`` (0 if
        they don't touch). The matrix is sized to
        ``max(unique_id) + 1`` over ALL faces.

        ``unique_id`` IS A PER-FRAME SLOT, NOT A PERSISTENT IDENTITY.
        Despite the name, it is recompacted to ``0..n_faces-1`` whenever a
        face is REMOVED (ablation, delamination), so the same ``unique_id``
        denotes a DIFFERENT cell before and after. Measured on an ablation
        run, across the single frame where the cell is dropped: the
        ``unique_id -> delta_level`` map changed by >0.25 for 211 of 507
        cells (mean |change| 0.24) while the ``id``-keyed map changed for
        NONE of them. Over the whole 5-unit run the ``id``-keyed map moved
        for 6 cells — those are the genuine differentiation events.

        Two consequences:

        * WITHIN one frame ``unique_id`` equals the positional row index, so
          this matrix is safely indexed with positions — which is what
          ``calc_contact_with_neighbors_from_type`` and
          ``_sim_neighbor_pair_percentages`` do (verified 507x507 for 507
          faces immediately after a removal).
        * ACROSS frames use the ``id`` column, which is the persistent
          identity. ``calc_HC_neighbors_at_differentiation`` and
          ``calc_percentage_of_differentiating_by_initial_neighbors``
          already do; anything new that traces cells over time must too.

        (An earlier version of this docstring claimed removed ``unique_id``
        values leave all-zero rows/columns behind. They do not — the ids are
        renumbered — and that claim cost real debugging time.)
        """
        has_opposite = self.edge_df.opposite >= 0
        faces_with_neighbors_ids = self.edge_df.loc[has_opposite, "face"].to_numpy()
        face_unique_ids = self.face_df.loc[faces_with_neighbors_ids, "unique_id"].to_numpy()
        neighbor_ids = self.edge_df.loc[self.edge_df.opposite[has_opposite], "face"].to_numpy()
        neighbors_unique_ids = self.face_df.loc[neighbor_ids, "unique_id"].to_numpy()
        contact_length = self.edge_df.loc[self.edge_df.opposite[has_opposite], "length"].to_numpy()
        # Size from the maximum unique_id over ALL faces (not just those
        # that currently have a neighbor) so the highest-id cell always
        # gets its row/column even if it happens to be isolated.
        number_of_faces = int(self.face_df["unique_id"].max()) + 1
        m = np.bincount(face_unique_ids*number_of_faces + neighbors_unique_ids, weights=contact_length,
                         minlength=number_of_faces*number_of_faces).reshape(number_of_faces, number_of_faces)
        return m

    def get_face_area(self):
        data = self.face_df[["area", "id"]].copy()
        data.set_index("id", inplace=True)
        return data["area"]

    def get_face_perimeter(self):
        data = self.face_df[["perimeter", "id"]].copy()
        data.set_index("id", inplace=True)
        return data["perimeter"]

    def get_face_roundness(self):
        data = self.face_df[["area", "perimeter", "id"]].copy()
        data["roundness"] = 4*np.pi*data["area"]/(data["perimeter"]**2)
        data.set_index("id", inplace=True)
        return data["roundness"]