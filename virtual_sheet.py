from scipy.spatial import Voronoi
from tyssue import Sheet, config
from tyssue.generation import from_2d_voronoi, hexa_grid2d
from tyssue.config.geometry import planar_spec
from periodic_sheet import PeriodicBoundarySheet
from periodic_sheet import PeriodicPlanarGeometry as PeriodicGeom
from tyssue.topology.base_topology import collapse_edge
from topological_events import index_preserving_add_vert, _min_image_midpoint
from tyssue.draw import sheet_view

import numpy as np
# Hard cap on how many virtual-vertex insertions we attempt in one call.
# Each iteration splits one over-long edge, so this bounds the work by
# a (large) multiple of the initial edge count.
MAX_VIRTUAL_INSERTIONS = 100_000

class VirtualSheet(Sheet):
    """ An epithelium tissue with virtual vertices, to allow for rounded apical morphology"""

    def __init__(self, identifier, datasets, specs=None, coords=None, maximal_bond_length=0.1,
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
                        minimal_bond_length=0.05, periodic=False, draw_debug=True):
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

    def get_opposite(self):
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

    def arrange_sheet_from_history(self, two_dim=True):
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

        Resilient to one bad face: if ``order_edges`` raises for a
        particular face (e.g. a transient bad topology mid-T1), the
        other faces are still re-ordered. The previous version
        propagated the exception, which left faces processed AFTER
        the bad one with their OLD (now-stale) order values — exactly
        the "order gaps in face N" symptom we kept hitting in long
        runs.
        """
        import logging
        log = logging.getLogger(__name__)
        failed = []
        for face in self.face_df.index.values:
            try:
                self.order_edges(int(face))
            except (IndexError, ValueError, KeyError) as exc:
                failed.append((int(face), type(exc).__name__))
        if failed:
            log.warning(
                "order_all_edges: %d face(s) couldn't be reordered "
                "(broken perimeter): %s",
                len(failed), failed[:5],
            )

    def order_edges(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number)
        if not len(edges):
            return
        self.edge_df.loc[edges.index, "order"] = 0
        current_edge = edges.iloc[0]
        current_edge_order = 1
        visited = 0
        n_face_edges = len(edges)
        while self.edge_df.at[current_edge.name, "order"] < 1:
            self.edge_df.at[current_edge.name, "order"] = current_edge_order
            edge_trgt = current_edge.trgt
            next_match = edges.query("srce == %d" % edge_trgt)
            if not len(next_match):
                # Perimeter doesn't close — break instead of looping
                # forever or raising. The face will be flagged in the
                # caller's "failed" list via the assertion below.
                raise IndexError(
                    f"face {face_number}: perimeter walk broke at "
                    f"vertex {edge_trgt} (no out-edge in this face)"
                )
            current_edge = next_match.iloc[0]
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
            increase_order = self.edge_df.query("face == %d and order > %d" %(edge_face, edge_order))
            self.edge_df.at[new_edge, "order"] = edge_order + 1
            self.edge_df.loc[increase_order.index, "order"] += 1
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
                    increase_order = self.edge_df.query("face == %d and order > %d" % (opposite_face, opposite_order))
                    self.edge_df.at[opposite, "order"] = opposite_order + 1
                    self.edge_df.at[new_opposite_edge, "order"] = opposite_order
                    self.edge_df.loc[increase_order.index, "order"] += 1
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
        while short.size > 0:
            self.remove_virtual_vertex(short[0])
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

    def get_update_virtual_vertices_function(self):
        def update_virtual_vertices(sheet, manager):
            sheet.remove_virtual_vertices()
            sheet.add_virtual_vertices()
            manager.append(update_virtual_vertices)
            return
        return update_virtual_vertices

    def get_neighbors(self, face, elem="face"):
        face_edges = self.edge_df.query("face == %d" % face)
        opposite_edges = face_edges.opposite.to_numpy()
        neighbors = self.edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
        return np.unique(neighbors)

    def get_contact_matrix(self):
        has_opposite = self.edge_df.opposite >= 0
        faces_with_neighbors_ids = self.edge_df.loc[has_opposite, "face"].to_numpy()
        neighbor_ids = self.edge_df.loc[self.edge_df.opposite[has_opposite], "face"].to_numpy()
        contact_length = self.edge_df.loc[self.edge_df.opposite[has_opposite], "length"].to_numpy()
        number_of_faces = self.face_df.shape[0]
        m = np.bincount(faces_with_neighbors_ids*number_of_faces + neighbor_ids, weights=contact_length,
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