"""Topology changes that PRESERVE existing indices, plus periodic variants.

tyssue's topology operations renumber vertices, edges and faces as they go. That
is fine for a single relaxation, but it makes a cell impossible to follow across
frames — and following cells is the whole point when the question is which cell
differentiated, or how one responded to its neighbour being removed. Every
operation here therefore appends new elements and rewires in place, leaving the
existing indices untouched.

What that buys, and what it costs:

* A cell keeps its identity for the lifetime of a run, so a history archive can
  be read as trajectories rather than as unrelated frames.
* Face labels are NOT compacted on removal, so the index set becomes sparse. Code
  that assumes ``range(n_faces)`` will be wrong.

Each operation is also ATOMIC: it snapshots the affected tables and restores them
if any stage fails, so a partial rearrangement never survives. A half-applied T1
leaves duplicate (source, target) pairs across two faces, which then surfaces far
away as a confusing "duplicated edge" warning.

Periodic sheets take separate paths (``_periodic_*``) because a neighbourhood can
wrap the box: distances use the minimum image, and a T1 near the seam must
consolidate the vertex labels of both images before rewiring.
"""
import numpy as np
import pandas as pd
from tyssue.topology.base_topology import drop_two_sided_faces, collapse_edge
from tyssue.topology.sheet_topology import get_division_edges
import logging
import warnings
logger = logging.getLogger(name=__name__)
# Alias so that pre-existing `log.warning(...)` / `log.error(...)` calls in
# the topology event handlers don't NameError on first use. The previous
# code mixed `logger` and `log` — only `logger` was defined, so any of
# the warning paths would crash before logging anything useful.
log = logger
MAX_ITER = 100


def log_topo_event(event_logger, sheet, succeeded, message, *args):
    """Document a topological event per the run's verbosity policy
    (``sheet.verbose_log``, set by ``simulate`` from ``run(verbose_log=...)``):

      * a SUCCESS is logged at INFO, but ONLY in a verbose run;
      * a FAILURE is logged at DEBUG, ALWAYS — so a quiet run's log still
        records every failed/rejected event (the DEBUG-level debug.log handler
        captures it).

    ``event_logger`` is the CALLING module's logger so ``%(name)s`` stays
    accurate; ``stacklevel=2`` makes ``%(filename)s:%(lineno)d`` point at the
    call site rather than this helper.
    """
    if succeeded:
        if getattr(sheet, "verbose_log", False):
            event_logger.info(message, *args, stacklevel=2)
    else:
        event_logger.debug(message, *args, stacklevel=2)


def log_topo_trace(event_logger, sheet, message, *args):
    """Low-level DEBUG trace of a topological SUB-step (closing a face,
    splitting a vertex, ...), emitted only in a verbose run so a quiet run's
    log holds just failed/rejected events."""
    if getattr(sheet, "verbose_log", False):
        event_logger.debug(message, *args, stacklevel=2)

def _min_image_midpoint(sheet, vert_indices):
    coords_arr = sheet.vert_df.loc[vert_indices, sheet.coords].to_numpy()
    if not getattr(sheet, "periodic", False):
        return coords_arr.mean(axis=0)

    periods = np.array([sheet.Lx, sheet.Ly])
    ref = coords_arr[0].copy()
    result = ref.copy()
    for i in range(1, len(coords_arr)):
        disp = coords_arr[i] - ref
        disp -= periods * np.round(disp / periods)
        result += ref + disp
    result = result / len(coords_arr)
    result %= periods

    # Snap values that floating-point pushed just below 0 or just above Lx/Ly
    # back to the canonical 0, so the lookup key is consistent.
    result[result > periods * (1.0 - 1e-10)] = 0.0
    result[np.abs(result) < 1e-10] = 0.0

    return result


class TopologicalEventsHandler:
    def __init__(self, model):
        self.model = model

    def get_ablation_function(self, cell_id, shrink_rate=1.5, critical_area=0.01):

        def ablation(sheet, manager):
            sheet.face_df.loc[cell_id, "type"] = -1
            sheet.face_df.loc[cell_id, "contractility"] = 10
            sheet.face_df.loc[cell_id, "area_elasticity"] = 20
            # The model uses tyssue's (misspelled) "prefered_area"/"prefered_vol"
            # columns — matching the delamination handler. The previous correctly
            # spelled "preferred_area"/"preferred_volume" wrote a dead column, so
            # the ablated cell kept its normal (large) preferred area and, with
            # the raised area_elasticity, BALLOONED instead of collapsing —
            # squeezing a neighbour into a negative area and crashing the solver.
            sheet.face_df.loc[cell_id, "prefered_area"] = 0
            sheet.face_df.loc[cell_id, "prefered_vol"] = 0
            # With a nonzero target perimeter (shape_index>0), a dying cell would
            # otherwise shrink its area to 0 but hold perimeter at P0 -> a
            # degenerate sliver. Drive its perimeter to 0 too so it collapses to a
            # point (no-op when P0=0 / the column is absent).
            if "prefered_perimeter" in sheet.face_df.columns:
                sheet.face_df.loc[cell_id, "prefered_perimeter"] = 0
            log_topo_event(logger, sheet, True,
                           "ablation: tagged cell %d for removal", cell_id)
            return
        return ablation


    def get_delamination_function(self, crit_area=0.5, shrink_rate=1.2):

        def delamination(sheet, manager):
            # First pass: tag every cell below the threshold so its
            # mechanical parameters drive it to shrink. This is a pure
            # column update — no topology change, no reset_index — so
            # iterating over a snapshot is safe.
            for cell_id in sheet.face_df.query(
                f"area < {crit_area}"
            ).index.tolist():
                if cell_id not in sheet.face_df.index:
                    continue
                sheet.face_df.loc[cell_id, "type"] = -1
                sheet.face_df.loc[cell_id, "area_elasticity"] = 20
                sheet.face_df.loc[cell_id, "contractility"] = 10
                sheet.face_df.at[cell_id, "prefered_area"] = 0
                sheet.face_df.at[cell_id, "prefered_vol"] = 0
                # Also drive target perimeter to 0 (shape_index>0 case) so the
                # cell collapses to a point rather than a P0-perimeter sliver.
                if "prefered_perimeter" in sheet.face_df.columns:
                    sheet.face_df.at[cell_id, "prefered_perimeter"] = 0

            # Second pass: remove cells that have shrunk to a triangle (<=3
            # sides) OR collapsed to a degenerate (<=0) area. We re-query after
            # each removal — like the division handler, we can't trust a
            # snapshot because the inner reset_index renumbers face labels.
            tried_ids = set()
            while True:
                # num_sides is ONLY refreshed by the division handler; with
                # divisions off it stays frozen at the initial (large) values,
                # so the `num_sides <= 3` test never fires and shrinking
                # delaminating cells are never removed — they keep collapsing
                # until one inverts (negative area) and kills the solver.
                # Recompute it from the live edge counts every pass. The
                # `area <= 0` clause is a backstop: a delaminating cell that
                # inverts before reaching <=3 sides is still removed (the
                # solver tolerates a type==-1 negative area precisely so this
                # handler gets the chance to clean it up).
                sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
                candidates = sheet.face_df.query(
                    f"area < {crit_area} & ((num_sides <= 3) | (area <= 0))"
                    " & is_alive == 1"
                )
                if "id" in candidates.columns and len(tried_ids):
                    candidates = candidates[~candidates["id"].isin(tried_ids)]
                if not len(candidates):
                    break
                cell_id = int(candidates.index[0])
                cell_uid = (int(sheet.face_df.at[cell_id, "id"])
                            if "id" in sheet.face_df.columns else cell_id)
                try:
                    index_preserving_remove(
                        sheet, cell_id, self.model.sheet.geom,
                    )
                except Exception as exc:
                    log_topo_event(logger, sheet, False,
                                   "delamination: remove_face on cell %d raised "
                                   "%s; skipping", cell_id, type(exc).__name__)
                    tried_ids.add(cell_uid)
                    continue
                sheet.reset_index(order=False)
                sheet.order_all_edges()
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                sheet.geom.update_all(sheet)
                log_topo_event(logger, sheet, True,
                               "delamination: removed cell (uid %d)", cell_uid)
            manager.append(delamination)
            return
        return delamination



    def get_division_function(self, crit_area):
        def division(sheet, manager):
            """Defines a division behavior.

            We re-query the dividing faces after EVERY division. The
            previous version iterated over a snapshot of
            ``dividing_faces`` and called ``reset_index`` inside the
            loop — but ``reset_index`` renumbers face labels, so the
            snapshot's ``cell_id`` for the second iteration could point
            to a completely different cell after the first division.
            That stale label can then be passed to
            ``index_preserving_cell_division``, which dutifully divides
            the WRONG cell — leaving the topology with order gaps,
            negative-area faces, and stretched edges across the whole
            cell. Re-querying makes each iteration use the current,
            valid labelling."""
            # Track cells we've tried but couldn't divide so we don't
            # spin forever on a cell whose geometry doesn't permit a
            # division (get_division_edges returned None).
            tried_ids = set()
            while True:
                candidates = sheet.face_df.query(
                    "area > %f & type == 0 & is_alive == 1" % crit_area
                )
                # Filter out cells we've already failed on this turn.
                # Use the stable "id" column when available so the skip
                # set survives a reset_index from a successful division.
                if "id" in candidates.columns and len(tried_ids):
                    candidates = candidates[~candidates["id"].isin(tried_ids)]
                if not len(candidates):
                    break
                cell_id = int(candidates.index[0])
                cell_uid = (int(sheet.face_df.at[cell_id, "id"])
                            if "id" in sheet.face_df.columns else cell_id)
                daughter = index_preserving_cell_division(
                    sheet, cell_id, sheet.geom,
                )
                if daughter is None:
                    log_topo_event(logger, sheet, False,
                                   "division: cell (uid %d) found no valid "
                                   "split; skipping", cell_uid)
                    tried_ids.add(cell_uid)
                    continue
                sheet.face_df.at[daughter, "id"] = daughter
                # Give the daughter a fresh unique_id. The default
                # ``pd.concat`` in ``index_preserving_face_division``
                # copies mother's row wholesale, so without this fix
                # mother and daughter share unique_id, which (a) collapses
                # them into a single node in ``sheet.lineage`` and (b)
                # makes downstream id-based bookkeeping ambiguous. We
                # mirror tyssue's stock division (basic_events.py) and
                # bump ``specs['face']['unique_id_max']``.
                if "unique_id" in sheet.face_df.columns:
                    if "face" in sheet.specs and "unique_id_max" in sheet.specs["face"]:
                        new_uid = sheet.specs["face"]["unique_id_max"] + 1
                        sheet.specs["face"]["unique_id_max"] = new_uid
                    else:
                        # Fallback: max existing + 1 (test/stub sheets
                        # may not carry specs['face']['unique_id_max']).
                        try:
                            new_uid = int(sheet.face_df["unique_id"].max()) + 1
                        except (TypeError, ValueError):
                            new_uid = daughter
                    sheet.face_df.at[daughter, "unique_id"] = new_uid
                sheet.reset_index(order=False)
                sheet.order_all_edges()
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                sheet.geom.update_all(sheet)
                log_topo_event(logger, sheet, True,
                               "division: cell (uid %d) divided into a new "
                               "daughter", cell_uid)
                # Sanity check: a successful division should leave BOTH
                # mother and daughter with strictly positive area. If
                # the perimeter walk in ``index_preserving_face_division``
                # picked the wrong continuation edge (or some upstream
                # invariant was violated) the resulting polygons cross
                # themselves and we get a negative signed area. Surface
                # this immediately rather than waiting for the solver to
                # explode several steps later with a confusing message.
                m_area = float(sheet.face_df.at[cell_id, "area"]) if cell_id in sheet.face_df.index else None
                d_area = float(sheet.face_df.at[daughter, "area"]) if daughter in sheet.face_df.index else None
                if (m_area is not None and m_area <= 0) or (d_area is not None and d_area <= 0):
                    log.error(
                        "Division of cell %d produced inverted polygon(s): "
                        "mother area=%s, daughter area=%s. "
                        "This indicates wrong edge→face attribution in "
                        "index_preserving_face_division.",
                        cell_id, m_area, d_area,
                    )
            # Keep num_sides in sync — some downstream operations (notably
            # remove_virtual_vertices) read it.
            sheet.face_df["num_sides"] = sheet.edge_df.groupby("face").size()
            manager.append(division)
        return division

    def get_intercalation_function(self, crit_edge_length):
        """Defines an intercalation behavior.

        Periodic boundary-crossing edges are fully supported:
        ``index_preserving_type1_transition`` detects ``is_periodic``
        edges and dispatches to ``_periodic_t1_transition`` which first
        consolidates periodic-image vertex labels around the rosette
        (turning the wrapping rosette into a clean topological rosette)
        and then runs the standard bulk T1."""
        def intercalation(sheet, manager):
            # Re-query after EACH T1. The previous snapshot-based loop
            # would happily call ``index_preserving_type1_transition``
            # with a stale edge_id after ``reset_index`` had renumbered
            # the edges, leading to a T1 on the wrong edge (or on a
            # vertex that no longer exists).
            def _resync():
                # Restore a clean, contiguous index and a consistent derived
                # state (order / opposite / geometry).
                sheet.reset_index(order=False)
                sheet.order_all_edges()
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                sheet.geom.update_all(sheet)

            # A T1 is NOT atomic: ``index_preserving_type1_transition`` runs
            # ``collapse_edge(..., reindex=False)`` — dropping edges and leaving
            # a GAPPY index — BEFORE the split / tri-face removal that may
            # raise. A skipped T1 therefore leaves a partially collapsed sheet
            # with a non-contiguous edge index.
            #
            # We must NOT reset_index between skipped attempts: reset_index
            # renumbers vertices, which would invalidate the (srce, trgt) keys
            # in ``tried`` and let the same failing edge be retried forever
            # (each retry collapses another edge). So we keep the index gappy
            # through the loop — every operation here is label-based, and
            # ``is_virtual_edge`` is given the ACTUAL labels (NOT
            # ``np.arange(shape[0])``, which assumes a contiguous 0..N-1 range
            # and raised "KeyError: [...] not in index" on the gappy index).
            # Any partial collapse is cleaned up by a single resync at the end.
            tried = set()
            pending_resync = False
            while True:
                is_virtual = sheet.is_virtual_edge(sheet.edge_df.index.to_numpy())
                real_edges = sheet.edge_df[~is_virtual]
                candidates = real_edges.query(
                    "is_active > 0 & length < %f" % crit_edge_length
                )
                # Skip edges we already tried (and that failed to T1).
                # Match by (srce, trgt) since edge labels change after
                # reset_index.
                candidate_ids = [
                    eid for eid in candidates.index
                    if (int(candidates.at[eid, "srce"]),
                        int(candidates.at[eid, "trgt"])) not in tried
                ]
                if not candidate_ids:
                    break
                edge_id = int(candidate_ids[0])
                s = int(sheet.edge_df.at[edge_id, "srce"])
                t = int(sheet.edge_df.at[edge_id, "trgt"])
                try:
                    ret = index_preserving_type1_transition(sheet, edge_id)
                except Exception as exc:
                    log_topo_event(logger, sheet, False,
                                   "intercalation: T1 on edge %d (srce=%d, "
                                   "trgt=%d) raised %s; skipping",
                                   edge_id, s, t, type(exc).__name__)
                    tried.add((s, t))
                    pending_resync = True  # collapse may have run before raising
                    continue
                if ret is not None and ret < 0:
                    log_topo_event(logger, sheet, False,
                                   "intercalation: T1 on edge (srce=%d, trgt=%d) "
                                   "returned %s; skipping", s, t, ret)
                    tried.add((s, t))
                    pending_resync = True  # collapse may have partially run
                    continue
                # Successful T1: clean up now. reset_index renumbers, so the
                # stale (srce, trgt) keys in ``tried`` no longer match — clear
                # it (the T1'd edge is gone and won't reappear as a candidate).
                _resync()
                log_topo_event(logger, sheet, True,
                               "intercalation: T1 on edge (srce=%d, trgt=%d)", s, t)
                tried = set()
                pending_resync = False
            # Leave a clean, contiguous index for the solver even when the final
            # attempts were skipped partial collapses.
            if pending_resync:
                _resync()
            manager.append(intercalation)
        return intercalation

# Fixes to tyssue bugs
def index_preserving_add_vert(eptm, edge):
    """Adds a vertex in the middle of the edge,

    which is split as is its opposite(s)

    Parameters
    ----------
    eptm : a :class:`Epithelium` instance
    edge : int
    the index of one of the half-edges to split

    Returns
    -------
    new_vert : int
    the index to the new vertex
    new_edges : int or list of ints
    index to the new edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges
    new_opp_edges : int or list of ints
    index to the new opposite edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges


    In the simple case whith two half-edge, returns
    indices to the new edges, with the following convention:

    s    e    t
      ------>
    * <------ *
    oe

    s    e       ne   t
      ------   ----->
    * <----- * ------ *
        oe   nv   noe

    where "e" is the passed edge as argument, "s" its source "t" its
    target and "oe" its opposite. The returned edges are the ones
    between the new vertex and the input edge's original target.
    """

    srce, trgt = eptm.edge_df.loc[edge, ["srce", "trgt"]]
    stored_opp = int(eptm.edge_df.at[edge, "opposite"])

    # --- Always use the known edge index directly, not a positional search ---
    # This prevents accidentally modifying co-located non-periodic edges.
    parallels = eptm.edge_df.loc[[edge]].copy()

    if stored_opp >= 0:
        opp_srce = int(eptm.edge_df.at[stored_opp, "srce"])
        opposites = eptm.edge_df.loc[[stored_opp]].copy()
    else:
        opp_srce = trgt
        opposites = pd.DataFrame()  # no opposite

    # --- Create midpoint vertex for the main edge ---
    new_vert_data = eptm.vert_df.loc[srce:srce].copy()
    start = eptm.vert_df.index.max() + 1
    new_vert_data.index = range(start, start + len(new_vert_data))
    eptm.vert_df = pd.concat([eptm.vert_df, new_vert_data])
    new_vert = eptm.vert_df.index[-1]
    eptm.vert_df.loc[new_vert, eptm.coords] = _min_image_midpoint(eptm, [srce, trgt])

    # --- Split the main edge ---
    eptm.edge_df.loc[parallels.index, "trgt"] = new_vert
    start = eptm.edge_df.index.max() + 1
    parallels.index = range(start, start + len(parallels))
    eptm.edge_df = pd.concat([eptm.edge_df, parallels])
    new_edges = eptm.edge_df.index[-parallels.index.size:]
    eptm.edge_df.loc[new_edges, "srce"] = new_vert
    eptm.edge_df.loc[new_edges, "trgt"] = trgt

    # --- Split the opposite ---
    # We use the SAME new_vert label as the opposite-side midpoint for
    # both interior AND periodic edges. For interior pairs this is the
    # obvious "shared midpoint". For periodic pairs it works because we
    # consolidate periodic-image vertex labels at construction time
    # (planar_periodic_sheet_2d) — both half-edges of a wrapping pair
    # therefore reference the SAME srce/trgt labels (just reversed), so
    # a single midpoint label keeps the labels-as-opposites invariant.
    # Creating a separate periodic-image midpoint would re-introduce the
    # drift-prone "two separate labels at the same canonical position"
    # situation that consolidation was supposed to eliminate.
    new_opp_edges = []
    new_opp_vert = None

    if len(opposites.index):
        new_opp_vert = new_vert

        eptm.edge_df.loc[opposites.index, "srce"] = new_opp_vert
        start = eptm.edge_df.index.max() + 1
        opposites.index = range(start, start + len(opposites))
        eptm.edge_df = pd.concat([eptm.edge_df, opposites])
        new_opp_edges = eptm.edge_df.index[-opposites.index.size:]
        eptm.edge_df.loc[new_opp_edges, "trgt"] = new_opp_vert
        eptm.edge_df.loc[new_opp_edges, "srce"] = opp_srce
        # No is_periodic mutation here: it will be recomputed from
        # geometry by PeriodicPlanarGeometry.update_dcoords.

    if len(new_edges) == 1:
        new_edges = new_edges[0]
    if len(new_opp_edges) == 1:
        new_opp_edges = new_opp_edges[0]
    elif len(new_opp_edges) == 0:
        new_opp_edges = None

    # Return new_opp_vert so callers can set is_virtual correctly
    return new_vert, new_edges, new_opp_edges, new_opp_vert


def index_preserving_close_face(eptm, face):
    """Closes the face if a single edge is missing.

    This function **does not** close the adjacent and opposite
    faces. Returns the index of the new edge if created, otherwise None
    """
    log_topo_trace(logger, eptm, "closing face %s", face)
    face_edges = eptm.edge_df[eptm.edge_df["face"] == face]
    srces = set(face_edges["srce"])
    trgts = set(face_edges["trgt"])

    if srces == trgts:
        log_topo_trace(logger, eptm, "face %d already closed", face)
        return None
    try:
        (single_srce,) = srces.difference(trgts)
        (single_trgt,) = trgts.difference(srces)
    except ValueError as err:
        print("Closing only possible with exactly two dangling vertices")
        raise err

    face_edges = face_edges.iloc[0:1].copy()
    start = eptm.edge_df.index.max() + 1
    face_edges.index = range(start, start + len(face_edges))
    eptm.edge_df = pd.concat([eptm.edge_df, face_edges])
    eptm.edge_df.index.name = "edge"
    new_edge = eptm.edge_df.index[-1]
    eptm.edge_df.loc[new_edge, ["srce", "trgt"]] = single_trgt, single_srce
    return new_edge

def index_preserving_remove(sheet, face, geom):
    """
    Removes the face and updates the geometry

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class

    """
    index_preserving_remove_face(sheet, face)
    geom.update_all(sheet)

def _drop_antenna_spikes(sheet):
    """Remove degenerate 'antenna' vertices — those joined to a SINGLE distinct
    other vertex — together with their incident half-edges, cascading until
    none remain.

    Such spikes are created when :func:`index_preserving_remove_face` collapses
    all of a removed cell's vertices onto one point ``new_vert``: any surviving
    vertex left with a single distinct neighbour (e.g. a virtual mid-edge vertex
    whose two endpoints both merged, or a corner left dangling) is no longer a
    real polygon corner but a backtracking ``A->v->A`` antenna. Because that
    spike vertex is shared by the faces on both sides, it shows up as DUPLICATE
    ``(srce, trgt)`` half-edges in two faces — which ``drop_two_sided_faces``
    can't see (those faces still have >2 sides) and which makes ``get_opposite``
    emit "Duplicated (`srce`, `trgt`) values in edge_df". A real polygon corner
    always has >=2 distinct neighbours, so peeling off the <=1 case is safe and
    a no-op on a clean removal."""
    while True:
        e = sheet.edge_df
        if e.empty:
            return
        nb = pd.concat([
            e[["srce", "trgt"]].rename(columns={"srce": "v", "trgt": "n"}),
            e[["trgt", "srce"]].rename(columns={"trgt": "v", "srce": "n"}),
        ], ignore_index=True)
        distinct_neighbours = nb.groupby("v")["n"].nunique()
        spikes = distinct_neighbours.index[distinct_neighbours <= 1]
        if not len(spikes):
            return
        drop = e.index[e["srce"].isin(spikes) | e["trgt"].isin(spikes)]
        if not len(drop):
            return
        sheet.edge_df.drop(drop, axis=0, inplace=True)


def index_preserving_remove_face(sheet, face):
    """Removes a face from the mesh.

    Returns the index of the new vert that replaces the face."""
    log_topo_trace(logger, sheet, "removing face %d", face)

    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    verts = edges["srce"].unique()

    new_vert_data = sheet.vert_df.loc[verts[0] : verts[0]].copy()
    new_vert_data[sheet.coords] = _min_image_midpoint(sheet, verts)
    new_vert_data = pd.DataFrame(new_vert_data)
    start = sheet.vert_df.index.max() + 1
    new_vert_data.index = range(start, start + len(new_vert_data))
    sheet.vert_df = pd.concat([sheet.vert_df, new_vert_data])
    new_vert = sheet.vert_df.index[-1]

    # collapse all edges connected to the face vertices
    sheet.edge_df.replace({"srce": verts, "trgt": verts}, new_vert, inplace=True)

    collapsed = sheet.edge_df.query("srce == trgt")

    sheet.edge_df.drop(collapsed.index, axis=0, inplace=True)
    remanent = sheet.edge_df.query(f"face == {face}").index
    if remanent.shape[0]:
        warnings.warn(f"something fishy with face {face}")
        sheet.edge_df.drop(remanent, axis=0, inplace=True)

    # Peel off any degenerate antenna spikes the collapse created (the source of
    # the "Duplicated (`srce`, `trgt`)" warnings during ablation/delamination).
    _drop_antenna_spikes(sheet)

    sheet.lineage.add_node(str(sheet.face_df.loc[face]['unique_id']),
                           color='black')

    sheet.face_df.drop(face, axis=0, inplace=True)

    log_topo_event(logger, sheet, True,
                   "removed face %d (dropped %d vertices; cell now dead)",
                   face, len(verts))
    drop_two_sided_faces(sheet)

    # Keep only vertices still referenced by an edge: the removed face's
    # collapsed verts, antenna-spike orphans, and anything drop_two_sided_faces
    # freed. (Replaces the old ``drop(verts)``, which dropped only the first of
    # those three and so left the spike orphans behind.)
    used = pd.unique(sheet.edge_df[["srce", "trgt"]].values.ravel())
    sheet.vert_df = sheet.vert_df.loc[sheet.vert_df.index.intersection(used)]

    sheet.reset_index()
    sheet.reset_topo()

    return new_vert

def index_preserving_cell_division(sheet, mother, geom, angle=None,
                                    min_area_ratio=0.01):
    """Divide ``mother`` along a cleavage line at ``angle`` (random
    when ``None``).

    Returns the daughter face label on success, ``None`` on failure
    (cell not alive, no valid cleavage edges, or the division
    produced a degenerate polygon).

    Degenerate-result rollback
    --------------------------
    ``index_preserving_face_division`` only guarantees the
    combinatorial topology of the result (≥3 edges per side). The
    GEOMETRIC consequence — a daughter whose perimeter self-crosses,
    so the signed area is microscopically negative — only shows up
    after ``geom.update_all`` recomputes the polygon areas. That
    case manifested in ``results/random_periodic_array1/`` at
    t=17.654: ``get_division_edges`` picked two virtual sub-edges so
    close together on the mother's perimeter that the resulting
    5-vertex daughter was a self-touching sliver with signed area
    -4.5e-3. The solver then rejected every subsequent dt down to
    dt_min and crashed.

    To avoid that, after the division we evaluate both polygons.
    A degenerate split is one where:
      (a) either side has a non-positive area, OR
      (b) the smaller-area side is less than
          ``min_area_ratio`` × the larger-area side.

    Both checks are pure RATIOS between the two post-division
    polygons — they're indifferent to whatever
    ``mother_area_before`` happened to be, which is important
    because callers sometimes pre-set ``face_df["area"]`` to an
    artificial sentinel (e.g. the multi-division tests load
    ``area = 999`` to mark "must divide"; the real area only
    materialises after ``geom.update_all``).

    On failure we roll back by calling
    ``index_preserving_remove_face`` on the daughter. ``remove_face``
    collapses the daughter's vertices into a single centroid which
    the surrounding mother absorbs — the cell ends up unchanged-ish,
    slightly remeshed, and the caller (``get_division_function``)
    marks it as tried and continues. On the next manager pass
    mechanics will have shifted things, and a fresh random angle is
    likely to succeed.
    """
    if not sheet.face_df.loc[mother, "is_alive"]:
        log_topo_event(logger, sheet, False,
                       "division: cell %s is not alive and cannot divide", mother)
        return None

    # For periodic sheets there's no global vertex shift to worry about:
    # PeriodicPlanarGeometry.face_projected_pos already projects the
    # mother in its unfolded frame, so get_division_edges sees a
    # contiguous polygon. The new vertices created by index_preserving_add_vert
    # are placed at minimum-image midpoints and wrapped to canonical
    # coords by _min_image_midpoint, so the sheet stays in [0, L)
    # throughout.
    edge_a, edge_b = get_division_edges(sheet, mother, geom, angle=angle, axis="x")
    if edge_a is None:
        return None

    vert_a, *_ = index_preserving_add_vert(sheet, edge_a)
    vert_b, *_ = index_preserving_add_vert(sheet, edge_b)
    sheet.vert_df.index.name = "vert"
    daughter = index_preserving_face_division(sheet, mother, vert_a, vert_b)
    if daughter is None:
        return None

    # --- Degeneracy check ----------------------------------------------
    # Areas are computed by geom.update_all after the division. Without
    # this update, face_df.area still holds the pre-division mother area
    # and the freshly-appended daughter row inherits that value via
    # pd.concat — both look fine and the check would let the bad
    # polygon through.
    #
    # IMPORTANT: repair the mother's and daughter's edge ``order`` first.
    # ``index_preserving_face_division`` appends the new edges by copying
    # an existing edge row (so they inherit a stale ``order``) and never
    # rewrites the column, leaving e.g. order = [1, 2, 7, 6, 1] on the
    # mother. The periodic geometry (PeriodicPlanarGeometry.update_dcoords)
    # builds each face polygon by walking its edges in ``order`` sequence
    # — with a broken order it produces a self-tangled polygon whose area
    # collapses to ~0, which would trip the degeneracy test below and roll
    # back a perfectly valid division (observed on adjacent same-category
    # divisions in scenario 9). Re-walking just these two faces' perimeters
    # restores a correct order so the area measurement is meaningful. The
    # caller re-orders the whole sheet afterwards; this only fixes what the
    # check itself needs.
    if hasattr(sheet, "order_edges"):
        for _f in (mother, daughter):
            try:
                sheet.order_edges(int(_f))
            except (IndexError, ValueError, KeyError):
                # A genuinely broken perimeter will surface as a
                # degenerate area below and be rolled back as before.
                pass
    geom.update_all(sheet)
    if daughter not in sheet.face_df.index or mother not in sheet.face_df.index:
        # face_division shouldn't be dropping rows, but if for any
        # reason the labels disappeared just bail without further
        # damage.
        return None
    d_area = float(sheet.face_df.at[daughter, "area"])
    m_area = float(sheet.face_df.at[mother, "area"])

    # Pure RATIO check between the two post-division polygons. This
    # is independent of any pre-division ``face_df["area"]`` value
    # (which can be a sentinel — see the docstring).
    small = min(d_area, m_area)
    large = max(d_area, m_area)
    is_degenerate = (
        small <= 0.0
        or (large > 0 and small < float(min_area_ratio) * large)
    )
    if is_degenerate:
        log_topo_event(
            logger, sheet, False,
            "division: cell %s gave degenerate polygon "
            "(mother area=%.4g, daughter area=%.4g, "
            "min/max ratio=%.4g, threshold=%.2g); rolling back via "
            "index_preserving_remove_face on the daughter so the bad "
            "geometry doesn't reach the solver.",
            mother, m_area, d_area,
            (small / large) if large > 0 else float("-inf"),
            min_area_ratio,
        )
        try:
            index_preserving_remove_face(sheet, daughter)
        except Exception as exc:
            logger.error(
                "Rollback of degenerate division on cell %s failed (%s); "
                "the daughter row may still be present on the sheet.",
                mother, exc,
            )
        return None

    return daughter

def index_preserving_face_division(sheet, mother, vert_a, vert_b):
    """
    Divides the face associated with edges
    indexed by `edge_a` and `edge_b`, splitting it
    in the middle of those edes.

    The walk that discovers the daughter side of the cleavage line is
    SENSITIVE to the row order of ``edge_df`` after the new edges have
    been appended. The vertex ``vert_b`` has TWO outgoing mother-face
    edges right at this moment — the continuation half-edge created by
    ``index_preserving_add_vert`` (vert_b → original_trgt_b) AND the
    just-created ``new_edge_m`` (vert_b → vert_a). If ``new_edge_m``
    were picked first, the walk would terminate immediately, putting
    BOTH new central-line edges into ``daughter`` and leaving the
    mother with no closing edge — its polygon would degenerate, the
    daughter polygon would self-cross (negative area), and the next
    geometry update would produce wildly long edges on the broken
    perimeter.

    To make this robust independent of insertion order, we explicitly
    EXCLUDE the two just-created central-line edges from the candidate
    set used for the walk lookup. They are added to ``daughter_edges``
    only once via the explicit seed ``daughter_edges = [new_edge_d]``.
    """
    # mother = sheet.edge_df.loc[edge_a, 'face']

    face_cols = sheet.face_df.loc[mother:mother]

    sheet.face_df = pd.concat([sheet.face_df, face_cols], ignore_index=True)
    sheet.face_df.index.name = "face"
    daughter = int(sheet.face_df.index[-1])

    edge_cols = sheet.edge_df[sheet.edge_df["face"] == mother].iloc[0:1]
    mother_edges = edge_cols.copy()
    daughter_edges = edge_cols.copy()
    mother_start = sheet.edge_df.index.max() + 1
    mother_edges.index = range(mother_start, mother_start + len(mother_edges))
    daughter_start = mother_edges.index.max() + 1
    daughter_edges.index = range(daughter_start, daughter_start + len(daughter_edges))
    sheet.edge_df = pd.concat([sheet.edge_df, mother_edges, daughter_edges])
    new_edge_m = sheet.edge_df.index[-2]
    sheet.edge_df.loc[new_edge_m, "srce"] = vert_b
    sheet.edge_df.loc[new_edge_m, "trgt"] = vert_a
    new_edge_d = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_d, "srce"] = vert_a
    sheet.edge_df.loc[new_edge_d, "trgt"] = vert_b

    # ## Discover daughter edges
    # Walk the daughter side of the cleavage line. We EXCLUDE the two
    # new central-line edges (new_edge_m, new_edge_d) from the lookup
    # so that the [0] selection at each step picks the next perimeter
    # edge (the continuation from add_vert) and never accidentally the
    # central line itself.
    m_data = sheet.edge_df[
        (sheet.edge_df["face"] == mother)
        & (~sheet.edge_df.index.isin([new_edge_m, new_edge_d]))
    ]
    daughter_edges_walk = [new_edge_d]
    srce, trgt = vert_a, vert_b
    srces, trgts = m_data[["srce", "trgt"]].values.T
    edge_index_arr = m_data.index.to_numpy()
    spins = 0

    while trgt != vert_a:
        next_mask = srces == trgt
        if not next_mask.any():
            raise ValueError(
                f"Division of face {mother}: no outgoing mother edge "
                f"at vertex {trgt} during daughter walk (mother edges: "
                f"{len(m_data)})."
            )
        srce, trgt = trgt, trgts[next_mask][0]

        # Find the edge that took us from srce → trgt. Restrict to the
        # m_data view so we never pick up new_edge_m / new_edge_d.
        match_mask = (srces == srce) & (trgts == trgt)
        if not match_mask.any():
            raise ValueError(
                f"Division of face {mother}: no edge srce={srce}, "
                f"trgt={trgt} in mother's perimeter."
            )
        daughter_edges_walk.append(int(edge_index_arr[match_mask][0]))
        spins += 1
        if spins > m_data.shape[0]:
            raise ValueError(f"The face {mother} has an invalid topology, \n")
    sheet.edge_df.loc[daughter_edges_walk, "face"] = daughter
    sheet.edge_df.index.name = "edge"

    # Defensive: a successful division MUST leave the mother and the
    # daughter with at least 3 edges each (a closed simple polygon).
    # Anything less is a topology bug — raise now so the caller can
    # roll back / skip rather than letting the broken state seep into
    # the next solver step.
    m_count = int((sheet.edge_df["face"] == mother).sum())
    d_count = int((sheet.edge_df["face"] == daughter).sum())
    if m_count < 3 or d_count < 3:
        raise ValueError(
            f"Division of face {mother} left a degenerate polygon: "
            f"mother has {m_count} edges, daughter has {d_count}. "
            f"new_edge_m={new_edge_m}, new_edge_d={new_edge_d}, "
            f"walked {len(daughter_edges_walk)} daughter edges."
        )

    sheet.reset_topo()
    return daughter

def index_preserving_type1_transition(sheet, edge01, *, remove_tri_faces=True, multiplier=1.5):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    See Finegan et al. for a description of the algotithm https://doi.org/10.1101/704932


    Parameters
    ----------
    sheet : a `Sheet` instance
    edge_01 : int
       index of the edge around which the transition takes place
    epsilon : float, optional, deprecated
       default 0.1, the initial length of the new edge, in case "threshold_length"
       is not in the sheet.settings
    remove_tri_faces : bool, optional
       if True (the default), will remove triangular cells
       after the T1 transition is performed
    multiplier : float, optional
       default 1.5, the multiplier to the threshold length, so that the
       length of the new edge is set to multiplier * threshold_length


    """
    # A T1 is NOT atomic internally: ``collapse_edge(reindex=False)`` drops edges
    # and merges vertices BEFORE ``index_preserving_split_vert`` / tri-face
    # removal, any of which can raise. A partially-applied T1 leaves DUPLICATE
    # (srce, trgt) half-edges — a non-manifold mesh that ``reset_topo`` /
    # antenna-spike healing CAN'T repair (seen as the "NON-spike duplicate ...
    # [cross-face]" warnings during intercalation). Snapshot the topology and
    # restore it on ANY failure so a failed T1 is a clean no-op; the caller
    # (e.g. the intercalation handler) then just marks the edge as tried.
    _edge_bak = sheet.edge_df.copy()
    _vert_bak = sheet.vert_df.copy()
    _face_bak = sheet.face_df.copy()

    def _restore():
        sheet.edge_df = _edge_bak
        sheet.vert_df = _vert_bak
        sheet.face_df = _face_bak

    try:
        if (
            getattr(sheet, "periodic", False)
            and "is_periodic" in sheet.edge_df.columns
            and bool(sheet.edge_df.at[edge01, "is_periodic"])
        ):
            ret = _periodic_t1_transition(
                sheet, edge01,
                remove_tri_faces=remove_tri_faces,
                multiplier=multiplier,
            )
        else:
            ret = _bulk_t1_transition(
                sheet, edge01,
                remove_tri_faces=remove_tri_faces,
                multiplier=multiplier,
            )
    except Exception:
        _restore()
        raise
    # A negative return code means the T1 declined partway (e.g. the collapse
    # failed); restore so no partial change is left behind either.
    if ret is not None and ret < 0:
        _restore()
    return ret


def _bulk_t1_transition(sheet, edge01, *, remove_tri_faces=True, multiplier=1.5):
    """The standard non-periodic T1 transition. Called directly for
    bulk edges, and inside the wrapper for periodic edges after a
    global shift that brings the edge into the bulk."""
    srce, trgt, face = sheet.edge_df.loc[edge01, ["srce", "trgt", "face"]].astype(int)

    vert = min(srce, trgt)  # find the vertex that won't be reindexed
    ret_code = collapse_edge(sheet, edge01, reindex=False, allow_two_sided=True)
    if ret_code < 0:
        warnings.warn(f"Collapse of edge {edge01} failed")
        return ret_code

    index_preserving_split_vert(
        sheet,
        vert,
        face,
        multiplier=multiplier,
        reindex=False,
        recenter=True,
    )

    if not remove_tri_faces:
        return 0
    # Type 1 transitions might create 3 or 2 sided cells, we remove those
    tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
    i = 0
    while len(tri_faces):
        index_preserving_remove_face(sheet, tri_faces[0])
        tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
        i += 1
        if i > MAX_ITER:
            raise RecursionError
    return 0


def _consolidate_periodic_image_labels(sheet, target_labels, prec=6):
    """Merge periodic-image vertex labels at the canonical positions of
    ``target_labels`` into a single label per position.

    When a periodic sheet is built via Voronoi on a 3×3 supercell, the
    same physical point at the boundary of the central tile appears as
    SEVERAL distinct vertex labels — one per cell that touches that
    point. e.g. vertex 4 at canonical (1, 3.625) is "face 0's view"
    while vertex 37 at the same (1, 3.625) is "face 13's view".

    Standard T1 only knows about vertex labels; it can't see that those
    two labels refer to the same physical corner. Consolidating them
    BEFORE the T1 turns the rosette into a clean topological rosette
    that the standard T1 handles correctly.

    Returns a dict mapping each input label to the label it was
    consolidated into.
    """
    Lx, Ly = float(sheet.Lx), float(sheet.Ly)
    # Bucket every vertex by its (rounded, wrapped) canonical position
    pos_to_labels = {}
    for v in sheet.vert_df.index:
        x = round(float(sheet.vert_df.at[v, "x"]) % Lx, prec)
        y = round(float(sheet.vert_df.at[v, "y"]) % Ly, prec)
        pos_to_labels.setdefault((x, y), []).append(v)

    # For each target, find the equivalence class and pick a canonical label
    label_map = {}
    drops = {}  # drop -> keep
    for target in target_labels:
        if target not in sheet.vert_df.index:
            label_map[target] = target
            continue
        x = round(float(sheet.vert_df.at[target, "x"]) % Lx, prec)
        y = round(float(sheet.vert_df.at[target, "y"]) % Ly, prec)
        equiv = pos_to_labels.get((x, y), [target])
        keep = min(equiv)
        label_map[target] = keep
        for v in equiv:
            if v != keep:
                drops[v] = keep

    if not drops:
        return label_map

    # Rewire every edge referencing a dropped label to use its kept label.
    srce_remap = sheet.edge_df["srce"].map(drops)
    trgt_remap = sheet.edge_df["trgt"].map(drops)
    sheet.edge_df["srce"] = srce_remap.fillna(sheet.edge_df["srce"]).astype(int)
    sheet.edge_df["trgt"] = trgt_remap.fillna(sheet.edge_df["trgt"]).astype(int)

    # Drop any self-loop edges that the rewiring may have created (two
    # vertices that previously bracketed the boundary now coincide).
    self_loops = sheet.edge_df.query("srce == trgt").index
    sheet.edge_df.drop(self_loops, inplace=True)

    # Drop the now-redundant vertex labels.
    drop_labels = [v for v in drops if v in sheet.vert_df.index]
    if drop_labels:
        sheet.vert_df.drop(drop_labels, inplace=True)

    return label_map


def _close_all_open_faces(sheet, max_passes=8):
    """Patch every face that has dangling vertex pairs by adding edges
    between them.

    ``index_preserving_close_face`` only handles faces with EXACTLY one
    dangling srce/trgt pair. After a periodic T1 a face can have several
    dangling pairs (the new T1 edges in periodic-image cells were never
    created by split_vert, which only sweeps its own connected set).
    For each face, repeatedly pair the dangling vertices and add the
    missing edges until the face's perimeter is closed (srces == trgts).
    """
    for _ in range(max_passes):
        changed = False
        for face_id in list(sheet.face_df.index):
            face_edges = sheet.edge_df[sheet.edge_df["face"] == face_id]
            if len(face_edges) == 0:
                continue
            srces = list(face_edges["srce"])
            trgts = list(face_edges["trgt"])
            # Dangling srces (not consumed as a trgt elsewhere) and
            # dangling trgts (not produced as a srce elsewhere).
            srce_counts = {}
            for s in srces:
                srce_counts[s] = srce_counts.get(s, 0) + 1
            trgt_counts = {}
            for t in trgts:
                trgt_counts[t] = trgt_counts.get(t, 0) + 1
            dangling_srce = []  # vertices appearing as srce more often than trgt
            dangling_trgt = []  # vertices appearing as trgt more often than srce
            for v in set(srces) | set(trgts):
                ds = srce_counts.get(v, 0) - trgt_counts.get(v, 0)
                if ds > 0:
                    dangling_srce.extend([v] * ds)
                elif ds < 0:
                    dangling_trgt.extend([v] * (-ds))
            if not dangling_srce and not dangling_trgt:
                continue
            if len(dangling_srce) != len(dangling_trgt):
                # Topology too broken for a simple patch; bail.
                continue
            # Pair each dangling_trgt → dangling_srce by nearest position
            # so the added edges are short.
            vx = sheet.vert_df["x"].to_dict()
            vy = sheet.vert_df["y"].to_dict()
            template = face_edges.iloc[0:1].copy()
            for _ in range(len(dangling_srce)):
                if not dangling_srce or not dangling_trgt:
                    break
                # Pair the first dangling_trgt with the nearest dangling_srce
                t = dangling_trgt.pop(0)
                tx, ty = vx[t], vy[t]
                best_i = 0
                best_d = float("inf")
                Lx = float(getattr(sheet, "Lx", 0.0)) or 1.0
                Ly = float(getattr(sheet, "Ly", 0.0)) or 1.0
                for i, s in enumerate(dangling_srce):
                    sx, sy = vx[s], vy[s]
                    dx = sx - tx
                    dy = sy - ty
                    if getattr(sheet, "periodic", False):
                        dx -= Lx * round(dx / Lx)
                        dy -= Ly * round(dy / Ly)
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best_d = d
                        best_i = i
                s = dangling_srce.pop(best_i)
                # Add edge t → s in this face
                start = int(sheet.edge_df.index.max()) + 1
                new = template.copy()
                new.index = [start]
                new.loc[start, "srce"] = t
                new.loc[start, "trgt"] = s
                sheet.edge_df = pd.concat([sheet.edge_df, new])
                changed = True
        if not changed:
            break


def _periodic_t1_transition(sheet, edge01, *, remove_tri_faces=True, multiplier=1.5):
    """T1 on a wrapping edge — option (a), via rosette consolidation.

    The problem with the naive approach is that when a Voronoi periodic
    sheet is built from a 3×3 supercell, vertices on the central tile's
    boundary appear as MULTIPLE distinct labels — one per cell that
    touches that physical point. e.g. canonical (1, 3.625) might be
    vertex 4 (from face 0/1's perspective) AND vertex 37 (from face 13's
    perspective). The standard T1 only operates on one label and leaves
    the other(s) dangling, breaking the topology.

    Algorithm:

    1. Identify every vertex label at the canonical positions of
       edge01's srce and trgt. This includes face_A/face_B's labels
       AND any periodic-image labels used by other neighbouring cells.
    2. Consolidate: keep one label per position, rewire all edges,
       drop the redundant labels. After this the rosette around the
       wrapping edge is a clean topological rosette with single labels.
    3. Pre-shift trgt to be the min-image of srce so collapse_edge's
       arithmetic-mean midpoint lands at the right physical location.
    4. Run the standard bulk T1 (collapse + split + close_face). It
       handles the (now-consolidated) rosette correctly.
    5. Re-wrap into [0, L), re-stitch periodic opposites by position,
       re-order edges anti-clockwise.
    """
    Lx, Ly = float(sheet.Lx), float(sheet.Ly)
    srce_orig, trgt_orig = sheet.edge_df.loc[edge01, ["srce", "trgt"]].astype(int)
    face_A = int(sheet.edge_df.at[edge01, "face"])
    opp = int(sheet.edge_df.at[edge01, "opposite"])
    face_B = int(sheet.edge_df.at[opp, "face"]) if opp >= 0 else -1

    # 1-2: consolidate periodic-image labels at EVERY vertex used by
    # face_A or face_B. The T1 affects the whole rosette around edge01,
    # which includes the perpendicular cells' shared corners — those
    # also need single labels per physical position.
    rosette_labels = set()
    for f in (face_A, face_B):
        if f >= 0 and f in sheet.face_df.index:
            verts = sheet.edge_df[sheet.edge_df["face"] == f]["srce"].to_numpy()
            rosette_labels.update(int(v) for v in verts)
    label_map = _consolidate_periodic_image_labels(sheet, sorted(rosette_labels))
    srce = int(label_map.get(srce_orig, srce_orig))
    trgt = int(label_map.get(trgt_orig, trgt_orig))

    # edge01 might have been dropped if consolidation made it a self-loop
    # (impossible for a wrapping edge since srce and trgt are at
    # different canonical positions, but defensive).
    if edge01 not in sheet.edge_df.index:
        # Find an edge with the (now-consolidated) srce/trgt in face_A
        face_A = None  # face was lost
        warnings.warn(
            f"Periodic T1: edge {edge01} disappeared during consolidation",
            RuntimeWarning,
        )
        return -1

    # 3: pre-shift trgt to the min-image of srce so collapse_edge's
    # arithmetic mean produces the right midpoint.
    sx_ = float(sheet.vert_df.at[srce, "x"])
    sy_ = float(sheet.vert_df.at[srce, "y"])
    tx_ = float(sheet.vert_df.at[trgt, "x"])
    ty_ = float(sheet.vert_df.at[trgt, "y"])
    dx = tx_ - sx_
    dy = ty_ - sy_
    shift_x = -Lx if dx > Lx / 2 else (Lx if dx < -Lx / 2 else 0.0)
    shift_y = -Ly if dy > Ly / 2 else (Ly if dy < -Ly / 2 else 0.0)
    if shift_x or shift_y:
        sheet.vert_df.at[trgt, "x"] = tx_ + shift_x
        sheet.vert_df.at[trgt, "y"] = ty_ + shift_y

    # 4: standard bulk T1. With the rosette consolidated and trgt
    # pre-shifted, this behaves like an interior T1.
    ret = _bulk_t1_transition(
        sheet, edge01,
        remove_tri_faces=remove_tri_faces,
        multiplier=multiplier,
    )

    # 5: re-wrap into canonical box, re-stitch periodic opposites,
    # re-order edges anti-clockwise BEFORE the geometry update so the
    # per-face unfolding walks the perimeter in the correct order
    # (otherwise the unfolded sx, sy can land in the wrong image and
    # produce a self-intersecting polygon with the wrong area).
    sheet.vert_df["x"] = sheet.vert_df["x"] % Lx
    sheet.vert_df["y"] = sheet.vert_df["y"] % Ly
    sheet.reset_index(order=False)
    sheet.get_opposite()
    if hasattr(sheet, "order_all_edges"):
        sheet.order_all_edges()
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
    sheet.geom.update_all(sheet)
    return ret

def index_preserving_split_vert(
    sheet, vert, face=None, multiplier=1.5, reindex=True, recenter=False, epsilon=None
):
    """Splits a vertex towards the center of the face.

    This operation removes the  face `face` from the neighborhood of the vertex.

    Returns a list of the new edge's indices in edge_df. This should be two:
        the edge with vert as the srce and the newly created vertex as the trgt
        and the reverse edge with vert as the trgt and the new one as the srce
    """
    # Get the value for the length of the new edge
    if epsilon is None:
        epsilon = sheet.settings.get("threshold_length", 0.1) * multiplier
    else:
        warnings.warn(
            "The epsilon argument is deprecated and will be removed"
            " in a future version. "
            "The length of the new edge should be set by "
            "`sheet.settings['threshold_length]*multiplier` "
        )
    if face is None:
        face = np.random.choice(sheet.edge_df[sheet.edge_df["srce"] == vert]["face"])

    face_edges = sheet.edge_df.query(f"face == {face}")
    (prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]
    (next_v,) = face_edges[face_edges["srce"] == vert]["trgt"]
    connected = sheet.edge_df[
        sheet.edge_df["trgt"].isin((next_v, prev_v))
        | sheet.edge_df["srce"].isin((next_v, prev_v))
    ]

    index_preserving_base_split_vert(sheet, vert, face, connected, epsilon, recenter)
    new_edges = []
    for face_ in connected["face"]:
        new_edge = index_preserving_close_face(sheet, face_)
        if new_edge is not None:
            new_edges.append(new_edge)

    if reindex:
        sheet.reset_index()
        sheet.reset_topo()

    return new_edges

def index_preserving_base_split_vert(sheet, vert, face, to_rewire, epsilon, recenter=False):
    """Creates a new vertex and moves it towards the center of face.

    The edges in to_rewire will be connected to the new vertex.

    Parameters
    ----------

    sheet : a :class:`tyssue.Sheet` instance
    vert : int, the index of the vertex to split
    face : int, the index of the face where to move the vertex
    to_rewire : :class:`pd.DataFrame` a subset of `sheet.edge_df`
        where all the edges pointing to (or from) the old vertex will point
        to (or from) the new.

    Note
    ----

    This will leave opened faces and cells

    """
    log_topo_trace(logger, sheet, "splitting vertex %d", vert)

    # Add a vertex
    this_vert = sheet.vert_df.loc[vert:vert].copy()  # avoid type munching
    start = sheet.vert_df.index.max() + 1
    this_vert.index = range(start, start + len(this_vert))
    sheet.vert_df = pd.concat([sheet.vert_df, this_vert])

    new_vert = sheet.vert_df.index[-1]
    # Move it towards the face center. Cast to plain float arrays —
    # face_df may have mixed-dtype columns (e.g. the periodic metadata
    # stash) which turn .to_numpy() into an object array, breaking np.round.
    face_pos = sheet.face_df.loc[face, sheet.coords].to_numpy().astype(float)
    vert_pos = sheet.vert_df.loc[vert, sheet.coords].to_numpy().astype(float)
    r_ia = face_pos - vert_pos
    if getattr(sheet, "periodic", False):
        periods = np.array([float(sheet.Lx), float(sheet.Ly)])
        r_ia -= periods * np.round(r_ia / periods)
    shift = r_ia * epsilon / np.linalg.norm(r_ia)
    if recenter:
        sheet.vert_df.loc[new_vert, sheet.coords] += shift / 2.0
        sheet.vert_df.loc[vert, sheet.coords] -= shift / 2.0

    else:
        sheet.vert_df.loc[new_vert, sheet.coords] += shift

    # rewire
    sheet.edge_df.loc[to_rewire.index] = to_rewire.replace(
        {"srce": vert, "trgt": vert}, new_vert
    )