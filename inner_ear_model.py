"""A two-cell-type vertex model with differentiation, and the effectors it needs.

Extends a tyssue vertex model in three ways:

1. **Virtual vertices**, so cell borders can curve (see ``virtual_sheet``).
2. **Per-type mechanics** — tension, contractility, area elasticity, preferred
   area and repulsion are looked up per cell type, so primary and secondary cells
   can differ mechanically, and a cell's parameters follow it when it
   differentiates.
3. **Differentiation**, by lateral inhibition, optionally gated by mechanical
   stress (see ``lateral_inhibition_model``).

The class assembles the effector stack, drives the solver, and exposes the
measurements the coupling needs — per-cell stress, neighbour relations, contact
lengths — plus archive and drawing helpers.

Two effectors are defined here rather than taken from tyssue:

* :class:`ContractilityPerimeterElasticity` — perimeter elasticity about a
  preferred perimeter. With shape index 0 the preferred perimeter is 0 and this
  reduces exactly to face contractility, Gamma/2 * P^2.
* :class:`BoundaryBending` — penalises CURVATURE at virtual vertices rather than
  length. Line tension is the obvious way to smooth a bond, but it also shortens
  it, so it fights any elongation the geometry calls for; bending smooths without
  that conflict.

Vocabulary: cell types are stored as ``'HC'`` and ``'SC'`` and the signal columns
as ``atoh_level`` / ``delta_level``, named for the biological system this was
written for; read them as primary / secondary and signal.
"""
import os.path
import sys
import tyssue
from tyssue import config, History, HistoryHdf5
from tyssue.behaviors import EventManager
from solvers import IVPSolver
from tyssue.dynamics import model_factory
from tyssue.dynamics.effectors import (LineTension, FaceAreaElasticity, FaceContractility,
                                       PerimeterElasticity, AbstractEffector,
                                       elastic_energy, elastic_force)
from tyssue.dynamics import units
from tyssue.utils import to_nd

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from empty_effector import EmptyAffector
from topological_events import TopologicalEventsHandler


class ContractilityPerimeterElasticity(PerimeterElasticity):
    """Perimeter elasticity ``1/2 * contractility * (perimeter - prefered_perimeter)**2``.

    A drop-in generalization of :class:`FaceContractility`
    (``1/2*contractility*perimeter**2``): it reuses the SAME per-face
    ``contractility`` column as the elasticity magnitude — so it stays
    delta/atoh-dependent (via ``update_cell_type_parameters``) and delamination /
    ablation keep driving cells to shrink by raising ``contractility`` — but adds
    a target perimeter ``prefered_perimeter`` (= P0). With ``prefered_perimeter``
    = 0 it is IDENTICAL to ``FaceContractility``.

    P0 sets the target shape index p0 = P0 / sqrt(A0), the vertex-model control
    parameter for cell shape (rigidity transition ~3.81): p0 below it -> round,
    uniform cells; above it -> elongated, variable cells. It is thus the lever
    for fitting the roundness distribution.
    """
    magnitude = "contractility"
    label = "Contractility perimeter elasticity"
    specs = {
        "face": {
            "is_alive": 1,
            "perimeter": 1.0,
            "contractility": 1.0,
            "prefered_perimeter": 0.0,
        }
    }

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.face_df, "perimeter", "contractility * is_alive", "prefered_perimeter")

    @staticmethod
    def gradient(eptm):
        gamma_ = elastic_force(
            eptm.face_df, "perimeter", "contractility * is_alive", "prefered_perimeter")
        gamma = eptm.upcast_face(gamma_)
        grad_srce = -eptm.edge_df[eptm.ucoords] * to_nd(gamma, len(eptm.coords))
        grad_srce.columns = ["g" + u for u in eptm.coords]
        grad_trgt = -grad_srce
        return grad_srce, grad_trgt


class BoundaryBending(AbstractEffector):
    """Uniform BENDING (curvature) energy along cell boundaries::

        E = sum over corners of  kappa * (1 - cos(theta))

    where ``theta`` is the turn between the two consecutive boundary edges meeting
    at a corner. E = 0 for a perfectly straight boundary and grows as it kinks.

    Applied ONLY at VIRTUAL vertices — the mid-bond subdivision points. Real
    vertices are genuine 3-cell junctions whose ~120 deg corners are physical;
    penalising those would simply round the cells off, which is what we are trying
    to avoid.

    Penalising CURVATURE rather than LENGTH is the whole point. Line tension
    shortens the perimeter, so it fights the very elongation needed to reach the
    experimental roundness (measured: smoothing and roundness trade off directly).
    Bending instead keeps bonds STRAIGHT while leaving the perimeter free — the
    combination needed for smooth but elongated cells.

    ``kappa`` is the per-face ``bending`` column; 0 (the default) makes this
    effector an exact no-op.
    """
    dimensions = units.line_elasticity
    magnitude = "bending"
    label = "Boundary bending"
    element = "face"
    specs = {"face": {"is_alive": 1, "bending": 0.0}}

    _EPS = 1e-12

    @staticmethod
    def _corners(eptm):
        """Geometry of every VIRTUAL-vertex corner, in edge ROW-POSITION space.

        Returns ``(pi, pj, u, w, kappa)`` where ``pi`` / ``pj`` are row positions
        of the incoming (A->V) and outgoing (V->B) edge of each corner, and
        ``u`` / ``w`` are their edge vectors. Uses the stored dx/dy, so the
        periodic min-image geometry is already accounted for.
        """
        e = eptm.edge_df
        if e.shape[0] == 0:
            return (np.array([], int), np.array([], int),
                    np.zeros((0, 2)), np.zeros((0, 2)), np.array([]))
        cols = ["face", "dx", "dy", "trgt"] + (["order"] if "order" in e.columns else [])
        ed = e[cols]
        ed = ed.sort_values(["face", "order"]) if "order" in e.columns else ed.sort_values(["face"])
        pos = np.asarray(e.index.get_indexer(ed.index))
        face = ed["face"].to_numpy()
        vecs = ed[["dx", "dy"]].to_numpy(float)
        trgt = ed["trgt"].to_numpy()
        n = len(face)
        change = np.empty(n, bool); change[0] = True; change[1:] = face[1:] != face[:-1]
        run_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
        is_last = np.empty(n, bool); is_last[:-1] = face[1:] != face[:-1]; is_last[-1] = True
        nxt = np.arange(n) + 1; nxt[is_last] = run_start[is_last]
        # corner lives at V = trgt(edge i) = srce(edge nxt[i]); keep virtual only
        isv = eptm.vert_df["is_virtual"].astype(bool)
        virt = np.nan_to_num(isv.reindex(trgt).to_numpy(), nan=0).astype(bool)
        kappa_face = eptm.face_df["bending"] * eptm.face_df.get(
            "is_alive", pd.Series(1.0, index=eptm.face_df.index))
        kappa = kappa_face.reindex(face).to_numpy(float)
        keep = virt & np.isfinite(kappa) & (np.abs(kappa) > 0)
        return (pos[keep], pos[nxt[keep]], vecs[keep], vecs[nxt[keep]], kappa[keep])

    @classmethod
    def energy(cls, eptm):
        pi, pj, u, w, kappa = cls._corners(eptm)
        out = pd.Series(0.0, index=eptm.face_df.index)
        if len(pi) == 0:
            return out
        nu = np.linalg.norm(u, axis=1); nw = np.linalg.norm(w, axis=1)
        good = (nu > cls._EPS) & (nw > cls._EPS)
        c = np.zeros(len(pi))
        c[good] = (u[good] * w[good]).sum(1) / (nu[good] * nw[good])
        c = np.clip(c, -1.0, 1.0)
        e_corner = kappa * (1.0 - c)
        faces = eptm.edge_df["face"].to_numpy()[pi]
        agg = pd.Series(e_corner).groupby(faces).sum()
        out.loc[agg.index] = agg.to_numpy()
        return out

    @classmethod
    def gradient(cls, eptm):
        ncoords = len(eptm.coords)
        n_edges = eptm.edge_df.shape[0]
        gs = np.zeros((n_edges, ncoords))
        gt = np.zeros((n_edges, ncoords))
        pi, pj, u, w, kappa = cls._corners(eptm)
        if len(pi):
            nu = np.linalg.norm(u, axis=1); nw = np.linalg.norm(w, axis=1)
            good = (nu > cls._EPS) & (nw > cls._EPS)
            pi, pj, u, w, kappa = pi[good], pj[good], u[good], w[good], kappa[good]
            nu, nw = nu[good], nw[good]
            uh = u / nu[:, None]; wh = w / nw[:, None]
            c = np.clip((uh * wh).sum(1), -1.0, 1.0)
            k = kappa[:, None]
            # E = kappa*(1-c);  dE/du = -kappa*(wh - c*uh)/|u|,  dE/dw = -kappa*(uh - c*wh)/|w|
            dEdu = -k * (wh - c[:, None] * uh) / nu[:, None]
            dEdw = -k * (uh - c[:, None] * wh) / nw[:, None]
            # u = V - A,  w = B - V
            gA = -dEdu
            gV = dEdu - dEdw
            gB = dEdw
            np.add.at(gs, pi, gA)      # A is srce of the incoming edge
            np.add.at(gt, pi, gV)      # V is trgt of the incoming edge
            np.add.at(gt, pj, gB)      # B is trgt of the outgoing edge
        cols = ["g" + c_ for c_ in eptm.coords]
        grad_srce = pd.DataFrame(gs, index=eptm.edge_df.index, columns=cols)
        grad_trgt = pd.DataFrame(gt, index=eptm.edge_df.index, columns=cols)
        return grad_srce, grad_trgt


from lateral_inhibition_model import LateralInhibitionModel


def _truncate_history_file(path, continue_from_time):
    """Drop every row whose ``time > continue_from_time`` from every
    table in the HDF5 archive at ``path``.

    Called by :meth:`InnerEarModel.simulate` when resuming an
    interrupted run. The snapshot AT ``continue_from_time`` stays
    intact — only the bits AFTER it are removed so the resumed
    simulation can write fresh snapshots over the discarded tail
    without leaving a gap or a contradictory history.

    Each element (``/vert``, ``/edge``, ``/face``) has ``time``
    registered as a ``data_column`` by ``HistoryHdf5.record``, which
    means ``HDFStore.remove(key, where=...)`` works directly. If for
    any reason ``time`` isn't an indexable column (e.g. a hand-built
    archive), we fall back to a read-filter-rewrite pass.
    """
    if not os.path.isfile(path):
        return
    where_clause = "time > %.17g" % float(continue_from_time)
    with pd.HDFStore(path, "a") as store:
        for key in list(store.keys()):
            element = key.strip("/")
            if element == "settings":
                # ``settings`` is a single Series, not time-indexed.
                continue
            try:
                store.remove(key=element, where=where_clause)
            except (KeyError, ValueError, TypeError, NotImplementedError):
                # Fallback for tables without an indexed ``time``
                # column: read everything, filter, rewrite.
                try:
                    df = store.select(element)
                except KeyError:
                    continue
                if "time" not in df.columns:
                    continue
                kept = df[df["time"] <= float(continue_from_time)].copy()
                store.remove(element)
                if not kept.empty:
                    store.put(
                        element, kept, format="table",
                        data_columns=["time"],
                    )


def _rewrite_history_for_resume(path, continue_from_time, history):
    """Rewrite the kept portion (``time <= continue_from_time``) of the
    HDF5 archive at ``path`` so its table structure exactly matches
    what ``history.record()`` will append.

    Why a plain truncate isn't enough
    ---------------------------------
    The original archive was written by a run whose live sheet still
    carried 3D coordinate columns (``z``, ``sz``, ``tz``, ``dz``,
    ``fz``), the periodic-metadata columns, and a particular column
    ORDER. The resumed run's live sheet is loaded via
    ``arrange_sheet_from_history(two_dim=True)``, which DROPS the 3D
    columns, can add spec-leaked columns (``unique_id_max``), and
    re-orders everything. ``pandas``/``PyTables`` ``store.append``
    requires the appended frame's block structure (column set +
    order + dtypes) to match the existing table EXACTLY — so the
    first ``record()`` after a bare truncate dies with
    ``cannot match existing table structure for [...]``.

    This function transcribes each kept table into precisely the
    column set / order / dtypes that ``record()`` produces (i.e.
    ``history.columns[element] + ['time']`` with dtypes taken from
    ``history.sheet``), then writes them as the new archive's first
    rows — so every subsequent ``record()`` append lines up.

    Must be called AFTER the ``HistoryHdf5`` object is built (so
    ``history.columns`` / ``history.sheet`` reflect the structure
    the resumed run will record) but BEFORE the solver runs.
    """
    if not os.path.isfile(path):
        return
    t0 = float(continue_from_time)

    rewritten = {}
    settings = None
    with pd.HDFStore(path, "r") as src:
        keys = [k.strip("/") for k in src.keys()]
        for element in keys:
            if element == "settings":
                settings = src["settings"]
                continue
            if element not in history.columns:
                # The resumed history doesn't track this element;
                # drop it rather than carry a table record() will
                # never touch.
                continue
            kept = src.select(element, where=f"time <= {t0}")
            ref = history.sheet.datasets[element]
            target_cols = list(history.columns[element])

            # Build the non-time part column-by-column, mirroring the
            # dtypes record() will write. ``pd.DataFrame(data,
            # columns=...)`` then consolidates same-dtype columns into
            # the same blocks record() produces.
            data = {}
            n = len(kept)
            for col in target_cols:
                if col in kept.columns:
                    arr = kept[col].to_numpy()
                elif col in ref.columns and len(ref):
                    # Column the resumed structure expects but the old
                    # archive lacked (e.g. a spec-leaked unique_id_max
                    # or a newer-model column). Broadcast a
                    # representative value — the historical value of
                    # such metadata columns isn't meaningful.
                    arr = np.full(n, ref[col].iloc[0])
                else:
                    arr = np.zeros(n)
                if col in ref.columns:
                    arr = arr.astype(ref[col].dtype)
                data[col] = arr
            df_part = pd.DataFrame(data, index=kept.index, columns=target_cols)

            # Append the time column LAST, exactly as record() does
            # (``pd.concat([df, times], axis=1, sort=False)``).
            times = pd.Series(
                kept["time"].to_numpy().astype(float),
                name="time", index=kept.index,
            )
            rewritten[element] = pd.concat([df_part, times], axis=1, sort=False)

    os.remove(path)
    with pd.HDFStore(path, "w") as dst:
        if settings is not None:
            dst["settings"] = settings
        for element, out in rewritten.items():
            kwargs = {"data_columns": ["time"]}
            if "segment" in out.columns:
                # Mirror HistoryHdf5.record's variable-length string
                # handling so the table re-opens cleanly.
                kwargs["min_itemsize"] = {"segment": 8}
            dst.append(key=element, value=out, **kwargs)

    # The HistoryHdf5 cached its time stamps at construction (from the
    # pre-rewrite file). Invalidate the cache so the next read re-scans
    # the rewritten archive.
    history._time_stamps = np.empty((0,))


class InnerEarModel:
    """
    A wrapping class for epithilium model, for easy execution of inner ear model simulations.
    """
    def __init__(self, sheet, tension=None, repulsion=None, repulsion_distance=None, repulsion_exp=7,
                 preferred_area=None, preferred_area_override=None,
                 contractility=None, elasticity=None, shape_index=0.0,
                 hc_shape_index=None, sc_shape_index=None, bending=0.0,
                 differentiation_threshold=0.5, random_sensitivity=False,
                 l=3, m=3, betaN=1, betaD=1, inhibition=False,
                 notch_repressor_degradation_ratio=1, repressor_sensitivity=1, atoh_sensitivity=1,
                 delta_repressor_degradation_ratio=1, notch_delta_production_ratio=1,
                 stress_effectors=None, mechanosensitivity=0, stress_shift=0.0,
                 stress_hill_exponent=None,
                 notch_sensitivity=1, atoh_by_repressor=False,
                 randomize_notch_delta_levels=False,
                 notch_levels=None, delta_levels=None, repressor_levels=None,
                 initial_notch_delta_level=None):
        # Setting class constants
        self.CELL_TYPES = ['SC', 'HC']
        self.DIMENSIONS = ['2D']
        self.topological_events_handler = TopologicalEventsHandler(self)
        # Setting default behavior
        if tension is None:
            tension = {('HC', 'HC'): 0.05,
                       ('HC', 'SC'): 0.05,
                       ('SC', 'SC'): 0.05
                       }
        if preferred_area is None:
            preferred_area = {'HC': 1.,
                              'SC': 1.}
        if contractility is None:
            contractility = {'HC': 0.4,
                             'SC': 0.1}
        if repulsion is None:
            repulsion = {'HC': 0.001, #0.001
                         'SC': 0}
        if repulsion_distance is None:
            repulsion_distance = {'HC': 2.0,
                                  'SC': 0}
        if elasticity is None:
            elasticity = {'HC': 5.,
                          'SC': 1.}
        if ('SC', 'HC') in tension:
            tension[('HC', 'SC')] = tension[('SC', 'HC')]
        elif ('HC', 'SC') in tension:
            tension[('SC', 'HC')] = tension[('HC', 'SC')]
        sheet.repulsion_exp = repulsion_exp
        self.sheet = sheet
        # L0 (mean cell perimeter) is still computed and passed to the
        # lateral-inhibition model below, where the LI equations KEEP their L0
        # normalization (face stress / L0, and the L0/perimeter contact terms).
        # But MECHANICAL parameters are now used AS GIVEN — NOT length-normalized.
        # The old ``preferred_area *= L0**2`` / ``repulsion_distance *= L0``
        # scalings inflated the preferred area by ~L0^2 (~12x here), leaving the
        # whole tissue compressed (it would expand on a cut, opposite to the
        # experiment's ~10% shrinkage).
        length_normalization_factor = self.get_average_face_perimeter()
        if preferred_area_override is not None:
            # Optional: set the effective preferred area DIRECTLY (e.g. a fraction
            # of the measured mean cell area) instead of the caller's per-type dict.
            preferred_area = {'HC': float(preferred_area_override),
                              'SC': float(preferred_area_override)}
        # Target perimeter P0 = shape_index * sqrt(A0), PER CELL TYPE. A0 is
        # already length-normalized above, so P0 scales with it. Consumed by
        # ContractilityPerimeterElasticity; shape_index=0 -> P0=0 -> the perimeter
        # effector reduces exactly to plain contractility (Gamma/2 P^2).
        #
        # ``hc_shape_index`` / ``sc_shape_index`` each default to the shared
        # ``shape_index``, so a caller that passes only ``shape_index`` keeps the
        # historical uniform behaviour exactly. Giving them different values is
        # the type-dependent shape index: it sets the HC and SC target perimeters
        # independently, which is what lets HC and SC roundness be matched at the
        # same time (one shared P0 drives them together).
        p0_hc = float(shape_index if hc_shape_index is None else hc_shape_index)
        p0_sc = float(shape_index if sc_shape_index is None else sc_shape_index)
        self.face_params = {"contractility": contractility, "repulsion": repulsion,
                            "repulsion_distance": repulsion_distance,
                            "prefered_area": preferred_area,
                            "prefered_vol": preferred_area,
                            "area_elasticity": elasticity,
                            "prefered_perimeter": {
                                'HC': p0_hc * float(np.sqrt(preferred_area['HC'])),
                                'SC': p0_sc * float(np.sqrt(preferred_area['SC']))},
                            # UNIFORM boundary bending stiffness (same for both
                            # cell types); 0 makes BoundaryBending a no-op.
                            "bending": {'HC': float(bending), 'SC': float(bending)}}
        self.differentiation_threshold = differentiation_threshold
        self.edge_params = {"line_tension": tension}
        self.dimensionality = '2D'
        specs = self.get_specs_2d(notch_sensitivity)
        self.sheet.update_specs(specs, reset=True)
        self.lateral_inhibition_model = LateralInhibitionModel(
            self, l=l, m=m, betaN=betaN, betaD=betaD, inhibition=inhibition,
            notch_repressor_degradation_ratio=notch_repressor_degradation_ratio,
            length_normalization_factor=length_normalization_factor,
            repressor_sensitivity=repressor_sensitivity, atoh_sensitivity=atoh_sensitivity,
            delta_repressor_degradation_ratio=delta_repressor_degradation_ratio,
            notch_delta_production_ratio=notch_delta_production_ratio,
            stress_effectors=stress_effectors, mechanosensitivity=mechanosensitivity,
            stress_shift=stress_shift, stress_hill_exponent=stress_hill_exponent)
        # If ``sheet`` came in with LI columns already populated
        # (typical of a sheet retrieved from HDF5 history), those
        # values are preserved by ``initialize_notch_delta``;
        # otherwise it seeds them from a uniform random distribution.
        # Pass ``randomize_notch_delta_levels=True`` to force a
        # fresh random seed even when the loaded sheet already
        # carries values — useful for parameter sweeps that re-use
        # an existing geometry but want a fresh LI starting state.
        # The old ``saved_notch_delta_levels_file`` side-channel
        # (a pickle workaround for the LI-not-surviving-history bug)
        # is gone — see :meth:`get_specs_2d` for the actual fix.
        # ``notch_levels`` / ``delta_levels`` / ``repressor_levels`` let a
        # caller seed the LI initial state from per-cell arrays (or .npy
        # paths) indexed by ``unique_id``; when given they take precedence
        # over both the preserve-loaded and random-seed paths.
        self.initialize_notch_delta(
            random_sensitivity, contact_dependent=True,
            force_random=randomize_notch_delta_levels,
            notch_levels=notch_levels, delta_levels=delta_levels,
            repressor_levels=repressor_levels,
            initial_level_range=initial_notch_delta_level,
        )
        active_edges = (self.sheet.edge_df.opposite.values >= 0).astype(int)
        self.sheet.edge_df.loc[:, 'is_active'] = active_edges
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.srce.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.trgt.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.face_df.loc[:,"id"] = self.sheet.face_df.index
        self.sheet.active_verts = np.where(self.sheet.vert_df.is_active.values)[0]
        if atoh_by_repressor:
            if "repressor_level" in self.sheet.face_df.columns:
                self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                    self.sheet.face_df.repressor_level.values)
            else:
                self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                    self.sheet.face_df.notch_level.values)
        else:
            self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                self.sheet.face_df.delta_level.values, activation=True)
        self.update_cell_type_parameters(self.sheet.face_df.atoh_level)
        self.sheet.order_all_edges()

        return

    def update_cell_type_parameters(self, atoh_level):
        differentiating_cells = self.sheet.face_df.query('type >= 0').index
        for param in self.face_params.keys():
            new_values = atoh_level * self.face_params[param]["HC"] + (1 - atoh_level) * self.face_params[param]["SC"]
            self.sheet.face_df.loc[differentiating_cells,param] = new_values[differentiating_cells]

        new_types = (atoh_level > self.differentiation_threshold).astype(np.int32)
        self.sheet.face_df.loc[differentiating_cells, 'type'] = new_types[differentiating_cells]
        first_faces = self.sheet.edge_df.face.values
        opposite_to_first = self.sheet.edge_df.opposite.values
        second_faces = - np.ones(opposite_to_first.shape)
        second_faces[opposite_to_first >= 0] = self.sheet.edge_df.loc[opposite_to_first[opposite_to_first >= 0], "face"].values
        first_types = self.sheet.face_df.loc[first_faces, "type"].values
        second_types = - np.ones(second_faces.shape)
        second_types[second_faces >= 0] = self.sheet.face_df.loc[second_faces[second_faces >= 0], "type"].values
        for param in self.edge_params.keys():
            new_vals = self.sheet.edge_df.loc[:, param].values
            new_vals[np.logical_and(first_types == 1, second_types == 1)] = self.edge_params[param][("HC", "HC")]
            new_vals[np.logical_and(first_types == 1, second_types == 0)] = self.edge_params[param][("HC", "SC")]
            new_vals[np.logical_and(first_types == 0, second_types == 1)] = self.edge_params[param][("SC", "HC")]
            new_vals[np.logical_and(first_types == 0, second_types == 0)] = self.edge_params[param][("SC", "SC")]
            self.sheet.edge_df.loc[:, param] = new_vals

    def set_random_parameters(self):
        self.sheet.face_df.loc[:, "prefered_area"] = np.random.rand(self.sheet.face_df.shape[0],)
        self.sheet.face_df.loc[:, "prefered_vol"] = self.sheet.face_df.loc[:, "prefered_area"]
        self.sheet.face_df.loc[:, "contractility"] = np.random.rand(self.sheet.face_df.shape[0],)/5

    def get_specs_2d(self, notch_sensitivity):
        # NOTE: ``notch_level`` / ``delta_level`` / ``repressor_level``
        # are deliberately NOT in the spec dict. ``update_specs`` is
        # called with ``reset=True`` in ``__init__``, which would
        # otherwise overwrite any LI values the sheet brought in
        # from a loaded HDF5 history. The LI columns are managed
        # explicitly by :meth:`initialize_notch_delta` — which now
        # preserves them when they're already populated (continued
        # run) and randomises them only on a genuinely fresh sheet.
        specs = {'vert': {'is_active':1,
                          'radial_tension': 0},
                 'edge': {'is_active': 1,
                          'sub_area': 6,
                          },
                 'face': {'is_alive': 1,
                          'type': 0,
                          'radial_tension': 0,
                          'notch_sensitivity': notch_sensitivity
                          }
                 }
        for param in self.edge_params.keys():
            specs['edge'][param] = self.edge_params[param][("SC", "SC")]
        for param in self.face_params.keys():
            specs['face'][param] = self.face_params[param]["SC"]
        return specs

    def mean_notch(self, indices):
        return self.sheet.face_df.loc[indices[indices >= 0], 'notch_level'].mean()

    def mean_delta(self, indices):
        return self.sheet.face_df.loc[indices[indices >= 0], 'delta_level'].mean()

    def get_neighbors(self, face):
        return self.sheet.get_neighbors(face)


    def get_neighbors_data(self, func_list):
        def apply_on_real_neighbors(func):
            def f(neighbors):
                indices = neighbors.to_numpy()
                return func(self.sheet.edge_df.loc[indices[indices >= 0], "face"].unique())
            return f
        if hasattr(func_list, "__len__"):
            return self.sheet.edge_df.groupby("face")["opposite"].agg([apply_on_real_neighbors(func) for func in func_list])
        else:
            return self.sheet.edge_df.groupby("face")["opposite"].agg(apply_on_real_neighbors(func_list))

    def get_neighbor_types(self, neighbors_id):
        return self.sheet.face_df.type[neighbors_id].values

    def get_num_of_HC_neighbors(self, neighbors_id):
        types = self.get_neighbor_types(neighbors_id)
        counts = np.bincount(types[types>=0])
        if counts.size > 1:
            return counts[1]
        else:
            return 0

    def get_edge_stress(self, relevant_effectors):
        edge_data = self.sheet.edge_df[["ux", "uy", "opposite"]].copy()
        stress_model = model_factory(relevant_effectors)
        grads = stress_model.compute_gradient(self.sheet, components=True)
        norm_factor = self.sheet.specs["settings"].get("nrj_norm_factor", 1)
        srce_grads = [g[0] for g in grads if g[0].shape[0] == self.sheet.Ne]
        if srce_grads:
            edge_data["srce_gx"] = np.array([grad.gx.values for grad in srce_grads]).sum(axis=0)
            edge_data["srce_gy"] = np.array([grad.gy.values for grad in srce_grads]).sum(axis=0)
        trgt_grads = [
            g[1] for g in grads if (g[1] is not None) and (g[1].shape[0] == self.sheet.Ne)
        ]
        if trgt_grads:
            edge_data["trgt_gx"] = np.array([grad.gx.values for grad in trgt_grads]).sum(axis=0)
            edge_data["trgt_gy"] = np.array([grad.gy.values for grad in trgt_grads]).sum(axis=0)
        vert_grads = [g[0] for g in grads if g[0].shape[0] == self.sheet.Nv]
        if vert_grads:
            raise NotImplementedError
        edge_data["stress"] = edge_data.eval("((trgt_gx - srce_gx) * ux + (trgt_gy - srce_gy) * uy)/ %s" % str(norm_factor))
        # Couple each edge to its OPPOSITE (the shared junction carries both):
        # stress[e] = base[e] + base[opposite(e)]. Snapshot the base stress first
        # so BOTH terms use pre-coupling values, and map opposite LABELS to array
        # positions (get_indexer) so it is correct even for a non-contiguous index.
        base_stress = edge_data["stress"].to_numpy().copy()
        opp = edge_data["opposite"].to_numpy()
        has = opp > 0
        new_stress = base_stress.copy()
        new_stress[has] = base_stress[has] + base_stress[edge_data.index.get_indexer(opp[has])]
        edge_data["stress"] = new_stress
        return edge_data.stress

    def get_face_stress(self, relevant_effectors):
        edge_stress = self.get_edge_stress(relevant_effectors)
        stress_df = self.sheet.edge_df[["face", "length"]].copy()
        stress_df["stress"] = edge_stress.values
        stress_df["weighted_stress"] = stress_df.eval("stress * length")
        face_stress = stress_df[["face", "weighted_stress"]].groupby("face").sum()
        return face_stress.weighted_stress.values

    def get_average_edge_stress_by_type(self, relevant_effectors=None):
        if relevant_effectors is None:
            relevant_effectors = self.lateral_inhibition_model.stress_effectors
        edge_stress = self.get_edge_stress(relevant_effectors).values
        first_faces = self.sheet.edge_df.face.values
        opposite_to_first = self.sheet.edge_df.opposite.values
        second_faces = - np.ones(opposite_to_first.shape)
        second_faces[opposite_to_first >= 0] = self.sheet.edge_df.loc[
            opposite_to_first[opposite_to_first >= 0], "face"].values
        first_types = self.sheet.face_df.loc[first_faces, "type"].values
        second_types = - np.ones(second_faces.shape)
        second_types[second_faces >= 0] = self.sheet.face_df.loc[second_faces[second_faces >= 0], "type"].values
        HC_HC_stress = edge_stress[np.logical_and(first_types == 1, second_types == 1)]
        SC_SC_stress = edge_stress[np.logical_and(first_types == 0, second_types == 0)]
        HC_SC_stress = edge_stress[np.logical_and(first_types == 1, second_types == 0)]
        SC_HC_stress = edge_stress[np.logical_and(first_types == 0, second_types == 1)]
        res = {"HC:HC": np.average(HC_HC_stress), "SC:SC": np.average(SC_SC_stress),
               "HC:SC": np.average(HC_SC_stress), "SC:HC": np.average(SC_HC_stress),
               "all": np.std(edge_stress),
               "HC:HC std": np.std(HC_HC_stress), "SC:SC std": np.std(SC_SC_stress),
               "HC:SC std": np.std(HC_SC_stress), "SC:HC std": np.std(SC_HC_stress),
               "all std": np.std(edge_stress),
               }
        return res

    def get_average_face_stress_by_number_of_HC_neighbors(self, relevant_effectors=None):
        if relevant_effectors is None:
            relevant_effectors = self.lateral_inhibition_model.stress_effectors
        face_stress = self.get_face_stress(relevant_effectors)
        face_types = self.sheet.face_df.type.values
        number_of_HC_neighbors = self.get_neighbors_data(self.get_num_of_HC_neighbors)
        res = dict()
        for n in range(np.max(number_of_HC_neighbors)+1):
            relevant_stresses = face_stress[np.logical_and(number_of_HC_neighbors == n, face_types == 0)]
            res["SC with %d HC neighbors N" % n] = relevant_stresses.size
            if relevant_stresses.size > 0:
                res["SC with %d HC neighbors avg" % n] = np.average(relevant_stresses)
            if relevant_stresses.size > 1:
                res["SC with %d HC neighbors std" % n] = np.std(relevant_stresses)
            relevant_stresses = face_stress[np.logical_and(number_of_HC_neighbors == n, face_types == 1)]
            res["HC with %d HC neighbors N" % n] = relevant_stresses.size
            if relevant_stresses.size > 0:
                res["HC with %d HC neighbors avg" % n] = np.average(relevant_stresses)
            if relevant_stresses.size > 1:
                res["HC with %d HC neighbors std" % n] = np.std(relevant_stresses)
        return res

    def get_average_edge_length(self, std=False):
        if std:
            return np.average(self.sheet.edge_df.length.values), np.std(self.sheet.edge_df.length.values)
        else:
            return np.average(self.sheet.edge_df.length.values)

    def get_average_face_perimeter(self, std=False):
        if std:
            return np.average(self.sheet.face_df.perimeter.values), np.std(self.sheet.face_df.perimeter.values)
        else:
            return np.average(self.sheet.face_df.perimeter.values)

    def get_contact_matrix(self):
        return self.sheet.get_contact_matrix()

    @staticmethod
    def get_model(only_differentiation=False, effectors=None):
        if only_differentiation:
            model = model_factory([EmptyAffector])
        else:
            if effectors is None:
                model = model_factory([LineTension, FaceContractility, FaceAreaElasticity])
            else:
                model = model_factory(effectors)
        return model

    def get_random_initializer(self, wait_time=5, dt=1.):
        self.time_to_random = wait_time
        def random_initializer(sheet, manager):
            if self.time_to_random <= 0:
                self.set_random_parameters()
                self.time_to_random = wait_time
            else:
                self.time_to_random -= dt
            manager.append(random_initializer)
        return random_initializer

    def save_notch_delta(self, file_path):
        relevant_columns = ['notch_level', 'delta_level']
        if 'repressor_level' in self.sheet.face_df.columns:
            relevant_columns.append('repressor_level')
        levels = self.sheet.face_df.loc[:, relevant_columns]

        levels.to_pickle(file_path)

    def load_li_levels_from_numpy(self, notch_levels=None, delta_levels=None,
                                  repressor_levels=None):
        """Set the lateral-inhibition initial values on ``face_df`` from
        per-cell data indexed by ``unique_id``.

        Each of ``notch_levels`` / ``delta_levels`` / ``repressor_levels``
        may be:

        - a path to a ``.npy`` file,
        - a 1-D numpy array (or anything ``np.asarray`` accepts), or
        - ``None`` to leave that channel untouched.

        Entry ``i`` of an array is the value for the cell whose
        ``unique_id == i`` — the same convention used by
        ``save_face_data_to_df`` (``cells_info`` is indexed by
        ``unique_id``), ``labels.npy`` and ``contact_matrix.npy``. On a
        fresh sheet ``unique_id`` equals the ``face_df`` index, but the
        mapping is done explicitly so it stays correct after
        divisions / delaminations leave gaps in ``unique_id``.

        This replaces the old pickled-DataFrame side-channel — the data
        is now plain numpy arrays keyed by the stable ``unique_id``.
        """
        uids = self.sheet.face_df["unique_id"].to_numpy()
        uids = uids.astype(int)
        max_uid = int(uids.max()) if uids.size else -1

        def _resolve(values, label):
            if values is None:
                return None
            if isinstance(values, str):
                values = np.load(values)
            values = np.asarray(values).ravel()
            if max_uid >= values.shape[0]:
                raise ValueError(
                    f"{label}: array has {values.shape[0]} entries but the "
                    f"sheet has a cell with unique_id {max_uid}. Entry i must "
                    f"hold the value for the cell with unique_id i, so the "
                    f"array needs at least {max_uid + 1} entries."
                )
            # Gather one value per face, indexed by that face's unique_id.
            return values[uids]

        notch = _resolve(notch_levels, "notch_levels")
        delta = _resolve(delta_levels, "delta_levels")
        repressor = _resolve(repressor_levels, "repressor_levels")
        if notch is not None:
            self.sheet.face_df["notch_level"] = notch
        if delta is not None:
            self.sheet.face_df["delta_level"] = delta
        if repressor is not None:
            self.sheet.face_df["repressor_level"] = repressor

    def initialize_notch_delta(self, random_sensitivity=False,
                               contact_dependent=False, force_random=False,
                               notch_levels=None, delta_levels=None,
                               repressor_levels=None, initial_level_range=None):
        """Seed the lateral-inhibition columns on ``face_df``.

        Behaviour
        ---------
        - If any of ``notch_levels`` / ``delta_levels`` /
          ``repressor_levels`` is given (per-cell arrays or ``.npy``
          paths indexed by ``unique_id`` — see
          :meth:`load_li_levels_from_numpy`), those values are loaded
          and take PRECEDENCE over everything below. Any channel not
          supplied falls back to the rules below (preserve if already
          present, else random).
        - If ``notch_level`` and ``delta_level`` are already present
          on ``face_df`` (the sheet was loaded from a saved HDF5
          history that carries those columns) AND ``force_random``
          is False (the default), they are PRESERVED as-is. Only
          ``repressor_level`` is filled with a fresh random column
          when ``contact_dependent`` is requested but the loaded
          sheet pre-dates the repressor channel (an old archive
          without that column).
        - Otherwise — i.e. on a genuinely fresh sheet built from
          ``planar_virtual_sheet_2d`` OR when the caller explicitly
          asks for a fresh seed via ``force_random=True`` — every
          relevant column is initialised from a uniform random
          distribution scaled by the relevant maximum.

        The ``force_random`` opt-in supports parameter sweeps that
        share a geometry (e.g. resuming a saved sheet) but want
        different LI initial conditions per sweep point.

        This used to do a third thing: load the levels from a
        separate ``_notch_delta_levels.pkl`` side-file when one
        existed. That code-path was a workaround for a now-fixed
        bug where the LI columns weren't surviving the
        ``HistoryHdf5`` round-trip — see ``get_specs_2d`` for the
        actual fix (the LI columns are no longer in the spec dict
        that ``update_specs(reset=True)`` resets). The pickle
        branch has been removed; the history file is the single
        source of truth.
        """
        n_faces = self.sheet.face_df.shape[0]

        # Explicit per-cell initial values (arrays or .npy paths, indexed
        # by unique_id) take precedence over both the preserve-loaded and
        # the random-seed paths below. Any channel the caller did NOT
        # supply falls back to those rules (preserve if already present on
        # the sheet, else random).
        if (notch_levels is not None or delta_levels is not None
                or repressor_levels is not None):
            self.load_li_levels_from_numpy(
                notch_levels, delta_levels, repressor_levels
            )
            if "notch_level" not in self.sheet.face_df.columns:
                maximal_notch_level = self.lateral_inhibition_model.get_maximal_notch_level()
                self.sheet.face_df["notch_level"] = (
                    np.random.rand(n_faces) * maximal_notch_level
                )
            if "delta_level" not in self.sheet.face_df.columns:
                maximal_delta_level = self.lateral_inhibition_model.get_maximal_delta_level()
                self.sheet.face_df["delta_level"] = (
                    np.random.rand(n_faces) * maximal_delta_level
                )
            if contact_dependent and "repressor_level" not in self.sheet.face_df.columns:
                maximal_repressor_level = self.lateral_inhibition_model.get_maximal_repressor_level()
                self.sheet.face_df["repressor_level"] = (
                    np.random.rand(n_faces) * maximal_repressor_level
                )
            if random_sensitivity:
                self.sheet.face_df["notch_sensitivity"] = np.random.rand(n_faces)
            return

        already_loaded = (
            not force_random
            and "notch_level" in self.sheet.face_df.columns
            and "delta_level" in self.sheet.face_df.columns
        )
        if already_loaded:
            # Preserve the loaded notch/delta values. Top up
            # repressor only if the caller asked for the
            # contact-dependent variant and the archive didn't
            # carry that column (older archives wouldn't).
            if contact_dependent and "repressor_level" not in self.sheet.face_df.columns:
                maximal_repressor_level = self.lateral_inhibition_model.get_maximal_repressor_level()
                self.sheet.face_df["repressor_level"] = (
                    np.random.rand(n_faces) * maximal_repressor_level
                )
        elif initial_level_range is not None:
            # Near-zero UNDIFFERENTIATED start: draw every channel from
            # U(0, initial_level_range) (e.g. 0.01), matching the collaborators'
            # model, instead of scaling by each channel's maximum. The tiny
            # random spread is the symmetry-breaking the lateral inhibition then
            # amplifies into the salt-and-pepper pattern.
            rng = float(initial_level_range)
            self.sheet.face_df["notch_level"] = np.random.rand(n_faces) * rng
            self.sheet.face_df["delta_level"] = np.random.rand(n_faces) * rng
            if contact_dependent:
                self.sheet.face_df["repressor_level"] = np.random.rand(n_faces) * rng
        else:
            maximal_delta_level = self.lateral_inhibition_model.get_maximal_delta_level()
            maximal_notch_level = self.lateral_inhibition_model.get_maximal_notch_level()
            self.sheet.face_df["notch_level"] = (
                np.random.rand(n_faces) * maximal_notch_level
            )
            self.sheet.face_df["delta_level"] = (
                np.random.rand(n_faces) * maximal_delta_level
            )
            if contact_dependent:
                maximal_repressor_level = self.lateral_inhibition_model.get_maximal_repressor_level()
                self.sheet.face_df["repressor_level"] = (
                    np.random.rand(n_faces) * maximal_repressor_level
                )
        if random_sensitivity:
            self.sheet.face_df["notch_sensitivity"] = np.random.rand(n_faces)

    def simulate(self, t_end, dt, only_differentiation=False, random_forces=False,
                 aging_sensitivity=False, no_differentiation=False, contact_dependent_differentiation=False,
                 divisions=True, intercalations=True, delaminations=True, ablated_cells=[], sensitivity_aging_rate=0,
                 division_area=1.3, intercalation_length=0.04, delamination_area=0.1, delamination_rate=1.2,
                 sharp_angle_threshold=None, verbose_log=False,
                 viscosity=3, effectors=None, quasi_static=False, quasi_static_threshold=0.01, atoh_by_repressor=True,
                 history_file=None, save_interval=None,
                 max_displacement=None, max_disp_factor=0.25,
                 dt_min_factor=0.0001, dt_increase_factor=1.1,
                 until_steady_state=False, lateral_inhibition_threshold=1e-3, save_every=0.1,
                 continue_from_time=None, steady_state_min_steps=4,
                 max_wall_seconds=None, min_progress_rate=None, progress_window_seconds=30.0):
        """Run the simulation until ``t_end`` (or, if
        ``until_steady_state`` is True, until the dynamics settle).

        Steady-state stopping (``until_steady_state=True``)
        ---------------------------------------------------
        Stops as soon as the per-step change in the relevant
        quantities falls below threshold. ``t_end`` then acts as a
        wall-clock safety cap.

        - Mechanical criterion: ``max(|new_pos - old_pos|) <
          quasi_static_threshold`` for every active vertex.
        - Lateral-inhibition criterion:
          ``max(|new - old|) < lateral_inhibition_threshold`` across
          whichever of ``notch_level``, ``delta_level``,
          ``repressor_level`` are present on ``face_df``.

        Which criterion is required depends on the existing
        differentiation flags:

        - ``only_differentiation=True``: ONLY the lateral-inhibition
          criterion (positions don't move when there are no
          mechanics) — the mechanical check is skipped.
        - ``no_differentiation=True``: ONLY the mechanical criterion
          (LI levels don't change when there's no differentiation
          manager) — the LI check is skipped.
        - Otherwise: BOTH criteria must hold simultaneously.

        A topology change during a step always fails the LI check
        for that step (so the system has to settle for at least one
        full step AFTER any division / delamination / T1 before it
        can declare steady state).

        The criteria must hold for ``steady_state_min_steps``
        CONSECUTIVE accepted steps before the run halts (default 4 —
        "no significant change for more than 3 steps"). A single
        drifting step, a rejected step, or any topology change resets
        the counter, so a brief lull can't trigger a premature stop.

        Resuming an interrupted run (``continue_from_time``)
        ----------------------------------------------------
        Pass ``continue_from_time=<t0>`` to pick up a partially-
        completed run. The caller is expected to have:

          1. Loaded ``self.sheet`` from the existing history file at
             time ``t0`` (typically via
             :func:`run_model.load_sheet_from_file`).
          2. Pointed ``history_file`` at the SAME archive that was
             being written before.

        ``simulate`` will then:

          - Truncate the archive in-place, dropping every recorded
            snapshot whose ``time > t0``. The snapshot AT ``t0`` is
            preserved (and re-stamped when the solver records the
            initial state — same data, no duplicate row, because
            ``HistoryHdf5.record`` removes any existing row at the
            new ``time`` before appending).
          - Seed the solver's clock with ``prev_t = t0`` so
            ``solver.solve(tf=t_end, ...)`` runs from ``t0`` to
            ``t_end`` (i.e. ``t_end`` is interpreted as the
            CUMULATIVE end time, not "another t_end units from
            now"). If ``until_steady_state=True`` the simulation can
            stop earlier as usual.

        When ``continue_from_time`` is ``None`` (the default) the
        normal "fresh run" behaviour is preserved: any pre-existing
        archive at ``history_file`` is deleted before recording.
        """
        manager = EventManager("face")
        # Interior-angle threshold (radians) for the incipient-fold sharp-corner
        # collapse run every step inside update_virtual_vertices. When left at
        # the default (None) derive it from the sheet's length scales (the angle
        # at which a corner's two neighbours, on max_bond_length-long edges, sit
        # max(min_bond, intercalation) apart — see
        # VirtualSheet.default_sharp_angle_threshold); otherwise use the value
        # the caller supplied.
        if sharp_angle_threshold is None:
            sharp_angle_threshold = self.sheet.default_sharp_angle_threshold(
                self.sheet.minimal_bond_length, intercalation_length,
                self.sheet.maximal_bond_length)
        self.sheet.sharp_angle_threshold = sharp_angle_threshold
        # Topological-event log verbosity: successes at INFO only when True;
        # failures/rejections always at DEBUG (see topological_events.log_topo_event).
        self.sheet.verbose_log = verbose_log
        # manager.append(self.get_ablation_function(2))
        if no_differentiation:
            quasi_static = False
        else:
            if contact_dependent_differentiation:
                manager.append(self.lateral_inhibition_model.get_length_dependent_differentiation_function(dt=dt,
                                                                                                           quasi_static=quasi_static,
                                                                                                           atoh_by_repressor=atoh_by_repressor))
            else:
                manager.append(self.lateral_inhibition_model.get_differentiation_function(dt=dt))
        if aging_sensitivity:
            manager.append(self.lateral_inhibition_model.get_aging_sensitivity_function(rate=sensitivity_aging_rate, dt=dt))
        if not only_differentiation:
            if divisions:
                manager.append(self.topological_events_handler.get_division_function(crit_area=division_area))
            if intercalations:
                manager.append(self.topological_events_handler.get_intercalation_function(crit_edge_length=intercalation_length))
            if delaminations:
                manager.append(self.topological_events_handler.get_delamination_function(crit_area=delamination_area,
                                                                                         shrink_rate=delamination_rate))
            if len(ablated_cells) > 0:
                for cell in ablated_cells:
                    manager.append(
                        self.topological_events_handler.get_ablation_function(cell, shrink_rate=delamination_rate,
                                                                              critical_area=delamination_area))
            manager.append(self.sheet.get_update_virtual_vertices_function())

        if random_forces:
            manager.append(self.get_random_initializer(wait_time=5, dt=dt))

        # Set viscosity BEFORE creating HistoryHdf5 so its dtype is
        # captured at init.
        self.sheet.vert_df['viscosity'] = viscosity

        # IMPORTANT: run a geometry update before constructing
        # HistoryHdf5. ``InnerEarModel.__init__`` calls
        # ``update_specs(reset=True)`` which resets derived edge
        # columns (notably ``sub_area``) back to their integer defaults
        # from the spec dict. Once the solver runs ``set_pos`` for the
        # first time it calls ``geom.update_all`` which recomputes
        # ``sub_area = nz/2`` as a float, and HistoryHdf5's strict
        # dtype check (captured at HistoryHdf5 init time) then trips
        # with "There is a change of datatype in edge table in
        # {'sub_area': dtype('int64')} columns". Updating the geometry
        # here makes the captured dtypes match the steady state.
        self.sheet.geom.update_all(self.sheet)

        # Stash the periodic metadata onto face_df BEFORE any history is
        # created, so EVERY recorded snapshot carries ``_periodic_flag``
        # / ``_periodic_Lx`` / ``_periodic_Ly`` and the archive stays
        # self-describing. This matters for ALL runs, not just resumes:
        # a run whose sheet was loaded from an archive went through
        # ``arrange_sheet_from_history``, which READS and then DROPS
        # those columns. Without re-stashing here, that run's OWN output
        # archive would lose the periodic flag, and a later resume /
        # fork off it would reload the sheet as NON-periodic — the
        # periodic geometry would then never run, boundary-crossing
        # faces would unwrap to domain-spanning edges (length ~ Lx) and
        # the first force evaluation would explode (negative areas,
        # huge per-vertex displacement). ``_stash_periodic_metadata`` is
        # a no-op on non-periodic sheets.
        if hasattr(self.sheet, "_stash_periodic_metadata"):
            self.sheet._stash_periodic_metadata()

        # Pass history_file="some/path.hf5" to record on-the-fly to disk
        # — useful for diagnosing hangs (e.g. add_virtual_vertices infinite
        # loop): the partial archive can be opened from another process
        # while the simulation is still running. Without it we fall back
        # to in-memory History (faster for short runs).
        if history_file is not None:
            if continue_from_time is not None:
                # Resuming a previous run: keep the snapshots up to
                # continue_from_time and append the new ones onto the
                # SAME archive. (Periodic metadata was already stashed
                # above, so the rewritten kept snapshots and the new
                # ones all carry the flag.)
                #
                # Build the history first so it captures the column
                # set / dtypes the resumed run will record, THEN
                # rewrite the kept portion of the archive to match
                # that structure. A bare row-level truncate leaves the
                # original (3D-coords, different-order) table layout
                # in place, and the first ``record`` append then dies
                # with pandas' "cannot match existing table structure".
                history = HistoryHdf5(
                    self.sheet, save_every=save_every, dt=dt,
                    hf5file=history_file, overwrite=True,
                )
                _rewrite_history_for_resume(
                    history_file, continue_from_time, history,
                )
            else:
                # Remove an existing file so HistoryHdf5 doesn't auto-rename.
                if os.path.isfile(history_file):
                    os.remove(history_file)
                history = HistoryHdf5(self.sheet, save_every=save_every, dt=dt,
                                       hf5file=history_file, overwrite=True)
        else:
            history = History(self.sheet, save_every=save_every, save_all=False, dt=dt)
        model = self.get_model(only_differentiation, effectors=effectors)
        solver = IVPSolver(self, self.sheet, self.sheet.geom, model, manager=manager, history=history, auto_reconnect=False)
        # When resuming, seed the solver's clock so its ``while
        # current_t < tf`` loop starts at the resume time. Without
        # this the solver would happily begin at t=0 and double-cover
        # everything up to t_end.
        if continue_from_time is not None:
            solver.prev_t = float(continue_from_time)
            # Also stamp the resume time on the in-memory ``History``
            # so the first ``_record_at`` call doesn't mis-tag the
            # initial snapshot.
            history.time = float(continue_from_time)

        # Translate the differentiation flags into the solver's
        # steady-state check flags. ``only_differentiation`` means
        # there's no mechanical evolution to wait on (only LI must
        # converge). ``no_differentiation`` means LI levels don't
        # evolve (only mechanics must settle). The remaining default
        # asks for BOTH to converge before halting.
        check_mech = bool(until_steady_state) and not bool(only_differentiation)
        check_li = bool(until_steady_state) and not bool(no_differentiation)
        if until_steady_state and not (check_mech or check_li):
            # only_differentiation AND no_differentiation are both True
            # — that's contradictory. Warn rather than silently looping
            # forever with no halt criterion.
            import warnings as _w
            _w.warn(
                "until_steady_state=True but BOTH only_differentiation "
                "and no_differentiation are set; steady-state stop "
                "cannot fire and the simulation will run to t_end."
            )

        solver.solve(
            tf=t_end, dt=dt,
            quasi_static=quasi_static, quasi_static_threshold=quasi_static_threshold,
            max_displacement=max_displacement,
            max_disp_factor=max_disp_factor,
            dt_min_factor=dt_min_factor,
            dt_increase_factor=dt_increase_factor,
            save_interval=save_interval,
            until_steady_state=until_steady_state,
            lateral_inhibition_threshold=lateral_inhibition_threshold,
            check_mechanical_steady=check_mech,
            check_lateral_inhibition_steady=check_li,
            steady_state_min_steps=steady_state_min_steps,
            max_wall_seconds=max_wall_seconds,
            min_progress_rate=min_progress_rate,
            progress_window_seconds=progress_window_seconds,
        )
        # fig, ax = plot_forces(self.sheet, geom, model, ['x', 'y'], 1)
        # plt.show()
        return history

    @staticmethod
    def get_draw_sheet_method(number_vertices=False, number_edges=False, number_faces=False, is_ordered=True,
                    maximal_level=1, color_by="atoh", arrange_sheet=False):
        def draw_sheet(sheet):
            if arrange_sheet:
                sheet.arrange_sheet_from_history()
            # if not sheet.check_all_edge_order():
            #     print("bug in drawing")
            #     sheet.order_all_edges()
            draw_specs = tyssue.config.draw.sheet_spec()
            cmap = plt.cm.get_cmap('Greens').reversed()
            if color_by == "atoh":
                cmap_scale = sheet.face_df.atoh_level.to_numpy()
            elif color_by == "delta":
                cmap_scale = sheet.face_df.delta_level.to_numpy() / maximal_level
            elif color_by == "inverse_repressor":
                cmap_scale = (maximal_level - sheet.face_df.repressor_level.to_numpy()) / maximal_level
            color_cmap = cmap(0.7*(cmap_scale - 1) + 1)
            draw_specs['face']['color'] = color_cmap
            draw_specs['face']['alpha'] = 0.5
            draw_specs['face']['visible'] = True
            if is_ordered:
                sheet.is_ordered = True
                sheet.edge_df.sort_values(["face", "order"], inplace=True)

            sheet_view = sheet.get_sheet_view_method()
            fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
            fig.set_size_inches((8, 8))

            if number_faces:
                for face, data in sheet.face_df.iterrows():
                    ax.text(data.x, data.y, face, fontsize=14, color="red")

            if number_vertices:
                for vert, data in sheet.vert_df.iterrows():
                    ax.text(data.x, data.y + 0.02, vert, weight="bold", color="blue")

            if number_edges:
                for edge, data in sheet.edge_df.iterrows():
                    ax.text((data.tx + data.sx)/2 - (data.tx - data.sx)/4,
                            (data.ty + data.sy)/2 - (data.ty - data.sy)/4 + 0.02,
                            edge, weight="bold", color="green")

            return fig, ax
        return draw_sheet
    @staticmethod
    def save_sheet_labels_to_numpy(sheet, path, pixels_per_unit=40):
        """Save a labeled segmentation image of the sheet.

        Each interior pixel holds ``unique_id + 1`` of the face it
        belongs to; cell-cell boundaries (membranes) are 0. Labels are
        1-based (``unique_id + 1``) so the value 0 is reserved
        unambiguously for boundaries — matching the experimental
        segmentation convention (labels 1..N, 0 = membrane) and
        avoiding any clash with the 0-based ``unique_id``. A pixel
        value ``v`` therefore maps to the cell with
        ``unique_id == v - 1`` (== the ``cells_info`` row index, which
        is the ``unique_id``).

        The previous implementation rendered the sheet to an RGB
        canvas and tried to recover labels from pixel colors. That was
        fundamentally broken: ``alpha=0.5`` blended every face colour
        with the white background and the edges, antialiasing smeared
        the boundaries, and the decode loop assigned cells sequential
        integers (1, 2, 3, ...) in colour-sort order rather than their
        ``unique_id``. The result was ~99 % boundary with scrambled,
        non-``unique_id`` labels.

        This version rasterizes the face polygons directly via exact
        point-in-polygon tests (``matplotlib.path.Path``), so the
        label of every interior pixel is exactly the face's
        ``unique_id`` — no colour round-trip, no antialiasing, no
        relabeling. Periodic boundary-crossing faces are unfolded
        (min-image, the same construction
        ``PeriodicPlanarGeometry`` / ``periodic_sheet_view`` use) and
        their pixels wrapped back into ``[0, Lx) x [0, Ly)`` so they
        appear on both sides of the tile, as they should.

        Parameters
        ----------
        pixels_per_unit : int, default 40
            Image resolution. For the usual 20x20 periodic box this
            gives an 800x800 image (matching the previous output
            size).
        """
        from matplotlib.path import Path as _MplPath

        periodic = bool(getattr(sheet, "periodic", False))
        Lx = getattr(sheet, "Lx", None)
        Ly = getattr(sheet, "Ly", None)
        # Defensive: a sheet straight from ``history.retrieve`` (not run
        # through ``arrange_sheet_from_history``) keeps the default
        # ``periodic=False`` and no ``Lx``/``Ly`` even though its
        # ``face_df`` still carries the periodic metadata. Recover it
        # here so the segmentation can't silently fall back to the
        # non-periodic branch — which leaves boundary-crossing faces
        # unfolded and scrambles the image at the seam.
        if (not periodic) and ("_periodic_flag" in sheet.face_df.columns) \
                and bool(sheet.face_df["_periodic_flag"].iloc[0]):
            periodic = True
            Lx = float(sheet.face_df["_periodic_Lx"].iloc[0])
            Ly = float(sheet.face_df["_periodic_Ly"].iloc[0])

        if periodic:
            x0 = y0 = 0.0
            Lx, Ly = float(Lx), float(Ly)
        else:
            xv = sheet.vert_df["x"].to_numpy()
            yv = sheet.vert_df["y"].to_numpy()
            x0, y0 = float(xv.min()), float(yv.min())
            Lx, Ly = float(xv.max()) - x0, float(yv.max()) - y0

        W = max(1, int(round(Lx * pixels_per_unit)))
        H = max(1, int(round(Ly * pixels_per_unit)))
        labels = np.zeros((H, W), dtype=np.int64)

        vx = sheet.vert_df["x"]
        vy = sheet.vert_df["y"]
        edge_df = sheet.edge_df

        for face_id in sheet.face_df.index:
            face_edges = edge_df[edge_df["face"] == face_id]
            if "order" in face_edges.columns:
                face_edges = face_edges.sort_values("order")
            srce = face_edges["srce"].to_numpy()
            if srce.size < 3:
                continue
            xs = vx.loc[srce].to_numpy().astype(float).copy()
            ys = vy.loc[srce].to_numpy().astype(float).copy()
            if periodic:
                # Unfold the perimeter so the polygon is contiguous
                # (no edge spans more than half a period).
                for i in range(1, xs.size):
                    while xs[i] - xs[i - 1] > Lx / 2: xs[i] -= Lx
                    while xs[i] - xs[i - 1] < -Lx / 2: xs[i] += Lx
                    while ys[i] - ys[i - 1] > Ly / 2: ys[i] -= Ly
                    while ys[i] - ys[i - 1] < -Ly / 2: ys[i] += Ly
            px = (xs - x0) * pixels_per_unit
            py = (ys - y0) * pixels_per_unit
            c0 = int(np.floor(px.min())); c1 = int(np.ceil(px.max())) + 1
            r0 = int(np.floor(py.min())); r1 = int(np.ceil(py.max())) + 1
            cols = np.arange(c0, c1)
            rows = np.arange(r0, r1)
            if cols.size == 0 or rows.size == 0:
                continue
            gx, gy = np.meshgrid(cols + 0.5, rows + 0.5)
            inside = _MplPath(np.column_stack([px, py])).contains_points(
                np.column_stack([gx.ravel(), gy.ravel()])
            ).reshape(gy.shape)
            cc, rr = np.meshgrid(cols, rows)
            # 1-based label so 0 stays reserved for boundaries.
            label_val = int(sheet.face_df.at[face_id, "unique_id"]) + 1
            if periodic:
                cc = np.mod(cc, W)
                rr = np.mod(rr, H)
                sel = inside
            else:
                sel = inside & (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            labels[rr[sel], cc[sel]] = label_val

        # Guarantee every face contributes at least its centroid pixel
        # (so a cell smaller than the membrane width can't vanish). The
        # centroid is deep in the interior, so this never lands on a
        # membrane.
        fx = sheet.face_df["x"].to_numpy()
        fy = sheet.face_df["y"].to_numpy()
        uids = sheet.face_df["unique_id"].to_numpy()
        crow = np.round((fy - y0) * pixels_per_unit).astype(int)
        ccol = np.round((fx - x0) * pixels_per_unit).astype(int)
        if periodic:
            crow = np.mod(crow, H); ccol = np.mod(ccol, W)
        else:
            crow = np.clip(crow, 0, H - 1); ccol = np.clip(ccol, 0, W - 1)
        labels[crow, ccol] = uids + 1  # 1-based, matching the fill above

        # Draw 1-px cell-cell membranes as 0. ``np.roll`` makes the
        # comparison wrap for a periodic tile; for a non-periodic sheet
        # we use a non-wrapping shifted diff so the image edge isn't
        # spuriously marked.
        membrane = np.zeros_like(labels, dtype=bool)
        for axis in (0, 1):
            if periodic:
                membrane |= labels != np.roll(labels, 1, axis=axis)
            else:
                diff = np.zeros_like(labels, dtype=bool)
                if axis == 0:
                    diff[1:, :] = labels[1:, :] != labels[:-1, :]
                else:
                    diff[:, 1:] = labels[:, 1:] != labels[:, :-1]
                membrane |= diff
        labels[membrane] = 0

        out_dtype = "uint16" if int(labels.max()) < 2 ** 16 else "uint32"
        np.save(path, labels.astype(out_dtype))
        return 0

    @staticmethod
    def save_contact_matrix_to_numpy(sheet, path):
        contact_matrix = sheet.get_contact_matrix()
        np.save(path, contact_matrix)
        return 0

    @staticmethod
    def save_face_data_to_df(sheet, path):
        face_data = sheet.face_df
        # ``get_neighbors`` returns face labels (face_df index values);
        # map them to ``unique_id`` so the neighbor lists are in the
        # same id-space as this table's index and the contact matrix
        # (they coincide only for a fresh contiguous sheet, but diverge
        # once divisions / delaminations leave gaps in unique_id).
        uid_of_face = face_data["unique_id"]
        neighbors = []
        for id, face in face_data.iterrows():
            neighbor_faces = sheet.get_neighbors(id)
            neighbors.append(set(uid_of_face.loc[neighbor_faces].to_numpy().tolist()))
        # ``unique_id`` is the table INDEX: ``cells_info.loc[u]`` gives
        # the cell whose ``unique_id`` is ``u``. That ties the table to
        # both ``labels.npy`` (pixel value ``v`` -> ``unique_id v - 1``)
        # and ``contact_matrix`` (row/col ``u`` -> cell ``unique_id u``).
        # ``label`` keeps the 1-based ``unique_id + 1`` convention used
        # by ``labels.npy`` / the experimental data.
        cells_info_dict = {"unique_id": face_data.unique_id.to_numpy(),
                           "label": face_data.unique_id.to_numpy() + 1,
                           "cx":face_data.x.to_numpy(),
                           "cy": face_data.y.to_numpy(), "type": face_data.type.to_numpy(),
                           "perimeter": face_data.perimeter.to_numpy(), "valid": face_data.is_alive.to_numpy(),
                           "notch_level": face_data.notch_level.to_numpy(),
                           "delta_level": face_data.delta_level.to_numpy(),
                           "neighbors": neighbors}
        cells_info = pd.DataFrame(cells_info_dict).set_index("unique_id")
        cells_info.to_pickle(path)
        return 0

