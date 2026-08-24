"""Per-cell measurements on a single frame: neighbours, contacts, shape, area.

Everything here takes a sheet (one retrieved frame) and returns per-cell arrays,
so the caller decides how to aggregate. Two conventions run through the module:

* **Boundary cells are excluded.** On a non-periodic sheet the outermost cells
  have no complete neighbourhood, so their neighbour counts and roundness are not
  comparable to interior ones. On a periodic sheet there is no boundary and the
  selection is a no-op.
* **Cell type is a threshold on a signal column**, not a stored label, so the
  same frame can be classified by different thresholds without re-running
  anything. ``type_by`` names the column and ``threshold`` splits it.
"""

import os, sys
import numpy as np
import pandas as pd
from tyssue import History, HistoryHdf5
from virtual_sheet import VirtualSheet
from history_io import get_time_points

RESULTS_DIR = os.environ.get("TISSUE_RESULTS_DIR", r"D:\Kasirer\results")


def find_non_boundary_cells(time_point_data):
    boundary_cells = np.unique(time_point_data.edge_df.loc[time_point_data.edge_df.opposite < 0, "face"])
    neighbors_of_boundary_cells = np.unique(time_point_data.edge_df.face[time_point_data.edge_df.opposite.isin(boundary_cells)])
    exclude_cells = np.union1d(boundary_cells, neighbors_of_boundary_cells)
    face_idx = time_point_data.face_df.index
    non_boundary_cells = np.setdiff1d(face_idx, exclude_cells)
    return non_boundary_cells

def find_maximal_level_final_frame(load_name,  type_by='atoh_level'):
    load_path = os.path.join(RESULTS_DIR, load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    last_time_point = np.max(history.time_stamps)
    final_sheet = history.retrieve(last_time_point)
    final_sheet.arrange_sheet_from_history()
    face_ids = find_non_boundary_cells(final_sheet)
    level = final_sheet.face_df.loc[face_ids, type_by]
    return np.max(level)

def get_non_boundary_cell_ids_from_type(time_point_data, cell_type='all',
                                     type_by='atoh_level', threshold=None,
                                     HC_above_threshold=True, only_for_these_cells=None):
    """
    only_for_these_cells - Index of subset of cells. Only cells from this subset would be returned
    Returns - cells ordinal index, cells ids
    """
    face_idx = find_non_boundary_cells(time_point_data)
    if only_for_these_cells is not None:
        only_for_these_cells_idx = time_point_data.face_df[time_point_data.face_df['id'].isin(only_for_these_cells)].index
        face_idx = np.intersect1d(face_idx, only_for_these_cells_idx)
    if cell_type == "all":
        relevant_cells = face_idx
    else:
        type_data = time_point_data.face_df.loc[face_idx, type_by]
        if threshold is None:
            threshold = (np.max(type_data) + np.min(type_data)) / 2
            print("Using calculated threshold = %f" % threshold)
        if HC_above_threshold:
            is_HC = type_data > threshold
        else:
            is_HC = type_data < threshold
        if cell_type == "HC":
            relevant_cells = face_idx[is_HC]
        elif cell_type == "SC":
            relevant_cells = face_idx[~is_HC]
        else:
            raise "not implemented cell type"
    return time_point_data.face_df.index.get_indexer(relevant_cells), time_point_data.face_df.loc[relevant_cells, "id"].values

def calc_contact_with_neighbors_from_type(time_point_data, cell_type='all', neighbor_type='all',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None):

    relevant_cells_idx, _ = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=cell_type,
                                                      type_by=type_by, threshold=threshold,
                                                      HC_above_threshold=HC_above_threshold,
                                                      only_for_these_cells=only_for_these_cells)
    if neighbor_type==cell_type:
        relevant_neighbors_idx = relevant_cells_idx
    else:
        relevant_neighbors_idx, _ = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=neighbor_type,
                                                                 type_by=type_by, threshold=threshold,
                                                                 HC_above_threshold=HC_above_threshold,
                                                                 only_for_these_cells=only_for_these_cells)
    contact_matrix = time_point_data.get_contact_matrix()
    relevant_contacts = contact_matrix[np.ix_(relevant_cells_idx, relevant_neighbors_idx)]
    contact_length = relevant_contacts.sum(axis=1)
    binary_relevant_contacts = (relevant_contacts > 0).astype(int)
    number_of_neighbors = binary_relevant_contacts.sum(axis=1)
    return number_of_neighbors, contact_length

def calc_roundness_for_type(time_point_data, cell_type='all',
                            type_by='atoh_level', threshold=None,
                            HC_above_threshold=True, only_for_these_cells=None):
    _, relevant_cells = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=cell_type,
                                                         type_by=type_by, threshold=threshold,
                                                         HC_above_threshold=HC_above_threshold,
                                                         only_for_these_cells=only_for_these_cells)
    roundness = time_point_data.get_face_roundness()
    relevant_values = roundness.loc[relevant_cells].values
    return relevant_values

def calc_area_for_type(time_point_data, cell_type='all',
                       type_by='atoh_level', threshold=None,
                       HC_above_threshold=True, only_for_these_cells=None):
    """Face areas of the non-boundary cells of ``cell_type`` (mirror of
    :func:`calc_roundness_for_type`)."""
    _, relevant_cells = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=cell_type,
                                                         type_by=type_by, threshold=threshold,
                                                         HC_above_threshold=HC_above_threshold,
                                                         only_for_these_cells=only_for_these_cells)
    area = time_point_data.get_face_area()
    relevant_values = area.loc[relevant_cells].values
    return relevant_values

def calc_roundness_for_time_point(history, load_name, cell_type='HC',
                                       type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                       only_for_these_cells=None, time_point=-1):
    if time_point == -1:
        time_point = np.max(history.time_stamps)
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history()
    res = calc_roundness_for_type(sheet, cell_type=cell_type,
                                                type_by=type_by, threshold=threshold,
                                                HC_above_threshold=HC_above_threshold,
                                                only_for_these_cells=only_for_these_cells)
    np.save("%s results %s roundness" % (load_name, cell_type), res)
    return res

def calc_contacts_for_time_point(history, save_name, cell_type='HC', neighbor_type='HC',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None, time_point=-1):

    unique_stamps = get_time_points(history)
    if time_point == -1:
        time_point = unique_stamps[time_point]
    if time_point > np.max(unique_stamps):
        raise ValueError("time_point must be less than last_time_point")
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history()
    res = calc_contact_with_neighbors_from_type(sheet, cell_type=cell_type, neighbor_type=neighbor_type,
                                          type_by=type_by, threshold=threshold, HC_above_threshold=HC_above_threshold,
                                          only_for_these_cells=only_for_these_cells)
    np.save("%s results %s with %s neighbors"%(save_name, cell_type, neighbor_type), res)
    return res

def calc_HC_neighbors_at_differentiation(history, initial_time_point=0,
                                         type_by='atoh_level', threshold=None,
                                         HC_above_threshold=True):
    """Number of HC neighbors each final-frame hair cell had at the moment
    it differentiated.

    Every hair cell present (above ``threshold`` of ``type_by``) in the
    last frame is traced backwards through time to its *last threshold
    crossing* — the most recent frame at which it became a HC after not
    being one. That frame is taken as the cell's differentiation time, and
    the number of HC neighbors it had then is recorded.

    Tracing starts no earlier than ``initial_time_point`` (which need not
    be ``t=0``). Cells that were *already* HCs at ``initial_time_point``
    are excluded from the output — they did not differentiate within the
    traced window so no differentiation time exists for them. A single
    fixed ``threshold`` is used in every frame so that "crossing" means the
    same thing throughout (defaults to the mid-range of ``type_by`` over the
    final frame's non-boundary cells, matching the convention in
    :func:`get_non_boundary_cell_ids_from_type`).

    Parameters
    ----------
    history : :class:`HistoryHdf5` or :class:`tyssue.core.history.History`
        Loaded simulation history.
    initial_time_point : float
        Earliest time stamp to trace back to. Frames before it are ignored.
    type_by : str
        Face column whose value defines hair-cell identity.
    threshold : float, optional
        Threshold on ``type_by``. Computed from the final frame when None.
    HC_above_threshold : bool
        If True a HC is ``value > threshold``; if False ``value < threshold``.

    Returns
    -------
    numpy.ndarray
        1D array, one entry per final-frame HC that differentiated within
        the traced window, giving its HC-neighbor count at its
        differentiation time (NaN only when the cell was a boundary cell in
        that frame and so has no recorded neighbor count — which cannot
        happen for a periodic sheet).
    """
    time_points = get_time_points(history)
    time_points = time_points[time_points >= initial_time_point]
    if time_points.size == 0:
        raise ValueError("No time points at or after initial_time_point=%f" % initial_time_point)
    last_time_point = time_points[-1]

    final_sheet = history.retrieve(last_time_point)
    final_sheet.arrange_sheet_from_history()
    if threshold is None:
        non_boundary = find_non_boundary_cells(final_sheet)
        type_data = final_sheet.face_df.loc[non_boundary, type_by]
        threshold = (np.max(type_data) + np.min(type_data)) / 2
        print("Using calculated threshold = %f" % threshold)

    # Walk forward once over every frame, caching per persistent cell ``id``:
    #   value_by_id[f]          -> Series id -> value of ``type_by``
    #   n_HC_neighbors_by_id[f] -> Series id -> number of HC neighbors
    # so the backward differentiation-time search below is pure lookups.
    value_by_id = []
    n_HC_neighbors_by_id = []
    for time_point in time_points:
        sheet = history.retrieve(time_point)
        sheet.arrange_sheet_from_history()
        value_by_id.append(sheet.face_df.set_index("id")[type_by])
        _, HC_ids = get_non_boundary_cell_ids_from_type(
            sheet, cell_type="HC", type_by=type_by, threshold=threshold,
            HC_above_threshold=HC_above_threshold)
        number_of_HC_neighbors, _ = calc_contact_with_neighbors_from_type(
            sheet, cell_type="HC", neighbor_type="HC", type_by=type_by,
            threshold=threshold, HC_above_threshold=HC_above_threshold)
        n_HC_neighbors_by_id.append(pd.Series(number_of_HC_neighbors, index=HC_ids))

    # The cells we report on: HCs in the final frame.
    _, final_HC_ids = get_non_boundary_cell_ids_from_type(
        final_sheet, cell_type="HC", type_by=type_by, threshold=threshold,
        HC_above_threshold=HC_above_threshold)

    def is_HC(value):
        if value is None or np.isnan(value):
            return False
        return value > threshold if HC_above_threshold else value < threshold

    last_frame = time_points.size - 1
    result = []
    for cell_id in final_HC_ids:
        # Step back through the uninterrupted run of HC frames ending at the
        # final frame; ``diff_frame`` ends on its first frame == the last
        # below->above crossing (or ``initial_time_point`` if it was already
        # a HC there).
        diff_frame = last_frame
        while diff_frame > 0 and is_HC(value_by_id[diff_frame - 1].get(cell_id)):
            diff_frame -= 1
        # diff_frame == 0 means the cell was a HC continuously from the
        # initial frame onward, i.e. it was already differentiated at
        # ``initial_time_point`` -> no crossing in the window, skip it.
        if diff_frame == 0:
            continue
        result.append(n_HC_neighbors_by_id[diff_frame].get(cell_id, np.nan))
    return np.array(result)

def calc_percentage_of_differentiating_by_initial_neighbors(
        history, initial_time, max_number_of_neighbors,
        type_by='atoh_level', threshold=None, HC_above_threshold=True):
    """For each initial HC-neighbor count ``n`` (0, 1, ..., ``max`` where
    ``max`` means *>= max*), the percentage of the support cells (SCs) that
    had ``n`` HC neighbors at ``initial_time`` and went on to differentiate
    (became HCs by the final frame):

        100 * #(differentiated SCs with n initial HC neighbors)
            / #(SCs with n initial HC neighbors at initial_time)

    Returns a 1D array of length ``max_number_of_neighbors + 1`` (NaN for a
    bin that has no SCs at ``initial_time``)."""
    time_points = get_time_points(history)
    last_time_point = time_points[time_points >= initial_time][-1]

    initial_sheet = history.retrieve(initial_time)
    initial_sheet.arrange_sheet_from_history()
    final_sheet = history.retrieve(last_time_point)
    final_sheet.arrange_sheet_from_history()

    # SCs at initial_time and each one's HC-neighbor count, clumped at the cap.
    _, sc_ids = get_non_boundary_cell_ids_from_type(
        initial_sheet, cell_type="SC", type_by=type_by, threshold=threshold,
        HC_above_threshold=HC_above_threshold)
    n_HC_neighbors, _ = calc_contact_with_neighbors_from_type(
        initial_sheet, cell_type="SC", neighbor_type="HC", type_by=type_by,
        threshold=threshold, HC_above_threshold=HC_above_threshold)
    clumped = np.minimum(n_HC_neighbors, max_number_of_neighbors)

    # Which of those SCs are HCs by the final frame == they differentiated.
    _, final_HC_ids = get_non_boundary_cell_ids_from_type(
        final_sheet, cell_type="HC", type_by=type_by, threshold=threshold,
        HC_above_threshold=HC_above_threshold)
    differentiated = np.isin(sc_ids, final_HC_ids)

    percentages = np.full(max_number_of_neighbors + 1, np.nan)
    for n in range(max_number_of_neighbors + 1):
        in_bin = clumped == n
        denominator = np.count_nonzero(in_bin)
        if denominator > 0:
            percentages[n] = 100.0 * np.count_nonzero(in_bin & differentiated) / denominator
    return percentages

def calc_area_change_after_ablation(history, load_name, ablated_cells=[], end_time=-1, type_by='atoh_level', threshold=None,
                                    HC_above_threshold=True):
    if end_time is None or end_time == -1:
        end_time = get_time_points(history)[-1]
    initial_sheet = history.retrieve(0)
    initial_sheet.arrange_sheet_from_history()
    HC_neighbors_of_ablated = []
    SC_neighbors_of_ablated = []
    for ablated in ablated_cells:
        neighbors = initial_sheet.get_neighbors(ablated)
        neighbors = np.setdiff1d(neighbors, ablated, assume_unique=True)
        # get_neighbors returns face INDEX labels, but ``only_for_these_cells``
        # below is matched against the ``id`` COLUMN. The two coincide only
        # while no face has been removed — loading an archive resets the index
        # to 0..N-1 while ``id`` keeps its original values — so a base run that
        # had delaminated a cell before this ablation would silently select the
        # WRONG neighbours here, with no error. Translate explicitly; this is a
        # no-op whenever index == id, which is the case for every run scored so
        # far (verified), so no existing result changes.
        neighbors = initial_sheet.face_df.loc[neighbors, "id"].to_numpy()
        # ``get_non_boundary_cell_ids_from_type`` returns (ordinal_index, ids);
        # we accumulate the IDS. ``np.union1d`` returns a SINGLE array (not a
        # tuple) — the previous code passed the whole (ordinal, ids) tuple in
        # AND unpacked the union as ``_, x``, which only didn't crash when the
        # union happened to have exactly two elements (and even then produced
        # garbage). A 1-element union raised "not enough values to unpack".
        _, hc_ids = get_non_boundary_cell_ids_from_type(
            initial_sheet, cell_type="HC", type_by=type_by, threshold=threshold,
            HC_above_threshold=HC_above_threshold, only_for_these_cells=neighbors)
        HC_neighbors_of_ablated = np.union1d(HC_neighbors_of_ablated, hc_ids)
        _, sc_ids = get_non_boundary_cell_ids_from_type(
            initial_sheet, cell_type="SC", type_by=type_by, threshold=threshold,
            HC_above_threshold=HC_above_threshold, only_for_these_cells=neighbors)
        SC_neighbors_of_ablated = np.union1d(SC_neighbors_of_ablated, sc_ids)
    final_sheet = history.retrieve(end_time)
    final_sheet.arrange_sheet_from_history()
    exist_in_final_frame = final_sheet.face_df.id.values
    HC_neighbors_of_ablated = np.intersect1d(HC_neighbors_of_ablated, exist_in_final_frame)
    SC_neighbors_of_ablated = np.intersect1d(SC_neighbors_of_ablated, exist_in_final_frame)
    initial_face_area = initial_sheet.get_face_area()
    initial_HC_area_next_to_ablated = initial_face_area.loc[HC_neighbors_of_ablated].values
    initial_SC_area_next_to_ablated = initial_face_area.loc[SC_neighbors_of_ablated].values
    final_face_area = final_sheet.get_face_area()
    final_HC_area_next_to_ablated = final_face_area.loc[HC_neighbors_of_ablated].values
    final_SC_area_next_to_ablated = final_face_area.loc[SC_neighbors_of_ablated].values
    HC_area_ratio = final_HC_area_next_to_ablated / initial_HC_area_next_to_ablated
    SC_area_ratio = final_SC_area_next_to_ablated / initial_SC_area_next_to_ablated
    return HC_area_ratio, SC_area_ratio

def _li_length_for_model(folder):
    """Number of cells (``max(unique_id) + 1``) in the model whose archive
    lives in ``folder``. Reads the cheap sidecar artifacts first
    (``cells_info.pkl`` -> ``contact_matrix.npy``) and only falls back to
    loading ``history.hf5`` if neither is present. Used to validate that an
    extracted per-cell array has exactly one entry per cell."""
    ci_path = os.path.join(folder, "cells_info.pkl")
    if os.path.isfile(ci_path):
        return int(pd.read_pickle(ci_path).index.max()) + 1
    cm_path = os.path.join(folder, "contact_matrix.npy")
    if os.path.isfile(cm_path):
        return int(np.load(cm_path).shape[0])
    hist = HistoryHdf5.from_archive(os.path.join(folder, "history.hf5"),
                                    eptm_class=VirtualSheet)
    sheet = hist.retrieve(np.max(hist.time_stamps))
    sheet.arrange_sheet_from_history()
    return int(sheet.face_df["unique_id"].astype(int).max()) + 1
