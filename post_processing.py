import hashlib
import os, sys
import re
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tyssue import History, HistoryHdf5
from tyssue.draw.plt_draw import create_gif
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

# Where the sibling ``tissue_analyzing_tool`` package (providing
# ``statistical_analysis``) lives. Override with the ``TISSUE_ANALYZER_PATH``
# environment variable (e.g. on an Azure VM); defaults to the local checkout.
_TISSUE_ANALYZER_PATH = os.environ.get(
    "TISSUE_ANALYZER_PATH",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_image_processing\tissue_analyzing_tool")
sys.path.insert(0, _TISSUE_ANALYZER_PATH)


# Conservative cap on the FULL gif output path length. Windows' classic
# MAX_PATH is 260, and ImageMagick's ``convert`` (which create_gif
# shells out to) fails to write an output file when the path gets long.
# We keep a safety margin below 260.
_MAX_GIF_PATH_LEN = 200


def _shorten_gif_output(output, max_path_len=_MAX_GIF_PATH_LEN):
    """Return a gif ``output`` path whose total length is at most
    ``max_path_len`` by truncating the FILE NAME only — the directory
    is preserved exactly.

    A short hash of the original stem is appended so two distinct long
    names don't collide after truncation. If ``output`` already fits,
    it's returned unchanged.
    """
    output = os.fspath(output)
    if len(output) <= max_path_len:
        return output

    directory = os.path.dirname(output)
    stem, ext = os.path.splitext(os.path.basename(output))
    digest = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]

    # Budget for the (truncated) stem:
    #   total = directory + sep + stem[:keep] + "_" + digest + ext
    sep = os.sep if directory else ""
    fixed = len(directory) + len(sep) + 1 + len(digest) + len(ext)  # +1 for "_"
    keep = max_path_len - fixed
    if keep < 1:
        # The directory alone is already near the limit — keep just the
        # hash + extension so the name is at least unique and valid.
        short_base = digest + ext
    else:
        short_base = stem[:keep] + "_" + digest + ext
    return os.path.join(directory, short_base)


def create_gif_safe(history, output, **kwargs):
    """Drop-in wrapper around :func:`tyssue.draw.plt_draw.create_gif`
    that shortens an over-long output FILE NAME (keeping its
    directory) before handing it to ImageMagick's ``convert`` — which
    otherwise fails on long output paths. All other arguments are
    forwarded unchanged.
    """
    safe = _shorten_gif_output(output)
    if safe != output:
        print(
            "gif output name shortened to fit path limits: %s"
            % os.path.basename(safe)
        )
    return create_gif(history, safe, **kwargs)


# Folder holding the experimental measurements (npy / pickle / ODS). Override
# with the ``EXPERIMENTAL_DATA_DIR`` environment variable; defaults to the
# local checkout.
experimental_results_folder = os.environ.get(
    "EXPERIMENTAL_DATA_DIR",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data")
E17_number_of_HC_neighbors_file_name = r"E17.5 differentiating cells number of HC neighbors.npy"
E17_contact_length_with_HC_neighbors_file_name = r"E17.5 differentiating cells contacts length with HC.npy"
E17_HC_roundness_file_name = r"E17.5 +24h HC roundness.npy"
E17_SC_roundness_file_name = r"E17.5 +24h SC roundness.npy"
P0_number_of_HC_neighbors_file_name = r"P0 differentiating cells number of HC neighbors.npy"
P0_contact_length_with_HC_neighbors_file_name = r"P0 differentiating cells contact length with HC.npy"
P0_HC_roundness_file_name = r"P0 +24h HC roundness.npy"
P0_SC_roundness_file_name = r"P0 +24h SC roundness.npy"
ablation_area_change_file_name = r"ablation_area_change_summary.ods"


# Naming convention for the simulated random arrays, kept here (the dependency
# root; run_model imports from post_processing, not the other way round) so the
# integer-index <-> folder-name mapping lives in exactly ONE place. This is what
# lets a single Azure Batch task address one array by its index.
_STAGE_SHEET_SUFFIX = {"E17.5": "_for_E17", "P0": "_for_P0"}


def random_array_name(index):
    """Folder name of the raw random array with the given integer ``index``."""
    return "random_periodic_array%d" % index


def initial_morphology_name(index, stage):
    """Folder name of the fitted initial-morphology array: raw random array
    ``index`` snapped to ``stage``'s best-matching time point (the input handed
    to a differentiation run)."""
    return "%s%s" % (random_array_name(index), _STAGE_SHEET_SUFFIX[stage])


def redraw(load_name, save_name, movie=True, maximal_number_of_frames_to_save=100, color_by="atoh", maximal_level=1):

    load_path = os.path.join("results", load_name)
    history = HistoryHdf5.from_archive(os.path.join(load_path, "history.hf5"), eptm_class=VirtualSheet)
    initial_sheet = history.retrieve(0)
    last_time_point = np.max(history.time_stamps)
    number_of_time_points = np.unique(history.time_stamps).size
    final_sheet = history.retrieve(last_time_point)
    number_of_frames_to_save = min(number_of_time_points, maximal_number_of_frames_to_save)

    save_path = os.path.join("results", load_name, save_name)
    static_draw_func = InnerEarModel.get_draw_sheet_method(number_faces=False, number_edges=False, number_vertices=False,
                                         arrange_sheet=False, color_by=color_by, maximal_level=maximal_level)
    fig1, ax1 = static_draw_func(initial_sheet)
    plt.savefig("%s_initial.png" % save_path)
    fig2, ax2 =static_draw_func(final_sheet)
    plt.savefig("%s_finale.png" % save_path)
    if movie:
        gif_draw_func = InnerEarModel.get_draw_sheet_method(number_faces=False, number_edges=False, number_vertices=False,
                                             arrange_sheet=False, color_by=color_by, maximal_level=maximal_level)
        create_gif_safe(history, os.path.join(os.getcwd(), "%s_movie.gif" % save_path), num_frames=number_of_frames_to_save,
                   draw_func=gif_draw_func)
    return 0

def drop_corrupted_snapshots(archive_path, dry_run=False):
    """Remove snapshots with non-positive face areas from a HistoryHdf5
    archive in place.

    A successful periodic run never records a snapshot with a
    non-positive face area — the solver rejects any step that flips a
    face's signed area. So a stored snapshot with ``area <= 0`` is a
    sure sign of corruption. This happens, for example, when a resume
    accidentally loaded the sheet as NON-periodic (legacy archive
    missing the ``_periodic_flag``): ``geom.update_all`` then unwrapped
    boundary-crossing faces into domain-spanning polygons (huge
    perimeters, negative areas) and that bad geometry was re-recorded
    over the originally-good snapshot. Re-running off such a snapshot
    poisons ``InnerEarModel``'s ``length_normalization_factor`` (it
    averages the corrupted ``perimeter`` column), so the corrupted
    snapshot must be dropped first.

    Parameters
    ----------
    archive_path : str
        Path to the ``history.hf5`` archive to clean (modified in
        place).
    dry_run : bool, default False
        When True, only report which time stamps WOULD be dropped
        without touching the file.

    Returns
    -------
    dropped_times : list of float
        The time stamps that were (or, under ``dry_run``, would be)
        removed.
    """
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(archive_path)

    # Identify corrupted time stamps from the face table.
    with pd.HDFStore(archive_path, "r") as store:
        face = store.select("face", columns=["time", "area"])
    bad_mask = face["area"] <= 0
    dropped_times = sorted(face.loc[bad_mask, "time"].unique().tolist())

    if dry_run or not dropped_times:
        return dropped_times

    # Remove every row at a corrupted time from every time-indexed
    # table. ``time`` is a data_column, so the where-clause delete is
    # cheap and targeted.
    with pd.HDFStore(archive_path, "a") as store:
        for key in list(store.keys()):
            element = key.strip("/")
            if element == "settings":
                continue
            for t in dropped_times:
                try:
                    store.remove(key=element, where="time == %r" % float(t))
                except (KeyError, ValueError, TypeError, NotImplementedError):
                    # Fallback: read-filter-rewrite for tables whose
                    # ``time`` isn't indexable.
                    df = store.select(element)
                    if "time" not in df.columns:
                        continue
                    kept = df[~np.isclose(df["time"], float(t))].copy()
                    store.remove(element)
                    if not kept.empty:
                        store.put(element, kept, format="table",
                                  data_columns=["time"])
    return dropped_times


def extract_time_point_to_new_history(source_history, time_point):
    """Build an in-memory one-snapshot :class:`History` from a
    multi-time history object.

    Defers all the heavy lifting to ``source_history.retrieve(...)``:
    that's where the snap-to-nearest semantics (return the closest
    recorded time when no row matches exactly) already live. We just
    wrap the returned sheet in a fresh :class:`History` and re-stamp
    the seed time so the new object reads back as a one-frame
    archive at the chosen moment.

    Parameters
    ----------
    source_history : :class:`tyssue.core.history.History` or :class:`HistoryHdf5`
        An already-loaded history object — typically obtained via
        :meth:`HistoryHdf5.from_archive`. Anything that exposes
        ``time_stamps`` and ``retrieve(t)`` works.
    time_point : float
        Time stamp to extract. If no recorded time matches exactly,
        the snap-to-nearest behaviour baked into
        :meth:`HistoryHdf5.retrieve` picks the closest sample (and
        in the ``time > time_stamps[-1]`` case warns through
        ``warnings.warn``).

    Returns
    -------
    new_history : :class:`tyssue.core.history.History`
        A fresh in-memory history holding exactly one snapshot at
        the chosen / snapped time, ready to be passed to
        ``create_gif``, ``retrieve``, ``browse``, or any other
        history-aware utility.
    """
    sheet = source_history.retrieve(time_point)
    # ``History`` expects each dataset's index to carry its element
    # name (``'vert'``, ``'edge'``, ``'face'``) so that the
    # ``reset_index(drop=False)`` inside ``History.__init__`` lands
    # the labels in a column named after the element — which the
    # subsequent ``History.retrieve`` then ``set_index``-es back.
    # ``HistoryHdf5.retrieve`` only renames ``edge_df``; ensure all
    # three are set so the returned history's OWN ``retrieve()``
    # works without a ``KeyError``.
    for elem_name, df in (("vert", sheet.vert_df),
                          ("edge", sheet.edge_df),
                          ("face", sheet.face_df)):
        if df.index.name != elem_name:
            df.index.rename(elem_name, inplace=True)

    # ``retrieve`` doesn't return the snapped time directly; recompute
    # it the same way ``HistoryHdf5.retrieve`` does so the new
    # history's seed row carries the right stamp.
    recorded_times = np.asarray(source_history.time_stamps)
    actual_time = float(
        recorded_times[np.argmin(np.abs(recorded_times - float(time_point)))]
    )

    # ``History.__init__`` seeds itself with a t=0 snapshot of the
    # given sheet; patch the seed to live at ``actual_time`` so
    # ``new_history.retrieve(actual_time)`` and
    # ``new_history.time_stamps`` agree with the source.
    new_history = History(sheet)
    new_history.time = actual_time
    for element_df in new_history.datasets.values():
        if "time" in element_df.columns:
            element_df["time"] = actual_time
    return new_history

def save_data_of_a_given_time_point(history, time_point, output_name):
    output_dir = os.path.join("results", output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sheet = history.retrieve(time_point)
    time_point_history = extract_time_point_to_new_history(history, time_point)
    # ``History.to_archive`` opens the file in APPEND mode and
    # ``store.append``-s every table — so writing to a path that
    # already holds an archive (e.g. re-running this extraction for
    # the same ``output_name``) stacks the new snapshot ON TOP of
    # the old one, leaving every row duplicated. Remove any stale
    # archive first so each call produces a clean single-snapshot
    # file.
    archive_path = os.path.join(output_dir, "history.hf5")
    if os.path.isfile(archive_path):
        os.remove(archive_path)
    time_point_history.to_archive(archive_path)
    InnerEarModel.save_sheet_labels_to_numpy(sheet, path=os.path.join(output_dir, "labels.npy"))
    InnerEarModel.save_contact_matrix_to_numpy(sheet, path=os.path.join(output_dir, "contact_matrix.npy"))
    InnerEarModel.save_face_data_to_df(sheet, path=os.path.join(output_dir, "cells_info.pkl"))
    draw_func = InnerEarModel.get_draw_sheet_method(number_faces=False, number_edges=False, number_vertices=False,
                                            color_by="delta")
    fig, ax = draw_func(sheet)
    plt.savefig(os.path.join(output_dir, "final.png"))
    plt.close(fig)
    return time_point_history.time



def find_non_boundary_cells(time_point_data):
    boundary_cells = np.unique(time_point_data.edge_df.loc[time_point_data.edge_df.opposite < 0, "face"])
    neighbors_of_boundary_cells = np.unique(time_point_data.edge_df.face[time_point_data.edge_df.opposite.isin(boundary_cells)])
    exclude_cells = np.union1d(boundary_cells, neighbors_of_boundary_cells)
    face_idx = time_point_data.face_df.index
    non_boundary_cells = np.setdiff1d(face_idx, exclude_cells)
    return non_boundary_cells

def find_maximal_level_final_frame(load_name,  type_by='atoh_level'):
    load_path = os.path.join("results", load_name, load_name)
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
    type_data = time_point_data.face_df.loc[face_idx, type_by]
    if threshold is None:
        threshold = (np.max(type_data) + np.min(type_data)) / 2
        print("Using calculated threshold = %f" % threshold)

    if HC_above_threshold:
        is_HC = type_data > threshold
    else:
        is_HC = type_data < threshold

    if cell_type == "all":
        relevant_cells = face_idx
    elif cell_type == "HC":
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

def load_history_file(load_name):
    load_path = os.path.join("results", load_name)
    history = HistoryHdf5.from_archive(os.path.join(load_path, "history.hf5"), eptm_class=VirtualSheet)
    return history

def get_time_points(history):
    return np.unique(history.time_stamps)

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


def calc_area_change_after_ablation(history, load_name, ablated_cells=[], end_frame=3, type_by='atoh_level', threshold=None,
                                    HC_above_threshold=True):
    initial_sheet = history.retrieve(0)
    initial_sheet.arrange_sheet_from_history()
    HC_neighbors_of_ablated = []
    SC_neighbors_of_ablated = []
    for ablated in ablated_cells:
        neighbors = initial_sheet.get_neighbors(ablated)
        neighbors = np.setdiff1d(neighbors, ablated, assume_unique=True)
        _, HC_neighbors_of_ablated = np.union1d(HC_neighbors_of_ablated,
                                             get_non_boundary_cell_ids_from_type(initial_sheet,
                                                                                 cell_type="HC",
                                                                                 type_by=type_by,
                                                                                 threshold=threshold,
                                                                                 HC_above_threshold=HC_above_threshold,
                                                                                 only_for_these_cells=neighbors))
        _, SC_neighbors_of_ablated = np.union1d(SC_neighbors_of_ablated,
                                             get_non_boundary_cell_ids_from_type(initial_sheet,
                                                                                 cell_type="SC",
                                                                                 type_by=type_by,
                                                                                 threshold=threshold,
                                                                                 HC_above_threshold=HC_above_threshold,
                                                                                 only_for_these_cells=neighbors))
    final_sheet = history.retrieve(end_frame)
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


def _experimental_late_cells_info_path(stage, experiment):
    """Path to the *latest* (``+24h``) ``cells_info`` pickle for a given
    experiment, i.e. the recorded frame other than the initial ``frame_1``.
    Found by globbing so the late-frame number (191/199/120 for E17.5, …)
    doesn't have to be hard-coded per experiment."""
    prefix = "E17" if stage == "E17.5" else "P0"
    pattern = os.path.join(experimental_results_folder, stage,
                           "%s_experiment%d_cells_info_frame_*" % (prefix, experiment))
    best_path, best_frame = None, -1
    for path in glob.glob(pattern):
        match = re.search(r"frame_(\d+)$", path)
        if match is None:
            continue
        frame = int(match.group(1))
        if frame != 1 and frame > best_frame:
            best_frame, best_path = frame, path
    if best_path is None:
        raise FileNotFoundError("No +24h cells_info frame found for %s experiment %d" % (stage, experiment))
    return best_path


def _read_ablation_area_ratios(stage):
    """Parse the ablation-summary ODS and return ``(hc_ratios, sc_ratios)``
    for ``stage`` — the per-cell *area after / area before* ablation ratios
    of the HC and SC rows respectively.

    Read by unzipping the ODS and walking ``content.xml`` directly so no
    optional ``odfpy`` dependency is required (it isn't installed in the
    ``tyssue`` environment).
    """
    import zipfile
    from xml.etree import ElementTree as ET
    table_ns = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    ods_path = os.path.join(experimental_results_folder, ablation_area_change_file_name)
    with zipfile.ZipFile(ods_path) as archive:
        root = ET.fromstring(archive.read("content.xml").decode("utf-8"))

    rows = []
    for table in root.iter("{%s}table" % table_ns):
        for row in table.findall("{%s}table-row" % table_ns):
            cells = []
            for cell in row.findall("{%s}table-cell" % table_ns):
                repeat = int(cell.get("{%s}number-columns-repeated" % table_ns, "1"))
                text = "".join(cell.itertext()).strip()
                cells.extend([text] * repeat)
            rows.append(cells)

    # First non-empty row is the header; map the columns we need by name.
    header = next(r for r in rows if any(c for c in r))
    col = {name: idx for idx, name in enumerate(header)}
    stage_i, type_i, ratio_i = col["stage"], col["cell type"], col["area ratio"]

    hc_ratios, sc_ratios = [], []
    for cells in rows:
        if len(cells) <= ratio_i or cells is header or cells[stage_i] != stage:
            continue
        try:
            ratio = float(cells[ratio_i])
        except (ValueError, IndexError):
            continue
        if cells[type_i] == "HC":
            hc_ratios.append(ratio)
        elif cells[type_i] == "SC":
            sc_ratios.append(ratio)
    return np.array(hc_ratios), np.array(sc_ratios)


def load_experimental_results(stage, type, cell_type='all', neighbor_type='all'):
    """``cell_type`` / ``neighbor_type`` (``'all'`` / ``'HC'`` / ``'SC'``)
    restrict, for the ``"number of neighbors"`` and ``"contact length"``
    branches, which cells are counted and which of their neighbors are
    counted. The default ``'all'`` / ``'all'`` reproduces the original
    behavior (total neighbors of every valid cell); ``cell_type='HC'`` with
    ``neighbor_type='HC'`` yields the number of HC neighbors for each HC.
    The experimental ``type`` column encodes HCs as ``1`` and SCs as ``0``.
    """
    res = []
    if type == "number of neighbors" or type == "contact length":
        for experiment in range(1, 4):
            contact_matrix_file_name = "E17_experiment%d_contact_matrix_frame_1.npy"%experiment if stage=="E17.5"\
                else "P0_experiment%d_contact_matrix_frame_1.npy"%experiment
            cells_info_file_name = "E17_experiment%d_cells_info_frame_1"%experiment if stage=="E17.5"\
                else "P0_experiment%d_cells_info_frame_1"%experiment
            contact_matrix = np.load(os.path.join(experimental_results_folder, stage,contact_matrix_file_name))
            cells_info = pd.read_pickle(os.path.join(experimental_results_folder, stage,cells_info_file_name))
            valid_cells = cells_info.valid.values
            is_HC = cells_info.type.values == 1

            def _mask(which):
                if which == "HC":
                    return valid_cells & is_HC
                elif which == "SC":
                    return valid_cells & ~is_HC
                return valid_cells

            row_mask = _mask(cell_type)
            col_mask = _mask(neighbor_type)
            valid_contacts = contact_matrix[np.ix_(row_mask, col_mask)]
            if type == "number of neighbors":
                valid_contacts = (valid_contacts > 0).astype(int)
            res.append(np.sum(valid_contacts, axis=1))
        return res
    elif type == "HC to SC area ratio":
        # Per experiment (one replicate each), each HC's area divided by the
        # mean SC area at the +24h frame. A ratio, so unit-free and directly
        # comparable to the model.
        for experiment in range(1, 4):
            cells_info = pd.read_pickle(_experimental_late_cells_info_path(stage, experiment))
            valid = cells_info.valid.values
            is_HC = cells_info.type.values == 1
            area = cells_info.area.values
            res.append(area[valid & is_HC] / np.average(area[valid & ~is_HC]))
        return res
    elif type == "HC to SC roundness ratio":
        # Uses the precomputed +24h roundness arrays (same ones the existing
        # "HC roundness"/"SC roundness" comparisons use): each HC roundness
        # divided by the mean SC roundness.
        hc_file, sc_file = (E17_HC_roundness_file_name, E17_SC_roundness_file_name) if stage == "E17.5" \
            else (P0_HC_roundness_file_name, P0_SC_roundness_file_name)
        hc_roundness = np.load(os.path.join(experimental_results_folder, hc_file))
        sc_roundness = np.load(os.path.join(experimental_results_folder, sc_file))
        return [hc_roundness / np.average(sc_roundness)]
    elif type == "HC to SC area change ratio after ablation":
        # From the ablation-summary ODS: each ablation-adjacent HC's
        # area-change ratio divided by the mean SC area-change ratio.
        hc_ratios, sc_ratios = _read_ablation_area_ratios(stage)
        return [hc_ratios / np.average(sc_ratios)]
    elif stage == "E17.5":
        if type == "HC number of HC neighbors":
            return [np.load(os.path.join(experimental_results_folder, E17_number_of_HC_neighbors_file_name)).astype(int)]
        elif type == "HC contact length with HC":
            return [np.load(os.path.join(experimental_results_folder, E17_contact_length_with_HC_neighbors_file_name))]
        elif type == "HC roundness":
            return [np.load(os.path.join(experimental_results_folder, E17_HC_roundness_file_name))]
        elif type == "SC roundness":
            return [np.load(os.path.join(experimental_results_folder, E17_SC_roundness_file_name))]
        else:
            raise "Not implemented for type %s"%type
    elif stage == "P0":
        if type == "HC number of HC neighbors":
            return [np.load(os.path.join(experimental_results_folder, P0_number_of_HC_neighbors_file_name)).astype(int)]
        elif type == "HC contact length with HC":
            return [np.load(os.path.join(experimental_results_folder, P0_contact_length_with_HC_neighbors_file_name))]
        elif type == "HC roundness":
            return [np.load(os.path.join(experimental_results_folder, P0_HC_roundness_file_name))]
        elif type == "SC roundness":
            return [np.load(os.path.join(experimental_results_folder, P0_SC_roundness_file_name))]
        else:
            raise "Not implemented for type %s"%type
    else:
        raise "Not implemented for stage %s"%stage
def calc_normalized_hist(dist, continues, maximal_n=None):
    norm_hist = []
    min_dist = np.min(np.hstack(dist))
    max_dist = np.max(np.hstack(dist))
    for repeat in dist:
        if continues:
            kde = gaussian_kde(repeat.reshape(1, -1))
            grid = np.linspace(min_dist, max_dist, 100)
            hist = kde(grid)
            # Normalize
            hist /= hist.sum()
        else:
            if maximal_n is not None:
                repeat = np.clip(repeat, a_min=None, a_max=maximal_n)
            else:
                maximal_n = max(np.max(repeat), np.max(repeat))
            hist = np.bincount(repeat, minlength=(min(maximal_n, max_dist) + 1)) / repeat.size
        norm_hist.append(hist)
    return np.array(norm_hist)


def _align_histograms(norm_hists):
    """Right-pad every ``(n_repeats, n_bins)`` array in ``norm_hists``
    with trailing zero bins so they all share a common column count.

    Both ``calc_vectorial_distance`` and ``plot_distributions`` need
    to compare normalized histograms whose bin counts can differ
    (the discrete branch of ``calc_normalized_hist`` uses
    ``minlength = min(maximal_n, max_dist) + 1``, which depends on
    the dataset). Pulled out into a helper so the padding logic
    lives in exactly one place."""
    n_bins = max(nh.shape[1] for nh in norm_hists)
    aligned = []
    for nh in norm_hists:
        if nh.shape[1] < n_bins:
            padded = np.zeros((nh.shape[0], n_bins))
            padded[:, :nh.shape[1]] = nh
            nh = padded
        aligned.append(nh)
    return aligned


def plot_distributions(distribution_lists, continues=False, maximal_n=None,
                       labels=None, colors=None, edge_colors=None,
                       ax=None, xlabel=None, ylabel="Frequency",
                       title=None):
    """Grouped bar chart comparing K lists of distribution samples.

    For each input group ``distribution_lists[k]`` (a list of 1D
    arrays — one per repeat) we compute the normalized histogram of
    every repeat via :func:`calc_normalized_hist`, then average
    across repeats per bin. One bar per ``(bin, group)`` is drawn
    at height = mean frequency, with the K group-bars at each bin
    centred around the bin's integer position. The individual
    per-repeat frequencies are scattered on top of each bar so the
    spread within each group is visible. When a group carries more
    than 2 repeats an error bar showing the standard error of the
    mean (``std(ddof=1) / sqrt(n_repeats)``) is added — matching
    the "n > 2" SEM gate already used by
    :func:`calc_vectorial_distance`.

    Parameters
    ----------
    distribution_lists : list of list-of-arrays
        Outer list: K groups to compare against each other. Inner
        list: one 1D ndarray of samples per repeat of that group.
    continues : bool
        Forwarded to :func:`calc_normalized_hist`. ``False`` →
        discrete bincount; ``True`` → 100-point KDE.
    maximal_n : int, optional
        Discrete-mode value cap forwarded to
        :func:`calc_normalized_hist`.
    labels : list of str, optional
        Legend label per group. Length must match
        ``distribution_lists``.
    colors, edge_colors : list, optional
        Matplotlib face / edge color per group.
    ax : matplotlib axes, optional
        Target axes. A fresh figure is created when ``None``.
    xlabel, ylabel, title : str, optional
        Axis labels and title.

    Returns
    -------
    fig, ax
    """
    n_groups = len(distribution_lists)
    if n_groups == 0:
        raise ValueError("distribution_lists must contain at least one group")

    aligned = _align_histograms(
        [calc_normalized_hist(dl, continues, maximal_n)
         for dl in distribution_lists]
    )
    n_bins = aligned[0].shape[1]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # K bars per bin, centred around the integer bin index, with a
    # 0.8 total span so adjacent bins keep a visible gap.
    bar_width = 0.8 / n_groups
    bin_indices = np.arange(n_bins)

    for k, nh in enumerate(aligned):
        offset = (k - (n_groups - 1) / 2) * bar_width
        x = bin_indices + offset

        means = nh.mean(axis=0)
        # SEM gate: identical to the one calc_vectorial_distance
        # uses when forming the chi-square denominator. With <= 2
        # repeats the sample variance is too noisy to be useful.
        if nh.shape[0] > 2:
            sem = nh.std(axis=0, ddof=1) / np.sqrt(nh.shape[0])
        else:
            sem = None

        bar_kwargs = {"width": bar_width}
        if colors is not None:
            bar_kwargs["color"] = colors[k]
        if edge_colors is not None:
            bar_kwargs["edgecolor"] = edge_colors[k]
        if labels is not None:
            bar_kwargs["label"] = labels[k]
        if sem is not None:
            bar_kwargs["yerr"] = sem
            bar_kwargs["capsize"] = 3
        ax.bar(x, means, **bar_kwargs)

        # Scatter the per-repeat frequencies on top of each bar.
        # ``zorder=3`` puts them above the bar faces; the per-bin
        # x positions stack repeats vertically at the same x so
        # the spread within the group is obvious.
        for repeat in nh:
            ax.scatter(x, repeat, color="black", s=12,
                       zorder=3, alpha=0.7)

    if labels is not None:
        ax.legend()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def calc_vectorial_distance(dist1, dist2, maximal_n=None, continues=False,
                            plot=False, **plot_kwargs):
    norm_hist1, norm_hist2 = _align_histograms([
        calc_normalized_hist(dist1, continues, maximal_n),
        calc_normalized_hist(dist2, continues, maximal_n),
    ])
    averages1 = np.average(norm_hist1, axis=0)
    averages2 = np.average(norm_hist2, axis=0)
    chi_sqr = (averages1 - averages2)**2
    sem1_sqr = np.var(norm_hist1, axis=0)/norm_hist1.shape[0] if norm_hist1.shape[0] > 2 else np.zeros_like(averages1)
    sem2_sqr = np.var(norm_hist2, axis=0)/norm_hist2.shape[0] if norm_hist2.shape[0] > 2 else np.zeros_like(averages2)
    sem1_sqr[chi_sqr==0] = 1 # Avoiding 0/0
    reduced_chi_sqr = np.sum(chi_sqr/(sem1_sqr + sem2_sqr))
    if plot:
        # ``plot_kwargs`` lets callers (e.g.
        # ``compare_to_experimental_results``) forward labels /
        # colors / a target ``ax`` without burying that knowledge
        # in the distance computation itself.
        plot_distributions([dist1, dist2], continues=continues,
                           maximal_n=maximal_n, **plot_kwargs)
    return reduced_chi_sqr

def calc_pval(dist1, dist2, maximal_n=None, continues=False, plot=False,
              **plot_kwargs):
    from statistical_analysis import HierarchicalTwoSamplesCompare
    if maximal_n is not None:
        for i in range(len(dist1)):
            np.clip(dist1[i], a_min=None, a_max=maximal_n, out=dist1[i])
        for i in range(len(dist2)):
            np.clip(dist2[i], a_min=None, a_max=maximal_n, out=dist2[i])
    comparer = HierarchicalTwoSamplesCompare(dist1, dist2, continues=continues)
    pval = comparer.compare_samples()
    if plot:
        # NOTE: dist1 / dist2 have already been clipped above when
        # maximal_n was given, so the bars reflect exactly what
        # ``HierarchicalTwoSamplesCompare`` was handed.
        plot_distributions([dist1, dist2], continues=continues,
                           maximal_n=maximal_n, **plot_kwargs)
    return pval

def compare_to_experimental_results(history, model_name, experimental_stage, results_type="HC number of HC neighbors",
                                    type_by='atoh_level', threshold=None,
                                    max_number_of_neighbors=2, plot=False, time_point=-1, ablated_cells=[], post_ablation_frame=None):
    # Right now it is implemented only for the number of HC neighbors of HCs and roundness
    experimental_results = load_experimental_results(experimental_stage, results_type)
    if experimental_stage == "E17.5":
        color = "cyan"
        edge_color = "blue"
    elif experimental_stage == "P0":
        color = "pink"
        edge_color = "red"
    if results_type == "HC number of HC neighbors" or results_type == "number of neighbors":
        cell_type = "HC" if results_type == "HC number of HC neighbors" else "all"
        model_results, _ = calc_contacts_for_time_point(history, model_name, cell_type=cell_type, neighbor_type=cell_type,
                                              type_by=type_by, threshold=threshold, time_point=time_point)
        continues = False

    elif results_type == "HC roundness":
        model_results = calc_roundness_for_time_point(history, model_name, cell_type="HC",
                                                          type_by=type_by, threshold=threshold, time_point=time_point)
        continues = True
    elif results_type == "SC roundness":
        model_results = calc_roundness_for_time_point(history, model_name, cell_type="SC",
                                                          type_by=type_by, threshold=threshold, time_point=time_point)
        continues = True
    elif results_type == "contact length":
        _, model_results = calc_contacts_for_time_point(history, model_name, cell_type="all", neighbor_type="all", time_point=time_point)
    elif results_type == "HC area change after ablation":
        model_results, _ = calc_area_change_after_ablation(history, model_name, ablated_cells=ablated_cells,
                                                        end_frame=post_ablation_frame, type_by=type_by,
                                                        threshold=threshold)
    elif results_type == "SC area change after ablation":
        _, model_results = calc_area_change_after_ablation(history, model_name, ablated_cells=ablated_cells,
                                                        end_frame=post_ablation_frame, type_by=type_by,
                                                        threshold=threshold)
    elif results_type == "HC to SC area change ratio after ablation":
        HC_change, SC_change = calc_area_change_after_ablation(history, model_name, ablated_cells=ablated_cells,
                                                        end_frame=post_ablation_frame, type_by=type_by,
                                                        threshold=threshold)
        model_results = HC_change/np.average(SC_change)
    else:
        raise "Not implemented for stage %s"%experimental_stage
    print("Experimental average = %f\nModel average = %f"%(np.average(np.hstack(experimental_results)), np.average(model_results)))

    # The keyword used to be ``continous=`` here, which doesn't match
    # ``calc_vectorial_distance``'s ``continues=`` parameter — a
    # ``TypeError`` waiting to happen the moment plotting got far
    # enough for the comparison to run.
    return calc_pval([model_results], experimental_results, maximal_n=max_number_of_neighbors,
                                   continues=continues, plot=plot)

def find_best_timepoint_according_to_n_neighbors(model_name):
    history = load_history_file(model_name)
    pval_E17 = []
    pval_P0 = []
    time = get_time_points(history)
    for time_point in time:
        try:
            pval_E17.append(compare_to_experimental_results(history, model_name,"E17.5", results_type="number of neighbors",
                                                            type_by='atoh_level', threshold=None,
                                                            max_number_of_neighbors=9, plot=False,
                                                            time_point=time_point))
            pval_P0.append(compare_to_experimental_results(history, model_name, "P0", results_type="number of neighbors",
                                                           type_by='atoh_level', threshold=None,
                                                           max_number_of_neighbors=9, plot=False, time_point=time_point))
            print("E17 pval:%f, P0 pval:%f" % (pval_E17[-1], pval_P0[-1]))
        except ValueError:
            break
    best_E17_idx = np.argmax(pval_E17)
    best_P0_idx = np.argmax(pval_P0)
    fig, ax = plt.subplots()
    print("Best fit for E17.5 is time=%f with pval=%f" % (time[best_E17_idx], pval_E17[best_E17_idx]))
    print("Best fit for P0 is time=%f with pval=%f" % (time[best_P0_idx], pval_P0[best_P0_idx]))
    ax.plot(time, pval_E17, label="E17.5")
    ax.plot(time, pval_P0, label="P0")
    ax.legend()
    return fig, ax, time[best_E17_idx], time[best_P0_idx]

def compare_differentiation_to_experiments(model_name, experimental_stage="E17.5",
                                           type_by='atoh_level', threshold=None,
                                           max_number_of_neighbors=2, plot=False):
    """Match a simulation to the experimental tissue by HC arrangement, then
    compare differentiation-time HC neighborhoods.

    Pipeline:

    1. Load the experimental *number of HC neighbors per HC* at frame 1 for
       all 3 biological repeats (via the extended ``"number of neighbors"``
       branch of :func:`load_experimental_results` with
       ``cell_type=neighbor_type='HC'``).
    2. For every simulation frame, build the same distribution (HC neighbors
       per HC) and compute the p value against the 3 experimental repeats
       with :func:`calc_pval`. The frame with the largest p value is the
       best-matching frame; its time stamp becomes the ``initial_time``.
    3. Trace differentiation from ``initial_time`` with
       :func:`calc_HC_neighbors_at_differentiation` to get the simulation
       distribution of HC neighbors at differentiation.
    4. Compare that distribution to the experimental *differentiating cells
       number of HC neighbors* (the ``"HC number of HC neighbors"`` branch)
       and return the resulting p value.

    Parameters
    ----------
    model_name : str
        Results sub-folder holding ``history.hf5``.
    experimental_stage : str
        ``"E17.5"`` or ``"P0"``.
    type_by, threshold, HC_above_threshold
        Forwarded to the HC-identity helpers. ``threshold=None`` lets each
        step pick its own mid-range threshold (per frame for matching, from
        the final frame for the differentiation trace).
    max_number_of_neighbors : int
        Neighbor-count cap forwarded to :func:`calc_pval` (the experimental
        distributions top out at 2-3 HC neighbors).
    plot : bool
        Plot the final differentiation-distribution comparison.

    Returns
    -------
    pval : float
        p value of the differentiation comparison.
    initial_time : float
        Time stamp of the best-matching frame.
    sim_differentiation : numpy.ndarray
        Simulation HC-neighbors-at-differentiation distribution.
    """
    history = load_history_file(model_name)
    time_points = get_time_points(history)

    # A single fixed threshold (from the final frame, when not supplied) so
    # "HC" means the same thing in the frame-matching sweep and in the
    # differentiation trace below — and so the per-frame mid-range threshold
    # doesn't relabel ~half of every early frame as HCs.
    if threshold is None:
        final_sheet = history.retrieve(time_points[-1])
        final_sheet.arrange_sheet_from_history()
        non_boundary = find_non_boundary_cells(final_sheet)
        type_data = final_sheet.face_df.loc[non_boundary, type_by]
        threshold = (np.max(type_data) + np.min(type_data)) / 2
        print("Using fixed threshold = %f" % threshold)

    # (1) Experimental per-HC HC-neighbor counts at frame 1 (3 repeats).
    experimental_frame1 = load_experimental_results(
        experimental_stage, "number of neighbors",
        cell_type="HC", neighbor_type="HC")

    # (2) Per-frame match: p value of simulation HC-HC neighbor distribution
    #     against the experimental frame-1 repeats.
    pvals = []
    valid_times = []
    for time_point in time_points:
        sheet = history.retrieve(time_point)
        sheet.arrange_sheet_from_history()
        try:
            number_of_HC_neighbors, _ = calc_contact_with_neighbors_from_type(
                sheet, cell_type="HC", neighbor_type="HC",
                type_by=type_by, threshold=threshold)
        except Exception:
            # Frames with no/degenerate HC population can't form a
            # distribution; skip them rather than aborting the sweep.
            continue
        if number_of_HC_neighbors.size == 0:
            continue
        pval = calc_pval([number_of_HC_neighbors], experimental_frame1,
                         maximal_n=max_number_of_neighbors, continues=False)
        pvals.append(pval)
        valid_times.append(time_point)
    if not pvals:
        raise ValueError("No simulation frame produced a HC neighbor distribution")
    pvals = np.array(pvals)
    valid_times = np.array(valid_times)
    best_idx = int(np.argmax(pvals))
    initial_time = float(valid_times[best_idx])
    print("Best matching frame: t=%f (HC-arrangement pval=%f)" % (initial_time, pvals[best_idx]))

    # (3) Differentiation distribution from the best-matching frame onward.
    sim_differentiation = calc_HC_neighbors_at_differentiation(
        history, initial_time_point=initial_time,
        type_by=type_by, threshold=threshold)

    # (4) Compare to the experimental differentiating-cells distribution.
    experimental_differentiation = load_experimental_results(
        experimental_stage, "HC number of HC neighbors")
    pval = calc_pval([sim_differentiation], experimental_differentiation,
                     maximal_n=max_number_of_neighbors, continues=False, plot=plot)
    print("Differentiation comparison p value = %f" % pval)
    return pval, initial_time, sim_differentiation


def find_best_timepoint_for_random_arrays(model_names=None, indices=None):
    """Find each raw random array's best-matching E17.5 / P0 time point.

    Pass explicit ``model_names`` or, equivalently, the integer ``indices`` of
    the arrays (``indices=[7]`` addresses ``random_periodic_array7``) — the
    latter is what an Azure Batch fan-out task uses to handle one array.
    """
    if model_names is None:
        if indices is None:
            raise ValueError("Provide either model_names or indices")
        model_names = [random_array_name(i) for i in indices]
    out_file = os.path.join("results", "pval_results.txt")
    for model_name in model_names:
        fig, ax, bestE17, bestP0 = find_best_timepoint_according_to_n_neighbors(model_name)
        plt.savefig(os.path.join("results", model_name + "pval_results.png"))
        plt.close(fig)
        with open(out_file, "a") as f:
            f.write("Best time point for " + model_name + " E17.5: %f, P0: %f\n" % (bestE17, bestP0))

def store_best_time_point_for_random_arrays(only_for=None):
    # ``only_for`` may be a model name or an integer array index.
    if isinstance(only_for, int):
        only_for = random_array_name(only_for)
    time_point_file = os.path.join("results", "pval_results.txt")
    model_names = []
    E17_times = []
    P0_times = []
    with open(time_point_file, "r") as f:
        for line in f:
            if "Best time point for" in line:
                line = line.replace("Best time point for ", "")
                line = line.replace("E17.5:", "")
                line = line.replace("P0:", "")
                line = line.replace(",", "")
                line = line.replace("\n", "")
                splitted = line.split()
                model_names.append(splitted[0])
                E17_times.append(float(splitted[1]))
                P0_times.append(float(splitted[2]))
    for model_name, E17_time, P0_time in zip(model_names, E17_times, P0_times):
        if only_for is None or only_for == model_name:
            history = load_history_file(model_name)
            output_name = model_name + _STAGE_SHEET_SUFFIX["E17.5"]
            save_data_of_a_given_time_point(history, E17_time, output_name)
            output_name = model_name + _STAGE_SHEET_SUFFIX["P0"]
            save_data_of_a_given_time_point(history, P0_time, output_name)
    return 0

def compare_model_mechanics_to_experiments(model_name, experimental_stage,
                                           type_by='atoh_level', threshold=None,
                                           time_point=-1, plot=False,
                                           ablation_model_name=None, ablated_cells=[],
                                           post_ablation_frame=None):
    """Compare a model's HC/SC mechanics to experiments via the product of
    three p values:

    1. **HC area / mean SC area** — relative HC size.
    2. **HC roundness / mean SC roundness** — relative HC shape.
    3. **HC / mean SC area change after ablation** — relative HC mechanical
       response to ablation.

    Each model distribution (taken from frame ``time_point``, the last frame
    by default) is compared to the matching experimental distribution with
    :func:`calc_pval` (continuous data), and the three p values are
    multiplied into a single combined score. All three quantities are
    ratios, so they are dimensionless and comparable across model and
    experiment despite different absolute units.

    The ablation comparison (3) needs a before/after-ablation run, which a
    normal differentiation simulation is not. Provide it via
    ``ablation_model_name`` (a results folder holding an ablation
    ``history.hf5``), ``ablated_cells`` and ``post_ablation_frame``. When
    these are omitted the ablation term is skipped and contributes a neutral
    ``p = 1.0`` to the product.

    Returns
    -------
    combined_pval : float
        ``p_area * p_roundness * p_ablation``.
    pvals : dict
        The individual ``{"area", "roundness", "ablation"}`` p values.
    """
    history = load_history_file(model_name)
    if time_point == -1:
        time_point = np.max(history.time_stamps)
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history()

    # (1) HC area / mean SC area.
    model_hc_area = calc_area_for_type(sheet, cell_type="HC", type_by=type_by, threshold=threshold)
    model_sc_area = calc_area_for_type(sheet, cell_type="SC", type_by=type_by, threshold=threshold)
    model_area_ratio = model_hc_area / np.average(model_sc_area)
    experimental_area_ratio = load_experimental_results(experimental_stage, "HC to SC area ratio")
    p_area = calc_pval([model_area_ratio], experimental_area_ratio, continues=True, plot=plot)

    # (2) HC roundness / mean SC roundness.
    model_hc_roundness = calc_roundness_for_type(sheet, cell_type="HC", type_by=type_by, threshold=threshold)
    model_sc_roundness = calc_roundness_for_type(sheet, cell_type="SC", type_by=type_by, threshold=threshold)
    model_roundness_ratio = model_hc_roundness / np.average(model_sc_roundness)
    experimental_roundness_ratio = load_experimental_results(experimental_stage, "HC to SC roundness ratio")
    p_roundness = calc_pval([model_roundness_ratio], experimental_roundness_ratio, continues=True, plot=plot)

    # (3) HC / mean SC area change after ablation (only if an ablation run is given).
    if ablation_model_name is not None and post_ablation_frame is not None and len(ablated_cells) > 0:
        ablation_history = load_history_file(ablation_model_name)
        hc_change, sc_change = calc_area_change_after_ablation(
            ablation_history, ablation_model_name, ablated_cells=ablated_cells,
            end_frame=post_ablation_frame, type_by=type_by, threshold=threshold)
        model_ablation_ratio = hc_change / np.average(sc_change)
        experimental_ablation_ratio = load_experimental_results(
            experimental_stage, "HC to SC area change ratio after ablation")
        p_ablation = calc_pval([model_ablation_ratio], experimental_ablation_ratio, continues=True, plot=plot)
    else:
        print("No ablation run provided (ablation_model_name/ablated_cells/post_ablation_frame); "
              "skipping ablation comparison (p=1.0).")
        p_ablation = 1.0

    combined_pval = p_area * p_roundness * p_ablation
    print("p(area ratio)=%g, p(roundness ratio)=%g, p(ablation ratio)=%g  =>  product=%g"
          % (p_area, p_roundness, p_ablation, combined_pval))
    return combined_pval, {"area": p_area, "roundness": p_roundness, "ablation": p_ablation}




if __name__ == "__main__":
#     find_best_timepoint_for_random_arrays(["random_periodic_array0"])
#     store_best_time_point_for_random_arrays()
    for name in ["periodic_fromrandom_periodic_array%d_for_E17_gammaSC-0.01_gammaHC_ratio-10.00_alphaHC_ratio-1.00_psigma-0.00"% i for i in range (1,10)]:
        redraw(name,"movie", movie=True,
               color_by="delta")