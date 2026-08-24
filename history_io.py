"""Reading, trimming and re-writing simulation history archives.

A tyssue ``HistoryHdf5`` archive is append-only and can reach several GB, so the
helpers here avoid loading more than they need: opening an archive reads only its
time column, and extracting a frame copies that frame rather than rewriting the
file. The repair helpers exist because a run killed mid-write leaves a partially
flushed final frame, which reads back as a valid but truncated sheet.
"""

import hashlib
import os, sys
import shutil
import tempfile
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tyssue import History, HistoryHdf5
from tyssue.draw.plt_draw import sheet_view
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel

# ImageMagick's ``convert`` silently fails to write when the output path grows
# long, and Windows' classic MAX_PATH is 260; this leaves a margin below it.
_MAX_GIF_PATH_LEN = 200

# Where run folders live: one sub-directory per run, each holding history.hf5 and
# its parameters. Simulation archives reach several GB, so this usually points at
# a data drive rather than the working copy.
RESULTS_DIR = os.environ.get("TISSUE_RESULTS_DIR",
                             os.path.join(os.getcwd(), "results"))


def load_history_file(load_name):
    load_path = os.path.join(RESULTS_DIR, load_name)
    history = HistoryHdf5.from_archive(os.path.join(load_path, "history.hf5"), eptm_class=VirtualSheet)
    return history

def get_time_points(history):
    return np.unique(history.time_stamps)

def random_array_name(index):
    """Folder name of the raw random array with the given integer ``index``."""
    return "random_periodic_array%d" % index

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

def create_gif_safe(history, output, num_frames=None, interval=None,
                    draw_func=None, margin=5, duration=100, **draw_kwds):
    """Drop-in replacement for :func:`tyssue.draw.plt_draw.create_gif` that
    works on Windows, plus the long-output-name shortening it always did.

    WHY NOT tyssue's create_gif. It ends with

        subprocess.run(["convert", (graph_dir / "movie_*.png").as_posix(), output])

    which fails on Windows in three compounding ways:

    1. ``convert`` is ambiguous. ``shutil.which("convert")`` finds ImageMagick,
       but ``subprocess`` launches via ``CreateProcess``, which searches
       System32 BEFORE PATH — so the process that actually runs is Windows'
       own ``convert.exe`` (the FAT->NTFS converter). It answers
       ``rc=4, "Invalid drive specification."``.
    2. There is no ``check=True``, so that failure is SILENT: create_gif
       returns normally and the caller believes a gif was written.
    3. ``movie_*.png`` is handed over unexpanded, relying on the callee to
       glob.

    Rather than paper over which ImageMagick binary to call (``magick`` vs
    ``convert``, v6 vs v7), assemble the gif with Pillow — already a
    matplotlib dependency, so this removes the external system dependency
    altogether and makes failures raise instead of vanish.

    The frame loop mirrors tyssue's (same bounds/margin handling, same
    ``history.browse`` semantics) so output is unchanged apart from the
    encoder.

    Parameters are as tyssue's, plus ``duration`` (ms per frame).
    """
    from PIL import Image

    output = _shorten_gif_output(output)
    if draw_func is None:
        draw_func = sheet_view

    graph_dir = tempfile.mkdtemp()
    try:
        x, y = coords = draw_kwds.get("coords", history.sheet.coords[:2])
        sheet0 = history.retrieve(0)
        bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
        delta = (bounds.loc["max"] - bounds.loc["min"]).max()
        pad = delta * margin / 100
        xlim = bounds.loc["min", x] - pad, bounds.loc["max", x] + pad
        ylim = bounds.loc["min", y] - pad, bounds.loc["max", y] + pad
        start, stop = (None, None) if interval is None else (interval[0], interval[1])

        frames = []
        for i, (_t, sheet) in enumerate(history.browse(start, stop, num_frames)):
            try:
                fig, ax = draw_func(sheet, **draw_kwds)
            except Exception as exc:            # one bad frame must not kill the gif
                print("dropped frame %d: %s" % (i, exc))
                continue
            if isinstance(ax, plt.Axes) and margin >= 0:
                ax.set(xlim=xlim, ylim=ylim)
            path = os.path.join(graph_dir, "movie_%04d.png" % i)
            fig.savefig(path)
            plt.close(fig)
            frames.append(path)

        if not frames:
            raise RuntimeError("no frames could be drawn for %s" % output)

        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                  for p in sorted(frames)]
        images[0].save(output, save_all=True, append_images=images[1:],
                       duration=duration, loop=0, optimize=False)
        for im in images:
            im.close()
    finally:
        shutil.rmtree(graph_dir, ignore_errors=True)
    return output

def redraw(load_name, save_name, movie=True, maximal_number_of_frames_to_save=100, color_by="atoh", maximal_level=1):

    load_path = os.path.join(RESULTS_DIR, load_name)
    history = HistoryHdf5.from_archive(os.path.join(load_path, "history.hf5"), eptm_class=VirtualSheet)
    initial_sheet = history.retrieve(0)
    last_time_point = np.max(history.time_stamps)
    number_of_time_points = np.unique(history.time_stamps).size
    final_sheet = history.retrieve(last_time_point)
    number_of_frames_to_save = min(number_of_time_points, maximal_number_of_frames_to_save)

    save_path = os.path.join(RESULTS_DIR, load_name, save_name)
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
    output_dir = os.path.join(RESULTS_DIR, output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    sheet = history.retrieve(time_point)
    # ``history.retrieve`` returns a raw VirtualSheet with the default
    # ``periodic=False`` and no ``Lx``/``Ly`` — the periodic metadata
    # sits unread in ``face_df['_periodic_flag']``. Without arranging
    # it the artifacts below (especially the segmentation image and
    # the drawn ``final.png``) take the NON-periodic code path, so
    # boundary-crossing faces are never unfolded/wrapped and the image
    # is scrambled at the periodic seam. ``arrange_sheet_from_history``
    # restores ``periodic`` / ``Lx`` / ``Ly`` and the proper indices,
    # exactly as ``load_sheet_from_file`` does.
    sheet.arrange_sheet_from_history()
    sheet.initiate_edge_order()
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

def save_li_levels_at_time_point(history, time_point, output_dir="."):
    """Extract the lateral-inhibition levels of every cell at ``time_point``
    and save them as three numpy arrays — ``notch_levels.npy``,
    ``delta_levels.npy`` and ``repressor_levels.npy`` — in ``output_dir``.

    Each array is indexed by ``unique_id``: entry ``i`` holds the value for the
    cell whose ``unique_id == i`` (the same convention as ``contact_matrix.npy``
    / ``labels.npy`` and the inverse of
    :meth:`InnerEarModel.load_li_levels_from_numpy`). The array length is
    ``max(unique_id) + 1``; any gap in ``unique_id`` left by a division /
    delamination is filled with ``NaN``.

    Parameters
    ----------
    history : :class:`HistoryHdf5` or :class:`tyssue.core.history.History`
        Loaded simulation history.
    time_point : float
        Time stamp to extract (snapped to the nearest recorded frame).
    output_dir : str
        Directory to write the three ``.npy`` files into (created if needed).

    Returns
    -------
    dict
        ``{"notch_level": ndarray, "delta_level": ndarray,
        "repressor_level": ndarray}`` — the three saved arrays.
    """
    sheet = history.retrieve(time_point)
    uids = sheet.face_df["unique_id"].to_numpy().astype(int)
    n_cells = int(uids.max()) + 1 if uids.size else 0

    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for column, file_name in (("notch_level", "notch_levels.npy"),
                              ("delta_level", "delta_levels.npy"),
                              ("repressor_level", "repressor_levels.npy")):
        # Scatter each cell's value to the slot of its unique_id — the inverse
        # of load_li_levels_from_numpy's ``values[uids]`` gather. Gaps in
        # unique_id stay NaN.
        arranged = np.full(n_cells, np.nan)
        arranged[uids] = sheet.face_df[column].to_numpy()
        np.save(os.path.join(output_dir, file_name), arranged)
        saved[column] = arranged
    return saved
