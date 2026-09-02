import hashlib
import os, sys
import shutil
import tempfile
import re
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tyssue import History, HistoryHdf5
from tyssue.draw.plt_draw import sheet_view
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
# Archive reading and single-frame measurement live in the reusable layer; they
# are re-exported here so existing callers keep working against post_processing.
from history_io import (RESULTS_DIR, load_history_file, get_time_points,
                        random_array_name, redraw, create_gif_safe,
                        drop_corrupted_snapshots, extract_time_point_to_new_history,
                        save_data_of_a_given_time_point, save_li_levels_at_time_point,
                        _shorten_gif_output, _MAX_GIF_PATH_LEN)
from cell_measures import (find_non_boundary_cells, find_maximal_level_final_frame,
                           get_non_boundary_cell_ids_from_type,
                           calc_contact_with_neighbors_from_type,
                           calc_roundness_for_type, calc_area_for_type,
                           calc_roundness_for_time_point, calc_contacts_for_time_point,
                           calc_HC_neighbors_at_differentiation,
                           calc_percentage_of_differentiating_by_initial_neighbors,
                           calc_area_change_after_ablation, _li_length_for_model)

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






# Folder holding the experimental measurements (npy / pickle / ODS). Override
# with the ``EXPERIMENTAL_DATA_DIR`` environment variable; defaults to the
# local checkout.
experimental_results_folder = os.environ.get(
    "EXPERIMENTAL_DATA_DIR",
    r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data")

# Root folder holding every simulation result folder (one sub-folder per run,
# each with its ``history.hf5`` etc.). Imported from history_io above rather than
# redefined: archives are large, so this usually points at a data drive, and two
# copies of the default would drift apart. Set TISSUE_RESULTS_DIR to override —
# this project uses D:\Kasirer\results.
# Circular-ablation raw data (initial radius 60 um, per-ablation final radius).
# Lives in a sibling "Raw Data" folder by default; also looked for INSIDE the
# experimental data dir, which is how it is staged on the VM. Override with
# CIRCULAR_ABLATION_FILE.
_CIRCULAR_ABLATION_NAME = "circular_ablation_raw_data(figure 3 +S4).xlsx"
CIRCULAR_ABLATION_FILE = os.environ.get(
    "CIRCULAR_ABLATION_FILE",
    os.path.join(os.path.dirname(experimental_results_folder), "Raw Data",
                 _CIRCULAR_ABLATION_NAME))
CIRCULAR_ABLATION_INITIAL_RADIUS = 60.0


def resolve_circular_ablation_file():
    """Path to the circular-ablation workbook, trying both known layouts.

    The default assumes a "Raw Data" folder SIBLING to the experimental data
    dir, which is the local layout. On the VM the workbook is staged INSIDE the
    experimental data dir instead, so that default resolves to a path that does
    not exist (e.g. /home/azureuser/Raw Data/... when the file is really in
    /home/azureuser/experimental_data/...).

    This resolution used to be inline inside load_experimental_results, so
    anything reading the workbook directly - like the step-5 stiffness ratio -
    silently got the unresolved default and failed. Both now go through here.

    Returns the first path that exists, else CIRCULAR_ABLATION_FILE so the
    caller's error names the configured location.
    """
    candidates = [CIRCULAR_ABLATION_FILE,
                  os.path.join(experimental_results_folder, _CIRCULAR_ABLATION_NAME)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return CIRCULAR_ABLATION_FILE

E17_number_of_HC_neighbors_file_name = r"E17.5 differentiating cells number of HC neighbors.npy"
E17_contact_length_with_HC_neighbors_file_name = r"E17.5 differentiating cells contacts length with HC.npy"
E17_HC_roundness_file_name = r"E17.5 +24h HC roundness.npy"
E17_SC_roundness_file_name = r"E17.5 +24h SC roundness.npy"
P0_number_of_HC_neighbors_file_name = r"P0 differentiating cells number of HC neighbors.npy"
P0_contact_length_with_HC_neighbors_file_name = r"P0 differentiating cells contact length with HC.npy"
P0_HC_roundness_file_name = r"P0 +24h HC roundness.npy"
P0_SC_roundness_file_name = r"P0 +24h SC roundness.npy"
ablation_area_change_file_name = r"ablation_area_change_summary.ods"
percentage_of_differentiating_cells_file_name = r"percentage_of_differentiating_cells.xlsx"


# Naming convention for the simulated random arrays, kept here (the dependency
# root; run_model imports from post_processing, not the other way round) so the
# integer-index <-> folder-name mapping lives in exactly ONE place. This is what
# lets a single Azure Batch task address one array by its index.
_STAGE_SHEET_SUFFIX = {"E17.5": "_for_E17", "P0": "_for_P0"}






def save_li_levels_from_best_pval_jsonl(jsonl_path, results_dir=RESULTS_DIR,
                                        overwrite=True, write_threshold=True):
    """Extract per-array data from a ``*_best_pval_per_array.jsonl`` file and
    write it into each matching model's results folder.

    Each non-empty line of ``jsonl_path`` is one JSON record carrying:

    - ``array_index`` (int) and ``dev_stage`` (``"E17"`` / ``"P0"``),
    - ``N_final`` / ``D_final`` / ``R_final`` — per-cell final notch / delta /
      repressor levels, where entry ``i`` is the value for the cell with
      ``unique_id == i`` (0-based),
    - ``D_threshold_mean`` — the scalar delta threshold to classify the
      array's cells into HC / SC.

    For a record with ``array_index = k`` and ``dev_stage = s`` the files::

        notch_levels.npy      <- N_final
        delta_levels.npy      <- D_final
        repressor_levels.npy  <- R_final
        threshold.npy         <- D_threshold_mean   (when write_threshold)

    are written into ``<results_dir>/random_periodic_array{k}_for_{s}/`` — the
    same folder that holds that model's ``history.hf5`` — so that ``run`` /
    ``find_mechanical_parameters`` pick them up automatically. ``threshold.npy``
    is a 0-d float array; ``find_mechanical_parameters(use_saved_threshold=True)``
    reads it as the per-array HC/SC classification threshold. Because it is a
    DELTA threshold it should be paired with ``type_by='delta_level'``.

    The per-cell list length is checked against the model's cell count
    (``max(unique_id) + 1``); a mismatch raises rather than writing
    misaligned data. A record whose target folder is missing is skipped with
    a message. Returns the list of folders written.
    """
    import json

    with open(jsonl_path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return _save_li_levels_records(records, results_dir=results_dir,
                                   overwrite=overwrite, write_threshold=write_threshold)


def _save_li_levels_records(records, results_dir=RESULTS_DIR, overwrite=True,
                            write_threshold=True):
    """Shared writer for the per-array LI extractors. Each record uses the OLD
    field names: ``array_index``, ``dev_stage``, ``N_final`` / ``D_final`` /
    ``R_final`` (per cell, entry ``i`` -> cell with ``unique_id == i``) and an
    optional ``D_threshold_mean``. Writes the four sidecar files into each
    ``<results_dir>/random_periodic_array{array_index}_for_{dev_stage}/`` folder
    (see :func:`save_li_levels_from_best_pval_jsonl`). Returns folders written."""
    written = []
    for rec in records:
        array_index = rec["array_index"]
        dev_stage = rec["dev_stage"]
        folder = os.path.join(
            results_dir, "%s_for_%s" % (random_array_name(array_index), dev_stage))
        if not os.path.isdir(folder):
            print("  [skip] %s: folder does not exist" % folder)
            continue

        channels = {
            "notch_levels.npy": np.asarray(rec["N_final"], dtype=float),
            "delta_levels.npy": np.asarray(rec["D_final"], dtype=float),
            "repressor_levels.npy": np.asarray(rec["R_final"], dtype=float),
        }
        lengths = {name: arr.shape[0] for name, arr in channels.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "array_index %s: N/D/R lengths differ: %s" % (array_index, lengths))
        n = next(iter(lengths.values()))
        expected = _li_length_for_model(folder)
        if n != expected:
            raise ValueError(
                "array_index %s (%s): per-cell list length %d != model cell "
                "count %d (max unique_id + 1). Entry i must be the value for "
                "the cell with unique_id i." % (array_index, folder, n, expected))

        for fname, arr in channels.items():
            path = os.path.join(folder, fname)
            if os.path.exists(path) and not overwrite:
                print("  [skip] %s exists (overwrite=False)" % path)
                continue
            np.save(path, arr)

        # Per-array HC/SC classification threshold (a single scalar). Saved
        # as a 0-d float array next to the LI levels; matched by name in
        # run_model._load_saved_threshold.
        thr = None
        if write_threshold and rec.get("D_threshold_mean") is not None:
            thr_path = os.path.join(folder, "threshold.npy")
            if not (os.path.exists(thr_path) and not overwrite):
                np.save(thr_path, np.asarray(float(rec["D_threshold_mean"])))
                thr = float(rec["D_threshold_mean"])

        written.append(folder)
        print("  [ok]   %s  (%d cells%s)" % (
            folder, n, "" if thr is None else ", threshold=%.4g" % thr))
    return written


def save_li_levels_from_best_row_json(json_path, results_dir=RESULTS_DIR,
                                      overwrite=True, write_threshold=True):
    """Extract per-array LI initial levels + delta threshold from a
    ``best_row_per_morphology.json`` file into each model's results folder.

    Same output as :func:`save_li_levels_from_best_pval_jsonl` (writes
    ``notch_levels.npy`` / ``delta_levels.npy`` / ``repressor_levels.npy`` keyed
    by ``unique_id`` plus ``threshold.npy``), but for the NEW file layout: a JSON
    LIST (not JSONL) of records, one per ``(dev_stage, array_id)``, whose per-cell
    channels are ``N_final`` / ``D_final`` / ``R_final`` and whose delta threshold
    is ``D_threshold`` (vs the old ``array_index`` / ``D_threshold_mean``). Extra
    fields (``pS``, ``pR``, ``repeat_idx``, ``chi2_*``) are ignored. The per-cell
    length is still checked against each model's cell count. Returns folders
    written."""
    import json
    with open(json_path) as fh:
        raw = json.load(fh)
    records = [{
        "array_index": rec["array_id"],
        "dev_stage": rec["dev_stage"],
        "N_final": rec["N_final"],
        "D_final": rec["D_final"],
        "R_final": rec["R_final"],
        "D_threshold_mean": rec.get("D_threshold"),
    } for rec in raw]
    return _save_li_levels_records(records, results_dir=results_dir,
                                   overwrite=overwrite, write_threshold=write_threshold)


def initial_morphology_name(index, stage):
    """Folder name of the fitted initial-morphology array: raw random array
    ``index`` snapped to ``stage``'s best-matching time point (the input handed
    to a differentiation run)."""
    return "%s%s" % (random_array_name(index), _STAGE_SHEET_SUFFIX[stage])


























def plot_percentage_of_differentiating(experimental_percentages, sim_percentages,
                                       stage="", ax=None):
    """Grouped bar chart of the % of differentiating cells by initial
    HC-neighbor count: experimental (mean over repeats, with the per-repeat
    values scattered on top) vs the simulation. ``experimental_percentages``
    is a list of per-repeat arrays; ``sim_percentages`` is one array. Both are
    indexed by bin (0, 1, ..., >=max)."""
    experimental = np.vstack(experimental_percentages)  # (n_repeats, n_bins)
    n_bins = experimental.shape[1]
    x = np.arange(n_bins)
    width = 0.35
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.bar(x - width / 2, experimental.mean(axis=0), width,
           color="cyan", edgecolor="blue", label="Experiment")
    for repeat in experimental:
        ax.scatter(x - width / 2, repeat, color="black", s=14, zorder=3, alpha=0.7)
    ax.bar(x + width / 2, sim_percentages[:n_bins], width,
           color="lightgray", edgecolor="black", label="Simulation")
    ax.scatter(x + width / 2, sim_percentages[:n_bins], color="black", s=14, zorder=3)

    labels = [str(n) for n in range(n_bins)]
    labels[-1] = ">=%d" % (n_bins - 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# HC neighbors at initial time")
    ax.set_ylabel("% differentiating")
    ax.set_title(("%s: " % stage if stage else "") + "differentiation by initial HC neighbors")
    ax.legend()
    return fig, ax




def _experimental_late_cells_info_path(stage, experiment, data_type_str="cells_info"):
    """Path to the *latest* (``+24h``) ``cells_info`` pickle for a given
    experiment, i.e. the recorded frame other than the initial ``frame_1``.
    Found by globbing so the late-frame number (191/199/120 for E17.5, …)
    doesn't have to be hard-coded per experiment."""
    prefix = "E17" if stage == "E17.5" else "P0"
    pattern = os.path.join(experimental_results_folder, stage,
                           "%s_experiment%d_%s_frame_*" % (prefix, experiment, data_type_str))
    best_path, best_frame = None, -1
    for path in glob.glob(pattern):
        match = re.search(r"frame_(\d+)", path)
        if match is None:
            continue
        frame = int(match.group(1))
        if frame != 1 and frame > best_frame:
            best_frame, best_path = frame, path
    if best_path is None:
        raise FileNotFoundError("No +24h cells_info frame found for %s experiment %d" % (stage, experiment))
    return best_path


def _read_ablation_area_ratios(stage):
    """Parse the ablation-summary ODS and return the per-cell *area after / area
    before* ablation ratios for ``stage``, GROUPED BY BIOLOGICAL REPEAT.

    A biological repeat is one ablation experiment, identified by its
    ``(Date, position)`` — a distinct utricle imaged on a given date at a given
    position. Returns a list of ``(hc_ratios, sc_ratios)`` NumPy-array pairs, one
    per repeat in first-seen order, so the HC/SC area-change ratio can be formed
    WITHIN each repeat (each HC normalized by its own repeat's mean SC change) and
    the repeats used as replicates in the hierarchical comparison — matching how
    the area/roundness metrics are grouped. (The previous version pooled every
    cell into one array, which mislabelled a whole stage as a single replicate.)

    Read by unzipping the ODS and walking ``content.xml`` directly so no optional
    ``odfpy`` dependency is required (it isn't installed in the ``tyssue``
    environment).
    """
    import zipfile
    from collections import OrderedDict
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
    date_i, stage_i, pos_i = col["Date"], col["stage"], col["position"]
    type_i, ratio_i = col["cell type"], col["area ratio"]
    need = max(date_i, stage_i, pos_i, type_i, ratio_i)

    # Group each cell's area ratio into its biological repeat (Date, position),
    # keeping HC and SC separate. OrderedDict preserves first-seen repeat order.
    repeats = OrderedDict()
    for cells in rows:
        if cells is header or len(cells) <= need or cells[stage_i] != stage:
            continue
        if cells[type_i] not in ("HC", "SC"):
            continue
        try:
            ratio = float(cells[ratio_i])
        except (ValueError, IndexError):
            continue
        hc, sc = repeats.setdefault((cells[date_i], cells[pos_i]), ([], []))
        (hc if cells[type_i] == "HC" else sc).append(ratio)
    return [(np.array(hc), np.array(sc)) for hc, sc in repeats.values()]


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
        # Per experiment (one replicate each), each HC's roundness divided by the
        # mean SC roundness at the +24h frame. Roundness is computed PER CELL
        # from that experiment's cells_info as 4*pi*area / perimeter**2 — the
        # same formula the model uses (VirtualSheet.get_face_roundness) — so it's
        # unit-free and directly comparable to the model. Mirrors the
        # "HC to SC area ratio" branch above.
        for experiment in range(1, 4):
            cells_info = pd.read_pickle(_experimental_late_cells_info_path(stage, experiment))
            valid = cells_info.valid.values
            is_HC = cells_info.type.values == 1
            roundness = 4 * np.pi * cells_info.area.values / cells_info.perimeter.values ** 2
            res.append(roundness[valid & is_HC] / np.average(roundness[valid & ~is_HC]))
        return res
    elif type == "HC roundness" or type == "SC roundness":
        # ABSOLUTE per-cell roundness (4*pi*area/perimeter**2) of the valid HC
        # (or SC) cells, one array per experiment — computed from each
        # experiment's +24h cells_info, the SAME source/formula as the ratio
        # branch above and the model (VirtualSheet.get_face_roundness). Roundness
        # is dimensionless, so unlike area the absolute value IS directly
        # comparable model<->experiment. (Replaces the old precomputed-npy path.)
        want_hc = type == "HC roundness"
        for experiment in range(1, 4):
            cells_info = pd.read_pickle(_experimental_late_cells_info_path(stage, experiment))
            valid = cells_info.valid.values
            is_HC = cells_info.type.values == 1
            roundness = 4 * np.pi * cells_info.area.values / cells_info.perimeter.values ** 2
            res.append(roundness[valid & (is_HC if want_hc else ~is_HC)])
        return res
    elif type == "lonely SC percentage":
        for experiment in range(1, 4):
            cells_info = pd.read_pickle(_experimental_late_cells_info_path(stage, experiment))
            contact_matrix = np.load(_experimental_late_cells_info_path(stage, experiment, "contact_matrix"))
            valid = cells_info.valid.values
            is_HC = cells_info.type.values == 1
            SC_mask = valid & ~is_HC
            HC_mask = valid & is_HC
            n_SC = np.sum(SC_mask.astype(int))
            valid_contacts = contact_matrix[np.ix_(SC_mask, HC_mask)]
            valid_contacts = (valid_contacts > 0).astype(int)
            n_HC_neighbors = np.sum(valid_contacts, axis=1)
            n_lonely_SC = np.sum((n_HC_neighbors == 0).astype(int))
            res.append(100*n_lonely_SC/n_SC)
        return res

    elif type == "HC to SC area change ratio after ablation":
        # From the ablation-summary ODS, GROUPED BY BIOLOGICAL REPEAT
        # ((Date, position)): within each repeat, each ablation-adjacent HC's
        # area-change ratio divided by that repeat's mean SC area-change ratio.
        # One array per repeat (the ablation experiments per stage), so the
        # repeats act as replicates in the hierarchical comparison, like the
        # area/roundness metrics. A repeat with no HC (or no SC) cells can't form
        # the ratio and is dropped.
        res = []
        for hc_ratios, sc_ratios in _read_ablation_area_ratios(stage):
            if len(hc_ratios) > 0 and len(sc_ratios) > 0:
                res.append(hc_ratios / np.average(sc_ratios))
        return res
    elif type == "HC area change after ablation" or type == "SC area change after ablation":
        # The ablation area-change (area after / area before) of the ablation-
        # adjacent HC (resp. SC) cells, one array per biological repeat
        # ((Date, position)) — so the repeats act as replicates in the comparison.
        # HC and SC are separate fit objectives (the old HC-over-SC ratio was
        # split). Note SC has cells in every repeat while HC may not (a repeat
        # with no cells of that type yields no array).
        want_hc = type == "HC area change after ablation"
        res = []
        for hc_ratios, sc_ratios in _read_ablation_area_ratios(stage):
            vals = hc_ratios if want_hc else sc_ratios
            if len(vals) > 0:
                res.append(vals)
        return res
    elif type == "cut shrinkage":
        # LINEAR shrinkage (%) of a cut disc, one value per ablation experiment
        # (each a biological replicate). Stage-specific: the two stages have
        # similar means but very different spreads, and the SEM of THIS stage's
        # repeats is what sets the tolerance on the fit term.
        table = pd.read_excel(resolve_circular_ablation_file(),
                              sheet_name="Overall data")
        rows = table[table["Stage"] == stage]
        final = rows["Final radius (um)"].to_numpy(dtype=float)
        r0 = CIRCULAR_ABLATION_INITIAL_RADIUS
        return [np.array([100.0 * (r0 - f) / r0]) for f in final]
    elif type == "percentage of differentiating cells":
        # One row per biological repeat (normal development) of the chosen
        # stage; each returned array is the % of cells that differentiated
        # binned by their initial HC-neighbor count (0, 1, ..., >=max). The
        # number of bins is fixed by the spreadsheet's columns.
        table = pd.read_excel(os.path.join(experimental_results_folder,
                                           percentage_of_differentiating_cells_file_name))
        rows = table[(table["Stage"] == stage) & (table["Condition"] == "Normal development")]
        percentage_columns = [c for c in table.columns if str(c).startswith("% differentiating")]
        return [row[percentage_columns].to_numpy(dtype=float) for _, row in rows.iterrows()]
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
                                                        end_time=post_ablation_frame, type_by=type_by,
                                                        threshold=threshold)
    elif results_type == "SC area change after ablation":
        _, model_results = calc_area_change_after_ablation(history, model_name, ablated_cells=ablated_cells,
                                                        end_time=post_ablation_frame, type_by=type_by,
                                                        threshold=threshold)
    elif results_type == "HC to SC area change ratio after ablation":
        HC_change, SC_change = calc_area_change_after_ablation(history, model_name, ablated_cells=ablated_cells,
                                                        end_time=post_ablation_frame, type_by=type_by,
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


def find_best_matching_frame(history, experimental_stage="E17.5",
                             type_by='atoh_level', threshold=None,
                             max_number_of_neighbors=2, plot=False,
                             save_plot_path=None):
    """Find the simulation frame whose per-HC HC-neighbor distribution best
    matches the experimental *frame-1* tissue.

    For every frame the per-HC HC-neighbor distribution is compared against the
    3 experimental frame-1 repeats with :func:`calc_pval`; the frame with the
    largest p value wins. A single fixed ``threshold`` (computed from the final
    frame when not given) labels HCs the same way in every frame, so the
    per-frame mid-range threshold doesn't relabel ~half of every early frame.

    Parameters
    ----------
    history : :class:`HistoryHdf5` or :class:`tyssue.core.history.History`
        Loaded simulation history.
    experimental_stage : str
        ``"E17.5"`` or ``"P0"``.
    type_by, threshold, max_number_of_neighbors
        As in :func:`compare_differentiation_to_experiments`.
    plot : bool
        Show the p-value-vs-time curve interactively.
    save_plot_path : str, optional
        When given, save the p-value-vs-time curve to this path (PNG) instead
        of showing it.

    Returns
    -------
    initial_time : float
        Time stamp of the best-matching frame.
    threshold : float
        The fixed HC threshold used (echoed back so downstream steps reuse it).
    """
    time_points = get_time_points(history)
    if threshold is None:
        final_sheet = history.retrieve(time_points[-1])
        final_sheet.arrange_sheet_from_history()
        non_boundary = find_non_boundary_cells(final_sheet)
        type_data = final_sheet.face_df.loc[non_boundary, type_by]
        threshold = (np.max(type_data) + np.min(type_data)) / 2
        print("Using fixed threshold = %f" % threshold)

    # Experimental per-HC HC-neighbor counts at frame 1 (3 repeats).
    experimental_frame1 = load_experimental_results(
        experimental_stage, "number of neighbors",
        cell_type="HC", neighbor_type="HC")

    # Per-frame match: p value of the simulation HC-HC neighbor distribution
    # against the experimental frame-1 repeats.
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
    if plot or save_plot_path is not None:
        fig1, ax1 = plt.subplots()
        ax1.plot(valid_times, pvals, "b.-")
        ax1.plot(valid_times[best_idx], pvals[best_idx], "*r", markersize=12,
                 label="best (t=%.2f)" % initial_time)
        ax1.set_yscale("log")
        ax1.set_ylabel("p value (sim vs experiment frame 1)")
        ax1.set_xlabel("Time (a.u.)")
        ax1.legend()
        if save_plot_path is not None:
            fig1.savefig(save_plot_path)
            plt.close(fig1)
            print("Saved pval-vs-time plot to %s" % save_plot_path)
        else:
            plt.show()
    return initial_time, threshold


def save_li_levels_at_best_matching_frame(model_names, experimental_stage="E17.5",
                                           type_by='atoh_level', threshold=None,
                                           max_number_of_neighbors=2, plot=False):
    """For each model in ``model_names``, find its best-matching frame (vs the
    experimental frame-1 tissue, via :func:`find_best_matching_frame`) and save
    that frame's notch / delta / repressor levels as ``notch_levels.npy`` /
    ``delta_levels.npy`` / ``repressor_levels.npy`` inside the model's
    ``results/<model_name>`` folder (via :func:`save_li_levels_at_time_point`).

    Parameters
    ----------
    model_names : sequence of str
        Result folder names, each holding a ``history.hf5``.
    experimental_stage, type_by, threshold, max_number_of_neighbors
        Forwarded to :func:`find_best_matching_frame`.
    plot : bool
        When True, also save the simulation-to-experiment p-value-vs-time curve
        as ``best_frame_pval_vs_time.png`` in each model's folder.

    Returns
    -------
    dict
        ``{model_name: best_matching_time}`` for every model processed.
    """
    best_times = {}
    for model_name in model_names:
        history = load_history_file(model_name)
        output_dir = os.path.join(RESULTS_DIR, model_name)
        save_plot_path = os.path.join(output_dir, "best_frame_pval_vs_time.png") if plot else None
        initial_time, _ = find_best_matching_frame(
            history, experimental_stage, type_by=type_by, threshold=threshold,
            max_number_of_neighbors=max_number_of_neighbors,
            save_plot_path=save_plot_path)
        save_li_levels_at_time_point(history, initial_time, output_dir=output_dir)
        best_times[model_name] = initial_time
        print("Saved notch/delta/repressor levels for %s at t=%f into %s"
              % (model_name, initial_time, output_dir))
    return best_times


def compare_differentiation_to_experiments(model_name, experimental_stage="E17.5",
                                           type_by='atoh_level', threshold=None,
                                           max_number_of_neighbors=2, plot=False):
    """Match a simulation to the experimental tissue by HC arrangement, then
    compare differentiation-time HC neighborhoods.

    Pipeline:

    1.+2. Find the best-matching frame (its time stamp becomes ``initial_time``)
       by comparing each simulation frame's per-HC HC-neighbor distribution to
       the experimental frame-1 tissue — delegated to
       :func:`find_best_matching_frame`.
    3. Trace differentiation from ``initial_time`` with
       :func:`calc_HC_neighbors_at_differentiation` to get the simulation
       distribution of HC neighbors at differentiation.
    4. Compare that distribution to the experimental *differentiating cells
       number of HC neighbors* (the ``"HC number of HC neighbors"`` branch)
       and return the resulting p value.
    5. Compute the percentage of cells that differentiate grouped by their
       HC-neighbor count at ``initial_time`` (via
       :func:`calc_percentage_of_differentiating_by_initial_neighbors`) and
       compare it to the experimental spreadsheet with
       ``statistical_analysis.TwoSampleCompare.compare_samples``.

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
        p value of the differentiation (HC-neighbors-at-differentiation) comparison.
    initial_time : float
        Time stamp of the best-matching frame.
    sim_differentiation : numpy.ndarray
        Simulation HC-neighbors-at-differentiation distribution.
    percentage_pval : float
        p value of the percentage-of-differentiating-cells comparison (step 5).
    sim_percentages : numpy.ndarray
        Simulation % differentiating per initial HC-neighbor bin (0..>=max).
    """
    history = load_history_file(model_name)

    # (1)+(2) Best-matching frame vs the experimental frame-1 tissue. The fixed
    # threshold it picks is reused below so "HC" means the same thing in the
    # differentiation trace and the percentage comparison.
    initial_time, threshold = find_best_matching_frame(
        history, experimental_stage, type_by=type_by, threshold=threshold,
        max_number_of_neighbors=max_number_of_neighbors, plot=plot)

    # (3) Differentiation distribution from the best-matching frame onward.
    sim_differentiation = calc_HC_neighbors_at_differentiation(
        history, initial_time_point=initial_time,
        type_by=type_by, threshold=threshold)

    # (4) Compare to the experimental differentiating-cells distribution.
    experimental_differentiation = load_experimental_results(
        experimental_stage, "HC number of HC neighbors")
    pval = calc_pval([sim_differentiation], experimental_differentiation,
                     maximal_n=max_number_of_neighbors, continues=False, plot=plot)
    if plot:
        plt.show()
    print("Differentiation comparison p value = %f" % pval)

    # (5) Percentage of differentiating cells grouped by their initial
    #     HC-neighbor count, vs the experimental spreadsheet.
    from statistical_analysis import TwoSampleCompare
    experimental_percentages = load_experimental_results(
        experimental_stage, "percentage of differentiating cells")
    # The spreadsheet's columns fix the binning (0, 1, ..., >=max); clump the
    # simulation to the same number of bins so the two are comparable.
    n_bins = experimental_percentages[0].size
    sim_percentages = calc_percentage_of_differentiating_by_initial_neighbors(
        history, initial_time, max_number_of_neighbors=n_bins - 1,
        type_by=type_by, threshold=threshold)
    # Pool experimental values (all repeats x all bins) and compare against the
    # simulation's per-bin percentages (dropping bins with no SCs).
    experimental_flat = np.concatenate(experimental_percentages)
    sim_flat = sim_percentages[~np.isnan(sim_percentages)]
    percentage_pval = TwoSampleCompare(experimental_flat, sim_flat,
                                       continues=True).compare_samples()
    print("Percentage-of-differentiating comparison p value = %f" % percentage_pval)
    if plot:
        plot_percentage_of_differentiating(experimental_percentages, sim_percentages,
                                           stage=experimental_stage)
        plt.show()

    return pval, initial_time, sim_differentiation, percentage_pval, sim_percentages


# --------------------------------------------------------------------------- #
# FULL-MODEL differentiation comparison: pools the 10 full-model runs and the  #
# 3 experiments into three n-sigma / chi^2 scores (see                         #
# compare_full_model_differentiation_to_experiments).                          #
# --------------------------------------------------------------------------- #
def _neighbor_pair_percentages(is_HC, binary_adjacency):
    """HC:HC, HC:SC and SC:SC neighbor-PAIR percentages (each pair counted once;
    the three sum to 100). ``binary_adjacency`` is the symmetric NxN neighbor
    matrix over the same N cells that the boolean ``is_HC`` indexes."""
    A = (np.asarray(binary_adjacency) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    hc = np.asarray(is_HC, bool)
    sc = ~hc
    total = A.sum() / 2.0
    if total <= 0:
        return np.nan, np.nan, np.nan
    hchc = A[np.ix_(hc, hc)].sum() / 2.0
    scsc = A[np.ix_(sc, sc)].sum() / 2.0
    hcsc = A[np.ix_(hc, sc)].sum()          # each HC-SC edge appears once here
    return 100.0 * hchc / total, 100.0 * hcsc / total, 100.0 * scsc / total


def _sim_neighbor_pair_percentages(sheet, type_by, threshold, HC_above_threshold=True):
    """(HC:HC%, HC:SC%, SC:SC%) among the non-boundary cells of a sim frame."""
    all_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="all", type_by=type_by, threshold=threshold,
        HC_above_threshold=HC_above_threshold)
    hc_idx, _ = get_non_boundary_cell_ids_from_type(
        sheet, cell_type="HC", type_by=type_by, threshold=threshold,
        HC_above_threshold=HC_above_threshold)
    sub = sheet.get_contact_matrix()[np.ix_(all_idx, all_idx)]
    return _neighbor_pair_percentages(np.isin(all_idx, hc_idx), sub)


def _exp_neighbor_pair_percentages(cells_info, contact_matrix):
    """(HC:HC%, HC:SC%, SC:SC%) among the valid experimental cells of a frame."""
    valid = cells_info.valid.values.astype(bool)
    is_HC = (cells_info.type.values == 1)[valid]
    A = np.asarray(contact_matrix)[np.ix_(valid, valid)]
    return _neighbor_pair_percentages(is_HC, A)


def _best_matching_frame_by_neighbor_pairs(history, target_hchc, target_hcsc,
                                           type_by, threshold, HC_above_threshold=True):
    """Sim time-point whose (HC:HC%, HC:SC%) neighbor-pair composition is closest
    (least squared distance) to the experimental target. Returns (time, hchc%, hcsc%)."""
    best = None
    for t in get_time_points(history):
        sheet = history.retrieve(t)
        sheet.arrange_sheet_from_history()
        hchc, hcsc, _ = _sim_neighbor_pair_percentages(
            sheet, type_by, threshold, HC_above_threshold)
        if np.isnan(hchc):
            continue
        d = (hchc - target_hchc) ** 2 + (hcsc - target_hcsc) ** 2
        if best is None or d < best[0]:
            best = (d, float(t), hchc, hcsc)
    if best is None:
        raise ValueError("no frame with a valid neighbor-pair composition")
    return best[1], best[2], best[3]


def _nsigma_and_chi2(sim_vals, exp_vals):
    """Signed n-sigma = (mean_sim - mean_exp) / sqrt(SEM_sim^2 + SEM_exp^2), and
    chi^2 = n-sigma^2. NaNs (undefined bins) are dropped from each group."""
    s = np.asarray(sim_vals, float); s = s[~np.isnan(s)]
    e = np.asarray(exp_vals, float); e = e[~np.isnan(e)]
    ms = float(np.mean(s)) if s.size else np.nan
    me = float(np.mean(e)) if e.size else np.nan
    sem_s = float(np.std(s, ddof=1) / np.sqrt(s.size)) if s.size > 1 else 0.0
    sem_e = float(np.std(e, ddof=1) / np.sqrt(e.size)) if e.size > 1 else 0.0
    denom = np.sqrt(sem_s ** 2 + sem_e ** 2)
    z = (ms - me) / denom if denom > 0 else np.nan
    return z, (z * z if np.isfinite(z) else np.nan), ms, me


def full_model_run_names(experimental_stage, indices=None, n_arrays=10,
                        run_prefix="fullmodel"):
    """Default full-model run names for a stage (``<run_prefix>_<array>``).

    ``run_prefix`` selects which set of full-model runs to read. It exists
    because the psigma=0 folder name encodes NO mechanics, so the pre-v2 runs
    (perimeter elasticity, bending, A0=0.4657) and the v2 runs (contractility,
    no bending, self-consistent A0) would otherwise share names. v2 runs are
    written under ``fullmodel_v2`` — see run_fitted_full_model.py."""
    prefix = "E17" if experimental_stage == "E17.5" else "P0"
    if indices is None:
        indices = range(n_arrays)
    return ["%s_random_periodic_array%d_for_%s" % (run_prefix, i, prefix)
            for i in indices]


def compare_full_model_differentiation_to_experiments(
        experimental_stage, model_names=None, type_by='delta_level',
        threshold=0.355079, max_number_of_neighbors=2, model_groups=None):
    """Compare the FULL-model runs (``model_names``, default the 10 ``fullmodel_*``
    runs of the stage) to experiment via three differentiation scores, each an
    n-sigma / chi^2 that POOLS the simulations and the experiments:

      1. Neighbor-pair composition at the best-matching initial frame:
         chi^2(HC:HC%) + chi^2(HC:SC%)  (SC:SC% is complementary -> ignored).
      2. HC neighbors AT DIFFERENTIATION as % of ALL differentiating cells:
         chi^2(%-with-0) + chi^2(%-with-1)  (>=2 complementary -> ignored).
      3. % of initial SCs that differentiate, grouped by initial HC-neighbor
         count: chi^2(0) + chi^2(1) + chi^2(>=2)  (independent ratios).

    Per bucket: n-sigma = (mean over the 10 sim %s - mean over the 3 exp %s)
    / sqrt(SEM_sim^2 + SEM_exp^2); chi^2 = n-sigma^2. HC/SC identity uses
    ``type_by`` above ``threshold`` (delta-level threshold 0.355079). The
    best-matching frame (score 1) sets each sim's initial time for scores 2/3.
    Prints the report and returns a dict of the three scores and the total."""
    if model_names is None:
        model_names = full_model_run_names(experimental_stage)
    prefix = "E17" if experimental_stage == "E17.5" else "P0"

    # ----- experimental values, per experiment -------------------------------
    e1_hchc, e1_hcsc = [], []                       # score 1: frame-1 pair %
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(
            experimental_results_folder, experimental_stage,
            "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(
            experimental_results_folder, experimental_stage,
            "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        hchc, hcsc, _ = _exp_neighbor_pair_percentages(ci, cm)
        e1_hchc.append(hchc); e1_hcsc.append(hcsc)

    e2_p0, e2_p1 = [], []                           # score 2: read per-exp counts
    for i in range(3):
        counts = np.asarray(np.load(os.path.join(
            experimental_results_folder,
            "%s differentiating cells_experiment%d.npy" % (experimental_stage, i))),
            dtype=float)
        if counts.size == 0:                        # no differentiating cells -> no %
            continue
        e2_p0.append(100.0 * np.mean(counts == 0))
        e2_p1.append(100.0 * np.mean(counts == 1))

    e3 = np.array(load_experimental_results(         # score 3: spreadsheet, per repeat
        experimental_stage, "percentage of differentiating cells"), dtype=float)

    # ----- simulation values, per model --------------------------------------
    # ``model_groups`` (list of lists of run names) averages REPEATS OF THE SAME
    # ARRAY into one data point before any statistic is taken. That matters
    # because SEM_sim sits in the n-sigma denominator: pooling 30 runs as 30
    # independent points would shrink it by ~sqrt(3) and inflate every score,
    # even though the extra runs only re-roll the lateral-inhibition seed. The
    # array-to-array spread is the variation that belongs in the SEM; the
    # seed-to-seed spread is noise to average out. Grouping keeps the point count
    # (and therefore the score scale) comparable to the single-repeat results.
    if model_groups is not None:
        model_names = [n for grp in model_groups for n in grp]
        owner = [g for g, grp in enumerate(model_groups) for _ in grp]
        n_groups = len(model_groups)
    else:
        owner = list(range(len(model_names)))
        n_groups = len(model_names)

    def _by_group(vals, owners):
        """Mean within each group; groups with no value are dropped.

        Values are carried WITH their owning group index rather than by
        position: the score-2 lists skip runs that produced no differentiating
        cells, so positional grouping would silently misalign them."""
        out = []
        for g in range(n_groups):
            v = [x for x, o in zip(vals, owners) if o == g]
            if v:
                out.append(np.mean(np.asarray(v, dtype=float), axis=0))
        return np.array(out, dtype=float)

    tgt_hchc = float(np.nanmean(e1_hchc)); tgt_hcsc = float(np.nanmean(e1_hcsc))
    s1_hchc, s1_hcsc, s2_p0, s2_p1, s3 = [], [], [], [], []
    o1, o2, o3 = [], [], []
    for name, g in zip(model_names, owner):
        history = load_history_file(name)
        t0, hchc, hcsc = _best_matching_frame_by_neighbor_pairs(
            history, tgt_hchc, tgt_hcsc, type_by, threshold)
        s1_hchc.append(hchc); s1_hcsc.append(hcsc); o1.append(g)
        counts = calc_HC_neighbors_at_differentiation(
            history, initial_time_point=t0, type_by=type_by, threshold=threshold)
        counts = np.asarray(counts, float); counts = counts[~np.isnan(counts)]
        if counts.size > 0:
            s2_p0.append(100.0 * np.mean(counts == 0))
            s2_p1.append(100.0 * np.mean(counts == 1))
            o2.append(g)
        s3.append(calc_percentage_of_differentiating_by_initial_neighbors(
            history, t0, max_number_of_neighbors=max_number_of_neighbors,
            type_by=type_by, threshold=threshold))
        o3.append(g)
    s1_hchc = _by_group(s1_hchc, o1); s1_hcsc = _by_group(s1_hcsc, o1)
    s2_p0 = _by_group(s2_p0, o2); s2_p1 = _by_group(s2_p1, o2)
    s3 = _by_group(s3, o3)                           # (n_groups, 3)

    # ----- scores + report ---------------------------------------------------
    def block(title, buckets):
        print("\n" + title)
        total = 0.0
        for label, sim_vals, exp_vals in buckets:
            z, chi2, ms, me = _nsigma_and_chi2(sim_vals, exp_vals)
            total += 0.0 if not np.isfinite(chi2) else chi2
            print("    %-9s  sim%%=%6.2f  exp%%=%6.2f   n-sigma=%+7.3f   chi^2=%8.3f"
                  % (label, ms, me, z, chi2))
        print("    -> score = %.3f" % total)
        return total

    print("=" * 80)
    print("FULL-MODEL DIFFERENTIATION vs EXPERIMENT   %s   (%d data point%s from %d "
          "run%s, 3 experiments)"
          % (experimental_stage, n_groups, "" if n_groups == 1 else "s",
             len(model_names), "" if len(model_names) == 1 else "s"))
    print("  n-sigma = (mean_sim%% - mean_exp%%) / sqrt(SEM_sim^2 + SEM_exp^2);  chi^2 = n-sigma^2")
    print("=" * 80)
    sc1 = block("1) neighbor pairs at best-matching initial frame:",
                [("HC:HC", s1_hchc, e1_hchc), ("HC:SC", s1_hcsc, e1_hcsc)])
    sc2 = block("2) HC neighbors at differentiation (% of all differentiating cells):",
                [("0 HC nb", s2_p0, e2_p0), ("1 HC nb", s2_p1, e2_p1)])
    sc3 = block("3) % of initial SCs differentiating, by initial HC-neighbor count:",
                [("0 HC nb", s3[:, 0], e3[:, 0]), ("1 HC nb", s3[:, 1], e3[:, 1]),
                 (">=2 HC nb", s3[:, 2], e3[:, 2])])
    total = sc1 + sc2 + sc3
    print("\n" + "=" * 80)
    print("  SCORE 1 = %.3f    SCORE 2 = %.3f    SCORE 3 = %.3f    TOTAL = %.3f"
          % (sc1, sc2, sc3, total))
    print("=" * 80)
    return {"score1": sc1, "score2": sc2, "score3": sc3, "total": total}


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
    out_file = os.path.join(RESULTS_DIR, "pval_results.txt")
    for model_name in model_names:
        fig, ax, bestE17, bestP0 = find_best_timepoint_according_to_n_neighbors(model_name)
        plt.savefig(os.path.join(RESULTS_DIR, model_name + "pval_results.png"))
        plt.close(fig)
        with open(out_file, "a") as f:
            f.write("Best time point for " + model_name + " E17.5: %f, P0: %f\n" % (bestE17, bestP0))

def store_best_time_point_for_random_arrays(only_for=None):
    # ``only_for`` may be a model name or an integer array index.
    if isinstance(only_for, int):
        only_for = random_array_name(only_for)
    time_point_file = os.path.join(RESULTS_DIR, "pval_results.txt")
    model_names = []
    E17_times = []
    P0_times = []
    with open(time_point_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
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


# Arrays whose stored best-matching frame is self-intersecting (folded). The
# E17.5 extracts of arrays 2/7/8 are already fold-free, so only their P0 frame
# needs replacing.
CORRUPTED_EXTRACTS = [
    ("random_periodic_array1", "E17.5"), ("random_periodic_array1", "P0"),
    ("random_periodic_array2", "P0"),
    ("random_periodic_array3", "E17.5"), ("random_periodic_array3", "P0"),
    ("random_periodic_array5", "E17.5"), ("random_periodic_array5", "P0"),
    ("random_periodic_array7", "P0"),
    ("random_periodic_array8", "P0"),
]


def _count_self_intersecting_faces(sheet):
    """Complete count of faces whose polygon self-intersects (any two
    non-adjacent perimeter edges cross). Walks each face's unfolded ``sx/sy``
    in ``order``. Complete (unlike the turning-number proxy in
    ``solvers.count_folded_faces``, which misses small self-loops)."""
    def seg_cross(p1, p2, p3, p4):
        d = (p2[0]-p1[0])*(p4[1]-p3[1]) - (p2[1]-p1[1])*(p4[0]-p3[0])
        if abs(d) < 1e-12:
            return False
        t = ((p3[0]-p1[0])*(p4[1]-p3[1]) - (p3[1]-p1[1])*(p4[0]-p3[0])) / d
        u = ((p3[0]-p1[0])*(p2[1]-p1[1]) - (p3[1]-p1[1])*(p2[0]-p1[0])) / d
        return 1e-9 < t < 1-1e-9 and 1e-9 < u < 1-1e-9

    ed = sheet.edge_df.sort_values(["face", "order"])
    fcol = ed["face"].to_numpy(); sx = ed["sx"].to_numpy(); sy = ed["sy"].to_numpy()
    cnt = 0; start = 0; n = len(fcol)
    for k in range(1, n + 1):
        if k == n or fcol[k] != fcol[start]:
            P = np.column_stack((sx[start:k], sy[start:k])); m = len(P)
            if m >= 4:
                bad = False
                for i in range(m):
                    for j in range(i + 2, m):
                        if i == 0 and j == m - 1:
                            continue
                        if seg_cross(P[i], P[(i+1) % m], P[j], P[(j+1) % m]):
                            bad = True; break
                    if bad:
                        break
                if bad:
                    cnt += 1
            start = k
    return cnt


def _frame_is_fold_free(history, time_point):
    """Whether the frame at ``time_point`` has NO self-intersecting face. Uses
    the cheap turning-number check (``solvers.count_folded_faces``) as a fast
    reject, then the complete segment test to confirm."""
    from solvers import count_folded_faces
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history()
    sheet.initiate_edge_order()
    if count_folded_faces(sheet) > 0:        # definitely folded
        return False
    return _count_self_intersecting_faces(sheet) == 0


def closest_fold_free_frames(history, t_best):
    """Latest fold-free time strictly before ``t_best`` and earliest fold-free
    time strictly after it (complete self-intersection check, scanning outward
    from ``t_best``). Either may be ``None`` if no fold-free frame exists on
    that side."""
    times = np.asarray(get_time_points(history), dtype=float)
    t_before = None
    for t in sorted(times[times < t_best], reverse=True):
        if _frame_is_fold_free(history, float(t)):
            t_before = float(t); break
    t_after = None
    for t in sorted(times[times > t_best]):
        if _frame_is_fold_free(history, float(t)):
            t_after = float(t); break
    return t_before, t_after


def extract_fold_free_best_timepoints(corrupted_extracts=None, output_suffix="_foldfree",
                                      max_number_of_neighbors=9, type_by='atoh_level'):
    """Re-extract each corrupted (folded) best-time-point as a FOLD-FREE frame.

    Strategy (the p-value-vs-time curve has a single maximum, so the best frame
    sits on the peak): for each corrupted ``(model, stage)`` whose stored best
    frame self-intersects, find the closest fold-free frame BEFORE and AFTER it,
    compute each one's p-value against the experiment with the SAME metric used
    to pick the original best frame
    (``compare_to_experimental_results(..., "number of neighbors")``), take the
    higher-p-value fold-free frame, and save its artifacts to a NEW folder
    ``<model>_for_<stage><output_suffix>`` via
    :func:`save_data_of_a_given_time_point`. The original (folded) extract is
    left untouched.

    Returns a list of dicts (one per extract) and prints, for each, the p-value
    drop from the folded best frame to the chosen fold-free frame.
    """
    if corrupted_extracts is None:
        corrupted_extracts = CORRUPTED_EXTRACTS

    report = []
    for model_name, stage in corrupted_extracts:
        history = load_history_file(model_name)
        suffix = _STAGE_SHEET_SUFFIX[stage]
        # The time of the existing (folded) extract is the original best time.
        extract_hist = HistoryHdf5.from_archive(
            os.path.join(RESULTS_DIR, model_name + suffix, "history.hf5"),
            eptm_class=VirtualSheet)
        t_best = float(np.max(extract_hist.time_stamps))

        t_before, t_after = closest_fold_free_frames(history, t_best)

        def pval(t):
            return compare_to_experimental_results(
                history, model_name, stage, results_type="number of neighbors",
                type_by=type_by, threshold=None,
                max_number_of_neighbors=max_number_of_neighbors,
                plot=False, time_point=t)

        p_best = pval(t_best)
        candidates = [(t, pval(t)) for t in (t_before, t_after) if t is not None]
        if not candidates:
            print("%s %s: NO fold-free frame found around t=%.3f; skipped"
                  % (model_name, stage, t_best))
            continue
        t_clean, p_clean = max(candidates, key=lambda x: x[1])

        output_name = model_name + suffix + output_suffix
        save_data_of_a_given_time_point(history, t_clean, output_name)

        reduction = p_best - p_clean
        rel = (reduction / p_best * 100.0) if p_best else float("nan")
        print("%s %s: best t=%.3f p=%.4g  ->  fold-free t=%.3f p=%.4g  "
              "(drop %.4g, %.1f%%) -> %s"
              % (model_name, stage, t_best, p_best, t_clean, p_clean,
                 reduction, rel, output_name))
        report.append({"model": model_name, "stage": stage,
                       "t_best": t_best, "p_best": p_best,
                       "t_before": t_before, "t_after": t_after,
                       "t_clean": t_clean, "p_clean": p_clean,
                       "reduction": reduction, "output": output_name})
    return report

def relaxed_cut_scale(face_df):
    """Isotropic scale ``lambda`` a freely-relaxing cut piece would settle at,
    from the affine-relaxation model (see SHRINKAGE_ESTIMATE_METHOD.md): minimize
    ``sum K/2 (l^2 A - A0)^2 + G/2 (l P - P0)^2`` over ``l``. Bending is
    scale-invariant (it depends only on angles) so it does not enter. Returns nan
    if the sheet lacks the mechanical columns or has no interior minimum."""
    need = ("area", "perimeter", "prefered_area", "prefered_perimeter",
            "area_elasticity", "contractility")
    if any(col not in face_df.columns for col in need):
        return float("nan")
    A = face_df["area"].to_numpy(float); P = face_df["perimeter"].to_numpy(float)
    A0 = face_df["prefered_area"].to_numpy(float)
    P0 = face_df["prefered_perimeter"].to_numpy(float)
    K = face_df["area_elasticity"].to_numpy(float)
    G = face_df["contractility"].to_numpy(float)
    ok = np.isfinite(A) & np.isfinite(P) & np.isfinite(A0) & np.isfinite(P0)
    if not ok.any():
        return float("nan")
    A, P, A0, P0, K, G = A[ok], P[ok], A0[ok], P0[ok], K[ok], G[ok]

    def dE(l):
        return float(np.sum(K * (l * l * A - A0) * 2 * l * A + G * (l * P - P0) * P))

    lo, hi = 0.2, 1.6
    if dE(lo) * dE(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dE(lo) * dE(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _hc_over_mean_sc(hc_values, sc_values):
    """Each HC value divided by the MEAN SC value of the same replicate.

    ``None`` when either side has no finite data or the SC mean is not positive
    — the caller drops the term for that replicate rather than scoring garbage.
    """
    if hc_values is None or sc_values is None:
        return None
    hc = np.asarray(hc_values, dtype=float)
    sc = np.asarray(sc_values, dtype=float)
    hc = hc[np.isfinite(hc)]
    sc = sc[np.isfinite(sc)]
    if hc.size == 0 or sc.size == 0:
        return None
    denominator = sc.mean()
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    return hc / denominator


def extract_model_mechanics(model_name, type_by='atoh_level', threshold=None,
                            time_point=-1, ablation_model_name=None,
                            ablated_cells=[], post_ablation_frame=None):
    """Extract ONE run's per-term model DISTRIBUTIONS for the mechanics comparison.

    Splitting extraction from comparison lets several runs — the SAME parameters
    on DIFFERENT initial arrays — be POOLED and compared to the experiments once
    per term (see :func:`compare_pooled_model_mechanics_to_experiments`). Returns
    ``{"hc_roundness", "sc_roundness", "hc_ablation", "sc_ablation"}`` where each
    value is a model distribution at frame ``time_point`` (the last by default):

    * ``hc_roundness`` / ``sc_roundness`` — absolute per-cell roundness
      (4*pi*A/P**2) of HCs / SCs; dimensionless.
    * ``hc_ablation`` / ``sc_ablation`` — per-cell area change (area after / area
      before ablation) of the ablation-adjacent HCs / SCs, or ``None`` when no
      ablation run is supplied (``ablation_model_name`` / ``post_ablation_frame`` /
      ``ablated_cells``).

    (The HC/SC-area-ratio term was dropped — its experimental spread is
    small-sample noise the model can't reproduce — and the single ablation
    HC-over-SC ratio was split into separate HC and SC area-change terms.)
    """
    history = load_history_file(model_name)
    if time_point == -1:
        time_point = np.max(history.time_stamps)
    sheet = history.retrieve(time_point)
    sheet.arrange_sheet_from_history()

    hc_roundness = calc_roundness_for_type(sheet, cell_type="HC", type_by=type_by, threshold=threshold)
    sc_roundness = calc_roundness_for_type(sheet, cell_type="SC", type_by=type_by, threshold=threshold)

    hc_ablation = sc_ablation = None
    if ablation_model_name is not None and post_ablation_frame is not None and len(ablated_cells) > 0:
        ablation_history = load_history_file(ablation_model_name)
        hc_ablation, sc_ablation = calc_area_change_after_ablation(
            ablation_history, ablation_model_name, ablated_cells=ablated_cells,
            end_time=post_ablation_frame, type_by=type_by, threshold=threshold)
    # Predicted LINEAR shrinkage (%) of a cut disc — a whole-tissue observable
    # taken from the SAME final frame, so it costs no extra simulation. One value
    # per run, wrapped in an array so it pools like the per-cell terms.
    lam = relaxed_cut_scale(sheet.face_df)
    shrinkage = (np.array([100.0 * (1.0 - lam)]) if np.isfinite(lam)
                 else np.array([], dtype=float))

    # HC-over-SC RATIO terms, built to MIRROR the experimental construction
    # exactly (see load_experimental_results, "HC to SC roundness ratio" and
    # "HC to SC area change ratio after ablation"): each HC value divided by the
    # MEAN SC value of the SAME replicate — one simulation run here, one
    # experiment there. Doing it per replicate matters: the comparison averages
    # per-replicate means, so a ratio of POOLED means would mix replicates and
    # is a different statistic. Absolute terms are still returned for
    # diagnostics; only the ratios are scored (see _MECHANICS_EXPERIMENTAL_TYPE).
    roundness_ratio = _hc_over_mean_sc(hc_roundness, sc_roundness)
    ablation_ratio = _hc_over_mean_sc(hc_ablation, sc_ablation)
    return {"hc_roundness": hc_roundness, "sc_roundness": sc_roundness,
            "hc_ablation": hc_ablation, "sc_ablation": sc_ablation,
            "roundness_ratio": roundness_ratio,
            "ablation_ratio": ablation_ratio,
            "shrinkage": shrinkage}


# The fit objectives -> the experimental "type" each is compared against.
# Insertion order is the canonical term order used everywhere downstream.
#
# THREE terms as of the v2 (contractility) fit: the two absolute roundness terms
# were replaced by their HC/SC RATIO, and likewise the two ablation terms. The
# ratio cancels whatever sets the overall scale of a replicate, so what is
# compared is the HC-vs-SC CONTRAST the model is actually being asked to
# reproduce, rather than two absolute levels that move together.
#
# Both experimental ratio loaders build one array PER EXPERIMENT (each HC value
# over that experiment's mean SC value), so compare_pooled_model_mechanics_to_
# experiments takes 3 per-experiment means and their SEM — which is exactly the
# "average each experiment separately, 3 data points" the score is specified on.
# The model side mirrors it in extract_model_mechanics via _hc_over_mean_sc.
#
# The absolute terms are still returned by extract_model_mechanics for
# diagnostics; they are simply no longer scored.
_MECHANICS_EXPERIMENTAL_TYPE = {
    "roundness_ratio": "HC to SC roundness ratio",
    "ablation_ratio": "HC to SC area change ratio after ablation",
    "shrinkage": "cut shrinkage",
}

# A degenerate term (no model data / no usable experimental SEM) is scored this
# many sigma so the point gets a large but FINITE penalty (the objective is a sum
# of z**2, so this must be finite to stay comparable).
_WORST_CASE_NSIGMA = 1e3


def _finite_arrays(arrays):
    """Coerce a list of arrays to 1-D float, drop non-finite cells, and drop any
    array left empty. Returns a list of float arrays (possibly the empty list)."""
    out = []
    for a in (arrays or []):
        if a is None:
            continue
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size > 0:
            out.append(a)
    return out


def compare_pooled_model_mechanics_to_experiments(model_terms, experimental_stage,
                                                  plot=False):
    """Standardized discrepancy (n-sigma) between the POOLED model distributions
    and the experimental biological repeats — ONE signed ``z`` per term.

    For each term ``z = (mean_model - mean_exp) / SEM_exp``, where the means are
    grand means over replicates (per-run means for the model, per-repeat means for
    the experiment) and ``SEM_exp = std(repeat means, ddof=1) / sqrt(n_repeats)``
    is the standard error of the experimental repeat means. This REPLACES the
    p-value objective: the pooled hierarchical p saturated at ~0 across almost all
    parameters (a flat, un-gradiented landscape that pinned the optimizer at the
    bounds), whereas the n-sigma distance varies smoothly and is zero exactly when
    the model mean lands on the experimental mean. The optimizer minimizes
    ``sum(z**2)`` over the terms.

    ``model_terms`` maps each term (see ``_MECHANICS_EXPERIMENTAL_TYPE``) to a LIST
    of per-run arrays. Returns ``{term: z}`` (signed). ``z`` is ``nan`` when the
    term has no finite model data, fewer than 2 experimental repeats, or a
    zero/non-finite experimental SEM — the CALLER decides whether a nan is a
    neutral skip (e.g. ablation not simulated) or a penalized miss. Non-finite
    cells are dropped before any statistic.
    """
    zscores = {}
    for term, exp_type in _MECHANICS_EXPERIMENTAL_TYPE.items():
        model_list = _finite_arrays(model_terms.get(term))
        if not model_list:
            zscores[term] = float("nan")
            continue
        try:
            exp = _finite_arrays(load_experimental_results(experimental_stage, exp_type))
            if len(exp) < 2:
                zscores[term] = float("nan")
                continue
            exp_means = np.array([e.mean() for e in exp], dtype=float)
            exp_mean = float(exp_means.mean())
            exp_sem = float(np.std(exp_means, ddof=1) / np.sqrt(exp_means.size))
            model_mean = float(np.mean([a.mean() for a in model_list]))
            z = (model_mean - exp_mean) / exp_sem if exp_sem > 0 else float("nan")
        except Exception as exc:  # noqa: BLE001 - a degenerate term must not kill the fit
            print("[mechanics] %s discrepancy failed (%s: %s); nan"
                  % (term, type(exc).__name__, exc))
            z = float("nan")
        zscores[term] = float(z)
    print("n-sigma: " + "  ".join("%s=%.3g" % (t, zscores[t])
                                  for t in _MECHANICS_EXPERIMENTAL_TYPE))
    return zscores


def compare_model_mechanics_to_experiments(model_name, experimental_stage,
                                           type_by='atoh_level', threshold=None,
                                           time_point=-1, plot=False,
                                           ablation_model_name=None, ablated_cells=[],
                                           post_ablation_frame=None):
    """Score a SINGLE model run against experiments with the n-sigma objective.

    Thin wrapper over :func:`extract_model_mechanics` +
    :func:`compare_pooled_model_mechanics_to_experiments` with a one-element pool.
    The fit pools ALL initial-array runs and calls those two directly; this is kept
    for interactive single-run checks.

    Returns ``(objective, {term: z})`` where ``objective`` is the sum of ``z**2``
    over the terms whose ``z`` is finite (nan terms — no data — are skipped).
    """
    terms = extract_model_mechanics(
        model_name, type_by=type_by, threshold=threshold, time_point=time_point,
        ablation_model_name=ablation_model_name, ablated_cells=ablated_cells,
        post_ablation_frame=post_ablation_frame)
    model_terms = {t: ([] if terms[t] is None else [terms[t]])
                   for t in _MECHANICS_EXPERIMENTAL_TYPE}
    z = compare_pooled_model_mechanics_to_experiments(model_terms, experimental_stage, plot=plot)
    objective = float(np.nansum([zz ** 2 for zz in z.values()]))
    return objective, z


# --------------------------------------------------------------------------- #
# Visualizing the mechanical-parameter Bayesian-optimization trace            #
# --------------------------------------------------------------------------- #
def load_mechanical_optimization(stage, results_dir=RESULTS_DIR,
                                 n_params=3):
    """Load the optimizer trace ``find_mechanical_parameters`` saved for
    ``stage`` — the ``<stage>_optimization_params.npy`` (evaluated points,
    shape ``(n_calls, n_params)``) and ``<stage>_optimization_objective.npy``
    (their objective values, shape ``(n_calls,)``).

    The objective is ``sum_terms z**2`` (``z`` = standardized model-vs-experiment
    mean discrepancy per term: HC/SC roundness and HC/SC area-change-after-
    ablation) — LOWER is a better fit, ~0 meaning the model means land within the
    experimental replicate uncertainty.

    Raises a clear error if the files are absent or EMPTY (a completed fit
    writes ``n_calls`` rows; a 0-row file means the trace never reached disk)."""
    px = os.path.join(results_dir, "%s_optimization_params.npy" % stage)
    py = os.path.join(results_dir, "%s_optimization_objective.npy" % stage)
    for p in (px, py):
        if not os.path.isfile(p):
            raise FileNotFoundError("No optimization result at %s" % p)
    X = np.load(px)
    y = np.asarray(np.load(py), float).ravel()
    if X.size == 0 or y.size == 0:
        raise ValueError(
            "Optimization result files for %r are EMPTY (params shape %s, "
            "objective shape %s). A completed find_mechanical_parameters writes "
            "an (n_calls x %d) params array and an (n_calls,) objective array — "
            "0 rows means the trace was never saved (the run didn't finish, or "
            "result['X']/result['y'] were empty). Re-run the fit, or re-save "
            "result['X']/result['y'], before plotting."
            % (stage, X.shape, y.shape, n_params))
    X = np.atleast_2d(X).astype(float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("params/objective length mismatch: %d vs %d"
                         % (X.shape[0], y.shape[0]))
    return X, y


def load_mechanical_optimization_trace(stage, results_dir=RESULTS_DIR):
    """Read the per-evaluation JSONL trace ``find_mechanical_parameters`` writes
    incrementally (``<stage>_optimization_trace.jsonl``) into a DataFrame.

    One row per evaluated point, with the parameters, the ``objective``
    (``sum_terms z**2`` over the active terms), the signed per-term discrepancy
    (``nsigma_hc_roundness`` / ``nsigma_sc_roundness`` / ``nsigma_hc_ablation`` /
    ``nsigma_sc_ablation`` — standardized model-vs-experiment mean gap; may be
    NaN when a term has no data), its ``z**2`` contribution (``obj_<term>``, which
    sum to ``objective``), the pooled model sample size per term (``n_<term>``),
    how many sheets contributed (``n_contributing`` of ``n_sheets``) and the
    per-sheet ok/dropped status (``sheets``). This is the crash-resistant record —
    it exists even if the run was killed before the final ``.npy``/landscape
    save."""
    import json
    path = os.path.join(results_dir, "%s_optimization_trace.jsonl" % stage)
    if not os.path.isfile(path):
        raise FileNotFoundError("No optimization trace at %s" % path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Optimization trace %s is empty (0 evaluations)." % path)
    return pd.DataFrame(rows)


def load_mechanical_optimization_landscape(stage, results_dir=RESULTS_DIR):
    """Load the GP-surrogate landscape bundle
    (``<stage>_optimization_landscape.npz``) saved at the end of
    ``find_mechanical_parameters``.

    Returns a dict with: ``param_names`` (d,), ``bounds`` (d, 2), ``axes``
    (d, res) the per-parameter grid coordinates, ``mean`` / ``std`` the GP
    posterior objective (shape ``res**d``, indexed ``[i_gammaSC, i_gammaHC, ...]``,
    ``indexing='ij'``), and ``X`` / ``y`` the evaluated training points. The
    objective is ``sum of z**2`` (per-term standardized mean discrepancy)."""
    path = os.path.join(results_dir, "%s_optimization_landscape.npz" % stage)
    if not os.path.isfile(path):
        raise FileNotFoundError("No optimization landscape at %s" % path)
    return dict(np.load(path, allow_pickle=True))


def plot_mechanical_optimization(
        stage, results_dir=RESULTS_DIR,
        param_names=("gammaSC", "gammaHC_ratio", "alphaHC_ratio"),
        bounds=None, pval_floor=1e-300, save_path=None, show=True):
    """Visualize the mechanical-parameter Bayesian-optimization trace saved by
    :func:`run_model.find_mechanical_parameters` for ``stage`` (e.g. ``"E17.5"``).

    Produces a 2x2 figure:

    1. **Convergence** — every evaluation's objective (``sum of z**2`` over the
       active terms; lower is better) plus the running best, with the overall
       best starred and any degenerate evaluations (a term worst-cased at ~1e6)
       flagged off-scale, so you can see how much of the search actually improved.
    2-4. **Per-parameter marginals** — objective vs each parameter, coloured by
       evaluation order (to see where the optimizer concentrated), best point
       starred, and the bound edges drawn (``bounds`` optional; falls back to
       the explored min/max) so a fit pinned AT a bound is obvious.

    Also prints a summary (best params, best objective, per-parameter value,
    boundary-pinning, and the fraction of evaluations that degenerated). Returns
    ``(fig, (X, y))``.

    ``bounds`` : optional list of ``(lo, hi)`` per parameter — pass the same
    ``gammaSC_bounds`` / ``gammaHC_ratio_bounds`` / ``alphaHC_ratio_bounds``
    used for the fit to mark the true search box.
    """
    X, y = load_mechanical_optimization(stage, results_dir, n_params=len(param_names))
    n, d = X.shape
    evals = np.arange(1, n + 1)
    running_best = np.minimum.accumulate(y)
    best_idx = int(np.argmin(y))
    # A point with any worst-cased (degenerate) term has objective >= 1e6
    # (_WORST_CASE_NSIGMA**2); a real fit sits at O(1-100). So this cleanly flags
    # the degenerate evaluations that starve the optimizer.
    degenerate = 1e5
    at_ceiling = int(np.sum(y >= degenerate))

    fig = plt.figure(figsize=(14, 9))

    # (1) Convergence.
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(evals, y, s=18, c="0.65", label="evaluations")
    ax.plot(evals, running_best, "b-", lw=2, label="running best")
    ax.scatter([best_idx + 1], [y[best_idx]], marker="*", s=220,
               c="crimson", zorder=5, label="best")
    if at_ceiling:
        ax.text(0.02, 0.96, "%d degenerate eval(s) off-scale (a term worst-cased)"
                % at_ceiling, transform=ax.transAxes, va="top", fontsize=8,
                color="0.3")
        # keep the informative range visible despite the 1e6 outliers.
        good = y[y < degenerate]
        if good.size:
            ax.set_ylim(0, float(good.max()) * 1.1)
    ax.set_xlabel("evaluation #")
    ax.set_ylabel(r"objective = $\sum z^2$  (lower is better)")
    ax.set_title(r"Convergence   best $\sum z^2$ = %.3g   (%d/%d degenerate)"
                 % (y[best_idx], at_ceiling, n))
    ax.legend(loc="upper right", fontsize=8)

    # (2-4) Per-parameter marginals.
    for j in range(d):
        ax = fig.add_subplot(2, 2, 2 + j)
        sc = ax.scatter(X[:, j], y, c=evals, cmap="viridis", s=28,
                        edgecolors="none")
        ax.scatter([X[best_idx, j]], [y[best_idx]], marker="*", s=220,
                   c="crimson", zorder=5)
        lo, hi = (bounds[j] if bounds is not None
                  else (float(X[:, j].min()), float(X[:, j].max())))
        for edge in (lo, hi):
            ax.axvline(edge, color="0.6", ls=":", lw=1)
        # Flag a best value sitting on a bound (fit wants to escape the box).
        span = (hi - lo) or 1.0
        pinned = ""
        if bounds is not None and min(abs(X[best_idx, j] - lo),
                                      abs(X[best_idx, j] - hi)) <= 0.02 * span:
            pinned = "  [PINNED at bound]"
        ax.set_xlabel(param_names[j])
        ax.set_ylabel(r"objective ($\sum z^2$)")
        ax.set_title("%s   best = %.4g%s" % (param_names[j], X[best_idx, j], pinned))
    cbar = fig.colorbar(sc, ax=fig.axes[1:], fraction=0.025, pad=0.02)
    cbar.set_label("evaluation order")

    fig.suptitle("Mechanical-parameter fit — %s" % stage, fontsize=13)

    # Console summary — the actionable bits.
    print("=== Mechanical optimization summary (%s) ===" % stage)
    print("evaluations: %d   best objective (sum z^2): %.4g   "
          "(~%.2f sigma per active term)"
          % (n, y[best_idx], (y[best_idx] / 4.0) ** 0.5))
    for j in range(d):
        lo, hi = (bounds[j] if bounds is not None
                  else (float(X[:, j].min()), float(X[:, j].max())))
        tag = ""
        if bounds is not None and min(abs(X[best_idx, j] - lo),
                                      abs(X[best_idx, j] - hi)) <= 0.02 * ((hi - lo) or 1.0):
            tag = "  <-- at bound (consider widening)"
        print("  %-14s best=%.4g   explored [%.4g, %.4g]%s"
              % (param_names[j], X[best_idx, j], X[:, j].min(), X[:, j].max(), tag))
    print("  degenerate (a term worst-cased) evaluations: %d / %d (%.0f%%)"
          % (at_ceiling, n, 100.0 * at_ceiling / n))
    if at_ceiling > 0.3 * n:
        print("  NOTE: a large fraction of the search degenerated (a term had no "
              "usable data — dropped sheets or empty ablation), which starves the "
              "optimizer. Check the debug logs (stall guard) and narrow the bounds "
              "to the solver-tractable region.")

    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (X, y)


if __name__ == "__main__":
    #model_names = ["periodic_fromrandom_periodic_array%d_for_P0_gammaSC-0.01_gammaHC_ratio-10.00_alphaHC_ratio-1.00_psigma-0.00"%i for i in range(10)]
    #save_li_levels_at_best_matching_frame(model_names, "P0", type_by="delta_level",
     #                                     threshold=0.377, max_number_of_neighbors=2, plot=True)
    # compare_differentiation_to_experiments(
    #    "periodic_fromrandom_periodic_array0_for_P0_gammaSC-0.01_gammaHC_ratio-10.00_alphaHC_ratio-1.00_psigma-0.00",
    #    experimental_stage="P0",
    #    type_by='delta_level',
    #    threshold=0.377,
    #    max_number_of_neighbors=2,
    #    plot=True)
    # find_best_timepoint_for_random_arrays(["random_periodic_array0"])
    # store_best_time_point_for_random_arrays()
    # for name in ["periodic_fromrandom_periodic_array%d_for_E17_gammaSC-0.01_gammaHC_ratio-10.00_alphaHC_ratio-1.00_psigma-0.00"% i for i in range (1,10)]:
    #     redraw(name,"movie", movie=True,
    #            color_by="delta")
    names = ["fullmodel_v2_random_periodic_array2_for_P0"]
    for name in names:
        redraw(name,"delta", movie=True, maximal_number_of_frames_to_save=100,
               color_by="delta", maximal_level=find_maximal_level_final_frame(name, "delta_level"))
    # extract_fold_free_best_timepoints(corrupted_extracts=[("random_periodic_array6", "E17.5")])
    # compare_model_mechanics_to_experiments("fit_gSC1.63_gHC1.12_aHC2.00_ps0.00_eeca8ad8fd", "E17.5",
    #                                        type_by='delta_level', threshold=0.3350058877,
    #                                        time_point=-1, plot=True,
    #                                        ablation_model_name=None,
    #                                        ablated_cells=[337, 304, 65, 114],
    #                                        post_ablation_frame=-1)
    # for stage in ["E17.5", "P0"]:
    #     res = load_experimental_results(stage, "lonely SC percentage")
    #     print(res)
    # plt.show()