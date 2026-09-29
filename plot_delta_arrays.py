"""Draw an initial array coloured by delta levels supplied as a .npy file.

    python plot_delta_arrays.py
    python plot_delta_arrays.py --dir "D:/.../lateral inhibition only results"

A colleague's lateral-inhibition-only runs record only the delta level per cell,
one 1D array per saved frame, because that model has no mechanics: the geometry
never moves, so one drawing of the initial array serves every time point and
only the colouring changes.

Each file is drawn with InnerEarModel.get_draw_sheet_method — the same routine
history_io.redraw uses — by loading the initial array the run started from and
overwriting its ``delta_level`` column before drawing.

HOW THE VALUES LINE UP WITH THE CELLS. The arrays are indexed by unique face id,
which on these sheets is the row position: ``unique_id`` runs 0..N-1 in face_df
order, and the file length equals the face count exactly (508 for E17 array 0,
538 for P0 array 1). The check that this is really the right ordering is that
each first-frame file correlates 0.92 (E17) and 0.87 (P0) with the array's own
delta_levels.npy seed; a mismatched ordering would correlate around zero.

ONE COLOUR SCALE FOR EVERY PANEL. maximal_level defaults to the largest delta
across all the files being drawn, so first and last frame are directly
comparable. Pass --maximal-level to fix it by hand.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from post_processing import RESULTS_DIR
from run_model import load_sheet_from_file
from inner_ear_model import InnerEarModel
from history_io import annotate_time
from ablate_single_hc import BOX

LI_DIR = r"D:\Kasirer\results\lateral inhibition only results"
# the file names carry everything needed to find the array they belong to
NAME_RE = re.compile(r"D_(?P<frame>\w+?)_frame_(?P<stage>E17|P0)_array_(?P<array>\d+)"
                     r"_repeat_(?P<repeat>\d+)_pS_(?P<pS>[\d.]+)_pR_(?P<pR>[\d.]+)\.npy$")


def parse(path):
    m = NAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    d = m.groupdict()
    d["path"] = path
    d["initial_array"] = "random_periodic_array%s_for_%s" % (d["array"], d["stage"])
    return d


def draw_one(spec, maximal_level, results_dir=RESULTS_DIR, out_dir=None):
    """Load the array, colour it by the file's deltas, save the figure."""
    values = np.load(spec["path"])
    sheet = load_sheet_from_file(os.path.join(results_dir, spec["initial_array"]),
                                 force_periodic_box=BOX)
    if len(sheet.face_df) != values.size:
        raise SystemExit("%s has %d values but %s has %d faces"
                         % (os.path.basename(spec["path"]), values.size,
                            spec["initial_array"], len(sheet.face_df)))
    uid = sheet.face_df["unique_id"].to_numpy()
    if not np.array_equal(uid, np.arange(len(sheet.face_df))):
        # the file is indexed by unique face id; only when that IS the row
        # position can the values be assigned in order
        order = np.argsort(uid)
        values = values[np.argsort(order)]
    sheet.face_df["delta_level"] = values

    draw = InnerEarModel.get_draw_sheet_method(
        number_faces=False, number_edges=False, number_vertices=False,
        arrange_sheet=False, color_by="delta", maximal_level=maximal_level)
    fig, ax = draw(sheet)
    ax.set_title("%s  array %s  %s frame\ndelta 0-%.3f, pS=%s pR=%s"
                 % (spec["stage"], spec["array"], spec["frame"], maximal_level,
                    spec["pS"], spec["pR"]), fontsize=11)
    stem = os.path.join(out_dir or results_dir,
                        "delta_%s_array%s_%s_frame"
                        % (spec["stage"], spec["array"], spec["frame"]))
    for ext in ("png", "svg"):
        fig.savefig("%s.%s" % (stem, ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stem, values


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=LI_DIR, help="where the .npy files are")
    ap.add_argument("--out-dir", default=None, help="default: the results dir")
    ap.add_argument("--maximal-level", type=float, default=None,
                    help="delta value mapped to the darkest colour "
                         "(default: the largest across the files drawn)")
    a = ap.parse_args()

    specs = [s for s in (parse(p) for p in sorted(glob.glob(os.path.join(a.dir, "*.npy"))))
             if s is not None]
    if not specs:
        raise SystemExit("no D_*_frame_*.npy files in %s" % a.dir)

    top = a.maximal_level
    if top is None:
        top = float(max(np.load(s["path"]).max() for s in specs))
        print("  common colour scale: delta 0 to %.4f (largest over %d file(s))"
              % (top, len(specs)))

    for s in specs:
        stem, v = draw_one(s, top, out_dir=a.out_dir)
        print("  %-6s array %s  %-5s frame: %d cells, delta %.4f-%.4f -> %s.png/.svg"
              % (s["stage"], s["array"], s["frame"], v.size, v.min(), v.max(),
                 os.path.basename(stem)))


if __name__ == "__main__":
    main()
