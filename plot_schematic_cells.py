"""A zoomed view of two cells, sharp against a blurred neighbourhood.

    python plot_schematic_cells.py
    python plot_schematic_cells.py --cells 147 148 --blur 9 --delta 0.5

Raw material for a hand-drawn schematic: two real cells of a real initial array,
with their neighbourhood behind them thrown out of focus so it reads as
background rather than as data.

EVERY CELL IS GIVEN THE SAME DELTA, so the colour carries nothing — this is a
picture of geometry, not of a pattern. The level is --delta as a fraction of
maximal_level, and the project's own colour map runs DARK GREEN at delta 0 to
WHITE at delta 1 (InnerEarModel.get_draw_sheet_method builds it as
Greens-reversed, sampled at 0.7*(x-1)+1, over faces drawn at alpha 0.5). So 0.5
is the midpoint the request asks for, a light green; lower is greener, higher is
paler.

HOW THE BLUR IS DONE. The figure is rendered once, sharp. A mask is then
rendered from the SAME figure and axes — an opaque cover with the two chosen
cells painted white over it — so the mask lines up with the image exactly, with
no coordinate transform of our own to get wrong, and matplotlib's antialiasing
gives the mask a feathered edge for free. The output is sharp inside the mask
and Gaussian-blurred outside it. The mask polygons are stroked as well as
filled, so each chosen cell keeps its own boundary sharp.

THE SVG IS BLURRED TOO, by a real filter rather than a raster. Matplotlib cannot
emit one — its agg_filter rasterises the artist it is applied to — so the SVG is
built differently from the PNG: the sheet is tagged, the two cells are REDRAWN
sharp over it, and an feGaussianBlur element is injected into the saved file and
referenced from the tagged groups. Everything stays a path, so the file is still
editable and the blur strength is one stdDeviation attribute. See
write_blurred_svg.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from post_processing import RESULTS_DIR
from run_model import load_sheet_from_file
from inner_ear_model import InnerEarModel
from ablate_single_hc import BOX

ARRAY = "random_periodic_array0_for_E17"
CELLS = (147, 148)
DELTA_FRACTION = 0.5      # of maximal_level; 0 = dark green, 1 = white
# 0 frames the neighbourhood exactly, so its outer ring is the last thing fully
# in view; a little more keeps those cells off the frame edge
PAD = 0.04
BLUR_SIGMA = 9.0          # pixels, at the rendered dpi
BORDER = 2.2              # cell-border width; tyssue's own default is 0.5


def neighbourhood(sheet, cells):
    """The chosen cells plus everything touching them."""
    keep = set(int(c) for c in cells)
    for c in cells:
        keep.update(int(n) for n in np.atleast_1d(sheet.get_neighbors(int(c))))
    return sorted(keep)


def _blur(img, sigma):
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:                                 # pragma: no cover
        raise SystemExit("scipy is needed for the blur; pass --blur 0 to skip")
    out = np.empty_like(img)
    for k in range(img.shape[2]):                       # per channel, 2-D blur
        out[..., k] = gaussian_filter(img[..., k], sigma)
    return out


def render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255.0


FILTER_ID = "schematic-blur"


def write_blurred_svg(fig, ax, polys, cells, sigma_px, dpi, border, path):
    """An SVG whose background is blurred by a real feGaussianBlur filter.

    Matplotlib cannot emit a blur: an agg_filter would make it rasterise the
    artist, which defeats the point of a vector file. So the blur is built in
    SVG terms instead — the whole sheet goes into tagged groups, the two chosen
    cells are REDRAWN sharp on top of it, and a filter element is injected into
    the saved file and referenced from the tagged groups.

    The result stays editable: the blur is a renderer-side effect, so its
    strength can be changed by editing stdDeviation, and the cells underneath
    are still paths.

    stdDeviation is in SVG user units, which matplotlib writes as points, so
    the pixel sigma is converted at 72/dpi.
    """
    # THE SHARP COPIES MUST BE OPAQUE. The sheet draws faces at alpha 0.5 over
    # white and borders at 0.8 over those faces; drawing a second translucent
    # copy on top would composite twice and come out visibly darker than its
    # surroundings. So each colour is flattened to what it already looks like on
    # screen and redrawn at alpha 1, which also hides the blurred original
    # underneath instead of blending with it.
    face_rgba = np.asarray(ax.collections[0].get_facecolor()[0], float)
    edge_rgba = np.asarray(np.atleast_2d(ax.collections[1].get_color())[0], float)
    face_rgb = face_rgba[:3] * face_rgba[3] + (1.0 - face_rgba[3])      # over white
    edge_rgb = edge_rgba[:3] * edge_rgba[3] + face_rgb * (1.0 - edge_rgba[3])
    face_rgba = tuple(face_rgb) + (1.0,)
    edge_rgba = tuple(edge_rgb) + (1.0,)

    for i, coll in enumerate(ax.collections):
        coll.set_gid("blur-layer-%d" % i)
    sharp = []
    for c in cells:
        p = Polygon(np.asarray(polys[c]), closed=True, facecolor=face_rgba,
                    edgecolor=edge_rgba, linewidth=border, zorder=900)
        ax.add_patch(p)
        sharp.append(p)
    fig.savefig(path)
    for p in sharp:
        p.remove()
    for coll in ax.collections:
        coll.set_gid(None)

    with open(path, "r", encoding="utf-8") as fh:
        svg = fh.read()
    std = sigma_px * 72.0 / dpi
    defs = ('<defs><filter id="%s" x="-20%%" y="-20%%" width="140%%" '
            'height="140%%"><feGaussianBlur stdDeviation="%.4f"/></filter></defs>'
            % (FILTER_ID, std))
    head = svg.index(">", svg.index("<svg")) + 1
    svg = svg[:head] + defs + svg[head:]
    n = 0
    for i in range(len(ax.collections)):
        tag = '<g id="blur-layer-%d"' % i
        if tag in svg:
            svg = svg.replace(tag, tag + ' filter="url(#%s)"' % FILTER_ID)
            n += 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return n, std


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", default=ARRAY)
    ap.add_argument("--cells", type=int, nargs="+", default=list(CELLS))
    ap.add_argument("--delta", type=float, default=DELTA_FRACTION,
                    help="uniform delta as a fraction of maximal_level "
                         "(0 dark green .. 1 white; default %g)" % DELTA_FRACTION)
    ap.add_argument("--pad", type=float, default=PAD)
    ap.add_argument("--blur", type=float, default=BLUR_SIGMA,
                    help="Gaussian sigma in pixels; 0 leaves the view sharp")
    ap.add_argument("--border", type=float, default=BORDER,
                    help="cell-border width (tyssue's default is 0.5)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    a = ap.parse_args()

    sheet = load_sheet_from_file(os.path.join(a.results_dir, a.array),
                                 force_periodic_box=BOX)
    missing = [c for c in a.cells if c not in sheet.face_df.index]
    if missing:
        raise SystemExit("no such face(s) in %s: %s" % (a.array, missing))
    # one colour for every cell: the picture is about shape, not about pattern
    sheet.face_df["delta_level"] = float(a.delta)

    draw = InnerEarModel.get_draw_sheet_method(
        number_faces=False, number_edges=False, number_vertices=False,
        arrange_sheet=False, color_by="delta", maximal_level=1.0)
    fig, ax = draw(sheet)
    fig.set_dpi(a.dpi)
    # the cell borders are one LineCollection; get_draw_sheet_method does not
    # expose its width, so it is widened on the artist after the fact
    from matplotlib.collections import LineCollection
    for coll in ax.collections:
        if isinstance(coll, LineCollection):
            coll.set_linewidth(a.border)

    # draw() sorts the edges and sets is_ordered, which face_polygons requires
    polys = sheet.face_polygons(["x", "y"])
    around = neighbourhood(sheet, a.cells)
    pts = np.concatenate([np.asarray(polys[f]) for f in around])
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    pad = a.pad * max(x1 - x0, y1 - y0)                 # same margin both ways
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    # the figure is shaped like the view, so an equal aspect fills the canvas
    # instead of leaving white bands down two sides
    width = 8.0
    fig.set_size_inches(width, width * (y1 - y0) / (x1 - x0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    stem = a.out or os.path.join(a.results_dir, "schematic_cells_%s"
                                 % "_".join(str(c) for c in a.cells))
    stem = os.path.splitext(stem)[0]
    if a.blur > 0:
        n_layers, std = write_blurred_svg(fig, ax, polys, a.cells, a.blur, a.dpi,
                                          a.border, stem + ".svg")
    else:
        fig.savefig(stem + ".svg")
        n_layers, std = 0, 0.0

    sharp = render(fig)
    if a.blur > 0:
        # the mask comes off the SAME axes, so it is aligned by construction
        cover = Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="black",
                          edgecolor="none", zorder=1000)
        ax.add_patch(cover)
        marks = []
        for c in a.cells:
            # The mask stroke has to cover the drawn border, which straddles the
            # polygon path and so sticks out by border/2 — but only just. Any
            # wider and it keeps a ring of plain background sharp against the
            # blur, which reads as a pale halo around the cell.
            p = Polygon(np.asarray(polys[c]), closed=True, facecolor="white",
                        edgecolor="white", linewidth=a.border + 2.0, zorder=1001)
            ax.add_patch(p)
            marks.append(p)
        mask = render(fig)[..., 0][..., None]           # white = keep sharp
        for p in [cover] + marks:
            p.remove()
        out = sharp * mask + _blur(sharp, a.blur) * (1.0 - mask)
    else:
        out = sharp

    plt.imsave(stem + ".png", np.clip(out, 0, 1))
    print("  array %s, cells %s, neighbourhood %d cell(s)"
          % (a.array, a.cells, len(around)))
    print("  delta %.2f of maximal -> colour %.2f of the way from green to white"
          % (a.delta, a.delta))
    print("  view x %.2f..%.2f  y %.2f..%.2f   blur sigma %.1f px"
          % (x0, x1, y0, y1, a.blur))
    if n_layers:
        print("  svg: feGaussianBlur stdDeviation %.3f on %d layer(s)"
              % (std, n_layers))
    print("\n  wrote %s.png and %s.svg" % (stem, stem))


if __name__ == "__main__":
    main()
