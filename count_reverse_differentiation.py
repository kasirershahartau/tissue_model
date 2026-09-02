"""Do hair cells ever turn back into support cells?

    python count_reverse_differentiation.py
    python count_reverse_differentiation.py --runs-per-point 3 --seed 1

Counts HC -> SC crossings in one randomly chosen run per (stage, pT), two ways:

MIRRORED — the same rule build_fullmodel_table.differentiation_events uses for
SC -> HC, with the direction flipped: take every non-boundary SC of the FINAL
frame, walk back through its uninterrupted run of SC frames, and call the first
of them the crossing. This is the number directly comparable to the forward
event counts, because it is the same definition.

It inherits the same blind spot, which matters more in this direction: it only
sees a cell's LAST crossing, so a cell that went HC -> SC -> HC ends as a HC and
its reversal is invisible. That is exactly the transient this question is about.

ALL CROSSINGS — every frame-to-frame sign change of (delta - threshold) for every
cell, in both directions, over the same window. Nothing is missed, cells are not
required to end in any particular state, and the forward count from this pass is
printed too so the two directions can be compared on equal terms.

Both use the run's own t0 (the frame the scoring matches to the experiment) as
the start of the window, and the same delta threshold the scores use.
"""
import argparse
import os

import numpy as np
import pandas as pd

from post_processing import (RESULTS_DIR, load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type)
from build_experimental_tables import read_table

TYPE_BY = "delta_level"
THRESHOLD = 0.355079
POINTS = [(stage, ps) for stage in ("E17.5", "P0") for ps in (0.0, 0.162)]


#: t0 comes back from the saved table as a float that has been through a pickle
#: and a CSV, so it can sit a few ulp BELOW the history stamp it names (seen:
#: 10.209999999999829 against 10.209999999999827). A bare ``>= t0`` then drops the
#: t0 frame itself, and every cell that crossed between t0 and the next frame
#: looks like it was already a HC when the window opened — which cost 2 of 54
#: events on the first run tested. The tolerance is far below the ~0.1 frame
#: spacing, so it can only ever re-admit the intended frame.
T0_TOL = 1e-9


def _delta_by_frame(history, t0, type_by=TYPE_BY):
    """Per frame from t0 on: delta indexed by persistent cell id."""
    stamps = np.asarray(get_time_points(history), float)
    stamps = stamps[stamps >= t0 - T0_TOL]
    frames = []
    for t in stamps:
        s = history.retrieve(float(t))
        s.arrange_sheet_from_history()
        frames.append(s.face_df.set_index("id")[type_by])
    return stamps, frames


def mirrored_reverse(history, stamps, frames, threshold, type_by=TYPE_BY):
    """HC -> SC events by the forward rule, run backwards.

    Mirror of differentiation_events: the final frame's non-boundary SCs, each
    walked back through its uninterrupted SC run; a cell already SC at t0 did
    not revert inside the window and is skipped.

    Both directions test the FIRST FRAME rather than asking whether the walk
    reached it, which is what rejects a transient — a cell that left its
    starting state and returned to it. The model produces none of those at
    pT 0 or 0.162, in either direction, so the guard changes no count here; it
    stops the rule depending on that being true.
    """
    if stamps.size == 0:
        return []
    final = history.retrieve(float(stamps[-1]))
    final.arrange_sheet_from_history()
    _idx, final_sc_ids = get_non_boundary_cell_ids_from_type(
        final, cell_type="SC", type_by=type_by, threshold=threshold)

    def is_sc(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v)) \
            and v <= threshold

    out = []
    last = stamps.size - 1
    for cid in final_sc_ids:
        # Mirror of the forward guard: a cell already below threshold when the
        # window opened did not revert inside it. Read off the first frame
        # rather than inferred from the walk reaching it, so a cell that was an
        # SC at t0, rose above threshold and fell back is rejected too — the
        # walk stops at the recrossing and would otherwise call that a reversal.
        if is_sc(frames[0].get(cid)):
            continue
        f = last
        while f > 0 and is_sc(frames[f - 1].get(cid)):
            f -= 1
        out.append((int(cid), float(stamps[f])))
    return out


def all_crossings(stamps, frames, threshold):
    """Every frame-to-frame threshold crossing, both directions.

    A cell is followed only while it exists in consecutive frames; a face that
    was removed simply stops contributing.
    """
    fwd, rev = [], []
    for f in range(1, stamps.size):
        before, after = frames[f - 1], frames[f]
        common = before.index.intersection(after.index)
        was_hc = before.loc[common].to_numpy(float) > threshold
        is_hc = after.loc[common].to_numpy(float) > threshold
        ids = np.asarray(common, int)
        for cid in ids[~was_hc & is_hc]:
            fwd.append((int(cid), float(stamps[f])))
        for cid in ids[was_hc & ~is_hc]:
            rev.append((int(cid), float(stamps[f])))
    return fwd, rev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-per-point", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--out", default="reverse_differentiation.csv")
    a = ap.parse_args()

    runs = read_table(os.path.join(RESULTS_DIR, "fullmodel_runs.pkl"))
    rng = np.random.default_rng(a.seed)

    rows = []
    for stage, ps in POINTS:
        g = runs[(runs["stage"] == stage) & np.isclose(runs["psigma"], ps)]
        if not len(g):
            print("  no runs for %s pT=%g" % (stage, ps))
            continue
        pick = g.iloc[rng.choice(len(g), size=min(a.runs_per_point, len(g)),
                                 replace=False)]
        for _i, r in pick.iterrows():
            name = r["model_name"]
            print("\n%s  pT = %g" % (stage, ps))
            print("  run %s" % name)
            try:
                history = load_history_file(name)
            except Exception as exc:                        # noqa: BLE001
                print("   could not load: %s: %s" % (type(exc).__name__, exc))
                continue
            stamps, frames = _delta_by_frame(history, float(r["t0"]))
            mirrored = mirrored_reverse(history, stamps, frames, a.threshold)
            fwd, rev = all_crossings(stamps, frames, a.threshold)
            rev_cells = len({c for c, _t in rev})
            print("   window t0=%.2f to %.2f, %d frames, %d cells at t0"
                  % (r["t0"], stamps[-1] if stamps.size else np.nan,
                     stamps.size, len(frames[0]) if frames else 0))
            print("   forward SC->HC : %4d events (table says %d)"
                  % (len(fwd), r["n_differentiation_events"]))
            print("   reverse HC->SC : %4d mirrored, %4d crossings in"
                  " %d distinct cell(s)" % (len(mirrored), len(rev), rev_cells))
            rows.append(dict(
                stage=stage, pT=ps, model_name=name,
                initial_array=r["initial_array"], repeat=r["repeat"],
                t0=float(r["t0"]),
                t_final=float(stamps[-1]) if stamps.size else np.nan,
                n_frames=int(stamps.size),
                n_forward_crossings=len(fwd),
                n_forward_events_table=int(r["n_differentiation_events"]),
                n_reverse_mirrored=len(mirrored),
                n_reverse_crossings=len(rev),
                n_reverse_cells=rev_cells,
                reverse_cell_ids=";".join(str(c) for c in sorted({c for c, _t in rev})),
                reverse_times=";".join("%.2f" % t for _c, t in rev)))

    if not rows:
        raise SystemExit("nothing measured")
    out = pd.DataFrame(rows)
    print("\n  %-6s %-6s %8s %10s %10s %10s"
          % ("stage", "pT", "frames", "forward", "rev mirr", "rev cross"))
    for _i, r in out.iterrows():
        print("  %-6s %-6.3f %8d %10d %10d %10d"
              % (r["stage"], r["pT"], r["n_frames"], r["n_forward_crossings"],
                 r["n_reverse_mirrored"], r["n_reverse_crossings"]))
    path = a.out if os.path.isabs(a.out) else os.path.join(RESULTS_DIR, a.out)
    out.to_csv(path, index=False)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
