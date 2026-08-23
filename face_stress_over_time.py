"""Average face_stress OVER TIME in the full-model runs, per cell group.

    python face_stress_over_time.py                       # E17.5, every 1.0 t
    python face_stress_over_time.py --interval 0.5 --workers 6
    python face_stress_over_time.py --stage P0 --out ~/stress_P0.csv

Groups: HC and SC, each split by HC-neighbour count (0, 1, >=2). Cell type and
neighbour counts are recomputed AT EVERY SAMPLED FRAME, so cells move between
groups as they differentiate — which is the point.

``face_stress`` is exactly the quantity the lateral-inhibition ODE sees
(lateral_inhibition_model.py): ``sum_edges(edge_stress * length) / L0`` per face,
with ``L0`` the mean cell perimeter of the run's FIRST frame (the model's
``length_normalization_factor``). The Hill gate is
``increasing_hill(max(face_stress, 0), psigma)``, so psigma should sit near the
typical POSITIVE stress — that column is reported explicitly.

BOTH effector sets are computed in one pass, because they differ by more than an
order of magnitude and only one of them is what the simulation actually gates on:

  * ``perimeter``  - ContractilityPerimeterElasticity only. THIS is what
    ``run_model.stress_effectors`` is currently set to, so it is the set that
    determines psigma's effect in a stress-dependent run.
  * ``all``        - perimeter + area elasticity + bending, i.e. total mechanical
    stress. Larger and almost entirely positive (the area term dominates once the
    tissue is under tension).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from post_processing import (load_history_file, get_time_points,
                             get_non_boundary_cell_ids_from_type,
                             calc_contact_with_neighbors_from_type,
                             full_model_run_names)
from inner_ear_model import (InnerEarModel, ContractilityPerimeterElasticity,
                             BoundaryBending)
from tyssue.dynamics.effectors import FaceAreaElasticity

# In the v2 model shape index p0 = 0, so prefered_perimeter == 0 and
# ContractilityPerimeterElasticity reduces EXACTLY to face contractility,
# Gamma/2 * P^2 (verified on the stored face_df). Bending is 0 there too, so it
# contributes nothing to "all" and is kept only so the same code reads pre-v2
# runs. The names below are the v2 names for these two sets.
EFFECTOR_SETS = {
    "contractility": [ContractilityPerimeterElasticity],
    "all": [ContractilityPerimeterElasticity, FaceAreaElasticity, BoundaryBending],
}
GROUPS = [("HC", 0), ("HC", 1), ("HC", 2), ("SC", 0), ("SC", 1), ("SC", 2)]


class _Shim:
    """Exposes InnerEarModel's stress methods on a bare sheet."""
    def __init__(self, sheet):
        self.sheet = sheet
    get_edge_stress = InnerEarModel.get_edge_stress


def _face_stress(sheet, effectors, L0):
    """Per-face sum(edge_stress * length) / L0, indexed by face label."""
    edge_stress = _Shim(sheet).get_edge_stress(effectors)
    tmp = sheet.edge_df[["face", "length"]].copy()
    tmp["ws"] = np.asarray(edge_stress.values) * tmp["length"].values
    return tmp.groupby("face")["ws"].sum() / L0


def one_run(args):
    """Sample one run: returns rows of (time, effector_set, cell_type, nbrs, stress)."""
    name, interval, type_by, threshold = args
    rows = []
    try:
        history = load_history_file(name)
        stamps = get_time_points(history)
        first = history.retrieve(float(stamps[0]))
        first.arrange_sheet_from_history(); first.geom.update_all(first)
        L0 = float(np.mean(first.face_df["perimeter"].values))
        # Requested sample times; retrieve() snaps to the nearest recorded frame.
        t_max = float(stamps[-1])
        wanted = np.arange(0.0, t_max + 1e-9, interval)
        for t in wanted:
            sheet = history.retrieve(float(t))
            sheet.arrange_sheet_from_history(); sheet.geom.update_all(sheet)
            all_idx, _ = get_non_boundary_cell_ids_from_type(
                sheet, "all", type_by=type_by, threshold=threshold)
            if all_idx.size == 0:
                continue
            hc_idx, _ = get_non_boundary_cell_ids_from_type(
                sheet, "HC", type_by=type_by, threshold=threshold)
            n_hc, _ = calc_contact_with_neighbors_from_type(
                sheet, "all", "HC", type_by=type_by, threshold=threshold)
            is_hc = np.isin(all_idx, hc_idx)
            bins = np.minimum(np.asarray(n_hc), 2)
            labels = sheet.face_df.index.values[all_idx]
            # Cell SIZE per group, to test the "more stress == bigger cell"
            # reading directly (the all-effectors stress is ~97% area term, and
            # every cell sits above its preferred area, so the two should track).
            area = sheet.face_df["area"].to_numpy(float)[all_idx]
            perim = sheet.face_df["perimeter"].to_numpy(float)[all_idx]
            for cell_type, nb in GROUPS:
                m = (is_hc if cell_type == "HC" else ~is_hc) & (bins == nb)
                if m.sum():
                    rows.append((float(t), "geometry", cell_type, nb, int(m.sum()),
                                 float(np.nanmean(area[m])), float(np.nanmean(perim[m])),
                                 float("nan")))
            for set_name, effectors in EFFECTOR_SETS.items():
                stress = _face_stress(sheet, effectors, L0).reindex(labels).values
                for cell_type, nb in GROUPS:
                    mask = (is_hc if cell_type == "HC" else ~is_hc) & (bins == nb)
                    vals = stress[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        rows.append((float(t), set_name, cell_type, nb,
                                     vals.size, float(vals.mean()), float(vals.std()),
                                     float(np.mean(vals > 0))))
        return name, rows, None
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the sweep
        return name, rows, "%s: %s" % (type(exc).__name__, exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="E17.5", choices=["E17.5", "P0"])
    ap.add_argument("--interval", type=float, default=1.0,
                    help="sample every N units of SIMULATION time (default 1.0)")
    ap.add_argument("--arrays", type=int, nargs="+", default=None,
                    help="array indices (default 0..9)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel runs; keep low if a fit is using the machine")
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=0.355079,
                    help="delta threshold separating HC from SC (the full model's "
                         "atoh_sensitivity)")
    ap.add_argument("--run-prefix", dest="run_prefix", default="fullmodel_v2",
                    help="which full-model run set to read (default the v2 runs)")
    ap.add_argument("--out", default=None, help="CSV path (default ~/face_stress_<stage>.csv)")
    a = ap.parse_args()

    names = full_model_run_names(a.stage, indices=a.arrays, run_prefix=a.run_prefix)
    out = a.out or os.path.expanduser("~/face_stress_%s.csv" % a.stage)
    print("face_stress over time | %s | %d runs | every %.3g t | %d worker(s)"
          % (a.stage, len(names), a.interval, a.workers), flush=True)

    tasks = [(n, a.interval, a.type_by, a.threshold) for n in names]
    if a.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(one_run, tasks))
    else:
        results = [one_run(t) for t in tasks]

    rows = []
    for name, run_rows, err in results:
        print("  %-52s %s" % (name[:52], "OK (%d rows)" % len(run_rows) if not err
                              else "FAILED " + err), flush=True)
        rows.extend(run_rows)
    if not rows:
        raise SystemExit("no data collected")

    # "geometry" rows carry mean AREA in `mean` and mean PERIMETER in `std`.
    df = pd.DataFrame(rows, columns=["time", "effectors", "cell_type", "hc_neighbors",
                                     "n", "mean", "std", "frac_positive"])
    # Pool the per-run group means at each sampled time (n-weighted).
    df["wsum"] = df["mean"] * df["n"]
    df["wpos"] = df["frac_positive"] * df["n"]
    g = df.groupby(["time", "effectors", "cell_type", "hc_neighbors"], as_index=False).agg(
        n=("n", "sum"), wsum=("wsum", "sum"), wpos=("wpos", "sum"),
        std=("std", "mean"), n_runs=("mean", "size"))
    g["mean"] = g["wsum"] / g["n"]
    g["frac_positive"] = g["wpos"] / g["n"]
    g = g.drop(columns=["wsum", "wpos"])
    g.to_csv(out, index=False)
    print("\nwrote %s (%d rows)" % (out, len(g)))

    geo = g[g.effectors == "geometry"]
    if len(geo):
        print("\n=== mean CELL AREA by group over time (preferred area A0 = 0.4657) ===")
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            print(geo.pivot_table(index="time", columns=["cell_type", "hc_neighbors"],
                                  values="mean").round(4))
        print("\n=== group SIZES (n cells, pooled over runs) ===")
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            print(geo.pivot_table(index="time", columns=["cell_type", "hc_neighbors"],
                                  values="n").fillna(0).astype(int))
    for set_name in EFFECTOR_SETS:
        sub = g[g.effectors == set_name]
        print("\n=== %s effectors: mean face_stress by group over time ===" % set_name.upper())
        piv = sub.pivot_table(index="time", columns=["cell_type", "hc_neighbors"],
                              values="mean")
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            print(piv.round(4))
        pos = sub[sub["mean"] > 0]
        print("  positive-stress fraction (all groups, last frame): %.0f%%"
              % (100 * sub[sub.time == sub.time.max()]["frac_positive"].mean()))
        if len(pos):
            print("  psigma scale (Hill half-max ~ typical POSITIVE stress): "
                  "median of positive group-means = %.4f" % pos["mean"].median())


if __name__ == "__main__":
    main()
