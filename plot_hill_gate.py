"""Delta-production Hill gate vs psigma, with the measured face_stress overlaid.

    python plot_hill_gate.py                     # both K values, both stages
    python plot_hill_gate.py --workers 8

The gate the lateral-inhibition ODE applies to Delta production is

    gate = increasing_hill(max(face_stress - K, 0), psigma, m)
         = x^m / (psigma^m + x^m),      x = max(face_stress - K, 0)

so for a cell of a given stress the gate crosses 0.5 exactly at psigma = x. That
makes psigma directly comparable to a stress excess, and it is why the measured
per-group stresses can be drawn as VERTICAL LINES on a gate-vs-psigma plot: the
line marks the psigma at which that group's Delta production is half shut off.

Stress is measured with the PERIMETER effector set only, matching
run_model.stress_effectors, and normalised by L0 (first-frame mean perimeter)
exactly as the model does. It is sampled at two times per run:

  * t = 5           - early, while differentiation is still being decided
  * the best-matching initial frame - the per-run t0 that
    compare_full_model_differentiation_to_experiments picks by neighbour-pair
    composition, i.e. the frame scores 2 and 3 are measured from

Both the gate at the GROUP-MEAN stress (solid, pairs with the vertical line) and
the true MEAN GATE over individual cells (dotted) are drawn. They differ because
the Hill is convex-then-concave and the per-cell stress spread is wide - the
mean-stress reading is the one used for back-of-envelope selectivity arguments,
so it is worth seeing how far it is from the honest per-cell average.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from post_processing import (RESULTS_DIR, load_history_file,
                             get_non_boundary_cell_ids_from_type,
                             calc_contact_with_neighbors_from_type,
                             full_model_run_names, experimental_results_folder,
                             _exp_neighbor_pair_percentages,
                             _best_matching_frame_by_neighbor_pairs)
from face_stress_over_time import _face_stress, EFFECTOR_SETS, GROUPS

STAGES = ("E17.5", "P0")
K_VALUES = (-0.080, -0.060)
M_DEFAULT = 3
T_EARLY = 5.0
GROUP_COLOURS = {("SC", 0): "#b2182b", ("SC", 1): "#ef8a62", ("SC", 2): "#fddbc7",
                 ("HC", 0): "#2166ac", ("HC", 1): "#67a9cf", ("HC", 2): "#d1e5f0"}
STAGE_LS = {"E17.5": "-", "P0": "--"}


def _experimental_targets(stage):
    """Mean experimental HC:HC% / HC:SC% at frame 1 - the t0 matching target."""
    prefix = "E17" if stage == "E17.5" else "P0"
    hchc, hcsc = [], []
    for e in range(1, 4):
        ci = pd.read_pickle(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_cells_info_frame_1" % (prefix, e)))
        cm = np.load(os.path.join(
            experimental_results_folder, stage,
            "%s_experiment%d_contact_matrix_frame_1.npy" % (prefix, e)))
        a, b, _ = _exp_neighbor_pair_percentages(ci, cm)
        hchc.append(a); hcsc.append(b)
    return float(np.nanmean(hchc)), float(np.nanmean(hcsc))


def one_run(args):
    """Per-cell stresses per group at t=5 and at the run's best-matching frame."""
    name, tgt_hchc, tgt_hcsc, type_by, threshold = args
    try:
        history = load_history_file(name)
        first = history.retrieve(0.0)
        first.arrange_sheet_from_history(); first.geom.update_all(first)
        L0 = float(np.mean(first.face_df["perimeter"].values))
        t0, _, _ = _best_matching_frame_by_neighbor_pairs(
            history, tgt_hchc, tgt_hcsc, type_by, threshold)

        rows = []
        for when, t in (("t=5", T_EARLY), ("initial frame", float(t0))):
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
            stress = _face_stress(sheet, EFFECTOR_SETS["perimeter"],
                                  L0).reindex(labels).values
            for cell_type, nb in GROUPS:
                mask = (is_hc if cell_type == "HC" else ~is_hc) & (bins == nb)
                vals = stress[mask]
                vals = vals[np.isfinite(vals)]
                for v in vals:
                    rows.append((when, float(t0), cell_type, nb, float(v)))
        return name, rows, None
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the rest
        return name, [], "%s: %s" % (type(exc).__name__, exc)


def measure(stage, workers, type_by, threshold, arrays=None):
    names = full_model_run_names(stage, indices=arrays)
    tgt_hchc, tgt_hcsc = _experimental_targets(stage)
    print("\n%s | %d runs | target HC:HC=%.2f%% HC:SC=%.2f%%"
          % (stage, len(names), tgt_hchc, tgt_hcsc), flush=True)
    tasks = [(n, tgt_hchc, tgt_hcsc, type_by, threshold) for n in names]
    if workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one_run, tasks))
    else:
        results = [one_run(t) for t in tasks]

    rows, t0s = [], []
    for name, run_rows, err in results:
        if err:
            print("  %-52s FAILED %s" % (name[:52], err), flush=True)
            continue
        if run_rows:
            t0s.append(run_rows[0][1])
        rows.extend(run_rows)
    if not rows:
        raise SystemExit("no data for %s" % stage)
    print("  best-matching frames t0: %s  (mean %.2f)"
          % (", ".join("%.1f" % t for t in t0s), float(np.mean(t0s))), flush=True)
    df = pd.DataFrame(rows, columns=["when", "t0", "cell_type", "hc_neighbors",
                                     "stress"])
    df["stage"] = stage
    return df


def gate(x, psigma, m):
    """increasing_hill(x, psigma, m), vectorised over psigma; x >= 0 already."""
    x = np.asarray(x, float)[..., None]
    ps = np.asarray(psigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x ** m / (ps ** m + x ** m)
    return np.where(ps == 0, 1.0, np.nan_to_num(out, nan=0.0))


def plot(df, out, m=M_DEFAULT, min_cells=20):
    whens = ["t=5", "initial frame"]
    fig, axes = plt.subplots(len(K_VALUES), len(whens), figsize=(15.0, 9.5),
                             sharey=True)
    psigma = np.logspace(-4, -0.6, 400)

    summary = []
    for i, K in enumerate(K_VALUES):
        for j, when in enumerate(whens):
            ax = axes[i, j]
            for stage in STAGES:
                for cell_type, nb in GROUPS:
                    sel = df[(df.stage == stage) & (df.when == when)
                             & (df.cell_type == cell_type)
                             & (df.hc_neighbors == nb)]
                    if len(sel) < min_cells:
                        continue
                    s_mean = float(sel.stress.mean())
                    x_mean = max(s_mean - K, 0.0)
                    colour = GROUP_COLOURS[(cell_type, nb)]
                    lbl = "%s %s-%s" % (stage, cell_type,
                                        ">=2" if nb == 2 else nb)
                    # solid: gate evaluated at the group-mean stress
                    ax.plot(psigma, gate(x_mean, psigma, m).ravel(),
                            STAGE_LS[stage], color=colour, lw=2.0, label=lbl)
                    # dotted: honest mean of the per-cell gate
                    x_cells = np.maximum(sel.stress.to_numpy(float) - K, 0.0)
                    ax.plot(psigma, gate(x_cells, psigma, m).mean(axis=0),
                            ":", color=colour, lw=1.3, alpha=0.85)
                    if x_mean > 0:
                        ax.axvline(x_mean, color=colour, lw=1.0, alpha=0.55,
                                   ls=STAGE_LS[stage])
                    summary.append(dict(K=K, when=when, stage=stage,
                                        group="%s-%d" % (cell_type, nb),
                                        n=len(sel), stress=s_mean,
                                        x=x_mean, half_at=x_mean))
            ax.set_xscale("log")
            ax.set_xlim(psigma[0], psigma[-1])
            ax.set_ylim(-0.03, 1.05)
            ax.axhline(0.5, color="k", lw=0.7, ls=":", alpha=0.5)
            ax.grid(alpha=0.25, which="both", lw=0.5)
            ax.set_title("K = %.3f   |   %s" % (K, when), fontsize=11)
            ax.set_xlabel(r"$p_\sigma$")
            if j == 0:
                ax.set_ylabel("Delta-production gate\n(1 = untouched, 0 = shut off)")
            ax.legend(fontsize=6.5, ncol=2, framealpha=0.9, loc="lower left")

    fig.suptitle("Delta-production Hill gate vs $p_\\sigma$   (m = %d)\n"
                 "solid = gate at group-mean stress (vertical line = its half-max)"
                 "   |   dotted = mean of per-cell gates" % m, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n-> %s" % out)
    return pd.DataFrame(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--arrays", type=int, nargs="+", default=None)
    ap.add_argument("--type-by", dest="type_by", default="delta_level")
    ap.add_argument("--threshold", type=float, default=0.355079)
    ap.add_argument("-m", type=int, default=M_DEFAULT, help="Hill exponent")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-plot from the cached per-cell stresses instead of "
                         "re-measuring. The stresses do not depend on the Hill "
                         "exponent - only the gate curves do - so sweeping m is "
                         "free once the measurement has been done once.")
    ap.add_argument("--out", default=None,
                    help="figure path (default hill_gate_vs_psigma[_m<N>].png)")
    a = ap.parse_args()

    csv = os.path.join(RESULTS_DIR, "hill_gate_stress.csv")
    if a.from_csv:
        df = pd.read_csv(csv)
        print("re-plotting from %s (%d cells, no re-measurement)" % (csv, len(df)))
    else:
        df = pd.concat([measure(s, a.workers, a.type_by, a.threshold, a.arrays)
                        for s in STAGES], ignore_index=True)
        df.to_csv(csv, index=False)

    print("\n=== mean face_stress (perimeter effectors) per group ===")
    piv = df.pivot_table(index=["when", "stage"],
                         columns=["cell_type", "hc_neighbors"],
                         values="stress", aggfunc="mean")
    with pd.option_context("display.width", 220):
        print(piv.round(4).to_string())
    print("\n=== cells per group ===")
    with pd.option_context("display.width", 220):
        print(df.pivot_table(index=["when", "stage"],
                             columns=["cell_type", "hc_neighbors"],
                             values="stress", aggfunc="size").fillna(0)
                .astype(int).to_string())

    out = a.out or os.path.join(
        RESULTS_DIR,
        "hill_gate_vs_psigma.png" if a.m == M_DEFAULT
        else "hill_gate_vs_psigma_m%d.png" % a.m)
    summary = plot(df, out, m=a.m)
    print("\n=== psigma at which each group is HALF shut off (= mean stress - K) ===")
    with pd.option_context("display.width", 220):
        print(summary.pivot_table(index=["K", "when"], columns=["stage", "group"],
                                  values="half_at").round(4).to_string())
    print("\nper-cell stresses: %s" % csv)
