"""Re-run the BEST E17.5 mechanical-fit point locally, base + ablation, with a
DENSE history (a snapshot every 0.01 of simulation time) for morphology figures
and movies.

    python run_best_point.py            # array 0
    python run_best_point.py 3          # array 3
    python run_best_point.py 0 1 2      # several

Output folders: ``bestpoint_E17.5_a<i>`` and ``bestpoint_E17.5_a<i>_abl``
(the stage prefix comes from ``STAGE`` below, so P0 later is a one-line change).

WARNING - SIZE. save_interval=0.01 with dt=0.01 records EVERY solver step:
~2500 frames over t_end=25, and a frame of these sheets is ~5 MB, so expect
roughly **12 GB per run** (~25 GB for a base+ablation pair). A movie needs only
~100 frames; raise SAVE_INTERVAL to 0.25 for ~500 MB if you don't specifically
need every step.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import run_model as rm

STAGE = "E17.5"                           # folder prefix: bestpoint_<STAGE>_a<i>

# best E17.5 fit (objective 1.771; eval 1 of the 70-call re-fit)
GAMMA_SC, GAMMA_HC_RATIO, ALPHA_HC = 0.2461, 1.0, 1.00
HC_SHAPE_INDEX, SC_SHAPE_INDEX = 4.86, 5.72
PREFERRED_AREA = 0.593 * np.pi / 4        # 0.4657
BENDING = 0.02
BASE_QST, ABLATION_QST = 0.03, 0.02
ABLATED_CELLS = (337, 304, 65, 114)
T_END, DT = 25, 0.01
SAVE_INTERVAL = 0.01                      # dense; see the size warning above


def one(index):
    suffix = "E17" if STAGE == "E17.5" else "P0"
    initial = "random_periodic_array%d_for_%s" % (index, suffix)
    li = rm._li_levels_kwargs_for_initial_sheet(initial)
    threshold = float(rm._load_saved_threshold(initial))
    common = dict(shape_index=0.0, hc_shape_index=HC_SHAPE_INDEX,
                  sc_shape_index=SC_SHAPE_INDEX, bending=BENDING, line_tension=None,
                  preferred_area_override=PREFERRED_AREA, atoh_sensitivity=threshold,
                  no_differentiation=True, end_on_steady_state=True,
                  t_end=T_END, dt=DT, save_interval=SAVE_INTERVAL)
    base = rm._strip_results_prefix(rm.run(
        GAMMA_SC, GAMMA_HC_RATIO, ALPHA_HC, 0, initial_sheet_name=initial,
        name="bestpoint_%s_a%d" % (STAGE, index), quasi_static_threshold=BASE_QST,
        **li, **common))
    # The ablation run forks from the BASE run's archive (not the raw array), so
    # it starts from the same relaxed state the fit used.
    abl = rm._strip_results_prefix(rm.run(
        GAMMA_SC, GAMMA_HC_RATIO, ALPHA_HC, 0, initial_sheet_name=base,
        name="bestpoint_%s_a%d_abl" % (STAGE, index), ablated_cells=list(ABLATED_CELLS),
        quasi_static_threshold=ABLATION_QST, **common))
    return index, base, abl


if __name__ == "__main__":
    indices = [int(a) for a in sys.argv[1:]] or [0]
    print("best %s point:" % STAGE, " gammaSC=%.4f alphaHC=%.2f hc_p0=%.2f sc_p0=%.2f A0=%.4f"
          % (GAMMA_SC, ALPHA_HC, HC_SHAPE_INDEX, SC_SHAPE_INDEX, PREFERRED_AREA))
    print("arrays %s | save_interval=%.3g -> ~%d frames, ~%.0f GB per run"
          % (indices, SAVE_INTERVAL, T_END / SAVE_INTERVAL,
             T_END / SAVE_INTERVAL * 5e-3), flush=True)
    n = min(len(indices), os.cpu_count() or 1)
    results = ([one(i) for i in indices] if n <= 1 else
               list(ProcessPoolExecutor(max_workers=n).map(one, indices)))
    print()
    for index, base, abl in results:
        print("array %d:\n  base     %s\n  ablation %s" % (index, base, abl))
