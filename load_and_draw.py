import os
import numpy as np
from matplotlib import pyplot as plt
from tyssue import HistoryHdf5
from tyssue.draw.plt_draw import create_gif
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel

if __name__ == "__main__":
    # Name of existing model results to load
    load_name = "stress_dependent_on_random_0_mechanosensitivity2_8"
    save_name = "stress_dependent_on_random_0__mechanosensitivity2_8_repressor_vals"
    number_of_frames_to_save = 100

    load_path = os.path.join("results", load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    initial_sheet = history.retrieve(0)
    last_time_point = np.max(history.time_stamps)
    final_sheet = history.retrieve(last_time_point)

    save_path = load_path = os.path.join("results", load_name, save_name)
    fig1, ax1 = InnerEarModel.draw_sheet(initial_sheet, number_faces=False, number_edges=False, number_vertices=False,
                                         arrange_sheet=True)
    plt.savefig("%s_initial.png" % save_path)
    fig2, ax2 = InnerEarModel.draw_sheet(final_sheet, number_faces=False, number_edges=False, number_vertices=False,
                                         arrange_sheet=True)
    plt.savefig("%s_finale.png" % save_path)
    create_gif(history, "%s.gif" % save_path, num_frames=number_of_frames_to_save,
               draw_func=InnerEarModel.draw_sheet, arrange_sheet=True)