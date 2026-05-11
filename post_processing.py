import os, sys
import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from tyssue import HistoryHdf5
from tyssue.draw.plt_draw import create_gif
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_image_processing\tissue_analyzing_tool")

experimental_results_folder = r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data"
E17_number_of_HC_neighbors_file_name = r"E17.5 differentiating cells number of HC neighbors.npy"
E17_contact_length_with_HC_neighbors_file_name = r"E17.5 differentiating cells contacts length with HC.npy"
E17_HC_roundness_file_name = r"E17.5 +24h HC roundness.npy"
E17_SC_roundness_file_name = r"E17.5 +24h SC roundness.npy"
P0_number_of_HC_neighbors_file_name = r"P0 differentiating cells number of HC neighbors.npy"
P0_contact_length_with_HC_neighbors_file_name = r"P0 differentiating cells contact length with HC.npy"
P0_HC_roundness_file_name = r"P0 +24h HC roundness.npy"
P0_SC_roundness_file_name = r"P0 +24h SC roundness.npy"


def redraw(load_name, save_name, movie=True, maximal_number_of_frames_to_save=100, color_by="atoh", maximal_level=1):

    load_path = os.path.join("results", load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    initial_sheet = history.retrieve(0)
    last_time_point = np.max(history.time_stamps)
    number_of_time_points = np.unique(history.time_stamps).size
    final_sheet = history.retrieve(last_time_point)
    number_of_frames_to_save = min(number_of_time_points, maximal_number_of_frames_to_save)

    save_path = os.path.join("results", load_name, save_name)
    static_draw_func = InnerEarModel.get_draw_sheet_method(number_faces=True, number_edges=False, number_vertices=False,
                                         arrange_sheet=True, color_by=color_by, maximal_level=maximal_level)
    fig1, ax1 = static_draw_func(initial_sheet)
    plt.savefig("%s_initial.png" % save_path)
    fig2, ax2 =static_draw_func(final_sheet)
    plt.savefig("%s_finale.png" % save_path)
    if movie:
        gif_draw_func = InnerEarModel.get_draw_sheet_method(number_faces=False, number_edges=False, number_vertices=False,
                                             arrange_sheet=True, color_by=color_by, maximal_level=maximal_level)
        create_gif(history, os.path.join(os.getcwd(), "%s.gif" % save_path), num_frames=number_of_frames_to_save,
                   draw_func=gif_draw_func)
    return 0

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
    final_sheet = InnerEarModel.arrange_sheet_from_history(final_sheet)
    face_ids = find_non_boundary_cells(final_sheet)
    level = final_sheet.face_df.loc[face_ids, type_by]
    return np.max(level)

def get_non_boundary_cell_ids_from_type(time_point_data, cell_type='all',
                                     type_by='atoh_level', threshold=None,
                                     HC_above_threshold=True, only_for_these_cells=None):
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
    return time_point_data.face_df.loc[relevant_cells, "id"].values

def calc_contact_with_neighbors_from_type(time_point_data, cell_type='all', neighbor_type='all',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None):

    relevant_cells = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=cell_type,
                                                      type_by=type_by, threshold=threshold,
                                                      HC_above_threshold=HC_above_threshold,
                                                      only_for_these_cells=only_for_these_cells)
    if neighbor_type==cell_type:
        relevant_neighbors = relevant_cells
    else:
        relevant_neighbors = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=neighbor_type,
                                                                 type_by=type_by, threshold=threshold,
                                                                 HC_above_threshold=HC_above_threshold,
                                                                 only_for_these_cells=only_for_these_cells)
    contact_matrix = time_point_data.get_contact_matrix()
    relevant_contacts = contact_matrix[np.ix_(relevant_cells, relevant_neighbors)]
    contact_length = relevant_contacts.sum(axis=1)
    binary_relevant_contacts = (relevant_contacts > 0).astype(int)
    number_of_neighbors = binary_relevant_contacts.sum(axis=1)
    return number_of_neighbors, contact_length

def calc_roundness_for_type(time_point_data, cell_type='all',
                            type_by='atoh_level', threshold=None,
                            HC_above_threshold=True, only_for_these_cells=None):
    relevant_cells = get_non_boundary_cell_ids_from_type(time_point_data, cell_type=cell_type,
                                                         type_by=type_by, threshold=threshold,
                                                         HC_above_threshold=HC_above_threshold,
                                                         only_for_these_cells=only_for_these_cells)
    roundness = time_point_data.get_face_roundness()
    relevant_values = roundness.loc[relevant_cells].values
    return relevant_values


def calc_roundness_for_last_time_point(load_name, cell_type='HC',
                                       type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                       only_for_these_cells=None):
    load_path = os.path.join("results", load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    last_time_point = np.max(history.time_stamps)
    final_sheet = history.retrieve(last_time_point)
    final_sheet = InnerEarModel.arrange_sheet_from_history(final_sheet)
    res = calc_roundness_for_type(final_sheet, cell_type=cell_type,
                                                type_by=type_by, threshold=threshold,
                                                HC_above_threshold=HC_above_threshold,
                                                only_for_these_cells=only_for_these_cells)
    np.save("%s results %s roundness" % (load_name, cell_type), res)
    return res

def calc_contacts_for_last_time_point(load_name, cell_type='HC', neighbor_type='HC',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None):
    load_path = os.path.join("results", load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    last_time_point = np.max(history.time_stamps)
    final_sheet = history.retrieve(last_time_point)
    final_sheet = InnerEarModel.arrange_sheet_from_history(final_sheet)
    res = calc_contact_with_neighbors_from_type(final_sheet, cell_type=cell_type, neighbor_type=neighbor_type,
                                          type_by=type_by, threshold=threshold, HC_above_threshold=HC_above_threshold,
                                          only_for_these_cells=only_for_these_cells)
    np.save("%s results %s with %s neighbors"%(load_name, cell_type, neighbor_type), res)
    return res

def calc_area_change_after_ablation(load_name, ablated_cells=[], end_frame=3, type_by='atoh_level', threshold=None,
                                    HC_above_threshold=True):
    load_path = os.path.join("results", load_name, load_name)
    history = HistoryHdf5.from_archive("%s.hf5" % load_path, eptm_class=VirtualSheet)
    initial_sheet = history.retrieve(0)
    initial_sheet = InnerEarModel.arrange_sheet_from_history(initial_sheet)
    HC_neighbors_of_ablated = []
    SC_neighbors_of_ablated = []
    for ablated in ablated_cells:
        neighbors = initial_sheet.get_neighbors(ablated)
        neighbors = np.setdiff1d(neighbors, ablated, assume_unique=True)
        HC_neighbors_of_ablated = np.union1d(HC_neighbors_of_ablated,
                                             get_non_boundary_cell_ids_from_type(initial_sheet,
                                                                                 cell_type="HC",
                                                                                 type_by=type_by,
                                                                                 threshold=threshold,
                                                                                 HC_above_threshold=HC_above_threshold,
                                                                                 only_for_these_cells=neighbors))
        SC_neighbors_of_ablated = np.union1d(SC_neighbors_of_ablated,
                                             get_non_boundary_cell_ids_from_type(initial_sheet,
                                                                                 cell_type="SC",
                                                                                 type_by=type_by,
                                                                                 threshold=threshold,
                                                                                 HC_above_threshold=HC_above_threshold,
                                                                                 only_for_these_cells=neighbors))
    final_sheet = history.retrieve(end_frame)
    final_sheet = InnerEarModel.arrange_sheet_from_history(final_sheet)
    initial_face_area = initial_sheet.get_face_area()
    initial_HC_area_next_to_ablated = initial_face_area.loc[HC_neighbors_of_ablated].values
    initial_SC_area_next_to_ablated = initial_face_area.loc[SC_neighbors_of_ablated].values
    final_face_area = final_sheet.get_face_area()
    final_HC_area_next_to_ablated = final_face_area.loc[HC_neighbors_of_ablated].values
    final_SC_area_next_to_ablated = final_face_area.loc[SC_neighbors_of_ablated].values
    HC_area_ratio = final_HC_area_next_to_ablated / initial_HC_area_next_to_ablated
    SC_area_ratio = final_SC_area_next_to_ablated / initial_SC_area_next_to_ablated
    return HC_area_ratio, SC_area_ratio


def load_experimental_results(stage, type):
    if stage == "E17.5":
        if type == "HC number of HC neighbors":
            return np.load(os.path.join(experimental_results_folder, E17_number_of_HC_neighbors_file_name)).astype(int)
        elif type == "HC contact length with HC":
            return np.load(os.path.join(experimental_results_folder, E17_contact_length_with_HC_neighbors_file_name))
        elif type == "HC roundness":
            return np.load(os.path.join(experimental_results_folder, E17_HC_roundness_file_name))
        elif type == "SC roundness":
            return np.load(os.path.join(experimental_results_folder, E17_SC_roundness_file_name))
        else:
            raise "Not implemented for type %s"%type
    elif stage == "P0":
        if type == "HC number of HC neighbors":
            return np.load(os.path.join(experimental_results_folder, P0_number_of_HC_neighbors_file_name)).astype(int)
        elif type == "HC contact length with HC":
            return np.load(os.path.join(experimental_results_folder, P0_contact_length_with_HC_neighbors_file_name))
        elif type == "HC roundness":
            return np.load(os.path.join(experimental_results_folder, P0_HC_roundness_file_name))
        elif type == "SC roundness":
            return np.load(os.path.join(experimental_results_folder, P0_SC_roundness_file_name))
        else:
            raise "Not implemented for type %s"%type
    else:
        raise "Not implemented for stage %s"%stage

def calc_vectorial_distance(dist1, dist2, maximal_n=None, continous=False):
    if dist1.size < 2 or dist2.size < 2:
        return None
    if continous:
        kde1 = gaussian_kde(dist1.reshape(1, -1))
        kde2 = gaussian_kde(dist2.reshape(1, -1))

        # Shared evaluation grid
        grid = np.linspace(
            min(dist1.min(), dist2.min()),
            max(dist1.max(), dist2.max()),
            500
        )

        p = kde1(grid)
        q = kde2(grid)

        # Normalize
        p /= p.sum()
        q /= q.sum()

        return jensenshannon(p, q)
    else:
        if maximal_n is not None:
            dist1 = np.clip(dist1, a_min=None, a_max=maximal_n)
            dist2 = np.clip(dist2, a_min=None, a_max=maximal_n)
        else:
            maximal_n = max(np.max(dist1), np.max(dist2))
        hist1 = np.bincount(dist1, minlength=maximal_n + 1)/dist1.size
        hist2 = np.bincount(dist2, minlength=maximal_n + 1)/dist2.size
        return np.sqrt(np.sum((hist1 - hist2)**2))

def compare_to_experimental_results(model_name, experimental_stage, results_type="HC number of HC neighbors",
                                    type_by='atoh_level', threshold=None,
                                    max_number_of_neighbors=2, plot=False):
    # Right now it is implemented only for the number of HC neighbors of HCs and roundness
    experimental_results = load_experimental_results(experimental_stage, results_type)
    if experimental_stage == "E17.5":
        color = "cyan"
        edge_color = "blue"
    elif experimental_stage == "P0":
        color = "pink"
        edge_color = "red"
    if results_type == "HC number of HC neighbors":
        model_results, _ = calc_contacts_for_last_time_point(model_name, cell_type="HC", neighbor_type="HC",
                                              type_by=type_by, threshold=threshold)
        experimental_results = np.clip(experimental_results, a_min=None, a_max=max_number_of_neighbors)
        model_results = np.clip(model_results, a_min=None, a_max=max_number_of_neighbors)
        experimental_hist = np.bincount(experimental_results, minlength=max_number_of_neighbors + 1)
        model_hist = np.bincount(model_results, minlength=max_number_of_neighbors + 1)
        experimental_percent = 100 * (experimental_hist / experimental_results.size)
        model_percent = 100 * (model_hist / model_results.size)
        if plot:
            fig, ax = plt.subplots()
            ax.bar(np.arange(experimental_percent.size) - 0.125, experimental_percent, width=0.25, color=color,
                   edgecolor=edge_color, label="Experiment %s" % experimental_stage)
            ax.bar(np.arange(model_percent.size) + 0.125, model_percent, width=0.25, color="white",
                   edgecolor=edge_color, label="Model %s" % model_name)
            ax.set_xlabel('HC number of with HC neighbors')
            ax.set_ylabel('Frequency')
            ax.legend()
        continues = False

    elif results_type == "HC roundness":
        model_results = calc_roundness_for_last_time_point(model_name, cell_type="HC",
                                                          type_by=type_by, threshold=threshold)
        continues = True
    elif results_type == "SC roundness":
        model_results = calc_roundness_for_last_time_point(model_name, cell_type="SC",
                                                          type_by=type_by, threshold=threshold)
        continues = True

    else:
        raise "Not implemented for stage %s"%experimental_stage
    print("Experimental average = %f\nModel average = %f"%(np.average(experimental_results), np.average(model_results)))

    return calc_vectorial_distance(model_results, experimental_results, maximal_n=max_number_of_neighbors,
                                   continous=continues)

if __name__ == "__main__":
    # gammaSC_vals = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # psigma_vals = [8.0]
    # # gammaSC_vals = [0.5]
    # # psigma_vals = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    # gammaSC_vals_for_E17 = gammaSC_vals.copy()
    # psigma_vals_for_E17 = psigma_vals.copy()
    # gammaSC_vals_for_P0 = gammaSC_vals.copy()
    # psigma_vals_for_P0 = psigma_vals.copy()
    #
    #
    # best_E17_model = ""
    # best_E17_dist = np.inf
    # best_P0_model = ""
    # best_P0_dist = np.inf
    # E17_dists = []
    # P0_dists = []
    # lonely_SCs = []
    # for gammaSC in gammaSC_vals:
    #     for psigma in psigma_vals:
    #         load_name = "stress_dependent_on_random_0_psigma-%.1f_gammaSC-%.1f_patoh-0.31"%(psigma, gammaSC)
    #         E17_comparison = compare_to_experimental_results(load_name, "E17.5", results_type="HC roundness",
    #                                                          type_by="delta_level", threshold=0.31,
    #                                                          max_number_of_neighbors=2, plot=True)
    #         if E17_comparison is not None:
    #             E17_dists.append(E17_comparison)
    #             print("Comparison with E17.5 distance:%f"%E17_comparison)
    #             if E17_comparison < best_E17_dist:
    #                 best_E17_dist = E17_comparison
    #                 best_E17_model = load_name
    #         else:
    #             gammaSC_vals_for_E17.remove(psigma)
    #         P0_comparison = compare_to_experimental_results(load_name, "P0", results_type="HC roundness", type_by="delta_level",
    #                                                          threshold=0.31, max_number_of_neighbors=2, plot=True)
    #         if P0_comparison is not None:
    #             P0_dists.append(P0_comparison)
    #             print("Comparison with P0 distance:%f"%P0_comparison)
    #             if P0_comparison < best_P0_dist:
    #                 best_P0_dist = P0_comparison
    #                 best_P0_model = load_name
    #         else:
    #             gammaSC_vals_for_P0.remove(psigma)
            # number_of_SC_neighbors, _ = calc_contacts_for_last_time_point(load_name, cell_type='SC',
            #                                                               neighbor_type='HC',
            #                                                               type_by="delta_level",
            #                                                               threshold=0.31)
            # number_of_SC_with_no_HC_neighbors = np.count_nonzero(number_of_SC_neighbors==0)
            # percent_of_SC_with_no_HC_neighbors = 100*number_of_SC_with_no_HC_neighbors/number_of_SC_neighbors.size
            # print("SC without HC neighbors for model %s: %f which is %f percent of all SCs"%(load_name,
            #                                                                                    number_of_SC_with_no_HC_neighbors,
            #                                                                                    percent_of_SC_with_no_HC_neighbors))
            # lonely_SCs.append(percent_of_SC_with_no_HC_neighbors)
    # print("Best E17.5 model is %s with score %f"%(best_E17_model, best_E17_dist))
    # print("Best P0 model is %s with score %f"%(best_P0_model, best_P0_dist))
    # fig, ax = plt.subplots()
    # ax.plot(gammaSC_vals_for_E17, E17_dists, "b*-", label="E17.5 distances")
    # ax.plot(gammaSC_vals_for_P0, P0_dists, "r*-", label="P0 distances")
    # ax.set_xlabel("gammaSC")
    # ax.set_ylabel("Vector distance between experimental and model results")
    # ax.legend()
    # fig1, ax1 = plt.subplots()
    # ax1.plot(gammaSC_vals, lonely_SCs, "b*-")
    # ax1.set_xlabel("gammaSC")
    # # ax1.set_ylabel("SCs with no HC neighbrs")
    # ax1.set_ylabel("SCs with no HC neighbrs")
    # plt.show()
    # number_of_neighbors, contact_length = calc_contacts_for_last_time_point(load_name, cell_type='SC',
    #                                                                         neighbor_type='HC',
    #                                                                         type_by="delta_level")
    # fig1, ax1 = plt.subplots()
    # ax1.hist(contact_length)
    # ax1.set_xlabel('SC contact with HC neighbors')
    # ax1.set_ylabel('Frequency')
    # fig2, ax2 = plt.subplots()
    # ax2.hist(number_of_neighbors)
    # ax2.set_xlabel('SC number of with HC neighbors')
    # ax2.set_ylabel('Frequency')
    # plt.show()
    # psigma = 8.0
    # gammaSC = 0.5
    # load_name = "stress_dependent_on_random_0_psigma-8.0_gammaSC-0.5_patoh-0.31"

    # load_name = "stress_dependent_on_psigma-8.0_gammaSC-0.5_ablation-211-149-116-v2"
    # HC_res, SC_res = calc_area_change_after_ablation(load_name,ablated_cells=[211,149,116], end_frame=4)
    # print("HC avg area change:%f"%np.average(HC_res))
    # print("SC avg area change:%f"%np.average(SC_res))

    gammaSC_vals = [0.01]
    psigma_vals = [0.0]
    gammaHC_ratio_vals = [2.0, 4.0, 6.0, 8.0, 10.0, 20.0]
    alphaHC_ratio_vals = [1.0]
    results = []
    for gammaSC in gammaSC_vals:
        for psigma in psigma_vals:
            for gammaHC_ratio in gammaHC_ratio_vals:
                for alphaHC_ratio in alphaHC_ratio_vals:
                    name = "stress_dependent_on_random_0_psigma-%.1f_gammaSC-%.1f_pR-0.35_gammaHC_ratio-%.1f_alphaHC_ratio-%.1f" % (
                    psigma, gammaSC, gammaHC_ratio, alphaHC_ratio)
                    load_name = name
                    try:
                        # save_name = "delta"
                        # redraw(load_name,
                        #        save_name,movie=True, maximal_number_of_frames_to_save=100, color_by="delta",
                        #        maximal_level=find_maximal_level_final_frame(load_name, "delta_level"))
                        # save_name = "atoh"
                        # redraw(load_name,
                        #        save_name, movie=True, maximal_number_of_frames_to_save=100, color_by="atoh",
                        #        maximal_level=find_maximal_level_final_frame(load_name, "atoh_level"))
                        HC_res = calc_roundness_for_last_time_point(load_name, cell_type='HC',
                                           type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                           only_for_these_cells=None)
                        SC_res = calc_roundness_for_last_time_point(load_name, cell_type='SC',
                                                                    type_by='atoh_level', threshold=None,
                                                                    HC_above_threshold=True,
                                                                    only_for_these_cells=None)
                        HC_avg = np.average(HC_res)
                        SC_avg = np.average(SC_res)
                        ratio_avg = HC_avg/SC_avg
                        results.append((gammaSC, gammaHC_ratio, alphaHC_ratio, HC_avg, SC_avg, ratio_avg))
                    except Exception as e:
                        print(e)
                        continue
    results = np.array(results)
    gammaSC_arr = results[:, 0]
    gammaHC_arr = results[:, 1]
    alphaHC_arr = results[:, 2]
    HC_roundness_arr = results[:, 3]
    SC_roundness_arr = results[:, 4]
    roundness_ratio_arr = results[:, 5]

    # plt.plot(gammaHC_arr, HC_roundness_arr, "*", label="HC roundness")
    # plt.plot(gammaHC_arr, SC_roundness_arr, "*", label="SC roundness")
    # plt.plot(gammaHC_arr, roundness_ratio_arr, "*", label="HC:SC roundness ratio")
    # df = pd.DataFrame({
    #     "gammaSC": gammaSC_arr,
    #     "gammaHC_ratio": gammaHC_arr,
    #     "alphaHC_ratio": alphaHC_arr,
    #     "HC roundness": HC_roundness_arr,
    #     "SC roundness": HC_roundness_arr,
    #     "HC/SC roundness ratio": roundness_ratio_arr,
    # })
    #
    # fig1 = px.scatter_3d(
    #     df,
    #     x="gammaSC", y="gammaHC_ratio", z="alphaHC_ratio",
    #     color="HC roundness",
    #     color_continuous_scale="Viridis",
    #     size_max=10,
    #     title="3D Heat Map of HC Roundness"
    # )
    #
    # fig2 = px.scatter_3d(
    #     df,
    #     x="gammaSC", y="gammaHC_ratio", z="alphaHC_ratio",
    #     color="SC roundness",
    #     color_continuous_scale="Viridis",
    #     size_max=10,
    #     title="3D Heat Map of SC Roundness"
    # )
    #
    # fig3 = px.scatter_3d(
    #     df,
    #     x="gammaSC", y="gammaHC_ratio", z="alphaHC_ratio",
    #     color="HC/SC roundness ratio",
    #     color_continuous_scale="Viridis",
    #     size_max=10,
    #     title="3D Heat Map of HC/SC Roundness ratio"
    # )
    #
    # fig1.show()
    # fig2.show()
    # fig3.show()
