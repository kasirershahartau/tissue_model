import os, sys
import numpy as np
from matplotlib import pyplot as plt
from tyssue import HistoryHdf5
from tyssue.draw.plt_draw import create_gif
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
sys.path.insert(0, r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_image_processing\tissue_analyzing_tool")

experimental_results_folder = r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data"
E17_number_of_HC_neighbors_file_name = r"E17.5 differentiating cells number of HC neighbors.npy"
E17_contact_length_with_HC_neighbors_file_name = r"E17.5 differentiating cells contacts length with HC.npy"
P0_number_of_HC_neighbors_file_name = r"P0 differentiating cells number of HC neighbors.npy"
P0_contact_length_with_HC_neighbors_file_name = r"P0 differentiating cells contact length with HC.npy"


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
        create_gif(history, "%s.gif" % save_path, num_frames=number_of_frames_to_save,
                   draw_func=gif_draw_func)
    return 0

def find_non_boundary_cells(time_point_data):
    boundary_cells = np.unique(time_point_data.edge_df.loc[time_point_data.edge_df.opposite < 0, "face"])
    neighbors_of_boundary_cells = np.unique(time_point_data.edge_df.face[time_point_data.edge_df.opposite.isin(boundary_cells)])
    exclude_cells = np.union1d(boundary_cells, neighbors_of_boundary_cells)
    face_ids = time_point_data.face_df.index
    non_boundary_cells = np.setdiff1d(face_ids, exclude_cells)
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

def calc_contact_with_neighbors_from_type(time_point_data, cell_type='all', neighbor_type='all',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None):
    face_ids = find_non_boundary_cells(time_point_data)
    if only_for_these_cells is not None:
        face_ids = np.intersect1d(face_ids, only_for_these_cells)
    type_data = time_point_data.face_df.loc[face_ids, type_by]
    if threshold is None:
        threshold = (np.max(type_data) + np.min(type_data))/2
        print("Using calculated threshold = %f"%threshold)

    if HC_above_threshold:
        is_HC = type_data > threshold
    else:
        is_HC = type_data < threshold

    if cell_type == "all":
        relevant_cells = face_ids
    elif cell_type == "HC":
        relevant_cells = face_ids[is_HC]
    elif cell_type == "SC":
        relevant_cells = face_ids[~is_HC]
    else:
        raise "not implemented cell type"

    if neighbor_type == "all":
        relevant_neighbors = face_ids
    elif neighbor_type == "HC":
        relevant_neighbors = face_ids[is_HC]
    elif neighbor_type == "SC":
        relevant_neighbors = face_ids[~is_HC]
    else:
        raise "not implemented neighbors type"
    contact_matrix = time_point_data.get_contact_matrix()
    relevant_contacts = contact_matrix[np.ix_(relevant_cells, relevant_neighbors)]
    contact_length = relevant_contacts.sum(axis=1)
    binary_relevant_contacts = (relevant_contacts > 0).astype(int)
    number_of_neighbors = binary_relevant_contacts.sum(axis=1)
    return number_of_neighbors, contact_length

def calc_contacts_for_last_time_point(load_name, cell_type='HC', neighbor_type='HC',
                                          type_by='atoh_level', threshold=None, HC_above_threshold=True,
                                          only_for_these_cells=None, save=True):
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


def load_experimental_results(stage, type):
    if stage == "E17.5":
        if type == "number of neighbors":
            return np.load(os.path.join(experimental_results_folder, E17_number_of_HC_neighbors_file_name)).astype(int)
        elif type == "contact length":
            return np.load(os.path.join(experimental_results_folder, E17_contact_length_with_HC_neighbors_file_name))
        else:
            raise "Not implemented for type %s"%type
    elif stage == "P0":
        if type == "number of neighbors":
            return np.load(os.path.join(experimental_results_folder, P0_number_of_HC_neighbors_file_name)).astype(int)
        elif type == "contact length":
            return np.load(os.path.join(experimental_results_folder, P0_contact_length_with_HC_neighbors_file_name))
        else:
            raise "Not implemented for type %s"%type
    else:
        raise "Not implemented for stage %s"%stage

def calc_vectorial_distance(dist1, dist2, maximal_n=None):
    if maximal_n is not None:
        dist1 = np.clip(dist1, a_min=None, a_max=maximal_n)
        dist2 = np.clip(dist2, a_min=None, a_max=maximal_n)
    else:
        maximal_n = max(np.max(dist1), np.max(dist2))
    hist1 = np.bincount(dist1, minlength=maximal_n + 1)/dist1.size
    hist2 = np.bincount(dist2, minlength=maximal_n + 1)/dist2.size
    return np.sqrt(np.sum((hist1 - hist2)**2))

def compare_to_experimental_results(model_name, experimental_stage, results_type="number of neighbors",
                                    cell_type='HC', neighbor_type='HC', type_by='atoh_level', threshold=None,
                                    max_number_of_neighbors=2, plot=False):
    # Right now it is implemented only for the number of HC neighbors of HCs
    experimental_results = load_experimental_results(experimental_stage, results_type)
    model_results, _ = calc_contacts_for_last_time_point(model_name, cell_type=cell_type, neighbor_type=neighbor_type,
                                          type_by=type_by, threshold=threshold)
    experimental_results = np.clip(experimental_results, a_min=None, a_max=max_number_of_neighbors)
    model_results = np.clip(model_results, a_min=None, a_max=max_number_of_neighbors)
    if experimental_stage == "E17.5":
        color = "cyan"
        edge_color = "blue"
    elif experimental_stage == "P0":
        color = "pink"
        edge_color = "red"
    else:
        raise "Not implemented for stage %s"%experimental_stage
    if plot:
        experimental_hist = np.bincount(experimental_results, minlength=max_number_of_neighbors + 1)
        model_hist = np.bincount(model_results, minlength=max_number_of_neighbors + 1)
        experimental_percent = 100*(experimental_hist/experimental_results.size)
        model_percent = 100*(model_hist/model_results.size)
        fig, ax = plt.subplots()
        ax.bar(np.arange(experimental_percent.size)-0.125, experimental_percent, width=0.25, color=color,
               edgecolor=edge_color, label="Experiment %s"%experimental_stage)
        ax.bar(np.arange(model_percent.size) + 0.125, model_percent, width=0.25, color="white",
               edgecolor=edge_color, label="Model %s"%model_name)
        ax.set_xlabel('HC number of with HC neighbors')
        ax.set_ylabel('Frequency')
        ax.legend()
    return calc_vectorial_distance(model_results, experimental_results, maximal_n=max_number_of_neighbors)

if __name__ == "__main__":
    gammaSC_vals = [0.1, 0.2, 0.3, 0.4]
    psigma_vals = [2.0]
    best_E17_model = ""
    best_E17_dist = np.inf
    best_P0_model = ""
    best_P0_dist = np.inf
    E17_dists = []
    P0_dists = []
    lonely_SCs = []
    for gammaSC in gammaSC_vals:
        for psigma in psigma_vals:
            load_name = "stress_dependent_on_random_0_psigma-%.1f_gammaSC-%.1f_patoh-0.31"%(psigma, gammaSC)

            E17_comparison = compare_to_experimental_results(load_name, "E17.5",type_by="delta_level",
                                                             threshold=0.31, max_number_of_neighbors=2, plot=True)
            E17_dists.append(E17_comparison)
            print("Comparison with E17.5 distance:%f"%E17_comparison)
            if E17_comparison < best_E17_dist:
                best_E17_dist = E17_comparison
                best_E17_model = load_name
            P0_comparison = compare_to_experimental_results(load_name, "P0", type_by="delta_level",
                                                             threshold=0.31, max_number_of_neighbors=2, plot=True)
            P0_dists.append(P0_comparison)
            print("Comparison with P0 distance:%f"%P0_comparison)
            if P0_comparison < best_P0_dist:
                best_P0_dist = P0_comparison
                best_P0_model = load_name
            number_of_SC_neighbors, _ = calc_contacts_for_last_time_point(load_name, cell_type='SC',
                                                                          neighbor_type='HC',
                                                                          type_by="delta_level",
                                                                          threshold=0.31)
            number_of_SC_with_no_HC_neighbors = np.count_nonzero(number_of_SC_neighbors==0)
            percent_of_SC_with_no_HC_neighbors = 100*number_of_SC_with_no_HC_neighbors/number_of_SC_neighbors.size
            print("SC without HC neighbors for model %s: %f which is %f percent of all SCs"%(load_name,
                                                                                               number_of_SC_with_no_HC_neighbors,
                                                                                               percent_of_SC_with_no_HC_neighbors))
            lonely_SCs.append(percent_of_SC_with_no_HC_neighbors)
    print("Best E17.5 model is %s with score %f"%(best_E17_model, best_E17_dist))
    print("Best P0 model is %s with score %f"%(best_P0_model, best_P0_dist))
    fig, ax = plt.subplots()
    ax.plot(gammaSC_vals, E17_dists, "b*-", label="E17.5 distances")
    ax.plot(gammaSC_vals, P0_dists, "r*-", label="P0 distances")
    ax.set_xlabel("gammaSC")
    ax.set_ylabel("Vector distance between experimental and model results")
    ax.legend()
    fig1, ax1 = plt.subplots()
    ax1.plot(gammaSC_vals, lonely_SCs, "b*-")
    ax1.set_xlabel("gammaSC")
    ax1.set_ylabel("SCs with no HC neighbrs")
    plt.show()
    number_of_neighbors, contact_length = calc_contacts_for_last_time_point(load_name, cell_type='SC',
                                                                            neighbor_type='HC',
                                                                            type_by="delta_level")
    fig1, ax1 = plt.subplots()
    ax1.hist(contact_length)
    ax1.set_xlabel('SC contact with HC neighbors')
    ax1.set_ylabel('Frequency')
    fig2, ax2 = plt.subplots()
    ax2.hist(number_of_neighbors)
    ax2.set_xlabel('SC number of with HC neighbors')
    ax2.set_ylabel('Frequency')
    plt.show()
    # psigma = 8.0
    # gammaSC = 0.5
    # load_name = "stress_dependent_on_random_0_psigma-8.0_gammaSC-0.5_patoh-0.31"
    # save_name = load_name + "_delta"
    # redraw(load_name, save_name,movie=True, maximal_number_of_frames_to_save=100, color_by="delta",
    #        maximal_level=find_maximal_level_final_frame(load_name, "delta_level"))