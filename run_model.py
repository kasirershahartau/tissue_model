import os
import numpy as np
from tyssue import HistoryHdf5
from tyssue.draw.plt_draw import create_gif
from matplotlib import pyplot as plt
from virtual_sheet import VirtualSheet
from inner_ear_model import InnerEarModel
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility


def initialize_sheet(nx, ny, distx=1, disty=1, max_bond_length=0.5, min_bond_length=0.05):
    sheet = VirtualSheet.planar_sheet_2d(
        'basic2D',  # a name or identifier for this sheet
        nx=nx,  # approximate number of cells on the x axis
        ny=ny,  # approximae number of cells along the y axis
        distx=distx,  # distance between 2 cells along x
        disty=disty  # distance between 2 cells along y
    )
    sheet.set_maximal_bond_length(max_bond_length)  # was 0.2
    sheet.set_minimal_bond_length(min_bond_length)  # was 0.05
    sheet.initiate_edge_order()
    sheet.add_virtual_vertices()
    return sheet


def load_sheet_from_file(initial_sheet_name):
    history = HistoryHdf5.from_archive("%s.hf5" % initial_sheet_name, eptm_class=VirtualSheet)
    last_time_point = np.max(history.time_stamps)
    return history.retrieve(last_time_point)


if __name__ == "__main__":

    # Sheet Parameters
    initial_sheet_name = ""
    load_lateral_inhibition_data_from_file = True
    name = "stress_dependent_small_check"
    max_bond_length = 0.5
    min_bond_length = 0.05

    # In case initial_sheet_name == "", creating a new sheet with the following parameters
    nx = 7
    ny = 7
    distx = 1
    disty = 1

    # Model version select
    random_sensitivity = False
    aging_sensitivity = False
    only_differentiation = False
    no_differentiation = False
    contact_dependent_differentiation = True
    notch_inhibition = False
    stress_dependent = True
    divisions = False
    intercalations = True
    delaminations = True
    ablated_cells = []
    random_forces = False
    quasi_static = False
    quasi_static_threshold=0.001

    # Model Parameters
    # General parameters
    t_end = 10
    dt = 0.001
    movie_frames = 10

    # 2D vertex related parameters
    effectors = [FaceContractility, FaceAreaElasticity]
    tension = {('HC', 'HC'): 0.05,
               ('HC', 'SC'): 0.05,
               ('SC', 'SC'): 0.05
               }
    preferred_area = {'HC': 1/(4*np.pi),
                      'SC': 1/(4*np.pi)}
    contractility = {'HC': 4.,
                     'SC': 2.}

    repulsion = {'HC': 0.001,
                 'SC': 0.}
    repulsion_distance = {'HC': 2.0,
                          'SC': 0.}
    repulsion_exponent = 7.
    elasticity = {'HC': 5.,
                  'SC': 1.}

    # Topological events related parameters
    division_area = 1.3
    intercalation_length = 0.04
    delamination_area = 0.1
    delamination_rate = 1.2
    viscosity = 0.5

    # Lateral Inhibition parameters
    differentiation_threshold = 0.5
    l = 3  # decreasing Hill exponent
    m = 3  # increasing Hill exponent
    betaN = 1  # maximum production rate Notch for classical model
    betaD = 1  # maximum production rate Delta for classical model
    notch_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    repressor_sensitivity = 2  # how much Delta production is sensitive to repressor level (1 / (1 + sensitivity * repressor)^l)
    atoh_sensitivity = 100  # how much Atoh1 production is sensitive to repressor level (1 / (1 + sensitivity * repressor)^l)
    notch_sensitivity = 0.5  # how much Repressor production is sensitive to signaling level (signaling^m / (sensitivity^m + signaling^m))
    delta_repressor_degradation_ratio = 1  # notch degradation rate / repressor degradation rate
    notch_delta_production_ratio = 1
    sensitivity_aging_rate = 10  # Notch sensitivity change rate (for aging sensitivity version)
    mechanosensitivity = 10  # Sensitivity to mechanical stress (for stress dependent version)
    stress_effectors = [FaceContractility]  # effectors to calculate stress (for stress dependent version)

    if not stress_dependent:
        mechanosensitivity = 0


    results_dir = os.path.join("results", name)
    if os.path.exists(results_dir):
        overwrite = input("overwriting existing results, are you sure? (y/n)")
        if overwrite not in ["y", "Y", "yes", "Yes"]:
            exit(0)
    else:
        os.mkdir(results_dir)


    #  Saving model  parameters
    params_file = os.path.join(os.path.join("results", name, name + "_parameters.txt"))
    variables = globals().copy().items()
    with open(params_file, "w") as f:
        for var_name, var_value in variables:
            # Exclude built-in and special variables (e.g., those starting with '__')
            if not var_name.startswith("__") and not callable(var_value) and not isinstance(var_value, type(os)):
                f.write(f"{var_name}: {repr(var_value)}\n")

    initial_sheet_name = os.path.join("results", initial_sheet_name, initial_sheet_name)
    name = os.path.join("results", name, name)

    # Load or initialize sheet
    if os.path.isfile("%s.hf5" % initial_sheet_name):
        sheet = load_sheet_from_file(initial_sheet_name)
    else:
        sheet = initialize_sheet(nx, ny, distx, disty, max_bond_length, min_bond_length)
    if load_lateral_inhibition_data_from_file:
        lateral_inhibition_data_file = "%s_notch_delta_levels.pkl" % initial_sheet_name
    else:
        lateral_inhibition_data_file = None
    sheet.set_maximal_bond_length(max_bond_length)
    sheet.set_minimal_bond_length(min_bond_length)

    # Initialize model
    inner = InnerEarModel(sheet, tension=tension, repulsion=repulsion, repulsion_distance=repulsion_distance,
                          repulsion_exp=repulsion_exponent, preferred_area=preferred_area, contractility=contractility,
                          elasticity=elasticity, differentiation_threshold=differentiation_threshold,
                          random_sensitivity=random_sensitivity,
                          saved_notch_delta_levels_file=lateral_inhibition_data_file,
                          l=l, m=m, betaN=betaN, betaD=betaD, inhibition=notch_inhibition,
                          notch_repressor_degradation_ratio=notch_repressor_degradation_ratio,
                          repressor_sensitivity=repressor_sensitivity, atoh_sensitivity=atoh_sensitivity,
                          delta_repressor_degradation_ratio=delta_repressor_degradation_ratio,
                          notch_delta_production_ratio=notch_delta_production_ratio,
                          stress_effectors=stress_effectors, mechanosensitivity=mechanosensitivity)

    fig1, ax1 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=False)
    plt.savefig("%s_initial.png" % name)
    history = inner.simulate(t_end=t_end, dt=dt, only_differentiation=only_differentiation,
                             random_forces=random_forces, aging_sensitivity=aging_sensitivity,
                             no_differentiation=no_differentiation,
                             contact_dependent_differentiation=contact_dependent_differentiation, divisions=divisions,
                             intercalations=intercalations, delaminations=delaminations, ablated_cells=ablated_cells,
                             sensitivity_aging_rate=sensitivity_aging_rate,
                             division_area=division_area, intercalation_length=intercalation_length,
                             delamination_area=delamination_area, delamination_rate=delamination_rate,
                             viscosity=viscosity, effectors=effectors, quasi_static=quasi_static,
                             quasi_static_threshold=quasi_static_threshold)
    if os.path.isfile("%s.hf5" % name):
        os.remove("%s.hf5" % name)
    history.to_archive("%s.hf5" % name)
    inner.save_notch_delta("%s_notch_delta_levels.pkl" % name)
    fig2, ax2 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=False)
    plt.savefig("%s_finale.png" % name)
    inner.save_sheet_labels_to_numpy(inner.sheet, path="%s_labels.npy" % name)
    inner.save_contact_matrix_to_numpy(inner.sheet, path="%s_contact_matrix.npy" % name)
    inner.save_face_data_to_df(inner.sheet, path="%s_cells_info.pkl" % name)
    create_gif(history, "%s.gif" % name, num_frames=movie_frames, draw_func=inner.draw_sheet)