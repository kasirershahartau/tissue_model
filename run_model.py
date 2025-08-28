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
    name = "contact_dependent_diff_only"
    nx = 5
    ny = 5
    distx = 1
    disty = 1
    max_bond_length = 0.5
    min_bond_length = 0.05

    # Model version select
    random_sensitivity = False
    aging_sensitivity = False
    only_differentiation = True
    no_differentiation = False
    contact_dependent_differentiation = True
    random_forces = False
    notch_inhibition = False
    tension_dependent = False

    # Model Parameters
    # General parameters
    t_end = 10
    dt = 0.001
    movie_frames = 100

    # 2D vertex related parameters
    effectors = [LineTension, FaceContractility, FaceAreaElasticity]
    tension = {('HC', 'HC'): 0.05,
               ('HC', 'SC'): 0.05,
               ('SC', 'SC'): 0.05
               }
    preferred_area = {'HC': 1,
                      'SC': 1}
    contractility = {'HC': 0.4,
                     'SC': 0.1}

    repulsion = {'HC': 0.001,  # 0.001
                 'SC': 0}
    repulsion_distance = {'HC': 2.0,
                          'SC': 0}
    repulsion_exponent = 7
    elasticity = {'HC': 5,
                  'SC': 1}

    # Topological events related parameters
    division_area = 1.3
    intercalation_length = 0.04
    delamination_area = 0.1
    delamination_rate = 1.2
    viscosity = 3

    # Lateral Inhibition parameters
    differentiation_threshold = 0.5
    l = 3
    m = 3
    mu = 10
    rho = 10
    xhi = 10
    betaN = 3.9
    betaD = 3.9
    betaR = 194
    kt = 5
    gammaR = 60
    sensitivity_aging_rate = 10
    mechanosensitivity = 1
    tension_effectors = [LineTension, FaceContractility]

    if not tension_dependent:
        mechanosensitivity = 0

    # Load or initialize sheet
    if os.path.isfile("%s.hf5" % initial_sheet_name):
        sheet = load_sheet_from_file(initial_sheet_name)
    else:
        sheet = initialize_sheet(nx, ny, distx, disty, max_bond_length, min_bond_length)

    # Initialize model
    inner = InnerEarModel(sheet, tension=tension, repulsion=repulsion, repulsion_distance=repulsion_distance,
                          repulsion_exp=repulsion_exponent, preferred_area=preferred_area, contractility=contractility,
                          elasticity=elasticity, differentiation_threshold=differentiation_threshold,
                          random_sensitivity=random_sensitivity,
                          saved_notch_delta_levels_file="%s_notch_delta_levels.pkl" % initial_sheet_name)

    fig1, ax1 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=False)
    plt.savefig("%s_initial.png" % name)
    history = inner.simulate(t_end=t_end, dt=dt, notch_inhibition=notch_inhibition, only_differentiation=only_differentiation,
                             random_forces=random_forces, aging_sensitivity=aging_sensitivity,
                             no_differentiation=no_differentiation,
                             contact_dependent_differentiation=contact_dependent_differentiation,
                             l=l, m=m, mu=mu, rho=rho, xhi=xhi, betaN=betaN, betaD=betaD, betaR=betaR, kt=kt,
                             gammaR=gammaR, sensitivity_aging_rate=sensitivity_aging_rate,
                             division_area=division_area, intercalation_length=intercalation_length,
                             delamination_area=delamination_area, delamination_rate=delamination_rate,
                             viscosity=viscosity, effectors=effectors, mechanosensitivity=mechanosensitivity,
                             tension_effectors=tension_effectors
                             )
    if os.path.isfile("%s.hf5" % name):
        os.remove("%s.hf5" % name)
    history.to_archive("%s.hf5" % name)
    inner.save_notch_delta("%s_notch_delta_levels.pkl" % name)
    fig2, ax2 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=False)
    plt.savefig("%s_finale.png" % name)
    inner.save_sheet_labels_to_numpy(inner.sheet, path="%s_labels.npy" % name)
    inner.save_contact_matrix_to_numpy(inner.sheet, path="%s_contact_matrix.npy" % name)
    inner.save_face_data_to_df(inner.sheet, path="%s_cells_info.npy" % name)
    create_gif(history, "%s.gif" % name, num_frames=movie_frames, draw_func=inner.draw_sheet)