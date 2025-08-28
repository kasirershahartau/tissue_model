import sys
sys.path.insert(1, 'C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_model\\tyssue\\tyssue')
from tyssue.dynamics.effectors import AbstractEffector, units, to_nd
from tyssue.dynamics.planar_gradients import area_grad as area_grad_2d
import numpy as np
import pandas as pd

def _get_centroid_diff_matrix(face_df):
    location_x = face_df.x.to_numpy()
    location_y = face_df.y.to_numpy()
    # Creating distance matrix such that diffx[i,j] holds cx_i-cx_j and  diffy[i,j] holds cy_i - cy_j
    return location_x.reshape((1, len(location_x))) - location_x.reshape((len(location_x), 1)),\
           location_y.reshape((1, len(location_y))) - location_y.reshape((len(location_y), 1))

def _get_distance_matrix(face_df):
    '''
    # Creating distance matrix such that distance[i,j] holds the distance between i and j faces (Rij from
    Roei's paper)
    '''
    diff_x, diff_y = _get_centroid_diff_matrix(face_df)
    # Creating distance matrix such that distance[i,j] holds the distance between i and j faces
    return np.sqrt(diff_x**2 + diff_y**2)


def _get_geometric_mean_of_all_pairs(arr):
    '''
    Returns a matrix such that M[i,j] = sqrt(arr[i]*arr[j])
    '''
    return np.sqrt(arr.reshape((len(arr), 1)).dot(arr.reshape((1, len(arr)))))


def _get_repulsion_strength_matrix(face_df):
    '''
     Creating repulsion matrix such that repulsion_strength[i,j] = sigma_ij from Roei's paper
    '''
    repulsion_strength = face_df.repulsion.values * face_df.is_alive.values
    return _get_geometric_mean_of_all_pairs(repulsion_strength)


def _get_repulsion_distance_matrix(face_df):
    '''
    Creating repulsion distance matrix such that repulsion_distance[i,j] = D_ij from Roei's paper
    '''
    repulsion_distance = face_df.repulsion_distance.to_numpy()
    return _get_geometric_mean_of_all_pairs(repulsion_distance)

def _get_repulsion_energy_matrix(eptm):
    distance = _get_repulsion_distance_matrix(eptm.face_df)
    repulsion_distance = _get_repulsion_distance_matrix(eptm.face_df)
    repulsion_strength = _get_repulsion_strength_matrix(eptm.face_df)
    distance[distance == 0] = np.inf  # To avoid division by zero
    return repulsion_strength * (repulsion_distance / distance) ** eptm.repulsion_exp

def centroid_gradient(sheet):

    coords = sheet.coords
    area = sheet.face_df.loc[sheet.edge_df.face.values, "area"]
    inv_area = 1.0/(12*area)
    face_pos = sheet.edge_df[["f" + c for c in coords]]
    srce_pos = sheet.edge_df[["s" + c for c in coords]]
    trgt_pos = sheet.edge_df[["t" + c for c in coords]]

    # Area gradient
    grad_a_srce, grad_a_trgt = area_grad_2d(sheet)

    # Derivative of centroid x coordinate
    grad_cx_srce = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_cx_srce["gx"] = (2 * srce_pos["sx"] * trgt_pos["ty"] + trgt_pos["tx"]*(trgt_pos["ty"] - srce_pos["sy"])
                          - face_pos["fx"] * grad_a_srce["gx"]) * sheet.edge_df["nz"]
    grad_cx_srce["gy"] = -((trgt_pos["tx"] + srce_pos["sx"]) * trgt_pos["tx"]
                           + face_pos["fx"] * grad_a_srce["gy"]) * sheet.edge_df["nz"]
    grad_cx_trgt = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_cx_trgt["gx"] = - (2 * trgt_pos["tx"] * srce_pos["sy"] + srce_pos["sx"]*(srce_pos["sy"] - trgt_pos["ty"])
                            + face_pos["fx"] * grad_a_trgt["gx"]) * sheet.edge_df["nz"]
    grad_cx_trgt["gy"] = ((srce_pos["sx"] + trgt_pos["tx"]) * srce_pos["sx"]
                          - face_pos["fx"] * grad_a_trgt["gy"]) * sheet.edge_df["nz"]

    # Derivative of centroid y coordinate
    grad_cy_srce = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_cy_srce["gx"] = ((trgt_pos["ty"] + srce_pos["sy"]) * trgt_pos["ty"]
                          - face_pos["fy"] * grad_a_srce["gx"]) * sheet.edge_df["nz"]
    grad_cy_srce["gy"] = - (2 * srce_pos["sy"] * trgt_pos["tx"] + trgt_pos["ty"] * (trgt_pos["tx"] - srce_pos["sx"])
                            + face_pos["fy"] * grad_a_srce["gy"]) * sheet.edge_df["nz"]
    grad_cy_trgt = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_cy_trgt["gx"] = - ((srce_pos["sy"] + trgt_pos["ty"]) * srce_pos["sy"]
                            + face_pos["fy"] * grad_a_trgt["gx"]) * sheet.edge_df["nz"]
    grad_cy_trgt["gy"] = (2 * trgt_pos["ty"] * srce_pos["sx"] + srce_pos["sy"] * (srce_pos["sx"] - trgt_pos["tx"])
                          - face_pos["fy"] * grad_a_trgt["gy"]) * sheet.edge_df["nz"]

    grad_cx_srce = to_nd(inv_area, 2) * grad_cx_srce
    grad_cx_trgt = to_nd(inv_area, 2) * grad_cx_trgt
    grad_cy_srce = to_nd(inv_area, 2) * grad_cy_srce
    grad_cy_trgt = to_nd(inv_area, 2) * grad_cy_trgt

    return grad_cx_srce, grad_cx_trgt, grad_cy_srce, grad_cy_trgt

class FaceRepulsion(AbstractEffector):
    dimensionless = False
    dimensions = units.energy
    magnitude = "face_repulsion"
    label = "Face repulsion"
    element = "face"
    specs = {
        "face": {
            "is_alive": 1,
            "repulsion": 1.0,
            "x": 1,
            "y": 1
        },
        "egde": {
            "sub_area": 1.0/6.0
        }
    }

    spatial_ref = "repulsion_distance", units.length

    @staticmethod
    def get_nrj_norm(specs):
        return specs["face"]["repulsion"]

    @staticmethod
    def energy(eptm):
        return 0.5*np.sum(_get_repulsion_energy_matrix(eptm))

    @staticmethod
    def gradient(eptm):
        repulsion_energy_matrix = _get_repulsion_energy_matrix(eptm)
        distance_matrix = _get_distance_matrix(eptm.face_df)
        distance_matrix[distance_matrix == 0] = np.inf  # to avoid division by zero
        centroid_x_diff, centroid_y_diff = _get_centroid_diff_matrix(eptm.face_df)
        face_prefix_matrix = - eptm.repulsion_exp * repulsion_energy_matrix/distance_matrix**2
        face_prefix_cx = np.sum(face_prefix_matrix*centroid_x_diff, axis=0)
        face_prefix_cy = np.sum(face_prefix_matrix*centroid_y_diff, axis=0)
        edge_prefix_cx = eptm.edge_df.is_active.values * face_prefix_cx[eptm.edge_df.face.values]
        edge_prefix_cy = eptm.edge_df.is_active.values * face_prefix_cy[eptm.edge_df.face.values]

        grad_cx_srce, grad_cx_trgt, grad_cy_srce, grad_cy_trgt = centroid_gradient(eptm)

        grad_srce = grad_cx_srce.multiply(edge_prefix_cx, axis=0) + grad_cy_srce.multiply(edge_prefix_cy, axis=0)
        grad_trgt = grad_cx_trgt.multiply(edge_prefix_cx, axis=0) + grad_cy_trgt.multiply(edge_prefix_cy, axis=0)

        grad_srce.columns = ["g" + u for u in eptm.coords]
        grad_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_srce, grad_trgt