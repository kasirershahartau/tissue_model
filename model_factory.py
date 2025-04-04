###### Creating varius types of models ##########
# This file is a derivative of tyssue project with few chages:
# 1. Adding virtual vertices to allow for round apical morphology
# 2. Adding differential features by cell types
# 3. Adding external forces
import os.path
import pickle
import random
import sys
#sys.path.insert(1, 'C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_model\\tyssue')
import tyssue
from tyssue import Sheet, config, History
from tyssue import PlanarGeometry as geom
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
from tyssue.topology.base_topology import add_vert, collapse_edge
from tyssue.topology.sheet_topology import cell_division, type1_transition
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet import apoptosis
from tyssue.behaviors.sheet.actions import merge_vertices
from tyssue.solvers import QSSolver
from tyssue.solvers.viscous import EulerSolver
from tyssue.draw.plt_draw import create_gif
from tyssue.dynamics import model_factory
from tyssue.dynamics.effectors import AbstractEffector, units, to_nd, LineTension, FaceAreaElasticity, FaceContractility
from tyssue.dynamics.planar_gradients import area_grad as area_grad_2d
from tyssue import HistoryHdf5

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class VirtualSheet(Sheet):
    """ An epithelium tissue with virtual vertices, to allow for rounded apical morphology"""

    def __init__(self, identifier, datasets, specs=None, coords=None, maximal_bond_length=0.1,
                 minimal_bond_length=0.05):
        """
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        datasets : dictionary of dataframes
            The keys correspond to the different geometrical elements
            constituting the epithelium:

            * `vert` contains a dataframe of vertices,
            * `edge` contains a dataframe of *oriented* half-edges between vertices,
            * `face` contains a dataframe of polygonal faces enclosed by half-edges,
            * `cell` contains a dataframe of polyhedral cells delimited by faces,
        virtual_vert_it: The number of virtual vertices adding iterations. On each iteration each edge is splitted
         into 2. For example: 2 iterations add 3 virtual vertices to each edge.

        """
        super().__init__(identifier, datasets, specs, coords)
        self.update_specs({"vert": {"is_virtual": int(0)}, "edge": {"order": int(0)}})
        self.maximal_bond_length = maximal_bond_length
        self.minimal_bond_length = minimal_bond_length
        self.geom = geom

    def initiate_edge_order(self):
        self.sanitize(trim_borders=True, order_edges=True)
        self.reset_index(order=True)
        self.geom.update_all(self)
        self.get_opposite()
        face_list = self.edge_df.face.to_numpy()
        edges_order = np.zeros((len(face_list,)))
        counter = 0
        current_face = -1
        for idx in range(edges_order.size):
            if face_list[idx] != current_face:
                current_face = face_list[idx]
                counter = 0
            counter += 1
            edges_order[idx] = counter
        self.edge_df.loc[:, 'order'] = edges_order.astype(int)

    def set_maximal_bond_length(self, length):
        self.maximal_bond_length = length

    def set_minimal_bond_length(self, length):
        self.minimal_bond_length = length

    def order_all_edges(self):
        for face in self.face_df.index.values:
            self.order_edges(face)

    def order_edges(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number)
        self.edge_df.loc[edges.index, "order"] = 0
        current_edge = edges.iloc[0]
        current_edge_order = 1
        while self.edge_df.at[current_edge.name, "order"] < 1:
            self.edge_df.at[current_edge.name, "order"] = current_edge_order
            edge_trgt = current_edge.trgt
            current_edge = edges.query("srce == %d" %edge_trgt).iloc[0]
            current_edge_order += 1

    def check_edge_order(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number).loc[:,["order", "srce", "trgt"]]
        edges.sort_values(["order"], inplace=True)
        first_srce = -1
        current_trgt = -1
        for index, row in edges.iterrows():
            if first_srce < 0:
                first_srce = row.srce
            if current_trgt > 0 and current_trgt != row.srce:
                return False
            current_trgt = row.trgt
        return row.trgt == first_srce

    def check_all_edge_order(self):
        for face in self.face_df.index.values:
            if not self.check_edge_order(face):
                print("wrong order in face %d" %face)
                return False
        return True



    def add_virtual_vertices(self):
        long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
        np.random.shuffle(long)
        while long.size > 0:
            edge_ind = long[0]
            edge_order = self.edge_df.at[edge_ind, "order"]
            edge_face = self.edge_df.at[edge_ind, "face"]
            new_vert, new_edge, new_opposite_edge = add_vert(self, edge_ind)
            self.vert_df.at[new_vert, "is_virtual"] = 1
            self.edge_df.at[edge_ind, "length"] /= 2
            self.edge_df.at[new_edge, "length"] /= 2
            increase_order = self.edge_df.query("face == %d and order > %d" %(edge_face, edge_order))
            self.edge_df.at[new_edge, "order"] = edge_order + 1
            self.edge_df.loc[increase_order.index, "order"] += 1
            opposite = int(self.edge_df.loc[edge_ind, "opposite"])
            if opposite >= 0:
                self.edge_df.at[opposite, "length"] /= 2
            if new_opposite_edge is None:
                self.edge_df.at[new_edge, "opposite"] = -1
            else:
                opposite_order = self.edge_df.at[opposite, "order"]
                opposite_face = self.edge_df.at[opposite, "face"]
                self.edge_df.at[new_edge, "opposite"] = new_opposite_edge
                self.edge_df.at[new_opposite_edge, "opposite"] = new_edge
                self.edge_df.at[new_opposite_edge, "length"] /= 2
                increase_order = self.edge_df.query("face == %d and order > %d" % (opposite_face, opposite_order))
                self.edge_df.at[opposite, "order"] = opposite_order + 1
                self.edge_df.loc[increase_order.index, "order"] += 1
            long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
            np.random.shuffle(long)
        self.edge_df.index.name = 'edge'
        self.geom.update_all(self)
        # self.reset_index(order=False)
        self.edge_df.sort_values(["face", "order"], inplace=True)
        self.get_opposite()
        if not self.check_all_edge_order():
            print("bug in adding virtual vertices")

    def remove_virtual_vertex(self, edge_id):
        srce_idx = self.edge_df.loc[edge_id].srce
        trgt_idx = self.edge_df.loc[edge_id].trgt
        srce = self.vert_df.loc[srce_idx]
        trgt = self.vert_df.loc[trgt_idx]
        if srce.is_virtual == 1 and trgt.is_virtual != 1:  # if only one is virtual, collapse to the real vertex
            self.vert_df.loc[srce_idx, self.coords] = self.vert_df.loc[trgt_idx, self.coords]
        elif trgt.is_virtual == 1 and srce.is_virtual != 1:
            self.vert_df.loc[trgt_idx, self.coords] = self.vert_df.loc[srce_idx, self.coords]
        # involved_faces.append(self.edge_df.at[short[0], "face"])
        # opposite_edge = self.edge_df.at[short[0], "opposite"]
        # if opposite_edge > 0:
        #     involved_faces.append(self.edge_df.at[opposite_edge, "face"])
        collapse_edge(self, edge_id, allow_two_sided=False, reindex=True)
        return 0

    def remove_virtual_vertices(self):
        # involved_faces = []
        short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
        if short.size > 0:
            short = short[self.is_virtual_edge(short)]
        np.random.shuffle(short)
        while short.size > 0:
            self.remove_virtual_vertex(short[0])
            short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
            if short.size > 0:
                short = short[self.is_virtual_edge(short)]
            np.random.shuffle(short)
        # for face in np.unique(involved_faces):
        #     self.order_edges(face)
        # sheet.edge_df.sort_values(["face", "order"], inplace=True)
        # sheet.get_opposite()
        # self.geom.update_all(self)
        if not self.check_all_edge_order():
            print("bug in removing virtual vertices")
        return 0

    def is_virtual_edge(self, edge_indices):
        """
        Checks if an edge contains a virtual vertex
        """
        if hasattr(edge_indices, "__len__"):
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual.to_numpy() == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual.to_numpy() == 1
            return np.logical_or(srce_is_virtual, trgt_is_virtual)
        else:
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual == 1
            return srce_is_virtual or trgt_is_virtual

    def get_update_virtual_vertices_function(self):
        def update_virtual_vertices(sheet, manager):
            sheet.remove_virtual_vertices()
            sheet.add_virtual_vertices()
            manager.append(update_virtual_vertices)
            return
        return update_virtual_vertices

    def get_neighbors(self, face, elem="face"):
        face_edges = self.edge_df.query("face == %d" % face)
        opposite_edges = face_edges.opposite.to_numpy()
        neighbors = self.edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
        return np.unique(neighbors)

# Helping methods for FaceRepulsion effector #
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

class EmptyAffector(AbstractEffector):
    dimensionless = False
    dimensions = units.energy
    magnitude = "empty"
    label = "Empty"
    element = "face"
    spatial_ref = "distance", units.length

    @staticmethod
    def get_nrj_norm(specs):
        return specs["face"]["area"]

    @staticmethod
    def energy(eptm):
        return 0

    @staticmethod
    def gradient(eptm):

        return (pd.DataFrame.from_dict({"gx":[0]*eptm.edge_df.shape[0],"gy":[0]*eptm.edge_df.shape[0]}),
               pd.DataFrame.from_dict({"gx": [0] * eptm.edge_df.shape[0], "gy": [0] * eptm.edge_df.shape[0]}))

class RandomAffector(AbstractEffector):
    dimensionless = False
    dimensions = units.energy
    magnitude = "random"
    label = "Random"
    element = "edge"
    spatial_ref = "distance", units.length

    @staticmethod
    def get_nrj_norm(specs):
        return specs["edge"]["length"]

    @staticmethod
    def energy(eptm):
        return np.random.rand()

    @staticmethod
    def gradient(eptm):
        gx = np.random.rand(eptm.edge_df.shape[0])
        gy = np.random.rand(eptm.edge_df.shape[0])
        return (pd.DataFrame.from_dict({"gx":gx,"gy":gy}),
               pd.DataFrame.from_dict({"gx":-gx, "gy":-gy}))


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


class InnerEarModel:
    """
    A wrapping class for epithilium model, for easy execution of inner ear model simulations.
    """
    # TODO:
    # 1. Add model creating function with line-tension, area elasticity, contractility, and repulsion forces
    #    between HCs. All forces should be cell-type dependent.
    # 2. Add event manager creating function with intercalation, division, delamination and differentiation.
    # 3. Add simulation routine that handles energy minimisation, differentiation and other active events.

    def __init__(self, sheet, tension=None, repulsion=None, repulsion_distance=None, repulsion_exp=7,
                 preferred_area=None, contractility=None, elasticity=None,
                 differentiation_threshold=0.5, random_sensitivity=False, saved_notch_delta_levels_file=None):
        # Setting class constants
        self.CELL_TYPES = ['SC', 'HC']
        self.DIMENSIONS = ['2D']


        # Setting default behavior
        if tension is None:
            tension = {('HC', 'HC'): 0.05,
                       ('HC', 'SC'): 0.05,
                       ('SC', 'SC'): 0.05
                       }
        if preferred_area is None:
            preferred_area = {'HC': 1,
                              'SC': 1}
        if contractility is None:
            contractility = {'HC': 0.15, #0.4
                             'SC': 0.1}
        if repulsion is None:
            repulsion = {'HC': 0.001, #0.001
                         'SC': 0}
        if repulsion_distance is None:
            repulsion_distance = {'HC': 2.0,
                                  'SC': 0}
        if elasticity is None:
            elasticity = {'HC': 1, #5
                          'SC': 1}
        if ('SC', 'HC') in tension:
            tension[('HC', 'SC')] = tension[('SC', 'HC')]
        elif ('HC', 'SC') in tension:
            tension[('SC', 'HC')] = tension[('HC', 'SC')]
        self.sheet = self.arrange_sheet_from_history(sheet)
        sheet.repulsion_exp = repulsion_exp
        self.face_params = {"contractility": contractility, "repulsion": repulsion,
                            "repulsion_distance": repulsion_distance, "prefered_area": preferred_area,
                            "prefered_vol": preferred_area,
                            "area_elasticity": elasticity}
        self.differentiation_threshold = differentiation_threshold
        self.edge_params = {"line_tension": tension}
        self.dimensionality = '2D'
        specs = self.get_specs_2d()
        self.sheet.update_specs(specs)
        self.initialize_notch_delta(random_sensitivity, saved_levels_file_path=saved_notch_delta_levels_file)
        active_edges = (self.sheet.edge_df.opposite.values >= 0).astype(int)
        self.sheet.edge_df.loc[:, 'is_active'] = active_edges
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.srce.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.trgt.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.face_df.loc[:,"id"] = self.sheet.face_df.index
        self.sheet.active_verts = np.where(self.sheet.vert_df.is_active.values)[0]
        self.update_cell_type_parameters(self.sheet.face_df.delta_level)
        self.sheet.initiate_edge_order()
        self.sheet.order_all_edges()
        return

    @staticmethod
    def arrange_sheet_from_history(sheet):
        if 'vert' in sheet.vert_df.columns:
            sheet.vert_df.drop('time', inplace=True, axis=1)
            if np.isnan(sheet.vert_df['vert']).any():
                sheet.vert_df.set_index('index', inplace=True)
            else:
                sheet.vert_df.set_index('vert', inplace=True)
        if 'edge' in sheet.edge_df.columns:
            sheet.edge_df.set_index('edge', inplace=True)
            sheet.edge_df.drop('time', inplace=True, axis=1)
        if 'face' in sheet.face_df.columns:
            sheet.face_df.set_index('face', inplace=True)
            sheet.face_df.drop('time', inplace=True, axis=1)
        return sheet

    def update_cell_type_parameters(self, delta_level):
        differentiating_cells = self.sheet.face_df.query('type >= 0').index
        for param in self.face_params.keys():
            new_values = delta_level * self.face_params[param]["HC"] + (1 - delta_level) * self.face_params[param]["SC"]
            self.sheet.face_df.loc[differentiating_cells,param] = new_values[differentiating_cells]

        new_types = (delta_level > self.differentiation_threshold).astype(int)
        self.sheet.face_df.loc[differentiating_cells, 'type'] = new_types[differentiating_cells]
        first_faces = self.sheet.edge_df.face.values
        opposite_to_first = self.sheet.edge_df.opposite.values
        second_faces = - np.ones(opposite_to_first.shape)
        second_faces[opposite_to_first >= 0] = self.sheet.edge_df.loc[opposite_to_first[opposite_to_first >= 0], "face"].values
        first_types = self.sheet.face_df.loc[first_faces, "type"].values
        second_types = - np.ones(second_faces.shape)
        second_types[second_faces >= 0] = self.sheet.face_df.loc[second_faces[second_faces >= 0], "type"].values
        for param in self.edge_params.keys():
            new_vals = self.sheet.edge_df.loc[:, param].values
            new_vals[np.logical_and(first_types == 1, second_types == 1)] = self.edge_params[param][("HC", "HC")]
            new_vals[np.logical_and(first_types == 1, second_types == 0)] = self.edge_params[param][("HC", "SC")]
            new_vals[np.logical_and(first_types == 0, second_types == 1)] = self.edge_params[param][("SC", "HC")]
            new_vals[np.logical_and(first_types == 0, second_types == 0)] = self.edge_params[param][("SC", "SC")]
            self.sheet.edge_df.loc[:, param] = new_vals

    def get_specs_2d(self):
        specs = {'vert': {'is_active':1,
                          'radial_tension': 0},
                 'edge': {'is_active': 1,
                          'sub_area': 6,
                          },
                 'face': {'notch_level': 1.,
                          'delta_level': 1.,
                          'repressor_level': 1.,
                          'is_alive': 1,
                          'type': 0,
                          'radial_tension': 0,
                          'notch_sensitivity': 1
                          }
                 }
        for param in self.edge_params.keys():
            specs['edge'][param] = self.edge_params[param][("SC", "SC")]
        for param in self.face_params.keys():
            specs['face'][param] = self.face_params[param]["SC"]
        return specs

    def mean_notch(self, indices):
        return self.sheet.face_df.loc[indices[indices >= 0], 'notch_level'].mean()

    def mean_delta(self, indices):
        return self.sheet.face_df.loc[indices[indices >= 0], 'delta_level'].mean()

    def get_neighbors(self, face):
        return sheet.get_neighbors(face)


    def get_neighbors_data(self, func_list):
        def apply_on_real_neighbors(func):
            def f(neighbours):
                indices = neighbours.to_numpy()
                return func(self.sheet.edge_df.loc[indices[indices >= 0], "face"].unique())
            return f
        if hasattr(func_list, "__len__"):
            return self.sheet.edge_df.groupby("face")["opposite"].agg([apply_on_real_neighbors(func) for func in func_list])
        else:
            return self.sheet.edge_df.groupby("face")["opposite"].agg(apply_on_real_neighbors(func_list))

    @staticmethod
    def get_model(only_differentiation=False, add_random_forces=False):
        if only_differentiation:
            model = model_factory([EmptyAffector])
        elif add_random_forces:
            model = model_factory([LineTension, FaceContractility, FaceAreaElasticity, FaceRepulsion, RandomAffector])
        else:
            model = model_factory([LineTension, FaceContractility, FaceAreaElasticity])
        return model

    def get_ablation_function(self, cell_id, shrink_rate=1.5):
        self.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': 0.01,
            'radial_tension': 0.2,
            'contractile_increase': 0.3,
            'contract_span': 2,
            'geom': self.sheet.geom,
            'neighbors': self.get_neighbors(cell_id)
        }
        def ablation(sheet, manager):
            # Do delamination
            sheet.face_df.loc[cell_id, "type"] = -1
            sheet.face_df.loc[cell_id, "area_elasticity"] = 5
            manager.append(apoptosis, face_id=cell_id, **sheet.settings['apoptosis'])

            return
        return ablation


    def get_delamination_function(self, crit_area=0.5, shrink_rate=1.2):
        self.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': crit_area/2,
            'radial_tension': 0.2,
            'contractile_increase': 0.3,
            'contract_span': 2,
            'geom': self.sheet.geom
        }

        def delamination(sheet, manager):
            for cell_id, row in sheet.face_df.iterrows():
                if row.area < crit_area:
                    # Do delamination
                    sheet.face_df.loc[cell_id, "type"] = -1
                    manager.append(apoptosis, face_id=cell_id, **sheet.settings['apoptosis'])
                    sheet.face_df.at[cell_id, "prefered_area"] = sheet.face_df.at[cell_id, "prefered_vol"]
                    involved_faces = self.get_neighbors(cell_id)
                    for face in involved_faces:
                        sheet.order_edges(face)
                    # sheet.reset_index(order=False)
                    sheet.edge_df.sort_values(["face", "order"], inplace=True)
                    sheet.get_opposite()
                    # update geometry
                    sheet.geom.update_all(sheet)
                    if not sheet.check_all_edge_order():
                        print("bug in delamination")
            return
        return delamination



    def get_division_function(self, crit_area):
        def division(sheet, manager):
            """Defines a division behavior."""
            for cell_id, row in sheet.face_df.iterrows():
                if row.area > crit_area and row.type == 0:
                    # Do division
                    daughter = cell_division(sheet, cell_id, sheet.geom)[0]
                    # Update the topology
                    sheet.get_opposite()
                    involved_faces = np.intersect1d(self.get_neighbors(cell_id), self.get_neighbors(daughter))
                    involved_faces = np.hstack([involved_faces, np.array([cell_id, daughter])]).astype(int)
                    for face in involved_faces:
                        sheet.order_edges(face)
                    # sheet.reset_index(order=False)
                    sheet.edge_df.sort_values(["face", "order"], inplace=True)
                    sheet.get_opposite()
                    # update geometry
                    sheet.geom.update_all(sheet)
            manager.append(division)
        return division

    def get_intercalation_function(self, crit_edge_length):
        """Defines an intercalation behavior."""
        def intercalation(sheet, manager):
            is_virtual = sheet.is_virtual_edge(np.arange(sheet.edge_df.shape[0]))
            for edge_id, row in sheet.edge_df.iterrows():
                if row.is_active > 0 and row.length < crit_edge_length and not is_virtual[edge_id]:
                    # find involved cells
                    vertices = sheet.edge_df.loc[edge_id, ["srce", "trgt"]]
                    involved_edges = sheet.edge_df.query("srce in [%d, %d] or trgt in [%d, %d]" % (vertices.values[0],
                                                                                                   vertices.values[1],
                                                                                                   vertices.values[0],
                                                                                                   vertices.values[1]))
                    involved_faces = np.unique(involved_edges.face.to_numpy())
                    # Do intercalation
                    type1_transition(sheet, edge_id)
                    # Update the topology
                    for face in involved_faces:
                        sheet.order_edges(face)
                    sheet.reset_index(order=False)
                    sheet.edge_df.sort_values(["face", "order"], inplace=True)
                    sheet.get_opposite()
                    # update geometry
                    sheet.geom.update_all(sheet)
                    if not sheet.check_all_edge_order():
                        print("bug in intercalation")
                    break
            manager.append(intercalation)
        return intercalation

    def get_differentiation_function(self, l, m, mu, rho, betaN, betaD, inhibition=False):
        def differentiation(sheet, manager, dt=1.):
            # Notch and delta levels of each cell
            levels = sheet.face_df.loc[:,['notch_level', 'delta_level', 'notch_sensitivity']]
            notch_level = levels.notch_level.to_numpy()
            delta_level = levels.delta_level.to_numpy()
            sensitivity = levels.notch_sensitivity.to_numpy()
            # Mean notch and delta of neighboring cells
            neigh_delta = self.get_neighbors_data(self.mean_delta).to_numpy()

            def f(x, a):
                return (x**l)/(a + x**l)

            def g(x):
                return 1/(1 + x**m)
            if inhibition:
                new_notch = notch_level - dt * mu * notch_level
                new_delta = delta_level + dt * rho * (1 - delta_level)
            else:
                new_notch = notch_level + dt * mu * (betaN * f(neigh_delta, sensitivity) - notch_level)
                new_delta = delta_level + dt * rho * (betaD * g(notch_level) - delta_level)
            sheet.face_df.loc[:, 'notch_level'] = new_notch
            sheet.face_df.loc[:, 'delta_level'] = new_delta
            self.update_cell_type_parameters(new_delta)
            manager.append(differentiation, dt=dt)
        return differentiation

    def get_length_dependent_differentiation_function(self, l, m, mu, rho, xhi, betaN, betaD, betaR, kt, gammaR, inhibition=False):
        if inhibition:
            return self.get_differentiation_function(l, m, mu, rho, inhibition=inhibition)
        def differentiation(sheet, manager, dt=1.):
            # Notch and delta levels of each edge
            edge_data = sheet.edge_df[['face', 'opposite', 'length']]
            face_data = sheet.face_df[['repressor_level', 'notch_level', 'delta_level', 'perimeter', 'notch_sensitivity']]
            face_list, number_of_edges = np.unique(edge_data.face.values, return_counts=True)
            face_data.loc[face_list, 'n_edges'] = number_of_edges
            face_data['mean_notch_level'] = face_data.eval('notch_level / n_edges')
            face_data['mean_delta_level'] = face_data.eval('delta_level / n_edges')
            edge_data.loc[:, ['notch_level', 'delta_level', 'face_perimeter', 'face_repressor_level']] =\
                face_data.loc[edge_data.face.values, ['mean_notch_level','mean_delta_level', 'perimeter', 'repressor_level']]
            edge_data.loc[:, ['opposite_notch_level','opposite_delta_level']] = \
                edge_data.loc[edge_data.opposite.values, ['notch_level', 'delta_level']]
            edge_data.loc[:, "interaction_level"] = edge_data.eval('notch_level * opposite_delta_level * edge_length')
            face_data = face_data.join(edge_data.group_by("face")["interaction_level"].sum(), on="face", how="left")

            edge_notch_levels = edge_data.notch_level.to_numpy()
            edge_delta_levels = edge_data.delta_level.to_numpy()
            matching_faces_perimeter = edge_data.face_perimeter.to_numpy()
            matching_face_repressor = edge_data.repressor_level.to_numpy()
            opposite_notch_levels = edge_data.opposite_notch_level.to_numpy()
            opposite_delta_levels = edge_data.opposite_delta_level.to_numpy()
            face_repressor_levels = face_data.repressor_level.to_numpy()
            face_interactions_level = face_data.interaction_level.to_numpy()
            face_sensitivity = face_data.notch_sensitivity.to_numpy()

            def f(x, a):
                return (x**m)/(a + x**m)

            def g(x):
                return 1/(1 + (x**l))
            notch_change = betaN/matching_faces_perimeter - edge_notch_levels - edge_notch_levels*opposite_delta_levels/kt
            delta_change = g(matching_face_repressor)*betaD/matching_faces_perimeter - edge_delta_levels - edge_delta_levels*opposite_notch_levels/kt
            repressor_change = f(face_interactions_level, face_sensitivity)*betaR - gammaR*face_repressor_levels

            edge_data.loc[:, "notch_level"] = edge_data.loc[:, "notch_level"] + notch_change * mu * dt
            edge_data.loc[:, "delta_level"] = edge_data.loc[:, "notch_level"] + delta_change * rho * dt
            new_notch_delta_levels = edge_data.group_by("face")["notch_level", "delta_level"].sum()
            new_notch_delta_levels.set_index("face", inplace=True)

            sheet.face_df.update(new_notch_delta_levels.notch_level)
            sheet.face_df.loc[:, "repressor_level"] = sheet.face_df.repressor_level.values + repressor_change * xhi * dt

            self.update_cell_type_parameters(sheet.face_df.delta_level.values)
            manager.append(differentiation, dt=dt)
        return differentiation

    def get_aging_sensitivity_function(self, rate):
        def aging_sensitivity(sheet, manager, dt=1.):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
        return aging_sensitivity

    def save_notch_delta(self, file_path):
        levels = self.sheet.face_df.loc[:, ['notch_level', 'delta_level']]
        levels.to_pickle(file_path)

    def initialize_notch_delta(self, random_sensitivity=False, saved_levels_file_path=None):
        if saved_levels_file_path is not None and os.path.isfile(saved_levels_file_path):
            levels = pd.read_pickle(saved_levels_file_path)
            self.sheet.face_df.loc[:, 'notch_level'] = levels.notch_level.values
            self.sheet.face_df.loc[:, 'delta_level'] = levels.delta_level.values
        else:
            self.sheet.face_df.loc[:, 'notch_level'] = np.random.rand(self.sheet.face_df.shape[0])
            self.sheet.face_df.loc[:, 'delta_level'] = np.random.rand(self.sheet.face_df.shape[0])
            # self.sheet.face_df.loc[:, 'notch_level'] = 1
            # self.sheet.face_df.loc[:, 'delta_level'] = 0
        if random_sensitivity:
            self.sheet.face_df.loc[:, 'notch_sensitivity'] = np.random.rand(self.sheet.face_df.shape[0])


    def simulate(self, t_end, dt, notch_inhibition=False, only_differentiation=False, random_forces=False, aging_sensitivity=False):
        manager = EventManager("face")
        manager.append(self.get_ablation_function(2))
        manager.append(self.get_differentiation_function(l=2, m=2, mu=10, rho=10, betaN=100, betaD=1,
                                                        inhibition=notch_inhibition), dt=dt)
        if aging_sensitivity:
            manager.append(self.get_aging_sensitivity_function(rate=10))
        if not only_differentiation:
            manager.append(self.get_division_function(crit_area=1.01))
            manager.append(self.get_intercalation_function(crit_edge_length=0.04))
            manager.append(self.get_delamination_function(crit_area=0.1, shrink_rate=1.2))
            manager.append(self.sheet.get_update_virtual_vertices_function())
        history = History(self.sheet, save_all=True)
        model = self.get_model(only_differentiation, random_forces)
        solver = EulerSolver(self.sheet, self.sheet.geom, model, manager=manager, history=history,auto_reconnect=True)
        self.sheet.vert_df['viscosity'] = 3.0
        # for diff in range(100):
        #     manager.execute(self.sheet)
        #     manager.update()
        solver.solve(tf=t_end, dt=dt)
        # fig, ax = plot_forces(self.sheet, geom, model, ['x', 'y'], 1)
        # plt.show()
        return history

    @staticmethod
    def draw_sheet(sheet, number_vertices=False, number_edges=False, number_faces=False, is_ordered=True):
        if not sheet.check_all_edge_order():
            print("bug in drawing")
            sheet.order_all_edges()
        draw_specs = tyssue.config.draw.sheet_spec()
        cmap = plt.cm.get_cmap('Greens').reversed()
        cmap_scale = sheet.face_df.delta_level.to_numpy()
        color_cmap = cmap(0.7*(cmap_scale - 1) + 1)
        draw_specs['face']['color'] = color_cmap
        draw_specs['face']['alpha'] = 0.5
        draw_specs['face']['visible'] = True
        if is_ordered:
            sheet.is_ordered = True
            sheet.edge_df.sort_values(["face", "order"], inplace=True)

        fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
        fig.set_size_inches((8, 8))

        if number_faces:
            for face, data in sheet.face_df.iterrows():
                ax.text(data.x, data.y, face, fontsize=14, color="red")

        if number_vertices:
            for vert, data in sheet.vert_df.iterrows():
                ax.text(data.x, data.y + 0.02, vert, weight="bold", color="blue")

        if number_edges:
            for edge, data in sheet.edge_df.iterrows():
                ax.text((data.tx + data.sx)/2 - (data.tx - data.sx)/4,
                        (data.ty + data.sy)/2 - (data.ty - data.sy)/4 + 0.02,
                        edge, weight="bold", color="green")

        return fig, ax


def get_neighbors_from_history(history, face, time):
        edge_time_df = history.edge_h.get(['face', 'oposite', 'time'])
        edge_df = edge_time_df.query("time == %f" % time)
        face_edges = edge_time_df.query("face == %d" % face)
        opposite_edges = face_edges.opposite.to_numpy()
        neighbors = edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
        return np.unique(neighbors)

def find_differentiation_events(history):
    face_time_df = history.face_h.get(['face', 'type', 'time'])
    previously_SC = set()
    differentiation_df = pd.DataFrame(columns=["time", "face", "HC_neighbors"])
    for time in np.sort(np.unique(face_time_df.time.to_numpy())):
        currently_HC = set(face_time_df.query("type == 1 and time == %f" % time).face)
        differentiating_faces = previously_SC.intersection(currently_HC)
        for face in differentiating_faces:
            neighbors = set(get_neighbors_from_history(history, face, time))
            HC_neighbors_number = len(neighbors.intersection(currently_HC))
            differentiation_df.append({"time": time, "face": face, "HC_neighbors_number": HC_neighbors_number},
                                      ignore_index=True)
        previously_SC = set(face_time_df.query("type == 0 and time == %f" % time).face)
    return differentiation_df

if __name__ == "__main__":
    initial_sheet_name = ""
    name = "ablation_trial"
    random_sensitivity = False
    aging_sensitivity = False
    only_differentiation = False
    random_forces = False

    if os.path.isfile("%s.hf5" % initial_sheet_name):
        history = HistoryHdf5.from_archive("%s.hf5" % initial_sheet_name, eptm_class=VirtualSheet)
        last_time_point = np.max(history.time_stamps)
        sheet = history.retrieve(last_time_point)
    else:
        sheet = VirtualSheet.planar_sheet_2d(
            'basic2D',  # a name or identifier for this sheet
            nx=5,  # approximate number of cells on the x axis
            ny=6,  # approximae number of cells along the y axis
            distx=1,  # distance between 2 cells along x
            disty=1  # distance between 2 cells along y
        )
        sheet.initiate_edge_order()
        sheet.add_virtual_vertices()
    sheet.set_maximal_bond_length(0.2)  # was 0.2
    sheet.set_minimal_bond_length(0.05)  # was 0.05
    inner = InnerEarModel(sheet, random_sensitivity=random_sensitivity,
                          saved_notch_delta_levels_file="%s_notch_delta_levels.pkl" % initial_sheet_name)
    fig1, ax1 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=False)
    plt.savefig("%s_initial.png" % name)
    history = inner.simulate(t_end=10, dt=0.01, notch_inhibition=False, only_differentiation=only_differentiation,
                             random_forces=random_forces, aging_sensitivity=aging_sensitivity)
    # history2 = inner.simulate(t_end=5, dt=0.01, notch_inhibition=True)
    if os.path.isfile("%s.hf5" % name):
        os.remove("%s.hf5" % name)
    # if os.path.isfile("%s_after.hf5" % name):
    #     os.remove("%s_after.hf5" % name)
    history.to_archive("%s.hf5" % name)
    inner.save_notch_delta("%s_notch_delta_levels.pkl" % name)
    # history2.to_archive("%s_after.hf5" % name)
    fig2, ax2 = inner.draw_sheet(inner.sheet, number_faces=True, number_edges=False, number_vertices=False)
    plt.savefig("%s_finale.png" % name)
    create_gif(history, "%s.gif" % name, num_frames=len(history), draw_func=inner.draw_sheet)
    # create_gif(history2, "%s_after.gif" % name, num_frames=len(history), draw_func=inner.draw_sheet)

