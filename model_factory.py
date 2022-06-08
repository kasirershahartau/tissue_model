###### Creating varius types of models ##########
# This file is a derivative of tyssue project with few chages:
# 1. Adding virtual vertices to allow for round apical morphology
# 2. Adding differential features by cell types
# 3. Adding external forces

import tyssue
from tyssue import Sheet, config, SheetGeometry, History, EventManager
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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class VirtualSheet(Sheet):
    """ An epithelium tissue with virtual vertices, to allow for rounded apical morphology"""

    def __init__(self, identifier, datasets, specs=None, coords=None, maximal_bond_length=0.5,
                 minimal_bond_length=0.1):
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
        self.update_specs({"vert": {"is_virtual": 0}})
        self.sanitize(trim_borders=True, order_edges=True)
        self.get_opposite()
        self.maximal_bond_length = maximal_bond_length
        self.minimal_bond_length = minimal_bond_length
        geom.update_all(self)

    def add_virtual_vertices(self):
        long = self.edge_df[sheet.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
        np.random.shuffle(long)
        while long.size > 0:
            edge_ind = long[0]
            new_vert, new_edge, new_opposite_edge = add_vert(self, edge_ind)
            self.vert_df.at[new_vert, "is_virtual"] = 1
            self.edge_df.at[edge_ind, "length"] /= 2
            self.edge_df.at[new_edge, "length"] /= 2
            opposite = int(self.edge_df.loc[edge_ind, "opposite"])
            if opposite >= 0:
                self.edge_df.at[opposite, "length"] /= 2
            if new_opposite_edge is None:
                self.edge_df.at[new_edge, "opposite"] = -1
            else:
                self.edge_df.at[new_edge, "opposite"] = new_opposite_edge
                self.edge_df.at[new_opposite_edge, "opposite"] = new_edge
                self.edge_df.at[new_opposite_edge, "length"] /= 2
            long = self.edge_df[sheet.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
            np.random.shuffle(long)
        self.edge_df.index.name = 'edge'
        geom.update_all(self)
        self.reset_index(order=True)
        self.get_opposite()

    def remove_virtual_vertices(self):
        short = self.edge_df[sheet.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
        if short.size > 0:
            short = short[self.is_virtual_edge(short)]
        np.random.shuffle(short)
        while short.size > 0:
            collapse_edge(sheet, short[0], allow_two_sided=False)
            short = self.edge_df[sheet.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
            if short.size > 0:
                short = short[self.is_virtual_edge(short)]
            np.random.shuffle(short)
        geom.update_all(self)
        return 0

    def is_virtual_edge(self, edge_indices):
        """
        Checks if an edge contains a virtual vertex
        """
        srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual.to_numpy() == 1
        trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual.to_numpy() == 1
        return np.logical_or(srce_is_virtual, trgt_is_virtual)

    def get_update_virtual_vertices_function(self):
        def update_virtual_vertices(sheet, manager):
            sheet.remove_virtual_vertices()
            sheet.add_virtual_vertices()
            manager.append(update_virtual_vertices)
            return
        return update_virtual_vertices


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
                 differentiation_threshold=0.5):
        # Setting class constants
        self.CELL_TYPES = ['SC', 'HC']
        self.DIMENSIONS = ['2D']


        # Setting default behavior
        if tension is None:
            tension = {('HC', 'HC'): 1,
                       ('HC', 'SC'): 2,
                       ('SC', 'SC'): 0.5
                       }
        if preferred_area is None:
            preferred_area = {'HC': 3,
                              'SC': 0.5}
        if contractility is None:
            contractility = {'HC': 0.5,
                             'SC': 1}
        if repulsion is None:
            repulsion = {'HC': 0.001,
                         'SC': 0}
        if repulsion_distance is None:
            repulsion_distance = {'HC': 2.0,
                                  'SC': 0}
        if elasticity is None:
            elasticity = {'HC': 1,
                          'SC': 1}
        if ('SC', 'HC') in tension:
            tension[('HC', 'SC')] = tension[('SC', 'HC')]
        elif ('HC', 'SC') in tension:
            tension[('SC', 'HC')] = tension[('HC', 'SC')]
        self.sheet = sheet
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
        self.initialize_random_notch_delta()
        active_edges = (self.sheet.edge_df.opposite.values >= 0).astype(int)
        self.sheet.edge_df.at[:, 'is_active'] = active_edges
        self.sheet.vert_df.at[set(self.sheet.edge_df.srce.values[np.logical_not(active_edges)]), 'is_active'] = 0
        self.sheet.vert_df.at[set(self.sheet.edge_df.trgt.values[np.logical_not(active_edges)]), 'is_active'] = 0
        self.sheet.active_verts = np.where(self.sheet.vert_df.is_active.values)[0]
        self.update_cell_type_parameters(self.sheet.face_df.delta_level)


    def update_cell_type_parameters(self, delta_level):
        for param in self.face_params.keys():
            self.sheet.face_df.at[:,param] = delta_level*self.face_params[param]["HC"] +\
                                                   (1-delta_level)*self.face_params[param]["SC"]
        self.sheet.face_df.at[:, 'type'] = (delta_level > self.differentiation_threshold).astype(int)
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
            self.sheet.edge_df.at[:, param] = new_vals

    def get_specs_2d(self):
        specs = {'vert': {'is_active':1},
                 'edge': {'is_active': 1,
                          'sub_area': 6,
                          'notch_level': 1.,
                          'delta_level': 1.,
                          },
                 'face': {'notch_level': 1.,
                          'delta_level': 1.,
                          'repressor_level': 1.,
                          'is_alive': 1,
                          'type': 0
                          }
                 }
        for param in self.edge_params.keys():
            specs['edge'][param] = self.edge_params[param][("SC", "SC")]
        for param in self.face_params.keys():
            specs['face'][param] = self.face_params[param]["SC"]
        return specs

    @staticmethod
    def mean_notch(indices):
        return sheet.face_df.loc[indices[indices >= 0], 'notch_level'].mean()

    @staticmethod
    def mean_delta(indices):
        return sheet.face_df.loc[indices[indices >= 0], 'delta_level'].mean()

    @staticmethod
    def get_neighbors_data(func_list):
        def apply_on_real_neighbors(func):
            def f(neighbours):
                indices = neighbours.to_numpy()
                return func(inner.sheet.edge_df.loc[indices[indices >= 0], "face"].unique())
            return f
        if hasattr(func_list, "__len__"):
            return sheet.edge_df.groupby("face")["opposite"].agg([apply_on_real_neighbors(func) for func in func_list])
        else:
            return sheet.edge_df.groupby("face")["opposite"].agg(apply_on_real_neighbors(func_list))

    @staticmethod
    def get_model():
        model = model_factory([LineTension, FaceContractility, FaceAreaElasticity, FaceRepulsion])
        return model

    def get_delamination_function(self, crit_area=0.5, shrink_rate=1.2):
        self.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': crit_area/2,
            'radial_tension': 0.2,
            'contractile_increase': 0.3,
            'contract_span': 2
        }

        def delamination(sheet, manager):
            for cell_id, row in sheet.face_df.iterrows():
                if row.area < crit_area:
                    # Do delamination
                    manager.append(apoptosis, face_id=cell_id, **sheet.settings['apoptosis'])
                    sheet.face_df.at[cell_id, "prefered_area"] = sheet.face_df.at[cell_id, "prefered_vol"]
            return
        return delamination



    def get_division_function(self, crit_area):
        def division(sheet, manager):
            """Defines a division behavior."""
            for cell_id, row in sheet.face_df.iterrows():
                if row.area > crit_area and row.type == 0:
                    # Do division
                    daughter = cell_division(sheet, cell_id, geom)
                    # Update the topology
                    sheet.reset_index(order=True)
                    sheet.get_opposite()
                    # update geometry
                    geom.update_all(sheet)
            manager.append(division)
        return division

    def get_intercalation_function(self, crit_edge_length):
        """Defines an intercalation behavior."""
        def intercalation(sheet, manager):
            is_virtual = sheet.is_virtual_edge(np.arange(sheet.edge_df.shape[0]))
            for edge_id, row in sheet.edge_df.iterrows():
                if row.is_active > 0 and row.length < crit_edge_length and not is_virtual[edge_id]:
                    # Do intercalation
                    type1_transition(sheet, edge_id)
                    # Update the topology
                    sheet.reset_index(order=True)
                    sheet.get_opposite()
                    # update geometry
                    geom.update_all(sheet)
                    break
            manager.append(intercalation)
        return intercalation


    def get_differentiation_function(self, k, h, a, b, mu, rho):
        def differentiation(sheet, manager, dt=1.):
            # Notch and delta levels of each cell
            levels = sheet.face_df.loc[:,['notch_level', 'delta_level']]
            notch_level = levels.notch_level.to_numpy()
            delta_level = levels.delta_level.to_numpy()
            # Mean notch and delta of neighboring cells
            neigh_delta = self.get_neighbors_data(self.mean_delta).to_numpy()

            def f(x):
                return (x**k)/(a + x**k)

            def g(x):
                return 1/(1 + b*(x**h))
            new_notch = notch_level + dt*mu*(f(neigh_delta) - notch_level)
            new_delta = delta_level + dt*rho*(g(notch_level) - delta_level)
            sheet.face_df.at[:, 'notch_level'] = new_notch
            sheet.face_df.at[:, 'delta_level'] = new_delta
            self.update_cell_type_parameters(new_delta)
            manager.append(differentiation, dt=dt)
        return differentiation

    # def get_length_dependent_differentiation_function(self, betaN, betaD, betaR, kt, m, l):
    #     def differentiation(sheet, manager, dt=1.):
    #         # Notch and delta levels of each edge
    #         edge_levels = sheet.edge_df.loc[:,['notch_level', 'delta_level']]
    #         repressor_level = sheet.face_df.repressor_level.values()
    #         notch_level = edge_levels.notch_level.to_numpy()
    #         delta_level = edge_levels.delta_level.to_numpy()
    #         weighted_mean_of_notch =
    #         def f(x):
    #             return (x**m)/(1 + x**m)
    #
    #         def g(x):
    #             return 1/(1 + (x**l))
    #         notch_change = betaN/
    #
    #
    #
    #         sheet.edge_df.at[:, 'notch_level'] =
    #         sheet.face_df.at[:, 'delta_level'] = new_delta
    #         self.update_cell_type_parameters(new_delta)
    #         manager.append(differentiation, dt=dt)
    #     return differentiation


    def initialize_random_notch_delta(self):
        self.sheet.face_df.at[:, 'notch_level'] = np.random.rand(self.sheet.face_df.shape[0])
        self.sheet.face_df.at[:, 'delta_level'] = np.random.rand(self.sheet.face_df.shape[0])

    def simulate(self, t_end, dt):
        manager = EventManager("face")
        manager.append(self.get_differentiation_function(k=2, h=2, a=0.01, b=100, mu=10, rho=10), dt=dt)
        manager.append(self.get_division_function(crit_area=1))
        manager.append(self.get_intercalation_function(crit_edge_length=0.1))
        manager.append(self.get_delamination_function(crit_area=0.5, shrink_rate=1.2))
        manager.append(sheet.get_update_virtual_vertices_function())
        history = History(self.sheet)
        model = self.get_model()
        solver = EulerSolver(self.sheet, geom, model, manager=manager, history=history,auto_reconnect=True)
        self.sheet.vert_df['viscosity'] = 3.0
        # for diff in range(100):
        #     manager.execute(self.sheet)
        #     manager.update()
        solver.solve(tf=t_end, dt=dt)
        # fig, ax = plot_forces(self.sheet, geom, model, ['x', 'y'], 1)
        # plt.show()
        return history

    @staticmethod
    def draw_sheet(sheet, number_vertices=False, number_edges=False, number_faces=False):
        draw_specs = tyssue.config.draw.sheet_spec()
        cmap = plt.cm.get_cmap('Reds')
        color_cmap = cmap(sheet.face_df.delta_level.to_numpy())
        draw_specs['face']['color'] = color_cmap
        draw_specs['face']['alpha'] = 0.5
        draw_specs['face']['visible'] = True
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

if __name__ == "__main__":
    sheet = VirtualSheet.planar_sheet_2d(
        'basic2D', # a name or identifier for this sheet
        nx=6, # approximate number of cells on the x axis
        ny=7, # approximate number of cells along the y axis
        distx=1, # distance between 2 cells along x
        disty=1 # distance between 2 cells along y
    )

    sheet.add_virtual_vertices()
    inner = InnerEarModel(sheet)
    fig1, ax1 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=True)
    plt.savefig("initial.png")
    history = inner.simulate(t_end=0.1, dt=0.01)
    fig2, ax2 = inner.draw_sheet(inner.sheet, number_faces=False, number_edges=False, number_vertices=True)
    plt.savefig("finale.png")
    create_gif(history, "differentiation_and_division.gif", num_frames=len(history), draw_func=inner.draw_sheet)

