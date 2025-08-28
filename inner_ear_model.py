###### Creating varius types of models ##########
# This file is a derivative of tyssue project with few chages:
# 1. Adding virtual vertices to allow for round apical morphology
# 2. Adding differential features by cell types
# 3. Adding external forces
import os.path
import sys
sys.path.insert(1, 'C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_model\\tyssue\\tyssue')
import tyssue
from tyssue import config, History
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division, type1_transition
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet import apoptosis
from tyssue.solvers.viscous import EulerSolver

from tyssue.dynamics import model_factory
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from virtual_sheet import VirtualSheet
from face_repulsion_efffector import FaceRepulsion
from empty_effector import EmptyAffector
from random_effector import RandomAffector

class InnerEarModel:
    """
    A wrapping class for epithilium model, for easy execution of inner ear model simulations.
    """

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
            contractility = {'HC': 0.4,
                             'SC': 0.1}
        if repulsion is None:
            repulsion = {'HC': 0.001, #0.001
                         'SC': 0}
        if repulsion_distance is None:
            repulsion_distance = {'HC': 2.0,
                                  'SC': 0}
        if elasticity is None:
            elasticity = {'HC': 5,
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

    def set_random_parameters(self):
        self.sheet.face_df.loc[:, "prefered_area"] = np.random.rand(self.sheet.face_df.shape[0],)
        self.sheet.face_df.loc[:, "prefered_vol"] = self.sheet.face_df.loc[:, "prefered_area"]
        self.sheet.face_df.loc[:, "contractility"] = np.random.rand(self.sheet.face_df.shape[0],)/5
        self.sheet.edge_df.loc[:, "line_tension"] = np.random.rand(self.sheet.edge_df.shape[0],)/10

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
            def f(neighbors):
                indices = neighbors.to_numpy()
                return func(self.sheet.edge_df.loc[indices[indices >= 0], "face"].unique())
            return f
        if hasattr(func_list, "__len__"):
            return self.sheet.edge_df.groupby("face")["opposite"].agg([apply_on_real_neighbors(func) for func in func_list])
        else:
            return self.sheet.edge_df.groupby("face")["opposite"].agg(apply_on_real_neighbors(func_list))

    @staticmethod
    def get_model(only_differentiation=False, effectors=None):
        if only_differentiation:
            model = model_factory([EmptyAffector])
        else:
            if effectors is None:
                model = model_factory([LineTension, FaceContractility, FaceAreaElasticity])
            else:
                model = model_factory(effectors)
        return model

    def get_ablation_function(self, cell_id, shrink_rate=1.5, critical_area=0.01):
        self.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': critical_area,
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

    def get_differentiation_function(self, l, m, mu, rho, betaN, betaD, inhibition=False, dt=1.):
        def differentiation(sheet, manager):
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
            manager.append(differentiation)
        return differentiation

    def get_length_dependent_differentiation_function(self, l, m, mu, rho, xhi, betaN, betaD, betaR, kt, gammaR, inhibition=False):
        if inhibition:
            return self.get_differentiation_function(l, m, mu, rho, inhibition=inhibition)
        def differentiation(sheet, manager, dt=1.):
            # Notch and delta levels of each edge
            edge_data = sheet.edge_df[['face', 'opposite', 'length']].copy()
            face_data = sheet.face_df[['repressor_level', 'notch_level', 'delta_level', 'perimeter', 'notch_sensitivity']].copy()
            face_list, number_of_edges = np.unique(edge_data.face.values, return_counts=True)
            face_data.loc[face_list, 'n_edges'] = number_of_edges
            face_data['notch_level_per_length'] = face_data.eval('notch_level / perimeter')
            face_data['delta_level_per_length'] = face_data.eval('delta_level / perimeter')
            for to_key, from_key in zip(['notch_level_per_length', 'delta_level_per_length', 'face_perimeter', 'face_repressor_level'],
                                        ['notch_level_per_length','delta_level_per_length', 'perimeter', 'repressor_level']):
                edge_data.loc[:, to_key] = face_data.loc[edge_data.face.values, from_key].to_numpy()
            edge_data["notch_level"] = edge_data.eval('notch_level_per_length * length')
            edge_data["delta_level"] = edge_data.eval('delta_level_per_length * length')
            has_opposite = edge_data.opposite.values >= 0
            edge_data.loc[edge_data.index[has_opposite], ['opposite_notch_level','opposite_delta_level']] = \
                edge_data.loc[edge_data.opposite.values[has_opposite], ['notch_level', 'delta_level']].to_numpy()
            edge_data["interaction_level"] = edge_data.eval('notch_level * opposite_delta_level * length').fillna(0)
            face_data = face_data.join(edge_data.groupby("face")["interaction_level"].sum(), on="face", how="left")

            edge_notch_levels = edge_data.notch_level.to_numpy()
            edge_delta_levels = edge_data.delta_level.to_numpy()
            matching_faces_perimeter = edge_data.face_perimeter.to_numpy()
            matching_face_repressor = edge_data.face_repressor_level.to_numpy()
            opposite_notch_levels = edge_data.opposite_notch_level.to_numpy()
            opposite_notch_levels[np.isnan(opposite_notch_levels)] = 0
            opposite_delta_levels = edge_data.opposite_delta_level.to_numpy()
            opposite_delta_levels[np.isnan(opposite_delta_levels)] = 0
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

            edge_data.loc[:, "notch_level"] = edge_data.notch_level.values + notch_change * mu * dt
            edge_data.loc[:, "delta_level"] = edge_data.delta_level.values + delta_change * rho * dt
            new_notch_delta_levels = edge_data.groupby("face")[["notch_level", "delta_level"]].sum()
            sheet.face_df.update(new_notch_delta_levels)
            sheet.face_df.loc[:, "repressor_level"] = sheet.face_df.repressor_level.values + repressor_change * xhi * dt
            self.update_cell_type_parameters(sheet.face_df.delta_level.values)
            manager.append(differentiation, dt=dt)
        return differentiation

    def get_aging_sensitivity_function(self, rate, dt):
        def aging_sensitivity(sheet, manager):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
        return aging_sensitivity


    def get_random_initializer(self, wait_time=5, dt=1.):
        self.time_to_random = wait_time
        def random_initializer(sheet, manager):
            if self.time_to_random <= 0:
                self.set_random_parameters()
                self.time_to_random = wait_time
            else:
                self.time_to_random -= dt
            manager.append(random_initializer)
        return random_initializer

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


    def simulate(self, t_end, dt, notch_inhibition=False, only_differentiation=False, random_forces=False,
                 aging_sensitivity=False, no_differentiation=False, contact_dependent_differentiation=False,
                 l=3, m=3, mu=10, rho=10, xhi=10, betaN=1, betaD=1, betaR=1, kt=1, gammaR=1, sensitivity_aging_rate=10,
                 division_area=1.3, intercalation_length=0.04, delamination_area=0.1, delamination_rate=1.2,
                 viscosity=3, effectors=None):
        manager = EventManager("face")
        # manager.append(self.get_ablation_function(2))
        if not no_differentiation:
            if contact_dependent_differentiation:
                manager.append(self.get_length_dependent_differentiation_function(l=l, m=m, mu=mu, rho=rho, xhi=xhi, betaN=betaN,
                                                                                  betaD=betaD, betaR=betaR, kt=kt, gammaR=gammaR,
                                                                                  inhibition=notch_inhibition), dt=dt)
            else:
                manager.append(self.get_differentiation_function(l=l, m=m, mu=mu, rho=rho, betaN=betaN, betaD=betaD,
                                                                 inhibition=notch_inhibition), dt=dt)
        if aging_sensitivity:
            manager.append(self.get_aging_sensitivity_function(rate=sensitivity_aging_rate, dt=dt))
        if not only_differentiation:
            manager.append(self.get_division_function(crit_area=division_area))
            manager.append(self.get_intercalation_function(crit_edge_length=intercalation_length))
            manager.append(self.get_delamination_function(crit_area=delamination_area, shrink_rate=delamination_rate))
            manager.append(self.sheet.get_update_virtual_vertices_function())
        if random_forces:
            manager.append(self.get_random_initializer(wait_time=1, dt=dt))
        history = History(self.sheet, save_all=True)
        model = self.get_model(only_differentiation)
        solver = EulerSolver(self.sheet, self.sheet.geom, model, manager=manager, history=history,auto_reconnect=True)
        self.sheet.vert_df['viscosity'] = viscosity
        # for diff in range(100):
        #     manager.execute(self.sheet)
        #     manager.update()
        solver.solve(tf=t_end, dt=dt)
        # fig, ax = plot_forces(self.sheet, geom, model, ['x', 'y'], 1)
        # plt.show()
        return history

    @staticmethod
    def draw_sheet(sheet, number_vertices=False, number_edges=False, number_faces=False, is_ordered=True,
                   for_labels=False):
        if not sheet.check_all_edge_order():
            print("bug in drawing")
            sheet.order_all_edges()
        draw_specs = tyssue.config.draw.sheet_spec()
        if for_labels:
            cmap_scale = sheet.face_df.id.to_numpy()
            color_cmap = np.zeros((cmap_scale.size, 4))
            color_cmap[:,0] = (cmap_scale%127) / 127
            color_cmap[:,1] = ((cmap_scale//127)%127) / 127
            color_cmap[:,2] = 0.5
            color_cmap[:,3] = 1
        else:
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
    @staticmethod
    def save_sheet_labels_to_numpy(sheet, path):
        # Creating labeld image
        Nc = sheet.Nc
        fig, ax = InnerEarModel.draw_sheet(sheet, for_labels=True)
        fig.tight_layout(pad=0)
        ax.set_axis_off()
        canvas = fig.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        data0 = data[:,:,0].copy()
        data0[data[:,:,2] != 191] = 0
        data1 = data[:,:,1].copy()
        data1[data[:,:,2] != 191] = 0
        labels = np.zeros(data0.shape)
        values1 = np.unique(data1)
        values0 = np.unique(data0)
        i = 1
        for v1 in values1:
            for v0 in values0:
                if i <= Nc:
                    if v0 != 0 or v1 != 0:
                        cell_pixels = np.logical_and(data0 == v0, data1 == v1)
                        if cell_pixels.any():
                            labels[np.logical_and(data0 == v0, data1 == v1)] = i
                            i += 1
        np.save(path, labels.astype("uint16"))
        return 0

    @staticmethod
    def save_contact_matrix_to_numpy(sheet, path):
        contact_matrix = sheet.get_contact_matrix()
        np.save(path, contact_matrix)
        return 0

    @staticmethod
    def save_face_data_to_df(sheet, path):
        face_data = sheet.face_df
        neighbors = []
        for id, face in face_data.iterrows():
            neighbors.append(set(sheet.get_neighbors(id)))
        cells_info_dict = {"label": face_data.id.to_numpy() + 1, "cx":face_data.x.to_numpy(),
                           "cy": face_data.y.to_numpy(), "type": face_data.type.to_numpy(),
                           "perimeter": face_data.perimeter.to_numpy(), "valid": face_data.is_alive.to_numpy(),
                           "notch_level": face_data.notch_level.to_numpy(),
                           "delta_level": face_data.delta_level.to_numpy(),
                           "neighbors": neighbors}
        cells_info = pd.DataFrame(cells_info_dict)
        cells_info.to_pickle(path)
        return 0

