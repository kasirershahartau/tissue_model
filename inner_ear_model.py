###### Creating varius types of models ##########
# This file is a derivative of tyssue project with few chages:
# 1. Adding virtual vertices to allow for round apical morphology
# 2. Adding differential features by cell types
# 3. Adding external forces
import os.path
import sys
import tyssue
from tyssue import config, History
from tyssue.draw import sheet_view
from tyssue.behaviors import EventManager
from solvers import IVPSolver
from tyssue.dynamics import model_factory
from tyssue.dynamics.effectors import LineTension, FaceAreaElasticity, FaceContractility

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from empty_effector import EmptyAffector
from topologocal_events import TopologicalEventsHandler
from lateral_inhibition_model import LateralInhibitionModel

class InnerEarModel:
    """
    A wrapping class for epithilium model, for easy execution of inner ear model simulations.
    """
    def __init__(self, sheet, tension=None, repulsion=None, repulsion_distance=None, repulsion_exp=7,
                 preferred_area=None, contractility=None, elasticity=None,
                 differentiation_threshold=0.5, random_sensitivity=False, saved_notch_delta_levels_file=None,
                 l=3, m=3, betaN=1, betaD=1, inhibition=False,
                 notch_repressor_degradation_ratio=1, repressor_sensitivity=1, atoh_sensitivity=1,
                 delta_repressor_degradation_ratio=1, notch_delta_production_ratio=1,
                 stress_effectors=None, mechanosensitivity=0, notch_sensitivity=1, atoh_by_repressor=True):
        # Setting class constants
        self.CELL_TYPES = ['SC', 'HC']
        self.DIMENSIONS = ['2D']
        self.topological_events_handler = TopologicalEventsHandler(self)
        # Setting default behavior
        if tension is None:
            tension = {('HC', 'HC'): 0.05,
                       ('HC', 'SC'): 0.05,
                       ('SC', 'SC'): 0.05
                       }
        if preferred_area is None:
            preferred_area = {'HC': 1.,
                              'SC': 1.}
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
            elasticity = {'HC': 5.,
                          'SC': 1.}
        if ('SC', 'HC') in tension:
            tension[('HC', 'SC')] = tension[('SC', 'HC')]
        elif ('HC', 'SC') in tension:
            tension[('SC', 'HC')] = tension[('HC', 'SC')]
        self.sheet = self.arrange_sheet_from_history(sheet)
        self.sheet.initiate_edge_order()
        sheet.repulsion_exp = repulsion_exp
        length_normalization_factor = self.get_average_face_perimeter()
        preferred_area['HC'] *= length_normalization_factor ** 2
        preferred_area['SC'] *= length_normalization_factor ** 2
        repulsion_distance['HC'] *= length_normalization_factor
        repulsion_distance['SC'] *= length_normalization_factor
        self.face_params = {"contractility": contractility, "repulsion": repulsion,
                            "repulsion_distance": repulsion_distance,
                            "prefered_area": preferred_area,
                            "prefered_vol": preferred_area,
                            "area_elasticity": elasticity}
        self.differentiation_threshold = differentiation_threshold
        self.edge_params = {"line_tension": tension}
        self.dimensionality = '2D'
        specs = self.get_specs_2d(notch_sensitivity)
        self.sheet.update_specs(specs, reset=True)
        self.lateral_inhibition_model = LateralInhibitionModel(
            self, l=l, m=m, betaN=betaN, betaD=betaD, inhibition=inhibition,
            notch_repressor_degradation_ratio=notch_repressor_degradation_ratio,
            length_normalization_factor=length_normalization_factor,
            repressor_sensitivity=repressor_sensitivity, atoh_sensitivity=atoh_sensitivity,
            delta_repressor_degradation_ratio=delta_repressor_degradation_ratio,
            notch_delta_production_ratio=notch_delta_production_ratio,
            stress_effectors=stress_effectors, mechanosensitivity=mechanosensitivity)
        self.initialize_notch_delta(random_sensitivity, saved_levels_file_path=saved_notch_delta_levels_file, contact_dependent=True)
        active_edges = (self.sheet.edge_df.opposite.values >= 0).astype(int)
        self.sheet.edge_df.loc[:, 'is_active'] = active_edges
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.srce.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.vert_df.loc[list(set(self.sheet.edge_df.trgt.values[np.logical_not(active_edges)])), 'is_active'] = 0
        self.sheet.face_df.loc[:,"id"] = self.sheet.face_df.index
        self.sheet.active_verts = np.where(self.sheet.vert_df.is_active.values)[0]
        if atoh_by_repressor:
            if "repressor_level" in self.sheet.face_df.columns:
                self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                    self.sheet.face_df.repressor_level.values)
            else:
                self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                    self.sheet.face_df.notch_level.values)
        else:
            self.sheet.face_df["atoh_level"] = self.lateral_inhibition_model.get_atoh_level(
                self.sheet.face_df.delta_level.values, activation=True)
        self.update_cell_type_parameters(self.sheet.face_df.atoh_level)

        self.sheet.order_all_edges()

        return

    @staticmethod
    def arrange_sheet_from_history(sheet):
        if 'vert' in sheet.vert_df.columns:
            if np.isnan(sheet.vert_df['vert']).any():
                sheet.vert_df.set_index('index', inplace=True)
                sheet.vert_df.drop(columns=['vert'], inplace=True)
            else:
                sheet.vert_df.set_index('vert', inplace=True)
        if 'time' in sheet.vert_df.columns:
            sheet.vert_df.drop('time', inplace=True, axis=1)
        if 'edge' in sheet.edge_df.columns:
            sheet.edge_df.set_index('edge', inplace=True)
        if 'time' in sheet.edge_df.columns:
            sheet.edge_df.drop('time', inplace=True, axis=1)
        if 'face' in sheet.face_df.columns:
            sheet.face_df.set_index('face', inplace=True)
        if 'time' in sheet.face_df.columns:
            sheet.face_df.drop('time', inplace=True, axis=1)
        return sheet

    def update_cell_type_parameters(self, atoh_level):
        differentiating_cells = self.sheet.face_df.query('type >= 0').index
        for param in self.face_params.keys():
            new_values = atoh_level * self.face_params[param]["HC"] + (1 - atoh_level) * self.face_params[param]["SC"]
            self.sheet.face_df.loc[differentiating_cells,param] = new_values[differentiating_cells]

        new_types = (atoh_level > self.differentiation_threshold).astype(np.int32)
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

    def get_specs_2d(self, notch_sensitivity):
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
                          'notch_sensitivity': notch_sensitivity
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
        return self.sheet.get_neighbors(face)


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

    def get_neighbor_types(self, neighbors_id):
        return self.sheet.face_df.type[neighbors_id].values

    def get_num_of_HC_neighbors(self, neighbors_id):
        types = self.get_neighbor_types(neighbors_id)
        counts = np.bincount(types[types>=0])
        if counts.size > 1:
            return counts[1]
        else:
            return 0

    def get_edge_stress(self, relevant_effectors):
        edge_data = self.sheet.edge_df[["ux", "uy", "opposite"]].copy()
        stress_model = model_factory(relevant_effectors)
        grads = stress_model.compute_gradient(self.sheet, components=True)
        norm_factor = self.sheet.specs["settings"].get("nrj_norm_factor", 1)
        srce_grads = [g[0] for g in grads if g[0].shape[0] == self.sheet.Ne]
        if srce_grads:
            edge_data["srce_gx"] = np.array([grad.gx.values for grad in srce_grads]).sum(axis=0)
            edge_data["srce_gy"] = np.array([grad.gy.values for grad in srce_grads]).sum(axis=0)
        trgt_grads = [
            g[1] for g in grads if (g[1] is not None) and (g[1].shape[0] == self.sheet.Ne)
        ]
        if trgt_grads:
            edge_data["trgt_gx"] = np.array([grad.gx.values for grad in trgt_grads]).sum(axis=0)
            edge_data["trgt_gy"] = np.array([grad.gy.values for grad in trgt_grads]).sum(axis=0)
        vert_grads = [g[0] for g in grads if g[0].shape[0] == self.sheet.Nv]
        if vert_grads:
            raise NotImplementedError
        edge_data["stress"] = edge_data.eval("((trgt_gx - srce_gx) * ux + (trgt_gy - srce_gy) * uy)/ %s" % str(norm_factor))
        # Including the opposite edge stress (treating edges as mechanically coupled)
        edge_data.loc[edge_data["opposite"].values > 0, "stress"] += edge_data.loc[edge_data["opposite"].index[edge_data["opposite"].values > 0], "stress"]
        return edge_data.stress

    def get_face_stress(self, relevant_effectors):
        edge_stress = self.get_edge_stress(relevant_effectors)
        stress_df = self.sheet.edge_df[["face", "length"]].copy()
        stress_df["stress"] = edge_stress.values
        stress_df["weighted_stress"] = stress_df.eval("stress * length")
        face_stress = stress_df[["face", "weighted_stress"]].groupby("face").sum()
        return face_stress.weighted_stress.values

    def get_average_edge_stress_by_type(self, relevant_effectors=None):
        if relevant_effectors is None:
            relevant_effectors = self.lateral_inhibition_model.stress_effectors
        edge_stress = self.get_edge_stress(relevant_effectors).values
        first_faces = self.sheet.edge_df.face.values
        opposite_to_first = self.sheet.edge_df.opposite.values
        second_faces = - np.ones(opposite_to_first.shape)
        second_faces[opposite_to_first >= 0] = self.sheet.edge_df.loc[
            opposite_to_first[opposite_to_first >= 0], "face"].values
        first_types = self.sheet.face_df.loc[first_faces, "type"].values
        second_types = - np.ones(second_faces.shape)
        second_types[second_faces >= 0] = self.sheet.face_df.loc[second_faces[second_faces >= 0], "type"].values
        HC_HC_stress = edge_stress[np.logical_and(first_types == 1, second_types == 1)]
        SC_SC_stress = edge_stress[np.logical_and(first_types == 0, second_types == 0)]
        HC_SC_stress = edge_stress[np.logical_and(first_types == 1, second_types == 0)]
        SC_HC_stress = edge_stress[np.logical_and(first_types == 0, second_types == 1)]
        res = {"HC:HC": np.average(HC_HC_stress), "SC:SC": np.average(SC_SC_stress),
               "HC:SC": np.average(HC_SC_stress), "SC:HC": np.average(SC_HC_stress),
               "all": np.std(edge_stress),
               "HC:HC std": np.std(HC_HC_stress), "SC:SC std": np.std(SC_SC_stress),
               "HC:SC std": np.std(HC_SC_stress), "SC:HC std": np.std(SC_HC_stress),
               "all std": np.std(edge_stress),
               }
        return res

    def get_average_face_stress_by_number_of_HC_neighbors(self, relevant_effectors=None):
        if relevant_effectors is None:
            relevant_effectors = self.lateral_inhibition_model.stress_effectors
        face_stress = self.get_face_stress(relevant_effectors)
        face_types = self.sheet.face_df.type.values
        number_of_HC_neighbors = self.get_neighbors_data(self.get_num_of_HC_neighbors)
        res = dict()
        for n in range(np.max(number_of_HC_neighbors)+1):
            relevant_stresses = face_stress[np.logical_and(number_of_HC_neighbors == n, face_types == 0)]
            res["SC with %d HC neighbors N" % n] = relevant_stresses.size
            if relevant_stresses.size > 0:
                res["SC with %d HC neighbors avg" % n] = np.average(relevant_stresses)
            if relevant_stresses.size > 1:
                res["SC with %d HC neighbors std" % n] = np.std(relevant_stresses)
            relevant_stresses = face_stress[np.logical_and(number_of_HC_neighbors == n, face_types == 1)]
            res["HC with %d HC neighbors N" % n] = relevant_stresses.size
            if relevant_stresses.size > 0:
                res["HC with %d HC neighbors avg" % n] = np.average(relevant_stresses)
            if relevant_stresses.size > 1:
                res["HC with %d HC neighbors std" % n] = np.std(relevant_stresses)
        return res

    def get_average_edge_length(self, std=False):
        if std:
            return np.average(self.sheet.edge_df.length.values), np.std(self.sheet.edge_df.length.values)
        else:
            return np.average(self.sheet.edge_df.length.values)

    def get_average_face_perimeter(self, std=False):
        if std:
            return np.average(self.sheet.face_df.perimeter.values), np.std(self.sheet.face_df.perimeter.values)
        else:
            return np.average(self.sheet.face_df.perimeter.values)

    def get_contact_matrix(self):
        return self.sheet.get_contact_matrix()

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
        relevant_columns = ['notch_level', 'delta_level']
        if 'repressor_level' in self.sheet.face_df.columns:
            relevant_columns.append('repressor_level')
        levels = self.sheet.face_df.loc[:, relevant_columns]

        levels.to_pickle(file_path)

    def initialize_notch_delta(self, random_sensitivity=False, contact_dependent=False, saved_levels_file_path=None):
        maximal_delta_level = self.lateral_inhibition_model.get_maximal_delta_level()
        maximal_notch_level = self.lateral_inhibition_model.get_maximal_notch_level()
        maximal_repressor_level = self.lateral_inhibition_model.get_maximal_repressor_level()
        if saved_levels_file_path is not None and os.path.isfile(saved_levels_file_path):
            levels = pd.read_pickle(saved_levels_file_path)
            self.sheet.face_df.loc[:, 'notch_level'] = levels.notch_level.values
            self.sheet.face_df.loc[:, 'delta_level'] = levels.delta_level.values
            if contact_dependent:
                if 'repressor_level' in levels.columns:
                    self.sheet.face_df.loc[:, 'repressor_level'] = levels.repressor_level.values
                else:
                    self.sheet.face_df.loc[:, 'repressor_level'] = np.random.rand(self.sheet.face_df.shape[0]) * maximal_repressor_level
        else:
            self.sheet.face_df.loc[:, 'notch_level'] = np.random.rand(self.sheet.face_df.shape[0]) * maximal_notch_level
            self.sheet.face_df.loc[:, 'delta_level'] = np.random.rand(self.sheet.face_df.shape[0]) * maximal_delta_level
            if contact_dependent:
                self.sheet.face_df.loc[:, 'repressor_level'] = np.random.rand(self.sheet.face_df.shape[0]) * maximal_repressor_level
            # self.sheet.face_df.loc[:, 'notch_level'] = 1
            # self.sheet.face_df.loc[:, 'delta_level'] = 0
        if random_sensitivity:
            self.sheet.face_df.loc[:, 'notch_sensitivity'] = np.random.rand(self.sheet.face_df.shape[0])

    def simulate(self, t_end, dt, only_differentiation=False, random_forces=False,
                 aging_sensitivity=False, no_differentiation=False, contact_dependent_differentiation=False,
                 divisions=True, intercalations=True, delaminations=True, ablated_cells=[], sensitivity_aging_rate=0,
                 division_area=1.3, intercalation_length=0.04, delamination_area=0.1, delamination_rate=1.2,
                 viscosity=3, effectors=None, quasi_static=False, quasi_static_threshold=0.01, atoh_by_repressor=True):
        manager = EventManager("face")
        # manager.append(self.get_ablation_function(2))
        if not no_differentiation:
            if contact_dependent_differentiation:
                manager.append(self.lateral_inhibition_model.get_length_dependent_differentiation_function(dt=dt,
                                                                                                           quasi_static=quasi_static,
                                                                                                           atoh_by_repressor=atoh_by_repressor))
            else:
                manager.append(self.lateral_inhibition_model.get_differentiation_function(dt=dt))
        if aging_sensitivity:
            manager.append(self.lateral_inhibition_model.get_aging_sensitivity_function(rate=sensitivity_aging_rate, dt=dt))
        if not only_differentiation:
            if divisions:
                manager.append(self.topological_events_handler.get_division_function(crit_area=division_area))
            if intercalations:
                manager.append(self.topological_events_handler.get_intercalation_function(crit_edge_length=intercalation_length))
            if delaminations:
                manager.append(self.topological_events_handler.get_delamination_function(crit_area=delamination_area,
                                                                                         shrink_rate=delamination_rate))
            if len(ablated_cells) > 0:
                for cell in ablated_cells:
                    manager.append(
                        self.topological_events_handler.get_ablation_function(cell, shrink_rate=delamination_rate,
                                                                              critical_area=delamination_area))
            manager.append(self.sheet.get_update_virtual_vertices_function())

        if random_forces:
            manager.append(self.get_random_initializer(wait_time=1, dt=dt))
        history = History(self.sheet, save_every=0.1,save_all=False, dt=dt)
        model = self.get_model(only_differentiation, effectors=effectors)
        solver = IVPSolver(self, self.sheet, self.sheet.geom, model, manager=manager, history=history, auto_reconnect=False)
        self.sheet.vert_df['viscosity'] = viscosity
        # for diff in range(100):
        #     manager.execute(self.sheet)
        #     manager.update()
        solver.solve(tf=t_end, dt=dt, quasi_static=quasi_static, quasi_static_threshold=quasi_static_threshold)
        # fig, ax = plot_forces(self.sheet, geom, model, ['x', 'y'], 1)
        # plt.show()
        return history

    @staticmethod
    def get_draw_sheet_method( number_vertices=False, number_edges=False, number_faces=False, is_ordered=True,
                   for_labels=False, maximal_level=1, color_by="atoh", arrange_sheet=False):
        def draw_sheet(sheet):
            if arrange_sheet:
                sheet = InnerEarModel.arrange_sheet_from_history(sheet)
            # if not sheet.check_all_edge_order():
            #     print("bug in drawing")
            #     sheet.order_all_edges()
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
                if color_by == "atoh":
                    cmap_scale = sheet.face_df.atoh_level.to_numpy()
                elif color_by == "delta":
                    cmap_scale = sheet.face_df.delta_level.to_numpy() / maximal_level
                elif color_by == "inverse_repressor":
                    cmap_scale = (maximal_level - sheet.face_df.repressor_level.to_numpy()) / maximal_level
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
        return draw_sheet
    @staticmethod
    def save_sheet_labels_to_numpy(sheet, path):
        # Creating labeld image
        Nc = sheet.Nc
        draw_func =  InnerEarModel.get_draw_sheet_method(sheet, for_labels=True)
        fig, ax = draw_func(sheet)
        fig.tight_layout(pad=0)
        ax.set_axis_off()
        canvas = fig.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (4,))
        datar = data[:, :, 1].copy()
        datag = data[:, :, 2].copy()
        datab = data[:, :, 3].copy()
        datar[datab != 191] = 0
        datag[datab != 191] = 0
        labels = np.zeros(datar.shape)
        values1 = np.unique(datag)
        values0 = np.unique(datar)
        i = 1
        for v1 in values1:
            for v0 in values0:
                if i <= Nc:
                    if v0 != 0 or v1 != 0:
                        cell_pixels = np.logical_and(datar == v0, datag == v1)
                        if cell_pixels.any():
                            labels[np.logical_and(datar == v0, datag == v1)] = i
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

