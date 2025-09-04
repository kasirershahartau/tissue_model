import numpy as np
from scipy.integrate import solve_ivp

class LateralInhibitionModel:
    def __init__(self, model):
        self.model = model

    def get_differentiation_function(self, l, m, mu, rho, betaN, betaD, inhibition=False, dt=1.):
        def differentiation(sheet, manager):
            # Notch and delta levels of each cell
            levels = sheet.face_df.loc[:,['notch_level', 'delta_level', 'notch_sensitivity']]
            notch_level = levels.notch_level.to_numpy()
            delta_level = levels.delta_level.to_numpy()
            sensitivity = levels.notch_sensitivity.to_numpy()
            # Mean notch and delta of neighboring cells
            neigh_delta = self.model.get_neighbors_data(self.model.mean_delta).to_numpy()

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
            self.model.update_cell_type_parameters(new_delta)
            manager.append(differentiation)
        return differentiation

    def get_length_dependent_differentiation_function(self, l, m, mu, rho, xhi, betaN, betaD, betaR, kt, gammaR,
                                                      inhibition=False, tension_effectors=None,
                                                      mechanosensitivity=0):
        if inhibition:
            return self.get_differentiation_function(l, m, mu, rho, inhibition=inhibition)
        def differentiation(sheet, manager, dt=1.):
            # Notch and delta levels of each edge
            edge_data = sheet.edge_df[['face', 'opposite', 'length']].copy()
            face_data = sheet.face_df[['repressor_level', 'notch_level', 'delta_level', 'perimeter', 'notch_sensitivity']].copy()
            n_edges = edge_data.shape[0]
            n_faces = face_data.shape[0]
            face_list, number_of_edges = np.unique(edge_data.face.values, return_counts=True)
            face_data.loc[face_list, 'n_edges'] = number_of_edges
            face_data['notch_level_per_length'] = face_data.eval('notch_level / perimeter')
            face_data['delta_level_per_length'] = face_data.eval('delta_level / perimeter')
            for to_key, from_key in zip(['notch_level_per_length', 'delta_level_per_length', 'face_perimeter', 'face_repressor_level'],
                                        ['notch_level_per_length','delta_level_per_length', 'perimeter', 'repressor_level']):
                edge_data.loc[:, to_key] = face_data.loc[edge_data.face.values, from_key].to_numpy()
            # Redistributing ligands on membranes according to fast diffusion assumption
            edge_data["notch_level"] = edge_data.eval('notch_level_per_length * length')
            edge_data["delta_level"] = edge_data.eval('delta_level_per_length * length')
            # Rearranging data to use scipy's solve_ivp y=[notch, delta, repressor]
            edge_data["cardinal_index"] = np.arange(n_edges)
            face_data["cardinal_index"] = np.arange(n_faces)
            has_opposite = edge_data.opposite.values >= 0
            opposite_index = edge_data.cardinal_index[edge_data.opposite.values[has_opposite]].values
            matching_faces = face_data.loc[edge_data["face"].values, "cardinal_index"]

            initial_y = np.hstack((edge_data.notch_level.to_numpy(),
                                   edge_data.delta_level.to_numpy(),
                                   face_data.repressor_level.to_numpy()))
            if mechanosensitivity > 0:
                edge_data["tension"] = self.model.get_edge_tension(tension_effectors)

            edge_length = edge_data.length.to_numpy()
            matching_faces_perimeter = edge_data.face_perimeter.to_numpy()
            face_sensitivity = face_data.notch_sensitivity.to_numpy()

            def f(x, a):
                return (x**m)/(a + x**m)

            def g(x):
                return 1/(1 + (x**l))

            def lateral_inhibition_ode(t, y):
                notch_level = y[:n_edges]
                delta_level = y[n_edges:2*n_edges]
                repressor_level = y[2*n_edges:]
                matching_face_repressor = repressor_level[matching_faces]
                delta_production = g(matching_face_repressor) * betaD / matching_faces_perimeter
                if mechanosensitivity > 0:
                    edge_tension = edge_data.tension.to_numpy()
                    delta_production = delta_production * f(np.maximum(edge_tension, 0), mechanosensitivity)
                notch_production = betaN / matching_faces_perimeter
                opposite_delta_level = np.zeros((n_edges,))
                opposite_delta_level[has_opposite] = delta_level[opposite_index]
                edge_notch_delta_interaction_level = notch_level * opposite_delta_level * edge_length
                opposite_notch_level = np.zeros((n_edges,))
                opposite_notch_level[has_opposite] = notch_level[opposite_index]
                face_interactions_level = np.bincount(matching_faces, weights=edge_notch_delta_interaction_level,
                                                      minlength=n_faces)
                repressor_production = f(face_interactions_level, face_sensitivity) * betaR
                notch_change = mu*(notch_production - notch_level - notch_level * opposite_delta_level / kt)
                delta_change = rho*(delta_production - delta_level - delta_level * opposite_notch_level / kt)
                repressor_change = xhi*(repressor_production - gammaR * repressor_level)
                return np.hstack((notch_change, delta_change, repressor_change))

            final_y = solve_ivp(lateral_inhibition_ode, (0, dt), initial_y, t_eval=[dt]).y[:,0]
            final_notch_level = final_y[:n_edges]
            final_delta_level = final_y[n_edges:2 * n_edges]
            final_repressor_level = final_y[2 * n_edges:]
            edge_data.loc[:, "notch_level"] = final_notch_level
            edge_data.loc[:, "delta_level"] = final_delta_level
            new_notch_delta_levels = edge_data.groupby("face")[["notch_level", "delta_level"]].sum()
            sheet.face_df.update(new_notch_delta_levels)
            sheet.face_df.loc[:, "repressor_level"] = final_repressor_level
            self.model.update_cell_type_parameters(sheet.face_df.delta_level.values)
            manager.append(differentiation, dt=dt)
        return differentiation

    def get_aging_sensitivity_function(self, rate, dt):
        def aging_sensitivity(sheet, manager):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
        return aging_sensitivity