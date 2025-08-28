import numpy as np

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
            self.model.update_cell_type_parameters(sheet.face_df.delta_level.values)
            manager.append(differentiation, dt=dt)
        return differentiation

    def get_aging_sensitivity_function(self, rate, dt):
        def aging_sensitivity(sheet, manager):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
        return aging_sensitivity