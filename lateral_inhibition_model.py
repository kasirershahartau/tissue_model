import numpy as np
from scipy.integrate import solve_ivp

class LateralInhibitionModel:
    def __init__(self, model, l=3, m=3, betaN=1, betaD=1, inhibition=False,
                 notch_repressor_degradation_ratio=1, length_normalization_factor=1, repressor_sensitivity=1,
                 atoh_sensitivity=1, delta_repressor_degradation_ratio=1, notch_delta_production_ratio=1,
                 stress_effectors=None, mechanosensitivity=0):
        self.model = model
        # General parameters
        self.inhibition = inhibition
        self.l = l
        self.m = m

        # Classical model (no contact dependent) parameters
        self.betaN = betaN
        self.betaD = betaD

        # Contact dependent model parameters
        self.notch_repressor_degradation_ratio = notch_repressor_degradation_ratio  # tauN
        self.delta_repressor_degradation_ratio = delta_repressor_degradation_ratio  # tauR
        self.length_normalization_factor = length_normalization_factor  # L0
        self.repressor_sensitivity = repressor_sensitivity  # pR
        self.atoh_sensitivity = atoh_sensitivity  # patoh
        self.notch_delta_production_ratio = notch_delta_production_ratio  # alpha

        self.stress_effectors = stress_effectors
        self.mechanosensitivity = mechanosensitivity

    def get_maximal_delta_level(self):
        return self.betaD

    def get_maximal_notch_level(self):
        return self.betaN

    def get_maximal_repressor_level(self):
        return self.betaN / self.notch_repressor_degradation_ratio

    def get_differentiation_function(self, dt=1.):
        def differentiation(sheet, manager):
            # Notch and delta levels of each cell
            levels = sheet.face_df.loc[:,['notch_level', 'delta_level', 'notch_sensitivity']]
            notch_level = levels.notch_level.to_numpy()
            delta_level = levels.delta_level.to_numpy()
            sensitivity = levels.notch_sensitivity.to_numpy()
            # Mean notch and delta of neighboring cells
            neigh_delta = self.model.get_neighbors_data(self.model.mean_delta).to_numpy()

            def f(x, a):
                return (x**self.l)/(a + x**self.l)

            def g(x):
                return 1/(1 + x**self.m)
            if self.inhibition:
                new_notch = notch_level - dt * notch_level
                new_delta = delta_level + dt * (1 - delta_level)
            else:
                new_notch = notch_level + dt * (self.betaN * f(neigh_delta, sensitivity) - notch_level)
                new_delta = delta_level + dt * (self.betaD * g(notch_level) - delta_level)
            sheet.face_df.loc[:, 'notch_level'] = new_notch
            sheet.face_df.loc[:, 'delta_level'] = new_delta
            self.model.update_cell_type_parameters(new_delta)
            manager.append(differentiation)
        return differentiation

    def get_length_dependent_differentiation_function(self, dt=1., quasi_static=False):

        def differentiation(sheet, manager):
            # Notch and delta levels of each edge
            contact_matrix = self.model.get_contact_matrix()
            face_data = sheet.face_df[['repressor_level', 'notch_level', 'delta_level', 'perimeter', 'notch_sensitivity']].copy()
            n_faces = face_data.shape[0]

            initial_y = np.hstack((face_data.notch_level.to_numpy(),
                                   face_data.delta_level.to_numpy(),
                                   face_data.repressor_level.to_numpy()))
            if self.mechanosensitivity > 0:
                face_data["stress"] = self.model.get_face_stress(self.stress_effectors)

            face_sensitivity = face_data.notch_sensitivity.to_numpy()
            face_perimeter = face_data.perimeter.to_numpy()

            def lateral_inhibition_ode(t, y):
                notch_level = y[:n_faces]
                delta_level = y[n_faces:2*n_faces]
                repressor_level = y[2*n_faces:]

                delta_production = self.decreasing_hill(repressor_level, self.repressor_sensitivity)
                if self.mechanosensitivity > 0:
                    face_stress = face_data.stress.to_numpy() / self.length_normalization_factor
                    delta_production = delta_production * self.increasing_hill(np.maximum(face_stress, 0), self.mechanosensitivity)
                delta_notch_interaction = (notch_level * (self.length_normalization_factor / face_perimeter) *
                                           np.matmul(contact_matrix, (delta_level / face_perimeter)))
                notch_delta_interaction = (delta_level * (self.length_normalization_factor / face_perimeter) *
                                           np.matmul(contact_matrix, (notch_level / face_perimeter)))
                repressor_production = 0 if self.inhibition else self.increasing_hill(notch_delta_interaction, face_sensitivity)
                notch_change = 1 - self.notch_repressor_degradation_ratio * notch_level - notch_delta_interaction
                delta_change = (delta_production - self.delta_repressor_degradation_ratio * delta_level -
                                self.notch_delta_production_ratio * delta_notch_interaction)
                repressor_change = repressor_production - repressor_level
                return np.hstack((notch_change, delta_change, repressor_change))

            final_y = solve_ivp(lateral_inhibition_ode, (0, dt), initial_y, t_eval=[dt]).y[:,0]
            final_notch_level = final_y[:n_faces]
            final_delta_level = final_y[n_faces:2 * n_faces]
            final_repressor_level = final_y[2 * n_faces:]
            sheet.face_df.loc[:, "notch_level"] = np.clip(final_notch_level, a_min=0, a_max=1)
            sheet.face_df.loc[:, "delta_level"] = np.clip(final_delta_level, a_min=0, a_max=1)
            final_repressor_level = np.clip(final_repressor_level, a_min=0, a_max=1)
            sheet.face_df.loc[:, "repressor_level"] = final_repressor_level
            atoh_levels = self.get_atoh_level(final_repressor_level)
            sheet.face_df.loc[:,"atoh_level"] = atoh_levels
            self.model.update_cell_type_parameters(atoh_levels)
            if not quasi_static:
                manager.append(differentiation)
        return differentiation

    def increasing_hill(self, x, a):
        return (x ** self.m) / (a ** self.m + x ** self.m)

    def decreasing_hill(self, x, a):
        return 1 / (1 + (a * x) ** self.l)

    def get_atoh_level(self, repressor_level):
        return self.decreasing_hill(repressor_level, self.atoh_sensitivity)

    def get_aging_sensitivity_function(self, rate, dt=1.):
        def aging_sensitivity(sheet, manager):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
            manager.append(aging_sensitivity)
        return aging_sensitivity