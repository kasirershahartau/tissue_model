import numpy as np
from scipy.integrate import solve_ivp

class LateralInhibitionModel:
    def __init__(self, model, l=3, m=3, betaN=1, betaD=1, inhibition=False,
                 notch_repressor_degradation_ratio=1, length_normalization_factor=1, repressor_sensitivity=1,
                 atoh_sensitivity=1, delta_repressor_degradation_ratio=1, notch_delta_production_ratio=1,
                 stress_effectors=None, mechanosensitivity=0, stress_shift=0.0,
                 stress_hill_exponent=None):
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
        self.mechanosensitivity = mechanosensitivity  # psigma - the Hill constant
        # Stress SHIFT (K). The mechanosensitivity gate is
        #     increasing_hill(max(face_stress - K, 0), psigma)
        # so production is fully off below K and half-max at K + psigma. K exists
        # because the junction (perimeter-only) stress this gates on is NEGATIVE
        # for support cells, and the old max(stress, 0) clipped every SC to zero,
        # destroying the discrimination. K = 0 reproduces the historical gate
        # exactly. NOTE K is not a redundant offset: the Hill depends on the RATIO
        # psigma/(s-K), so K also sets the gate's STEEPNESS (slope at half-max =
        # 3/(4*psigma)). Set K just below the stress you intend to block; putting
        # it far below flattens the gate and washes out the selectivity.
        self.stress_shift = stress_shift
        # Hill exponent for the MECHANOSENSITIVITY gate only. None -> self.m (3),
        # an exact no-op. It is separate from self.m because that exponent is also
        # used by the repressor and Atoh1 Hills, so raising it globally would
        # change the whole LI model. A larger value sharpens the stress switch,
        # which is what lets one psigma leave E17.5 untouched while gating P0
        # (their isolated-SC stresses differ by only ~0.003).
        self.stress_hill_exponent = stress_hill_exponent

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

    def get_length_dependent_differentiation_function(self, dt=1., quasi_static=False, atoh_by_repressor=False):

        def differentiation(sheet, manager):
            # Notch and delta levels of each edge.
            # get_contact_matrix() is indexed by unique_id and sized
            # max(unique_id)+1, while every vector below comes from face_df in
            # ROW order. Those agree only while unique_id == row position for
            # every face - true at t=0, false as soon as a face is REMOVED
            # (ablation, delamination), which left the matrix one row/column
            # larger than the state vectors and blew up the matmul. Take the
            # submatrix for the unique_ids actually present, in face_df order,
            # so the two are aligned by construction.
            uids = sheet.face_df["unique_id"].to_numpy(int)
            contact_matrix = self.model.get_contact_matrix()[np.ix_(uids, uids)]
            face_data = sheet.face_df[['repressor_level', 'notch_level', 'delta_level', 'perimeter', 'notch_sensitivity']].copy()
            n_faces = face_data.shape[0]

            initial_y = np.hstack((face_data.notch_level.to_numpy(),
                                   face_data.delta_level.to_numpy(),
                                   face_data.repressor_level.to_numpy()))
            if self.mechanosensitivity > 0:
                face_data["stress"] = self.model.get_face_stress(self.stress_effectors)

            face_sensitivity = face_data.notch_sensitivity.to_numpy()
            face_perimeter = face_data.perimeter.to_numpy()

            # A cell tagged for removal (type == -1, set by BOTH the ablation and
            # the delamination handlers) is collapsing: its prefered_area and
            # prefered_perimeter are driven to 0, so its perimeter tends to 0 and
            # the 1/face_perimeter factors below diverge. That makes the ODE stiff
            # enough that solve_ivp never returns - a HANG, not a crash, which is
            # exactly how it presented. A dying cell is not signalling anyway, so
            # drop it from the coupling and give it a harmless unit perimeter.
            # No-op whenever nothing is being ablated or delaminated, which is why
            # this never surfaced: the mechanical fit's ablation runs all pass
            # no_differentiation=True, so this ODE never ran beside a dying cell.
            dying = (sheet.face_df["type"].to_numpy() == -1)
            if dying.any():
                contact_matrix = contact_matrix.copy()
                contact_matrix[dying, :] = 0.0
                contact_matrix[:, dying] = 0.0
                face_perimeter = face_perimeter.copy()
                face_perimeter[dying] = 1.0

            def lateral_inhibition_ode(t, y):
                notch_level = y[:n_faces]
                delta_level = y[n_faces:2*n_faces]
                repressor_level = y[2*n_faces:]

                delta_production = self.decreasing_hill(repressor_level, self.repressor_sensitivity)
                if self.mechanosensitivity > 0:
                    face_stress = face_data.stress.to_numpy() / self.length_normalization_factor
                    delta_production = delta_production * self.increasing_hill(
                            np.maximum(face_stress - self.stress_shift, 0), self.mechanosensitivity,
                            self.stress_hill_exponent)
                notch_with_neighboring_delta_interaction = (notch_level * (self.length_normalization_factor / face_perimeter) *
                                           np.matmul(contact_matrix, (delta_level / face_perimeter))) # Si - notch in cell with delta in neighbors
                delta_with_neighboring_notch_interaction = (delta_level * (self.length_normalization_factor / face_perimeter) *
                                           np.matmul(contact_matrix, (notch_level / face_perimeter))) # Ti - delta in cell with notch in neighbors
                repressor_production = 0 if self.inhibition else self.increasing_hill(notch_with_neighboring_delta_interaction, face_sensitivity)
                notch_change = 1 - self.notch_repressor_degradation_ratio * notch_level - notch_with_neighboring_delta_interaction
                delta_change = (delta_production - self.delta_repressor_degradation_ratio * delta_level -
                                self.notch_delta_production_ratio * delta_with_neighboring_notch_interaction)
                repressor_change = repressor_production - repressor_level
                return np.hstack((notch_change, delta_change, repressor_change))

            final_y = solve_ivp(lateral_inhibition_ode, (0, dt), initial_y, t_eval=[dt]).y[:,0]
            final_notch_level = final_y[:n_faces]
            final_notch_level = np.clip(final_notch_level, a_min=0, a_max=1)
            final_delta_level = final_y[n_faces:2 * n_faces]
            final_delta_level = np.clip(final_delta_level, a_min=0, a_max=1)
            final_repressor_level = final_y[2 * n_faces:]
            final_repressor_level = np.clip(final_repressor_level, a_min=0, a_max=1)
            sheet.face_df.loc[:, "notch_level"] = final_notch_level
            sheet.face_df.loc[:, "delta_level"] = final_delta_level
            sheet.face_df.loc[:, "repressor_level"] = final_repressor_level
            if atoh_by_repressor:
                atoh_levels = self.get_atoh_level(final_repressor_level, activation=False)
            else:
                atoh_levels = self.get_atoh_level(final_delta_level, activation=True)
            sheet.face_df.loc[:,"atoh_level"] = atoh_levels
            self.model.update_cell_type_parameters(atoh_levels)
            if not quasi_static:
                manager.append(differentiation)
        return differentiation

    def increasing_hill(self, x, a, m=None):
        m = self.m if m is None else m
        return (x ** m) / (a ** m + x ** m)

    def decreasing_hill(self, x, a):
        return (a**self.l) / ((a**self.l) + (x ** self.l))

    def get_atoh_level(self, level, activation=False):
        if activation:
            return self.increasing_hill(level, self.atoh_sensitivity)
        else:
            return self.decreasing_hill(level, self.atoh_sensitivity)


    def get_aging_sensitivity_function(self, rate, dt=1.):
        def aging_sensitivity(sheet, manager):
            sheet.face_df.loc[:, "notch_sensitivity"] = sheet.face_df.notch_sensitivity.values + rate**dt
            manager.append(aging_sensitivity)
        return aging_sensitivity