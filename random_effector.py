"""An effector applying an uncorrelated random force to every vertex.

Adds noise to the dynamics — enough to dislodge a tissue sitting in a shallow
local minimum, or to test that a result does not depend on a perfectly symmetric
initial condition. It has no energy to speak of: only the gradient is meaningful,
so it must not be used where the energy is minimised rather than integrated.
"""
from tyssue.dynamics.effectors import AbstractEffector, units
import numpy as np
import pandas as pd

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