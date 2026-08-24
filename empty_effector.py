"""An effector contributing no energy and no gradient.

Useful as a placeholder when a model is assembled from a variable list of
effectors: including this keeps the stack's shape fixed without changing the
dynamics, which avoids branching on an empty list.
"""
from tyssue.dynamics.effectors import AbstractEffector, units
import pandas as pd


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