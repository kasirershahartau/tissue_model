import sys
sys.path.insert(1, 'C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\tissue_model\\tyssue\\tyssue')
from tyssue.dynamics.effectors import AbstractEffector, units, to_nd, LineTension, FaceAreaElasticity, FaceContractility
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