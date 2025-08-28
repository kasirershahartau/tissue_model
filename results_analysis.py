import numpy as np
import pandas as pd

def get_neighbors_from_history(history, face, time):
    edge_time_df = history.edge_h.get(['face', 'oposite', 'time'])
    edge_df = edge_time_df.query("time == %f" % time)
    face_edges = edge_time_df.query("face == %d" % face)
    opposite_edges = face_edges.opposite.to_numpy()
    neighbors = edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
    return np.unique(neighbors)


def find_differentiation_events(history):
    face_time_df = history.face_h.get(['face', 'type', 'time'])
    previously_SC = set()
    differentiation_df = pd.DataFrame(columns=["time", "face", "HC_neighbors"])
    for time in np.sort(np.unique(face_time_df.time.to_numpy())):
        currently_HC = set(face_time_df.query("type == 1 and time == %f" % time).face)
        differentiating_faces = previously_SC.intersection(currently_HC)
        for face in differentiating_faces:
            neighbors = set(get_neighbors_from_history(history, face, time))
            HC_neighbors_number = len(neighbors.intersection(currently_HC))
            differentiation_df.append({"time": time, "face": face, "HC_neighbors_number": HC_neighbors_number},
                                      ignore_index=True)
        previously_SC = set(face_time_df.query("type == 0 and time == %f" % time).face)
    return differentiation_df