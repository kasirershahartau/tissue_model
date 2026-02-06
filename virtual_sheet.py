from tyssue import Sheet, config
from tyssue import PlanarGeometry as geom
from tyssue.topology.base_topology import add_vert, collapse_edge
import numpy as np

class VirtualSheet(Sheet):
    """ An epithelium tissue with virtual vertices, to allow for rounded apical morphology"""

    def __init__(self, identifier, datasets, specs=None, coords=None, maximal_bond_length=0.1,
                 minimal_bond_length=0.05):
        """
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        datasets : dictionary of dataframes
            The keys correspond to the different geometrical elements
            constituting the epithelium:

            * `vert` contains a dataframe of vertices,
            * `edge` contains a dataframe of *oriented* half-edges between vertices,
            * `face` contains a dataframe of polygonal faces enclosed by half-edges,
            * `cell` contains a dataframe of polyhedral cells delimited by faces,
        virtual_vert_it: The number of virtual vertices adding iterations. On each iteration each edge is splitted
         into 2. For example: 2 iterations add 3 virtual vertices to each edge.

        """
        super().__init__(identifier, datasets, specs, coords)
        self.update_specs({"vert": {"is_virtual": int(0)}, "edge": {"order": int(0)}})
        self.maximal_bond_length = maximal_bond_length
        self.minimal_bond_length = minimal_bond_length
        self.geom = geom

    def set_periodic_boundary_condition(self):
        self.specs = config.geometry.planar_periodic_sheet()

    def initiate_edge_order(self):
        self.sanitize(trim_borders=True, order_edges=True)
        self.reset_index(order=True)
        self.geom.update_all(self)
        self.get_opposite()
        face_list = self.edge_df.face.to_numpy()
        edges_order = np.zeros((len(face_list,)))
        counter = 0
        current_face = -1
        for idx in range(edges_order.size):
            if face_list[idx] != current_face:
                current_face = face_list[idx]
                counter = 0
            counter += 1
            edges_order[idx] = counter
        self.edge_df.loc[:, 'order'] = edges_order.astype(int)

    def set_maximal_bond_length(self, length):
        self.maximal_bond_length = length

    def set_minimal_bond_length(self, length):
        self.minimal_bond_length = length

    def order_all_edges(self):
        for face in self.face_df.index.values:
            self.order_edges(face)

    def order_edges(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number)
        self.edge_df.loc[edges.index, "order"] = 0
        current_edge = edges.iloc[0]
        current_edge_order = 1
        while self.edge_df.at[current_edge.name, "order"] < 1:
            self.edge_df.at[current_edge.name, "order"] = current_edge_order
            edge_trgt = current_edge.trgt
            current_edge = edges.query("srce == %d" %edge_trgt).iloc[0]
            current_edge_order += 1

    def check_edge_order(self, face_number):
        edges = self.edge_df.query("face == %d" % face_number).loc[:,["order", "srce", "trgt"]]
        edges.sort_values(["order"], inplace=True)
        first_srce = -1
        current_trgt = -1
        for index, row in edges.iterrows():
            if first_srce < 0:
                first_srce = row.srce
            if current_trgt > 0 and current_trgt != row.srce:
                return False
            current_trgt = row.trgt
        return row.trgt == first_srce

    def check_all_edge_order(self):
        for face in self.face_df.index.values:
            if not self.check_edge_order(face):
                print("wrong order in face %d" %face)
                return False
        return True



    def add_virtual_vertices(self):
        long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
        # np.random.shuffle(long)
        while long.size > 0:
            edge_ind = long[0]
            edge_order = self.edge_df.at[edge_ind, "order"]
            edge_face = self.edge_df.at[edge_ind, "face"]
            new_vert, new_edge, new_opposite_edge = add_vert(self, edge_ind)
            self.vert_df.at[new_vert, "is_virtual"] = 1
            self.edge_df.at[edge_ind, "length"] /= 2
            self.edge_df.at[new_edge, "length"] /= 2
            increase_order = self.edge_df.query("face == %d and order > %d" %(edge_face, edge_order))
            self.edge_df.at[new_edge, "order"] = edge_order + 1
            self.edge_df.loc[increase_order.index, "order"] += 1
            opposite = int(self.edge_df.loc[edge_ind, "opposite"])
            if opposite >= 0:
                self.edge_df.at[opposite, "length"] /= 2
                if new_opposite_edge is None:
                    self.edge_df.at[new_edge, "opposite"] = -1
                else:
                    opposite_order = self.edge_df.at[opposite, "order"]
                    opposite_face = self.edge_df.at[opposite, "face"]
                    self.edge_df.at[new_edge, "opposite"] = new_opposite_edge
                    self.edge_df.at[new_opposite_edge, "opposite"] = new_edge
                    self.edge_df.at[new_opposite_edge, "length"] /= 2
                    increase_order = self.edge_df.query("face == %d and order > %d" % (opposite_face, opposite_order))
                    self.edge_df.at[opposite, "order"] = opposite_order + 1
                    self.edge_df.loc[increase_order.index, "order"] += 1
            long = self.edge_df[self.edge_df["length"] > self.maximal_bond_length].index.to_numpy()
            np.random.shuffle(long)
        self.edge_df.index.name = 'edge'
        self.vert_df.index.name = 'vert'
        self.geom.update_all(self)
        # self.reset_index(order=False)
        self.edge_df.sort_values(["face", "order"], inplace=True)
        self.get_opposite()
        # if not self.check_all_edge_order():
        #     print("bug in adding virtual vertices")

    def remove_virtual_vertex(self, edge_id):
        srce_idx = self.edge_df.loc[edge_id].srce
        trgt_idx = self.edge_df.loc[edge_id].trgt
        srce = self.vert_df.loc[srce_idx]
        trgt = self.vert_df.loc[trgt_idx]
        if srce.is_virtual == 1 and trgt.is_virtual != 1:  # if only one is virtual, collapse to the real vertex
            self.vert_df.loc[srce_idx, self.coords] = self.vert_df.loc[trgt_idx, self.coords]
        elif trgt.is_virtual == 1 and srce.is_virtual != 1:
            self.vert_df.loc[trgt_idx, self.coords] = self.vert_df.loc[srce_idx, self.coords]
        # involved_faces.append(self.edge_df.at[short[0], "face"])
        # opposite_edge = self.edge_df.at[short[0], "opposite"]
        # if opposite_edge > 0:
        #     involved_faces.append(self.edge_df.at[opposite_edge, "face"])
        collapse_edge(self, edge_id, allow_two_sided=False, reindex=True)
        return 0

    def remove_virtual_vertices(self):
        # involved_faces = []
        short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
        if short.size > 0:
            short = short[self.is_virtual_edge(short)]
        np.random.shuffle(short)
        while short.size > 0:
            self.remove_virtual_vertex(short[0])
            short = self.edge_df[self.edge_df["length"] < self.minimal_bond_length].index.to_numpy()
            if short.size > 0:
                short = short[self.is_virtual_edge(short)]
            np.random.shuffle(short)
        # for face in np.unique(involved_faces):
        #     self.order_edges(face)
        # sheet.edge_df.sort_values(["face", "order"], inplace=True)
        # sheet.get_opposite()
        # self.geom.update_all(self)
        # if not self.check_all_edge_order():
        #     print("bug in removing virtual vertices")
        return 0

    def is_virtual_edge(self, edge_indices):
        """
        Checks if an edge contains a virtual vertex
        """
        if hasattr(edge_indices, "__len__"):
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual.to_numpy() == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual.to_numpy() == 1
            return np.logical_or(srce_is_virtual, trgt_is_virtual)
        else:
            srce_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].srce].is_virtual == 1
            trgt_is_virtual = self.vert_df.loc[self.edge_df.loc[edge_indices].trgt].is_virtual == 1
            return srce_is_virtual or trgt_is_virtual

    def get_update_virtual_vertices_function(self):
        def update_virtual_vertices(sheet, manager):
            sheet.remove_virtual_vertices()
            sheet.add_virtual_vertices()
            manager.append(update_virtual_vertices)
            return
        return update_virtual_vertices

    def get_neighbors(self, face, elem="face"):
        face_edges = self.edge_df.query("face == %d" % face)
        opposite_edges = face_edges.opposite.to_numpy()
        neighbors = self.edge_df.loc[opposite_edges[opposite_edges >= 0], "face"].to_numpy()
        return np.unique(neighbors)

    def get_contact_matrix(self):
        has_opposite = self.edge_df.opposite >= 0
        faces_with_neighbors_ids = self.edge_df.loc[has_opposite, "face"].to_numpy()
        neighbor_ids = self.edge_df.loc[self.edge_df.opposite[has_opposite], "face"].to_numpy()
        contact_length = self.edge_df.loc[self.edge_df.opposite[has_opposite], "length"].to_numpy()
        number_of_faces = self.face_df.shape[0]
        m = np.bincount(faces_with_neighbors_ids*number_of_faces + neighbor_ids, weights=contact_length,
                         minlength=number_of_faces*number_of_faces).reshape(number_of_faces, number_of_faces)
        return m