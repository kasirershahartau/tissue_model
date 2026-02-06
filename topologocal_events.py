from tyssue.topology.sheet_topology import cell_division, type1_transition
from tyssue.behaviors.sheet.actions import remove
import numpy as np

class TopologicalEventsHandler:
    def __init__(self, model):
        self.model = model

    def get_ablation_function(self, cell_id, shrink_rate=1.5, critical_area=0.01):

        def ablation(sheet, manager):
            sheet.face_df.loc[cell_id, "type"] = -1
            sheet.face_df.loc[cell_id, "contractility"] = 10
            sheet.face_df.loc[cell_id, "area_elasticity"] = 20
            sheet.face_df.loc[cell_id, "preferred_area"] = 0
            sheet.face_df.loc[cell_id, "preferred_volume"] = 0
            return
        return ablation


    def get_delamination_function(self, crit_area=0.5, shrink_rate=1.2):

        def delamination(sheet, manager):
            delaminating_faces = sheet.face_df.query("area < %f" % crit_area)
            for cell_id, row in delaminating_faces.iterrows():
                # Do delamination
                sheet.face_df.loc[cell_id, "type"] = -1
                sheet.face_df.loc[cell_id, "area_elasticity"] = 20
                sheet.face_df.loc[cell_id, "contractility"] = 10
                sheet.face_df.at[cell_id, "prefered_area"] = 0
                sheet.face_df.at[cell_id, "prefered_vol"] = 0
                if sheet.face_df.loc[cell_id, "num_sides"] <= 3:
                    remove(sheet, cell_id, self.model.sheet.geom)
                    sheet.reset_index(order=False)
                    sheet.order_all_edges()
                    sheet.edge_df.sort_values(["face", "order"], inplace=True)
                    sheet.get_opposite()
                    # update geometry
                    sheet.geom.update_all(sheet)
            manager.append(delamination)
            return
        return delamination



    def get_division_function(self, crit_area):
        def division(sheet, manager):
            """Defines a division behavior."""
            dividing_faces = sheet.face_df.query("area > %f & type == 0" % crit_area)
            for cell_id, row in dividing_faces.iterrows():
                # Do division
                daughter = cell_division(sheet, cell_id, sheet.geom)[0]
                sheet.face_df.at[daughter, "id"] = daughter
                # Update the topology
                sheet.get_opposite()
                sheet.reset_index(order=False)
                sheet.order_all_edges()
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                # update geometry
                sheet.geom.update_all(sheet)
            manager.append(division)
        return division

    def get_intercalation_function(self, crit_edge_length):
        """Defines an intercalation behavior."""
        def intercalation(sheet, manager):
            is_virtual = sheet.is_virtual_edge(np.arange(sheet.edge_df.shape[0]))
            real_edges = sheet.edge_df[~is_virtual]
            intercalating_edges = real_edges.query("is_active > 0 & length < %f" % crit_edge_length)
            for edge_id, row in intercalating_edges.iterrows():
                # find involved cells
                vertices = intercalating_edges.loc[edge_id, ["srce", "trgt"]]
                involved_edges = intercalating_edges.query("srce in [%d, %d] or trgt in [%d, %d]" % (vertices.values[0],
                                                                                               vertices.values[1],
                                                                                               vertices.values[0],
                                                                                               vertices.values[1]))
                involved_faces = np.unique(involved_edges.face.to_numpy())
                # Do intercalation
                type1_transition(sheet, edge_id)
                # Update the topology
                sheet.reset_index(order=False)
                sheet.order_all_edges()
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                # update geometry
                sheet.geom.update_all(sheet)
            manager.append(intercalation)
        return intercalation