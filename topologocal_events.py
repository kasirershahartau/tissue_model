from tyssue.topology.sheet_topology import cell_division, type1_transition
from tyssue.behaviors.sheet import apoptosis
import numpy as np

class TopologicalEventsHandler:
    def __init__(self, model):
        self.model = model

    def get_ablation_function(self, cell_id, shrink_rate=1.5, critical_area=0.01):
        self.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': critical_area,
            'radial_tension': 0.2,
            'contractile_increase': 0.3,
            'contract_span': 2,
            'geom': self.model.sheet.geom,
            'neighbors': self.model.get_neighbors(cell_id)
        }
        def ablation(sheet, manager):
            # Do delamination
            sheet.face_df.loc[cell_id, "type"] = -1
            sheet.face_df.loc[cell_id, "area_elasticity"] = 5
            manager.append(apoptosis, face_id=cell_id, **sheet.settings['apoptosis'])
            return
        return ablation


    def get_delamination_function(self, crit_area=0.5, shrink_rate=1.2):
        self.model.sheet.settings['apoptosis'] = {
            'shrink_rate': shrink_rate,
            'critical_area': crit_area/2,
            'radial_tension': 0.2,
            'contractile_increase': 0.3,
            'contract_span': 2,
            'geom': self.model.sheet.geom
        }

        def delamination(sheet, manager):
            delaminating_faces = sheet.face_df.query("area < %f" % crit_area)
            for cell_id, row in delaminating_faces.iterrows():
                # Do delamination
                sheet.face_df.loc[cell_id, "type"] = -1
                manager.append(apoptosis, face_id=cell_id, **sheet.settings['apoptosis'])
                sheet.face_df.at[cell_id, "prefered_area"] = sheet.face_df.at[cell_id, "prefered_vol"]
                involved_faces = self.model.get_neighbors(cell_id)
                for face in involved_faces:
                    sheet.order_edges(face)
                # sheet.reset_index(order=False)
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                # update geometry
                sheet.geom.update_all(sheet)
                if not sheet.check_all_edge_order():
                    print("bug in delamination")
            return
        return delamination



    def get_division_function(self, crit_area):
        def division(sheet, manager):
            """Defines a division behavior."""
            dividing_faces = sheet.face_df.query("area > %f & type == 0" % crit_area)
            for cell_id, row in dividing_faces.iterrows():
                # Do division
                daughter = cell_division(sheet, cell_id, sheet.geom)[0]
                # Update the topology
                sheet.get_opposite()
                involved_faces = np.intersect1d(self.model.get_neighbors(cell_id), self.model.get_neighbors(daughter))
                involved_faces = np.hstack([involved_faces, np.array([cell_id, daughter])]).astype(int)
                for face in involved_faces:
                    sheet.order_edges(face)
                # sheet.reset_index(order=False)
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
                for face in involved_faces:
                    sheet.order_edges(face)
                sheet.reset_index(order=False)
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                # update geometry
                sheet.geom.update_all(sheet)
                if not sheet.check_all_edge_order():
                    print("bug in intercalation")
                break
            manager.append(intercalation)
        return intercalation