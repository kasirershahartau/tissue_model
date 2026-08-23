import numpy as np


def get_specs_2d(self):
    specs = {'vert': {'is_active':1,
                      'radial_tension': 0.},
             'edge': {'is_active': 1,
                      'sub_area': 6.,
                      'notch_level': 1.,
                      'delta_level': 1.,
                      'tension_before_division': 0.,
                      },
             'face': {'notch_level': 1.,
                      'delta_level': 1.,
                      'repressor_level': 1.,
                      'is_alive': 1,
                      'type': 0
                      }
             }
    for param in self.edge_params.keys():
        specs['edge'][param] = self.edge_params[param][("SC", "SC")]
    for param in self.face_params.keys():
        specs['face'][param] = self.face_params[param]["SC"]
    return specs


def get_division_function(self, model, crit_area):
    def division(sheet, manager):
        """Defines a division behavior."""
        #for major axis angles:
        #angles_list = major_axis_angles(sheet)
        # for elongataing angles
        angles_list = elongating_angles(sheet, model)
        index = 0
        # for random division orientation, angle=None
        for cell_id, row in sheet.face_df.iterrows():
            if row.area > crit_area and row.type == 0:
                # Do division, elongating_angle:
                daughter, edge_d_index, edge_m_index = cell_division(sheet, cell_id, geom,
                                                                     angle=angles_list[index])
                # Update the topology
                before_division_tension_d = sheet.edge_df.at[edge_d_index,'line_tension']
                before_division_tension_m = sheet.edge_df.at[edge_m_index,'line_tension']
                sheet.edge_df.at[edge_d_index,'line_tension'] = 100*before_division_tension_d # was 20
                sheet.edge_df.at[edge_m_index,'line_tension'] = 100*before_division_tension_m # was 20
                # new *10:
                sheet.edge_df.at[edge_d_index, 'tension_before_division'] = before_division_tension_d
                sheet.edge_df.at[edge_m_index, 'tension_before_division'] = before_division_tension_m
                #end new *10
                if np.abs(before_division_tension_m - before_division_tension_d) > 1e-6:
                    print("warning- tension in edges not equal")
                    sheet.edge_df.at[edge_d_index, 'line_tension'] = 100*before_division_tension_m
                sheet.get_opposite()
                involved_faces = np.intersect1d(self.get_neighbors(cell_id), self.get_neighbors(daughter))
                involved_faces = np.hstack([involved_faces, np.array([cell_id, daughter])])
                for face in involved_faces:
                    sheet.order_edges(face)
                sheet.reset_index(order=False)
                sheet.edge_df.sort_values(["face", "order"], inplace=True)
                sheet.get_opposite()
                # update geometry
                geom.update_all(sheet)
                if not sheet.check_all_edge_order():
                    print("bug in division")
            index +=1
        manager.append(division)
    return division

def get_reduced_tension_function(tau, dt):
    def get_reduced_tension(sheet,manager):
        for index, row in sheet.edge_df.iterrows():
            if row.tension_before_division != 0:
                tension = row.line_tension
                tension_before_division = row.tension_before_division
                after_division_tension = tension_before_division + (np.exp(-dt/tau))*(tension-tension_before_division)
                sheet.edge_df.at[index, "line_tension"] = after_division_tension
                epsilon = 0.0001
                if after_division_tension - tension_before_division < epsilon:
                    sheet.edge_df.at[index, "line_tension"] = tension_before_division
                    sheet.edge_df.at[index,"tension_before_division"] = 0
        manager.append(get_reduced_tension)

    return get_reduced_tension

# new

    # def get_forces_function(self):
    #     def get_forces(sheet,manager):
    #         model = self.get_model()
    #         grad_E = model.compute_gradient(sheet)
    #         grad_E.head()
    #         force = -grad_E
    #         manager.append(get_forces)
    #
    #     return get_forces

    # end new
# new
#     def get_forces_function(self):
#         def get_forces(sheet, manager):
#             model = self.get_model()
#             coords = sheet.coords
#             grad_each_vert = calculate_forces(sheet, geom, model, coords, scaling=2, ax=None, approx_grad=None)
#             manager.append(get_forces())
#         return grad_each_vert

# end new


    # def get_force_angles_function(self):
    #     def get_force_angles(sheet,manager):
    #         model = self.get_model()
    #         grad_E = model.compute_gradient(sheet)
    #         grad_E.head()
    #         force = -grad_E
    #         import cmath
    #         table_srce_face = sheet.datasets['edge']['face', 'srce']
    #         table_srce_face.groupby('face')
    #         face_column = table_srce_face['face']
    #         N_face = face_column.max()
    #         list_of_theta = []
    #         for n in range(0, N_face + 1):
    #             rows_each_face = table_srce_face[table_srce_face['face'] == n]
    #             sum_of_forces_per_face = 0
    #             for first_edge_index, first_row in rows_each_face.iterrows():
    #                 force_x = force.at[first_row.srce, 'gx']
    #                 force_y = force.at[first_row.srce, 'gy']
    #                 angle_of_force = np.arctan2(force_y, force_x)
    #                 abs_of_force = np.sqrt(force_x ** 2 + force_y ** 2)
    #                 sum_of_forces_per_face += abs_of_force * np.exp(2j * angle_of_force)
    #             angle_of_force_per_face = cmath.phase(sum_of_forces_per_face)
    #             list_of_theta.append(angle_of_force_per_face)
    #         manager.append(get_force_angles)
    #     return get_force_angles

def draw_sheet(sheet, number_vertices=False, number_edges=False, number_faces=True, is_ordered=True, print_angles=False):
    draw_specs = tyssue.config.draw.sheet_spec()
    cmap = plt.cm.get_cmap('Greens').reversed()
    cmap_scale = sheet.face_df.delta_level.to_numpy()
    color_cmap = cmap(0.7*(cmap_scale - 1) + 1)
    draw_specs['face']['color'] = color_cmap
    draw_specs['face']['alpha'] = 0.5
    draw_specs['face']['visible'] = True
    if is_ordered:
        sheet.is_ordered = True
        sheet.edge_df.sort_values(["face", "order"], inplace=True)
    if not sheet.check_all_edge_order():
        print("bug in drawing")
    fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
    fig.set_size_inches((8, 8))

    if number_faces:
        for face, data in sheet.face_df.iterrows():
            ax.text(data.x, data.y, face, fontsize=14, color="red")
    if print_angles:
        angles_list = major_axis_angles(sheet)
        index = 0
        for face, data in sheet.face_df.iterrows():
            ax.text(data.x, data.y-0.1, "%.3f" % angles_list[index], fontsize=14, color="magenta")
            index += 1
    if number_vertices:
        for vert, data in sheet.vert_df.iterrows():
            ax.text(data.x, data.y + 0.02, vert, weight="bold", color="blue")

    if number_edges:
        for edge, data in sheet.edge_df.iterrows():
            ax.text((data.tx + data.sx)/2 - (data.tx - data.sx)/4,
                    (data.ty + data.sy)/2 - (data.ty - data.sy)/4 + 0.02,
                    edge, weight="bold", color="green")
    return fig, ax


#print(sheet.upcast_face(sheet.face_df['area']))
# import h5py
# import numpy as np
# f1 = h5py.File('history_file.hdf5','r+')
# #history_face = pd.DataFrame(np.array(h5py.File('history_file.hdf5')['face']))
# #history_face_df = pd.read_hdf('history_file.hdf5')['face']
# #print(history_face)
# # print(history_face)
# # print(type(history_face))
# # print(type(history_face.at[0, 0]))
# # print(type(history_face.at[1, 0]))
# print(list(f1.keys()))
# group_face = f1['face']
# print(list(group_face.keys()))
# dset_face_i_table = group_face['_i_table']
# dset_face_table = group_face['table']
#
# print(dset_face_i_table.keys())
#
# print(dset_face_table.shape)
# print(dset_face_table.dtype)
# print(dset_face_table)
# print(dset_face_table.columns)
#print(list(history.face_h.columns))
#print(history.face_h)
#print(list(history.edge_h.columns))
#print(list(history.vert_h.columns))
# edge_time_df = history.edge_h
# edge_face_time = edge_time_df.get(['face','line_tension','tension_before_division','time'])
# edge_time_df.to_pickle("edge_time_file.pkl")
#edge_time = pd.read_pickle("edge_time_file.pkl")

# face_time_df = history.face_h
# face_time_df.to_pickle("face_time_file.pkl")
#
#
# divided = edge_face_time.loc[edge_face_time['tension_before_division'] != 0]
# divided = divided.drop_duplicates()
# list_of_faces = np.unique(divided.face)
#print(len(divided))
#print(len(edge_face_time))
#grouped_divided = divided.groupby('face')
# calculate the division timing per cell assuming no more than 1 division per cell.
# the table is groped by timing so the first timing per face is the division time.
# cells_divide = []
#
# division_timing = []
#
# for face in list_of_faces:
#     current_face_divide = divided.query('face == %d' % face)
#     indices_division = current_face_divide.diff().query("line_tension > 0 or line_tension != line_tension").index
#     real_divisions = current_face_divide.loc[indices_division]
#     for index,row in real_divisions.iterrows():
#         cells_divide.append((row['face']))
#         division_timing.append(row['time'])
#del cells_divide[0]
#del division_timing[0]
# timing_and_faces_divisions = sorted(zip(division_timing, cells_divide))
# print(timing_and_faces_divisions)

#new:
# new_timing_and_faces_divisions = timing_and_faces_divisions
# for e in range(0,len(timing_and_faces_divisions)):
#     for s in range(0,len(timing_and_faces_divisions)):
#         if e != s:
#             if timing_and_faces_divisions[e][1] == timing_and_faces_divisions[s][1] and timing_and_faces_divisions[e][0] == timing_and_faces_divisions[s][0]-1:
#                 break
#             else:
#                 new_timing_and_faces_divisions.append(timing_and_faces_divisions[e])

# print(new_timing_and_faces_divisions)
# face_list = []
#mother_list = []
#daughter_list = []
# timing_list = []
#make lists of the division timing and of the division face:
# for index in range(0,len(new_timing_and_faces_divisions)):
#     timing_list.append(new_timing_and_faces_divisions[index][0])
#     face_list.append(new_timing_and_faces_divisions[index][1])
# for ti in range(0,len(timing_list)):
#     if timing_list.count(timing_list[ti]) % 2 == 0 and face_list.count(face_list[ti]) <= 2:
#         new_timing_and_faces_divisions.append(timing_and_faces_divisions[ti])
# make a list of the mother cell ], a list of the relevant daughter cell and the relevant division timing:
# for i in range(0,len(new_timing_and_faces_divisions)-1):
#     division_index_list = []
#     count_of_faces_per_timing = timing_list.count(new_timing_and_faces_divisions[i][0])
# note the indices of the divisions if the count is bigger than 2:
# the index in the face list is the index of the division in the list of tuples - timing_and_faces_divisions
#     count_of_times_per_face = face_list.count(new_timing_and_faces_divisions[i][1])
#     if count_of_times_per_face >= 2:
#         division_index_list.append(face_list.index(new_timing_and_faces_divisions[i][1]))
# if there was one division per timing, the count of faces==2 (one for daughter and one for mother)
# and the mother cell will be the first and the daughter the second:
#     if i % 2 == 0 and count_of_faces_per_timing == 2:
#         mother_face = new_timing_and_faces_divisions[i][1]
#        face_list.append(mother_face)
#        mother_list.append(mother_face)
#         daughter_face = new_timing_and_faces_divisions[i+1][1]
#        face_list.append(daughter_face)
#        daughter_list.append(daughter_face)
# if there is only one division of this cell, it will appear in the face list once:
#         if face_list.count(new_timing_and_faces_divisions[i][1]) == 1:
#             time_of_division = new_timing_and_faces_divisions[i][0]
#             face_area_time = history.face_h.get(['face','area','time'])
#             division_mother_face_time = face_area_time.loc[face_area_time['face'] == mother_face]
#             division_daughter_face_time = face_area_time.loc[face_area_time['face'] == daughter_face]
#
#             division_mother_face_before_division = division_mother_face_time.loc[division_mother_face_time['time'] < time_of_division]
#             area_before_division = division_mother_face_before_division.area
#             time_before_division = division_mother_face_before_division.time
#
#
#             division_mother_face_since_division = division_mother_face_time.loc[division_mother_face_time['time'] >= time_of_division]
#             area_after_division_mother = division_mother_face_since_division.area
#             time_after_division = division_mother_face_since_division.time
#             area_after_division_daughter = division_daughter_face_time.area

#if there is more than one division of a cell, it will appear more than once in the face list:
        # else:
        #     time_of_division = new_timing_and_faces_divisions[i][0]
        #     face_area_time = history.face_h.get(['face','area','time'])
        #     division_mother_face_time = face_area_time.loc[face_area_time['face'] == mother_face]

# print(np.array(time_before_division))
# print(len(time_before_division))
# print(np.array(time_after_division))
# print(len(time_after_division))
# print(np.array(area_before_division))
# print(len(area_before_division))
# print(np.array(area_after_division_mother))
# print(len(area_after_division_mother))
# print(np.array(area_after_division_daughter))
# print(len(area_after_division_daughter))
# area_after_division = np.add(np.array(area_after_division_daughter), np.array(area_after_division_mother))
# time_axis = np.append(np.array(time_before_division), np.array(time_after_division))
# area_axis = np.append(np.array(area_before_division), area_after_division)
# print(time_axis)
# print(len(time_axis))
# print(area_axis)
# print(len(area_axis))
#plt.plot(time_axis,area_axis)
#plt.show()








# print(cells_divide)
# print(division_timing)
# initial_face = history.face_h[history.face_h['time'] == 0.0]
# last_face = history.face_h[history.face_h['time'] == 1.0]
# first_iter_face = history.face_h[history.face_h['time'] == 0.01]
# almost_last_face = history.face_h[history.face_h['time'] == 0.99]
# print(len(initial_face))
# first_division_face_time = history.face_h[history.face_h['time'] == division_timing[-1]]
# first_division_face_time_before = history.face_h[history.face_h['time'] == (division_timing[-1]-0.01)]
# #first_division_face_time_after = history.face_h[history.face_h['time'] == (division_timing[-1]+0.01)]
# print(len(first_division_face_time_before))
# print(len(first_division_face_time))
#print(len(first_division_face_time_after))
#cells_divide.append(divided.at[i,'face'])
#division_timing.append(divided.at[i,'time'])
#print(cells_divide)
#print(division_timing)
#for first_edge_index, first_row in rows_each_face.iterrows():
#    force_x = force.at[first_row.srce, 'gx']
#    force_y = force.at[first_row.srce, 'gy']




# for col in history.vert_h.columns:
#     print(col)
# print('end of ver columns')
# for col in history.edge_h.columns:
#     print(col)
# print('end of edge columns')
# for col in history.face_h.columns:
#     print(col)
# print('end of face columns')
#     #for v in history.edge_h['tension_before_division']:
#         #print(v)
# print(history.edge_h[['tension_before_division','time']])
# #print(history.edge_h.query('tension_before_division'!=0))
# for edge in history.edge_h['tension_before_division']:
#     if edge != 0:
#         print("yey")



# elongating axis tryouts:
#     import cmath
#     table_srce_face = sheet.datasets['edge'][['face', 'srce']]
#     table_srce_face.groupby('face')
#     face_column = table_srce_face['face']
#     N_face = face_column.max()
#     list_of_theta = []
#     print(table_srce_face)
#     for n in range(0, N_face + 1):
#         rows_each_face = table_srce_face[table_srce_face['face'] == n]
#         sum_of_forces_per_face = 0
#         for first_edge_index, first_row in rows_each_face.iterrows():
#             force_x = force.at[first_row.srce, 'gx']
#             force_y = force.at[first_row.srce, 'gy']
#             angle_of_force = np.arctan2(force_y, force_x)
#             abs_of_force = np.sqrt(force_x ** 2 + force_y ** 2)
#             sum_of_forces_per_face += abs_of_force * np.exp(2j * angle_of_force)
#         angle_of_force_per_face = cmath.phase(sum_of_forces_per_face)
#         list_of_theta.append(angle_of_force_per_face)





#print(sheet.datasets['edge'].head())
# tab_co = sheet.datasets['edge'][['face', 'sx', 'sy']]
# # tab_co.groupby('face')
# # rows_each_face = tab_co[tab_co['face'] == 0]
# print(rows_each_face)
# list_of_max = []
# list_of_x_1 = []
# list_of_x_2 = []
# list_of_y_1 = []
# list_of_y_2 = []
#
# for e in range(0,len(rows_each_face.index)):
#     major_x = rows_each_face.at[e,'sx']
#     major_y = rows_each_face.at[e,'sy']
#     dist_list = []
#     list_of_x_1.append(major_x)
#     list_of_y_1.append(major_y)
#     for i in range(0,len(rows_each_face.index)):
#         minor_x = rows_each_face.at[i,'sx']
#         minor_y = rows_each_face.at[i,'sy']
#         dist_i_sq = (major_y-minor_y)**2 + (major_x-minor_x)**2
#         dist_list.append(dist_i_sq)
#
#     max_dist = max(dist_list)
#     max_index = dist_list.index(max_dist)
#     max_coord_x = rows_each_face.iat[max_index,1]
#     max_coord_y = rows_each_face.iat[max_index,2]
#     list_of_max.append(max_dist)
#     list_of_x_2.append(max_coord_x)
#     list_of_y_2.append(max_coord_y)
#
# the_max_dist = max(list_of_max)
# the_max_index = list_of_max.index(the_max_dist)
# x_1_coord_max = list_of_x_1[the_max_index]
# x_2_coord_max = list_of_x_2[the_max_index]
# y_1_coord_max = list_of_y_1[the_max_index]
# y_2_coord_max = list_of_y_2[the_max_index]
# delta_x = abs(x_2_coord_max - x_1_coord_max)
# delta_y = abs(y_2_coord_max - y_1_coord_max)
# theta = np.arctan2(delta_y,delta_x)
# print(theta)



# (len(tab_co.index))
#for i in range(0, 1):
    #dist_sq_list = []
    #for e in range(0, (len(tab_co.index))):
        #if tab_co.loc[e, 'face'] == tab_co.loc[i, 'face']:
            #dist_sq = ((tab_co.loc[e, 'sx'] - tab_co.loc[i, 'sx']) ** 2) + (
                        #(tab_co.loc[e, 'sy'] - tab_co.loc[i, 'sy']) ** 2)
            #dist_sq_list.append(dist_sq)
            #print(sheet.datasets['edge'].head())
            #tab_co = sheet.datasets['edge'][['face', 'dx', 'sx', 'dy', 'sy']]
            # tab_dist = tab_co.diff(periods=1, axis=1)[['sx','sy']]
            #tab_co.groupby('face')
            #print(tab_co)
            # (len(tab_co.index))
            #for i in range(0, 2):
                #dist_sq_list = []
                #for e in range(0, (len(tab_co.index))):
                    #if tab_co.loc[e, 'face'] == tab_co.loc[i, 'face']:
                        #dist_sq = ((tab_co.loc[e, 'sx'] - tab_co.loc[i, 'sx']) ** 2) + (
                                    #(tab_co.loc[e, 'sy'] - tab_co.loc[i, 'sy']) ** 2)
                        #dist_sq_list.append(dist_sq)
                        #print(tab_co.loc[i, 'face'] )

                #print("the list is", dist_sq_list)

    #print("the list is", dist_sq_list)

