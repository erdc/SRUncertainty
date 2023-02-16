from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator
import copy
import glob
import numpy as np
import settings
import os

path = os.getcwd()
parent = os.path.join(path, os.pardir)
parent = os.path.join(parent, os.pardir)


class TurbulenceDataset:
    def __init__(self):
        pass

    def load_gridded_dataset(self, sensors, model, training_case, input_fields, test_case, target_fields):
        print('Loading features and labels from model type: ' + model)
        train_x_coord, train_y_coord, test_x_coord, test_y_coord, train_input_features, train_target_features, test_input_features, test_target_features,\
        wall_train_features, wall_test_features = self._load_dimension_features(model, training_case, input_fields, test_case, target_fields)

        print('Tessellating training cases... ')
        train_centroids, train_centroids_indices, train_voronoi_features, train_gridded_inputs, train_gridded_targets, train_sensor_mask, train_sensor_gridded = self._tessellate_features_2d(
            train_input_features, train_target_features, train_x_coord, train_y_coord, sensors)

        print('Tessellating testing cases... ')
        test_centroids, test_centroids_indices, test_voronoi_features, test_gridded_inputs, test_gridded_targets, test_sensor_mask, test_sensor_gridded = self._tessellate_features_2d(
            test_input_features, test_target_features, test_x_coord, test_y_coord, sensors)

        x_coord = {**train_x_coord, **test_x_coord}
        y_coord = {**train_y_coord, **test_y_coord}

        print('Creating wall_distance meshgrids...')
        wall_train_features = \
            self._iterate_2d_cases(training_case, ['wallDistance'], x_coord, y_coord, wall_train_features)

        wall_test_features = \
            self._iterate_2d_cases(test_case, ['wallDistance'], x_coord, y_coord, wall_test_features)

        train_gridded_inputs_list = []
        train_gridded_targets_list = []
        test_gridded_inputs_list = []
        test_gridded_targets_list = []
        wall_train_features_list = []
        wall_test_features_list = []

        for case in training_case:
            train_gridded_inputs[case] = np.where(np.isnan(wall_train_features[case]), np.nan, train_gridded_inputs[case])
            train_gridded_inputs[case] = np.where(wall_train_features[case] <= 0, np.nan, train_gridded_inputs[case])
            train_gridded_inputs[case] = np.where(np.isnan(train_gridded_inputs[case]), 0, train_gridded_inputs[case])
            train_gridded_inputs_list.append(train_gridded_inputs[case])
            train_gridded_targets[case] = np.where(np.isnan(wall_train_features[case]), np.nan, train_gridded_targets[case])
            train_gridded_targets[case] = np.where(wall_train_features[case] <= 0, np.nan, train_gridded_targets[case])
            train_gridded_targets[case] = np.where(np.isnan(train_gridded_targets[case]), 0, train_gridded_targets[case])
            train_gridded_targets_list.append(train_gridded_targets[case])
            wall_train_features_list.append(wall_train_features[case])

        for case in test_case:
            test_gridded_inputs[case] = np.where(np.isnan(wall_test_features[case]), np.nan, test_gridded_inputs[case])
            test_gridded_inputs[case] = np.where(wall_test_features[case] <= 0, np.nan, test_gridded_inputs[case])
            test_gridded_inputs[case] = np.where(np.isnan(test_gridded_inputs[case]), 0, test_gridded_inputs[case])
            test_gridded_inputs_list.append(test_gridded_inputs[case])
            test_gridded_targets[case] = np.where(np.isnan(wall_test_features[case]), np.nan, test_gridded_targets[case])
            test_gridded_targets[case] = np.where(wall_test_features[case] <= 0, np.nan, test_gridded_targets[case])
            test_gridded_targets[case] = np.where(np.isnan(test_gridded_targets[case]), 0, test_gridded_targets[case])
            test_gridded_targets_list.append(test_gridded_targets[case])
            wall_test_features_list.append(wall_test_features[case])

        print('Saving dataset to file for later runs...')
        dataset = [x_coord, y_coord, train_sensor_mask, test_sensor_mask,
                   train_sensor_gridded, test_sensor_gridded,
                   train_voronoi_features, test_voronoi_features,
                   train_gridded_inputs_list, train_gridded_targets_list,
                   test_gridded_inputs_list, test_gridded_targets_list,
                   wall_train_features_list, wall_test_features_list]
        
        if not os.path.isdir(parent + '/data/npy'):
            os.makedirs(parent + '/data/npy')
            
        np.save(parent + '/data/npy/2D_' + str(training_case) + str(test_case) + str(input_fields) + str(target_fields) + str(sensors) + ".npy", dataset)
        return dataset

    def load_1d_dataset(self, sensors, model, training_case, input_fields, test_case, target_fields):
        train_x_coord, train_y_coord, test_x_coord, test_y_coord, train_features, test_features, \
        wall_train_features, wall_test_features = self._load_dimension_features(model, training_case, input_fields, test_case, target_fields)

        x_coord = {**train_x_coord, **test_x_coord}
        y_coord = {**train_y_coord, **test_y_coord}
        print(len(x_coord))

        if sensors > 0:
            print('Tessellating training cases... ')
            train_centroids, train_centroids_indices, train_voronoi_features, train_sensor_mask = self._tessellate_features(
                train_features, train_x_coord, train_y_coord, sensors)

            print('Tessellating testing cases... ')
            test_centroids, test_centroids_indices, test_voronoi_features, test_sensor_mask = self._tessellate_features(
                test_features, test_x_coord, test_y_coord, sensors)

            train_voronoi_features, test_voronoi_features, train_target_features, test_target_features, = \
                self._serialize_dataset(training_case, test_case, input_fields, target_fields, train_voronoi_features,
                                       train_features, test_voronoi_features, test_features)

            print('Saving dataset to file for later runs...')
            dataset = [x_coord, y_coord,
                       train_features, test_features,
                       train_voronoi_features, test_voronoi_features,
                       train_target_features, test_target_features,
                       wall_train_features, wall_test_features]
            #np.save('./data/npy/dataset_' + str(training_case) + str(test_case) + str(input_fields) + str(target_fields) + str(sensors) + ".npy", dataset)
            return dataset
        else:
            print("Sensor count: 0, no tesselation.")

            train_input_features, train_target_features, test_input_features, test_target_features = \
                self._serialize_dataset(training_case, test_case, input_fields, target_fields, train_features, test_features, train_features, test_features)
            print('Saving dataset to file for later runs...')
            dataset = [x_coord, y_coord,
                       train_features, test_features,
                       train_input_features, test_input_features,
                       train_target_features, test_target_features,
                       wall_train_features, wall_test_features]
            if not os.path.isdir(parent + '/data/npy'):
                os.makedirs(parent + '/data/npy')            
            np.save(parent + '/data/npy/dataset_1D_' + str(training_case) + str(test_case) + str(input_fields) + str(target_fields) + str(sensors) + ".npy", dataset)
            return dataset

    def _load_dimension_features(self, model, training_case, input_fields, test_case, target_fields):
        train_x_coord = self._load_from_file(model, training_case, 'Cx')
        train_y_coord = self._load_from_file(model, training_case, 'Cy')
        test_x_coord = self._load_from_file(model, test_case, 'Cx')
        test_y_coord = self._load_from_file(model, test_case, 'Cy')

        train_input_features = self._load_from_file(model, training_case, input_fields)
        train_target_features = self._load_from_file(model, training_case, target_fields)
        test_input_features = self._load_from_file(model, test_case, input_fields)
        test_target_features = self._load_from_file(model, test_case, target_fields)

        wall_train_features = self._load_from_file(model, training_case, 'wallDistance')
        wall_test_features = self._load_from_file(model, test_case, 'wallDistance')

        return train_x_coord, train_y_coord, test_x_coord, test_y_coord, train_input_features, train_target_features, test_input_features, test_target_features, wall_train_features, wall_test_features

    def _iterate_2d_cases(self, cases, fields, x_coord, y_coord, features):
        gridded_features = {}
        for case in cases:
            temp_x_coord, temp_y_coord = np.squeeze(x_coord[case]['Cx']), np.squeeze(y_coord[case]['Cy'])

            x = np.linspace(np.amin(temp_x_coord), np.amax(temp_x_coord), settings.input_cols)
            y = np.linspace(np.amin(temp_y_coord), np.amax(temp_y_coord), settings.input_rows)
            X, Y = np.meshgrid(x, y, indexing='xy')
            if case in gridded_features:
                pass
            else:
                gridded_features[case] = []
            field_no=0
            for field in fields:
                temp_features = features[case][field]
                if isinstance(temp_features, list):
                    temp_features = temp_features[0]
                if temp_features.ndim < 2:
                    temp_features = np.expand_dims(temp_features, axis=-1)
                temp_gridded_features_cubic = griddata(np.asarray([temp_x_coord, temp_y_coord]).T, temp_features, (X, Y), method='cubic')
                temp_gridded_features_linear = griddata(np.asarray([temp_x_coord, temp_y_coord]).T, temp_features, (X, Y),
                                                 method='linear')
                temp_gridded_features_nearest = griddata(np.asarray([temp_x_coord, temp_y_coord]).T, temp_features, (X, Y),
                                                 method='nearest')
                """print(case)
                print(field)
                fig = plt.figure()
                norm = mpl.cm.colors.Normalize(vmax=.002, vmin=-.002)
                ax = fig.add_subplot(221)
                ax.imshow(temp_gridded_features_cubic, cmap='jet', norm=norm)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax)
                ax1 = fig.add_subplot(222)
                ax1.imshow(temp_gridded_features_linear, cmap='jet', norm=norm)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax1)
                ax2 = fig.add_subplot(223)
                ax2.imshow(temp_gridded_features_nearest, cmap='jet', norm=norm)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax2)
                ax3 = fig.add_subplot(224)
                ax3.tricontour(np.squeeze(temp_x_coord), np.squeeze(temp_y_coord), features[case][field], levels=14, linewidths=0.5,
                                cmap='jet', norm=norm)
                ax3.tricontourf(np.squeeze(temp_x_coord), np.squeeze(temp_y_coord), features[case][field], levels=14, cmap='jet', norm=norm)
                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax3)
                plt.show()"""

                temp_gridded_features = temp_gridded_features_cubic

                if field_no == 0:
                    gridded_features[case] = temp_gridded_features
                else:
                    gridded_features[case] = np.concatenate((gridded_features[case], temp_gridded_features), axis=-1)
                field_no += 1

        return gridded_features

    def _iterate_1d_cases(self, cases, input_fields, target_fields, input_features, target_features):
        x_list = []
        y_list = []

        for case in cases:
            print(case)
            field_no = 0
            x = 0
            for field in input_fields:
                x, field_no = self._iterate_field_lists(x, input_features, case, field, field_no)
            x_list.append(x)

            field_no = 0
            y = 0
            for field in target_fields:
                y, field_no = self._iterate_field_lists(y, target_features, case, field, field_no)
            y_list.append(y)


        return x_list, y_list

    def _serialize_dataset(self, training_case, test_case, input_fields, target_fields, train_input_features, train_target_features, test_input_features, test_target_features):

        print('Creating input/target features for ann model...')
        x_train_list, y_train_list, = \
            self._iterate_1d_cases(training_case, input_fields, target_fields, train_input_features, train_target_features)
        x_test_list, y_test_list = \
            self._iterate_1d_cases(test_case, input_fields, target_fields, test_input_features, test_target_features)

        for i in range(len(x_test_list)):
            temp_x = x_test_list[i]
            temp_y = y_test_list[i]
            if i == 0:
                temp_x_test_list = np.copy(temp_x)
                temp_y_test_list = np.copy(temp_y)
            else:
                temp_x_test_list = np.concatenate((temp_x_test_list, temp_x), axis=0)
                temp_y_test_list = np.concatenate((temp_y_test_list, temp_y), axis=0)

        for i in range(len(x_train_list)):
            temp_x = x_train_list[i]
            temp_y = y_train_list[i]
            if i == 0:
                temp_x_train_list = np.copy(temp_x)
                temp_y_train_list = np.copy(temp_y)
            else:
                temp_x_train_list = np.concatenate((temp_x_train_list, temp_x), axis=0)
                temp_y_train_list = np.concatenate((temp_y_train_list, temp_y), axis=0)

        x_train_list = temp_x_train_list
        x_test_list = temp_x_test_list
        y_train_list = temp_y_train_list
        y_test_list = temp_y_test_list

        return x_train_list, x_test_list, y_train_list, y_test_list

    @staticmethod
    def _load_from_file(model, cases, fields):
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(cases, str):
            cases = [cases]
        data = {}
        for field in fields:
            for case in cases:
                if case in data:
                    pass
                else:
                    data[case] = {}
                if field in data[case]:
                    pass
                else:
                    data[case][field] = []
                try:
                    field_data = np.concatenate([np.load(glob.glob(parent + '/data/' + model + '/' + model + '_' + case + '_' + field + '*.npy')[0], allow_pickle=True)])
                except:
                    field_data = np.concatenate(
                        [np.load(glob.glob(parent + '/data/labels/' + case + '_' + field + '*.npy')[0], allow_pickle=True)])

                field_data = np.asarray(field_data)

                if field_data.ndim == 2:
                    field_data = field_data.reshape(field_data.shape[0], (field_data.shape[1]))
                if field_data.ndim == 3:
                    field_data = field_data.reshape(field_data.shape[0],
                                                    (field_data.shape[1] * field_data.shape[2]))
                if field_data.ndim == 4:
                    field_data = field_data.reshape(field_data.shape[0],
                                                    (field_data.shape[1] * field_data.shape[2] *
                                                     field_data.shape[3]))
                data[case][field].append(field_data)
        return data

    @staticmethod
    def _tessellate_features(features, x_coord, y_coord, num_sen):
        centroids_indices = {}
        centroids = {}
        sensor_mask = {}
        voronoi_features = copy.deepcopy(features)

        for case in features.keys():
            print('Tessellating features for case: ' + case)
            tempx = x_coord[case][list(x_coord[case].keys())[0]]
            tempy = y_coord[case][list(y_coord[case].keys())[0]]
            tempx = np.asarray(tempx)
            tempy = np.asarray(tempy)
            if case in centroids_indices:
                pass
            else:
                centroids_indices[case] = []
            if case in sensor_mask:
                pass
            else:
                sensor_mask[case] = []

            # for clustered sensor locations
            # points = np.concatenate((tempx, tempy), axis=-1)
            # centroids, clusters = kmeans2(points, k=num_sen)

            if case in centroids:
                pass
            else:
                centroids[case] = []

            # for uniform sensor locations
            x_sensor_values = np.linspace(.05*np.amax(tempx), np.amax(tempx)-.05*np.amax(tempx), int(np.sqrt(num_sen)))
            y_sensor_values = np.linspace(.05*np.amax(tempy), np.amax(tempy)-.05*np.amax(tempy), int(np.sqrt(num_sen)))

            #for ylog sensor locations
            x_sensor_values = np.linspace(np.amin(tempx) + .01*np.amax(tempx), np.amax(tempx)-.01*np.amax(tempx), int(np.sqrt(num_sen)))
            y_sensor_values = np.geomspace(np.amin(tempy) + .01*np.amax(tempy), np.amax(tempy)-.01*np.amax(tempy), int(np.sqrt(num_sen)))

            for i in range(int(np.sqrt(num_sen))):
                for j in range(int(np.sqrt(num_sen))):
                    centroids[case].append([x_sensor_values[i], y_sensor_values[j]])

            for centroid in centroids[case]:
                lowest_distance = 10 ^ 100
                lowest_distance_point = 0
                for j in range(len(tempx[0])):
                    distance = np.sqrt((tempx[0, j] - centroid[0]) ** 2 + (tempy[0, j] - centroid[1]) ** 2)
                    if distance < lowest_distance:
                        lowest_distance = distance
                        lowest_distance_point = j
                centroids_indices[case].append(lowest_distance_point)
            for j in range(len(tempx[0])):
                lowest_distance = 10 ^ 100
                lowest_distance_centroid = 0
                for i in range(len(centroids[case])):
                    distance = np.sqrt(
                        (tempx[0, j] - centroids[case][i][0]) ** 2 + (tempy[0, j] - centroids[case][i][1]) ** 2)
                    if distance < lowest_distance:
                        lowest_distance = distance
                        lowest_distance_centroid = i
                for field in voronoi_features[case]:
                    voronoi_features[case][field] = np.squeeze(voronoi_features[case][field])
                    features[case][field] = np.squeeze(features[case][field])
                    voronoi_features[case][field][j] = features[case][field][centroids_indices[case][lowest_distance_centroid]]

            sensor_mask[case] = np.zeros(len(tempx[0]))
            sensor_mask[case][centroids_indices[case]] = 1

        return centroids, centroids_indices, voronoi_features, sensor_mask

    @staticmethod
    def _tessellate_features_2d(input_features, target_features, x_coord, y_coord, num_sen):
        voronoi_features = copy.deepcopy(input_features)
        centroids = {}
        sensor_mask = {}
        sensor_gridded = {}
        features_2d = {}
        targets_2d = {}
        #create input features grid
        for case in input_features.keys():
            print('Tessellating features for case: ' + case)
            tempx = x_coord[case][list(x_coord[case].keys())[0]]
            tempy = y_coord[case][list(y_coord[case].keys())[0]]
            tempx = np.asarray(tempx)
            tempy = np.asarray(tempy)
            x = np.linspace(np.amin(tempx), np.amax(tempx), settings.input_cols)
            y = np.linspace(np.amin(tempy), np.amax(tempy), settings.input_rows)
            X, Y = np.meshgrid(x, y, indexing='xy')
            if case in sensor_mask:
                pass
            else:
                sensor_mask[case] = []
            if case in sensor_gridded:
                pass
            else:
                sensor_gridded[case] = []
            if case in centroids:
                pass
            else:
                centroids[case] = []
            if case in features_2d:
                pass
            else:
                features_2d[case] = []
            if case in targets_2d:
                pass
            else:
                targets_2d[case] = []
            # for clustered sensor locations
            # points = np.concatenate((tempx, tempy), axis=-1)
            # centroids, clusters = kmeans2(points, k=num_sen)

            # for uniform sensor locations
            x_sensor_values = np.linspace(np.amin(tempx)+.05*np.amax(tempx), np.amax(tempx)-.05*np.amax(tempx), int(np.sqrt(num_sen)))
            y_sensor_values = np.linspace(np.amin(tempy)+.05*np.amax(tempy), np.amax(tempy)-.05*np.amax(tempy), int(np.sqrt(num_sen)))
            #for ylog sensor locations
            x_sensor_values = np.linspace(np.amin(tempx) + .05*np.amax(tempx), np.amax(tempx)-.05*np.amax(tempx), int(np.sqrt(num_sen)))
            y_sensor_values = np.geomspace(np.amin(tempy) + .05*np.amax(tempy), np.amax(tempy)-.05*np.amax(tempy), int(np.sqrt(num_sen)))

            for i in range(int(np.sqrt(num_sen))):
                for j in range(int(np.sqrt(num_sen))):
                    centroids[case].append([x_sensor_values[i], y_sensor_values[j]])
            centroids_indices = copy.deepcopy(centroids)
            def find_centroids(centroids, centroids_indices, tempx, tempy):
                n = 0
                bad_centroid_indices = []
                for centroid in centroids:
                    # find centroids that are too close to other centroids
                    """lowest_distance = 10 ^ 100
                    for j in range(len(centroids)):
                        distance = np.sqrt(((centroids[j][0] - centroid[0])/(np.amax(tempx)-np.amin(tempx))) ** 2 + ((centroids[j][1] - centroid[1])/(np.amax(tempy)-np.amin(tempy))) ** 2)
                        if distance !=0:
                            if distance < lowest_distance:
                                lowest_distance = distance
                                lowest_distance_point = j
                    if lowest_distance < np.sqrt((.01)**2 + (.01)**2):
                        centroid[1] = centroid[1] * 2
                        bad_centroid_indices.append(n)
                    else:
                        centroids_indices[n] = lowest_distance_point"""
                    #check that each centroid is not too close to the edge of the grid
                    lowest_distance = 10 ^ 100
                    lowest_distance_point = 0
                    for idj, j in enumerate(tempx[0, ::10]):
                        idj = idj * 10
                        distance = np.sqrt(((tempx[0, idj] - centroid[0])/(np.amax(tempx)-np.amin(tempx))) ** 2 + ((tempy[0, idj] - centroid[1])/(np.amax(tempy)-np.amin(tempy))) ** 2)
                        if distance < lowest_distance:
                            lowest_distance = distance
                            lowest_distance_point = idj
                    #if lowest_distance > np.sqrt((.01)**2 + (.01)**2):
                        #bad_centroid_indices.append(n)
                        #centroids[n][1] = centroids[n][1] * 2
                    #else:
                    centroids_indices[n] = lowest_distance_point
                    n+= 1
                return centroids_indices, bad_centroid_indices
            for v in range(1):
                centroids_indices[case], centroids_offgrid = find_centroids(centroids[case], centroids_indices[case], tempx, tempy)
                if len(centroids_offgrid) == 0:
                    break

            centroids_offgrid = [ele for ele in reversed(centroids_offgrid)]
            for i in centroids_offgrid:
                centroids[case].pop(i)
                centroids_indices[case].pop(i)

            m = 0
            for field in voronoi_features[case]:
                value_indices_temp = np.array(centroids_indices[case]).astype(int).T
                voronoi_features[case][field] = np.squeeze(voronoi_features[case][field])
                interp = NearestNDInterpolator(centroids[case], voronoi_features[case][field][value_indices_temp])
                Z = interp(X, Y)
                if np.ndim(Z) < 3:
                    Z = np.expand_dims(Z, axis=-1)
                if m == 0:
                    features_2d[case] = np.copy(Z)
                else:
                    features_2d[case] = np.concatenate((features_2d[case], Z), axis=-1)
                m+=1
                if field == 'tau':
                    np.delete(features_2d[case], 7, axis=-1)
                    np.delete(features_2d[case], 6, axis=-1)
                    np.delete(features_2d[case], 5, axis=-1)
                    np.delete(features_2d[case], 2, axis=-1)
                    np.delete(features_2d[case], 1, axis=-1)

            #create sensor mask
            sensor_mask[case] = np.zeros(len(tempx[0]))
            sensor_mask[case][centroids_indices[case]] = 1
            temp_x_coord, temp_y_coord = np.squeeze(x_coord[case]['Cx']), np.squeeze(y_coord[case]['Cy'])
            sensor_gridded[case] = griddata(np.asarray([temp_x_coord, temp_y_coord]).T, sensor_mask[case], (X, Y), method='linear')

        #create target features grid
        for case in target_features.keys():
            print('Tessellating features for case: ' + case)
            tempx = x_coord[case][list(x_coord[case].keys())[0]]
            tempy = y_coord[case][list(y_coord[case].keys())[0]]
            tempx = np.asarray(tempx)
            tempy = np.asarray(tempy)
            x = np.linspace(np.amin(tempx), np.amax(tempx), settings.input_cols)
            y = np.linspace(np.amin(tempy), np.amax(tempy), settings.input_rows)
            X, Y = np.meshgrid(x, y, indexing='xy')
            m = 0
            for field in target_features[case]:
                temp_x_coord, temp_y_coord = np.squeeze(x_coord[case]['Cx']), np.squeeze(y_coord[case]['Cy'])
                target_features[case][field] = np.squeeze(target_features[case][field])
                gridded_features_cubic = griddata(np.asarray([temp_x_coord, temp_y_coord]).T, target_features[case][field], (X, Y), method='nearest')
                if np.ndim(gridded_features_cubic) < 3:
                    gridded_features_cubic = np.expand_dims(gridded_features_cubic, axis=-1)
                if m == 0:
                    targets_2d[case] = np.copy(gridded_features_cubic)
                else:
                    targets_2d[case] = np.concatenate((targets_2d[case], gridded_features_cubic), axis=-1)
                #plt.imshow(gridded_features_cubic[:, :, 0])
                #plt.show()
                m+=1
                if field == 'tau':
                    print(np.shape(targets_2d[case]))
                    targets_2d[case] = np.delete(targets_2d[case], 7, axis=-1)
                    targets_2d[case] = np.delete(targets_2d[case], 6, axis=-1)
                    targets_2d[case] = np.delete(targets_2d[case], 5, axis=-1)
                    targets_2d[case] = np.delete(targets_2d[case], 2, axis=-1)
                    targets_2d[case] = np.delete(targets_2d[case], 1, axis=-1)
                print(np.shape(targets_2d[case]))

        return centroids, centroids_indices, voronoi_features, features_2d, targets_2d, sensor_mask, sensor_gridded

    @staticmethod
    def _iterate_field_lists(x, features, case, field, field_no):
        if field_no == 0:
            temp_x = features[case][field]
            temp_x = np.asarray(temp_x)
            if temp_x.ndim > 1:
                x = np.moveaxis(np.squeeze(temp_x), 0, -1)
            else:
                x = temp_x
                x = np.expand_dims(x, axis=-1)
        else:
            temp_x = features[case][field]
            temp_x = np.asarray(temp_x)
            if temp_x.ndim > 1:
                x = np.concatenate((x, temp_x), axis=-1)
            else:
                temp_x = np.expand_dims(temp_x, axis=-1)
                x = np.concatenate((x, temp_x), axis=-1)
        return x, field_no