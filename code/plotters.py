import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import settings
import sys
import os

path = os.getcwd()
parent = os.path.join(path, os.pardir)
parent2 = os.path.join(parent, os.pardir)

class plotter():
    def __init__(self, **kwargs):
        super(plotter,self).__init__(**kwargs)   
        
    def predict_and_plot_mf(self, arch, model, dataset, args):
        
        ml_model, rans_model, training_case, input_fields, test_case, target_fields, mode = args
                
        x_coord = dataset[0] 
        y_coord = dataset[1]
        train_sensor_mask = dataset[2]
        test_sensor_mask = dataset[3]
        train_sensor_gridded = dataset[4]
        test_sensor_gridded = dataset[5]
        train_features = dataset[6]
        test_features = dataset[7]
        train_input_features = dataset[8]
        train_target_features = dataset[9]
        test_input_features = dataset[10]
        test_target_features = dataset[11]
        wall_train_features = dataset[12]
        wall_test_features = dataset[13]
        
        error = []
        total_prediction_list = []
        target_list = []
        input_list = []
        uncertainty_list = []
        range_list = []
        print(np.shape(test_target_features))
        print(np.shape(test_input_features))
        for case_indice, case in enumerate(test_case):
            print("Case: " + str(case))
            prediction_list = []
            for sensor_indice, no_sensors in enumerate(settings.sensor_list):
                if no_sensors < settings.sensor_list[-1]:
                    continue
                for j in tqdm(range(settings.ensembles)):
                    input_batch = []
                    target_batch = []
                    for k in range(settings.batch_size):
                        if k == 0:
                            index = case_indice
                            sensor_indice = sensor_indice
                        else:
                            index = np.random.randint(0, len(test_case))
                            sensor_indice = np.random.randint(0, len(settings.sensor_list))
                        #noise = np.random.normal(0, settings.data_noise_std, size=(settings.input_rows, settings.input_cols, settings.input_features-1)) #gaussian noise
                        target_min = np.nanpercentile(np.where(test_target_features[sensor_indice][index] == 0, np.nan, test_target_features[sensor_indice][index]), 5, axis=[0, 1])
                        target_max = np.nanpercentile(np.where(test_target_features[sensor_indice][index] == 0, np.nan, test_target_features[sensor_indice][index]), 95, axis=[0, 1])
                        input_min = np.nanpercentile(np.where(test_input_features[sensor_indice][index] == 0, np.nan, test_input_features[sensor_indice][index]), 5, axis=[0, 1])
                        input_max = np.nanpercentile(np.where(test_input_features[sensor_indice][index] == 0, np.nan, test_input_features[sensor_indice][index]), 95, axis=[0, 1])
                        temp_input_features = (test_input_features[sensor_indice][index] - input_min)/(input_max - input_min)
                        #temp_input_features += noise
                        temp_wall_features = np.where(np.isnan(wall_test_features[sensor_indice][index]), 0, wall_test_features[sensor_indice][index])
                        temp_wall_features = np.where(temp_wall_features < 0, 0, temp_wall_features)
                        temp_input_batch = np.concatenate((temp_input_features, temp_wall_features), axis=-1)
                        temp_target_batch = (test_target_features[sensor_indice][index] - target_min)/(target_max - target_min)
                        input_batch.append(temp_input_batch)
                        target_batch.append(temp_target_batch)
                    prediction = model.predict_on_batch(np.asarray(input_batch))
                    prediction = np.squeeze(prediction[0])
                    prediction_list.append(prediction)

                prediction_ensemble = np.asarray(prediction_list)
                pred_std = np.std(prediction_ensemble, axis=0)
                pred_mean = np.nanmean(prediction_ensemble, axis=0)

                prediction = pred_mean
                uncertainty = pred_std * 2
                # define uncertainty as the range of the prediction ensemble *experimental*
                prediction_rangeux = [np.nanmin(prediction_ensemble, axis=0)[:, :, 0], np.nanmax(prediction_ensemble, axis=0)[:, :, 0]]
                prediction_rangeuy = [np.nanmin(prediction_ensemble, axis=0)[:, :, 1], np.nanmax(prediction_ensemble, axis=0)[:, :, 1]]

                plot_gridded_input = input_batch[0]
                plot_gridded_target = target_batch[0]
                if plot_gridded_input.ndim < 3:
                    plot_gridded_input = np.expand_dims(plot_gridded_input, axis=-1)
                if plot_gridded_target.ndim < 3:
                    plot_gridded_target = np.expand_dims(plot_gridded_target, axis=-1)
                if uncertainty.ndim < 3:
                    uncertainty = np.expand_dims(uncertainty, axis=-1)
                if prediction.ndim < 3:
                    prediction = np.expand_dims(prediction, axis=-1)

                #test_sensor_gridded[sensor_indice][case] = np.where(test_sensor_gridded[sensor_indice][case] > 1, np.nan, test_sensor_gridded[sensor_indice][case])
                test_sensor_gridded[sensor_indice][case] = np.where(test_sensor_gridded[sensor_indice][case] > .1, 1, np.nan)
                test_sensor_gridded[sensor_indice][case] = np.where(np.isnan(np.squeeze(wall_test_features[sensor_indice][case_indice])), np.nan, test_sensor_gridded[sensor_indice][case])
                test_sensor_gridded[sensor_indice][case] = np.where(np.squeeze(wall_test_features[sensor_indice][case_indice]) <= .01, np.nan, test_sensor_gridded[sensor_indice][case])
                sensor_indices_y, sensor_indices_x = np.where(test_sensor_gridded[sensor_indice][case] == 1)
                # test_sensor_gridded[0][case] = np.where(test_sensor_gridded[0][case] > 1, np.nan, 1)

                #temp_wall_test_features = np.pad(wall_test_features[sensor_indice][case_indice], ((4, 4), (4, 4), (0, 0)), 'edge')
                temp_wall_test_features = wall_test_features[sensor_indice][case_indice]
                plot_gridded_input = np.where(np.isnan(temp_wall_test_features), np.nan, plot_gridded_input)
                plot_gridded_input = np.where(temp_wall_test_features <= 0, np.nan, plot_gridded_input)
                plot_gridded_target = np.where(np.isnan(temp_wall_test_features), np.nan, plot_gridded_target)
                plot_gridded_target = np.where(temp_wall_test_features <= 0, np.nan, plot_gridded_target)
                prediction = np.where(np.isnan(temp_wall_test_features), np.nan, prediction)
                prediction = np.where(temp_wall_test_features <= 0, np.nan, prediction)
                uncertainty = np.where(np.isnan(temp_wall_test_features), np.nan, uncertainty)
                uncertainty = np.where(temp_wall_test_features <= 0, np.nan, uncertainty)

                try:
                    control_ae = np.abs(np.squeeze(plot_gridded_input[:, :, :-1]) - np.squeeze(plot_gridded_target))
                    control_ae = np.where(np.squeeze(np.isnan(temp_wall_test_features)), np.nan, np.squeeze(control_ae))
                    control_mae = np.nanmean(control_ae)
                except:
                    #print("\ncontrol mae failed")
                    control_mae = 0.0

                ae = np.abs(prediction - plot_gridded_target)
                ae = np.where(np.isnan(temp_wall_test_features), np.nan, ae)
                mae = np.nanmean(ae)
                rmse = np.sqrt(np.nanmean(np.square(prediction - plot_gridded_target)))
                print("MAE: ", mae)
                print("RmSE: ", rmse)

                bias = prediction - plot_gridded_target
                bias = np.where(np.isnan(temp_wall_test_features), np.nan, bias)
                meanbias = np.nanmean(bias)
                print("Bias: ", meanbias)
                for k in range(settings.target_features):
                    try:
                        plot_field = target_fields[k]
                    except:
                        plot_field = target_fields[0]

                    if plot_field == 'tau':
                        if k == 0:
                            plot_field = 'tau uu'
                        if k == 1:
                            plot_field = 'tau vu'
                        if k == 2:
                            plot_field = 'tau vv'
                        if k == 3:
                            plot_field = 'tau ww'


                    ae_1d = np.where(np.isnan(ae[:, :, k]), 0, ae[:, :, k]).flatten()
                    uc_1d = np.where(np.isnan(uncertainty[:, :, k]), 0, uncertainty[:, :, k]).flatten()
                    R = np.corrcoef(ae_1d, uc_1d)
                    R = R[0][1]
                    print("R: ", R)
                    fig = plt.figure(figsize=(16, 9))
                    ax = fig.add_subplot(331)
                    ax1 = fig.add_subplot(332)
                    ax2 = fig.add_subplot(333)
                    ax3 = fig.add_subplot(334)
                    ax4 = fig.add_subplot(335)
                    ax5 = fig.add_subplot(336)
                    ax6 = fig.add_subplot(337)
                    ax7 = fig.add_subplot(338)
                    ax8 = fig.add_subplot(339)
                    fig.suptitle('Super-resolution of Flow Fields with unet \n'  
                                 'train case(s): ' + str(training_case) + ' | test case(s): ' + str(case) + '\n' +
                                 'train field(s): ' + str(input_fields) + ' | test field(s): ' + str(target_fields) + '\n' +
                                 'inner_noise: ' + str(settings.noise_std) + ' | data_noise: ' + str(settings.data_noise_std))

                    Uxnorm = mpl.cm.colors.Normalize(vmax=np.nanmax(plot_gridded_input[:, :, 0]), vmin=np.nanmin(plot_gridded_input[:, :, 0]))
                    Uynorm = mpl.cm.colors.Normalize(vmax=np.nanmax(plot_gridded_input[:, :, 1]), vmin=np.nanmin(plot_gridded_input[:, :, 1]))
                    if plot_field == 'Ux':
                        norm = Uxnorm
                        prediction_range = prediction_rangeux
                    if plot_field == 'Uy':
                        norm = Uynorm
                        prediction_range = prediction_rangeuy
                    wall_norm = mpl.cm.colors.Normalize(vmax=np.nanmax(plot_gridded_input[:, :, 2]), vmin=0)
                    abs_err_norm = mpl.cm.colors.Normalize(vmax=.1*np.nanmax(plot_gridded_input[:, :, k]), vmin=0)
                    bias_norm = mpl.cm.colors.Normalize(vmax=.1*np.nanmax(plot_gridded_input[:, :, k]), vmin=-.1*np.nanmax(plot_gridded_input[:, :, k]))
                    cmap = mpl.cm.jet
                    cmap_abs_err = mpl.cm.Reds

                    ax.imshow(plot_gridded_input[:, :, 0], norm=Uxnorm, cmap=cmap, origin='lower',)
                    ax.set_title('Tesselized Input Ux')
                    ax.set_ylabel('Height (m)')
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Uxnorm, cmap=cmap), ax=ax)
                    cbar.set_label('Ux (norm)')

                    ax1.imshow(plot_gridded_target[:, :, k], norm=norm, cmap=cmap, origin='lower')# extent=[0, np.amax(x_coord[case]['Cx'][0]), 0, np.amax(y_coord[case]['Cy'][0])])
                    ax1.set_title('Gridded Target ' + plot_field)
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
                    cbar.set_label(plot_field + ' (norm)')

                    ax2.imshow(prediction[:, :, k], norm=norm, cmap=cmap, origin='lower')#, extent=[0, np.amax(x_coord[case]['Cx'][0]), 0, np.amax(y_coord[case]['Cy'][0])])
                    ax2.set_title('Prediction ' + plot_field)
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2)
                    cbar.set_label(plot_field + ' (norm)')

                    ax3.imshow(plot_gridded_input[:, :, 1], norm=Uynorm, cmap=cmap, origin='lower',)
                    #          extent=[np.amin(x_coord[sensor_indice][case]['Cx']), np.amax(x_coord[sensor_indice][case]['Cx']),
                    #                  np.amin(y_coord[sensor_indice][case]['Cy']), np.amax(y_coord[sensor_indice][case]['Cy'])])
                    #ax.contourf(np.flip(test_sensor_gridded[0][case], axis=0), norm=norm, cmap='Greys')
                    #ax.scatter(sensor_indices_x, sensor_indices_y, c='black', s=10)
                    #ax.scatter(x_coord[sensor_indice][case]['Cx'][0][np.where(test_sensor_mask[sensor_indice][case] == 1)],
                    #           y_coord[sensor_indice][case]['Cy'][0][np.where(test_sensor_mask[sensor_indice][case] == 1)], c='black', s=5)
                    ax3.set_title('Tesselized Input Uy')
                    ax3.set_ylabel('Height (m)')
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=Uynorm, cmap=cmap), ax=ax3)
                    cbar.set_label('Uy (norm)')

                    ax4.imshow(ae[:, :, k], cmap=cmap_abs_err, norm=abs_err_norm, origin='lower')#, extent=[0, np.amax(x_coord[case]['Cx'][0]), 0, np.amax(y_coord[case]['Cy'][0])])
                    ax4.set_title('Absolute Error | MAE: ' + f"{mae:.5f}")
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_abs_err, norm=abs_err_norm), ax=ax4)
                    cbar.set_label('Abs Error (norm)')

                    ax5.imshow(uncertainty[:, :, k], norm=abs_err_norm, cmap=cmap_abs_err, origin='lower')#, extent=[0, np.amax(x_coord[case]['Cx'][0]), 0, np.amax(y_coord[case]['Cy'][0])])
                    ax5.set_title('Uncertainty ' + plot_field)# + ' | R: ' + f"{R:.5f}")
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=abs_err_norm, cmap=cmap_abs_err), ax=ax5)
                    cbar.set_label(plot_field + ' (norm)')

                    ax6.imshow(plot_gridded_input[:, :, 2], norm=wall_norm, cmap=cmap, origin='lower',)
                    ax6.set_title('Wall Distance')
                    ax6.set_xlabel('Length (m)')
                    ax6.set_ylabel('Height (m)')
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=wall_norm, cmap=cmap), ax=ax6)
                    cbar.set_label('Wall Distance (m)')

                    ax7.imshow(bias[:, :, k], cmap='bwr', norm=bias_norm, origin='lower')#, extent=[0, np.amax(x_coord[case]['Cx'][0]), 0, np.amax(y_coord[case]['Cy'][0])])
                    ax7.set_xlabel('Length (m)')
                    ax7.set_title('Bias | meanBias: ' + f"{meanbias:.5f}")
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap='bwr', norm=bias_norm), ax=ax7)
                    cbar.set_label('Error (norm)')

                    ax8.imshow(np.where(np.logical_and(plot_gridded_target[:, :, k] > prediction_range[0],
                                                       plot_gridded_target[:, :, k] < prediction_range[1]),
                                        1, 0),
                               vmin=0, vmax=1, cmap=cmap, origin='lower')
                    ax8.set_title('Within ' + str(np.nanmean(np.where(np.logical_and(plot_gridded_target[:, :, k] > prediction_range[0],
                    plot_gridded_target[:, :, k] < prediction_range[1]),1, 0))) + '%')
                    ax8.set_xlabel('Length (m)')

                    plt.subplots_adjust(top=.775, hspace=.38, wspace=.203, right=.97, left=0.03, bottom=0.07)
                    
                    if not os.path.isdir('./plots'):
                        os.makedirs('./plots')
                        
                    plt.savefig('./plots/'+ arch + '_' + case + '_' + str(plot_field) + '_' + str(no_sensors) + '.png')
                    plt.close('all')
                    if no_sensors == settings.sensor_list[-1]:
                        input_list.append(plot_gridded_input[:, :, k])
                        target_list.append(plot_gridded_target[:, :, k])
                        total_prediction_list.append(prediction[:, :, k])
                        range_list.append(prediction_range)
                        uncertainty_list.append(uncertainty[:, :, k])
                error.append(rmse)
        np.savez_compressed(parent2 + '/data/npy/meanflow_' + str(settings.ensembles) + 'input_mean_list.npz', input_list)
        np.savez_compressed(parent2 + '/data/npy/meanflow_' + str(settings.ensembles) + 'target_mean_list.npz', target_list)
        np.savez_compressed(parent2 + '/data/npy/meanflow_' + str(settings.ensembles) + 'prediction_mean_list.npz', total_prediction_list)
        np.savez_compressed(parent2 + '/data/npy/meanflow_' + str(settings.ensembles) + 'range_list.npz', range_list)
        np.savez_compressed(parent2 + '/data/npy/meanflow_' + str(settings.ensembles) + 'uncertainty_mean_list.npz', uncertainty_list)
        return error