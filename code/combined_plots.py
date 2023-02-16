import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_scatter_density
import netCDF4 as nc
import sm_params
import sst_params
import cv2
import glob
import re
import settings
from natsort import natsorted, ns

def load_sm_mask():
    layer_fn = './data/sm/layer1/layer1/SoMo.ml_v1_layer1_2000.nc'
    layer_data = nc.Dataset(layer_fn)
    sm = layer_data.variables['layer1'][:]
    mask = np.where(np.isnan(sm[0]), 0.0, 1.0)
    mask = np.squeeze(mask)
    resized_mask = cv2.resize(mask, (sm_params.input_cols, sm_params.input_rows))
    resized_mask = np.expand_dims(resized_mask, axis=-1)
    return mask, resized_mask


def load_sst_mask():
    # land-sea mask
    mask = nc.Dataset('./data/noaa/lsmask.nc')
    mask = mask.variables['mask'][:][0]
    mask = mask.astype(float)
    mask[mask == 0] = np.nan
    resized_mask = cv2.resize(mask, (360, 180), interpolation=cv2.INTER_CUBIC)
    return mask, resized_mask


sm_mask, sm_resized_mask = load_sm_mask()
sm_prediction_mean = np.load('./data/npy/SoMoprediction_mean.npz')['arr_0']
sm_prediction_std = np.load('./data/npy/SoMoprediction_std.npz')['arr_0']
sm_prediction_range = np.load('./data/npy/SoMoprediction_range.npz')['arr_0']
sm_uncertainty = np.load('./data/npy/SoMouncertainty.npz')['arr_0']
sm_targets = np.load('./data/npy/SoMotargets.npz')['arr_0']
sm_inputs = np.load('./data/npy/SoMoinputs.npz')['arr_0']
sm_plot_mask = cv2.resize(sm_resized_mask, (360, 180))
sm_plot_mask = np.expand_dims(sm_plot_mask, axis=-1)
sm_plot_mask = np.concatenate((sm_plot_mask, sm_plot_mask, sm_plot_mask), axis=2)
sm_plot_mask = np.where(sm_plot_mask == 0, np.nan, sm_plot_mask)
inputs_downsized_list = []
uc_downsized_list = []
targets_downsized_list = []
prediction_mean_downsized_list = []
prediction_range_downsized_list = []
for i in range(len(sm_prediction_mean)):
    prediction_mean_downsized = cv2.resize(sm_prediction_mean[i], (360, 180))
    prediction_std_downsized = cv2.resize(sm_prediction_std[i], (360, 180))
    prediction_range_downsized = [cv2.resize(sm_prediction_range[0][i], (360, 180)), cv2.resize(sm_prediction_range[1][i], (360, 180))]
    uncertainty_downsized = cv2.resize(sm_uncertainty[i], (360, 180))
    targets_downsized = cv2.resize(sm_targets[i], (360, 180))
    inputs_downsized = cv2.resize(sm_inputs[i], (360, 180))
    inputs_downsized = np.where(np.isnan(sm_plot_mask), np.nan, inputs_downsized[:, :, :3])
    prediction_mean_downsized = np.where(np.isnan(sm_plot_mask), np.nan, prediction_mean_downsized)
    prediction_std_downsized = np.where(np.isnan(sm_plot_mask), np.nan, prediction_std_downsized)
    uncertainty_downsized = np.where(np.isnan(sm_plot_mask), np.nan, uncertainty_downsized)
    targets_downsized = np.where(np.isnan(sm_plot_mask), np.nan, targets_downsized)
    inputs_downsized = np.where(np.isnan(sm_plot_mask), np.nan, inputs_downsized[:, :, :sm_params.target_features])
    uc_downsized_list.append(uncertainty_downsized)
    targets_downsized_list.append(targets_downsized)
    prediction_mean_downsized_list.append(prediction_mean_downsized)
    inputs_downsized_list.append(inputs_downsized)
    prediction_range_downsized_list.append(prediction_range_downsized)

sm_inputs = np.array(inputs_downsized_list)
sm_targets = np.array(targets_downsized_list)
sm_prediction_mean = np.array(prediction_mean_downsized_list)
sm_prediction_range = np.array(prediction_range_downsized_list)
sm_uncertainty = np.array(uc_downsized_list)

sst_mask, sst_resized_mask = load_sst_mask()
sst_prediction_mean = np.load('./data/npy/sst.wkmean_500_prediction_mean.npz')['arr_0']
sst_prediction_std = np.load('./data/npy/sst.wkmean_500_prediction_std.npz')['arr_0']
sst_prediction_range = np.load('./data/npy/sst.wkmean_500_prediction_range.npz')['arr_0']
sst_uncertainty = np.load('./data/npy/sst.wkmean_500_uncertainty.npz')['arr_0']
sst_targets = np.load('./data/npy/sst.wkmean_500_targets.npz')['arr_0']
sst_inputs = np.load('./data/npy/sst.wkmean_500_inputs.npz')['arr_0']

inputs_downsized_list = []
uc_downsized_list = []
targets_downsized_list = []
prediction_mean_downsized_list = []
prediction_range_downsized_list = []
for i in range(len(sst_prediction_mean)):
    prediction_mean_downsized = cv2.resize(sst_prediction_mean[i], (360, 180))
    prediction_std_downsized = cv2.resize(sst_prediction_std[i], (360, 180))
    prediction_range_downsized = [cv2.resize(sst_prediction_range[0][i], (360, 180)), cv2.resize(sst_prediction_range[1][i], (360, 180))]
    uncertainty_downsized = cv2.resize(sst_uncertainty[i], (360, 180))
    targets_downsized = cv2.resize(sst_targets[i], (360, 180))
    inputs_downsized = cv2.resize(sst_inputs[i], (360, 180))
    inputs_downsized = np.where(np.isnan(sst_mask), np.nan, inputs_downsized)
    prediction_mean_downsized = np.where(np.isnan(sst_mask), np.nan, prediction_mean_downsized)
    prediction_std_downsized = np.where(np.isnan(sst_mask), np.nan, prediction_std_downsized)
    uncertainty_downsized = np.where(np.isnan(sst_mask), np.nan, uncertainty_downsized)
    targets_downsized = np.where(np.isnan(sst_mask), np.nan, targets_downsized)
    inputs_downsized_list.append(inputs_downsized)
    uc_downsized_list.append(uncertainty_downsized)
    targets_downsized_list.append(targets_downsized)
    prediction_mean_downsized_list.append(prediction_mean_downsized)
    prediction_range_downsized_list.append(prediction_range_downsized)

sst_inputs = np.array(inputs_downsized_list)
sst_targets = np.array(targets_downsized_list)
sst_prediction_mean = np.array(prediction_mean_downsized_list)
sst_prediction_range = np.array(prediction_range_downsized_list)
sst_uncertainty = np.array(uc_downsized_list)

print('SST inputs shape: ', sst_inputs.shape)
print('SST targets shape: ', sst_targets.shape)
print('SST prediction mean shape: ', sst_prediction_mean.shape)
print('SST prediction range shape: ', sst_prediction_range.shape)
print('SST uncertainty shape: ', sst_uncertainty.shape)
meanflow_inputs = np.load('./data/npy/meanflow_' + str(settings.ensembles) + 'input_mean_list.npz')['arr_0']
meanflow_targets = np.load('./data/npy/meanflow_' + str(settings.ensembles) + 'target_mean_list.npz')['arr_0']
meanflow_prediction_mean = np.load('./data/npy/meanflow_' + str(settings.ensembles) + 'prediction_mean_list.npz')['arr_0']
meanflow_prediction_range = np.load('./data/npy/meanflow_' + str(settings.ensembles) + 'range_list.npz')['arr_0']
meanflow_uncertainty = np.load('./data/npy/meanflow_' + str(settings.ensembles) + 'uncertainty_mean_list.npz')['arr_0']

turbulence_inputs = np.load('./data/npy/inputs_isotropic.npy')[:, :, :, 0]
turbulence_targets = np.load('./data/npy/targets_isotropic.npy')[:, :, :, 0]
turbulence_prediction = np.load('./data/npy/predictions_isotropic.npy')[:, :, :, 0]
turbulence_prediction_range = np.load('./data/npy/range_isotropic.npy')[:, :, :, 0]
turbulence_uncertainty = np.load('./data/npy/uncertainty_isotropic.npy')[:, :, :, 0]

# calculate RMSE, MAE, and bias of prediction list
rmse = np.sqrt(np.nanmean((np.asarray(sm_prediction_mean) - np.asarray(sm_targets))**2))
mae = np.nanmean(np.abs(np.asarray(sm_prediction_mean) - np.asarray(sm_targets)))
bias = np.nanmean(np.asarray(sm_prediction_mean) - np.asarray(sm_targets))
print("SM RMSE: " + str(rmse))
print("SM MAE: " + str(mae))
print("SM Bias: " + str(bias))

rmse = np.sqrt(np.nanmean((sst_prediction_mean - sst_targets)**2))
mae = np.nanmean(np.abs(sst_prediction_mean - sst_targets))
bias = np.nanmean(sst_prediction_mean - sst_targets)
print("SST RMSE: " + str(rmse))
print("SST MAE: " + str(mae))
print("SST Bias: " + str(bias))

rmse = np.sqrt(np.nanmean((meanflow_prediction_mean - meanflow_targets)**2))
mae = np.nanmean(np.abs(meanflow_prediction_mean - meanflow_targets))
bias = np.nanmean(meanflow_prediction_mean - meanflow_targets)
print("Meanflow RMSE: " + str(rmse))
print("Meanflow MAE: " + str(mae))
print("Meanflow Bias: " + str(bias))

rmse = np.sqrt(np.nanmean((turbulence_prediction - turbulence_targets)**2))
mae = np.nanmean(np.abs(turbulence_prediction - turbulence_targets))
bias = np.nanmean(turbulence_prediction - turbulence_targets)
print("Turbulence RMSE: " + str(rmse))
print("Turbulence MAE: " + str(mae))
print("Turbulence Bias: " + str(bias))

# create meanflow summary plot
"""fig, axs = plt.subplots(6, 3, figsize=(20, 10), constrained_layout=True)
ax0 = axs[0, 0]
ax0.pcolormesh(meanflow_inputs[0], cmap='jet')
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel('Input', fontsize=14)

ax1 = axs[0, 1]
ax1.pcolormesh(meanflow_inputs[2], cmap='jet')
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = axs[0, 2]
ax2.pcolormesh(meanflow_inputs[4], cmap='jet')
ax2.set_xticks([])
ax2.set_yticks([])

ax3 = axs[1, 0]
ax3.pcolormesh(meanflow_targets[0], cmap='jet')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel('Target', fontsize=14)

ax4 = axs[1, 1]
ax4.pcolormesh(meanflow_targets[2], cmap='jet')
ax4.set_xticks([])
ax4.set_yticks([])

ax5 = axs[1, 2]
ax5.pcolormesh(meanflow_targets[4], cmap='jet')
ax5.set_xticks([])
ax5.set_yticks([])

ax6 = axs[2, 0]
ax6.pcolormesh(meanflow_prediction_mean[0], cmap='jet')
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_ylabel('Prediction', fontsize=14)
ax7 = axs[2, 1]
ax7.pcolormesh(meanflow_prediction_mean[2], cmap='jet')
ax7.set_xticks([])
ax7.set_yticks([])
ax8 = axs[2, 2]
ax8.pcolormesh(meanflow_prediction_mean[4], cmap='jet')
ax8.set_xticks([])
ax8.set_yticks([])
ax9 = axs[3, 0]
ax9.pcolormesh(np.abs(meanflow_prediction_mean[0] - meanflow_targets[0])/np.abs(meanflow_prediction_mean[0] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax9.set_xticks([])
ax9.set_yticks([])
ax9.set_ylabel('Error', fontsize=14)
ax10 = axs[3, 1]
ax10.pcolormesh(np.abs(meanflow_prediction_mean[2] - meanflow_targets[2])/np.abs(meanflow_prediction_mean[2] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax10.set_xticks([])
ax10.set_yticks([])
ax11 = axs[3, 2]
ax11.pcolormesh(np.abs(meanflow_prediction_mean[4] - meanflow_targets[4])/np.abs(meanflow_prediction_mean[4] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax11.set_xticks([])
ax11.set_yticks([])
ax12 = axs[4, 0]
ax12.pcolormesh(meanflow_uncertainty[0]/np.abs(meanflow_prediction_mean[0] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax12.set_xticks([])
ax12.set_yticks([])
ax12.set_ylabel('Uncertainty', fontsize=14)
ax13 = axs[4, 1]
ax13.pcolormesh(meanflow_uncertainty[2]/np.abs(meanflow_prediction_mean[2] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax13.set_xticks([])
ax13.set_yticks([])
ax14 = axs[4, 2]
ax14.pcolormesh(meanflow_uncertainty[4]/np.abs(meanflow_prediction_mean[4] + 0.001), cmap='jet', vmin=0, vmax=0.2)
ax14.set_xticks([])
ax14.set_yticks([])
ax15 = axs[5, 0]
ax15.pcolormesh(np.where(np.abs(meanflow_prediction_mean[0] - meanflow_targets[0]) < meanflow_uncertainty[0], 1, 0), cmap='jet')
ax15.set_xticks([])
ax15.set_yticks([])
ax15.set_ylabel('Bounded', fontsize=14)
ax16 = axs[5, 1]
ax16.pcolormesh(np.where(np.abs(meanflow_prediction_mean[2] - meanflow_targets[2]) < meanflow_uncertainty[2], 1, 0), cmap='jet')
ax16.set_xticks([])
ax16.set_yticks([])
ax17 = axs[5, 2]
ax17.pcolormesh(np.where(np.abs(meanflow_prediction_mean[4] - meanflow_targets[4]) < meanflow_uncertainty[4], 1, 0), cmap='jet')
ax17.set_xticks([])
ax17.set_yticks([])
norm = mpl.colors.Normalize(vmin=0, vmax=1)
errnorm = mpl.colors.Normalize(vmin=0, vmax=0.2)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=axs[:3, -1], aspect=45, shrink=0.7,
                    orientation='vertical')
cbar.set_label('Ux (norm)', fontsize=14)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=errnorm, cmap='jet'), ax=axs[3:5, -1], aspect=45, shrink=0.7,
                    orientation='vertical')
cbar.set_label('Error (norm)', fontsize=14)

plt.show()
plt.savefig('./data/figures/meanflow_summary.png', dpi=300)
plt.close()"""

# calculate the bounding percentage for each case
sst_bounded = np.nansum(np.where(np.abs(sst_prediction_mean - sst_targets)/np.abs(sst_prediction_mean + .00001) <= sst_uncertainty/np.abs(sst_prediction_mean + .00001), 1, 0)) / np.nansum(np.where(np.isnan(sst_targets) == False, 1, 0))
sm_bounded = np.nansum(np.where(np.abs(np.asarray(sm_prediction_mean) - np.asarray(sm_targets))/np.abs(sm_prediction_mean + .00001) <= np.asarray(sm_uncertainty)/np.abs(sm_prediction_mean + .00001), 1, 0)) / np.nansum(np.where(np.isnan(sm_targets) == False, 1, 0))
meanflow_bounded = np.nansum(np.where(np.abs(meanflow_prediction_mean - meanflow_targets)/np.abs(meanflow_prediction_mean + .00001) <= meanflow_uncertainty/np.abs(meanflow_prediction_mean + .00001), 1, 0)) / np.nansum(np.where(np.isnan(meanflow_targets) == False, 1, 0))
turbulence_bounded = np.nansum(np.where(np.logical_or(np.abs(turbulence_prediction - turbulence_targets)/np.abs(turbulence_prediction) <= turbulence_uncertainty/np.abs(turbulence_prediction), (np.abs(turbulence_prediction - turbulence_targets) <= .02)), 1, 0)) / np.nansum(np.where(np.isnan(turbulence_targets) == False, 1, 0))

print("SST Bounded: " + str(sst_bounded))
print("SM Bounded: " + str(sm_bounded))
print("Meanflow Bounded: " + str(meanflow_bounded))
print("Turbulence Bounded: " + str(turbulence_bounded))
print(np.shape(sst_targets))

# calculate whether the target is within the ensemble range
sst_range_bounded = np.nansum(np.where(np.logical_and(sst_targets >= sst_prediction_range[:, 0], sst_targets <= sst_prediction_range[:, 1]), 1, 0)) / np.nansum(np.where(np.isnan(sst_targets) == False, 1, 0))
sm_range_bounded = np.nansum(np.where(np.logical_and(sm_targets >= sm_prediction_range[:, 0], sm_targets <= sm_prediction_range[:, 1]), 1, 0)) / np.nansum(np.where(np.isnan(sm_targets) == False, 1, 0))
meanflow_range_bounded = np.nansum(np.where(np.logical_and(meanflow_targets >= meanflow_prediction_range[:, 0], meanflow_targets <= meanflow_prediction_range[:, 1]), 1, 0)) / np.nansum(np.where(np.isnan(meanflow_targets) == False, 1, 0))
#turbulence_range_bounded = np.nansum(np.where(np.logical_and(turbulence_targets >= turbulence_prediction_range[:, 0], turbulence_targets <= turbulence_prediction_range[:, 1]), 1, 0)) / np.nansum(np.where(np.isnan(turbulence_targets) == False, 1, 0))
print(np.nansum(np.where(np.isnan(meanflow_targets) == True, 1, 0))/np.nansum(np.where(np.isnan(meanflow_targets) == False, 1, 0)))
print("SST Range Bounded: " + str(sst_range_bounded))
print("SM Range Bounded: " + str(sm_range_bounded))
print("Meanflow Range Bounded: " + str(meanflow_range_bounded))
#print("Turbulence Range Bounded: " + str(turbulence_range_bounded))

# plot the variance on the y axis and the error on the x axis, plot 45 degree lines for negative and positive bias, plot scatter density map of points
fig = plt.figure()

# plot the variance on the y axis and the error on the x axis, plot 45 degree lines for negative and positive bias, plot scatter density map of points
ax = fig.add_subplot(421, projection='scatter_density')
error = meanflow_prediction_mean - meanflow_targets
uncertainty = meanflow_uncertainty
density = ax.scatter_density(np.where(np.isnan(error), 0, error), np.where(np.isnan(uncertainty), 0, uncertainty), cmap='viridis', vmax=500)
fig.colorbar(density, ax=ax)
ax.set_xlabel('Error')
ax.set_ylabel('Uncertainty')
ax.set_xlim(-.1, .1)
ax.set_ylim(.001, .1)
ax.set_title('a) Mean Flow')
ax.plot([0, .1], [0, .1], color='red')
ax.plot([0, -.1], [0, .1], color='red')

ax = fig.add_subplot(422, projection='scatter_density')
density = ax.scatter_density(np.where(np.isnan(meanflow_prediction_mean), 0, meanflow_prediction_mean).flatten(),
                        np.where(np.isnan(meanflow_targets), 0, meanflow_targets).flatten(), cmap='viridis', vmax=500)
fig.colorbar(density, ax=ax, label='Number of points')
ax.set_xlabel('Predicted')
ax.set_ylabel('Target')
ax.set_xlim(-.5, 2)
ax.set_ylim(-.5, 2)
ax.set_title('b) Mean Flow')
ax.plot([-.5, 2], [-.5, 2], color='red')

ax = fig.add_subplot(425, projection='scatter_density')
error = np.asarray(sm_prediction_mean) - np.asarray(sm_targets)
uncertainty = np.asarray(sm_uncertainty)
density = ax.scatter_density(np.where(np.isnan(error), 0, error), np.where(np.isnan(uncertainty), 0, uncertainty), cmap='viridis', vmax=100)
fig.colorbar(density, ax=ax)
ax.set_xlim(-.05, .05)
ax.set_ylim(.001, .05)
ax.set_xlabel('Error')
ax.set_ylabel('Uncertainty')
ax.set_title('e) Soil Moisture')
ax.plot([0, .1], [0, .1], color='red')
ax.plot([0, -.05], [0, .05], color='red')

# display a scatter plot of the prediction and targets
ax = fig.add_subplot(426, projection='scatter_density')
density = ax.scatter_density(np.where(np.isnan(sm_prediction_mean), 0, sm_prediction_mean).flatten(),
                   np.where(np.isnan(sm_targets), 0, sm_targets).flatten(), cmap='viridis', vmax=100)
fig.colorbar(density, ax=ax, label='Number of points')
ax.set_xlabel('Predicted')
ax.set_ylabel('Target')
ax.set_title('f) Soil Moisture')
ax.plot([0, 1], [0, 1], color='red')

# plot the variance on the y axis and the error on the x axis, plot 45 degree lines for negative and positive bias, plot scatter density map of points
ax = fig.add_subplot(423, projection='scatter_density')
error = sst_prediction_mean - sst_targets
uncertainty = sst_uncertainty
density = ax.scatter_density(np.where(np.isnan(error), 0, error), np.where(np.isnan(uncertainty), 0, uncertainty), cmap='viridis', vmax=200)
fig.colorbar(density, ax=ax)
ax.set_xlabel('Error')
ax.set_ylabel('Uncertainty')
ax.set_xlim(-.05, .05)
ax.set_ylim(.001, .05)
ax.set_title('c) Sea-surface Temperature')
ax.plot([0, .1], [0, .1], color='red')
ax.plot([0, -.05], [0, .05], color='red')

# display a scatter plot of the prediction and targets
ax = fig.add_subplot(424, projection='scatter_density')
density = ax.scatter_density(np.where(np.isnan(sst_prediction_mean), 0, sst_prediction_mean).flatten(),
                     np.where(np.isnan(sst_targets), 0, sst_targets).flatten(), cmap='viridis', vmax=200)
fig.colorbar(density, ax=ax, label='Number of points')
ax.set_xlabel('Predicted')
ax.set_ylabel('Target')
ax.set_title('d) Sea-surface Temperature')
ax.plot([0, 1], [0, 1], color='red')


ax = fig.add_subplot(427, projection='scatter_density')
error = turbulence_prediction - turbulence_targets
uncertainty = turbulence_uncertainty
density = ax.scatter_density(np.where(np.isnan(error), 0, error), np.where(np.isnan(uncertainty), 0, uncertainty), cmap='viridis', vmax=500)
fig.colorbar(density, ax=ax)
ax.set_xlabel('Error')
ax.set_ylabel('Uncertainty')
ax.set_xlim(-.1, .1)
ax.set_ylim(.001, .1)
ax.set_title('Turbulence')
ax.plot([0, .1], [0, .1], color='red')
ax.plot([0, -.1], [0, .1], color='red')

ax = fig.add_subplot(428, projection='scatter_density')
density = ax.scatter_density(np.where(np.isnan(turbulence_prediction), 0, turbulence_prediction).flatten(),
                        np.where(np.isnan(turbulence_targets), 0, turbulence_targets).flatten(), cmap='viridis', vmax=500)
fig.colorbar(density, ax=ax, label='Number of points')
ax.set_xlabel('Predicted')
ax.set_ylabel('Target')
ax.set_xlim(-.5, 2)
ax.set_ylim(-.5, 2)
ax.set_title('Turbulence')
ax.plot([-.5, 2], [-.5, 2], color='red')

plt.subplots_adjust(hspace=0.45, right=.7, left=.3)
plt.show()


# plot the uncertainty and range bounds of each different ensemble count for the SST case
ensemble_counts = glob.glob('./data/npy/sst.wkmean_*_prediction_range.npz')
# grab the ensemble counts from the file names
ensemble_counts = [re.findall(r'\d+', file)[0] for file in ensemble_counts]
ensemble_counts = natsorted(ensemble_counts)
# create lists for the ensemble ranges, uncertainties, targets, prediction means, and errors
ensemble_ranges = []
ensemble_uncertainties = []
ensemble_targets = []
ensemble_prediction_means = []
ensemble_errors = []
# loop through each ensemble count
for ensemble_count in ensemble_counts:
    # load the data
    srange = np.load('./data/npy/sst.wkmean_{}_prediction_range.npz'.format(ensemble_count))['arr_0']
    uncertainty = np.load('./data/npy/sst.wkmean_{}_uncertainty.npz'.format(ensemble_count))['arr_0']
    target = np.load('./data/npy/sst.wkmean_{}_targets.npz'.format(ensemble_count))['arr_0']
    prediction_mean = np.load('./data/npy/sst.wkmean_{}_prediction_mean.npz'.format(ensemble_count))['arr_0']

    # append the data to the lists
    ensemble_ranges.append(srange)
    ensemble_uncertainties.append(uncertainty)
    ensemble_targets.append(target)
    ensemble_prediction_means.append(prediction_mean)
    ensemble_errors.append(prediction_mean - target)

# calculate the bounding percentage and whether the target is inside the range for each ensemble number
ensemble_bounding_percentages = []
ensemble_range_bounding_percentages = []

for i in range(len(ensemble_counts)):
    # calculate the bounding percentage
    bounding_percentage = np.sum(np.abs(ensemble_errors[i]) < ensemble_uncertainties[i]) / np.nansum(np.isfinite(ensemble_errors[i]))
    # calculate the range bounding percentage
    range_bounding_percentage = np.sum((ensemble_targets[i] > ensemble_ranges[i][0]) & (ensemble_targets[i] < ensemble_ranges[i][1])) / np.nansum(np.isfinite(ensemble_errors[i]))
    # append the percentages to the list
    ensemble_bounding_percentages.append(bounding_percentage)
    ensemble_range_bounding_percentages.append(range_bounding_percentage)

prediction_list_300 = np.squeeze(np.load('./data/npy/sstprediction_list300_1.npy'))
target_list_300 = np.load('./data/npy/ssttarget_list300_1.npy')
input_list_300 = np.load('./data/npy/sstinput_list300_1.npy')
print(prediction_list_300.shape)
print(target_list_300.shape)
print(input_list_300.shape)
prediction_list_100 = np.squeeze(np.load('./data/npy/sstprediction_list100_1.npy'))
target_list_100 = np.load('./data/npy/ssttarget_list100_1.npy')
input_list_100 = np.load('./data/npy/sstinput_list100_1.npy')
print(prediction_list_100.shape)
print(target_list_100.shape)
print(input_list_100.shape)

from scipy.stats import norm

# plot all the predictions in the ensemble for a few different points in four subplots
fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(321)
ax1.hist(prediction_list_100[:, 25, 50], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_100[:, 25, 50])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
print("plot a, mean: {}, std: {}".format(mu, std))
ax1.plot(x, p, 'k', linewidth=2)
ax1.axvline(x=target_list_100[0, 25, 50], color='r')
ax1 = fig.add_subplot(322)
ax1.hist(prediction_list_300[:, 25, 50], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_300[:, 25, 50])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
print("plot b, mean: {}, std: {}".format(mu, std))
ax1.plot(x, p, 'k', linewidth=2)
ax1.axvline(x=target_list_300[0, 25, 50], color='r')
ax3 = fig.add_subplot(323)
ax3.hist(prediction_list_100[:, 186, 150], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_100[:, 186, 150])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
print("plot c, mean: {}, std: {}".format(mu, std))
ax3.plot(x, p, 'k', linewidth=2)
ax3.axvline(x=target_list_100[0, 186, 150], color='r')
ax4 = fig.add_subplot(324)
ax4.hist(prediction_list_300[:, 186, 150], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_300[:, 186, 150])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 300)
p = norm.pdf(x, mu, std)
print("plot d, mean: {}, std: {}".format(mu, std))
ax4.plot(x, p, 'k', linewidth=2)
ax4.axvline(x=target_list_300[0, 186, 150], color='r')
ax5 = fig.add_subplot(325)
ax5.hist(prediction_list_100[:, 150, 300], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_100[:, 150, 300])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 300)
p = norm.pdf(x, mu, std)
print("plot e, mean: {}, std: {}".format(mu, std))
ax5.plot(x, p, 'k', linewidth=2)
ax5.axvline(x=target_list_100[0, 150, 300], color='r')
ax6 = fig.add_subplot(326)
ax6.hist(prediction_list_300[:, 150, 300], bins=10, edgecolor='black', facecolor='green')#, density=True, stacked=True)
mu, std = norm.fit(prediction_list_300[:, 150, 300])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 300)
p = norm.pdf(x, mu, std)
print("plot f, mean: {}, std: {}".format(mu, std))
ax6.plot(x, p, 'k', linewidth=2)
ax6.axvline(x=target_list_300[0, 150, 300], color='r')
plt.show()


