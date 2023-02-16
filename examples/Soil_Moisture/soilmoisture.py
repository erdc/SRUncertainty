import sys
sys.path.insert(0, '/media/amcoast/Work_SSD/Superresolution')
import os
import glob
import matplotlib as mpl
import tensorflow as tf
import sm_params
import netCDF4 as nc
import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import NearestNDInterpolator
import cv2
import mpl_scatter_density
import logging

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.keras.backend.clear_session()

logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)

def load_unet():

    inputs = tf.keras.layers.Input((sm_params.input_rows, sm_params.input_cols, sm_params.input_features))
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv2)
    drop2 = tf.keras.layers.Dropout(0.1)(conv2, training=True)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv3)
    drop3 = tf.keras.layers.Dropout(0.1)(conv3, training=True)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv4)
    drop4 = tf.keras.layers.Dropout(0.1)(conv4, training=True)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.BatchNormalization(trainable=True)(conv5)
    conv5 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.BatchNormalization(trainable=True)(conv5)
    conv5 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                          kernel_initializer='he_normal')((conv5))
    conv5 = tf.keras.layers.BatchNormalization(trainable=True)(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5, training=True)
    merge6 = tf.keras.layers.concatenate([drop4, drop5], axis=3)

    conv6 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.BatchNormalization(trainable=True)(conv6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = tf.keras.layers.BatchNormalization(trainable=True)(conv6)
    conv6 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                            kernel_initializer='he_normal')((conv6))
    conv6 = tf.keras.layers.BatchNormalization(trainable=True)(conv6)
    conv6 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv6)
    drop6 = tf.keras.layers.Dropout(0.1)(conv6, training=True)

    merge7 = tf.keras.layers.concatenate([drop3, drop6], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                            kernel_initializer='he_normal')((conv7))
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv7)
    drop7 = tf.keras.layers.Dropout(0.1)(conv7, training=True)

    merge8 = tf.keras.layers.concatenate([drop2, drop7], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                            kernel_initializer='he_normal')((conv8))
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.GaussianNoise(sm_params.noise_std)(conv8)
    drop8 = tf.keras.layers.Dropout(0.1)(conv8, training=True)

    merge9 = tf.keras.layers.concatenate([conv1, drop8], axis=3)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(sm_params.target_features, 1, activation=None)(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=[conv10])
    optimizer = tf.keras.optimizers.Nadam()

    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mae', 'mse'])

    return model


def load_model():
    try:
        saves = glob.glob('./model_chk/sm.h5')
        #saves = glob.glob('./model_chk/sm (copy).h5')
        #saves = glob.glob('./model_chk/sm_iter.h5')
        model = load_unet()
        model.load_weights(saves[-1])
        print("unet checkpoint loaded")
        print(saves[-1])
    except:
        print("Didn't find saved sm checkpoint")
        model = load_unet()
    return model


def load_data():
    try:
        sm_data = np.load('./data/sm/sm_dataset.npy', allow_pickle=True)
    except:
        print("loading data from scratch")
        layer_list = []
        for layer in range(1, 4):
            sm_list = []
            lat_list = []
            lon_list = []
            time_list = []
            for yr in range(2000, 2019, 3):
                layer_fn = './data/sm/layer' + str(layer) + '/layer' + str(layer) + '/SoMo.ml_v1_layer' + str(layer) + '_' + str(yr) + '.nc'
                layer_data = nc.Dataset(layer_fn)
                sm = layer_data.variables['layer' + str(layer)][::7]
                lat = layer_data.variables['lat'][::7]
                lon = layer_data.variables['lon'][::7]
                time = layer_data.variables['time'][::7]
                global mask
                mask = np.where(np.isnan(sm[0]), 1, 0)
                sm_list.append(sm)
                lat_list.append(lat)
                lon_list.append(lon)
                time_list.append(time)
                layer_data.close()

            sm_list = np.array(sm_list)
            lat_list = np.array(lat_list)
            lon_list = np.array(lon_list)
            time_list = np.array(time_list)
            # create dictionary of sm_list lat_list lon_list time_list
            layer_dict = {'sm': sm_list, 'lat': lat_list, 'lon': lon_list, 'time': time_list}
            layer_list.append(layer_dict)
        np.save('./data/sm/sm_dataset.npy', layer_list)
        print("saved")

    return sm_data


def load_mask():
    layer_fn = './data/sm/layer1/layer1/SoMo.ml_v1_layer1_2000.nc'
    layer_data = nc.Dataset(layer_fn)
    sm = layer_data.variables['layer1'][:]
    mask = np.where(np.isnan(sm[0]), 0.0, 1.0)
    mask = np.squeeze(mask)
    resized_mask = cv2.resize(mask, (sm_params.input_cols, sm_params.input_rows))
    resized_mask = np.expand_dims(resized_mask, axis=-1)
    return mask, resized_mask


def tesselize_data():
    tess_layer = []
    resized_layer = []
    try:
        tess_input_data = np.load('./data/sm/' + sm_params.dataset + '_tess.npy')
        resized_input_data = np.load('./data/sm/' + sm_params.dataset + '_resized.npy')
    except:
        print("tessellating data")
        for layer_no in range(len(data)):
            yearly_list = []
            resized_yearly_list = []
            print("layer " + str(layer_no + 1))
            for yr in range(data[layer_no]['sm'].shape[0])[::3]:
                print("year " + str(yr + 1))
                sm_values = data[layer_no]['sm'][yr]
                tesselized_feature_list = []
                resized_features_list = []
                for sensor_no in sm_params.sensor_list:
                    print("sensor " + str(sensor_no))
                    tess_features = np.zeros((len(sm_values), 256, 512))
                    resized_features = np.zeros((len(sm_values), 256, 512))
                    x = np.linspace(0, 512, 512)
                    y = np.linspace(0, 256, 256)
                    X, Y = np.meshgrid(x, y, indexing='ij')
                    centroids_x = np.linspace(0, 512, int(np.sqrt(sensor_no)))
                    centroids_y = np.linspace(0, 256, int(np.sqrt(sensor_no)))
                    centroids_X, centroids_Y = np.meshgrid(centroids_x, centroids_y, indexing='ij')
                    centroids = np.vstack((centroids_X.flatten(), centroids_Y.flatten())).T

                    for i in range(sm_values.shape[0]):
                        sm_values[i] = np.where(np.isnan(sm_values[i]), 0, sm_values[i])
                        temp_sm = cv2.resize(sm_values[i], (sm_params.input_cols, sm_params.input_rows))
                        tess_interp = NearestNDInterpolator(centroids, temp_sm[
                            centroids[:, 1].astype(int) - 1, centroids[:, 0].astype(int) - 1])
                        tess_interp = tess_interp(X, Y)
                        tess_features[i] = tess_interp.T
                        resized_features[i] = temp_sm

                        # plot tess features with original features subplot
                        """plt.subplot(1, 2, 1)
                        plt.imshow(resized_features[i])
                        plt.subplot(1, 2, 2)
                        plt.imshow(tess_features[i])
                        plt.show()"""

                    tesselized_feature_list.append(tess_features)
                    resized_features_list.append(resized_features)
                tesselized_features = np.asarray(tesselized_feature_list)
                resized_features = np.asarray(resized_features_list)
                yearly_list.append(tesselized_features)
                resized_yearly_list.append(resized_features)
            yearly_list = np.asarray(yearly_list)
            resized_yearly_list = np.asarray(resized_yearly_list)
            tess_layer.append(yearly_list)
            resized_layer.append(resized_yearly_list)
        tess_layer = np.asarray(tess_layer)
        resized_layer = np.asarray(resized_layer)
        tess_layer = np.swapaxes(tess_layer, 0, 1)
        tess_layer = np.swapaxes(tess_layer, 1, 2)
        tess_layer = np.swapaxes(tess_layer, 2, 3)
        tess_layer = np.swapaxes(tess_layer, 3, 4)
        tess_dataset = np.swapaxes(tess_layer, 4, 5)

        resized_layer = np.swapaxes(resized_layer, 0, 1)
        resized_layer = np.swapaxes(resized_layer, 1, 2)
        resized_layer = np.swapaxes(resized_layer, 2, 3)
        resized_layer = np.swapaxes(resized_layer, 3, 4)
        resized_dataset = np.swapaxes(resized_layer, 4, 5)

        np.save('./data/sm/' + sm_params.dataset + '_tess.npy', tess_dataset)
        np.save('./data/sm/' + sm_params.dataset + '_resized.npy', resized_dataset)
        print("tessellated data saved")
        tess_input_data = tess_dataset
        resized_input_data = resized_dataset
    return tess_input_data, resized_input_data


def split_data():
    test_times = data[0]['time'][-1]
    train_times = data[0]['time'][:-1]
    global tess_data, resized_data

    test_tess_data = tess_data[-1:, :, :]
    test_resized_data = resized_data[-1:, :, :]
    tess_data = tess_data[:-1, :, :]
    resized_data = resized_data[:-1, :, :]

    tess_reshaped = np.reshape(tess_data, (tess_data.shape[0]*tess_data.shape[1]*tess_data.shape[2], tess_data.shape[3], tess_data.shape[4], tess_data.shape[5]))
    resized_reshaped = np.reshape(resized_data, (resized_data.shape[0]*resized_data.shape[1]*resized_data.shape[2], resized_data.shape[3], resized_data.shape[4], resized_data.shape[5]))
    test_tess_data = np.reshape(test_tess_data, (test_tess_data.shape[0]*test_tess_data.shape[1]*test_tess_data.shape[2], test_tess_data.shape[3], test_tess_data.shape[4], test_tess_data.shape[5]))
    test_resized_data = np.reshape(test_resized_data, (test_resized_data.shape[0]*test_resized_data.shape[1]*test_resized_data.shape[2], test_resized_data.shape[3], test_resized_data.shape[4], test_resized_data.shape[5]))
    # merge first two dimensions with reshape
    train_targets = resized_reshaped
    test_targets = test_resized_data
    train_inputs = tess_reshaped
    test_inputs = test_tess_data

    return train_inputs, train_targets, test_inputs, test_targets, test_times


def get_batch(time, input, target):
    temp_input_batch = (input[time, :, :, :sm_params.target_features] - target_min) / (target_max - target_min)
    temp_target_batch = (target[time, :, :, :sm_params.target_features] - target_min) / (target_max - target_min)

    temp_input_batch = np.concatenate((temp_input_batch, resized_mask), axis=-1)
    temp_input_batch = np.where(np.isnan(temp_input_batch), -1, temp_input_batch)
    temp_target_batch = np.where(np.isnan(temp_target_batch), -1, temp_target_batch)
    """fig = plt.figure(figsize=(7, 4))
    ax0 = fig.add_subplot(121)
    ax0.imshow(temp_input_batch[:, :, 0])
    ax1 = fig.add_subplot(122)
    ax1.imshow(temp_target_batch)
    plt.show()"""
    return temp_input_batch, temp_target_batch


def train():
    print("Training for " + str(sm_params.epochs) + " epochs")
    old_best = 1000
    train_log_dir = './logs/train'
    test_log_dir = './logs/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    test_summary_writer = tf.summary.create_file_writer(test_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    X_train, X_val, temp_y_train, temp_y_val = train_test_split(train_data[::2], train_targets[::2], test_size=0.2, random_state=42)
    #y_train = X_train
    #y_val = X_val
    y_train = temp_y_train
    y_val = temp_y_val

    for epoch in range(sm_params.epochs):
        avg_train_loss = []
        #if epoch > 10:
        #    y_train = temp_y_train
        #    y_val = temp_y_val
        if epoch % 1 == 0:
            avg_val_loss = []
            print(str(epoch) + "/" + str(sm_params.epochs))
            for j in tqdm(range(len(X_val))):
                input_batch = []
                target_batch = []
                for k in range(sm_params.batch_size):
                    time = np.random.randint(0, len(X_val))
                    temp_input_batch, temp_target_batch = get_batch(time, X_val, y_val)
                    input_batch.append(temp_input_batch)
                    target_batch.append(temp_target_batch)
                target_batch = np.expand_dims(target_batch, axis=-1)
                history = model.test_on_batch(np.asarray(input_batch), target_batch)
                avg_val_loss.append(history[0])
            if np.nanmean(avg_val_loss) < old_best:
                print("New best validation loss: ", np.nanmean(np.asarray(avg_val_loss)))
                old_best = np.nanmean(avg_val_loss)
                val_loss = np.nanmean(avg_val_loss)
                model.save('./model_chk/sm.h5', overwrite=True)
                test()
                plot_results()
            else:
                print("val loss failed to improve")
                print("Current LR: " + str(model.optimizer.lr) + "reducing learning rate by 10%")
                tf.keras.backend.set_value(model.optimizer.lr, model.optimizer.lr * .95)
                print("New LR: " + str(tf.keras.backend.get_value(model.loss)))
            with test_summary_writer.as_default():
                tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch)
        for j in tqdm(range(int(len(X_train)/sm_params.batch_size))):
            input_batch = []
            target_batch = []
            for k in range(sm_params.batch_size):
                time = np.random.randint(0, len(X_train))
                temp_input_batch, temp_target_batch = get_batch(time, X_train, y_train)
                # plot all 7 panels of the input and target batches
                """fig = plt.figure(figsize=(7, 4))
                ax0 = fig.add_subplot(241)
                ax0.imshow(temp_input_batch[:, :, 0])
                ax1 = fig.add_subplot(242)
                ax1.imshow(temp_input_batch[:, :, 1])
                ax2 = fig.add_subplot(243)
                ax2.imshow(temp_input_batch[:, :, 2])
                ax3 = fig.add_subplot(244)
                ax3.imshow(temp_input_batch[:, :, 3])
                ax4 = fig.add_subplot(245)
                ax4.imshow(temp_target_batch[:, :, 0])
                ax5 = fig.add_subplot(246)
                ax5.imshow(temp_target_batch[:, :, 1])
                ax6 = fig.add_subplot(247)
                ax6.imshow(temp_target_batch[:, :, 2])
                plt.show()"""
                input_batch.append(temp_input_batch)
                target_batch.append(temp_target_batch)
            input_batch = np.array(input_batch)
            target_batch = np.array(target_batch)
            target_batch = np.expand_dims(target_batch, axis=-1)
            history = model.train_on_batch(input_batch, target_batch)
            avg_train_loss.append(history[0])
        model.save('./model_chk/sm_iter.h5', overwrite=True)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', np.mean(avg_train_loss), step=epoch)
    return model


def test():
    X_test, y_test = test_data[-int(len(test_data)/len(sm_params.sensor_list))::15], test_targets[-int(len(test_data)/len(sm_params.sensor_list))::15]
    print("Testing on " + str(len(X_test)) + " samples")
    ensemble_prediction_list = []
    ensemble_target_list = []
    ensemble_input_list = []

    for i in tqdm(range(int(len(X_test)))):
        prediction_list = []
        target_list = []
        input_list = []
        for j in range(sm_params.ensembles):
            input_batch = []
            target_batch = []
            for k in range(sm_params.batch_size):
                if k == 0:
                    time = i
                    temp_input_batch, temp_target_batch = get_batch(time, X_test, y_test)
                else:
                    time = np.random.randint(0, len(X_test))
                    temp_input_batch, temp_target_batch = get_batch(time, test_data, test_targets)
                input_batch.append(temp_input_batch)
                target_batch.append(temp_target_batch)
            prediction = model.predict_on_batch(np.asarray(input_batch))
            prediction = prediction[0]
            prediction = np.where(np.isnan(resized_mask), np.nan, prediction)
            target = target_batch[0]
            prediction_list.append(prediction)
            target_list.append(target)
            input = input_batch[0]
            input = np.where(np.isnan(resized_mask), np.nan, input)
            input_list.append(input)
        prediction_list = np.asarray(prediction_list)
        target_list = np.asarray(target_list)
        input_list = np.asarray(input_list)
        ensemble_prediction_list.append(prediction_list)
        ensemble_target_list.append(target_list)
        ensemble_input_list.append(input_list)

    prediction_mean = np.nanmean(ensemble_prediction_list, axis=1)
    prediction_range = [np.nanmin(ensemble_prediction_list, axis=1), np.nanmax(ensemble_prediction_list, axis=1)]
    prediction_std = np.nanstd(ensemble_prediction_list, axis=1)
    uncertainty = prediction_std * 2
    targets = np.nanmean(ensemble_target_list, axis=1)
    inputs = np.nanmean(ensemble_input_list, axis=1)

    np.savez_compressed('./data/npy/' + sm_params.dataset + 'prediction_mean.npz', prediction_mean)
    np.savez_compressed('./data/npy/' + sm_params.dataset + 'prediction_range.npz', prediction_range)
    np.savez_compressed('./data/npy/' + sm_params.dataset + 'prediction_std.npz', prediction_std)
    np.savez_compressed('./data/npy/' + sm_params.dataset + 'uncertainty.npz', uncertainty)
    np.savez_compressed('./data/npy/' + sm_params.dataset + 'targets.npz', targets)
    np.savez_compressed('./data/npy/' + sm_params.dataset + 'inputs.npz', inputs)


def calc_stats():
    prediction_mean = np.load('./data/npy/' + sm_params.dataset + 'prediction_mean.npz')['arr_0']
    prediction_std = np.load('./data/npy/' + sm_params.dataset + 'prediction_std.npz')['arr_0']
    uncertainty = np.load('./data/npy/' + sm_params.dataset + 'uncertainty.npz')['arr_0']
    targets = np.load('./data/npy/' + sm_params.dataset + 'targets.npz')['arr_0']
    inputs = np.load('./data/npy/' + sm_params.dataset + 'inputs.npz')['arr_0']
    plot_mask = cv2.resize(resized_mask, (360, 180))
    plot_mask = np.expand_dims(plot_mask, axis=-1)
    plot_mask = np.concatenate((plot_mask, plot_mask, plot_mask), axis=2)
    plot_mask = np.where(plot_mask == 0, np.nan, plot_mask)
    inputs_downsized_list = []
    uc_downsized_list = []
    targets_downsized_list = []
    prediction_mean_downsized_list = []
    for i in range(len(prediction_mean[-int(len(test_data)/len(sm_params.sensor_list)):])):
        prediction_mean_downsized = cv2.resize(prediction_mean[i], (360, 180))
        prediction_std_downsized = cv2.resize(prediction_std[i], (360, 180))
        uncertainty_downsized = cv2.resize(uncertainty[i], (360, 180))
        targets_downsized = cv2.resize(targets[i], (360, 180))
        inputs_downsized = cv2.resize(inputs[i], (360, 180))
        prediction_mean_downsized = np.where(np.isnan(plot_mask), np.nan, prediction_mean_downsized)
        prediction_std_downsized = np.where(np.isnan(plot_mask), np.nan, prediction_std_downsized)
        uncertainty_downsized = np.where(np.isnan(plot_mask), np.nan, uncertainty_downsized)
        targets_downsized = np.where(np.isnan(plot_mask), np.nan, targets_downsized)
        inputs_downsized = np.where(np.isnan(plot_mask), np.nan, inputs_downsized[:, :, :sm_params.target_features])
        uc_downsized_list.append(uncertainty_downsized)
        targets_downsized_list.append(targets_downsized)
        prediction_mean_downsized_list.append(prediction_mean_downsized)
        inputs_downsized_list.append(inputs_downsized)

    # calculate RMSE, MAE, and bias of prediction list
    rmse = np.sqrt(np.nanmean((np.asarray(prediction_mean_downsized_list) - np.asarray(targets_downsized_list))**2))
    mae = np.nanmean(np.abs(np.asarray(prediction_mean_downsized_list) - np.asarray(targets_downsized_list)))
    bias = np.nanmean(np.asarray(prediction_mean_downsized_list) - np.asarray(targets_downsized_list))
    print("SM RMSE: " + str(rmse))
    print("SM MAE: " + str(mae))
    print("SM Bias: " + str(bias))

    # plot average of uncertainty within values at each pixel
    uc_downsized_list = np.asarray(uc_downsized_list)
    prediction_mean_downsized_list = np.asarray(prediction_mean_downsized_list)
    targets_downsized_list = np.asarray(targets_downsized_list)
    inputs_downsized_list = np.asarray(inputs_downsized_list)
    inputs_downsized_mean = np.nanmean(inputs_downsized_list, axis=0)
    inputs_downsized_mean = np.nanmean(inputs_downsized_mean, axis=-1)

    within_list = np.where(uc_downsized_list > (prediction_mean_downsized_list - targets_downsized_list), 1, 0)
    within_list = np.nanmean(within_list, axis=0)
    within_list = np.nanmean(within_list, axis=-1)

    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(211)
    ax0.pcolormesh(inputs_downsized_mean, cmap='viridis', vmax=1, vmin=0)
    ax0.invert_yaxis()
    ax0.set_title('Layer Averaged Inputs')
    ax0.set_xticks([])
    ax0.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
    fig.colorbar(ax0.get_children()[0], ax=ax0)
    ax = fig.add_subplot(212)
    ax.pcolormesh(within_list, cmap='viridis', vmax=1, vmin=0)
    ax.invert_yaxis()
    ax.set_title('Layer Averaged Uncertainty')
    ax.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
    ax.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
    fig.colorbar(ax.get_children()[0], ax=ax)
    plt.show()

    # calculate spatial average of the predictions, errors, and uncertainties over the test set
    """prediction_mean_downsized_list = np.asarray(prediction_mean_downsized_list)
    targets_downsized_list = np.asarray(targets_downsized_list)
    uc_downsized_list = np.asarray(uc_downsized_list)
    prediction_mean_downsized_list = np.nanmean(prediction_mean_downsized_list, axis=0)
    targets_downsized_list = np.nanmean(targets_downsized_list, axis=0)
    uc_downsized_list = np.nanmean(uc_downsized_list, axis=0)
    inputs_downsized_list = np.nanmean(inputs_downsized_list, axis=0)

    # plot spatial averages of the predictions, errors, and uncertaitnies
    fig = plt.figure(figsize=(12, 4))
    ax0 = fig.add_subplot(321)
    ax0.imshow(inputs_downsized_list)
    ax0.set_title('Inputs')
    ax1 = fig.add_subplot(322)
    ax1.imshow(prediction_mean_downsized_list)
    ax1.set_title('Prediction')
    ax2 = fig.add_subplot(323)
    ax2.imshow(targets_downsized_list)
    ax2.set_title('Target')
    ax3 = fig.add_subplot(324)
    ax3.pcolormesh(np.abs(np.nanmean(prediction_mean_downsized_list - targets_downsized_list, axis=-1)), cmap='jet', vmax=0.1, vmin=0)
    ax3.invert_yaxis()
    ax3.set_title('Error')
    ax4 = fig.add_subplot(325)
    ax4.pcolormesh(np.abs(np.nanmean(uc_downsized_list, axis=-1)), cmap='jet', vmax=0.1, vmin=0)
    ax4.invert_yaxis()
    ax4.set_title('Uncertainty')
    ax5 = fig.add_subplot(326)
    ax5.pcolormesh(np.where(np.abs(np.nanmean(prediction_mean_downsized_list - targets_downsized_list, axis=-1)) < np.abs(np.nanmean(uc_downsized_list, axis=-1)), 1, 0), cmap='jet', vmax=1, vmin=0)
    ax5.invert_yaxis()
    ax5.set_title('Error < Uncertainty | UC%: ' + str(np.round(np.nansum(np.where(np.abs(np.nanmean(prediction_mean_downsized_list - targets_downsized_list, axis=-1)) < np.abs(np.nanmean(uc_downsized_list, axis=-1)), 1, 0))/np.nansum(np.where(np.isnan(np.nanmean(prediction_mean_downsized_list - targets_downsized_list, axis=-1)), 0, 1))*100, 2)))
    plt.tight_layout()
    plt.show()"""


def plot_results():
    prediction_mean = np.load('./data/npy/' + sm_params.dataset + 'prediction_mean.npz')['arr_0']
    prediction_range = np.load('./data/npy/' + sm_params.dataset + 'prediction_range.npz')['arr_0']
    prediction_std = np.load('./data/npy/' + sm_params.dataset + 'prediction_std.npz')['arr_0']
    uncertainty = np.load('./data/npy/' + sm_params.dataset + 'uncertainty.npz')['arr_0']
    targets = np.load('./data/npy/' + sm_params.dataset + 'targets.npz')['arr_0']
    inputs = np.load('./data/npy/' + sm_params.dataset + 'inputs.npz')['arr_0']
    plot_mask = cv2.resize(resized_mask, (360, 180))
    plot_mask = np.expand_dims(plot_mask, axis=-1)
    plot_mask = np.concatenate((plot_mask, plot_mask, plot_mask), axis=2)
    plot_mask = np.where(plot_mask == 0, np.nan, plot_mask)[:, :, :]

    for i in range(len(prediction_mean[-int(len(test_data)/len(sm_params.sensor_list)):])):
        prediction_mean_downsized = cv2.resize(prediction_mean[i], (360, 180))
        prediction_std_downsized = cv2.resize(prediction_std[i], (360, 180))
        prediction_range_downsized = [cv2.resize(prediction_range[0][i], (360, 180)), cv2.resize(prediction_range[1][i], (360, 180))]
        uncertainty_downsized = cv2.resize(uncertainty[i], (360, 180))
        targets_downsized = cv2.resize(targets[i], (360, 180))
        inputs_downsized = cv2.resize(inputs[i], (360, 180))

        if len(prediction_mean_downsized.shape) == 2:
            prediction_mean_downsized = np.expand_dims(prediction_mean_downsized, axis=-1)
            prediction_std_downsized = np.expand_dims(prediction_std_downsized, axis=-1)
            prediction_range_downsized = np.expand_dims(prediction_range_downsized, axis=-1)
            uncertainty_downsized = np.expand_dims(uncertainty_downsized, axis=-1)
            targets_downsized = np.expand_dims(targets_downsized, axis=-1)

        prediction_mean_downsized = np.where(np.isnan(plot_mask[:, :, :sm_params.target_features]), np.nan, prediction_mean_downsized)
        prediction_std_downsized = np.where(np.isnan(plot_mask[:, :, :sm_params.target_features]), np.nan, prediction_std_downsized)
        uncertainty_downsized = np.where(np.isnan(plot_mask[:, :, :sm_params.target_features]), np.nan, uncertainty_downsized)
        targets_downsized = np.where(np.isnan(plot_mask[:, :, :sm_params.target_features]), np.nan, targets_downsized)
        inputs_downsized = np.where(np.isnan(plot_mask[:, :, :sm_params.target_features]), np.nan, inputs_downsized[:, :, :sm_params.target_features])

        while(prediction_mean_downsized.shape[-1] < 3):
            prediction_mean_downsized = np.concatenate((prediction_mean_downsized, np.zeros((prediction_mean_downsized.shape[0], prediction_mean_downsized.shape[1], 1))), axis=2)
            prediction_std_downsized = np.concatenate((prediction_std_downsized, np.zeros((prediction_std_downsized.shape[0], prediction_std_downsized.shape[1], 1))), axis=2)
            uncertainty_downsized = np.concatenate((uncertainty_downsized, np.zeros((uncertainty_downsized.shape[0], uncertainty_downsized.shape[1], 1))), axis=2)
            targets_downsized = np.concatenate((targets_downsized, np.zeros((targets_downsized.shape[0], targets_downsized.shape[1], 1))), axis=2)

        while(inputs_downsized.shape[-1] < 3):
            inputs_downsized = np.concatenate((inputs_downsized, np.zeros((inputs_downsized.shape[0], inputs_downsized.shape[1], 1))), axis=2)

        # make range the uncertainty

        # plot example predictions
        norm = mpl.cm.colors.Normalize(vmax=.1, vmin=0)

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Soil Moisture 3 Channels\n Shown as RGB images, with R correlating with shallowest depth, G with medium depth, B with greatest depth \n', fontsize=16)
        ax0 = fig.add_subplot(4, 3, 1)
        ax0.imshow(np.where(np.isnan(plot_mask[:, :, ]), 1, inputs_downsized))
        ax0.set_xticks([])
        ax0.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
        ax0.set_title("a) Model Input (norm)")

        ax1 = fig.add_subplot(4, 3, 2)
        ax1.imshow(np.where(np.isnan(plot_mask[:, :, :]), 1, prediction_mean_downsized))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("b) Model Prediction (norm)")

        ax2 = fig.add_subplot(4, 3, 3)
        ax2.imshow(np.where(np.isnan(plot_mask[:, :, :]), 1, targets_downsized))
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("c) Model Target (norm)")

        ax3 = fig.add_subplot(4, 3, 4)
        ax3.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.abs(prediction_mean_downsized[:, :, 0] - targets_downsized[:, :, 0])/np.abs(prediction_mean_downsized[:, :, 1]+.00001)), norm=norm, cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
        ax3.set_title("d) Layer 1 Relative Error")

        ax3 = fig.add_subplot(4, 3, 5)
        ax3.imshow(np.where(np.isnan(plot_mask[:, :, 1]), np.nan, np.abs(prediction_mean_downsized[:, :, 1] - targets_downsized[:, :, 1])/np.abs(prediction_mean_downsized[:, :, 1]+.00001)), norm=norm, cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("e) Layer 2 Relative Error")

        ax3 = fig.add_subplot(4, 3, 6)
        ax3.imshow(np.where(np.isnan(plot_mask[:, :, 2]), np.nan, np.abs(prediction_mean_downsized[:, :, 2] - targets_downsized[:, :, 2])/np.abs(prediction_mean_downsized[:, :, 1]+.00001)), norm=norm, cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("f) Layer 3 Relative Error")

        ax4 = fig.add_subplot(4, 3, 7)
        ax4.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, uncertainty_downsized[:, :, 0])/np.abs(prediction_mean_downsized[:, :, 1]+.00001), norm=norm, cmap='jet')
        ax4.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
        ax4.set_xticks([])
        ax4.set_title("g) Layer 1 Relative Uncertainty")

        ax4 = fig.add_subplot(4, 3, 8)
        ax4.imshow(np.where(np.isnan(plot_mask[:, :, 1]), np.nan, uncertainty_downsized[:, :, 1])/np.abs(prediction_mean_downsized[:, :, 1]+.00001), norm=norm, cmap='jet')
        ax4.set_yticks([])
        ax4.set_xticks([])
        ax4.set_title("h) Layer 2 Relative Uncertainty")

        ax4 = fig.add_subplot(4, 3, 9)
        ax4.imshow(np.where(np.isnan(plot_mask[:, :, 2]), np.nan, uncertainty_downsized[:, :, 2])/np.abs(prediction_mean_downsized[:, :, 2]+.00001), norm=norm, cmap='jet')
        ax4.set_yticks([])
        ax4.set_xticks([])
        ax4.set_title("i) Layer 3 Relative Uncertainty")

        """ax5 = fig.add_subplot(4, 3, 10)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(uncertainty_downsized[:, :, 0] > np.abs(prediction_mean_downsized[:, :, 0] - targets_downsized[:, :, 0]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
        ax5.set_title("Error Bounded (binary)")

        ax5 = fig.add_subplot(4, 3, 11)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(uncertainty_downsized[:, :, 1] > np.abs(prediction_mean_downsized[:, :, 1] - targets_downsized[:, :, 1]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks([])
        ax5.set_title("Error Bounded (binary)")

        ax5 = fig.add_subplot(4, 3, 12)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(uncertainty_downsized[:, :, 2] > np.abs(prediction_mean_downsized[:, :, 2] - targets_downsized[:, :, 2]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks([])
        ax5.set_title("Error Bounded (binary)")"""

        ax5 = fig.add_subplot(4, 3, 10)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(np.logical_and(targets_downsized[:, :, 0] > prediction_range_downsized[0][:, :, 0],
                                                                                          targets_downsized[:, :, 0] < prediction_range_downsized[1][:, :, 0]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks(np.arange(30, 180, 30), ['60N', '30N', 'EQ', '30S', '60S'])
        ax5.set_title("j) Ens. Range Bounded (binary)")

        ax5 = fig.add_subplot(4, 3, 11)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(np.logical_and(targets_downsized[:, :, 1] > prediction_range_downsized[0][:, :, 1],
                                                                                            targets_downsized[:, :, 1] < prediction_range_downsized[1][:, :, 1]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks([])
        ax5.set_title("k) Ens. Range Bounded (binary)")

        ax5 = fig.add_subplot(4, 3, 12)
        ax5.imshow(np.where(np.isnan(plot_mask[:, :, 0]), np.nan, np.where(np.logical_and(targets_downsized[:, :, 2] > prediction_range_downsized[0][:, :, 2],
                                                                                        targets_downsized[:, :, 2] < prediction_range_downsized[1][:, :, 2]), 1, 0).astype(float)))
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks([])
        ax5.set_title("l) Ens. Range Bounded (binary)")

        #plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=.17, top=.9, bottom=0.05, left=.21, right=.78)
        plt.savefig('./plots/' + sm_params.dataset + '_' + str(i) + '.png')
        plt.show()
        plt.close('all')


def plot_variation():
    prediction_mean = np.load('./data/npy/' + sm_params.dataset + 'prediction_mean.npz')['arr_0']
    prediction_range = np.load('./data/npy/' + sm_params.dataset + 'prediction_range.npz')['arr_0']
    prediction_std = np.load('./data/npy/' + sm_params.dataset + 'prediction_std.npz')['arr_0']
    uncertainty = np.load('./data/npy/' + sm_params.dataset + 'uncertainty.npz')['arr_0']
    targets = np.load('./data/npy/' + sm_params.dataset + 'targets.npz')['arr_0']
    inputs = np.load('./data/npy/' + sm_params.dataset + 'inputs.npz')['arr_0']
    print(np.shape(uncertainty))
    print(np.shape(train_targets))
    plot_pred = []
    for i in range(len(prediction_mean)):
        index = i
        plot_pred.append(prediction_mean[index])
    # plot the spatial variation of the train targets and the average uncertainty at each location
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, np.nanstd(train_targets[:, :, :, 0], axis=0)*2))
    ax1.set_title("b) Train Target Std")
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, np.nanmean(uncertainty[:, :, :, 0], axis=0)))
    ax2.set_title("c) Prediction Std")
    ax3 = fig.add_subplot(4, 2, 3)
    ax3.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, targets[0, :, :, 0]))
    ax3.set_title("d) Test Target A")
    ax5 = fig.add_subplot(4, 2, 4)
    ax5.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, targets[3, :, :, 0]))
    ax5.set_title("f) Test Target C")
    ax6 = fig.add_subplot(4, 2, 5)
    ax6.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, np.nanmean(uncertainty[0, :, :, :], axis=-1)/np.nanmean(prediction_mean[0, :, :, :] + .01, axis=-1)), vmax=.1)
    ax6.set_title("g) Relative Uncertainty A")
    ax8 = fig.add_subplot(4, 2, 6)
    ax8.imshow(np.where(np.isnan(resized_mask[:, :, 0]), np.nan, np.nanmean(uncertainty[-1, :, :, :], axis=-1)/np.nanmean(prediction_mean[3, :, :, :] + .01, axis=-1)), vmax=.1)
    ax8.set_title("i) Relative Uncertainty C")
    ax9 = fig.add_subplot(4,2,7, projection='scatter_density')
    density = ax9.scatter_density(np.nanstd(train_targets, axis=0).flatten()*2, uncertainty[0].flatten(), vmax=50)
    fig.colorbar(density, ax=ax9)
    ax9.set_title("j) Correlation A")
    ax9.set_xlabel("Train Target Std")
    ax9.set_ylabel("Prediction Std")
    ax11 = fig.add_subplot(4,2,8, projection='scatter_density')
    density = ax11.scatter_density(np.nanstd(train_targets, axis=0).flatten()*2, uncertainty[3].flatten(), vmax=50)
    fig.colorbar(density, ax=ax11, label="Count")
    ax11.set_title("l) Correlation C")
    ax11.set_xlabel("Train Target Std")
    plt.tight_layout()
    traintargets_mask = np.isfinite(np.nanstd(train_targets[:, :, :, 0], axis=0))
    uncertainty_mask = np.isfinite(np.nanmean(uncertainty[:, :, :, 0], axis=0))
    print(np.shape(traintargets_mask))
    print(np.shape(uncertainty_mask))
    train_targets_masked = np.where(np.nanstd(train_targets[:, :, :, 0], axis=0)[traintargets_mask] > .01)
    uncertainty_masked = np.where(np.nanmean(uncertainty[:, :, :, 0], axis=0)[uncertainty_mask] > .01)
    print(np.shape(train_targets_masked))
    print(np.shape(uncertainty_masked))
    #print R and R^2 of each correlation
    #print("R: ", np.corrcoef(train_targets.flatten()[train_targets_masked], uncertainty_masked.flatten()[uncertainty_mask]))
    #print("R^2: ", np.corrcoef(train_targets.flatten()[train_targets_masked], uncertainty_masked.flatten()[uncertainty_mask])**2)
    plt.show()


if __name__ == "__main__":
    data = load_data()
    mask, resized_mask = load_mask()
    tess_data, resized_data = tesselize_data()
    train_data, train_targets, test_data, test_targets, test_times = split_data()
    del(data)
    del(tess_data)
    del(resized_data)
    #target_max = np.nanmax(train_data)
    #target_min = np.nanmin(train_data)

    #model = load_model()
    #train()
    #test()
    #calc_stats()
    plot_results()
    #plot_variation()
