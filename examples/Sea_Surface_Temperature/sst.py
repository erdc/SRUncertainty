import sys
sys.path.insert(0, '/media/amcoast/Work_SSD/Superresolution')
import os
import glob
import matplotlib as mpl
import tensorflow as tf
import sst_params
import netCDF4 as nc
import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import NearestNDInterpolator
import cv2
from scipy.stats import norm
import statistics
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.keras.backend.clear_session()


def load_unet():

    inputs = tf.keras.layers.Input((sst_params.input_rows, sst_params.input_cols, sst_params.input_features))
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
    conv1 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
    conv2 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv2)
    drop2 = tf.keras.layers.Dropout(0.1)(conv2, training=True)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
    conv3 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv3)
    drop3 = tf.keras.layers.Dropout(0.1)(conv3, training=True)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
    conv4 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv4)
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
    conv6 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv6)
    drop6 = tf.keras.layers.Dropout(0.1)(conv6, training=True)

    merge7 = tf.keras.layers.concatenate([drop3, drop6], axis=3)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                            kernel_initializer='he_normal')((conv7))
    conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
    conv7 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv7)
    drop7 = tf.keras.layers.Dropout(0.1)(conv7, training=True)

    merge8 = tf.keras.layers.concatenate([drop2, drop7], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                            kernel_initializer='he_normal')((conv8))
    conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
    conv8 = tf.keras.layers.GaussianNoise(sst_params.noise_std)(conv8)
    drop8 = tf.keras.layers.Dropout(0.1)(conv8, training=True)

    merge9 = tf.keras.layers.concatenate([conv1, drop8], axis=3)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.Conv2D(sst_params.target_features, 1, activation=None)(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=[conv10])
    optimizer = tf.keras.optimizers.Nadam()

    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mae', 'mse'])

    return model


def load_model():
    if sst_params.dataset == 'icec.wkmean':
        try:
            saves = glob.glob('./model_chk/icec.wkmean.h5')
            model = load_unet()
            model.load_weights(saves[-1])
            print("unet checkpoint loaded")
            print(saves[-1])
        except:
            print("Didn't find saved unet checkpoint")
            model = load_unet()
        return model
    if sst_params.dataset == 'sst.wkmean':
        try:
            saves = glob.glob('./model_chk/sst.wkmean.h5')
            model = load_unet()
            model.load_weights(saves[-1])
            print("unet checkpoint loaded")
            print(saves[-1])
        except:
            print("Didn't find saved sst checkpoint")
            model = load_unet()
        return model

def load_mask():
    # land-sea mask
    mask = nc.Dataset('./data/noaa/lsmask.nc')
    mask = mask.variables['mask'][:][0]
    mask = mask.astype(float)
    mask[mask == 0] = np.nan
    resized_mask = cv2.resize(mask, (sst_params.input_cols, sst_params.input_rows))
    resized_mask = np.expand_dims(resized_mask, axis=-1)
    return mask, resized_mask


def load_data(mask):
    try:
        input_data = np.load('./data/noaa/' + sst_params.dataset + '.npz')['arr_0']
        lat = np.load('./data/noaa/' + sst_params.dataset + '_lat.npz')['arr_0']
        lon = np.load('./data/noaa/' + sst_params.dataset + '_lon.npz')['arr_0']
        time = np.load('./data/noaa/' + sst_params.dataset + '_time.npz')['arr_0']
        time_bnds = np.load('./data/noaa/' + sst_params.dataset + '_time_bnds.npz')['arr_0']
    except:
        if sst_params.dataset == 'icec.wkmean':
            fn = './data/noaa/icec.wkmean.1990-present.nc'
            data = nc.Dataset(fn)
            icec = data.variables['icec'][:]
            for i in range(icec.shape[0]):
                icec[i] = np.where(np.isnan(mask), np.nan, icec[i])
            input_data = icec
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            time = data.variables['time'][:]
            time_bnds = data.variables['time_bnds'][:]
            data.close()
        if sst_params.dataset == 'sst_params.wkmean':
            fn = './data/noaa/sst.wkmean.1990-present.nc'
            data = nc.Dataset(fn)
            sst = data.variables['sst'][:]
            for i in range(sst_params.shape[0]):
                sst[i] = np.where(np.isnan(mask), np.nan, sst[i])
            input_data = sst
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            time = data.variables['time'][:]
            time_bnds = data.variables['time_bnds'][:]
            data.close()

        np.savez_compressed('./data/noaa/' + sst_params.dataset + '.npz', input_data)
        np.savez_compressed('./data/noaa/' + sst_params.dataset + '_lat.npz', lat)
        np.savez_compressed('./data/noaa/' + sst_params.dataset + '_lon.npz', lon)
        np.savez_compressed('./data/noaa/' + sst_params.dataset + '_time.npz', time)
        np.savez_compressed('./data/noaa/' + sst_params.dataset + '_time_bnds.npz', time_bnds)

    return input_data, lat, lon, time, time_bnds


def tesselize_data(data):
    try:
        tess_input_data = np.load('./data/noaa/' + sst_params.dataset + '_tess.npz')['arr_0']
    except:
        tesselized_feature_list = []
        for sensor_no in sst_params.sensor_list:
            tess_features = np.zeros(np.shape(data))
            x = np.linspace(0, 360, 360)
            y = np.linspace(0, 180, 180)
            X, Y = np.meshgrid(x, y, indexing='ij')
            centroids_x = np.linspace(0, 360, int(np.sqrt(sensor_no)))
            centroids_y = np.linspace(0, 180, int(np.sqrt(sensor_no)))
            centroids_X, centroids_Y = np.meshgrid(centroids_x, centroids_y, indexing='ij')
            centroids = np.vstack((centroids_X.flatten(), centroids_Y.flatten())).T

            for i in range(data.shape[0]):
                tess_interp = NearestNDInterpolator(centroids, data[i, centroids[:, 1].astype(int)-1, centroids[:, 0].astype(int)-1])
                tess_interp = tess_interp(X, Y)
                tess_features[i] = tess_interp.T

            tesselized_feature_list.append(tess_features)

        tesselized_features = np.asarray(tesselized_feature_list)
        np.savez_compressed('./data/noaa/' + sst_params.dataset + '_tess.npz', tesselized_features)
        tess_input_data = tesselized_features
    return tess_input_data


def split_data():
    print("Splitting data")
    train_targets = np.zeros(tess_data[:, :-302].shape)
    test_targets = np.zeros(tess_data[:, -252::15].shape)
    test_times = time[-252::15]
    for sensor_no in range(len(train_targets)):
        train_targets[sensor_no] = data[:-302]
    for sensor_no in range(len(test_targets)):
        test_targets[sensor_no] = data[-252::15]
    train_data = np.reshape(tess_data[:, :-302], (
    (tess_data[:, :-302].shape[0] * tess_data[:, :-302].shape[1]), tess_data[:, :-302].shape[2],
    tess_data[:, :-302].shape[3]))
    test_data = np.reshape(tess_data[:, -252::15], (
    (tess_data[:, -252::15].shape[0] * tess_data[:, -252::15].shape[1]), tess_data[:, -252::15].shape[2],
    tess_data[:, -252::15].shape[3]))

    train_targets = np.reshape(train_targets, (
    (train_targets.shape[0] * train_targets.shape[1]), train_targets.shape[2], train_targets.shape[3]))
    test_targets = np.reshape(test_targets, (
    (test_targets.shape[0] * test_targets.shape[1]), test_targets.shape[2], test_targets.shape[3]))

    print("Train data shape: ", train_data.shape)
    print("Train targets shape: ", train_targets.shape)
    print("Test data shape: ", test_data.shape)
    print("Test targets shape: ", test_targets.shape)
    return train_data, train_targets, test_data, test_targets, test_times


def get_batch(time, input, target):
    temp_input_batch = (input[time, :, :] - target_min) / (target_max - target_min)
    temp_target_batch = (target[time, :, :] - target_min) / (target_max - target_min)
    temp_input_batch = cv2.resize(temp_input_batch, (sst_params.input_cols, sst_params.input_rows))
    temp_target_batch = cv2.resize(temp_target_batch, (sst_params.input_cols, sst_params.input_rows))
    temp_input_batch = np.expand_dims(temp_input_batch, axis=-1)
    temp_input_batch = np.concatenate((temp_input_batch, resized_mask), axis=-1)
    temp_input_batch = np.where(np.isnan(temp_input_batch), -1, temp_input_batch)
    temp_target_batch = np.where(np.isnan(temp_target_batch), -1, temp_target_batch)
    return temp_input_batch, temp_target_batch


def train():
    print("Training for " + str(sst_params.epochs) + " epochs")
    old_best = 1000
    train_log_dir = './logs/train'
    test_log_dir = './logs/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    test_summary_writer = tf.summary.create_file_writer(test_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    X_train, X_val, y_train, y_val = train_test_split(train_data[::], train_targets[::], test_size=0.1, random_state=42)

    for epoch in range(sst_params.epochs):
        avg_train_loss = []
        if epoch % 1 == 0:
            avg_val_loss = []
            print(str(epoch) + "/" + str(sst_params.epochs))
            for j in range(len(X_val)):
                input_batch = []
                target_batch = []
                for k in range(sst_params.batch_size):
                    time = np.random.randint(0, len(X_val))
                    temp_input_batch, temp_target_batch = get_batch(time, X_val, y_val)
                    input_batch.append(temp_input_batch)
                    target_batch.append(temp_target_batch)
                target_batch = np.expand_dims(target_batch, axis=-1)
                history = model.test_on_batch(np.asarray(input_batch), target_batch)
                avg_val_loss.append(history[0])
            if np.nanmean(np.asarray(avg_val_loss)) < old_best:
                print("New best validation loss: ", np.nanmean(np.asarray(avg_val_loss)))
                old_best = np.mean(np.asarray(avg_val_loss))
                val_loss = history[0]
                if sst_params.dataset == 'sst.wkmean':
                    model.save('./model_chk/sst_params.wkmean.h5', overwrite=True)
                    test()
                    plot_results()
            else:
                print("Fail at improving loss with score: ", np.nanmean(np.asarray(avg_val_loss)))
                val_loss = np.nanmean(np.asarray(avg_val_loss))
                print("Current LR: " + str(model.optimizer.lr) + "reducing learning rate by 5%")
                tf.keras.backend.set_value(model.optimizer.lr, model.optimizer.lr * .95)
                print("New LR: " + str(tf.keras.backend.get_value(model.loss)))
            with test_summary_writer.as_default():
                tf.summary.scalar('val_loss', np.mean(val_loss), step=epoch)
        for j in tqdm(range(int(len(X_train)/sst_params.batch_size/3))):
            input_batch = []
            target_batch = []
            for k in range(sst_params.batch_size):
                time = np.random.randint(0, len(X_train))
                temp_input_batch, temp_target_batch = get_batch(time, X_train, y_train)
                input_batch.append(temp_input_batch)
                target_batch.append(temp_target_batch)
            input_batch = np.array(input_batch)
            target_batch = np.array(target_batch)
            target_batch = np.expand_dims(target_batch, axis=-1)
            history = model.train_on_batch(input_batch, target_batch)
            avg_train_loss.append(history[0])
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', np.mean(avg_train_loss), step=epoch)
    return model


def test():
    print("Testing")

    X_test, y_test = test_data, test_targets
    print("Test data shape: ", X_test.shape)
    print("Test targets shape: ", y_test.shape)
    ensemble_prediction_list = []
    ensemble_target_list = []
    ensemble_input_list = []

    for i in tqdm(range(int(len(X_test[-int(len(X_test)/len(sst_params.sensor_list))::5])))):
        prediction_list = []
        target_list = []
        input_list = []
        for j in range(sst_params.ensembles):
            input_batch = []
            target_batch = []
            for k in range(sst_params.batch_size):
                if k == 0:
                    time = i
                    temp_input_batch, temp_target_batch = get_batch(time, X_test[-int(len(X_test)/len(sst_params.sensor_list)):], y_test[-int(len(X_test)/len(sst_params.sensor_list)):])
                else:
                    time = np.random.randint(0, len(X_test))
                    temp_input_batch, temp_target_batch = get_batch(time, X_test, y_test)
                input_batch.append(temp_input_batch)
                target_batch.append(temp_target_batch)
            prediction = model.predict_on_batch(np.asarray(input_batch))
            prediction = prediction[0]
            prediction = np.where(np.isnan(resized_mask), np.nan, prediction)
            target = target_batch[0]
            target = np.where(np.isnan(resized_mask[:, :, 0]), np.nan, target)
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

        np.save('./data/npy/sstprediction_list' + str(sst_params.ensembles) + '_' + str(i) + '.npy', prediction_list)
        np.save('./data/npy/ssttarget_list' + str(sst_params.ensembles) + '_' + str(i) + '.npy', target_list)
        np.save('./data/npy/sstinput_list' + str(sst_params.ensembles) + '_' + str(i) + '.npy', input_list)

        """fig = plt.figure(figsize=(20, 10))
        ax5 = fig.add_subplot(111)
        fig.colorbar(ax5.pcolormesh(prediction_list[0][:, :, 0], cmap='jet'), cmap='jet', label='Temperature (norm)')
        ax5.scatter([50, 150, 300], [25, 150, 150], color='black')
        ax5.invert_yaxis()
        plt.show()

        # plot all the predictions in the ensemble for a few different points in four subplots
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(421)
        ax1.hist(prediction_list[:, 25, 50, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 25, 50, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax1.plot(x, p, 'k', linewidth=2)
        ax1.axvline(x=target_list[0, 25, 50], color='r')
        ax1.set_ylabel('Count')
        # add error to 3 digits
        ax1.set_title("a) 25, 50 | Error: " + str(round(prediction_list[:, 25, 50, 0].mean() - target_list[0, 25, 50], 3)))
        ax2 = fig.add_subplot(422)
        ax2.hist(prediction_list[:, 100, 100, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 100, 100, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax2.plot(x, p, 'k', linewidth=2)
        ax2.axvline(x=target_list[0, 100, 100], color='r')
        ax2.set_title("b) 100, 100 | Error: " + str(round(prediction_list[:, 100, 100, 0].mean() - target_list[0, 100, 100], 3)))
        ax3 = fig.add_subplot(423)
        ax3.hist(prediction_list[:, 150, 150, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 150, 150, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax3.plot(x, p, 'k', linewidth=2)
        ax3.axvline(x=target_list[0, 150, 150], color='r')
        ax3.set_ylabel('Count')
        ax3.set_title("c) 150, 150 | Error: " + str(round(prediction_list[:, 150, 150, 0].mean() - target_list[0, 150, 150], 3)))
        ax4 = fig.add_subplot(424)
        ax4.hist(prediction_list[:, 200, 200, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 200, 200, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax4.plot(x, p, 'k', linewidth=2)
        ax4.axvline(x=target_list[0, 200, 200], color='r')
        ax4.set_title("d) 200, 200 | Error: " + str(round(prediction_list[:, 200, 200, 0].mean() - target_list[0, 200, 200], 3)))
        ax5 = fig.add_subplot(425)
        ax5.hist(prediction_list[:, 150, 250, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 150, 250, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax5.plot(x, p, 'k', linewidth=2)
        ax5.axvline(x=target_list[0, 150, 250], color='r')
        ax5.set_ylabel('Count')
        ax5.set_title("e) 150, 250 | Error: " + str(round(prediction_list[:, 150, 250, 0].mean() - target_list[0, 150, 250], 3)))
        ax6 = fig.add_subplot(426)
        ax6.hist(prediction_list[:, 150, 300, 0], bins=10, edgecolor='black', facecolor='green')
        mu, std = norm.fit(prediction_list[:, 150, 300, 0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax6.plot(x, p, 'k', linewidth=2)
        ax6.axvline(x=target_list[0, 150, 300], color='r')
        ax6.set_title("f) 150, 300 | Error: " + str(round(prediction_list[:, 150, 300, 0].mean() - target_list[0, 150, 300], 3)))
        ax5 = fig.add_subplot(427)
        fig.colorbar(ax5.pcolormesh(prediction_list[0][:, :, 0], cmap='jet'), cmap='jet')
        ax5.scatter([50, 150, 300], [25, 150, 150], color='red')
        ax5.invert_yaxis()
        ax5.set_title("g) Predictions")
        #ax5.set_xticks(np.arange(int(60*(512/360)), int(350*(512/360)), int(60*(512/360))), ['60E', '120E', '180', '120W', '60W'])
        #ax5.set_yticks(np.arange(int(30*(256/180)), int(180*(256/180)), int(30*(256/180))), ['60N', '30N', 'EQ', '30S', '60S'])
        ax6 = fig.add_subplot(428)
        fig.colorbar(ax6.pcolormesh(np.abs(prediction_list[0][:, :, 0] - target_list[0][:, :]), vmax=.1))
        ax6.scatter([50, 100, 150, 200, 250, 300], [25, 100, 150, 200, 150, 150], color='red')
        ax6.invert_yaxis()
        ax6.set_title("h) Absolute Error")
        #ax6.set_xticks(np.arange(60, 360, 60), ['60E', '120E', '180', '120W', '60W'])
        plt.show()
        plt.subplots_adjust(hspace=.38, right=.56)
        plt.savefig('./plots/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' + str(i) + '.png')"""

    ensemble_prediction_list = np.asarray(ensemble_prediction_list)
    ensemble_target_list = np.asarray(ensemble_target_list)
    ensemble_input_list = np.asarray(ensemble_input_list)

    prediction_mean = np.nanmean(ensemble_prediction_list[:, :, :, :, 0], axis=1)
    prediction_std = np.nanstd(ensemble_prediction_list[:, :, :, :, 0], axis=1)
    # calculate range of predictions
    prediction_range = [ensemble_prediction_list[:, :, :, :, 0].min(axis=1), ensemble_prediction_list[:, :, :, :, 0].max(axis=1)]
    uncertainty = prediction_std * 2
    targets = np.nanmean(ensemble_target_list, axis=1)
    inputs = np.nanmean(ensemble_input_list[:, :, :, :, 0], axis=1)

    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_mean.npz', prediction_mean)
    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_std.npz', prediction_std)
    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_range.npz', prediction_range)
    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'uncertainty.npz', uncertainty)
    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'targets.npz', targets)
    np.savez_compressed('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'inputs.npz', inputs)

    # create a histogram plot that shows the distribution of a histogram of the predictions
    histogram_list = []
    error_hist_list = []

    """for i in tqdm(range(len(ensemble_prediction_list))):
        for j in range(len(ensemble_prediction_list[0][0])):
            for k in range(len(ensemble_prediction_list[0][0][0])):
                if not np.isnan(ensemble_prediction_list[0][i][j][k][0]):
                    values, bins = np.histogram(ensemble_prediction_list[i, :, j, k, 0], bins=10)
                    # plot histogram
                    fig = plt.figure(figsize=(20, 20))
                    print("values: ", values)
                    print("bins: ", bins)
                    ax1 = fig.add_subplot(211)
                    ax1.hist(ensemble_prediction_list[i, :, j, k, 0], bins=10)
                    ax1.set_title("prediction histogram")
                    ax2 = fig.add_subplot(212)
                    ax2.pcolormesh(ensemble_prediction_list[i, :, j, k, 0].reshape(10, 10))
                    ax2.set_title("prediction")
                    plt.show()
                    histogram_list.append(values)
                    error_hist_list.append(np.abs(prediction_mean[i, j, k] - targets[i, j, k]))
"""
    np.save("sst_histogram_list.npy", histogram_list)
    np.save("sst_error_hist_list.npy", error_hist_list)


def calc_stats():
    prediction_mean = np.load('./data/npy/' + sst_params.dataset + 'prediction_mean.npz')['arr_0']
    prediction_std = np.load('./data/npy/' + sst_params.dataset + 'prediction_std.npz')['arr_0']
    uncertainty = np.load('./data/npy/' + sst_params.dataset + 'uncertainty.npz')['arr_0']
    targets = np.load('./data/npy/' + sst_params.dataset + 'targets.npz')['arr_0']
    inputs = np.load('./data/npy/' + sst_params.dataset + 'inputs.npz')['arr_0']

    # calculate RMSE, MAE, and bias of prediction list
    rmse = np.sqrt(np.nanmean((prediction_mean - targets)**2))
    mae = np.nanmean(np.abs(prediction_mean - targets))
    bias = np.nanmean(prediction_mean - targets)
    print("RMSE: " + str(rmse))
    print("MAE: " + str(mae))
    print("Bias: " + str(bias))


def plot_results():
    prediction_mean = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_mean.npz')['arr_0']
    prediction_std = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_std.npz')['arr_0']
    prediction_range = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'prediction_range.npz')['arr_0']
    uncertainty = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'uncertainty.npz')['arr_0']
    targets = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'targets.npz')['arr_0']
    inputs = np.load('./data/npy/' + sst_params.dataset + '_' + str(sst_params.ensembles) + '_' +'inputs.npz')['arr_0']

    for i in range(len(prediction_mean)):
        print(i)
        prediction_mean_downsized = cv2.resize(prediction_mean[i], (360, 180))
        prediction_std_downsized = cv2.resize(prediction_std[i], (360, 180))
        prediction_range_downsized = [cv2.resize(prediction_range[0][i], (360, 180)), cv2.resize(prediction_range[1][i], (360, 180))]
        uncertainty_downsized = cv2.resize(uncertainty[i], (360, 180))
        targets_downsized = cv2.resize(targets[i], (360, 180))
        inputs_downsized = cv2.resize(inputs[i], (360, 180))

        # plot example predictions
        norm = mpl.cm.colors.Normalize(vmax=np.nanmax(targets_downsized), vmin=0)
        errnorm = mpl.cm.colors.Normalize(vmax=.1, vmin=0)

        fig, axs = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        ax0 = axs[0,0]
        ax0.pcolormesh(lon, lat, inputs_downsized, vmin=0, vmax=1, cmap='jet')
        ax0.set_xticks([])
        ax0.set_yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N'])
        ax0.set_title("a) Model Input (norm)")

        ax1 = axs[0,1]
        ax1.contourf(lon, lat, prediction_mean_downsized * mask, levels=20, linewidths=1, vmin=0, vmax=1, cmap='jet')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("b) Model Prediction (norm)")

        ax2 = axs[0,2]
        ax2.contourf(lon, lat, targets_downsized * mask, levels=20, linewidths=1, vmin=0, vmax=1, cmap='jet')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("c) Model Target (norm)")

        ## Place the colorbar below the first row
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=axs[0, :], aspect=45, shrink=0.7,
                            orientation='horizontal')
        cbar.set_label('Sea-Surface Temperature (norm)')

        errnorm = mpl.cm.colors.Normalize(vmax=.2, vmin=0)
        ax3 = axs[1,0]
        ax3.pcolormesh(lon, lat, (np.abs(prediction_mean_downsized - targets_downsized)/np.abs(prediction_mean_downsized+0.001)) * mask, cmap='jet', norm=errnorm)
        ax3.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax3.set_yticks(np.arange(-60, 90, 30), ['60S', '30S', 'EQ', '30N', '60N'])
        ax3.set_title("d) Relative Error")

        ax4 = axs[1,1]
        ax4.pcolormesh(lon, lat, uncertainty_downsized/np.abs(prediction_mean_downsized+0.001) * mask, cmap='jet', norm=errnorm)
        ax4.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax4.set_yticks([])
        ax4.set_title("e) Relative Uncertainty")

        ax5 = axs[1,2]
        ax5.pcolormesh(lon, lat, np.where((prediction_range_downsized[0] < targets_downsized)
                                          & (targets_downsized < prediction_range_downsized[1]), 1, 0) * mask, cmap='jet')
        ax5.set_xticks(np.arange(60, 350, 60), ['60E', '120E', '180', '120W', '60W'])
        ax5.set_yticks([])
        ax5.set_title("f) Ens. Range Bounded (binary)")

        # print total percentage of pixels that have uncertainty over prediction error
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=errnorm, cmap='jet'), ax=axs[1, :2], aspect=45, shrink=0.7,
                            orientation='horizontal')
        cbar.set_label('Error')

        plt.subplots_adjust(bottom=.195, right=.9, top=.882, hspace=.399)
        plt.savefig('./plots/' + sst_params.dataset + '_' + str(i) + '.png')
        #plt.show()
        plt.close('all')
        #plt.show()


def plot_summary():
    print('plotting summary')
    histogram_list = np.load("sst_histogram_list.npy", allow_pickle=True)
    error_hist_list = np.load("sst_error_hist_list.npy", allow_pickle=True)
    histogram_list = np.asarray(histogram_list)
    error_hist_list = np.asarray(error_hist_list)

    error_hist_list = np.abs(error_hist_list)
    where_0to25 = np.where((error_hist_list >= 0) & (error_hist_list < 0.025))
    where_25to50 = np.where((error_hist_list >= 0.025) & (error_hist_list < 0.05))
    where_50to75 = np.where((error_hist_list >= 0.05) & (error_hist_list < 0.075))
    where_75to100 = np.where((error_hist_list >= 0.1) & (error_hist_list < 1))
    print(np.percentile(error_hist_list, 99))
    print(np.percentile(error_hist_list, 95))
    where_top10 = np.where(error_hist_list >= np.percentile(error_hist_list, 99.9))

    hist_0to25 = histogram_list[where_0to25]
    hist_25to50 = histogram_list[where_25to50]
    hist_50to75 = histogram_list[where_50to75]
    hist_75to100 = histogram_list[where_75to100]
    hist_top10 = histogram_list[where_top10]

    hist_sum_0to25 = np.sum(hist_0to25, axis=0)
    hist_sum_25to50 = np.sum(hist_25to50, axis=0)
    hist_sum_50to75 = np.sum(hist_50to75, axis=0)
    hist_sum_75to100 = np.sum(hist_75to100, axis=0)
    hist_sum_top10 = np.sum(hist_top10, axis=0)

    # create figure plotting all the histogram of frequency with a subplot for different ranges of error
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Histograms of Uncertainty and Error', fontsize=16)
    ax1 = fig.add_subplot(421)
    ax1.bar(np.arange(0, 10), hist_sum_0to25/np.sum(hist_sum_0to25), color='blue')
    ax1.set_ylim([0, 0.5])
    ax2 = fig.add_subplot(422)
    ax2.bar(np.arange(0, 10), hist_sum_25to50/np.sum(hist_sum_25to50), color='blue')
    ax2.set_ylim([0, 0.5])
    ax3 = fig.add_subplot(423)
    ax3.bar(np.arange(0, 10), hist_sum_50to75/np.sum(hist_sum_50to75), color='blue')
    ax3.set_ylim([0, 0.5])
    ax4 = fig.add_subplot(424)
    ax4.bar(np.arange(0, 10), hist_sum_75to100/np.sum(hist_sum_75to100), color='blue')
    ax4.set_ylim([0, 0.5])
    ax5 = fig.add_subplot(425)
    ax5.bar(np.arange(0, 10), hist_sum_top10/np.sum(hist_sum_top10), color='blue')
    ax5.set_ylim([0, 0.5])
    ax1.set_title("0.0 - 0.025")
    ax2.set_title("0.025 - 0.05")
    ax3.set_title("0.05 - 0.075")
    ax4.set_title("0.075 - 0.1")
    ax5.set_title("top 5%tile")
    plt.show()

    # create a histogram of histograms showing how often each count shows up for each bin
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle('Count of Prediction Distributions', fontsize=16)
    histogram_2da = np.zeros((60, len(histogram_list[0])))
    histogram_2db = np.zeros((60, len(histogram_list[0])))
    histogram_2dc = np.zeros((60, len(histogram_list[0])))
    histogram_2dd = np.zeros((60, len(histogram_list[0])))
    ax2 = fig.add_subplot(321)
    for i in range(len(histogram_2da)):
        for j in range(len(histogram_list[i])):
            histogram_2da[hist_0to25[i][j], j] += hist_0to25[i][j]
    ax2.imshow(histogram_2da, cmap='jet', origin='lower', extent=[0, 10, 0, 60], aspect=1/6)
    ax2.set_xlabel('Prediction Bin')
    ax2.set_ylabel('Count')
    ax2.set_title("0.0 - 0.025")
    ax3 = fig.add_subplot(322)
    for i in range(len(histogram_2db)):
        for j in range(len(histogram_list[i])):
            histogram_2db[hist_25to50[i][j], j] += hist_25to50[i][j]
    ax3.imshow(histogram_2db, cmap='jet', origin='lower', extent=[0, 10, 0, 60], aspect=1/6)
    ax3.set_xlabel('Prediction Bin')
    ax3.set_ylabel('Count')
    ax3.set_title("0.025 - 0.05")
    ax4 = fig.add_subplot(323)
    for i in range(len(histogram_2dc)):
        for j in range(len(histogram_list[i])):
            histogram_2dc[hist_50to75[i][j], j] += hist_50to75[i][j]
    ax4.imshow(histogram_2dc, cmap='jet', origin='lower', extent=[0, 10, 0, 60], aspect=1/6)
    ax4.set_xlabel('Prediction Bin')
    ax4.set_ylabel('Count')
    ax4.set_title("0.05 - 0.075")
    ax5 = fig.add_subplot(324)
    for i in range(len(histogram_2dd)):
        for j in range(len(histogram_list[i])):
            histogram_2dd[hist_75to100[i][j], j] += hist_75to100[i][j]
    ax5.imshow(histogram_2dd, cmap='jet', origin='lower', extent=[0, 10, 0, 60], aspect=1/6)
    ax5.set_xlabel('Prediction Bin')
    ax5.set_ylabel('Count')
    ax5.set_title("0.075 - 0.1")
    for i in range(len(histogram_2dd)):
        for j in range(len(histogram_list[i])):
            histogram_2dd[hist_top10[i][j], j] += hist_top10[i][j]
    ax6 = fig.add_subplot(325)
    ax6.imshow(histogram_2dd, cmap='jet', origin='lower', extent=[0, 10, 0, 60], aspect=1/6)
    ax6.set_xlabel('Prediction Bin')
    ax6.set_ylabel('Count')
    ax6.set_title("top 5%tile")
    plt.show()


if __name__ == "__main__":
    mask, resized_mask = load_mask()
    data, lat, lon, time, time_bnds = load_data(mask)
    tess_data = tesselize_data(data)
    train_data, train_targets, test_data, test_targets, test_times = split_data()
    target_max = np.nanmax(train_data)
    target_min = np.nanmin(train_data)

    model = load_model()
    #train()
    test()
    calc_stats()
    plot_results()
    plot_summary()

