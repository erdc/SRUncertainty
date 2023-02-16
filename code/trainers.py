import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import settings
import datetime
import sys
import os
from tqdm import tqdm


path = os.getcwd()
parent = os.path.join(path, os.pardir)
parent2 = os.path.join(parent, os.pardir)

from plotters import plotter

class trainer:
    def __init__(self, **kwargs):
        super(trainer, self).__init__(**kwargs)   

        self.old_best = 500

    def train_and_evaluate_mf(self, arch, model, dataset, args):
        
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
        
        train_log_dir = './logs/notpadded'
        test_log_dir = './logs/notpadded'
        train_summary_writer = tf.summary.create_file_writer(
            train_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        test_summary_writer = tf.summary.create_file_writer(
            test_log_dir + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        print("Training for " + str(settings.epochs) + " epochs.")
        fail = 0
        loss_list = []
        val_loss_list = []
#         self.old_best = 500
        for i in tqdm(range(settings.epochs)):
            settings.current_epoch = i
            if i % 10 == 0:
                #val_loss = np.n
                avg_val_loss = []
                print(str(i) + "/" + str(settings.epochs))
                for j in range(len(test_case)):
                    input_batch = []
                    target_batch = []
                    for k in range(settings.batch_size):
                        random_sensor = np.random.randint(0, len(settings.sensor_list))
                        target_min = np.nanpercentile(np.where(test_target_features[random_sensor][j] == 0, np.nan, test_target_features[random_sensor][j]), 5, axis=[0, 1])
                        target_max = np.nanpercentile(np.where(test_target_features[random_sensor][j] == 0, np.nan, test_target_features[random_sensor][j]), 95, axis=[0, 1])
                        input_min = np.nanpercentile(np.where(test_input_features[random_sensor][j] == 0, np.nan, test_input_features[random_sensor][j]), 5, axis=[0, 1])
                        input_max = np.nanpercentile(np.where(test_input_features[random_sensor][j] == 0, np.nan, test_input_features[random_sensor][j]), 95, axis=[0, 1])
                        temp_input_features = (test_input_features[random_sensor][j] - input_min)/(input_max - input_min)
                        temp_target_features = (test_target_features[random_sensor][j] - target_min)/(target_max - target_min)
                        temp_wall_features = np.where(np.isnan(wall_test_features[random_sensor][j]), 0, wall_test_features[random_sensor][j])
                        temp_wall_features = np.where(temp_wall_features < 0, 0, temp_wall_features)
                        #temp_input_features = np.pad(temp_input_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                        #temp_target_features = np.pad(temp_target_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                        #temp_wall_features = np.pad(temp_wall_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                        temp_input_batch = np.concatenate((temp_input_features, temp_wall_features), axis=-1)
                        temp_input_batch = np.expand_dims(temp_input_batch, axis=0)
                        temp_target_batch = np.expand_dims(temp_target_features, axis=0)
                        input_batch.append(temp_input_batch)
                        target_batch.append(temp_target_batch)
                    history = model.test_on_batch(np.squeeze(np.asarray(input_batch)), np.squeeze(np.asarray(target_batch)))
                    avg_val_loss.append(history)
                if np.nanmean(np.asarray(avg_val_loss)) < self.old_best:
                    fail = 0
                    print("New best validation loss: ", np.nanmean(np.asarray(avg_val_loss)))
                    val_loss = np.nanmean(np.asarray(avg_val_loss))
                    self.old_best = np.mean(np.asarray(avg_val_loss))
                    model.save('./model_chk/' + arch + '_' + str(settings.sensor_list[-1]) + str(training_case) + str(input_fields) + str(target_fields) + '.h5', overwrite=True)  # + str(i) + '.h5', overwrite=True)
                    plotter().predict_and_plot_mf(arch, model, dataset, args)

                else:
                    fail += 1
                    print("Fail at improving loss with score: ", np.nanmean(np.asarray(avg_val_loss)))
                    val_loss = np.nan
#                     if fail % 10 == 0:
#                         print("val loss failed to improve 10 validations in a row")
#                         print("Current LR: " + str(model.optimizer.lr) + "reducing learning rate by 10%")
#                         tf.keras.backend.set_value(model.optimizer.lr, model.optimizer.lr * .90)
#                         print("New LR: " + str(tf.keras.backend.get_value(model.loss)))
                val_loss_list.append(val_loss)
                with test_summary_writer.as_default():
                    tf.summary.scalar('val_loss', np.mean(val_loss), step=i)
            avg_train_loss = []
            for j in range(len(training_case)):
                input_batch = []
                target_batch = []
                for k in range(settings.batch_size):
                    random_sensor = np.random.randint(0, len(settings.sensor_list))
                    index = np.random.randint(0, len(training_case))
                    target_min = np.nanpercentile(np.where(train_target_features[random_sensor][index] == 0, np.nan, train_target_features[random_sensor][index]), 5, axis=[0, 1])
                    target_max = np.nanpercentile(np.where(train_target_features[random_sensor][index] == 0, np.nan, train_target_features[random_sensor][index]), 95, axis=[0, 1])
                    input_min = np.nanpercentile(np.where(train_input_features[random_sensor][index] == 0, np.nan, train_input_features[random_sensor][index]), 5, axis=[0, 1])
                    input_max = np.nanpercentile(np.where(train_input_features[random_sensor][index] == 0, np.nan, train_input_features[random_sensor][index]), 95, axis=[0, 1])
                    temp_input_features = (train_input_features[random_sensor][index] - input_min) / (input_max - input_min)
                    temp_target_features = (train_target_features[random_sensor][index] - target_min) / (target_max - target_min)
                    temp_wall_features = np.where(np.isnan(wall_train_features[random_sensor][index]), 0, wall_train_features[random_sensor][index])
                    temp_wall_features = np.where(temp_wall_features < 0, 0, temp_wall_features)
                    #temp_input_features = np.pad(temp_input_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                    #temp_target_features = np.pad(temp_target_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                    #temp_wall_features = np.pad(temp_wall_features, ((4, 4), (4, 4), (0, 0)), 'edge')
                    temp_input_batch = np.concatenate((temp_input_features, temp_wall_features), axis=-1)
                    temp_input_batch = np.expand_dims(temp_input_batch, axis=0)
                    temp_target_batch = np.expand_dims(temp_target_features, axis=0)
                    input_batch.append(temp_input_batch)
                    target_batch.append(temp_target_batch)
                history = model.train_on_batch(np.squeeze(np.asarray(input_batch)), np.squeeze(np.asarray(target_batch)))
                avg_train_loss.append(history[0])
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', np.mean(avg_train_loss), step=i)
            loss_list.append(np.sqrt(np.mean(np.square(np.asarray(avg_train_loss)))))
        print('\n' +str(np.mean(np.asarray(loss_list))))