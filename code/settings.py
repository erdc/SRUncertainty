import re
import tensorflow as tf
import numpy as np
import cv2
tf.compat.v1.enable_eager_execution()
# options you can adjust here
input_rows = 512  # set here
input_cols = 768  # set here
target_rows = 512  # set here
target_cols = 768 
channels = 3
learning_rate = 0.00001
jh_input_rows = 256  # set here
jh_input_cols = 256  # set here
epochs = 5000 # set here
ensembles = 500 # set here
noise_std = 0.0 # set here
data_noise_std = 0.0  # set here
batch_size = 4  # set here
sensor_list = [49, 64, 81, 100, 121, 144, 169, 196, 225]  # set here

# global init
input_features = 0  # set  from args
target_features = 0  # set from args
current_epoch = 0  # init for saving plots during training

@tf.autograph.experimental.do_not_convert
def error_grad_cont(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    square_error = tf.math.square(error)

    dz, dy, dx = tf.image.image_gradients(y_pred[:, :, :, :3])

    u_x = dx[:, :, :, 0]
    v_y = dy[:, :, :, 1]
    w_z = dz[:, :, :, 2]

    cont_loss = tf.math.abs(u_x) + tf.math.abs(v_y) + tf.math.abs(w_z)
    cont_loss = tf.expand_dims(cont_loss, axis=-1)
    full_error = tf.math.add(abs_error, cont_loss)

    return full_error
    loss.__name__ = "error_grad_cont"


# from stack overflow for sorting numerics
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]
