import tensorflow as tf
import numpy as np
import os
from models import GAN
import settings
import glob

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.keras.backend.clear_session()


class network():
    def __init__(self, input_shape, output_shape, arch='unet', gaussian=False, noise=0, avg=False, **kwargs):
        super(network,self).__init__(**kwargs)   
        
        self.arch = arch
        self.inpt_shape = input_shape
        self.outpt_shape = output_shape
        self.gaussian = gaussian
        self.noise = noise
        self.avg = avg
            
    def get(self, args=None, path='None'):
        """
        Loads the ml model data. First attempts to load a file containing saved weights,
        then loads the model and tries to insert the weights into the model. If this fails it uses
        the initialized weights.

        Options for:
        gan
        unet
        """
        try:
            ml_model, rans_model, training_case, input_fields, test_case, target_fields, mode = args
        except:
            print()
            
        if self.arch == 'unet':
            self.model = self.unet(self.inpt_shape, self.outpt_shape, self.gaussian, self.noise, self.avg)
            try:
                saves = glob.glob(path + '/model_chk/unet_' + str(settings.sensor_list[-1]) + str(training_case) + str(input_fields) + str(target_fields) + '*.h5')
                saves.sort(key=settings.natural_keys)
                print(saves[-1])
                self.model.load_weights(saves[-1])
                print("unet checkpoint loaded")
            except:
                print("Didn't find saved unet checkpoint")
                
        elif self.arch == 'gan':   
            self.gen = self.unet(self.inpt_shape, self.outpt_shape, self.gaussian, self.noise, self.avg)
            self.disc = self.discriminator(self.inpt_shape, self.outpt_shape)
            try:
                gsaves = glob.glob(path + '/model_chk/generator_' + str(settings.sensor_list[-1]) + str(training_case) + str(input_fields) + str(target_fields) + '*.h5')
                dsaves = glob.glob(path + '/model_chk/discriminator_' + str(settings.sensor_list[-1]) + str(training_case) + str(input_fields) + str(target_fields) + '*.h5')
                gsaves.sort(key=settings.natural_keys)
                dsaves.sort(key=settings.natural_keys)
                print(gsaves[-1])
                print(dsaves[-1])
                self.gen.load_weights(gsaves[-1])
                self.disc.load_weights(dsaves[-1])
                print("gan checkpoint loaded")
            except:
                print("Didn't find saved gan checkpoint")
            
            self.model = GAN(discriminator=self.disc, generator=self.gen)
               
        
        else:
            print('Please select a valid architecture')

        return self.model

                    
    def unet(self, input_shape, output_shape, gaussian, noise, avg):

        inputs = tf.keras.layers.Input((input_shape))
        conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
        conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
        conv1 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = tf.keras.layers.BatchNormalization(trainable=True)(conv1)
        if gaussian == True:
            conv1 = tf.keras.layers.GaussianNoise(noise)(conv1)
        if avg == True:
            pool1 = tf.keras.layers.AvgPool2D(pool_size=(2, 2))(conv1)
        else:
            pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        
        conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = tf.keras.layers.BatchNormalization(trainable=True)(conv2)
        if gaussian == True:
            conv2 = tf.keras.layers.GaussianNoise(noise)(conv2)
        drop2 = tf.keras.layers.Dropout(0.1)(conv2, training=True)
        if avg == True:
            pool2 = tf.keras.layers.AvgPool2D(pool_size=(2, 2))(drop2)
       
        else:
            pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop2)
        
        conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
        conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
        conv3 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = tf.keras.layers.BatchNormalization(trainable=True)(conv3)
        if gaussian == True:
            conv3 = tf.keras.layers.GaussianNoise(noise)(conv3)
        drop3 = tf.keras.layers.Dropout(0.1)(conv3, training=True)
        if avg == True:
            pool3 = tf.keras.layers.AvgPool2D(pool_size=(2, 2))(drop3)
        else:        
            pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop3)

        conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
        conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
        conv4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = tf.keras.layers.BatchNormalization(trainable=True)(conv4)
        if gaussian == True:
            conv4 = tf.keras.layers.GaussianNoise(noise)(conv4)
        drop4 = tf.keras.layers.Dropout(0.1)(conv4, training=True)
        if avg == True:
            pool4 = tf.keras.layers.AvgPool2D(pool_size=(2, 2))(drop4)
        else:   
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
        if gaussian == True:
            conv6 = tf.keras.layers.GaussianNoise(noise)(conv6)
        drop6 = tf.keras.layers.Dropout(0.1)(conv6, training=True)

        merge7 = tf.keras.layers.concatenate([drop3, drop6], axis=3)
        conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
        conv7 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
        conv7 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                                kernel_initializer='he_normal')((conv7))
        conv7 = tf.keras.layers.BatchNormalization(trainable=True)(conv7)
        if gaussian == True:
            conv7 = tf.keras.layers.GaussianNoise(noise)(conv7)
        drop7 = tf.keras.layers.Dropout(0.1)(conv7, training=True)

        merge8 = tf.keras.layers.concatenate([drop2, drop7], axis=3)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
        conv8 = tf.keras.layers.Conv2DTranspose(128, 3, (2, 2), activation="relu", padding='same',
                                                kernel_initializer='he_normal')((conv8))
        conv8 = tf.keras.layers.BatchNormalization(trainable=True)(conv8)
        if gaussian == True:
            conv8 = tf.keras.layers.GaussianNoise(noise)(conv8)
        drop8 = tf.keras.layers.Dropout(0.1)(conv8, training=True)

        merge9 = tf.keras.layers.concatenate([conv1, drop8], axis=3)
        conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_normal')(conv9)
        #conv9 = tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = tf.keras.layers.Conv2D(output_shape[-1], 1, activation=None)(conv9)

        model = tf.keras.models.Model(inputs=inputs, outputs=[conv10])

        return model

    def discriminator(self, input_shape, output_shape):

        inputs = tf.keras.layers.Input(input_shape)
        targets = tf.keras.layers.Input((output_shape))

        x = tf.keras.layers.concatenate([inputs, targets])

        d1 = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        d1 = tf.keras.layers.BatchNormalization()(d1)
        d1 = tf.keras.layers.LeakyReLU()(d1)
        d2 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(d1)
        d2 = tf.keras.layers.BatchNormalization()(d2)
        d2 = tf.keras.layers.LeakyReLU()(d2)
        d3 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(d2)
        d3 = tf.keras.layers.BatchNormalization()(d3)
        d3 = tf.keras.layers.LeakyReLU()(d3)
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(d3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(output_shape[-1], 4, strides=1)(zero_pad2)  # (batch_size, 30, 30, 1)
  
        model = tf.keras.Model(inputs=[inputs, targets], outputs=last)
    
        return model
    
