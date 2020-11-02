import numpy as np
import os
import argparse
import json
import h5py
import csv
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, Concatenate, Dropout, UpSampling2D, add
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam

os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train_deepvel(object):

    def __init__(self, root, noise, option, norm_filename='network/simulation_properties.npz'):
        """
        Class used to train DeepVel

        Parameters
        ----------
        root : string
            Name of the output files. Some extensions will be added for different files (weights, configuration, etc.)
        noise : float
            Noise standard deviation to be added during training. This helps avoid overfitting and
            makes the training more robust
        option : string
            Indicates what needs to be done
        """

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.root = root
        self.option = option

        # -----------------
        # Data properties:
        # -----------------
        # Patch dimensions
        self.nx = 48
        self.ny = 48
        # Number of types of inputs (intensitygrams, magnetograms, Dopplergrams)
        self.n_types = 1
        # Number of consecutive frames of a given input
        self.n_times = 2
        # Number of inputs
        self.n_inputs = self.n_types * self.n_times
        # Number of inferred velocity components
        self.n_depths = 1
        self.n_components = 2
        self.n_outputs = self.n_depths*self.n_components

        # -----------------
        # Network properties:
        # -----------------
        # Architecture
        self.n_filters = 64
        self.kernel_size = 3
        self.noise_level = noise
        # Number of batches
        self.batch_size = 10
        self.n_training = 3600
        self.n_validation = 900
        self.batches_per_epoch_training = int(self.n_training / self.batch_size)
        self.batches_per_epoch_validation = int(self.n_validation / self.batch_size)
        self.n_training = self.batches_per_epoch_training*self.batch_size
        self.n_validation = self.batches_per_epoch_validation*self.batch_size

        # --------------------
        # Training & test sets:
        # --------------------
        # Training
        self.input_files_training = 'training_validation_sets/input_training.npz'
        self.input_keys_training = 'input_train'
        self.output_files_training = 'training_validation_sets/output_training.npz'
        self.output_keys_training = 'output_train'
        # Validation
        self.input_files_validation = 'training_validation_sets/input_validation.npz'
        self.input_keys_validation = 'input_valid'
        self.output_files_validation = 'training_validation_sets/output_validation.npz'
        self.output_keys_validation = 'output_valid'

        # Normalization
        tmp = np.load(norm_filename)
        self.ic_tau_1_min = tmp['ic_tau_1_min']
        self.ic_tau_1_max = tmp['ic_tau_1_max']
        self.ic_tau_1_mean = tmp['ic_tau_1_mean']
        self.ic_tau_1_var = tmp['ic_tau_1_var']
        self.ic_tau_1_median = tmp['ic_tau_1_median']
        self.ic_tau_1_stddev = tmp['ic_tau_1_stddev']

        self.vx_tau_1_min = tmp['vx_tau_1_min']
        self.vx_tau_1_max = tmp['vx_tau_1_max']
        self.vx_tau_1_mean = tmp['vx_tau_1_mean']
        self.vx_tau_1_var = tmp['vx_tau_1_var']
        self.vx_tau_1_median = tmp['vx_tau_1_median']
        self.vx_tau_1_stddev = tmp['vx_tau_1_stddev']
        self.vy_tau_1_min = tmp['vy_tau_1_min']
        self.vy_tau_1_max = tmp['vy_tau_1_max']
        self.vy_tau_1_mean = tmp['vy_tau_1_mean']
        self.vy_tau_1_var = tmp['vy_tau_1_var']
        self.vy_tau_1_median = tmp['vy_tau_1_median']
        self.vy_tau_1_stddev = tmp['vy_tau_1_stddev']

        self.vx_tau_01_min = tmp['vx_tau_01_min']
        self.vx_tau_01_max = tmp['vx_tau_01_max']
        self.vx_tau_01_mean = tmp['vx_tau_01_mean']
        self.vx_tau_01_var = tmp['vx_tau_01_var']
        self.vx_tau_01_median = tmp['vx_tau_01_median']
        self.vx_tau_01_stddev = tmp['vx_tau_01_stddev']
        self.vy_tau_01_min = tmp['vy_tau_01_min']
        self.vy_tau_01_max = tmp['vy_tau_01_max']
        self.vy_tau_01_mean = tmp['vy_tau_01_mean']
        self.vy_tau_01_var = tmp['vy_tau_01_var']
        self.vy_tau_01_median = tmp['vy_tau_01_median']
        self.vy_tau_01_stddev = tmp['vy_tau_01_stddev']

        self.vx_tau_001_min = tmp['vx_tau_001_min']
        self.vx_tau_001_max = tmp['vx_tau_001_max']
        self.vx_tau_001_mean = tmp['vx_tau_001_mean']
        self.vx_tau_001_var = tmp['vx_tau_001_var']
        self.vx_tau_001_median = tmp['vx_tau_001_median']
        self.vx_tau_001_stddev = tmp['vx_tau_001_stddev']
        self.vy_tau_001_min = tmp['vy_tau_001_min']
        self.vy_tau_001_max = tmp['vy_tau_001_max']
        self.vy_tau_001_mean = tmp['vy_tau_001_mean']
        self.vy_tau_001_var = tmp['vy_tau_001_var']
        self.vy_tau_001_median = tmp['vy_tau_001_median']
        self.vy_tau_001_stddev = tmp['vy_tau_001_stddev']

        self.vx_tau_0001_min = tmp['vx_tau_0001_min']
        self.vx_tau_0001_max = tmp['vx_tau_0001_max']
        self.vx_tau_0001_mean = tmp['vx_tau_0001_mean']
        self.vx_tau_0001_var = tmp['vx_tau_0001_var']
        self.vx_tau_0001_median = tmp['vx_tau_0001_median']
        self.vx_tau_0001_stddev = tmp['vx_tau_0001_stddev']
        self.vy_tau_0001_min = tmp['vy_tau_0001_min']
        self.vy_tau_0001_max = tmp['vy_tau_0001_max']
        self.vy_tau_0001_mean = tmp['vy_tau_0001_mean']
        self.vy_tau_0001_var = tmp['vy_tau_0001_var']
        self.vy_tau_0001_median = tmp['vy_tau_0001_median']
        self.vy_tau_0001_stddev = tmp['vy_tau_0001_stddev']

        self.vx_d_1000km_min = tmp['vx_d_1000km_min']
        self.vx_d_1000km_max = tmp['vx_d_1000km_max']
        self.vx_d_1000km_mean = tmp['vx_d_1000km_mean']
        self.vx_d_1000km_var = tmp['vx_d_1000km_var']
        self.vx_d_1000km_median = tmp['vx_d_1000km_median']
        self.vx_d_1000km_stddev = tmp['vx_d_1000km_stddev']
        self.vy_d_1000km_min = tmp['vy_d_1000km_min']
        self.vy_d_1000km_max = tmp['vy_d_1000km_max']
        self.vy_d_1000km_mean = tmp['vy_d_1000km_mean']
        self.vy_d_1000km_var = tmp['vy_d_1000km_var']
        self.vy_d_1000km_median = tmp['vy_d_1000km_median']
        self.vy_d_1000km_stddev = tmp['vy_d_1000km_stddev']

    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_inputs))

        x = inputs
        conv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal')(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Dropout(0.5)(conv1)
        stri1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
                       kernel_initializer='he_normal')(conv1)
        stri1 = BatchNormalization()(stri1)
        stri1 = Activation('relu')(stri1)

        conv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal')(stri1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Dropout(0.5)(conv2)
        stri2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
                       kernel_initializer='he_normal')(conv2)
        stri2 = BatchNormalization()(stri2)
        stri2 = Activation('relu')(stri2)

        conv3 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal')(stri2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(0.5)(conv3)
        stri3 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
                       kernel_initializer='he_normal')(conv3)
        stri3 = BatchNormalization()(stri3)
        stri3 = Activation('relu')(stri3)

        convc = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal')(stri3)
        convc = BatchNormalization()(convc)
        convc = Activation('relu')(convc)
        convc = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal')(convc)
        convc = BatchNormalization()(convc)
        convc = Activation('relu')(convc)
        convc = Dropout(0.5)(convc)

        upconv3 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(convc))
        upconv3 = Concatenate(axis=3)([conv3, upconv3])
        upconv3 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same',
                         kernel_initializer='he_normal')(upconv3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation('relu')(upconv3)
        upconv3 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal')(upconv3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Activation('relu')(upconv3)
        upconv3 = Dropout(0.5)(upconv3)

        upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(upconv3))
        upconv2 = Concatenate(axis=3)([conv2, upconv2])
        upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same',
                         kernel_initializer='he_normal')(upconv2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation('relu')(upconv2)
        upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal')(upconv2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Activation('relu')(upconv2)
        upconv2 = Dropout(0.5)(upconv2)

        upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal', activation='relu')(UpSampling2D(size=(2, 2))(upconv2))
        upconv1 = Concatenate(axis=3)([conv1, upconv1])
        upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), padding='same',
                         kernel_initializer='he_normal')(upconv1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation('relu')(upconv1)
        upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                         kernel_initializer='he_normal')(upconv1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Activation('relu')(upconv1)
        upconv1 = Dropout(0.5)(upconv1)

        final = Conv2D(self.n_components, (1, 1), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', activation='linear')(upconv1)

        self.model = Model(inputs=inputs, outputs=final)

        json_string = self.model.to_json()
        f = open('{0}_model.json'.format(self.root), 'w')
        f.write(json_string)
        f.close()

    def compile_network(self):
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))

    def read_network(self):
        print("Reading previous network...")

        f = open('{0}_model.json'.format(self.root), 'r')
        json_string = f.read()
        f.close()

        self.model = model_from_json(json_string)
        self.model.load_weights("{0}_weights.hdf5".format(self.root))

    def train(self, n_iterations):
        print("Training network...")

        # Callbacks
        self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1,
                                            save_best_only=True)

        # Load loss History
        n_val_loss = 0
        if self.option == 'continue':
            self.csv_logger = CSVLogger("{0}_loss.csv".format(self.root), separator=',', append=True)
            list_val_loss = np.zeros(1)
            with open("{0}_loss.csv".format(self.root)) as csvfile:
                readcsv = csv.reader(csvfile, delimiter=',')
                cnt = 0
                for row in readcsv:
                    vl = row[2]
                    if cnt > 1:
                        list_val_loss = np.append(list_val_loss, np.float64(vl))
                    elif cnt == 1:
                        list_val_loss[0] = np.float64(vl)
                    cnt = cnt+1
            n_val_loss = len(list_val_loss)
            if n_val_loss > 1:
                self.checkpointer.best = np.nanmin(list_val_loss, axis=None)
            else:
                self.checkpointer.best = list_val_loss[0]
            print('Best val_loss: {0}'.format(self.checkpointer.best))
        else:
            self.csv_logger = CSVLogger("{0}_loss.csv".format(self.root), separator=',', append=False)

        # Read training and validation sets
        tmp = np.load(self.input_files_training)
        input_train = tmp[self.input_keys_training]
        tmp = np.load(self.output_files_training)
        output_train = tmp[self.output_keys_training]
        tmp = np.load(self.input_files_validation)
        input_valid = tmp[self.input_keys_validation]
        tmp = np.load(self.output_files_validdation)
        output_valid = tmp[self.output_keys_validation]
        # Adjust size
        input_train = input_train[0:self.n_training, 0:self.nx, 0:self.ny, :]
        output_train = output_train[0:self.n_training, 0:self.nx, 0:self.ny, :]
        input_valid = input_valid[0:self.n_validation, 0:self.nx, 0:self.ny, :]
        output_valid = output_valid[0:self.n_validation, 0:self.nx, 0:self.ny, :]
        # Normalization
        input_train = input_train / self.ic_tau_1_median
        input_valid = input_valid / self.ic_tau_1_median
        output_train[:, :, :, 0] = (output_train[:, :, :, 0] -
                                    self.vx_d_1000km_min) / (self.vx_d_1000km_max - self.vx_d_1000km_min)
        output_train[:, :, :, 1] = (output_train[:, :, :, 1] -
                                    self.vy_d_1000km_min) / (self.vy_d_1000km_max - self.vy_d_1000km_min)
        output_valid[:, :, :, 0] = (output_valid[:, :, :, 0] -
                                    self.vx_d_1000km_min) / (self.vx_d_1000km_min - self.vx_d_1000km_min)
        output_valid[:, :, :, 1] = (output_valid[:, :, :, 1] -
                                    self.vy_d_1000km_min) / (self.vy_d_1000km_min - self.vy_d_1000km_min)

        # Training process
        self.model.fit(input_train, output_train, batch_size=self.batch_size, epochs=n_iterations, verbose=1,
                       steps_per_epoch=self.batches_per_epoch_training,
                       callbacks=[self.checkpointer, self.csv_logger],
                       validation_data=(input_valid, output_valid), validation_batch_size=self.batch_size,
                       validation_steps=self.batches_per_epoch_validation, initial_epoch=n_val_loss, shuffle=True)

        # Aftermath
        cnt = 0
        with open("{0}_loss.csv".format(self.root)) as csvfile:
            readcsv = csv.reader(csvfile, delimiter=',')
            list_val_loss = np.zeros(1)
            for row in readcsv:
                vl = row[2]
                if cnt > 1:
                    list_val_loss = np.append(list_val_loss, np.float64(vl))
                elif cnt == 1:
                    list_val_loss[0] = np.float64(vl)
                cnt = cnt+1
        # Print min loss value
        print("Nb. of iterations performed: {0}; Best value: {1}".format(cnt-1, np.amin(list_val_loss)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train DeepVel')
    parser.add_argument('-o', '--out', help='Output files')
    parser.add_argument('-e', '--epochs', help='Number of epochs', default=10)
    parser.add_argument('-n', '--noise', help='Noise to add during training', default=0.0)
    parser.add_argument('-a', '--action', help='Action', choices=['start', 'continue'], required=True)
    parser.add_argument('-p', '--properties',
                        help='File containing the simulation properties for normalization',
                        default='network/simulation_properties.npz')
    parsed = vars(parser.parse_args())

    root = parsed['out']
    nEpochs = int(parsed['epochs'])
    option = parsed['action']
    noise = parsed['noise']
    norm_filename = parsed['properties']

    out = train_deepvel(root, noise, option, norm_filename)

    if option == 'start':
        out.define_network()

    if option == 'continue':
        out.read_network()

    if option == 'start' or option == 'continue':
        out.compile_network()
        out.train(nEpochs)
