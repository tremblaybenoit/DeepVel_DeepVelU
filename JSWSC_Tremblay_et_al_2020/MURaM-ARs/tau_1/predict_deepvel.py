import numpy as np
import platform
import os
import time
import argparse
from astropy.io import fits
import tensorflow as tf
from keras.layers import Input, Conv2D, Activation, BatchNormalization, add
from keras.models import Model

os.environ["KERAS_BACKEND"] = "tensorflow"

if platform.node() != 'vena':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class deepvel(object):

    def __init__(self, observations, output, border_x1=0, border_x2=0, border_y1=0, border_y2=0, norm_simulation=0,
                 norm_filename='network/simulation_properties.npz', network_weights='network/deepvel_weights.hdf5'):
        """
        ---------
        Keywords:
        ---------

        observations: Input array of shape (nx, ny, n_times*n_inputs) where
                        nx & ny: Image dimensions
                        n_times: Number of consecutive timesteps
                        n_inputs: Number of types of inputs

        output: Output array of dimensions (nx, ny, n_depths*n_comp) where
                        nx & ny: Image dimensions
                        n_depths: Number of optical/geometrical depths to infer
                        n_comp: Number of components of the velocity vector to infer

        border: Number of pixels to crop from the image in each direction
                        border_x1: Number of pixels to remove from the left of the image
                        border_x2: Number of pixels to remove from the right of the image
                        border_y1: Number of pixels to remove from the bottom of the image
                        border_y2: Number of pixels to remove from the top of the image

        same_as_training: Set to 1 if using data from the same simulation as the one used for training.
                            -> The inputs will be normalized using the same values as the inputs in the
                                training set because the values are known.

        network: Provide path to the network weights and normalization values

        """

        # GPU
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

        # -----------------
        # Input properties:
        # -----------------
        # Read
        self.observations = observations
        n_timesteps, nx, ny = observations.shape
        # Number of images to generate
        self.n_frames = n_timesteps - 1
        # Number of types of inputs (intensitygrams, magnetograms, Dopplergrams)
        self.n_types = 1
        # Number of consecutive frames of a given input
        self.n_times = 2
        # Number of inputs
        self.n_inputs = self.n_types*self.n_times
        # Image dimensions (remove borders)
        self.border_x1 = border_x1
        self.border_x2 = border_x2
        self.border_y1 = border_y1
        self.border_y2 = border_y2
        self.nx = nx - self.border_x1 - self.border_x2
        self.ny = ny - self.border_y1 - self.border_y2

        # ------------------
        # Output properties:
        # ------------------
        # Filename
        self.output_filename = output
        # Number of inferred depths
        self.n_depths = 1
        # Number of inferred velocity components
        self.n_components = 2
        self.n_outputs = self.n_depths*self.n_components

        # -----------------
        # Network properties:
        # -----------------
        # Load training weights
        self.weights_filename = network_weights
        # Architecture
        self.n_filters = 64
        self.kernel_size = 3
        self.n_conv_layers = 20
        self.batch_size = 1

        # --------------------
        # Test set properties:
        # --------------------
        # Use same normalization values as for the training and validation sets
        self.norm_simulation = norm_simulation

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

    def residual(self, inputs):

        x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, inputs])

        return x

    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_inputs))
        conv = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                      kernel_initializer='he_normal', activation='relu')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = add([x, conv])

        final = Conv2D(self.n_outputs, (1, 1), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=final)
        self.model.load_weights(self.weights_filename)

    def predict(self):
        print("Predicting velocities with DeepVel...")

        # Input and output arrays
        inputs = np.zeros((self.batch_size, self.nx, self.ny, self.n_inputs), dtype='float32')
        outputs = np.zeros((self.n_frames, self.nx, self.ny, self.n_outputs), dtype='float32')

        # Normalization of the input data:
        # If keyword is set, we use the median of the continuum intensity at tau = 1 as computed from
        # the training simulation. Otherwise, we compute the median from the input data.
        if self.norm_simulation == 0:
            self.ic_tau_1_median = np.median(self.observations[:, self.border_x1:self.border_x1 + self.nx,
                                             self.border_y1:self.border_y1 + self.ny])
        # Computation time (start)
        start = time.time()

        # Loop over all frames (timesteps)
        for i in range(self.n_frames):
            inputs[:, :, :, 0] = self.observations[i * self.batch_size:(i + 1) * self.batch_size,
                                                   self.border_x1:self.border_x1 + self.nx,
                                                   self.border_y1:self.border_y1 + self.ny] / self.ic_tau_1_median
            inputs[:, :, :, 1] = self.observations[i * self.batch_size + 1:(i + 1) * self.batch_size + 1,
                                                   self.border_x1:self.border_x1 + self.nx,
                                                   self.border_y1:self.border_y1 + self.ny] / self.ic_tau_1_median

            outputs[i, :, :, :] = self.model.predict(inputs, batch_size=self.batch_size, max_queue_size=1,
                                                     verbose=1)

        # Computation time (end)
        end = time.time()
        print("Prediction took {0} seconds...".format(end - start))

        # Output data is normalized -> Reverse process
        outputs[:, :, :, 0] = outputs[:, :, :, 0] * (self.vx_tau_1_max - self.vx_tau_1_min) + self.vx_tau_1_min
        outputs[:, :, :, 1] = outputs[:, :, :, 1] * (self.vy_tau_1_max - self.vy_tau_1_min) + self.vy_tau_1_min
        
        # Save inferred flows in a .fits file
        # Format: (self.n_frames, self.border_x1:self.border_x1 + self.nx, self.border_y1:self.border_y1 + self.ny,
        #          self.n_outputs)
        hdu = fits.PrimaryHDU(outputs)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(self.output_filename, overwrite=True)


# Main
if __name__ == '__main__':

    # Input parameters/keywords
    parser = argparse.ArgumentParser(description='DeepVel prediction')
    parser.add_argument('-o', '--out', help='Output file')
    parser.add_argument('-i', '--in', help='Input file')
    parser.add_argument('-bx1', '--border_x1', help='Border size in pixels', default=0)
    parser.add_argument('-bx2', '--border_x2', help='Border size in pixels', default=0)
    parser.add_argument('-by1', '--border_y1', help='Border size in pixels', default=0)
    parser.add_argument('-by2', '--border_y2', help='Border size in pixels', default=0)
    parser.add_argument('-norm_simulation', '--normalization_simulation',
                        help='Set to 1 if data is from the same simulation as the training set', default=0)
    parser.add_argument('-norm_filename', '--normalization_filename',
                        help='Normalization file for the inputs/outputs',
                        default='network/simulation_properties.npz')
    parser.add_argument('-n', '--network', help='Path to network weights and normalization values',
                        default='network/deepvel_weights.hdf5')
    parsed = vars(parser.parse_args())

    # Open file with observations and read
    f = fits.open(parsed['in'])
    imgs = f[0].data

    # Initialization
    out = deepvel(imgs, parsed['out'],
                  border_x1=int(parsed['border_x1']),
                  border_x2=int(parsed['border_x2']),
                  border_y1=int(parsed['border_y1']),
                  border_y2=int(parsed['border_y2']),
                  norm_simulation=int(parsed['normalization_simulation']),
                  norm_filename=parsed['normalization_filename'],
                  network_weights=parsed['network'])

    # Neural network architecture
    out.define_network()
    # Make predictions
    out.predict()
