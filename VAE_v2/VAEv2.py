'''
This file is the primary training file for the VAE. 
Requires proper manifest and the 'model' folder

File Usage:

	python vae.py [args]

	Command Line Args:
		--load [int],     -l [int]			
			Loads checkpoint model and start at a specified epcoh
		
		--arch, 		  -a
			Shown model architecture only, do not execute

File Configuration: 
	To properly configure this file, you may edit the configuration
	and constant values.
		VARIABLES			- Line 35
		CONSTANTS			- Line 42

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import os

from mpl_toolkits.axes_grid1 import ImageGrid
from pipeline import load_manifest, load_manifest_count, load_manifest_rand
from tensorflow import keras
from tensorflow.keras import layers



# Configuration Variables
number_of_pics = 10
max_epochs = 10000
num_rows_plots = 20
traing_mf_name = "train.manifest"
validation_mf_name = "val.manifest"

# Constants Configuration
LATENT_DIM = 512
HIDDEN_LAYER_DIM = 2048
IMAGE_DIMENSIONS = (512, 512)
input_shape = IMAGE_DIMENSIONS + (3,)


# Command line argument
architecture_only = False
reload_previous = False
start_at_epoch = 0
if (len(sys.argv) > 1):
	# If arch
	if (sys.argv[1] == "-a" or sys.argv[1] == "--arch"):
		architecture_only = True
		print("Only Displaying Architecture - Model will not be run!")
	# If load
	if (sys.argv[1] == "-l" or sys.argv[1] == "--load"):
		reload_previous = True
		start_at_epoch = int(sys.argv[2])
		print("Restarting Model at Epoch: " + str(start_at_epoch))

# Sampling function
def sampling(args):
	z_mean, z_log_var = args
	epsilon = tf.random.normal(shape = (tf.shape(z_mean)[0], LATENT_DIM), mean =0, stddev = 1.0)
	return z_mean + tf.math.exp(z_log_var) * epsilon

# Encoder 
# Task make this shorter by importing layers
# Check what dataset we are using and if it needs to be normalized
encoder_input = tf.keras.layers.Input(shape = input_shape, name = "encoder_input")

x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (4, 4), padding = 'same', activation = 'relu', name = 'RGB_Layer')(encoder_input)
x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same', activation = 'relu', strides = (4, 4), name = 'Conv_Layer_1')(x)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'same', name = 'Pooling_Layer_1')(x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (4, 4), padding = 'same', activation = 'relu', strides = 1, name = 'Conv_Layer_2')(x)
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), padding = 'same', name = 'Pooling_Layer_2')(x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = 1, padding = 'same', activation = 'relu', strides = 1, name = 'Conv_Layer_3')(x)
x = tf.keras.layers.Flatten(name = 'Flatten_Layer')(x)
x = tf.keras.layers.Dense(units = HIDDEN_LAYER_DIM, name = 'Hidden_Layer')(x)

z_mean = tf.keras.layers.Dense(units = LATENT_DIM, name = 'Z_MEAN')(x)
z_log_var = tf.keras.layers.Dense(units = LATENT_DIM, name = 'Z_LOG_VAR')(x)

encoder_outputs = tf.keras.layers.Lambda(function = sampling, output_shape = (LATENT_DIM, ), name = 'Latent_Space')([z_mean, z_log_var])
encoder = tf.keras.Model(inputs = encoder_input, outputs = encoder_outputs, name = 'encoder')
# encoder.summary()

# Decoder
decoder_inputs = tf.keras.Input(shape = (LATENT_DIM,))

x = tf.keras.layers.Dense(units = HIDDEN_LAYER_DIM, name = 'Hidden_Layer')(decoder_inputs)
x = tf.keras.layers.Dense(units = 65536, name = 'Upscale_Layer')(x)
x = tf.keras.layers.Reshape((32, 32, 64))(x)
x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = 1, padding = 'same', strides = 1, activation = 'relu', name = 'TP_Layer_3')(x)
x = tf.keras.layers.UpSampling2D(size = (2, 2), name = 'UpSample_Layer_2')(x)
x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (4, 4), padding = 'same', strides = 1, activation = 'relu', name = 'TP_Layer_2')(x)
x = tf.keras.layers.UpSampling2D(size = (2, 2), name = 'UpSample_Layer_1')(x)
x = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = (4, 4), padding = 'same', activation = 'sigmoid', name = 'TP_Layer_1')(x)

decoder_outputs = tf.keras.layers.Conv2D(filters = 3, kernel_size = (4, 4), padding = 'same', activation = 'sigmoid', name = 'Transpose_RGB_Layer')(x)
decoder = tf.keras.Model(inputs = decoder_inputs, outputs = decoder_outputs, name = 'decoder')
# decoder.summary()

# Model Compilation
z = encoder(encoder_input)
outputs = decoder(z)

VAE = tf.keras.Model(inputs = encoder_input, outputs = outputs, name = "VAE")
# VAE.summary()

if architecture_only:
	exit()

base_truth = tf.reshape(encoder_input, [-1])
predicted_truth = tf.reshape(outputs, [-1])
bc_loss = 1056 * tf.keras.losses.binary_crossentropy(base_truth, predicted_truth)
kl_loss = (-0.5) * tf.math.reduce_mean((1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)), axis = -1)
total_loss = tf.math.reduce_mean(bc_loss, kl_loss)

VAE.add_loss(total_loss)
VAE.compile(optimizer = 'adam')

# Load Manifest Data
mf_file = open(traing_mf_name, "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

mf_file = open(validation_mf_name, "r")
data = mf_file.read()
validation_manifest = data.split(" ")
mf_file.close()

# Manifest Sample Data Loading -- uses pipeline.py
sample_data = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 10)
sample_data_v = load_manifest_rand(validation_manifest, IMAGE_DIMENSIONS, 10)


# Plotting Function
def plot_step(plot_data, g, n, plot_i):
	results, offset = plot_data, n * (plot_i, + 1)
	g[offset].set_ylabel('EPOCH {}'.format(plot_i * (max_epochs // num_rows_plot)))
	for i in range(n):
        g[offset + 1].set_aspect('equal')
        g[offset + i].imshow(result[i], cmap = plt.cm.binary)
        g[offset + i].set_xticklabels([])
        g[offset + i].set_yticklabels([])
	return 0

# Returns a numpy array representing the predicted image
def gen_sample(vae, input_ims):
	result = VAE.predict(input_ims)
	return results













print("It works. It works. It works. It works. It works. It works.")
print("It works. It works. It works. It works. It works. It works.")
print("It works. It works. It works. It works. It works. It works.")
print("It works. It works. It works. It works. It works. It works.")


