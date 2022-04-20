'''
This file is the primary training file for the VAE. 
Requires proper manifest and the 'model' folder

File Usage:

	python vae.py 
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import gc
import os

from mpl_toolkits.axes_grid1 import ImageGrid


print(" ---------------------")
print("|VAE starting . . . . |")
print(" ---------------------")

# Configuration Variables
number_of_pics = 10
max_epochs = 10000
num_rows_plots = 20
train_dir = "../datasets/train"
test_dir = "../testDataset/test"
checkpoint_path = "modelCheckpoint"

# Constants Configuration
LATENT_DIM = 512
HIDDEN_LAYER_DIM = 2048
IMAGE_DIMENSIONS = (512, 512)
BATCH_SIZE = 32
input_shape = IMAGE_DIMENSIONS + (3,)
trainDataset = tf.keras.utils.image_dataset_from_directory(directory = train_dir, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = IMG_SIZE)
testDataset = tf.keras.utils.image_dataset_from_directory(directory = test_dir, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = IMG_SIZE)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 100)

# Sampling function
def sampling(args):
	z_mean, z_log_var = args
	epsilon = tf.random.normal(shape = (tf.shape(z_mean)[0], LATENT_DIM), mean = 0, stddev = 1.0)
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
VAE.summary()


base_truth = tf.reshape(encoder_input, [-1])
predicted_truth = tf.reshape(outputs, [-1])
bc_loss = 1056 * tf.keras.losses.binary_crossentropy(base_truth, predicted_truth)
bc_loss_tensor = tf.convert_to_tensor(bc_loss)
kl_loss = (-0.5) * tf.math.reduce_mean((1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)), axis = -1)
total_loss = tf.math.reduce_mean(bc_loss + kl_loss)

VAE.add_loss(total_loss)
VAE.compile(optimizer = 'adam')

history = VAE.fit(trainDataset, epochs = max_epochs, steps_per_epoch = len(training_data), validation_data = testDataset, validation_steps = len(trainDataset), callbacks = [model_checkpoint])

encoder.save('model/VAE_encoder')
decoder.save('model/VAE_decoder')
VAE.save('model/VAE_full')

print(" ----------------------")
print("|VAE Completed . . . . |")
print(" ----------------------")
