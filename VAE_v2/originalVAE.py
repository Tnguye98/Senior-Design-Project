# vae.py
#
# This file is the primary training file for the VAE.
# Training will fail unless a proper manifest is provided and the 'model' folder exists in the base directory
#
# File Usage:
#	python vae.py [args]
#
#	Command Line Arguments:
#	--load [int]	| Load checkpoint model and start at specified epoch
#	-l [int]		| Alias for --load
#	--arch			| Shown model architecture only, do not execute
#	-a 				| Alias for --arch
#
# File Configuraion:
#	To properly configure this file, you may edit the configuration and constants values
#	These variables can be found under the following sections:
#		CONFIGURATION VARIABLES 	- Line 70
#		CONSTANTS CONFIGURATION 	- Line 90
#





import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import sys
import pickle
import os
import gc

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid



#Data Builder File: ./data_builder.py - No longer in use
import data_builder as datab

#Data Pipeline for FMOW dataset: ./pipeline.py
from pipeline import load_im, load_manifest, load_manifest_count, load_manifest_rand


#MNIST
#NOTE: MNIST usage is depreciated. Do not use
from keras.datasets import mnist


#CIFAR10 Filename List for importer
#NOTE: Depreciated
CIFAR10_Filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

#Import data from CIFAR10 Dataset. Expected 5000 images total.
# NOTE: Depreciated. load_data_sets() is implemented in data_builder.py
#	This code is left in for reference but is now non-functional
#	See older versions of 'experiemental' branch for implementations
# load_data_sets(file_list, data_id)
# Default data ID is 3 for Cats - See data_builder.py for details

# pic_data = datab.load_data_sets(CIFAR10_Filenames)



################################################################
## CONFIGURATION VARIABLES ##

# These values are user editable
# They will change how your model runs, as well as the output
# data and sampled images

# Determins the number of pictures to plot per plotting step
number_of_pics = 10

# Number of epochs to run for
max_epochs = 10000

# Number of rows to plot.
# Will plot one row every (max_epochs/num_rows_plot) epochs
num_rows_plot = 20

#NOTE: Model checkpointing intervals are currently determined by (max_epochs/num_rows_plot)

# Manifest filenames
# These should be changed to the values of your generated manifests
traing_mf_name = "train.manifest"
validation_mf_name = "val.manifest"


################################################################
## CONSTANTS CONFIGURAION ##

# These values are user editable
# It is recommended not to edit the LATENT_DIM and HIDDEN_LAYER_DIM values
# IMAGE_DIMENSIONS should be editted to accomodate differently sized images.
# Note that dimensions lower than 256x256 are not supported by this architecture.

# The size of the latent dim (May affect model performance)
LATENT_DIM = 512
# The intermediate layer (May affect model performance)
HIDDEN_LAYER_DIM = 2048

# The dimensions of the input images
IMAGE_DIMENSIONS = (512,512)

# The number of channels (Do not edit this value if you are already using 3-channel RGB images)
input_shape = IMAGE_DIMENSIONS + (3,)

################################################################
## COMMAND LINE ARG LOGIC ##

#Get Args
architecture_only = False
reload_previous = False
start_at_epoch = 0
# If have any args
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


################################################################
## SAMPLING FUNCTION ##

# Used to implement epislon, needed for the variaional part
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM),mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

################################################################
## MODEL ARCHITECTURE - ENCODER ##

#Make the encoder here
encoder_input = keras.Input(shape=input_shape)
#Convolutional Layer to make sure 3-channel RGB is represented
x = layers.Conv2D(3, kernel_size=(4,4), padding='same', activation='relu', name='RGB_Layer')(encoder_input)

#Convolutional Layer 1, 4x4 kernel 4x4 stride
x = layers.Conv2D(32, kernel_size=(4,4), padding='same', activation='relu', strides=(4,4), name='Conv_Layer_1')(x)
x = layers.MaxPooling2D((2,2), padding='same', name='Pooling_Layer_1')(x)

#Convolutional Layer 2, 4x4 kernel no stride
x = layers.Conv2D(64, kernel_size=(4,4), padding='same', activation='relu', strides=1, name='Conv_Layer_2')(x)
x = layers.MaxPooling2D((2,2), padding='same', name='Pooling_Layer_2')(x)

#Convolutional Layer 3, 1x1 kernel no stride
x = layers.Conv2D(64, kernel_size=1, padding='same', activation='relu', strides=1, name='Conv_Layer_3')(x)


#Flatten Data and pass to Hidden Layer
flat_layer = layers.Flatten(name='Flatten_Layer')(x)
hidden_layer = layers.Dense(HIDDEN_LAYER_DIM, name='Hidden_Layer')(flat_layer)

#Latent Space is Built Here
# z_mean = mu
# z_log_var = sigma
z_mean = layers.Dense(LATENT_DIM, name='Z_MEAN')(hidden_layer)
z_log_var = layers.Dense(LATENT_DIM, name='Z_LOG_VAR')(hidden_layer)

# Insert epsolon via sampling layer
encoder_output = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='Latent_Space')([z_mean, z_log_var])

#Complete encoder model
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
#Print model summary
encoder.summary()


##########################################################
## MODEL ARCHITECTURE - DECODER ##

#Make the decoder Here
decoder_input = keras.Input(shape=(LATENT_DIM,))

#Reverse Hidden Layers
#Upsacle to hidden layer size
x = layers.Dense(HIDDEN_LAYER_DIM, name='Hidden_Layer')(decoder_input)
#Additional hidden layer to scale to our first convolutional layer
#This must be the size of the total data output by Conv_Layer_3
#In this case (64, 32, 32)
x = layers.Dense(64 * 32 * 32, name='Upscale_Layer')(x)

#Reshape for Conv Layers
#Must be reshaped to the same size and dimensions as Conv_Layer_3
x = layers.Reshape((32, 32, 64))(x)

#Convolutional Layers Transpose, Inverse Conv_Layer_3
x = layers.Conv2DTranspose(64, kernel_size=1, padding='same', strides=1, activation='relu', name='TP_Layer_3')(x)

#Convolutional Layers Transpose, Inverse Conv_layer_2
x = layers.UpSampling2D((2,2), name="UpSample_Layer_2")(x)
x = layers.Conv2DTranspose(64, kernel_size=(4,4), padding='same', strides=1, activation='relu', name='TP_Layer_2')(x)

#Convolutional Layers Transpose, Inverse Conv_Layer_1
x = layers.UpSampling2D((2,2), name="UpSample_Layer_1")(x)
x = layers.Conv2DTranspose(32, kernel_size=(4,4), padding='same', strides=(4,4), activation='relu', name='TP_Layer_1')(x)

#Convolutional Layers Transpose, Inverse RGB_Layer
decoder_output = layers.Conv2D(3, kernel_size=(4,4), padding='same', activation='sigmoid', name='Transpose_RGB_Layer')(x)

#Create Decoder Model
decoder = keras.Model(decoder_input, decoder_output, name="decoder")
#Print model summary
decoder.summary()


##########################################################
## MODEL COMPILATION ##


#Make the full VAE here
# [input] -> (Encoder) -> [z] -> (Decoder) -> [output]
z = encoder(encoder_input)
output = decoder(z)

vae = keras.Model(encoder_input, output, name="vae")
vae.summary()

#Quit if only showing the architecture. Do not continue.
if architecture_only:
    exit()


# Custom Loss Function
# Implements KL_Loss to the VAE
# Refernced Equations: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
base_truth = K.flatten(encoder_input)
predicted_truth = K.flatten(output)
bc_loss = 32 * 32 * keras.losses.binary_crossentropy(base_truth, predicted_truth)
kl_loss = (-0.5) * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
total_loss = K.mean(bc_loss + kl_loss)

# Add loss to the model
vae.add_loss(total_loss)
vae.compile(optimizer='adam')


##########################################################
## LOAD MANIFEST DATA ##


#Load Manifest
mf_file = open(traing_mf_name, "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

#Load Validation Manifest
mf_file = open(validation_mf_name, "r")
data = mf_file.read()
validation_manifest = data.split(" ")
mf_file.close()


### Load Data From Static Location
# NOTE: These Functions are depreciated; May not work.
# 	If implemented they will load all data from a manifest. Not recommended.
# print("Loading Training Data...")
# training_data = load_manifest(training_manifest, IMAGE_DIMENSIONS)
# print("Loading Validation Data...")
# validation_data = load_manifest(validation_manifest, IMAGE_DIMENSIONS)


################################################################
## MANIFEST SAMPLE DATA LOADING ##

# Loads data specifically for CIFAR10
# number_of_pics = 10
# sample_data = training_data[0:number_of_pics]
# sample_data_v = validation_data[0:number_of_pics]

# Loads data from a specified manifest file. Always picks randomly.
# If deterministic images are required, consider using load_manifest_count()
sample_data = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 10)
sample_data_v = load_manifest_rand(validation_manifest, IMAGE_DIMENSIONS, 10)




################################################################
## PLOTTING FUNCTIONS ##

# Create Plotter Function
def plot_step(plot_data, g, n, plot_i):
    #Function here
    # Plot and display result

    # Simulate Predictions
    # Run encoder and grab variable [2] (Latent data representation)
    # Run decoder on latent space
    result = plot_data
    offset = n*(plot_i+1)
    g[offset].set_ylabel('EPOCH {}'.format(plot_i*(max_epochs//num_rows_plot)))

    #Plot for each grid value
    for i in range(n):
        g[offset+1].set_aspect('equal')
        g[offset+i].imshow(result[i], cmap=plt.cm.binary)
        g[offset+i].set_xticklabels([])
        g[offset+i].set_yticklabels([])
        

#Returns only the numpy array representing the predicted image, should be reconstructed at the end
def gen_sample(vae, input_ims):
    result = vae.predict(input_ims)
    return result


################################################################
## PLOTTING CONFIGURATION ##

# Number of Rows to plot
epoch_plot_step = [i for i in range(0,max_epochs,max_epochs // num_rows_plot)]


# Setup Plot for Training Images
# Should have the same number of rows as the sample data length
fig = plt.figure(figsize=(number_of_pics, num_rows_plot+1))
fig.set_size_inches(40,40)
grid = ImageGrid(fig, 111, nrows_ncols=(num_rows_plot+1, number_of_pics), axes_pad=0.1)

grid[0].set_ylabel('BASE TRUTH')
for i in range(number_of_pics):
    grid[i].set_aspect('equal')
    grid[i].imshow(sample_data[i], cmap=plt.cm.binary)
    grid[i].set_xticklabels([])
    grid[i].set_yticklabels([])


# Setup Plot for Validation Images
# Should have the same number of rows as the sample data length
fig_v = plt.figure(figsize=(number_of_pics, num_rows_plot+1))
fig_v.set_size_inches(40,40)
grid_v = ImageGrid(fig_v, 111, nrows_ncols=(num_rows_plot+1, number_of_pics), axes_pad=0.1)

grid_v[0].set_ylabel('BASE TRUTH')
for i in range(number_of_pics):
    grid_v[i].set_aspect('equal')
    grid_v[i].imshow(sample_data_v[i], cmap=plt.cm.binary)
    grid_v[i].set_xticklabels([])
    grid_v[i].set_yticklabels([])


################################################################
## TRAINING CONFIGURATION ##

#Setup Checkpoint Path
checkpoint_path = "model/model.ckpt"

#NOTE: Checkpoint callbacks are depreciated in favor of incremental model saving.
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


#Init plot iterator
plot_iter = 0
#Init plotting data
plot_data = np.empty((0,number_of_pics) + IMAGE_DIMENSIONS + (3,))
plot_data_v = np.empty((0,number_of_pics) + IMAGE_DIMENSIONS + (3,))
plot_reshape = (-1,number_of_pics) + IMAGE_DIMENSIONS + (3,)

#Reload old plots if asked
if reload_previous:
    plot_data = np.load("training_plot_data_checkpoint.npy")
    plot_data_v = np.load("validation_plot_data_checkpoint.npy")

#Reload previous model if asked
if reload_previous:
    #Reload Model From Checkpoint
    vae.load_weights(checkpoint_path)
    print("Loaded Last Checkpoint for VAE")


################################################################
## TRAINING ##


#start_at_epoch, defaults to 0, will start at a later epoch if specified by command line args
for epoch in range(start_at_epoch, max_epochs):

    
	#Load Training and Validation Data here
	# Currently loads 64 training images and 32 validation images.
	# NOTE: This is bottlenecked by I/O speed. Image size may also play a role.
	# 	If RAM allows, consider loading all data at once?
	# IDEA: Implement these asynchronously for less bottlenecking?
    training_data = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 64)
    validation_data = load_manifest_rand(validation_manifest, IMAGE_DIMENSIONS, 32)

    #Notify what epoch we are running
    print("Running Epoch: " + str(epoch))
    #Model is trained here
    history = vae.fit(training_data, training_data, epochs=1, validation_data=(validation_data, validation_data))

    # Start plotting step
    # NOTE: Nothing is actually plotted here
    if epoch in epoch_plot_step:
    	# Grab a sample and save it to our plot data
    	# NOTE: the more plotting epochs we have, the more RAM that plot_data will use
        sp = gen_sample(vae, sample_data)
        sp = sp.reshape(plot_reshape)
        plot_data = np.concatenate((plot_data, sp))

        # Brab a smaple and save it to validation plot data
        sp = gen_sample(vae, sample_data_v)
        sp = sp.reshape(plot_reshape)
        plot_data_v = np.concatenate((plot_data_v, sp))
        
        # Increment our plot iterator
        plot_iter += 1
        print("Epoch " + str(epoch) + " Plotted.")

        #Save Numpy Data to File Here
        print("-----Saving Plot Data-----")
        np.save("training_plot_data_checkpoint.npy", plot_data)
        np.save("validation_plot_data_checkpoint.npy", plot_data_v)

        # Save our model checkpoint here
        print("-----Saving Model Weights-----")
        vae.save_weights(checkpoint_path)
        print("Saved Mode Step: " + str(epoch))
    

    #Garbage Collection
    print("Clearing Memory...")
    tf.keras.backend.clear_session()
    gc.collect()


################################################################
## PLOT IMAGES ##
i = 0
for im_row in plot_data:
    plot_step(im_row, grid, number_of_pics, i)
    i += 1

i = 0
for im_row_v in plot_data_v:
    plot_step(im_row_v, grid_v, number_of_pics, i)
    i += 1


################################################################
## SAVE MODELS ##

encoder.save('model/VAE_encoder') 
decoder.save('model/VAE_decoder') 
vae.save('model/VAE_full')


################################################################
## SAVE IMAGE RESULTS ##

fig.savefig("training-results.png")
fig.show()

fig_v.savefig("validation-results.png")
fig_v.show()


################################################################
## STATISTICS ##

#FIXME: Stats not working???
exit()


#FIXME: In theory these should work, in practice they do not right now.
#	All necessary data should be stored in 'history'

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

plt.savefig("loss.png")




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

plt.savefig("acc.png")