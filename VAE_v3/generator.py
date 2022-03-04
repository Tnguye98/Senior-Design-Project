# generator.py
# 
# This file is used to generate unique imagery from a trained vae mode
# 	It is required to run vae.py before this program can be used
#

# Dependancies
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import sys
import pickle
import os
import random

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid



#Data Builder File: ./data_builder.py
import data_builder as datab

#Data Pipeline for FMOW dataset: ./pipeline.py
from pipeline import load_im, load_manifest, load_manifest_count, load_manifest_rand


#MNIST
from keras.datasets import mnist

#Image Processing
from PIL import Image


# Random Data Generator
# Returns a random normal distribution of a specified size
def genRandData(size):
	g1 = tf.random.get_global_generator()
	o = g1.normal(shape=[1,size])
	return o

# Loads a locally specified file
# Not in use/depreciated
def loadLocal():
	l = []
	pim = Image.open("inf.jpg")
	pim_np = np.asarray(pim)
	pim_np = pim_np/255.
	l.append(pim_np)
	return np.asarray(l)

# Takes a base image and converts to the latent space. Then performs an add operation with a random normal distribution and reconstructs
# encoder : encoder model
# decoder : decoder model
# base_im : A raw image in numpy (x,y,RGB) format
# dim : The size of the latent space
def perturbGen(encoder, decoder, base_im, dim):
	enc_im = encoder.predict(base_im)
	p_data = genRandData(dim)
	c_im = enc_im + p_data
	pred_im = decoder.predict(c_im)
	return pred_im


# Takes an input image and perturbs a single dimension within a random threshold
# encoder : encoder model
# decoder : decoder model
# base_im : A raw image in numpy (x,y,RGB) format
# target_dimension : Specified dimension to perturb
# threshold : A 2 length tuple containing the min and max of a random integer, (min, max)
def perturbGenSingleThreshold(encoder, decoder, base_im, target_dimension, threshold):
	enc_im = encoder.predict(base_im)
	enc_im[0][target_dimension] = random.randint(threshold[0], threshold[1])
	pred_im = decoder.predict(enc_im)
	return pred_im


# Randomly generates a random normal distribution in a give size and generates an image from it
# decoder : decoder model
# dim : Size of the latent space
def randomGen(decoder, dim):
	r_data = genRandData(dim) 
	pred_im = decoder.predict(r_data)
	return pred_im



# - Data Needed for Loading the VAE
#CIFAR10 Filename List for importer
CIFAR10_Filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
LATENT_DIM = 512
HIDDEN_LAYER_DIM = 2048

IMAGE_DIMENSIONS = (512,512)

input_shape = IMAGE_DIMENSIONS + (3,)

#CIFAR 10 Data
#pic_data = datab.load_data_sets(CIFAR10_Filenames)


# Load the model parts

decoder = tf.keras.models.load_model('model/VAE_decoder')
encoder = tf.keras.models.load_model('model/VAE_encoder')
#VAE = tf.keras.models.load_model('model/VAE_full')


##########################################

## CONFIGURATION VALUES ##
# Change these to edit the number of images created

# Number of rows
rows = 5
# Number of images per row
ims_per_row = 5



#Image Plotting Here
total_plot = rows*ims_per_row

# Setup a grid
fig = plt.figure(figsize=(ims_per_row, rows))
fig.set_size_inches(40,40)
grid = ImageGrid(fig, 111, nrows_ncols=(ims_per_row, rows), axes_pad=0.1)


##########################################

#Load Manifest
mf_file = open("train.manifest", "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

# Generate new images here

for i in range(0, total_plot, 3):

	base_im = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 1)

	#Generation happens here. Three predifined functions are availible. Any configuration can be created, however.
	gen_im = perturbGen(encoder, decoder, base_im, 512)
	# gen_im = perturbGenSingleThreshold(encoder, decoder, base_im, 0, (-10,10))
	# gen_im = randomGen(decoder, 512)
	
	#Plot images here
	grid[0].set_aspect('equal')
	grid[0].imshow(gen_im[0], cmap = plt.cm.binary)


#Output image data here
#plt.show()
fig.savefig("GenImages.png")
