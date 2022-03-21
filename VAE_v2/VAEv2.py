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
from mpl_toolkits.axes_grid1 import ImageGrid
from pipeline import load_manifest, load_manifest_count, load_manifest_rand



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







