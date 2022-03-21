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
		VARIABLES			- Line 70
		CONSTANTS			- Line 90

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
import gc

from mpl_toolkits.axes_grid1 import ImageGrid











