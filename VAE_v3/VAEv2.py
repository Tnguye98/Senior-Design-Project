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