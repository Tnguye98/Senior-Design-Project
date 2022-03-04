###################################################################
#                                                                 #
# ./data_builder.py                                               #
#                                                                 #
# Code for loading the CIFAR10 dataset 					          #
# 	Also has a fancy header										  #
#                                                                 #
###################################################################

# NOTE: This file is kept as a refernce for loading CIFAR10 data
#	The modern VAE no longer implements any of this code.



# Imports
import pickle
import numpy as np


#Function for returning dictionary data from Pickle
#Format info here: http://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


#load_data_sets(list[str])
# This function takes a list of strings (str) as input.
# Each string should point to a path relative file that contains pickeled data.
# This function is designed to work only with the CIFAR10 dataset, a new function should be adapted to use custom datasets.
# target_id = 3 (Default for Cat Images)
def load_data_sets(dataset_list, target_id=3):

		#Create an empty numpy array that will be used to return data
		#Expected shape is in the form of (0,32,32,3)
		#return_data[IMG Number][Y Location][X Location][RGB value]
		return_data = np.empty((0,32,32,3))

		#Note: Does not check file list, filenames should be correct on input
		for dataset in dataset_list:
			#Unpickle dataset
			data = unpickle(dataset)
			
			#Seperate image data and label data
			image_data = data[b'data']
			label_data = data[b'labels']

			#Reshape image data into a usable format
			#Format, each pic = 3072 entries
			#1024 - R values
			#1024 - G values
			#1024 - B values
			# Total 3x32x32 = 3072
			image_data = image_data.reshape(-1,3,32,32)

			#Normalize Data Values for Tensorflow
			image_data = image_data /255.0

			#Transpose data so we get 32x32 sets of pixels in format [R,G,B]
			#Data Format:
			#image_data[IMG Number][Y Location][X Location][RGB value]
			image_data = np.transpose(image_data, (0,2,3,1))

			#Grab only selected ID pictures:
			#Explanation:
			#For every i in range(length of dataset)
			#If corresponding label == target_id (Is set above, default 3 for cats)
			#Add data into datset
			image_data = [image_data[i] for i in range(len(image_data)) if label_data[i] == 3]

			#Concatenate retrived data into our return list
			return_data = np.concatenate((return_data, image_data))

		#All data should now be concatenated in one large array following the specified format.
		return return_data


