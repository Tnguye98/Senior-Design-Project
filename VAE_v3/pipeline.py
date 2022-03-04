# pipeline.py
#
# Pipeline helper for setting up data transfer
#
# Manifest Specifications:
#	A manifest is a file that includes a series of space ' ' seperated absolute filepaths
#
# Functions Implemented Here:
#	load_im - Depreciated, do not use
#	load_manifest - Loads all images from a manifest
#	load_manifest_count - Loads a set number of images in a deterministic way from a manifest
#	load_manifest_rand - Loads a set number of images randomly from a manifest
#


# Necessary Imports
from PIL import Image
import numpy as np

import random


# Hardcoded Directory Path
# NOTE: This is utilzed in depreciated functions only
# directory = "/home/ubuntu/fmow-rgb-dataset/"
# Commented out due to not being used


# load_im(): Loads a number of images based on a manifest
# NOTE: This function is depreciated. It is not recommended to use it
# 	It implements on the fly image normalization which is not efficient
# manifest - List of all possible filepaths
# num_imgs - How many images to return
# dim - The dimensions of the image to load, should be in size (X,Y)
## DEPRECIATED ##
# def load_im(manifest, num_imgs, dim):
#
# 	return_data = np.empty((0,) + dim + (3,))
# 	reshape_size = (-1,) + dim + (3,)
#
# 	max_rand = len(manifest) - 1
# 	for i in range(num_imgs):
#
# 		choice = random.randint(0, max_rand)
# 		#Open chosen file from hardcoded file path
# 		im = Image.open(directory + manifest[choice])
#
# 		#Resize Image
# 		im = im.resize(dim)
#
# 		#Convert to np array
# 		im_np = np.asarray(im)
#
# 		#Convert to correct shape
# 		im_np = im_np.reshape(reshape_size)
#
# 		im_np = im_np /255.
#
# 		#Add to return data
# 		return_data = np.concatenate((return_data, im_np))
#
# 	return return_data

# Commented out lines 36-62 due to function being labeled as deprciated


# Loads all images from a specified Manifest

# manifest - List of all possible filepaths
# dim - Size of the images to load (VARIABLE NOT IN USE/DEPRECIATED)

def load_manifest(manifest, dim):
    # return_data = np.empty((0,) + dim + (3,))
    # reshape_size = dim + (3,)

    # Create an empty list for concatenate work around
    image_list = []

    # Load each image individually
    for obj in manifest:
        # Sanity Check
        # FIXME: Manifest likely shouldn't include this in the first place?
        if obj == "":
            continue
        # Debug Print
        # print("Loading: " + obj)
        im = Image.open(obj)
        im_np = np.asarray(im)

        # im_np = im_np.reshape(reshape_size)

        im_np = im_np / 255.

        # NOTE: Concatenate is slow and BAD, do not use!!!
        # return_data = np.concatenate((return_data, im_np))

        image_list.append(im_np)

    return_data = np.array(image_list)
    return return_data


# Loads a specified number of images from a manifest in a deterministic order

# manifest - List of all possible filepaths
# dim - Size of the images to load (VARIABLE NOT IN USE/DEPRECIATED)
# count - Number of images to return

def load_manifest_count(manifest, dim, count):
    # return_data = np.empty((0,) + dim + (3,))
    # reshape_size = dim + (3,)

    # Create an empty list for concatenate work around
    image_list = []

    # Load each image individually
    for i in range(count):
        obj = manifest[i]
        # Sanity Check
        # FIXME: Manifest likely shouldn't include this in the first place?
        if obj == "":
            continue
        # Debug Print
        # print("Loading: " + obj)
        im = Image.open(obj)
        im_np = np.asarray(im)

        # im_np = im_np.reshape(reshape_size)

        im_np = im_np / 255.

        # NOTE: Concatenate is slow and BAD, do not use!!!
        # return_data = np.concatenate((return_data, im_np))

        image_list.append(im_np)

    return_data = np.array(image_list)
    return return_data


# Loads random images from a manifest
# The best function implemented here for training

# manifest - List of all possible filepaths
# dim - Size of the images to load (VARIABLE NOT IN USE/DEPRECIATED)
# count - Number of images to return

def load_manifest_rand(manifest, dim, count):
    manifest = manifest[:-1]
    image_list = []
    max_rand = len(manifest) - 1
    for i in range(count):
        choice = random.randint(0, max_rand)
        obj = manifest[random.randint(0, max_rand)]
        # print("Loading: " + obj)
        im = Image.open(obj)
        im_np = np.asarray(im)

        # im_np = im_np.reshape(reshape_size)

        im_np = im_np / 255.

        # NOTE: Concatenate is slow and BAD, do not use!!!
        # return_data = np.concatenate((return_data, im_np))

        image_list.append(im_np)

    return_data = np.array(image_list)
    return return_data
