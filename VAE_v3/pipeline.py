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
