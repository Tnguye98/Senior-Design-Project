import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import ImageGrid

LATENT_DIM = 512
HIDDEN_LAYER_DIM = 2048
IMAGE_DIMENSIONS = (512, 512)
input_shape = IMAGE_DIMENSIONS + (3,)

decoder = tf.keras.models.load_model('model/VAE_decoder')
encoder = tf.keras.models.load_model('model/VAE_encoder')

def genRandData(size): 
	g1 = tf.random.get_global_generator()
	o = g1.normal(shape = [1, size])

def perturbGen(encoder, decoder, base_im, dim):
	enc_im = encoder.predict(base_im)
	p_data = genRandData(dim)
	c_im = enc_im + p_data
	pred_im = decoder.predict(c_im)
	return pred_im

def perturbGenSingleThreshold(encoder, decoder, base_im, target_dimension, threshold):
	enc_im = encoder.predict(base_im)
	enc_im[0][target_dimension] = random.randint(threshold[0], threshold[1])
	pred_im = decoder.predict(enc_im)
	return pred_im

def randomGen(decoder, dim):
	r_data = genRandData(dim) 
	pred_im = decoder.predict(r_data)
	return pred_im

def load_manifest_rand(manifest, dim, count):
	manifest = manifest[:-1]
	image_list = []
	max_rand = len(manifest) - 1
	for i in range(count):
		choice = random.randint(0, max_rand)
		obj = manifest[random.randint(0, max_rand)]
		im = Image.open(obj)
		im_np = np.asarray(im)
		im_np = im_np / 255.
		image_list.append(im_np)
	return_data = np.array(image_list)
	return return_data


rows = 5
ims_per_row = 5

total_plot = rows * ims_per_row

fig = plt.figure(figsize = (ims_per_row, rows))
fig.set_size_inches(40, 40)
grid = ImageGrid(fig, 111, nrows_ncols = (ims_per_row, rows), axes_pad = 0.1)

mf_file = open("train.manifest", "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

for i in range(0, total_plot, 3):
	base_im = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 1)
	gen_im = perturbGen(encoder, decoder, base_im, 512)
	grid[0].set_aspect('equal')
	grid[0].imshow(gen_im[0], cmap = plt.cm.binary)

# plt.show()
fig.savefig("GenImages.png")