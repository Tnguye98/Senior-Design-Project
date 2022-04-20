import tensorflow as tf
import tensorflow_hub as hub

def createModelFromHub(modelURL, numClasses, imgShape):
	modelLayer = hub.KerasLayer(modelURL, trainable = True, name = "hub_layer", input_shape = imgShape + (3,))
	model = tf.keras.Sequential([modelLayer, tf.keras.Dense(numClasses, activation = 'softmax', name = output_layer)])
	return model