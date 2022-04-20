import os
import pathlib
import matplotlib
import matplotlib.pyplot as pathlib
import numpy as np
import tensorflow as tf

train_dir = "../datasets/train"
test_dir = "../datasets/val"
modelCheckpoint_dir = "../modelCheckpoint"
max_epochs = 1000

IMG_SIZE = (512, 512)
BATCH_SIZE = 32
trainDataset = tf.keras.utils.image_dataset_from_directory(directory = train_dir, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = IMG_SIZE)
testDataset = tf.keras.utils.image_dataset_from_directory(directory = test_dir, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = IMG_SIZE)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 100)

baseModel = tf.keras.applications.resnet_v2.ResNet101V2(include_top = False)
inputs = tf.keras.layers.Input(shape = (512, 512, 3), name = 'input_layer')
x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 512)(inputs)
x = baseModel(x)
x = layers.GlobalAveragePooling2D(name = "global_avg_pool_layer")(x)
outputs = tf.keras.layers.Dense(len(trainDataset.class_names), activation = 'softmax', name = 'output_layer')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.0001), metrics = ['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy'])

history = model.fit(trainDataset, epochs = max_epochs, steps_per_epoch = len(trainDataset), validation_data = testDataset, validation_steps = len(trainDataset), callbacks = [model_checkpoint])

model.save('model/ObjectDetection')