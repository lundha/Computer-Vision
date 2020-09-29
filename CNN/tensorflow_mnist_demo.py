#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Connor D. Sanchez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow MNIST Demo"""

import json
import os
os.environ['CUDA_​DEVICE_​ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_​VISIBLE_​DEVICES'] = '0'  # Use the gpu in the 1st PCIe slot.
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import random

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Try to use the gpu if available and cuda installed. [not required]
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    try:
        logical_gpus = tf.config.experimental.list_logical_devices(
            device_type='GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as err:
        print(err)

# Set a constant random seed for deterministic runs.
SEED = 1021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

'''
for i in range(0,10):
    pyplot.subplot(2,5,i+1)
    pyplot.imshow(x_train[i])
pyplot.show()
'''


# Normalize the pixel values between [-1.0, 1.0].
x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1

# Add an extra dimension to the data so each image is 28x28x1.
#x_train, x_test = x_train[..., None], x_test[..., None]

# Mix up the training data.
perm_idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[perm_idx], y_train[perm_idx]

# Remember to grab some data for validation (4:1 split).
num_val = len(x_train) // 5
x_val, y_val = x_train[:num_val], y_train[:num_val]

# Don't train on the validation set.
x_train, y_train = x_train[num_val:], y_train[num_val:]

# This is the exact network as described in the homework.
# Try to change things around and see how it affects the training/results.
model = tf.keras.models.Sequential([
    # Notice the input shape, MNIST contains 28x28 images (grayscale). # Changed to input (32, 32, 1)
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
    ),
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid',
    ),
    tf.keras.layers.Flatten(data_format='channels_last'),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

# Print a summary of the network (layer names, shapes, and #parameters).
model.summary()

# Compile the computational graph.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the network using the Adam Optimizer (with cross entropy loss).
# Prints the training+validation loss+accuracy per epoch.
# Saves all of this in the "history".
# Try tweaking the batch size and the number of epochs.
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=8,
    validation_data=(x_val, y_val),
)

# Print the history dict.
print('epoch=', history.epoch)
print('history=', history.history)

# Serialize and save the history for later use.
h = {k: np.array(v).tolist() for k, v in history.history.items()}
with open('./history.json', 'w') as fp:
    json.dump(h, fp)

# Grab the epoch list and the training+validation loss+accuracy lists.
epoch = history.epoch
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training & validation accuracy values
plt.plot(epoch, accuracy)
plt.plot(epoch, val_accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(epoch, loss)
plt.plot(epoch, val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate and print the training loss and accuracy.
test_loss, test_accuracy = model.evaluate(
    x=x_test,
    y=y_test,
    batch_size=32,
)

# Now we are ready to save the network.

# Load the model when we want to use it again.
model = tf.keras.models.load_model('./model')

# Let's visualize some predictions.
# These are the soft predictions ("probabilities").
y_test = model.predict(
    x=x_test,
    batch_size=32,
)

# Get the hard predictions (digit 0-9).
# This corresponds to the digit with the maximum "probability".
y_test = y_test.argmax(-1)

###############################################################################
# TODO(students): Finish the rest of the assignment (plot errors, etc...).
###############################################################################
