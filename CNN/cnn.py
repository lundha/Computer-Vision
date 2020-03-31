import json
import os
import random

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot
import numpy as np
import tensorflow as tf

# Load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck']


# Set a constant random seed for deterministic runs.
SEED = 1021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


'''
for i in range(0,10):
    pyplot.subplot(2,5,i+1)
    pyplot.imshow(x_train[i])
    pyplot.title("class: %s \n class2: %s" % (classes[int(y_train[i])], classes[int(y_train[i+1])]))
pyplot.show()

'''

# Normalize the pixel values between [-1.0, 1.0].
x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1

# Add an extra dimension to the data so each image is 28x28x1.
# x_train, x_test = x_train[..., None], x_test[..., None]

# Mix up the training data.
perm_idx = np.random.permutation(len(x_train))
x_train, y_train = x_train[perm_idx], y_train[perm_idx]

# Remember to grab some data for validation (4:1 split).
num_val = len(x_train) // 5
x_val, y_val = x_train[:num_val], y_train[:num_val]

# Don't train on the validation set.
x_train, y_train = x_train[num_val:], y_train[num_val:]


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

'''
for i in range(10):
    pyplot.subplot(2,5,i+1)
    img  = x_val[i]
    img_reshaped = img.reshape(1,32,32,3)
    result = model.predict_classes(img_reshaped)
    real_label = classes[int(y_val[i])]
    predicted_label = classes[int(result)]
    pyplot.imshow(img)
    pyplot.title("real: %s \n predicted: %s" % (real_label, predicted_label))
pyplot.show()
'''
wrong_classified = [[0,0]] * 10
classified = [[0,0]] * len(x_test)
counter = 0

for i in range(len(x_val)):

    img = x_val[i]
    img_reshaped = img.reshape(1,32,32,3)
    result = model.predict_classes(img_reshaped)

    real_label = int(y_val[i])
    predicted_label = int(result)
    classified[i] = [real_label, predicted_label]

    if real_label != predicted_label and wrong_classified[real_label] == [0,0]:
        wrong_classified[real_label] = [predicted_label,i]
        counter = counter + 1
        if counter == 10:
            break

print(wrong_classified)
np.savetxt("wrong_classified", wrong_classified, fmt="%s")
np.savetxt("classified.txt", classified, fmt="%s")

for i in range(10):
    pyplot.subplot(2,5,i+1)
    im = int(wrong_classified[i][1])
    real_label = classes[int(y_val[im])]
    predicted_label = classes[int(wrong_classified[i][0])]
    img = x_val[im]
    pyplot.imshow(img)
    pyplot.title("real: %s \n predicted: %s" % (real_label, predicted_label))
pyplot.show()