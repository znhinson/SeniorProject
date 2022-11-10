# https://medium.com/analytics-vidhya/image-classification-with-vgg-convolutional-neural-network-using-keras-for-beginners-61767950c5dd

# import packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import regularizers
from keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau
# from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import random
import pickle
import cv2
import os
import copy
import sys

from keras.utils import img_to_array
import numpy as np

def build(width, height, depth):
    # initialize model, input shape, and channel dimension
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1    

    # using "channels first"
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
        chanDim = 1

    # CONV -> RELU -> POOL layer set
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    # (CONV -> RELU) * 2 -> POOL layer set
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    # (CONV -> RELU) * 3 -> POOL layer set
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    """
    # (CONV -> RELU) * 3 -> POOL layer set
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    """

    # FC -> RELU layer set
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(11))
    model.add(Activation("softmax"))

    # return constructed model
    return model

def train_model():
    # pre-processing
    DIRECTORY = r'new_dataset'
    CATEGORIES = ['1', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C', 'no_hair']
    ENCODINGS = {
        '1': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '2A': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        '2B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        '2C': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        '3A': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        '3B': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        '3C': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        '4A': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        '4B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        '4C': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'no_hair': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    data = []

    for category in CATEGORIES:
        folder = os.path.join(DIRECTORY, category)

        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img_arr = cv2.imread(img_path)
            img_arr = cv2.resize(img_arr, (224, 224))
            encoding = ENCODINGS.get(category)
            data.append([img_arr, encoding])

    random.shuffle(data)

    X = []
    Y = []

    for features, labels in data:
        X.append(features)
        Y.append(labels)

    X = np.array(X)
    Y = np.array(Y)

    x_remainder, x_test, y_remainder, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_remainder, y_remainder, test_size = 0.1, random_state = 42)

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    x_test = x_test / 255.0     

    model = build(224, 224, 3)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs = 12, batch_size = 21, validation_data = (x_valid, y_valid))

train_model()

"""
Epoch 1/12
40/40 [==============================] - 90s 2s/step - loss: 3.1909 - accuracy: 0.1418 - val_loss: 5.8126 - val_accuracy: 0.0978
Epoch 2/12
40/40 [==============================] - 93s 2s/step - loss: 2.7901 - accuracy: 0.2012 - val_loss: 8.2452 - val_accuracy: 0.0978
Epoch 3/12
40/40 [==============================] - 87s 2s/step - loss: 2.2343 - accuracy: 0.3588 - val_loss: 8.1645 - val_accuracy: 0.0978
Epoch 4/12
40/40 [==============================] - 88s 2s/step - loss: 1.8411 - accuracy: 0.4558 - val_loss: 6.1324 - val_accuracy: 0.0978
Epoch 5/12
40/40 [==============================] - 99s 2s/step - loss: 1.4193 - accuracy: 0.5673 - val_loss: 7.2528 - val_accuracy: 0.0978
Epoch 6/12
40/40 [==============================] - 88s 2s/step - loss: 1.0991 - accuracy: 0.6485 - val_loss: 9.9630 - val_accuracy: 0.0978
Epoch 7/12
40/40 [==============================] - 86s 2s/step - loss: 0.9395 - accuracy: 0.7115 - val_loss: 9.5009 - val_accuracy: 0.0978
Epoch 8/12
40/40 [==============================] - 85s 2s/step - loss: 0.8609 - accuracy: 0.7248 - val_loss: 6.3861 - val_accuracy: 0.1087
Epoch 9/12
40/40 [==============================] - 85s 2s/step - loss: 0.7579 - accuracy: 0.7636 - val_loss: 6.9022 - val_accuracy: 0.1196
Epoch 10/12
40/40 [==============================] - 96s 2s/step - loss: 0.4162 - accuracy: 0.8788 - val_loss: 4.6549 - val_accuracy: 0.1957
Epoch 11/12
40/40 [==============================] - 89s 2s/step - loss: 0.3629 - accuracy: 0.8970 - val_loss: 4.3052 - val_accuracy: 0.2174
Epoch 12/12
40/40 [==============================] - 85s 2s/step - loss: 0.2139 - accuracy: 0.9430 - val_loss: 3.4867 - val_accuracy: 0.2500
"""

"""
def process_image(img_path):
    # load input image and resize it
    img = cv2.imread(img_path)
    img = cv2.resize(img, 224, 224)

    # image to array
    img_arr = img_to_array(img)

    # normalize image
    processed_img = np.array(img_arr, dtype = "float") / 255.0
    return processed_img

def make_prediction(img_path):
    # process image
    img = process_image(img_path)

    # read image for output
    output = cv2.imread(img_path)
"""