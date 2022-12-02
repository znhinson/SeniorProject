import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import tensorboard
import time

# pre-processing data
DIRECTORY = r'hair_dataset'
CATEGORIES = ['1', '2A', '2B', '2C', '3A', '3B', '3C', '4A', '4B', '4C', 'no_hair']
IMG_SIZE = 300
NUM_HAIR_CATS = 11

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])

random.shuffle(data)

X = []
Y = []

for features, labels in data:
    X.append(features)
    Y.append(labels)

X = np.array(X)
Y = np.array(Y)

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 42)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)

print(y_train[0])
plt.imshow(x_train[0])
plt.show()

# creating model
model = Sequential()
model.add(Conv2D(filters = 75, kernel_size = (3, 3), padding = 'same', input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Conv2D(filters = 75, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Conv2D(150, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(150, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(MaxPooling2D(pool_size = 2, strides = 2))
model.add(Flatten())
model.add(Dense(600, activation = 'relu'))
model.add(Dense(600, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_HAIR_CATS, activation = 'softmax'))

NAME = f'hair-type-prediction-{int(time.time())}' 
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
model.summary()
model.fit(x_train, y_train, epochs = 12, batch_size = 21, validation_data = (x_valid, y_valid), callbacks = tensorboard)