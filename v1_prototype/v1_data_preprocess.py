import numpy as np # used for array operations
import cv2 # used for converting images into array
import os # used for location of images
import random
import matplotlib.pyplot as plt  # used for initializing
import pickle # used for saving data

# ADDITIONAL INSTALLATION REQUIREMENTS FOR ABOVE IMPORTS
# pip install opencv-python
# pip install matplotlib

#DATA PRE-PROCESSING
DIRECTORY = r'dataset'
CATEGORIES = ['hair', 'no_hair']
IMG_SIZE = 300 # smaller image size may lead to bad model due to over-pixelated images 

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY,category)
    # print(folder) # prints path for both categories
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img) # random order
        img_arr = cv2.imread(img_path) # image read to an array
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        # plt.imshow(img_arr) # array --> image
        # plt.show() # shows image
        # break
        data.append([img_arr, label]) # passing an array of an image with a corresponding label to data list
random.shuffle(data) # shuffles data 

X = []
Y = []

for features, labels in data:
    X.append(features)
    Y.append(labels)

X = np.array(X)
print(X)
Y = np.array(Y)

# pickle saves X and Y
pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('Y.pkl', 'wb'))