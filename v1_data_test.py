from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# "import" model to be used for testing purposes
model = keras.models.load_model(r'CHC_model')

# identify folder containing testing data set
folder = 'dataset/test'

for img in os.listdir(folder):
    # get each image in testing data set and convert to array format
    img_path = os.path.join(folder, img)
    img_arr = cv2.imread(img_path)
    img_arr = cv2.resize(img_arr, (300, 300))

    # have model make a prediction of the image
    # .reshape() is not modifying the model; included it to forgo errors
    prediction = model.predict(img_arr.reshape(1, 300, 300, 3))

    # convert array to image to show in pyplot
    img = Image.fromarray(np.uint8(img_arr)).convert('RGB')

    # print out model's prediction
    if prediction[0][0] == 1:
        plt.imshow(img)
        plt.figtext(0.5, 0.01, "hair", horizontalalignment='center', fontsize=14)
        plt.show()
    else:
        plt.imshow(img)
        plt.figtext(0.5, 0.01, "no_hair", horizontalalignment='center', fontsize=14)
        plt.show()