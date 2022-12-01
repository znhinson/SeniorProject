# https://medium.com/analytics-vidhya/image-classification-with-vgg-convolutional-neural-network-using-keras-for-beginners-61767950c5dd

# import packages
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import regularizers
from sklearn.model_selection import train_test_split
import random
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# construct model
def build(width, height, depth):
    # initialize model, input shape, and channel dimension
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1    

    # CONV -> RELU -> POOL layer set
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape = inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.4))

    # (CONV -> RELU) * 2 -> POOL layer set
    model.add(Conv2D(64, (3, 3), padding = "same", kernel_regularizer = regularizers.l2(l=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.4))

    # (CONV -> RELU) * 3 -> POOL layer set
    model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = regularizers.l2(l=0.01)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = chanDim))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.4))

    # FC -> RELU layer set
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))

    # softmax classifier
    model.add(Dense(11, kernel_regularizer = 'l2'))
    model.add(Activation("softmax"))

    # return constructed model
    return model

# preprocessing
def preprocessing():
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
            if (".DS_Store" in img_path):
                continue
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

    return (X, Y)

# training and validating
def train_and_validate():
    tup = preprocessing()
    X = tup[0]
    Y = tup[1]
    
    x_remainder, x_test, y_remainder, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_remainder, y_remainder, test_size = 0.1, random_state = 42)

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0
    x_test = x_test / 255.0     
    
    model = build(224, 224, 3)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs = 20, batch_size = 15, validation_data = (x_valid, y_valid))

    test(x_test, y_test, model)

# testing
def test(x_test, y_test, model):
    for i in range(len(x_test)):
        img = x_test[i]
        plt.imshow(img)
        plt.show()
        prediction = model.predict(img)
        print("Prediction = " + prediction)
        print("Correct label = " + y_test[i])

train_and_validate()

""" batch size = 20, l2 regularizer
Epoch 1/50
42/42 [==============================] - 85s 2s/step - loss: 3.8496 - accuracy: 0.1515 - val_loss: 4.4077 - val_accuracy: 0.1196
Epoch 2/50
42/42 [==============================] - 85s 2s/step - loss: 2.7143 - accuracy: 0.2885 - val_loss: 7.0769 - val_accuracy: 0.1196
Epoch 3/50
42/42 [==============================] - 84s 2s/step - loss: 2.0389 - accuracy: 0.4109 - val_loss: 5.7652 - val_accuracy: 0.1196
Epoch 4/50
42/42 [==============================] - 86s 2s/step - loss: 1.7965 - accuracy: 0.4812 - val_loss: 7.3921 - val_accuracy: 0.1196
Epoch 5/50
42/42 [==============================] - 95s 2s/step - loss: 1.2479 - accuracy: 0.6436 - val_loss: 5.1264 - val_accuracy: 0.1196
Epoch 6/50
42/42 [==============================] - 84s 2s/step - loss: 1.1419 - accuracy: 0.6861 - val_loss: 3.0904 - val_accuracy: 0.3696
Epoch 7/50
42/42 [==============================] - 84s 2s/step - loss: 2.0217 - accuracy: 0.4339 - val_loss: 4.2993 - val_accuracy: 0.2391
Epoch 8/50
42/42 [==============================] - 85s 2s/step - loss: 1.2842 - accuracy: 0.6315 - val_loss: 2.7084 - val_accuracy: 0.2717
Epoch 9/50
42/42 [==============================] - 88s 2s/step - loss: 0.8981 - accuracy: 0.7697 - val_loss: 2.4168 - val_accuracy: 0.3587
Epoch 10/50
42/42 [==============================] - 91s 2s/step - loss: 0.6146 - accuracy: 0.8691 - val_loss: 2.6017 - val_accuracy: 0.3696
Epoch 11/50
42/42 [==============================] - 92s 2s/step - loss: 0.5026 - accuracy: 0.8897 - val_loss: 2.7477 - val_accuracy: 0.3152
Epoch 12/50
42/42 [==============================] - 88s 2s/step - loss: 0.4164 - accuracy: 0.9273 - val_loss: 2.4142 - val_accuracy: 0.4348
Epoch 13/50
42/42 [==============================] - 92s 2s/step - loss: 0.3662 - accuracy: 0.9491 - val_loss: 2.6178 - val_accuracy: 0.4239
Epoch 14/50
42/42 [==============================] - 84s 2s/step - loss: 0.3162 - accuracy: 0.9600 - val_loss: 2.3258 - val_accuracy: 0.4130
Epoch 15/50
42/42 [==============================] - 85s 2s/step - loss: 0.3171 - accuracy: 0.9539 - val_loss: 2.3654 - val_accuracy: 0.4239
Epoch 16/50
42/42 [==============================] - 84s 2s/step - loss: 0.2542 - accuracy: 0.9770 - val_loss: 2.6009 - val_accuracy: 0.4348
Epoch 17/50
42/42 [==============================] - 84s 2s/step - loss: 0.2280 - accuracy: 0.9806 - val_loss: 2.6016 - val_accuracy: 0.4130
Epoch 18/50
42/42 [==============================] - 84s 2s/step - loss: 0.2251 - accuracy: 0.9794 - val_loss: 2.4585 - val_accuracy: 0.4348
Epoch 19/50
42/42 [==============================] - 84s 2s/step - loss: 0.2271 - accuracy: 0.9806 - val_loss: 2.4896 - val_accuracy: 0.4130
Epoch 20/50
42/42 [==============================] - 93s 2s/step - loss: 0.2000 - accuracy: 0.9903 - val_loss: 2.5869 - val_accuracy: 0.4130
Epoch 21/50
42/42 [==============================] - 102s 2s/step - loss: 0.2004 - accuracy: 0.9891 - val_loss: 2.4746 - val_accuracy: 0.4348
Epoch 22/50
42/42 [==============================] - 93s 2s/step - loss: 0.2032 - accuracy: 0.9830 - val_loss: 2.6724 - val_accuracy: 0.4239
Epoch 23/50
42/42 [==============================] - 102s 2s/step - loss: 0.1787 - accuracy: 0.9903 - val_loss: 2.4387 - val_accuracy: 0.4565
Epoch 24/50
42/42 [==============================] - 85s 2s/step - loss: 0.1732 - accuracy: 0.9891 - val_loss: 2.4061 - val_accuracy: 0.4348
Epoch 25/50
42/42 [==============================] - 86s 2s/step - loss: 0.1647 - accuracy: 0.9939 - val_loss: 2.5568 - val_accuracy: 0.3587
Epoch 26/50
42/42 [==============================] - 84s 2s/step - loss: 0.1752 - accuracy: 0.9903 - val_loss: 2.5052 - val_accuracy: 0.3804
Epoch 27/50
42/42 [==============================] - 84s 2s/step - loss: 0.1603 - accuracy: 0.9927 - val_loss: 2.5741 - val_accuracy: 0.3913
Epoch 28/50
42/42 [==============================] - 84s 2s/step - loss: 0.1596 - accuracy: 0.9952 - val_loss: 2.5807 - val_accuracy: 0.4348
Epoch 29/50
42/42 [==============================] - 85s 2s/step - loss: 0.1436 - accuracy: 0.9976 - val_loss: 2.3630 - val_accuracy: 0.3913
Epoch 30/50
42/42 [==============================] - 84s 2s/step - loss: 0.1455 - accuracy: 0.9952 - val_loss: 2.4200 - val_accuracy: 0.4239
Epoch 31/50
42/42 [==============================] - 84s 2s/step - loss: 0.1603 - accuracy: 0.9903 - val_loss: 2.4451 - val_accuracy: 0.4457
Epoch 32/50
42/42 [==============================] - 84s 2s/step - loss: 0.1443 - accuracy: 0.9964 - val_loss: 2.5434 - val_accuracy: 0.4239
Epoch 33/50
42/42 [==============================] - 84s 2s/step - loss: 0.1296 - accuracy: 0.9976 - val_loss: 2.5032 - val_accuracy: 0.4239
Epoch 34/50
42/42 [==============================] - 84s 2s/step - loss: 0.1245 - accuracy: 0.9964 - val_loss: 2.3729 - val_accuracy: 0.4348
Epoch 35/50
42/42 [==============================] - 84s 2s/step - loss: 0.1192 - accuracy: 0.9976 - val_loss: 2.3199 - val_accuracy: 0.4457
Epoch 36/50
42/42 [==============================] - 116s 3s/step - loss: 0.1233 - accuracy: 0.9976 - val_loss: 2.2593 - val_accuracy: 0.4565
Epoch 37/50
42/42 [==============================] - 125s 3s/step - loss: 0.1197 - accuracy: 0.9927 - val_loss: 2.3267 - val_accuracy: 0.4239
Epoch 38/50
42/42 [==============================] - 132s 3s/step - loss: 0.1233 - accuracy: 0.9976 - val_loss: 2.3858 - val_accuracy: 0.3696
Epoch 39/50
42/42 [==============================] - 125s 3s/step - loss: 0.1165 - accuracy: 0.9964 - val_loss: 2.3442 - val_accuracy: 0.4022
Epoch 40/50
42/42 [==============================] - 119s 3s/step - loss: 0.1101 - accuracy: 0.9952 - val_loss: 2.2392 - val_accuracy: 0.4674
Epoch 41/50
42/42 [==============================] - 109s 3s/step - loss: 0.1117 - accuracy: 0.9976 - val_loss: 2.3136 - val_accuracy: 0.4457
Epoch 42/50
42/42 [==============================] - 108s 3s/step - loss: 0.1098 - accuracy: 0.9976 - val_loss: 2.2327 - val_accuracy: 0.4239
Epoch 43/50
42/42 [==============================] - 102s 2s/step - loss: 0.1064 - accuracy: 0.9976 - val_loss: 2.2396 - val_accuracy: 0.4348
Epoch 44/50
42/42 [==============================] - 106s 3s/step - loss: 0.1075 - accuracy: 0.9976 - val_loss: 2.2344 - val_accuracy: 0.4457
Epoch 45/50
42/42 [==============================] - 103s 2s/step - loss: 0.1087 - accuracy: 0.9988 - val_loss: 2.2796 - val_accuracy: 0.4130
Epoch 46/50
42/42 [==============================] - 101s 2s/step - loss: 0.1006 - accuracy: 0.9976 - val_loss: 2.1885 - val_accuracy: 0.4239
Epoch 47/50
42/42 [==============================] - 104s 2s/step - loss: 0.0981 - accuracy: 0.9952 - val_loss: 2.0915 - val_accuracy: 0.4130
Epoch 48/50
42/42 [==============================] - 106s 3s/step - loss: 0.0888 - accuracy: 0.9988 - val_loss: 2.0734 - val_accuracy: 0.4348
Epoch 49/50
42/42 [==============================] - 87s 2s/step - loss: 0.0871 - accuracy: 0.9988 - val_loss: 2.1237 - val_accuracy: 0.4457
Epoch 50/50
42/42 [==============================] - 81s 2s/step - loss: 0.0902 - accuracy: 0.9976 - val_loss: 2.0451 - val_accuracy: 0.4348
"""

""" diff dataset, batch size = 5
Epoch 1/50
45/45 [==============================] - 29s 624ms/step - loss: 4.1847 - accuracy: 0.1306 - val_loss: 6.2354 - val_accuracy: 0.1200
Epoch 2/50
45/45 [==============================] - 29s 645ms/step - loss: 3.1860 - accuracy: 0.2342 - val_loss: 5.0885 - val_accuracy: 0.1200
Epoch 3/50
45/45 [==============================] - 29s 636ms/step - loss: 2.5605 - accuracy: 0.2928 - val_loss: 5.1760 - val_accuracy: 0.2000
Epoch 4/50
45/45 [==============================] - 28s 632ms/step - loss: 2.9117 - accuracy: 0.2838 - val_loss: 4.9634 - val_accuracy: 0.1600
Epoch 5/50
45/45 [==============================] - 28s 628ms/step - loss: 2.7474 - accuracy: 0.2793 - val_loss: 3.7681 - val_accuracy: 0.1600
Epoch 6/50
45/45 [==============================] - 28s 630ms/step - loss: 2.8372 - accuracy: 0.2432 - val_loss: 3.7164 - val_accuracy: 0.0000e+00
Epoch 7/50
45/45 [==============================] - 29s 636ms/step - loss: 2.7006 - accuracy: 0.2658 - val_loss: 4.5932 - val_accuracy: 0.0800
Epoch 8/50
45/45 [==============================] - 29s 642ms/step - loss: 2.3170 - accuracy: 0.3378 - val_loss: 4.3551 - val_accuracy: 0.1200
Epoch 9/50
45/45 [==============================] - 29s 640ms/step - loss: 3.7306 - accuracy: 0.1126 - val_loss: 7.3659 - val_accuracy: 0.1200
Epoch 10/50
45/45 [==============================] - 29s 634ms/step - loss: 3.2270 - accuracy: 0.1081 - val_loss: 4.6822 - val_accuracy: 0.0800
Epoch 11/50
45/45 [==============================] - 29s 637ms/step - loss: 2.5183 - accuracy: 0.2883 - val_loss: 4.1857 - val_accuracy: 0.0800
Epoch 12/50
45/45 [==============================] - 29s 647ms/step - loss: 2.5633 - accuracy: 0.2342 - val_loss: 3.7273 - val_accuracy: 0.1600
Epoch 13/50
45/45 [==============================] - 29s 638ms/step - loss: 2.2489 - accuracy: 0.3468 - val_loss: 3.6109 - val_accuracy: 0.0800
Epoch 14/50
45/45 [==============================] - 29s 634ms/step - loss: 2.3056 - accuracy: 0.3423 - val_loss: 3.3926 - val_accuracy: 0.0400
Epoch 15/50
45/45 [==============================] - 29s 644ms/step - loss: 2.0350 - accuracy: 0.3604 - val_loss: 3.7918 - val_accuracy: 0.1200
Epoch 16/50
45/45 [==============================] - 29s 640ms/step - loss: 1.8019 - accuracy: 0.4550 - val_loss: 4.3122 - val_accuracy: 0.2000
Epoch 17/50
45/45 [==============================] - 29s 643ms/step - loss: 1.7397 - accuracy: 0.4279 - val_loss: 3.6999 - val_accuracy: 0.0800
Epoch 18/50
45/45 [==============================] - 29s 637ms/step - loss: 1.5555 - accuracy: 0.5045 - val_loss: 3.5341 - val_accuracy: 0.1200
Epoch 19/50
45/45 [==============================] - 29s 642ms/step - loss: 1.3435 - accuracy: 0.5766 - val_loss: 3.7748 - val_accuracy: 0.1200
Epoch 20/50
45/45 [==============================] - 28s 629ms/step - loss: 1.4440 - accuracy: 0.6036 - val_loss: 12.6863 - val_accuracy: 0.0400
Epoch 21/50
45/45 [==============================] - 29s 637ms/step - loss: 1.5705 - accuracy: 0.5180 - val_loss: 7.7700 - val_accuracy: 0.0400
Epoch 22/50
45/45 [==============================] - 29s 635ms/step - loss: 1.5116 - accuracy: 0.5495 - val_loss: 3.8875 - val_accuracy: 0.2000
Epoch 23/50
45/45 [==============================] - 28s 632ms/step - loss: 1.4857 - accuracy: 0.5180 - val_loss: 6.4529 - val_accuracy: 0.1600
Epoch 24/50
45/45 [==============================] - 28s 630ms/step - loss: 1.4841 - accuracy: 0.5811 - val_loss: 5.8811 - val_accuracy: 0.1200
Epoch 25/50
45/45 [==============================] - 28s 630ms/step - loss: 1.4376 - accuracy: 0.5225 - val_loss: 5.5164 - val_accuracy: 0.0800
Epoch 26/50
45/45 [==============================] - 29s 634ms/step - loss: 1.4266 - accuracy: 0.5946 - val_loss: 4.1414 - val_accuracy: 0.1200
Epoch 27/50
45/45 [==============================] - 28s 625ms/step - loss: 1.2613 - accuracy: 0.6261 - val_loss: 3.6806 - val_accuracy: 0.1600
Epoch 28/50
45/45 [==============================] - 29s 636ms/step - loss: 1.0932 - accuracy: 0.7162 - val_loss: 4.0173 - val_accuracy: 0.1600
Epoch 29/50
45/45 [==============================] - 29s 640ms/step - loss: 1.0253 - accuracy: 0.7432 - val_loss: 3.5349 - val_accuracy: 0.1600
Epoch 30/50
45/45 [==============================] - 29s 633ms/step - loss: 1.4472 - accuracy: 0.5946 - val_loss: 3.6608 - val_accuracy: 0.1200
Epoch 31/50
45/45 [==============================] - 28s 628ms/step - loss: 1.3832 - accuracy: 0.5856 - val_loss: 3.6980 - val_accuracy: 0.1200
Epoch 32/50
45/45 [==============================] - 28s 630ms/step - loss: 1.3372 - accuracy: 0.5991 - val_loss: 4.3951 - val_accuracy: 0.1200
Epoch 33/50
45/45 [==============================] - 29s 635ms/step - loss: 1.3063 - accuracy: 0.5946 - val_loss: 3.7389 - val_accuracy: 0.1200
Epoch 34/50
45/45 [==============================] - 28s 626ms/step - loss: 1.1026 - accuracy: 0.6937 - val_loss: 3.9524 - val_accuracy: 0.1200
Epoch 35/50
45/45 [==============================] - 28s 629ms/step - loss: 1.0737 - accuracy: 0.6982 - val_loss: 3.7030 - val_accuracy: 0.0800
Epoch 36/50
45/45 [==============================] - 28s 629ms/step - loss: 0.8963 - accuracy: 0.7703 - val_loss: 3.8945 - val_accuracy: 0.1200
Epoch 37/50
45/45 [==============================] - 28s 632ms/step - loss: 0.9317 - accuracy: 0.7658 - val_loss: 3.6443 - val_accuracy: 0.1600
Epoch 38/50
45/45 [==============================] - 28s 620ms/step - loss: 0.9589 - accuracy: 0.7432 - val_loss: 3.7341 - val_accuracy: 0.1200
Epoch 39/50
45/45 [==============================] - 29s 637ms/step - loss: 0.8609 - accuracy: 0.7928 - val_loss: 3.7549 - val_accuracy: 0.0800
Epoch 40/50
45/45 [==============================] - 28s 624ms/step - loss: 0.9652 - accuracy: 0.7252 - val_loss: 3.5836 - val_accuracy: 0.1200
Epoch 41/50
45/45 [==============================] - 28s 622ms/step - loss: 0.8177 - accuracy: 0.8018 - val_loss: 3.7400 - val_accuracy: 0.0800
Epoch 42/50
45/45 [==============================] - 28s 621ms/step - loss: 0.7369 - accuracy: 0.8288 - val_loss: 3.9104 - val_accuracy: 0.1600
Epoch 43/50
45/45 [==============================] - 28s 622ms/step - loss: 0.7365 - accuracy: 0.8423 - val_loss: 3.9042 - val_accuracy: 0.1600
Epoch 44/50
45/45 [==============================] - 28s 619ms/step - loss: 0.6947 - accuracy: 0.8423 - val_loss: 3.8607 - val_accuracy: 0.1600
Epoch 45/50
45/45 [==============================] - 28s 617ms/step - loss: 0.6307 - accuracy: 0.8559 - val_loss: 3.8024 - val_accuracy: 0.2000
Epoch 46/50
45/45 [==============================] - 28s 625ms/step - loss: 0.7368 - accuracy: 0.8198 - val_loss: 4.0157 - val_accuracy: 0.1200
Epoch 47/50
45/45 [==============================] - 28s 620ms/step - loss: 0.6416 - accuracy: 0.8423 - val_loss: 4.0556 - val_accuracy: 0.2000
Epoch 48/50
45/45 [==============================] - 28s 624ms/step - loss: 0.5836 - accuracy: 0.8739 - val_loss: 3.9646 - val_accuracy: 0.2000
Epoch 49/50
45/45 [==============================] - 28s 627ms/step - loss: 0.5766 - accuracy: 0.8919 - val_loss: 4.0666 - val_accuracy: 0.0800
Epoch 50/50
45/45 [==============================] - 28s 629ms/step - loss: 0.5426 - accuracy: 0.8604 - val_loss: 4.3436 - val_accuracy: 0.1200
"""

""" Dropout 0.5 for FC, L2 regularizer for FC
Epoch 1/12
40/40 [==============================] - 96s 2s/step - loss: 3.5790 - accuracy: 0.1758 - val_loss: 8.3838 - val_accuracy: 0.0978
Epoch 2/12
40/40 [==============================] - 99s 2s/step - loss: 2.8954 - accuracy: 0.2655 - val_loss: 7.5236 - val_accuracy: 0.0978
Epoch 3/12
40/40 [==============================] - 100s 2s/step - loss: 2.2299 - accuracy: 0.3564 - val_loss: 9.9510 - val_accuracy: 0.0978
Epoch 4/12
40/40 [==============================] - 96s 2s/step - loss: 1.7092 - accuracy: 0.5212 - val_loss: 12.3147 - val_accuracy: 0.0978
Epoch 5/12
40/40 [==============================] - 94s 2s/step - loss: 1.5641 - accuracy: 0.5600 - val_loss: 10.2058 - val_accuracy: 0.0978
Epoch 6/12
40/40 [==============================] - 91s 2s/step - loss: 1.3232 - accuracy: 0.6170 - val_loss: 7.2878 - val_accuracy: 0.0978
Epoch 7/12
40/40 [==============================] - 99s 2s/step - loss: 0.9565 - accuracy: 0.7455 - val_loss: 7.7198 - val_accuracy: 0.0978
Epoch 8/12
40/40 [==============================] - 86s 2s/step - loss: 0.7111 - accuracy: 0.8194 - val_loss: 4.4762 - val_accuracy: 0.2283
Epoch 9/12
40/40 [==============================] - 90s 2s/step - loss: 0.8209 - accuracy: 0.7915 - val_loss: 5.0805 - val_accuracy: 0.1522
Epoch 10/12
40/40 [==============================] - 87s 2s/step - loss: 0.5628 - accuracy: 0.8727 - val_loss: 4.2604 - val_accuracy: 0.2609
Epoch 11/12
40/40 [==============================] - 86s 2s/step - loss: 0.4160 - accuracy: 0.9188 - val_loss: 4.2665 - val_accuracy: 0.2826
Epoch 12/12
40/40 [==============================] - 94s 2s/step - loss: 0.8775 - accuracy: 0.7697 - val_loss: 3.5591 - val_accuracy: 0.1957
"""

""" Dropout 0.5 for FC, L1 regularizer for FC
Epoch 1/12
40/40 [==============================] - 102s 3s/step - loss: 6.2964 - accuracy: 0.1430 - val_loss: 6.5603 - val_accuracy: 0.2935
Epoch 2/12
40/40 [==============================] - 90s 2s/step - loss: 4.3839 - accuracy: 0.3758 - val_loss: 6.2841 - val_accuracy: 0.1196
Epoch 3/12
40/40 [==============================] - 97s 2s/step - loss: 3.8433 - accuracy: 0.4145 - val_loss: 8.3931 - val_accuracy: 0.0543
Epoch 4/12
40/40 [==============================] - 97s 2s/step - loss: 3.1487 - accuracy: 0.5527 - val_loss: 6.0627 - val_accuracy: 0.0543
Epoch 5/12
40/40 [==============================] - 95s 2s/step - loss: 2.4094 - accuracy: 0.7152 - val_loss: 9.7366 - val_accuracy: 0.0652
Epoch 6/12
40/40 [==============================] - 93s 2s/step - loss: 2.0191 - accuracy: 0.8218 - val_loss: 5.4020 - val_accuracy: 0.0761
Epoch 7/12
40/40 [==============================] - 95s 2s/step - loss: 1.6472 - accuracy: 0.9006 - val_loss: 4.7042 - val_accuracy: 0.1630
Epoch 8/12
40/40 [==============================] - 98s 2s/step - loss: 1.3700 - accuracy: 0.9406 - val_loss: 5.0927 - val_accuracy: 0.0870
Epoch 9/12
40/40 [==============================] - 98s 2s/step - loss: 1.1542 - accuracy: 0.9782 - val_loss: 4.1463 - val_accuracy: 0.1630
Epoch 10/12
40/40 [==============================] - 98s 2s/step - loss: 1.0162 - accuracy: 0.9830 - val_loss: 3.7576 - val_accuracy: 0.1087
Epoch 11/12
40/40 [==============================] - 91s 2s/step - loss: 0.8829 - accuracy: 0.9842 - val_loss: 3.2439 - val_accuracy: 0.2283
Epoch 12/12
40/40 [==============================] - 93s 2s/step - loss: 0.7962 - accuracy: 0.9891 - val_loss: 2.9093 - val_accuracy: 0.2065
"""

""" Dropout 0.5 for FC
Epoch 1/12
40/40 [==============================] - 100s 2s/step - loss: 3.3981 - accuracy: 0.1903 - val_loss: 5.6718 - val_accuracy: 0.0870
Epoch 2/12
40/40 [==============================] - 94s 2s/step - loss: 2.3932 - accuracy: 0.3176 - val_loss: 8.4275 - val_accuracy: 0.0870
Epoch 3/12
40/40 [==============================] - 89s 2s/step - loss: 1.9557 - accuracy: 0.4121 - val_loss: 12.0641 - val_accuracy: 0.0870
Epoch 4/12
40/40 [==============================] - 91s 2s/step - loss: 1.6068 - accuracy: 0.4824 - val_loss: 16.1905 - val_accuracy: 0.0870
Epoch 5/12
40/40 [==============================] - 90s 2s/step - loss: 1.1508 - accuracy: 0.6327 - val_loss: 10.1111 - val_accuracy: 0.0870
Epoch 6/12
40/40 [==============================] - 101s 3s/step - loss: 0.7522 - accuracy: 0.7745 - val_loss: 8.5945 - val_accuracy: 0.0870
Epoch 7/12
40/40 [==============================] - 101s 3s/step - loss: 0.5170 - accuracy: 0.8352 - val_loss: 7.0591 - val_accuracy: 0.1304
Epoch 8/12
40/40 [==============================] - 97s 2s/step - loss: 0.2826 - accuracy: 0.9139 - val_loss: 4.8786 - val_accuracy: 0.1413
Epoch 9/12
40/40 [==============================] - 101s 3s/step - loss: 0.6538 - accuracy: 0.7939 - val_loss: 8.5924 - val_accuracy: 0.1087
Epoch 10/12
40/40 [==============================] - 99s 2s/step - loss: 0.3743 - accuracy: 0.8776 - val_loss: 3.5024 - val_accuracy: 0.2174
Epoch 11/12
40/40 [==============================] - 94s 2s/step - loss: 0.1940 - accuracy: 0.9479 - val_loss: 3.7893 - val_accuracy: 0.2609
Epoch 12/12
40/40 [==============================] - 100s 3s/step - loss: 0.1158 - accuracy: 0.9721 - val_loss: 3.4711 - val_accuracy: 0.3152
"""

""" with everything
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

""" using data augmentation
Epoch 1/25
30/30 [==============================] - 79s 3s/step - loss: 3.3358 - accuracy: 0.1375
Epoch 2/25
30/30 [==============================] - 77s 3s/step - loss: 2.9753 - accuracy: 0.1952
Epoch 3/25
30/30 [==============================] - 75s 2s/step - loss: 2.8321 - accuracy: 0.2455
Epoch 4/25
30/30 [==============================] - 76s 3s/step - loss: 2.4519 - accuracy: 0.3143
Epoch 5/25
30/30 [==============================] - 75s 2s/step - loss: 2.3963 - accuracy: 0.2782
Epoch 6/25
30/30 [==============================] - 68s 2s/step - loss: 2.2482 - accuracy: 0.3224
Epoch 7/25
30/30 [==============================] - 67s 2s/step - loss: 2.1265 - accuracy: 0.3715
Epoch 8/25
30/30 [==============================] - 67s 2s/step - loss: 2.6458 - accuracy: 0.2602
Epoch 9/25
30/30 [==============================] - 70s 2s/step - loss: 2.3523 - accuracy: 0.3143
Epoch 10/25
30/30 [==============================] - 73s 2s/step - loss: 2.1753 - accuracy: 0.3142
Epoch 11/25
30/30 [==============================] - 74s 2s/step - loss: 2.1304 - accuracy: 0.3486
Epoch 12/25
30/30 [==============================] - 74s 2s/step - loss: 2.0085 - accuracy: 0.3666
Epoch 13/25
30/30 [==============================] - 75s 2s/step - loss: 2.1492 - accuracy: 0.3273
Epoch 14/25
30/30 [==============================] - 74s 2s/step - loss: 2.0714 - accuracy: 0.3535
Epoch 15/25
30/30 [==============================] - 75s 2s/step - loss: 1.9189 - accuracy: 0.3846
Epoch 16/25
30/30 [==============================] - 71s 2s/step - loss: 1.8842 - accuracy: 0.3846
Epoch 17/25
30/30 [==============================] - 75s 2s/step - loss: 1.7901 - accuracy: 0.4255
Epoch 18/25
30/30 [==============================] - 87s 3s/step - loss: 1.8360 - accuracy: 0.4175
Epoch 19/25
30/30 [==============================] - 79s 3s/step - loss: 1.6737 - accuracy: 0.4603
Epoch 20/25
30/30 [==============================] - 73s 2s/step - loss: 1.4842 - accuracy: 0.5041
Epoch 21/25
30/30 [==============================] - 87s 3s/step - loss: 1.4555 - accuracy: 0.5041
Epoch 22/25
30/30 [==============================] - 85s 3s/step - loss: 1.6639 - accuracy: 0.4681
Epoch 23/25
30/30 [==============================] - 83s 3s/step - loss: 1.4734 - accuracy: 0.5123
Epoch 24/25
30/30 [==============================] - 85s 3s/step - loss: 1.3615 - accuracy: 0.5333
Epoch 25/25
30/30 [==============================] - 76s 3s/step - loss: 1.4449 - accuracy: 0.5205
"""