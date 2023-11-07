# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
#!pip install imutils

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from imutils.contours import sort_contours
import imutils
import scipy

from tensorflow.keras.models import Sequential  # Tensorflow의 keras로 변경
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten  # Tensorflow의 keras로 변경
from tensorflow.keras.optimizers import Adam  # Tensorflow의 keras로 변경
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.25
)

data_path = 'input/handwritten-digits-and-operators/CompleteImages/All data (Compressed)'
train_set = train_datagen.flow_from_directory(
    data_path,
    target_size=(40, 40),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=123
)
valid_set = train_datagen.flow_from_directory(
    data_path,
    target_size=(40, 40),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=123
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 40, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(17, activation='softmax'))

# compile model
adam = Adam(learning_rate=5e-4)  # lr -> learning_rate로 변경
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_set, validation_data=valid_set, epochs=5, verbose=1)

val_loss, val_accuracy = model.evaluate(valid_set)
print(val_loss, val_accuracy)

model.save("hbec_model.h5")



