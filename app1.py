# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:28:25 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = (80, 3, 1)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from keras.models import save_model
save_model(model,'best_model1.hdf5')

from keras.models import load_model
model.load_weights('best_model.hdf5')


label = joblib.load("LabelEncoder.pickle") 

scaler = joblib.load("StandardScaler.pickle") 
scaler.mean_

x1=np.asarray(X["x"])
y1=np.asarray(X["y"])
z1=np.asarray(X["z"])

import statistics
statistics.stdev(x1)
statistics.stdev(y1)
statistics.stdev(z1)

converter = tf.lite.TFLiteConverter.from_keras_model_file('best_model2.hdf5')
tflite_model = converter.convert()
open("converted_model1.tflite", "wb").write(tflite_model)


