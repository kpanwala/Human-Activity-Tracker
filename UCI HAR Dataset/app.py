# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 22:34:33 2020

@author: Admin
"""

import pandas as pd
import numpy as np

acc_x=pd.read_table("total_acc_x_train.txt",delim_whitespace=True,header=None)
acc_y=pd.read_table("total_acc_y_train.txt",delim_whitespace=True,header=None)
acc_z=pd.read_table("total_acc_z_train.txt",delim_whitespace=True,header=None)

gyro_x=pd.read_table("body_gyro_x_train.txt",delim_whitespace=True,header=None)
gyro_y=pd.read_table("body_gyro_y_train.txt",delim_whitespace=True,header=None)
gyro_z=pd.read_table("body_gyro_z_train.txt",delim_whitespace=True,header=None)

y=pd.read_table("../y_train.txt",delim_whitespace=True,header=None)

acc_x1=np.asarray(acc_x)
acc_x1=acc_x1.reshape(-1,1)

from array import array 

#  *****
acc_x2=[]
for i in range(0,len(acc_x1)):
    if int(i/64)%2==0:
        acc_x2.append(float(acc_x1[i]))

ac_x=np.asarray(acc_x2)
ac_x=ac_x.reshape(-1,1)

#  *****

acc_y1=np.asarray(acc_y)
acc_y1=acc_y1.reshape(-1,1)

acc_y2=[]
for i in range(0,len(acc_y1)):
    if int(i/64)%2==0:
        acc_y2.append(float(acc_y1[i]))

ac_y=np.asarray(acc_y2)
ac_y=ac_y.reshape(-1,1)

#  ******

acc_z1=np.asarray(acc_z)
acc_z1=acc_z1.reshape(-1,1)

acc_z2=[]
for i in range(0,len(acc_z1)):
    if int(i/64)%2==0:
        acc_z2.append(float(acc_z1[i]))
        
ac_z=np.asarray(acc_z2)
ac_z=ac_z.reshape(-1,1)

#  ******

gyro_x1=np.asarray(gyro_x)
gyro_x1=gyro_x1.reshape(-1,1)

gyro_x2=[]
for i in range(0,len(gyro_x1)):
    if int(i/64)%2==0:
        gyro_x2.append(float(gyro_x1[i]))
        
gy_x=np.asarray(gyro_x2)
gy_x=gy_x.reshape(-1,1)

#  ****

gyro_y1=np.asarray(gyro_y)
gyro_y1=gyro_y1.reshape(-1,1)

gyro_y2=[]
for i in range(0,len(gyro_y1)):
    if int(i/64)%2==0:
        gyro_y2.append(float(gyro_y1[i]))
        
gy_y=np.asarray(gyro_y2)
gy_y=gy_y.reshape(-1,1)

#  ****

gyro_z1=np.asarray(gyro_z)
gyro_z1=gyro_z1.reshape(-1,1)

gyro_z2=[]
for i in range(0,len(gyro_z1)):
    if int(i/64)%2==0:
        gyro_z2.append(float(gyro_z1[i]))
        
gy_z=np.asarray(gyro_z2)
gy_z=gy_z.reshape(-1,1)

# *** 
data=pd.DataFrame(ac_x)
data=pd.concat([pd.DataFrame(ac_x),pd.DataFrame(ac_y),pd.DataFrame(ac_z),pd.DataFrame(gy_x),pd.DataFrame(gy_y),pd.DataFrame(gy_z)],axis=1)
data.columns=["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

data["activity"]=0

for j in range(0,7352):
    data["activity"][j*64:(j+1)*64]=y[0][j]

frame_size = 64

import scipy.stats as stats
def get_frames(df, frame_size):

    N_FEATURES = 6

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size,64):
        acc_x = df['acc_x'].values[i: i + frame_size]
        acc_y = df['acc_y'].values[i: i + frame_size]
        acc_z = df['acc_z'].values[i: i + frame_size]
        
        gyro_x = df['gyro_x'].values[i: i + frame_size]
        gyro_y = df['gyro_y'].values[i: i + frame_size]
        gyro_z = df['gyro_z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['activity'][i: i + frame_size])[0][0]
        frames.append([acc_x, acc_y, acc_z,gyro_x,gyro_y,gyro_z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(data, frame_size)
y=y.reshape(-1,1)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

X_train.shape, X_test.shape

X_train[0].shape, X_test[0].shape

X_train = X_train.reshape(5880, 64, 6, 1)
X_test = X_test.reshape(1471, 64, 6, 1)

from keras.callbacks import EarlyStopping, ModelCheckpoint

#del(model)
model = Sequential()
model.add(Conv2D(16, (2, 2), activation = 'relu', input_shape = (64 , 6, 1)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(X_train, y_train ,epochs=50, callbacks=[mc], validation_data=(X_test,y_test))