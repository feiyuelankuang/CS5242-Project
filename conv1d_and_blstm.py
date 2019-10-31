import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D

import os

names = []
directory = "/home/e/evan2133/cs5242project/train/train/"
listNeed = os.listdir(directory)
listNeed = list(filter(lambda k: '.npy' in k, listNeed))
listNeed.sort(key= lambda x: float(x.strip('.npy')))
pad = np.zeros([1000,102]) # for 0 padding
data = np.zeros([0,1000,102]) # initialize data
print("Start 0 pad")
for filename in listNeed:
    if filename.endswith(".npy"):
        tempFileName = "/home/e/evan2133/cs5242project/train/train/" + filename
        value = np.load(tempFileName) # load 1000,102 matrix
        value_pad = value + pad # pad it, this is allowed due to broadcasting
        value_pad = value_pad.reshape(1, 1000, 102) # reshape for np.concatenate
        data = np.concatenate((data, value_pad), axis=0)

print("Finish 0 pad")
print("This is printing data.Shape. It should be 18662 * 1000 * 102")
print(data.shape)

labels = pd.read_csv("/home/e/evan2133/cs5242project/train_kaggle.csv")
labels = labels.drop(labels.columns[[0]], axis = 1)

model = Sequential() # to be able to add several models at once
# removed this line. Given the input shape is (18662, 1000, 102), for extracting the first 92 columns (out of 102) for hashtricks embedding/feature selection, please use tf.slice(input, [0, 0, 0], [-1, -1, 92]) for tensorflow 3d tensor or input[:,:,:92] for numpy 3d array
model.add(BatchNormalization()) # do batch normalization ??
model.add(Conv1D(filters=64, kernel_size=2, stride=1, padding='same', activation='relu')) # conv1d here, try power of 2 for filters (32, 64, 128), try 2, 3, 4 for kernel_size, may try conv2d also for comparison
model.add(Bidirectional(LSTM(128, return_sequences=True))) # this is BLSTM. Try a "good number" (50, 100, 200) or "power of 2" (32, 64, 128, 256) to compare and see, try comment it out and compare and see, return full sequences here
model.add(GlobalMaxPooling1D(pool_size=2)) # do global max pooling (try 2, 3, 4 for pool size)
model.add(Dense(256, activation='relu')) # fully connected with relu (try with powers of 2 or some other good numbers)
model.add(Dropout(0.5)) # want to add dropout??
model.add(Dense(1, activation='sigmoid')) # fully connected with sigmoid
