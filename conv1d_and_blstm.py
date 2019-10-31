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
from keras import optimizers
from keras import metrics

import os

directory = "/home/e/evan2133/cs5242project/train/train/"
listNeed = os.listdir(directory)
listNeed = list(filter(lambda k: '.npy' in k, listNeed))
listNeed.sort(key= lambda x: float(x.strip('.npy')))
data_raw = {} # initialize data
print("Start 0 pad train")
for filename in listNeed: # do not hardcode! hardcoding will cause the code to not work if the number of samples is different!
    if filename.endswith(".npy"):
        tempFileName = "/home/e/evan2133/cs5242project/train/train/" + filename # load each file        
        print(filename) # print the name file
        value = np.load(tempFileName) # load 1000,102 matrix
        value_columns = value.shape[0]
        if value_columns < 1000:
            padding = [[0 for i in range(102)] for j in range(1000 - value_columns)]
        value_pad = np.concatenate((value, padding), axis = 0) # pad the matrix
        index = int(filename.strip('.npy')) # get the numerical file name
        data_raw[index] = value_pad # add to the data

data = np.array(list(data_raw.values()))

print("Finish 0 pad train")
print("This is printing train data shape. It should be 18662 * 1000 * 102")
print(data.shape)

testdirectory = "/home/e/evan2133/cs5242project/test/test/"
testlistNeed = os.listdir(testdirectory)
testlistNeed = list(filter(lambda k: '.npy' in k, testlistNeed))
testlistNeed.sort(key= lambda x: float(x.strip('.npy')))
test_raw = {} # initialize data
print("Start 0 pad test")
for filename in testlistNeed: # do not hardcode! hardcoding will cause the code to not work if the number of samples is different!
    if filename.endswith(".npy"):
        tempFileName = "/home/e/evan2133/cs5242project/test/test/" + filename # load each file        
        print(filename) # print the name file
        testvalue = np.load(tempFileName) # load 1000,102 matrix
        testvalue_columns = testvalue.shape[0]
        if testvalue_columns < 1000:
            testpadding = [[0 for i in range(102)] for j in range(1000 - testvalue_columns)]
        testvalue_pad = np.concatenate((testvalue, testpadding), axis = 0) # pad the matrix
        testindex = int(filename.strip('.npy')) # get the numerical file name
        test_raw[index] = value_pad # add to the data

test = np.array(list(test_raw.values()))

print("Finish 0 pad test")
print("This is printing test data shape. It should be 6051 * 1000 * 102")
print(test.shape)

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
myadam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) # this is the Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=myadam) # using binary cross-entropy loss (since it is a binary classification) and the Adam optimizer stated above
model.fit(data, labels, epochs=1000, batch_size=64) # batch_size is recommended to be in the power of 2
model.predict(testdata, batch_size=64) # test it
