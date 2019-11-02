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
from sklearn import metrics
from keras import backend as K
# import tensorflow.python.keras.backend as K (for tensorflow 2.0)

import os

directory = "C:/Users/Evan/Documents/S2/deep learning/project/data/train/"
listNeed = os.listdir(directory)
listNeed = list(filter(lambda k: '.npy' in k, listNeed))
listNeed.sort(key= lambda x: float(x.strip('.npy')))
data = {} # initialize data
print("Start 0 pad train")
for filename in listNeed: # do not hardcode! hardcoding will cause the code to not work if the number of samples is different!
    if filename.endswith(".npy"):
        tempFileName = "C:/Users/Evan/Documents/S2/deep learning/project/data/train/" + filename # load each file        
        print(filename) # print the name file
        value = np.load(tempFileName) # load 1000,102 matrix
        value_columns = value.shape[0]
        if value_columns < 1000:
            padding = [[0 for i in range(102)] for j in range(1000 - value_columns)]
            value = np.concatenate((value, padding), axis = 0) # pad the matrix
        index = int(filename.strip('.npy')) # get the numerical file name
        data[index] = value # add to the data

data = np.array(list(data.values()))

print("Finish 0 pad train")
print("This is printing train data shape. It should be 18662 * 1000 * 102")
print(data.shape)

testdirectory = "C:/Users/Evan/Documents/S2/deep learning/project/data/test/"
testlistNeed = os.listdir(testdirectory)
testlistNeed = list(filter(lambda k: '.npy' in k, testlistNeed))
testlistNeed.sort(key= lambda x: float(x.strip('.npy')))
test = {} # initialize data
print("Start 0 pad test")
for filename in testlistNeed: # do not hardcode! hardcoding will cause the code to not work if the number of samples is different!
    if filename.endswith(".npy"):
        tempFileName = "C:/Users/Evan/Documents/S2/deep learning/project/data/test/" + filename # load each file        
        print(filename) # print the name file
        testvalue = np.load(tempFileName) # load 1000,102 matrix
        testvalue_columns = testvalue.shape[0]
        if testvalue_columns < 1000:
            testpadding = [[0 for i in range(102)] for j in range(1000 - testvalue_columns)]
            testvalue = np.concatenate((testvalue, testpadding), axis = 0) # pad the matrix
        testindex = int(filename.strip('.npy')) # get the numerical file name
        test[testindex] = testvalue # add to the data

test = np.array(list(test.values()))

print("Finish 0 pad test")
print("This is printing test data shape. It should be 6051 * 1000 * 102")
print(test.shape)

#TODO (to save time): save the 0-padded train and test data so that it can be reloaded

#data = data[:,:,:92]
#print("This is printing feature selected (hashtricking) train data shape. It should be 18662 * 1000 * 92")
#print(data.shape)
print("Start to retrieve train labels")
labels = pd.read_csv("C:/Users/Evan/Documents/S2/deep learning/project/train_kaggle.csv")
labels = labels.drop(labels.columns[[0]], axis = 1).to_numpy()
print("Finish to retrieve train labels")

#OPTIONAL TODO: add train-test split for validation of the model (80%:20%)

#to use auc
def auc(actual, predict):
#    auc = tf.compat.v1.metrics.auc(actual, predict)[1] # for tensorflow 2.0
    auc = tf.metrics.auc(actual, predict)[1] # for tensorflow 1.x
    K.get_session().run(tf.local_variables_initializer())
    return auc

print("Adding model")
model = Sequential() # to be able to add several models at once
model.add(BatchNormalization()) # do batch normalization first
model.add(Conv1D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu')) # conv1d here, try power of 2 for filters (32, 64, 128), try 2, 3, 4 for kernel_size, compare same vs valid padding, may try conv2d also for comparison
#model.add(Bidirectional(LSTM(128, return_sequences=True))) # this is BLSTM. Try a "good number" (50, 100, 200) or "power of 2" (32, 64, 128, 256) to compare and see, try comment this layer out and compare and see, return full sequences here
model.add(GlobalMaxPooling1D()) # do global max pooling, for conv2d make sure to change 1d to 2d! (global max pooling takes no parameters), or try max pooling which has parameters (pool_size, default/recommended:2)
model.add(Dense(256, activation='relu')) # fully connected with relu (try with powers of 2 or some other good numbers)
model.add(Dropout(0.5)) # dropout the layers. Change appropiately if you have time and attempts.
model.add(Dense(1, activation='sigmoid')) # fully connected with sigmoid (to cover some decimals), to 1 because we are dealing with a single number for the target values (technically this is a binary classification whether a file is malware or not)
print("Finish adding model")
myadam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False) # this is the Adam optimizer
print("Compile model")
model.compile(loss='binary_crossentropy', optimizer=myadam, metrics=[auc]) # using binary cross-entropy loss (since it is a binary classification) and the Adam optimizer stated above, use AUC (optional, but strongly recommended) for determining quality of the learner (consistent with Kaggle)
print("Finish compile model. Now fit model")
model.fit(data, labels, epochs=10, batch_size=64, verbose=2) # batch_size is recommended to be in the power of 2
print("Finish fit model. Now predict model")
results = model.predict(test, batch_size=64, verbose=1) # test it
print("Finish predict model. Now saving to csv")
results_df = pd.DataFrame(results, columns=['Predicted']) # kaggle format
results_df.to_csv('results.csv', index=True, index_label='Id') # save for Kaggle submission :)
print("Everything is done!")
