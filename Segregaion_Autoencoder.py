# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 11:21:44 2018

@author: SmartGridData
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:21:25 2017

@author: SmartGridData
"""

"============================================================================="
" Code for compressing data with autoencoder with ONE layer"
"============================================================================="

import numpy as np
import keras as kr
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from sklearn import metrics


"============================================================================="
"============================= ARCHITECTURE =================================="
"============================================================================="

#inp = kr.layers.Input(shape=(299,1))
#
#Enc1 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
#                                      padding = "valid", activation = 'relu')(inp)
#
#P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)
#
##Enc2 = kr.layers.convolutional.Conv1D(filters = 16, kernel_size = 15,
##                                      padding = "valid", activation = "relu")(P1)
##
##P2 = kr.layers.pooling.MaxPool1D(pool_size=5, strides =a 3, padding="valid")(Enc2)
#
#Enc3 = kr.layers.convolutional.Conv1D(filters = 8, kernel_size = 9,
#                                      padding = "valid", activation = 'relu')(P1) 
#
#P3 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc3)
#
#F1 = kr.layers.Flatten()(P3)
#
#D1 = kr.layers.Dense(10, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(F1)
#
#D1 = kr.layers.Dropout(rate=0.2)(D1)
#
#D2 = kr.layers.Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform')(D1)
#
#Enc4 = kr.layers.convolutional.Conv1D(filters = 64, kernel_size = 15,
#                                      padding = "valid", activation = "relu")(P1)
#Enc4=kr.layers.BatchNormalization(momentum=0.8)(Enc4)
#
#P4 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc4)
#
#Enc5 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
#                                      padding = "valid", activation = 'relu')(P4) 
#
#Enc5=kr.layers.BatchNormalization(momentum=0.8)(Enc5)
#
#P5 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc5)
#
#F2 = kr.layers.Flatten()(P5)
#
#D3 = kr.layers.Dense(200, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(F2)
#
#D3=kr.layers.BatchNormalization(momentum=0.8)(D3)
#
#D3 = kr.layers.Dropout(rate=0.2)(D3)
#
#D4 = kr.layers.Dense(50, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(D3)
##D3 = kr.layers.advanced_activations.PReLU(alpha_initializer='glorot_uniform')(D3)
#D4 = kr.layers.LeakyReLU(alpha=0.1)(D4)
#
#Df = kr.layers.concatenate([D2, D4], axis=-1)
#
#
#m = kr.models.Model(inp, output=[D2, D4])

#%%  Architecture 2

inp = kr.layers.Input(shape=(299,1))

Enc1 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 20,
                                      padding = "valid", activation = 'relu')(inp)

P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)

Enc4 = kr.layers.convolutional.Conv1D(filters = 16, kernel_size = 15,
                                      padding = "valid", activation = "relu")(P1)
Enc4=kr.layers.BatchNormalization(momentum=0.8)(Enc4)

P4 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc4)

UpSamp1=kr.layers.UpSampling1D(size=2)(P4)

Enc5 = kr.layers.convolutional.Conv1D(filters = 16, kernel_size = 20,
                                      padding = "valid", activation = 'relu')(UpSamp1) 

Enc5=kr.layers.BatchNormalization(momentum=0.8)(Enc5)

UpSamp2=kr.layers.UpSampling1D(size=2)(Enc5)

Enc6=kr.layers.convolutional.Conv1D(filters=5, kernel_size=15, padding="valid", activation="relu")(UpSamp2)

Enc6=kr.layers.BatchNormalization(momentum=0.8)(Enc6)

#UpSamp3=kr.layers.UpSampling1D(size=2)(P4)

F2 = kr.layers.Flatten()(Enc6)

F2=kr.layers.Reshape((240,1))(F2)

P1 = kr.layers.pooling.MaxPool1D(pool_size=93, strides = 3, padding="valid")(F2)

#
#D3 = kr.layers.Dense(150, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(F2)
#
#D3=kr.layers.BatchNormalization(momentum=0.8)(D3)
#
#D3 = kr.layers.Dropout(rate=0.3)(D3)

#D4 = kr.layers.Dense(50, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(D3)
#D3 = kr.layers.advanced_activations.PReLU(alpha_initializer='glorot_uniform')(D3)
D4 = kr.layers.LeakyReLU(alpha=0.1)(P1)

D4=kr.layers.Reshape((50,))(D4)
#Df = kr.layers.concatenate([D2, D4], axis=-1)

m = kr.models.Model(inp, output= D4)

#%% 

"============================================================================="
"=============================== TRAINING ===================================="
"============================================================================="

"loss function and compilation"
def NILM_Loss(y_true,y_pred):
    
#    s = tf.shape(y_true)
#    y_class = K.reshape(y_true[:,0],[s[0],1])
#    temp = tf.cast(tf.equal(y_class,1), tf.int32)
#    y_true1 = y_true[temp,1:s[1]]
#    y_pred1 = y_pred[temp,1:s[1]]
#    
#    first_log = K.log(K.clip(y_pred1, K.epsilon(), None) + 1.) # mean square log error
#    second_log = K.log(K.clip(y_true1, K.epsilon(), None) + 1.)
#    RAE_Loss= K.mean(K.abs(first_log - second_log), axis=-1)
    
    s = tf.shape(y_true)
    y_class = tf.cast(tf.reshape(y_true[:,0],[s[0],1]), dtype=tf.float32)
    y_true = y_true[:,1:s[1]]
    y_pred = y_pred[:,1:s[1]]
    y_err = tf.abs(y_true - y_pred) #/ tf.clip_by_value(y_true, 1, 10000000)
    temp1 = tf.multiply(y_err, y_class)

#    RAE_Loss = tf.reduce_sum(temp1, axis=-1) / tf.cast((tf.count_nonzero(y_class)*50),tf.float32)
    
    temp2=tf.reduce_sum(temp1, axis=1)
    RAE_Loss = tf.reduce_sum(temp2, axis=0) / tf.cast((tf.count_nonzero(y_class)), tf.float32)
    return RAE_Loss

opt = kr.optimizers.adam(lr=0.001, decay=1e-7)
#opt=kr.optimizers.adagrad()
#m.compile(optimizer = opt, loss = ['binary_crossentropy', 'mae'], metrics = ['accuracy'])
m.compile(optimizer = opt, loss = 'mae', metrics = ['accuracy'])


#m.compile(optimizer=opt, loss=['binary_crossentropy','mae'])

"saving options"
FileLoc = './Models/Kettle/MAE/'
ModelName = 'Segnet_MAE'
ModelCheckPointer = ModelCheckpoint(filepath=FileLoc+ModelName+'{epoch:02d}-{val_loss:.2f}',
                                  monitor='val_loss',
                                  verbose=0,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=1)

CSVFileName = 'Training_1Layer'+ModelName+'.csv'
csv_logger = CSVLogger(filename=FileLoc+CSVFileName, separator=',', append=False)

"read data"
x_train = np.load('./Data/Processed/XTrainKettle.npy')
x_val = np.load('./Data/Processed/XValKettle.npy')
x_test = np.load('./Data/Processed/XTestKettle.npy')

x_train = np.diff(x_train)
x_val = np.diff(x_val)
x_test = np.diff(x_test)

# Normalizing data 

#Mean=np.reshape(np.mean(x_train,axis=1),(len(x_train),1))
#Std=np.reshape(np.std(x_train,axis=1),(len(x_train),1))
#x_train=(x_train-Mean)/Std
#del Mean,Std

#Mean=np.reshape(np.mean(x_val,axis=1),(len(x_val),1))
#Std=np.reshape(np.std(x_val,axis=1),(len(x_val),1))
#x_val=(x_val-Mean)/Std
#del Mean,Std

#Mean=np.reshape(np.mean(x_test,axis=1),(len(x_test),1))
#Std=np.reshape(np.std(x_test,axis=1),(len(x_test),1))
#x_test=(x_test-Mean)/Std

y_train = np.load('./Data/Processed/YTrainKettle.npy')
#np.place(y_train[:,1:51], y_train[:,1:51]==0, 1)
y_val = np.load('./Data/Processed/YValKettle.npy')
#np.place(y_val[:,1:51], y_val[:,1:51]==0, 1)
y_test = np.load('./Data/Processed/YTestKettle.npy')

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
x_val = x_val.reshape((x_val.shape[0],x_val.shape[1],1))

"train model"
#h1 = m.fit(x_train, [y_train[:,0], y_train[:,1:51]],
#           epochs = 3000,
#           batch_size = 128,
#           shuffle = True,
#           validation_data = (x_val, [y_val[:,0],y_val[:,1:51]]), verbose = 1)#,
           #callbacks=[ModelCheckPointer,csv_logger])
           
h1 = m.fit(x_train, y_train[:,1:51],
           epochs = 3000,
           batch_size = 128,
           shuffle = True,
           validation_data = (x_val, y_val[:,1:51]),verbose = 1,
           callbacks=[ModelCheckPointer,csv_logger])
         

"============================================================================="
"================================ TESTING ===================================="
"============================================================================="
#
#y_pred = m.predict(x_test, verbose = 1)
#
#yp = y_pred[1]
#for i in range(5):
#    ind = np.random.randint(low=0,high=y_test.shape[0])
#    if y_test[ind,0] == 1: 
#        plt.figure() 
#        plt.plot(y_test[ind,1:51])
#        plt.plot(y_pred[ind,:])
#        plt.title('Model Predictions')
#        plt.ylabel('Power (W)')
#        plt.xlabel('Sample')
#        plt.legend(['Measured', 'Estimated'], loc='upper left')
#        plt.show()
#
#error=mean_absolute_error(y_test[:,1:51],y_pred)
