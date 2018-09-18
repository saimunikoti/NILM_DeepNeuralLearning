# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:54:10 2018

@author: SmartGridData """

import numpy as np
import keras as kr
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from sklearn import metrics
from keras import regularizers

#%%##################################### Architecture with skip connection_1 ##################################

inp = kr.layers.Input(shape=(300,1))

Enc1 = kr.layers.convolutional.Conv1D(filters = 64, kernel_size = 15,
                                      padding = "valid", activation = 'relu',
                                      kernel_regularizer=regularizers.l2(0.01))(inp)

Enc1=kr.layers.BatchNormalization(momentum=0.8)(Enc1)

P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)

Enc2 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
                                      padding = "valid", activation = 'relu', 
                                      kernel_regularizer=regularizers.l2(0.01))(P1)

Enc2=kr.layers.BatchNormalization(momentum=0.8)(Enc2)

P2 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc2)

Lstm1=kr.layers.LSTM(90)(P2)

Lstm1=kr.layers.BatchNormalization(momentum=0.8)(Lstm1)

P2=kr.layers.Flatten()(P2)

Skip1 = kr.layers.concatenate([Lstm1, P2], axis=-1)


D1 = kr.layers.Dense(300, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
                     kernel_regularizer=regularizers.l2(0.01))(Skip1)

D1=kr.layers.BatchNormalization(momentum=0.8)(D1)

D1 = kr.layers.Dropout(rate=0.3)(D1)

D2 = kr.layers.Dense(50, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
                     kernel_regularizer=regularizers.l2(0.01))(D1)
#D3 = kr.layers.advanced_activations.PReLU(alpha_initializer='glorot_uniform')(D3)
D2 = kr.layers.LeakyReLU(alpha=0.1)(D2)

#Df = kr.layers.concatenate([D2, D4], axis=-1)

m = kr.models.Model(inp, output= D2)

m.summary()

#%%################################### Architercture-2 with skip connection_2 #####################################################

inp = kr.layers.Input(shape=(300,1))

Enc1 = kr.layers.convolutional.Conv1D(filters = 64, kernel_size = 15,
                                      padding = "valid", activation = 'relu')(inp)

Enc1=kr.layers.BatchNormalization(momentum=0.8)(Enc1)

P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)

Enc2 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
                                      padding = "valid", activation = 'relu')(P1)

Enc2=kr.layers.BatchNormalization(momentum=0.8)(Enc2)

P2 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc2)


SkipConv=kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 1,
                                      padding = "valid", activation = 'relu')(P1)
#Lstm1=kr.layers.LSTM(100)(P2)

#Lstm1=kr.layers.BatchNormalization(momentum=0.8)(Lstm1)

#P1=kr.layers.Flatten()(P1)

#P2=kr.layers.Flatten()(P2)

Skip1 = kr.layers.concatenate([P2, SkipConv], axis=1)

Enc3= kr.layers.convolutional.Conv1D(filters = 1, kernel_size = 15,
                                      padding = "valid", activation = 'relu')(Skip1)

P3 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc3)

P3=kr.layers.Reshape((34,))(P3)

D1 = kr.layers.Dense(300, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
                     kernel_regularizer=regularizers.l2(0.01))(P3)

D1=kr.layers.BatchNormalization(momentum=0.8)(D1)

D1 = kr.layers.Dropout(rate=0.3)(D1)

D2 = kr.layers.Dense(50, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', 
                     kernel_regularizer=regularizers.l2(0.01))(D1)
#D3 = kr.layers.advanced_activations.PReLU(alpha_initializer='glorot_uniform')(D3)
D2 = kr.layers.LeakyReLU(alpha=0.1)(D2)

#Df = kr.layers.concatenate([D2, D4], axis=-1)

m = kr.models.Model(inp, output= D2)

#%%####################################### Architecture-3 with skip connection_3 #################################################

inp = kr.layers.Input(shape=(300,1))

Enc1 = kr.layers.convolutional.Conv1D(filters = 64, kernel_size = 15,
                                      padding = "valid")(inp)

Enc1=kr.layers.BatchNormalization(momentum=0.8)(Enc1)

Enc1=kr.layers.LeakyReLU(alpha=0)(Enc1)

P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)

Enc2 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
                                      padding = "valid")(P1)

Enc2=kr.layers.BatchNormalization(momentum=0.8)(Enc2)

Enc2=kr.layers.LeakyReLU(alpha=0)(Enc2)

P2 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc2)


SkipConv=kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 1,
                                      padding = "valid", activation = 'relu')(P1)

Skip1 = kr.layers.concatenate([P2, SkipConv], axis=1)

Lstm1=kr.layers.LSTM(90)(Skip1)

Lstm1=kr.layers.BatchNormalization(momentum=0.8)(Lstm1)

D1 = kr.layers.Dense(300, use_bias=True, kernel_initializer='glorot_uniform',)(Lstm1)

D1=kr.layers.BatchNormalization(momentum=0.8)(D1)

D1=kr.layers.LeakyReLU(alpha=0)(D1)

D1 = kr.layers.Dropout(rate=0.3)(D1)

D2 = kr.layers.Dense(50, use_bias=True, kernel_initializer='glorot_uniform')(D1)

D2 = kr.layers.LeakyReLU(alpha=0.1)(D2)

m = kr.models.Model(inp, output= D2)

m.summary()


#%%####################################### Architecture-4 with skip connection_4 #################################################

inp = kr.layers.Input(shape=(300,1)) 

Enc1 = kr.layers.convolutional.Conv1D(filters = 64, kernel_size = 15,
                                      padding = "valid")(inp)

Enc1=kr.layers.BatchNormalization(momentum=0.8)(Enc1)

Enc1=kr.layers.LeakyReLU(alpha=0)(Enc1)

P1 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc1)

Enc2 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,           
                                      padding = "valid")(P1) 

Enc2=kr.layers.BatchNormalization(momentum=0.8)(Enc2)

Enc2=kr.layers.LeakyReLU(alpha=0)(Enc2)

P2 = kr.layers.pooling.MaxPool1D(pool_size=5, strides = 3, padding="valid")(Enc2)

Lstm1=kr.layers.LSTM(90)(P2)

Lstm1=kr.layers.BatchNormalization(momentum=0.8)(Lstm1)

Enc3 = kr.layers.convolutional.Conv1D(filters = 1, kernel_size = 15,           
                                      padding = "valid")(P1) 

Enc3=kr.layers.BatchNormalization(momentum=0.8)(Enc3)

Enc3=kr.layers.LeakyReLU(alpha=0)(Enc3)

Enc3=kr.layers.Reshape((80,))(Enc3)

Enc4 = kr.layers.convolutional.Conv1D(filters = 1, kernel_size = 15,           
                                      padding = "valid")(P2) 

Enc4=kr.layers.BatchNormalization(momentum=0.8)(Enc4)

Enc4=kr.layers.LeakyReLU(alpha=0)(Enc4)

Enc4=kr.layers.Reshape((12,))(Enc4)

Skip1 = kr.layers.concatenate([Lstm1, Enc3, Enc4], axis=1)


D1 = kr.layers.Dense(300, use_bias=True, kernel_initializer='glorot_uniform')(Skip1)

D1=kr.layers.BatchNormalization(momentum=0.8)(D1)

D1=kr.layers.LeakyReLU(alpha=0)(D1)

D1 = kr.layers.Dropout(rate=0.3)(D1)

D2 = kr.layers.Dense(50, use_bias=True, kernel_initializer='glorot_uniform')(D1)

D2 = kr.layers.LeakyReLU(alpha=0.1)(D2)

m = kr.models.Model(inp, output= D2)

m.summary()

#%%###################################### Architecture-5 with Direct connection_5 #################################################

inp = kr.layers.Input(shape=(300,1)) 

Enc1 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
                                      padding = "valid")(inp)

Enc1=kr.layers.BatchNormalization(momentum=0.8)(Enc1)

Enc1=kr.layers.LeakyReLU(alpha=0)(Enc1)

P1 = kr.layers.pooling.MaxPool1D(pool_size=7, strides = 3, padding="valid")(Enc1)

#P1=kr.layers.GlobalAveragePooling1D()(Enc1)

Enc2 = kr.layers.convolutional.Conv1D(filters = 32, kernel_size = 15,
                                      padding = "valid")(P1)

Enc2=kr.layers.BatchNormalization(momentum=0.8)(Enc2)

Enc2=kr.layers.LeakyReLU(alpha=0)(Enc2)

P2 = kr.layers.pooling.MaxPool1D(pool_size=7, strides = 3, padding="valid")(Enc2)

P2=kr.layers.Flatten()(P2)

D2 = kr.layers.Dense(50, use_bias=True, kernel_initializer='glorot_uniform')(P2)

D2 = kr.layers.LeakyReLU(alpha=0.1)(D2)

m = kr.models.Model(inp, output= D2)

m.summary()


#%%######################################### Model Traning ######################################################

"loss function and compilation"

opt = kr.optimizers.adam(lr=0.001, decay=1e-7)
#opt=kr.optimizers.adagrad()
#m.compile(optimizer = opt, loss = ['binary_crossentropy', 'mae'], metrics = ['accuracy'])
m.compile(optimizer = opt, loss = 'mae', metrics = ['accuracy'])


#m.compile(optimizer=opt, loss=['binary_crossentropy','mae'])
aply='dw'

"saving options"
FileLoc = './Models/Dishwasher/Dishwasher_Skip4_Trsfrlrng_h5/'
ModelName = 'Segdw'+aply+'_'
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
#x_train = np.load('./Data/Processed/XTrainkettle.npy')
x_train = np.load('./Data/New_Processed/XTrain'+aply+'h5.npy')
x_train = np.load('./Data/New_Processed/house_5/XTrain'+aply+'_h5.npy')


#x_val = np.load('./Data/Processed/XValkettle.npy')
x_val = np.load('./Data/New_Processed/XVal'+aply+'.npy')
x_val = np.load('./Data/New_Processed/house_5/XVal'+aply+'_h5.npy')

#x_test = np.load('./Data/Processed/XTestkettle.npy')
x_test = np.load('./Data/New_Processed/house_1/XTest'+aply+'.npy')
x_test = np.load('./Data/New_Processed/house_5/XTest'+aply+'_h5.npy')

x_train = np.diff(x_train)
x_val = np.diff(x_val)
x_test = np.diff(x_test)

# Normalizing data 

#Mean=np.reshape(np.mean(x_train,axis=1),(len(x_train),1))
##Std=np.reshape(np.std(x_train,axis=1),(len(x_train),1))
#x_train=(x_train-Mean)#/Std
#del Mean,Std
#
#Mean=np.reshape(np.mean(x_val,axis=1),(len(x_val),1))
##Std=np.reshape(np.std(x_val,axis=1),(len(x_val),1))
#x_val=(x_val-Mean)#/Std
#del Mean,Std
#
#Mean=np.reshape(np.mean(x_test,axis=1),(len(x_test),1))
##Std=np.reshape(np.std(x_test,axis=1),(len(x_test),1))
#x_test=(x_test-Mean)#/Std

#y_train = np.load('./Data/Processed/YTrainkettle.npy')
y_train = np.load('./Data/New_Processed/YTrain'+aply+'.npy')
y_train = np.load('./Data/New_Processed/house_5/YTrain'+aply+'_h5.npy')

#np.place(y_train[:,1:51], y_train[:,1:51]==0, 1)

#y_val = np.load('./Data/Processed/YValkettle.npy')
y_val = np.load('./Data/New_Processed/YVal'+aply+'.npy')
y_val = np.load('./Data/New_Processed/house_5/YVal'+aply+'_h5.npy')

#np.place(y_val[:,1:51], y_val[:,1:51]==0, 1)

#y_test = np.load('./Data/Processed/YTestkettle.npy')
y_val = np.load('./Data/New_Processed/house_2/YVal'+aply+'.npy')
y_test = np.load('./Data/New_Processed/house_5/YTest'+aply+'_h5.npy')

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
           batch_size = 256,
           shuffle = True,
           validation_data = (x_val, y_val[:,1:51]),verbose = 1,
           callbacks=[ModelCheckPointer,csv_logger])

         
#%%######################################### Model testing ################################################

y_pred = m.predict(x_test, verbose = 1)
##
##yp = y_pred[1]
for i in range(10):
    ind = np.random.randint(low=0,high=y_test.shape[0])
    if y_test[ind,0] == 1: 
        plt.figure() 
        plt.plot(y_test[ind,1:51])
        plt.plot(y_pred[ind,:])
        plt.title('Model Predictions')
        plt.ylabel('Power (W)')
        plt.xlabel('Sample')
        plt.legend(['Measured', 'Estimated'], loc='upper left')
        plt.show()

error=mean_absolute_error(y_test[:,1:51],y_pred)

#%%############################### normalised absolute error #################################################

#t1=np.array(np.where(y_test[:,0]==0))
#t1=np.reshape(t1,(len(t1[0,:]),))
#
#y_test_ones=np.delete(y_test,t1, axis=0)
#
#y_pred_ones=np.delete(y_pred,t1, axis=0)
#
#diff=y_test_ones[:,1:51]-y_pred_ones
#
#dfsum=np.sum(diff, axis=1)
#
#ytest_sum=np.sum(y_test_ones, axis=1)
#
#Custom_error=abs(dfsum/ytest_sum)
#
#NAE=np.mean(Custom_error)

#%%################ Total aggregate error in estimating appliance activation #

Sum_Pred=np.zeros((len(y_pred),1))
Sum_Truth=np.zeros((len(y_test),1))
for i in range(len(y_test)):
    Sum_Truth[i,0]=np.sum(y_test[i,1:])
    Sum_Pred[i,0]=np.sum(y_pred[i,:])
    
    
    
    