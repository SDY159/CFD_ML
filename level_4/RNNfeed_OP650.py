import sys
import os
import pandas as pd
import numpy as np
import glob
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from numpy import genfromtxt
from numpy import array
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, RepeatVector, Reshape, TimeDistributed, InputLayer
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.python.keras.backend as K
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=32,
                          allow_soft_placement=True,
                          device_count={'CPU': 32})
session = tf.compat.v1.Session(config=config)
sess = tf.compat.v1.Session
K.set_session(sess)


#Print all of content
np.set_printoptions(threshold=5000, edgeitems=20, linewidth=100)

#Use delimeter as comma
#np.set_string_function(lambda x: repr(x), repr=False)

# fix random seed for reproducibility
seed = 7
np.random.seed(7)


#check list as np array
#np.array(arr_final).shape

#plot model architecture
#os.chdir(path)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#Generalization
Number_of_simulations = 3125
Number_of_particle = 196
Number_of_timesteps = 50
Min_particle_size = 0.0000400
Particle_size_interval = 0.000002
Min_main_inlet_speed = 8.45
Main_inlet_speed_interval = 0.8
Min_sub_inlet_speed = 25
Sub_inlet_speed_interval = 4
Min_main_inlet_pressure = 12000000
Main_inlet_pressure_interval = 960000
Min_sub_inlet_pressure = 1000000
Sub_inlet_pressure_interval = 400000
Number_of_parameters_for_the_X = 8

Batch_Size = 100
Epoch = 300

# read x
path = ('/bigdata/wonglab/syang159/OP650_1/level_3/')
os.chdir(path)

X = genfromtxt('X.csv', delimiter=',')

# read y
y = []

y = genfromtxt('y.csv', delimiter=',')
y = y.reshape((Number_of_simulations), Number_of_timesteps, Number_of_particle, 3)



cvscores=[]
predicted_y=[]
#kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

#for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
#i=0
#for train, test in kfold.split(X, y):
    #i = i+1
model = Sequential()
model.add(Dense(Number_of_particle*Number_of_parameters_for_the_X, input_shape=(Number_of_particle*Number_of_parameters_for_the_X,)))
model.add(RepeatVector(Number_of_timesteps))
model.add(LSTM((Number_of_particle*3), return_sequences=True))
model.add(LSTM((Number_of_particle*3), return_sequences=True))
model.add(LSTM((Number_of_particle*3), return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dense(Number_of_particle*3))
model.add(Reshape(target_shape=(Number_of_timesteps, Number_of_particle, 3)))


model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, batch_size=Batch_Size, validation_split = 0.3, epochs=Epoch , verbose=1)
history
# list all data in history
os.chdir('/bigdata/wonglab/syang159/OP650_1/level_5')

predicted_y = np.array(model.predict(X))
flat_pre_y = predicted_y.flatten()
flat_ori_y = y.flatten()


a = np.asarray(flat_pre_y)
np.savetxt('X2prime.csv', a, fmt='%.8e', delimiter=",")


# evaluate the model
print("r2_score: %.5f" % r2_score(flat_ori_y, flat_pre_y))
print("rmse: %.5f" % mean_squared_error(flat_ori_y, flat_pre_y), squared=False)

