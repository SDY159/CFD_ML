import sys
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
import glob
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from numpy import array
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, Conv3D, MaxPooling3D
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
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
#np.array(arr_split_by_time).shape


#Generalization
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
Epoch = 50


# read X
path = ('/bigdata/wonglab/syang159/OP650_1/level_4/')
os.chdir(path)
pre = []

pre = genfromtxt('X2prime.csv', delimiter=',')
pre = pre.reshape((Number_of_simulations), Number_of_timesteps, Number_of_particle, 3)

X = pre


# read y
liy=[]


for i in np.arange(1, Number_of_simulations+1, 1):
    df_1  = pd.read_csv(path + "cell_%i" % i + "/corrosion_pressure")
    lst_1 = range(0,2371)
    df_1 = df_1.drop(lst_1)
    lst_2 = range (40683,42858)
    df_1 = df_1.drop(lst_2)
    erosion = np.array(df_1["dpm-erosion-rate-finnie"])
    liy.append(erosion)


# split into samples

y = np.array(liy)
y = preprocessing.normalize(y, norm='max')

X = np.expand_dims(X, axis=0)

X = np.transpose(X, (2, 3, 1, 4, 0))


cvscores_ero=[]
predicted_y=[]
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

#for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

for train, test in kfold.split(X, y):

    model = Sequential()
    model.add(Conv3D(100, kernel_size=(3, 3, 3), activation='tanh', padding='same', data_format='channels_last', input_shape= (Number_of_timesteps, Number_of_particle , 3, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    
    model.add(Conv3D(200, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    
    model.add(Conv3D(400, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(800, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(1600, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(3200, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))



    model.add(Flatten())
    model.add(Dense(38312, activation='relu'))
   
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    model.fit(X[train], y[train], epochs=Epoch, batch_size=Batch_Size , verbose=1)
    #print(model.predict(X[test]))
    #print("==================================================")
    #print(y[test])
    predicted_y = np.array(model.predict(X[test]))

	# evaluate the model
    scores = r2_score(y[test], model.predict(X[test]))
    print("r2_score: %.5f" % r2_score(y[test], model.predict(X[test])))
    cvscores_ero.append(scores)

    original_array = np.transpose(np.nonzero(y[test]))
    predicted_array = np.transpose(np.nonzero(predicted_y))
    print(len(original_array))
    print(len(predicted_array))
    os.chdir(path + 'level_5/')

        # predicted results and test set into csv
    a = []
    b = []
    printori=[]
    printpre=[]
        # predicted results and test set into csv
    os.chdir(path + 'level_5/original/')
    for j in range (0, len(y[test]), 2):
        np.savetxt('original_%i_kfold_array_from_RNN%i.csv'%(i, j), y[test][j], fmt='%.8e', delimiter=",")

    os.chdir(path + 'level_5/predicted/')
    for j in range (0, len(predicted_y), 2):
        np.savetxt('predicted_%i_kfold_array_from_RNN%i.csv'%(i, j), predicted_y[j], fmt='%.8e', delimiter=",")


print("Average_r2_score: %.5f (+/- %.5f)" % (np.mean(cvscores_ero), np.std(cvscores_ero)))


