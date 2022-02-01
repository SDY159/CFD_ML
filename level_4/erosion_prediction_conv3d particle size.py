import sys
import os
import pandas as pd
import numpy as np
import glob
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from numpy import array
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, RepeatVector, Reshape, TimeDistributed, InputLayer
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
Number_of_simulations = 1000
Number_of_particle = 208
Number_of_timesteps = 50
Min_particle_size = 0.0000400
Particle_size_interval = 0.000002
Min_main_inlet_speed = 8.45
Main_inlet_speed_interval = 0.8
Min_sub_inlet_speed = 25
Sub_inlet_speed_interval = 4
Number_of_parameters_for_the_X = 9
Batch_Size = 100
Epoch = 150


# read X
path = ('/bigdata/wonglab/syang159/New_geo/')
dataframe = []
for i in range(1, 6, 1):
    df = pd.read_csv("predicted%i.csv" %i, index_col=0, header=0)
    dataframe.append(df)
frame = pd.concat(dataframe)

# read y
liy=[]


for i in np.arange(1, Number_of_simulations, 1):
    df_1  = pd.read_csv(path + "cell_%i" % i + "/corrosion_pressure")
    lst_1 = range(0,360)
    df_1 = df_1.drop(lst_1)
    lst_2 = range (9624,9776)
    df_1 = df_1.drop(lst_2)
    erosion = np.array(df_1["dpm-erosion-rate-finnie"])
    liy.append(erosion)


# split into samples
X = arr_final

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
    model.add(Conv3D(30, kernel_size=(3, 3, 3), activation='tanh', padding='same', data_format='channels_last', input_shape= (50, 180, 7, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(60, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(90, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(120, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(150, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(180, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(210, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))



    model.add(Flatten())
    model.add(Dense(8012, activation='relu'))
   
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    model.fit(X[train], y[train], epochs=20, batch_size=1001, verbose=0)
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

    print("==================================================")
    print("original_non_zero_values:")
    for i in range (len(original_array)):
        print ('{:>10}'.format(original_array[i, 1]), '{:>10}'.format(y[test][original_array[i, 0], original_array[i, 1]]))
    print("==================================================")
    print("Predicted_non_zero_values:")
    for i in range (len(predicted_array)):
        print ('{:>10}'.format(predicted_array[i,1]), '{:>10}'.format(predicted_y[predicted_array[i,0],predicted_array[i,1]]))

print("Average_r2_score: %.5f (+/- %.5f)" % (np.mean(cvscores_ero), np.std(cvscores_ero)))

