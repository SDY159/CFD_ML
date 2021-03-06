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
from sklearn.metrics import mean_squared_error
import tensorflow
import keras_tuner
from keras_tuner.tuners import RandomSearch
import keras_tuner as kt
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
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                          allow_soft_placement=True,
                          device_count={'CPU': 8})
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
Number_of_simulations = 3125
Number_of_particle = 196
Number_of_timesteps = 50
Min_particle_size = 0.0000400
Particle_size_interval = 0.000005
Min_main_inlet_speed = 16.75
Main_inlet_speed_interval = 2.5925
Min_sub_inlet_speed = 237.03
Sub_inlet_speed_interval = 48.2575
Min_main_inlet_pressure = 14000000
Main_inlet_pressure_interval = 500000
Min_sub_inlet_pressure = 2300000
Sub_inlet_pressure_interval = 175000
Number_of_parameters_for_the_X = 8

Batch_Size = 100
Epoch = 50

# read X
path = ('/bigdata/wonglab/syang159/OP650_1/')
os.chdir(path+'level_5/')
pre = []

pre = genfromtxt('X2prime.csv', delimiter=',')
pre = pre.reshape((Number_of_simulations), Number_of_timesteps, Number_of_particle, 3)

X = pre
X = np.transpose(X, (2, 0, 1, 3))
X = np.dstack(X)
X = X.reshape((Number_of_simulations*Number_of_timesteps*Number_of_particle),3)
frame = pd.DataFrame(X)

#adding particle size info, 1st added parameter (Max Quotient is 5)
frame.insert(3, "particle_size", 0.0)
frame['particle_size'] = pd.to_numeric(frame['particle_size'])
frame = frame.reset_index()
frame = frame.reset_index()
temppar = frame['level_0']/((Number_of_timesteps*Number_of_particle)*625)
temppar=temppar.astype('int32')
frame['particle_size'] = (temppar*Particle_size_interval) + Min_particle_size
del frame['level_0']


normalized_particlesize= np.array(frame['particle_size'])
scaler = MinMaxScaler()
normalized_particlesize= frame['particle_size'].values.reshape(-1, 1)
normalized_particlesize = scaler.fit_transform(normalized_particlesize)
normalized_particlesize=np.squeeze(normalized_particlesize)
frame['particle_size']=normalized_particlesize
frame['particle_size']= pd.to_numeric(frame['particle_size'])

#adding initial main inlet speed info, 2nd added parameter (Max Quotient is 25)
frame.insert(5, "main_inlet_speed", 0.0)
frame['main_inlet_speed'] = pd.to_numeric(frame['main_inlet_speed'])
frame = frame.reset_index()
temppar = frame['level_0']/((Number_of_timesteps*Number_of_particle)*125)
temppar = temppar.astype('int32')
loop=0
for i in range(0, (Number_of_timesteps*Number_of_particle*Number_of_simulations), ((Number_of_timesteps*Number_of_particle)*625)):
    itv=5
    temppar[i:i+((Number_of_timesteps*Number_of_particle)*625)] = (temppar[i:i+((Number_of_timesteps*Number_of_particle)*625)] - loop)
    loop = loop + itv

frame['main_inlet_speed'] = (temppar*Main_inlet_speed_interval) + Min_main_inlet_speed
del frame['level_0']


normalized_main_inlet_speed = np.array(frame['main_inlet_speed'])
scaler = MinMaxScaler()
normalized_main_inlet_speed = frame['main_inlet_speed'].values.reshape(-1, 1)
normalized_main_inlet_speed = scaler.fit_transform(normalized_main_inlet_speed)
normalized_main_inlet_speed = np.squeeze(normalized_main_inlet_speed)
frame['main_inlet_speed'] = normalized_main_inlet_speed
frame['main_inlet_speed'] = pd.to_numeric(frame['main_inlet_speed'])


#adding initial sub inlet speed info, 3rd added parameter (Max Quotient is 125)
frame.insert(6, "sub_inlet_speed", 0.0)
frame['sub_inlet_speed'] = pd.to_numeric(frame['sub_inlet_speed'])
frame = frame.reset_index()
temppar = frame['level_0']/((Number_of_timesteps*Number_of_particle)*25)
temppar = temppar.astype('int32')
loop=0
for i in range(0, (Number_of_timesteps*Number_of_particle*Number_of_simulations), ((Number_of_timesteps*Number_of_particle)*125)):
    itv=5
    temppar[i:i+((Number_of_timesteps*Number_of_particle)*125)] = (temppar[i:i+((Number_of_timesteps*Number_of_particle)*125)] - loop)
    loop = loop + itv
frame['sub_inlet_speed'] = (temppar*Sub_inlet_speed_interval) + Min_sub_inlet_speed

del frame['level_0']


normalized_sub_inlet_speed= np.array(frame['sub_inlet_speed'])
scaler = MinMaxScaler()
normalized_sub_inlet_speed = frame['sub_inlet_speed'].values.reshape(-1, 1)
normalized_sub_inlet_speed = scaler.fit_transform(normalized_sub_inlet_speed)
normalized_sub_inlet_speed = np.squeeze(normalized_sub_inlet_speed)
frame['sub_inlet_speed'] = normalized_sub_inlet_speed
frame['sub_inlet_speed'] = pd.to_numeric(frame['sub_inlet_speed'])

#adding initial main inlet pressure info, 4th added parameter (Max Quotient is 625)
frame.insert(7, "main_inlet_pressure", 0.0)
frame['main_inlet_pressure'] = pd.to_numeric(frame['main_inlet_pressure'])
frame = frame.reset_index()
temppar = frame['level_0']/((Number_of_timesteps*Number_of_particle)*5)
temppar = temppar.astype('int32')
loop=0
for i in range(0, (Number_of_timesteps*Number_of_particle*Number_of_simulations), ((Number_of_timesteps*Number_of_particle)*25)):
    itv=5
    temppar[i:i+((Number_of_timesteps*Number_of_particle)*25)] = (temppar[i:i+((Number_of_timesteps*Number_of_particle)*25)] - loop)
    loop = loop + itv
frame['main_inlet_pressure'] = (temppar*Main_inlet_pressure_interval) + Min_main_inlet_pressure

del frame['level_0']


normalized_main_inlet_pressure = np.array(frame['main_inlet_pressure'])
scaler = MinMaxScaler()
normalized_main_inlet_pressure = frame['main_inlet_pressure'].values.reshape(-1, 1)
normalized_main_inlet_pressure = scaler.fit_transform(normalized_main_inlet_pressure)
normalized_main_inlet_pressure = np.squeeze(normalized_main_inlet_pressure)
frame['main_inlet_pressure'] = normalized_main_inlet_pressure
frame['main_inlet_pressure'] = pd.to_numeric(frame['main_inlet_pressure'])

#adding initial sub inlet pressure info, 5th added parameter (Max Quotient is 3125)
frame.insert(8, "sub_inlet_pressure", 0.0)
frame['sub_inlet_pressure'] = pd.to_numeric(frame['sub_inlet_pressure'])
frame = frame.reset_index()
temppar = frame['level_0']/(Number_of_timesteps*Number_of_particle)
temppar = temppar.astype('int32')
loop=0
for i in range(0, (Number_of_timesteps*Number_of_particle*Number_of_simulations), ((Number_of_timesteps*Number_of_particle)*5)):
    itv=5
    temppar[i:i+((Number_of_timesteps*Number_of_particle)*5)] = (temppar[i:i+((Number_of_timesteps*Number_of_particle)*5)] - loop)
    loop = loop + itv
frame['sub_inlet_pressure'] = (temppar*Sub_inlet_pressure_interval) + Min_sub_inlet_pressure

del frame['level_0']


normalized_sub_inlet_pressure = np.array(frame['sub_inlet_pressure'])
scaler = MinMaxScaler()
normalized_sub_inlet_pressure = frame['sub_inlet_pressure'].values.reshape(-1, 1)
normalized_sub_inlet_pressure = scaler.fit_transform(normalized_sub_inlet_pressure)
normalized_sub_inlet_pressure = np.squeeze(normalized_sub_inlet_pressure)
frame['sub_inlet_pressure'] = normalized_sub_inlet_pressure
frame['sub_inlet_pressure'] = pd.to_numeric(frame['sub_inlet_pressure'])

del frame['index']

A=frame.to_numpy()
arr_split_by_simulations = np.array_split(A, Number_of_simulations)
X = np.array(arr_split_by_simulations).reshape(Number_of_simulations, Number_of_timesteps, Number_of_particle,8)


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

X = np.expand_dims(X, axis=4)

def create_model(hyperparam):
	model = Sequential()
	model.add(Conv3D(filters=hyperparam.Int('convolution_1', min_value=50, max_value=100, step=10), kernel_size=(3, 3, 3),activation='tanh', padding='same',input_shape= (Number_of_timesteps, Number_of_particle , 8, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_2', min_value=100, max_value=400, step=50),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_3', min_value=150, max_value=800, step=50),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_4', min_value=200, max_value=1200, step=50),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_5', min_value=250, max_value=1600, step=100),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_6', min_value=300, max_value=1800, step=100),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Conv3D(filters=hyperparam.Int('convolution_7', min_value=350, max_value=2000, step=150),kernel_size=(2, 2, 2),activation='tanh', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
	model.add(Flatten())
	model.add(Dense(38312, activation='relu'))
	model.compile(optimizer='adam',loss='mse')
	return model


tuner_search=RandomSearch(create_model, objective='val_loss', max_trials=100,directory='output',project_name="mnist")


tuner_search.search(X,y,epochs=5,validation_data=(X,y))

model=tuner_search.get_best_models(num_models=2)
model.fit(X,y, epochs=Epoch, validation_data=(X,y))
