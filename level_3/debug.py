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
Number_of_simulations = 500
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
path = ('/bigdata/wonglab/syang159/OP650_1/')
li = []

# reading files

for i in np.arange(1, Number_of_simulations+1, 1):
    os.chdir(path + "cell_%i" % i + '/time/')
    all_files = glob.glob(path + "cell_%i" % i + '/time/' + "/*.csv")
    all_files = sorted(os.listdir(path + "cell_%i" % i + '/time/'), key=lambda x: int(x.replace(".csv", "")))
    for filename in all_files:
        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df)
frame = pd.concat(li)


#getting data to make (180x10x6) arrays
grpdat = frame.groupby('particle')

#delete 1st and 8th column
del frame['particle']
del frame['time']

#indepent input normalization
normalized_pos_x = np.array(frame['pos_x'])
normalized_pos_x = preprocessing.normalize([normalized_pos_x], norm='max')
normalized_pos_x= np.squeeze(normalized_pos_x)
frame['pos_x'] =normalized_pos_x
frame['pos_x'] = pd.to_numeric(frame['pos_x'])

normalized_pos_y = np.array(frame['pos_y'])
normalized_pos_y = preprocessing.normalize([normalized_pos_y], norm='max')
normalized_pos_y= np.squeeze(normalized_pos_y)
frame['pos_y'] =normalized_pos_y
frame['pos_y'] = pd.to_numeric(frame['pos_y'])

normalized_pos_z = np.array(frame['pos_z'])
normalized_pos_z = preprocessing.normalize([normalized_pos_z], norm='max')
normalized_pos_z= np.squeeze(normalized_pos_z)
frame['pos_z'] =normalized_pos_z
frame['pos_z'] = pd.to_numeric(frame['pos_z'])

normalized_vel_x = np.array(frame['vel_x'])
normalized_vel_x = preprocessing.normalize([normalized_vel_x], norm='max')
normalized_vel_x= np.squeeze(normalized_vel_x)
frame['vel_x'] =normalized_vel_x
frame['vel_x'] = pd.to_numeric(frame['vel_x'])

normalized_vel_y = np.array(frame['vel_y'])
normalized_vel_y = preprocessing.normalize([normalized_vel_y], norm='max')
normalized_vel_y= np.squeeze(normalized_vel_y)
frame['vel_y'] =normalized_vel_y
frame['vel_y'] = pd.to_numeric(frame['vel_y'])

normalized_vel_z = np.array(frame['vel_z'])
normalized_vel_z = preprocessing.normalize([normalized_vel_z], norm='max')
normalized_vel_z= np.squeeze(normalized_vel_z)
frame['vel_z'] =normalized_vel_z
frame['vel_z'] = pd.to_numeric(frame['vel_z'])


del frame['vel_x']
del frame['vel_y']
del frame['vel_z']

#make 4D input array
arr_final = []


for i in range (1, (Number_of_particle)+1 , 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, Number_of_simulations)
        arr_final.append(arr_split_by_time)

arr_final = np.transpose(arr_final, (2, 1, 0, 3))

y = np.transpose(arr_final, (1, 0, 2, 3))

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



#get the X

grpdat = frame.groupby('index')
del frame['index']

arr_final2 = []

for i in range (0, Number_of_particle, 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, Number_of_simulations)
        arr_final2.append(arr_split_by_time)

arr_final2 = np.transpose(arr_final2, (2, 1, 0, 3))

X = arr_final2[0]

X = np.reshape(X, (Number_of_simulations, Number_of_particle*Number_of_parameters_for_the_X))

print(X.shape)
print(y.shape)
