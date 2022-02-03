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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

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

# read X
path = ('/bigdata/wonglab/syang159/OP650_1/level_4/')
os.chdir(path)
pre = []

pre = genfromtxt('X2prime.csv', delimiter=',')
X = pre

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
y = y.flatten()

print(mean_squared_error(X, y, squared=False))
