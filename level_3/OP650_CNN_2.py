import os
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv3D, MaxPooling3D
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
Epoch = 150


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

grpdat = frame.groupby('index')
del frame['index']

arr_final2 =[]

for i in range (0, Number_of_particle, 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, Number_of_simulations)
        arr_final2.append(arr_split_by_time)

arr_final2 = np.transpose(arr_final2, (2, 1, 0, 3))

X = arr_final2 


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

X = np.transpose(X, (2, 1, 3, 4, 0))


cvscores_ero=[]
predicted_y=[]
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

#for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]
i=0
for train, test in kfold.split(X, y):
    
    model = Sequential()
    model.add(Conv3D(100, kernel_size=(3, 3, 3), activation='tanh', padding='same', data_format='channels_last', input_shape= (Number_of_timesteps, Number_of_particle , 8, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(150, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(200, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(250, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(300, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(350, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

    model.add(Conv3D(400, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))



    model.add(Flatten())
    model.add(Dense(38312, activation='relu'))
   
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    model.fit(X[train], y[train], epochs=Epoch, batch_size=Batch_Size, verbose=1)
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


        # predicted results and test set into csv
    os.chdir(path + 'level_2/kernel/original/')
    for j in range (0, len(y[test]), 25):
        np.savetxt('original_%i_kfold_array_from_RNN%i.csv'%(i, j), y[test][j], fmt='%.8e', delimiter=",")

    os.chdir(path + 'level_2/kernel/predicted/')
    for j in range (0, len(predicted_y), 2):
        np.savetxt('predicted_%i_kfold_array_from_RNN%i.csv'%(i, j), predicted_y[j], fmt='%.8e', delimiter=",")
    i=i+1
print("Average_r2_score: %.5f (+/- %.5f)" % (np.mean(cvscores_ero), np.std(cvscores_ero)))

