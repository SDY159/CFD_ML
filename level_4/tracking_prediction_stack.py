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
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

#Print all of content
np.set_printoptions(threshold=5000, edgeitems=20, linewidth=100)

#Use delimeter as comma
#np.set_string_function(lambda x: repr(x), repr=False)

# fix random seed for reproducibility
seed = 7
np.random.seed(7)


#check list as np array
#np.array(arr_final).shape


#load files

# read x
path = ('/bigdata/wonglab/syang159/particle_size3/')
li = []

# reading files

for i in np.arange(4000, 6001, 2):
    os.chdir(path + "cell_0.%08i" % i + '/time/')
    all_files = glob.glob(path + "cell_0.%08i" % i + '/time/' + "/*.csv")
    all_files = sorted(os.listdir(path + "cell_0.%08i" % i + '/time/'), key=lambda x: int(x.replace(".csv", "")))
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

#make 4D out array
arr_final = []


for i in range (1, 181, 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, 1001)
        arr_final.append(arr_split_by_time)

arr_final = np.transpose(arr_final, (2, 1, 0, 3))

y = np.transpose(arr_final, (1, 0, 2, 3))

#adding particle size info
frame.insert(6, "particle_size", 0.0)
frame['particle_size'] = pd.to_numeric(frame['particle_size'])
frame = frame.reset_index()
frame = frame.reset_index()
frame['particle_size'] = frame['level_0']/9000*0.00000002 + 0.00004000
del frame['level_0']

#make 4D input array with normalized particle size
normalized_particle_size = np.array(frame['particle_size'])
normalized_particle_size = preprocessing.normalize([normalized_particle_size], norm='max')
normalized_particle_size= np.squeeze(normalized_particle_size)
frame['particle_size'] =normalized_particle_size
frame['particle_size'] = pd.to_numeric(frame['particle_size'])

arr_final = []


for i in range (1, 181, 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, 1001)
        arr_final.append(arr_split_by_time)

arr_final = np.transpose(arr_final, (2, 1, 0, 3))


X = arr_final[0]

X = np.reshape(X, (1001, 180*7))


#declare k fold splits
cvscores=[]
predicted_y=[]
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

#for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = X[train_index], X[test_index]
    #y_train, y_test = y[train_index], y[test_index]

for train, test in kfold.split(X, y):

    model = Sequential()
    model.add(Dense(1260, input_shape=(1260,)))
    model.add(RepeatVector(50))
    model.add(LSTM(1080, return_sequences=True))
    model.add(LSTM(1080, return_sequences=True))
    model.add(LSTM(1080, return_sequences=True))
    model.add(TimeDistributed(Dense(180*6)))
    model.add(TimeDistributed(Reshape((180, 6))))
    model.add(Reshape(target_shape=(50, 180, 6)))


    model.compile(optimizer='adam', loss='mse')
    model.fit(X[train], y[train], epochs=20, verbose=0)

    predicted_y = np.array(model.predict(X[test]))
    flat_pre_y = predicted_y.flatten()
    flat_ori_y = y[test].flatten()
    os.chdir(path)


    # predicted results and test set into csv
    a = np.asarray(flat_pre_y)
    np.savetxt('predicted.csv', a, fmt='%.8e', delimiter=",")
    b = np.asarray(flat_ori_y)
    np.savetxt('original.csv', b, fmt='%.8e', delimiter=",")

	# evaluate the model
    scores = r2_score(flat_ori_y, flat_pre_y)

    print("r2_score: %.5f" % r2_score(flat_ori_y, flat_pre_y))
    cvscores.append(scores)
		
print("Average_r2_score: %.5f (+/- %.5f)" % (np.mean(cvscores), np.std(cvscores)))
