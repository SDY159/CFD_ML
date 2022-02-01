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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LSTM, RepeatVector, Reshape, TimeDistributed, InputLayer, Conv3D, MaxPooling3D
from keras.regularizers import l2
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

#print the model architecture
#os.chdir('/bigdata/wonglab/syang159/particle_size3/rnn_codes')
#tensorflow.keras.utils.plot_model(model2, to_file="model_arch2.png", show_shapes=True, show_layer_names=True)



#check list as np array
#np.array(arr_final).shape


#load files

# read x
path = ('/bigdata/wonglab/syang159/particle_size3/')
li = []

# reading files

for i in np.arange(4000, 4101, 2):
    os.chdir(path + "cell_0.%08i" % i + '/time/')
    all_files = glob.glob(path + "cell_0.%08i" % i + '/time/' + "/*.csv")
    all_files = sorted(os.listdir(path + "cell_0.%08i" % i + '/time/'), key=lambda x: int(x.replace(".csv", "")))
    for filename in all_files:
        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df)
frame = pd.concat(li)


#delete 1st and 8th column
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

#adding particle size info
frame.insert(7, "particle_size", 0.0)
frame['particle_size'] = pd.to_numeric(frame['particle_size'])
frame = frame.reset_index()
frame = frame.reset_index()
temppar = frame['level_0']/9000
temppar=temppar.astype('int32')
frame['particle_size'] = temppar*0.00000002 + 0.00004000
del frame['level_0']


normalized_particlesize= np.array(frame['particle_size'])
scaler = MinMaxScaler()
normalized_particlesize= frame['particle_size'].values.reshape(-1, 1)
normalized_particlesize = scaler.fit_transform(normalized_particlesize)
normalized_particlesize=np.squeeze(normalized_particlesize)
frame['particle_size']=normalized_particlesize
frame['particle_size']= pd.to_numeric(frame['particle_size'])


#get the X1

grpdat = frame.groupby('index')
del frame['index']
del frame['particle']

arr_final2 = []

for i in range (0, 180, 1):

        arr_i = grpdat.get_group(i).values
        arr_split_by_time = np.array_split(arr_i, 51)
        arr_final2.append(arr_split_by_time)

arr_final2 = np.transpose(arr_final2, (2, 1, 0, 3))

X1 = arr_final2[0]

X1 = np.reshape(X1, (51, 180*7))

X2 = np.transpose(arr_final2, (1, 0, 2, 3))

# read y
liy=[]


for i in np.arange(4000, 4101, 2):
    df_1  = pd.read_csv(path + "cell_0.%08i" % i + "/corrosion_pressure")
    lst_1 = range(0,414)
    df_1 = df_1.drop(lst_1)
    lst_2 = range (8426,8840)
    df_1 = df_1.drop(lst_2)
    erosion = np.array(df_1["dpm-erosion-rate-finnie"])
    liy.append(erosion)

y = np.array(liy)
y = preprocessing.normalize(y, norm='max')



#declare k fold splits
cvscores=[]
predicted_y=[]
X2prime=[]
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

#RNN layer (X1,X2)
x1 = Input(shape =(1260,)) 
RV =RepeatVector(50) (x1)
LSTM1 = LSTM(1260, return_sequences=True) (RV)
LSTM2 = LSTM(1260, return_sequences=True) (LSTM1)
LSTM3 = LSTM(1260, return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)) (LSTM2)
Dense1 = Dense(180*7) (LSTM3)
reshape = Reshape(target_shape=(50, 180, 7)) (Dense1)
output = Reshape(target_shape=(50, 180, 7, 1)) (reshape)

model1 = Model(inputs=x1, outputs=output)



model1.compile(optimizer='adam', loss='mse')
model1.fit(X1, X2, epochs=2, batch_size=10)
#Generate X2' dataset
X2prime = np.array(model1.predict(X1))




#CNN layer (X2', y)
X2pr = Input(shape =(50, 180, 7, 1))
Conv1 = Conv3D(30, kernel_size=(3, 3, 3), activation='tanh', padding='same', data_format='channels_last') (X2pr)
MP1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv1)
Conv2 =Conv3D(60, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP1)
MP2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv2)
Conv3 = Conv3D(90, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP2)
MP3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv3)
Conv4 = Conv3D(120, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP3)
MP4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv4)
Conv5 = Conv3D(150, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP4)
MP5 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv5)
Conv6 = Conv3D(180, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP5)
MP6 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv6)
Conv7 = Conv3D(210, kernel_size=(2, 2, 2), activation='tanh',padding='same',  data_format='channels_last') (MP6)
MP7 = MaxPooling3D(pool_size=(2, 2, 2), padding='same') (Conv7)
Flatten = Flatten() (MP7)
DenseOut = Dense(8012, activation='relu') (Flatten)

#Declare the model with inputs and outputs

model2 = Model(inputs=X2pr, outputs=DenseOut)

for train, test in kfold.split(X2prime, y):
 
    model2.compile(optimizer='adam', loss='mse')
    model2.fit(X2prime, y, epochs=2, batch_size=10)

    #print(model.predict(X[test]))
    #print("==================================================")
    #print(y[test])


    predicted_y = np.array(model2.predict(X2prime[test]))
    flat_pre_y = predicted_y.flatten()
    flat_ori_y = y[test].flatten()

	# evaluate the model
    scores = r2_score(flat_ori_y, flat_pre_y)

    print("r2_score: %.5f" % r2_score(flat_ori_y, flat_pre_y))
    cvscores.append(scores)
		
print("Average_r2_score: %.5f (+/- %.5f)" % (np.mean(cvscores), np.std(cvscores)))
