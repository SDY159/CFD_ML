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

from numpy import genfromtxt
np.set_printoptions(threshold=5000, edgeitems=200, linewidth=500)


# read x
path = ('/bigdata/wonglab/syang159/New_geo/')
os.chdir(path)

X = genfromtxt('inputX.csv', delimiter=',')
X = np.reshape(X, (1000, 208, 9))




for i in range (0,1000,1):
    print('X[%i,0]= '%i, X[i,207])
    print('X[%i,1]= '%i, X[i,207])
    print('X[%i,2]= '%i, X[i,207])
    print('X[%i,3]= '%i, X[i,207])
    print('X[%i,4]= '%i, X[i,207])
    print('X[%i,5]= '%i, X[i,207])
    print('X[%i,6]= '%i, X[i,207])
    print('X[%i,7]= '%i, X[i,207])
    print('X[%i,8]= '%i, X[i,207])
    print('X[%i,9]= '%i, X[i,207])
    print('-------------------------------------------------')
