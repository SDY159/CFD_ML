import pandas as pd
import numpy as np
import matplotlib
import os
from numpy import genfromtxt
matplotlib.use('Agg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from sklearn import preprocessing

from mpl_toolkits.mplot3d import axes3d

path = ('/bigdata/wonglab/syang159/particle_size3/')
os.chdir(path)

pre = genfromtxt('predictedd1.csv', delimiter=',')
pre = pre.reshape(20, 50, 208, 6)

ori = genfromtxt('originald1.csv', delimiter=',')
ori = ori.reshape(20, 50, 208, 6)

os.chdir(path+'/rnn_codes/plot')


for i in range(0, 20, 2):

    g1 = pre[i]
    g2 = ori[i]

#thin out the array
    g1 = np.delete(g1, np.s_[3:6], 2)
    g1 = np.delete(g1, np.s_[0:208:2], 1)
    g2 = np.delete(g2, np.s_[3:6], 2)
    g2 = np.delete(g2, np.s_[0:208:2], 1)


# Creating dataset
    z = g1[:, :, 2]
    x = g1[:, :, 0]
    y = g1[:, :, 1]

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    w = g2[:, :, 2]
    u = g2[:, :, 0]
    v = g2[:, :, 1]

    u = u.flatten()
    v = v.flatten()
    w = w.flatten()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, lw=2, c='b', s=2, alpha=0.2)
    ax.scatter(u, v, w, lw=2, c='r', s=2, alpha=0.2)

    plt.savefig('ptplot_comparison%i.png' %i)




