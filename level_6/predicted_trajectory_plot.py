import sys
import os
import pandas as pd
import numpy as np
import glob
import csv
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from numpy import array
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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


# read x
path = ('/bigdata/wonglab/syang159/CFD1/OP650_2/')
os.chdir(path + 'level_7/')


X = pd.read_csv('denormalized_X2prime.csv', delimiter=',')
X = pd.DataFrame(X).to_numpy()
X= X.reshape((Number_of_simulations),  Number_of_particle*Number_of_timesteps, 3)

sim_num=296
X_s = X[sim_num]

X_tr = np.reshape(X_s,(196, 50 ,3), order='F')

X_cut = np.delete(X_tr, np.s_[0:196:2], 0)
X_cut = np.delete(X_cut, np.s_[0:196:2], 0)
# X_cut = np.delete(X_cut, np.s_[2:196:2], 1)


def data_for_cylinder_main(center_y,center_z,radius,height_x):
       x = np.linspace(0, height_x, 50)-0.64025
       theta = np.linspace(0, 2*np.pi, 50)
       theta_grid, x_grid=np.meshgrid(theta, x)
       y_grid = radius*np.cos(theta_grid) + center_y
       z_grid = radius*np.sin(theta_grid) + center_z
       return x_grid,y_grid,z_grid
Xm,Ym,Zm = data_for_cylinder_main(0,0,0.269,1.8805)
def data_for_cylinder_inlet(center_x,center_z,radius,height_y):
       y = np.linspace(0, -height_y, 50)-0.269
       theta = np.linspace(0, 2*np.pi, 50)
       theta_grid, y_grid=np.meshgrid(theta, y)
       x_grid = radius*np.cos(theta_grid) + center_x
       z_grid = radius*np.sin(theta_grid) + center_z
       return x_grid,y_grid,z_grid
Xi1,Yi1,Zi1 = data_for_cylinder_inlet(-0.5435,0,0.01875,0.269)
Xi2,Yi2,Zi2 = data_for_cylinder_inlet(0.0,0,0.01875,0.269)
Xi3,Yi3,Zi3 = data_for_cylinder_inlet(0.5435,0,0.01875,0.269)

fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
ax.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
ax.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
ax.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.zaxis.set_major_locator(plt.MaxNLocator(5))

ax.set_xlabel('X-coordinate (m)', fontsize=12)
ax.set_ylabel('Y-coordinate (m)', fontsize=12)
ax.set_zlabel('Z-coordinate (m)', fontsize=12)

for i in range(0,18,1):
    X = X_cut[i][:,0]
    Y = X_cut[i][:,1]
    Z = X_cut[i][:,2]



    ax.plot(X, Z, Y, color='cyan', linestyle='-',linewidth=1, markersize=2.5)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

for i in range(18,34,1):
    X = X_cut[i][:,0]
    Y = X_cut[i][:,1]
    Z = X_cut[i][:,2]



    ax.plot(X, Z, Y, color='orange', linestyle='-',linewidth=1, markersize=2.5)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


for i in range(34,49,1):
    X = X_cut[i][:,0]
    Y = X_cut[i][:,1]
    Z = X_cut[i][:,2]



    ax.plot(X, Z, Y, color='magenta', linestyle='-',linewidth=1, markersize=2.5)

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

# plt.title("Particle trajectory (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)

ax.axes.set_xlim3d(left=-0.8, right=0.8)
ax.axes.set_ylim3d(bottom=-0.8, top=0.8)
ax.axes.set_zlim3d(bottom=-0.8, top=0.8)
# plt.grid()
# Hide grid lines
ax.grid(False)

plt.tight_layout()
# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
os.chdir('/bigdata/wonglab/syang159/CFD1/OP650_2/level_7/')
plt.savefig('plot_predicted_t_%i.png'%sim_num)
