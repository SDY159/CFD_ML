import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
import matplotlib.ticker as ticker
import pandas as pd
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

path = ('/bigdata/wonglab/syang159/OP650_1/')

os.chdir(path + 'cell_1/')
df = pd.read_csv("corrosion_pressure")

inlet_1=df.iloc[:2175,:]
inlet_2=df.iloc[2175:2247,:]
inlet_3=df.iloc[2247:2311,:]
inlet_4=df.iloc[2311:2371,:]
main=df.iloc[2371:40683,:]
outlet=df.iloc[40683:,:]


X = main.iloc[:,1].values
Y = main.iloc[:,2].values
Z = main.iloc[:,3].values


os.chdir(path + 'level_7/predicted')

plot_num=50
sim_num=296

df_1 = pd.read_csv("predicted_1_kfold_array_from_RNN%i.csv"%plot_num ,header=None)

os.chdir(path + 'level_8')
df_2 = pd.read_csv("erosion_denominator.csv" ,header=None)
df_2  = pd.DataFrame(df_2).to_numpy()
df_1 = df_1 * df_2[sim_num]
W = df_1[0].values

fig = plt.figure()

ax=fig.add_subplot(111,projection='3d')


ax.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
ax.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
ax.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
ax.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')
# ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# ax.yaxis.set_major_locator(plt.MaxNLocator(5))
# ax.zaxis.set_major_locator(plt.MaxNLocator(5))



cm = plt.get_cmap('rainbow')
cNorm = matplotlib.colors.Normalize(vmin=min(W), vmax=max(W))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
# ax.set_xlabel('X-coordinate (m)', fontsize=12)
# ax.set_ylabel('Y-coordinate (m)', fontsize=12)
# ax.set_zlabel('Z-coordinate (m)', fontsize=12)


ax.scatter(X,Z,Y, s = 1000000*W, c=scalarMap.to_rgba(W), label='Erosion profile')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')


# plt.title("Erosion profile (RNN+CNN) on %ith dataset"%sim_num, fontsize=15)
scalarMap.set_array(W)
def fmt(x, pos):
    if x>0:
       a, b = '{:.2e}'.format(x).split('e')
       b = int(b)
       return r'${} \times 10^{{{}}}$'.format(a, b)
cbar = plt.colorbar(scalarMap, shrink=0.7, pad=0.1, location='right', format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title('Erosion rate \n (kg/$m^2$·s)',fontsize=12)

ax.axes.set_xlim3d(left=-0.8, right=0.8)
ax.axes.set_ylim3d(bottom=-0.8, top=0.8)
ax.axes.set_zlim3d(bottom=-0.8, top=0.8)
# plt.grid()

# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

plt.tight_layout()


os.chdir('/bigdata/wonglab/syang159/OP650_1/level_8/new_2')
plt.savefig('plot_predicted_ero_%i.png'%sim_num)