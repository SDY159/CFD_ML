# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
import matplotlib.ticker as ticker
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

sim_num = 296

path = ('/bigdata/wonglab/syang159/CFD1/OP650_2/')
os.chdir(path + 'cell_%i/'%sim_num)
import pandas as pd
df = pd.read_csv("corrosion_pressure")
X = df.iloc[:,1].values
Y = df.iloc[:,2].values
Z = df.iloc[:,3].values
W = df["dpm-erosion-rate-finnie"].values
fig = plt.figure()

ax=fig.add_subplot(111,projection='3d')
fig.set_size_inches(18.5, 10.5)

ax.plot_surface(Xm,Zm,Ym,alpha=0.1,color='blue')
ax.plot_surface(Xi1,Zi1,Yi1,alpha=0.1,color='blue')
ax.plot_surface(Xi2,Zi2,Yi2,alpha=0.1,color='blue')
ax.plot_surface(Xi3,Zi3,Yi3,alpha=0.1,color='blue')




cm = plt.get_cmap('rainbow')
cNorm = matplotlib.colors.Normalize(vmin=min(W), vmax=max(W))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
# ax.set_xlabel('X-coordinate (m)', fontsize=12)
# ax.set_ylabel('Y-coordinate (m)', fontsize=12)
# ax.set_zlabel('Z-coordinate (m)', fontsize=12)


ax.scatter(X,Z,Y, s = 1000000*W, c=scalarMap.to_rgba(W), label='Erosion profile')

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + (X.max()+X.min())
Yb = max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + (Y.max()+Y.min())
Zb = max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + (Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')


#plt.title("Erosion profile (CFD)", fontsize=15)
scalarMap.set_array(W)
def fmt(x, pos):
    if x>0:
       a, b = '{:.2e}'.format(x).split('e')
       b = int(b)
       return r'${} \times 10^{{{}}}$'.format(a, b)
cbar = plt.colorbar(scalarMap, shrink=0.7, pad=0.1, location='right', format=ticker.FuncFormatter(fmt))
cbar.ax.tick_params(labelsize=25)
cbar.ax.set_title('Erosion rate \n (kg/$m^2$·s)',fontsize=25)


ax.axes.set_xlim3d(left=-0.8, right=0.8)
ax.axes.set_ylim3d(bottom=-0.8, top=0.8)
ax.axes.set_zlim3d(bottom=-0.8, top=0.8)
# plt.grid()

#plt.tight_layout()


#plt.axis('off')

#ax.set_xlim(xmin=-1.0, xmax= 0.9)
#ax.set_ylim(ymin=0, ymax= 0.9) 
#ax.set_zlim(zmin=-0.9, zmax= 0) 
#ax.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax.yaxis.set_major_locator(plt.MaxNLocator(5))
#ax.zaxis.set_major_locator(plt.MaxNLocator(5))
#ax.tick_params(axis='both', which='major', labelsize=15)

# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

plt.tight_layout()

os.chdir('/bigdata/wonglab/syang159/CFD1/OP650_2/level_7')
plt.savefig('plot_original_%i.png'%sim_num)

