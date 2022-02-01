import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib
from sklearn import preprocessing
import os

#see data type
#df.dtypes


path = ('/bigdata/wonglab/syang159/OP650_1/')
os.chdir(path + "cell_1")

df = pd.read_fwf("corrosion_pressure", colspecs=[(0,10), (11, 27) , (28, 44), (45, 61), (96,114)], header=None, names = ["cellnumber", "x", "y", "z", "dpm-erosion-rate-finnie"])
df = df.iloc[1:]
df = df.reset_index(drop=True)

lst_1 = range(0,2371)
df = df.drop(lst_1)
lst_2 = range (40683,42858)
df = df.drop(lst_2)


normalized_original = np.array(df['dpm-erosion-rate-finnie'])
normalized_original = preprocessing.normalize([normalized_original])
normalized_original = np.squeeze(normalized_original)
df["dpm-erosion-rate-finnie"] =normalized_original
df = df.reset_index(drop=True)

os.chdir(path + "level_6/original/")

for i in range (0, 625, 25):
	original_df = pd.read_fwf("original_3126_kfold_array_from_RNN%i.csv"%i, colspecs=[(0,15)], names = ["normalized-dpm-erosion-rate-finnie"])
	df['original']= original_df["normalized-dpm-erosion-rate-finnie"]
	# Creating dataset
	y = np.array(df["z"], dtype = float)
	x = np.array(df["x"], dtype = float)
	z = np.array(df["y"], dtype = float)
	cs = np.array(df['original'], dtype = float)

 
# Creating figure
	fig = plt.figure(figsize = (200, 100))
	ax = plt.axes(projection ="3d")
	ax.patch.set_facecolor('black')
 
# Add x, y gridlines
	ax.grid(b = True, linestyle ='-.', linewidth = 0.3, alpha = 0.2)

# Creating color map
	colorsMap ='jet'
	cm = plt.get_cmap('rainbow')
	cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=cs*10000+200)
	scalarMap.set_array(cs)
	fig.colorbar(scalarMap)

# Creating plot

	plt.savefig('2nd_plot_original%i.png'%i)



os.chdir(path + "level_6/predicted/")
for i in range (0, 625, 25):
	predicted_df = pd.read_fwf("predicted_3126_kfold_array_from_RNN%i.csv"%i, colspecs=[(0,15)], names = ["normalized-dpm-erosion-rate-finnie"])

	df['predicted']= predicted_df["normalized-dpm-erosion-rate-finnie"]

# Creating dataset
	y = np.array(df["z"], dtype = float)
	x = np.array(df["x"], dtype = float)
	z = np.array(df["y"], dtype = float)
	cs = np.array(df['predicted'], dtype = float)

 
# Creating figure
	fig = plt.figure(figsize = (200, 100))
	ax = plt.axes(projection ="3d")
	ax.patch.set_facecolor('black')
 
# Add x, y gridlines
	ax.grid(b = True, linestyle ='-.', linewidth = 0.3, alpha = 0.2)

# Creating color map
	colorsMap ='jet'
	cm = plt.get_cmap('rainbow')
	cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
	ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), s=cs*10000+200)
	scalarMap.set_array(cs)
	fig.colorbar(scalarMap)

# Creating plot

	plt.savefig('2nd_plot_predicted%i.png'%i)


