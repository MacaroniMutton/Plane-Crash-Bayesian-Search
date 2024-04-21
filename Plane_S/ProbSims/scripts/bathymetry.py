import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from sklearn import preprocessing
import seaborn as sns

data = xr.load_dataset('gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
elevation = data.elevation

li = np.array([[0]*84 for _ in range(75)])
mini = float('inf')
maxi = float('-inf')
for lat in range(75):
    for lon in range(84):
        li[74-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/75):(lat+1)*int(data.sizes['lat']/75), (lon)*int(data.sizes['lon']/84):(lon+1)*int(data.sizes['lon']/84)].mean().load())
        mini = min(mini, li[74-lat][lon])
        maxi = max(maxi, li[74-lat][lon])

# norm = [[(float(j)-mini)/(maxi-mini) for j in range(len(li[i]))] for i in range(len(li))]
# print(norm[-1][1])
# norm = [(float(i)-min(li))/(max(li)-min(li)) for i in li]
# normalized_arr = preprocessing.normalize(li)
# print(normalized_arr)
# for i in range(len(li)):
#     for j in range(len(li[i])):
#         li[i][j] += abs(mini)
# scaler = preprocessing.MinMaxScaler()
# d = scaler.fit_transform(li)
# print(d[-1])
# print(li[-1])
# sns.heatmap(d)
# plt.plot()

li = (li-mini)
li = li/(maxi-mini)

cmap = matplotlib.colormaps.get_cmap('coolwarm')
print(cmap(li[0,0])*255)