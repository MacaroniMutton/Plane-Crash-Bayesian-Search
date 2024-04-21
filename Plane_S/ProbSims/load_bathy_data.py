import numpy as np
import xarray as xr
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

CELL_SIZE = 8
ROWS = 96
COLUMNS = 96

data = xr.load_dataset('C:\\Users\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\ProbSims\\gebco_2023_n10.0063_s5.0845_w79.4883_e84.9023.nc')

elevation = data.elevation
li = np.array([[0]*COLUMNS for _ in range(ROWS)])
lat_lng_li = [[None]*COLUMNS for _ in range(ROWS)]
mini = float('inf')
maxi = float('-inf')
for lat in range(96):
    for lon in range(96):
        lat_lng_li[95-lat][lon] = [[data.lat[(lat)*int(data.sizes['lat']/96)], data.lat[(lat+1)*int(data.sizes['lat']/96)]], [data.lon[(lon)*int(data.sizes['lon']/96)], data.lon[(lon+1)*int(data.sizes['lon']/96)]]]
        li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
        mini = min(mini, li[95-lat][lon])
        maxi = max(maxi, li[95-lat][lon])
lat_lng_li = np.array(lat_lng_li)
print(lat_lng_li)
li = (li-mini)
li = li/(maxi-mini)
print(li)
sns.heatmap(li)
plt.show()
# with open("test.txt", "wb") as fp:   #Pickling
#     pickle.dump(li, fp)

