import numpy as np
import xarray as xr
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

CELL_SIZE = 8
ROWS = 96
COLUMNS = 96

data = xr.load_dataset('C:\\Users\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\ProbSims\\gebco_2023_n10.0063_s5.0845_w79.4883_e84.9023.nc')
print(int(95*(data.sizes['lat']/96)))



elevation = data.elevation
li = np.array([[0]*COLUMNS for _ in range(ROWS)])
lat_lng_li = [[None]*COLUMNS for _ in range(ROWS)]
mini = float('inf')
maxi = float('-inf')
for lat in range(96):
    for lon in range(96):
        lat_lng_li[95-lat][lon] = [[data.lat[int(lat*(data.sizes['lat']/96))], data.lat[int((lat+1)*(data.sizes['lat']/96))-1]], [data.lon[int(lon*(data.sizes['lon']/96))], data.lon[int((lon+1)*(data.sizes['lon']/96))-1]]]
        li[95-lat][lon] = (elevation[int(lat*(data.sizes['lat']/96)):int((lat+1)*(data.sizes['lat']/96))-1, int(lon*(data.sizes['lon']/96)):int((lon+1)*(data.sizes['lon']/96))-1].mean().load())
        mini = min(mini, li[95-lat][lon])
        maxi = max(maxi, li[95-lat][lon])
lat_lng_li = np.array(lat_lng_li)
print(lat_lng_li)
li = (li-mini)
li = li/(maxi-mini)
print(li)




