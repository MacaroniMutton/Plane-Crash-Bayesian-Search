import numpy as np
import xarray as xr
import pickle

CELL_SIZE = 8
ROWS = 96
COLUMNS = 96

data = xr.load_dataset('C:\\Users\Manan Kher\\OneDrive\\Documents\\Plane-Crash-Bayesian-Search\\ProbSims\\gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
elevation = data.elevation
li = np.array([[0]*COLUMNS for _ in range(ROWS)])
mini = float('inf')
maxi = float('-inf')
for lat in range(96):
    for lon in range(96):
        li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
        mini = min(mini, li[95-lat][lon])
        maxi = max(maxi, li[95-lat][lon])
li = (li-mini)
li = li/(maxi-mini)
print(li)
with open("test.txt", "wb") as fp:   #Pickling
    pickle.dump(li, fp)
