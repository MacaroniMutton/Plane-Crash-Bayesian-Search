import numpy as np
import xarray as xr
import pickle

# CELL_SIZE = 8
# ROWS = 96
# COLUMNS = 96

# data = xr.load_dataset('gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
# elevation = data.elevation
# li = np.array([[0]*COLUMNS for _ in range(ROWS)])
# mini = float('inf')
# maxi = float('-inf')
# for lat in range(96):
#     for lon in range(96):
#         li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
#         mini = min(mini, li[95-lat][lon])
#         maxi = max(maxi, li[95-lat][lon])
# li = (li-mini)
# li = li/(maxi-mini)
# print(li)
# with open("test.txt", "wb") as fp:   #Pickling
#     pickle.dump(li, fp)


def load_bathymetry_data(dataset_path):
    CELL_SIZE = 8
    ROWS = 96
    COLUMNS = 96

    data = xr.load_dataset(dataset_path)
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
    li = (li-mini)
    li = li/(maxi-mini)
    print(li)
    with open("C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\bathymetry_data.txt", "wb") as fp:   #Pickling
        pickle.dump(li, fp)
    return li, lat_lng_li

li, lat_lng_li = load_bathymetry_data('C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\ProbSims\\gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
print(lat_lng_li)
