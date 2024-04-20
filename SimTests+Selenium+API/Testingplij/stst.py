import xarray as xr

data = xr.load_dataset('gebco_2023_n2.0_s0.0_w0.0_e2.0.nc')
print(data.lon[0:5])