import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 1. Open both datasets
ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc")
ds_air = xr.open_dataset("air.sfc.2000.nc")

# 2. Extract target coordinates from H3 grid file
h3_lon_raw = ds_h3['longitude'].values
h3_lat = ds_h3['latitude'].values

# Shift longitudes from [-180, 180] to [0, 360] to align with NOAA format
h3_lon = np.where(h3_lon_raw < 0, h3_lon_raw + 360, h3_lon_raw)

# 3. Isolate the first time step slice
air_slice = ds_air['air'].isel(time=0).compute()

# CRITICAL FIX 1: Enforce explicit (lat, lon) row/column order via transpose
air_slice = air_slice.transpose('lat', 'lon')

# CRITICAL FIX 2: Sort latitude to strictly increasing order
if air_slice.lat[0] > air_slice.lat[-1]:
    air_slice = air_slice.sortby('lat')

src_lats = air_slice.lat.values
src_lons = air_slice.lon.values
src_values = air_slice.values  # This is now guaranteed to match shape (len(lat), len(lon))

# 4. Initialize the bilinear interpolator
# We use period handling or clamping bounds for edge points near 0/360
interpolator = RegularGridInterpolator(
    (src_lats, src_lons), 
    src_values, 
    method='linear', 
    bounds_error=False, 
    fill_value=None  # Use nearest edge extrapolation if slightly out of bounds
)

# 5. Route query matrix combinations (Latitude, Longitude)
query_points = np.column_stack((h3_lat, h3_lon))
interpolated_air = interpolator(query_points)

# 6. Build the output container matching original configurations
ds_output = ds_h3.copy()
ds_output['air_h3'] = (['h3_index'], interpolated_air, {
    "units": "degK",
    "long_name": "Surface Air Temperature Interpolated to H3 Centroids",
    "coordinates": "longitude latitude"
})

output_filename = "global_h3_res2_air.nc"
ds_output.to_netcdf(output_filename, format="NETCDF4")
print(f"Successfully mapped air temperature data over to: {output_filename}")

