import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# 1. Open both datasets
ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc")
ds_slp = xr.open_dataset("slp.mnmean.nc")

# 2. Extract target coordinates from the H3 grid file
# Convert H3 lons from [-180, 180] to match NOAA's [0, 360] system
h3_lon = np.where(ds_h3['longitude'].values < 0, ds_h3['longitude'].values + 360, ds_h3['longitude'].values)
h3_lat = ds_h3['latitude'].values

# 3. Prepare the source regular grid data (Time slice 0)
slp_slice = ds_slp['slp'].isel(time=0).compute()

# Ensure latitude is strictly increasing for the interpolator engine
if slp_slice.lat[0] > slp_slice.lat[-1]:
    slp_slice = slp_slice.sortby('lat')

src_lats = slp_slice.lat.values
src_lons = slp_slice.lon.values
src_values = slp_slice.values

# 4. Initialize the 2D RegularGridInterpolator
# We use 'bounds_error=False' to handle areas near boundary limits gracefully
interpolator = RegularGridInterpolator(
    (src_lats, src_lons), 
    src_values, 
    method='linear', 
    bounds_error=False, 
    fill_value=np.nan
)

# 5. Execute interpolation at the H3 centroid coordinate pairs
# Input format for RegularGridInterpolator points must be (Latitude, Longitude)
query_points = np.column_stack((h3_lat, h3_lon))
interpolated_slp = interpolator(query_points)

# 6. Save the results back into a new H3 NetCDF file
ds_output = ds_h3.copy()
ds_output['slp_h3'] = (['h3_index'], interpolated_slp, {
    "units": "mb",
    "long_name": "Sea Level Pressure Interpolated to H3 Centroids",
    "coordinates": "longitude latitude"
})

output_filename = "global_h3_res2_slp.nc"
ds_output.to_netcdf(output_filename, format="NETCDF4")
print(f"Successfully interpolated data and saved to: {output_filename}")

