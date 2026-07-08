import xarray as xr
import numpy as np

print("Opening datasets...")
ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc")
ds_air = xr.open_dataset("air.sfc.2000.nc")

# 1. Extract the first time step and drop unused coordinates
air_slice = ds_air['air'].isel(time=0).drop_vars('time', errors='ignore').compute()

# 2. Sort the latitude dimension so it runs strictly South-to-North (-90 to 90)
if air_slice.lat[0] > air_slice.lat[-1]:
    air_slice = air_slice.sortby('lat')

# 3. Circular Longitude Padding to close the 358.125 to 360 degree gap
new_lon_coords = np.append(air_slice.lon.values, 360.0)

# Copy the 0-degree data array column over to the new 360-degree position
padded_data = np.concatenate([air_slice.values, air_slice.values[:, :1]], axis=1)

# Rebuild a clean, continuous xarray dataset
clean_air = xr.DataArray(
    padded_data,
    coords={'lat': air_slice.lat.values, 'lon': new_lon_coords},
    dims=['lat', 'lon']
)

# 4. Extract H3 centroid coordinates and convert longitudes to 0-360 space
h3_lon_raw = ds_h3['longitude'].values
h3_lat_raw = ds_h3['latitude'].values
h3_lon_converted = np.where(h3_lon_raw < 0, h3_lon_raw + 360, h3_lon_raw)

# 5. Pack targets into matching 1D DataArrays for positional lookup
target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
target_lat = xr.DataArray(h3_lat_raw, dims=["h3_index"])

print("Running seamless coordinate-aligned interpolation...")
# FIX: Use bounds_error=False and fill_value=None to trigger scipy's native extrapolation engine safely
interpolated_air_ds = clean_air.interp(
    lon=target_lon, 
    lat=target_lat, 
    method="linear", 
    kwargs={'bounds_error': False, 'fill_value': None}
)
interpolated_values = interpolated_air_ds.values

# --- CONSOLE DIAGNOSTICS ---
valid_mask = ~np.isnan(interpolated_values)
valid_data = interpolated_values[valid_mask]
print("\n" + "="*40)
print("UPDATED INTERPOLATION REPORT")
print("="*40)
print(f"Total H3 cells processed : {len(interpolated_values)}")
print(f"Valid data cells mapped  : {len(valid_data)} (Goal: 5882)")
print(f"Minimum Temperature      : {np.min(valid_data):.2f} K")
print(f"Maximum Temperature      : {np.max(valid_data):.2f} K")
print("="*40 + "\n")

# 6. Save results to the NetCDF file
ds_output = ds_h3.copy()
ds_output['air_h3'] = (['h3_index'], interpolated_values, {
    "units": "degK",
    "long_name": "Surface Air Temperature Interpolated to H3 Centroids",
    "coordinates": "longitude latitude"
})

output_filename = "global_h3_res2_air.nc"
ds_output.to_netcdf(output_filename, format="NETCDF4")
print(f"Saved successful output file to: {output_filename}")

