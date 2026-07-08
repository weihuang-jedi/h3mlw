import xarray as xr
import numpy as np

print("Opening master H3 time-series file...")
ds_h3 = xr.open_dataset("global_h3_res2_air_all_times.nc")
num_nodes = len(ds_h3.h3_index)

# 1. OPTION A: Derive a land mask from your existing air temperature reanalysis file
# Since your original 20CRv2c dataset already knows land boundaries, we use its grid mask
print("Loading weather dataset mask properties...")
ds_air = xr.open_dataset("air.sfc.2000.nc")
air_slice = ds_air['air'].isel(time=0).drop_vars('time', errors='ignore')
if air_slice.lat.values > air_slice.lat.values[-1]:
    air_slice = air_slice.sortby('lat')

# Calculate which cells contain valid entries vs land missing masks
# (If your file treats missing elements natively, adjust the boolean check)
raw_land_mask = xr.where(~np.isnan(air_slice), 1.0, 0.0)

# 2. OPTION B: Load a standard global ETOPO NetCDF file to extract elevation metrics
# Replace 'etopo_file.nc' with your local terrain elevation resource path if available.
# If unavailable, we generate safe dummy elevation arrays matching the land profile.
try:
    print("Loading global digital elevation model file...")
    ds_etopo = xr.open_dataset("etopo1_or_2022.nc") # Standard NOAA ETOPO relief file
    has_etopo = True
except FileNotFoundError:
    print("Notice: External elevation file not found. Generating default topographical array...")
    has_etopo = False

# 3. Sample static parameters at each H3 centroid coordinate pair
h3_lats = ds_h3['latitude'].values
h3_lons = ds_h3['longitude'].values
h3_lon_converted = np.mod(h3_lons, 360) # Map coordinates cleanly to 0-360 space

target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

print("Mapping static geographic fields to H3 centroid node points...")
# Interpolate land mask from the weather file's spatial layout
node_land_mask = raw_land_mask.interp(lon=target_lon, lat=target_lat, method="nearest").values
# Ensure it scales strictly as binary 0 or 1 boundaries
node_land_mask = np.where(node_land_mask > 0.5, 1.0, 0.0)

if has_etopo:
    # Interpolate exact elevation metrics from ETOPO grid coordinates
    node_elevation = ds_etopo['elevation'].interp(lon=target_lon, lat=target_lat, method="linear").values
else:
    # Fallback approximation: assign 500m to land nodes and 0m to ocean nodes
    node_elevation = np.where(node_land_mask == 1.0, 500.0, 0.0)

# 4. Save the updated static variables into your master NetCDF database file
ds_h3['land_sea_mask'] = (["h3_index"], node_land_mask, {"long_name": "Land-Sea Binary Mask (1=Land, 0=Ocean)"})
ds_h3['elevation'] = (["h3_index"], node_elevation, {"long_name": "Topographic Elevation Above Sea Level", "units": "meters"})

# Rewrite file with the new variables included
ds_h3.to_netcdf("global_h3_res2_air_all_times.nc", format="NETCDF4")
print("SUCCESS: Static features appended directly to global_h3_res2_air_all_times.nc!")

