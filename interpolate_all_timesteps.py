import xarray as xr
import numpy as np

# ===================================================
# STEP 1: LOAD & ALIGN GRID STRUCTURES
# ===================================================
print("Loading master H3 geometry template definitions...")
ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc")

print("Loading full 3-hourly time series weather profile...")
# Open dataset with chunking enabled (Dask background handles parallel operations)
ds_air = xr.open_dataset("air.sfc.2000.nc", chunks={"time": 500})

# Isolate coordinate markers
h3_lons = ds_h3['longitude'].values
h3_lats = ds_h3['latitude'].values

# Force longitudes into continuous 0 to 360 space to match NOAA format
h3_lon_converted = np.mod(h3_lons, 360)

# Check and sort dataset latitudes to ensure ascending order (-90 to 90)
if ds_air.lat.values[0] > ds_air.lat.values[-1]:
    print("Inverting input latitude coordinates axis grid...")
    ds_air = ds_air.sortby('lat')

# ===================================================
# STEP 2: APPLY TEMPORAL CIRCULAR LONGITUDE PADDING
# ===================================================
print("Applying circular longitude padding across all time steps...")
# Extract dimension variables
raw_lons = ds_air.lon.values
padded_lons = np.append(raw_lons, 360.0)

# Extract the full 3D array data chunk (time, lat, lon)
raw_air_array = ds_air['air']

# Append the 0-degree longitude column slice to the end of the array
# index axis=2 corresponds to the 'lon' dimension in the (time, lat, lon) structure
padded_air_array = xr.concat([raw_air_array, raw_air_array.isel(lon=slice(0, 1))], dim="lon")
# Re-assign coordinate markers to account for the new column position
padded_air_array = padded_air_array.assign_coords(lon=padded_lons)

# ===================================================
# STEP 3: EXECUTE VECTORIZED 3D TIME SERIES LOOKUP
# ===================================================
print(f"Executing vectorized interpolation across {len(ds_air.time)} time steps...")
# Build 1D target lookup arrays mapped down to the h3_index dimension
target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"], coords={"h3_index": ds_h3.h3_index})
target_lat = xr.DataArray(h3_lats, dims=["h3_index"], coords={"h3_index": ds_h3.h3_index})

# Run the vectorized interpolation
# This calculates spatial interpolation for all 2,928 steps simultaneously
interpolated_ds = padded_air_array.interp(
    lon=target_lon, 
    lat=target_lat, 
    method="linear", 
    kwargs={'bounds_error': False, 'fill_value': None}
)

# ===================================================
# STEP 4: ASSEMBLE & EXPORT TEMPORAL NETCDF FILE
# ===================================================
print("Structuring compliant output metadata variables...")
# Prepare the output container dataset
ds_output = xr.Dataset(
    data_vars={
        "air_h3": (
            ["time", "h3_index"], 
            interpolated_ds.values, 
            {
                "units": "degK", 
                "long_name": "3-hourly Surface Air Temperature Interpolated to H3 Mesh",
                "coordinates": "longitude latitude"
            }
        ),
        "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
        "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
        "longitude_bounds": (["h3_index", "vertices"], ds_h3['longitude_bounds'].values, {"units": "degrees_east"}),
        "latitude_bounds": (["h3_index", "vertices"], ds_h3['latitude_bounds'].values, {"units": "degrees_north"}),
    },
    coords={
        "time": ds_air.time.values,
        "h3_index": ds_h3.h3_index.values,
        "vertices": np.arange(6)
    },
    attrs={
        "title": "Full Time-Series Global H3 Grid (Res 2)",
        "source_dataset": "NOAA-CIRES 20th Century Reanalysis V2c",
        "total_time_steps": len(ds_air.time)
    }
)

output_filename = "global_h3_res2_air_all_times.nc"
print(f"Writing time-series file to disk: {output_filename}...")
# Use NETCDF4 formatting to handle large multi-dimensional matrices cleanly
ds_output.to_netcdf(output_filename, format="NETCDF4")

print("\n" + "="*50)
print("TIME SERIES TEMPORAL INTERPOLATION SUCCESSFUL")
print("="*50)
print(f"Output File Name      : {output_filename}")
print(f"Result Matrix Shape   : {ds_output['air_h3'].shape} (time x h3_index)")
print(f"Processed Time Steps  : {len(ds_output.time)} frames")
print(f"Total H3 Cells/Frame  : {len(ds_output.h3_index)} hexagons")
print("="*50 + "\n")

