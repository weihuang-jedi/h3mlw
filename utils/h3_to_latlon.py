import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# ===================================================
# STEP 1: LOAD SOURCE UNSTRUCTURED H3 & TARGET SHAPE
# ===================================================
print("Loading interpolated H3 dataset and target grid metadata...")
# Load the unstructured H3 file we generated
ds_h3 = xr.open_dataset("global_h3_res2_air.nc")

# Load your original air template file to match its exact dimensions (lat: 94, lon: 192)
ds_template = xr.open_dataset("air.sfc.2000.nc")

# Isolate coordinate parameters
h3_lons = ds_h3['longitude'].values
h3_lats = ds_h3['latitude'].values
h3_vals = ds_h3['air_h3'].values

# The target output coordinates matching the template
# Note: Keep target longitudes in the original 0-360 range for alignment
target_lats = ds_template['lat'].values
target_lons = ds_template['lon'].values

# Ensure target lats are sorted to work with standard spatial visualization setups
if target_lats[0] > target_lats[-1]:
    target_lats_sorted = np.flip(target_lats)
else:
    target_lats_sorted = target_lats

# ===================================================
# STEP 2: PREPARE COORDINATE POINTS FOR GRIDDATA
# ===================================================
# Shift H3 longitudes from [-180, 180] to [0, 360] to align with the weather file
h3_lons_converted = np.mod(h3_lons, 360)

# Build the (N, 2) unstructured coordinate input array
points = np.column_stack((h3_lons_converted, h3_lats))

# Create the structured 2D coordinate mesh grids
lon_mesh, lat_mesh = np.meshgrid(target_lons, target_lats_sorted)

# ===================================================
# STEP 3: PERFORM UNSTRUCTURED-TO-GRID INTERPOLATION
# ===================================================
print("Interpolating unstructured hexagons onto the regular 2D matrix mesh...")
# We use linear interpolation. Near the extreme edges, we can fill remaining gaps
# using a fast nearest-neighbor pass or nearest-edge padding.
reconstructed_matrix = griddata(
    points, 
    h3_vals, 
    (lon_mesh, lat_mesh), 
    method='linear'
)

# Clean up any leftover boundary pixels near the extreme poles or dateline edges
# by applying a secondary nearest-neighbor pass for any NaN cells
nan_mask = np.isnan(reconstructed_matrix)
if np.any(nan_mask):
    print("Filling edge boundary gaps using nearest-neighbor lookup...")
    fallback_matrix = griddata(
        points, 
        h3_vals, 
        (lon_mesh, lat_mesh), 
        method='nearest'
    )
    reconstructed_matrix[nan_mask] = fallback_matrix[nan_mask]

# ===================================================
# STEP 4: REBUILD COMPLIANT STRUCTURED NETCDF
# ===================================================
print("Assembling structured output container...")

# Re-add the time axis dimension with length 1 to preserve downstream compatibility
# with pipelines or ingestion viewers expecting a (time, lat, lon) shape.
final_matrix_3d = np.expand_dims(reconstructed_matrix, axis=0) # Shape becomes (1, 94, 192)

# Copy the original coordinate variables and structural properties
ds_output = xr.Dataset(
    data_vars={
        "air": (
            ["time", "lat", "lon"], 
            final_matrix_3d, 
            {
                "long_name": "3-hourly Air Temperature at Surface (Reconstructed from H3)",
                "units": "degK",
                "standard_name": "air_temperature",
                "dataset": "Reconstructed from Unstructured H3 Grid"
            }
        )
    },
    coords={
        "time": ds_template['time'].isel(time=[0]), # Extract the first time value element
        "lat": target_lats_sorted,
        "lon": target_lons
    },
    attrs={
        "title": "NOAA-CIRES Reconstructed from H3 Resolution 2 Mesh",
        "Conventions": "CF-1.2"
    }
)

# Export back to disk
output_nc_file = "air.sfc.2000.reconstructed.nc"
ds_output.to_netcdf(output_nc_file, format="NETCDF4")

print("\n" + "="*50)
print("REVERSE GRID INTERPOLATION SUCCESSFUL")
print("="*50)
print(f"Output Matrix Shape  : {reconstructed_matrix.shape} (lat x lon)")
print(f"File Saved To        : {output_nc_file}")
print(f"Target Array Structure: ['air'] with dimensions (time, lat, lon)")
print("="*50 + "\n")

