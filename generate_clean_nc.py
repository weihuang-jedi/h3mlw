import xarray as xr
import h3
import numpy as np

print("Generating clean, strictly sorted global H3 grid...")
try:
    base_cells = h3.get_res0_cells()  # v4 API
    v4_api = True
except AttributeError:
    base_cells = h3.get_res0_indexes()  # v3 API
    v4_api = False

global_cells = set()
for base in base_cells:
    children = h3.cell_to_children(base, 2) if v4_api else h3.h3_to_children(base, 2)
    global_cells.update(children)

sorted_h3_indices = sorted(list(global_cells))
num_hexagons = len(sorted_h3_indices)

h3_lons = np.zeros(num_hexagons)
h3_lats = np.zeros(num_hexagons)
lon_bounds = np.zeros((num_hexagons, 6))
lat_bounds = np.zeros((num_hexagons, 6))

print("Computing exact geometry bounds and centroids with strict WGS84 limits...")
for i, cell in enumerate(sorted_h3_indices):
    lat, lng = h3.cell_to_latlng(cell) if v4_api else h3.h3_to_geo(cell)
    h3_lats[i] = lat
    # Wrap centroid longitude cleanly to [-180, 180]
    h3_lons[i] = np.mod(lng + 180, 360) - 180
    
    boundary = h3.cell_to_boundary(cell) if v4_api else h3.h3_to_geo_boundary(cell)
    for v_idx in range(6):
        b_idx = v_idx if v_idx < len(boundary) else -1
        
        # FIX: Explicitly index the tuple indices for legacy H3 V3 support
        if v4_api:
            # v4 returns: [lat, lon] sequence elements directly
            raw_v_lat = boundary[b_idx][0] if isinstance(boundary[b_idx], (tuple, list, np.ndarray)) else boundary[b_idx]
            raw_v_lon = boundary[b_idx][1] if isinstance(boundary[b_idx], (tuple, list, np.ndarray)) else boundary[b_idx]
        else:
            # v3 returns: ((lat0, lon0), (lat1, lon1), ...) 2D tuples
            raw_v_lat = boundary[b_idx][0]
            raw_v_lon = boundary[b_idx][1]
            
        lat_bounds[i, v_idx] = raw_v_lat
        # CRITICAL SEAMLESS CLAMPING: Force every single boundary vertex into standard [-180, 180] space
        lon_bounds[i, v_idx] = np.mod(raw_v_lon + 180, 360) - 180

# Load weather dataset
# ===================================================
# STEP 2: PREPARE AND FIX THE WEATHER INPUT GRID
# ===================================================
print("Loading weather dataset...")
ds_air = xr.open_dataset("air.sfc.2000.nc")
air_slice = ds_air['air'].isel(time=0).drop_vars('time', errors='ignore')

# PATCH: Check only the first scalar element to detect descending order safely
if air_slice.lat.values[0] > air_slice.lat.values[-1]:
    air_slice = air_slice.sortby('lat')
air_slice = air_slice.compute()

# Close the 358.125° to 360° global edge wrap-around gap
new_lon_coords = np.append(air_slice.lon.values, 360.0)
padded_data = np.concatenate([air_slice.values, air_slice.values[:, :1]], axis=1)

clean_air_grid = xr.DataArray(
    padded_data,
    coords={'lat': air_slice.lat.values, 'lon': new_lon_coords},
    dims=['lat', 'lon']
)

print("Running bilinear interpolation onto clean targets...")
h3_lon_converted = np.mod(h3_lons, 360)
target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

interpolated_air_ds = clean_air_grid.interp(
    lon=target_lon, lat=target_lat, method="linear", 
    kwargs={'bounds_error': False, 'fill_value': None}
)
interpolated_values = interpolated_air_ds.values

# Save clean structures
ds_output = xr.Dataset(
    data_vars={
        "air_h3": (["h3_index"], interpolated_values, {"units": "degK", "coordinates": "longitude latitude"}),
        "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
        "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
        "longitude_bounds": (["h3_index", "vertices"], lon_bounds, {"units": "degrees_east"}),
        "latitude_bounds": (["h3_index", "vertices"], lat_bounds, {"units": "degrees_north"}),
    },
    coords={"h3_index": sorted_h3_indices, "vertices": np.arange(6)},
    attrs={"title": "Clean Verified Global H3 Grid with Clamped Coordinates"}
)
ds_output.to_netcdf("global_h3_res2_air.nc", format="NETCDF4")
print("Saved clean verified dataset to: global_h3_res2_air.nc")

