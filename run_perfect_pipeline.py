import xarray as xr
import h3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon as MtplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon

def split_dateline_polygon(poly):
    """
    Cleans topology and safely splits a polygon if it crosses the antimeridian (-180/180).
    """
    poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return []

    x, y = poly.exterior.coords.xy
    if np.max(x) - np.min(x) > 180:
        shifted_coords = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in zip(x, y)]
        shifted_poly = ShapelyPolygon(shifted_coords).buffer(0)
        
        west_box = ShapelyPolygon([(0, -90), (180, -90), (180, 90), (0, 90)])
        east_box = ShapelyPolygon([(180, -90), (360, -90), (360, 90), (180, 90)])
        
        poly_west = shifted_poly.intersection(west_box)
        poly_east = shifted_poly.intersection(east_box)
        
        from shapely.affinity import translate
        poly_east_shifted = translate(poly_east, xoff=-360)
        
        result_polys = []
        for p in [poly_west, poly_east_shifted]:
            if p.is_empty:
                continue
            if p.geom_type == 'MultiPolygon':
                result_polys.extend(list(p.geoms))
            else:
                result_polys.append(p)
        return result_polys
    return [poly]

# ===================================================
# STEP 1: GENERATE A PERFECTLY SORTED GLOBAL H3 GRID
# ===================================================
print("Generating clean, strictly sorted global H3 grid...")
# Pull global cells from base cells
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

# CRITICAL SECURITY FIX: Convert to a list and SORT strictly.
# This guarantees that index i matches boundaries i across all arrays!
sorted_h3_indices = sorted(list(global_cells))
num_hexagons = len(sorted_h3_indices)

h3_lons = np.zeros(num_hexagons)
h3_lats = np.zeros(num_hexagons)
lon_bounds = np.zeros((num_hexagons, 6))
lat_bounds = np.zeros((num_hexagons, 6))

print("Computing exact geometry bounds and centroids...")
for i, cell in enumerate(sorted_h3_indices):
    # Centroids (PATCHED: cell_to_latlng with 'g')
    lat, lng = h3.cell_to_latlng(cell) if v4_api else h3.h3_to_geo(cell)
    h3_lats[i] = lat
    h3_lons[i] = lng
    
    # Boundary vertices
    boundary = h3.cell_to_boundary(cell) if v4_api else h3.h3_to_geo_boundary(cell)
    for v_idx in range(6):
        # H3 handles exactly 12 pentagons globally. If encountered, pad the final vertex
        if v_idx < len(boundary):
            lat_bounds[i, v_idx] = boundary[v_idx][0]
            lon_bounds[i, v_idx] = boundary[v_idx][1]
        else:
            lat_bounds[i, v_idx] = boundary[-1][0]
            lon_bounds[i, v_idx] = boundary[-1][1]

# ===================================================
# STEP 2: PREPARE AND FIX THE WEATHER INPUT GRID
# ===================================================
print("Loading weather dataset and preparing continuous spatial matrices...")
ds_air = xr.open_dataset("air.sfc.2000.nc")
air_slice = ds_air['air'].isel(time=0).drop_vars('time', errors='ignore')

# Handle descending latitude arrays cleanly via native xarray ordering
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

# ===================================================
# STEP 3: EXECUTE ACCURATE BILINEAR INTERPOLATION
# ===================================================
print("Running bilinear interpolation onto strictly matched target points...")
# Normalize H3 coordinates to match 0-360 space
h3_lon_converted = np.mod(h3_lons, 360)

target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

interpolated_air_ds = clean_air_grid.interp(
    lon=target_lon, 
    lat=target_lat, 
    method="linear", 
    kwargs={'bounds_error': False, 'fill_value': None}
)
interpolated_values = interpolated_air_ds.values

# ===================================================
# STEP 4: SAVE COMPLIANT STRUCTURED NETCDF FILE
# ===================================================
ds_output = xr.Dataset(
    data_vars={
        "air_h3": (["h3_index"], interpolated_values, {"units": "degK", "coordinates": "longitude latitude"}),
        "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
        "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
        "longitude_bounds": (["h3_index", "vertices"], lon_bounds, {"units": "degrees_east"}),
        "latitude_bounds": (["h3_index", "vertices"], lat_bounds, {"units": "degrees_north"}),
    },
    coords={
        "h3_index": sorted_h3_indices,
        "vertices": np.arange(6)
    },
    attrs={"title": "Perfectly Aligned Global H3 Grid for Ingestion"}
)
ds_output.to_netcdf("global_h3_res2_air.nc", format="NETCDF4")
print("Saved sorted and interpolated data structure to: global_h3_res2_air.nc")

# ===================================================
# STEP 5: VISUALIZE RESULTS
# ===================================================
print("Generating verified global raster plot map...")
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global()

patches = []
valid_data_values = []

for i in range(num_hexagons):
    val = interpolated_values[i]
    if np.isnan(val):
        continue
        
    cell_lons = lon_bounds[i, :]
    cell_lats = lat_bounds[i, :]
    
    raw_poly = ShapelyPolygon(np.column_stack((cell_lons, cell_lats)))
    split_polys = split_dateline_polygon(raw_poly)
    
    for sub_poly in split_polys:
        sub_x, sub_y = sub_poly.exterior.coords.xy
        patches.append(MtplPolygon(np.column_stack((sub_x, sub_y)), closed=True))
        valid_data_values.append(val)

valid_data_values = np.array(valid_data_values)

p_collection = PatchCollection(patches, transform=ccrs.PlateCarree(), cmap='plasma')
p_collection.set_array(valid_data_values)
p_collection.set_clim(vmin=np.min(valid_data_values), vmax=np.max(valid_data_values))
ax.add_collection(p_collection)

cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Surface Temperature (K)')

plt.title("Verified Aligned Global H3 Hexagon Grid (Res 2)")
output_img = "global_h3_temp_perfect.png"
plt.savefig(output_img, bbox_inches='tight', dpi=150)
print(f"Successfully generated clean global visualization map at: {output_img}")
plt.show()

