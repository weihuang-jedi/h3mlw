import xarray as xr
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

# ==========================================
# STEP 1: LOAD & HARD-ALIGN DATA STRUCTURES
# ==========================================
print("Loading datasets...")
ds_h3_raw = xr.open_dataset("global_h3_res2_with_bounds.nc")
ds_air = xr.open_dataset("air.sfc.2000.nc")

# 1. Force strict sorting of the H3 dataset by its index coordinate
ds_h3 = ds_h3_raw.sortby('h3_index')

# 2. Extract raw elements from the weather dataset
raw_lats = ds_air['lat'].values
raw_lons = ds_air['lon'].values
# Shape is (lat=94, lon=192)
raw_air_matrix = ds_air['air'].isel(time=0).values 

# 3. CRITICAL STRUCTURAL FIX: If lats run North-to-South, reverse the lats array 
# AND explicitly flip the rows of the data matrix using np.flipud()
if raw_lats[0] > raw_lats[-1]:
    print("Detected descending North-to-South grid. Flipping data matrix rows...")
    sorted_lats = np.flip(raw_lats)
    sorted_air_matrix = np.flipud(raw_air_matrix)
else:
    sorted_lats = raw_lats
    sorted_air_matrix = raw_air_matrix

# 4. Circular Longitude Padding to close the 358.125 to 360 degree gap
padded_lons = np.append(raw_lons, 360.0)
# Duplicate the 0-degree data column over to the 360-degree position
final_air_matrix = np.concatenate([sorted_air_matrix, sorted_air_matrix[:, :1]], axis=1)

# Rebuild a perfectly clean, aligned xarray dataset container
clean_air = xr.DataArray(
    final_air_matrix,
    coords={'lat': sorted_lats, 'lon': padded_lons},
    dims=['lat', 'lon']
)

# ==========================================
# STEP 2: RUN SAFE ACCURATE INTERPOLATION
# ==========================================
print("Interpolating data onto aligned H3 targets...")
h3_lon_raw = ds_h3['longitude'].values
h3_lat_raw = ds_h3['latitude'].values

# Force longitudes into a continuous 0 to 360 space to eliminate vertical strips
h3_lon_converted = np.mod(h3_lon_raw, 360)

# Map explicit 1D DataArrays for positional lookup
target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
target_lat = xr.DataArray(h3_lat_raw, dims=["h3_index"])

# Perform the bilinear interpolation
interpolated_air_ds = clean_air.interp(
    lon=target_lon, 
    lat=target_lat, 
    method="linear", 
    kwargs={'bounds_error': False, 'fill_value': None}
)
interpolated_values = interpolated_air_ds.values

# Save the properly sorted and interpolated dataset to a new file
ds_output = ds_h3.copy()
ds_output['air_h3'] = (['h3_index'], interpolated_values, {
    "units": "degK",
    "long_name": "Surface Air Temperature Interpolated to H3 Centroids",
    "coordinates": "longitude latitude"
})
ds_output.to_netcdf("global_h3_res2_air.nc", format="NETCDF4")
print("Saved sorted and interpolated data structure to: global_h3_res2_air.nc")

# ==========================================
# STEP 3: RENDERING STABLE RASTER MAP
# ==========================================
print("Assembling geometric plot maps...")
h3_indices = ds_output['h3_index'].values
lon_bnds = ds_output['longitude_bounds'].values
lat_bnds = ds_output['latitude_bounds'].values

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global() 

patches = []
valid_data_values = []

for i in range(len(h3_indices)):
    val = interpolated_values[i]
    if np.isnan(val):
        continue
        
    cell_lons = lon_bnds[i, :]
    cell_lats = lat_bnds[i, :]
    
    # Generate spatial boundaries
    raw_poly = ShapelyPolygon(np.column_stack((cell_lons, cell_lats)))
    split_polys = split_dateline_polygon(raw_poly)
    
    for sub_poly in split_polys:
        sub_x, sub_y = sub_poly.exterior.coords.xy
        polygon_vertices = np.column_stack((sub_x, sub_y))
        
        patches.append(MtplPolygon(polygon_vertices, closed=True))
        valid_data_values.append(val)

valid_data_values = np.array(valid_data_values)

print(f"Rendering {len(patches)} aligned hexagons...")
p_collection = PatchCollection(patches, transform=ccrs.PlateCarree(), cmap='plasma')
p_collection.set_array(valid_data_values)
p_collection.set_clim(vmin=np.min(valid_data_values), vmax=np.max(valid_data_values))
ax.add_collection(p_collection)

cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Surface Temperature (K)')

plt.title("Seamless Interpolated Global H3 Hexagon Grid (Res 2)")
output_img = "global_h3_temp_perfect.png"
plt.savefig(output_img, bbox_inches='tight', dpi=150)
print(f"Successfully generated clean global visualization at: {output_img}")
plt.show()

