import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon as MtplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

def split_dateline_polygon_perfect(poly):
    """
    Splits a polygon crossing the antimeridian and clips vertices 
    exactly at the -180 and 180 map boundaries.
    """
    poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return []

    x, y = poly.exterior.coords.xy
    if np.max(x) - np.min(x) > 180:
        shifted_coords = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in zip(x, y)]
        shifted_poly = ShapelyPolygon(shifted_coords).buffer(0)
        
        # Clip strictly between 0-180 and 180-360 space
        west_box = ShapelyPolygon([(0, -90), (180, -90), (180, 90), (0, 90)])
        east_box = ShapelyPolygon([(180, -90), (360, -90), (360, 90), (180, 90)])
        
        poly_west = shifted_poly.intersection(west_box)
        poly_east = shifted_poly.intersection(east_box)
        
        # Shift the 180-360 part back to the standard -180 to 0 grid space
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

# 1. Load your verified dataset
print("Reading aligned global H3 NetCDF file...")
ds = xr.open_dataset("global_h3_res2_air.nc")

h3_indices = ds['h3_index'].values
air_vals = ds['air_h3'].values
lon_bnds = ds['longitude_bounds'].values
lat_bnds = ds['latitude_bounds'].values
num_hexagons = len(h3_indices)

# 2. Setup Map Projection Canvas
fig = plt.figure(figsize=(12, 6))

map_proj = ccrs.PlateCarree()
# CRITICAL FIX: Drastically increase Cartopy's resolution threshold
# This forces the projection engine to slice edge-crossing shapes with perfect precision.
map_proj._threshold /= 100.0

ax = plt.axes(projection=map_proj)
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global()

patches = []
valid_data_values = []

print("Processing dateline edges for clean visual continuity...")
for i in range(num_hexagons):
    val = air_vals[i]
    if np.isnan(val):
        continue
        
    cell_lons = lon_bnds[i, :]
    cell_lats = lat_bnds[i, :]
    
    raw_poly = ShapelyPolygon(np.column_stack((cell_lons, cell_lats)))
    split_polys = split_dateline_polygon_perfect(raw_poly)
    
    for sub_poly in split_polys:
        sub_x, sub_y = sub_poly.exterior.coords.xy
        
        # Strict clipping to guarantee vertices match the map bounds exactly
        sub_x_clipped = np.clip(sub_x, -180.0, 180.0)
        
        patches.append(MtplPolygon(np.column_stack((sub_x_clipped, sub_y)), closed=True))
        valid_data_values.append(val)

valid_data_values = np.array(valid_data_values)

# 3. Create collection and draw shapes
p_collection = PatchCollection(patches, transform=map_proj, cmap='plasma')
p_collection.set_array(valid_data_values)
p_collection.set_clim(vmin=np.min(valid_data_values), vmax=np.max(valid_data_values))

# Force colors to bleed slightly into their sub-pixel seams to hide gaps completely
p_collection.set_edgecolor('face')
p_collection.set_linewidth(0.0)
ax.add_collection(p_collection)

# 4. Finalize framing dimensions
cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Surface Temperature (K)')

plt.title("Verified Aligned Global H3 Hexagon Grid (Perfect Seams)")
output_img = "global_h3_temp_perfect_seams.png"
plt.savefig(output_img, bbox_inches='tight', dpi=150)
print(f"Successfully generated a clear visualization at: {output_img}")
plt.show()

