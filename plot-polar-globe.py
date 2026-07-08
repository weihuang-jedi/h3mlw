import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon as MtplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

def split_dateline_polygon_perfect(poly):
    """Splits and cleans geometries crossing the international dateline."""
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

# 1. Load your verified dataset
print("Reading aligned global H3 NetCDF file...")
ds = xr.open_dataset("global_h3_res2_air.nc")

h3_indices = ds['h3_index'].values
air_vals = ds['air_h3'].values
lon_bnds = ds['longitude_bounds'].values
lat_bnds = ds['latitude_bounds'].values
num_hexagons = len(h3_indices)

# 2. Setup a Spherical Globe View Canvas
fig = plt.figure(figsize=(10, 10))

# FIX: Switch to Robinson or Orthographic to view the poles with correct spherical topology
# Orthographic(central_longitude=0, central_latitude=45) shows a realistic 3D space globe view.
view_proj = ccrs.Robinson(central_longitude=0)
data_proj = ccrs.PlateCarree() # Your source data coordinates remain flat lon/lat

ax = plt.axes(projection=view_proj)
ax.coastlines(color='black', linewidth=1)
ax.gridlines(linestyle='--')
ax.set_global()

patches = []
valid_data_values = []

print("Processing geometries for spherical rendering...")
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
        sub_x_clipped = np.clip(sub_x, -180.0, 180.0)
        
        patches.append(MtplPolygon(np.column_stack((sub_x_clipped, sub_y)), closed=True))
        valid_data_values.append(val)

valid_data_values = np.array(valid_data_values)

# 3. Create collection and draw shapes
# Passing transform=data_proj tells Cartopy to project your flat data onto the spherical globe
p_collection = PatchCollection(patches, transform=data_proj, cmap='plasma')
p_collection.set_array(valid_data_values)
p_collection.set_clim(vmin=np.min(valid_data_values), vmax=np.max(valid_data_values))

p_collection.set_edgecolor('face')
p_collection.set_linewidth(0.0)
ax.add_collection(p_collection)

# 4. Colorbar and Save
cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cbar.set_label('Surface Temperature (K)')

plt.title("Spherical Global H3 Hexagon Grid (Robinson View - No Sawtooth)")
output_img = "global_h3_spherical_view.png"
plt.savefig(output_img, bbox_inches='tight', dpi=150)
print(f"Successfully generated a clear visualization at: {output_img}")
plt.show()

