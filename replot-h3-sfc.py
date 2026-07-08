import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

# 1. Load the corrected data structure
ds = xr.open_dataset("global_h3_res2_air.nc")

h3_indices = ds['h3_index'].values
air_vals = ds['air_h3'].values
lon_bnds = ds['longitude_bounds'].values
lat_bnds = ds['latitude_bounds'].values

# 2. Setup the Map View layout environment
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global()  # Keeps map perspective locked to full global frame

# 3. Assemble and extract geometric hexagon contours
patches = []
valid_data_values = []

for i in range(len(h3_indices)):
    if np.isnan(air_vals[i]):
        continue
        
    cell_lons = lon_bnds[i, :]
    cell_lats = lat_bnds[i, :]
    
    # Check for the antimeridian line split to prevent distorted visual stretching
    if np.max(cell_lons) - np.min(cell_lons) > 180:
        continue 
        
    polygon_vertices = np.column_stack((cell_lons, cell_lats))
    patches.append(Polygon(polygon_vertices, closed=True))
    valid_data_values.append(air_vals[i])

# Convert to array to calculate robust color scaling bounds
valid_data_values = np.array(valid_data_values)

# 4. Pack polygon elements into a fast unified vector rendering block
p_collection = PatchCollection(patches, transform=ccrs.PlateCarree(), cmap='plasma')
p_collection.set_array(valid_data_values)

# Explicitly set color scale limits based on your temperature dataset
p_collection.set_clim(vmin=np.nanmin(valid_data_values), vmax=np.nanmax(valid_data_values))
ax.add_collection(p_collection)

# 5. Formatting color ranges
cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Surface Temperature (K)')

plt.title("Corrected Interpolated Unstructured H3 Hexagon Grid (Res 2) - Air Temp")
plt.savefig("interpolated_h3_temp_map_fixed.png", bbox_inches='tight', dpi=150)
plt.show()

