import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon as MtplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np

def split_dateline_polygon_perfect(poly):
    """Splits a polygon cleanly if it crosses the ±180 degree antimeridian."""
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

# 1. Load data variables
print("Loading forecast structures for animation rendering...")
ds = xr.open_dataset("h3_autoregressive_forecast.nc")
ds_geom = xr.open_dataset("global_h3_res2_air_all_times.nc")

forecast_timeline = ds['air_forecast'].values  # Shape: (time, h3_index)
lon_bnds = ds_geom['longitude_bounds'].values
lat_bnds = ds_geom['latitude_bounds'].values
timestamps = ds.time.values
num_hexagons = len(ds.h3_index)

# 2. Preconstruct and clean spatial polygon shards
print("Pre-building geometric patch wireframes...")
polygon_mappings = [] # Will hold a list of (list_of_patches, source_hexagon_index)
for i in range(num_hexagons):
    raw_poly = ShapelyPolygon(np.column_stack((lon_bnds[i, :], lat_bnds[i, :])))
    split_polys = split_dateline_polygon_perfect(raw_poly)
    
    sub_patches = []
    for sub_poly in split_polys:
        sx, sy = sub_poly.exterior.coords.xy
        sub_patches.append(MtplPolygon(np.column_stack((np.clip(sx, -180.0, 180.0), sy)), closed=True))
    polygon_mappings.append((sub_patches, i))

# 3. Setup global animation map viewport
fig = plt.figure(figsize=(12, 7))
map_proj = ccrs.PlateCarree()
map_proj._threshold /= 100.0  # Eliminate pixel jitter on edges

ax = plt.axes(projection=map_proj)
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global()

# Fix static color bar thresholds across the entire dynamic dataset limits
vmin_limit = np.nanmin(forecast_timeline)
vmax_limit = np.nanmax(forecast_timeline)

# Initialize an empty PatchCollection slot to modify during playback loops
p_collection = PatchCollection([], transform=map_proj, cmap='plasma')
p_collection.set_clim(vmin=vmin_limit, vmax=vmax_limit)
p_collection.set_edgecolor('face')
p_collection.set_linewidth(0.0)
ax.add_collection(p_collection)

cbar = fig.colorbar(p_collection, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
cbar.set_label('Surface Air Temperature (K)')
title_text = ax.set_title("", fontsize=14)

# 4. Define the sequential animation update frame loop
def update_animation_frame(step_idx):
    step_data = forecast_timeline[step_idx]
    
    frame_patches = []
    frame_values = []
    
    # Re-map temperature data arrays onto pre-constructed geometry shards
    for sub_patches, hex_idx in polygon_mappings:
        val = step_data[hex_idx]
        if np.isnan(val):
            continue
        for patch in sub_patches:
            frame_patches.append(patch)
            frame_values.append(val)
            
    # Swap the internal geometry paths and color array updates instantly
    p_collection.set_paths(frame_patches)
    p_collection.set_array(np.array(frame_values))
    
    # Update title to show current forecast time step
    lead_hours = (step_idx + 1) * 3
    title_text.set_text(f"Auto-Regressive GATv2 Forecast | Lead Time: +{lead_hours} Hours")
    
    if (step_idx + 1) % 8 == 0:
        print(f" -> Rendered frame step {step_idx + 1}/{len(timestamps)}")
        
    return p_collection, title_text

# 5. Compile and export animation file
print(f"Compiling {len(timestamps)} forecast steps into a single rollout animation...")
anim = FuncAnimation(fig, update_animation_frame, frames=len(timestamps), interval=200, blit=True)

output_gif = "h3_autoregressive_rollout.gif"
# PillowWriter handles image grouping and exports cleanly on headless HPC clusters
anim.save(output_gif, writer=PillowWriter(fps=5))
plt.close()

print("\n" + "="*50)
print(f"SUCCESS: Visual rollout animation saved to: {output_gif}")
print("="*50 + "\n")

