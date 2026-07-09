import xarray as xr
import matplotlib
# Use a non-interactive backend for headless cluster nodes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon as ShapelyPolygon
import numpy as np
import glob
import os
from PIL import Image

# Create a clean folder to store temporary frames
os.makedirs("tmp_frames", exist_ok=True)

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
            if p.is_empty or p.geom_type == 'Point' or p.geom_type == 'LineString':
                continue
            if p.geom_type == 'MultiPolygon':
                result_polys.extend([sub_p for sub_p in p.geoms if not sub_p.is_empty])
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
h3_indices = ds.h3_index.values
num_hexagons = len(h3_indices)
num_frames = len(timestamps)

# Fix static color bar thresholds across the entire dynamic dataset limits
vmin_limit = np.nanmin(forecast_timeline)
vmax_limit = np.nanmax(forecast_timeline)

# 2. Render each time step frame to an isolated static image file using GeoPandas
print(f"Rendering {num_frames} sequential forecast frames to disk...")
for step_idx in range(num_frames):
    step_data = forecast_timeline[step_idx]
    grid_records = []
    
    for i in range(num_hexagons):
        val = step_data[i]
        if np.isnan(val):
            continue
            
        raw_poly = ShapelyPolygon(np.column_stack((lon_bnds[i, :], lat_bnds[i, :])))
        split_polys = split_dateline_polygon_perfect(raw_poly)
        
        for sub_poly in split_polys:
            if sub_poly.is_empty:
                continue
            grid_records.append({
                "air_temp": val,
                "geometry": sub_poly
            })
            
    # Pack into a GeoDataFrame for resilient rendering
    gdf = gpd.GeoDataFrame(grid_records, crs="EPSG:4326")
    
    # Initialize Map Plot
    fig = plt.figure(figsize=(12, 7))
    map_proj = ccrs.PlateCarree()
    map_proj._threshold /= 100.0  # Eliminate pixel jitter on edges

    ax = plt.axes(projection=map_proj)
    ax.coastlines(color='black', linewidth=1)
    ax.gridlines(draw_labels=True, linestyle='--')
    ax.set_global()

    # Draw using GeoPandas native matplot engine (highly resilient to empty geometries)
    gdf.plot(
        column='air_temp',
        ax=ax,
        cmap='plasma',
        transform=ccrs.PlateCarree(),
        edgecolor='face',
        linewidth=0.0,
        vmin=vmin_limit,
        vmax=vmax_limit
    )

    # Re-draw the colorbar manually to maintain consistent plotting bounds
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=vmin_limit, vmax=vmax_limit))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
    cbar.set_label('Surface Air Temperature (K)')
    
    lead_hours = (step_idx + 1) * 3
    ax.set_title(f"Auto-Regressive GATv2 Forecast | Lead Time: +{lead_hours} Hours", fontsize=14)
    
    # Save the frame
    frame_filename = f"tmp_frames/frame_{step_idx:03d}.png"
    plt.savefig(frame_filename, bbox_inches='tight', dpi=120)
    plt.close(fig)
    
    if (step_idx + 1) % 8 == 0:
        print(f" -> Saved frame {step_idx + 1}/{num_frames}")

# =====================================================================
# 3. STITCH THE COMPLETED FRAMES INTO A UNIFIED GIF FILE USING PILLOW
# =====================================================================
print("Compiling frames into a seamless rollout GIF animation...")
frame_files = sorted(glob.glob("tmp_frames/frame_*.png"))
images = [Image.open(f) for f in frame_files]

output_gif = "h3_autoregressive_rollout.gif"

# FIX: Call .save on the first image object, not the list wrapper itself
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:], # Append the rest of the frames
    duration=200,              # 200 milliseconds per frame (5 frames per second)
    loop=0                     # 0 means loop infinitely
)

# 4. Clean up temporary directory
print("Cleaning up temporary workspace files...")
for f in frame_files:
    os.remove(f)
os.rmdir("tmp_frames")

print("\n" + "="*50)
print(f"SUCCESS: Visual rollout animation saved to: {output_gif}")
print("="*50 + "\n")

