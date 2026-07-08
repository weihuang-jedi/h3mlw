import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 1. Load the newly reverse-interpolated structured dataset
print("Loading reconstructed regular lat/lon dataset...")
ds_recon = xr.open_dataset("air.sfc.2000.reconstructed.nc")

# 2. Extract the data array variable
# Extract the single time step slice to match your original grid plot
air_slice = ds_recon['air'].isel(time=0)

# 3. Initialize the standard Cartopy projection map layout
plt.figure(figsize=(12, 6))

# Use the classic PlateCarree visual canvas
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(color='black', linewidth=1)
ax.gridlines(draw_labels=True, linestyle='--')
ax.set_global() # Keep map locked to a clean full global earth frame

# 4. Plot using standard xarray quadmesh mapping
# Passing transform=ccrs.PlateCarree() handles the 0-360 longitude mapping automatically
print("Rendering regular 2D raster mesh to canvas...")
im = air_slice.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cmap='plasma', # Keeping the same temperature color template for consistency
    cbar_kwargs={
        'label': 'Surface Air Temperature (K)', 
        'orientation': 'horizontal', 
        'pad': 0.08, 
        'shrink': 0.7
    }
)

plt.title("NOAA-CIRES Reconstructed 2D Grid from Unstructured H3 (Res 2)")

# 5. Save the output visual chart
output_img = "reconstructed_latlon_temp_map.png"
plt.savefig(output_img, bbox_inches='tight', dpi=150)
print(f"Successfully generated a clear visualization at: {output_img}")
plt.show()

