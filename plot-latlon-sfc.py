import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 1. Load the surface temperature regular lat/lon dataset
ds_air = xr.open_dataset("air.sfc.2000.nc")

# 2. Extract the first time step slice
# xarray automatically handles missing values using the metadata flag
air_slice = ds_air['air'].isel(time=0)

# 3. Initialize the cartopy map canvas
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True, linestyle='--')

# 4. Plot the regular grid scalar field
im = air_slice.plot(
    ax=ax, 
    transform=ccrs.PlateCarree(), 
    cmap='plasma', # Switching map color style for temperature metrics
    cbar_kwargs={'label': 'Surface Air Temperature (K)'}
)

plt.title(f"Original 20CRv2c Surface Temp - Time: {str(air_slice.time.values)[:13]}")
plt.savefig("original_temp_map.png", bbox_inches='tight', dpi=150)
plt.show()

