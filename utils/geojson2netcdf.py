import geopandas as gpd
import xarray as xr
import numpy as np

# 1. Load the generated H3 GeoJSON file
geojson_file = "global_h3_res2.geojson"
print(f"Reading {geojson_file}...")
gdf = gpd.read_file(geojson_file)

# 2. Extract indices and calculate centroids
h3_indices = gdf["h3_index"].values
centroids = gdf.geometry.centroid
lons = centroids.x.values
lats = centroids.y.values

# 3. Preallocate empty arrays for boundary vertices
# Dimensions: (number_of_hexagons, 6 vertices)
num_hexagons = len(gdf)
lon_bounds = np.zeros((num_hexagons, 6))
lat_bounds = np.zeros((num_hexagons, 6))

# 4. Extract and normalize vertices for every hexagon
for i, geom in enumerate(gdf.geometry):
    # Get exterior coordinates (Shapely repeats the 1st vertex at the end, making 7 total)
    coords = list(geom.exterior.coords)
    
    # We only need the unique 6 vertices of the hexagon
    for vertex_idx in range(6):
        lon_bounds[i, vertex_idx] = coords[vertex_idx][0]
        lat_bounds[i, vertex_idx] = coords[vertex_idx][1]

# 5. Define dummy metric data for your variables
dummy_metric = np.random.uniform(10.0, 35.0, size=num_hexagons)

# 6. Build the CF-compliant xarray Dataset
dataset = xr.Dataset(
    data_vars={
        # Primary metric linked to geographic coordinates
        "surface_metric": (
            ["h3_index"], 
            dummy_metric, 
            {
                "units": "degrees_C", 
                "long_name": "Sample Surface Temperature",
                "coordinates": "longitude latitude"
            }
        ),
        
        # Centroid coordinate mappings with CF 'bounds' markers
        "longitude": (
            ["h3_index"], 
            lons, 
            {
                "units": "degrees_east", 
                "standard_name": "longitude", 
                "bounds": "longitude_bounds"
            }
        ),
        "latitude": (
            ["h3_index"], 
            lats, 
            {
                "units": "degrees_north", 
                "standard_name": "latitude", 
                "bounds": "latitude_bounds"
            }
        ),
        
        # Explicit geometric boundary coordinates
        "longitude_bounds": (
            ["h3_index", "vertices"], 
            lon_bounds, 
            {"units": "degrees_east"}
        ),
        "latitude_bounds": (
            ["h3_index", "vertices"], 
            lat_bounds, 
            {"units": "degrees_north"}
        ),
    },
    coords={
        "h3_index": h3_indices,
        "vertices": np.arange(6)  # The explicit 6-vertex structural index
    },
    attrs={
        "title": "Global H3 Resolution 2 Hexagonal Grid with Geometry Bounds",
        "spatial_resolution": "H3 Resolution 2 (~250km spacing)",
        "conventions": "CF-1.8"  # Crucial for software to read the bounds attribute
    }
)

# 7. Export to netCDF format
output_nc = "global_h3_res2_with_bounds.nc"
dataset.to_netcdf(output_nc, format="NETCDF4")

print(f"Successfully generated a NetCDF file with preserved geometry bounds!")
print(f"Saved to: {output_nc}")

