import geopandas as gpd
import h3
from shapely.geometry import Polygon

# 1. Define target resolution (Resolution 2 provides ~158km edge lengths)
TARGET_RESOLUTION = 2

# 2. Safely get ALL H3 cells across the entire globe
# Start with the 122 base cells at Resolution 0
base_cells = h3.get_res0_cells()

global_cells = set()
for base_cell in base_cells:
    # Scale down from Resolution 0 to Resolution 2
    children = h3.cell_to_children(base_cell, TARGET_RESOLUTION)
    global_cells.update(children)

print(f"Generating global grid. Total hexagons to process: {len(global_cells)}")

# 3. Transform H3 cell indices into geometries
grid_data = []
for cell in global_cells:
    # Get the bounding vertices (Lat, Lon)
    boundary = h3.cell_to_boundary(cell)
    
    # Flip to (Lon, Lat) required by standard GeoJSON/GIS standards
    geometry_coords = [(lng, lat) for lat, lng in boundary]
    
    # Handle the antimeridian wrap-around to prevent stretched polygons
    lons = [coords[0] for coords in geometry_coords]
    if max(lons) - min(lons) > 180:
        # Shift negative longitudes to avoid clipping lines across the map
        geometry_coords = [(lng + 360 if lng < 0 else lng, lat) for lat, lng in geometry_coords]
        
    shapely_poly = Polygon(geometry_coords)
    
    grid_data.append({
        "h3_index": cell,
        "geometry": shapely_poly
    })

# 4. Convert to GeoDataFrame and assign standard coordinate reference system
gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

# 5. Export to disk
output_file = f"global_h3_res{TARGET_RESOLUTION}.geojson"
gdf.to_file(output_file, driver="GeoJSON")

print(f"Successfully saved global H3 grid to {output_file}!")

