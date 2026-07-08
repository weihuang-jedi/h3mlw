import geopandas as gpd
import xarray as xr
import numpy as np

# 1. Load the generated H3 GeoJSON file
geojson_file = "global_h3_res2.geojson"
print(f"Reading {geojson_file}...")
gdf = gpd.read_file(geojson_file)

num_hexagons = len(gdf)

# 2. Build explicit Triangle Meshes for OpenGL
# Each hexagon = 1 center vertex + 6 boundary vertices = 7 vertices total
# Each hexagon = 6 triangles * 3 indices = 18 elements in index array
vertices_list = []
vertex_data_list = []
indices_list = []

vertex_counter = 0

# Sample source metric data on per-hexagon level (e.g. Temperature)
# We will assign this value to the center and outer corner vertices
gdf["dummy_metric"] = np.random.uniform(10.0, 35.0, size=num_hexagons)

for i, row in enumerate(gdf.itertuples()):
    geom = row.geometry
    cell_value = row.dummy_metric
    
    # Calculate Center Point (Centroid)
    center_lon = geom.centroid.x
    center_lat = geom.centroid.y
    
    # Extract outer boundary coords (Ignore 7th repeated point, we need the exact 6)
    outer_coords = list(geom.exterior.coords)[:6]
    
    # --- VERTEX BUFFER DATA ---
    # Store Vertex 0: Centroid
    vertices_list.append([center_lon, center_lat, 0.0]) # Z=0 for flat maps, or project to 3D sphere
    vertex_data_list.append(cell_value)
    center_idx = vertex_counter
    vertex_counter += 1
    
    # Store Vertices 1-6: Outer Corners
    start_outer_idx = vertex_counter
    for lon, lat in outer_coords:
        vertices_list.append([lon, lat, 0.0])
        vertex_data_list.append(cell_value) # Assigning metric data directly to vertex
        vertex_counter += 1
        
    # --- INDEX BUFFER DATA (EBO) ---
    # Create the 6 triangles connecting the center to the outer pairs
    for t in range(6):
        v1 = start_outer_idx + t
        v2 = start_outer_idx + ((t + 1) % 6) # Loop back to the first outer boundary vertex
        
        indices_list.append([center_idx, v1, v2])

# Convert mesh constructs to clean NumPy arrays
out_vertices = np.array(vertices_list, dtype=np.float32)       # Shape: (Total Vertices, 3)
out_vertex_data = np.array(vertex_data_list, dtype=np.float32) # Shape: (Total Vertices,)
out_indices = np.array(indices_list, dtype=np.int32)           # Shape: (Total Triangles, 3)

# 3. Create OpenGL Structural NetCDF Dataset
dataset = xr.Dataset(
    data_vars={
        # Primary vector array for glVertexAttribPointer(XYZ Position)
        "vertex_positions": (
            ["num_vertices", "spatial_coordinates"], 
            out_vertices, 
            {"long_name": "Vertex Spatial Positions Lon Lat Z", "units": "degrees_and_meters"}
        ),
        # Target data mapped directly to vertices for shader color interpolation
        "vertex_metric_data": (
            ["num_vertices"], 
            out_vertex_data, 
            {"long_name": "Interpolated Metric Value at Vertex", "units": "custom_units"}
        ),
        # The exact Element Index Array for glDrawElements(GL_TRIANGLES)
        "triangle_indices": (
            ["num_triangles", "nodes_per_triangle"], 
            out_indices, 
            {"long_name": "Zero-Indexed Triangle Connectivity Table Map"}
        )
    },
    coords={
        "spatial_coordinates": ["lon", "lat", "z"],
        "nodes_per_triangle": [0, 1, 2]
    },
    attrs={
        "title": "Mesh Grid optimized for Starviewer OpenGL Engine Rendering",
        "mesh_type": "Unstructured Triangle Mesh (Tesselated Hexagons)",
        "total_vertices": len(out_vertices),
        "total_triangles": len(out_indices)
    }
)

# 4. Save to Disk
output_nc = "starviewer_opengl_grid.nc"
dataset.to_netcdf(output_nc, format="NETCDF4")

print(f"File saved successfully for starviewer graphics execution!")
print(f"Total Vertices: {len(out_vertices)} | Total Triangles: {len(out_indices)}")

