import xarray as xr
import numpy as np

# 1. Load the verified interpolated dataset
print("Loading global_h3_res2_air.nc...")
ds = xr.open_dataset("global_h3_res2_air.nc")

h3_indices = ds['h3_index'].values
air_vals = ds['air_h3'].values
lons = ds['longitude'].values
lats = ds['latitude'].values
lon_bnds = ds['longitude_bounds'].values
lat_bnds = ds['latitude_bounds'].values

num_hexagons = len(h3_indices)

# 2. Allocate lists for the OpenGL Vertex and Element arrays
vertices_list = []       # Will hold flat [Lon, Lat, 0.0] positions
vertex_data_list = []   # Will hold per-vertex temperature data values
triangle_indices = []   # Element Index Buffer array (EBO pointers)

vertex_counter = 0

print("Triangulating hexagons into primitive vertex and index buffers...")
for i in range(num_hexagons):
    val = air_vals[i]
    
    # Handle missing value data records gracefully
    if np.isnan(val):
        val = 0.0 # or pass a default mask value expected by your shader
        
    center_lon = lons[i]
    center_lat = lats[i]
    
    # Extract the 6 outer corner coordinates
    cell_lon_bounds = lon_bnds[i, :]
    cell_lat_bounds = lat_bnds[i, :]
    
    # --- WRITE VERTEX POSITION & DATA BUFFERS ---
    # Index V0: Centroid/Center point
    vertices_list.append([center_lon, center_lat, 0.0]) # Flat Z=0 plane (or map to a 3D sphere)
    vertex_data_list.append(val)
    center_idx = vertex_counter
    vertex_counter += 1
    
    # Indices V1-V6: Outer Boundary Corners
    start_outer_idx = vertex_counter
    for v in range(6):
        vertices_list.append([cell_lon_bounds[v], cell_lat_bounds[v], 0.0])
        vertex_data_list.append(val)
        vertex_counter += 1
        
    # --- WRITE ELEMENT INDEX CONNECTIVITY TABLE (GL_TRIANGLES) ---
    # Construct a 6-part triangle fan around the center point
    for t in range(6):
        v1 = start_outer_idx + t
        v2 = start_outer_idx + ((t + 1) % 6) # Wrap the final triangle back to the first outer vertex
        
        # Store index triplet as a distinct triangle face
        triangle_indices.append([center_idx, v1, v2])

# Convert mesh data arrays to high-performance NumPy blocks
out_vertices = np.array(vertices_list, dtype=np.float32)       # Shape: (Total Vertices, 3)
out_vertex_data = np.array(vertex_data_list, dtype=np.float32) # Shape: (Total Vertices,)
out_indices = np.array(triangle_indices, dtype=np.int32)       # Shape: (Total Triangles, 3)

# 3. Create the OpenGL-structured NetCDF File
print("Assembling NetCDF structure...")
ds_mesh = xr.Dataset(
    data_vars={
        # Primary spatial buffer input for: glVertexAttribPointer(Location_XYZ_Position)
        "vertex_positions": (
            ["num_vertices", "spatial_coordinates"], 
            out_vertices, 
            {"long_name": "Vertex Flat Spatial Positions Lon Lat Z", "units": "degrees_and_meters"}
        ),
        # Metric buffer input for: glVertexAttribPointer(Location_Custom_Data_Scalar)
        "vertex_temperature": (
            ["num_vertices"], 
            out_vertex_data, 
            {"long_name": "Interpolated Temperature data at Node Vertex", "units": "degK"}
        ),
        # Connectivity lookup input for: glDrawElements(GL_TRIANGLES, ...)
        "triangle_indices": (
            ["num_triangles", "nodes_per_triangle"], 
            out_indices, 
            {"long_name": "Zero-Indexed OpenGL Triangle Element connectivity Map"}
        )
    },
    coords={
        "spatial_coordinates": ["lon", "lat", "z"],
        "nodes_per_triangle": [0, 1, 2]
    },
    attrs={
        "title": "Mesh Grid optimized for Starviewer OpenGL Engine Ingestion",
        "mesh_type": "Unstructured Triangulated H3 Hexagons Mesh",
        "total_vertices": len(out_vertices),
        "total_triangles": len(out_indices),
        "interpolation_source": "NOAA air.sfc.2000.nc"
    }
)

# Export the mesh file
output_mesh_file = "starviewer_opengl_grid.nc"
ds_mesh.to_netcdf(output_mesh_file, format="NETCDF4")

print("\n" + "="*50)
print("OPENGL MESH FILE GENERATION SUCCESSFUL")
print("="*50)
print(f"Output Saved To   : {output_mesh_file}")
print(f"Total Vertices    : {len(out_vertices)}  (Ready for VBO allocation)")
print(f"Total Triangles   : {len(out_indices)}  (Ready for EBO allocation)")
print(f"Indices Count     : {len(out_indices) * 3} elements total")
print("="*50 + "\n")

