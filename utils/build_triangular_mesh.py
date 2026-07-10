import argparse
import os
import xarray as xr
import torch
import numpy as np
from scipy.spatial import ConvexHull

class H3ToTriangularMeshCompiler:
    """
    Extracts unstructured H3 cells, projects them to 3D Cartesian space,
    computes a spherical Delaunay triangulation, and exports both a PyTorch (.pt)
    edge index and a standard UGRID CF-compliant NetCDF mesh file.
    """
    def __init__(self, input_h3_path: str, output_dir: str = "."):
        self.input_path = input_h3_path
        self.output_dir = output_dir
        self.ds = None

    def compile_triangulation(self, nc_filename: str = "global_triangular_mesh.nc") -> None:
        print(f"[STAGE 1] Opening core H3 master grid file: {self.input_path}")
        self.ds = xr.open_dataset(self.input_path)

        # Extract centroid coordinates
        lons = self.ds['longitude'].values
        lats = self.ds['latitude'].values
        h3_indices = self.ds['h3_index'].values

        print("[STAGE 2] Projecting coordinates to 3D Cartesian space...")
        lon_rad = np.radians(lons)
        lat_rad = np.radians(lats)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        points_3d = np.column_stack((x, y, z))

        print("[STAGE 3] Computing global 3D convex hull triangulation...")
        hull = ConvexHull(points_3d)
        simplices = hull.simplices  # Shape: (num_faces, 3) representing node indices of each triangle

        print(f" -> Successfully constructed {len(simplices)} triangular faces.")

        # =================================================================
        # GENERATE PYTORCH DIRECTED EDGES
        # =================================================================
        print("[STAGE 4] Converting face arrays into directed edge indices...")
        edges_src = []
        edges_dst = []
        for face in simplices:
            n0, n1, n2 = face[0], face[1], face[2]
            edges_src.extend([n0, n1, n1, n2, n2, n0])
            edges_dst.extend([n1, n0, n2, n1, n0, n2])

        edge_stack = np.vstack((edges_src, edges_dst))
        unique_edges = np.unique(edge_stack, axis=1)
        triangular_edge_index = torch.from_numpy(unique_edges).long()

        os.makedirs(self.output_dir, exist_ok=True)
        pt_path = os.path.join(self.output_dir, "triangular_edge_index.pt")
        torch.save(triangular_edge_index, pt_path)
        print(f" -> Saved PyTorch edge index to {pt_path} (Shape: {triangular_edge_index.shape})")

        # =================================================================
        # GENERATE UGRID-COMPLIANT NETCDF FILE
        # =================================================================
        print("[STAGE 5] Formatting UGRID-compliant NetCDF mesh file...")
        
        # UGRID requires 1-based indexing for face connections, or an explicit start_index attribute set to 0.
        # We will use 0-based indexing and declare start_index = 0.
        face_nodes = simplices.astype(np.int32)

        ds_ugrid = xr.Dataset(
            data_vars={
                # Unstructured topology container variable
                "triangular_mesh": (
                    [], 
                    0, # Dummy value
                    {
                        "cf_role": "mesh_topology",
                        "topology_dimension": 2,
                        "node_coordinates": "longitude latitude",
                        "face_node_connectivity": "face_nodes",
                        "face_dimension": "face"
                    }
                ),
                # Face connectivity mapping (triangles linking back to node dimension)
                "face_nodes": (
                    ["face", "three"], 
                    face_nodes, 
                    {
                        "cf_role": "face_node_connectivity",
                        "start_index": 0,
                        "long_name": "Node indices defining each triangular face element"
                    }
                ),
                # Inherit the original node metrics if available, or generate a sample
                "elevation": (
                    ["node"],
                    self.ds["elevation"].values if "elevation" in self.ds else np.zeros_like(lons),
                    {
                        "long_name": "Surface Elevation Height Above Sea Level",
                        "units": "meters",
                        "coordinates": "longitude latitude",
                        "mesh": "triangular_mesh"
                    }
                ),
                "land_sea_mask": (
                    ["node"],
                    self.ds["land_sea_mask"].values if "land_sea_mask" in self.ds else np.zeros_like(lons),
                    {
                        "long_name": "Land Sea Mask",
                        "units": "fraction",
                        "coordinates": "longitude latitude",
                        "mesh": "triangular_mesh"
                    }
                ),
                # Point-location coordinates
                "longitude": (["node"], lons, {"units": "degrees_east", "standard_name": "longitude"}),
                "latitude": (["node"], lats, {"units": "degrees_north", "standard_name": "latitude"}),
            },
            coords={
                "node": np.arange(len(lons)),
                "face": np.arange(len(simplices)),
                "three": np.arange(3),
                "h3_index": (["node"], h3_indices)
            },
            attrs={
                "title": "Global Triangular Mesh generated from H3 Grid (UGRID Conformed)",
                "conventions": "CF-1.8 UGRID-1.0",
                "source_h3_file": self.input_path
            }
        )

        nc_path = os.path.join(self.output_dir, nc_filename)
        ds_ugrid.to_netcdf(nc_path, format="NETCDF4")
        print(f"SUCCESS: Compiled UGRID NetCDF file saved to: {nc_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Compile a global triangular UGRID NetCDF & edge .pt file.")
    parser.add_argument("-i", "--input", required=True, help="Path to master global_h3_res2_with_bounds.nc file")
    parser.add_argument("-o", "--output_dir", default=".", help="Target output folder")
    parser.add_argument("-f", "--filename", default="global_triangular_mesh.nc", help="Output NetCDF name")
    args = parser.parse_args()

    compiler = H3ToTriangularMeshCompiler(input_h3_path=args.input, output_dir=args.output_dir)
    compiler.compile_triangulation(nc_filename=args.filename)

if __name__ == "__main__":
    main()

