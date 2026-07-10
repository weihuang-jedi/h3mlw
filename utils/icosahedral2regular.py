import argparse
import os
import xarray as xr
import numpy as np
import torch
from scipy.spatial import cKDTree

class IcosahedralToRegularInterpolator:
    """
    An object-oriented transformation module that reconstructs standard 2D regular 
    latitude-longitude NetCDF fields from unstructured 1D GraphCast-style 
    icosahedral mesh sequence predictions.
    """
    def __init__(self, input_preds_path: str, output_path: str, target_resolution: float = 1.0):
        """
        Args:
            input_preds_path (str): Path to your compiled icosahedral weather history or prediction Zarr/NetCDF file
            output_path (str): Target path for the output regular grid NetCDF file
            target_resolution (float): Spacing of the target regular lat/lon grid in degrees (default: 1.0)
        """
        self.input_preds_path = input_preds_path
        self.output_path = output_path
        self.res = target_resolution
        
        self.ds_src = None
        self.var_name = None
        self.grid_lons = None
        self.grid_lats = None

    def build_target_latlon_matrix(self):
        """Generates the coordinate arrays for the regular destination grid."""
        print(f"[STAGE 1] Creating target regular grid layout at resolution: {self.res}°...")
        # Ascending layout matching standard meteorological conventions
        self.grid_lats = np.arange(-90.0, 90.0 + self.res, self.res)
        self.grid_lons = np.arange(0.0, 360.0, self.res)
        print(f" -> Grid Matrix Footprint Dimensions: ({len(self.grid_lats)} lats x {len(self.grid_lons)} lons)")

    def load_unstructured_source(self):
        """Opens the source file and discovers the active weather variable payload."""
        print(f"[STAGE 2] Loading unstructured icosahedral dataset: {self.input_preds_path}")
        if self.input_preds_path.endswith('.zarr') or os.path.isdir(self.input_preds_path):
            self.ds_src = xr.open_zarr(self.input_preds_path, consolidated=True)
        else:
            self.ds_src = xr.open_dataset(self.input_preds_path)

        # Discovers variable payload name automatically while bypassing static geographic variables
        ignore_keys = {"land_sea_mask", "elevation", "longitude", "latitude", "face_nodes", "x_cartesian", "y_cartesian", "z_cartesian", "icosahedral_mesh"}
        payload_vars = list(set(self.ds_src.data_vars.keys()) - ignore_keys)
        if not payload_vars:
            raise KeyError("Could not isolate an active weather variable payload in the input file.")
        
        self.var_name = payload_vars[0]
        print(f" -> Detected target weather variable: '{self.var_name}'")

    def compile_spatial_lookup_table(self) -> np.ndarray:
        """
        Builds a 3D Cartesian KD-Tree over the icosahedral nodes to find the
        nearest node index for every pixel on the regular grid without meridian distortion.
        """
        print("[STAGE 3] Building 3D Cartesian KD-Tree over icosahedral nodes...")
        mesh_lons = self.ds_src['longitude'].values
        mesh_lats = self.ds_src['latitude'].values

        # Convert 1D icosahedral coordinates to 3D Cartesian coordinates to prevent 180/-180 meridian wrap errors
        mesh_lon_rad = np.radians(np.mod(mesh_lons, 360))
        mesh_lat_rad = np.radians(mesh_lats)
        m_x = np.cos(mesh_lat_rad) * np.cos(mesh_lon_rad)
        m_y = np.cos(mesh_lat_rad) * np.sin(mesh_lon_rad)
        m_z = np.sin(mesh_lat_rad)
        mesh_cartesian = np.column_stack((m_x, m_y, m_z))

        # Build lookup tree
        tree = cKDTree(mesh_cartesian)

        # Generate 2D flat coordinates for the regular target grid mesh
        lon_mesh, lat_mesh = np.meshgrid(self.grid_lons, self.grid_lats)
        
        # Convert the regular grid mesh coordinates to 3D Cartesian coordinates
        grid_lon_rad = np.radians(lon_mesh.ravel())
        grid_lat_rad = np.radians(lat_mesh.ravel())
        g_x = np.cos(grid_lat_rad) * np.cos(grid_lon_rad)
        g_y = np.cos(grid_lat_rad) * np.sin(grid_lon_rad)
        g_z = np.sin(grid_lat_rad)
        grid_cartesian = np.column_stack((g_x, g_y, g_z))

        print(" -> Querying spatial indexes to connect grid elements...")
        _, closest_node_indices = tree.query(grid_cartesian, k=1)
        
        # Reshape the flat 1D indices back into the original 2D (lat, lon) target grid shape
        return closest_node_indices.reshape(lon_mesh.shape)

    def execute_reprojection(self) -> None:
        """Executes the spatial mapping loop over the timeline and writes the regular NetCDF file."""
        self.build_target_latlon_matrix()
        self.load_unstructured_source()
        
        # Compile index translation array
        lookup_matrix = self.compile_spatial_lookup_table()

        print(f"[STAGE 4] Mapping {len(self.ds_src.time)} timesteps to the 2D regular grid layout...")
        
        # Pull the 1D unstructured data payload block into memory
        raw_payload = self.ds_src[self.var_name].values  # Shape: [time, node]
        
        # Vectorized array indexing maps all nodes to the 2D regular grid instantly
        reprojected_data = raw_payload[:, lookup_matrix]  # Shape: [time, lat, lon]

        print("[STAGE 5] Packaging outputs into standard CF-compliant NetCDF dataset...")
        ds_regular = xr.Dataset(
            data_vars={
                self.var_name.replace("_icosahedral", ""): (
                    ["time", "latitude", "longitude"],
                    reprojected_data.astype(np.float32),
                    {
                        "units": self.ds_src[self.var_name].attrs.get("units", "unknown"),
                        "long_name": f"Regular Grid Reconstructed {self.var_name.upper()}"
                    }
                )
            },
            coords={
                "time": self.ds_src.time.values,
                "latitude": self.grid_lats,
                "longitude": self.grid_lons
            },
            attrs={
                "title": f"Regular Grid Reconstructed Climate Field (Variable: {self.var_name.upper()})",
                "horizontal_resolution": f"{self.res} degrees",
                "conventions": "CF-1.6"
            }
        )

        print(f"[SAVE] Serializing regular grid file to target destination: {self.output_path}")
        ds_regular.to_netcdf(self.output_path, format="NETCDF4")
        
        self.ds_src.close()
        ds_regular.close()
        print("SUCCESS: Interpolation pipeline finished successfully!\n")


# =====================================================================
# SCRIPT CONTROLLER INTERFACE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Transform 1D Unstructured predictions back to standard 2D NetCDF Regular Grids."
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to input unstructured icosahedral prediction file (.nc or .zarr)")
    parser.add_argument("-o", "--output", required=True, 
                        help="Destination output path for your 2D regular file (.nc)")
    parser.add_argument("-r", "--resolution", type=float, default=1.0, 
                        help="Target regular grid resolution spacing in degrees (default: 1.0)")

    args = parser.parse_args()

    interpolator = IcosahedralToRegularInterpolator(
        input_preds_path=args.input,
        output_path=args.output,
        target_resolution=args.resolution
    )
    interpolator.execute_reprojection()

if __name__ == "__main__":
    main()
