import argparse
import os
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

class IcosahedralToRegularInterpolator:
    """
    Transforms unstructured 1D GraphCast mesh projections back into conformed
    2D regular lat/lon fields matching an operational template footprint.
    """
    def __init__(self, input_preds_path: str, output_path: str, target_resolution: float = 1.0, template_path: str = None):
        self.input_preds_path = input_preds_path
        self.output_path = output_path
        self.res = target_resolution
        self.template_path = template_path
        
        self.ds_src = None
        self.var_name = None
        self.grid_lons = None
        self.grid_lats = None
        self.lon_name = 'longitude'
        self.lat_name = 'latitude'

    def build_target_latlon_matrix(self):
        """Generates or imports the target grid coordinate arrays."""
        if self.template_path and os.path.exists(self.template_path):
            print(f"[STAGE 1] Extracting template grid layout dimensions from: {self.template_path}")
            with xr.open_dataset(self.template_path) as ds_temp:
                self.lat_name = 'lat' if 'lat' in ds_temp.coords else 'latitude'
                self.lon_name = 'lon' if 'lon' in ds_temp.coords else 'longitude'
                self.grid_lats = ds_temp[self.lat_name].values
                self.grid_lons = ds_temp[self.lon_name].values
            print(f" -> Imported Template Footprint: ({len(self.grid_lats)} lats x {len(self.grid_lons)} lons)")
        else:
            print(f"[STAGE 1] Creating standard uniform grid layout at resolution: {self.res}°...")
            self.grid_lats = np.arange(-90.0, 90.0 + self.res, self.res)
            self.grid_lons = np.arange(0.0, 360.0, self.res)
            self.lat_name, self.lon_name = 'latitude', 'longitude'
            print(f" -> Generic Grid Footprint: ({len(self.grid_lats)} lats x {len(self.grid_lons)} lons)")

    def load_unstructured_source(self):
        print(f"[STAGE 2] Loading unstructured icosahedral dataset: {self.input_preds_path}")
        self.ds_src = xr.open_dataset(self.input_preds_path)

        ignore_keys = {"land_sea_mask", "elevation", "longitude", "latitude", "face_nodes", "x_cartesian", "y_cartesian", "z_cartesian", "icosahedral_mesh", "time", "node", "face", "three"}
        payload_vars = list(set(self.ds_src.data_vars.keys()) - ignore_keys)
        if not payload_vars:
            raise KeyError("Could not isolate an active weather variable payload in the input file.")
        
        self.var_name = payload_vars[0]
        print(f" -> Detected target weather variable: '{self.var_name}'")

    def compile_spatial_lookup_table(self) -> np.ndarray:
        print("[STAGE 3] Building 3D Cartesian KD-Tree over icosahedral nodes...")
        mesh_lons = self.ds_src['longitude'].values
        mesh_lats = self.ds_src['latitude'].values

        mesh_lon_rad = np.radians(np.mod(mesh_lons, 360))
        mesh_lat_rad = np.radians(mesh_lats)
        m_x = np.cos(mesh_lat_rad) * np.cos(mesh_lon_rad)
        m_y = np.cos(mesh_lat_rad) * np.sin(mesh_lon_rad)
        m_z = np.sin(mesh_lat_rad)
        mesh_cartesian = np.column_stack((m_x, m_y, m_z))

        tree = cKDTree(mesh_cartesian)

        # Build mesh grid matching coordinate mapping variables
        lon_mesh, lat_mesh = np.meshgrid(self.grid_lons, self.grid_lats)
        
        grid_lon_rad = np.radians(np.mod(lon_mesh.ravel(), 360))
        grid_lat_rad = np.radians(lat_mesh.ravel())
        g_x = np.cos(grid_lat_rad) * np.cos(grid_lon_rad)
        g_y = np.cos(grid_lat_rad) * np.sin(grid_lon_rad)
        g_z = np.sin(grid_lat_rad)
        grid_cartesian = np.column_stack((g_x, g_y, g_z))

        print(" -> Querying spatial indexes to connect grid elements...")
        _, closest_node_indices = tree.query(grid_cartesian, k=1)
        return closest_node_indices.reshape(lon_mesh.shape)

    def execute_reprojection(self) -> None:
        self.build_target_latlon_matrix()
        self.load_unstructured_source()
        
        lookup_matrix = self.compile_spatial_lookup_table()

        print(f"[STAGE 4] Mapping {len(self.ds_src.time)} timesteps to the 2D regular grid layout...")
        raw_payload = self.ds_src[self.var_name].values
        reprojected_data = raw_payload[:, lookup_matrix]

        print("[STAGE 5] Packaging outputs into standard CF-compliant NetCDF dataset...")
        
        # Strip suffix to match verification script assumptions
        clean_var_name = self.var_name.replace("_icosahedral_forecast", "").replace("_forecast", "")
        
        ds_regular = xr.Dataset(
            data_vars={
                clean_var_name: (
                    ["time", self.lat_name, self.lon_name],
                    reprojected_data.astype(np.float32),
                    {
                        "units": self.ds_src[self.var_name].attrs.get("units", "degK"),
                        "long_name": f"Regular Grid Reconstructed {clean_var_name.upper()}"
                    }
                )
            },
            coords={
                "time": self.ds_src.time.values,
                self.lat_name: self.grid_lats,
                self.lon_name: self.grid_lons
            },
            attrs={
                "title": "Regular Grid Reconstructed Climate Field",
                "conventions": "CF-1.6"
            }
        )

        print(f"[SAVE] Serializing conformed regular grid file to: {self.output_path}")
        ds_regular.to_netcdf(self.output_path, format="NETCDF4")
        
        self.ds_src.close()
        ds_regular.close()
        print("SUCCESS: Interpolation footprint conformed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Transform icosahedral variables back to match target grid shapes.")
    parser.add_argument("-i", "--input", required=True, help="Input unstructured file")
    parser.add_argument("-o", "--output", required=True, help="Output regular file")
    parser.add_argument("-r", "--resolution", type=float, default=1.0, help="Fallback generic resolution")
    parser.add_argument("-t", "--template", default=None, help="Path to ground truth template file to extract dimensions from")

    args = parser.parse_args()

    interpolator = IcosahedralToRegularInterpolator(
        input_preds_path=args.input,
        output_path=args.output,
        target_resolution=args.resolution,
        template_path=args.template
    )
    interpolator.execute_reprojection()

if __name__ == "__main__":
    main()
