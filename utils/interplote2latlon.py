import argparse
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

class H3ToLatLonInterpolator:
    """
    A class module to reverse-interpolate unstructured 1D H3 grid data 
    back into a structured 2D lat/lon rectangular matrix using a template.
    """
    def __init__(self, input_path: str, output_path: str, template_path: str):
        """
        Initializes the reverse interpolator with file configurations.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.template_path = template_path
        
        self.ds_h3 = None
        self.ds_template = None
        self.var_name = None
        self.h3_var_key = None

    def load_datasets(self) -> None:
        """
        Loads the NetCDF datasets and automatically detects variable naming layouts.
        """
        print(f"Loading unstructured H3 file: {self.input_path}")
        self.ds_h3 = xr.open_dataset(self.input_path)

        print(f"Loading grid template file: {self.template_path}")
        self.ds_template = xr.open_dataset(self.template_path)

        # DYNAMIC DETECTOR: Find the primary meteorological variable key in the template
        coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
        template_vars = list(set(self.ds_template.data_vars.keys()) - coords_keys)
        if not template_vars:
            raise KeyError("Could not detect a valid variable in the template NetCDF file.")
        self.var_name = template_vars[0]
        print(f" -> Detected weather variable name from template: '{self.var_name}'")

        # Dynamic detector for the matching data column inside the H3 input file
        h3_coords_keys = {"longitude", "latitude", "longitude_bounds", "latitude_bounds", "vertices"}
        h3_vars = list(set(self.ds_h3.data_vars.keys()) - h3_coords_keys)
        if not h3_vars:
            raise KeyError("Could not detect a valid data variable in the H3 input file.")
        
        # Prefer an exact variable name match or fall back to the first available data column
        self.h3_var_key = self.var_name if self.var_name in h3_vars else h3_vars[0]
        print(f" -> Mapping H3 source variable column: '{self.h3_var_key}'")

    def interpolate(self) -> None:
        """
        Performs the unstructured-to-structured matrix interpolation.
        """
        if self.ds_h3 is None or self.ds_template is None:
            raise ValueError("Datasets not loaded. Please call load_datasets() first.")

        # Extract source coordinates and values
        h3_lons = self.ds_h3['longitude'].values
        h3_lats = self.ds_h3['latitude'].values
        h3_vals = self.ds_h3[self.h3_var_key].values

        # If data is 2D (time, h3_index), slice the first time step for single-frame interpolation
        if h3_vals.ndim == 2:
            h3_vals = h3_vals[0, :]

        # Isolate destination coordinate structures
        target_lats = self.ds_template['lat'].values
        target_lons = self.ds_template['lon'].values

        # Enforce ascending order on latitudes for proper orientation tracking
        if target_lats[0] > target_lats[-1]:
            target_lats_sorted = np.flip(target_lats)
        else:
            target_lats_sorted = target_lats

        # Shift longitudes continuously to 0-360 space to avoid dateline gaps
        h3_lons_converted = np.mod(h3_lons, 360)
        points = np.column_stack((h3_lons_converted, h3_lats))

        # Build structural coordinate mesh grids
        lon_mesh, lat_mesh = np.meshgrid(target_lons, target_lats_sorted)

        print("Interpolating unstructured hexagons onto the regular 2D matrix mesh...")
        reconstructed_matrix = griddata(points, h3_vals, (lon_mesh, lat_mesh), method='linear')

        # Fallback pass to fill extreme boundaries using a nearest-neighbor strategy
        nan_mask = np.isnan(reconstructed_matrix)
        if np.any(nan_mask):
            print("Filling edge boundary gaps using nearest-neighbor lookup...")
            fallback_matrix = griddata(points, h3_vals, (lon_mesh, lat_mesh), method='nearest')
            reconstructed_matrix[nan_mask] = fallback_matrix[nan_mask]

        self._save_netcdf(reconstructed_matrix, target_lats_sorted, target_lons)

    def _save_netcdf(self, matrix: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
        """
        Saves the structured matrix back into a standard temporal NetCDF file.
        """
        print("Assembling structured output container...")
        final_matrix_3d = np.expand_dims(matrix, axis=0) # Shape: (1, lat, lon)

        # Pull original metadata attributes from the template to preserve units
        template_attr = self.ds_template[self.var_name].attrs
        long_name = template_attr.get("long_name", self.var_name.upper())
        units = template_attr.get("units", "unknown")

        ds_output = xr.Dataset(
            data_vars={
                self.var_name: (
                    ["time", "lat", "lon"],
                    final_matrix_3d,
                    {
                        "long_name": f"{long_name} (Reconstructed from H3)",
                        "units": units,
                        "standard_name": self.ds_template[self.var_name].attrs.get("standard_name", self.var_name),
                        "dataset": "Reconstructed Unstructured Mesh"
                    }
                )
            },
            coords={
                "time": self.ds_template['time'].isel(time=[0]),
                "lat": lats,
                "lon": lons
            },
            attrs={
                "title": f"Reconstructed {self.var_name.upper()} Layout from Global H3 Grid",
                "Conventions": "CF-1.2"
            }
        )

        print(f"Writing time-series file to disk: {self.output_path}...")
        ds_output.to_netcdf(self.output_path, format="NETCDF4")

        print("\n" + "="*50)
        print("REVERSE GRID INTERPOLATION SUCCESSFUL")
        print("="*50)
        print(f"Output Matrix Shape  : {matrix.shape} (lat x lon)")
        print(f"File Saved To        : {self.output_path}")
        print(f"Target Array Variable: ['{self.var_name}'] with dimensions (time, lat, lon)")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Object-oriented unstructured H3 to regular Lat/Lon reverse interpolator.")
    parser.add_argument("-i", "--input", required=True, help="Path to input unstructured H3 grid NetCDF file (e.g., global_h3_res2_air.nc)")
    parser.add_argument("-o", "--output", required=True, help="Path to save output reconstructed structured NetCDF file")
    parser.add_argument("-t", "--template", required=True, help="Path to original lat/lon template file (e.g., air.sfc.2000.nc)")
    args = parser.parse_args()

    processor = H3ToLatLonInterpolator(
        input_path=args.input, 
        output_path=args.output, 
        template_path=args.template
    )
    processor.load_datasets()
    processor.interpolate()

if __name__ == "__main__":
    main()

