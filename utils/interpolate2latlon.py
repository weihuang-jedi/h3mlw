import argparse
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
import functools

def _interpolate_single_time_step(t_idx, h3_vals_3d, points, lon_mesh, lat_mesh):
    """
    Worker function to process a single time step in parallel.
    Extracted to the top level so it can be cleanly pickled by multiprocessing.
    """
    # Isolate the 1D spatial vector for this specific time index
    h3_slice = h3_vals_3d[t_idx, :]
    
    # 1. Primary linear interpolation pass
    recon_2d = griddata(points, h3_slice, (lon_mesh, lat_mesh), method='linear')

    # 2. Fallback pass to fill extreme boundaries using nearest-neighbor strategy
    nan_mask = np.isnan(recon_2d)
    if np.any(nan_mask):
        fallback_2d = griddata(points, h3_slice, (lon_mesh, lat_mesh), method='nearest')
        recon_2d[nan_mask] = fallback_2d[nan_mask]
        
    return t_idx, recon_2d

class H3ToLatLonInterpolator:
    """
    A class module to reverse-interpolate multi-temporal unstructured 1D H3 grid data 
    back into a structured 3D lat/lon rectangular matrix series using a template.
    """
    def __init__(self, input_path: str, output_path: str, template_path: str, num_workers: int = None):
        """
        Initializes the reverse interpolator with file configurations.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.template_path = template_path
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        
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

        # 1. STRUCTURAL FIX: Extract the core target variable name from the TEMPLATE file
        coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
        template_vars = list(set(self.ds_template.data_vars.keys()) - coords_keys)
        if not template_vars:
            raise KeyError("Could not detect a valid variable in the template NetCDF file.")
        
        # Enforce raw string unpacking from list
        self.var_name = template_vars[0] if isinstance(template_vars, list) else template_vars
        print(f" -> Detected weather variable name from template: '{self.var_name}'")

        # 2. STRUCTURAL FIX: Extract the core target data variable from the H3 INPUT file
        h3_coords_keys = {"longitude", "latitude", "longitude_bounds", "latitude_bounds", "vertices"}
        h3_vars = list(set(self.ds_h3.data_vars.keys()) - h3_coords_keys)
        if not h3_vars:
            raise KeyError("Could not detect a valid data variable in the H3 input file.")
        
        # Pull the raw string element directly from the list mapping sequence
        self.h3_var_key = h3_vars[0] if isinstance(h3_vars, list) else h3_vars
        print(f" -> Mapping H3 source variable column: '{self.h3_var_key}'")

    def interpolate(self) -> None:
        """
        Performs the multi-temporal unstructured-to-structured matrix interpolation in parallel.
        """
        if self.ds_h3 is None or self.ds_template is None:
            raise ValueError("Datasets not loaded. Please call load_datasets() first.")

        # Extract source coordinates and full multi-temporal values explicitly from the DataArray
        h3_lons = self.ds_h3['longitude'].values
        h3_lats = self.ds_h3['latitude'].values
        
        # Pull data array from the isolated DataArray container specifically
        h3_vals_3d = self.ds_h3[self.h3_var_key].values 

        if h3_vals_3d.ndim == 1:
            # Fallback if a single-frame file is passed: expand to 2D matrix
            h3_vals_3d = np.expand_dims(h3_vals_3d, axis=0)

        num_times = h3_vals_3d.shape[0] # Explicitly target the temporal axis length integer
        print(f"Processing total timeline depth: {num_times} operational time levels.")

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

        print(f"Initializing parallel Pool worker array using {self.num_workers} CPU cores...")
        # Bind invariant geometry configurations using a partial function wrapper
        worker_func = functools.partial(
            _interpolate_single_time_step,
            h3_vals_3d=h3_vals_3d,
            points=points,
            lon_mesh=lon_mesh,
            lat_mesh=lat_mesh
        )

        # Allocate empty container matrix matching target spatial layout: (time, lat, lon)
        reconstructed_cube = np.zeros((num_times, len(target_lats_sorted), len(target_lons)), dtype=np.float32)

        # Map execution steps across the multiprocessing pool
        with Pool(processes=self.num_workers) as pool:
            for t_idx, recon_2d in pool.imap_unordered(worker_func, range(num_times)):
                reconstructed_cube[t_idx, :, :] = recon_2d
                if (t_idx + 1) % 10 == 0 or (t_idx + 1) == num_times:
                    print(f" -> Processed and compiled time levels: {t_idx + 1}/{num_times}")

        self._save_netcdf(reconstructed_cube, target_lats_sorted, target_lons)

    def _save_netcdf(self, matrix_3d: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> None:
        """
        Saves the structured 3D cube back into a temporal NetCDF file.
        """
        print("Assembling multi-temporal structured output container...")

        # Pull original metadata attributes from the template to preserve units
        template_attr = self.ds_template[self.var_name].attrs
        long_name = template_attr.get("long_name", self.var_name.upper())
        units = template_attr.get("units", "unknown")

        # Slice the matching time coordinates up to the generated rollout depth
        time_coords = self.ds_template['time'].values[:matrix_3d.shape[0]]

        ds_output = xr.Dataset(
            data_vars={
                self.var_name: (
                    ["time", "lat", "lon"],
                    matrix_3d,
                    {
                        "long_name": f"{long_name} (Reconstructed from H3 Timeline)",
                        "units": units,
                        "standard_name": self.ds_template[self.var_name].attrs.get("standard_name", self.var_name),
                        "dataset": "Reconstructed Multi-Temporal Unstructured Mesh"
                    }
                )
            },
            coords={
                "time": time_coords,
                "lat": lats,
                "lon": lons
            },
            attrs={
                "title": f"Multi-Temporal Reconstructed {self.var_name.upper()} Layout from Global H3 Grid",
                "Conventions": "CF-1.2",
                "total_time_steps": matrix_3d.shape[0]
            }
        )

        print(f"Writing full time-series grid file to disk: {self.output_path}...")
        ds_output.to_netcdf(self.output_path, format="NETCDF4")

        print("\n" + "="*50)
        print("MULTI-TEMPORAL REVERSE GRID INTERPOLATION SUCCESSFUL")
        print("="*50)
        print(f"Output Array Shape   : {matrix_3d.shape} (time x lat x lon)")
        print(f"File Saved To        : {self.output_path}")
        print(f"Target Variable Key  : ['{self.var_name}']")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-temporal object-oriented unstructured H3 to regular Lat/Lon reverse interpolator.")
    parser.add_argument("-i", "--input", required=True, help="Path to input unstructured H3 timeline file (e.g., global_h3_res2_air_all_times.nc)")
    parser.add_argument("-o", "--output", required=True, help="Path to save output reconstructed 3D structured NetCDF file")
    parser.add_argument("-t", "--template", required=True, help="Path to original lat/lon template file (e.g., air.sfc.2000.nc)")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Number of parallel CPU worker processes to spin up (default: max available)")
    args = parser.parse_args()

    processor = H3ToLatLonInterpolator(
        input_path=args.input, 
        output_path=args.output, 
        template_path=args.template,
        num_workers=args.workers
    )
    processor.load_datasets()
    processor.interpolate()

if __name__ == "__main__":
    main()

