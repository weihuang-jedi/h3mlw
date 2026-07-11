import argparse
import os
import numpy as np
import xarray as xr

class RegularGridWindowExtractor:
    """
    Slices and extracts a continuous 2-step history conditioning window 
    from a standard 2D regular lat/lon netCDF file based on a target start time.
    """
    def __init__(self, input_path: str, output_path: str, start_time_str: str):
        self.input_path = input_path
        self.output_path = output_path
        self.start_time_str = start_time_str

    def execute_extraction(self) -> None:
        print(f"[LOAD] Opening regular grid netCDF source file: {self.input_path}")
        # xarray automatically handles decoding the "hours since 1800-01-01" baseline
        ds = xr.open_dataset(self.input_path)

        print(f"[DATETIME] Searching for target timestamp signature: '{self.start_time_str}'")
        try:
            target_dt = np.datetime64(self.start_time_str)
        except ValueError:
            raise ValueError(f"Could not parse '{self.start_time_str}'. Use standard ISO layout: YYYY-MM-DDTHH:MM:SS")

        time_values = ds.time.values
        matching_indices = np.where(time_values == target_dt)[0]

        if len(matching_indices) == 0:
            min_time = str(time_values[0])[:19]
            max_time = str(time_values[-1])[:19]
            raise IndexError(
                f"\n[CRITICAL] The requested start time '{self.start_time_str}' was not found in this file.\n"
                f" -> Valid range for this specific netCDF dataset: [{min_time}] to [{max_time}].\n"
            )

        start_idx = int(matching_indices[0])
        
        # Guard check to ensure a second consecutive time step is available
        if start_idx + 1 >= len(time_values):
            raise IndexError("[CRITICAL] Not enough timesteps remaining after the target start time to extract a 2-step window.")

        print(f" -> Found matching array indices: Step 1 = index {start_idx} | Step 2 = index {start_idx + 1}")
        print(f" -> Step 1 Time: {str(time_values[start_idx])[:16]} UTC")
        print(f" -> Step 2 Time: {str(time_values[start_idx + 1])[:16]} UTC")

        # Slice the 2 continuous timesteps across the 'time' dimension
        print("[SLICE] Extracting the 2-step regular grid history window...")
        ds_window = ds.isel(time=slice(start_idx, start_idx + 2))

        # Explicitly preserve variable definitions and structure metadata
        ds_window.attrs["title"] = f"Regular Grid History Conditioning Window (Initialized {self.start_time_str})"
        ds_window.attrs["history_source_file"] = os.path.basename(self.input_path)

        print(f"[SAVE] Exporting regular grid initialization file to: {self.output_path}")
        ds_window.to_netcdf(self.output_path, format="NETCDF4")
        
        ds.close()
        ds_window.close()
        print("SUCCESS: Regular grid initial condition file created successfully!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a 2-step history initialization window from a regular grid NetCDF file.")
    parser.add_argument("-i", "--input", required=True, help="Input regular grid file path (e.g., air.sfc.2000.nc)")
    parser.add_argument("-o", "--output", required=True, help="Output regular grid file path (e.g., init_regular_2000.nc)")
    parser.add_argument("-t", "--start_time", required=True, help="ISO format start time (e.g., 2000-07-01T00:00:00)")
    
    args = parser.parse_args()
    
    extractor = RegularGridWindowExtractor(
        input_path=args.input,
        output_path=args.output,
        start_time_str=args.start_time
    )
    extractor.execute_extraction()

