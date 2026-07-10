import argparse
import os
import xarray as xr
import numpy as np

class LatLonToH3Interpolator:
    """
    An object-oriented multi-temporal spatial processing engine designed to reverse-interpolate
    historical regular lat/lon weather time-series records onto a 1D unstructured global H3
    hexagonal grid topology layout.
    """
    def __init__(self, input_weather_path: str, output_path: str, master_grid_path: str = "global_h3_res2_with_bounds.nc"):
        """
        Initializes the spatial interpolation engine.

        Args:
            input_weather_path (str): Path to raw lat-lon historical climate records (.nc or .zarr)
            output_path (str): Destination file path for the interpolated time-series data
            master_grid_path (str): Path to your pre-generated global H3 structural grid layout file
        """
        self.input_weather_path = input_weather_path
        self.output_path = output_path
        self.master_grid_path = master_grid_path

        self.ds_grid = None
        self.ds_src = None
        self.var_name = None

    def load_and_align_structures(self) -> None:
        """
        Loads the grid topology template definitions and opens the time-series 
        weather profiles using parallel chunk configurations.
        """
        print(f"[STAGE 1] Loading master H3 geometry definitions: {self.master_grid_path}")
        self.ds_grid = xr.open_dataset(self.master_grid_path).load()

        print(f"[STAGE 2] Opening historical climate dataset (Dask Enabled): {self.input_weather_path}")
        # Optimize chunk limits to fit comfortably inside multi-core workers
        if self.input_weather_path.endswith('.zarr') or os.path.isdir(self.input_weather_path):
            self.ds_src = xr.open_zarr(self.input_weather_path, consolidated=True)
        else:
            self.ds_src = xr.open_dataset(self.input_weather_path, chunks={"time": 500})

        # Dynamically discover the underlying data payload variable name
        coords_keys = {"time", "lat", "lon", "latitude", "longitude", "time_bnds", "nbnds"}
        data_vars = list(set(self.ds_src.data_vars.keys()) - coords_keys)
        if not data_vars:
            raise KeyError(f"Could not identify a valid weather variable in file: {self.input_weather_path}")
        
        self.var_name = data_vars[0]
        print(f" -> Automatically detected target data payload key: '{self.var_name}'")

        # Standardize latitude tracking direction to match ascending orientations
        lat_key = 'lat' if 'lat' in self.ds_src.coords else 'latitude'
        if self.ds_src[lat_key].values[0] > self.ds_src[lat_key].values[-1]:
            print(" -> Reorienting regular latitude grid axis to strict ascending layout (-90 to +90)...")
            self.ds_src = self.ds_src.sortby(lat_key)

    def apply_circular_padding(self) -> xr.DataArray:
        """
        Appends a 0-degree periodic longitudinal boundary column block to the end 
        of the array to eliminate prime-meridian stitching gaps.
        """
        print("[STAGE 3] Computing circular longitude wrapping tensors...")
        lon_key = 'lon' if 'lon' in self.ds_src.coords else 'longitude'
        raw_lons = self.ds_src[lon_key].values
        
        # Create an explicit 360-degree boundary array marker
        padded_lons = np.append(raw_lons, 360.0)
        raw_data_array = self.ds_src[self.var_name]

        # Extract the 0-degree index column and append it to the tail
        padded_array = xr.concat([raw_data_array, raw_data_array.isel({lon_key: slice(0, 1)})], dim=lon_key)
        padded_array = padded_array.assign_coords({lon_key: padded_lons})
        return padded_array

    def run_interpolation_pipeline(self) -> None:
        """
        Runs the full multi-scale vectorized spatial extraction matrix sequence.
        """
        self.load_and_align_structures()
        padded_air_array = self.apply_circular_padding()

        # Isolate spatial tracking parameters from master grid
        h3_lons = self.ds_grid['longitude'].values
        h3_lats = self.ds_grid['latitude'].values
        h3_lon_converted = np.mod(h3_lons, 360)

        print(f"[STAGE 4] Executing vectorized spatial search over {len(self.ds_src.time)} frames...")
        target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"], coords={"h3_index": self.ds_grid.h3_index})
        target_lat = xr.DataArray(h3_lats, dims=["h3_index"], coords={"h3_index": self.ds_grid.h3_index})

        lon_key = 'lon' if 'lon' in padded_air_array.coords else 'longitude'
        lat_key = 'lat' if 'lat' in padded_air_array.coords else 'latitude'

        # Perform the interpolation lazily to preserve chunk scaling properties
        interpolated_cube = padded_air_array.interp(
            {lon_key: target_lon, lat_key: target_lat},
            method="linear",
            kwargs={'bounds_error': False, 'fill_value': None}
        )

        print("[STAGE 5] Formatting CF-1.8 compliant output dataset buffers...")
        output_var_name = f"{self.var_name}_h3"

        # Explicitly preserve standard unit descriptions from master file layers
        lsm_attrs = self.ds_grid['land_sea_mask'].attrs if 'land_sea_mask' in self.ds_grid else {"long_name": "Land-Sea Binary Mask", "units": "fraction"}
        elv_attrs = self.ds_grid['elevation'].attrs if 'elevation' in self.ds_grid else {"long_name": "Topographic Elevation", "units": "meters"}

        ds_output = xr.Dataset(
            data_vars={
                output_var_name: (
                    ["time", "h3_index"],
                    interpolated_cube.data, # Safe lazy Dask array reference mapping block
                    {
                        "units": self.ds_src[self.var_name].attrs.get("units", "unknown"),
                        "long_name": f"3-hourly {self.var_name.upper()} Interpolated to H3 Mesh",
                        "coordinates": "longitude latitude"
                    }
                ),
                # FIX: Explicitly forward static geography variables directly from the master grid model mapping layout
                "land_sea_mask": (["h3_index"], self.ds_grid['land_sea_mask'].values, lsm_attrs),
                "elevation": (["h3_index"], self.ds_grid['elevation'].values, elv_attrs),

                "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
                "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
                "longitude_bounds": (["h3_index", "vertices"], self.ds_grid['longitude_bounds'].values, {"units": "degrees_east"}),
                "latitude_bounds": (["h3_index", "vertices"], self.ds_grid['latitude_bounds'].values, {"units": "degrees_north"}),
            },
            coords={
                "time": self.ds_src.time.values,
                "h3_index": self.ds_grid.h3_index.values,
                "vertices": np.arange(6)
            },
            attrs={
                "title": f"Full Time-Series Global H3 Grid Layout (Variable: {self.var_name.upper()})",
                "source_dataset": self.ds_src.attrs.get("title", "Historical Weather Archive"),
                "total_time_steps": len(self.ds_src.time)
            }
        )

        # Output branch check
        if self.output_path.endswith('.zarr') or os.path.isdir(self.output_path):
            print(f"Serializing output time-series directly to Zarr store: {self.output_path}")
            ds_output.to_zarr(self.output_path, mode="w", consolidated=True)
        else:
            print(f"Serializing output time-series directly to NetCDF file: {self.output_path}")
            ds_output.to_netcdf(self.output_path, format="NETCDF4")

        # Cleanup memory descriptors explicitly
        self.ds_grid.close()
        self.ds_src.close()
        print(f"SUCCESS: Completed interpolation loop. Shape: {ds_output[output_var_name].shape}\n")


# =====================================================================
# SCRIPT CONTROLLER ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Object-Oriented Vectorized Multi-Temporal Lat-Lon to Unstructured H3 Grid Interpolator."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input climate data time series file (.nc or .zarr)")
    parser.add_argument("-o", "--output", required=True, help="Destination path for the compiled H3 output file")
    parser.add_argument("-g", "--grid", default="global_h3_res2_with_bounds.nc", help="Path to master grid file with bounds")
    args = parser.parse_args()

    interpolator = LatLonToH3Interpolator(
        input_weather_path=args.input,
        output_path=args.output,
        master_grid_path=args.grid
    )
    interpolator.run_interpolation_pipeline()

if __name__ == "__main__":
    main()

