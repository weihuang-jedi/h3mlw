import argparse
import os
import xarray as xr
import numpy as np

class LatLonToIcosahedralInterpolator:
    """
    An object-oriented multi-temporal spatial processing engine designed to reverse-interpolate
    historical regular lat/lon weather time-series records onto a 1D unstructured global
    icosahedral grid topology layout (e.g., GraphCast Mk mesh structure).
    """
    def __init__(self, input_weather_path: str, output_path: str, master_mesh_path: str):
        """
        Args:
            input_weather_path (str): Path to raw lat-lon historical climate records (.nc or .zarr)
            output_path (str): Destination file path for the interpolated time-series data
            master_mesh_path (str): Path to your pre-generated global icosahedral UGRID master file
        """
        self.input_weather_path = input_weather_path
        self.output_path = output_path
        self.master_mesh_path = master_mesh_path

        self.ds_mesh = None
        self.ds_src = None
        self.var_name = None

    def load_and_align_structures(self) -> None:
        """Loads grid definitions and opens weather profiles with parallel chunk setups."""
        print(f"[STAGE 1] Loading master icosahedral mesh definitions: {self.master_mesh_path}")
        self.ds_mesh = xr.open_dataset(self.master_mesh_path).load()

        print(f"[STAGE 2] Opening historical climate dataset (Dask Enabled): {self.input_weather_path}")
        if self.input_weather_path.endswith('.zarr') or os.path.isdir(self.input_weather_path):
            self.ds_src = xr.open_zarr(self.input_weather_path, consolidated=True)
        else:
            self.ds_src = xr.open_dataset(self.input_weather_path, chunks={"time": 500})

        # Dynamically discover data variable payload name
        coords_keys = {"time", "lat", "lon", "latitude", "longitude", "time_bnds", "nbnds"}
        data_vars = list(set(self.ds_src.data_vars.keys()) - coords_keys)
        if not data_vars:
            raise KeyError(f"Could not identify a valid weather variable in: {self.input_weather_path}")
        
        self.var_name = data_vars[0]
        print(f" -> Automatically detected target data payload key: '{self.var_name}'")

        # Standardize latitude direction
        lat_key = 'lat' if 'lat' in self.ds_src.coords else 'latitude'
        if self.ds_src[lat_key].values[0] > self.ds_src[lat_key].values[-1]:
            print(" -> Reorienting regular latitude grid axis to strict ascending layout (-90 to +90)...")
            self.ds_src = self.ds_src.sortby(lat_key)

    def apply_circular_padding(self) -> xr.DataArray:
        """Appends a 360-degree boundary column to eliminate prime-meridian interpolation seams."""
        print("[STAGE 3] Computing circular longitude wrapping tensors...")
        lon_key = 'lon' if 'lon' in self.ds_src.coords else 'longitude'
        raw_lons = self.ds_src[lon_key].values
        
        padded_lons = np.append(raw_lons, 360.0)
        raw_data_array = self.ds_src[self.var_name]

        padded_array = xr.concat([raw_data_array, raw_data_array.isel({lon_key: slice(0, 1)})], dim=lon_key)
        padded_array = padded_array.assign_coords({lon_key: padded_lons})
        return padded_array

    def run_interpolation_pipeline(self) -> None:
        """Executes vectorized linear interpolation over the icosahedral grid vertices."""
        self.load_and_align_structures()
        padded_weather_array = self.apply_circular_padding()

        # Isolate spatial tracking coordinates from icosahedral mesh vertices
        mesh_lons = self.ds_mesh['longitude'].values
        mesh_lats = self.ds_mesh['latitude'].values
        mesh_lon_converted = np.mod(mesh_lons, 360)

        print(f"[STAGE 4] Executing vectorized spatial interpolation over {len(self.ds_src.time)} frames...")
        target_lon = xr.DataArray(mesh_lon_converted, dims=["node"], coords={"node": self.ds_mesh.node})
        target_lat = xr.DataArray(mesh_lats, dims=["node"], coords={"node": self.ds_mesh.node})

        lon_key = 'lon' if 'lon' in padded_weather_array.coords else 'longitude'
        lat_key = 'lat' if 'lat' in padded_weather_array.coords else 'latitude'

        # Lazy execution pipeline preservations via Dask
        interpolated_cube = padded_weather_array.interp(
            {lon_key: target_lon, lat_key: target_lat},
            method="linear",
            kwargs={'bounds_error': False, 'fill_value': None}
        )

        print("[STAGE 5] Formatting CF-1.8/UGRID compliant output dataset buffers...")
        output_var_name = f"{self.var_name}_icosahedral"

        # Safe metadata extraction
        lsm_attrs = self.ds_mesh['land_sea_mask'].attrs if 'land_sea_mask' in self.ds_mesh else {"long_name": "Land-Sea Mask", "units": "fraction"}
        elv_attrs = self.ds_mesh['elevation'].attrs if 'elevation' in self.ds_mesh else {"long_name": "Elevation", "units": "meters"}

        ds_output = xr.Dataset(
            data_vars={
                output_var_name: (
                    ["time", "node"],
                    interpolated_cube.data,  # .data passes the lazy Dask array reference safely
                    {
                        "units": self.ds_src[self.var_name].attrs.get("units", "unknown"),
                        "long_name": f"3-hourly {self.var_name.upper()} Interpolated to Icosahedral Mesh",
                        "coordinates": "longitude latitude",
                        "mesh": "icosahedral_mesh"
                    }
                ),
                # Forward physical static parameters mapped previously
                "land_sea_mask": (["node"], self.ds_mesh['land_sea_mask'].values, lsm_attrs),
                "elevation": (["node"], self.ds_mesh['elevation'].values, elv_attrs),
                
                # UGRID Structural parameters
                "face_nodes": (["face", "three"], self.ds_mesh['face_nodes'].values, self.ds_mesh['face_nodes'].attrs),
                "longitude": (["node"], mesh_lons, {"units": "degrees_east", "standard_name": "longitude"}),
                "latitude": (["node"], mesh_lats, {"units": "degrees_north", "standard_name": "latitude"}),
                "x_cartesian": (["node"], self.ds_mesh['x_cartesian'].values, {"units": "m"}),
                "y_cartesian": (["node"], self.ds_mesh['y_cartesian'].values, {"units": "m"}),
                "z_cartesian": (["node"], self.ds_mesh['z_cartesian'].values, {"units": "m"}),
                
                "icosahedral_mesh": ([], 0, self.ds_mesh['icosahedral_mesh'].attrs)
            },
            coords={
                "time": self.ds_src.time.values,
                "node": self.ds_mesh.node.values,
                "face": self.ds_mesh.face.values,
                "three": np.arange(3)
            },
            attrs={
                "title": f"Time-Series Global Icosahedral Grid Layout (Variable: {self.var_name.upper()})",
                "source_dataset": self.ds_src.attrs.get("title", "Weather Reanalysis Archive"),
                "total_time_steps": len(self.ds_src.time),
                "conventions": "CF-1.8 UGRID-1.0"
            }
        )

        if self.output_path.endswith('.zarr') or os.path.isdir(self.output_path):
            print(f"Serializing output dataset to Zarr store: {self.output_path}")
            ds_output.to_zarr(self.output_path, mode="w", consolidated=True)
        else:
            print(f"Serializing output dataset to NetCDF: {self.output_path}")
            ds_output.to_netcdf(self.output_path, format="NETCDF4")

        self.ds_mesh.close()
        self.ds_src.close()
        print(f"SUCCESS: Completed interpolation pipeline. Array shape: {ds_output[output_var_name].shape}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Temporal Lat-Lon to Unstructured Icosahedral Mesh Interpolator.")
    parser.add_argument("-i", "--input", required=True, help="Path to input climate data time series file (.nc or .zarr)")
    parser.add_argument("-m", "--mesh", required=True, help="Path to master icosahedral mesh file containing static geography")
    parser.add_argument("-o", "--output", required=True, help="Destination path for compiled output file")
    args = parser.parse_args()

    interpolator = LatLonToIcosahedralInterpolator(
        input_weather_path=args.input,
        master_mesh_path=args.mesh,
        output_path=args.output
    )
    interpolator.run_interpolation_pipeline()

if __name__ == "__main__":
    main()
