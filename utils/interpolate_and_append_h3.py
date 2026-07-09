import argparse
import xarray as xr
import numpy as np
import os

class H3UnifiedPipeline:
    """
    Unified class module to interpolate a full 2D regular lat/lon weather time-series 
    onto an unstructured H3 mesh grid, sample static geographical properties 
    (land-sea mask and elevation data), and compile them into a stabilized master NetCDF.
    """
    def __init__(self, input_path: str, etopo_path: str, output_path: str, h3_template: str = "global_h3_res2_with_bounds.nc"):
        self.input_path = input_path
        self.etopo_path = etopo_path
        self.output_path = output_path
        self.h3_template = h3_template

        self.ds_h3 = None
        self.ds_src = None
        self.ds_etopo = None
        
        self.var_name = None
        self.has_etopo = False

    def load_base_structures(self) -> None:
        """Loads reference H3 grids and handles dynamic variable detection inside the source file."""
        print(f"Loading master H3 geometry template definitions from: {self.h3_template}")
        self.ds_h3 = xr.open_dataset(self.h3_template).load()

        print(f"Loading full time-series weather profile from: {self.input_path}")
        self.ds_src = xr.open_dataset(self.input_path, chunks={"time": 500})

        # Dynamic detector: Identify primary meteorological data key payload (skips coordinates)
        coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
        detected_vars = list(set(self.ds_src.data_vars.keys()) - coords_keys)
        if not detected_vars:
            raise KeyError("Could not detect a valid target weather variable inside the input file.")
        self.var_name = detected_vars[0]
        print(f" -> Automatically detected target weather data variable: '{self.var_name}'")

        # Ensure source latitudes are sorted in ascending order (-90 to 90)
        if self.ds_src.lat.values[0] > self.ds_src.lat.values[-1]:
            print("Inverting input latitude coordinates axis grid...")
            self.ds_src = self.ds_src.sortby('lat')

    def load_elevation_model(self) -> None:
        """Loads the external digital elevation model layout with error protection."""
        try:
            print(f"Loading global digital elevation model relief file: {self.etopo_path}")
            self.ds_etopo = xr.open_dataset(self.etopo_path).load()
            self.has_etopo = True
        except FileNotFoundError:
            print("Notice: External elevation model file not found. Generating basic topography fallback rules...")
            self.has_etopo = False

    def run_pipeline(self) -> None:
        """Executes full vectorized space-time lookup and geography feature attachment loops."""
        if self.ds_h3 is None or self.ds_src is None:
            raise ValueError("Base data structures not loaded. Please execute load_base_structures() first.")

        # Extract unstructured target coordinates
        h3_lons = self.ds_h3['longitude'].values
        h3_lats = self.ds_h3['latitude'].values
        h3_lon_converted = np.mod(h3_lons, 360) # Normalize to 0-360 range

        # =====================================================================
        # PART 1: APPLY TEMPORAL CIRCULAR LONGITUDE PADDING & SPATIAL INTERP
        # =====================================================================
        print("Applying circular longitude padding across all time steps...")
        raw_lons = self.ds_src.lon.values
        padded_lons = np.append(raw_lons, 360.0)
        raw_data_array = self.ds_src[self.var_name]

        # Duplicate the 0-degree data boundary column over to the 360-degree point
        padded_air_array = xr.concat([raw_data_array, raw_data_array.isel(lon=slice(0, 1))], dim="lon")
        padded_air_array = padded_air_array.assign_coords(lon=padded_lons)

        print(f"Executing vectorized interpolation across {len(self.ds_src.time)} time steps...")
        target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"], coords={"h3_index": self.ds_h3.h3_index})
        target_lat = xr.DataArray(h3_lats, dims=["h3_index"], coords={"h3_index": self.ds_h3.h3_index})

        interpolated_ds = padded_air_array.interp(
            lon=target_lon, lat=target_lat, method="linear", 
            kwargs={'bounds_error': False, 'fill_value': None}
        )

        # =====================================================================
        # PART 2: SAMPLE AND COMPILE STATIC GEOGRAPHY ATTRIBUTES
        # =====================================================================
        print("Deriving land-sea boundary mask layout configurations...")
        # Build raw grid mask from the weather data's first spatial slice frame
        air_slice = self.ds_src[self.var_name].isel(time=0).drop_vars('time', errors='ignore').load()
        raw_land_mask = xr.where(~np.isnan(air_slice), 1.0, 0.0)
        
        print("Mapping static geographic fields to H3 centroid node points...")
        node_land_mask = raw_land_mask.interp(lon=target_lon, lat=target_lat, method="nearest").values
        node_land_mask = np.where(node_land_mask > 0.5, 1.0, 0.0)

        if self.has_etopo:
            print("Interpolating topographical height maps using variable 'z'...")
            node_elevation = self.ds_etopo['z'].interp(lon=target_lon, lat=target_lat, method="linear").values
            node_elevation = np.nan_to_num(node_elevation, nan=0.0)
            self.ds_etopo.close()
        else:
            print(" -> Applying static rule baseline fallback matrix approximations...")
            node_elevation = np.where(node_land_mask == 1.0, 500.0, 0.0)

        # =====================================================================
        # PART 3: ASSEMBLE COMPLIANT COMPACT OUTPUT NETCDF DATASET
        # =====================================================================
        print("Structuring unified multi-dimensional output metadata container...")
        output_var_key = f"{self.var_name}_h3"

        ds_output = xr.Dataset(
            data_vars={
                output_var_key: (
                    ["time", "h3_index"],
                    interpolated_ds.values,
                    {
                        "units": self.ds_src[self.var_name].attrs.get("units", "unknown"),
                        "long_name": f"3-hourly {self.var_name.upper()} Interpolated to H3 Mesh",
                        "coordinates": "longitude latitude"
                    }
                ),
                "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
                "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
                "longitude_bounds": (["h3_index", "vertices"], self.ds_h3['longitude_bounds'].values, {"units": "degrees_east"}),
                "latitude_bounds": (["h3_index", "vertices"], self.ds_h3['latitude_bounds'].values, {"units": "degrees_north"}),
                "land_sea_mask": (["h3_index"], node_land_mask, {"long_name": "Land-Sea Binary Mask (1=Land, 0=Ocean)"}),
                "elevation": (["h3_index"], node_elevation, {"long_name": "Topographic Elevation Above Sea Level", "units": "meters"}),
            },
            coords={
                "time": self.ds_src.time.values,
                "h3_index": self.ds_h3.h3_index.values,
                "vertices": np.arange(6)
            },
            attrs={
                "title": f"Full Time-Series Global H3 Grid (Res 2) with Static Geography",
                "source_dataset": self.ds_src.attrs.get("title", "NOAA-CIRES Reanalysis Map"),
                "total_time_steps": len(self.ds_src.time)
            }
        )

        # =====================================================================
        # PART 4: ATOMIC DISK WRITE AND CLEANUP EXECUTIONS
        # =====================================================================
        staging_path = self.output_path + ".tmp_pipeline_stage"
        print(f"Writing integrated dataset file safely to staging path: {staging_path}...")
        ds_output.to_netcdf(staging_path, format="NETCDF4")
        ds_output.close()
        self.ds_src.close()
        self.ds_h3.close()

        print(f"Committing atomic file replacement to permanent target: {self.output_path}...")
        os.replace(staging_path, self.output_path)
        
        print("\n" + "="*50)
        print("UNIFIED INTERPOLATION & APPEND RUN SUCCESSFUL")
        print("="*50)
        print(f"Master Output File  : {self.output_path}")
        print(f"Data Matrix Shape   : {ds_output[output_var_key].shape} (time x h3_index)")
        print(f"Enrolled Static Keys: ['land_sea_mask', 'elevation']")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Vectorized Weather Interpolation & Static Geography Injection Engine.")
    parser.add_argument("-i", "--input", required=True, help="Path to your input regular regular lat-lon NetCDF file (e.g., air.sfc.1978.nc)")
    parser.add_argument("-e", "--etopo", required=True, help="Path to your NOAA ETOPO digital elevation relief NetCDF file")
    parser.add_argument("-o", "--output", required=True, help="Path destination where you want to write the completed output NetCDF dataset file")
    parser.add_argument("-t", "--template", default="global_h3_res2_with_bounds.nc", help="Base H3 definitions reference template file path")
    args = parser.parse_args()

    pipeline = H3UnifiedPipeline(
        input_path=args.input,
        etopo_path=args.etopo,
        output_path=args.output,
        h3_template=args.template
    )
    pipeline.load_base_structures()
    pipeline.load_elevation_model()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()

