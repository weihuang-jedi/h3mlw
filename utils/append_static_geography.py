import argparse
import xarray as xr
import numpy as np
import os

class H3StaticGeographyAppender:
    """
    A class module to sample, interpolate, and append fixed geographic features 
    (land-sea masks and digital elevation data) onto an unstructured global H3 NetCDF file.
    """
    def __init__(self, h3_input_path: str, weather_mask_path: str, etopo_path: str, output_path: str):
        """
        Initializes the appender with required input and output file configurations.
        """
        self.h3_input_path = h3_input_path
        self.weather_mask_path = weather_mask_path
        self.etopo_path = etopo_path
        self.output_path = output_path
        
        self.ds_h3 = None
        self.raw_land_mask = None
        self.ds_etopo = None
        self.has_etopo = False

    def load_base_data(self) -> None:
        """
        Loads the master H3 file and the reference weather file to build the initial land-sea mask.
        """
        print(f"Opening master H3 grid timeline structure: {self.h3_input_path}")
        # Force high-priority memory loading
        self.ds_h3 = xr.open_dataset(self.h3_input_path).load()

        print(f"Loading reference weather file for masking layers: {self.weather_mask_path}")
        with xr.open_dataset(self.weather_mask_path) as ds_air:
            air_slice = ds_air['air'].isel(time=0).drop_vars('time', errors='ignore').load()
        
        # Securely check only the first element to catch descending order grids
        if air_slice.lat.values[0] > air_slice.lat.values[-1]:
            print(" -> Reorienting descending reference grid coordinates...")
            air_slice = air_slice.sortby('lat')
            
        # Classify cells into binary mask indicators
        self.raw_land_mask = xr.where(~np.isnan(air_slice), 1.0, 0.0)

    def load_elevation_model(self) -> None:
        """
        Attempts to load the ETOPO digital elevation model file. Sets a fallback flag if missing.
        """
        try:
            print(f"Loading global digital elevation model relief file: {self.etopo_path}")
            self.ds_etopo = xr.open_dataset(self.etopo_path).load()
            self.has_etopo = True
        except FileNotFoundError:
            print("Notice: External elevation model file not found. Generating basic topography fallback rules...")
            self.has_etopo = False

    def process_and_append(self) -> None:
        """
        Interpolates spatial properties over the H3 grid points and writes the output NetCDF safely.
        """
        # FIX: Removed the typo 'refinement' that broke Python syntax
        if self.ds_h3 is None or self.raw_land_mask is None:
            raise ValueError("Base datasets not loaded. Please call load_base_data() first.")

        # Extract unstructured coordinate parameters from the destination H3 grid
        h3_lats = self.ds_h3['latitude'].values
        h3_lons = self.ds_h3['longitude'].values
        h3_lon_converted = np.mod(h3_lons, 360) 

        # Package destination points into xarray lookup components
        target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
        target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

        print("Mapping static geographic fields to H3 centroid node points...")
        node_land_mask = self.raw_land_mask.interp(lon=target_lon, lat=target_lat, method="nearest").values
        node_land_mask = np.where(node_land_mask > 0.5, 1.0, 0.0)

        if self.has_etopo:
            print("Interpolating topographical height maps using variable 'z'...")
            node_elevation = self.ds_etopo['z'].interp(lon=target_lon, lat=target_lat, method="linear").values
            node_elevation = np.nan_to_num(node_elevation, nan=0.0)
        else:
            print(" -> Applying static rule baseline fallback matrix approximations...")
            node_elevation = np.where(node_land_mask == 1.0, 500.0, 0.0)

        print("Structuring compliant static metadata payload dimensions...")
        self.ds_h3['land_sea_mask'] = (["h3_index"], node_land_mask, {"long_name": "Land-Sea Binary Mask (1=Land, 0=Ocean)"})
        self.ds_h3['elevation'] = (["h3_index"], node_elevation, {"long_name": "Topographic Elevation Above Sea Level", "units": "meters"})

        # Close references if they exist to unlock files
        if self.ds_etopo is not None:
            self.ds_etopo.close()

        # Atomic staging overwrite implementation
        staging_path = self.output_path + ".tmp_staging"
        print(f"Writing updated dataset safely to staging space: {staging_path}...")
        self.ds_h3.to_netcdf(staging_path, format="NETCDF4")
        self.ds_h3.close()

        print(f"Committing atomic overwrite operation to final target: {self.output_path}...")
        os.replace(staging_path, self.output_path)
        print(f"SUCCESS: Static geography layers appended directly to: {self.output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Object-oriented static geography appender tool for unstructured H3 grids.")
    parser.add_argument("-i", "--input", required=True, help="Path to your master base time-series H3 NetCDF file.")
    parser.add_argument("-m", "--mask", required=True, help="Path to your regular lat-lon weather reference file.")
    parser.add_argument("-e", "--etopo", required=True, help="Path to your NOAA ETOPO elevation relief NetCDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path where you want to write the completed output NetCDF file.")
    args = parser.parse_args()

    appender = H3StaticGeographyAppender(
        h3_input_path=args.input,
        weather_mask_path=args.mask,
        etopo_path=args.etopo,
        output_path=args.output
    )
    appender.load_base_data()
    appender.load_elevation_model()
    appender.process_and_append()

if __name__ == "__main__":
    main()

