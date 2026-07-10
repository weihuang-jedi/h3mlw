import argparse
import xarray as xr
import numpy as np
import os

class MasterGridStaticAppender:
    """
    Utility module to derive and append static geographic parameters 
    (elevation height and land-sea mask) directly into the master H3 geometry file
    using only the ETOPO digital elevation model.
    """
    def __init__(self, grid_path: str, etopo_path: str):
        self.grid_path = grid_path
        self.etopo_path = etopo_path

        self.ds_grid = None
        self.ds_etopo = None

    def load_source_layers(self) -> None:
        print(f"[STAGE 1] Loading master H3 geometry file: {self.grid_path}")
        self.ds_grid = xr.open_dataset(self.grid_path).load()

        print(f"[STAGE 2] Loading ETOPO digital elevation model: {self.etopo_path}")
        # ETOPO files can be quite large, we load it dynamically
        self.ds_etopo = xr.open_dataset(self.etopo_path)

    def process_and_append(self) -> None:
        if self.ds_grid is None or self.ds_etopo is None:
            raise ValueError("Datasets not loaded. Please execute load_source_layers() first.")

        # Extract mesh coordinate points from your H3 grid
        h3_lats = self.ds_grid['latitude'].values
        h3_lons = self.ds_grid['longitude'].values
        h3_lon_converted = np.mod(h3_lons, 360)

        target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
        target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

        # Detect the coordinate keys inside ETOPO (usually 'lon'/'lat' or 'longitude'/'latitude')
        etopo_lon_key = 'lon' if 'lon' in self.ds_etopo.coords else 'longitude'
        etopo_lat_key = 'lat' if 'lat' in self.ds_etopo.coords else 'latitude'
        
        # ETOPO2022 standard height variable is typically 'z' or 'elevation'
        elv_var = 'z' if 'z' in self.ds_etopo.data_vars else 'elevation'

        print(f"[PROCESSING] Interpolating topographic height using ETOPO variable '{elv_var}'...")
        # We use linear interpolation to get a smooth altitude approximation at the hex center
        node_elevation = self.ds_etopo[elv_var].interp(
            {etopo_lon_key: target_lon, etopo_lat_key: target_lat}, 
            method="linear"
        ).values
        node_elevation = np.nan_to_num(node_elevation, nan=0.0)

        # MATHEMATICAL DERIVATION: 
        # Any point on the earth's crust >= 0 meters above sea level is classified as land (1.0).
        # Any point < 0 meters is classified as water/ocean (0.0).
        print("[PROCESSING] Deriving binary land-sea mask from physical elevation contours...")
        node_land_mask = np.where(node_elevation >= 0.0, 1.0, 0.0)

        # Append variables directly into the grid dataset structures
        self.ds_grid['land_sea_mask'] = (["h3_index"], node_land_mask.astype(np.float32), 
                                         {"long_name": "Land-Sea Binary Mask (1=Land, 0=Ocean)", "units": "fraction"})
        self.ds_grid['elevation'] = (["h3_index"], node_elevation.astype(np.float32), 
                                     {"long_name": "Topographic Elevation Above Sea Level", "units": "meters"})

        self.ds_etopo.close()

        # Atomic overwrite operation
        staging_path = self.grid_path + ".tmp_staging"
        print(f"[SAVE] Serializing updated blueprint to staging file: {staging_path}")
        self.ds_grid.to_netcdf(staging_path, format="NETCDF4")
        self.ds_grid.close()

        os.replace(staging_path, self.grid_path)
        print(f"SUCCESS: Master grid file '{self.grid_path}' now permanently contains land_sea_mask and elevation!\n")


def main():
    parser = argparse.ArgumentParser(description="Append land-sea mask and topography to the core H3 grid file using ETOPO.")
    parser.add_argument("-g", "--grid", required=True, help="Path to your master global_h3_res2_with_bounds.nc file")
    parser.add_argument("-e", "--etopo", required=True, help="Path to NOAA ETOPO relief file")
    args = parser.parse_args()

    appender = MasterGridStaticAppender(grid_path=args.grid, etopo_path=args.etopo)
    appender.load_source_layers()
    appender.process_and_append()

if __name__ == "__main__":
    main()
