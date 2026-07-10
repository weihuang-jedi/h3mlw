import os
import argparse
import geopandas as gpd
import xarray as xr
import numpy as np

class H3GeoJSONToNetCDFConverter:
    """
    A class module to convert an unstructured H3 spatial grid from GeoJSON format
    into a CF-1.8 compliant structured NetCDF file containing precise coordinate boundary vertices.
    """
    def __init__(self, input_path: str, output_path: str, var_name: str = "surface_metric"):
        """
        Initializes the converter with file and variable paths.

        Args:
            input_path (str): Path to the H3 GeoJSON input file.
            output_path (str): Path where the output NetCDF file will be saved.
            var_name (str): The name of the dummy metric variable created inside the file (default: "surface_metric").
        """
        self.input_path = input_path
        self.output_path = output_path
        self.var_name = var_name

        self.gdf = None
        self.h3_indices = None
        self.lons = None
        self.lats = None
        self.lon_bounds = None
        self.lat_bounds = None
        self.num_hexagons = 0

    def load_and_parse_geojson(self) -> None:
        """
        Loads the GeoJSON file and extracts H3 indices, centroid coordinates, and bounds metadata.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"CRITICAL: The input file does not exist: {self.input_path}")

        print(f"[INIT] Reading GeoJSON data source: {self.input_path}...")
        self.gdf = gpd.read_file(self.input_path)
        self.num_hexagons = len(self.gdf)

        # 1. Extract raw H3 index arrays and compute centroids
        self.h3_indices = self.gdf["h3_index"].values
        centroids = self.gdf.geometry.centroid
        self.lons = centroids.x.values
        self.lats = centroids.y.values

        # 2. Preallocate geometry vertex arrays (H3 cells consistently have up to 6 unique vertices)
        self.lon_bounds = np.zeros((self.num_hexagons, 6))
        self.lat_bounds = np.zeros((self.num_hexagons, 6))

        # 3. Extract boundary coordinates from geometric shapes
        print(f" -> Extracting unique boundary vertices for {self.num_hexagons} hexagonal cells...")
        for i, geom in enumerate(self.gdf.geometry):
            coords = list(geom.exterior.coords)
            
            # H3 cells technically contain exactly 12 pentagons globally at any resolution.
            # To remain fully robust to pentagons (which only have 5 coords before repeating), 
            # we pad with the final coordinate to keep a consistent vertex shape of 6.
            num_coords = len(coords) - 1 # Subtract repeated boundary coordinate
            for vertex_idx in range(6):
                use_idx = min(vertex_idx, num_coords - 1)
                self.lon_bounds[i, vertex_idx] = coords[use_idx][0]
                self.lat_bounds[i, vertex_idx] = coords[use_idx][1]

    def build_and_export_dataset(self) -> None:
        """
        Constructs the CF-compliant xarray dataset and serializes it to a NetCDF file.
        """
        if self.gdf is None:
            self.load_and_parse_geojson()

        # Generate standard baseline mock metrics
        dummy_metric = np.random.uniform(10.0, 35.0, size=self.num_hexagons)

        print("Compiling CF-1.8 compliant metadata layouts...")
        dataset = xr.Dataset(
            data_vars={
                self.var_name: (
                    ["h3_index"],
                    dummy_metric,
                    {
                        "units": "degrees_C",
                        "long_name": "Sample Surface Temperature",
                        "coordinates": "longitude latitude"
                    }
                ),
                "longitude": (
                    ["h3_index"],
                    self.lons,
                    {
                        "units": "degrees_east",
                        "standard_name": "longitude",
                        "bounds": "longitude_bounds"
                    }
                ),
                "latitude": (
                    ["h3_index"],
                    self.lats,
                    {
                        "units": "degrees_north",
                        "standard_name": "latitude",
                        "bounds": "latitude_bounds"
                    }
                ),
                "longitude_bounds": (
                    ["h3_index", "vertices"],
                    self.lon_bounds,
                    {"units": "degrees_east"}
                ),
                "latitude_bounds": (
                    ["h3_index", "vertices"],
                    self.lat_bounds,
                    {"units": "degrees_north"}
                ),
            },
            coords={
                "h3_index": self.h3_indices,
                "vertices": np.arange(6)
            },
            attrs={
                "title": "Global H3 Hexagonal Grid with Geometry Bounds",
                "conventions": "CF-1.8"
            }
        )

        print(f"Writing dataset to NetCDF: {self.output_path}...")
        dataset.to_netcdf(self.output_path, format="NETCDF4")
        print(f"SUCCESS: Geometry database created successfully!")


# =====================================================================
# CLI SCRIPT INTERFACE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert an H3 Spatial GeoJSON file to a CF-1.8 boundary NetCDF."
    )
    parser.add_argument("-i", "--input", default="global_h3_res2.geojson", help="Path to input H3 GeoJSON (default: global_h3_res2.geojson)")
    parser.add_argument("-o", "--output", default="global_h3_res2_with_bounds.nc", help="Path to save output NetCDF (default: global_h3_res2_with_bounds.nc)")
    parser.add_argument("-v", "--var_name", default="surface_metric", help="Variable metric field name (default: surface_metric)")

    args = parser.parse_args()

    converter = H3GeoJSONToNetCDFConverter(
        input_path=args.input,
        output_path=args.output,
        var_name=args.var_name
    )
    converter.build_and_export_dataset()

if __name__ == "__main__":
    main()

