import os
import argparse
import geopandas as gpd
import h3
from shapely.geometry import Polygon

class GlobalH3GridGenerator:
    """
    An object-oriented spatial utility module designed to construct, correct,
    and serialize a global conformal H3 hexagonal grid at any specified resolution
    into standard GIS GeoJSON format.
    """
    def __init__(self, resolution: int = 2, output_dir: str = "."):
        """
        Initializes the global H3 grid generator.

        Args:
            resolution (int): Uber H3 target resolution level (default: 2, ~158km edge lengths).
            output_dir (str): Destination directory for the generated GeoJSON file (default: current dir).
        """
        self.resolution = resolution
        self.output_dir = output_dir
        self.global_cells = set()
        self.gdf = None

    def discover_global_cells(self) -> None:
        """
        Traverses the base H3 resolution-0 cells and populates a set containing
        every child cell on the globe at the target resolution.
        """
        print(f"[INIT] Discovering H3 base cells at target resolution: Res {self.resolution}")
        base_cells = h3.get_res0_cells()
        
        self.global_cells = set()
        for base_cell in base_cells:
            children = h3.cell_to_children(base_cell, self.resolution)
            self.global_cells.update(children)
            
        print(f" -> Found a total of {len(self.global_cells)} hexagons spanning the globe.")

    def compile_geometries(self) -> None:
        """
        Transforms the flat cell indices into Shapely Polygons, resolving 
        antimeridian boundary stretching.
        """
        if not self.global_cells:
            self.discover_global_cells()

        print("Compiling spatial geometries and correcting antimeridian boundaries...")
        grid_data = []
        
        for cell in self.global_cells:
            # Fetch boundary coordinates (latitude, longitude)
            boundary = h3.cell_to_boundary(cell)
            
            # Flip ordering to (longitude, latitude) for standard GIS compliance
            geometry_coords = [(lng, lat) for lat, lng in boundary]
            
            # Identify and resolve antimeridian (180° longitude) boundary crossings
            lons = [coords[0] for coords in geometry_coords]
            if max(lons) - min(lons) > 180:
                # Map negative longitudes across the 360-degree boundary to prevent stretch artifacts
                geometry_coords = [
                    (lng + 360 if lng < 0 else lng, lat) 
                    for lat, lng in geometry_coords
                ]
                
            shapely_poly = Polygon(geometry_coords)
            grid_data.append({
                "h3_index": cell,
                "geometry": shapely_poly
            })
            
        # Bind database to a GeoDataFrame using the WGS84 projection
        self.gdf = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

    def export_geojson(self, filename: str = None) -> None:
        """
        Exports the compiled global grid dataset to disk.

        Args:
            filename (str): Optional custom filename. Defaults to 'global_h3_res[resolution].geojson'.
        """
        if self.gdf is None:
            self.compile_geometries()

        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"global_h3_res{self.resolution}.geojson"
            
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"Writing global H3 database to disk: {output_path}...")
        self.gdf.to_file(output_path, driver="GeoJSON")
        print("SUCCESS: Global geographic H3 boundary file generated!")


# =====================================================================
# CLI SCRIPT INTERFACE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Object-Oriented Conformal Global H3 Grid GeoJSON Generator."
    )
    parser.add_argument("-r", "--resolution", type=int, default=2, help="Uber H3 grid resolution level (default: 2)")
    parser.add_argument("-o", "--output_dir", default=".", help="Target directory to save the output file (default: current dir)")
    parser.add_argument("-f", "--filename", default=None, help="Optional custom name for the output file")

    args = parser.parse_args()

    # Instantiate and run the generator class module
    generator = GlobalH3GridGenerator(
        resolution=args.resolution,
        output_dir=args.output_dir
    )
    generator.export_geojson(filename=args.filename)

if __name__ == "__main__":
    main()
