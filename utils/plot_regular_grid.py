import argparse
import xarray as xr
import yaml
import numpy as np
import os

# =====================================================================
# A. SYSTEM BACKEND MANAGED INITIALIZATION
# =====================================================================
# We parse the '-s' flag early *before* loading matplotlib to prevent
# backend locking conflicts across different cluster login node types.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-s", "--show", action="store_true")
early_args, _ = parser.parse_known_args()

import matplotlib
if not early_args.show:
    # Quietly enable a non-interactive backend *only* if the user does not want to see the plot window
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# =====================================================================
# B. OBJECT-ORIENTED PLOTTER CLASS STYLE
# =====================================================================
class RegularGridPlotter:
    def __init__(self, input_path, output_path=None, show=False):
        self.input_path = input_path
        self.output_path = output_path
        self.show = show  
        self.dataset = None
        self.var_name = None
        self.var_slice = None

    def load_data(self):
        print(f"Reading regular grid data from: {self.input_path}")
        self.dataset = xr.open_dataset(self.input_path)

        coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
        var_names = list(set(self.dataset.data_vars.keys()) - coords_keys)
        
        if not var_names:
            raise KeyError("Could not detect a valid data variable in the NetCDF file.")
            
        # Target the first payload element
        self.var_name = var_names[0]
        print(f" -> Automatically detected target variable: '{self.var_name}'")
        self.var_slice = self.dataset[self.var_name].isel(time=0)

    def render_map(self):
        if self.var_slice is None:
            raise ValueError("Data not loaded. Please call load_data() before rendering.")

        long_name = self.var_slice.attrs.get("long_name", f"Surface {self.var_name.upper()}")
        units = self.var_slice.attrs.get("units", "unknown")
        time_str = str(self.var_slice.time.values)[:13]

        # FIX 1: Establish default output path matching variable parameters securely
        final_output = self.output_path if self.output_path else f"original_{self.var_name}_map.png"

        print("Initializing map projection and canvas layers...")
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True, linestyle='--')

        print(f"Rendering 2D raster field to {final_output}...")
        self.var_slice.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='plasma',
            cbar_kwargs={'label': f"{long_name} ({units})"}
        )

        plt.title(f"{long_name} - Time: {time_str}")
        plt.savefig(final_output, bbox_inches='tight', dpi=150)
        
        # FIX 2: Only call plt.show() if the user explicitly requested it and an interactive backend is available
        if self.show:
            print("Displaying interactive map window overlay...")
            plt.show()
            
        plt.close(fig)
        # FIX 3: Print the correct variable string 'final_output' rather than self.output_path
        print(f"SUCCESS: Visualization map cleanly generated at: {final_output}\n")

# =====================================================================
# C. ENTRY PARSING WORKFLOW
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Object-oriented regular weather grid visualization tool.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input regular lat-lon NetCDF file.")
    parser.add_argument("-o", "--output", default=None, help="Path to save the output map image.")
    parser.add_argument("-s", "--show", action="store_true", help="Display the plot interactively via desktop window popup.")
    args = parser.parse_args()

    plotter = RegularGridPlotter(input_path=args.input, output_path=args.output, show=args.show)
    plotter.load_data()
    plotter.render_map()

if __name__ == "__main__":
    main()

