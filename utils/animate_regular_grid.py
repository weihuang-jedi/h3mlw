import argparse
import xarray as xr
import matplotlib
# Headless backend for cluster nodes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import os
from PIL import Image

class RegularGridAnimator:
    """
    A class to animate standard 2D regular (time, lat, lon) netCDF grids
    by saving static frames and compiling them into a GIF.
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.tmp_dir = "tmp_regular_frames"
        
        self.ds = None
        self.var_name = None
        self.vmin = 0.0
        self.vmax = 0.0

    def initialize(self):
        print(f"Reading structured 2D grid from: {self.input_path}")
        self.ds = xr.open_dataset(self.input_path)
        
        # Detect core data variable (skips coordinates)
        coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
        var_names = list(set(self.ds.data_vars.keys()) - coords_keys)
        
        if not var_names:
            raise KeyError("Could not detect any valid data variables in the NetCDF container.")
            
        # FIX: Extract the raw string element directly from the list wrapper
        self.var_name = var_names[0] if isinstance(var_names, list) else var_names
        print(f" -> Successfully isolated target variable key: '{self.var_name}'")
        
        # Determine global thresholds safely from the DataArray for color scale stability
        self.vmin = float(self.ds[self.var_name].min())
        self.vmax = float(self.ds[self.var_name].max())
        
        os.makedirs(self.tmp_dir, exist_ok=True)

    def render_frames(self):
        num_frames = len(self.ds.time)
        long_name = self.ds[self.var_name].attrs.get("long_name", self.var_name.upper())
        units = self.ds[self.var_name].attrs.get("units", "")
        
        print(f"Rendering {num_frames} standard regular 2D raster frames...")
        for t_idx in range(num_frames):
            fig = plt.figure(figsize=(12, 7))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(color='black', linewidth=1)
            ax.gridlines(draw_labels=True, linestyle='--')
            ax.set_global()
            
            # Fast native 2D quadmesh raster plot
            self.ds[self.var_name].isel(time=t_idx).plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap='plasma',
                vmin=self.vmin,
                vmax=self.vmax,
                add_colorbar=False # Managed manually below
            )
            
            # Add consistent colorbar
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7)
            cbar.set_label(f"{long_name} ({units})")
            
            lead_hours = (t_idx + 1) * 3
            ax.set_title(f"Model Forecast Evolution (Regular Grid) | Lead Time: +{lead_hours} Hours", fontsize=14)
            
            frame_filename = os.path.join(self.tmp_dir, f"frame_{t_idx:03d}.png")
            plt.savefig(frame_filename, bbox_inches='tight', dpi=120)
            plt.close(fig)
            
            if (t_idx + 1) % 10 == 0 or (t_idx + 1) == num_frames:
                print(f" -> Compiled frame {t_idx + 1}/{num_frames}")

    def compile_gif(self):
        print("Stitching regular raster frames into a seamless rollout GIF...")
        frame_files = sorted(glob.glob(os.path.join(self.tmp_dir, "frame_*.png")))
        if not frame_files:
            raise FileNotFoundError("Could not locate any pre-rendered frame files to compile.")
            
        images = [Image.open(f) for f in frame_files]
        
        images[0].save(
            self.output_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0
        )
        
        # Cleanup
        for f in frame_files:
            os.remove(f)
        os.rmdir(self.tmp_dir)
        print(f"\nSUCCESS: Unified raster animation saved to: {self.output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Animate a reconstructed regular lat/lon weather grid.")
    parser.add_argument("-i", "--input", required=True, help="Path to your reconstructed regular NetCDF file (e.g., autoregressive_forecast.nc)")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output GIF animation")
    args = parser.parse_args()
    
    animator = RegularGridAnimator(input_path=args.input, output_path=args.output)
    animator.initialize()
    animator.render_frames()
    animator.compile_gif()

if __name__ == "__main__":
    main()

