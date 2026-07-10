import argparse
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def generate_comparison_panel(orig_path, recon_path, output_img_path, time_idx=0):
    print(f"[STAGE 1] Loading spatial datasets...")
    ds_orig = xr.open_dataset(orig_path)
    ds_recon = xr.open_dataset(recon_path)

    # 1. Automatically detect primary weather variable keys
    ignore_keys = {"time", "lat", "lon", "latitude", "longitude"}
    var_orig = list(set(ds_orig.data_vars.keys()) - ignore_keys)[0]
    var_recon = list(set(ds_recon.data_vars.keys()) - ignore_keys)[0]

    print(f" -> Original variable variable: '{var_orig}'")
    print(f" -> Reconstructed variable: '{var_recon}'")

    # 2. Extract a slice at the requested time step index
    da_orig = ds_orig[var_orig].isel(time=time_idx)
    da_recon = ds_recon[var_recon].isel(time=time_idx)
    time_str = str(da_orig.time.values)[:16]

    # 3. Standardize lat/lon dimension names to ensure clean matching
    lat_key_orig = 'lat' if 'lat' in da_orig.coords else 'latitude'
    lon_key_orig = 'lon' if 'lon' in da_orig.coords else 'longitude'
    lat_key_recon = 'lat' if 'lat' in da_recon.coords else 'latitude'
    lon_key_recon = 'lon' if 'lon' in da_recon.coords else 'longitude'

    # 4. Interpolate the original grid onto the reconstructed grid to compute a direct cell-by-cell difference
    print("[STAGE 2] Aligning spatial grid arrays...")
    da_orig_aligned = da_orig.interp(
        {lat_key_orig: ds_recon[lat_key_recon], lon_key_orig: ds_recon[lon_key_recon]}, 
        method="linear"
    )

    # Calculate absolute difference residuals
    da_diff = da_recon - da_orig_aligned

    # 5. Initialize the 3-panel horizontal plot configuration
    print("[STAGE 3] Rendering geographic panel arrays...")
    fig, axes = plt.subplots(
        1, 3, 
        figsize=(22, 5.5), 
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}
    )

    # Determine dynamic colorbar scaling for the raw weather data fields
    v_min = min(float(da_orig.min()), float(da_recon.min()))
    v_max = max(float(da_orig.max()), float(da_recon.max()))
    units = da_orig.attrs.get("units", "unknown")

    # Panel A: Original Baseline Data
    ax_a = axes[0]
    ax_a.coastlines(resolution='110m', color='black', linewidth=0.8)
    mesh_a = ax_a.pcolormesh(
        ds_recon[lon_key_recon], ds_recon[lat_key_recon], da_orig_aligned,
        cmap='turbo', vmin=v_min, vmax=v_max, transform=ccrs.PlateCarree()
    )
    ax_a.set_title(f"A) Original Grid Field ({var_orig.upper()})", fontsize=12, fontweight='bold')
    fig.colorbar(mesh_a, ax=ax_a, orientation='horizontal', pad=0.06, shrink=0.85, label=f"Units: {units}")

    # Panel B: Reconstructed Field from your Icosahedral Mesh
    ax_b = axes[1]
    ax_b.coastlines(resolution='110m', color='black', linewidth=0.8)
    mesh_b = ax_b.pcolormesh(
        ds_recon[lon_key_recon], ds_recon[lat_key_recon], da_recon,
        cmap='turbo', vmin=v_min, vmax=v_max, transform=ccrs.PlateCarree()
    )
    ax_b.set_title(f"B) Reconstructed Regular Grid", fontsize=12, fontweight='bold')
    fig.colorbar(mesh_b, ax=ax_b, orientation='horizontal', pad=0.06, shrink=0.85, label=f"Units: {units}")

    # Panel C: Spatial Residual Bias Error (Reconstructed minus Original)
    ax_c = axes[2]
    ax_c.coastlines(resolution='110m', color='black', linewidth=0.8)
    
    # Use a divergent colormap centered at zero for the difference panel
    max_bias = max(abs(float(da_diff.min())), abs(float(da_diff.max())))
    if max_bias == 0: max_bias = 1.0 # Bypasses edge-case division errors on empty files
    
    mesh_c = ax_c.pcolormesh(
        ds_recon[lon_key_recon], ds_recon[lat_key_recon], da_diff,
        cmap='bwr', vmin=-max_bias, vmax=max_bias, transform=ccrs.PlateCarree()
    )
    ax_c.set_title(f"C) Residual Bias Mapping (B - A)", fontsize=12, fontweight='bold')
    fig.colorbar(mesh_c, ax=ax_c, orientation='horizontal', pad=0.06, shrink=0.85, label=f"Delta Bias Value ({units})")

    # Add overarching figure annotations
    plt.suptitle(f"Global Multi-Scale Mesh Verification Diagnostic Analysis\nTime Step Sequence Target: {time_str} UTC", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    print(f"[SAVE] Exporting comparison diagnostic image asset to: {output_img_path}")
    plt.savefig(output_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Close datasets safely
    ds_orig.close()
    ds_recon.close()
    print("SUCCESS: Metric check panel compiled successfully!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a 3-panel comparison map checking interpolation errors.")
    parser.add_argument("-o", "--original", required=True, help="Path to original regular grid netCDF file")
    parser.add_argument("-r", "--reconstructed", required=True, help="Path to the output file from your icosahedral converter")
    parser.add_argument("-i", "--image", default="grid_comparison_panel.png", help="Output PNG image destination")
    parser.add_argument("-t", "--time_index", type=int, default=0, help="Timeline index step sequence location to plot (default: 0)")
    
    args = parser.parse_args()
    generate_comparison_panel(args.original, args.reconstructed, args.image, args.time_index)
