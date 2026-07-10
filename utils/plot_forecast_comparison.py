import argparse
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

class ForecastVisualizer:
    def __init__(self, pred_path, true_path, output_img="forecast_comparison.png"):
        self.pred_path = pred_path
        self.true_path = true_path
        self.output_img = output_img

    def generate_panel_plot(self, time_index=0):
        print("Loading regular-grid NetCDF datasets...")
        ds_pred = xr.open_dataset(self.pred_path)
        ds_true = xr.open_dataset(self.true_path)

        # Standardize latitude coordinate naming if needed
        if 'lat' in ds_pred.coords and 'latitude' not in ds_pred.coords:
            ds_pred = ds_pred.rename({'lat': 'latitude', 'lon': 'longitude'})
        if 'lat' in ds_true.coords and 'latitude' not in ds_true.coords:
            ds_true = ds_true.rename({'lat': 'latitude', 'lon': 'longitude'})

        # Extract variable keys dynamically
        pred_var = 'air_forecast' if 'air_forecast' in ds_pred.data_vars else 'air'
        true_var = 'air' if 'air' in ds_true.data_vars else 'air_h3'

        # Fetch target timestamp from the forecast slice index
        target_time = ds_pred.time.values[time_index]
        print(f"Targeting evaluation timeline slice: {str(target_time)}")

        # Align slices and force latitude sorting from South to North (-90 to +90)
        try:
            pred_slice = ds_pred[pred_var].sortby('latitude').isel(time=time_index)
            true_slice = ds_true[true_var].sortby('latitude').sel(time=target_time)
        except KeyError:
            print("WARNING: Exact timestamp match failed. Falling back to simple structural index matching.")
            pred_slice = ds_pred[pred_var].sortby('latitude').isel(time=time_index)
            true_slice = ds_true[true_var].sortby('latitude').isel(time=time_index)

        # Extract the safely aligned, ascending coordinates
        lon = pred_slice['longitude'].values
        lat = pred_slice['latitude'].values

        # Compute exact spatial grid residuals (Forecast - Target Truth)
        difference = pred_slice.values - true_slice.values

        # Determine shared color bounds for absolute temperature comparisons (Kelvin)
        vmin_abs = min(float(pred_slice.min()), float(true_slice.min()))
        vmax_abs = max(float(pred_slice.max()), float(true_slice.max()))
        
        # Balance divergence threshold boundaries around 0 for the error maps
        vmax_diff = np.max(np.abs(difference))
        vmin_diff = -vmax_diff

        # Initialize Matplotlib Canvas (1 Row, 3 Columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
        
        # -----------------------------------------------------------------
        # PANEL 1: AUTOREGRESSIVE MODEL PREDICTION
        # -----------------------------------------------------------------
        im1 = axes[0].pcolormesh(lon, lat, pred_slice.values, cmap='turbo', vmin=vmin_abs, vmax=vmax_abs, shading='auto')
        axes[0].set_title(f"AI Autoregressive Forecast\n(Lead Step: {time_index + 1})", fontsize=12, fontweight='bold')
        fig.colorbar(im1, ax=axes[0], orientation='vertical', label='Temperature (K)', shrink=0.7)

        # -----------------------------------------------------------------
        # PANEL 2: REANALYSIS GROUND TRUTH RECORD (Flipped right side up)
        # -----------------------------------------------------------------
        im2 = axes[1].pcolormesh(lon, lat, true_slice.values, cmap='turbo', vmin=vmin_abs, vmax=vmax_abs, shading='auto')
        axes[1].set_title(f"Ground Truth Reanalysis\n({np.datetime_as_string(target_time, unit='h')})", fontsize=12, fontweight='bold')
        fig.colorbar(im2, ax=axes[1], orientation='vertical', label='Temperature (K)', shrink=0.7)

        # -----------------------------------------------------------------
        # PANEL 3: SPATIAL RESIDUAL ERROR MAP (Prediction - Truth)
        # -----------------------------------------------------------------
        im3 = axes[2].pcolormesh(lon, lat, difference, cmap='RdBu_r', vmin=vmin_diff, vmax=vmax_diff, shading='auto')
        axes[2].set_title("Forecast Error Residuals\n(Model - Ground Truth)", fontsize=12, fontweight='bold')
        fig.colorbar(im3, ax=axes[2], orientation='vertical', label='Error Delta (K)', shrink=0.7)

        # Global Canvas Labels
        for ax in axes:
            ax.set_xlabel("Longitude (°E)")
        axes[0].set_ylabel("Latitude (°N)")

        plt.suptitle(f"Global Surface Grid Comparison Matrix — Target Step Timeline: {str(target_time)[:16]}", 
                     fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(self.output_img, bbox_inches='tight', dpi=200)
        print(f"SUCCESS: Comparison canvas saved directly to: {self.output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an operational panel plot for model comparison metrics.")
    parser.add_argument("-p", "--pred", default="autoregressive_forecast.nc", help="Reconstructed regular grid forecast file")
    parser.add_argument("-t", "--true", default="../data/air.sfc.2000.nc", help="Baseline ground truth data file")
    parser.add_argument("-s", "--step", type=int, default=0, help="Timeline forecast lead slice index (default: 0 = +3 Hours)")
    parser.add_argument("-o", "--output", default="forecast_comparison_panel.png", help="Name of output visualization file")
    
    args = parser.parse_args()
    
    visualizer = ForecastVisualizer(pred_path=args.pred, true_path=args.true, output_img=args.output)
    visualizer.generate_panel_plot(time_index=args.step)
