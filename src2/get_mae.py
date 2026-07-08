import xarray as xr
import numpy as np

# Load verification grids
ds_true = xr.open_dataset("global_h3_res2_air_all_times.nc")
ds_pred = xr.open_dataset("h3_autoregressive_forecast.nc")

# Align timestamps
true_slice = ds_true['air_h3'].sel(time=ds_pred.time)
pred_slice = ds_pred['air_forecast']

# Compute the Absolute Error matrix block
error_matrix = np.abs(pred_slice.values - true_slice.values)

# Average across all global nodes to find the MAE for each future time step
mae_per_step = error_matrix.mean(axis=1)

print("--- FORECAST ERROR DRIFT REPORT ---")
for step_idx in range(len(mae_per_step)):
    hour = (step_idx + 1) * 3
    print(f"Hour +{hour:02d} Forecast MAE: {mae_per_step[step_idx]:.3f} K")

