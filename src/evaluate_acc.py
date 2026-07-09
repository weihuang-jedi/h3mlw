import xarray as xr
import numpy as np

print("Loading forecast rollout and ground truth records...")
ds_true = xr.open_dataset("../data/global_h3_res2_air_sfc_1976.nc")
ds_pred = xr.open_dataset("h3_autoregressive_forecast.nc")

# Align matching timestamps using a safe intersection filter
matching_times = np.intersect1d(ds_true.time.values, ds_pred.time.values)

if len(matching_times) == 0:
    raise ValueError("CRITICAL ERROR: Zero overlapping timestamps found between prediction and ground truth files. check time indexes manually.")

# Pull metrics matching only the mutual valid timeframe windows
true_vals = ds_true['air_h3'].sel(time=matching_times).values
pred_vals = ds_pred['air_forecast'].sel(time=matching_times).values

# Update the loop tracking array to use the unified time index length
times = matching_times

# Compute a simple historical climate baseline for this time window
# (In production, this is a multi-decade average. Here we use the window mean as a proxy)
climatology = true_vals.mean(axis=0) # Average over time for each individual node

print("Calculating Anomaly Correlation Coefficient (ACC) step drift...")

for step in range(len(times)):
    # Calculate the anomalies (variations from the baseline)
    true_anomaly = true_vals[step] - climatology
    pred_anomaly = pred_vals[step] - climatology
    
    # Remove any spatial means to isolate the anomalies cleanly
    true_anomaly_prime = true_anomaly - true_anomaly.mean()
    pred_anomaly_prime = pred_anomaly - pred_anomaly.mean()
    
    # Compute the centered anomaly correlation coefficient (Pearson Correlation)
    numerator = np.sum(true_anomaly_prime * pred_anomaly_prime)
    denominator = np.sqrt(np.sum(true_anomaly_prime**2) * np.sum(pred_anomaly_prime**2))
    
    acc = numerator / (denominator + 1e-8)
    hour = (step + 1) * 3
    print(f"Lead Time +{hour:02d} Hours | Anomaly Correlation Coefficient (ACC): {acc:.4f}")

