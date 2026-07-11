import argparse
import os
import xarray as xr
import pandas as pd
import numpy as np

class H3AccuracyEvaluator:
    def __init__(self, true_path, pred_path, varname="unknown"):
        self.true_path = true_path
        self.pred_path = pred_path
        self.varname = varname

        self.ds_true = None
        self.ds_pred = None
        self.matching_times = None

    def load_and_align_datasets(self):
        print("Loading forecast rollout and ground truth records...")
        if self.true_path.endswith('.zarr') or os.path.isdir(self.true_path):
            self.ds_true = xr.open_zarr(self.true_path, consolidated=True)
        else:
            self.ds_true = xr.open_dataset(self.true_path)

        self.ds_pred = xr.open_dataset(self.pred_path)

        print("Aligning matching timestamps across files...")
        self.matching_times = np.intersect1d(self.ds_true.time.values, self.ds_pred.time.values)

        if len(self.matching_times) == 0:
            raise ValueError("CRITICAL ERROR: Zero overlapping timestamps found.")
        print(f" -> Found {len(self.matching_times)} aligned overlapping forecast steps for evaluation.")

    def compute_anomaly_correlation(self):
        if self.ds_true is None or self.ds_pred is None:
            self.load_and_align_datasets()

        true_key = 'air_h3' if 'air_h3' in self.ds_true else self.varname

        # -----------------------------------------------------------------
        # FIXED: Dynamic variable discovery to handle any suffix structure
        # -----------------------------------------------------------------
        ignore_keys = {"land_sea_mask", "elevation", "face_nodes", "x_cartesian", 
                       "y_cartesian", "z_cartesian", "longitude", "latitude", 
                       "time", "node", "face", "three", "icosahedral_mesh"}
        
        # Discover the real active payload keys
        true_payloads = list(set(self.ds_true.data_vars.keys()) - ignore_keys)
        pred_payloads = list(set(self.ds_pred.data_vars.keys()) - ignore_keys)
        
        if not true_payloads or not pred_payloads:
            raise KeyError("Could not isolate a weather payload variable in one of your NetCDF files.")
            
        true_key = true_payloads[0]
        pred_key = pred_payloads[0]
        
        print(f" -> Mapping True Variable Core: '{true_key}'")
        print(f" -> Mapping Predicted Variable Core: '{pred_key}'")

        # Slice matching timestamps across arrays dynamically using the discovered keys
        true_ds_sliced = self.ds_true[true_key].sel(time=self.matching_times)
        pred_ds_sliced = self.ds_pred[pred_key].sel(time=self.matching_times)

        # Convert times to a pandas DatetimeIndex to easily extract diurnal hours
        pd_times = pd.to_datetime(self.matching_times)
        unique_hours = np.unique(pd_times.hour)

        # FIX: Pre-compute an hour-specific climatology dictionary (00Z, 03Z, 06Z, etc.)
        hourly_climatology = {}
        for hr in unique_hours:
            # Find all frames matching this specific UTC hour inside the truth file
            hr_mask = pd_times.hour == hr
            hourly_climatology[hr] = true_ds_sliced.values[hr_mask].mean(axis=0)

        print("\nCalculating Diurnal-Corrected Anomaly Correlation Coefficient (ACC) step drift...")
        print("=" * 115)
        print(f"{'Lead Time':<15} | {'Forecast Frame (UTC)':<25} | {'Ground Truth Frame (UTC)':<25} | {'ACC':<10}")
        print("=" * 115)

        for step in range(len(self.matching_times)):
            current_dt = pd_times[step]
            pred_time_str = str(current_dt)
            true_time_str = str(pd.to_datetime(true_ds_sliced.time.values[step]))
            
            # Fetch the baseline belonging strictly to this specific hour of the day
            climatology = hourly_climatology[current_dt.hour]

            # 1. Calculate the true anomalies relative to the time-of-day baseline
            true_anomaly = true_ds_sliced.values[step] - climatology
            pred_anomaly = pred_ds_sliced.values[step] - climatology

            # 2. Remove spatial means
            true_anomaly_prime = true_anomaly - true_anomaly.mean()
            pred_anomaly_prime = pred_anomaly - pred_anomaly.mean()

            # 3. Compute Pearson correlation
            numerator = np.sum(true_anomaly_prime * pred_anomaly_prime)
            denominator = np.sqrt(np.sum(true_anomaly_prime**2) * np.sum(pred_anomaly_prime**2))

            acc = numerator / (denominator + 1e-8)

            hour = (step + 1) * 3
            lead_str = f"+{hour:02d} Hours"
            
            print(f"{lead_str:<15} | {pred_time_str:<25} | {true_time_str:<25} | {acc:.4f}")
        print("=" * 115)

def main():
    parser = argparse.ArgumentParser(description="Calculate Anomaly Correlation Coefficient (ACC) for Weather AI.")
    parser.add_argument("-t", "--true", required=True, help="Path to verification ground-truth archive")
    parser.add_argument("-p", "--pred", default="h3_autoregressive_forecast.nc", help="Path to forecast file")
    parser.add_argument("-v", "--varname", default="air_forecast", help="forecast variable name")

    args = parser.parse_args()
    evaluator = H3AccuracyEvaluator(true_path=args.true, pred_path=args.pred, varname=args.varname)
    evaluator.compute_anomaly_correlation()

if __name__ == "__main__":
    main()
