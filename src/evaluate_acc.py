import argparse
import os
import xarray as xr
import numpy as np

class H3AccuracyEvaluator:
    def __init__(self, true_path, pred_path):
        """
        Initializes the accuracy evaluation pipeline.
        
        Args:
            true_path (str): Path to the ground truth validation file (NetCDF/Zarr)
            pred_path (str): Path to the generated forecast rollout file (NetCDF)
        """
        self.true_path = true_path
        self.pred_path = pred_path
        
        self.ds_true = None
        self.ds_pred = None
        self.matching_times = None
        
    def load_and_align_datasets(self):
        """Loads verification archives and aligns timelines using a safe intersection filter."""
        print("Loading forecast rollout and ground truth records...")
        
        # Open True Dataset (Supports Zarr folders or NetCDF files)
        if self.true_path.endswith('.zarr') or os.path.isdir(self.true_path):
            self.ds_true = xr.open_zarr(self.true_path, consolidated=True)
        else:
            self.ds_true = xr.open_dataset(self.true_path)
            
        # Open Predicted Dataset
        self.ds_pred = xr.open_dataset(self.pred_path)

        # Intersect matching timestamps to secure index tracking bounds
        print("Aligning matching timestamps across files...")
        self.matching_times = np.intersect1d(self.ds_true.time.values, self.ds_pred.time.values)

        if len(self.matching_times) == 0:
            raise ValueError(
                "CRITICAL ERROR: Zero overlapping timestamps found between prediction "
                "and ground truth files. Please check your tracking bounds manually."
            )
        print(f" -> Found {len(self.matching_times)} aligned overlapping forecast steps for evaluation.")

    def compute_anomaly_correlation(self):
        """Computes the Anomaly Correlation Coefficient (ACC) step drift over the timeline."""
        if self.ds_true is None or self.ds_pred is None:
            self.load_and_align_datasets()

        # Pull arrays matching only mutual valid timeframe windows
        # Dynamic variable extraction: matches air_h3 or whatever key your target uses
        true_key = 'air_h3' if 'air_h3' in self.ds_true else 'air_forecast'
        
        true_vals = self.ds_true[true_key].sel(time=self.matching_times).values
        pred_vals = self.ds_pred['air_forecast'].sel(time=self.matching_times).values

        # Compute historical climate baseline (climatology proxy) across this time window
        climatology = true_vals.mean(axis=0)  # Average over time for each individual grid node

        print("\nCalculating Anomaly Correlation Coefficient (ACC) step drift...")
        print("-" * 75)

        for step in range(len(self.matching_times)):
            # 1. Calculate the anomalies (variations from the baseline)
            true_anomaly = true_vals[step] - climatology
            pred_anomaly = pred_vals[step] - climatology

            # 2. Remove spatial means to isolate anomalies cleanly
            true_anomaly_prime = true_anomaly - true_anomaly.mean()
            pred_anomaly_prime = pred_anomaly - pred_anomaly.mean()

            # 3. Compute centered anomaly correlation coefficient (Pearson Correlation)
            numerator = np.sum(true_anomaly_prime * pred_anomaly_prime)
            denominator = np.sqrt(np.sum(true_anomaly_prime**2) * np.sum(pred_anomaly_prime**2))

            acc = numerator / (denominator + 1e-8)
            
            # Assuming a standard 3-hourly time interval footprint sequence
            hour = (step + 1) * 3
            print(f"Lead Time +{hour:02d} Hours | Anomaly Correlation Coefficient (ACC): {acc:.4f}")
        print("-" * 75)

# =====================================================================
# EXECUTION CONTROLLER ROUTINE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate Anomaly Correlation Coefficient (ACC) step drift parameters for Weather AI."
    )
    parser.add_argument(
        "-t", "--true", 
        required=True, 
        help="Path to verification ground-truth archive (.nc or .zarr directory)"
    )
    parser.add_argument(
        "-p", "--pred", 
        default="h3_autoregressive_forecast.nc", 
        help="Path to generated forecast model file (default: h3_autoregressive_forecast.nc)"
    )
    
    args = parser.parse_args()

    # Instantiate and execute evaluator class pipeline
    evaluator = H3AccuracyEvaluator(true_path=args.true, pred_path=args.pred)
    evaluator.compute_anomaly_correlation()

if __name__ == "__main__":
    main()
