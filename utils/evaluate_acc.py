import argparse
import numpy as np
import pandas as pd
import xarray as xr

class GraphCastPerformanceEvaluator:
    """
    Computes the Diurnal-Corrected Anomaly Correlation Coefficient (ACC)
    between GraphCast multi-step rollouts and true verifying analytical fields.
    """
    def __init__(self, true_path: str, pred_path: str):
        self.true_path = true_path
        self.pred_path = pred_path
        self.ds_true = None
        self.ds_pred = None
        self.matching_times = None

    def load_and_align_datasets(self):
        print("Loading forecast rollout and ground truth records...")
        self.ds_true = xr.open_dataset(self.true_path)
        self.ds_pred = xr.open_dataset(self.pred_path)

        print("Aligning matching timestamps across files...")
        # Intersect dates regardless of relative time units or calendar attributes
        true_times = pd.to_datetime(self.ds_true.time.values)
        pred_times = pd.to_datetime(self.ds_pred.time.values)
        
        self.matching_times = np.intersect1d(true_times, pred_times)
        
        if len(self.matching_times) == 0:
            raise ValueError(
                f"\n[CRITICAL] Absolute time intersection is empty.\n"
                f" -> Truth bounds: [{true_times[0]} to {true_times[-1]}]\n"
                f" -> Forecast bounds: [{pred_times[0]} to {pred_times[-1]}]\n"
            )
        print(f" -> Found {len(self.matching_times)} aligned overlapping forecast steps for evaluation.")

    def compute_anomaly_correlation(self):
        # Filter structural mesh/grid metadata out to capture pure meteorological payload fields
        ignore_keys = {
            "land_sea_mask", "elevation", "face_nodes", "x_cartesian", 
            "y_cartesian", "z_cartesian", "longitude", "latitude", 
            "time", "node", "face", "three", "icosahedral_mesh", "nbnds", "time_bnds"
        }
        
        true_payloads = list(set(self.ds_true.data_vars.keys()) - ignore_keys)
        pred_payloads = list(set(self.ds_pred.data_vars.keys()) - ignore_keys)
        
        if not true_payloads or not pred_payloads:
            raise KeyError("Could not isolate a target weather variable in one or both of your input files.")
            
        true_key = true_payloads[0]
        pred_key = pred_payloads[0]
        
        print(f" -> Mapping True Variable Core: '{true_key}'")
        print(f" -> Mapping Predicted Variable Core: '{pred_key}'")

        # Slice datasets down to intersecting time scales using native xarray alignment
        true_ds_sliced = self.ds_true[true_key].sel(time=self.matching_times)
        pred_ds_sliced = self.ds_pred[pred_key].sel(time=self.matching_times)

        # Convert temporal metadata pointers explicitly into standard pandas structures
        true_times = pd.to_datetime(true_ds_sliced.time.values)
        pred_times = pd.to_datetime(pred_ds_sliced.time.values)

        print("\nCalculating Diurnal-Corrected Anomaly Correlation Coefficient (ACC) step drift...")
        print("===================================================================================================================")
        print(f"{'Lead Time':<15} | {'Forecast Frame (UTC)':<25} | {'Ground Truth Frame (UTC)':<25} | {'ACC'}")
        print("===================================================================================================================")

        for step in range(len(self.matching_times)):
            # Capture the hour code corresponding to the current rollout state
            target_hour = pred_times[step].hour
            
            # Compute climatology baseline specifically for this hour code to account for diurnal cycle variations
            matching_climatology = self.ds_true[true_key].dropna(dim='time')
            hour_mask = matching_climatology.time.dt.hour == target_hour
            climatology = matching_climatology.isel(time=hour_mask).mean(dim='time').values

            # Calculate raw spatial field anomaly representations
            true_anomaly = true_ds_sliced.values[step] - climatology
            pred_anomaly = pred_ds_sliced.values[step] - climatology

            # Compute spatial pearson correlation parameters across the active dimensions
            numerator = np.sum(true_anomaly * pred_anomaly)
            denominator = np.sqrt(np.sum(true_anomaly**2) * np.sum(pred_anomaly**2))
            
            acc = numerator / (denominator + 1e-8) if denominator > 0 else 0.0

            # Output step lines smoothly into stdout matrix tracks
            lead_hours = (step + 1) * 3
            pred_time_str = str(pred_times[step])[:19]
            true_time_str = str(true_times[step])[:19]
            
            print(f"+{lead_hours:02d} Hours        | {pred_time_str:<25} | {true_time_str:<25} | {acc:.4f}")
            
        print("===================================================================================================================")
        
        # Free resource descriptors cleanly
        self.ds_true.close()
        self.ds_pred.close()


def main():
    parser = argparse.ArgumentParser(description="Calculate Diurnal-Corrected Anomaly Correlation Coefficients (ACC).")
    parser.add_argument("-t", "--true", required=True, help="Path to verifying Ground Truth file (.nc or .zarr)")
    parser.add_argument("-p", "--pred", required=True, help="Path to GraphCast Model Prediction rollout file (.nc or .zarr)")
    args = parser.parse_args()

    evaluator = GraphCastPerformanceEvaluator(true_path=args.true, pred_path=args.pred)
    evaluator.load_and_align_datasets()
    evaluator.compute_anomaly_correlation()

if __name__ == "__main__":
    main()
