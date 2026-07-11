import os
import yaml
import argparse
import torch
import numpy as np
import xarray as xr
from training_on_icosahedral_grid import DeepGraphCastModel

class GraphCastForecastEngine:
    def __init__(self, config_path: str, checkpoint_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.checkpoint_path = checkpoint_path
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ds_zarr = None

    def initialize_system(self):
        print(f"[INIT] Loading compiled DeepGraphCast model checkpoint: {self.checkpoint_path}")
        self.model = DeepGraphCastModel.load_from_checkpoint(self.checkpoint_path, config=self.cfg)
        self.model.to(self.device)
        self.model.eval()

        target_store = self.cfg['paths']['zarr_store']
        print(f"[INIT] Opening active data warehouse: {target_store}")

        # -----------------------------------------------------------------
        # FIXED: Automatically adapt data loader to .nc or .zarr formats
        # -----------------------------------------------------------------
        if target_store.endswith('.nc') or target_store.endswith('.netcdf'):
            print(" -> Detected NetCDF source file layout format. Initializing standard reader...")
            self.ds_zarr = xr.open_dataset(target_store)
        else:
            print(" -> Detected Zarr database store layout format. Initializing parallel stream reader...")
            self.ds_zarr = xr.open_zarr(target_store, consolidated=True)

    def find_index_by_timestamp(self, target_time_str: str) -> int:
        """Looks up the integer index matching a target ISO datetime string."""
        print(f"[DATETIME] Searching for initializing step matching timestamp: {target_time_str}")
        
        # Convert the string target into a native numpy datetime64 object
        try:
            target_dt = np.datetime64(target_time_str)
        except ValueError:
            raise ValueError(f"Could not parse string format '{target_time_str}'. Use standard ISO format: YYYY-MM-DDTHH:MM:SS")

        zarr_times = self.ds_zarr.time.values
        matching_indices = np.where(zarr_times == target_dt)[0]

        if len(matching_indices) == 0:
            # Provide diagnostic support if the target date is outside the data window
            min_time = str(zarr_times[0])[:19]
            max_time = str(zarr_times[-1])[:19]
            raise IndexError(
                f"\n[CRITICAL] Target timestamp '{target_time_str}' is absent from this dataset repository.\n"
                f" -> Your Zarr file contains historical parameters spanning from [{min_time}] to [{max_time}].\n"
            )

        start_idx = int(matching_indices[0])
        print(f" -> Found matching index coordinate entry: {start_idx}")
        return start_idx

    def generate_rollout(self, start_idx: int, forecast_steps: int) -> tuple:
        history_steps = self.cfg['model_params']['history_steps']
        var_name = [k for k in self.ds_zarr.data_vars if 'mask' not in k and 'elevation' not in k and 'face' not in k][0]
        
        # Fetch initial conditioning history frames from the located index
        init_slice = slice(start_idx, start_idx + history_steps)
        raw_history = self.ds_zarr[var_name].isel(time=init_slice).values
        init_time_str = str(self.ds_zarr.time.values[start_idx])[:16]
        
        print(f"[FORECAST] Conditioning loop initialized on weather state from: {init_time_str} UTC")
        
        lsm = torch.from_numpy(self.ds_zarr['land_sea_mask'].values).float().to(self.device)
        elevation = torch.from_numpy(self.ds_zarr['elevation'].values).float().to(self.device)
        elevation = (elevation - elevation.mean()) / (elevation.std() + 1e-5)
        static_features = torch.stack([lsm, elevation], dim=-1)

        x_weather = torch.tensor(raw_history, dtype=torch.float32).permute(1, 0).to(self.device)
        current_input = torch.cat([x_weather, static_features], dim=-1).unsqueeze(0)

        node_counts = {lvl: int(getattr(self.model, f"edge_index_m{lvl}").max() + 1) for lvl in self.cfg['model_params']['hierarchy_levels']}
        fine_nodes = node_counts[self.cfg['model_params']['hierarchy_levels'][0]]

        forecast_history = []
        timestamps = self.ds_zarr.time.values[start_idx + history_steps : start_idx + history_steps + forecast_steps]

        print(f"[ROLLOUT] Beginning {forecast_steps}-step iterative autoregressive loop...")
        with torch.no_grad():
            for step in range(forecast_steps):
                pred_delta = self.model(current_input, node_counts)
                prev_field = current_input[0, :, history_steps - 1]
                pred_field = prev_field + pred_delta.squeeze(0)

                forecast_history.append(pred_field.cpu().numpy())

                next_input = torch.zeros_like(current_input)
                next_input[0, :, :-3] = current_input[0, :, 1:-2]
                next_input[0, :, history_steps - 1] = pred_field
                next_input[0, :, -2:] = current_input[0, :, -2:]
                current_input = next_input
                
                print(f" -> Computed step {step+1}/{forecast_steps} | Forecast Target: {str(timestamps[step])[:16]}")

        return np.array(forecast_history), timestamps

    def export_to_ugrid_nc(self, predictions: np.ndarray, timestamps: np.array, out_path: str):
        print(f"[EXPORT] Packing forecast variables into file target: {out_path}")
        var_name = [k for k in self.ds_zarr.data_vars if 'mask' not in k and 'elevation' not in k and 'face' not in k][0]
        
        ds_out = xr.Dataset(
            data_vars={
                f"{var_name}_forecast": (["time", "node"], predictions.astype(np.float32)),
                "land_sea_mask": (["node"], self.ds_zarr['land_sea_mask'].values),
                "elevation": (["node"], self.ds_zarr['elevation'].values),
                "face_nodes": (["face", "three"], self.ds_zarr['face_nodes'].values),
                "longitude": (["node"], self.ds_zarr['longitude'].values, {"units": "degrees_east"}),
                "latitude": (["node"], self.ds_zarr['latitude'].values, {"units": "degrees_north"}),
                "icosahedral_mesh": ([], 0, self.ds_zarr['icosahedral_mesh'].attrs)
            },
            coords={
                "time": timestamps,
                "node": self.ds_zarr.node.values,
                "face": self.ds_zarr.face.values,
                "three": np.arange(3)
            }
        )
        ds_out.to_netcdf(out_path, format="NETCDF4")
        print("SUCCESS: Forecast compilation finalized.\n")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-step autoregressive weather forecasts by timestamp.")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-o", "--output", default="model_forecast_output.nc")
    # Added string date-time lookup parameter
    parser.add_argument("-t", "--start_time", default="2000-07-01T00:00:00", 
                        help="ISO format starting time target (e.g., 2000-07-01T00:00:00)")
    parser.add_argument("-n", "--steps", type=int, default=8, help="Number of forward rollout forecast steps")
    args = parser.parse_args()

    engine = GraphCastForecastEngine(config_path=args.config, checkpoint_path=args.weights)
    engine.initialize_system()
    
    # Resolve the user's requested date-time string into a Zarr matrix index location
    start_idx = engine.find_index_by_timestamp(args.start_time)
    
    preds, times = engine.generate_rollout(start_idx=start_idx, forecast_steps=args.steps)
    engine.export_to_ugrid_nc(predictions=preds, timestamps=times, out_path=args.output)

if __name__ == "__main__":
    main()
