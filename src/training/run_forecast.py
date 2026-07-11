import os
import yaml
import argparse
import torch
import numpy as np
import xarray as xr
from training_on_icosahedral_grid import DeepGraphCastModel

class GraphCastForecastEngine:
    """
    Loads trained hierarchical GraphCast checkpoints and runs an autoregressive 
    iterative forecast rollout sequence driven entirely by config settings.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ds_init = None

    def initialize_system(self):
        checkpoint_path = self.cfg['inference_params']['weights']
        print(f"[INIT] Loading compiled DeepGraphCast model checkpoint: {checkpoint_path}")
        self.model = DeepGraphCastModel.load_from_checkpoint(checkpoint_path, config=self.cfg)
        self.model.to(self.device)
        self.model.eval()

        # Swapped key target parameter boundary to initial_condition
        target_store = self.cfg['paths']['initial_condition']
        print(f"[INIT] Opening active data warehouse: {target_store}")
        
        if target_store.endswith('.nc') or target_store.endswith('.netcdf'):
            print(" -> Detected NetCDF initial condition format. Initializing standard reader...")
            self.ds_init = xr.open_dataset(target_store)
        else:
            print(" -> Detected Zarr database initial condition format. Initializing parallel stream reader...")
            self.ds_init = xr.open_zarr(target_store, consolidated=True)

    def find_index_by_timestamp(self, target_time_str: str) -> int:
        """Looks up the integer index matching a target ISO datetime string."""
        print(f"[DATETIME] Searching for initializing step matching timestamp: {target_time_str}")
        try:
            target_dt = np.datetime64(target_time_str)
        except ValueError:
            raise ValueError(f"Could not parse '{target_time_str}'. Use standard ISO format: YYYY-MM-DDTHH:MM:SS")

        init_times = self.ds_init.time.values
        matching_indices = np.where(init_times == target_dt)[0]

        if len(matching_indices) == 0:
            min_time = str(init_times[0])[:19]
            max_time = str(init_times[-1])[:19]
            raise IndexError(
                f"\n[CRITICAL] Target timestamp '{target_time_str}' is absent from this initial condition file.\n"
                f" -> File spans from [{min_time}] to [{max_time}].\n"
            )

        start_idx = int(matching_indices[0])
        print(f" -> Found matching index coordinate entry: {start_idx}")
        return start_idx

    def generate_rollout(self) -> tuple:
        """Runs the autoregressive forecast loop over the configured step limits."""
        start_time_str = self.cfg['inference_params']['start_time']
        forecast_steps = self.cfg['inference_params']['steps']
        history_steps = self.cfg['model_params']['history_steps']
        
        start_idx = self.find_index_by_timestamp(start_time_str)
        var_name = [k for k in self.ds_init.data_vars if 'mask' not in k and 'elevation' not in k and 'face' not in k][0]
        
        # -----------------------------------------------------------------
        # FIXED: Slice backward from start_idx to guarantee exactly history_steps
        # -----------------------------------------------------------------
        if start_idx + 1 >= history_steps:
            # If there is enough history behind start_idx, slice backwards to include it
            init_slice = slice(start_idx - history_steps + 1, start_idx + 1)
        else:
            # Fallback for small dedicated files: grab the first history_steps of the file
            print(f" -> [WARN] Insufficient history behind index {start_idx}. Defaulting to first {history_steps} steps of the file.")
            init_slice = slice(0, history_steps)
            
        raw_history = self.ds_init[var_name].isel(time=init_slice).values
        
        # Guard check to ensure channel sizes are perfect before hitting the GNN
        if raw_history.shape[0] != history_steps:
            raise ValueError(
                f"\n[CRITICAL] Slicing failure: Got {raw_history.shape[0]} steps, "
                f"but model requires exactly history_steps: {history_steps}.\n"
                f" -> Ensure your initial condition file has at least {history_steps} total timesteps.\n"
            )
        
        lsm = torch.from_numpy(self.ds_init['land_sea_mask'].values).float().to(self.device)
        elevation = torch.from_numpy(self.ds_init['elevation'].values).float().to(self.device)
        elevation = (elevation - elevation.mean()) / (elevation.std() + 1e-5)
        static_features = torch.stack([lsm, elevation], dim=-1)

        x_weather = torch.tensor(raw_history, dtype=torch.float32).permute(1, 0).to(self.device)
        current_input = torch.cat([x_weather, static_features], dim=-1).unsqueeze(0)

        node_counts = {lvl: int(getattr(self.model, f"edge_index_m{lvl}").max() + 1) for lvl in self.cfg['model_params']['hierarchy_levels']}

        forecast_history = []
        
        # Setup timestamps for the forward steps
        if start_idx + 1 >= history_steps:
            timestamps = self.ds_init.time.values[start_idx + 1 : start_idx + 1 + forecast_steps]
        else:
            timestamps = self.ds_init.time.values[history_steps : history_steps + forecast_steps]

        # Fallback if the small initialization file doesn't contain future target timestamps
        if len(timestamps) < forecast_steps:
            print(" -> [INFO] Generating future synthetic timeline steps for forecast export...")
            time_delta = self.ds_init.time.values[1] - self.ds_init.time.values[0]
            base_time = self.ds_init.time.values[-1]
            timestamps = [base_time + (i + 1) * time_delta for i in range(forecast_steps)]
            timestamps = np.array(timestamps)

        print(f"[ROLLOUT] Beginning {forecast_steps}-step iterative autoregressive loop...")
        with torch.no_grad():
            for step in range(forecast_steps):
                pred_delta = self.model(current_input, node_counts)
                prev_field = current_input[0, :, history_steps - 1]
                pred_field = prev_field + pred_delta.squeeze(0)

                forecast_history.append(pred_field.cpu().numpy())

                next_input = torch.zeros_like(current_input)
                # Slide history channels over by 1 step
                next_input[0, :, :history_steps - 1] = current_input[0, :, 1:history_steps]
                # Inject the newly predicted field
                next_input[0, :, history_steps - 1] = pred_field
                # Keep static land/topo channels completely intact at the end
                next_input[0, :, -2:] = current_input[0, :, -2:]
                current_input = next_input
                
                print(f" -> Computed step {step+1}/{forecast_steps} | Forecast Target: {str(timestamps[step])[:16]}")

        return np.array(forecast_history), timestamps


    def export_to_ugrid_nc(self, predictions: np.ndarray, timestamps: np.array):
        out_path = self.cfg['inference_params']['output']
        print(f"[EXPORT] Packing forecast variables into file target: {out_path}")
        var_name = [k for k in self.ds_init.data_vars if 'mask' not in k and 'elevation' not in k and 'face' not in k][0]
        
        ds_out = xr.Dataset(
            data_vars={
                f"{var_name}_forecast": (["time", "node"], predictions.astype(np.float32)),
                "land_sea_mask": (["node"], self.ds_init['land_sea_mask'].values),
                "elevation": (["node"], self.ds_init['elevation'].values),
                "face_nodes": (["face", "three"], self.ds_init['face_nodes'].values),
                "longitude": (["node"], self.ds_init['longitude'].values, {"units": "degrees_east"}),
                "latitude": (["node"], self.ds_init['latitude'].values, {"units": "degrees_north"}),
                "icosahedral_mesh": ([], 0, self.ds_init['icosahedral_mesh'].attrs)
            },
            coords={
                "time": timestamps,
                "node": self.ds_init.node.values,
                "face": self.ds_init.face.values,
                "three": np.arange(3)
            }
        )
        ds_out.to_netcdf(out_path, format="NETCDF4")
        print("SUCCESS: Forecast execution pipeline completed successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="Generate multi-step forecast using parameters defined in config.")
    parser.add_argument("-c", "--config", default="config_inference.yaml", help="Path to config yaml file")
    args = parser.parse_args()

    engine = GraphCastForecastEngine(config_path=args.config)
    engine.initialize_system()
    preds, times = engine.generate_rollout()
    engine.export_to_ugrid_nc(predictions=preds, timestamps=times)

if __name__ == "__main__":
    main()

