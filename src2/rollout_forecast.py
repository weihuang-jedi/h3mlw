import yaml
import xarray as xr
import numpy as np
import torch
from train_distributed import DistributedH3WeatherGAT, GlobalH3WeatherDataset

def run_autoregressive_rollout(checkpoint_path="checkpoints/best-h3-diurnal-gat-model.ckpt", forecast_days=5):
    # 1. Load System Configuration Parameters
    print("Reading configuration file...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize and Load Trained Weights from Checkpoint
    print(f"Loading trained weights from checkpoint: {checkpoint_path}...")
    # map_location='cpu' ensures it loads safely before transferring to the active GPU
    model = DistributedH3WeatherGAT.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval() # Freeze layers and deactivate dropout for inference
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Initialize Dataset to Extract Normalization Parameters and Initial States
    print("Initializing dataset structures...")
    dataset = GlobalH3WeatherDataset(
        nc_path=config['paths']['data_nc'],
        history_steps=config['model_params']['history_steps'],
        forecast_offset=config['model_params']['forecast_offset']
    )
    
    num_nodes = dataset.static_features.shape[0]
    history_steps = config['model_params']['history_steps']
    
    # 4. Define the Initial Conditions (Start from index 0 of your dataset array)
    start_idx = 0
    # Isolate initial historical window data tensor shape: (num_nodes, history_steps)
    initial_history = dataset.norm_air[start_idx : start_idx + history_steps].t().to(device)
    static_features = dataset.static_features.to(device) # Shape: (num_nodes, 2)
    
    # 20CRv2c utilizes 3-hourly time steps. Compute total loops required for the target window
    steps_per_day = 8 # 24 hours / 3 hours
    total_rollout_steps = forecast_days * steps_per_day
    print(f"Starting a {forecast_days}-day forecast rollout ({total_rollout_steps} sequential steps)...")

    # Allocate a tensor to store all future predictions: shape (total_rollout_steps, num_nodes)
    predictions_history = torch.zeros((total_rollout_steps, num_nodes), dtype=torch.float32)
    
    # Current active historical frame matrix pointer that updates dynamically inside the loop
    current_history = initial_history.clone()

    with torch.no_grad():
        for step in range(total_rollout_steps):
            # Calculate the explicit index footprint along the global dataset timeline
            current_time_idx = start_idx + history_steps + step
            
            # Extract the correct solar forcing features array for the current time step
            x_solar = dataset.solar_forcings[current_time_idx].to(device) # Shape: (num_nodes, 4)
            
            # Concatenate current dynamic history, static features, and solar forcings
            # Input shape matches the model's expected layout: (num_nodes, history_steps + 2 + 4)
            x_input = torch.cat([current_history, static_features, x_solar], dim=1)
            
            # Add batch dimension to simulate batch size 1: shape (1, num_nodes, 8)
            x_input = x_input.unsqueeze(0)
            
            # Execute inference pass to predict the next time step (normalized value)
            y_hat = model(x_input).squeeze(0) # Shape: (num_nodes,)
            
            # Store the prediction in our tracking history tensor
            predictions_history[step] = y_hat.cpu()
            
            # --- AUTO-REGRESSIVE UPDATE LOOP ---
            # Drop the oldest historical time step (column 0) and slide the rest to the left
            updated_history = current_history[:, 1:]
            # Append the new prediction as the most recent historical step (column -1)
            current_history = torch.cat([updated_history, y_hat.unsqueeze(1)], dim=1)
            
            if (step + 1) % steps_per_day == 0:
                print(f" -> Completed Forecast Day {(step + 1) // steps_per_day}/{forecast_days}")

    # =====================================================================
    # 5. DENORMALIZE PREDICTIONS AND EXPORT COMPLIANT NETCDF
    # =====================================================================
    print("Denormalizing generated data back to Kelvin scales...")
    # Reverse the Z-score transformation to convert values back to original units
    final_predictions_kelvin = (predictions_history * dataset.air_std.item()) + dataset.air_mean.item()
    
    # Extract output timestamps for the forecast period
    rollout_times = dataset.ds['time'].values[start_idx + history_steps : start_idx + history_steps + total_rollout_steps]
    
    print("Structuring rollout output NetCDF file...")
    ds_rollout = xr.Dataset(
        data_vars={
            "air_forecast": (
                ["time", "h3_index"], 
                final_predictions_kelvin.numpy(), 
                {
                    "units": "degK", 
                    "long_name": "Auto-Regressive Multi-Day Surface Air Temperature Forecast",
                    "coordinates": "longitude latitude"
                }
            ),
            "longitude": (["h3_index"], dataset.ds['longitude'].values, {"units": "degrees_east"}),
            "latitude": (["h3_index"], dataset.ds['latitude'].values, {"units": "degrees_north"}),
        },
        coords={
            "time": rollout_times,
            "h3_index": dataset.ds['h3_index'].values
        },
        attrs={
            "title": f"{forecast_days}-Day Graph Attention Auto-Regressive Rollout Model Output",
            "history_window_input": f"{history_steps} steps",
            "forecast_resolution": "3-hourly increments"
        }
    )
    
    output_filename = "h3_autoregressive_forecast.nc"
    ds_rollout.to_netcdf(output_filename, format="NETCDF4")
    print(f"SUCCESS: Auto-regressive forecast rollout saved to: {output_filename}")

if __name__ == "__main__":
    # Execute a 5-day auto-regressive forecast run
    run_autoregressive_rollout(checkpoint_path="checkpoints/best-h3-diurnal-gat-model.ckpt", forecast_days=5)

