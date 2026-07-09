import yaml
import xarray as xr
import pandas as pd
import numpy as np
import torch
from train_single_gpu import SingleGPUH3WeatherGAT, GlobalH3WeatherDataset

def run_autoregressive_rollout(checkpoint_path="checkpoints/best-h3-diurnal-gat-model.ckpt", forecast_days=5):
    # 1. Load System Configuration Parameters
    print("Reading configuration file...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize and Load Trained Weights from Checkpoint
    print(f"Loading trained weights from checkpoint: {checkpoint_path}...")
    model = SingleGPUH3WeatherGAT.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval() # Freeze layers and deactivate dropout for inference
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. HIGH-SPEED OPTIMIZED DATASET INITIALIZATION
    print("Initializing dataset structures...")
    # Open dataset and read coordinates
    ds = xr.open_dataset(config['paths']['data_nc'])
    times = ds['time'].values
    lons = ds['longitude'].values
    lats = ds['latitude'].values
    num_times = len(times)
    num_nodes = len(lons)
    
    # Instantiate an empty class mockup structure to parse statistical mean variables quickly
    dataset = GlobalH3WeatherDataset(
        nc_path=config['paths']['data_nc'],
        history_steps=config['model_params']['history_steps'],
        forecast_offset=config['model_params']['forecast_offset']
    )
    
    # 4. HIGH-SPEED VECTORIZED SOLAR CALCULATION (Drops processing from 4 mins -> 1 second)
    print("Vectorizing cyclical solar forcing matrix arrays across the timeline...")
    datetime_index = pd.to_datetime(times)
    
    # Build 1D time component arrays
    day_of_year = datetime_index.dayofyear.values[:, np.newaxis] # Shape (Time, 1)
    utc_hour = (datetime_index.hour + datetime_index.minute / 60.0).values[:, np.newaxis] # Shape (Time, 1)
    
    # Broadcast time variables against the 1D longitude array (1, Nodes)
    lon_grid = lons[np.newaxis, :]
    
    # Execute matrix operations
    annual_phase = 2.0 * np.pi * day_of_year / 365.25
    local_solar_hour = (utc_hour + lon_grid / 15.0) % 24.0
    diurnal_phase = 2.0 * np.pi * local_solar_hour / 24.0
    
    # Pack everything directly into the master solar tensor layout
    solar_forcings = torch.zeros((num_times, num_nodes, 4), dtype=torch.float32)
    solar_forcings[:, :, 0] = torch.from_numpy(np.sin(diurnal_phase))
    solar_forcings[:, :, 1] = torch.from_numpy(np.cos(diurnal_phase))
    solar_forcings[:, :, 2] = torch.from_numpy(np.sin(annual_phase)).repeat(1, num_nodes)
    solar_forcings[:, :, 3] = torch.from_numpy(np.cos(annual_phase)).repeat(1, num_nodes)

    # 5. Define Initial Conditions
    start_idx = 0
    history_steps = config['model_params']['history_steps']
    
    initial_history = dataset.norm_air[start_idx : start_idx + history_steps].t().to(device)
    static_features = dataset.static_features.to(device)
    
    steps_per_day = 8
    total_rollout_steps = forecast_days * steps_per_day
    print(f"Starting a {forecast_days}-day forecast rollout ({total_rollout_steps} sequential steps)...")

    predictions_history = torch.zeros((total_rollout_steps, num_nodes), dtype=torch.float32)
    current_history = initial_history.clone()

    # 6. RUN HARDWARE INFUSION INTERACTIVE EVALUATION LOOP
    with torch.no_grad():
        for step in range(total_rollout_steps):
            current_time_idx = start_idx + history_steps + step
            
            # Pull solar data from our optimized pre-calculated matrix block
            x_solar = solar_forcings[current_time_idx].to(device)
            
            # Combine current dynamic history state, static geography features, and solar forcings
            x_input = torch.cat([current_history, static_features, x_solar], dim=1)
            x_input = x_input.unsqueeze(0) # Add simulated batch axis
            
            # Run model forward pass to generate forecast step
            y_hat = model(x_input).squeeze(0)
            predictions_history[step] = y_hat.cpu()
            
            # Auto-regressive queue shift
            updated_history = current_history[:, 1:]
            current_history = torch.cat([updated_history, y_hat.unsqueeze(1)], dim=1)
            
            if (step + 1) % steps_per_day == 0:
                print(f" -> Completed Forecast Day {(step + 1) // steps_per_day}/{forecast_days}")

    # 7. EXPORT COMPLIANT FORECAST DATASET FILE
    print("Denormalizing generated data matrix back to Kelvin scales...")
    final_predictions_kelvin = (predictions_history * dataset.air_std.item()) + dataset.air_mean.item()
    rollout_times = dataset.ds['time'].values[start_idx + history_steps : start_idx + history_steps + total_rollout_steps]
    
    print("Structuring rollout output NetCDF file...")
    ds_rollout = xr.Dataset(
        data_vars={
            "air_forecast": (["time", "h3_index"], final_predictions_kelvin.numpy(), {"units": "degK", "coordinates": "longitude latitude"}),
            "longitude": (["h3_index"], dataset.ds['longitude'].values, {"units": "degrees_east"}),
            "latitude": (["h3_index"], dataset.ds['latitude'].values, {"units": "degrees_north"}),
        },
        coords={"time": rollout_times, "h3_index": dataset.ds['h3_index'].values},
        attrs={"title": f"{forecast_days}-Day Graph Attention Auto-Regressive Rollout Model Output"}
    )
    
    output_filename = "h3_autoregressive_forecast.nc"
    ds_rollout.to_netcdf(output_filename, format="NETCDF4")
    print(f"SUCCESS: Auto-regressive forecast rollout saved to: {output_filename}")

if __name__ == "__main__":
    run_autoregressive_rollout()

