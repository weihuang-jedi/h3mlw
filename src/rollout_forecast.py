import yaml
import xarray as xr
import pandas as pd
import numpy as np
import torch
import os
from train_single_gpu import SingleGPUH3WeatherGAT, FastMultiYearH3Dataset

def run_autoregressive_rollout(checkpoint_path="checkpoints/best-h3-20year-cyclic-model.ckpt", forecast_days=5):
    # 1. Load System Configuration Parameters
    print("Reading configuration file...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Initialize and Load Trained Weights from Checkpoint
    print(f"Loading trained weights from checkpoint: {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        # Look into the configured directory fallback if dirpath is appended
        checkpoint_dir = config['paths'].get('checkpoint_dir', 'checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, "best-h3-20year-cyclic-model.ckpt")
        print(f" -> Adjusting target path to: {checkpoint_path}")

    model = SingleGPUH3WeatherGAT.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval() # Freeze layers and deactivate dropout for inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. HIGH-SPEED OPTIMIZED DATASET INITIALIZATION
    print("Initializing dataset structures...")
    data_path = config['paths']['data_pattern']
    
    # Handle both Zarr paths and standard netcdf configuration structures seamlessly
    if data_path.endswith('.zarr') or os.path.isdir(data_path) and not data_path.endswith('.nc'):
        ds = xr.open_zarr(data_path, consolidated=True)
    else:
        ds = xr.open_mfdataset(data_path, combine="by_coords", data_vars="minimal")

    # Instantiate our fast multi-year dataset to acquire the normalized features & stats
    dataset = FastMultiYearH3Dataset(
        file_pattern=data_path,
        history_steps=config['model_params']['history_steps'],
        forecast_offset=config['model_params']['forecast_offset']
    )

    times = dataset.times
    lons = dataset.lons
    num_times = dataset.num_times
    num_nodes = dataset.num_nodes

    # 4. HIGH-SPEED VECTORIZED SOLAR CALCULATION
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

    # Normalize raw input on-the-fly using calculated metrics from dataset instantiation
    raw_history = dataset.air_data[start_idx : start_idx + history_steps]
    norm_history = torch.nan_to_num((raw_history - dataset.air_mean) / (dataset.air_std + 1e-6), nan=0.0).t()
    
    initial_history = norm_history.to(device)
    static_features = dataset.static_features.to(device)

    steps_per_day = 8  # Assumes 3-hourly data checkpoints (24 hours / 3 hours)
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
    final_predictions_kelvin = (predictions_history * dataset.air_std) + dataset.air_mean
    rollout_times = dataset.times[start_idx + history_steps : start_idx + history_steps + total_rollout_steps]

    print("Structuring rollout output NetCDF file...")
    
    # Track spatial dimensions dynamically
    h3_dim_name = 'h3_index' if 'h3_index' in dataset.ds.dims else 'nodes'
    
    ds_rollout = xr.Dataset(
        data_vars={
            "air_forecast": ([ "time", h3_dim_name ], final_predictions_kelvin.numpy(), {"units": "degK"}),
            "longitude": ([ h3_dim_name ], dataset.ds['longitude'].values.squeeze(), {"units": "degrees_east"}),
            "latitude": ([ h3_dim_name ], dataset.ds['latitude'].values.squeeze(), {"units": "degrees_north"}),
        },
        coords={"time": rollout_times, h3_dim_name: dataset.ds[h3_dim_name].values},
        attrs={"title": f"{forecast_days}-Day Graph Attention Auto-Regressive Rollout Model Output"}
    )

    output_filename = "h3_autoregressive_forecast.nc"
    ds_rollout.to_netcdf(output_filename, format="NETCDF4")
    print(f"SUCCESS: Auto-regressive forecast rollout saved to: {output_filename}")

if __name__ == "__main__":
    run_autoregressive_rollout()
