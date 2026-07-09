import os
import yaml
import xarray as xr
import pandas as pd
import numpy as np
import torch
from train_single_gpu import SingleGPUH3WeatherGAT, FastMultiYearH3Dataset

class H3WeatherForecaster:
    def __init__(self, config_path="forecast.yaml"):
        """Initializes the forecaster module and loads system configurations."""
        print(f"Loading forecasting configuration from: {config_path}")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset = None
        
    def load_model(self):
        """Loads the trained weights from the specified PyTorch Lightning checkpoint."""
        ckpt_path = self.cfg['paths']['checkpoint_path']
        print(f"Loading trained weights from checkpoint: {ckpt_path}...")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
            
        # self.model = SingleGPUH3WeatherGAT.load_from_checkpoint(ckpt_path, map_location="cpu")
        # Pass strict=False to ignore the new training-only buffers safely
        self.model = SingleGPUH3WeatherGAT.load_from_checkpoint(ckpt_path, map_location="cpu", strict=False)
        self.model.eval()  # Freeze layers and deactivate dropout for inference
        self.model.to(self.device)
        print("Model successfully transferred to target hardware device.")

    def initialize_dataset(self):
        """Loads data indexes and computes required statistical parameters via FastMultiYearH3Dataset."""
        print("Initializing dataset structures...")
        data_path = self.cfg['paths']['data_pattern']
        
        # Instantiate our fast multi-year dataset to acquire the normalized features & stats
        self.dataset = FastMultiYearH3Dataset(
            file_pattern=data_path,
            history_steps=self.cfg['model_params']['history_steps'],
            forecast_offset=self.cfg['model_params']['forecast_offset']
        )

    def _generate_vectorized_solar_forcings(self, target_times):
        """Computes cyclical solar forcing matrix arrays optimized for performance."""
        print("Vectorizing cyclical solar forcing matrix arrays across the rollout timeline...")
        datetime_index = pd.to_datetime(target_times)
        num_times = len(target_times)
        num_nodes = self.dataset.num_nodes
        lons = self.dataset.lons

        day_of_year = datetime_index.dayofyear.values[:, np.newaxis]  # (Time, 1)
        utc_hour = (datetime_index.hour + datetime_index.minute / 60.0).values[:, np.newaxis]  # (Time, 1)
        lon_grid = lons[np.newaxis, :]  # (1, Nodes)

        annual_phase = 2.0 * np.pi * day_of_year / 365.25
        local_solar_hour = (utc_hour + lon_grid / 15.0) % 24.0
        diurnal_phase = 2.0 * np.pi * local_solar_hour / 24.0

        solar_forcings = torch.zeros((num_times, num_nodes, 4), dtype=torch.float32)
        solar_forcings[:, :, 0] = torch.from_numpy(np.sin(diurnal_phase))
        solar_forcings[:, :, 1] = torch.from_numpy(np.cos(diurnal_phase))
        solar_forcings[:, :, 2] = torch.from_numpy(np.sin(annual_phase)).repeat(1, num_nodes)
        solar_forcings[:, :, 3] = torch.from_numpy(np.cos(annual_phase)).repeat(1, num_nodes)
        return solar_forcings

    def run_rollout(self):
        if self.model is None or self.dataset is None:
            self.load_model()
            self.initialize_dataset()

        start_time_str = self.cfg['forecast_settings']['forecast_start_time']
        forecast_days = self.cfg['forecast_settings']['forecast_days']
        steps_per_day = self.cfg['model_params']['steps_per_day']
        history_steps = self.cfg['model_params']['history_steps']
        total_rollout_steps = forecast_days * steps_per_day
        
        target_timestamp = pd.to_datetime(start_time_str)

        init_file = self.cfg['paths'].get('initial_condition_nc')
        if init_file and os.path.exists(init_file):
            print(f"Loading external operational initial conditions from: {init_file}")
            ds_init = xr.open_dataset(init_file)
            
            # CRITICAL FIX: Ensure xarray time values are parsed cleanly as a pandas DatetimeIndex
            time_series = pd.to_datetime(ds_init.time.values)
            
            if target_timestamp not in time_series:
                raise KeyError(f"Requested start time {start_time_str} not found in initial condition file.")
                
            start_idx = np.where(time_series == target_timestamp)[0][0]
            print(f" -> Found target timestamp at index offset: {start_idx}")
            
            # Extract warm-up states from operational observations file
            raw_history = torch.tensor(ds_init['air_h3'].values[start_idx : start_idx + history_steps], dtype=torch.float32)
        else:
            print(f"Falling back to historical Zarr store index search for: {start_time_str}")
            time_series = pd.to_datetime(self.dataset.times)
            if target_timestamp not in time_series:
                raise KeyError(f"Requested start time {start_time_str} is not present within historical dataset timelines.")
            start_idx = np.where(time_series == target_timestamp)[0][0]
            raw_history = self.dataset.air_data[start_idx : start_idx + history_steps]

        print("Normalizing history context states...")
        norm_history = torch.nan_to_num((raw_history - self.dataset.air_mean) / (self.dataset.air_std + 1e-6), nan=0.0).t()
        current_history = norm_history.to(self.device)
        static_features = self.dataset.static_features.to(self.device)

        print("Constructing dynamic rolling future timelines...")
        rollout_time_window = pd.date_range(start=target_timestamp + pd.Timedelta(hours=3), 
                                             periods=total_rollout_steps, 
                                             freq='3h').values
        
        full_timeline_slice = pd.date_range(start=target_timestamp, 
                                             periods=history_steps + total_rollout_steps, 
                                             freq='3h').values
                                             
        # This is where your script was pausing:
        solar_forcings = self._generate_vectorized_solar_forcings(full_timeline_slice)
        print(f"Solar array generated successfully. Shape: {solar_forcings.shape}")

        print(f"Starting auto-regressive forecast rollout ({total_rollout_steps} sequential steps)...")
        predictions_history = torch.zeros((total_rollout_steps, self.dataset.num_nodes), dtype=torch.float32)

        # 4. Interactive evaluation loop execution block
        print("Entering GPU forward pass loop...")
        with torch.no_grad():
            for step in range(total_rollout_steps):
                current_solar_idx = history_steps + step
                x_solar = solar_forcings[current_solar_idx].to(self.device)

                # Force inputs to float32 to prevent Tensor Core conflicts
                x_input = torch.cat([current_history, static_features, x_solar], dim=1).float()
                x_input = x_input.unsqueeze(0)  

                # Run model forward pass
                y_hat = self.model(x_input).squeeze(0)
                predictions_history[step] = y_hat.cpu()

                # Execute Autoregressive Queue Shift
                updated_history = current_history[:, 1:]
                current_history = torch.cat([updated_history, y_hat.unsqueeze(1)], dim=1)

                print(f" -> Processed forecast step {step+1}/{total_rollout_steps}")
                if (step + 1) % steps_per_day == 0:
                    print(f"[PROGRESS] Completed Forecast Day {(step + 1) // steps_per_day}/{forecast_days}")

        # 5. Export compliant data
        self._export_to_netcdf(predictions_history, rollout_time_window)

    def _export_to_netcdf(self, predictions_history, rollout_times):
        """Denormalizes tensors back into Kelvin and writes out a clean NetCDF file."""
        print("Denormalizing generated data matrix back to Kelvin scales...")
        final_predictions_kelvin = (predictions_history * self.dataset.air_std) + self.dataset.air_mean
        output_filename = self.cfg['paths']['output_nc']

        print(f"Structuring rollout output file: {output_filename}...")
        h3_dim_name = 'h3_index' if 'h3_index' in self.dataset.ds.dims else 'nodes'
        
        ds_rollout = xr.Dataset(
            data_vars={
                "air_forecast": ([ "time", h3_dim_name ], final_predictions_kelvin.numpy(), {"units": "degK"}),
                "longitude": ([ h3_dim_name ], self.dataset.ds['longitude'].values.squeeze(), {"units": "degrees_east"}),
                "latitude": ([ h3_dim_name ], self.dataset.ds['latitude'].values.squeeze(), {"units": "degrees_north"}),
            },
            coords={"time": rollout_times, h3_dim_name: self.dataset.ds[h3_dim_name].values},
            attrs={"title": "Graph Attention Auto-Regressive Class Module Rollout Output"}
        )

        ds_rollout.to_netcdf(output_filename, format="NETCDF4")
        print(f"SUCCESS: Auto-regressive forecast rollout saved to: {output_filename}")

if __name__ == "__main__":
    # Provides default backward-compatibility CLI access
    forecaster = H3WeatherForecaster(config_path="forecast.yaml")
    forecaster.run_rollout()

# To use in other script:
# from rollout_forecast import H3WeatherForecaster

#     forecaster = H3WeatherForecaster(config_path="forecast.yaml")
#     forecaster.run_rollout()
