import xarray as xr
import torch
from torch.utils.data import Dataset

class GlobalH3WeatherDataset(Dataset):
    def __init__(self, nc_path="global_h3_res2_air_all_times.nc", history_steps=2, forecast_offset=1):
        """
        Args:
            nc_path: Path to your generated time-series NetCDF.
            history_steps: Number of past time steps to feed into the model (e.g., 2 frames = 6 hours).
            forecast_offset: How many steps ahead to predict (e.g., 1 step = +3 hours).
        """
        # Open using xarray
        self.ds = xr.open_dataset(nc_path)
        # Pull values array directly into RAM as a float32 tensor
        # Shape: (2928 steps, 5882 cells)
        self.data = torch.tensor(self.ds['air_h3'].values, dtype=torch.float32)
        
        # Calculate mean and standard deviation for Z-score feature normalization
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.normalized_data = (self.data - self.mean) / (self.std + 1e-6)
        
        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        
        # Total valid sequences available to sample
        self.total_samples = len(self.data) - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Extract past sequence frames: shape (history_steps, num_nodes)
        history_end = idx + self.history_steps
        x = self.normalized_data[idx:history_end]
        
        # In GNN models, feature matrices expect shape: (num_nodes, features)
        # We transpose (history_steps, num_nodes) into (num_nodes, history_steps)
        x = x.t() 
        
        # Extract target variable slice to evaluate predictions against
        target_idx = history_end + self.forecast_offset - 1
        y = self.normalized_data[target_idx] # Shape: (num_nodes,)
        
        return x, y

