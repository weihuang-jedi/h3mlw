import yaml
import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('high')

# =====================================================================
# A. MULTI-VARIABLE CONFIG-DRIVEN DATASET
# =====================================================================
class GlobalH3MultiVariableDataset(Dataset):
    def __init__(self, nc_path, history_steps, forecast_offset):
        self.ds = xr.open_dataset(nc_path)
        
        # 1. Dynamically load and norm multiple weather features independently
        self.feature_keys = ["air_h3", "uwnd_h3", "vwnd_h3", "rhum_h3"]
        num_features = len(self.feature_keys)
        num_times = len(self.ds.time)
        num_nodes = len(self.ds.h3_index)
        
        # Shape: (time, nodes, variables)
        self.weather_tensor = torch.zeros((num_times, num_nodes, num_features), dtype=torch.float32)
        self.means = {}
        self.stds = {}
        
        print("Loading and normalizing individual weather parameters...")
        for f_idx, key in enumerate(self.feature_keys):
            raw_matrix = torch.tensor(self.ds[key].values, dtype=torch.float32)
            # Filter NaNs or infs safely
            valid_mask = ~torch.isnan(raw_matrix) & ~torch.isinf(raw_matrix)
            f_mean = raw_matrix[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0)
            f_std = raw_matrix[valid_mask].std() if valid_mask.sum() > 0 else torch.tensor(1.0)
            
            self.means[key] = f_mean
            self.stds[key] = f_std
            
            norm_matrix = (raw_matrix - f_mean) / (f_std + 1e-6)
            self.weather_tensor[:, :, f_idx] = torch.nan_to_num(norm_matrix, nan=0.0)

        # 2. Static geographic attributes (shape: nodes, 2)
        raw_lsm = torch.tensor(self.ds['land_sea_mask'].values, dtype=torch.float32) - 0.5
        raw_elv = torch.tensor(self.ds['elevation'].values, dtype=torch.float32)
        norm_elv = torch.nan_to_num((raw_elv - raw_elv.mean()) / (raw_elv.std() + 1e-6), nan=0.0)
        self.static_features = torch.stack([raw_lsm, norm_elv], dim=1)
        
        # 3. Solar forcings (shape: time, nodes, 4)
        times = self.ds['time'].values
        lons = self.ds['longitude'].values
        datetime_index = pd.to_datetime(times)
        
        self.solar_forcings = torch.zeros((num_times, num_nodes, 4), dtype=torch.float32)
        print("Pre-calculating cyclical solar forcing attributes...")
        for t_idx, dt in enumerate(datetime_index):
            day_phase = 2.0 * np.pi * dt.dayofyear / 365.25
            utc_hour = dt.hour + dt.minute / 60.0
            
            for n_idx, lon in enumerate(lons):
                local_hour = (utc_hour + lon / 15.0) % 24.0
                diurnal_phase = 2.0 * np.pi * local_hour / 24.0
                
                self.solar_forcings[t_idx, n_idx, 0] = np.sin(diurnal_phase)
                self.solar_forcings[t_idx, n_idx, 1] = np.cos(diurnal_phase)
                self.solar_forcings[t_idx, n_idx, 2] = np.sin(day_phase)
                self.solar_forcings[t_idx, n_idx, 3] = np.cos(day_phase)

        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        self.total_samples = num_times - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        history_end = idx + self.history_steps
        
        # Extract dynamic history footprint: shape (history_steps, num_nodes, variables)
        x_dyn = self.weather_tensor[idx:history_end]
        # Reshape and unroll history to form features: shape (num_nodes, history_steps * variables)
        # Permute splits dimensions from (steps, nodes, vars) to (nodes, steps, vars)
        x_dyn = x_dyn.permute(1, 0, 2).reshape(len(self.ds.h3_index), -1)
        
        x_solar = self.solar_forcings[history_end - 1] # shape: (num_nodes, 4)
        
        # Combine all parameter vectors into a unified feature set
        # Combined size width: (nodes, history_steps * variables + static + solar)
        x_combined = torch.cat([x_dyn, self.static_features, x_solar], dim=1)
        
        target_idx = history_end + self.forecast_offset - 1
        y = self.weather_tensor[target_idx] # Target shape becomes multi-variable: (num_nodes, 4)
        
        return x_combined, y

# =====================================================================
# B. PYTORCH LIGHTNING DATAMODULE
# =====================================================================
class H3DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def setup(self, stage=None):
        full_dataset = GlobalH3MultiVariableDataset(
            nc_path=self.cfg['paths']['data_nc'],
            history_steps=self.cfg['model_params']['history_steps'],
            forecast_offset=self.cfg['model_params']['forecast_offset']
        )
        total = len(full_dataset)
        train_sz = int(total * 0.8)
        val_sz = int(total * 0.1)
        test_sz = total - train_sz - val_sz
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_sz, val_sz, test_sz])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg['training_params']['batch_size'], shuffle=True, num_workers=self.cfg['training_params']['num_workers'], pin_memory=True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['training_params']['batch_size'], shuffle=False, num_workers=self.cfg['training_params']['num_workers'], pin_memory=True)

# =====================================================================
# C. SINGLE-GPU MULTI-VARIABLE GRAPH ATTENTION WEATHER NETWORK
# =====================================================================
class SingleGPUMultiVarH3GAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        edge_index = torch.load(self.cfg['paths']['edge_index_pt'], map_location="cpu")
        self.register_buffer("edge_index", edge_index)
        
        # Calculate size parameters (2 history steps * 4 weather features + 2 static + 4 solar = 14)
        in_channels = (self.cfg['model_params']['history_steps'] * self.cfg['model_params']['weather_features'] + 
                       self.cfg['model_params']['static_features'] + 
                       self.cfg['model_params']['solar_features'])
        
        h_channels = self.cfg['model_params']['hidden_channels']
        heads = self.cfg['model_params']['attention_heads']
        out_features = self.cfg['model_params']['weather_features'] # Output predicts 4 channels
        
        self.feature_norm = nn.LayerNorm(in_channels)
        self.hidden_norm1 = nn.LayerNorm(h_channels * heads)
        self.hidden_norm2 = nn.LayerNorm(h_channels)
        
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=h_channels, heads=heads, concat=True, dropout=0.05, negative_slope=0.2)
        self.relu = nn.ReLU()
        self.gat2 = GATv2Conv(in_channels=h_channels * heads, out_channels=h_channels, heads=heads, concat=False, dropout=0.05, negative_slope=0.2)
        
        # Output Head: Now maps hidden dimensions to out_features channels (4 weather fields)
        self.linear = nn.Linear(h_channels, out_features)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, num_nodes, features = x.shape
        x_flat = x.view(-1, features)
        x_flat = self.feature_norm(x_flat)
        
        num_edges = self.edge_index.shape[1]
        local_edge_index = self.edge_index.to(x.device)
        
        offsets = torch.arange(batch_size, device=x.device).repeat_interleave(num_edges) * num_nodes
        edge_index_batched = local_edge_index.repeat(1, batch_size) + offsets
        
        with torch.amp.autocast('cuda', enabled=False):
            h = self.gat1(x_flat.float(), edge_index_batched)
            h = torch.clamp(h, min=-5.0, max=5.0)
            h = self.hidden_norm1(h)
            h = self.relu(h)
            
            h = self.gat2(h, edge_index_batched)
            h = torch.clamp(h, min=-5.0, max=5.0)
            h = self.hidden_norm2(h)
            h = self.relu(h)
            
            out_flat = self.linear(h)
            
        # Reshape output matrix to match batch shapes: (Batch_Size, Num_Nodes, 4)
        return out_flat.view(batch_size, num_nodes, -1).to(x.dtype)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-3)

def main():
    print("Reading configuration parameters...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    datamodule = H3DataModule(config)
    model = SingleGPUMultiVarH3GAT(config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=config['paths']['checkpoint_dir'],
        filename="best-h3-multi-variable-gat-model", save_top_k=1, mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
accelerator="gpu", devices=1, num_nodes=1, strategy="auto",callbacks=[checkpoint_callback], precision="32", log_every_n_steps=10,gradient_clip_val=0.3, gradient_clip_algorithm="norm")print("Launching Single-GPU Multi-Variable GATv2 Attention Forecast Engine...")

trainer.fit(model, datamodule=datamodule)if name == "main":main()


