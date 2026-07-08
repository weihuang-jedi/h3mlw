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

# OPTIMIZATION: Configure PyTorch to natively utilize NVIDIA H100 Tensor Cores
torch.set_float32_matmul_precision('medium')

# =====================================================================
# A. CONFIG-DRIVEN PYTORCH DATASET WITH SOLAR AND STATIC FEATURES
# =====================================================================
class GlobalH3WeatherDataset(Dataset):
    def __init__(self, nc_path, history_steps, forecast_offset):
        self.ds = xr.open_dataset(nc_path)
        
        # 1. Load and normalize time-series weather parameters (shape: time, nodes)
        raw_air = torch.tensor(self.ds['air_h3'].values, dtype=torch.float32)
        self.air_mean, self.air_std = raw_air.mean(), raw_air.std()
        self.norm_air = (raw_air - self.air_mean) / (self.air_std + 1e-6)
        
        # 2. Load static geographic features (shape: nodes, 2)
        raw_lsm = torch.tensor(self.ds['land_sea_mask'].values, dtype=torch.float32)
        raw_elv = torch.tensor(self.ds['elevation'].values, dtype=torch.float32)
        norm_elv = (raw_elv - raw_elv.mean()) / (raw_elv.std() + 1e-6)
        self.static_features = torch.stack([raw_lsm, norm_elv], dim=1) # (num_nodes, 2)
        
        # 3. Compute Solar Forcing variables dynamically across the timeline
        times = self.ds['time'].values
        lons = self.ds['longitude'].values
        
        # FIXED: Use pandas to safely decode the datetime index without relying on raw attributes
        datetime_index = pd.to_datetime(times)
        
        num_times = len(times)
        num_nodes = len(lons)
        
        self.solar_forcings = torch.zeros((num_times, num_nodes, 4), dtype=torch.float32)
        
        print("Pre-calculating cyclical solar forcing attributes for all nodes...")
        for t_idx, dt in enumerate(datetime_index):
            # Use pandas datetime attributes
            day_of_year = dt.dayofyear
            annual_phase = 2.0 * np.pi * day_of_year / 365.25
            ann_sin = np.sin(annual_phase)
            ann_cos = np.cos(annual_phase)
            
            utc_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            
            for n_idx, lon in enumerate(lons):
                local_solar_hour = (utc_hour + lon / 15.0) % 24.0
                diurnal_phase = 2.0 * np.pi * local_solar_hour / 24.0
                
                self.solar_forcings[t_idx, n_idx, 0] = np.sin(diurnal_phase)
                self.solar_forcings[t_idx, n_idx, 1] = np.cos(diurnal_phase)
                self.solar_forcings[t_idx, n_idx, 2] = ann_sin
                self.solar_forcings[t_idx, n_idx, 3] = ann_cos

        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        self.total_samples = len(self.norm_air) - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        history_end = idx + self.history_steps
        x_dynamic = self.norm_air[idx:history_end].t() # Shape: (num_nodes, history_steps)
        x_solar = self.solar_forcings[history_end - 1] # Shape: (num_nodes, 4)
        
        # Combine dynamic weather, fixed geographic fields, and solar metrics
        x_combined = torch.cat([x_dynamic, self.static_features, x_solar], dim=1)
        
        target_idx = history_end + self.forecast_offset - 1
        y = self.norm_air[target_idx] # Shape: (num_nodes,)
        
        return x_combined, y

# =====================================================================
# B. PYTORCH LIGHTNING DATAMODULE
# =====================================================================
class H3DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def setup(self, stage=None):
        full_dataset = GlobalH3WeatherDataset(
            nc_path=self.cfg['paths']['data_nc'],
            history_steps=self.cfg['model_params']['history_steps'],
            forecast_offset=self.cfg['model_params']['forecast_offset']
        )
        total = len(full_dataset)
        train_sz = int(total * 0.8)
        val_sz = int(total * 0.1)
        test_sz = total - train_sz - val_sz
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_sz, val_sz, test_sz]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg['training_params']['batch_size'], 
                          shuffle=True, num_workers=self.cfg['training_params']['num_workers'], 
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['training_params']['batch_size'], 
                          shuffle=False, num_workers=self.cfg['training_params']['num_workers'], 
                          pin_memory=True)

# =====================================================================
# C. SINGLE-GPU DYNAMIC GRAPH ATTENTION WEATHER NETWORK (GATv2)
# =====================================================================
class SingleGPUH3WeatherGAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        edge_index = torch.load(self.cfg['paths']['edge_index_pt'], map_location="cpu")
        self.register_buffer("edge_index", edge_index)
        
        in_channels = (self.cfg['model_params']['history_steps'] + 
                       self.cfg['model_params']['static_features'] + 
                       self.cfg['model_params']['solar_features'])
        
        h_channels = self.cfg['model_params']['hidden_channels']
        heads = self.cfg['model_params']['attention_heads']
        
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=h_channels, heads=heads, concat=True, dropout=0.05)
        self.relu = nn.ReLU()
        self.gat2 = GATv2Conv(in_channels=h_channels * heads, out_channels=h_channels, heads=heads, concat=False, dropout=0.05)
        self.linear = nn.Linear(h_channels, 1)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, num_nodes, features = x.shape
        x_flat = x.view(-1, features)
        
        num_edges = self.edge_index.shape[1]
        offsets = torch.arange(batch_size, device=x.device).repeat_interleave(num_edges) * num_nodes
        edge_index_batched = self.edge_index.repeat(1, batch_size) + offsets
        
        h = self.gat1(x_flat, edge_index_batched)
        h = self.relu(h)
        h = self.gat2(h, edge_index_batched)
        h = self.relu(h)
        
        out_flat = self.linear(h)
        return out_flat.view(batch_size, num_nodes)

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
        return torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-4)

# =====================================================================
# D. ENGINE RUNNER EXECUTION
# =====================================================================
def main():
    print("Reading configuration file...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    datamodule = H3DataModule(config)
    model = SingleGPUH3WeatherGAT(config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config['paths']['checkpoint_dir'],
        filename="best-h3-diurnal-gat-model",
        save_top_k=1,
        mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
        accelerator="gpu",
        devices=1,           
        num_nodes=1,
        strategy="auto",     
        callbacks=[checkpoint_callback],
        precision="16-mixed",     
        log_every_n_steps=10
    )
    
    print("Launching Single-GPU GATv2 Attention Training Model...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

