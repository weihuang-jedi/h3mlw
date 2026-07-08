import yaml
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

# =====================================================================
# A. CONFIG-DRIVEN PYTORCH DATASET WITH STATIC FEATURES
# =====================================================================
class GlobalH3WeatherDataset(Dataset):
    def __init__(self, nc_path, history_steps, forecast_offset):
        self.ds = xr.open_dataset(nc_path)
        
        # 1. Load and normalize time-series weather parameters
        raw_air = torch.tensor(self.ds['air_h3'].values, dtype=torch.float32)
        self.air_mean, self.air_std = raw_air.mean(), raw_air.std()
        self.norm_air = (raw_air - self.air_mean) / (self.air_std + 1e-6)
        
        # 2. Load and normalize static geographic features
        raw_lsm = torch.tensor(self.ds['land_sea_mask'].values, dtype=torch.float32)
        raw_elv = torch.tensor(self.ds['elevation'].values, dtype=torch.float32)
        
        # Normalize elevation metrics to protect optimization gradients
        self.elv_mean, self.elv_std = raw_elv.mean(), raw_elv.std()
        norm_elv = (raw_elv - self.elv_mean) / (self.elv_std + 1e-6)
        
        # Stack static features into a unified matrix layout of shape: (num_nodes, 2)
        # Land-sea mask does not require normalization since it is already binary (0 or 1)
        self.static_features = torch.stack([raw_lsm, norm_elv], dim=1)
        
        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        self.total_samples = len(self.norm_air) - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Slice historical frames: shape (history_steps, num_nodes)
        history_end = idx + self.history_steps
        x_dynamic = self.norm_air[idx:history_end].t() # Transpose to shape: (num_nodes, history_steps)
        
        # Concatenate time-series features and static features along the tracking axis
        # Resulting tensor shape: (num_nodes, history_steps + static_features)
        x_combined = torch.cat([x_dynamic, self.static_features], dim=1)
        
        target_idx = history_end + self.forecast_offset - 1
        y = self.norm_air[target_idx] # Target shape: (num_nodes,)
        
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
# C. DISTRIBUTED GEOGRAPHY-AWARE GATv2 NETWORK MODEL
# =====================================================================
class DistributedH3WeatherGAT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        
        edge_index = torch.load(self.cfg['paths']['edge_index_pt'], map_location="cpu")
        self.register_buffer("edge_index", edge_index)
        
        # Calculate combined total input channel width: (history_steps + static_features)
        in_channels = self.cfg['model_params']['history_steps'] + self.cfg['model_params']['static_features']
        h_channels = self.cfg['model_params']['hidden_channels']
        heads = self.cfg['model_params']['attention_heads']
        
        # The GATv2 layers will now weight messages based on both terrain and weather trends
        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=h_channels, heads=heads, concat=True, dropout=0.05)
        self.relu = nn.ReLU()
        self.gat2 = GATv2Conv(in_channels=h_channels * heads, out_channels=h_channels, heads=heads, concat=False, dropout=0.05)
        self.linear = nn.Linear(h_channels, 1)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, num_nodes, features = x.shape
        x_flat = x.view(-1, features)
        
        offsets = torch.arange(batch_size, device=x.device).repeat_interleave(self.edge_index.shape) * num_nodes
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-4)

# =====================================================================
# D. ENGINE EXECUTION RUNNER
# =====================================================================
def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    datamodule = H3DataModule(config)
    model = DistributedH3WeatherGAT(config)
    
    ddp_strategy = DDPStrategy(
        process_group_backend="nccl", 
        find_unused_parameters=False, 
        static_graph=True             
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config['paths']['checkpoint_dir'],
        filename="best-h3-gat-geography-model",
        save_top_k=1,
        mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
        accelerator="gpu",
        devices="auto",           
        num_nodes=config['training_params']['num_nodes'],
        strategy=ddp_strategy,     
        callbacks=[checkpoint_callback],
        precision="16-mixed",     
        log_every_n_steps=10
    )
    
    print("Launching Distributed Geography-Aware GATv2 Training Engine...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

