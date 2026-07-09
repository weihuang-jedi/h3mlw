import yaml
import xarray as xr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from torch_geometric.nn import GATv2Conv
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

# OPTIMIZATION: Configure PyTorch to natively utilize NVIDIA H100 Tensor Cores
torch.set_float32_matmul_precision('high')

# =====================================================================
# 1. OPTIMIZED HIGH-SPEED MULTI-YEAR DATASET
# =====================================================================
# =====================================================================
# 1. OPTIMIZED HIGH-SPEED MULTI-YEAR DATASET (ZARR VERSION)
# =====================================================================
class FastMultiYearH3Dataset(Dataset):
    def __init__(self, file_pattern, history_steps, forecast_offset):
        # NOTE: file_pattern can now just be the path to your zarr directory
        # e.g., "../data/global_h3_20years.zarr"
        print(f"Opening cloud-optimized Zarr store: {file_pattern}")

        # Open the consolidated Zarr archive
        self.ds = xr.open_zarr(file_pattern, consolidated=True)

        print("Extracting weather tracking metrics directly to high-speed memory arrays...")
        # Since Zarr reads fast, pulling .values into a Torch Tensor is clean
        self.air_data = torch.tensor(self.ds['air_h3'].values, dtype=torch.float32)

        print("Calculating exact data standard deviation and mean parameters...")
        valid_mask = ~torch.isnan(self.air_data) & ~torch.isinf(self.air_data)
        self.air_mean = float(self.air_data[valid_mask].mean())
        self.air_std = float(self.air_data[valid_mask].std())
        print(f" -> Baseline Mean: {self.air_mean:.3f} K | Standard Deviation: {self.air_std:.3f} K")

        # Load stable fixed static parameter variables (already flattened via previous error fixes)
        raw_lsm = torch.tensor(self.ds['land_sea_mask'].values.squeeze(), dtype=torch.float32) - 0.5
        raw_elv = torch.tensor(self.ds['elevation'].values.squeeze(), dtype=torch.float32)
        norm_elv = torch.nan_to_num((raw_elv - raw_elv.mean()) / (raw_elv.std() + 1e-6), nan=0.0)
        self.static_features = torch.stack([raw_lsm, norm_elv], dim=1)

        # Pull timeline parameters
        self.times = self.ds['time'].values
        self.lons = self.ds['longitude'].values[0, :] if 'time' in self.ds['longitude'].dims else self.ds['longitude'].values
        self.num_times = len(self.times)
        self.num_nodes = len(self.lons)

        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        self.total_samples = self.num_times - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        history_end = idx + self.history_steps

        # Extract features instantly from RAM without disk I/O latency
        raw_x_air = self.air_data[idx:history_end]
        raw_y_air = self.air_data[history_end + self.forecast_offset - 1]

        # Normalize fields
        norm_x_air = torch.nan_to_num((raw_x_air - self.air_mean) / (self.air_std + 1e-6), nan=0.0).t()
        y = torch.nan_to_num((raw_y_air - self.air_mean) / (self.air_std + 1e-6), nan=0.0)

        # Compute Solar Encodings on-the-fly via numpy/torch math
        dt = pd.to_datetime(self.times[history_end - 1])
        day_phase = 2.0 * np.pi * dt.dayofyear / 365.25
        utc_hour = dt.hour + dt.minute / 60.0

        local_hours = (utc_hour + self.lons / 15.0) % 24.0
        diurnal_phases = 2.0 * np.pi * local_hours / 24.0

        x_solar = torch.zeros((self.num_nodes, 4), dtype=torch.float32)
        x_solar[:, 0] = torch.from_numpy(np.sin(diurnal_phases))
        x_solar[:, 1] = torch.from_numpy(np.cos(diurnal_phases))
        x_solar[:, 2] = np.sin(day_phase)
        x_solar[:, 3] = np.cos(day_phase)

        x_combined = torch.cat([norm_x_air, self.static_features, x_solar], dim=1)
        return x_combined, y

# =====================================================================
# 2. SEQUENTIAL COMPLIANT LIGHTNING DATAMODULE
# =====================================================================
class MultiYearH3DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.full_dataset = FastMultiYearH3Dataset(
            file_pattern=self.cfg['paths']['data_pattern'],
            history_steps=self.cfg['model_params']['history_steps'],
            forecast_offset=self.cfg['model_params']['forecast_offset']
        )
        total = len(self.full_dataset)
        train_end = int(total * 0.85) # Use 17 years for training, 3 for validation

        self.train_dataset = torch.utils.data.Subset(self.full_dataset, range(0, train_end))
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, range(train_end, total))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg['training_params']['batch_size'],
                          shuffle=True, num_workers=self.cfg['training_params']['num_workers'],
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg['training_params']['batch_size'],
                          shuffle=False, num_workers=self.cfg['training_params']['num_workers'],
                          pin_memory=True)

# =====================================================================
# 3. GRAPH ATTENTION WEATHER NETWORK WITH CYCLIC LEARNING RATE
# =====================================================================
class SingleGPUH3WeatherGAT(pl.LightningModule):
    def __init__(self, config, total_train_steps=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.total_train_steps = total_train_steps

        edge_index = torch.load(self.cfg['paths']['edge_index_pt'], map_location="cpu")
        self.register_buffer("edge_index", edge_index)

        in_channels = (self.cfg['model_params']['history_steps'] +
                       self.cfg['model_params']['static_features'] +
                       self.cfg['model_params']['solar_features'])

        h_channels = self.cfg['model_params']['hidden_channels']
        heads = self.cfg['model_params']['attention_heads']

        self.feature_norm = nn.LayerNorm(in_channels)
        self.hidden_norm1 = nn.LayerNorm(h_channels * heads)
        self.hidden_norm2 = nn.LayerNorm(h_channels)

        self.gat1 = GATv2Conv(in_channels=in_channels, out_channels=h_channels, heads=heads, concat=True, dropout=0.05, negative_slope=0.2)
        self.relu = nn.ReLU()
        self.gat2 = GATv2Conv(in_channels=h_channels * heads, out_channels=h_channels, heads=heads, concat=False, dropout=0.05, negative_slope=0.2)
        self.linear = nn.Linear(h_channels, 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, num_nodes, features = x.shape
        x_flat = x.view(-1, features)
        x_flat = self.feature_norm(x_flat)

        num_edges = self.edge_index.shape[1]  # or self.edge_index.size(1)
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

        return out_flat.view(batch_size, num_nodes).to(x.dtype)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-3)

        # IMPLEMENTATION: Cyclical Learning Rate Scheduler
        # It smoothly scales the learning rate from the base configuration up to the maximum over the step window.
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.cfg['model_params']['learning_rate'],
            max_lr=self.cfg['model_params']['max_lr'],
            step_size_up=2000, # Number of steps to ramp up the learning rate
            mode='triangular2',
            cycle_momentum=False
        )

        # Configure Lightning to update the learning rate scheduler step-by-step
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler_config]

# =====================================================================
# 4. CUSTOM STAGE & WALL-CLOCK TIMING PROGRESS LOGGER
# =====================================================================
class ProgressStageTracker(Callback):
    """
    A validation callback that prints step processing speeds and elapsed times.
    Use this to accurately calculate your Slurm wall-time clock requirements.
    """
    def __init__(self):
        super().__init__()  # <--- Added the missing underscores
        self.epoch_start_time = 0
        self.batch_start_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        print(f"\n[STAGE PROGRESS] Beginning Epoch {trainer.current_epoch} loops...")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % 500 == 0:
            self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx > 0 and batch_idx % 500 == 0:
            elapsed = time.time() - self.batch_start_time
            steps_per_sec = 500.0 / elapsed
            print(f" -> Processed steps: {batch_idx} | Speed: {steps_per_sec:.2f} steps/sec")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"[STAGE COMPLETE] Epoch {trainer.current_epoch} finished in {epoch_duration/60.0:.2f} minutes.")
        print(f" -> Projected time required for 5 epochs: {(epoch_duration * 5) / 3600.0:.2f} hours.\n")

# =====================================================================
# 5. EXECUTION ENTRY BLOCK
# =====================================================================
def main():
    print("Reading configuration file...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    datamodule = MultiYearH3DataModule(config)

    # Run data layout discovery early to determine total batch steps for scheduler configurations
    datamodule.setup()

    model = SingleGPUH3WeatherGAT(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config['paths']['checkpoint_dir'],
        filename="best-h3-20year-cyclic-model",
        save_top_k=1,
        mode="min"
        )

    # Initialize the timing callback
    stage_tracker = ProgressStageTracker()

    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="auto",
        callbacks=[checkpoint_callback, stage_tracker],
        precision="32",
        log_every_n_steps=50,
        gradient_clip_val=0.3,
        gradient_clip_algorithm="norm"
        )

    print("Launching Vectorized Single-GPU GATv2 20-Year Climate Training Loop...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

