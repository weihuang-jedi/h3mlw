import os
import yaml
import argparse
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

torch.set_float32_matmul_precision('high')

# =====================================================================
# 1. OPTIMIZED HIGH-SPEED MULTI-YEAR DATASET (ZARR COMPLIANT)
# =====================================================================
class FastMultiYearH3Dataset(Dataset):
    def __init__(self, file_pattern, history_steps, forecast_offset):
        print(f"Opening cloud-optimized data store: {file_pattern}")
        if file_pattern.endswith('.zarr') or os.path.isdir(file_pattern):
            self.ds = xr.open_zarr(file_pattern, consolidated=True)
        else:
            self.ds = xr.open_mfdataset(file_pattern, combine="by_coords", data_vars="minimal", compat="override")

        print("Extracting weather tracking metrics directly to memory...")
        self.air_data = torch.tensor(self.ds['air_h3'].values, dtype=torch.float32)

        print("Calculating exact data standard deviation and mean parameters...")
        valid_mask = ~torch.isnan(self.air_data) & ~torch.isinf(self.air_data)
        self.air_mean = float(self.air_data[valid_mask].mean())
        self.air_std = float(self.air_data[valid_mask].std())
        print(f" -> Baseline Mean: {self.air_mean:.3f} K | Standard Deviation: {self.air_std:.3f} K")

        raw_lsm = torch.tensor(self.ds['land_sea_mask'].values.squeeze(), dtype=torch.float32) - 0.5
        raw_elv = torch.tensor(self.ds['elevation'].values.squeeze(), dtype=torch.float32)
        norm_elv = torch.nan_to_num((raw_elv - raw_elv.mean()) / (raw_elv.std() + 1e-6), nan=0.0)
        self.static_features = torch.stack([raw_lsm, norm_elv], dim=1) 

        self.times = self.ds['time'].values
        self.lons = self.ds['longitude'].values.squeeze()
        self.lats = self.ds['latitude'].values.squeeze() # Extracted for Latitude Weighting
        self.num_times = len(self.times)
        self.num_nodes = len(self.lons)

        self.history_steps = history_steps
        self.forecast_offset = forecast_offset
        self.total_samples = self.num_times - self.history_steps - self.forecast_offset + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        history_end = idx + self.history_steps
        raw_x_air = self.air_data[idx:history_end]
        raw_y_air = self.air_data[history_end + self.forecast_offset - 1]

        norm_x_air = torch.nan_to_num((raw_x_air - self.air_mean) / (self.air_std + 1e-6), nan=0.0).t()
        y = torch.nan_to_num((raw_y_air - self.air_mean) / (self.air_std + 1e-6), nan=0.0)

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
        return x_combined, y, idx # Pass index to help compute rolling time vectors inside training loops

class MultiYearH3DataModule(pl.LightningDataModule):
    def __init__(self, config, data_path):
        super().__init__()
        self.cfg = config
        self.data_path = data_path
        self.full_dataset = None

    def setup(self, stage=None):
        self.full_dataset = FastMultiYearH3Dataset(
            file_pattern=self.data_path,
            history_steps=self.cfg['model_params']['history_steps'],
            forecast_offset=self.cfg['model_params']['forecast_offset']
        )
        total = len(self.full_dataset)
        train_end = int(total * 0.85)
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
# 2. UPGRADED LIGHTNING MODULE WITH ADVANCED WEATHER FIXES
# =====================================================================
class SingleGPUH3WeatherGAT(pl.LightningModule):
    def __init__(self, config, lats=None, lons=None):
        super().__init__()
        # Explicitly save it to the class object right here
        self.cfg = config

        # Save to hparams container cleanly
        self.save_hyperparameters(ignore=['lats', 'lons'])

        # 1. LATITUDE LOSS WEIGHTING INITIALIZATION
        if lats is not None:
            weights = np.cos(np.radians(lats))
            weights /= weights.mean()  # Normalize weights around a baseline of 1.0
            self.register_buffer("spatial_loss_weights", torch.from_numpy(weights).float())
        else:
            self.register_buffer("spatial_loss_weights", None)

        if lons is not None:
            self.register_buffer("lons_deg", torch.from_numpy(lons).float())
        else:
            self.register_buffer("lons_deg", None)

        # Load Graph Topology
        edge_index = torch.load(self.cfg['paths']['edge_index_pt'], map_location="cpu")
        self.register_buffer("edge_index", edge_index)

        # Dimensions
        in_channels = 8  # (2 history + 2 static + 4 solar)
        latent_dim = config['model_params']['hidden_channels'] # e.g., 128

        # 1. ENCODER MLP: Compress raw inputs per-node into latent space
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # 2. PROCESSOR: Deep Graph Attention layers operating entirely in latent space
        self.gat1 = GATv2Conv(in_channels=latent_dim, out_channels=latent_dim, heads=4, concat=False)
        self.gat2 = GATv2Conv(in_channels=latent_dim, out_channels=latent_dim, heads=4, concat=False)

        # 3. DECODER MLP: Map latent space back down to physical weather variables (1 output: Temp)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Nodes, Features]
        batch_size, num_nodes, _ = x.shape

        # Step 1: Encode node states into latent vectors
        h = self.encoder(x) # Shape: [Batch, Nodes, Latent_Dim]

        # Flatten batch for PyG message passing
        h_flat = h.view(-1, h.shape[-1])

        # Step 2: Process spatial relationships across the graph index with inline batch math
        num_edges = self.edge_index.shape[1]
        local_edge_index = self.edge_index.to(x.device)
        offsets = torch.arange(batch_size, device=x.device).repeat_interleave(num_edges) * num_nodes
        edge_index_batched = local_edge_index.repeat(1, batch_size) + offsets

        # Apply GAT layers using torch.relu directly
        h_flat = torch.relu(self.gat1(h_flat, edge_index_batched))
        h_flat = torch.relu(self.gat2(h_flat, edge_index_batched))

        # Reshape back to batch format
        h = h_flat.view(batch_size, num_nodes, -1)

        # Step 3: Decode latent vectors back to single temperature variable predictions
        out = self.decoder(h) # Shape: [Batch, Nodes, 1]
        return out.squeeze(-1)

    def compute_weighted_loss(self, pred, target):
        """Calculates area-weighted MSE loss based on node latitude."""
        loss_matrix = (pred - target) ** 2
        if getattr(self, "spatial_loss_weights", None) is not None:
            # Broadcast weights across the batch axis
            loss_matrix = loss_matrix * self.spatial_loss_weights.unsqueeze(0)
        return loss_matrix.mean()

    def _generate_solar_for_step(self, base_idx, step_offset, batch_size, num_nodes, device):
        """Helper to generate solar arrays dynamically during multi-step unrolling."""
        hours_offset = (step_offset + 1) * 3
        x_solar = torch.zeros((batch_size, num_nodes, 4), device=device)
        return x_solar

    def training_step(self, batch, batch_idx):
        x, y_true_start, start_indices = batch
        batch_size, num_nodes, _ = x.shape
        history_steps = self.cfg['model_params']['history_steps']

        # Inject Gaussian Noise to prevent AR Explosions
        noise_scale = self.cfg['training_params'].get('noise_injection_scale', 0.02)
        if self.training and noise_scale > 0:
            noise = torch.randn_like(x[:, :, :history_steps]) * noise_scale
            x[:, :, :history_steps] = x[:, :, :history_steps] + noise

        # Multi-Step Rollout Training Horizon
        rollout_steps = self.cfg['training_params'].get('train_rollout_steps', 3)
        total_loss = 0.0

        current_history = x[:, :, :history_steps].clone()
        static_features = x[:, :, history_steps : history_steps+2].clone()

        for step in range(rollout_steps):
            x_solar = x[:, :, history_steps+2:].clone()

            x_input = torch.cat([current_history, static_features, x_solar], dim=2)
            y_hat = self(x_input)

            step_loss = self.compute_weighted_loss(y_hat, y_true_start)
            total_loss += step_loss

            # Autoregressive shift queue sequence
            updated_history = current_history[:, :, 1:]
            current_history = torch.cat([updated_history, y_hat.unsqueeze(2)], dim=2)

        loss = total_loss / rollout_steps
        self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.compute_weighted_loss(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.cfg['model_params']['learning_rate'],
            max_lr=self.cfg['model_params']['max_lr'],
            step_size_up=2000,
            mode='triangular2',
            cycle_momentum=False
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler_config]

# =====================================================================
# 3. CUSTOM STAGE & WALL-CLOCK TIMING PROGRESS LOGGER
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
# 4. ARGPARSE SCRIPT LAUNCH CONTROLLER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="NOAA EPIC-Style Vectorized GATv2 Weather Model Training.")
    parser.add_argument("-i", "--input", required=True, help="Path to input training data folder (.zarr) or NetCDF pattern")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to operational configuration YAML file")
    args = parser.parse_args()

    print(f"Reading configuration file: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    datamodule = MultiYearH3DataModule(config, data_path=args.input)
    datamodule.setup()

    # Extract coordinates directly from data layout layers to populate loss weighting metrics
    lats = datamodule.full_dataset.lats
    lons = datamodule.full_dataset.lons

    model = SingleGPUH3WeatherGAT(config, lats=lats, lons=lons)

    stage_tracker = ProgressStageTracker()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=config['paths']['checkpoint_dir'],
        filename="best-h3-20year-cyclic-model", save_top_k=1, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
        accelerator="gpu", devices=1,
        callbacks=[checkpoint_callback, stage_tracker],
        precision="32", log_every_n_steps=50, gradient_clip_val=0.3
    )

    print("Launching AI Pipeline Model Fit Execution Block...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
