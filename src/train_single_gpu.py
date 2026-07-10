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
        self.lats = self.ds['latitude'].values.squeeze() 
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
        return x_combined, y, idx

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
# 2. GRAPHCAST/AIFS INSPIRED DEEP RESIDUAL PROCESSOR BLOCK
# =====================================================================
class ResidualProcessorBlock(nn.Module):
    def __init__(self, latent_dim, heads=4):
        super().__init__()
        # Stable deep message passing with multi-head spatial attention
        self.gat = GATv2Conv(in_channels=latent_dim, out_channels=latent_dim, heads=heads, concat=False)
        self.norm = nn.LayerNorm(latent_dim)
        self.act = nn.SiLU() # SiLU (Swish) is standard in GraphCast/AIFS architectures

    def forward(self, x, edge_index):
        residual = x
        x = self.gat(x, edge_index)
        x = self.act(x)
        return self.norm(x + residual) # Clean residual addition stabilizes gradient flow


# =====================================================================
# UPGRADED 3-TIER DEEP HIERARCHICAL WEATHER MODEL PIPELINE
# =====================================================================
class SingleGPUH3WeatherGAT(pl.LightningModule):
    def __init__(self, config, lats=None, lons=None):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(ignore=['lats', 'lons'])

        # Spatial Loss Area Weighting Tracker
        if lats is not None:
            weights = np.cos(np.radians(lats))
            weights /= weights.mean()
            self.register_buffer("spatial_loss_weights", torch.from_numpy(weights).float())
        else:
            self.register_buffer("spatial_loss_weights", None)

        # Load 3-Tier Multi-Resolution Topologies
        self.register_buffer("edge_res2", torch.load(config['paths']['edge_res2'], map_location="cpu"))
        self.register_buffer("edge_res1", torch.load(config['paths']['edge_res1'], map_location="cpu"))
        self.register_buffer("edge_res0", torch.load(config['paths']['edge_res0'], map_location="cpu"))
        self.register_buffer("map_r2_to_r1", torch.load(config['paths']['map_r2_to_r1'], map_location="cpu"))
        self.register_buffer("map_r1_to_r0", torch.load(config['paths']['map_r1_to_r0'], map_location="cpu"))

        # Channel Parameter Calculations
        history_steps = config['model_params']['history_steps']
        in_channels = history_steps + 2 + 4  # Weather History + Static Topography + Solar Radiation
        latent_dim = config['model_params']['hidden_channels']
        attn_heads = config['model_params'].get('attention_heads', 4)

        # 1. Base Encoder Layers
        self.fine_encoder = nn.Linear(in_channels, latent_dim)
        
        # 2. Inter-Resolution Aggregation Networks (Pooling Layers)
        self.pool_r2_to_r1 = GATv2Conv(in_channels=(latent_dim, latent_dim), out_channels=latent_dim, heads=attn_heads, concat=False)
        self.pool_r1_to_r0 = GATv2Conv(in_channels=(latent_dim, latent_dim), out_channels=latent_dim, heads=attn_heads, concat=False)

        # 3. Macro Fluid Processor Core (Deep Residual Blocks at Global Root Resolution 0)
        self.processor_stack = nn.ModuleList([
            ResidualProcessorBlock(latent_dim=latent_dim, heads=attn_heads)
            for _ in range(config['training_params'].get('processor_layers', 4))
        ])

        # 4. Inter-Resolution Expansion Networks (Unpooling Layers)
        self.unpool_r0_to_r1 = GATv2Conv(in_channels=(latent_dim, latent_dim), out_channels=latent_dim, heads=attn_heads, concat=False)
        self.unpool_r1_to_res2 = GATv2Conv(in_channels=(latent_dim, latent_dim), out_channels=latent_dim, heads=attn_heads, concat=False)

        # 5. Base Decoder Output Layer
        self.fine_decoder = nn.Linear(latent_dim, 1)

    def _batch_bipartite_edges(self, base_map, batch_size, num_fine, num_coarse):
        """Vectorizes index tracking structures across variable batch configurations."""
        fine_offsets = torch.arange(batch_size, device=base_map.device).repeat_interleave(base_map.shape[1]) * num_fine
        coarse_offsets = torch.arange(batch_size, device=base_map.device).repeat_interleave(base_map.shape[1]) * num_coarse
        batched_map = base_map.repeat(1, batch_size)
        batched_map[0] += fine_offsets
        batched_map[1] += coarse_offsets
        return batched_map

    def forward(self, x):
        batch_size, num_nodes_r2, _ = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # Dynamic deduction of structural shape variables across layers
        num_nodes_r1 = int(self.map_r2_to_r1[1].max() + 1)
        num_nodes_r0 = int(self.map_r1_to_r0[1].max() + 1)

        # Step 1: Initialize local latent vectors on Fine Grid (Res 2)
        h_r2 = torch.relu(self.fine_encoder(x_flat))

        # Vectorize multi-tier inter-resolution bipartite edge collections
        map_b1 = self._batch_bipartite_edges(self.map_r2_to_r1, batch_size, num_nodes_r2, num_nodes_r1)
        map_b2 = self._batch_bipartite_edges(self.map_r1_to_r0, batch_size, num_nodes_r1, num_nodes_r0)

        # Step 2: Spatial Pooling Phase up the hierarchy
        h_r1_init = torch.zeros((num_nodes_r1 * batch_size, h_r2.shape[-1]), device=x.device)
        h_r1 = torch.relu(self.pool_r2_to_r1((h_r2, h_r1_init), map_b1))

        h_r0_init = torch.zeros((num_nodes_r0 * batch_size, h_r2.shape[-1]), device=x.device)
        h_r0 = torch.relu(self.pool_r1_to_r0((h_r1, h_r0_init), map_b2))

        # Step 3: Deep Residual Processor Core message-passing on Root Grid (Res 0)
        r0_edges = self.edge_res0.shape[1]
        r0_offsets = torch.arange(batch_size, device=x.device).repeat_interleave(r0_edges) * num_nodes_r0
        batched_r0_edges = self.edge_res0.repeat(1, batch_size) + r0_offsets

        for block in self.processor_stack:
            h_r0 = block(h_r0, batched_r0_edges)

        # Step 4: Spatial Unpooling Phase down the hierarchy (reversing bipartite edges)
        map_unb2 = torch.stack([map_b2[1], map_b2[0]], dim=0)
        h_r1_reconstructed = torch.relu(self.unpool_r0_to_r1((h_r0, h_r1), map_unb2))

        map_unb1 = torch.stack([map_b1[1], map_b1[0]], dim=0)
        h_r2_reconstructed = torch.relu(self.unpool_r1_to_res2((h_r1_reconstructed, h_r2), map_unb1))

        # Step 5: Map decoded states back to physical values
        out_flat = self.fine_decoder(h_r2_reconstructed)
        return out_flat.view(batch_size, num_nodes_r2)

    def compute_weighted_loss(self, pred, target):
        loss_matrix = (pred - target) ** 2
        if getattr(self, "spatial_loss_weights", None) is not None:
            loss_matrix = loss_matrix * self.spatial_loss_weights.unsqueeze(0)
        return loss_matrix.mean()

    def training_step(self, batch, batch_idx):
        x, y_true_start, _ = batch
        batch_size, num_nodes, _ = x.shape
        history_steps = self.cfg['model_params']['history_steps']

        # Inject Gaussian Noise to mitigate Auto-Regressive error compounding
        noise_scale = self.cfg['training_params'].get('noise_injection_scale', 0.02)
        if self.training and noise_scale > 0:
            noise = torch.randn_like(x[:, :, :history_steps]) * noise_scale
            x[:, :, :history_steps] = x[:, :, :history_steps] + noise

        # Multi-Step Rollout Training Horizon Loop
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
# 4. CUSTOM STAGE & WALL-CLOCK TIMING PROGRESS LOGGER
# =====================================================================
class ProgressStageTracker(Callback):
    def __init__(self):
        super().__init__()
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
# 5. ARGPARSE SCRIPT LAUNCH CONTROLLER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Operational Hierarchical H3 Graph Neural Weather Model.")
    parser.add_argument("-i", "--input", required=True, help="Path to input training data (.zarr)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    print(f"Reading configuration file: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    datamodule = MultiYearH3DataModule(config, data_path=args.input)
    datamodule.setup()

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

    # DEFINE THE PATH TO YOUR SAVED CHECKPOINT
    # For example, loading the best model saved by the ModelCheckpoint callback
    checkpoint_to_resume = "checkpoints/best-h3-20year-cyclic-model.ckpt"

    if os.path.exists(checkpoint_to_resume):
        print(f"FOUND CHECKPOINT! Resuming training from: {checkpoint_to_resume}")
        n = 1
        new_checkpoint_to_resume = f"checkpoints/best-h3-20year-cyclic-model.ckpt.{n}"
        while os.path.exists(new_checkpoint_to_resume):
            n += 1
            new_checkpoint_to_resume = f"checkpoints/best-h3-20year-cyclic-model.ckpt.{n}"
        os.rename(checkpoint_to_resume, new_checkpoint_to_resume)
        # Pass the path to ckpt_path to restore optimizers, schedulers, and epoch counters
        trainer.fit(model, datamodule=datamodule, ckpt_path=new_checkpoint_to_resume)
    else:
        print("No checkpoint found. Launching a brand fresh training loop...")
        trainer.fit(model, datamodule=datamodule)

    # print("Launching AI Pipeline Model Fit Execution Block...")
    # trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()

