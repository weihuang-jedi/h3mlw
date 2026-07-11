import os
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
import pytorch_lightning as pl
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. GRAPHCAST LANDSCAPE INTERPOLATED DATA CONTAINER
# =====================================================================
class FastMultiYearIcosahedralDataset(Dataset):
    def __init__(self, zarr_path: str, history_steps: int = 2, rollout_steps: int = 1):
        print(f"[DATASET] Opening GraphCast-aligned input target stream: {zarr_path}")
        self.ds = xr.open_zarr(zarr_path, consolidated=True)
        self.history_steps = history_steps
        self.rollout_steps = rollout_steps
        self.total_window_size = history_steps + rollout_steps
        
        self.var_name = [k for k in self.ds.data_vars if 'mask' not in k and 'elevation' not in k and 'face' not in k][0]
        self.weather_data = self.ds[self.var_name]
        self.total_time_steps = len(self.ds.time)

        # Cache geographical context variables natively
        self.lsm = torch.from_numpy(self.ds['land_sea_mask'].values).float()
        self.elevation = torch.from_numpy(self.ds['elevation'].values).float()
        self.elevation = (self.elevation - self.elevation.mean()) / (self.elevation.std() + 1e-5)

    def __len__(self):
        return self.total_time_steps - self.total_window_size + 1

    def __getitem__(self, idx):
        window_slice = slice(idx, idx + self.total_window_size)
        
        # Pulls the data pointer into memory
        raw_window = self.weather_data.isel(time=window_slice).data

        # -----------------------------------------------------------------
        # FIXED: Wrap with torch.tensor or cast explicitly to np.array
        # -----------------------------------------------------------------
        x_weather = torch.tensor(np.array(raw_window[:self.history_steps]), dtype=torch.float32) # [history, nodes]
        y_weather = torch.tensor(np.array(raw_window[self.history_steps:]), dtype=torch.float32) # [rollout, nodes]

        x_weather = x_weather.permute(1, 0) # [nodes, history]

        # Combine historical frames with static land features
        static_features = torch.stack([self.lsm, self.elevation], dim=-1)
        x_features = torch.cat([x_weather, static_features], dim=-1)

        return x_features, y_weather


# =====================================================================
# 2. CORE FEATURE GRAPHCAST LAYER BLOCKS (INTERACTION MLPS)
# =====================================================================
class GraphCastInteractionLayer(nn.Module):
    """
    Implements GraphCast's signature message passing: 
    Updates node states by aggregating neighboring edge features computed via an MLP.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # Edge Feature Constructor MLP (combines source node, destination node, and current edge attributes)
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # Node Aggregation MLP (takes current node state and sum of incoming edge features)
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x_nodes, edge_index):
        src, dst = edge_index[0], edge_index[1]
        
        # Concatenate source and destination node features to build the raw edge attributes
        raw_edge_inputs = torch.cat([x_nodes[src], x_nodes[dst]], dim=-1)
        edge_messages = self.edge_mlp(raw_edge_inputs)
        
        # Aggregate edge messages at destination nodes via scatter-sum
        aggregated_messages = torch.zeros_like(x_nodes)
        aggregated_messages.index_add_(0, dst, edge_messages)
        
        # Update node features via residual connection
        node_inputs = torch.cat([x_nodes, aggregated_messages], dim=-1)
        x_nodes_out = x_nodes + self.node_mlp(node_inputs)
        return x_nodes_out


# =====================================================================
# 3. CLASSICAL ENCODER-PROCESSOR-DECODER (EPD) SYSTEM PIPELINE
# =====================================================================
class DeepGraphCastModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters()

        self.levels = config['model_params']['hierarchy_levels'] 
        self.fine_level = self.levels[0]
        self.root_level = self.levels[-1]
        graph_dir = config['paths']['graph_dir']

        # Load multi-mesh horizontal layer mixing structures
        for lvl in self.levels:
            path = os.path.join(graph_dir, f"edge_index_m{lvl}.pt")
            self.register_buffer(f"edge_index_m{lvl}", torch.load(path, map_location="cpu"))

        # Load cross-resolution vertical pooling configurations
        self.pooling_pairs = []
        for i in range(len(self.levels) - 1):
            f_lvl = self.levels[i]
            c_lvl = self.levels[i+1]
            path = os.path.join(graph_dir, f"map_m{c_lvl}_to_m{f_lvl}.pt")
            self.register_buffer(f"map_m{f_lvl}_to_m{c_lvl}", torch.load(path, map_location="cpu"))
            self.pooling_pairs.append((f_lvl, c_lvl))

        history_steps = config['model_params']['history_steps']
        in_channels = history_steps + 2  
        latent_dim = config['model_params']['latent_dim']

        # -------------------------------------------------------------
        # GRAPHCAST SUBSYSTEM A: ENCODER
        # -------------------------------------------------------------
        self.grid_encoder = nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # Bipartite Grid-to-Mesh mapping layers
        self.grid_to_mesh_poolers = nn.ModuleDict({
            f"pool_m{f}_to_m{c}": GraphCastInteractionLayer(latent_dim)
            for f, c in self.pooling_pairs
        })

        # -------------------------------------------------------------
        # GRAPHCAST SUBSYSTEM B: HOMOGENEOUS MULTI-MESH PROCESSOR CORE
        # -------------------------------------------------------------
        self.processor_stack = nn.ModuleList([
            GraphCastInteractionLayer(latent_dim)
            for _ in range(config['model_params']['processor_layers'])
        ])

        # -------------------------------------------------------------
        # GRAPHCAST SUBSYSTEM C: DECODER
        # -------------------------------------------------------------
        self.mesh_to_grid_unpoolers = nn.ModuleDict({
            f"unpool_m{c}_to_m{f}": GraphCastInteractionLayer(latent_dim)
            for f, c in self.pooling_pairs
        })
        self.grid_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1) # Outputs raw spatial weather residual steps (Delta)
        )

    def _batch_map(self, base_map, batch_size, n_fine, n_coarse):
        f_off = torch.arange(batch_size, device=base_map.device).repeat_interleave(base_map.shape[1]) * n_fine
        c_off = torch.arange(batch_size, device=base_map.device).repeat_interleave(base_map.shape[1]) * n_coarse
        b_map = base_map.repeat(1, batch_size)
        b_map[0] += f_off
        b_map[1] += c_off
        return b_map

    def forward(self, x, node_counts):
        batch_size = x.shape[0]
        x_flat = x.view(-1, x.shape[-1])
        latent_dim = self.cfg['model_params']['latent_dim']

        # [1.0] ENCODER PHASE: Project initial sparse features to latents
        latent_registry = {}
        latent_registry[self.fine_level] = self.grid_encoder(x_flat)

        # Pool features up the bipartite multi-mesh hierarchy
        for f_lvl, c_lvl in self.pooling_pairs:
            b_map = getattr(self, f"map_m{f_lvl}_to_m{c_lvl}")
            n_fine = node_counts[f_lvl]
            n_coarse = node_counts[c_lvl]
            batched_map = self._batch_map(b_map, batch_size, n_fine, n_coarse)

            # Initialize parent tensors with zeros before applying spatial updates
            h_coarse_init = torch.zeros((n_coarse * batch_size, latent_dim), device=x.device)
            # Combine tensors and perform message passing
            combined_tensor = torch.cat([latent_registry[f_lvl], h_coarse_init], dim=0)
            
            # Map indices across combined tensor bounds
            shifted_map = batched_map.clone()
            shifted_map[1] += (n_fine * batch_size)
            
            updated_block = self.grid_to_mesh_poolers[f"pool_m{f_lvl}_to_m{c_lvl}"](combined_tensor, shifted_map)
            latent_registry[c_lvl] = updated_block[(n_fine * batch_size):]

        # [2.0] PROCESSOR PHASE: Message passing at the coarsest mesh level (M0)
        h_root = latent_registry[self.root_level]
        root_edges = getattr(self, f"edge_index_m{self.root_level}")
        root_offsets = torch.arange(batch_size, device=x.device).repeat_interleave(root_edges.shape[1]) * node_counts[self.root_level]
        batched_root_edges = root_edges.repeat(1, batch_size) + root_offsets

        for block in self.processor_stack:
            h_root = block(h_root, batched_root_edges)
        latent_registry[self.root_level] = h_root

        # [3.0] DECODER PHASE: Unpool features back down to the finest grid
        for f_lvl, c_lvl in reversed(self.pooling_pairs):
            b_map = getattr(self, f"map_m{f_lvl}_to_m{c_lvl}")
            n_fine = node_counts[f_lvl]
            n_coarse = node_counts[c_lvl]
            batched_map = self._batch_map(b_map, batch_size, n_fine, n_coarse)
            
            # Project downward by reversing the bipartite mapping layout
            batched_unpool_map = torch.stack([batched_map[1], batched_map[0]], dim=0)
            combined_tensor = torch.cat([latent_registry[c_lvl], latent_registry[f_lvl]], dim=0)
            
            shifted_map = batched_unpool_map.clone()
            shifted_map[1] += (n_coarse * batch_size)
            
            updated_block = self.mesh_to_grid_unpoolers[f"unpool_m{c_lvl}_to_m{f_lvl}"](combined_tensor, shifted_map)
            latent_registry[f_lvl] = updated_block[(n_coarse * batch_size):]

        # Decode final node latents into the predicted weather step updates (Delta)
        out_flat = self.grid_decoder(latent_registry[self.fine_level])
        return out_flat.view(batch_size, node_counts[self.fine_level])

    def training_step(self, batch, batch_idx):
        x, y = batch  
        batch_size, num_nodes, _ = x.shape
        
        node_counts = {lvl: int(getattr(self, f"edge_index_m{lvl}").max() + 1) for lvl in self.levels}

        current_input = x.clone()
        total_loss = 0.0
        rollout_steps = y.shape[1]

        # Autoregressive sequence execution rollout loop
        for step in range(rollout_steps):
            pred_delta = self(current_input, node_counts)
            prev_field = current_input[:, :, self.cfg['model_params']['history_steps'] - 1]
            pred_field = prev_field + pred_delta  # Residual update matching the target physics state

            target_field = y[:, step, :]
            step_loss = nn.functional.mse_loss(pred_field, target_field)
            total_loss += step_loss

            if step < rollout_steps - 1:
                next_input = torch.zeros_like(current_input)
                next_input[:, :, :-2] = current_input[:, :, 1:-1]
                next_input[:, :, self.cfg['model_params']['history_steps'] - 1] = pred_field
                next_input[:, :, -2:] = current_input[:, :, -2:]
                current_input = next_input

        loss = total_loss / rollout_steps
        self.log("train_graphcast_mse", loss, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg['model_params']['learning_rate'], weight_decay=1e-5)


# =====================================================================
# 4. RUNTIME OBJECT CONTROL CONTROLLER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train GraphCast-style network architecture over icosahedral grids.")
    parser.add_argument("-c", "--config", default="config.yaml")
    # New argument to point to a checkpoint file
    parser.add_argument("-r", "--resume", default=None, help="Path to a checkpoint file (.ckpt) to resume training from")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize data pipeline components
    dataset = FastMultiYearIcosahedralDataset(
        zarr_path=config['paths']['zarr_store'],
        history_steps=config['model_params']['history_steps'],
        rollout_steps=config['training_params']['rollout_steps']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['training_params']['num_workers'],
        shuffle=True,
        pin_memory=True
    )

    # 1. Handle Model Instantiation/Compilation
    if args.resume is not None:
        print(f"[RESUME] Loading model weights from checkpoint: {args.resume}")
        # When resuming a compiled model, pass the raw instantiated architecture to torch.compile
        raw_model = DeepGraphCastModel.load_from_checkpoint(args.resume, config=config)
    else:
        print("[START] Initializing new model architecture from scratch...")
        raw_model = DeepGraphCastModel(config=config)

    # Run graph optimization for Ursa's CUDA environment
    optimized_model = torch.compile(raw_model, mode="reduce-overhead")

    # 2. Setup Progress Tracking Callback
    from lightning.pytorch.callbacks import TQDMProgressBar
    pbar = TQDMProgressBar(
        refresh_rate=config['training_params'].get('progress_bar_refresh_rate', 10)
    )

    trainer = pl.Trainer(
        max_epochs=config['training_params']['max_epochs'],
        accelerator="gpu",
        devices=1,
        precision=config['training_params'].get('precision', '16-mixed'),
        callbacks=[pbar]
    )

    # 3. Trigger Training Pipeline
    if args.resume is not None:
        print(f"[START] Resuming training loop at restored epoch marker from: {args.resume}")
        # Passing ckpt_path tells the trainer to restore optimizer state and historical epoch metrics
        trainer.fit(optimized_model, dataloader, ckpt_path=args.resume)
    else:
        print(f"[START] Triggering fresh model execution loop using setup file: {args.config}")
        trainer.fit(optimized_model, dataloader)

if __name__ == "__main__":
    main()
