import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

class DistributedH3WeatherGCN(pl.LightningModule):
    def __init__(self, history_steps=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        
        # 1. Load the adjacency tensor setup
        edge_index = torch.load("h3_edge_index.pt", map_location="cpu")
        # register_buffer pins this static topology to the active GPU 
        # assigned to each local process thread automatically
        self.register_buffer("edge_index", edge_index)
        
        # 2. Network Layout
        self.conv1 = GCNConv(in_channels=history_steps, out_channels=64)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(in_channels=64, out_channels=32)
        self.linear = nn.Linear(32, 1)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x input shape from DataParallel loader: (Batch_Size, Num_Nodes, History_Steps)
        batch_size, num_nodes, features = x.shape
        
        # Flatten batch into a single large contiguous graph block
        # Shape: (Batch_Size * Num_Nodes, History_Steps)
        x_flat = x.view(-1, features)
        
        # EFFICIENT distributed edge broadcasting:
        # Instead of looping in python, compute batch offsets cleanly using matrix math.
        # This keeps communication and computation confined to the local GPU VRAM.
        offsets = torch.arange(batch_size, device=x.device).repeat_interleave(self.edge_index.shape[1]) * num_nodes
        edge_index_batched = self.edge_index.repeat(1, batch_size) + offsets
        
        # Run local message passing
        h = self.conv1(x_flat, edge_index_batched)
        h = self.relu(h)
        h = self.conv2(h, edge_index_batched)
        h = self.relu(h)
        
        out_flat = self.linear(h)
        # Reshape back to standard dimensions: (Batch_Size, Num_Nodes)
        return out_flat.view(batch_size, num_nodes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # sync_dist=True enforces mean aggregation across all GPUs 
        # to ensure unified metric logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

