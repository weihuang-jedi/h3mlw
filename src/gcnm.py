import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

class H3WeatherGCN(pl.LightningModule):
    def __init__(self, history_steps=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        
        # Load the global adjacency mapping we generated in Step 1
        # register_buffer ensures the graph layout is pinned correctly to CPU/GPU memory
        edge_index = torch.load("h3_edge_index.pt")
        self.register_buffer("edge_index", edge_index)
        
        # Layer 1: Aggregates historical metrics along local edges
        self.conv1 = GCNConv(in_channels=history_steps, out_channels=32)
        self.relu = nn.ReLU()
        
        # Layer 2: Deeper message passing step
        self.conv2 = GCNConv(in_channels=32, out_channels=16)
        
        # Output Linear Head: Projects hidden dimensions back down to 1 scalar forecast variable
        self.linear = nn.Linear(16, 1)
        
        # Loss metric tracking criteria
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x shape inside the batch: (Batch_Size, Num_Nodes, History_Steps)
        batch_size, num_nodes, features = x.shape
        
        # Unroll batch tensor to match torch_geometric's contiguous node stack requirements
        # Reshaping yields a flat contiguous block of shape (Batch_Size * Num_Nodes, Features)
        x_flat = x.view(-1, features)
        
        # Broadcast the global edge index mapping across the data blocks
        # This repeats edge indexing configurations to cover the batch array footprint
        edge_index_expanded = self._get_batch_edges(batch_size, num_nodes)
        
        # Message Passing Layer 1
        h = self.conv1(x_flat, edge_index_expanded)
        h = self.relu(h)
        
        # Message Passing Layer 2
        h = self.conv2(h, edge_index_expanded)
        h = self.relu(h)
        
        # Predict target values
        out_flat = self.linear(h) # Yields shape: (Batch_Size * Num_Nodes, 1)
        
        # Re-pack the flat outputs back into standard batch sizes: (Batch_Size, Num_Nodes)
        return out_flat.view(batch_size, num_nodes)

    def _get_batch_edges(self, batch_size, num_nodes):
        """Repeats edge connections across stacked batches."""
        edges = []
        for b in range(batch_size):
            offset = b * num_nodes
            edges.append(self.edge_index + offset)
        return torch.cat(edges, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch # x: inputs, y: true targets
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

