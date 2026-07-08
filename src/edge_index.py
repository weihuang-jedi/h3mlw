import xarray as xr
import h3
import torch
import numpy as np

# 1. Load the generated dataset coordinates
ds = xr.open_dataset("../data/global_h3_res2_air_all_times.nc")
h3_indices = ds['h3_index'].values
num_nodes = len(h3_indices)

# Create a fast hash map pointing an H3 hex string to its positional array index
h3_to_idx = {h3_str: idx for idx, h3_str in enumerate(h3_indices)}

# 2. Extract edge links
source_nodes = []
target_nodes = []

print("Building H3 grid adjacency topology graph...")
for idx, cell_str in enumerate(h3_indices):
    # Retrieve adjacent cell strings natively via H3
    try:
        neighbors = h3.grid_ring(cell_str, 1) # v4 API
    except AttributeError:
        neighbors = h3.k_ring(cell_str, 1)    # v3 API fallback
        
    for neighbor in neighbors:
        if neighbor in h3_to_idx:
            source_nodes.append(idx)
            target_nodes.append(h3_to_idx[neighbor])

# Convert to a PyTorch long tensor with shape (2, Num_Edges)
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
torch.save(edge_index, "h3_edge_index.pt")
print(f"Graph constructed! Enrolled {num_nodes} nodes with {edge_index.shape[1]} unique edge paths.")

