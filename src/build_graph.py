import xarray as xr
import h3
import torch
import yaml

def build_h3_graph():
    print("Loading parameters from config.yaml...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Reading H3 cell index list from: {config['paths']['data_nc']}...")
    ds = xr.open_dataset(config['paths']['data_nc'])
    h3_indices = ds['h3_index'].values
    num_nodes = len(h3_indices)

    # Map each H3 hex string to its positional array index for fast lookup
    h3_to_idx = {h3_str: idx for idx, h3_str in enumerate(h3_indices)}

    source_nodes = []
    target_nodes = []

    print("Building H3 grid adjacency topology graph...")
    # Determine the H3 library version to use the correct function call
    try:
        _ = h3.get_res0_cells()  # test for v4 API
        v4_api = True
    except AttributeError:
        v4_api = False

    for idx, cell_str in enumerate(h3_indices):
        neighbors = h3.grid_ring(cell_str, 1) if v4_api else h3.k_ring(cell_str, 1)
            
        for neighbor in neighbors:
            if neighbor in h3_to_idx:
                source_nodes.append(idx)
                target_nodes.append(h3_to_idx[neighbor])

    # Convert to a PyTorch long tensor with shape (2, Num_Edges)
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    
    output_path = config['paths']['edge_index_pt']
    torch.save(edge_index, output_path)
    print(f"SUCCESS: Graph built! Saved {edge_index.shape[1]} edges to {output_path}")

if __name__ == "__main__":
    build_h3_graph()

