import argparse
import xarray as xr
import torch
import numpy as np
import h3

def build_hierarchical_graphs(input_nc):
    print(f"Opening data layout file to discover spatial grid coordinates: {input_nc}")
    ds = xr.open_dataset(input_nc)
    
    # 1. READ FINE NODES (Resolution 2)
    # Ensure we have the actual hex string IDs. If they are stored as integers/strings, fetch them.
    if 'h3_index' in ds:
        fine_hexes = [str(x) for x in ds['h3_index'].values]
    else:
        # Fallback: if your dataset uses an index array, generate hexes from lat/lon positions
        lats = ds['latitude'].values.squeeze()
        lons = ds['longitude'].values.squeeze()
        fine_hexes = [h3.latlng_to_cell(lat, lon, 2) for lat, lon in zip(lats, lons)]
        
    num_fine_nodes = len(fine_hexes)
    # Create a mapping dictionary for quick index lookups
    fine_hex_to_idx = {hex_id: i for i, hex_id in enumerate(fine_hexes)}
    print(f" -> Successfully discovered {num_fine_nodes} Fine Nodes (H3 Resolution 2)")

    # =====================================================================
    # PHASE A: GENERATE FINE TO COARSE (BIPARTITE) MAPPINGS
    # =====================================================================
    print("Computing parent nodes at H3 Resolution 1 for the hierarchy...")
    # Find the unique parent hex string IDs at Resolution 1
    coarse_hexes = sorted(list(set([h3.cell_to_parent(h, 1) for h in fine_hexes])))
    num_coarse_nodes = len(coarse_hexes)
    coarse_hex_to_idx = {hex_id: i for i, hex_id in enumerate(coarse_hexes)}
    print(f" -> Generated {num_coarse_nodes} Coarse Nodes (H3 Resolution 1)")

    print("Building Fine-to-Coarse (Bipartite Pooling) edge indexing maps...")
    f2c_src = []
    f2c_dst = []
    
    for fine_hex in fine_hexes:
        parent_hex = h3.cell_to_parent(fine_hex, 1)
        
        fine_idx = fine_hex_to_idx[fine_hex]
        coarse_idx = coarse_hex_to_idx[parent_hex]
        
        # Direction: From Fine Node index to Coarse Node index
        f2c_src.append(fine_idx)
        f2c_dst.append(coarse_idx)
        
    edge_index_f2c = torch.tensor([f2c_src, f2c_dst], dtype=torch.long)

    # =====================================================================
    # PHASE B: GENERATE CORE FINE GRIDS CONNECTIONS (Original Layout)
    # =====================================================================
    print("Building localized Fine Grid (Res 2) edge connections...")
    fine_src = []
    fine_dst = []
    for fine_hex in fine_hexes:
        fine_idx = fine_hex_to_idx[fine_hex]
        # Get immediate neighboring cells at resolution 2
        neighbors = h3.grid_ring(fine_hex, 1)
        for nb in neighbors:
            if nb in fine_hex_to_idx:  # Check if neighbor is in our dataset
                fine_src.append(fine_idx)
                fine_dst.append(fine_hex_to_idx[nb])
                
    edge_index_fine = torch.tensor([fine_src, fine_dst], dtype=torch.long)

    # =====================================================================
    # PHASE C: GENERATE CORE COARSE GRIDS CONNECTIONS (Macro Layout)
    # =====================================================================
    print("Building global Coarse Grid (Res 1) edge connections...")
    coarse_src = []
    coarse_dst = []
    for coarse_hex in coarse_hexes:
        coarse_idx = coarse_hex_to_idx[coarse_hex]
        # Get immediate neighboring cells at resolution 1
        neighbors = h3.grid_ring(coarse_hex, 1)
        for nb in neighbors:
            if nb in coarse_hex_to_idx:
                coarse_src.append(coarse_idx)
                coarse_dst.append(coarse_hex_to_idx[nb])
                
    edge_index_coarse = torch.tensor([coarse_src, coarse_dst], dtype=torch.long)

    # =====================================================================
    # PHASE D: SERIALIZE TENSORS TO DISK
    # =====================================================================
    print("\nSaving completed structural geometry graphs to disk...")
    torch.save(edge_index_fine, "h3_edge_index.pt")
    torch.save(edge_index_coarse, "coarse_edge_index.pt")
    torch.save(edge_index_f2c, "fine_to_coarse_edge_index.pt")
    
    print("-" * 60)
    print(f"SUCCESS: Graphs compiled!")
    print(f" -> fine_edge_index.pt shape:          {edge_index_fine.shape}")
    print(f" -> coarse_edge_index.pt shape:        {edge_index_coarse.shape}")
    print(f" -> fine_to_coarse_edge_index.pt shape: {edge_index_f2c.shape}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Resolution Hierarchical H3 Edge Tensors.")
    parser.add_argument("-i", "--input", required=True, help="Path to sample reference netcdf file containing hex locations")
    args = parser.parse_args()
    
    build_hierarchical_graphs(args.input)
