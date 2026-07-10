import os
import argparse
import xarray as xr
import torch
import numpy as np
import h3

class TieredH3GraphBuilder:
    """
    Constructs a 3-tier deep hierarchical spatial mesh layout (Res 2 -> Res 1 -> Res 0)
    to drive multi-scale message-passing GNNs.
    """
    def __init__(self, res2_path: str, res1_path: str, res0_path: str, output_dir: str = "."):
        self.paths = {2: res2_path, 1: res1_path, 0: res0_path}
        self.output_dir = output_dir
        self.hex_lists = {}
        self.hex_to_idx = {}

    def load_all_resolutions(self) -> None:
        print("[INIT] Loading 3-Tier resolution structures from disk...")
        for res, path in self.paths.items():
            with xr.open_dataset(path) as ds:
                hexes = [str(x) for x in ds['h3_index'].values]
                # Keep original order to ensure perfect feature mapping later
                self.hex_lists[res] = hexes
                self.hex_to_idx[res] = {hex_id: i for i, hex_id in enumerate(hexes)}
            print(f" -> Resolution {res}: Found {len(self.hex_lists[res])} nodes.")

    def build_bipartite_map(self, fine_res: int, coarse_res: int) -> torch.Tensor:
        print(f"[MAP] Generating structural pooling index: Res {fine_res} -> Res {coarse_res}...")
        src, dst = [], []
        fine_hexes = self.hex_lists[fine_res]
        coarse_lookup = self.hex_to_idx[coarse_res]

        for fine_hex in fine_hexes:
            parent_hex = h3.cell_to_parent(fine_hex, coarse_res)
            if parent_hex in coarse_lookup:
                src.append(self.hex_to_idx[fine_res][fine_hex])
                dst.append(coarse_lookup[parent_hex])
        return torch.tensor([src, dst], dtype=torch.long)

    def build_grid_ring_edges(self, res: int) -> torch.Tensor:
        print(f"[EDGES] Generating horizontal mixing graph for Resolution {res}...")
        src, dst = [], []
        hex_list = self.hex_lists[res]
        lookup = self.hex_to_idx[res]

        for hex_id in hex_list:
            node_idx = lookup[hex_id]
            for neighbor in h3.grid_ring(hex_id, 1):
                if neighbor in lookup:
                    src.append(node_idx)
                    dst.append(lookup[neighbor])
        return torch.tensor([src, dst], dtype=torch.long)

    def compile(self) -> None:
        self.load_all_resolutions()
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Generate Horizontal mixing topologies for each graph layer
        edges_res2 = self.build_grid_ring_edges(2)
        edges_res1 = self.build_grid_ring_edges(1)
        edges_res0 = self.build_grid_ring_edges(0)

        # 2. Generate Vertical spatial pooling indexes
        map_res2_to_res1 = self.build_bipartite_map(2, 1)
        map_res1_to_res0 = self.build_bipartite_map(1, 0)

        # 3. Serialize all arrays to disk
        torch.save(edges_res2, os.path.join(self.output_dir, "edge_index_res2.pt"))
        torch.save(edges_res1, os.path.join(self.output_dir, "edge_index_res1.pt"))
        torch.save(edges_res0, os.path.join(self.output_dir, "edge_index_res0.pt"))
        torch.save(map_res2_to_res1, os.path.join(self.output_dir, "map_res2_to_res1.pt"))
        torch.save(map_res1_to_res0, os.path.join(self.output_dir, "map_res1_to_res0.pt"))
        print(f"\nSUCCESS: 3-Resolution Graph Architecture saved to '{self.output_dir}'!\n")

def main():
    parser = argparse.ArgumentParser(description="3-Tier Hierarchical Graph Component Compiler.")
    parser.add_argument("--res2", required=True, help="Path to global_h3_res2_with_bounds.nc")
    parser.add_argument("--res1", required=True, help="Path to global_h3_res1_with_bounds.nc")
    parser.add_argument("--res0", required=True, help="Path to global_h3_res0_with_bounds.nc")
    parser.add_argument("-o", "--output_dir", default=".", help="Where to save compiled graph tensors")
    args = parser.parse_args()

    builder = TieredH3GraphBuilder(args.res2, args.res1, args.res0, args.output_dir)
    builder.compile()

if __name__ == "__main__":
    main()

