import argparse
import os
import h3
import torch
import xarray as xr

class ArbitraryH3GraphBuilder:
    """
    Automated spatial compiler for arbitrary multi-scale graph networks (e.g., 4-tier or 5-tier pipelines).
    """
    def __init__(self, resolutions: list, reference_grid_paths: list, output_dir: str = "."):
        assert len(resolutions) == len(reference_grid_paths), "Each resolution requires a matching netcdf path footprint."
        self.resolutions = resolutions
        self.paths = dict(zip(resolutions, reference_grid_paths))
        self.output_dir = output_dir
        self.hex_lists = {}
        self.hex_to_idx = {}

    def load_all_nodes(self):
        print("[INIT] Loading arbitrary node sequences from target paths...")
        for res in self.resolutions:
            with xr.open_dataset(self.paths[res]) as ds:
                hexes = [str(x) for x in ds['h3_index'].values]
                self.hex_lists[res] = hexes
                self.hex_to_idx[res] = {hex_id: i for i, hex_id in enumerate(hexes)}
            print(f" -> Level Res {res}: Bound {len(self.hex_lists[res])} nodes.")

    def build_horizontal_edges(self, res: int) -> torch.Tensor:
        print(f"[GRID] Connecting ring neighbors at Resolution {res}...")
        src, dst = [], []
        lookup = self.hex_to_idx[res]
        for hex_id in self.hex_lists[res]:
            node_idx = lookup[hex_id]
            for neighbor in h3.grid_ring(hex_id, 1):
                if neighbor in lookup:
                    src.append(node_idx)
                    dst.append(lookup[neighbor])
        return torch.tensor([src, dst], dtype=torch.long)

    def build_bipartite_pool_map(self, fine_res: int, coarse_res: int) -> torch.Tensor:
        print(f"[POOL] Compiling bipartite link array: Res {fine_res} -> Res {coarse_res}...")
        src, dst = [], []
        coarse_lookup = self.hex_to_idx[coarse_res]
        for fine_hex in self.hex_lists[fine_res]:
            parent_hex = h3.cell_to_parent(fine_hex, coarse_res)
            if parent_hex in coarse_lookup:
                src.append(self.hex_to_idx[fine_res][fine_hex])
                dst.append(coarse_lookup[parent_hex])
        return torch.tensor([src, dst], dtype=torch.long)

    def compile_all(self):
        self.load_all_nodes()
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Compile all horizontal layer connections
        for res in self.resolutions:
            edges = self.build_horizontal_edges(res)
            torch.save(edges, os.path.join(self.output_dir, f"edge_index_res{res}.pt"))

        # 2. Compile all adjacent vertical pooling blocks
        for i in range(len(self.resolutions) - 1):
            fine_res = self.resolutions[i]
            coarse_res = self.resolutions[i+1]
            bipartite_map = self.build_bipartite_pool_map(fine_res, coarse_res)
            torch.save(bipartite_map, os.path.join(self.output_dir, f"map_res{fine_res}_to_res{coarse_res}.pt"))

        print(f"\nSUCCESS: Modular N-Tier Graph compiled inside '{self.output_dir}'!\n")

def main():
    parser = argparse.ArgumentParser(description="Modular Hierarchical N-Resolution H3 Graph Builder.")
    parser.add_argument("--resolutions", type=int, nargs="+", required=True, help="List of resolutions from fine to root (e.g. 3 2 1 0)")
    parser.add_argument("--paths", nargs="+", required=True, help="Matching reference grid netcdf file paths in identical sequence order")
    parser.add_argument("-o", "--output_dir", default=".", help="Target destination directory for PT files")
    args = parser.parse_args()

    builder = ArbitraryH3GraphBuilder(args.resolutions, args.paths, args.output_dir)
    builder.compile_all()

if __name__ == "__main__":
    main()

