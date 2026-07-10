import argparse
import os
import h3
import torch
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree

class VirtualTierGraphBuilder:
    """
    Constructs a 3-Tier message passing architecture (Res 2 -> Virtual Mid-Level -> Res 1)
    using only your existing Res 2 and Res 1 baseline grid files, resolving H3FailedErrors.
    """
    def __init__(self, res2_path: str, res1_path: str, output_dir: str = "."):
        self.res2_path = res2_path
        self.res1_path = res1_path
        self.output_dir = output_dir
        
        self.nodes_res2 = []
        self.nodes_res1 = []
        self.nodes_virtual = []

    def compile_virtual_pyramid(self):
        print(f"[STAGE 1] Extracting base physical footprints...")
        with xr.open_dataset(self.res2_path) as ds2:
            self.nodes_res2 = [str(x) for x in ds2['h3_index'].values]
            r2_lats = ds2['latitude'].values
            r2_lons = ds2['longitude'].values

        with xr.open_dataset(self.res1_path) as ds1:
            self.nodes_res1 = [str(x) for x in ds1['h3_index'].values]

        r2_lookup = {h: i for i, h in enumerate(self.nodes_res2)}
        r1_lookup = {h: i for i, h in enumerate(self.nodes_res1)}

        # =================================================================
        # MATHEMATICAL VIRTUAL GENERATION
        # =================================================================
        print("[STAGE 2] Programmatically deriving virtual cluster mid-levels...")
        processed = set()
        virtual_centers = []
        
        for cell in self.nodes_res2:
            if cell in processed:
                continue
            # Pick this cell as a virtual cluster center landmark
            virtual_centers.append(cell)
            neighbors = h3.grid_ring(cell, 1)
            processed.add(cell)
            processed.update(neighbors)

        self.nodes_virtual = sorted(virtual_centers)
        v_lookup = {h: i for i, h in enumerate(self.nodes_virtual)}
        print(f" -> Created {len(self.nodes_virtual)} virtual intermediate clusters.")

        # =================================================================
        # SAFE SPATIAL LOOKUP USING KD-TREE
        # =================================================================
        print("[STAGE 3] Building cross-tier bipartite mapping loops via KD-Tree...")
        
        # 1. Get coordinates of virtual centers for KD-Tree indexing
        v_coords = np.array([h3.cell_to_latlng(v) for v in self.nodes_virtual]) # Lat, Lon
        # Convert longitude/latitude coordinates to 3D Cartesian coordinates to avoid 180/-180 wrap errors
        v_rad = np.radians(v_coords)
        v_x = np.cos(v_rad[:, 0]) * np.cos(v_rad[:, 1])
        v_y = np.cos(v_rad[:, 0]) * np.sin(v_rad[:, 1])
        v_z = np.sin(v_rad[:, 0])
        v_cartesian = np.column_stack((v_x, v_y, v_z))

        # Build KD-Tree over the virtual center points
        tree = cKDTree(v_cartesian)

        # 2. Get Cartesian coordinates of Res 2 physical cells
        r2_coords = np.column_stack((r2_lats, r2_lons))
        r2_rad = np.radians(r2_coords)
        r2_x = np.cos(r2_rad[:, 0]) * np.cos(r2_rad[:, 1])
        r2_y = np.cos(r2_rad[:, 0]) * np.sin(r2_rad[:, 1])
        r2_z = np.sin(r2_rad[:, 0])
        r2_cartesian = np.column_stack((r2_x, r2_y, r2_z))

        # Perform nearest-neighbor search for all Res 2 points against Virtual Centers
        print(" -> Assigning Res 2 physical nodes to closest virtual centers...")
        _, closest_v_indices = tree.query(r2_cartesian, k=1)

        # Build the Map A Bipartite tensor
        map_r2_to_v_src = list(range(len(self.nodes_res2)))
        map_r2_to_v_dst = closest_v_indices.tolist()
        map_r2_to_virtual = torch.tensor([map_r2_to_v_src, map_r2_to_v_dst], dtype=torch.long)

        # Map B: Virtual Mid-Level -> Res 1 (Coarse Physical)
        print(" -> Assigning virtual centers to parent Res 1 cells...")
        map_v_to_r1_src, map_v_to_r1_dst = [], []
        for v_idx, v_cell in enumerate(self.nodes_virtual):
            parent = h3.cell_to_parent(v_cell, 1)
            if parent in r1_lookup:
                map_v_to_r1_src.append(v_idx)
                map_v_to_r1_dst.append(r1_lookup[parent])
                
        map_virtual_to_r1 = torch.tensor([map_v_to_r1_src, map_v_to_r1_dst], dtype=torch.long)

        # =================================================================
        # COMPILING HORIZONTAL EDGE CONNECTIONS
        # =================================================================
        print("[STAGE 4] Building horizontal mesh edges...")
        
        # Horizontal Edges for the Virtual Layer
        v_src, v_dst = [], []
        for v_cell in self.nodes_virtual:
            node_idx = v_lookup[v_cell]
            # Virtual neighbors are computed via an expanded search radius
            for neighbor in h3.grid_ring(v_cell, 2):
                if neighbor in v_lookup:
                    v_src.append(node_idx)
                    v_dst.append(v_lookup[neighbor])
        edge_index_virtual = torch.tensor([v_src, v_dst], dtype=torch.long)

        # Standard physical edges
        r2_src, r2_dst = [], []
        for c in self.nodes_res2:
            for nb in h3.grid_ring(c, 1):
                if nb in r2_lookup:
                    r2_src.append(r2_lookup[c])
                    r2_dst.append(r2_lookup[nb])
        edge_index_res2 = torch.tensor([r2_src, r2_dst], dtype=torch.long)

        r1_src, r1_dst = [], []
        for c in self.nodes_res1:
            for nb in h3.grid_ring(c, 1):
                if nb in r1_lookup:
                    r1_src.append(r1_lookup[c])
                    r1_dst.append(r1_lookup[nb])
        edge_index_res1 = torch.tensor([r1_src, r1_dst], dtype=torch.long)

        # =================================================================
        # SERIALIZE TO DISK
        # =================================================================
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(edge_index_res2, os.path.join(self.output_dir, "edge_index_res2.pt"))
        torch.save(edge_index_virtual, os.path.join(self.output_dir, "edge_index_virtual.pt"))
        torch.save(edge_index_res1, os.path.join(self.output_dir, "edge_index_res1.pt"))
        
        torch.save(map_r2_to_virtual, os.path.join(self.output_dir, "map_res2_to_virtual.pt"))
        torch.save(map_virtual_to_r1, os.path.join(self.output_dir, "map_virtual_to_res1.pt"))
        
        print(f"\nSUCCESS: Added virtual middle level successfully without creating new NetCDF files!")
        print(f" -> Target Directory: {self.output_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="Inject virtual mid-levels into the H3 network framework.")
    parser.add_argument("--res2", required=True, help="Path to global_h3_res2_with_bounds.nc")
    parser.add_argument("--res1", required=True, help="Path to global_h3_res1_with_bounds.nc")
    parser.add_argument("-o", "--output_dir", default=".", help="Output path for geometry files")
    args = parser.parse_args()

    builder = VirtualTierGraphBuilder(args.res2, args.res1, args.output_dir)
    builder.compile_virtual_pyramid()

if __name__ == "__main__":
    main()
