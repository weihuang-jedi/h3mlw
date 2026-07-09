import argparse
import xarray as xr
import h3
import torch
import yaml
import os

class H3GraphTopologyBuilder:
    """
    Class module to extract H3 spatial strings from a combined master weather NetCDF file,
    compute topological neighbor boundaries, and compile a PyTorch-ready edge connectivity map.
    """
    def __init__(self, config_path: str = "config.yaml", input_override: str = None, output_override: str = None):
        self.config_path = config_path
        self.input_override = input_override
        self.output_override = output_override
        
        self.config = {}
        self.data_nc_path = None
        self.edge_output_path = None
        self.h3_indices = None

    def load_configuration(self) -> None:
        """Parses path rules from the YAML config file or falls back to terminal flags."""
        if os.path.exists(self.config_path):
            print(f"Reading topology routing boundaries from: {self.config_path}")
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            print("Notice: config.yaml not found. Relying strictly on parameter overrides.")

        # Determine target file mapping layers using conditional fallback checks
        self.data_nc_path = self.input_override if self.input_override else self.config.get('paths', {}).get('data_nc')
        self.edge_output_path = self.output_override if self.output_override else self.config.get('paths', {}).get('edge_index_pt')

        if not self.data_nc_path or not self.edge_output_path:
            raise ValueError("Missing file configuration requirements. Provide valid path arguments or a config.yaml file.")

    def extract_nodes(self) -> None:
        """Loads the unified dataset and isolates the alphanumeric H3 index elements."""
        print(f"Opening combined master weather dataset: {self.data_nc_path}")
        with xr.open_dataset(self.data_nc_path) as ds:
            # NetCDF stores H3 index parameters as string elements or bytes arrays
            raw_indices = ds['h3_index'].values
            
            # Unpack bytes formatting safely if encountered across different Python/NetCDF compilers
            if isinstance(raw_indices[0], bytes):
                self.h3_indices = [idx.decode('utf-8') for idx in raw_indices]
            else:
                self.h3_indices = [str(idx) for idx in raw_indices]
                
        print(f" -> Enrolled {len(self.h3_indices)} unique hexagonal coordinate cells from the master file.")

    def construct_edge_index(self) -> None:
        """Computes adjacent cell ring connections using a version-agnostic H3 mapping layer."""
        if self.h3_indices is None:
            raise ValueError("H3 coordinate nodes not extracted. Run extract_nodes() first.")

        # Create a fast hash dictionary mapping an H3 index string to its integer row offset
        h3_to_idx = {h3_str: idx for idx, h3_str in enumerate(self.h3_indices)}

        source_nodes = []
        target_nodes = []

        # Version check the active python H3 library installation bounds to prevent runtime errors
        try:
            _ = h3.get_res0_cells()  # Modern v4 API test call
            v4_api = True
        except AttributeError:
            v4_api = False

        print(f"Processing neighbor connections using H3 Version {'4' if v4_api else '3'} library layer...")
        for idx, cell_str in enumerate(self.h3_indices):
            # Retrieve adjacent cell lists
            neighbors = h3.grid_ring(cell_str, 1) if v4_api else h3.k_ring(cell_str, 1)
            
            for neighbor in neighbors:
                # Add edge route only if the adjacent cell exists in our global database frame
                if neighbor in h3_to_idx:
                    source_nodes.append(idx)
                    target_nodes.append(h3_to_idx[neighbor])

        # Convert edge coordinates to a PyTorch long tensor with shape: (2, total_edges)
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Ensure destination directory structure exists
        output_dir = os.path.dirname(self.edge_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Writing PyTorch Element Buffer matrix array to disk: {self.edge_output_path}...")
        torch.save(edge_index, self.edge_output_path)
        
        print("\n" + "="*50)
        print("GRAPH TOPOLOGY CONNECTIVITY BUILD SUCCESSFUL")
        print("="*50)
        print(f"Target Input File: {self.data_nc_path}")
        print(f"Saved Edge Paths : {self.edge_output_path}")
        print(f"Total Unique Nodes: {len(self.h3_indices)} hexagons")
        print(f"Total Adjacency   : {edge_index.shape[1]} unique edge paths")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Integrated Object-Oriented H3 Adjacency Graph Builder Utility.")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to your system config.yaml file.")
    parser.add_argument("-i", "--input", default=None, help="Override path to master input NetCDF file.")
    parser.add_argument("-o", "--output", default=None, help="Override destination path for the compiled PyTorch .pt tensor.")
    args = parser.parse_args()

    # Execute graph compilation loop sequentially
    builder = H3GraphTopologyBuilder(
        config_path=args.config,
        input_override=args.input,
        output_override=args.output
    )
    builder.load_configuration()
    builder.extract_nodes()
    builder.construct_edge_index()


if __name__ == "__main__":
    main()

