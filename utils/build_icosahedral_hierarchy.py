import argparse
import os
import numpy as np
import torch

class IcosahedralHierarchyBuilder:
    """
    Generates a multi-scale GraphCast-style hierarchical mesh topology pipeline,
    compiling horizontal mixing graphs and vertical cross-tier bipartite maps from M0 to Mk.
    """
    def __init__(self, max_level: int = 4, output_dir: str = "."):
        self.max_level = max_level
        self.output_dir = output_dir

    def _get_base_icosahedron(self):
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ], dtype=np.float64)
        vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        return vertices, faces

    def _compile_horizontal_edges(self, faces: np.ndarray) -> torch.Tensor:
        """Extracts unique directed graph edges from a set of triangular faces."""
        edges_src = []
        edges_dst = []
        for face in faces:
            n0, n1, n2 = face[0], face[1], face[2]
            edges_src.extend([n0, n1, n1, n2, n2, n0])
            edges_dst.extend([n1, n0, n2, n1, n0, n2])
        edge_stack = np.vstack((edges_src, edges_dst))
        unique_edges = np.unique(edge_stack, axis=1)
        return torch.from_numpy(unique_edges).long()

    def compile_hierarchy(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize arrays with the base M0 state
        vertices, faces = self._get_base_icosahedron()
        
        print(f"[STAGE 1] Compiling horizontal components for Root Layer M0...")
        edge_index_m0 = self._compile_horizontal_edges(faces)
        torch.save(edge_index_m0, os.path.join(self.output_dir, "edge_index_m0.pt"))
        
        # Track past counts for vertical binding loops
        prev_num_nodes = len(vertices)

        # Loop through each refinement level step-by-step
        for level in range(1, self.max_level + 1):
            print(f"\n[STAGE 2] Processing Subdivision Layer: M{level-1} -> M{level}...")
            new_faces = []
            midpoint_cache = {}
            verts_list = vertices.tolist()

            def get_midpoint(p1_idx, p2_idx):
                edge = tuple(sorted((p1_idx, p2_idx)))
                if edge in midpoint_cache:
                    return midpoint_cache[edge]
                v1 = np.array(verts_list[p1_idx])
                v2 = np.array(verts_list[p2_idx])
                mid = v1 + v2
                mid /= np.linalg.norm(mid)
                verts_list.append(mid.tolist())
                new_idx = len(verts_list) - 1
                midpoint_cache[edge] = new_idx
                return new_idx

            for face in faces:
                v0, v1, v2 = face[0], face[1], face[2]
                m01 = get_midpoint(v0, v1)
                m12 = get_midpoint(v1, v2)
                m20 = get_midpoint(v2, v0)

                new_faces.append([v0, m01, m20])
                new_faces.append([v1, m12, m01])
                new_faces.append([v2, m20, m12])
                new_faces.append([m01, m12, m20])

            vertices = np.array(verts_list)
            faces = np.array(new_faces, dtype=np.int32)
            current_num_nodes = len(vertices)

            # 1. Compile and save active horizontal layer edges
            print(f" -> Saving horizontal mixing edge index for M{level}...")
            edge_index = self._compile_horizontal_edges(faces)
            torch.save(edge_index, os.path.join(self.output_dir, f"edge_index_m{level}.pt"))

            # 2. Compile and save vertical bipartite pooling map.
            # Due to the nested construction property, each parent index maps directly 
            # to an identical index value in the child layer.
            print(f" -> Compiling vertical mapping links: M{level-1} -> M{level}...")
            map_src = list(range(prev_num_nodes))
            map_dst = list(range(prev_num_nodes))
            bipartite_map = torch.tensor([map_src, map_dst], dtype=torch.long)
            torch.save(bipartite_map, os.path.join(self.output_dir, f"map_m{level-1}_to_m{level}.pt"))

            prev_num_nodes = current_num_nodes

        print(f"\nSUCCESS: Icosahedral hierarchy graphs saved to '{self.output_dir}'!")

def main():
    parser = argparse.ArgumentParser(description="Compile nested multi-scale graphs across icosahedral grid steps.")
    parser.add_argument("-k", "--max_level", type=int, default=4, help="Maximum target resolution scale level (default: 4)")
    parser.add_argument("-o", "--output_dir", default=".", help="Target output workspace directory for PT tensor blocks")
    args = parser.parse_args()

    builder = IcosahedralHierarchyBuilder(max_level=args.max_level, output_dir=args.output_dir)
    builder.compile_hierarchy()

if __name__ == "__main__":
    main()

