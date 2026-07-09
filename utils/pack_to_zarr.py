import xarray as xr
import zarr
from numcodecs import Blosc

def convert_netcdf_to_zarr(file_pattern, output_zarr_path):
    print("Opening NetCDF multi-file compilation...")
    # Open without computing yet, keeping it lazy
    ds = xr.open_mfdataset(file_pattern, combine="by_coords", data_vars="minimal")
    
    # Crucial: Chunk along the time dimension for ML sequencing optimization
    # e.g., chunk size of 32 or 64 steps works great for training sequences
    chunk_spec = {'time': 32, 'h3_index': -1} # Adjust 'h3_index' to your spatial dimension name
    ds = ds.chunk(chunk_spec)
    
    # Configure high-efficiency compression (Blosc Zstd)
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {'compressor': compressor} for var in ds.data_vars}
    
    print(f"Writing compressed Zarr store to: {output_zarr_path}...")
    ds.to_zarr(output_zarr_path, mode='w', encoding=encoding, consolidated=True)
    print("Pack complete!")

if __name__ == "__main__":
    convert_netcdf_to_zarr("../data/global_h3_res2_air_sfc_19*.nc", "../data/global_h3_20years.zarr")

