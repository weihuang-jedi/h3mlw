import xarray as xr
import numpy as np
import pandas as pd

def build_multi_variable_h3_dataset():
    print("Opening base geometry definition dataset...")
    ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc").sortby('h3_index')
    
    # Extract target coordinates
    h3_lats = ds_h3['latitude'].values
    h3_lons = ds_h3['longitude'].values
    h3_lon_converted = np.mod(h3_lons, 360)
    
    target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"])
    target_lat = xr.DataArray(h3_lats, dims=["h3_index"])

    # Define the source files and their matching internal variable keys
    # Replace these filenames with your active cluster dataset paths
    variables_map = {
        "air_h3": {"file": "air.sfc.2000.nc", "key": "air"},
        "uwnd_h3": {"file": "uwnd.sfc.2000.nc", "key": "uwnd"},
        "vwnd_h3": {"file": "vwnd.sfc.2000.nc", "key": "vwnd"},
        "rhum_h3": {"file": "rhum.sfc.2000.nc", "key": "rhum"}
    }

    interpolated_vars = {}
    time_coords = None

    for out_key, meta in variables_map.items():
        print(f"Processing source weather parameter matrix: {meta['file']}...")
        ds_src = xr.open_dataset(meta['file']).compute()
        
        # Pull or store unified temporal coordinates array
        if time_coords is None:
            time_coords = ds_src.time.values
            
        src_slice = ds_src[meta['key']].isel(time=slice(0, len(time_coords))).drop_vars('time', errors='ignore')
        
        # Handle descending coordinate orientation
        if src_slice.lat.values[0] > src_slice.lat.values[-1]:
            src_slice = src_slice.sortby('lat')
            
        # Close the 358.125° to 360° global edge wrap-around gap via padding column replication
        new_lon_coords = np.append(src_slice.lon.values, 360.0)
        padded_data = np.concatenate([src_slice.values, src_slice.values[:, :, :1]], axis=2) # time, lat, lon
        
        clean_grid = xr.DataArray(
            padded_data,
            coords={'time': time_coords, 'lat': src_slice.lat.values, 'lon': new_lon_coords},
            dims=['time', 'lat', 'lon']
        )
        
        print(f" -> Splicing 3D interpolation onto H3 graph topology grid...")
        interp_ds = clean_grid.interp(lon=target_lon, lat=target_lat, method="linear", kwargs={'bounds_error': False, 'fill_value': None})
        raw_vals = interp_ds.values
        
        # Clean missing values constants (-9.96921e+36f or isolated NaNs)
        corrupt_mask = (raw_vals < -900) | (raw_vals > 9000) | np.isnan(raw_vals)
        if np.any(corrupt_mask):
            safe_mean = np.nanmean(raw_vals[(raw_vals >= -500) & (raw_vals <= 5000)])
            raw_vals[corrupt_mask] = safe_mean if not np.isnan(safe_mean) else 0.0
            
        interpolated_vars[out_key] = (["time", "h3_index"], raw_vals)

    # 3. Append static boundaries
    interpolated_vars["longitude"] = (["h3_index"], h3_lons)
    interpolated_vars["latitude"] = (["h3_index"], h3_lats)
    interpolated_vars["longitude_bounds"] = (["h3_index", "vertices"], ds_h3['longitude_bounds'].values)
    interpolated_vars["latitude_bounds"] = (["h3_index", "vertices"], ds_h3['latitude_bounds'].values)
    interpolated_vars["land_sea_mask"] = (["h3_index"], ds_h3['land_sea_mask'].values)
    interpolated_vars["elevation"] = (["h3_index"], ds_h3['elevation'].values)

    print("Writing multi-variable master dataset to disk...")
    ds_master = xr.Dataset(
        data_vars=interpolated_vars,
        coords={"time": time_coords, "h3_index": ds_h3.h3_index.values, "vertices": np.arange(6)},
        attrs={"title": "Master Multi-Variable Global AI Ingestion Dataset"}
    )
    
    ds_master.to_netcdf("global_h3_res2_multi_variable_times.nc", format="NETCDF4")
    print("SUCCESS: File built! Ready for multi-variable deep learning optimization runs.")

if __name__ == "__main__":
    build_multi_variable_h3_dataset()

