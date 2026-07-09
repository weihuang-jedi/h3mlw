import argparse
import xarray as xr
import numpy as np

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Vectorized 3D Spatial Interpolation from Lat-Lon to H3 Grid.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input regular lat-lon NetCDF file (e.g., air.sfc.2000.nc)")
    parser.add_argument("-o", "--output", required=True, help="Path to save the interpolated H3 output NetCDF file")
    args = parser.parse_args()

    # ===================================================
    # STEP 1: LOAD & ALIGN GRID STRUCTURES
    # ===================================================
    print("Loading master H3 geometry template definitions...")
    ds_h3 = xr.open_dataset("global_h3_res2_with_bounds.nc")

    print(f"Loading full time series weather profile from: {args.input}")
    # Open dataset with chunking enabled
    ds_src = xr.open_dataset(args.input, chunks={"time": 500})

    # DYNAMIC VARIABLE DETECTOR: Identify the weather variable payload key (skips dimension keys)
    coords_keys = {"time", "lat", "lon", "time_bnds", "nbnds"}
    var_name = list(set(ds_src.data_vars.keys()) - coords_keys)[0]
    print(f" -> Automatically detected target data variable variable: '{var_name}'")

    # Isolate coordinate markers
    h3_lons = ds_h3['longitude'].values
    h3_lats = ds_h3['latitude'].values
    h3_lon_converted = np.mod(h3_lons, 360)

    # Check and sort dataset latitudes to ensure ascending order (-90 to 90)
    if ds_src.lat.values[0] > ds_src.lat.values[-1]:
        print("Inverting input latitude coordinates axis grid...")
        ds_src = ds_src.sortby('lat')

    # ===================================================
    # STEP 2: APPLY TEMPORAL CIRCULAR LONGITUDE PADDING
    # ===================================================
    print("Applying circular longitude padding across all time steps...")
    raw_lons = ds_src.lon.values
    padded_lons = np.append(raw_lons, 360.0)

    # Extract the full 3D array data chunk using the dynamic key variable name
    raw_data_array = ds_src[var_name]

    # Append the 0-degree longitude column slice to the end of the array
    padded_air_array = xr.concat([raw_data_array, raw_data_array.isel(lon=slice(0, 1))], dim="lon")
    padded_air_array = padded_air_array.assign_coords(lon=padded_lons)

    # ===================================================
    # STEP 3: EXECUTE VECTORIZED 3D TIME SERIES LOOKUP
    # ===================================================
    print(f"Executing vectorized interpolation across {len(ds_src.time)} time steps...")
    target_lon = xr.DataArray(h3_lon_converted, dims=["h3_index"], coords={"h3_index": ds_h3.h3_index})
    target_lat = xr.DataArray(h3_lats, dims=["h3_index"], coords={"h3_index": ds_h3.h3_index})

    interpolated_ds = padded_air_array.interp(
        lon=target_lon,
        lat=target_lat,
        method="linear",
        kwargs={'bounds_error': False, 'fill_value': None}
    )

    # ===================================================
    # STEP 4: ASSEMBLE & EXPORT TEMPORAL NETCDF FILE
    # ===================================================
    print("Structuring compliant output metadata variables...")
    
    # Dynamically name the output variable based on the input payload
    output_var_name = f"{var_name}_h3"
    
    ds_output = xr.Dataset(
        data_vars={
            output_var_name: (
                ["time", "h3_index"],
                interpolated_ds.values,
                {
                    "units": ds_src[var_name].attrs.get("units", "unknown"),
                    "long_name": f"3-hourly {var_name.upper()} Interpolated to H3 Mesh",
                    "coordinates": "longitude latitude"
                }
            ),
            "longitude": (["h3_index"], h3_lons, {"units": "degrees_east", "bounds": "longitude_bounds"}),
            "latitude": (["h3_index"], h3_lats, {"units": "degrees_north", "bounds": "latitude_bounds"}),
            "longitude_bounds": (["h3_index", "vertices"], ds_h3['longitude_bounds'].values, {"units": "degrees_east"}),
            "latitude_bounds": (["h3_index", "vertices"], ds_h3['latitude_bounds'].values, {"units": "degrees_north"}),
        },
        coords={
            "time": ds_src.time.values,
            "h3_index": ds_h3.h3_index.values,
            "vertices": np.arange(6)
        },
        attrs={
            "title": "Full Time-Series Global H3 Grid (Res 2)",
            "source_dataset": ds_src.attrs.get("title", "NOAA-CIRES Reanalysis"),
            "total_time_steps": len(ds_src.time)
        }
    )

    print(f"Writing time-series file to disk: {args.output}...")
    ds_output.to_netcdf(args.output, format="NETCDF4")

    print("\n" + "="*50)
    print("TIME SERIES TEMPORAL INTERPOLATION SUCCESSFUL")
    print("="*50)
    print(f"Output File Name      : {args.output}")
    print(f"Result Matrix Shape   : {ds_output[output_var_name].shape} (time x h3_index)")
    print(f"Processed Time Steps  : {len(ds_output.time)} frames")
    print(f"Total H3 Cells/Frame  : {len(ds_output.h3_index)} hexagons")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

