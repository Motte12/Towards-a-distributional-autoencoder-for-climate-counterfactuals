#%% import xarray as xr
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Apply Europe land mask.")
    parser.add_argument('--input_dataset', type=str, required=True, help='Path to input dataset (NetCDF)')
    parser.add_argument('--output_dataset', type=str, required=True, help='Path to output masked dataset (NetCDF)')
    args = parser.parse_args()

    input_path = args.input_dataset
    output_path = args.output_dataset

    # load temperature data
    trefht_europe = xr.open_dataset(input_path)

    #%%
    # load land mask (percentage of land in each grid cell)
    ds = xr.open_dataset("sftlf_fx_CESM2_historical_r1i1p1f1_gn.nc")


    # Flip and sort longitude coordinates to facilitate data subsetting
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180

    # Sort longitudes, so that subset operations end up being simpler.
    ds_pre = ds.sortby("lon")

    # Place latitudes in increasing order:
    sftlf = (ds_pre.sortby("lat", ascending=True).sftlf.isel(lat =slice(132,164), lon = slice(135,167))).assign_coords(lat=trefht_europe.lat)


    #%%
    # set land percentage to mask
    land_percentage = 10
    trefht_europe_masked = xr.where(sftlf > land_percentage, trefht_europe, np.nan)
    trefht_europe_masked.to_netcdf(output_path)
    print("saving masked data completed")
    


if __name__ == "__main__":
    main()
