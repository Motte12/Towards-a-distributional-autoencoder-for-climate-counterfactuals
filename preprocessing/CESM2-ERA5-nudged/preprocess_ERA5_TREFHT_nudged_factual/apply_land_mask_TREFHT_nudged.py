#%% import xarray as xr
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

#%%
def main():
    # load temperature data
    #%%
    parser = argparse.ArgumentParser(description="Input arguments")
    parser.add_argument("--input_dataset", type=str, help="Input directory")
    parser.add_argument("--output_dataset", type=str, help="Output .nc file")
    args = parser.parse_args()
    
    #%%
    trefht_europe = xr.open_dataset(args.input_dataset)


    #%%
    # load land mask (percentage of land in each grid cell)
    ds = xr.open_dataset("/climca/people/floer/data/land_mask/sftlf_fx_CESM2_historical_r1i1p1f1_gn.nc")


    # Flip and sort longitude coordinates to facilitate data subsetting
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180

    # Sort longitudes, so that subset operations end up being simpler.
    ds_pre = ds.sortby("lon")

    # Place latitudes in increasing order:
    sftlf = (ds_pre.sortby("lat", ascending=True).sftlf.isel(lat =slice(132,164), lon = slice(135,167))).assign_coords(lat=trefht_europe.lat)


    #%%
    # set land percentage to mask
    land_percentage = 10
    #sftlf_fin = sftlf.assign_coords(lat=llaae_cfs.lat, lon=llaae_cfs.lon)
    trefht_europe_masked = xr.where(sftlf > land_percentage, trefht_europe, np.nan)

    trefht_europe_masked.drop_vars("time_bnds").transpose("time", "lat", "lon").to_netcdf(args.output_dataset)
    print("saving masked data completed")

    #%%

if __name__ == "__main__":
    main()