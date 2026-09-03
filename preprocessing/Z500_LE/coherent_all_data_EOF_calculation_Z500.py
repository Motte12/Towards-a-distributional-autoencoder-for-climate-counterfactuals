#%%
import json
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
from netCDF4 import Dataset
from eofs.xarray import Eof
import pickle
import argparse
from dask.diagnostics import ProgressBar
import shutil
from datetime import datetime

# Add utility functions path
sys.path.append('/home/floer/Climate_Counterfactuals/climat-counterfactuals/utility_functions')
sys.path.append('/home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized')

import load_datasets as load
import analysis
import helper_functions as hf
#%%
def main():
    
    #%%
    parser = argparse.ArgumentParser(description="Process and stack TREFHT data.")
    parser.add_argument('--le_directory_path', type=str, required=True, help='Input directory')
    parser.add_argument('--eth_data_directory_path', type=str, required=True, help='Input directory')
    parser.add_argument('--era5_data_directory_path', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()
    le_directory_path = args.le_directory_path
    eth_data_directory_path = args.eth_data_directory_path
    era5_data_directory_path = args.era5_data_directory_path
    output_path = args.output_dir
    
    
    #%% 
    # read contents from the settings.json file
    settings_file_path = 'preprocessing_settings_Z500.json'
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)
    n_anno_indices = settings['n_anno_indices']
    plat_min = settings['plat_min']
    plat_max = settings['plat_max']
    plon_min = settings['plon_min']
    plon_max = settings['plon_max']

    #%%
    print("Command line arguments:")
    print(f"  le_directory_path: {le_directory_path}")
    print(f"  eth_data_directory_path: {eth_data_directory_path}")
    print(f"  era5_data_directory_path: {era5_data_directory_path}")
    print(f"  output_path: {output_path}")
    
    print("\nSettings from JSON file:")
    print(json.dumps(settings, indent=2))


    
    
    ####################
    #### Load  Data ####
    ####################
    # %%
    # LARGE ENSEMBLE
    chunk_shape = {'ensemble_member': 1, 'lat': 192, 'lon': 288}
    le_directory_path = "/climca/people/floer/data/automated_preprocessing_13012025/Z500/temporary2"
    psl_ds_le_pre2 = hf.load_data("5day", #only loads _anom.nc files 
                              le_directory_path, 
                              chunk_shape=chunk_shape).sel(
                                  lat = slice(plat_min,plat_max), 
                                  lon = slice(plon_min, plon_max))
    psl_ds_le_pre = psl_ds_le_pre2.where(psl_ds_le_pre2['time'].dt.month.isin([6, 7, 8]), drop=True).chunk({'ensemble_member': 1, 'time': 4769, 'lat': 53, 'lon': 85})
    #slp_ensemble_mean = psl_ds_le.mean(dim = "ensemble_member")
    psl_ds_le_pre

    # %% 
    # compute ensemble mean
    le_ens_mean = psl_ds_le_pre.Z500.mean(dim="ensemble_member")
    le_ens_mean

    # %%
    # subtract ensemble mean from each ensemble member
    psl_ds_le = psl_ds_le_pre.Z500 - le_ens_mean
    psl_ds_le 

    #%%
    # ETH
    eth_data_directory_path = "/climca/people/floer/data/automated_preprocessing_13012025/Z500_ETH/temporary2"
    psl_ds_eth_pre2 = hf.load_data("5day", 
                            eth_data_directory_path, 
                            chunk_shape=chunk_shape).sel(
                                lat = slice(plat_min,plat_max), 
                                lon = slice(plon_min, plon_max))
    psl_ds_eth_pre = psl_ds_eth_pre2.where(psl_ds_eth_pre2['time'].dt.month.isin([6, 7, 8]), drop=True).Z500.chunk({'ensemble_member': 1, 'time': 4769, 'lat': 53, 'lon': 85})
    psl_ds_eth = psl_ds_eth_pre.assign_coords(lat=le_ens_mean.lat) - le_ens_mean #subtract ensemble mean from ETH data 
    psl_ds_eth

    #%%
    # ERA5
    era5_data_directory_path = "/climca/people/floer/data/ERA5/Z500/temporary2"
    psl_ds_era5_pre2 = hf.load_data("5day", 
                            era5_data_directory_path, 
                            chunk_shape=chunk_shape).sel(
                                lat = slice(plat_min,plat_max), 
                                lon = slice(plon_min, plon_max))
    
    psl_ds_era5_pre = psl_ds_era5_pre2.where(psl_ds_era5_pre2['time'].dt.month.isin([6, 7, 8]), drop=True).var129.chunk({'ensemble_member': 1, 'time': 1596, 'lat': 53, 'lon': 85, 'plev': 1})
    psl_ds_era5_pre 
    
    #%%
    # LE
    transposed_combined_psl_le_pre = ((psl_ds_le.rename({'time': 't'})).stack(time=("ensemble_member", "t"))).transpose('time', 'lat', 'lon')
    transposed_combined_psl_le = transposed_combined_psl_le_pre.assign_coords(time=transposed_combined_psl_le_pre.t.values) #reassign time coordinates for projecting onto EOFs

    # ETH
    # with le ens mean subtracted
    transposed_combined_psl_eth_pre_detrend = ((psl_ds_eth.rename({'time': 't'})).stack(time=("ensemble_member", "t"))).transpose('time', 'lat', 'lon')
    transposed_combined_psl_eth_detrend = transposed_combined_psl_eth_pre_detrend.assign_coords(time=transposed_combined_psl_eth_pre_detrend.t.values)

    # without le ens mean subtracted
    transposed_combined_psl_eth_pre = ((psl_ds_eth_pre.rename({'time': 't'})).stack(time=("ensemble_member", "t"))).transpose('time', 'lat', 'lon')
    transposed_combined_psl_eth = transposed_combined_psl_eth_pre.assign_coords(time=transposed_combined_psl_eth_pre_detrend.t.values)
    
    # ERA5
    transposed_combined_psl_era5_pre = ((psl_ds_era5_pre.rename({'time': 't'})).stack(time=("ensemble_member", "t"))).isel(plev=0).transpose('time', 'lat', 'lon') 
    transposed_combined_psl_era5 = transposed_combined_psl_era5_pre.assign_coords(time=transposed_combined_psl_era5_pre.t.values)

    # ERA5 detrended with LE mean ###
    # detrend with LE ensemble mean
    slp_ensemble_mean_subset = le_ens_mean.sel(time=slice("1940","2023"))
    psl_ds_era5_detrended_pre = psl_ds_era5_pre.assign_coords(lat=slp_ensemble_mean_subset.lat, time=slp_ensemble_mean_subset.time).isel(plev=0) - slp_ensemble_mean_subset # remove ensemble mean from data, need to reassign coordinates so difference is calculated correctly

    transposed_combined_psl_era5_detrended_pre = ((psl_ds_era5_detrended_pre.rename({'time': 't'})).stack(time=("ensemble_member", "t"))).transpose('time', 'lat', 'lon') 
    transposed_combined_psl_era5_detrended = transposed_combined_psl_era5_detrended_pre.assign_coords(time=transposed_combined_psl_era5_detrended_pre.t.values)
    #################################        
    

    #%%
    #transposed_combined_psl_le.isel(time=0).plot()
    #plt.show()
    #transposed_combined_psl_eth.isel(time=0).plot()
    #plt.show()
    #transposed_combined_psl_era5.isel(time=0).plot()
    #plt.show()
    


    #%%
    #########################
    #### Calculate EOFs ####
    #########################

    # %%
    print("LE data for EOF calculation:", transposed_combined_psl_le)
    print("LE data time for EOF calculation:", transposed_combined_psl_le.time)
    print("ETH data to project onto EOFs:", transposed_combined_psl_eth)
    print("ETH detrended data to project onto EOFs:", transposed_combined_psl_eth_detrend)
    print("ERA5 data to project onto EOFs:", transposed_combined_psl_era5)
    print("ERA5 detrended data to project onto EOFs:", transposed_combined_psl_era5_detrended)
    print("PSL data loaded, now calculating EOFs")
    
    
    
    #%%
    ProgressBar(minimum=20.0).register()

    # compute EOFs
    # load solver
    # with open("/climca/people/floer/data/automated_preprocessing_13012025/streamfunction_26012026/LE_streamfunction/streamfunction_preprocessed/final_dataset_EOFs/EOFs_stream_5daily_100ensmembers_JJA_seasonal_anoms.pkl", "rb") as f:
    #         solver = pickle.load(f)
    # solver.eofs() 
    
    # compute EOFs
    solver = Eof(transposed_combined_psl_le) #create an Eof object
    print("EOFs calculated")

    # try to save solver with pickle
    try:
       with open(f"{output_path}/EOFs_z500_5daily_100ensmembers_JJA_seasonal_anoms_streamfunction.pkl", "wb") as f:
           pickle.dump(solver, f)
       print("Solver saved successfully.")
    except Exception as e:
       print(f"Error saving solver: {e}")  


    
    

    ###########################################
    #### Project LE PSL time series onto EOFs #
    ###########################################
    
    
    #%%
    # stack
    X_stacked_le_pre = transposed_combined_psl_le.stack(space=("lat", "lon")).chunk({"time": 200, "space": -1})   
    eofs_stacked = solver.eofs().isel(mode=slice(0,n_anno_indices)).stack(space=("lat", "lon")).chunk({"mode": 50, "space": -1}) 
    
    # assign same space coordinates to X_stacked_le as to eofs_stacked
    X_stacked_le = X_stacked_le_pre.assign_coords(space=eofs_stacked.space)
    

    print("Data and EOFs stacked")
    X_stacked_le 

    # project LE data onto EOFs
    pcts_le = xr.dot(X_stacked_le, eofs_stacked, dims="space") 


    #%% project ETH data onto EOFs

    # detrended
    X_stacked_eth_pre_detrend = transposed_combined_psl_eth_detrend.stack(space=("lat", "lon")).chunk({"time": 200, "space": -1})

    # assign same space coordinates to X_stacked_eth_detrend as to eofs_stacked
    X_stacked_eth_detrend = X_stacked_eth_pre_detrend.assign_coords(space=eofs_stacked.space)

    pcts_eth_detrend = xr.dot(X_stacked_eth_detrend, eofs_stacked, dims="space")


    
    # not detrended
    X_stacked_eth_pre = transposed_combined_psl_eth.stack(space=("lat", "lon")).chunk({"time": 200, "space": -1})   

    # assign same space coordinates to X_stacked_eth as to eofs_stacked
    X_stacked_eth = X_stacked_eth_pre.assign_coords(space=eofs_stacked.space)

    pcts_eth = xr.dot(X_stacked_eth, eofs_stacked, dims="space") 


    #%% project ERA5 data onto EOFs
    X_stacked_era5_pre = transposed_combined_psl_era5.stack(space=("lat", "lon")).chunk({"time": 200, "space": -1})
    
    # assign same space coordinates to X_stacked_era5 as to eofs_stacked
    X_stacked_era5 = X_stacked_era5_pre.assign_coords(space=eofs_stacked.space)
    
    pcts_era5 = xr.dot(X_stacked_era5, eofs_stacked, dims="space")


    # project ERA5 detrended onto EOFs
    X_stacked_era5_detrended_pre = transposed_combined_psl_era5_detrended.stack(space=("lat", "lon")).chunk({"time": 200, "space": -1})

    # assign same space coordinates to X_stacked_era5_detrended as to eofs_stacked
    X_stacked_era5_detrended = X_stacked_era5_detrended_pre.assign_coords(space=eofs_stacked.space)

    pcts_era5_detrended = xr.dot(X_stacked_era5_detrended, eofs_stacked, dims="space")



    #%%
    #X_stacked_le 

    #%%
    #X_stacked_eth

    #%% 
    #X_stacked_era5


    #%%
    #pcts_le.load()
    
    

    # %%

    # SAVE PC time series to disc

    # add note on ensemble member order
    pcts_le.attrs["Ensemble_member_order"] = transposed_combined_psl_le_pre.time.ensemble_member.values

    # LE PC time series
    print("saving LE PC time series to disc")
    pcts_le.to_netcdf(f"{output_path}/pseudoPCs_EOFs_z500_5daily_100ensmembers_JJA_not_scaled_no_ens_mean_subtracted.nc")

    # ETH PC time series
    print("saving ETH PC time series to disc")
    pcts_eth.to_netcdf(f"{output_path}/NOT_detrended_pseudoPCs_EOFs_z500_5daily_ETH_JJA_not_scaled_no_ens_mean_subtracted.nc")

    print("saving ETH detrended PC time series to disc")
    pcts_eth_detrend.to_netcdf(f"{output_path}/detrended_pseudoPCs_EOFs_z500_5daily_ETH_JJA_not_scaled_no_ens_mean_subtracted.nc")

    # ERA5 PC time series
    print("saving ERA5 PC time series to disc")
    pcts_era5.to_netcdf(f"{output_path}/pseudoPCs_EOFs_z500_5daily_ERA5_JJA_not_scaled_no_ens_mean_subtracted.nc")

    # ERA5 detrended PC time series
    print("saving ERA5 detrended PC time series to disc")
    pcts_era5_detrended.to_netcdf(f"{output_path}/pseudoPCs_EOFs_z500_5daily_ERA5_detrended_with_LE_mean_JJA_not_scaled_no_ens_mean_subtracted.nc")

    ###
    # save this script
    current_script = os.path.realpath(__file__)

    # timestamp string: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # output filename with timestamp
    out_file = os.path.join(
        args.output_dir,
        f"used_EOF_calculation_script_{timestamp}.py"
    )

    # copy the script
    shutil.copy(current_script, out_file)

    print(f"Script saved as: {out_file}")
    ###

if __name__ == "__main__":
    main()