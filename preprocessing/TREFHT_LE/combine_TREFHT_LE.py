
#%%
import xarray as xr
import numpy as np
import sys
import argparse
import pickle

# Add your helper module to path
sys.path.append('/home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized')
import helper_functions as hf

#%%
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process and stack TREFHT data.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing the ensemble data')
    parser.add_argument('--output_file', type=str, required=True, help='Output file path for stacked NetCDF')

    args = parser.parse_args()

    le_directory_path = args.input_dir
    output_file_path = args.output_file

    #%%
    # Load ensemble members
    chunk_shape = {'ensemble_member': 1, 'lat': 192, 'lon': 288}
    trefht_ds_le_pre2 = hf.load_data("5day", le_directory_path, chunk_shape=chunk_shape)

    # Select JJA months
    trefht_ds_le_pre = trefht_ds_le_pre2.where(trefht_ds_le_pre2['time'].dt.month.isin([6, 7, 8]), drop=True)

    #%%
    # Stack the data
    stacked_trefht_pre = (
        trefht_ds_le_pre.TREFHT.rename({'time': 't'})
        .stack(time=("ensemble_member", "t"))
    )
    stacked_trefht = stacked_trefht_pre.assign_coords(time=stacked_trefht_pre.t.values)

    #%%
    # Save to NetCDF
    stacked_trefht.to_netcdf(output_file_path)

#%%
if __name__ == "__main__":
    main()
