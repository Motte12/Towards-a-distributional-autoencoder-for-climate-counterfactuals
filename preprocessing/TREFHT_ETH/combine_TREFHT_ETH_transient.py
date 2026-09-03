#%%
import xarray as xr
import numpy as np
import sys
sys.path.append('/home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized')
import helper_functions as hf
import argparse
#%%
def main():
    #%%
    # input arguments
    parser = argparse.ArgumentParser(description="Input arguments")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_file", type=str, help="Output .nc file")
    args = parser.parse_args()


    
    #%%
    # load ensemble members
    le_directory_path = args.input_dir  #"/climca/people/floer/data/TREFHT/ETH_ensemble/5day_subset/transient/"  # directory where all ensemble members are stored
    #chunk_shape = {'ensemble_member': 1, 'lat': 192, 'lon': 288}
    trefht_ds_le_pre2 = hf.load_data("5day", le_directory_path)#, chunk_shape=chunk_shape)
    trefht_ds_le_pre = trefht_ds_le_pre2.where(trefht_ds_le_pre2['time'].dt.month.isin([6, 7, 8]), drop=True)
    #%%
    trefht_ds_le_pre

    #%%
    stacked_trefht_pre = ((trefht_ds_le_pre.TREFHT.rename({'time': 't'})).stack(time=("ensemble_member", "t")))
    stacked_trefht = stacked_trefht_pre.assign_coords(time=stacked_trefht_pre.t.values)
    #stacked_trefht

    #%%
    #with open("/climca/people/floer/data/TREFHT/5daily_TREFHT_combined_JJA/stacked_TREFHT_JJA.pkl", "wb") as f:
    #    pickle.dump(stacked_trefht, f)
    #stacked_trefht.to_netcdf("/climca/people/floer/data/TREFHT/ETH_ensemble/5day_subset/transient/stacked_ETH_transient_TREFHT_JJA.nc")
    stacked_trefht.to_netcdf(args.output_file)

    #%%

if __name__ == "__main__":
    main()