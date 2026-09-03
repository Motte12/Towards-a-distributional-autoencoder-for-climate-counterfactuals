#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append('/home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized')
import helper_functions as hf
import pickle
#%%
def main():
    #%%
    parser = argparse.ArgumentParser(description="Extract JJA GMT and combine with Z500 pseudoPCs")
    parser.add_argument("--gmt_path", type=str, 
                        default="/climca/people/floer/data/GMT/spatial_means/gmt_ensemble_mean.nc",
                        help="Path to GMT ensemble mean data")
    parser.add_argument("--pseudo_pcs_le_path", type=str,
                        default="/climca/people/floer/data/automated_preprocessing_13012025/Z500/final_dataset/pseudoPCs_EOFs_Z500_5daily_100ensmembers_JJA_not_scaled.nc",
                        help="Path to Z500 pseudo PCs data")
    parser.add_argument("--pseudo_pcs_eth_path", type=str,
                        default=None,
                        help="Path to ETH Z500 pseudo PCs data (optional)")
    parser.add_argument("--pseudo_pcs_era5_path", type=str,
                        default=None,
                        help="Path to ERA5 Z500 pseudo PCs data (optional)")
    parser.add_argument("--output_path", type=str,
                        default="/climca/people/floer/data/GMT/spatial_means/stacked_GMT_JJA.nc",
                        help="Path to save the combined output")
    parser.add_argument("--era5_gmt_path", type=str,
                        default="/climca/people/floer/data/ERA5/TREFHT/ERA5_TREFHT_GMT/ERA5_TREFHT_GMT_anom.nc", help="Path to ERA5 GMT data")
    parser.add_argument("--era5_gmt_smoothed", type=str,
                        default="/climca/people/floer/data/ERA5/TREFHT/ERA5_TREFHT_GMT/ERA5_TREFHT_GMT_anom_smoothed_rolling100.nc", help="Path to ERA5 GMT smoothed data")
    
    args = parser.parse_args()
    
    #%%
    # load GMT 
    gmt = xr.open_dataset(args.gmt_path).TREFHT
    gmt_extracted = gmt.where(gmt['time'].dt.month.isin([6, 7, 8]), drop=True)

    # era5 GMT
    if args.pseudo_pcs_era5_path is not None:
        era5_gmt_pre = xr.open_dataset(args.era5_gmt_path).var167
        print(era5_gmt_pre)
        era5_gmt = era5_gmt_pre.where(era5_gmt_pre['time'].dt.month.isin([6, 7, 8]), drop=True).squeeze(drop=True).expand_dims(dim='mode').transpose().assign_coords({'mode': [1000]})
        print("era5 gmt:", era5_gmt)


        # era5 GMT smoothed 100
        era5_gmt_smoothed_pre = xr.open_dataset(args.era5_gmt_smoothed).var167
        print(era5_gmt_smoothed_pre)
        era5_gmt_smoothed = era5_gmt_smoothed_pre.where(era5_gmt_pre['time'].dt.month.isin([6, 7, 8]), drop=True).squeeze(drop=True).expand_dims(dim='mode').transpose().assign_coords({'mode': [1000]})
        print("era5 gmt:", era5_gmt_smoothed)

    
 

    # drop lat and lon dimensions
    gmt_squeezed = gmt_extracted.squeeze(drop=True)

    ##########
    ### LE ### 
    ##########
    # repeat the array 100 times along the time dimension
    repeated_GMT_le = xr.concat([gmt_squeezed]*100, dim="time").expand_dims(dim='mode').transpose().assign_coords({'mode': [1000]})
    
    ###########
    ### ETH ###
    ###########
    repeated_GMT_eth = xr.concat([gmt_squeezed]*3, dim="time").expand_dims(dim='mode').transpose().assign_coords({'mode': [1000]})

    ############
    ### ERA5 ###
    ############

    # for ERA5 - only create if ERA5 path is provided
    if args.pseudo_pcs_era5_path is not None:
        pseudo_pcs_era5 = xr.open_dataset(args.pseudo_pcs_era5_path).pseudo_pcs
        print("ERA5 pseudoPCs shape:", pseudo_pcs_era5.shape)
        first_year = pseudo_pcs_era5.time[0].dt.year.values
        last_year = pseudo_pcs_era5.time[-1].dt.year.values
        print("First year ERA5:", first_year)
        print("Last year ERA5:", last_year)
        repeated_GMT_ERA5 = gmt_squeezed.sel(time=slice(str(first_year), str(last_year))).expand_dims(dim='mode').transpose().assign_coords({'mode': [1000]})
        print("GMT ERA5 shape:", repeated_GMT_ERA5.shape)
        
    #%%

    # check result
    #repeated_GMT
    #plt.plot(range(4769*5), repeated_GMT.isel(time=slice(0, 5*4769)).values, label='Repeated GMT')
    

    #with open("/climca/people/floer/data/TREFHT/5daily_TREFHT_combined_JJA/stacked_TREFHT_JJA.pkl", "wb") as f:
    #    pickle.dump(stacked_trefht, f)
    #repeated_GMT.to_netcdf("/climca/people/floer/data/GMT/spatial_means/stacked_GMT_JJA.nc")
    
    # Load LE and ETH Z500 pseudoPCs 
    if args.pseudo_pcs_le_path is not None:
        pseudo_pcs_le = xr.open_dataset(args.pseudo_pcs_le_path).pseudo_pcs

    if args.pseudo_pcs_eth_path is not None:
        pseudo_pcs_eth = xr.open_dataset(args.pseudo_pcs_eth_path).pseudo_pcs
    
    
    # concatenate GMT as last column
    if args.pseudo_pcs_le_path is not None:
        predictors_concatenated_le = xr.concat([pseudo_pcs_le, repeated_GMT_le], dim='mode')
    if args.pseudo_pcs_eth_path is not None:
        predictors_concatenated_eth = xr.concat([pseudo_pcs_eth, repeated_GMT_eth], dim='mode')
    if args.pseudo_pcs_era5_path is not None:
        print(pseudo_pcs_era5)
        print(repeated_GMT_ERA5)
        predictors_concatenated_era5 = xr.concat([pseudo_pcs_era5, repeated_GMT_ERA5.assign_coords({'time':pseudo_pcs_era5.time})], dim='mode')
        predictors_concatenated_era5_era5GMT = xr.concat([pseudo_pcs_era5, era5_gmt.assign_coords({'time':pseudo_pcs_era5.time})], dim='mode')
        predictors_concatenated_era5_era5GMT_smoothed = xr.concat([pseudo_pcs_era5, era5_gmt_smoothed.assign_coords({'time':pseudo_pcs_era5.time})], dim='mode')

        
   

    # save all 3 datasets to output directory
    if args.pseudo_pcs_le_path is not None:
        predictors_concatenated_le.to_netcdf(f"{args.output_path}/pcsts_and_fGMT_not_standardized_LE.nc")
    if args.pseudo_pcs_eth_path is not None:
        predictors_concatenated_eth.to_netcdf(f"{args.output_path}/pcsts_and_fGMT_not_standardized_ETH.nc")
    if args.pseudo_pcs_era5_path is not None:
        predictors_concatenated_era5.to_netcdf(f"{args.output_path}/pcsts_and_fGMT_not_standardized_ERA5_detrended_smooth_from_idx1000.nc")
        predictors_concatenated_era5_era5GMT.to_netcdf(f"{args.output_path}/pcsts_and_ERA5GMT_not_standardized_ERA5_detrended_smooth_from_idx1000.nc")
        predictors_concatenated_era5_era5GMT_smoothed.to_netcdf(f"{args.output_path}/pcsts_and_ERA5GMTsmoothed_not_standardized_ERA5_detrended_smooth_from_idx1000.nc")


if __name__ == "__main__":
    main()