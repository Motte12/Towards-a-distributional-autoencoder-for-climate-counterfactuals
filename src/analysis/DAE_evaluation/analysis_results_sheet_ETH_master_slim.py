import torch
import xarray as xr
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np
from datetime import datetime
import argparse

import sys
import os
sys.path.append('../../utils')
import dpa_ensemble as de
import utils as ut
import evaluation


def log_print(log_path, message):
    print(message) 
    with open(log_path, "a") as f:
        print(message, file=f)  


def main():
    print("script running")
    parser = argparse.ArgumentParser(description="Example script with arguments")
    
    parser.add_argument("--include_train_analysis", type=int, default=0, help="Whether to include analysis of train data.")
    parser.add_argument("--period_start", type=int, help="Start year of period to analyse")
    parser.add_argument("--period_end", type=int, help="End year of period to analyse")
    parser.add_argument("--ensemble_path", type=str, help="Path of DPA ensemble")
    parser.add_argument("--no_epochs", type=int, help="Number of epochs model was trained used for creating this DPA ensemble")
    parser.add_argument("--ens_members", type=int, default=100, help="Number of members in DPA ensemble")
    parser.add_argument("--save_path_le", type=str, help="Save path of LE train set analysis figures")
    parser.add_argument("--save_path_eth", type=str, help="Save path of ETH set analysis figures")
    parser.add_argument("--settings_file_path", type=str, help="Path of settings (datasets) to create ensemble.")
    parser.add_argument("--no_test_members", type=int, default=3, help="Number of members in the test set.")
    parser.add_argument("--calculate_e_loss_per_ti", type=int, default=1, help="Whether to calculate energy loss per time step.")
    parser.add_argument("--StoNet_ensemble", type=int, default=0, help="Whether to evaluate StoNet ensemble.")
    parser.add_argument("--eval_ERA5", type=int, default=0, help="Whether to evaluate ERA5.")
    parser.add_argument("--domain", type=str, default="FR", help="Domain to use for True-Pred scatterplot.")
    args = parser.parse_args()
    
    time_period = [str(args.period_start), str(args.period_end)]
    
    ens_members=args.ens_members
    no_epochs = args.no_epochs

    # Paths
    # ensemble path
    if args.eval_ERA5:
        ensemble_path = f"{args.ensemble_path}"
    else:
        ensemble_path = f"{args.ensemble_path}ETH_ensemble_after_{no_epochs}_epochs"

    # set save path
    if args.save_path_eth is not None:
        print("save path eth is given")
        save_path_eth = f"{args.save_path_eth}/period_{time_period[0]}_{time_period[1]}"
    else:
        print("save path eth is not given")
        save_path_eth = f"ETH_analysis_results/final_analysis_test_ETH/model_trained_for_{args.no_epochs}_epochs/period_{time_period[0]}_{time_period[1]}"

    # create save paths
    os.makedirs(save_path_eth, exist_ok=True)
    os.makedirs(f"{save_path_eth}/quantiles", exist_ok=True)
    os.makedirs(f"{save_path_eth}/data", exist_ok=True)
    
    print("save path ETH analysis results:", save_path_eth)
    print("include LE train analysis:", args.include_train_analysis)
    print("ensemble load path:", ensemble_path)
    
    # Log file
    #log_file = f"{save_path_eth}/test_log_metrics_{time_period[0]}-{time_period[1]}.txt"
    
    # Get current time and print it
    #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #log_print(log_file, f"=== Current Time: {current_time} ===")
    #log_print(log_file, f"=== Quantiles ===")

 
    # Plotting settings
    title_fontsize = 18
    figsize_map = (10,8)
    figsize_ts = (10,8)
    figsize_hist = (8,6)


    #################
    ### Load Data ###
    #################
    
    # Large Ensemble Data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced, pi_period_mean, _, _ = de.load_test_data(args.settings_file_path)
    print(ds)

    # ERA5 data
    if args.eval_ERA5:
        z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf, _, _ = de.load_era5_test_data(args.settings_file_path)
        
    # ETH Ensemble Test data
    else:
        z500, mask_x_te_eth_fact, ds_test_eth_fact, ds_test_eth_cf, x_te_reduced_eth_fact, x_te_reduced_eth_cf, _, _ = de.load_eth_test_data(args.settings_file_path)
    # z500                  -> test predictors
    # mask_x_te_eth_fact    -> land mask
    # ds_test_eth_fact      -> factual test temperatures (xarray dataset) lat: 32, lon: 32, time: 14307
    # x_te_reduced_eth_fact -> land grid cells factual temperature data
    # x_te_reduced_eth_cf   -> land grid cells counterfactual temperature data

    print("x_te_reduced_eth_fact:", x_te_reduced_eth_fact.shape)
    
    slice_end_index = int(x_te_reduced_eth_fact.shape[0]/args.no_test_members)
    print("Slice end index:", slice_end_index)
    
    # datasets
    ds_test_1300_eth_fact = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    ds_test_1300_eth_cf = ds_test_eth_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))

    # get indices of individual test time slices
    time_index = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).get_index("time")
    #print("Time index:", time_index)
    indices = time_index.get_indexer(ds_test_1300_eth_fact.time.values)
    start_idx, end_idx = indices[0], indices[-1]+1 # add 1 to include last index
    start_idx_1400, end_idx_1400 = end_idx, 2*end_idx
    start_idx_1500, end_idx_1500 = 2*end_idx, 3*end_idx
    
    print("Start index:", start_idx)
    print("End index:", end_idx)
    print("Start index 1400:", start_idx_1400)
    print("End index 1400:", end_idx_1400)
    print("Start index 1500:", start_idx_1500)
    print("End index 1500:", end_idx_1500)

    
    #################
    ### Test Data ###
    #################
    
    # PYTORCH arrays
    # Factual Test/True temperatures
    eth_fact_1300_test_reduced = x_te_reduced_eth_fact[:slice_end_index,:][start_idx:end_idx,:] # HERE
    eth_fact_1400_test_reduced = x_te_reduced_eth_fact[slice_end_index:2*slice_end_index,:][start_idx:end_idx,:]
    eth_fact_1500_test_reduced = x_te_reduced_eth_fact[-slice_end_index:14307,:][start_idx:end_idx,:]
    print("eth_fact_1300_test_reduced shape:", eth_fact_1300_test_reduced.shape)
    mask_x_te = mask_x_te_eth_fact

    # Counterfactual
    # Factual Test/True temperatures
    eth_cf_1300_test_reduced = x_te_reduced_eth_cf[:slice_end_index,:][start_idx:end_idx,:] # HERE
    eth_cf_1400_test_reduced = x_te_reduced_eth_cf[slice_end_index:2*slice_end_index,:][start_idx:end_idx,:]
    eth_cf_1500_test_reduced = x_te_reduced_eth_cf[-slice_end_index:14307,:][start_idx:end_idx,:]
    print("eth_fact_1300_test_reduced counterfactual shape:", eth_cf_1300_test_reduced.shape)
    
    #########################
    ### Load DAE Ensemble ###
    #########################
    
    
    print("DPA ensemble load paths:")
    # shape: ensemble_member: 100, time: 14307, lat_x_lon: 648
    print(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
    print(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    print(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
    print(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
        
    # load DAE ensembles
    # factual
    dpa_ensemble_fact_raw = xr.open_dataset(f"{ensemble_path}/raw_ETH_gen_dpa_ens_{no_epochs}_dataset.nc")
    dpa_ensemble_fact_restored = xr.open_dataset(f"{ensemble_path}/ETH_gen_dpa_ens_{no_epochs}_dataset_restored.nc")
    
    # counterfactual
    dpa_ensemble_raw_cf = xr.open_dataset(f"{ensemble_path}/raw_ETH_cf_gen_dpa_ens_{no_epochs}_dataset.nc")
    dpa_ensemble_restored_cf = xr.open_dataset(f"{ensemble_path}/ETH_cf_gen_dpa_ens_{no_epochs}_dataset_restored.nc")

    # subset to individual test members (1300, 1400, 1500)
    # FACTUAL
    dpa_1300_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    

    # shape: ensemble_member: 100, time: 14307, lat: 32, lon: 32
    dpa_1300_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_fact_restored = dpa_ensemble_fact_restored.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    

    # COUNTERFACTUAL
    dpa_1300_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_cf_raw = dpa_ensemble_raw_cf.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))

    dpa_1300_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_cf_restored = dpa_ensemble_restored_cf.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    
    

    #####################
    ### Scatter data ####
    #####################
    print("#################")
    print("### Test data ###")
    print("#################")
    print("ds_test_eth_fact:", ds_test_eth_fact)
    print("ds_test_eth_cf:", ds_test_eth_cf)

    print("################")
    print("### DAE data ###")
    print("################")
    print("dpa_ensemble_fact_restored:", dpa_ensemble_fact_restored)
    print("dpa_ensemble_restored_cf:", dpa_ensemble_restored_cf)

    ###########################
    ### Domain spatial mean ###
    ###########################
    if True:
        
        # Domain 
        if args.domain == "GER":
            # GER
            lat_min = 48
            lat_max = 54
            lon_min = 6
            lon_max = 15
    
        elif args.domain == "FR":
            # FR
            lat_min = 45
            lat_max = 50
            lon_min= 0
            lon_max= 5
    
        elif args.domain == "SP":
            lat_min = 38
            lat_max = 42
            lon_min = -8
            lon_max = 0
    
        # --- Factual ---
        true_fact_mean = ut.get_ger_1d_data(ds_test_eth_fact["TREFHT"], lat_min, lat_max, lon_min, lon_max)          # dims: (time,)
        dae_fact_mean = ut.get_ger_1d_data(dpa_ensemble_fact_restored["TREFHT"], lat_min, lat_max, lon_min, lon_max) # dims: (ensemble_member, time)
        
        # --- Counterfactual ---
        true_cf_mean = ut.get_ger_1d_data(ds_test_eth_cf["TREFHT"], lat_min, lat_max, lon_min, lon_max)
        dae_cf_mean = ut.get_ger_1d_data(dpa_ensemble_restored_cf["TREFHT"], lat_min, lat_max, lon_min, lon_max)
        
        # take DAE ensemble median across members to compare against single truth
        dae_fact_mean_ensmean = dae_fact_mean.median(dim="ensemble_member", skipna=True)
        dae_cf_mean_ensmean = dae_cf_mean.median(dim="ensemble_member", skipna=True)

        # save data
        ds_domain = xr.Dataset(
            {
                "true_fact_mean": ("time", true_fact_mean.values),
                "dae_fact_mean_ensmedian": ("time", dae_fact_mean_ensmean.values),
                "true_cf_mean": ("time_cf", true_cf_mean.values),
                "dae_cf_mean_ensmedian": ("time_cf", dae_cf_mean_ensmean.values),
            },
            coords={
                "time": true_fact_mean.time.values,
                "time_cf": true_cf_mean.time.values,
            },
            attrs={
                "domain": args.domain,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
        )
        
        #ds_domain.to_netcdf(f"{save_path_eth}/data/domain_mean_{args.domain}.nc")
        
        # --- Scatter plot: true vs predicted ---
        fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.scatter(true_fact_mean.values, dae_fact_mean_ensmean.values,
                   alpha=0.5, s=15, color="tab:orange", label="Factual")
        ax.scatter(true_cf_mean.values, dae_cf_mean_ensmean.values,
                   alpha=0.5, s=15, color="tab:blue", label="Counterfactual")
        
        # 1:1 reference line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel("True domain mean (CESM2)")
        ax.set_ylabel("Predicted domain mean (DAE ensemble mean)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        
        plt.tight_layout()
        #plt.savefig(f"true_vs_predicted_scatter_{args.domain}_domain.pdf")
        
    #############################
    ### Individual grid cells ###
    #############################

    if True:
        def ensemble_median_nan_safe(da, dim="ensemble_member"):
            """Compute median across ensemble members, skipping NaNs."""
            return da.median(dim=dim, skipna=True)
        
        # --- Factual ---
        dae_fact_median = ensemble_median_nan_safe(dpa_ensemble_fact_restored["TREFHT"])  # dims: (lat, lon, time)
        true_fact = ds_test_eth_fact["TREFHT"].transpose("time", "lat", "lon").assign_coords(time=dae_fact_median.time)                      # dims: (lat, lon, time)
        print("dae_fact_median.shape:", dae_fact_median.shape)
        
        # --- Counterfactual ---
        true_cf = ds_test_eth_cf["TREFHT"]
        dae_cf_median = ensemble_median_nan_safe(dpa_ensemble_restored_cf["TREFHT"])
        print("dae_cf_median.shape", dae_cf_median.shape)
        
        # --- Flatten to 1D, dropping NaNs pairwise ---
        def flatten_pair_dropna(true_da, pred_da):
            t = true_da.values.ravel()
            p = pred_da.values.ravel()
            mask = ~np.isnan(t) & ~np.isnan(p)
            return t[mask], p[mask]
        
        true_fact_flat, dae_fact_flat = flatten_pair_dropna(true_fact, dae_fact_median)
        true_cf_flat, dae_cf_flat = flatten_pair_dropna(true_cf, dae_cf_median)
    
    
        def r2_per_gridcell(true_da, pred_da):
            """
            Compute R^2 per grid cell (lat, lon), reducing over the time dimension.
            Returns a 2D DataArray of shape (lat, lon).
            """
            residual = true_da - pred_da
            ss_res = (residual ** 2).sum(dim="time", skipna=True)
            ss_tot = ((true_da - true_da.mean(dim="time", skipna=True)) ** 2).sum(dim="time", skipna=True)
            r2 = 1 - ss_res / ss_tot
            return r2
        
        # --- Compute per-gridcell R^2 (using factual data) ---
        print("true_fact:", true_fact)
        print("dae_fact_median:", dae_fact_median)
        r2_map = r2_per_gridcell(true_fact, dae_fact_median)  # dims: (lat, lon)
        print("r2_map shape:", r2_map.shape)
        
        r2_flat = r2_map.values.ravel()
        valid_mask = ~np.isnan(r2_flat)
        r2_valid = r2_flat[valid_mask]
        
        # --- Find R^2 values at 20th and 50th percentiles ---
        p20_val = np.percentile(r2_valid, 20)
        p50_val = np.percentile(r2_valid, 50)
        
        # find the grid cell whose R^2 is CLOSEST to each target percentile value
        def closest_gridcell_to_value(r2_map, target_value):
            diff = np.abs(r2_map.values - target_value)
            diff_flat = diff.ravel()
            diff_flat[np.isnan(diff_flat)] = np.inf  # ignore NaNs
            idx_flat = np.argmin(diff_flat)
            idx_2d = np.unravel_index(idx_flat, r2_map.shape)  # (lat_idx, lon_idx)
            return idx_2d

        # grid cells at 20th and 50th percentiles
        idx_p20 = closest_gridcell_to_value(r2_map, p20_val)
        idx_p50 = closest_gridcell_to_value(r2_map, p50_val)
        
        # best grid cell = argmax R^2
        r2_no_nan = np.nan_to_num(r2_map.values, nan=-np.inf)
        idx_best = np.unravel_index(np.argmax(r2_no_nan), r2_map.shape)
        
        # get lat/lon coordinate values for labeling
        def latlon_at_idx(idx):
            lat_val = r2_map.lat.values[idx[0]]
            lon_val = r2_map.lon.values[idx[1]]
            return lat_val, lon_val
        
        lat_p20, lon_p20 = latlon_at_idx(idx_p20)
        lat_p50, lon_p50 = latlon_at_idx(idx_p50)
        lat_best, lon_best = latlon_at_idx(idx_best)
        
        print(f"20th pct grid cell: lat={lat_p20:.2f}, lon={lon_p20:.2f}, R²={r2_map.values[idx_p20]:.3f}")
        print(f"50th pct grid cell: lat={lat_p50:.2f}, lon={lon_p50:.2f}, R²={r2_map.values[idx_p50]:.3f}")
        print(f"Best grid cell:     lat={lat_best:.2f}, lon={lon_best:.2f}, R²={r2_map.values[idx_best]:.3f}")
        
        # --- Extract true/predicted time series (factual) at each selected grid cell ---
        def gridcell_series(true_da, pred_da, idx):
            t = true_da.isel(lat=idx[0], lon=idx[1]).values
            p = pred_da.isel(lat=idx[0], lon=idx[1]).values
            mask = ~np.isnan(t) & ~np.isnan(p)
            return t[mask], p[mask]
        
        t_p20, p_p20 = gridcell_series(true_fact, dae_fact_median, idx_p20)
        t_p50, p_p50 = gridcell_series(true_fact, dae_fact_median, idx_p50)
        t_best, p_best = gridcell_series(true_fact, dae_fact_median, idx_best)
        
        # --- Extract true/predicted time series (counterfactual) at the SAME grid cells ---
        t_p20_cf, p_p20_cf = gridcell_series(true_cf, dae_cf_median, idx_p20)
        t_p50_cf, p_p50_cf = gridcell_series(true_cf, dae_cf_median, idx_p50)
        t_best_cf, p_best_cf = gridcell_series(true_cf, dae_cf_median, idx_best)

        # save data
        ds_out = xr.Dataset(
            {
                "true_fact_p20": ("obs_p20", t_p20),
                "pred_fact_p20": ("obs_p20", p_p20),
                "true_fact_p50": ("obs_p50", t_p50),
                "pred_fact_p50": ("obs_p50", p_p50),
                "true_fact_best": ("obs_best", t_best),
                "pred_fact_best": ("obs_best", p_best),
        
                "true_cf_p20": ("obs_p20_cf", t_p20_cf),
                "pred_cf_p20": ("obs_p20_cf", p_p20_cf),
                "true_cf_p50": ("obs_p50_cf", t_p50_cf),
                "pred_cf_p50": ("obs_p50_cf", p_p50_cf),
                "true_cf_best": ("obs_best_cf", t_best_cf),
                "pred_cf_best": ("obs_best_cf", p_best_cf),
            },
            coords={
                "lat_p20": lat_p20, "lon_p20": lon_p20, "r2_p20": r2_map.values[idx_p20],
                "lat_p50": lat_p50, "lon_p50": lon_p50, "r2_p50": r2_map.values[idx_p50],
                "lat_best": lat_best, "lon_best": lon_best, "r2_best": r2_map.values[idx_best],
            },
        )
        
        ds_out.to_netcdf(f"{save_path_eth}/data/percentile_selected_gridcells_fact_cf.nc")


    ########################
    ### End scatter data ###
    ########################
    
    # ensemble mean of restored factual DPA ensemble
    #dpa_ens_mean_restored = dpa_ensemble_fact_restored.TREFHT.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1300_restored = dpa_1300_fact_restored.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_restored = dpa_1400_fact_restored.mean(dim="ensemble_member")
        dpa_ens_mean_fact_1500_restored = dpa_1500_fact_restored.mean(dim="ensemble_member")

    # ensemble mean of restored counterfactual DPA ensemble
    dpa_ens_mean_cf_1300_restored = dpa_1300_cf_restored.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_restored = dpa_1400_cf_restored.mean(dim="ensemble_member")
        dpa_ens_mean_cf_1500_restored = dpa_1500_cf_restored.mean(dim="ensemble_member")

    # mean of raw factual ensemble
    dpa_ens_mean_fact_1300_raw = dpa_1300_fact_raw.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_raw = dpa_1400_fact_raw.mean(dim="ensemble_member")
        dpa_ens_mean_fact_1500_raw = dpa_1500_fact_raw.mean(dim="ensemble_member")

    # mean of raw counterfactual ensemble
    dpa_ens_mean_cf_1300_raw = dpa_1300_cf_raw.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_raw = dpa_1400_cf_raw.mean(dim="ensemble_member")
        dpa_ens_mean_cf_1500_raw = dpa_1500_cf_raw.mean(dim="ensemble_member")

    
    dpa_ens_mean_fact_1300_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1300_raw.values) #dpa_ens_mean_pt
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1400_raw.values)
        dpa_ens_mean_fact_1500_raw_pt = torch.from_numpy(dpa_ens_mean_fact_1500_raw.values)


    dpa_ens_mean_cf_1300_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1300_raw.values) #dpa_ens_mean_pt
    if args.no_test_members > 1:
        dpa_ens_mean_cf_1400_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1400_raw.values)
        dpa_ens_mean_cf_1500_raw_pt = torch.from_numpy(dpa_ens_mean_cf_1500_raw.values)


    ###################
    ### Calibration ###
    ###################
    # quantiles to evaluate
    quantiles_cq = torch.linspace(0.01, 0.99, 99)

    ### Factual ###
    if True:
        through = zip([eth_fact_1300_test_reduced], [dpa_1300_fact_raw])
        if args.no_test_members > 1:
            through = zip([eth_fact_1300_test_reduced, eth_fact_1400_test_reduced, eth_fact_1500_test_reduced], [dpa_1300_fact_raw.values, dpa_1400_fact_raw.values, dpa_1500_fact_raw.values])
        
        memb = 0
        for y_test_np, dpa_xxxx_fact_raw in through:
            #mae_list = []
            #mae099_list = []
            member_gc_coverages = np.zeros((648,99))
            # iterate through grid-cells
            for i in range(648):
                print(i)
                
                # compute DAE quantiles
                quantile_predictions_dpa = np.quantile(dpa_xxxx_fact_raw[:,:,i].T, np.linspace(0.01, 0.99, 99), axis=1).T
                
                # compute coverage per quantile
                cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)
                
                # save cq per grid-cell
                member_gc_coverages[i,:] = cover_dpa

            # save results per member
            np.save(f"{save_path_eth}/data/cq_spatial_test_member{memb}_factual_01-99.npy", member_gc_coverages)
            memb += 1
            
    
    ### Counterfactual ###
    print("Evaluating counterfactual calibration ...")
    through_cf = zip([eth_cf_1300_test_reduced], [dpa_1300_cf_raw])
    if args.no_test_members > 1:
        through_cf = zip([eth_cf_1300_test_reduced, eth_cf_1400_test_reduced, eth_cf_1500_test_reduced], [dpa_1300_cf_raw.values, dpa_1400_cf_raw.values, dpa_1500_cf_raw.values])
    memb = 0

    for y_test_np, dpa_xxxx_cf_raw in through_cf:
        member_gc_coverages_cf = np.zeros((648,99))
        for i in range(648):
            print(i)

            # compute DAE quantiles
            quantile_predictions_dpa = np.quantile(dpa_xxxx_cf_raw[:,:,i].T, np.linspace(0.01, 0.99, 99), axis=1).T

            # compute coverage per quantile
            cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)

            # save cq per grid-cell
            member_gc_coverages_cf[i,:] = cover_dpa
        
        # save results per member
        np.save(f"{save_path_eth}/data/cq_spatial_test_member{memb}_counterfactual_01-99.npy", member_gc_coverages_cf)
        memb += 1

    
    ###########
    ### MAE ###
    ###########

    mae_1300_fact = evaluation.mae_cols(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)
    if args.no_test_members > 1:
        mae_1400_fact = evaluation.mae_cols(eth_fact_1400_test_reduced, dpa_ens_mean_fact_1400_raw_pt, dim=0)
        mae_1500_fact = evaluation.mae_cols(eth_fact_1500_test_reduced, dpa_ens_mean_fact_1500_raw_pt, dim=0)

    mae_1300_cf = evaluation.mae_cols(eth_cf_1300_test_reduced, dpa_ens_mean_cf_1300_raw_pt, dim=0)
    if args.no_test_members > 1:
        mae_1400_cf = evaluation.mae_cols(eth_cf_1400_test_reduced, dpa_ens_mean_cf_1400_raw_pt, dim=0)
        mae_1500_cf = evaluation.mae_cols(eth_cf_1500_test_reduced, dpa_ens_mean_cf_1500_raw_pt, dim=0)

    # concatenate
    if args.no_test_members > 1:
        all_mae_fact = torch.cat((mae_1300_fact, mae_1400_fact, mae_1500_fact), dim=0)
        all_mae_cf = torch.cat((mae_1300_cf, mae_1400_cf, mae_1500_cf), dim=0)
    else:
        all_mae_fact = mae_1300_fact.clone()
        all_mae_cf = mae_1300_cf.clone()

    # save MAE arrays for later evaluation
    torch.save(all_mae_fact, f"{save_path_eth}/data/all_mae_fact.pt")
    torch.save(all_mae_cf, f"{save_path_eth}/data/all_mae_cf.pt")
    

if __name__ == "__main__":
    main()
    
