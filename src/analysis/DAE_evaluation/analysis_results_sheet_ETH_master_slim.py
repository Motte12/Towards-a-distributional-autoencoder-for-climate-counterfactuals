import torch
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from engression.models import StoNet, StoLayer
from engression.loss_func import energy_loss, energy_loss_two_sample

import xarray as xr
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.manifold import TSNE
import numpy as np
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import shutil
from datetime import datetime
import argparse
from matplotlib.patches import Patch

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


    args = parser.parse_args()
    
    time_period = [str(args.period_start), str(args.period_end)]
    
    no_epochs = args.no_epochs
    
    ensemble_path = f"{args.ensemble_path}ETH_ensemble_after_{no_epochs}_epochs"

    # save path
    if args.save_path_le is not None:
        print("save path LE is given")
        save_path_le = args.save_path_le
    else:
        print("save path LE is not given")
        save_path_le = f"ETH_analysis_results/final_analysis_train_LE/model_trained_for_{args.no_epochs}_epochs"
        

    if args.save_path_eth is not None:
        print("save path eth is given")
        save_path_eth = f"{args.save_path_eth}/period_{time_period[0]}_{time_period[1]}"
    else:
        print("save path eth is not given")
        save_path_eth = f"ETH_analysis_results/final_analysis_test_ETH/model_trained_for_{args.no_epochs}_epochs/period_{time_period[0]}_{time_period[1]}"

        

    print("save path ETH analysis results:", save_path_eth)
    print("save path LE analysis results:", save_path_le)
    print("include LE train analysis:", args.include_train_analysis)
    print("ensemble load path:", ensemble_path)
    
    
    os.makedirs(save_path_eth, exist_ok=True)
    os.makedirs(save_path_le, exist_ok=True)

    
    log_file = f"{save_path_eth}/test_log_metrics_{time_period[0]}-{time_period[1]}.txt"
    
    # Get current time and print it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(log_file, f"=== Current Time: {current_time} ===")
    log_print(log_file, f"=== Quantiles ===")
    #log_print(log_file, f"\n")


    print(f"{save_path_eth}/Germany")
    print(f"{save_path_eth}/Spain")
    
    # create germany and spain subdirs––±
    os.makedirs(f"{save_path_eth}/Germany", exist_ok=True)
    os.makedirs(f"{save_path_eth}/Spain", exist_ok=True)
    os.makedirs(f"{save_path_eth}/quantiles", exist_ok=True)
    os.makedirs(f"{save_path_eth}/data", exist_ok=True)
    
    
 
    # plotting settings
    title_fontsize = 18
    figsize_map = (10,8)
    figsize_ts = (10,8)
    figsize_hist = (8,6)

    #input_dim = (32, 32)


    # years for time series plotting
    years = ["2000", "2020", "2040", "2060"]#, "2000", "2020", "2040", "2060"]
    # subset to years that are contained in time period!!
    
    # Convert to integers
    start, end = map(int, time_period)
    
    # Filter years that fall within the range
    years = [y for y in map(int, years) if start <= y <= end]
    
    #print("years contained", years)
    
    ens_members=args.ens_members

    
    #################
    ### Load Data ###
    #################
    # create figures page
    #fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # 2x2 grid of subplots

    # load test data
    #print("Loading test data ...")
    
    # Large Ensemble Data
    z500_test, z500_train, mask_x_te, ds, ds_train, ds_test, x_te_reduced, x_tr_reduced, pi_period_mean, _, _ = de.load_test_data(args.settings_file_path)
    print(ds)
    #print("x_te_reduced shape:", x_te_reduced.shape)

    # ETH Ensemble Test data
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
    ds_test_1300_eth_fact = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1])) # HERE TP
    #print("ds_test_1300_eth_fact:", ds_test_1300_eth_fact)
    ds_test_1300_eth_cf = ds_test_eth_cf.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1])) # HERE TP

    # get indices of time slices
    time_index = ds_test_eth_fact.TREFHT.isel(time=slice(0, slice_end_index)).get_index("time")
    #print("Time index:", time_index)
    indices = time_index.get_indexer(ds_test_1300_eth_fact.time.values)
    start_idx, end_idx = indices[0], indices[-1]+1 # add 1 to include last index
    start_idx_1400, end_idx_1400 = end_idx, 2*end_idx
    start_idx_1500, end_idx_1500 = 2*end_idx, 3*end_idx
    
    
    #if time_period[-1] == "2100":
    #    end_idx = indices[-1]
    
    #############################

    print("Start index:", start_idx)
    print("End index:", end_idx)

    print("Start index 1400:", start_idx_1400)
    print("End index 1400:", end_idx_1400)

    print("Start index 1500:", start_idx_1500)
    print("End index 1500:", end_idx_1500)

    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[start_idx])
    #print(ds_test_eth_fact.TREFHT.isel(time=slice(0, 4769)).time[end_idx])

    
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
    ### Load DPA Ensemble ###
    #########################
    
    # including nan's
    #dpa_ensemble = xr.open_zarr(f"{save_path_ensemble_single}/dpa_ens_100_dataset_restored.zarr", consolidated=True)
    
    # RAW ensemble without NaNs
    # shape: ensemble_member: 100time: 64000lat_x_lon: 648
    print("Loading DPA ensemble ...")
    print("DPA ensemble load paths:")

    # FACTUAL 
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

    # subset to individual test members
    # FACTUAL
    dpa_1300_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(0, slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
    if args.no_test_members > 1:
        dpa_1400_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(slice_end_index,2*slice_end_index)).sel(time=slice(time_period[0], time_period[1]))
        dpa_1500_fact_raw = dpa_ensemble_fact_raw.TREFHT.isel(time=slice(-slice_end_index,14307)).sel(time=slice(time_period[0], time_period[1]))
    #print("dpa_1300_fact_raw:", dpa_1300_fact_raw)
    

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
    
    
    #############
    #############
    #############
    
    # mean of restored factual DPA ensemble
    #dpa_ens_mean_restored = dpa_ensemble_fact_restored.TREFHT.mean(dim="ensemble_member")
    dpa_ens_mean_fact_1300_restored = dpa_1300_fact_restored.mean(dim="ensemble_member")
    if args.no_test_members > 1:
        dpa_ens_mean_fact_1400_restored = dpa_1400_fact_restored.mean(dim="ensemble_member")
        dpa_ens_mean_fact_1500_restored = dpa_1500_fact_restored.mean(dim="ensemble_member")

    # mean of restored counterfactual DPA ensemble
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
    quantiles_cq = torch.linspace(0.01, 0.99, 99)
    quantiles_cq_np = np.linspace(0.01, 0.99, 99)


    ### Factual ###
    if True:
        mae_means = []
        mae099_means = []
        coverages_all_membs = np.zeros((args.no_test_members,99))
    
        through = zip([eth_fact_1300_test_reduced], [dpa_1300_fact_raw])
        if args.no_test_members > 1:
            through = zip([eth_fact_1300_test_reduced, eth_fact_1400_test_reduced, eth_fact_1500_test_reduced], [dpa_1300_fact_raw.values, dpa_1400_fact_raw.values, dpa_1500_fact_raw.values])
        memb = 0
        all_gc_coverages_fact = {}
        for y_test_np, dpa_xxxx_fact_raw in through:
            mae_list = []
            mae099_list = []
            member_gc_coverages = np.zeros((648,99))
            for i in range(648):
                print(i)
                print("Shapes truth, ensemble:", y_test_np[:,i].shape,dpa_xxxx_fact_raw[:,:,i].shape)
                quantile_predictions_dpa = np.quantile(dpa_xxxx_fact_raw[:,:,i].T, np.linspace(0.01, 0.99, 99), axis=1).T

                cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)
                # save cq per grid-cell
                member_gc_coverages[i,:] = cover_dpa

            ##############################
            np.save(f"{save_path_eth}/data/cq_spatial_test_member{memb}_factual_01-99.npy", member_gc_coverages)
            all_gc_coverages_fact[str(memb)] = member_gc_coverages
            spat_mean_coverage = np.mean(member_gc_coverages, axis=0) # compute mean coverage over all grid-cells
            coverages_all_membs[memb, :] = spat_mean_coverage
            memb += 1
            spat_mean_mae = np.mean(mae_list) 
            spat_mean_099 = np.mean(mae099_list)
            mae_means.append(spat_mean_mae)
            mae099_means.append(spat_mean_099)

        # compute mean coverage across members from: coverages_all_membs
        # mean of: mean of spatial covergae per grid-cell, per member
        mean_coverage_membs_gcs = np.mean(coverages_all_membs, axis = 0)
        print("mean_coverage_membs_gcs shape:", mean_coverage_membs_gcs.shape)
        mae_membs_gcs = np.mean(np.abs(quantiles_cq_np - mean_coverage_membs_gcs))
    
        log_print(log_file, f"factual mean calibration CQ MAE: {mae_membs_gcs}")
        log_print(log_file, f"coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae_means)}")
        log_print(log_file, f"095 coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae099_means)}")
    
        

    

    
    ### Counterfactual ###
    mae_means_cf = []
    mae099_means_cf = []
    coverages_all_membs_cf = np.zeros((args.no_test_members,99))
    
    through_cf = zip([eth_cf_1300_test_reduced], [dpa_1300_cf_raw])
    if args.no_test_members > 1:
        through_cf = zip([eth_cf_1300_test_reduced, eth_cf_1400_test_reduced, eth_cf_1500_test_reduced], [dpa_1300_cf_raw.values, dpa_1400_cf_raw.values, dpa_1500_cf_raw.values])
    memb = 0
    all_gc_coverages_cf = {}
    for y_test_np, dpa_xxxx_cf_raw in through_cf:
        mae_list = []
        mae099_list = []
        member_gc_coverages_cf = np.zeros((648,99))
        for i in range(648):
            print(i)
            quantile_predictions_dpa = np.quantile(dpa_xxxx_cf_raw[:,:,i].T, np.linspace(0.01, 0.99, 99), axis=1).T
            cover_dpa = evaluation.compute_coverage_per_quantile(y_test_np[:,i], quantile_predictions_dpa, quantiles_cq)

            # save cq per grid-cell
            member_gc_coverages_cf[i,:] = cover_dpa
    
            mae = np.mean(np.abs(quantiles_cq_np - cover_dpa))
            mae_list.append(mae)
            mae099 = np.abs(quantiles_cq_np[-1] - cover_dpa[-1])
            mae099_list.append(mae099)
        np.save(f"{save_path_eth}/data/cq_spatial_test_member{memb}_counterfactual_01-99.npy", member_gc_coverages_cf)
        all_gc_coverages_cf[str(memb)] = member_gc_coverages_cf
        spat_mean_coverage_cf = np.mean(member_gc_coverages_cf, axis=0) # compute mean coverage over all grid-cells
        coverages_all_membs_cf[memb, :] = spat_mean_coverage_cf
        memb += 1
        spat_mean_mae = np.mean(mae_list) 
        spat_mean_099 = np.mean(mae099_list)
        mae_means_cf.append(spat_mean_mae)
        mae099_means_cf.append(spat_mean_099)

    # compute mean coverage across members from: coverages_all_membs
    # mean of: mean of spatial covergae per grid-cell, per member
    mean_coverage_membs_gcs_cf = np.mean(coverages_all_membs_cf, axis = 0)
    print("mean_coverage_membs_gcs shape:", mean_coverage_membs_gcs_cf.shape)
    mae_membs_gcs_cf = np.mean(np.abs(quantiles_cq_np - mean_coverage_membs_gcs_cf))
        
    log_print(log_file, f"counterfactual mean calibration CQ MAE: {mae_membs_gcs_cf}")
    log_print(log_file, f"counterfactual coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae_means_cf)}")
    log_print(log_file, f"counterfactual 095 coverage-quantiles spatial mean, mean MAE across test members (ETH 1300,1400,1500): {np.mean(mae099_means_cf)}")


    ###################
    ###################
    ###################





    
    ###########
    ### MAE ###
    ###########

    mae_1300_fact = evaluation.mae_cols(eth_fact_1300_test_reduced, dpa_ens_mean_fact_1300_raw_pt, dim=0)
    mae_1400_fact = evaluation.mae_cols(eth_fact_1400_test_reduced, dpa_ens_mean_fact_1400_raw_pt, dim=0)
    mae_1500_fact = evaluation.mae_cols(eth_fact_1500_test_reduced, dpa_ens_mean_fact_1500_raw_pt, dim=0)

    mae_1300_cf = evaluation.mae_cols(eth_cf_1300_test_reduced, dpa_ens_mean_cf_1300_raw_pt, dim=0)
    mae_1400_cf = evaluation.mae_cols(eth_cf_1400_test_reduced, dpa_ens_mean_cf_1400_raw_pt, dim=0)
    mae_1500_cf = evaluation.mae_cols(eth_cf_1500_test_reduced, dpa_ens_mean_cf_1500_raw_pt, dim=0)

    # concatenate
    all_mae_fact = torch.cat((mae_1300_fact, mae_1400_fact, mae_1500_fact), dim=0)
    all_mae_cf = torch.cat((mae_1300_cf, mae_1400_cf, mae_1500_cf), dim=0)

    # save MAE arrays for later evaluation
    # f"{save_path_eth}/data/

    torch.save(all_mae_fact, f"{save_path_eth}/data/all_mae_fact.pt")
    torch.save(all_mae_cf, f"{save_path_eth}/data/all_mae_cf.pt")
    
    ### plot histograms ###

    data_mae = all_mae_fact.flatten().numpy()
    data_mae_cf = all_mae_cf.flatten().numpy()
    # compute medians
    median_mae = np.nanmedian(data_mae)
    median_mae_cf = np.nanmedian(data_mae_cf)

    
    # create figure and axis
    fig, ax = plt.subplots(figsize=(8,6))
    
    # plot histograms
    ax.hist(data_mae, bins=50, density=True, label="Fact", color="tab:red", alpha=0.5)
    ax.axvline(median_mae, color="tab:red", linestyle="--", linewidth=2, label=f"Fact median: {median_mae:.3f}")
    
    ax.hist(data_mae_cf, bins=50, density=True, label="CF", color="tab:blue", alpha=0.5)
    ax.axvline(median_mae_cf, color="tab:blue", linestyle="--", linewidth=2, label=f"CF median: {median_mae_cf:.3f}")
    
    # labels and ticks
    label_fs = 22
    ax.set_xlabel(r"MAE", fontsize=label_fs)
    ax.set_ylabel("Density", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=16)
    
    # legend and layout
    ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    
    # save and show
    fig.savefig(f"{save_path_eth}/MAE_spatial.pdf")
    plt.show()
    
    

    sys.exit()

    
    

    
    
    



if __name__ == "__main__":
    main()
    
