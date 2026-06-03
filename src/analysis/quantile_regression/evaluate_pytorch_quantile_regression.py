import numpy as np
from sklearn.linear_model import QuantileRegressor
import matplotlib.pyplot as plt
import xarray as xr
import json
from joblib import Parallel, delayed
import torch
import pytorch_quantile_regression as pqr
import pandas as pd
import argparse
import sys
#sys.path.append('/Users/friederl/Documents/EcoN_project/LLAAE/DPA/code/dpa_for_llaae')
import utils as ut
import evaluation as evaluation
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to quantile regression model to use.")
    parser.add_argument("--results_save_path", type=str, help="Path to save evaluation results.")
    parser.add_argument("--compare_model", type=str, help="Other quantile/ensemble model to compare to.")
    parser.add_argument("--data_version", type=str, help="Test data version.")
    parser.add_argument("--eval_counterfactuals", type=int, default=0, help="Whether to evaluate counterfactuals.")
    parser.add_argument("--one_dimensional_ger", type=int, default=0, help="Whether to evaluate for DPA trained on 1d ger data.")
    parser.add_argument("--analogues", type=int, default=0, help="Whether to evaluate analogues.")
    parser.add_argument("--standardize_predictors", type=int, default=0, help="Whether to standardize validation/test predictors.")
    parser.add_argument("--eval_validation_set", type=int, default=0, help="Whether to validate validation set or test set.")
    parser.add_argument("--qr_epoch", type=int, default=100, help="Quantile regression checkpoint/epoch to load.")
    parser.add_argument("--eval_era5", type=int, default=0, help="Whether to evaluate Era5.")
    parser.add_argument("--domain", type=str, default="GER", help="Which domain to evaluate")
    parser.add_argument("--eval_epochs", type=int, help="Number of epochs that model to be evaluated was trained on.")
    parser.add_argument("--dae_model", type=str, help="DAE model to evaluate.")

    args = parser.parse_args()
    print(args.data_version)
    print(type(args.data_version))

    save_string = f"{args.domain}"
    os.makedirs(args.results_save_path, exist_ok=True)

    # Domain 
    if args.domain == "GER":
        # GER
        lat_min = 48
        lat_max = 54
        lon_min = 6
        lon_max = 15

    elif args.domain == "FR":
        lat_min = 45
        lat_max = 50
        lon_min= 0
        lon_max= 5

    elif args.domain == "SP":
        lat_min = 38
        lat_max = 42
        lon_min = -8
        lon_max = 0

    else:
        print("Choose domain correctly!")
        

    ###########################################################
    ### Plot Training loss curves and smoothed pinball loss ###
    ###########################################################
    
    # Look at loss curves
    csv_path = f"{args.model_path}training_log.csv"
    
    # --- 1. Load parameters from metadata.json ---
    
    metadata_path = f"{args.model_path}metadata.json"  # adjust if needed
    print(metadata_path)
    
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    # load
    df = pd.read_csv(csv_path)
    
    # quick check
    print(df.head())
    # expected columns: epoch, train_loss, val_loss


    CM = 1 / 2.54
    
    # ~1/3 text width on A4 with 2.5 cm margins
    FIG_WIDTH = 5.3 * CM
    FIG_HEIGHT = 4.0 * CM
    
    LABEL_SIZE = 8
    TICK_SIZE = 7
    TITLE_SIZE = 9
    LEGEND_SIZE = 7
    
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True
    )
    
    ax.plot(
        df["epoch"],
        df["train_loss"],
        label="Train loss",
        linewidth=1.2
    )
    
    ax.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation loss",
        linewidth=1.2
    )
    
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
    
    #ax.set_title(
    #    "Training and validation loss",
    #    fontsize=TITLE_SIZE
    #)
    
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    
    ax.legend(
        fontsize=LEGEND_SIZE,
        frameon=False
    )
    
    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.5
    )
    
    #plt.savefig(f"{args.results_save_path}loss_curves_{args.domain}.pdf")

        
    
    # ---------------------------------------------------------
    # 1. Load quantiles & default delta from metadata.json
    # ---------------------------------------------------------
    quantiles = np.array(meta["quantiles"])
    delta_default = float(meta["delta"])
    
    print("Loaded quantiles:", quantiles)
    print("Default delta from metadata:", delta_default)
    
    # ---------------------------------------------------------
    # 2. Choose multiple delta values to plot
    # ---------------------------------------------------------
    
    # You can override or extend with your own list:
    deltas = [delta_default, 1e-1, 0.5]
        
    # ---------------------------------------------------------
    # 3. Define smoothed pinball loss
    # ---------------------------------------------------------
    
    
    
    # ---------------------------------------------------------
    # 4. Select quantiles to visualize
    # ---------------------------------------------------------
    
    # Pick 3 representative quantiles (low, mid, high)
    if len(quantiles) > 5:
        taus_to_plot = [
            quantiles[0],
            quantiles[len(quantiles)//2],
            quantiles[-1]
        ]
    else:
        taus_to_plot = quantiles
    
    print("Quantiles used for plotting:", taus_to_plot)
    
    # ---------------------------------------------------------
    # 5. Create grid for residuals u = y - y_pred
    # ---------------------------------------------------------
    
    u = np.linspace(-3, 3, 500)   # adjust range if needed
    
    # ---------------------------------------------------------
    # 7. Optional: Multi-panel plot for different quantiles
    # ---------------------------------------------------------

    CM = 1 / 2.54
    
    # ~1/3 text width
    FIG_HEIGHT = 5.3 * CM
    
    
    # adjust height depending on number of subplots
    FIG_WIDTH = (4.5 * len(taus_to_plot)) * CM
    
    LABEL_SIZE = 8
    TICK_SIZE = 7
    TITLE_SIZE = 9
    LEGEND_SIZE = 7
    
    fig, axes = plt.subplots(
        1,
        len(taus_to_plot),
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True
    )
    
    # ensure iterable if only one subplot
    if len(taus_to_plot) == 1:
        axes = [axes]
    
    for ax, tau in zip(axes, taus_to_plot):
    
        for delta in deltas:
            loss_vals = smooth_pinball_loss(u, tau, delta)
    
            ax.plot(
                u,
                loss_vals,
                label=fr"$\delta = {delta}$",
                linewidth=1.1
            )
    
        ax.axvline(
            0,
            color="k",
            linestyle=":",
            linewidth=0.8
        )
    
        ax.set_title(
            fr"$\tau = {tau}$",
            fontsize=TITLE_SIZE
        )
    
        ax.set_xlabel(
            "Residual $u$",
            fontsize=LABEL_SIZE
        )
    
        ax.set_ylabel(
            "Loss",
            fontsize=LABEL_SIZE
        )
    
        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE
        )
    
        ax.grid(
            True,
            linewidth=0.5,
            alpha=0.5
        )
    
        ax.legend(
            loc="upper center",
            fontsize=LEGEND_SIZE,
            frameon=False
        )
    
    #plt.savefig(
    #    f"{args.results_save_path}smoothed_pinball_loss_{args.domain}.pdf",
    #    bbox_inches="tight"
    #)


    ###############################################################
    ### End plot Training loss curves and smoothed pinball loss ###
    ###############################################################

    
    
    ##################################
    ### Quantile rergression model ###
    ##################################

    # Load model
    model_path = args.model_path
    
    # ---- Load metadata ----
    with open(f"{model_path}metadata.json", "r") as f:
        meta = json.load(f)
    
    #quantiles = meta["quantiles"]
    n_features = meta["n_features"]
    n_quantiles = len(quantiles)
    
    # ---- Load checkpoint ----
    ckpt_path = f"{args.model_path}checkpoint_epoch_{args.qr_epoch}.pth"#meta["last_checkpoint"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # ---- Rebuild model ----
    model = pqr.LinearMultiQuantileRegressor(
        n_features=n_features,
        n_quantiles=n_quantiles
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # ---- Move to GPU (optional) ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    
    
    #################
    ### Test data ###
    #################
    
    if args.data_version not in ["v1", "v2", "v3", "v4"]:
        settings_file_path = f"{args.data_version}"
    
    else:
        settings_file_path = f"{args.data_version}_dpa_train_settings.json" #used v2 here for a long time

    # open settings file
    with open(settings_file_path, 'r') as file:
            settings = json.load(file)
    
    #############################
    ### Temperature Test data ###
    #############################
    if args.eval_counterfactuals:
        ds_eth = os.path.join(settings['paths']['data'], settings['paths']['dataset_trefht_eth_nudged_shifted'])
        trefht_eth = xr.open_dataset(ds_eth)
    else:
        if bool(args.eval_validation_set):
            ds_eth = os.path.join(settings['paths']['data'], settings['paths']['dataset_trefht'])
            trefht_eth = xr.open_dataset(ds_eth).isel(time=slice(90*4769,476900))

        else:
            ds_eth = os.path.join(settings['paths']['data'], settings['paths']['dataset_trefht_eth_transient'])
            trefht_eth = xr.open_dataset(ds_eth)
            

    if bool(args.eval_era5):
        trefht_eth = trefht_eth.sel(time=slice("1940","2023"))
    

    
    
    # cut test data
    trefht_eth_ger = trefht_eth.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # calculate weighted means
    #weights
    weights_ger_pre = np.cos(np.deg2rad(trefht_eth["lat"]))
    weights_ger = weights_ger_pre / weights_ger_pre.sum()
    # test_data
    trefht_eth_ger_mean = trefht_eth_ger.TREFHT.weighted(weights_ger).mean(dim=("lat", "lon")).values
    
    #################################
    ### Load Test Predictors Z500 ###
    #################################
    z500_test_path = os.path.join(settings['paths']['data'], settings['paths']['dataset_z500_eth_test'])
    z500_test = xr.open_dataset(z500_test_path).pseudo_pcs

    ##################################
    ### Load Train Predictors Z500 ###
    ##################################
    z500_le_path = os.path.join(settings['paths']['data'], settings['paths']['dataset_z500'])
    predictors_combined_le_pre = xr.open_dataset(z500_le_path).pseudo_pcs
    #predictors_combined_le_pre = xr.open_dataset(settings['dataset_z500']).pseudo_pcs
    predictors_combined_le = predictors_combined_le_pre.values
    print("predictors combined shape:", predictors_combined_le.shape)

    ### STANDARDIZE DATA ###
    if args.data_version == "v1" or args.data_version == "v4" or args.standardize_predictors:
        print("Data standardized here")
        ## train
        train_predictors, train_mean, train_std = ut.standardize_numpy(predictors_combined_le[:90*4769, :])
        X_torch = torch.from_numpy(train_predictors)
        print(train_mean.shape, train_std.shape)
    
        ## validation
        validation_predictors, _, _ = ut.standardize_numpy(predictors_combined_le[90*4769:, :], train_mean, train_std)
        X_val_torch = torch.from_numpy(validation_predictors)
        
        ## test data
        z500_test_np_pre, _, _ = ut.standardize_numpy(z500_test.values, train_mean, train_std)

        # standardize only fGMT with train statistics
        #z500_test_np_pre_dummy, _, _ = ut.standardize_numpy(z500_test.values, train_mean[0,-1], train_std[0,-1])
        #z500_test_np_pre[:,-1] = z500_test_np_pre_dummy[:,-1]
        
    else:
        print("predictors already standardized")
        z500_test_np = z500_test.values
        

    # SET VALIDATION OR TEST DATA
    if args.eval_validation_set:
        print("#################################")
        print("### Validating VALIDATION set ###")
        print("#################################")
        X_test_torch = torch.from_numpy(validation_predictors.astype("float32")).to(device)
        z500_test_np = X_test_torch

    else:
        print("###########################")
        print("### Validating TEST set ###")
        print("###########################")
        X_test_torch = torch.from_numpy(z500_test_np_pre.astype("float32")).to(device)   
        z500_test_np = X_test_torch 
    


    # set counterfactual fGMT when evaluating counterfactuals
    if args.eval_counterfactuals:
        pi_period_mean = predictors_combined_le_pre.isel(time=slice(0,4769)).sel(time=slice("1850","1900")).isel(mode=1000).mean().values
        cf_fgmt = (pi_period_mean - train_mean[0,-1]) / train_std[0,-1]

        # assign value to test set
        X_test_torch[:,-1] = torch.tensor(cf_fgmt)
        print("cf_fgmt:", cf_fgmt)
    
    #########################
    ### Model predictions ###
    #########################

    # Quantile regression model 
    with torch.no_grad():
        preds = model(X_test_torch)   # shape (N_test, n_quantiles)
    quantile_predictions = preds.cpu().numpy() # trnasform into numpy

    # compute R2
    y_true_qr = trefht_eth_ger_mean
    y_pred_qr = quantile_predictions[:, 9]
    
    # Create mask: keep only entries where both are finite
    mask = np.isfinite(y_true_qr) & np.isfinite(y_pred_qr)
    
    y_t = torch.from_numpy(y_true_qr[mask])
    y_p = torch.from_numpy(y_pred_qr[mask])
    print("TRUE/PREDICTED shapes:", y_t.shape, y_p.shape)

    mae_t_qr = evaluation.mae_cols(y_t, y_p, dim=0)
    print("MAE QR:", mae_t_qr)


    ###################################
    ### Load comparison model (DAE) ###
    ###################################

    ###########
    ### DAE ###
    ###########
    dpa_ds = xr.open_dataset(args.compare_model)
    trefht_dpa_trans_ger = dpa_ds.TREFHT.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    
    # calculate weighted means
    #weights
    weights_ger_pre = np.cos(np.deg2rad(trefht_dpa_trans_ger["lat"]))
    weights_ger = weights_ger_pre / weights_ger_pre.sum()
    trefht_dpa_trans_ger_mean = trefht_dpa_trans_ger.weighted(weights_ger).mean(dim=("lat", "lon"))
    
    ##################
    ##################
    ##################

    # miscellaneous assignments
    # prepare data for validation 
    X_test_np = z500_test_np
    y_test_np = trefht_eth_ger_mean
    
    
    # DPA predicted quantiles
    dpa_trans_predicted_quantiles = np.quantile(trefht_dpa_trans_ger_mean.values.T, np.linspace(0.01, 0.99, 99), axis=1).T

    print("####################################")
    print("DPA trans predicted quantiles shape:",dpa_trans_predicted_quantiles.shape)
    print("DPA trans predicted quantiles:",dpa_trans_predicted_quantiles)
    print("####################################")

    if args.compare_model is not None:    
        quantile_predictions_dpa = dpa_trans_predicted_quantiles

    
    y = trefht_eth_ger_mean #y_test_np
    q_hat = quantile_predictions  # (N_test, n_quantiles)
    if args.compare_model is not None:
        q_hat_dpa = quantile_predictions_dpa


    # compute MAE
    y_true = trefht_eth_ger_mean
    y_pred = dpa_trans_predicted_quantiles[:, 49]
    
    # Create mask: keep only entries where both are finite
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    y_t = y_true[mask]
    y_p = y_pred[mask]

    y_t = torch.from_numpy(y_true[mask])
    y_p = torch.from_numpy(y_pred[mask])
    print("TRUE/PREDICTED shapes:", y_t.shape, y_p.shape)

    mae_t_dae = evaluation.mae_cols(y_t, y_p, dim=0)
    print("MAE DAE:", mae_t_dae)
    
    







    ####################
    ### Summary Both ###
    ####################

    if args.compare_model is not None:
        # calibration plot only
        FIG_WIDTH_MM = 153
        FIG_WIDTH_IN = 0.4 * (FIG_WIDTH_MM / 25.4)
        FIG_HEIGHT_IN = 0.3 * (FIG_WIDTH_MM / 25.4)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

        FONT_SIZE = 8

        plt.rcParams.update({
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
        
        # 1) CALIBRATION CURVE (top-left): QR vs DPA
        ax_cal = ax
    
        cover_qr  = compute_coverage_per_quantile(y_test_np, quantile_predictions,     quantiles)
        cover_dpa = compute_coverage_per_quantile(y_test_np, quantile_predictions_dpa, quantiles)
        
        
        if args.eval_counterfactuals:
            dae_color="tab:blue"
            climate = "CF"
        else:  
            dae_color="tab:red"
            climate="Factual"

        ax_cal.plot(quantiles, cover_qr, label=f"QR {climate}", color="darkcyan", linewidth=1)
        ax_cal.plot(quantiles, cover_dpa, label=f"DAE {climate}", color=dae_color, linewidth=1)

        # compute MAE ###
        mae_qr = np.mean(np.abs(quantiles-cover_qr))
        mae_dae = np.mean(np.abs(quantiles-cover_dpa))

        mae_qr_095 = np.mean(np.abs(quantiles[-1]-cover_qr[-1]))
        mae_dae_095 = np.mean(np.abs(quantiles[-1]-cover_dpa[-1]))

        # print mae
        ax_cal.text(0.99, 0.2,      # x=0.95, y=0.95 in axes coordinates (0-1)
                r" $\mathrm{QR\ MAE_{cal}}$:" + f" {mae_qr:.3f}",   # text to display
                transform=ax_cal.transAxes,  # use axes coordinates
                ha='right',      # horizontal alignment
                va='top',        # vertical alignment
                fontsize=6,
                color='black'
            )

        ax_cal.text(0.99, 0.1,      # x=0.95, y=0.95 in axes coordinates (0-1)
                r"$\mathrm{DAE\ MAE_{cal}}$:" + f" {mae_dae:.3f}",   # text to display
                transform=ax.transAxes,  # use axes coordinates
                ha='right',      # horizontal alignment
                va='top',        # vertical alignment
                fontsize=6,
                color='black'
            )

        if bool(args.eval_counterfactuals):
            subpanel="d)"
        else:
            subpanel="b)"
        
        ax.text(
                0.035, 0.96,      # x=0.95, y=0.95 in axes coordinates (0-1)
                f"{subpanel}",   # text to display
                transform=ax.transAxes,  # use axes coordinates
                ha='left',      # horizontal alignment
                va='top',        # vertical alignment
                fontsize=8,
                color='black'
            )

        
        ax_cal.plot([0, 1], [0, 1], "k--", label="Ideal 1:1", linewidth=1)
    
        ax_cal.set_xlabel("Nominal quantile τ", fontsize=8)
        #ax_cal.set_ylabel("Empirical fraction of points \n with y_true ≤ q̂_τ(x)", fontsize=10)
        ax_cal.set_ylabel("Points in quantile", fontsize=8)

        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='x', labelsize=8)  # x-axis labels font size 12
        ax.tick_params(axis='y', labelsize=8)
        ax_cal.legend(loc="upper left", frameon=False, fontsize = 6, bbox_to_anchor=(0.0, 0.92))

        plt.tight_layout()
        #plt.savefig(f"{args.results_save_path}CALIBRATION_{args.domain}_{args.eval_epochs}epochs_qr_vs_dpa_CF={bool(args.eval_counterfactuals)}_{args.dae_model}.pdf")
        sys.exit()

        


def smooth_pinball_loss(u, tau, delta):
        """
        Smoothed pinball loss:
            rho_τ^δ(u) = 0.5 * ( sqrt(u^2 + δ^2) + (2τ - 1)*u )
        """
        u = np.asarray(u)
        smooth_abs = np.sqrt(u**2 + delta**2)
        return 0.5 * (smooth_abs + (2*tau - 1.0) * u)

def compute_coverage_per_quantile(y_true, q_preds, quantiles):
    """
    y_true: (N,)
    q_preds: (N, Q)
    quantiles: (Q,)
    Returns: empirical coverages array of shape (Q,)
    """
    y_true = np.asarray(y_true).reshape(-1)
    q_preds = np.asarray(q_preds)
    quantiles = np.asarray(quantiles)

    coverages = []
    for j in range(len(quantiles)):
        tau = quantiles[j]
        q_tau = q_preds[:, j]
        cov = np.mean(y_true <= q_tau)
        coverages.append(cov)

    return np.array(coverages)

    
if __name__ == "__main__":
    main()