import torch
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

import sys
sys.path.append('../utils')
import utils as ut


def plot_dpa_time_series(
    ts,
    plot_year,
    FIG_WIDTH_IN, 
    FIG_HEIGHT_IN,
    ax=None,
    panel_label="(b)",
    ylabel="Temperature [°C]",
    xlabel="Time" 
    ):
    """
    Plot publication-style factual and counterfactual DPA time series.
    """

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
            constrained_layout=False,
        )
    else:
        fig = ax.figure

    year = str(plot_year)

    # Select year
    t0 = f"{year}-01-01"
    t1 = f"{year}-12-31"

    true_cf = ts["true_cf"].sel(time=slice(t0, t1))
    true_fact = ts["true_fact"].sel(time=slice(t0, t1))
    dpa_cf = ts["dpa_cf_mean"].sel(time=slice(t0, t1))
    dpa_fact = ts["dpa_fact_mean"].sel(time=slice(t0, t1))

    cf_lower = ts["cf_lower"].sel(time=slice(t0, t1))
    cf_upper = ts["cf_upper"].sel(time=slice(t0, t1))
    fact_lower = ts["fact_lower"].sel(time=slice(t0, t1))
    fact_upper = ts["fact_upper"].sel(time=slice(t0, t1))

    #x = np.arange(true_cf.sizes["time"])
    #labels = [f"{t.month:02d}-{t.day:02d}" for t in true_cf.time.dt]
    x = np.arange(true_cf.sizes["time"])
    labels = true_cf.time.dt.strftime("%m-%d").values
    
    lw = 1.0

    # Uncertainty envelopes first
    ax.fill_between(
        x,
        fact_lower.values,
        fact_upper.values,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
        label=rf"Factual DAE $\pm${ts['n_stds']} SD",
    )

    ax.fill_between(
        x,
        cf_lower.values,
        cf_upper.values,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
        label=rf"CF DAE $\pm${ts['n_stds']} SD",
    )

    # Time series
    ax.plot(
        x,
        true_cf.values,
        color="tab:cyan",
        linewidth=lw,
        label="CF truth",
    )

    ax.plot(
        x,
        dpa_cf.values,
        color="tab:blue",
        linestyle="--",
        linewidth=lw,
        label="CF DAE mean",
    )

    ax.plot(
        x,
        true_fact.values,
        color="tab:orange",
        linewidth=lw,
        label="Factual truth",
    )

    ax.plot(
        x,
        dpa_fact.values,
        color="tab:red",
        linestyle="--",
        linewidth=lw,
        label="Factual DAE mean",
    )

    # Axes
    ax.set_ylabel(ylabel, labelpad=1)
    ax.set_xlabel(xlabel, labelpad=1)

    ax.set_xticks(x[::3])
    ax.set_xticklabels(labels[::3])

    ax.set_xticks(x, minor=True)

    ax.tick_params(axis="both", which="major", length=2.5, width=0.5, pad=1)
    ax.tick_params(axis="x", which="minor", length=1.5, width=0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.text(
        0.03,
        0.92,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        bottom=0.18,
        top=0.82,
    )

    return fig, ax

def plot_multiple_dpa_time_series(true_t, 
                                  dpa_ens, 
                                  dpa_ens_mean, 
                                  true_t_fact, 
                                  dpa_ens_fact, 
                                  dpa_ens_mean_fact, 
                                  lat_min, 
                                  lat_max, 
                                  lon_min, 
                                  lon_max, 
                                  plot_year, 
                                  figsize_ts, 
                                  title_fontsize, 
                                  title, 
                                  climate):
    
    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    print("temp_true_ger")


    ### DPA Ensemble COUNTERFACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_std['lat']}, dims=['lat'])
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_pre = dpa_ens_mean.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_mean_ger_pre['lat']}, dims=['lat'])
    dpa_ens_mean_ger = dpa_ens_mean_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_pre = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))#.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_ger_1300_pre['lat']}, dims=['lat'])
    dpa_ens_ger_1300 = dpa_ens_ger_1300_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # counterfactual truth
    #fact_truth_ger = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    

    ### DPA Ensemble FACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member")
    dpa_ens_std_ger_fact = dpa_ens_std_fact.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_fact = dpa_ens_mean_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # factual truth
    fact_truth_ger = true_t_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    
    
    # plot
    n_stds = 2
    
    # cf
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger.TREFHT 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger.TREFHT
    print("ENVS:", lower_env, upper_env)

    # factual
    lower_env_fact = (dpa_ens_mean_ger_fact + n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    upper_env_fact = (dpa_ens_mean_ger_fact - n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    for year in plot_year:
        #year = plot_year
        lw=1.0
        times = temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values
        labels = [f'{t.month:02d}-{t.day:02d}' for t in times]
        print(temp_true_ger.sel(time="2003"))
        plt.plot(range(19), temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:cyan", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:blue", linestyle="--", linewidth=lw)
        plt.plot(range(19), fact_truth_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color = "tab:orange", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:red", linestyle="--", linewidth=lw)
        
        # factual
        ax.fill_between(
            range(19), 
            lower_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:red", alpha=0.2, label=r'$\pm$' + f"2 factual DAE ensemble standard deviations",
            linewidth=0
        )

        # counterfactual
        ax.fill_between(
            range(19),
            lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:blue", alpha=0.2, label=r'$\pm$' + f"2 CF DAE ensemble standard deviations",
            linewidth=0
        )
        
        # Major ticks: every 3rd time step (with labels)
        ax.set_xticks(range(0, len(times), 3))
        
        # Minor ticks: all time steps
        ax.set_xticks(range(len(times)), minor=True)
        
        ax.set_xticklabels([lab for i, lab in enumerate(labels[::3])])


    
    ax.legend(fontsize=6, ncol=1, frameon=False)

    if len(plot_year) > 1:
        multi_year = True
    else:
        multi_year = False

    return fig, ax

def old_plot_multiple_dpa_time_series(true_t, 
                                  dpa_ens, 
                                  dpa_ens_mean, 
                                  true_t_fact, 
                                  dpa_ens_fact, 
                                  dpa_ens_mean_fact, 
                                  lat_min, 
                                  lat_max, 
                                  lon_min, 
                                  lon_max, 
                                  plot_year, 
                                  figsize_ts, 
                                  title_fontsize, 
                                  title, 
                                  climate):
    
    ### True (Test) temperature ###
    # true temperature - germany spatial average
    temp_true_ger_pre = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    # create weights
    # 1) define weights as above
    weights_ger = np.cos(np.deg2rad(temp_true_ger_pre['lat']))
    
    # 2) wrap in a DataArray so xarray knows which dim it belongs to
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': temp_true_ger_pre['lat']}, dims=['lat'])
    
    temp_true_ger = temp_true_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))


    ### DPA Ensemble COUNTERFACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member") # before: dpa_ensemble_restored.TREFHT
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_std['lat']}, dims=['lat'])
    dpa_ens_std_ger = dpa_ens_std.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_pre = dpa_ens_mean.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_mean_ger_pre['lat']}, dims=['lat'])
    dpa_ens_mean_ger = dpa_ens_mean_ger_pre.weighted(w_da_ger).mean(dim=('lat', 'lon')) # before: dpa_ensemble_restored

    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_pre = dpa_ens.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))#.weighted(w_da_ger).mean(dim=('lat', 'lon'))
    w_da_ger = xr.DataArray(weights_ger, coords={'lat': dpa_ens_ger_1300_pre['lat']}, dims=['lat'])
    dpa_ens_ger_1300 = dpa_ens_ger_1300_pre.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # counterfactual truth
    fact_truth_ger = true_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    

    ### DPA Ensemble FACTUAL ###
    # standard deviation, germany mean
    dpa_ens_std_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).std(dim="ensemble_member")
    dpa_ens_std_ger_fact = dpa_ens_std_fact.weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # ensemble mean, germany mean 
    dpa_ens_mean_ger_fact = dpa_ens_mean_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    # ensemble germany average (ensemble_member, )
    dpa_ens_ger_1300_fact = dpa_ens_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))

    # factual truth
    fact_truth_ger = true_t_fact.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).weighted(w_da_ger).mean(dim=('lat', 'lon'))
    
    
    
    # plot
    n_stds = 2
    
    # cf
    lower_env = dpa_ens_mean_ger + n_stds * dpa_ens_std_ger 
    upper_env = dpa_ens_mean_ger - n_stds * dpa_ens_std_ger

    # factual
    lower_env_fact = (dpa_ens_mean_ger_fact + n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    upper_env_fact = (dpa_ens_mean_ger_fact - n_stds * dpa_ens_std_ger_fact).TREFHT #.to_numpy().astype("float64")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_ts)

    for year in plot_year:
        #year = plot_year
        lw=1.0
        times = temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).time.values
        labels = [f'{t.month:02d}-{t.day:02d}' for t in times]

        plt.plot(range(19), temp_true_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:cyan", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:blue", linestyle="--", linewidth=lw)
        plt.plot(range(19), fact_truth_ger.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color = "tab:orange", linewidth=lw)
        plt.plot(range(19), dpa_ens_mean_ger_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values, color="tab:red", linestyle="--", linewidth=lw)
        
        # factual
        ax.fill_between(
            range(19), 
            lower_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env_fact.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:red", alpha=0.2, label=r'$\pm$' + f"2 factual DAE ensemble standard deviations",
            linewidth=0
        )

        # counterfactual
        ax.fill_between(
            range(19),
            lower_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # lower bound
            upper_env.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).values,        # upper bound
            color="tab:blue", alpha=0.2, label=r'$\pm$' + f"2 CF DAE ensemble standard deviations",
            linewidth=0
        )
        
        # Major ticks: every 3rd time step (with labels)
        ax.set_xticks(range(0, len(times), 3))
        
        # Minor ticks: all time steps
        ax.set_xticks(range(len(times)), minor=True)
        
        ax.set_xticklabels([lab for i, lab in enumerate(labels[::3])])


    
    ax.legend(fontsize=6, ncol=1, frameon=False)

    if len(plot_year) > 1:
        multi_year = True
    else:
        multi_year = False

    return fig, ax

def r2_score(y_true, y_pred, dim=0):
    """
    Compute R² per feature along `dim`.
    y_true, y_pred: torch tensors of same shape
    """
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=dim)
    ss_tot = torch.sum((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2, dim=dim)
    r2 = 1 - ss_res / ss_tot
    return r2

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

def pearsonr_cols(x, y, dim=0, eps=1e-12):
    """
    Pearson correlation per feature along `dim` (default: samples axis).
    x, y: tensors of the same shape, e.g. (n_samples, n_features)
    Returns: tensor of shape equal to the non-reduced dims (e.g. (n_features,))
    """
    # center
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean

    # numerator: covariance (without / (n-1) since it cancels in correlation)
    num = (x_c * y_c).sum(dim=dim)

    # denominator: product of std devs
    x_ss = (x_c * x_c).sum(dim=dim)
    y_ss = (y_c * y_c).sum(dim=dim)
    den = (x_ss * y_ss).sqrt().clamp_min(eps)

    return num / den


def mae_cols(x, y, dim=0):
    """
    Mean Absolute Error per feature along `dim` (default: samples axis).
    x, y: tensors of the same shape, e.g. (n_samples, n_features)
    Returns: tensor of shape equal to the non-reduced dims (e.g. (n_features,))
    """
    # compute absolute differences
    abs_diff = torch.abs(x - y)

    # mean along the specified dimension
    mae = abs_diff.mean(dim=dim)

    return mae



def mse_cols(x, y, dim=0):
    """
    Mean Squared Error per feature along `dim` (default: samples axis).
    x, y: tensors of the same shape, e.g. (n_samples, n_features)
    Returns: tensor of shape equal to the non-reduced dims (e.g. (n_features,))
    """
    # compute squared differences
    sq_diff = (x - y) ** 2

    # mean along the specified dimension
    mse = sq_diff.mean(dim=dim)

    return mse

    import numpy as np

def snr(x, y, ddof=1):
    """
    Compute signal-to-noise ratio between two distributions.

    Parameters
    ----------
    x, y : array-like
        Samples from the two distributions.
    ddof : int
        Delta degrees of freedom for std calculation.

    Returns
    -------
    float
        Signal-to-noise ratio.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mu1 = np.mean(x)
    mu2 = np.mean(y)

    s1 = np.std(x, ddof=ddof)
    s2 = np.std(y, ddof=ddof)

    pooled_std = np.sqrt((s1**2 + s2**2) / 2)

    return abs(mu1 - mu2) / pooled_std
    





    
    