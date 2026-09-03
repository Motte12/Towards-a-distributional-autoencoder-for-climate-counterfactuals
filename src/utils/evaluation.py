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
    





    
    