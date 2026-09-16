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
from statsmodels.nonparametric.smoothers_lowess import lowess
import pickle
import glob


# function to load data
def load_data(resolution, data_path, chunk_shape=None):

    if chunk_shape != None:
        # Use xr.open_mfdataset to open and concatenate multiple datasets
        print("data path:", f"{data_path}/*.nc")
        t_combined_dataset_le = xr.open_mfdataset(f"{data_path}/*_anom.nc", chunk_shape, combine='nested', concat_dim='ensemble_member')#.sel(time=slice("1851","2099")) # probably already need to subet here 
    elif chunk_shape == None:
        t_combined_dataset_le = xr.open_mfdataset(f"{data_path}/*_anom.nc", combine='nested', concat_dim='ensemble_member') 

    
    print('raw data loaded')

    # Flip and sort longitude coordinates to facilitate data subsetting
    t_ds_le = sort_data(t_combined_dataset_le)
    return t_ds_le

def sort_data(t_combined_dataset_le):

    t_combined_dataset_le["lon"] = ((t_combined_dataset_le["lon"] + 180) % 360) - 180

    # Sort longitudes, so that subset operations end up being simpler.
    t_ds_le_pre = t_combined_dataset_le.sortby("lon")

    # Place latitudes in increasing order:
    t_ds_le = t_ds_le_pre.sortby("lat", ascending=True)
    
    return t_ds_le