# Data preprocessing

This README describes the routine that was used for preprocessing the data.

raw data is available at:
- CESM2 Large Ensemble: https://doi.org/10.26024/kgmp-c556
- CESM2 ETH-Ensemble (in parts): https://zenodo.org/records/18172330
- ERA5 data: https://cds.climate.copernicus.eu/datasets


## TREFHT

### LE

run: TREFHT_LE/main_preprocessing.sh

and for later calculating fGMT, run: TREFHT_LE/main_preprocessing_for_global_temperatures.sh

### ETH transient

run: TREFHT_ETH/ETH_TREFHT_main_preprocessing.sh

### ETH nudged-circulation

run: TREFHT_ETH_nudged/ETH_TREFHT_nudged_main_preprocessing.sh

### CESM2-ERA5 nudged-circulation

run:
    + CESM2-ERA5-nudged/preprocess_ERA5_TREFHT_nudged/ETH_TREFHT_main_nudged.sh
    + CESM2-ERA5-nudged/preprocess_ERA5_TREFHT_nudged_factual/ETH_TREFHT_main_nudged.sh

## Z500 

### LE

run: Z500_LE/z500_main_preprocessing.sh

### ETH

run: Z500_ETH/z500_eth_main_preprocessing.sh

### ERA5

run: Z500_ERA5/z500_era5_main_preprocessing.sh

Once the above three scripts for Z500 preprocessing completed, run: Z500_LE/submit_coherent_Z500_calculation.sh (**you need to adjust the input paths**)

## fGMT

run: fGMT_LE_combine_predictors/compute_GMT_from_5daily_TREFHT.sh 
    + adjust the input/output paths 
    + and the input paths to GMT_extract_JJA_and_combineZ500.py script to the outputs of Z500_LE/submit_coherent_Z500_calculation.sh 

This script also combined the Z500 predictors with the fGMT predictor.