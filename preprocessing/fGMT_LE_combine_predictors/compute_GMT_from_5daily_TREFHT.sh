#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --partition=pq3
#SBATCH --time=02-01:01

# Large Ensemble 

# uncomment 
input_directory="data/TREFHT/temporary2_global_temps/" 
output_directory="data/TREFHT/final_dataset_global_temps/"

mkdir -p "$output_directory"

for file in  "$input_directory"*_anom.nc; do
    filename=$(basename "$file")
    echo  "Processing $filename"
    cdo fldmean "$file" "$output_directory$filename"
done

cdo ensmean "$output_directory"*_anom.nc "$output_directory"gmt_ensemble_mean.nc

# ERA5 - GMT

#cdo fldmean "data/ERA5/TREFHT/temporary2/E5sf00_1D_T2M_1940-2025_remapped_to_CESM2_no_leap_anom.nc" "data/ERA5/TREFHT/ERA5_TREFHT_GMT/ERA5_TREFHT_GMT_anom.nc"

# combine GMT and pseudo_pcs
eval "$(conda shell.bash hook)"
conda activate deepL #oder geocat something env

# adjust paths below
python3 GMT_extract_JJA_and_combineZ500.py \
 --output_path "data/z500_predictors_combined/ERA5_detrended_with_smoothed_grid-cells" \
 --gmt_path "$output_directory"gmt_ensemble_mean.nc \
 --pseudo_pcs_era5_path "" \
 --pseudo_pcs_le_path "" \
 --pseudo_pcs_eth_path "" \
 