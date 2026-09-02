#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --partition=pq3
#SBATCH --time=02-01:01

eval "$(conda shell.bash hook)"
conda activate deepL

python3 coherent_all_data_EOF_calculation_Z500.py \
    --le_directory_path "/climca/people/floer/data/automated_preprocessing_13012025/Z500/temporary2" \
    --eth_data_directory_path "/climca/people/floer/data/automated_preprocessing_13012025/Z500_ETH/temporary2" \
    --era5_data_directory_path "/climca/people/floer/data/ERA5/Z500/temporary2" \
    --output_dir "/climca/people/floer/data/publication_testing"