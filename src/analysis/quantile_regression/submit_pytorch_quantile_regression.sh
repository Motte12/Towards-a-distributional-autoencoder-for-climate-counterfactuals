#!/bin/bash
#SBATCH --job-name=QR_SP
#SBATCH --partition=clara
#SBATCH --mem=100G
#SBATCH --time=0-08:00:00

# Optional: print job info
echo "Starting job on $(hostname) at $(date)"
echo "Running from: $(pwd)"

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate dpa

# create directory
# Get current date, hour and minute
timestamp=$(date +"%Y-%m-%d_%H-%M")

##################
### SET DOMAIN ###
##################
domain="SP"

# Name prefix
settings_file="../../../settings.json"

# output dir
output_dir=$(jq -r '.paths.output_dir' "$settings_file")
echo "Output dir: $output_dir"


# Create directory with timestamp
name="$output_dir/QR_baseline/v5_quantile_regression_${domain}"
dirname="${name}_${timestamp}"
mkdir -p "$dirname"

echo "Created directory: $dirname"




# Run the preprocessing script starting from line 72
python pytorch_quantile_regression.py \
    --settings_file_path $settings_file \
    --delta 0.00001 \
    --n_epochs 200 \
    --save_path "$dirname/" \
    --q_start 0.01 \
    --q_end 0.99 \
    --q_n 99 \
    --domain $domain \
    --standardize_predictors 1

echo "Job finished at $(date)"
