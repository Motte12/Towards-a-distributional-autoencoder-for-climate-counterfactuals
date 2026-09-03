#!/bin/bash

########################################
### Z500 Preprocessing Master Script ###
########################################

parent_directory="data/automated_preprocessing_13012025/Z500"
initial_input_directory="data/CESM2_LE/NEW_Z500/daily_raw"
variable1="Z500"

# 1. Z500

# 1.1 merge raw daiy files into single ensemble members 
# BASH SCRIPT: LLAAE/data_preprocessing/restructured_modularized/Z500_preprocessing/Z500_merge_files.sh
# INPUT:"data/CESM2_LE/NEW_Z500/daily_raw"
# OUTPUT: "data/tests/Z500/temporary1"

# directories
input1="$initial_input_directory"
output1="${parent_directory}/temporary1"

# HERE
mkdir "${output1}"

# RUN SCRIPT
# script input_directory output_directory variable 
./z500_merge_files.sh "${input1}" "${output1}" "${variable1}"

# 1.2 delete days? --> probably not needed

# 1.2 per ensemble member: 
## create 5daily averages, 
## choose reference period (1850-1900) 
## compute seasonal cycle (per grid-cell, location), use seasonal cycle to create anomalies

# create temporary output directory 2
output2="${parent_directory}/temporary2"
# HERE
mkdir "${output2}"

# RUN SCRIPT
./preprocessing_cdo_Z500.sh "${output1}" "${output2}"

# 1.3 
# detrend Z500 with ensemble mean, 
## compute EOFs, 
## subset to summer months JJA, 
## project Z500 onto EOFs to obtain pseudo PCs (use first 1000)

# directories
# remove temporary output directory 1
# for starting script later
parent_directory="/climca/people/floer/data/automated_preprocessing_13012025/Z500"
output2="${parent_directory}/temporary2"


exit 
# from here continue with the scripts 
# Z500_LE/submit_coherent_Z500_calculation.sh that executes Z500_LE/coherent_all_data_EOF_calculation_Z500.py

output3="${parent_directory}/final_dataset/"
mkdir "${output3}"

# activat python envs

eval "$(conda shell.bash hook)"
conda activate deepL

echo "Now calculating EOFs"
# python script input_directory output_directory
# ONLY HERE THE DATA IS SUBSET TO THE NORTH ATLANTIC DOMAIN
python3 EOF_calculation_Z500.py --input_dir "${output2}" --output_dir "${output3}" #include forwards slash at output directory path end!!!
# would actually not need slicing here any more, but should still work if data was subset correctly before in z500_merge_files.sh

# remove output directory 2

