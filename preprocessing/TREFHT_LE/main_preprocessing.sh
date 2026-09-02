#!/bin/bash

###################################
### Preprocessing Master Script ###
###################################
parent_directory="/climca/people/floer/data/automated_preprocessing_13012025/TREFHT"
initial_input_directory="/climca/data/CESM2_LE/TREFHT_new/day_raw" # the raw data files as downloaded from the large ensemble
variable1="TREFHT"
# 1 TREFHT

## 1.1) create ensemble members with daily data
## BASH SCRIPT: restructured_modularized/merge_files.sh
## INPUT: /climca/data/CESM2_LE/TREFHT_new/day_raw
## OUTPUT: /climca/people/floer/data/TREFHT/day_processed

# directories 
input1="$initial_input_directory"
output1="${parent_directory}/temporary1"
echo $output1

# create temporary output directory 1
mkdir "${output1}"

# RUN SCRIPT
# run script: script input_directory output_directory variable
./merge_files.sh "${input1}" "${output1}" "${variable1}"


## 1.2) per ensemble member: create 5daily averages, choose reference period (1850-1900), compute seasonal cycle (per grid-cell, location), use seasonal cycle to create anomalies
## BASH SCRIPT: temperature_preprocessing_cdo.sh
## INPUT: /climca/people/floer/data/TREFHT/day_processed
## OUTPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT

# create temporary output directory 2
output2="${parent_directory}/temporary2"
mkdir "${output2}"

# RUN SCRIPT
# script input_directory output_directory
./temperature_preprocessing_cdo.sh "${output1}" "${output2}"

# delete temporary output directory 1
#rm -r "${output1}"



## 1.3) subset data to European domain
## BASH SCRIPT: subset_TREFHT.sh
## INPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT
## OUTPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT_subset

# create temporary output directory 3
#output3="${parent_directory}/temporary3"
#mkdir "${output3}"

# RUN SCRIPT
# script input_dir output_dir variable
# SUBSETTING NOW ALREADY DONE IN merge_files.sh
#./subset_TREFHT.sh "${output2}" "${output3}" "${variable1}" 

# delete temporary output directory 2
#rm -r "${output2}"  


## 1.4) combine all 100 ens members into one file and only select **summer months JJA**
## PYTHON SCRIPT: /home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized/combine_TREFHT_LE.py
## INPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT_subset
## OUTPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT_combined_JJA

# create temporary output directory 4
output3="${parent_directory}/final_dataset"
outfile3="${output3}/stacked_TREFHT_JJA.nc"
mkdir "${output3}"

# activat python envs
#export PATH=/home/floer/anaconda3/envs/deepL/lib/python3.11/bin:$PATH
#source /home/floer/anaconda3/bin/activate deepL
# OR
eval "$(conda shell.bash hook)"
conda activate deepL

# execute python script
python3 combine_TREFHT_LE.py --input_dir "${output2}" --output_file "${outfile3}"


# delete temporary output directory 3
#rm -r "${output3}"

## 1.5) mask data
## PYTHON SCRIPT: /home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized/apply_land_mask_TREFHT.py
## INPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT_combined_JJA/stacked_TREFHT_JJA.nc
## OUTPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT_combined_JJA/europe_10percent_masked_stacked_TREFHT_JJA.nc

# output file
outfile4="${output3}/europe_10percent_masked_stacked_TREFHT_JJA.nc"

# execute python script
python3 apply_land_mask_TREFHT.py --input_dataset "${outfile3}" --output_dataset "${outfile4}" 






