#!/bin/bash

###################################
### Preprocessing Master Script ###
###################################
parent_directory="/climca/people/floer/data/automated_preprocessing_13012025/TREFHT"
initial_input_directory="/climca/data/CESM2_LE/TREFHT_new/day_raw"
variable1="TREFHT"
# 1 TREFHT

## 1.1) create ensemble members with daily data
## BASH SCRIPT: /home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized/merge_files.sh
## INPUT: /climca/data/CESM2_LE/TREFHT_new/day_raw
## OUTPUT: /climca/people/floer/data/TREFHT/day_processed

# directories 
input1="$initial_input_directory"
output1="${parent_directory}/temporary1_global_temps"
echo $output1

# create temporary output directory 1
mkdir "${output1}"

# RUN SCRIPT
# run script: script input_directory output_directory variable
./merge_files_global_no_subsetting.sh "${input1}" "${output1}" "${variable1}"


## 1.2) per ensemble member: create 5daily averages, choose reference period (1850-1900), compute seasonal cycle (per grid-cell, location), use seasonal cycle to create anomalies
## BASH SCRIPT: /home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized/temperature_preprocessing_cdo.sh
## INPUT: /climca/people/floer/data/TREFHT/day_processed
## OUTPUT: /climca/people/floer/data/TREFHT/5daily_TREFHT

# create temporary output directory 2
output2="${parent_directory}/temporary2_global_temps"
mkdir "${output2}"

# RUN SCRIPT
# script input_directory output_directory
./temperature_preprocessing_cdo.sh "${output1}" "${output2}"

# delete temporary output directory 1
#rm -r "${output1}"

exit 

