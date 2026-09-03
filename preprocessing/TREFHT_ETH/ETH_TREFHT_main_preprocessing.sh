#!/bin/bash

###################################
### Preprocessing Master Script ###
###################################
parent_directory="data/automated_preprocessing_13012025/TREFHT_ETH/ETH_transient/"
initial_input_directory="data/CESM2-ETH/"
variable1="TREFHT"
# 1 TREFHT

output1="${parent_directory}temporary1/"
mkdir "${output1}"

## 1.1) concatenate transient runs (historic and future) (1300, 1400, 1500) to obtain data for 1850-2100
# concatenate 1300
cdo cat "${initial_input_directory}b.e212.BHISTcmip6.f09_g17.1300/TREFHT_day_b.e212.BHISTcmip6.f09_g17.1300.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1300/TREFHT_day_b.e212.BSSP370cmip6.f09_g17.1300.nc" "${output1}TREFHT_day_b.e212.BHISTcmip6_BSSP370cmip6.f09_g17.1300_1850-2100.nc"

# concatenate 1400
cdo cat "${initial_input_directory}b.e212.BHISTcmip6.f09_g17.1400/TREFHT_day_b.e212.BHISTcmip6.f09_g17.1400.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1400/TREFHT_day_b.e212.BSSP370cmip6.f09_g17.1400.nc" "${output1}TREFHT_day_b.e212.BHISTcmip6_BSSP370cmip6.f09_g17.1400_1850-2100.nc"

# concatenate 1500
cdo cat "${initial_input_directory}b.e212.BHISTcmip6.f09_g17.1500/TREFHT_day_b.e212.BHISTcmip6.f09_g17.1500.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1500/TREFHT_day_b.e212.BSSP370cmip6.f09_g17.1500.nc" "${output1}TREFHT_day_b.e212.BHISTcmip6_BSSP370cmip6.f09_g17.1500_1850-2100.nc"


## 1.2) per ensemble member: create 5daily averages, choose reference period (1850-1900), compute seasonal cycle (per grid-cell, location), use seasonal cycle to create anomalies
## INPUT: 
## OUTPUT: 

# create temporary output directory 2
output2="${parent_directory}temporary2/"
mkdir "${output2}"

# Input and output directories
INPUT_DIR="${output1}"
OUTPUT_DIR="${output2}"
REF_PERIOD_START="1950-01-01"
REF_PERIOD_END="1980-12-31"

echo "$INPUT_DIR"
echo "$OUTPUT_DIR"
echo "Reference period: $REF_PERIOD_START to $REF_PERIOD_END"

for file in ${INPUT_DIR}/*.nc; do
    base=$(basename "$file" .nc)
    echo "Processing $base..."

    # Step 1: Apply non-overlapping 5-day means to full file
    cdo timselmean,5 "$file" "${base}_5d.nc"

    # Step 2: Extract reference period from 5-day mean file
    cdo seldate,${REF_PERIOD_START},${REF_PERIOD_END} "${base}_5d.nc" temp_ref.nc

    # Step 3: Compute seasonal cycle (daily or 5-day periods)
    # Here we treat each 5-day step as equivalent to "day of year" block
    cdo ydaymean temp_ref.nc "${OUTPUT_DIR}/${base}_clim.nc" #calculates the mean for each calendar day (or day of the year) across all years in the file
    #cdo timmean temp_ref.nc "${OUTPUT_DIR}/${base}_clim.nc"  # mean over matching 5-day periods in reference

    # Step 4: Subtract climatology from 5-day mean file
    cdo ydaysub "${base}_5d.nc" "${OUTPUT_DIR}/${base}_clim.nc" "${OUTPUT_DIR}/${base}_anom.nc"
    #cdo sub "${base}_5d.nc" "${OUTPUT_DIR}/${base}_clim.nc" "${OUTPUT_DIR}/${base}_anom.nc"

    # Cleanup
    rm -f "${base}_5d.nc" temp_ref.nc

    # Completion message
    echo "$file done"
done

# delete temporary output directory 1
#rm -r "${output1}"

## 1.3) subset to European domain
## BASH SCRIPT: 
## INPUT: 
## OUTPUT: 

output3="${parent_directory}temporary3/"
mkdir "${output3}"

INPUT_DIR="${output2}"
OUTPUT_DIR="${output3}"
VARIABLE="${variable1}"                # Change this to the variable you want to extract

LON_MIN=-12 #168
LON_MAX=28 #208
LAT_MIN=34
LAT_MAX=64

# -------------------------------
# Processing Loop
# -------------------------------

for file in "$INPUT_DIR"/*anom.nc; do
  filename=$(basename "$file")
  output_file="$OUTPUT_DIR/$filename"

  echo "Processing $filename..."

  # Extract variable and subdomain using CDO
  #cdo -selvar,$VARIABLE -selindexbox,136,167,133,164 "$file" "$output_file"
  cdo -selvar,$VARIABLE -sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX "$file" "$output_file"
  #cdo sinfo "$output_file"
done

echo "All files processed. Output saved to $OUTPUT_DIR"

#delete temporary output directory 2
#rm -r "${output2}"  

# create temporary output directory 3







## 1.4) combined all 3 ens member into one file and only select **summer months JJA**
## PYTHON SCRIPT: preprocessing_automated/TREFHT_ETH_preprocessing_automated/combine_TREFHT_ETH_transient.py

# create temporary output directory 4
output4="${parent_directory}final_dataset"
outfile4="${output3}stacked_TREFHT_JJA.nc"
mkdir "${output4}"

# activat python envs

eval "$(conda shell.bash hook)"
conda activate deepL

# execute python script
python3 combine_TREFHT_ETH_transient.py --input_dir "${output3}" --output_file "${outfile4}"

# delete temporary output directory 2
#rm -r "${output3}"





## 1.5) mask data to european domain
## coordinates are changed here to match -180 to 180 longitude
## PYTHON SCRIPT: 
#parent_directory="data/automated_preprocessing/TREFHT_ETH/ETH_transient/"
#initial_input_directory="data/CESM2-ETH/"
#variable1="TREFHT"
#output4="${parent_directory}final_dataset/"
#output3="${parent_directory}temporary3"
#outfile4="${output3}/stacked_TREFHT_JJA.nc"

echo "${output3}"
echo "${outfile4}"


# output file
outfile5="${output4}/ETH_europe_10percent_masked_stacked_TREFHT_JJA.nc"
echo "${outfile5}"

# execute python script
python3 apply_land_mask_TREFHT_transient.py --input_dataset "${outfile4}" --output_dataset "${outfile5}" # ADJUST THIS SCRIPT TO TAKE ARGUMENTS!