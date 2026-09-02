#!/bin/bash

###################################
### Preprocessing Master Script ###
###################################
parent_directory="/climca/people/floer/data/automated_preprocessing_13012025/Z500_ETH/"
initial_input_directory="/climca/data/CESM2-ETH/"
variable1="Z500"

# 1.1 concatenate

output1="${parent_directory}temporary1/"
mkdir "${output1}"

# concatenate 1300
cdo cat "${initial_input_directory}/b.e212.BHISTcmip6.f09_g17.1300/Z500_day_b.e212.BHISTcmip6.f09_g17.1300.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1300/Z500_day_b.e212.BSSP370cmip6.f09_g17.1300.nc" "${output1}Z500_day_b.e212.BHISTcmip6.f09_g17.1300_1850-2100.nc"

# concatenate 1400
cdo cat "${initial_input_directory}/b.e212.BHISTcmip6.f09_g17.1400/Z500_day_b.e212.BHISTcmip6.f09_g17.1400.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1400/Z500_day_b.e212.BSSP370cmip6.f09_g17.1400.nc" "${output1}Z500_day_b.e212.BHISTcmip6.f09_g17.1400_1850-2100.nc"

# concatenate 1500
cdo cat "${initial_input_directory}/b.e212.BHISTcmip6.f09_g17.1500/Z500_day_b.e212.BHISTcmip6.f09_g17.1500.nc" "${initial_input_directory}b.e212.BSSP370cmip6.f09_g17.1500/Z500_day_b.e212.BSSP370cmip6.f09_g17.1500.nc" "${output1}Z500_day_b.e212.BHISTcmip6.f09_g17.1500_1850-2100.nc"




#exit
# 1.2 

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
    if [[ "$file" == *trimmed* ]]; then
        continue
    fi
    base=$(basename "$file" .nc)
    echo "Processing $base..."

    # Step 1: Apply non-overlapping 5-day means to full file
    cdo timselmean,5 "$file" "${OUTPUT_DIR}/${base}_5d.nc"

    # Step 2: Extract reference period from 5-day mean file
    cdo seldate,${REF_PERIOD_START},${REF_PERIOD_END} "${OUTPUT_DIR}/${base}_5d.nc" "${OUTPUT_DIR}/${base}_Z500_ref.nc"

    # Step 3: Compute seasonal cycle (daily or 5-day periods)
    # Here we treat each 5-day step as equivalent to "day of year" block
    cdo ydaymean "${OUTPUT_DIR}/${base}_Z500_ref.nc" "${OUTPUT_DIR}/${base}_clim.nc" #calculates the mean for each calendar day (or day of the year) across all years in the file

    # Step 4: Subtract climatology from 5-day mean file
    cdo ydaysub "${OUTPUT_DIR}/${base}_5d.nc" "${OUTPUT_DIR}/${base}_clim.nc" "${OUTPUT_DIR}/${base}_anom.nc"
    #cdo sub "${base}_5d.nc" "${OUTPUT_DIR}/${base}_clim.nc" "${OUTPUT_DIR}/${base}_anom.nc"

    # Cleanup
    #rm -f "${base}_5d.nc" Z500_ref.nc

    # Completion message
    echo "$file done"
done

exit 
# from here continue with the scripts 
# publish_preprocessing/Z500_LE/submit_coherent_Z500_calculation.sh that executes publish_preprocessing/Z500_LE/coherent_all_data_EOF_calculation_Z500.py

# 1.3 calculate EOFs
parent_directory="/climca/people/floer/data/automated_preprocessing_13012025/Z500_ETH/"
output2="${parent_directory}temporary2"
output3="${parent_directory}EOFs/"
mkdir "${output3}"

# activate conde env
eval "$(conda shell.bash hook)"
conda activate deepL 

python /home/floer/Climate_Counterfactuals/climat-counterfactuals/LLAAE/data_preprocessing/restructured_modularized/Z500_preprocessing/ETH_Z500_preprocessing/EOF_calculation_ETH_Z500.py \
    --input_dir "${output2}" \
    --output_dir "${output3}" \
    --input_le_z500 "/climca/people/floer/data/automated_preprocessing_13012025/Z500/temporary2" \
    --compute_eofs_new 0 \
    --project_eth_data 0





# concatenate EOFs with GMT




