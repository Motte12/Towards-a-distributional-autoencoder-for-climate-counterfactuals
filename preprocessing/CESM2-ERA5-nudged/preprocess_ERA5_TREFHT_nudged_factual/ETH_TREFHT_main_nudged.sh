#!/bin/bash

###################################
### Preprocessing Master Script ###
###################################
parent_directory="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/"
initial_input_directory="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/raw_nudged_files/"
variable1="TSA"

# 1 TREFHT: extract TREFHT from each file

output1="${parent_directory}temporary1"
mkdir -p "${output1}"

# # --- User settings ---
INPUT_DIR=$initial_input_directory
OUTPUT_DIR=$output1
VAR_NAME=$variable1   # <-- variable to extract

# # --- Create output directory if needed ---
mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
FILES=("$INPUT_DIR"/*.nc)
shopt -u nullglob

echo "Extracting variable '$VAR_NAME' from ${#FILES[@]} files..."

for f in "${FILES[@]}"; do
    base=$(basename "$f")
    out="$OUTPUT_DIR/$base"

    # Check if variable exists in file
    if cdo -s showname "$f" | tr ' ' '\n' | grep -qx "$VAR_NAME"; then
        echo "  Processing $base"
        cdo -O selvar,"$VAR_NAME" "$f" "$out"
    else
        echo "  Skipping $base (missing $VAR_NAME)" >&2
    fi
done

echo "Done. Output written to $OUTPUT_DIR"

# 2: merge all files in time 

# ###
output2="${parent_directory}temporary2/"
mkdir -p $output2

OUTPUT_FILE=${output2}merged_TREFHT_nudged_1940-2024.nc

FILES=("$output1"/*.nc)

# --- Merge ---
echo "Merging ${#FILES[@]} NetCDF files from:"
echo "  $output1"
echo "into:"
echo "  $OUTPUT_FILE"

cdo -O mergetime "${FILES[@]}" "$OUTPUT_FILE"

echo "Done."


## 3: per ensemble member: create 5daily averages, choose reference period (1850-1900), compute seasonal cycle (per grid-cell, location), use seasonal cycle to create anomalies
## INPUT: 
## OUTPUT: 

# create temporary output directory 2
output3="${parent_directory}temporary3/"
mkdir -p "${output3}"

# Input and output directories
#INPUT_DIR="${output1}"
OUTPUT_DIR="${output3}"
REF_PERIOD_START="1950-01-01"
REF_PERIOD_END="1980-12-31"

file=$OUTPUT_FILE
      

#echo "$INPUT_DIR"
echo "Input file : $file"
echo "$OUTPUT_DIR"
echo "Reference period: $REF_PERIOD_START to $REF_PERIOD_END"




base=$(basename "$file" .nc)
echo "Processing $base ..."

# Step 1: Apply non-overlapping 5-day means to full file
cdo timselmean,5 "$file" "${OUTPUT_DIR}/${base}_5d.nc"

# Step 2: Extract reference period from 5-day mean file
cdo seldate,${REF_PERIOD_START},${REF_PERIOD_END} "${OUTPUT_DIR}/${base}_5d.nc" "${OUTPUT_DIR}/temp_ref.nc"

# Step 3: Compute seasonal cycle (daily or 5-day periods)
# Here we treat each 5-day step as equivalent to "day of year" block
cdo ydaymean "${OUTPUT_DIR}/temp_ref.nc" "${OUTPUT_DIR}/${base}_clim.nc" #calculates the mean for each calendar day (or day of the year) across all years in the file

# Step 4: Subtract climatology from 5-day mean file
# determine number of timesteps automatically (recommended)
NT=$(cdo -s ntime "${OUTPUT_DIR}/${base}_5d.nc")

# delete last timestep of the 5-day file
cdo -O delete,timestep=${NT} \
    "${OUTPUT_DIR}/${base}_5d.nc" \
    "${OUTPUT_DIR}/${base}_5d_trim.nc"

# subtract climatology
cdo -O ydaysub \
    "${OUTPUT_DIR}/${base}_5d_trim.nc" \
    "${OUTPUT_DIR}/${base}_clim.nc" \
    "${OUTPUT_DIR}/${base}_anom.nc"

# Completion message
echo "$file done"

## 1.3) subset to European domain
## BASH SCRIPT: 
## INPUT: 
## OUTPUT: 

output4="${parent_directory}temporary4/"
mkdir -p "${output4}"

INPUT_DIR="${output3}"
OUTPUT_DIR="${output4}"

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
  cdo -sellonlatbox,$LON_MIN,$LON_MAX,$LAT_MIN,$LAT_MAX "$file" "$output_file"
done

echo "All files processed. Output saved to $OUTPUT_DIR"



## 1.4) only select **summer months JJA**

# # create temporary output directory 4
output5="${parent_directory}final_dataset"
outfile5="${output5}/Era5_nudged_TREFHT_JJA.nc"
mkdir -p "${output5}"



shopt -s nullglob
FILES=("${output4}"*.nc)
echo $FILES
shopt -u nullglob

cdo selmon,6,7,8 "${FILES[@]}" "$outfile5"


## 1.5) mask data to european domain
## coordinates are changed here to match -180 to 180 longitude
## PYTHON SCRIPT: 
#parent_directory="/climca/people/floer/data/automated_preprocessing/TREFHT_ETH/ETH_transient/"
#initial_input_directory="/climca/data/CESM2-ETH/"
#variable1="TREFHT"
#output4="${parent_directory}final_dataset/"
#output3="${parent_directory}temporary3"
#outfile4="${output3}/stacked_TREFHT_JJA.nc"

echo "${output3}"
echo "${outfile4}"


# output file
outfile6="${output5}/ERA5_nudged_europe_10percent_masked_TREFHT_JJA.nc"
echo "${outfile6}"

eval "$(conda shell.bash hook)"
conda activate deepL

# execute python script
python3 apply_land_mask_TREFHT_nudged.py --input_dataset "${outfile5}" --output_dataset "${outfile6}" 