#!/bin/bash

# Input and output directories
INPUT_DIR="$1"
OUTPUT_DIR="$2"
REF_PERIOD_START="1950-01-01"
REF_PERIOD_END="1980-12-31"

echo "$INPUT_DIR"
echo "$OUTPUT_DIR"
echo "Reference period: $REF_PERIOD_START to $REF_PERIOD_END"

for file in ${INPUT_DIR}/*.nc; do

    # skip _pre.nc files
    if [[ "$file" == *_pre.nc ]]; then
        continue
    fi
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
