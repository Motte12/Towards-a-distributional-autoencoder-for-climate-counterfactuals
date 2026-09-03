#!/bin/bash

# --- Input and output files ---
INPUT_FILE="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/raw_nudged_files/starts_late_CO2h_LUh_Aerh_1940-2015.clm2.h2.1940-01-02-00000.nc"
FIRST_TIMESTEP="first_timestep.nc"
DUMMY_JAN1="dummy_jan1.nc"
OUTPUT_FILE="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/raw_nudged_files/CO2h_LUh_Aerh_1940-2015.clm2.h2.1940-01-02-00000.nc"

# Step 1: Extract first timestep (Jan 2nd)
cdo seltimestep,1 "$INPUT_FILE" "$FIRST_TIMESTEP"

# Step 2: Create dummy timestep with Jan 1st
cdo setdate,1940-01-01 "$FIRST_TIMESTEP" "$DUMMY_JAN1"

# Step 3: Prepend dummy to original file
cdo cat "$DUMMY_JAN1" "$INPUT_FILE" "$OUTPUT_FILE"

echo "New file with Jan 1st prepended created: $OUTPUT_FILE"