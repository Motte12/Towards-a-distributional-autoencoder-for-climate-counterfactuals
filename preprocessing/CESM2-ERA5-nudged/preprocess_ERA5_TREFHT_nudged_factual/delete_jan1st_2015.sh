#!/bin/bash

# Input and output files
INPUT_FILE="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/raw_nudged_files/raw_CO2h_LUh_Aerh_2015-2023.clm2.h2.2015-01-01-00000.nc"
OUTPUT_FILE="/climca/people/floer/data/ERA5/TREFHT_nudged_fact/raw_nudged_files/CO2h_LUh_Aerh_2015-2023.clm2.h2.2015-01-01-00000.nc"

# Remove the first timestep
cdo seldate,2015-01-02,2015-12-31 "$INPUT_FILE" "$OUTPUT_FILE"

echo "Done! First timestep (2015-01-01) removed."