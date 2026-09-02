#!/bin/bash
# Frieder Loer frieder.loer@yahoo.com
# 05.01.2023
# this script merges (in time) monthly data from the CESM2 Large Ensemble downloaded as .nc files containing different time periods 
# downloaded from https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2le.atm.proc.monthly_ave.html via provided wget script
# the directory should only include in the .nc format the files to be merged, files of different format are ok 
# execute the script with typing ./merge_raw_data.sh
# you will be prompted for input and output directory and for the variable name
#
#
#
#read -p "Enter the input directory (e.g., /path/to/input): " input_directory # Prompt the user for input directory
input_directory="$1"
#read -p "Enter the output directory (e.g., /path/to/output): " output_directory # Prompt the user for output directory
output_directory="$2"
#read -p "Enter the variable name as written in filenames (e.g. PSL or TREFHT): " variable # Prompt the user for the variable name
variable="$3"



files=$(ls "$input_directory"/*.nc) # List all .nc files in the input directory
common_sequences=$(echo "$files" | awk -F "cam" '{print $1}' | sort -u) # Make a list of all common prefixes
echo $common_sequences
output_txt="${output_directory}/output_log_combine_files_mergetime.txt" # Create a text file to store the output
echo "Output Log:" >> "$output_txt"

echo "Input directory: $input_directory" >> "$output_txt"
echo "Output directory: $output_directory" >> "$output_txt"

for prefix in $common_sequences; do # Iterate through each common sequence
    echo "Merge files for prefix $prefix ..." >> "$output_txt" # Append information to the output log
    matching_files=$(ls "${prefix}"*.nc) # Select files that belong to the current prefix
    #filename_only=$(basename "$prefix")
    filename_only=$(basename "${matching_files[0]}" | awk -F "$variable" '{print $1}')

    #new_filename="${output_directory}/${filename_only}${variable}.1850-2100.nc" # Create a new filename in the output directory
    new_filename="${output_directory}/${filename_only}${variable}.1850-2100.nc" # Create a new filename in the output directory


    cdo_output=$(cdo mergetime $matching_files "${new_filename%.nc}_pre.nc" 2>&1) # Merge files in time using cdo and append the output to the log
    
    # delete the last timestep from the merged file
    # cdo delete,timestep=-1 "${new_filename%.nc}_pre.nc" $new_filename
    # now extract variable, subset and delete last timestep all in one go
    # extract variable, subset to north atlantic domain, delete last timestep all in one go
    cdo delete,timestep=-1 \
        -selvar,$variable \
        "${new_filename%.nc}_pre.nc" \
        $new_filename

    echo "$cdo_output" >> "$output_txt"
    
    echo "Output file: $new_filename" >> "$output_txt" # Append information to the output log
    echo "" >> "$output_txt" 
done


###############################################################

files_merged=$(ls "$output_directory"/*.nc)
common_endings=$(echo "$files_merged" | awk -F '.f09' '{print $2}' | sort -u)

echo "Files are merged from $output_directory into: $output_directory" >> "$output_txt"

for ending in $common_endings; do # Iterate through each common sequence
    echo "Merge files for ending $ending ..." >> "$output_txt"

    matching_files=$(ls "$output_directory"/*"$ending")
    new_filename="${output_directory}/b.e21.f09${ending}"
    
    cdo_output=$(cdo mergetime $matching_files $new_filename 2>&1)   # Merge files in time using cdo and append the output to the log
    echo "$cdo_output" >> "$output_txt"
    
    echo "Output file: $new_filename" >> "$output_txt" # Append information to the output log
    echo "" >> "$output_txt" 
    echo "$matching_files"
done


# Check if the list is not empty
if [ -n "$files_merged" ]; then
  # Iterate through the files and remove them
  for file in $files_merged; do
    rm "$file"
    echo "Deleted: $file"
  done
else
  echo "File list is empty."
fi