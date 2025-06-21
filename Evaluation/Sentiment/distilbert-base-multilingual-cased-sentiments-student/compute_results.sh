#!/bin/bash

# Initialize variables
total=0
count=0
output_file="TestUA_results.txt"
table_file="README.md"

model_name=$(basename "$(pwd)")

# Clear the output files before starting
> "$output_file"
> "$table_file"

# Write the Markdown table header
echo "# ${model_name} Results" >> "$table_file"
echo " " >> "$table_file"

echo "| Directory   | Test UA          |" >> "$table_file"
echo "|-------------|------------------|" >> "$table_file"

# Loop through directories fold_0 to fold_9
for i in {0..9}; do
    dir="fold_$i"

    # Check if the directory exists
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir" | tee -a "$output_file"

        # Find the .log file in the directory
        log_file=$(find "$dir" -maxdepth 1 -name "*.log" | head -n 1)

        # Check if a .log file was found
        if [ -f "$log_file" ]; then
           
            # Extract the 'Test UA:' line and get the numeric value
            value=$(grep "Test UA:" "$log_file" | awk -F'Test UA: ' '{print $2}' | awk '{print $1}')
            
            # Check if a value was found and is numeric
            if [[ $value =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
                echo -e "Test UA value found: $value\n" | tee -a "$output_file"
		rounded_value=$(awk "BEGIN {printf \"%.6f\", $value}")
                echo "| $dir       | $rounded_value          |" >> "$table_file"
                total=$(awk "BEGIN {print $total + $value}")
                count=$((count + 1))
            else
                echo "No valid Test UA value found in $log_file" | tee -a "$output_file"
                echo "| $dir       | No valid value  |" >> "$table_file"
            fi
        else
            echo "No .log file found in $dir" | tee -a "$output_file"
            echo "| $dir       | No log file     |" >> "$table_file"
        fi
    else
        echo "Directory $dir does not exist" | tee -a "$output_file"
        echo "| $dir       | Directory missing |" >> "$table_file"
    fi
done

# Compute the average
if [ "$count" -gt 0 ]; then
    average=$(awk "BEGIN {print $total / $count}")
    echo "Average Test UA: $average" | tee -a "$output_file"
    echo "| **Average** | $average          |" >> "$table_file"
else
    echo "No valid Test UA values found" | tee -a "$output_file"
    echo "| **Average** | No valid values  |" >> "$table_file"
fi
