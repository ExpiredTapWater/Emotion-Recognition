#!/bin/bash

# Full path to compute_wer.py
COMPUTE_WER_PATH="C:\Users\ChenYi\Documents\Github\SIT-emotion-recognition\asr\compute_wer.py"

# Set the base directory to the current directory
BASE_DIR="$(pwd)"

# Ground truth file in the specified path
GROUND_TRUTH_FILE="C:\Users\ChenYi\Documents\Github\SIT-emotion-recognition\asr\groundtruth_IEMOCAP.txt"

# Output file to save logs
RESULTS_FILE="${BASE_DIR}/wer_results.txt"

# Check if compute_wer.py exists
if [[ ! -f "$COMPUTE_WER_PATH" ]]; then
    echo "compute_wer.py not found at $COMPUTE_WER_PATH."
    exit 1
fi

# Check if the ground truth file exists
if [[ ! -f "$GROUND_TRUTH_FILE" ]]; then
    echo "Ground truth file not found at $GROUND_TRUTH_FILE."
    exit 1
fi

# Clear the results file
> "$RESULTS_FILE"

# Initialize variables for WER calculation
total_wer=0
fold_count=0

# Loop through fold_0 to fold_9
for i in {0..9}; do
    FOLDER="${BASE_DIR}/fold_$i"
    PREDICTION_FILE="${FOLDER}/transcriptions.txt"

    # Check if the prediction file exists
    if [[ -f "$PREDICTION_FILE" ]]; then
        echo "------- Fold $i -------"
        # Run the compute_wer.py script and capture its output
        output=$(python "$COMPUTE_WER_PATH" --mode present "$GROUND_TRUTH_FILE" "$PREDICTION_FILE")

        # Extract the WER value using grep and awk
        wer=$(echo "$output" | grep -oP '%WER\s+\K[0-9.]+')

        # Append the output to the results file
        echo "Results for fold_$i:" >> "$RESULTS_FILE"
        echo "$output" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"  # Add a blank line for readability

        # Update the total WER and fold count
        if [[ ! -z "$wer" ]]; then
            total_wer=$(awk "BEGIN {print $total_wer + $wer}")
            fold_count=$((fold_count + 1))
        fi
    else
        echo "Skipping fold_$i: transcriptions.txt not found." >> "$RESULTS_FILE"
    fi
done

# Calculate and save the average WER
if [[ $fold_count -gt 0 ]]; then
    average_wer=$(awk "BEGIN {print $total_wer / $fold_count}")
    echo "Average WER across $fold_count folds: $average_wer" >> "$RESULTS_FILE"
    echo "Average WER: $average_wer"
else
    echo "No valid folds processed. No average WER calculated."
fi

echo " ------- Processing complete. Results saved to $RESULTS_FILE -------"
