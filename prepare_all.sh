#!/bin/bash

SCRIPT_PATH="/scratch/09295/naveens/hindcast/split_hindcast_WDCC.py"

INPUT_DIR="/scratch/09295/naveens/hindcast"

SKIPPED_LOG="skipped_files_all_years.txt"
> "$SKIPPED_LOG"  

for year in $(seq 1979 2024); do
    echo "======================"
    echo "Processing year: $year"
    echo "======================"

    python "$SCRIPT_PATH" --year "$year" --input_dir "$INPUT_DIR"

    if [ -f skipped_files.txt ]; then
        echo "----- $year -----" >> "$SKIPPED_LOG"
        cat skipped_files.txt >> "$SKIPPED_LOG"
        rm skipped_files.txt
    fi
done

echo "Processing complete for all years. See $SKIPPED_LOG for any skipped files."
