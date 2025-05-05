#!/bin/bash

# Paths
CSV_FILE="selected_records_ahi_results_with_severity.csv"
INPUT_DIR="selected_records"
DEST_BASE_DIR="organized_output"

# Create destination folders
mkdir -p "$DEST_BASE_DIR/No_OSA"
mkdir -p "$DEST_BASE_DIR/Mild_OSA"
mkdir -p "$DEST_BASE_DIR/Moderate_OSA"
mkdir -p "$DEST_BASE_DIR/Severe_OSA"

# Skip header and loop through CSV rows
tail -n +2 "$CSV_FILE" | while IFS=',' read -r record ahi severity; do
    # Replace spaces with underscores in severity for folder names
    safe_severity=$(echo "$severity" | sed 's/ /_/g')

    # Source and destination paths
    src="$INPUT_DIR/$record"
    dest="$DEST_BASE_DIR/$safe_severity"

    # Copy the folder if it exists
    if [[ -d "$src" ]]; then
        cp -r "$src" "$dest/"
        echo "Copied $record to $dest/"
    else
        echo "Warning: Folder $src not found!"
    fi
done
