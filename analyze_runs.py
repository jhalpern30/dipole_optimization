"""
File: analyze_runs.py
Author: Jake Halpern
Date: 02/10/2025
Description: This script analyzes a set of optimizations from the run_optimize_scan script by setting
             them up into a dataframe that can be further filtered
"""
import os
import pandas as pd
import glob
import json

# Define the common parent directory (e.g., 'geoscan_test')
parent_dir = '../outputs/20250207_geoscan_test'
# Get all result directories under the parent directory
result_dirs = [d for d in glob.glob(os.path.join(parent_dir, '*')) if os.path.isdir(d)]

df = pd.DataFrame()
# Loop through directories and read results.json
for result_dir in result_dirs:
    result_file = os.path.join(result_dir, 'results.json')
    if not os.path.isfile(result_file):
        continue  # Skip if results.json does not exist
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        # Extract the directory name relative to the parent directory
        relative_dirname = os.path.relpath(result_dir, parent_dir)
        # Add relative directory name to data
        data['dirname'] = relative_dirname
        # Normalize nested lists
        df = pd.concat([df, pd.json_normalize(data)], ignore_index=True)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading {result_file}: {e}")
        continue

# Print dataframe keys to check structure
print(df)
