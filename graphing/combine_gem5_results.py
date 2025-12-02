#!/usr/bin/env python3
"""
Combine gem5 Results Files

This script combines all gem5_comb_x_results.csv files into a single file
with an additional app_mix_list column for easier processing.

Usage:
    python3 graphing/combine_gem5_results.py
"""

import os
import sys
import pandas as pd
import ast

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_final')

# Input files
GEM5_FILES = [
    'gem5_comb_1_results.csv',
    'gem5_comb_2_results.csv',
    'gem5_comb_3_results.csv',
    'gem5_comb_4_results.csv',
]

# Output file
OUTPUT_FILE = 'gem5_combined_results.csv'


def parse_app_mix_str(app_mix_str):
    """
    Convert app_mix_str to list format.

    Args:
        app_mix_str: String like "canny_1_deblur_0_gru_0_harris_0_lstm_0_"

    Returns:
        List like [1, 0, 0, 0, 0]
    """
    if pd.isna(app_mix_str):
        return None

    # Remove trailing underscore and split
    parts = app_mix_str.strip('_').split('_')

    # Extract every other element starting from index 1 (the counts)
    # Format: app1_count1_app2_count2_...
    counts = []
    for i in range(1, len(parts), 2):
        try:
            counts.append(int(parts[i]))
        except (ValueError, IndexError):
            print(f"[WARNING] Could not parse count from app_mix_str: {app_mix_str}")
            return None

    return counts


def main():
    """Main function to combine gem5 results."""
    print("[INFO] Starting gem5 results combination...")

    # Check if all input files exist
    missing_files = []
    for filename in GEM5_FILES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)

    if missing_files:
        print(f"[ERROR] Missing input files: {', '.join(missing_files)}")
        print(f"[ERROR] Make sure files are in: {RESULTS_DIR}")
        return 1

    # Read and combine all files
    all_dataframes = []

    for filename in GEM5_FILES:
        filepath = os.path.join(RESULTS_DIR, filename)
        print(f"[INFO] Reading {filename}...")

        try:
            df = pd.read_csv(filepath)
            print(f"[INFO]   Loaded {len(df)} rows, {len(df.columns)} columns")

            # Add app_mix_list column
            df['app_mix_list'] = df['app_mix_str'].apply(parse_app_mix_str)

            # Verify parsing worked
            null_count = df['app_mix_list'].isna().sum()
            if null_count > 0:
                print(f"[WARNING]   {null_count} rows had invalid app_mix_str")

            all_dataframes.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to read {filename}: {e}")
            return 1

    # Concatenate all dataframes
    print(f"\n[INFO] Combining {len(all_dataframes)} dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"[INFO] Combined dataset: {len(combined_df)} total rows")

    # Calculate num_apps from app_mix_list where missing
    print(f"\n[INFO] Calculating num_apps where missing...")
    def calculate_num_apps(row):
        if pd.notna(row['num_apps']):
            return row['num_apps']
        if row['app_mix_list'] is not None:
            return sum(row['app_mix_list'])
        return None

    combined_df['num_apps'] = combined_df.apply(calculate_num_apps, axis=1)
    combined_df['num_apps'] = combined_df['num_apps'].astype('Int64')  # Use nullable integer type

    # Verify app_mix_list column
    print(f"\n[INFO] Verifying app_mix_list column...")
    null_count = combined_df['app_mix_list'].isna().sum()
    if null_count > 0:
        print(f"[WARNING] {null_count} rows have invalid app_mix_list")
    else:
        print(f"[INFO] All app_mix_list entries are valid")

    # Show sample rows
    print(f"\n[INFO] Sample rows:")
    print(combined_df[['policy', 'app_mix_str', 'app_mix_list', 'num_apps']].head())

    # Write output file
    output_path = os.path.join(RESULTS_DIR, OUTPUT_FILE)
    print(f"\n[INFO] Writing combined results to: {output_path}")

    try:
        combined_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Combined results saved successfully!")

        # Show output file info
        file_size = os.path.getsize(output_path)
        print(f"[INFO] Output file size: {file_size / 1024:.1f} KB")
        print(f"[INFO] Total rows: {len(combined_df)}")
        print(f"[INFO] Total columns: {len(combined_df.columns)}")

    except Exception as e:
        print(f"[ERROR] Failed to write output file: {e}")
        return 1

    # Summary statistics
    print(f"\n[INFO] Summary by scheduler:")
    print(combined_df.groupby('policy').size())

    print(f"\n[INFO] Summary by num_apps:")
    print(combined_df.groupby('num_apps').size())

    return 0


if __name__ == '__main__':
    sys.exit(main())
