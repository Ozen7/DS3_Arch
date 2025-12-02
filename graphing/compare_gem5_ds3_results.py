#!/usr/bin/env python3
"""
gem5 vs DS3 Results Comparison and Graphing

This script compares gem5 and DS3 execution times, calculates percentage
errors, and generates a bar graph showing average error by number of jobs.

Usage:
    python3 graphing/compare_gem5_ds3_results.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import ast

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_final')
GRAPHS_DIR = os.path.join(SCRIPT_DIR, 'graphs')

# Input files
GEM5_COMBINED_CSV = os.path.join(RESULTS_DIR, 'gem5_combined_results.csv')
DS3_RESULTS_CSV = os.path.join(RESULTS_DIR, 'ds3_gem5_matching_results.csv')

# Output files
DETAILED_COMPARISON_CSV = os.path.join(RESULTS_DIR, 'gem5_ds3_detailed_comparison.csv')
COMPARISON_GRAPH = os.path.join(GRAPHS_DIR, 'gem5_ds3_comparison.pdf')

# Scheduler mapping
SCHEDULER_MAPPING = {
    'LAX': 'LL',
    'ELFD': 'RELIEF',
    'ELF': 'RELIEF',  # Fallback
    'GEDF_D': 'GEDF_D',
    'GEDF_N': 'GEDF_N',
    'HetSched': 'HetSched',
}


def main():
    """Main function for comparison and graphing."""
    print("[INFO] gem5 vs DS3 Results Comparison")
    print("=" * 60)

    # Check input files exist
    if not os.path.exists(GEM5_COMBINED_CSV):
        print(f"[ERROR] gem5 results not found: {GEM5_COMBINED_CSV}")
        return 1

    if not os.path.exists(DS3_RESULTS_CSV):
        print(f"[ERROR] DS3 results not found: {DS3_RESULTS_CSV}")
        print("[ERROR] Please run graphing/run_gem5_matching_experiments.py first")
        return 1

    # Ensure graphs directory exists
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # Load results
    print(f"\n[INFO] Loading gem5 results...")
    gem5_df = pd.read_csv(GEM5_COMBINED_CSV)
    print(f"[INFO] Loaded {len(gem5_df)} gem5 results")

    print(f"\n[INFO] Loading DS3 results...")
    ds3_df = pd.read_csv(DS3_RESULTS_CSV)
    print(f"[INFO] Loaded {len(ds3_df)} DS3 results")

    # Filter gem5 for valid schedulers only
    print(f"\n[INFO] Filtering gem5 results for valid schedulers...")
    valid_schedulers = list(SCHEDULER_MAPPING.keys())
    gem5_df_filtered = gem5_df[gem5_df['policy'].isin(valid_schedulers)]
    print(f"[INFO] {len(gem5_df_filtered)} gem5 results after filtering")

    # Convert gem5 sim_seconds to microseconds
    print(f"\n[INFO] Converting gem5 times to microseconds...")
    gem5_df_filtered['gem5_execution_time_us'] = gem5_df_filtered['sim_seconds'] * 1_000_000

    # Map gem5 policy to DS3 scheduler
    gem5_df_filtered['ds3_scheduler'] = gem5_df_filtered['policy'].map(SCHEDULER_MAPPING)

    # Normalize app_mix_list to string for merging
    def safe_str_app_mix_list(val):
        if isinstance(val, str):
            # Already string, possibly string representation of list
            try:
                # Try to parse and re-stringify for consistency
                parsed = ast.literal_eval(val)
                return str(parsed)
            except:
                return val
        return str(val)

    gem5_df_filtered['app_mix_list_str'] = gem5_df_filtered['app_mix_list'].apply(safe_str_app_mix_list)
    ds3_df['app_mix_list_str'] = ds3_df['app_mix_list'].apply(safe_str_app_mix_list)

    # Filter DS3 for successful experiments only
    ds3_df_success = ds3_df[ds3_df['status'] == 'success'].copy()
    print(f"[INFO] {len(ds3_df_success)} successful DS3 experiments")

    if len(ds3_df_success) == 0:
        print("[ERROR] No successful DS3 experiments to compare")
        return 1

    # Merge gem5 and DS3 results
    print(f"\n[INFO] Merging gem5 and DS3 results...")
    merged_df = pd.merge(
        gem5_df_filtered[['app_mix_list_str', 'ds3_scheduler', 'gem5_execution_time_us', 'num_apps', 'policy']],
        ds3_df_success[['app_mix_list_str', 'ds3_scheduler', 'ds3_execution_time_us']],
        on=['app_mix_list_str', 'ds3_scheduler'],
        how='inner'
    )

    print(f"[INFO] Merged {len(merged_df)} matching experiments")

    if len(merged_df) == 0:
        print("[ERROR] No matching experiments found between gem5 and DS3")
        return 1

    # Calculate absolute percentage error
    print(f"\n[INFO] Calculating percentage errors...")
    merged_df['abs_percentage_error'] = (
        abs(merged_df['gem5_execution_time_us'] - merged_df['ds3_execution_time_us']) /
        merged_df['gem5_execution_time_us'] * 100
    )

    # Show some statistics
    print(f"\n[INFO] Error Statistics:")
    print(f"  Mean error: {merged_df['abs_percentage_error'].mean():.2f}%")
    print(f"  Median error: {merged_df['abs_percentage_error'].median():.2f}%")
    print(f"  Min error: {merged_df['abs_percentage_error'].min():.2f}%")
    print(f"  Max error: {merged_df['abs_percentage_error'].max():.2f}%")

    # Save detailed comparison
    print(f"\n[INFO] Saving detailed comparison to: {DETAILED_COMPARISON_CSV}")
    merged_df.to_csv(DETAILED_COMPARISON_CSV, index=False)

    # Group by num_apps and calculate average error
    print(f"\n[INFO] Calculating average error by number of jobs...")
    avg_errors = merged_df.groupby('num_apps')['abs_percentage_error'].mean().sort_index()

    print(f"\n[INFO] Average Error by Number of Jobs:")
    for num_apps, avg_error in avg_errors.items():
        count = len(merged_df[merged_df['num_apps'] == num_apps])
        print(f"  {int(num_apps)} jobs: {avg_error:.2f}% (n={count})")

    # Generate bar graph
    print(f"\n[INFO] Generating bar graph...")

    fig, ax = plt.subplots(figsize=(10, 6))

    num_apps = avg_errors.index.tolist()
    avg_error_pct = avg_errors.values.tolist()

    # Create bars with blue color scheme (matching other graphs)
    bars = ax.bar(num_apps, avg_error_pct, color='#4A90A4', edgecolor='black', alpha=0.8)

    # Customize appearance - NO title, NO axis labels
    ax.set_xticks(num_apps)
    ax.set_xticklabels([int(x) for x in num_apps], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add experiment count annotations
    for num_app in num_apps:
        count = len(merged_df[merged_df['num_apps'] == num_app])
        ax.text(num_app, -max(avg_error_pct)*0.05,
                f'n={count}',
                ha='center', va='top', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig(COMPARISON_GRAPH, format='pdf', dpi=300, bbox_inches='tight')
    print(f"[INFO] Graph saved to: {COMPARISON_GRAPH}")

    plt.close()

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"[SUCCESS] Comparison complete!")
    print(f"\n[INFO] Output files:")
    print(f"  Detailed comparison: {DETAILED_COMPARISON_CSV}")
    print(f"  Bar graph: {COMPARISON_GRAPH}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
