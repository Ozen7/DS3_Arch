#!/usr/bin/env python3
"""
gem5 vs DS3 Results Comparison and Graphing

This script compares gem5 and DS3 execution times, calculates percentage
errors and statistical measures (mean, median, variance, std dev, min, max),
and generates a box and whisker plot showing error distribution by number of jobs.

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
    'FCFS': 'FCFS'
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

    # Identify highest error cases
    print(f"\n{'='*80}")
    print(f"[INFO] HIGHEST ERROR CASES - Top 10")
    print(f"{'='*80}")

    # Sort by error descending and get top 10
    top_errors = merged_df.nlargest(10, 'abs_percentage_error')

    for idx, (_, row) in enumerate(top_errors.iterrows(), 1):
        print(f"\n{'─'*80}")
        print(f"RANK #{idx} - Error: {row['abs_percentage_error']:.2f}%")
        print(f"{'─'*80}")
        print(f"  Scheduler (gem5): {row['policy']}")
        print(f"  Scheduler (DS3):  {row['ds3_scheduler']}")
        print(f"  Number of Apps:   {int(row['num_apps'])}")
        print(f"  App Mix:          {row['app_mix_list_str']}")
        print(f"  gem5 Time:        {row['gem5_execution_time_us']:,.2f} μs")
        print(f"  DS3 Time:         {row['ds3_execution_time_us']:,.2f} μs")
        print(f"  Difference:       {abs(row['gem5_execution_time_us'] - row['ds3_execution_time_us']):,.2f} μs")

    # Analyze patterns in high-error cases
    print(f"\n{'='*80}")
    print(f"[INFO] HIGH ERROR PATTERN ANALYSIS (Top 20% of errors)")
    print(f"{'='*80}")

    # Define high error threshold (top 20%)
    error_threshold = merged_df['abs_percentage_error'].quantile(0.80)
    high_error_df = merged_df[merged_df['abs_percentage_error'] >= error_threshold]

    print(f"\nThreshold for high error: {error_threshold:.2f}%")
    print(f"Number of high-error cases: {len(high_error_df)} out of {len(merged_df)}")

    # Analyze by scheduler
    print(f"\n[ANALYSIS] High-Error Cases by Scheduler:")
    scheduler_counts = high_error_df.groupby('ds3_scheduler').size().sort_values(ascending=False)
    for scheduler, count in scheduler_counts.items():
        pct = (count / len(high_error_df)) * 100
        print(f"  {scheduler:12s}: {count:3d} cases ({pct:5.1f}%)")

    # Analyze by number of apps
    print(f"\n[ANALYSIS] High-Error Cases by Number of Apps:")
    app_counts = high_error_df.groupby('num_apps').size().sort_values(ascending=False)
    for num_apps, count in app_counts.items():
        pct = (count / len(high_error_df)) * 100
        print(f"  {int(num_apps):2d} apps: {count:3d} cases ({pct:5.1f}%)")

    # Analyze by app mix
    print(f"\n[ANALYSIS] Most Common App Mixes in High-Error Cases (Top 5):")
    mix_counts = high_error_df.groupby('app_mix_list_str').size().sort_values(ascending=False).head(5)
    for idx, (mix, count) in enumerate(mix_counts.items(), 1):
        avg_error = high_error_df[high_error_df['app_mix_list_str'] == mix]['abs_percentage_error'].mean()
        print(f"  #{idx} Mix: {mix}")
        print(f"      Count: {count}, Avg Error: {avg_error:.2f}%")

    # Save detailed comparison
    print(f"\n{'='*80}")
    print(f"[INFO] Saving detailed comparison to: {DETAILED_COMPARISON_CSV}")
    print(f"{'='*80}")
    merged_df.to_csv(DETAILED_COMPARISON_CSV, index=False)

    # Print ALL comparisons for verification
    print(f"\n{'='*80}")
    print(f"[INFO] COMPLETE COMPARISON DATA - All {len(merged_df)} Experiments")
    print(f"{'='*80}")

    # Sort by scheduler, then num_apps, then error for organized output
    sorted_df = merged_df.sort_values(['ds3_scheduler', 'num_apps', 'abs_percentage_error'],
                                       ascending=[True, True, False])

    current_scheduler = None
    for _, row in sorted_df.iterrows():
        # Print header when scheduler changes
        if row['ds3_scheduler'] != current_scheduler:
            current_scheduler = row['ds3_scheduler']
            print(f"\n{'='*80}")
            print(f"SCHEDULER: {current_scheduler}")
            print(f"{'='*80}")

        print(f"\n  Apps: {int(row['num_apps']):2d} | Error: {row['abs_percentage_error']:6.2f}% | Mix: {row['app_mix_list_str']}")
        print(f"    gem5: {row['gem5_execution_time_us']:12,.2f} μs | DS3: {row['ds3_execution_time_us']:12,.2f} μs | Diff: {abs(row['gem5_execution_time_us'] - row['ds3_execution_time_us']):10,.2f} μs")

    # Group by num_apps and calculate statistics
    print(f"\n{'='*80}")
    print(f"[INFO] STATISTICAL SUMMARY BY NUMBER OF JOBS")
    print(f"{'='*80}")

    # Group errors by num_apps
    grouped_errors = merged_df.groupby('num_apps')['abs_percentage_error']

    # Calculate comprehensive statistics
    stats = grouped_errors.agg(['mean', 'median', 'std', 'var', 'min', 'max', 'count'])

    print(f"\n{stats.to_string()}")
    print(f"\nStatistics shown: mean, median, std (standard deviation), var (variance), min, max, count")

    # Prepare data for box plot
    error_data_by_jobs = [merged_df[merged_df['num_apps'] == n]['abs_percentage_error'].values
                          for n in sorted(merged_df['num_apps'].unique())]
    num_apps_labels = sorted(merged_df['num_apps'].unique())

    # Generate box and whisker plot
    print(f"\n[INFO] Generating box and whisker plot...")

    # Set up publication-quality styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set background color to white for clean appearance
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create box plot with softer, publication-quality colors
    bp = ax.boxplot(error_data_by_jobs,
                    labels=[int(x) for x in num_apps_labels],
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D',
                                 markerfacecolor='#E57373',  # Soft red
                                 markersize=7,
                                 markeredgecolor='#C62828',
                                 markeredgewidth=1),
                    medianprops=dict(color='#37474F',  # Dark blue-gray
                                   linewidth=2),
                    boxprops=dict(facecolor='#90CAF9',  # Soft blue
                                edgecolor='#1565C0',  # Darker blue border
                                linewidth=1.5,
                                alpha=0.8),
                    whiskerprops=dict(linewidth=1.5,
                                    color='#424242',  # Dark gray
                                    linestyle='-'),
                    capprops=dict(linewidth=1.5,
                                color='#424242'),  # Dark gray
                    flierprops=dict(marker='o',
                                  markerfacecolor='#BDBDBD',  # Light gray
                                  markersize=5,
                                  markeredgecolor='#757575',
                                  markeredgewidth=0.5,
                                  alpha=0.6))

    # Clean up spines for publication quality
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['left'].set_color('#424242')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#424242')

    # Customize tick appearance
    ax.tick_params(axis='x', labelsize=18, length=6, width=1.5, color='#424242', pad=8)
    ax.tick_params(axis='y', labelsize=18, length=6, width=1.5, color='#424242')

    # Make y-axis ticks less frequent
    ax.locator_params(axis='y', nbins=6)

    # Add subtle grid for easier reading
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=1, color='#BDBDBD')
    ax.set_axisbelow(True)

    # Add variance and experiment count annotations
    # Adjust the y-position to be above/below the box plots without overlap
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add variance labels above each box plot and n= labels below
    for i, num_app in enumerate(num_apps_labels, 1):
        count = len(merged_df[merged_df['num_apps'] == num_app])
        data = merged_df[merged_df['num_apps'] == num_app]['abs_percentage_error']

        # Calculate standard deviation
        variance = data.var()
        sd = variance ** 0.5

        # Calculate standard deviation excluding outliers (using 1.5*IQR rule)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter out outliers
        data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]
        sd_no_outliers = data_no_outliers.std()

        # Standard deviation label (top line)
        ax.text(i, y_max + y_range * 0.08,
                f'σ={sd:.2f}',
                ha='center', va='bottom', fontsize=20, style='italic',
                color='#616161')

        # Standard deviation without outliers (second line)
        ax.text(i, y_max + y_range * 0.02,
                f'σ*={sd_no_outliers:.2f}',
                ha='center', va='bottom', fontsize=20, style='italic',
                color='#757575')

        # n= label below the x-axis
        ax.text(i, y_min - y_range * 0.08,
                f'n={count}',
                ha='center', va='top', fontsize=20, style='italic',
                color='#616161')

    # Extend y-axis to accommodate both variance and n= labels
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.14)

    # Ensure tight layout with extra padding at bottom for n= labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save with high quality
    plt.savefig(COMPARISON_GRAPH, format='pdf', dpi=300, bbox_inches='tight')
    print(f"[INFO] Graph saved to: {COMPARISON_GRAPH}")

    plt.close()

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"[SUCCESS] Comparison complete!")
    print(f"\n[INFO] Output files:")
    print(f"  Detailed comparison: {DETAILED_COMPARISON_CSV}")
    print(f"  Box and whisker plot: {COMPARISON_GRAPH}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
