#!/usr/bin/env python3
"""
SoC Resource Scaling Comparison Graphing Script

This script generates bar charts comparing single-workload performance across
different SoC resource configurations at 5280 MHz bandwidth.

Author: Generated for DS3 Architecture Research
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# ============================================================================
# Constants
# ============================================================================

# Path resolution function
def resolve_ds3_file(bandwidth):
    """Find DS3 results file for given bandwidth."""
    primary = f'../results_final/experiment_results_RELIEF_NoCrit_MinList_{bandwidth}.csv'
    secondary = f'../results/experiment_results_{bandwidth}.csv'

    if os.path.exists(primary):
        return primary
    elif os.path.exists(secondary):
        return secondary
    else:
        return None


# Resolve path to 5280 MHz data file
DATA_FILE = resolve_ds3_file(5280)
if DATA_FILE is None:
    print("[ERROR] Could not find experiment_results_5280.csv in results_final/ or results/")
    sys.exit(1)

# SoC configurations to compare
SOC_CONFIGS = {
    'all1': 'SoC.i1_g1_cv1_h1_e1_c1_el1.txt',
    'all2': 'SoC.i2_g2_cv2_h2_e2_c2_el2.txt',
    'all3': 'SoC.i3_g3_cv3_h3_e3_c3_el3.txt'
}

# Workload list (alphabetical order)
WORKLOADS = ['canny', 'deblur', 'gru', 'harris', 'lstm']

# Schedulers to process
SCHEDULERS = ['RELIEF', 'LL', 'GEDF_D', 'GEDF_N', 'HetSched']

# Graph configuration
COLORS = {
    'all1': '#90EE90',   # Light green
    'all2': '#4CAF50',   # Medium green
    'all3': '#1B5E20'    # Dark green
}

BAR_WIDTH = 0.25


# ============================================================================
# Data Loading Function
# ============================================================================

def load_soc_data(file_path, soc_configs):
    """
    Load DS3 experiment data from CSV file, filtering by SoC configurations.

    Args:
        file_path: Path to experiment_results_5280.csv
        soc_configs: Dictionary mapping labels to SoC config filenames

    Returns:
        Dictionary structure: {scheduler: {workload: {soc_label: execution_time}}}
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found: {file_path}")
        sys.exit(1)

    print(f"[INFO] Loading data from {file_path}")
    df = pd.read_csv(file_path)

    soc_data = {}

    # Filter for single-workload runs and specified SoC configs
    for soc_label, soc_config in soc_configs.items():
        single_workload_df = df[
            (df['workload_name'].str.startswith('single_')) &
            (df['soc_config'] == soc_config)
        ]

        for _, row in single_workload_df.iterrows():
            scheduler = row['scheduler']
            workload_full = row['workload_name']
            # Strip 'single_' prefix
            workload = workload_full.replace('single_', '')
            execution_time = row['execution_time']

            # Initialize nested dictionaries if needed
            if scheduler not in soc_data:
                soc_data[scheduler] = {}
            if workload not in soc_data[scheduler]:
                soc_data[scheduler][workload] = {}

            soc_data[scheduler][workload][soc_label] = execution_time

    return soc_data


# ============================================================================
# Graph Generation Function
# ============================================================================

def create_soc_comparison_graph(scheduler, soc_data, output_dir='graphs'):
    """
    Create a grouped bar chart comparing performance across SoC configurations.

    Args:
        scheduler: Name of the scheduler
        soc_data: SoC data dictionary for this scheduler
        output_dir: Directory to save the output PDF
    """
    # Prepare data arrays for plotting
    workload_labels = []
    all1_times = []
    all2_times = []
    all3_times = []

    for workload in WORKLOADS:
        # Check if workload exists in data
        if workload not in soc_data:
            print(f"[WARNING] Workload '{workload}' not found for scheduler '{scheduler}'")
            continue

        workload_labels.append(workload)

        # Get execution times for different SoC configs
        all1_times.append(soc_data[workload].get('all1', 0))
        all2_times.append(soc_data[workload].get('all2', 0))
        all3_times.append(soc_data[workload].get('all3', 0))

    if not workload_labels:
        print(f"[WARNING] No data available for scheduler '{scheduler}', skipping graph")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(workload_labels))

    # Calculate offsets for grouped bars
    offset1 = -BAR_WIDTH
    offset2 = 0
    offset3 = BAR_WIDTH

    # Create bars
    bars1 = ax.bar(x + offset1, all1_times, BAR_WIDTH,
                   label='all1 (1x Resources)', color=COLORS['all1'])
    bars2 = ax.bar(x + offset2, all2_times, BAR_WIDTH,
                   label='all2 (2x Resources)', color=COLORS['all2'])
    bars3 = ax.bar(x + offset3, all3_times, BAR_WIDTH,
                   label='all3 (3x Resources)', color=COLORS['all3'])

    # Configure chart elements
    ax.set_xlabel('Workload', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (μs)', fontsize=12, fontweight='bold')
    ax.set_title(f'SoC Resource Scaling Comparison - {scheduler}',
                fontsize=14, fontweight='bold', pad=35)
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels)

    # Place legend at top-left, outside the graphed area
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3, fontsize=9, frameon=True)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save to PDF
    output_path = os.path.join(output_dir, f'resource_scaling_{scheduler}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Generated graph: {output_path}")

    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("SoC Resource Scaling Comparison Script")
    print("=" * 70)

    # Validate that data file exists
    print("\n[INFO] Validating input file...")
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Data file missing: {DATA_FILE}")
        sys.exit(1)
    else:
        print(f"[OK] Found: {DATA_FILE}")

    # Display SoC configurations
    print("\n[INFO] Loading data for SoC configurations...")
    for label, config in SOC_CONFIGS.items():
        print(f"  - {label}: {config}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    soc_data = load_soc_data(DATA_FILE, SOC_CONFIGS)

    # Print summary of loaded data
    print("\n[INFO] Data loaded:")
    for scheduler in sorted(soc_data.keys()):
        workload_count = len(soc_data[scheduler])
        print(f"  - {scheduler}: {workload_count} workloads")

    # Generate graphs
    print("\n" + "=" * 70)
    print("Generating Graphs")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs('graphs', exist_ok=True)

    graph_count = 0
    for scheduler in sorted(soc_data.keys()):
        if scheduler in SCHEDULERS:
            print(f"\n[INFO] Processing scheduler: {scheduler}")
            create_soc_comparison_graph(scheduler, soc_data[scheduler])
            graph_count += 1

    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[INFO] Total graphs generated: {graph_count}")
    print(f"[INFO] Output directory: graphs/")
    print(f"[INFO] File pattern: resource_scaling_<scheduler>.pdf")
    print("\n[SUCCESS] Script completed successfully!")


if __name__ == "__main__":
    main()
