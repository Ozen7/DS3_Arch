#!/usr/bin/env python3
"""
Memory Saturation Comparison Graphing Script

This script generates bar charts comparing execution times across different
memory bandwidth saturation levels (DS3 results) with gem5 baseline results.

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

# Path resolution functions
def resolve_ds3_file(bandwidth):
    """
    Find DS3 results file for given bandwidth.
    Checks results_final/ first, then falls back to results/.
    """
    # Try results_final with full descriptive name
    primary = f'../results_final/experiment_results_RELIEF_NoCrit_MinList_{bandwidth}.csv'
    # Fall back to results directory with simple name
    secondary = f'../results/experiment_results_{bandwidth}.csv'

    if os.path.exists(primary):
        return primary
    elif os.path.exists(secondary):
        return secondary
    else:
        return None


def resolve_gem5_file():
    """Find gem5 baseline file in results_final/ or results_old/."""
    primary = '../results_final/gem5_comb_1_results.csv'
    secondary = '../results_old/gem5_comb_1_results.csv'
    fallback = '../results/gem5_comb_1_results.csv'

    for path in [primary, secondary, fallback]:
        if os.path.exists(path):
            return path
    return None


# Build file paths dynamically with fallback resolution
DS3_FILES = {}
for bw in [5280, 8000, 16000]:
    resolved_path = resolve_ds3_file(bw)
    if resolved_path:
        DS3_FILES[bw] = resolved_path

GEM5_FILE = resolve_gem5_file()

# Scheduler to gem5 policy mapping
SCHEDULER_POLICY_MAP = {
    'RELIEF': 'ELF',
    'LL': 'LAX',
    'GEDF_D': 'GEDF_D',
    'GEDF_N': 'GEDF_N',
    'HetSched': 'HetSched',
    'FCFS': 'FCFS'
}

# Workload list (alphabetical order)
WORKLOADS = ['canny', 'deblur', 'gru', 'harris', 'lstm']

# Baseline SoC configuration to filter for
BASELINE_SOC_CONFIG = 'SoC.i1_g1_cv1_h1_e1_c1_el1.txt'

# Graph configuration
COLORS = {
    5280: '#A8D8EA',   # Light blue
    8000: '#4A90A4',   # Medium blue
    16000: '#1B4F72',  # Dark blue
    'gem5': '#E67E22'  # Orange
}

BAR_WIDTH = 0.18


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_ds3_data(file_paths):
    """
    Load DS3 experiment data from multiple CSV files.

    Args:
        file_paths: Dictionary mapping bandwidth -> file path

    Returns:
        Dictionary structure: {scheduler: {workload: {bandwidth: time}}}
    """
    ds3_data = {}

    for bandwidth, filepath in file_paths.items():
        if not os.path.exists(filepath):
            print(f"[WARNING] DS3 file not found: {filepath}")
            continue

        print(f"[INFO] Loading DS3 data from {filepath} (BW: {bandwidth} MHz)")
        df = pd.read_csv(filepath)

        # Filter for single-workload runs with baseline SoC configuration
        single_workload_df = df[
            (df['workload_name'].str.startswith('single_')) &
            (df['soc_config'] == BASELINE_SOC_CONFIG)
        ]

        for _, row in single_workload_df.iterrows():
            scheduler = row['scheduler']
            workload_full = row['workload_name']
            # Strip 'single_' prefix
            workload = workload_full.replace('single_', '')
            execution_time = row['execution_time']

            # Initialize nested dictionaries if needed
            if scheduler not in ds3_data:
                ds3_data[scheduler] = {}
            if workload not in ds3_data[scheduler]:
                ds3_data[scheduler][workload] = {}

            ds3_data[scheduler][workload][bandwidth] = execution_time

    return ds3_data


def load_gem5_data(file_path):
    """
    Load gem5 simulation data from CSV file.

    Args:
        file_path: Path to gem5 results CSV

    Returns:
        Dictionary structure: {policy: {workload: time_us}}
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] gem5 file not found: {file_path}")
        sys.exit(1)

    print(f"[INFO] Loading gem5 data from {file_path}")
    df = pd.read_csv(file_path)

    gem5_data = {}

    # Process each row
    for _, row in df.iterrows():
        application = row['application']
        policy = row['policy']
        sim_seconds = row['sim_seconds']
        app_mix_str = row['app_mix_str']

        # Filter for single-application runs
        # Single-app runs have only one workload with count=1, others with count=0
        # Example: "canny_1_deblur_0_gru_0_harris_0_lstm_0_"
        app_counts = {}
        parts = app_mix_str.strip('_').split('_')
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                app_name = parts[i]
                count = int(parts[i + 1])
                app_counts[app_name] = count

        # Check if this is a single-application run
        total_apps = sum(app_counts.values())
        if total_apps == 1:
            # Convert sim_seconds to microseconds
            time_us = sim_seconds * 1_000_000

            # Initialize policy dictionary if needed
            if policy not in gem5_data:
                gem5_data[policy] = {}

            gem5_data[policy][application] = time_us

    return gem5_data


# ============================================================================
# Graph Generation Function
# ============================================================================

def create_comparison_graph(scheduler, ds3_data, gem5_data, output_dir='graphs'):
    """
    Create a grouped bar chart comparing DS3 saturation levels with gem5 baseline.

    Args:
        scheduler: Name of the DS3 scheduler
        ds3_data: DS3 data dictionary for this scheduler
        gem5_data: gem5 data dictionary
        output_dir: Directory to save the output PDF
    """
    # Get the corresponding gem5 policy
    if scheduler not in SCHEDULER_POLICY_MAP:
        print(f"[WARNING] No gem5 policy mapping for scheduler '{scheduler}', skipping")
        return

    gem5_policy = SCHEDULER_POLICY_MAP[scheduler]

    if gem5_policy not in gem5_data:
        print(f"[WARNING] gem5 policy '{gem5_policy}' not found in data, skipping '{scheduler}'")
        return

    # Prepare data arrays for plotting
    workload_labels = []
    ds3_5280_times = []
    ds3_8000_times = []
    ds3_16000_times = []
    gem5_times = []

    for workload in WORKLOADS:
        # Check if workload exists in DS3 data
        if workload not in ds3_data:
            print(f"[WARNING] Workload '{workload}' not found in DS3 data for scheduler '{scheduler}'")
            continue

        # Check if workload exists in gem5 data
        if workload not in gem5_data[gem5_policy]:
            print(f"[WARNING] Workload '{workload}' not found in gem5 data for policy '{gem5_policy}'")
            continue

        workload_labels.append(workload)

        # Get DS3 times for different bandwidths
        ds3_5280_times.append(ds3_data[workload].get(5280, 0))
        ds3_8000_times.append(ds3_data[workload].get(8000, 0))
        ds3_16000_times.append(ds3_data[workload].get(16000, 0))

        # Get gem5 time
        gem5_times.append(gem5_data[gem5_policy][workload])

    if not workload_labels:
        print(f"[WARNING] No data available for scheduler '{scheduler}', skipping graph")
        return

    # Normalize all times to gem5 baseline (gem5 becomes 1.0)
    ds3_5280_normalized = [ds3_5280_times[i] / gem5_times[i] for i in range(len(workload_labels))]
    ds3_8000_normalized = [ds3_8000_times[i] / gem5_times[i] for i in range(len(workload_labels))]
    ds3_16000_normalized = [ds3_16000_times[i] / gem5_times[i] for i in range(len(workload_labels))]
    gem5_normalized = [1.0] * len(workload_labels)  # gem5 baseline is always 1.0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(workload_labels))

    # Calculate offsets for grouped bars
    offset1 = -1.5 * BAR_WIDTH
    offset2 = -0.5 * BAR_WIDTH
    offset3 = 0.5 * BAR_WIDTH
    offset4 = 1.5 * BAR_WIDTH

    # Create bars with normalized values
    bars1 = ax.bar(x + offset1, ds3_5280_normalized, BAR_WIDTH,
                   label='DS3 @ 5280 MHz (1/3 BW)', color=COLORS[5280])
    bars2 = ax.bar(x + offset2, ds3_8000_normalized, BAR_WIDTH,
                   label='DS3 @ 8000 MHz (1/2 BW)', color=COLORS[8000])
    bars3 = ax.bar(x + offset3, ds3_16000_normalized, BAR_WIDTH,
                   label='DS3 @ 16000 MHz (Full BW)', color=COLORS[16000])
    bars4 = ax.bar(x + offset4, gem5_normalized, BAR_WIDTH,
                   label=f'gem5 Baseline ({gem5_policy})', color=COLORS['gem5'])

    # Configure chart elements - NO title, NO axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(workload_labels, fontsize=30)
    ax.tick_params(axis='y', labelsize=30)

    # Place legend at top center (where title was)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=20, frameon=True)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save to PDF
    output_path = os.path.join(output_dir, f'memory_saturation_{scheduler}.pdf')
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
    print("Memory Saturation Comparison Script")
    print("=" * 70)

    # Validate that required files exist
    print("\n[INFO] Validating input files...")
    all_files_exist = True

    for bandwidth, filepath in DS3_FILES.items():
        if not os.path.exists(filepath):
            print(f"[ERROR] DS3 file missing: {filepath}")
            all_files_exist = False
        else:
            print(f"[OK] Found: {filepath}")

    if not os.path.exists(GEM5_FILE):
        print(f"[ERROR] gem5 file missing: {GEM5_FILE}")
        all_files_exist = False
    else:
        print(f"[OK] Found: {GEM5_FILE}")

    if not all_files_exist:
        print("\n[ERROR] Missing required files. Please ensure all data files exist.")
        sys.exit(1)

    # Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)
    ds3_data = load_ds3_data(DS3_FILES)
    gem5_data = load_gem5_data(GEM5_FILE)

    # Print summary of loaded data
    print("\n[INFO] DS3 Data Summary:")
    for scheduler in sorted(ds3_data.keys()):
        workload_count = len(ds3_data[scheduler])
        print(f"  - {scheduler}: {workload_count} workloads")

    print("\n[INFO] gem5 Data Summary:")
    for policy in sorted(gem5_data.keys()):
        workload_count = len(gem5_data[policy])
        print(f"  - {policy}: {workload_count} workloads")

    # Generate graphs
    print("\n" + "=" * 70)
    print("Generating Graphs")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs('graphs', exist_ok=True)

    graph_count = 0
    for scheduler in sorted(ds3_data.keys()):
        print(f"\n[INFO] Processing scheduler: {scheduler}")
        create_comparison_graph(scheduler, ds3_data[scheduler], gem5_data)
        graph_count += 1

    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[INFO] Total graphs generated: {graph_count}")
    print(f"[INFO] Output directory: graphs/")
    print(f"[INFO] File pattern: memory_saturation_<scheduler>.pdf")
    print("\n[SUCCESS] Script completed successfully!")


if __name__ == "__main__":
    main()
