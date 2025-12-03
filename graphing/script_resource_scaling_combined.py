#!/usr/bin/env python3
"""
Combined SoC Resource Scaling Comparison Graph

Creates a single unified graph showing all schedulers and workloads together.
- Main bar groups: 5 workloads (canny, deblur, gru, harris, lstm)
- Sub-bars per workload: 6 schedulers
- Stacked bars per scheduler: 1x (baseline), 2x (doubled), 3x (tripled)

Author: Generated for DS3 Architecture Research
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import sys

# ============================================================================
# Constants
# ============================================================================

# Path resolution function
def resolve_ds3_file(bandwidth):
    """Find DS3 results file for given bandwidth."""
    primary = f'../results_final/experiment_results_RELIEF_NoCrit_min_{bandwidth}.csv'
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
    'all1': 'SoC.i1_g1_cv1_h1_e1_c1_el1.txt',  # Baseline
    'all2': 'SoC.i2_g2_cv2_h2_e2_c2_el2.txt',  # Doubled
    'all3': 'SoC.i3_g3_cv3_h3_e3_c3_el3.txt'   # Tripled
}

# Workload list (alphabetical order)
WORKLOADS = ['canny', 'deblur', 'gru', 'harris', 'lstm']

# Schedulers to process (in desired order)
SCHEDULERS = ['RELIEF', 'LL', 'GEDF_D', 'GEDF_N', 'HetSched', 'FCFS']

# Color scheme for schedulers - vibrant and distinct
SCHEDULER_COLORS = {
    'RELIEF': '#2E86AB',     # Deep blue
    'LL': '#FF6B35',         # Bright orange
    'GEDF_D': '#06A77D',     # Teal green
    'GEDF_N': '#D62828',     # Bright red
    'HetSched': '#9D4EDD',   # Purple
    'FCFS': '#8D5B4C'        # Brown
}

# Lightness variations for resource scaling (lighter = fewer resources)
# Each scheduler color will be shown in 3 shades
def get_color_for_resource(scheduler, resource_level):
    """Get color with appropriate lightness for resource level."""
    base_color = SCHEDULER_COLORS[scheduler]

    # Convert hex to RGB
    r = int(base_color[1:3], 16)
    g = int(base_color[3:5], 16)
    b = int(base_color[5:7], 16)

    # Apply lightness factor
    if resource_level == 'all1':  # Baseline - lighten
        factor = 1.5
    elif resource_level == 'all2':  # Doubled - medium
        factor = 1.15
    else:  # all3 - Tripled - full saturation
        factor = 1.0

    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))

    return f'#{r:02x}{g:02x}{b:02x}'

# Hatching patterns for 1x, 2x, 3x (for additional distinction)
HATCH_PATTERNS = {
    'all1': '',       # Baseline - no pattern
    'all2': '///',    # Doubled - diagonal lines
    'all3': 'xxx'     # Tripled - cross-hatch
}

BAR_WIDTH = 0.08  # Width of each stacked bar group
SCHEDULER_SPACING = 0.02  # Small gap between schedulers


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
        Dictionary structure: {workload: {scheduler: {soc_label: execution_time}}}
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

            # Initialize nested dictionaries if needed (workload first)
            if workload not in soc_data:
                soc_data[workload] = {}
            if scheduler not in soc_data[workload]:
                soc_data[workload][scheduler] = {}

            soc_data[workload][scheduler][soc_label] = execution_time

    return soc_data


# ============================================================================
# Graph Generation Function
# ============================================================================

def create_combined_resource_scaling_graph(soc_data, output_dir='graphs'):
    """
    Create a single combined bar chart showing all schedulers and workloads.
    Each scheduler has one stacked bar showing 1x, 2x, 3x performance.

    Args:
        soc_data: Dictionary {workload: {scheduler: {soc_label: execution_time}}}
        output_dir: Directory to save the output PDF
    """
    # Create figure with larger size for all data
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate positions
    num_workloads = len(WORKLOADS)
    num_schedulers = len(SCHEDULERS)

    # Space between workload groups
    workload_spacing = 0.5

    # Calculate total width per workload group
    group_width = num_schedulers * (BAR_WIDTH + SCHEDULER_SPACING)

    # X positions for workload groups
    workload_positions = np.arange(num_workloads) * (group_width + workload_spacing)

    # Store bar objects for legend
    legend_bars_scheduler = {}
    legend_bars_resource = {}

    # Iterate through workloads and schedulers
    for workload_idx, workload in enumerate(WORKLOADS):
        if workload not in soc_data:
            print(f"[WARNING] Workload '{workload}' not found in data")
            continue

        workload_data = soc_data[workload]
        base_x = workload_positions[workload_idx]

        for scheduler_idx, scheduler in enumerate(SCHEDULERS):
            if scheduler not in workload_data:
                print(f"[WARNING] Scheduler '{scheduler}' not found for workload '{workload}'")
                continue

            scheduler_data = workload_data[scheduler]

            # Get execution times for 1x, 2x, 3x
            time_1x = scheduler_data.get('all1', 0)
            time_2x = scheduler_data.get('all2', 0)
            time_3x = scheduler_data.get('all3', 0)

            # Calculate bar position for this scheduler
            bar_x = base_x + scheduler_idx * (BAR_WIDTH + SCHEDULER_SPACING)

            # Get colors for this scheduler with different lightness levels
            color_1x = get_color_for_resource(scheduler, 'all1')
            color_2x = get_color_for_resource(scheduler, 'all2')
            color_3x = get_color_for_resource(scheduler, 'all3')

            # Create stacked bars showing improvement from baseline (1x) down to tripled (3x)
            # Bottom segment: 3x (best performance, darkest)
            # Middle segment: 2x - 3x (improvement delta, medium)
            # Top segment: 1x - 2x (additional improvement delta, lightest)
            # Total height: 1x (baseline performance)

            bar_3x = ax.bar(bar_x, time_3x, BAR_WIDTH,
                           color=color_3x,
                           hatch=HATCH_PATTERNS['all3'],
                           edgecolor='black',
                           linewidth=0.5)

            bar_2x = ax.bar(bar_x, time_2x - time_3x, BAR_WIDTH,
                           bottom=time_3x,
                           color=color_2x,
                           hatch=HATCH_PATTERNS['all2'],
                           edgecolor='black',
                           linewidth=0.5)

            bar_1x = ax.bar(bar_x, time_1x - time_2x, BAR_WIDTH,
                           bottom=time_2x,
                           color=color_1x,
                           hatch=HATCH_PATTERNS['all1'],
                           edgecolor='black',
                           linewidth=0.5)

            # Store bars for legend (only once per scheduler/resource)
            if workload_idx == 0:
                if scheduler not in legend_bars_scheduler:
                    legend_bars_scheduler[scheduler] = Rectangle((0, 0), 1, 1,
                                                                  facecolor=color_2x,
                                                                  edgecolor='black',
                                                                  linewidth=1.0)
                if 'all1' not in legend_bars_resource:
                    legend_bars_resource['all1'] = bar_1x
                if 'all2' not in legend_bars_resource:
                    legend_bars_resource['all2'] = bar_2x
                if 'all3' not in legend_bars_resource:
                    legend_bars_resource['all3'] = bar_3x

    # Configure x-axis
    # Place tick at center of each workload group
    workload_centers = workload_positions + (group_width - SCHEDULER_SPACING) / 2
    ax.set_xticks(workload_centers)
    ax.set_xticklabels(WORKLOADS, fontsize=32, fontweight='bold')

    # Configure y-axis
    ax.tick_params(axis='y', labelsize=28)
    ax.set_ylabel('Execution Time (μs)', fontsize=32, fontweight='bold')

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

    # Create legend with two sections
    # Section 1: Schedulers (colors)
    scheduler_handles = [legend_bars_scheduler[s] for s in SCHEDULERS if s in legend_bars_scheduler]
    scheduler_labels = SCHEDULERS

    # Section 2: Resource scaling (hatching/position in stack)
    resource_handles = [
        legend_bars_resource['all1'],
        legend_bars_resource['all2'],
        legend_bars_resource['all3']
    ]
    resource_labels = ['Baseline (1x)', 'Doubled (2x)', 'Tripled (3x)']

    # Combine legends
    all_handles = scheduler_handles + resource_handles
    all_labels = scheduler_labels + resource_labels

    # Place legend at top
    ax.legend(all_handles, all_labels,
             loc='upper center',
             bbox_to_anchor=(0.5, 1.12),
             ncol=6,
             fontsize=20,
             frameon=True,
             columnspacing=1.0,
             handletextpad=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save to PDF
    output_path = os.path.join(output_dir, 'resource_scaling_combined.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Generated combined graph: {output_path}")

    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("Combined SoC Resource Scaling Graph Generator")
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
    for workload in sorted(soc_data.keys()):
        scheduler_count = len(soc_data[workload])
        print(f"  - {workload}: {scheduler_count} schedulers")

    # Generate combined graph
    print("\n" + "=" * 70)
    print("Generating Combined Graph")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs('graphs', exist_ok=True)

    create_combined_resource_scaling_graph(soc_data)

    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[INFO] Combined graph generated")
    print(f"[INFO] Output file: graphs/resource_scaling_combined.pdf")
    print("\n[SUCCESS] Script completed successfully!")


if __name__ == "__main__":
    main()
