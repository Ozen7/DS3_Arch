#!/usr/bin/env python3
"""
Execution Time Across Complexity Levels - Grouped Heatmap

This script generates a dense 4D visualization showing execution times across:
1. Workload complexity (rows)
2. Scheduler algorithms (column groups)
3. SoC resource complexity (sub-columns within each scheduler)
4. Execution time (color + annotation)

Author: Generated for DS3 Architecture Research
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

# ============================================================================
# Constants
# ============================================================================

# Path resolution function
def resolve_ds3_file(bandwidth):
    """Find DS3 results file for given bandwidth."""
    primary = f'../results_final/experiment_results_RELIEF_NoCrit_min_coloc_{bandwidth}.csv'
    secondary = f'../results/experiment_results_RELIEF_NoCrit_min_coloc_{bandwidth}.csv'
    fallback = f'../results_final/experiment_results_RELIEF_NoCrit_min_{bandwidth}.csv'

    if os.path.exists(primary):
        return primary
    elif os.path.exists(secondary):
        return secondary
    elif os.path.exists(fallback):
        return fallback
    else:
        return None


# Use 5280 MHz bandwidth data (most constrained, shows deadline pressure)
DATA_FILE = resolve_ds3_file(5280)
if DATA_FILE is None:
    print("[ERROR] Could not find experiment_results_RELIEF_NoCrit_min_coloc_5280.csv in results_final/ or results/")
    sys.exit(1)

# SoC configurations to compare
# Uniform scaling + individual 4x accelerator configs
SOC_CONFIGS = {
    '1x': 'SoC.i1_g1_cv1_h1_e1_c1_el1.txt',
    '2x': 'SoC.i2_g2_cv2_h2_e2_c2_el2.txt',
    '3x': 'SoC.i3_g3_cv3_h3_e3_c3_el3.txt',
    '4xI': 'SoC.i4_g1_cv1_h1_e1_c1_el1.txt',
    '4xG': 'SoC.i1_g4_cv1_h1_e1_c1_el1.txt',
    '4xC': 'SoC.i1_g1_cv4_h1_e1_c1_el1.txt',
    '4xHNM': 'SoC.i1_g1_cv1_h4_e1_c1_el1.txt',
    '4xET': 'SoC.i1_g1_cv1_h1_e4_c1_el1.txt',
    '4xCNM': 'SoC.i1_g1_cv1_h1_e1_c4_el1.txt',
    '4xEM': 'SoC.i1_g1_cv1_h1_e1_c1_el4.txt'
}

# Workload mixes to compare
WORKLOADS = ['mixed_balanced', 'mixed_vision_heavy', 'mixed_ml_heavy']

# Schedulers to process (order matters for visualization)
SCHEDULERS = ['FCFS', 'GEDF_D', 'GEDF_N', 'HetSched', 'LL', 'RELIEF']

# Colormap for heatmap (white to dark red - darker = slower/worse)
CMAP = 'Reds'


# ============================================================================
# Data Loading Function
# ============================================================================

def load_execution_time_data(file_path, soc_configs, schedulers, workloads):
    """
    Load execution time data from CSV file.

    Args:
        file_path: Path to experiment results CSV
        soc_configs: Dictionary mapping labels to SoC config filenames
        schedulers: List of scheduler names
        workloads: List of workload names

    Returns:
        Dictionary structure: {(workload, scheduler, soc_label): execution_time}
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Data file not found: {file_path}")
        sys.exit(1)

    print(f"[INFO] Loading data from {file_path}")
    df = pd.read_csv(file_path)

    execution_data = {}

    # Filter for mixed-workload runs
    for soc_label, soc_config in soc_configs.items():
        mixed_workload_df = df[
            (df['workload_name'].isin(workloads)) &
            (df['soc_config'] == soc_config)
        ]

        for _, row in mixed_workload_df.iterrows():
            scheduler = row['scheduler']
            workload = row['workload_name']

            # Skip if not in our lists
            if scheduler not in schedulers or workload not in workloads:
                continue

            execution_time = row['execution_time']

            # Store with composite key
            key = (workload, scheduler, soc_label)
            execution_data[key] = execution_time

    return execution_data


# ============================================================================
# Matrix Construction Function
# ============================================================================

def build_heatmap_matrix(execution_data, workloads, schedulers, soc_configs):
    """
    Build a 2D matrix for heatmap visualization.

    Rows: Workloads
    Columns: Scheduler × SoC complexity (flattened)

    Args:
        execution_data: Dictionary with (workload, scheduler, soc) -> execution_time
        workloads: List of workload names
        schedulers: List of scheduler names
        soc_configs: Dictionary of SoC configuration labels

    Returns:
        - matrix: 2D numpy array for heatmap
        - col_labels: List of column labels
        - row_labels: List of row labels
        - global_max: Maximum execution time across all data
    """
    num_workloads = len(workloads)
    num_schedulers = len(schedulers)
    num_soc_configs = len(soc_configs)
    num_cols = num_schedulers * num_soc_configs

    # Initialize matrix
    matrix = np.zeros((num_workloads, num_cols))

    # Build column labels
    col_labels = []
    soc_order = list(soc_configs.keys())  # ['1x', '2x', '3x', ...]

    for scheduler in schedulers:
        for soc_label in soc_order:
            col_labels.append(f"{scheduler}\n{soc_label}")

    # Create shortened labels for better readability
    row_labels = [w.replace('mixed_', '') for w in workloads]

    # Fill matrix and track global max
    global_max = 0
    for i, workload in enumerate(workloads):
        col_idx = 0
        for scheduler in schedulers:
            for soc_label in soc_order:
                key = (workload, scheduler, soc_label)
                exec_time = execution_data.get(key, 0.0)
                matrix[i, col_idx] = exec_time
                global_max = max(global_max, exec_time)
                col_idx += 1

    return matrix, col_labels, row_labels, global_max


# ============================================================================
# Graph Generation Function
# ============================================================================

def create_execution_time_heatmap(execution_data, workloads, schedulers, soc_configs,
                                  output_dir='graphs'):
    """
    Create a grouped heatmap showing execution time across all dimensions.

    Args:
        execution_data: Dictionary with execution times
        workloads: List of workload names
        schedulers: List of scheduler names
        soc_configs: Dictionary of SoC configuration labels
        output_dir: Directory to save the output PDF
    """
    # Build matrix
    matrix, col_labels, row_labels, global_max = build_heatmap_matrix(
        execution_data, workloads, schedulers, soc_configs
    )

    print(f"[INFO] Global maximum execution time: {global_max:.0f} μs")

    # Create figure with appropriate size
    num_cols = len(col_labels)
    num_rows = len(row_labels)

    # Wider figure to accommodate 10 SoC configs × 6 schedulers = 60 columns
    fig_width = max(24, num_cols * 0.5)
    fig_height = max(6, num_rows * 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create heatmap with global max as vmax
    im = ax.imshow(matrix, cmap=CMAP, aspect='auto', vmin=0, vmax=global_max)

    # Set ticks
    ax.set_xticks(np.arange(num_cols))
    ax.set_yticks(np.arange(num_rows))

    # Set tick labels (smaller fonts for many columns)
    ax.set_xticklabels(col_labels, fontsize=7, ha='right')
    ax.set_yticklabels(row_labels, fontsize=14)

    # Rotate x-axis labels to 90 degrees for readability
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    # Add text annotations showing execution times
    for i in range(num_rows):
        for j in range(num_cols):
            value = matrix[i, j]
            # Choose text color based on background intensity
            # Normalize value to 0-1 range for color decision
            normalized = value / global_max if global_max > 0 else 0
            text_color = 'white' if normalized > 0.5 else 'black'

            # Format large numbers with comma separator
            if value >= 1000:
                text = ax.text(j, i, f'{value:,.0f}',
                              ha="center", va="center",
                              color=text_color, fontsize=6, weight='bold')
            else:
                text = ax.text(j, i, f'{value:.0f}',
                              ha="center", va="center",
                              color=text_color, fontsize=6, weight='bold')

    # Add vertical lines to separate scheduler groups
    num_soc_configs = len(soc_configs)
    for i in range(1, len(schedulers)):
        x_pos = i * num_soc_configs - 0.5
        ax.axvline(x=x_pos, color='white', linewidth=3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Execution Time (μs)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add scheduler group labels at the top
    num_soc = len(soc_configs)
    for idx, scheduler in enumerate(schedulers):
        center_x = (idx * num_soc) + (num_soc - 1) / 2.0
        ax.text(center_x, -1.2, scheduler,
               ha='center', va='bottom', fontsize=12, weight='bold',
               transform=ax.transData)

    # Add title above everything
    fig.suptitle('Execution Time: Workload Mix × Scheduler × SoC Complexity',
                fontsize=16, weight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to PDF
    output_path = os.path.join(output_dir, 'execution_time_heatmap.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Generated graph: {output_path}")

    plt.close()


# ============================================================================
# Alternative: Individual Scheduler Heatmaps (Faceted View)
# ============================================================================

def create_faceted_execution_time_heatmaps(execution_data, workloads, schedulers,
                                           soc_configs, output_dir='graphs'):
    """
    Create a faceted view with one heatmap per scheduler.
    Each subplot shows: Workload (rows) × SoC Complexity (columns)

    Args:
        execution_data: Dictionary with execution times
        workloads: List of workload names
        schedulers: List of scheduler names
        soc_configs: Dictionary of SoC configuration labels
        output_dir: Directory to save the output PDF
    """
    num_schedulers = len(schedulers)
    soc_order = list(soc_configs.keys())
    num_soc = len(soc_order)
    num_workloads = len(workloads)

    # Calculate global max for consistent coloring
    global_max = max(execution_data.values())
    print(f"[INFO] Global maximum execution time for faceted view: {global_max:.0f} μs")

    # Create subplots (2 rows, 3 columns for 6 schedulers)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]

        # Build matrix for this scheduler
        matrix = np.zeros((num_workloads, num_soc))

        for i, workload in enumerate(workloads):
            for j, soc_label in enumerate(soc_order):
                key = (workload, scheduler, soc_label)
                exec_time = execution_data.get(key, 0.0)
                matrix[i, j] = exec_time

        # Create heatmap with global max
        im = ax.imshow(matrix, cmap=CMAP, aspect='auto', vmin=0, vmax=global_max)

        # Set ticks and labels (shortened workload names on left column only)
        ax.set_xticks(np.arange(num_soc))
        ax.set_yticks(np.arange(num_workloads))
        ax.set_xticklabels(soc_order, fontsize=12, rotation=90, ha='right')
        # Show shortened labels only on left column
        shortened_labels = [w.replace('mixed_', '') for w in workloads] if idx % 3 == 0 else []
        ax.set_yticklabels(shortened_labels, fontsize=11)

        # Add title
        ax.set_title(scheduler, fontsize=14, weight='bold', pad=10)

        # Add text annotations
        for i in range(num_workloads):
            for j in range(num_soc):
                value = matrix[i, j]
                normalized = value / global_max if global_max > 0 else 0
                text_color = 'white' if normalized > 0.5 else 'black'

                # Format large numbers
                if value >= 1000:
                    ax.text(j, i, f'{value:,.0f}',
                           ha="center", va="center",
                           color=text_color, fontsize=9, weight='bold')
                else:
                    ax.text(j, i, f'{value:.0f}',
                           ha="center", va="center",
                           color=text_color, fontsize=9, weight='bold')

        # Grid lines
        ax.set_xticks(np.arange(num_soc + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_workloads + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)

    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('Execution Time (μs)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Add overall title
    fig.suptitle('Execution Time by Scheduler (Workload Mix × SoC Complexity)',
                fontsize=16, weight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])

    # Save to PDF
    output_path = os.path.join(output_dir, 'execution_time_faceted.pdf')
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
    print("Execution Time Visualization Script")
    print("=" * 70)

    # Validate that data file exists
    print("\n[INFO] Validating input file...")
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Data file missing: {DATA_FILE}")
        sys.exit(1)
    else:
        print(f"[OK] Found: {DATA_FILE}")

    # Load data
    print("\n" + "=" * 70)
    print("Loading Execution Time Data")
    print("=" * 70)
    execution_data = load_execution_time_data(DATA_FILE, SOC_CONFIGS, SCHEDULERS, WORKLOADS)

    # Print summary
    print(f"\n[INFO] Loaded {len(execution_data)} data points")
    print(f"[INFO] Dimensions:")
    print(f"  - Workloads: {len(WORKLOADS)} ({', '.join(WORKLOADS)})")
    print(f"  - Schedulers: {len(SCHEDULERS)} ({', '.join(SCHEDULERS)})")
    print(f"  - SoC Configs: {len(SOC_CONFIGS)} ({', '.join(SOC_CONFIGS.keys())})")

    # Generate graphs
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs('graphs', exist_ok=True)

    # Create grouped heatmap (dense, single plot)
    print("\n[INFO] Generating grouped heatmap...")
    create_execution_time_heatmap(execution_data, WORKLOADS, SCHEDULERS, SOC_CONFIGS)

    # Create faceted view (one heatmap per scheduler)
    print("\n[INFO] Generating faceted heatmaps...")
    create_faceted_execution_time_heatmaps(execution_data, WORKLOADS, SCHEDULERS, SOC_CONFIGS)

    # Final summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[INFO] Generated 2 visualizations")
    print(f"[INFO] Output directory: graphs/")
    print(f"[INFO] Files generated:")
    print(f"  - execution_time_heatmap.pdf (grouped, dense)")
    print(f"  - execution_time_faceted.pdf (per-scheduler subplots)")
    print("\n[SUCCESS] Script completed successfully!")


if __name__ == "__main__":
    main()
