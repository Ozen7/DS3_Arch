#!/usr/bin/env python3
"""
gem5-Matching DS3 Experiment Runner

This script runs DS3 experiments matching gem5 configurations.
Based on scripts/run_experiment_sweep.py (simplified version).

Usage:
    python3 graphing/run_gem5_matching_experiments.py
"""

import subprocess
import configparser
import csv
import os
import re
import time
from datetime import datetime
import sys
import pandas as pd
import ast

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Scheduler mapping: gem5 → DS3
SCHEDULER_MAPPING = {
    'LAX': 'LL',
    'ELFD': 'RELIEF',
    'ELF': 'RELIEF',  # Fallback if ELFD not available
    'GEDF_D': 'GEDF_D',
    'GEDF_N': 'GEDF_N',
    'HetSched': 'HetSched',
    'FCFS': 'FCFS'
}

# Paths
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_final')
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config_file.ini')
DASH_SIM_SCRIPT = os.path.join(PROJECT_ROOT, 'DASH_Sim_v0.py')

# Input/output files
GEM5_COMBINED_CSV = os.path.join(RESULTS_DIR, 'gem5_combined_results.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'ds3_gem5_matching_results.csv')

# Configuration
SOC_CONFIG = 'SoC.i1_g1_cv1_h1_e1_c1_el1.txt'  # 1 of each accelerator
JOB_FILES = 'job_canny.txt,job_deblur.txt,job_gru.txt,job_harris.txt,job_lstm.txt'
SIMULATION_MODE = 'performance'
SIMULATION_TIMEOUT = 600  # 10 minutes


def app_mix_list_to_job_list(app_mix_list):
    """
    Convert app_mix_list to job_list string.

    Args:
        app_mix_list: List like [1, 0, 0, 0, 0] or string representation

    Returns:
        String like "[[1,0,0,0,0]]"
    """
    # Handle string representation of list
    if isinstance(app_mix_list, str):
        app_mix_list = ast.literal_eval(app_mix_list)

    return f"[[{','.join(map(str, app_mix_list))}]]"


def modify_config_for_experiment(scheduler, job_list_str):
    """
    Modify config_file.ini with experiment parameters.

    Args:
        scheduler: DS3 scheduler name
        job_list_str: Job list string like "[[1,0,0,0,0]]"
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    config['DEFAULT']['resource_file'] = SOC_CONFIG
    config['DEFAULT']['scheduler'] = scheduler
    config['DEFAULT']['job_file'] = JOB_FILES
    config['DEFAULT']['job_list'] = job_list_str
    config['DEFAULT']['job_probabilities'] = '[]'
    config['SIMULATION MODE']['simulation_mode'] = SIMULATION_MODE

    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)


def run_ds3_experiment():
    """
    Run DASH_Sim_v0.py and extract results.

    Returns:
        Dictionary with result metrics, or None if failed
    """
    try:
        result = subprocess.run(
            ['python3', DASH_SIM_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=SIMULATION_TIMEOUT,
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"[ERROR] Simulation failed with return code {result.returncode}")
            print(f"[ERROR] stderr: {result.stderr[:500]}")
            return None

        # Parse output
        output = result.stdout

        # Extract execution_time using regex
        execution_time_match = re.search(r'\[I\] Execution time\(us\)\s+:\s+([\d.]+)', output)
        if not execution_time_match:
            print("[ERROR] Could not find execution_time in output")
            return None

        execution_time = float(execution_time_match.group(1))

        # Extract completed_jobs
        completed_jobs_match = re.search(r'\[I\] Number of completed jobs:\s+(\d+)', output)
        completed_jobs = int(completed_jobs_match.group(1)) if completed_jobs_match else None

        # Extract deadlines_met and deadlines_missed
        deadlines_met_match = re.search(r'\[I\] Number of deadlines met:\s+(\d+)', output)
        deadlines_missed_match = re.search(r'\[I\] Number of deadlines missed:\s+(\d+)', output)

        deadlines_met = int(deadlines_met_match.group(1)) if deadlines_met_match else None
        deadlines_missed = int(deadlines_missed_match.group(1)) if deadlines_missed_match else None

        return {
            'ds3_execution_time_us': execution_time,
            'completed_jobs': completed_jobs,
            'deadlines_met': deadlines_met,
            'deadlines_missed': deadlines_missed,
        }

    except subprocess.TimeoutExpired:
        print("[ERROR] Simulation timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        return None


def main():
    """Main function to run gem5-matching experiments."""
    print("[INFO] gem5-Matching DS3 Experiment Runner")
    print("=" * 60)

    # Check if gem5_combined_results.csv exists
    if not os.path.exists(GEM5_COMBINED_CSV):
        print(f"[ERROR] gem5 combined results not found: {GEM5_COMBINED_CSV}")
        print("[ERROR] Please run graphing/combine_gem5_results.py first")
        return 1

    # Load gem5 results
    print(f"\n[INFO] Loading gem5 results from: {GEM5_COMBINED_CSV}")
    gem5_df = pd.read_csv(GEM5_COMBINED_CSV)
    print(f"[INFO] Loaded {len(gem5_df)} gem5 results")

    # Filter for valid schedulers
    print(f"\n[INFO] Filtering for valid schedulers...")
    valid_schedulers = list(SCHEDULER_MAPPING.keys())
    gem5_df_filtered = gem5_df[gem5_df['policy'].isin(valid_schedulers)].copy()  # ← Add .copy()
    print(f"[INFO] {len(gem5_df_filtered)} experiments after filtering")

    # Get unique configurations (app_mix_list + scheduler combinations)
    print(f"\n[INFO] Extracting unique experiment configurations...")

    # Convert app_mix_list from string representation if needed
    def safe_eval_app_mix_list(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    gem5_df_filtered['app_mix_list'] = gem5_df_filtered['app_mix_list'].apply(safe_eval_app_mix_list)

    # Create a string representation for grouping
    gem5_df_filtered['app_mix_str_key'] = gem5_df_filtered['app_mix_list'].apply(str)

    # Drop duplicates using only hashable columns
    unique_configs = gem5_df_filtered[['app_mix_str_key', 'policy', 'num_apps']].drop_duplicates()

    # Add back app_mix_list by merging
    unique_configs = unique_configs.merge(
        gem5_df_filtered[['app_mix_str_key', 'app_mix_list']].drop_duplicates(subset='app_mix_str_key'),
        on='app_mix_str_key',
        how='left'
    )
    print(f"[INFO] Found {len(unique_configs)} unique configurations to run")

    # Show summary
    print(f"\n[INFO] Configuration summary:")
    print(f"  Schedulers: {unique_configs['policy'].value_counts().to_dict()}")
    print(f"  By num_apps: {unique_configs['num_apps'].value_counts().to_dict()}")

    # Prepare results list
    results = []

    # Run experiments
    print(f"\n[INFO] Running DS3 experiments...")
    print("=" * 60)

    total = len(unique_configs)
    for idx, (_, row) in enumerate(unique_configs.iterrows(), 1):
        app_mix_list = row['app_mix_list']
        gem5_policy = row['policy']
        num_apps = row['num_apps']

        # Map scheduler
        ds3_scheduler = SCHEDULER_MAPPING[gem5_policy]

        # Convert app_mix_list to job_list string
        job_list_str = app_mix_list_to_job_list(app_mix_list)

        print(f"\n[{idx}/{total}] Experiment: policy={gem5_policy}→{ds3_scheduler}, apps={app_mix_list}, num={num_apps}")

        # Modify config
        modify_config_for_experiment(ds3_scheduler, job_list_str)

        # Run experiment
        print(f"[{idx}/{total}] Running simulation...")
        start_time = time.time()
        result = run_ds3_experiment()
        elapsed = time.time() - start_time

        if result is None:
            print(f"[{idx}/{total}] FAILED (after {elapsed:.1f}s)")
            # Record failure
            results.append({
                'experiment_num': idx,
                'app_mix_list': str(app_mix_list),
                'gem5_policy': gem5_policy,
                'ds3_scheduler': ds3_scheduler,
                'num_apps': num_apps,
                'ds3_execution_time_us': None,
                'completed_jobs': None,
                'deadlines_met': None,
                'deadlines_missed': None,
                'status': 'failed',
            })
        else:
            print(f"[{idx}/{total}] SUCCESS (after {elapsed:.1f}s)")
            print(f"[{idx}/{total}]   execution_time: {result['ds3_execution_time_us']:.2f} μs")
            print(f"[{idx}/{total}]   completed_jobs: {result['completed_jobs']}")

            # Record success
            results.append({
                'experiment_num': idx,
                'app_mix_list': str(app_mix_list),
                'gem5_policy': gem5_policy,
                'ds3_scheduler': ds3_scheduler,
                'num_apps': num_apps,
                'ds3_execution_time_us': result['ds3_execution_time_us'],
                'completed_jobs': result['completed_jobs'],
                'deadlines_met': result['deadlines_met'],
                'deadlines_missed': result['deadlines_missed'],
                'status': 'success',
            })

    # Write results to CSV
    print(f"\n" + "=" * 60)
    print(f"[INFO] Writing results to: {OUTPUT_CSV}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    print(f"\n[INFO] Experiment Summary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful: {len(results_df[results_df['status'] == 'success'])}")
    print(f"  Failed: {len(results_df[results_df['status'] == 'failed'])}")

    if len(results_df[results_df['status'] == 'failed']) > 0:
        print(f"\n[WARNING] Some experiments failed. Check the output above for details.")

    print(f"\n[SUCCESS] Results saved to: {OUTPUT_CSV}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
