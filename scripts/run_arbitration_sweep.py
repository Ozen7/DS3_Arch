#!/usr/bin/env python3
"""
Automated Experiment Runner for DS3 (DASH-Sim) - Arbitration Type Sweep

This script automates running experiments across multiple:
- SoC configurations (auto-generated via create_RELIEF_SoC.py)
- Workload configurations (job_list and job_probabilities)
- Schedulers (RELIEF, LL, GEDF_D, GEDF_N, HetSched)
- Arbitration types (min, min_coloc, random, exectime)

Fixed bandwidth: 5280 bytes/μs

Usage:
    python3 run_arbitration_sweep.py
"""

import subprocess
import configparser
import csv
import os
import re
import time
from datetime import datetime
import sys

# Get the directory containing this script and the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of scripts/ directory

# ========== EXPERIMENT CONFIGURATION (EDIT HERE) ==========

# Simulation mode: 'performance' or 'validation'
SIMULATION_MODE = 'performance'

# Fixed bandwidth for arbitration sweep (bytes/microsecond)
bandwidth = 5280

# SoC configurations to generate and test
# Format: (filename, accelerator_specs)
# accelerator_specs: dict of {acc_type: count}
SOC_CONFIGS = [
    # Baseline configurations
    ('SoC.i1_g1_cv1_h1_e1_c1_el1.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 1, 'harris': 1, 'edge': 1, 'canny': 1, 'elem': 1}),
    ('SoC.i2_g2_cv2_h2_e2_c2_el2.txt',
     {'isp': 2, 'grayscale': 2, 'conv': 2, 'harris': 2, 'edge': 2, 'canny': 2, 'elem': 2}),
    ('SoC.i3_g3_cv3_h3_e3_c3_el3.txt',
     {'isp': 3, 'grayscale': 3, 'conv': 3, 'harris': 3, 'edge': 3, 'canny': 3, 'elem': 3}),

    # Individual 4x variants (one accelerator type has 4, rest have 1)
    ('SoC.i4_g1_cv1_h1_e1_c1_el1.txt',
     {'isp': 4, 'grayscale': 1, 'conv': 1, 'harris': 1, 'edge': 1, 'canny': 1, 'elem': 1}),
    ('SoC.i1_g4_cv1_h1_e1_c1_el1.txt',
     {'isp': 1, 'grayscale': 4, 'conv': 1, 'harris': 1, 'edge': 1, 'canny': 1, 'elem': 1}),
    ('SoC.i1_g1_cv4_h1_e1_c1_el1.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 4, 'harris': 1, 'edge': 1, 'canny': 1, 'elem': 1}),
    ('SoC.i1_g1_cv1_h4_e1_c1_el1.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 1, 'harris': 4, 'edge': 1, 'canny': 1, 'elem': 1}),
    ('SoC.i1_g1_cv1_h1_e4_c1_el1.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 1, 'harris': 1, 'edge': 4, 'canny': 1, 'elem': 1}),
    ('SoC.i1_g1_cv1_h1_e1_c4_el1.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 1, 'harris': 1, 'edge': 1, 'canny': 4, 'elem': 1}),
    ('SoC.i1_g1_cv1_h1_e1_c1_el4.txt',
     {'isp': 1, 'grayscale': 1, 'conv': 1, 'harris': 1, 'edge': 1, 'canny': 1, 'elem': 4}),

    # Add more SoC configurations here easily...
]

# Schedulers to test
SCHEDULERS = [
    'RELIEF',
    'LL',
    'GEDF_D',
    'GEDF_N',
    'HetSched',
]

# Workload configurations (easily extensible)
# job_file order: canny, deblur, gru, harris, lstm
WORKLOAD_CONFIGS = [
    # Single workloads - one job of each type
    {'name': 'single_canny', 'job_list': '[[1,0,0,0,0]]', 'job_probabilities': '[]'},
    {'name': 'single_deblur', 'job_list': '[[0,1,0,0,0]]', 'job_probabilities': '[]'},
    {'name': 'single_gru', 'job_list': '[[0,0,1,0,0]]', 'job_probabilities': '[]'},
    {'name': 'single_harris', 'job_list': '[[0,0,0,1,0]]', 'job_probabilities': '[]'},
    {'name': 'single_lstm', 'job_list': '[[0,0,0,0,1]]', 'job_probabilities': '[]'},

    # Mixed workloads - various combinations
    {'name': 'mixed_balanced', 'job_list': '[[1,1,1,1,1]]', 'job_probabilities': '[]'},
    {'name': 'mixed_vision_heavy', 'job_list': '[[2,2,0,2,0]]', 'job_probabilities': '[]'},
    {'name': 'mixed_ml_heavy', 'job_list': '[[0,0,2,0,2]]', 'job_probabilities': '[]'},

    # Add more workload configurations here easily...
]

# Arbitration types to sweep through
ARBITRATION_TYPES = ['min', 'min_coloc', 'random', 'exectime']

# Paths (relative to PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_final')
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config_file.ini')
SOC_CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config_SoC')
DASH_SIM_SCRIPT = os.path.join(PROJECT_ROOT, 'DASH_Sim_v0.py')
CREATE_SOC_SCRIPT = os.path.join(SCRIPT_DIR, 'create_RELIEF_SoC.py')

# Ensure results_final directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simulation timeout (seconds)
SIMULATION_TIMEOUT = 600  # 10 minutes

# ========== END OF CONFIGURATION ==========


# ========== HELPER FUNCTIONS ==========

def generate_soc_config(soc_filename, acc_specs, bandwidth):
    """
    Generate SoC configuration file using create_RELIEF_SoC.py.

    Args:
        soc_filename: Name of the SoC configuration file
        acc_specs: Dictionary of {accelerator_type: count}
        bandwidth: Memory bandwidth in bytes/microsecond

    Returns:
        True if successful, False otherwise
    """
    soc_path = os.path.join(SOC_CONFIG_DIR, soc_filename)

    # Delete existing file if it exists
    if os.path.exists(soc_path):
        print(f"[SoC] Deleting existing {soc_filename}")
        os.remove(soc_path)

    print(f"[SoC] Generating {soc_filename}...")

    # Build command-line arguments for create_RELIEF_SoC.py
    # Format: python3 create_RELIEF_SoC.py isp 1 grayscale 1 conv 1 ...
    cmd_args = ['python3', CREATE_SOC_SCRIPT]

    # Add accelerator type and count pairs in the correct order
    # Order from create_RELIEF_SoC.py: isp, grayscale, conv, harris, edge, canny, elem
    acc_order = ['isp', 'grayscale', 'conv', 'harris', 'edge', 'canny', 'elem']

    for acc_type in acc_order:
        if acc_type in acc_specs and acc_specs[acc_type] > 0:
            cmd_args.extend([acc_type, str(acc_specs[acc_type])])

    # Set bandwidth via environment variable for create_RELIEF_SoC.py
    env = os.environ.copy()
    env['DS3_BANDWIDTH'] = str(bandwidth)

    try:
        result = subprocess.run(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30,
            env=env,  # Pass modified environment
            cwd=PROJECT_ROOT  # Run from project root
        )

        if result.returncode == 0:
            print(f"[SoC] Successfully generated {soc_filename}")
            return True
        else:
            print(f"[SoC] Failed to generate {soc_filename}")
            print(f"[SoC] Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[SoC] Timeout generating {soc_filename}")
        return False
    except Exception as e:
        print(f"[SoC] Error generating {soc_filename}: {e}")
        return False


def backup_config_file():
    """
    Create a backup of the original config_file.ini.

    Returns:
        Path to backup file
    """
    if os.path.exists(CONFIG_FILE):
        backup_file = f"{CONFIG_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        subprocess.run(['cp', CONFIG_FILE, backup_file])
        print(f"[Backup] Created backup: {backup_file}")
        return backup_file
    return None


def modify_config(soc_config, scheduler, workload_config, arbitration_type):
    """
    Modify the config_file.ini with the specified parameters.

    Args:
        soc_config: SoC configuration filename
        scheduler: Scheduler name
        workload_config: Dictionary with 'job_list' and 'job_probabilities'
        arbitration_type: Arbitration type (min, min_coloc, random, exectime)
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    # Modify the parameters
    config['DEFAULT']['resource_file'] = soc_config
    config['DEFAULT']['scheduler'] = scheduler
    config['DEFAULT']['job_list'] = workload_config['job_list']
    config['DEFAULT']['job_probabilities'] = workload_config['job_probabilities']
    config['DEFAULT']['arbitration_type'] = arbitration_type
    config['SIMULATION MODE']['simulation_mode'] = SIMULATION_MODE

    # Write back to the config file
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

    print(f"[Config] SoC={soc_config}, Scheduler={scheduler}, Workload={workload_config['name']}, Arbitration={arbitration_type}")


def run_simulation():
    """
    Run the DASH_Sim_v0.py simulation and capture output.

    Returns:
        Tuple of (success: bool, output: str, error: str)
    """
    try:
        print("[Run] Starting simulation...")
        result = subprocess.run(
            ['python3', DASH_SIM_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=SIMULATION_TIMEOUT,
            cwd=PROJECT_ROOT  # Run from project root
        )

        if result.returncode == 0:
            print("[Run] Simulation completed successfully")
            return True, result.stdout, result.stderr
        else:
            print(f"[Run] Simulation failed with return code {result.returncode}")
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print("[Run] Simulation timed out")
        return False, "", "Timeout"
    except Exception as e:
        print(f"[Run] Error running simulation: {e}")
        return False, "", str(e)


def parse_simulation_output(output):
    """
    Parse the simulation output to extract all relevant metrics.

    Args:
        output: String containing the simulation output

    Returns:
        Dictionary with all parsed metrics (None for missing values)
    """
    metrics = {
        'injected_jobs': None,
        'completed_jobs': None,
        'deadlines_met': None,
        'deadlines_missed': None,
        'data_colocated': None,
        'data_forwarded': None,
        'data_from_memory': None,
        'num_forwards': None,
        'num_RELIEF_forwards': None,
        'num_colocations': None,
        'ave_latency': None,
        'execution_time': None,
        'cumulative_execution_time': None,
        'time_moving_memory': None,
        'total_energy': None,
        'edp': None,
        'average_concurrent_jobs': None,
    }

    # Regular expressions to extract metrics
    patterns = {
        'injected_jobs': r'Number of injected jobs:\s+(\d+)',
        'completed_jobs': r'Number of completed jobs:\s+(\d+)',
        'deadlines_met': r'Number of deadlines met:\s+(\d+)',
        'deadlines_missed': r'Number of deadlines missed:\s+(\d+)',
        'num_forwards': r'Number of forwards:\s+(\d+)',
        'num_RELIEF_forwards': r'Number of RELIEF forwards:\s+(\d+)',
        'num_colocations': r'Number of colocations:\s+(\d+)',
        'ave_latency': r'Ave latency:\s+([\d.]+)',
        'execution_time': r'Execution time\(us\)\s+:\s+([\d.]+)',
        'cumulative_execution_time': r'Cumulative Execution time\(us\)\s+:\s+([\d.]+)',
        'time_moving_memory': r'Time Spent Moving Memory\(us\)\s+:\s+([\d.]+)',
        'total_energy': r'Total energy consumption\(J\)\s+:\s+([\d.]+)',
        'edp': r'EDP\s+:\s+([\d.]+)',
        'average_concurrent_jobs': r'Average concurrent jobs\s+:\s+([\d.]+)',
    }

    # Special pattern for the data colocated/forwarded/memory line
    data_pattern = r'Data Colocated/Forwarded/Pulled From Memory:\s+(\d+)\s+/\s+(\d+)\s+/\s+(\d+)'
    data_match = re.search(data_pattern, output)
    if data_match:
        metrics['data_colocated'] = int(data_match.group(1))
        metrics['data_forwarded'] = int(data_match.group(2))
        metrics['data_from_memory'] = int(data_match.group(3))

    # Extract all other metrics
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            # Convert to appropriate type
            if '.' in value:
                metrics[key] = float(value)
            else:
                metrics[key] = int(value)

    return metrics


def save_results_to_csv(results, output_file):
    """
    Save all experiment results to a CSV file.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    if not results:
        print("[Save] No results to save")
        return

    # Get all unique keys from all results
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())

    # Sort fieldnames for consistent output, with metadata columns first
    priority_fields = ['experiment_num', 'timestamp', 'soc_config', 'scheduler',
                      'workload_name', 'job_list', 'job_probabilities', 'memory_bandwidth',
                      'arbitration_type']

    # Separate priority fields and other fields
    other_fields = sorted(list(fieldnames - set(priority_fields)))

    # Priority fields that exist in results, then other fields
    final_fieldnames = [f for f in priority_fields if f in fieldnames] + other_fields

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[Save] Results saved to {output_file} ({len(results)} experiments)")


# ========== MAIN EXPERIMENT LOOP ==========

def run_experiments():
    """
    Main function to run all experiment combinations.
    """
    print("="*70)
    print("DS3 (DASH-Sim) Automated Experiment Runner - Arbitration Type Sweep")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Simulation mode: {SIMULATION_MODE}")
    print(f"Fixed bandwidth: {bandwidth} bytes/μs")
    print(f"SoC configs: {len(SOC_CONFIGS)}")
    print(f"Schedulers: {len(SCHEDULERS)}")
    print(f"Workloads: {len(WORKLOAD_CONFIGS)}")
    print(f"Arbitration types: {len(ARBITRATION_TYPES)}")
    experiments_per_arbitration = len(SOC_CONFIGS) * len(SCHEDULERS) * len(WORKLOAD_CONFIGS)
    total_experiments = experiments_per_arbitration * len(ARBITRATION_TYPES)
    print(f"Experiments per arbitration type: {experiments_per_arbitration}")
    print(f"Total experiments: {total_experiments}")
    print("="*70)
    print()

    # Step 1: Generate all SoC configurations
    print("\n" + "="*70)
    print("STEP 1: Generating SoC Configurations")
    print("="*70)
    failed_soc_generation = []

    for soc_filename, acc_specs in SOC_CONFIGS:
        success = generate_soc_config(soc_filename, acc_specs, bandwidth)
        if not success:
            failed_soc_generation.append(soc_filename)

    if failed_soc_generation:
        print(f"\n[Warning] Failed to generate {len(failed_soc_generation)} SoC configs:")
        for soc in failed_soc_generation:
            print(f"  - {soc}")
        print("\nExperiments using these SoCs will fail.")

    print()

    # Step 2: Backup original config file
    print("="*70)
    print("STEP 2: Backing up configuration")
    print("="*70)
    backup_file = backup_config_file()
    print()

    # Step 3: Run all experiments
    print("="*70)
    print("STEP 3: Running Experiments")
    print("="*70)

    failed_experiments = []
    experiment_num = 0

    # OUTER LOOP: Iterate through each arbitration type (4 separate sweeps)
    for arbitration_type in ARBITRATION_TYPES:
        # Set OUTPUT_CSV for this arbitration type's sweep
        OUTPUT_CSV = os.path.join(RESULTS_DIR, f'experiment_results_RELIEF_NoCrit_{arbitration_type}_5280.csv')

        # Results list for THIS arbitration type only
        results = []

        print(f"\n{'='*70}")
        print(f"ARBITRATION TYPE SWEEP: {arbitration_type}")
        print(f"Output file: {OUTPUT_CSV}")
        print(f"{'='*70}")

        # INNER LOOPS: Workload × SoC × Scheduler (standard sweep within each arbitration type)
        for soc_filename, _ in SOC_CONFIGS:
            for scheduler in SCHEDULERS:
                for workload_config in WORKLOAD_CONFIGS:
                    experiment_num += 1

                    print(f"\n{'='*70}")
                    print(f"Experiment {experiment_num}/{total_experiments}")
                    print(f"Arbitration Type: {arbitration_type}")
                    print(f"{'='*70}")

                    # Modify configuration
                    try:
                        modify_config(soc_filename, scheduler, workload_config, arbitration_type)
                    except Exception as e:
                        print(f"[Error] Failed to modify config: {e}")
                        failed_experiments.append({
                            'experiment_num': experiment_num,
                            'soc_config': soc_filename,
                            'scheduler': scheduler,
                            'workload': workload_config['name'],
                            'arbitration_type': arbitration_type,
                            'error': f'Config modification failed: {e}'
                        })
                        continue

                    # Run simulation
                    success, output, error = run_simulation()

                    if not success:
                        print(f"[Error] Simulation failed")
                        failed_experiments.append({
                            'experiment_num': experiment_num,
                            'soc_config': soc_filename,
                            'scheduler': scheduler,
                            'workload': workload_config['name'],
                            'arbitration_type': arbitration_type,
                            'error': error if error else 'Unknown error'
                        })

                    # Parse results
                    metrics = parse_simulation_output(output)

                    # Add experiment metadata to results
                    result = {
                        'experiment_num': experiment_num,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'soc_config': soc_filename,
                        'memory_bandwidth': bandwidth,
                        'scheduler': scheduler,
                        'workload_name': workload_config['name'],
                        'job_list': workload_config['job_list'],
                        'job_probabilities': workload_config['job_probabilities'],
                        'arbitration_type': arbitration_type,
                    }

                    # Add all metrics
                    result.update(metrics)

                    results.append(result)

                    # Print key metrics
                    print(f"\n[Results] Experiment {experiment_num} completed:")
                    print(f"  Completed Jobs: {metrics.get('completed_jobs', 'N/A')}")
                    print(f"  Deadlines Met: {metrics.get('deadlines_met', 'N/A')}")
                    print(f"  Deadlines Missed: {metrics.get('deadlines_missed', 'N/A')}")
                    print(f"  Execution Time: {metrics.get('execution_time', 'N/A')} us")
                    print(f"  Energy: {metrics.get('total_energy', 'N/A')} J")
                    print(f"  EDP: {metrics.get('edp', 'N/A')}")

                    # Save intermediate results to THIS arbitration type's CSV
                    save_results_to_csv(results, OUTPUT_CSV)

        # After completing sweep for this arbitration type, save final CSV
        print(f"\n{'='*70}")
        print(f"[Complete] Finished sweep for arbitration_type={arbitration_type}")
        print(f"[Complete] Results saved to: {OUTPUT_CSV}")
        print(f"[Complete] Experiments for this type: {len(results)}")
        print(f"{'='*70}")

    # Step 4: Final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total experiments planned: {total_experiments}")
    print(f"Successful: {total_experiments - len(failed_experiments)}")
    print(f"Failed: {len(failed_experiments)}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nGenerated {len(ARBITRATION_TYPES)} CSV files:")
    for arb_type in ARBITRATION_TYPES:
        csv_path = os.path.join(RESULTS_DIR, f'experiment_results_RELIEF_NoCrit_{arb_type}_5280.csv')
        print(f"  - {csv_path}")

    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}):")
        for fail in failed_experiments:
            print(f"  - Experiment {fail['experiment_num']}: SoC={fail['soc_config']}, "
                  f"Scheduler={fail['scheduler']}, Workload={fail['workload']}, "
                  f"Arbitration={fail['arbitration_type']}")
            print(f"    Error: {fail['error']}")

    print("="*70)

    # Restore backup if needed
    if backup_file and os.path.exists(backup_file):
        print(f"\nTo restore original config: cp {backup_file} {CONFIG_FILE}")


if __name__ == '__main__':
    try:
        run_experiments()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Experiment run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
