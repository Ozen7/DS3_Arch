#!/usr/bin/env python3
"""
DS3 Full Experiment Pipeline
=============================

Automates the complete experimental workflow:
1. Runs DS3 experiments at three bandwidth levels (5280, 8000, 16000 MHz) OR
   Runs arbitration type sweep experiments (min, min_coloc, random, exectime) for RELIEF or COMM
2. Generates memory saturation comparison graphs
3. Generates resource scaling comparison graphs

Usage:
    python3 run_full_experiment_pipeline.py [--sweep-type TYPE]

Options:
    --sweep-type TYPE     Type of sweep: 'bandwidth' (default), 'arbitration', 'comm-arbitration', or 'both'
    --skip-experiments    Skip running experiments, only generate graphs
    --bandwidth <bw>      Run only specific bandwidth (5280, 8000, or 16000) for bandwidth sweeps
    --graphs-only         Only generate graphs from existing data
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import argparse


# ============================================================================
# Configuration
# ============================================================================

BANDWIDTHS = [5280, 8000, 16000]
EXPERIMENT_SCRIPT = "run_experiment_sweep.py"
ARBITRATION_SWEEP_SCRIPT = "run_arbitration_sweep.py"
COMM_ARBITRATION_SWEEP_SCRIPT = "run_comm_arbitration_sweep.py"
GRAPHING_DIR = "../graphing"
MEMORY_SAT_SCRIPT = "script_memory_saturation.py"
RESOURCE_SCALE_SCRIPT = "script_resource_scaling.py"


# ============================================================================
# Helper Functions
# ============================================================================

def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70 + "\n")


def print_section(message):
    """Print a formatted section header."""
    print("\n" + "-" * 70)
    print(message)
    print("-" * 70 + "\n")


def run_command(cmd, description, timeout=600):
    """
    Run a shell command and handle output.

    Args:
        cmd: Command to run (list of strings)
        description: Description of what the command does
        timeout: Timeout in seconds (default: 600 = 10 minutes)

    Returns:
        True if successful, False otherwise
    """
    print(f"[RUN] {description}")
    print(f"[CMD] {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"[SUCCESS] Completed in {elapsed_time:.1f} seconds")
            # Print last few lines of output for context
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("[OUTPUT] Last 5 lines:")
                    for line in lines[-5:]:
                        print(f"  {line}")
            return True
        else:
            print(f"[ERROR] Command failed with return code {result.returncode}")
            if result.stderr:
                print("[STDERR]")
                print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return False


def verify_files_exist():
    """Verify required files exist."""
    print("[CHECK] Verifying required files exist...")

    required_files = [
        EXPERIMENT_SCRIPT,
        os.path.join(GRAPHING_DIR, MEMORY_SAT_SCRIPT),
        os.path.join(GRAPHING_DIR, RESOURCE_SCALE_SCRIPT)
    ]

    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
            print(f"[MISSING] {file_path}")
        else:
            print(f"[OK] {file_path}")

    if missing:
        print(f"\n[ERROR] {len(missing)} required file(s) missing!")
        return False

    print("[OK] All required files found\n")
    return True


def verify_results_exist(bandwidths):
    """Verify experiment result files exist."""
    print("[CHECK] Verifying experiment results exist...")

    missing = []
    for bw in bandwidths:
        filename = f"../results_final/experiment_results_RELIEF_NoCrit_MinList_{bw}.csv"
        if not os.path.exists(filename):
            missing.append(filename)
            print(f"[MISSING] {filename}")
        else:
            # Check file size
            size = os.path.getsize(filename)
            print(f"[OK] {filename} ({size} bytes)")

    if missing:
        print(f"\n[WARNING] {len(missing)} result file(s) missing!")
        print("[INFO] You may need to run experiments first")
        return False

    # Check gem5 baseline
    gem5_file = "../results_final/gem5_comb_1_results.csv"
    if not os.path.exists(gem5_file):
        print(f"[WARNING] {gem5_file} not found")
        print("[INFO] Memory saturation graphs will not include gem5 baseline")
    else:
        print(f"[OK] {gem5_file}")

    print("[OK] All result files verified\n")
    return True


# ============================================================================
# Experiment Running Functions
# ============================================================================

def run_bandwidth_experiment(bandwidth):
    """
    Run experiment for a specific bandwidth.

    Args:
        bandwidth: Bandwidth value (5280, 8000, or 16000)

    Returns:
        True if successful, False otherwise
    """
    print_section(f"Running Experiment: {bandwidth} MHz Bandwidth")

    cmd = ["python3", EXPERIMENT_SCRIPT, "--bandwidth", str(bandwidth)]
    description = f"DS3 experiment sweep at {bandwidth} MHz"

    # Experiments can take a long time - 10 minute timeout
    success = run_command(cmd, description, timeout=600)

    if success:
        # Verify output file was created
        output_file = f"../results_final/experiment_results_RELIEF_NoCrit_MinList_{bandwidth}.csv"
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            lines = 0
            with open(output_file, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"[VERIFY] Output file created: {output_file}")
            print(f"[VERIFY] File size: {size} bytes, {lines} lines")
        else:
            print(f"[ERROR] Expected output file not found: {output_file}")
            return False

    return success


def run_all_experiments(bandwidths):
    """
    Run experiments for all specified bandwidths.

    Args:
        bandwidths: List of bandwidth values to run

    Returns:
        Dictionary of {bandwidth: success_status}
    """
    print_header("STEP 1: Running DS3 Experiments")

    results = {}

    for i, bw in enumerate(bandwidths, 1):
        print(f"\n[PROGRESS] Experiment {i}/{len(bandwidths)}: {bw} MHz")
        success = run_bandwidth_experiment(bw)
        results[bw] = success

        if not success:
            print(f"[ERROR] Experiment at {bw} MHz failed!")
            user_input = input("Continue with remaining experiments? (y/n): ")
            if user_input.lower() != 'y':
                print("[ABORT] User aborted experiment pipeline")
                break

    # Print summary
    print_section("Experiment Summary")
    successful = [bw for bw, success in results.items() if success]
    failed = [bw for bw, success in results.items() if not success]

    print(f"[SUCCESS] {len(successful)}/{len(bandwidths)} experiments completed successfully")
    if successful:
        print(f"  Successful: {', '.join(map(str, successful))} MHz")
    if failed:
        print(f"  Failed: {', '.join(map(str, failed))} MHz")

    return results


def run_arbitration_sweep():
    """
    Run arbitration type sweep experiments.

    Returns:
        True if successful, False otherwise
    """
    print_header("STEP 1: Running Arbitration Type Sweep")
    print_section("Running Arbitration Type Sweep Experiment")

    cmd = ["python3", ARBITRATION_SWEEP_SCRIPT]
    description = "DS3 arbitration type sweep (min, min_coloc, random, exectime) at 5280 MHz"

    # This sweep runs many experiments - allow 60 minute timeout
    success = run_command(cmd, description, timeout=3600)

    if success:
        # Verify output files were created
        arbitration_types = ['min', 'min_coloc', 'random', 'exectime']
        print_section("Verifying Arbitration Sweep Results")

        all_found = True
        for arb_type in arbitration_types:
            output_file = f"../results_final/experiment_results_RELIEF_NoCrit_{arb_type}_5280.csv"
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                lines = 0
                with open(output_file, 'r') as f:
                    lines = sum(1 for _ in f)
                print(f"[VERIFY] {arb_type}: {output_file}")
                print(f"[VERIFY]   File size: {size} bytes, {lines} lines")
            else:
                print(f"[ERROR] Expected output file not found: {output_file}")
                all_found = False

        success = all_found

    return success


def run_comm_arbitration_sweep():
    """
    Run COMM arbitration type sweep experiments.

    Returns:
        True if successful, False otherwise
    """
    print_header("Running COMM Arbitration Type Sweep")
    print_section("Running COMM Arbitration Type Sweep Experiment")

    cmd = ["python3", COMM_ARBITRATION_SWEEP_SCRIPT]
    description = "DS3 COMM arbitration type sweep (min, min_coloc, random, exectime) at 5280 MHz"

    # This sweep runs many experiments - allow 60 minute timeout
    success = run_command(cmd, description, timeout=3600)

    if success:
        # Verify output files were created
        arbitration_types = ['min', 'min_coloc', 'random', 'exectime']
        print_section("Verifying COMM Arbitration Sweep Results")

        all_found = True
        for arb_type in arbitration_types:
            output_file = f"../results_final/experiment_results_COMM_NoCrit_{arb_type}_5280.csv"
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                lines = 0
                with open(output_file, 'r') as f:
                    lines = sum(1 for _ in f)
                print(f"[VERIFY] {arb_type}: {output_file}")
                print(f"[VERIFY]   File size: {size} bytes, {lines} lines")
            else:
                print(f"[ERROR] Expected output file not found: {output_file}")
                all_found = False

        success = all_found

    return success


# ============================================================================
# Graph Generation Functions
# ============================================================================

def generate_memory_saturation_graphs():
    """Generate memory saturation comparison graphs."""
    print_section("Generating Memory Saturation Graphs")

    cmd = ["python3", MEMORY_SAT_SCRIPT]
    description = "Memory saturation comparison graphs (5280 vs 8000 vs 16000 vs gem5)"

    # Change to graphing directory
    original_dir = os.getcwd()
    os.chdir(GRAPHING_DIR)

    try:
        success = run_command(cmd, description, timeout=120)
    finally:
        os.chdir(original_dir)

    if success:
        # Verify output files
        expected_graphs = [
            "memory_saturation_RELIEF.pdf",
            "memory_saturation_LL.pdf",
            "memory_saturation_GEDF_D.pdf",
            "memory_saturation_GEDF_N.pdf",
            "memory_saturation_HetSched.pdf"
        ]

        graphs_dir = os.path.join(GRAPHING_DIR, "graphs")
        created = []
        missing = []

        for graph in expected_graphs:
            graph_path = os.path.join(graphs_dir, graph)
            if os.path.exists(graph_path):
                size = os.path.getsize(graph_path)
                created.append(f"{graph} ({size} bytes)")
            else:
                missing.append(graph)

        if created:
            print(f"[VERIFY] {len(created)} graphs created:")
            for g in created:
                print(f"  - {g}")

        if missing:
            print(f"[WARNING] {len(missing)} expected graphs not found:")
            for g in missing:
                print(f"  - {g}")

    return success


def generate_resource_scaling_graphs():
    """Generate resource scaling comparison graphs."""
    print_section("Generating Resource Scaling Graphs")

    cmd = ["python3", RESOURCE_SCALE_SCRIPT]
    description = "Resource scaling comparison graphs (all1 vs all2 vs all3)"

    # Change to graphing directory
    original_dir = os.getcwd()
    os.chdir(GRAPHING_DIR)

    try:
        success = run_command(cmd, description, timeout=120)
    finally:
        os.chdir(original_dir)

    if success:
        # Verify output files
        expected_graphs = [
            "resource_scaling_RELIEF.pdf",
            "resource_scaling_LL.pdf",
            "resource_scaling_GEDF_D.pdf",
            "resource_scaling_GEDF_N.pdf",
            "resource_scaling_HetSched.pdf"
        ]

        graphs_dir = os.path.join(GRAPHING_DIR, "graphs")
        created = []
        missing = []

        for graph in expected_graphs:
            graph_path = os.path.join(graphs_dir, graph)
            if os.path.exists(graph_path):
                size = os.path.getsize(graph_path)
                created.append(f"{graph} ({size} bytes)")
            else:
                missing.append(graph)

        if created:
            print(f"[VERIFY] {len(created)} graphs created:")
            for g in created:
                print(f"  - {g}")

        if missing:
            print(f"[WARNING] {len(missing)} expected graphs not found:")
            for g in missing:
                print(f"  - {g}")

    return success


def generate_all_graphs():
    """Generate all comparison graphs."""
    print_header("STEP 2: Generating Comparison Graphs")

    mem_success = generate_memory_saturation_graphs()
    res_success = generate_resource_scaling_graphs()

    # Print summary
    print_section("Graph Generation Summary")
    if mem_success and res_success:
        print("[SUCCESS] All graphs generated successfully!")
        print("  - 5 memory saturation graphs")
        print("  - 5 resource scaling graphs")
        print("  - Total: 10 PDF files")
    else:
        if not mem_success:
            print("[ERROR] Memory saturation graph generation failed")
        if not res_success:
            print("[ERROR] Resource scaling graph generation failed")

    return mem_success and res_success


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run DS3 experiments and generate comparison graphs"
    )
    parser.add_argument(
        "--sweep-type",
        type=str,
        choices=['bandwidth', 'arbitration', 'comm-arbitration', 'both'],
        default='bandwidth',
        help="Type of sweep: 'bandwidth' (default), 'arbitration' (RELIEF), 'comm-arbitration' (COMM), or 'both' (RELIEF+COMM arbitration)"
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments, only generate graphs"
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        choices=[5280, 8000, 16000],
        help="Run only specific bandwidth experiment (only for sweep-type='bandwidth')"
    )
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Only generate graphs from existing data"
    )

    args = parser.parse_args()

    # Print pipeline header
    print_header("DS3 Full Experiment Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Sweep type: {args.sweep_type}")

    # Verify required files exist
    if not verify_files_exist():
        print("\n[ABORT] Required files missing. Please check your setup.")
        return 1

    # Determine which experiments to run
    run_bandwidth_sweep = args.sweep_type == 'bandwidth'
    run_arbitration = args.sweep_type == 'arbitration'
    run_comm_arbitration = args.sweep_type == 'comm-arbitration'
    run_both = args.sweep_type == 'both'

    if run_both:
        run_arbitration = True
        run_comm_arbitration = True

    # Determine which bandwidths to run (if running bandwidth sweep)
    if run_bandwidth_sweep:
        if args.bandwidth:
            bandwidths = [args.bandwidth]
            print(f"[MODE] Running single bandwidth: {args.bandwidth} MHz")
        else:
            bandwidths = BANDWIDTHS
            print(f"[MODE] Running all bandwidths: {', '.join(map(str, bandwidths))} MHz")

    # Run experiments (unless skipped)
    if args.skip_experiments or args.graphs_only:
        print("\n[SKIP] Skipping experiment execution (using existing data)")
        if run_bandwidth_sweep and not verify_results_exist(BANDWIDTHS):
            print("\n[ERROR] Cannot skip experiments - result files missing!")
            return 1
    else:
        print("\n[MODE] Running experiments")

        # Run bandwidth sweep
        if run_bandwidth_sweep:
            exp_results = run_all_experiments(bandwidths)

            # Check if all experiments succeeded
            if not all(exp_results.values()):
                print("\n[WARNING] Some bandwidth experiments failed!")
                if run_arbitration:
                    user_input = input("Continue with arbitration sweep? (y/n): ")
                    if user_input.lower() != 'y':
                        print("[ABORT] User aborted pipeline")
                        return 1
                else:
                    user_input = input("Continue with graph generation? (y/n): ")
                    if user_input.lower() != 'y':
                        print("[ABORT] User aborted pipeline")
                        return 1

        # Run arbitration sweep (RELIEF)
        if run_arbitration:
            arb_success = run_arbitration_sweep()

            if not arb_success:
                print("\n[WARNING] RELIEF arbitration sweep failed!")
                if run_comm_arbitration:
                    user_input = input("Continue with COMM arbitration sweep? (y/n): ")
                    if user_input.lower() != 'y':
                        print("[ABORT] User aborted pipeline")
                        return 1
                else:
                    user_input = input("Continue with graph generation? (y/n): ")
                    if user_input.lower() != 'y':
                        print("[ABORT] User aborted pipeline")
                        return 1

        # Run COMM arbitration sweep
        if run_comm_arbitration:
            comm_success = run_comm_arbitration_sweep()

            if not comm_success:
                print("\n[WARNING] COMM arbitration sweep failed!")
                user_input = input("Continue with graph generation? (y/n): ")
                if user_input.lower() != 'y':
                    print("[ABORT] User aborted pipeline")
                    return 1

    # Generate graphs
    graph_success = generate_all_graphs()

    # Final summary
    print_header("Pipeline Complete")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if graph_success:
        print("\n[SUCCESS] All tasks completed successfully!")
        print("\nGenerated files:")
        print("  CSV Results: ../results_final/experiment_results_RELIEF_NoCrit_MinList_*.csv")
        print("  Graphs: ../graphing/graphs/*.pdf")
        return 0
    else:
        print("\n[ERROR] Pipeline completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
