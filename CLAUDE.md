# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DS3 (DASH-Sim) is a discrete-event simulator for heterogeneous System-on-Chip (SoC) architectures, focused on evaluating task scheduling algorithms and dynamic power management (DTPM) policies. The simulator models resource-constrained embedded systems with multiple processing elements (PEs) executing directed acyclic graph (DAG) workloads.

This codebase implements the RELIEF scheduling algorithm and supports research on Dynamic Adaptive Scheduling (DAS) frameworks for energy-efficient heterogeneous SoCs.

## Running the Simulator

### Basic Simulation
```bash
python DASH_Sim_v0.py
```

### Case Study Execution
```bash
python run_Scheduling_Case_Study.py
```

### Environment Setup
```bash
pip install -r requirements.txt
```

Note: This codebase requires Python 3.6 and uses legacy package versions (TensorFlow 2.1.0, NumPy 1.17.3, etc.).

## Configuration

All simulation parameters are defined in `config_file.ini`:

- **Resource Configuration**: `resource_file` points to SoC configurations in `config_SoC/` directory
- **Application Configuration**: `job_file` points to task graph definitions in `config_Jobs/` directory
- **Scheduler Selection**: `scheduler` variable selects the scheduling algorithm (e.g., `RELIEF_BASE`, `MET`, `ETF`)
- **Workload Generation**: Control via `job_probabilities`, `job_list`, or `inject_fixed_num_jobs`
- **Simulation Mode**: `validation` (fixed job count) or `performance` (timed execution)
- **Communication Mode**: `PE_to_PE` or `shared_memory`

## Architecture

### Core Simulation Flow

1. **DASH_Sim_v0.py**: Entry point that orchestrates simulation setup
   - Parses SoC configuration via `DASH_SoC_parser.py`
   - Parses job/task graphs via `job_parser.py`
   - Creates SimPy environment and launches simulation

2. **DASH_Sim_core.py**: Contains `SimulationManager` class
   - Drives discrete-event simulation loop
   - Manages task queues: `outstanding`, `ready`, `running`, `completed`
   - Handles task state transitions and dependencies
   - Coordinates between scheduler, PEs, and job generator

3. **job_generator.py**: Contains `JobGenerator` class
   - Dynamically injects jobs into the system based on arrival patterns
   - Manages inter-arrival times using exponential distribution
   - Supports both probabilistic and snippet-based workload generation

4. **scheduler.py**: Contains `Scheduler` class with multiple scheduling algorithms
   - `MET`: Minimum Execution Time
   - `ETF`: Earliest Task First
   - `RELIEF_BASE`: Resource-Limited Execution with Intelligent Forwarding (current research focus)
   - Each scheduler assigns tasks to PEs based on different heuristics

5. **processing_element.py**: Contains `PE` class
   - Models individual processing elements (CPUs, accelerators, memory)
   - Tracks utilization, power consumption, thermal state
   - Maintains per-PE task queues and execution state
   - Supports DVFS (Dynamic Voltage and Frequency Scaling)

### Key Data Structures

- **ResourceManager** (`common.py`): Manages all PE resources and their power/performance characteristics
- **ApplicationManager** (`common.py`): Manages job definitions and task graphs
- **Task Queues** (`common.py`): Global task state tracking
  - `common.outstanding`: Tasks waiting for predecessors
  - `common.ready`: Tasks ready for scheduling
  - `common.running`: Tasks currently executing
  - `common.completed`: Finished tasks
- **DAG Representation**: Uses NetworkX (`common.current_dag`) to track task dependencies

### RELIEF Implementation

RELIEF (Resource-Limited Execution with Intelligent Forwarding) is the current research focus:

- **RELIEF_Sim_helpers.py**: Contains RELIEF-specific helper functions
- Key concept: Task forwarding between PEs to optimize data movement and reduce communication overhead
- Implements double buffering and intelligent memory management
- Recent commits indicate ongoing development and testing of RELIEF features

### Power Management

- **DTPM.py**: Dynamic Thermal and Power Management module
- **DTPM_policies.py**: Implements DVFS policies (performance, powersave, ondemand, etc.)
- **DTPM_power_models.py**: Power and thermal modeling functions
- Supports thermal throttling with configurable trip points
- Leakage power modeling based on Odroid XU3 board parameters

## Configuration File Formats

### SoC Configuration (`config_SoC/SoC.*.txt`)
```
add_new_resource resource_type <TYPE> resource_name <NAME> resource_ID <ID> capacity <N> num_supported_functionalities <M> DVFS_mode <MODE>
opp <frequency_MHz> <voltage_mV>
trip_freq <trip1> <trip2> <trip3>
power_profile <freq> <power_values...>
PG_profile <freq> <pg_values...>
<task_name> <execution_time>
```

### Job Configuration (`config_Jobs/job_*.txt`)
```
add_new_tasks <num_tasks>
<task_name> <task_id> <predecessor_ids...>
```

## Testing

The project uses pytest for testing:
```bash
pytest
```

## Output and Tracing

Results are written to `results.csv`. Optional trace files include:
- `trace_tasks.csv`: Per-task execution traces
- `trace_system.csv`: System-level events
- `trace_frequency.csv`: DVFS frequency changes
- `trace_PEs.csv`: Per-PE utilization/power
- `trace_temperature.csv`: Thermal traces

Control tracing via `[TRACE]` section in `config_file.ini`.

## Important Notes

- The codebase uses Python 3.6 with legacy package versions
- SimPy 3.0.11 provides the discrete-event simulation framework
- Constraint Programming (CP) models are available in `CP_models.py` using IBM DOCPLEX
- HTML documentation is auto-generated in the `html/` directory
- The project is under active development, particularly the RELIEF scheduler implementation
