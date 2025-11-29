# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DS3 (DASH-Sim) is a discrete-event simulator for heterogeneous System-on-Chip (SoC) architectures, focused on evaluating task scheduling algorithms and dynamic power management (DTPM) policies. The simulator models resource-constrained embedded systems with multiple processing elements (PEs) executing directed acyclic graph (DAG) workloads.

This codebase implements the RELIEF scheduling algorithm and supports research on Dynamic Adaptive Scheduling (DAS) frameworks for energy-efficient heterogeneous SoCs. Recent development focuses on intelligent task forwarding, scratchpad memory management, and deadline-aware scheduling.

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

## Communication Modes

DS3 supports three communication modes for data transfer between processing elements:

### Mode Selection
Set in `config_file.ini` under `[COMMUNICATION MODE]`:
```ini
communication_mode = forwarding  # Options: PE_to_PE, shared_memory, forwarding
```

### Available Modes

1. **PE_to_PE (Legacy)**
   - All data transfers use direct PE-to-PE communication timings
   - Uses point-to-point communication latencies from SoC configuration
   - No memory/scratchpad modeling
   - Best for: Systems with dedicated interconnects between PEs

2. **shared_memory (Legacy)**
   - All data transfers go through shared memory
   - Uses memory read/write latencies
   - No PE-to-PE direct communication
   - Best for: Systems with shared memory architectures

3. **forwarding (NEW - Recommended)**
   - **Dynamic communication mode** that optimizes data movement
   - Checks if data is available in source PE's scratchpad
   - If data available locally: Uses PE-to-PE communication timing
   - If data not available: Uses memory communication timing
   - Supports task colocation optimization
   - Tracks memory writeback operations
   - Best for: Realistic heterogeneous SoC modeling with local buffers

### How Forwarding Mode Works

When a task needs data from a predecessor:
1. Simulator checks if predecessor PE has data in its scratchpad (`has_data_in_scratchpad()`)
2. If YES: Data forwarded directly (PE-to-PE timing), scratchpad allocated on consumer PE
3. If NO: Data fetched from shared memory (memory timing)
4. Communication timing decision stored in task metadata (`task.comm_timing_mode`)

Implementation: `decide_comm_timing()` in common.py:524-558

## Memory Management

### Scratchpad Buffers

Each PE has a local scratchpad (buffer) for storing intermediate task data:

**PE Scratchpad Attributes** (processing_element.py:55-59):
- `scratchpad`: Dictionary mapping data_id → metadata (task_id, size, timestamp, dependencies)
- `scratchpad_capacity`: Total buffer size in bytes (currently hardcoded per PE type)
- `scratchpad_used`: Current bytes allocated
- `forwarding_enabled`: Boolean flag enabling scratchpad operations

**Configuration**:
- Currently hardcoded in PE initialization
- Future: Will be configurable per PE type in SoC configuration files

**Key Operations**:
- `has_data_in_scratchpad(data_id)`: Check if data is cached locally
- `allocate_scratchpad(data_id, size, task_id)`: Allocate space, evict if needed
- `free_scratchpad(data_id)`: Free space, conditionally writeback to memory

### LRU Eviction Policy

When scratchpad is full and new data must be allocated:
1. Find Least Recently Used (LRU) entry based on timestamps
2. Evict LRU data from scratchpad
3. Writeback to memory if needed (see Conditional Writeback)
4. Allocate space for new data

Implementation: `allocate_scratchpad()` in processing_element.py:288-342

### Task Colocation Optimization

When predecessor and successor tasks execute on the same PE:
- **No storage overhead**: Self-to-self bandwidth is high, no communication latency
- Data remains in scratchpad without reallocation
- Scratchpad space allocated once, shared between tasks
- Reduces memory traffic and communication overhead

Detected in: `calculate_memory_movement_latency()` in common.py:484-489

### Conditional Memory Writeback

Data is written back to memory only when necessary:
- Writeback occurs if successor tasks may execute on different PEs
- Checks task dependency graph and predecessor locations
- Avoids unnecessary memory traffic for colocated tasks

### Memory Writeback Tracking

Global dictionary tracks ongoing memory writeback operations:
- **Data Structure**: `common.memory_writeback = {}`
- **Purpose**: Prevents read-after-write conflicts
- **Usage**: When reading from memory, communication latency includes pending writeback time
- **Location**: common.py:414, DASH_Sim_core.py:307, 332-339

This ensures accurate modeling of memory contention and data movement overhead.

## Deadline Management

The simulator tracks both DAG-level and node-level deadlines for real-time workload analysis.

### Deadline Types

1. **DAG Deadline** (`Applications.deadline`, `Task.jobDeadline`)
   - Overall deadline for completing the entire job/application
   - Increments per repetition: `(repetition_count + 1) * BASE_DEADLINE`
   - Checked when tail tasks complete
   - Used for deadline hit/miss counting

2. **Node Deadline** (Task-level, from workload config)
   - Fine-grained per-task timing constraints
   - Defined in job configuration files
   - Used for detailed timing analysis

### Deadline Tracking

**Performance Statistics** (common.py:232-233):
- `common.results.deadlines_met`: Counter for jobs completed before deadline
- `common.results.deadlines_missed`: Counter for jobs that violated deadline

**Deadline Checking** (processing_element.py:213-217):
- Performed when tail tasks (final tasks in DAG) complete
- Compares task completion time against `task.jobDeadline`
- Increments appropriate counter

**Reported Metrics**:
```python
print('[I] Number of deadlines met: %d' %(common.results.deadlines_met))
print('[I] Number of deadlines missed: %d' %(common.results.deadlines_missed))
```

### Deadline Configuration

Deadlines are specified in job configuration files and parsed during application setup. See Workload Benchmarks section for example deadline values.

## Configuration

All simulation parameters are defined in `config_file.ini`:

### Core Configuration Parameters

- **Resource Configuration**: `resource_file` points to SoC configurations in `config_SoC/` directory
- **Application Configuration**: `job_file` points to task graph definitions in `config_Jobs/` directory
- **Scheduler Selection**: `scheduler` variable selects the scheduling algorithm (e.g., `RELIEF_BASE`, `MET`, `ETF`)
- **Workload Generation**: Control via `job_probabilities`, `job_list`, or `inject_fixed_num_jobs`
- **Simulation Mode**: `validation` (fixed job count) or `performance` (timed execution)
- **Communication Mode**: `PE_to_PE`, `shared_memory`, or `forwarding` (NEW)

### Communication Mode Configuration

```ini
[COMMUNICATION MODE]
# Communication mode: PE_to_PE, shared_memory, or forwarding
# - PE_to_PE: Always use direct PE-to-PE communication timings
# - shared_memory: Always use memory communication timings
# - forwarding: Dynamic mode - use PE_to_PE when data in scratchpad, else memory
communication_mode = forwarding
```

### Tracing Configuration

Control simulation tracing via `[TRACE]` section:
- Task-level traces: `trace_tasks.csv`
- System events: `trace_system.csv`
- DVFS changes: `trace_frequency.csv`
- PE utilization/power: `trace_PEs.csv`
- Thermal traces: `trace_temperature.csv`

## Architecture

### Core Simulation Flow

1. **DASH_Sim_v0.py**: Entry point that orchestrates simulation setup
   - Parses SoC configuration via `DASH_SoC_parser.py`
   - Parses job/task graphs via `job_parser.py`
   - Initializes scratchpad buffers in forwarding mode (lines 163-180)
   - Creates SimPy environment and launches simulation

2. **DASH_Sim_core.py**: Contains `SimulationManager` class
   - Drives discrete-event simulation loop
   - Manages task queues: `outstanding`, `ready`, `running`, `completed`
   - Handles task state transitions and dependencies
   - Coordinates between scheduler, PEs, and job generator
   - Updates memory writeback tracking (lines 307, 332-339)

3. **job_generator.py**: Contains `JobGenerator` class
   - Dynamically injects jobs into the system based on arrival patterns
   - Manages inter-arrival times using exponential distribution
   - Supports both probabilistic and snippet-based workload generation

4. **scheduler.py**: Contains `Scheduler` class with multiple scheduling algorithms
   - See Scheduling Algorithms section below

5. **processing_element.py**: Contains `PE` class
   - Models individual processing elements (CPUs, accelerators, memory)
   - Tracks utilization, power consumption, thermal state
   - Maintains per-PE task queues and execution state
   - Supports DVFS (Dynamic Voltage and Frequency Scaling)
   - **NEW**: Scratchpad buffer management (lines 55-59, 288-375)
   - **NEW**: Deadline checking for tail tasks (lines 213-217)

### Key Data Structures

**Core Managers** (common.py):
- `ResourceManager`: Manages all PE resources and their power/performance characteristics
- `ApplicationManager`: Manages job definitions and task graphs
- `PETypeManager` (NEW, lines 290-335): Efficient PE type lookup
  - `by_type`: Dict mapping PE type → list of PE IDs
  - `by_id`: Dict mapping PE ID → PE type
  - Methods: `register_PE()`, `get_PEs_of_type()`, `get_type_of_PE()`, `get_all_types()`

**Task Queues** (common.py):
- `common.outstanding`: Tasks waiting for predecessors
- `common.ready`: Tasks ready for scheduling
- `common.running`: Tasks currently executing
- `common.completed`: Finished tasks

**DAG Representation**:
- Uses NetworkX (`common.current_dag`) to track task dependencies

**Global State** (common.py):
- `memory_writeback = {}` (NEW, line 414): Tracks data writeback timestamps
- `executable = {}` (NEW, line 415): Per-PE executable task queues

### Updated Classes

**Tasks Class** (common.py:337-377):
- `jobDeadline`: DAG-level deadline for the job
- **NEW Forwarding Metadata**:
  - `comm_timing_mode`: 'PE_to_PE' or 'memory' decision
  - `data_locations`: Maps predecessor_task_ID → PE_ID with data
  - `forwarded_from_PE`: PE_ID if data was forwarded
  - `data_sizes`: Maps predecessor_task_ID → data size

**Applications Class** (common.py:387-397):
- `deadline` (NEW): Overall deadline for the application/job

**PerfStatics Class** (common.py:218-236):
- `deadlines_met` (NEW): Counter for jobs completed before deadline
- `deadlines_missed` (NEW): Counter for jobs that missed deadline

### Key Functions

**calculate_memory_movement_latency()** (common.py:418-522):
- Calculates data movement latency for task execution
- Handles both PE-to-PE and memory communication modes
- Checks memory writeback queue for data availability
- Allocates scratchpad for colocated tasks
- Returns maximum wait time among all predecessor data dependencies

**decide_comm_timing()** (common.py:524-558):
- Decides between PE_to_PE vs memory communication for each data transfer
- Legacy modes (`PE_to_PE`, `shared_memory`): Return fixed decision
- Forwarding mode: Checks scratchpad for data availability dynamically
- Returns `'PE_to_PE'` or `'memory'` string

## Scheduling Algorithms

The simulator supports multiple scheduling algorithms selectable via `config_file.ini`:

### Available Schedulers

1. **MET (Minimum Execution Time)**
   - Assigns each task to the PE with minimum execution time
   - Greedy heuristic, no lookahead
   - Good baseline for comparison

2. **ETF (Earliest Task First)**
   - Priority-based scheduling considering task criticality
   - Accounts for data dependencies
   - Static priority assignment

3. **RELIEF (Resource-Limited Execution with Intelligent Forwarding)**
   - **Current research focus**
   - Intelligent task forwarding to optimize data movement
   - Supports task colocation for communication reduction
   - Lookahead capability for better scheduling decisions
   - Integration with scratchpad memory management

### RELIEF Implementation

**Core Files**:
- `scheduler.py`: RELIEF_BASE algorithm implementation
- `RELIEF_Sim_helpers.py`: RELIEF-specific helper functions

**Key Features**:
- Task forwarding between PEs to reduce communication overhead
- Double buffering support
- Intelligent memory management with scratchpad integration
- **Recent Improvements**:
  - Fixed lookahead capability for accurate future state prediction
  - Support for task colocation optimization
  - Conditional data movement based on predecessor locations
  - Simplified implementation with improved code clarity

**How RELIEF Works**:
1. Evaluates candidate PEs for each ready task
2. Considers data locality and scratchpad availability
3. Estimates communication overhead with forwarding decisions
4. Makes colocation decisions for dependent tasks
5. Optimizes for both performance and energy efficiency

## Workload Benchmarks

DS3 includes embedded vision and machine learning benchmark workloads for realistic SoC evaluation.

### Available Workloads

Located in `config_Jobs/` directory (currently in `.h` format, will be converted to native configuration):

1. **canny.h** - Canny Edge Detection
   - 12 nodes per DAG iteration
   - Deadline: 16667 μs (~60 FPS)
   - Tasks: ISP, grayscale conversion, noise reduction, gradient calculation, non-maximum suppression, edge tracking
   - Typical use: Real-time vision processing

2. **deblur.h** - Iterative Image Deblurring
   - Richardson-Lucy deconvolution algorithm (5 iterations)
   - Deadline: 16667 μs
   - Tasks: ISP, grayscale, convolution with PSF, division, element-wise operations
   - Typical use: Image enhancement pipelines

3. **gru.h** - Gated Recurrent Unit (RNN)
   - 8-cell sequence, 15 nodes per cell (120 total nodes)
   - Deadline: 7000 μs
   - Tasks: Update gate, reset gate, candidate state, cell state operations
   - Typical use: Sequence modeling, time-series prediction

4. **harris.h** - Harris Corner Detection
   - 18 nodes per DAG iteration
   - Deadline: 16667 μs
   - Tasks: ISP, grayscale, spatial derivatives (Sobel), structure tensor, Harris response, non-maximum suppression
   - Typical use: Feature detection for visual odometry

5. **lstm.h** - Long Short-Term Memory (RNN)
   - 8-cell sequence, 18 nodes per cell (144 total nodes)
   - Deadline: 7000 μs
   - Tasks: Forget gate, input gate, output gate, cell state update, hidden state update
   - Typical use: Advanced sequence modeling, NLP tasks

### Workload Characteristics

- **DAG Structure**: Complex dependencies modeling realistic application dataflow
- **Deadline Constraints**: Real-time timing requirements for embedded systems
- **Repetition Support**: Multi-iteration execution with incremental deadlines
- **Accelerator Targets**: Designed for heterogeneous SoCs with specialized accelerators:
  - ISP (Image Signal Processor)
  - Grayscale conversion
  - Convolution engines
  - Edge detection (Canny, Harris)
  - Matrix operations
  - Element-wise operations

### Future Format Conversion

These workloads are currently in C header file format (`.h`) and will be converted to DS3's native job configuration format with proper accelerator mappings.

## Power Management

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

**Future Addition**: Per-PE scratchpad capacity configuration will be added to SoC config files.

### Job Configuration (`config_Jobs/job_*.txt`)

```
add_new_tasks <num_tasks>
<task_name> <task_id> <predecessor_ids...>
```

**Note**: New `.h` format workloads (canny, deblur, gru, harris, lstm) will be converted to this format.

## Testing

The project uses pytest for testing:
```bash
pytest
```

## Output and Tracing

### Results Output

Results are written to `results.csv` and terminal output includes:
- Total execution time
- Energy consumption
- Task completion statistics
- **NEW**: Deadline metrics
  - Number of deadlines met
  - Number of deadlines missed
- PE utilization statistics

### Trace Files

Optional trace files for detailed analysis:
- `trace_tasks.csv`: Per-task execution traces with timing
- `trace_system.csv`: System-level events
- `trace_frequency.csv`: DVFS frequency changes
- `trace_PEs.csv`: Per-PE utilization and power consumption
- `trace_temperature.csv`: Thermal traces

Control tracing via `[TRACE]` section in `config_file.ini`.

## Pending Job Configuration Updates

The following job configuration files have been updated with:
- Volume conversion (bits → bytes) to match bandwidth units
- Placeholder sub-deadline (sd) values set to 1

**Files updated:**
- `job_WIFI_5RXM.txt` (includes corrected deadlines from DS3_TIMING_REFERENCE.md)
- `job_WIFI_5TXM.txt`
- `job_LAG.txt`
- `job_SCR.txt`
- `job_SCT.txt`

**Future work:** User will provide correct sub-deadline (sd) values for tasks in:
- `job_WIFI_5TXM.txt`
- `job_LAG.txt`
- `job_SCR.txt`
- `job_SCT.txt`

These sd values will be computed based on the specific scheduling algorithm requirements and critical path analysis for each workload. The current placeholder value of 1 allows the simulator to run but does not reflect realistic sub-deadline distributions.

## Important Notes

- The codebase uses Python 3.6 with legacy package versions
- SimPy 3.0.11 provides the discrete-event simulation framework
- Constraint Programming (CP) models are available in `CP_models.py` using IBM DOCPLEX
- HTML documentation is auto-generated in the `html/` directory
- The project is under active development, particularly:
  - RELIEF scheduler enhancements
  - Forwarding communication mode optimization
  - Scratchpad management strategies
  - Deadline-aware scheduling policies
