# RELIEF Scheduler Architecture Plan
## Session Documentation: Complete Architectural Design and Decisions

---

## 1. Project Overview

This document captures the complete architectural plan for implementing RELIEF (Resource-Limited Execution with Intelligent Forwarding) scheduler enhancements to the DASH-Sim discrete-event simulator.

**Primary Goals:**
1. Enable RELIEF scheduler to have per-PE executable queues with dynamic reordering capability
2. Create data structures to organize PEs by type for efficient scheduling decisions
3. Support dynamic communication mode switching between memory and peer-to-peer based on forwarding feasibility
4. Maintain backward compatibility with existing schedulers (MET, EFT, DRL, CP)

---

## 2. Core Architecture Decisions

### 2.1 Executable Queue Architecture

**Decision:** Convert `common.executable` from a list to a dictionary structure

**Rationale:**
- Original list-based structure didn't support per-PE queues
- Schedulers need ability to reorder tasks dynamically
- State must persist between scheduler invocations and simulation loop
- Each PE needs independent queue for parallel execution tracking

**Structure:**
```python
common.executable = {
    PE_ID: [task1, task2, task3, ...],  # Queue for each PE
    ...
}
```

**Key Properties:**
- O(1) access to any PE's queue
- Schedulers can directly manipulate queue contents
- Simulation loop processes each PE's queue independently
- State preserved across scheduler/simulator boundary

### 2.2 Single Source of Truth Principle

**Decision:** Use `common.executable` as the ONLY executable queue structure

**Rejected Approach:** Creating separate `scheduler.pe_queues` structure

**Rationale:**
- Separate queues caused state loss between scheduler and simulator
- Tasks removed from `common.executable` during simulation weren't reflected in scheduler state
- Double processing risk when tasks exist in multiple queues
- Violates single source of truth principle

**Impact:**
- Schedulers manage `common.executable` directly
- Legacy `update_execution_queue()` still supported for compatibility
- RELIEF and advanced schedulers can bypass `update_execution_queue()` entirely

### 2.3 PE Type Organization

**Decision:** Create `PETypeManager` class for O(1) type-based PE lookup

**Structure:**
```python
class PETypeManager:
    def __init__(self):
        self.by_type = {}  # {'CPU': [PE_IDs], 'ACC_JPEG': [PE_IDs], ...}
        self.by_id = {}    # {PE_ID: 'PE_type'}

    def register_PE(self, pe_id, pe_type)
    def get_PEs_of_type(self, pe_type)
    def get_type(self, pe_id)
```

**Rationale:**
- RELIEF needs to quickly find all PEs of a given type
- O(1) lookup instead of linear search through PE list
- Supports future multi-type scheduling algorithms
- Clean separation of concerns (type organization vs. queue management)

**Note:** PE types should be granular (ACC_JPEG, ACC_MP3, etc.) not generic (ACC)

---

## 3. Communication Mode Architecture

### 3.1 Three Communication Modes

**Mode 1: PE_to_PE (Legacy)**
- Data transfers directly between PEs
- Used by original DASH-Sim schedulers
- Communication timing: `comm_band[source_PE_ID][dest_PE_ID]`

**Mode 2: shared_memory (Legacy)**
- Data transfers via shared memory
- Used when modeling memory-centric architectures
- Communication timing: `comm_band[MEM_ID][dest_PE_ID]`

**Mode 3: forwarding (New)**
- Hybrid mode enabling RELIEF's opportunistic forwarding
- Dynamic switching between PE_to_PE and memory timing
- Requires scratchpad management and forwarding metadata

### 3.2 Forwarding Mode Logic

**Critical Clarification:** PE_to_PE timing is used FOR forwarding operations

**Decision Logic:**
```
IF comm_mode == 'forwarding':
    IF task.isForwarded == True:
        USE PE_to_PE timing  # Task forwarded to idle accelerator
    ELSE:
        USE memory timing    # Normal scheduling path
ELSE:
    Use legacy mode (PE_to_PE or shared_memory)
```

**Key Insight:** Forwarding mode doesn't replace PE_to_PE mode—it uses PE_to_PE timing when forwarding occurs.

### 3.3 Forwarding Feasibility Conditions

**A task can be forwarded IF:**
1. Communication mode is 'forwarding'
2. Target PE is idle (`PE.idle == True`)
3. Target PE has forwarding enabled (`PE.forwarding_enabled == True`)
4. Target PE has scratchpad capacity (`PE.scratchpad_capacity > 0`)
5. Forwarding doesn't violate laxity constraints of other tasks
6. Target PE can execute the task type

**Laxity Constraint Check (`is_feasible()`):**
- For each non-forwarded task ahead in target PE's queue:
  - If that task has positive laxity: check if `laxity > new_task.runtime`
  - If insufficient laxity: forwarding not feasible
- If feasible: reduce laxity of all ahead tasks by `new_task.runtime`

---

## 4. Data Structure Enhancements

### 4.1 Task Metadata for Communication

**New Task Attributes:**
```python
class Tasks:
    # Communication metadata
    self.comm_timing_mode = None        # 'PE_to_PE' or 'memory'
    self.data_locations = {}            # {predecessor_ID: PE_ID or 'memory'}
    self.forwarded_from_PE = None       # PE_ID if forwarded
    self.data_sizes = {}                # {predecessor_ID: size_bytes}
    self.isForwarded = False            # True if task was forwarded

    # RELIEF metadata
    self.laxity = None                  # deadline - runtime
    self.time_stamp = None              # When added to executable queue
```

**Purpose:**
- Track where predecessor data resides
- Determine communication timing for each dependency
- Support scratchpad allocation decisions
- Enable forwarding metadata updates

### 4.2 Scratchpad Management in PE Class

**New PE Attributes:**
```python
class PE:
    self.scratchpad = {}                # {data_id: {'task_id': X, 'size': Y, 'timestamp': Z}}
    self.scratchpad_capacity = 0        # Bytes (configured per PE type)
    self.scratchpad_used = 0            # Bytes currently allocated
    self.forwarding_enabled = False     # Can this PE participate in forwarding?
```

**Scratchpad Methods:**
```python
def has_data_in_scratchpad(self, data_id) -> bool
def allocate_scratchpad(self, data_id, size, task_id) -> bool  # LRU eviction
def free_scratchpad(self, data_id)
def can_forward(self) -> bool  # idle + enabled + has capacity
```

**LRU Eviction Policy:**
- When allocation exceeds capacity
- Evict oldest data (by timestamp) until space available
- Update `scratchpad_used` accordingly

### 4.3 Configuration Extensions

**New config_file.ini Sections:**
```ini
[COMMUNICATION MODE]
communication_mode = forwarding  # or PE_to_PE or shared_memory

# Scratchpad capacity per PE type (bytes, forwarding mode only)
scratchpad_capacity_CPU = 0
scratchpad_capacity_BIG = 0
scratchpad_capacity_LTL = 0
scratchpad_capacity_ACC_JPEG = 8192
scratchpad_capacity_ACC_MP3 = 8192
scratchpad_capacity_MEM = 0
```

**Configuration Parsing:**
- Read communication mode
- Parse scratchpad capacities into dictionary `{PE_type: capacity}`
- Initialize PE instances with appropriate capacities
- Enable forwarding flag based on mode

---

## 5. RELIEF Scheduler Design

### 5.1 Two-Phase Algorithm

**Phase 1: Task-to-PE Mapping**
```
FOR each ready task:
    Find fastest PE for this task type
    Compute laxity = deadline - min_exec_time
    Assign task.PE_ID = fastest_PE
    Insert task into temporary bucket fwd_nodes[PE_ID] (sorted by laxity)
```

**Phase 2: Forwarding or Normal Scheduling**
```
FOR each PE:
    can_forward = (PE.idle AND forwarding_enabled)

    WHILE fwd_nodes[PE.ID] has tasks:
        task = fwd_nodes[PE.ID].pop()
        task.time_stamp = current_time

        # Find insertion point in common.executable[PE.ID] based on laxity
        insert_index = calculate_insertion_point(task.laxity)

        IF can_forward AND is_feasible(PE.ID, task, insert_index):
            # FORWARD: Insert at front (index 0)
            common.executable[task.PE_ID].insert(0, task)
            task.isForwarded = True
            task.forwarded_from_PE = PE.ID
        ELSE:
            # NORMAL: Insert at calculated position
            common.executable[task.PE_ID].insert(insert_index, task)
            task.isForwarded = False
```

### 5.2 Laxity-Based Insertion

**Finding Insertion Point:**
```python
insert_index = 0
if pe_id in common.executable:
    for exec_task in common.executable[pe_id]:
        if exec_task.laxity <= task.laxity:
            insert_index += 1
        else:
            break
```

**Purpose:** Maintain laxity-ordered queue (most urgent first)

### 5.3 Forwarding Feasibility Check

**`is_feasible(accelerator_id, task, index)` Logic:**
```
can_forward = True

FOR each executable_task before index in common.executable[accelerator_id]:
    IF executable_task is NOT forwarded AND has positive laxity:
        IF executable_task.laxity <= task.runtime:
            can_forward = False
            BREAK

IF can_forward:
    FOR each executable_task before index:
        executable_task.laxity -= task.runtime

RETURN can_forward
```

**Key Insight:** Only non-forwarded tasks contribute to feasibility check (forwarded tasks are already accounted for)

---

## 6. Communication Timing Logic

### 6.1 Dynamic Timing Decision

**Function:** `decide_comm_timing(task, predecessor_task, target_PE_ID)`

**Logic:**
```
IF comm_mode == 'PE_to_PE':
    RETURN 'PE_to_PE'

ELIF comm_mode == 'shared_memory':
    RETURN 'memory'

ELIF comm_mode == 'forwarding':
    IF task.isForwarded:
        RETURN 'PE_to_PE'  # Forwarded task uses direct transfer
    ELSE:
        RETURN 'memory'    # Normal task uses memory transfer
```

### 6.2 Communication Time Calculation

**For PE_to_PE Timing:**
```python
comm_band = ResourceManager.comm_band[source_PE_ID][dest_PE_ID]
comm_time = int(comm_vol / comm_band)
ready_time = predecessor_finish_time + comm_time
```

**For Memory Timing:**
```python
comm_band = ResourceManager.comm_band[MEM_ID][dest_PE_ID]
comm_time = int(comm_vol / comm_band)
ready_time = current_time + comm_time
```

### 6.3 Scratchpad-Aware Communication

**For Forwarded Tasks:**
```
FOR each predecessor:
    IF predecessor data in scratchpad:
        # Data already present, no transfer needed
        ready_time = max(ready_time, predecessor_finish_time)
    ELSE:
        # Calculate transfer time
        timing_mode = decide_comm_timing(task, predecessor, task.PE_ID)
        ready_time = max(ready_time, calculate_comm_time(timing_mode))

        # Allocate scratchpad if forwarding mode
        IF comm_mode == 'forwarding':
            PE.allocate_scratchpad(predecessor.output_id, data_size, task.ID)
```

---

## 7. Simulation Loop Integration

### 7.1 Dictionary-Based Execution Loop

**Main Loop Structure:**
```python
remove_from_executable = {}  # {PE_ID: [tasks_to_remove]}

FOR pe_id, pe_queue in common.executable.items():
    IF pe_queue is empty:
        CONTINUE

    PE = PEs[pe_id]
    tasks_to_remove = []

    FOR executable_task in pe_queue:
        # Check if ready to execute
        IF current_time >= executable_task.time_stamp:
            IF PE has available resources:
                IF dynamic dependencies met:
                    # Execute task
                    PE.queue.append(executable_task)
                    env.process(PE.run(executable_task))
                    tasks_to_remove.append(executable_task)

    IF tasks_to_remove:
        remove_from_executable[pe_id] = tasks_to_remove

# Clean up executed tasks
FOR pe_id, tasks in remove_from_executable.items():
    FOR task in tasks:
        common.executable[pe_id].remove(task)
```

### 7.2 Backward Compatibility

**Legacy Schedulers (MET, EFT):**
- Still use `update_execution_queue(ready_list)`
- This function now appends to `common.executable[task.PE_ID]`
- Simulation loop processes all tasks uniformly

**Advanced Schedulers (RELIEF, CP):**
- Directly populate `common.executable` dictionary
- Bypass `update_execution_queue()` entirely
- Full control over queue ordering and task placement

---

## 8. Implementation File Map

### 8.1 Core Data Structures (common.py)

**Changes:**
- Line 416: `executable = {}` (dictionary, not list)
- Lines 251-296: Add `PETypeManager` class
- Lines 127-182: Add communication mode config variables
- Lines 375-379: Add task communication metadata attributes

### 8.2 Job Generation (job_generator.py)

**Changes:**
- Lines 51-53: Unified initialization `common.executable = {pe.ID: [] for pe in self.PEs}`
- Lines 147-151: Update CP scheduler logic for dictionary access

### 8.3 Scheduler Logic (scheduler.py)

**Changes:**
- Lines 38-42: Remove `scheduler.pe_queues` (causes state loss)
- Lines 45-58: Add `populate_execution_queue()` helper
- Lines 834-862: Implement RELIEF_BASIC two-phase algorithm
- Lines 870-908: Implement `is_feasible()` laxity checking
- Lines 972-1058: Add `update_forward_metadata()` stub with TODO

### 8.4 PE Enhancements (processing_element.py)

**Changes:**
- Lines 53-57: Add scratchpad attributes
- Lines 271-346: Implement scratchpad management methods
  - `has_data_in_scratchpad()`
  - `allocate_scratchpad()` with LRU eviction
  - `free_scratchpad()`
  - `can_forward()`

### 8.5 Simulation Core (DASH_Sim_core.py)

**Changes:**
- Lines 143-183: Add `decide_comm_timing()` method
- Lines 303-305: Update `update_execution_queue()` for dictionary
- Lines 316-318: Sort each PE's queue individually
- Lines 418-469: Dictionary-based simulation loop (remove duplicate)
- Lines 472-477: Update DRL scheduler logic

### 8.6 RELIEF Helpers (RELIEF_Sim_helpers.py)

**Changes:**
- Lines 97-101: Remove invalid list access patterns
- Lines 110-112: Update sort for dictionary structure

### 8.7 Main Entry Point (DASH_Sim_v0.py)

**Changes:**
- Lines 157-180: Initialize PETypeManager and configure forwarding mode

---

## 9. Testing and Validation Plan

### 9.1 Unit Testing Priorities

**PETypeManager:**
- Test PE registration
- Verify O(1) lookup performance
- Test with multiple PEs of same type

**Scratchpad Management:**
- Test LRU eviction when capacity exceeded
- Verify allocation/deallocation tracking
- Test edge cases (zero capacity, oversized data)

**Communication Timing:**
- Verify mode switching logic
- Test PE_to_PE vs memory timing calculations
- Validate forwarded task timing

### 9.2 Integration Testing

**RELIEF Algorithm:**
- Test two-phase scheduling with forwarding enabled
- Verify laxity-based insertion ordering
- Test feasibility checking with various queue states
- Validate forwarding vs normal scheduling decisions

**Backward Compatibility:**
- Run existing benchmarks with MET, EFT schedulers
- Verify identical results to pre-refactoring baseline
- Test DRL and CP schedulers

### 9.3 Performance Testing

**Metrics to Track:**
- Average job completion time
- Deadline miss rate
- Forwarding success rate
- Scratchpad hit rate
- Average laxity at execution time

**Test Scenarios:**
- Varying workload intensities (scale parameter)
- Different PE type configurations
- Mixed job types (heterogeneous workloads)
- Forwarding mode vs legacy modes comparison

---

## 10. Known Limitations and Future Work

### 10.1 Current Limitations

**Scratchpad Model:**
- Assumes instant data availability if in scratchpad
- No modeling of scratchpad access latency
- LRU eviction may not be optimal for all workloads

**Forwarding Logic:**
- Single-level forwarding (no multi-hop)
- No priority inversion prevention beyond laxity checks
- Assumes homogeneous communication bandwidth

**PE Type Organization:**
- Requires manual PE type configuration
- No automatic type inference
- Type changes require reconfiguration

### 10.2 Future Enhancements

**Advanced Forwarding:**
- Multi-hop forwarding chains
- Predictive forwarding based on upcoming ready tasks
- Power-aware forwarding decisions

**Scratchpad Optimization:**
- Cache-aware eviction policies (LFU, ARC)
- Prefetching for predicted data needs
- Scratchpad sharing between PEs

**Type-Aware Scheduling:**
- Heterogeneity-aware task mapping
- Dynamic type affinity learning
- Cross-type migration support

**Real-Time Extensions:**
- Hard real-time guarantee mode
- Admission control for deadline guarantees
- Slack reclamation techniques

---

## 11. Architectural Principles

### 11.1 Design Patterns Used

**Single Source of Truth:**
- `common.executable` is THE executable queue
- No duplicate or shadow queue structures
- All components reference same state

**Separation of Concerns:**
- PETypeManager: Type organization
- Scratchpad: Data locality tracking
- Scheduler: Task ordering decisions
- Simulator: Execution timing

**Open-Closed Principle:**
- Legacy schedulers continue to work unchanged
- New schedulers can leverage enhanced features
- Configuration-driven behavior (no code changes for mode switching)

### 11.2 Key Invariants

**Queue Integrity:**
- Every task in `common.executable[pe_id]` must have `task.PE_ID == pe_id`
- No task should exist in multiple PE queues simultaneously
- Tasks in queue must have `time_stamp` set

**Laxity Ordering (RELIEF):**
- Tasks in each PE's queue ordered by increasing laxity
- Forwarded tasks always at front (index 0)
- Laxity updated as tasks are inserted

**Scratchpad Consistency:**
- `PE.scratchpad_used <= PE.scratchpad_capacity` always
- Data in scratchpad must correspond to completed predecessor
- Evicted data should not be referenced by pending tasks

### 11.3 Error Handling Philosophy

**Fail Fast:**
- Invalid PE IDs raise exceptions immediately
- Configuration errors detected at initialization
- Type mismatches caught before simulation starts

**Graceful Degradation:**
- If scratchpad full, fall back to memory transfer
- If forwarding infeasible, use normal scheduling
- Missing data triggers explicit error, not silent failure

---

## 12. Critical Architectural Insights from Session

### 12.1 The State Persistence Problem

**Original Issue:**
- Separate `scheduler.pe_queues` caused state loss
- Tasks removed from `common.executable` during simulation weren't tracked in scheduler
- Led to double processing and inconsistent state

**Solution:**
- Single shared dictionary `common.executable`
- Both scheduler and simulator operate on same structure
- State changes immediately visible to both components

### 12.2 PE_to_PE != Separate from Forwarding

**Original Misunderstanding:**
- Thought PE_to_PE and forwarding were orthogonal modes
- Believed forwarding decision based on scratchpad contents

**Correct Understanding:**
- Forwarding mode USES PE_to_PE timing when forwarding occurs
- Decision based on `task.isForwarded` flag (set by scheduler)
- Memory timing used when NOT forwarding (even in forwarding mode)

### 12.3 Per-PE-Type vs Per-PE Queues

**Evolution:**
- Initially planned: One queue per PE TYPE (all JPEGs share queue)
- Problem: Can't track individual PE idle status or capacity
- Solution: One queue per PE, but organize lookups by type via PETypeManager

**Benefit:**
- Fine-grained scheduling control
- Individual PE state tracking
- Type-based optimization still possible via PETypeManager

---

## 13. Summary

This architectural plan establishes a comprehensive framework for RELIEF scheduler implementation with the following key components:

1. **Dictionary-based executable queues** enabling per-PE task management
2. **PETypeManager** for efficient type-based PE organization
3. **Three communication modes** with dynamic switching logic
4. **Scratchpad management** with LRU eviction for data locality
5. **Two-phase RELIEF algorithm** with laxity-based forwarding
6. **Backward compatibility** with existing schedulers
7. **Single source of truth** architecture preventing state loss

The design maintains the flexibility to support future enhancements while preserving the existing DASH-Sim functionality and simulator architecture.
