'''!
@brief This file contains all the common parameters used in DASH_Sim.
'''
import configparser
import sys
import os
import ast
import networkx as nx
import pickle
import numpy as np
from typing import List


#time_at_sim_termination = -1

def str_to_list(x):
    # function to return a list based on a formatted string
    result = []
    if '[' in x:
        result = ast.literal_eval(x)
    else:
        for part in x.split(','):
            if ('-' in part):
                a, b, c = part.split('-')
                a, b, c = int(a), int(b), int(c)
                result.extend(range(a, b, c))
            elif ('txt' in part):
                result.append(part)
            else:
                a = int(part)
                result.append(a)
    return result
# end of def str_to_list(x)


config = configparser.ConfigParser()
config.read('config_file.ini')

# Assign debug variable to be true to check the flow of the program
DEBUG_CONFIG    = config.getboolean('DEBUG', 'debug_config')                    # Debug variable to check the DASH-Sim configuration related debug messages
DEBUG_SIM       = config.getboolean('DEBUG', 'debug_sim')                       # Debug variable to check the Simulation core related debug messages
DEBUG_JOB       = config.getboolean('DEBUG', 'debug_job')                       # Debug variable to check the Job generator related debug messages
DEBUG_SCH       = config.getboolean('DEBUG', 'debug_sch')                       # Debug variable to check the Scheduler related debug messages

# Assign info variable to be true to get the information about the flow of the program
INFO_SIM        = config.getboolean('INFO', 'info_sim')                         # Info variable to check the Simulation core related info messages
INFO_JOB        = config.getboolean('INFO', 'info_job')                         # Info variable to check the job generator related info messages
INFO_SCH        = config.getboolean('INFO', 'info_sch')                         # Info variable to check the Scheduler related info messages

## DEFAULT
scheduler               = config['DEFAULT']['scheduler']                        # Assign scheduler name variable
seed                    = int(config['DEFAULT']['random_seed'])                 # Specify a seed value for the random number generator
simulation_clk          = int(config['DEFAULT']['clock'])                       # The core simulation engine tick with simulation_clk
arbitration_type        = config['DEFAULT']['arbitration_type']
simulation_length       = int(config['DEFAULT']['simulation_length'])           # The length of the simulation (in us)
standard_deviation      = float(config['DEFAULT']['standard_deviation'])        # Standard deviation for randomization of execution time
job_probabilities       = str_to_list(config['DEFAULT']['job_probabilities'])   # Probability of each app for being selected as the new job
inject_jobs_ASAP        = config.getboolean('DEFAULT', 'inject_jobs_asap')
fixed_injection_rate    = config.getboolean('DEFAULT', 'fixed_injection_rate')
max_jobs_in_parallel    = int(config['DEFAULT']['max_jobs_in_parallel'])

job_list = str_to_list(config['DEFAULT']['job_list'])                           # List containing the number of jobs that should be executed for each application
if len(job_list) > 0:
    current_job_list = job_list[0]
    snippet_size = sum(job_list[0])
    max_num_jobs = snippet_size*len(job_list)
    inject_fixed_num_jobs = True
else:
    current_job_list = []
    max_num_jobs = int(config['DEFAULT']['max_jobs'])
    snippet_size = max_num_jobs
    inject_fixed_num_jobs = config.getboolean('DEFAULT', 'inject_fixed_num_jobs')
job_counter_list = [0] * len(current_job_list)                                  # List to count the number of injected jobs for each application

## TRACE
# Assign trace variable to be true to save traces from the execution
CLEAN_TRACES                        = config.getboolean('TRACE', 'clean_traces')              # Flag used to clear previous traces
TRACE_TASKS                         = config.getboolean('TRACE', 'trace_tasks')               # Trace information from each task
TRACE_SYSTEM                        = config.getboolean('TRACE', 'trace_system')              # Trace information from the whole system
TRACE_FREQUENCY                     = config.getboolean('TRACE', 'trace_frequency')           # Trace frequency variation information
TRACE_PES                           = config.getboolean('TRACE', 'trace_PEs')                 # Trace information from each PE
TRACE_IL_PREDICTIONS                = config.getboolean('TRACE', 'trace_IL_predictions')      # Trace the predictions of the IL policy
TRACE_TEMPERATURE                   = config.getboolean('TRACE', 'trace_temperature')         # Trace temperature information
TRACE_LOAD                          = config.getboolean('TRACE', 'trace_load')                # Trace system load information
CREATE_DATASET_DTPM                 = config.getboolean('TRACE', 'create_dataset_DTPM')       # Create dataset for the ML algorithm
TRACE_FILE_TASKS                    = config['TRACE']['trace_file_tasks']                     # Trace file name for the task trace
TRACE_FILE_SYSTEM                   = config['TRACE']['trace_file_system']                    # Trace file name for the system trace
TRACE_FILE_FREQUENCY                = config['TRACE']['trace_file_frequency']                 # Trace file name for the frequency trace
TRACE_FILE_PES                      = config['TRACE']['trace_file_PEs']                       # Trace file name for the PE trace
TRACE_FILE_TEMPERATURE              = config['TRACE']['trace_file_temperature']               # Trace file name for the temperature trace
TRACE_FILE_TEMPERATURE_WORKLOAD     = config['TRACE']['trace_file_temperature_workload']      # Trace file name for the temperature trace (workload)
TRACE_FILE_LOAD                     = config['TRACE']['trace_file_load']                      # Trace file name for the load trace
RESULTS                             = config['TRACE']['results']                              # Trace file name for the results of the simulation, including exec time, energy, etc.

## POWER MANAGEMENT
sampling_rate                   = int(config['POWER MANAGEMENT']['sampling_rate'])                      # Specify the sampling rate for the DVFS mechanism
sampling_rate_temperature       = int(config['POWER MANAGEMENT']['sampling_rate_temperature'])          # Specify the sampling rate for the temperature update
util_high_threshold             = float(config['POWER MANAGEMENT']['util_high_threshold'])              # Specify the high threshold (ondemand mode)
util_low_threshold              = float(config['POWER MANAGEMENT']['util_low_threshold'])               # Specify the low threshold  (ondemand mode)
enable_throttling               = config.getboolean('POWER MANAGEMENT', 'enable_throttling')            # Flag to enable the thermal throttling
enable_DTPM_throttling          = config.getboolean('POWER MANAGEMENT', 'enable_DTPM_throttling')       # Flag to enable the thermal throttling for the custom DTPM policies
C1                              = float(config['POWER MANAGEMENT']['C1'])                               # Coefficient for the leakage model
C2                              = int(config['POWER MANAGEMENT']['C2'])                                 # Coefficient for the leakage model
Igate                           = float(config['POWER MANAGEMENT']['Igate'])                            # Coefficient for the leakage model
T_ambient                       = float(config['POWER MANAGEMENT']['T_ambient'])                        # Ambient temperature

trip_temperature                = str_to_list(config['POWER MANAGEMENT']['trip_temperature'])           # List of temperature trip points
trip_hysteresis                 = str_to_list(config['POWER MANAGEMENT']['trip_hysteresis'])            # List of hysteresis trip points
DTPM_trip_temperature           = str_to_list(config['POWER MANAGEMENT']['DTPM_trip_temperature'])      # List of temperature trip points for the custom DTPM policies

## SIMULATION MODE
simulation_mode = config['SIMULATION MODE']['simulation_mode']                  # Defines under which mode, the simulation will be run

if simulation_mode not in ('validation','performance') :
    print('[E] Please choose a valid simulation mode')
    print(simulation_mode)
    sys.exit()

# variables used under performance mode
warmup_period       = int(config['SIMULATION MODE']['warmup_period'])                   # is the time period till which no result will be recorded
num_of_iterations   = int(config['SIMULATION MODE']['num_of_iterations'])               # The number of iteration at each job injection rate
config_scale_values = config['SIMULATION MODE']['scale_values']
scale_values_list = str_to_list(config_scale_values)                                    # List of scale values which will determine the job arrival rate under performance mode

# variables used under validation mode
scale = int(config['SIMULATION MODE']['scale'])                                 # The variable used to adjust the mean value of the job inter-arrival time
if (simulation_mode == 'validation'):
    warmup_period = 0                                                           # Warmup period is zero under validation mode

## COMMUNICATION MODE
packet_size      = int(config['COMMUNICATION MODE']['packet_size'])               # The packet size (in bits)

# Check if using new communication_mode parameter or legacy PE_to_PE/shared_memory flags
if 'communication_mode' in config['COMMUNICATION MODE']:
    # New mode: 'PE_to_PE', 'shared_memory', or 'forwarding'
    comm_mode = config['COMMUNICATION MODE']['communication_mode'].strip()

    # Validate mode
    if comm_mode not in ('PE_to_PE', 'shared_memory', 'forwarding'):
        print('[E] Invalid communication_mode. Must be one of: PE_to_PE, shared_memory, forwarding')
        sys.exit()

    # Set legacy flags for backwards compatibility
    if comm_mode == 'PE_to_PE':
        PE_to_PE = True
        shared_memory = False
        forwarding_enabled = False
    elif comm_mode == 'shared_memory':
        PE_to_PE = False
        shared_memory = True
        forwarding_enabled = False
    elif comm_mode == 'forwarding':
        # In forwarding mode, we don't use legacy paths
        PE_to_PE = False
        shared_memory = False
        forwarding_enabled = True
else:
    # Legacy mode: use PE_to_PE and shared_memory boolean flags
    PE_to_PE         = config.getboolean('COMMUNICATION MODE', 'PE_to_PE')            # The communication mode in which data is sent, directly, from a PE to a PE
    shared_memory    = config.getboolean('COMMUNICATION MODE', 'shared_memory')       # The communication mode in which data is sent from a PE to a PE through a shared memory
    forwarding_enabled = False

    # Set comm_mode based on legacy flags
    if PE_to_PE and shared_memory:
        print('[E] Please chose only one of the communication modes')
        sys.exit()
    elif not PE_to_PE and not shared_memory:
        print('[E] Please chose one of the communication modes')
        sys.exit()
    elif PE_to_PE:
        comm_mode = 'PE_to_PE'
    else:
        comm_mode = 'shared_memory'


write_time       = -1
read_time        = -1
PE_to_Cache      = {}

# Memory contention parameters (tunable)
MEMORY_DEGRADATION_RATE = 0.5  # Linear degradation coefficient
MEMORY_MIN_FACTOR = 0.3        # Minimum bandwidth multiplier (30% at full load)

iteration = 0

# The variables used by table-based schedulers
table   = -1
table_2 = -1
table_3 = -1
table_4 = -1
temp_list = []
# Additional variables used by list-based schedulers
current_dag      = nx.DiGraph()
computation_dict = {}
power_dict       = {}

## DTPM
current_temperature_vector  = [T_ambient,                  # Indicate the current PE temperature for each hotspot
                               T_ambient,
                               T_ambient,
                               T_ambient,
                               T_ambient]
B_model = []
throttling_state = -1
trace_file_num = 0
DVFS_cfg_list = []

# Snippet_inj is incremented every time a snippet finishes being injected
snippet_ID_inj                      = -1
# Snippet_exec is incremented every time a snippet finishes being executed
snippet_ID_exec                     = 0
snippet_throttle                    = -1
snippet_temp_list                   = []
snippet_initial_temp                = [T_ambient,
                                       T_ambient,
                                       T_ambient,
                                       T_ambient,
                                       T_ambient]

snippet_start_time                  = 0
## End of DTPM

class PerfStatics:
    '''!
    Define the PerfStatics class to calculate energy consumption and total execution time.
    '''
    def __init__(self):
        self.execution_time = 0.0                   # The total execution time (us)
        self.energy_consumption = 0.0               # The total energy consumption (uJ)
        self.cumulative_exe_time = 0.0              # Sum of the execution time of completed tasks (us)
        self.cumulative_energy_consumption = 0.0    # Sum of the energy consumption of completed tasks (us)
        self.injected_jobs = 0                      # Count the number of jobs that enter the system (i.e. the ready queue)
        self.completed_jobs = 0                     # Count the number of jobs that are completed 
        self.ave_execution_time = 0.0               # Average execution time for the jobs that are finished 
        self.job_counter = 0                        # Indicate the number of jobsin the system at any given time
        self.average_job_number = 0                 # Shows the average number of jobs in the system for a workload
        self.deadlines_met = 0                      # Shows the number of deadlines met
        self.deadlines_missed = 0                   # Shows the number of deadlines missed
        self.amount_deadlines_overrun = []           # shows the amount that deadlines were deferred before finishing
        self.job_counter_list = []
        self.memory_overhead = 0                    # Indicates time spend on memory transfers
        self.sampling_rate_list = []
        self.num_forwards = 0
        self.num_RELIEF_forwards = 0
        self.num_colocations = 0

        self.colocationData = 0
        self.forwardData = 0
        self.memoryData = 0
# end class PerfStatics

# Instantiate the object that will store the performance statistics
global results 

results = PerfStatics()

class Validation:
    '''!
    Define the Validation class to compare the generated and completed jobs.
    '''
    start_times = []
    finish_times = []
    generated_jobs = []
    injected_jobs = []
    completed_jobs = []
# end class Validation

class Resource:
    '''!
    Define the Resource class to define a resource.
    It stores properties of the resources.
	'''
    def __init__(self):
        self.type = ''                          # The type of the resource (CPU, FFT_ACC, etc.)
        self.accelerator_family = ''            # Family grouping for multiple instances (e.g., CONVOLUTION, ISP)
        self.scratchpad_size = 0                # Scratchpad buffer size in bytes (default: 256KB)
        self.name = ''                          # Name of the resource
        self.ID = -1                            # This is the unique ID of the resource. "-1" means it is not initialized
        self.cluster_ID = -1                    # ID of the cluster this PE belongs to
        self.capacity = 1                       # Number tasks that a resource can run simultaneously. Default value is 1.
        self.num_of_functionalites = 0          # This variable shows how many different task this resource can run
        self.supported_functionalities = []     # List of all tasks can be executed by Resource
        self.performance = []                   # List of runtime (in micro seconds) for each supported task
        self.idle = True                        # initial state of Resource which idle and ready for a task (normalized to the number of instructions)
        self.mesh_name = -1
        self.position = -1
        self.width = -1
        self.height = -1
# end class Resource

class ResourceManager:
    '''!
    Define the ResourceManager class to maintain the list of the resource in our DASH-SoC model.
    '''
    def __init__(self):
        self.list:list[Resource] = []                          # List of available resources
        self.comm_band = []                     # This variable represents the communication bandwidth matrix
# end class ResourceManager

class ClusterManager:
    '''!
    Define the ClusterManager class to maintain the list of clusters in our DASH-SoC model.
    '''
    def __init__(self):
        self.cluster_list = []                  # list of available clusters
# end class ClusterManager

class PETypeManager:
    '''!
    Define the PETypeManager class to organize PEs by type for efficient lookup.
    This enables O(1) access to all PEs of a specific type (e.g., all ACC_JPEG accelerators).
    '''
    def __init__(self):
        self.by_type = {}                       # Dictionary mapping PE type to list of PE IDs: {'CPU': [0,1,2], 'ACC_JPEG': [3,4], ...}
        self.by_id = {}                         # Dictionary mapping PE ID to PE type: {0: 'CPU', 1: 'CPU', 3: 'ACC_JPEG', ...}
        self.by_family = {}                     # Dictionary mapping accelerator family to list of PE IDs: {'CONVOLUTION': [6,7,8], ...}

    def register_PE(self, pe_id, pe_type):
        '''!
        Register a PE with its type for efficient lookup.
        @param pe_id: The ID of the PE to register
        @param pe_type: The type of the PE (e.g., 'CPU', 'ACC_JPEG', etc.)
        '''
        # Add to by_id mapping

        self.by_id[pe_id] = pe_type

        # Add to by_type mapping
        if pe_type not in self.by_type:
            self.by_type[pe_type] = []
        self.by_type[pe_type].append(pe_id)

    def get_PEs_of_type(self, pe_type):
        '''!
        Get list of all PE IDs of a specific type.
        @param pe_type: The type of PEs to retrieve
        @return: List of PE IDs, or empty list if type not found
        '''
        return self.by_type.get(pe_type, [])

    def get_type_of_PE(self, pe_id):
        '''!
        Get the type of a specific PE.
        @param pe_id: The ID of the PE
        @return: PE type string, or None if not found
        '''
        return self.by_id.get(pe_id, None)

    def get_all_types(self):
        '''!
        Get list of all PE types registered.
        @return: List of PE type strings
        '''
        return list(self.by_type.keys())

    def get_all_families(self):
        '''!
        Get list of all PE types registered.
        @return: List of PE type strings
        '''
        return list(self.by_family.keys())
    
    def register_PE_family(self, pe_id, family):
        '''!
        Register a PE's accelerator family for efficient multi-instance lookup.
        @param pe_id: The ID of the PE to register
        @param family: The accelerator family (e.g., 'CONVOLUTION', 'ISP')
        '''
        if family not in self.by_family:
            self.by_family[family] = []
        self.by_family[family].append(pe_id)

    def get_PEs_of_family(self, family):
        '''!
        Get list of all PE IDs belonging to an accelerator family.
        @param family: The accelerator family name
        @return: List of PE IDs, or empty list if family not found
        '''
        return self.by_family[family]
# end class PETypeManager

TypeManager = PETypeManager()

class Tasks:
    '''!
    Define the Tasks class to maintain the list of tasks.
    It stores properties of the tasks.
    '''
    def __init__(self):
        self.name = ''                          # The name of the task
        self.ID = -1                            # This is the unique ID of the task. "-1" means it is not initialized
        self.predecessors = []                  # List of all task IDs to identify task dependency
        self.est = -1                           # This variable represents the earliest time that a task can start
        self.deadline = -1                      # This variable represents the deadline for a task
        self.sd = -1                            # Sub-deadline ratio value in microseconds
        self.runtime = -1                       # Represents the runtime of the task - once a PE has been selected
        self.laxity = -1                        # Indicates the Laxity of the Task, once a PE has been selected 
        self.head = False                       # If head is true, this task is the leading (the first) element in a task graph
        self.tail = False                       # If tail is true, this task is the end (the last) element in a task graph
        self.jobID = -1                         # This task belongs to job with this ID
        self.jobDeadline = -1                   # The deadline of the overarching Job - TODO implement in job_generator
        self.jobname = ''                       # This task belongs to job with this name
        self.base_ID = -1                       # This ID will be used to calculate the data volume from one task to another
        self.PE_Family = ""                     # Holds the name of the PE Family that the task is bound to
        self.PE_ID = -1                         # Holds the PE ID on which the task will be executed
        self.start_time = -1                    # Execution start time of a task
        self.finish_time = -1                   # Execution finish time of a task
        self.order = -1                         # Relative ordering of this task on a particular PE
        self.dynamic_dependencies = []          # List of dynamic dependencies that a scheduler requests are satisfied before task launch
        self.ready_wait_times = []              # List holding wait times for a task for being ready due to communication time from its predecessor
        self.execution_wait_times = []          # List holding wait times for a task for being execution-ready due to communication time between memory and a PE 
        self.PE_to_PE_wait_time = []            # List holding wait times for a task for being execution-ready due to PE to PE communication time
        self.order = -1                         # Execution order if a list based scheduler is used, e.g., ILP
        self.task_elapsed_time_max_freq = 0     # Indicate the current execution time for a given task
        self.job_start = -1                     # Holds the execution start time of a head task (also execution start time for a job)
        self.time_stamp = -1                    # This values used to check whether all data for the task is transferred or not
        self.input_packet_size = -1
        self.output_packet_size = -1
        self.isForwarded = False                # Indicates whether this Task has inputs that are being forwarded.

        # Forwarding metadata (only populated in forwarding mode)
        self.comm_timing_mode = None            # 'PE_to_PE' or 'memory' - decided at scheduling time for this task
        self.data_locations = {}                # Maps predecessor_task_ID to PE_ID where that predecessor's output lives
        self.forwarded_from_PE = None           # PE_ID if this task was forwarded to a specific PE, else None
        self.data_sizes = {}                    # Maps predecessor_task_ID to size of data in bytes
# end class Tasks

class TaskManager:
    '''!
    Define the TaskManager class to maintain the list of the tasks in our DASH-SoC model.
    '''
    def __init__(self):
        self.list = []                          # List of available tasks
# end class TaskManager

class Applications:
    '''!
    Define the Applications class to maintain all information about an application (job)
    '''
    def __init__(self):
        self.name =  ''                         # The name of the application
        self.deadline = 0
        self.task_list = []                     # List of all tasks in an application
        self.comm_vol = []                      # This variable represents the communication volume matrix
        # i.e. each entry is data volume should be transferred from one task to another
# end class Applications

class ApplicationManager:
    '''!
    Define the ApplicationManager class to maintain the list of the applications (jobs) in our DASH-SoC model.
    '''
    def __init__(self):
        self.list = []                          # List of all applications
# end class ApplicationManager

#class TaskQueues:

outstanding:List[Tasks] = []                   # List of *all* tasks waiting to be processed
ready:List[Tasks] = []                         # List of tasks that are ready for processing
running:List[Tasks] = []                       # List of currently running tasks
completed:List[Tasks] = []                     # List of completed tasks
wait_ready:List[Tasks] = []                    # List of task waiting for being pushed into ready queue because of memory communication time
active_noc_transfers = []                      # List of active NoC transfers
executable = {}                    # Dictionary of per-PE executable queues: {PE_ID: [task_list]} 
new_schedulers = ['RELIEF', 'LL', 'GEDF_D', 'GEDF_N', 'HetSched', 'FCFS']         # List of schedulers introduced to take advantage of my reeimplementation of DS3
deprioritize_negative_laxity = ['RELIEF', 'LL']

# no actual self, but we need access to the jobs and env variables from wherever it is being called from (scheduler or dash sim core)
def calculate_memory_movement_latency(caller, executable_task:Tasks, PE_ID, canAllocate):
    isColocated = False

    # Get the Job Name
    for ind, job in enumerate(caller.jobs.list):
        if job.name == executable_task.jobname:
            job_ID = ind

    # Get the list of tasks 
    for task in caller.jobs.list[job_ID].task_list:
        if executable_task.base_ID == task.ID:
            

            bandwidth = []

            src_list = []

            data_ids = []

            data_volumes = []

            total_data = executable_task.input_packet_size

            cancel = canAllocate # only cancel if we can allocate


            if executable_task.head == True:
                if canAllocate and DEBUG_SIM:
                    print('task %d is a head task' %(executable_task.ID))
            elif canAllocate and DEBUG_SIM:
                print('[D] task %d num predecessors %d' %(executable_task.ID, len(task.predecessors)))
            
            # This tracks the total amount of input data to this task. If we need more data than is covered by the 
            # predecessors, then the assumption is that we need something from memory as well (weights, biases, etc.)

            #add predecessors for allocating output scratchpad
            if canAllocate:
                for predecessor in task.predecessors:
                    real_predecessor_ID = predecessor + executable_task.ID - executable_task.base_ID
                    caller.PEs[executable_task.PE_ID].dependencies.append(real_predecessor_ID) #removed in update ready queue 
                out = caller.PEs[executable_task.PE_ID].allocate_scratchpad(f"{executable_task.ID}_output",executable_task.output_packet_size*packet_size,executable_task.ID)                 # allocate room in the scratchpad for the output of this task.
                cancel = out and cancel
            for predecessor in task.predecessors:

                # data required from the predecessor for $ready_task
                comm_vol = caller.jobs.list[job_ID].comm_vol[predecessor, executable_task.base_ID]

                # subtract it from the data total we need
                total_data -= comm_vol

                # retrieve the real ID  of the predecessor based on the job ID
                real_predecessor_ID = predecessor + executable_task.ID - executable_task.base_ID

                # Initialize following two variables which will be used if
                # PE to PE communication is utilized
                predecessor_PE_ID = -1
                predecessor_task = None

                # Find the predecessor task to get its PE and finish time
                for cmp in completed:
                    if cmp.ID == real_predecessor_ID:
                        predecessor_PE_ID = cmp.PE_ID
                        predecessor_task = cmp
                        break

                assert predecessor_task is not None

                # Decide communication timing mode (PE_to_PE or memory)
                comm_timing = decide_comm_timing(caller, executable_task, predecessor_task, predecessor_task.PE_ID)

                if comm_timing == 'PE_to_PE' or PE_to_PE:
                    # Use PE-to-PE communication timing (includes forwarding in forwarding mode)
                    comm_band = ResourceManager.comm_band[predecessor_PE_ID, PE_ID]
                    data_id = f"{predecessor_task.ID}_output"

                    # data for combined data transfer
                    bandwidth.append(comm_band)
                    data_volumes.append(comm_vol)
                    src_list.append(predecessor_PE_ID)
                    data_ids.append(data_id)

                    # measurement data and scratchpad allocation
                    if canAllocate and predecessor_PE_ID != PE_ID:
                        results.forwardData += comm_vol
                        results.num_forwards += 1
                        out = caller.PEs[executable_task.PE_ID].allocate_scratchpad(data_id, comm_vol*packet_size, predecessor_task.ID)
                        cancel = cancel and out

                    elif canAllocate:
                        results.num_colocations += 1
                        results.colocationData += comm_vol

                    # handle colocation - data volume set to 0, no scratchpad allocation
                    if predecessor_PE_ID == PE_ID:
                        isColocated = True
                        data_volumes[-1] = 0
                # end of if comm_timing == 'PE_to_PE':

                if comm_timing == 'memory' or shared_memory:
                    # Use memory communication timing
                    comm_band = ResourceManager.comm_band[caller.resource_matrix.list[-1].ID, PE_ID]
                    data_id = f"{predecessor_task.ID}_output"

                    bandwidth.append(comm_band)
                    data_volumes.append(comm_vol)
                    src_list.append(-1)
                    data_ids.append(data_id)

                    if canAllocate:
                        results.memoryData += comm_vol
                        out = caller.PEs[executable_task.PE_ID].allocate_scratchpad(data_id, comm_vol*packet_size, predecessor_task.ID)
                        cancel = cancel and out
                # end of if comm_timing == 'memory'
            # end of for predecessor in task.predecessors:

            if total_data > 0:
                comm_band = ResourceManager.comm_band[caller.resource_matrix.list[-1].ID, PE_ID]
                data_id = f"{executable_task.ID}_input"

                bandwidth.append(comm_band)
                data_volumes.append(total_data)
                src_list.append(-1)
                data_ids.append(data_id)

                if canAllocate:
                    results.memoryData += total_data
                    out = caller.PEs[executable_task.PE_ID].allocate_scratchpad(data_id, total_data*packet_size, executable_task.ID)
                    cancel = cancel and out
            # end if total_data > 0

            # stall if we can't get enough scratchpad space
            if not cancel and canAllocate:
                caller.PEs[executable_task.PE_ID].dependencies = []
                return False
            assert total_data >= 0

            assert len(bandwidth) == len(data_volumes) == len(src_list) == len(data_ids)


            # Set the time stamp for when the task is ready for execution
            # Use sums since DMA is done back to back
            if canAllocate:
                increase_congestion(caller.env.now, data_volumes, src_list, executable_task.PE_ID, data_ids, bandwidth, caller, executable_task)
                return True
            else:
                return (isColocated, sum(int(x / y) for x, y in zip(data_volumes, bandwidth))) #return ideal numbers for the purposes of scheduling
            # end else
        # end if executable_task.base_ID == task.ID:
    # end  for task in self.jobs.list[job_ID].task_list:

# https://github.com/booksim/booksim2



def get_congested_bandwidth():
    '''!
    Calculate available NoC bandwidth per active transfer.
    Only counts transfers in 'active' state (queued transfers don't affect congestion).

    @return: Bandwidth in bytes/us available per active transfer
    '''
    global_bandwidth = 16000  # bytes/us

    # Only count ACTIVE transfers (queued transfers don't contribute to congestion)
    active_count = sum(1 for t in active_noc_transfers if t['state'] == 'active')

    if active_count == 0:
        return global_bandwidth

    return global_bandwidth / active_count


def increase_congestion(current_time, volumes, src_PEs, dst_PE, data_IDs, bandwidths, caller, task = None, ):
    '''!
    Add a new NoC transfer and update congestion state.
    Determines if transfer can start immediately (active) or must wait (queued).

    @param current_time: Current simulation time
    @param volumes: List of data volumes in bytes
    @param src_PEs: List of source PE IDs (-1 for memory)
    @param dst_PE: Destination PE ID
    @param data_IDs: List of data identifiers
    @param bandwidths: List of max bandwidths in bytes/us
    @param task: Associated task (optional)
    @param start_time: Requested start time (optional, legacy parameter)
    @param caller: SimulationManager instance (required for contention checking)
    '''
    # Determine transfer state using calculate_contention()
    state = calculate_contention(caller, src_PEs[0], dst_PE, data_IDs[0], current_time)

    transfer_dict = {
        'current_transfer': 0, # tracks which transfer we are currently working on.
        'total_transfers': len(volumes), # tracks the total number of transfers we need to do before we finish
        'volume': volumes.copy(),
        'src_PEs': src_PEs.copy(),
        'dst_PE': dst_PE,
        'data_IDs': data_IDs.copy(),
        'max_bandwidths': bandwidths.copy(),
        'congested_bandwidths': bandwidths.copy(),
        'task': task, # pointer to the task associated with the data transfer, to be able to delay / bring forward its wait time.
        'update_time': current_time, # Time since the late update to congestion - used for calculating the remaining volume of data.
        'finish_time': current_time, # Time to completion - updated as congestion increases and decreases.
        'state': state  # Transfer state: 'queued' or 'active'
    }

    active_noc_transfers.append(transfer_dict)

    # If active, update timers and set DMA ownership
    if state == 'active':
        if src_PEs[0] != -1:
            caller.PEs[src_PEs[0]].active_dma_transfer = transfer_dict
        if dst_PE != -1:
            caller.PEs[dst_PE].active_dma_transfer = transfer_dict
    update_noc_transfers(current_time)
    
def cleanup_noc_transfers(current_time, caller):
    '''!
    Remove completed transfers, free DMA channels, and promote queued transfers to active.

    @param current_time: Current simulation time
    @param caller: SimulationManager instance (required for DMA management)
    '''
    before = len(active_noc_transfers)

    # Remove completed transfers and free DMA channels
    completed_transfers = []
    for t in active_noc_transfers:
        if t['finish_time'] <= current_time and t['state']== 'active':
            completed_transfers.append(t)

            # Free DMA channels
            for i in range(len(t['src_PEs'])):
                if t['src_PEs'][i] != -1:
                    caller.PEs[t['src_PEs'][i]].active_dma_transfer = None
            if t['dst_PE'] != -1:
                caller.PEs[t['dst_PE']].active_dma_transfer = None
    
    # Remove completed from list
    active_noc_transfers[:] = [
        t for t in active_noc_transfers
        if t not in completed_transfers
    ]

    # Check for newly unblocked transfers (queued → active)
    recalc_needed = False
    for t in active_noc_transfers:
        if t['state'] == 'queued':
            # Check if transfer can now start
            current_seg = t['current_transfer']
            state = calculate_contention(
                caller, t['src_PEs'][current_seg],
                t['dst_PE'], t['data_IDs'][current_seg], current_time
            )
            if state == 'active':
                
                t['state'] = 'active'
                t['update_time'] = current_time

                # Set DMA ownership
                if t['src_PEs'][current_seg] != -1:
                    caller.PEs[t['src_PEs'][current_seg]].active_dma_transfer = t
                if t['dst_PE'] != -1:

                    caller.PEs[t['dst_PE']].active_dma_transfer = t

                recalc_needed = True

    # Only recalculate if something changed
    if len(active_noc_transfers) != before or recalc_needed:
        update_noc_transfers(current_time)

def update_noc_transfers(current_time):
    '''!
    Recalculate finish times for all active NoC transfers based on current congestion.
    Skips queued transfers (they don't contribute to congestion).

    @param current_time: Current simulation time
    '''
    c = get_congested_bandwidth()
    memory_factor = get_memory_contention_factor()

    # we calculate the amount of work done in hindsight, since the last update.
    # congested_bandwidth is used (previous c)
    for t in active_noc_transfers:
        # Skip queued transfers - they don't affect congestion yet
        if t['state'] == 'queued':
            t['finish_time'] = float('inf') # time_stamp is initialized at 0!
            if t['task'] != None:
                t['task'].time_stamp = t['finish_time']
            continue

        assert t['current_transfer'] <= t['total_transfers']
        elapsed = current_time - t['update_time']

        while elapsed > 0:
            idx = t['current_transfer']
            effective_bw = min(t['max_bandwidths'][idx], t['congested_bandwidths'][idx])

            time_for_segment = t['volume'][idx] / effective_bw

            if elapsed >= time_for_segment:
                elapsed -= time_for_segment
                t['volume'][idx] = 0
                t['current_transfer'] += 1
                # cleanup_noc_transfers is always run before increase_congestion. It isn't possible to have a job that has met its deadline be in update_noc_transfers
                assert t['current_transfer'] < t['total_transfers']

            else:
                t['volume'][idx] -= elapsed * effective_bw
                elapsed = 0
        # end while elapsed > 0

        # Calculate NoC transfer time
        total_time = 0
        for idx in range(t['current_transfer'], t['total_transfers']):
            # Calculate memory overhead time (parallel, not additive)
            if t['src_PEs'][idx] == -1 or t['dst_PE'] == -1:
                # calculate the time if the noc is the limiting factor and if the memory is the limiting factor
                seg_bw = min(t['max_bandwidths'][idx], c)
                noc_time = t['volume'][idx] / seg_bw
                memory_bw = t['max_bandwidths'][idx] * (1/memory_factor) # 1 / memory factor here is the bandwidth we can expect to recieve based on latency slowdown
                memory_time = t['volume'][idx] / memory_bw # the same as multiplying it into the memory time.

                total_time += max(memory_time, noc_time)
                if noc_time > memory_time:
                    t['congested_bandwidths'][idx] = int(seg_bw) # min of this is the same as max in total_time
                else:
                    t['congested_bandwidths'][idx] = int(memory_bw) # min of this is the same as max in total_time
            else:
                seg_bw = min(t['max_bandwidths'][idx], c)
                total_time += t['volume'][idx] / seg_bw
                t['congested_bandwidths'][idx] = seg_bw


        t['finish_time'] = int(current_time + total_time)

        

        t['update_time'] = int(current_time)

        if t['task'] != None:
            t['task'].time_stamp = t['finish_time']


def get_memory_contention_factor():
    """
    Calculate memory latency scaling factor based on bandwidth utilization.
    Uses queueing-theory model from Mess paper findings.
    
    @return: Multiplicative factor >= 1.0
             (1.0 = no contention, higher = more latency)
    """
    peak_bw = 12800  # bytes/us for LPDDR5-6400 (12.8 GB/s)
    
    # Calculate current memory bandwidth usage and read/write ratio
    # Only count ACTIVE transfers that touch memory (src or dst is -1)
    total_memory_bw = 0
    read_bw = 0
    write_bw = 0
    count = 0
    
    for t in active_noc_transfers:
        if t['state'] != 'active':
            continue
            
        idx = t['current_transfer']
        if idx >= t['total_transfers']:
            assert False
            
        # Get effective bandwidth for this transfer
        effective_bw = min(t['max_bandwidths'][idx], t['congested_bandwidths'][idx])
        
        src_pe = t['src_PEs'][idx]
        dst_pe = t['dst_PE']
        
        # Check if this transfer involves memory
        if src_pe == -1:
            # Reading from memory to accelerator
            read_bw += effective_bw
            total_memory_bw += effective_bw
            count += 1
        elif dst_pe == -1:
            # Writing from accelerator to memory
            write_bw += effective_bw
            total_memory_bw += effective_bw
            count += 1
        # else: SPAD-to-SPAD transfer, doesn't affect memory contention


    utilization = total_memory_bw / peak_bw

    if count == 1:
        return utilization # no contention - use the entire memory bandwidth


    if total_memory_bw > 0:
        read_ratio = (read_bw / total_memory_bw) 
    else:
        read_ratio = 1.0  # Default to reads when no traffic

    
    if utilization <= 0:
        return 1.0
    
    # Saturation point depends on read/write ratio
    # 100% reads: saturates at ~85-90% of peak
    # 50% reads: saturates at ~65-70% of peak
    # Since we don't need to read for each write, we can achieve 0% reads. Best we can do is continue the trend of saturation
    sat_point = 0.60 + 0.30 * read_ratio
    
    # Knee point: where latency starts increasing noticeably
    # Paper shows latency is ~flat until ~70-80% of saturation point
    knee_point = sat_point * 0.75
    
    if utilization < knee_point:
        # Flat region: minimal latency increase
        return 1.0
    
    if utilization > sat_point:
        utilization = 1.0
    
    # Beyond knee: sharp increase toward saturation
    # Map utilization from [knee, sat] to [0, 1] for the steep region
    steep_region = (utilization - knee_point) / (sat_point - knee_point)
    
    # Steep curve in saturation region
    # At knee: factor = 1.0
    # At saturation: factor approaches max_factor
    # We shouldn't be hitting this at all, if ever
    max_factor = 3.0  # Paper shows ~2-4x latency increase at saturation
    
    factor = 1.0 + (max_factor) * (steep_region ** 2) # approximately quadratic increase
    
    return factor


def calculate_contention(caller, src_PE, dst_PE, data_id, current_time):
    '''!
    Determines earliest start time for a transfer considering:
    1. DMA availability on source and destination PEs (single-channel)
    2. Writeback conflicts (PE-to-PE transfers must wait for writebacks)

    @param caller: SimulationManager instance (for accessing PEs)
    @param src_PE: Source PE ID (-1 for memory)
    @param dst_PE: Destination PE ID
    @param data_id: Data identifier for writeback conflict checking
    @param current_time: Current simulation time
    @return: Tuple (state, earliest_start_time) where state is 'queued' or 'active'
    '''
    # Check DMA availability for source PE (if not memory)
    if ((src_PE != -1 and caller.PEs[src_PE].active_dma_transfer != None) or
        (dst_PE != -1 and caller.PEs[dst_PE].active_dma_transfer != None)):
        return 'queued'

    return 'active'



            
def decide_comm_timing(caller, task:Tasks, predecessor_task, predecessor_PE_ID):
    '''!
    Decide whether to use PE_to_PE or memory timing for communication.
    This method supports three modes:
    - PE_to_PE (legacy): Always use direct PE-to-PE timings
    - shared_memory (legacy): Always use memory timings
    - forwarding: Dynamic decision based on data location and forwarding feasibility

    @param task: The task that needs data
    @param predecessor_task: The task that produced the data
    @param target_PE_ID: The PE ID where the task will execute
    @return: 'PE_to_PE' or 'memory' indicating which timing to use
    '''
    # Legacy modes - fixed decision
    if comm_mode == 'PE_to_PE':
        return 'PE_to_PE'
    elif comm_mode == 'shared_memory':
        return 'memory'

    # Forwarding mode - dynamic decision based on whether task is being forwarded to idle PE
    elif comm_mode == 'forwarding':
        predecessor_PE = caller.PEs[predecessor_PE_ID]

        # If task is marked as forwarded (by scheduler) and target PE was idle when scheduled, use PE_to_PE timing
        # Otherwise use memory timing
        if predecessor_PE.has_data_in_scratchpad(f"{predecessor_task.ID}_output"):
            # If the predecessor has the data in its scratchpad, we forward.
            return 'PE_to_PE'
        else:
            # Task is not being forwarded - use memory timing
            return 'memory'

    # Default fallback (should not reach here)
    return 'memory'
#end def decide_comm_timing


# =============================================================================
# def clear_screen():
#     '''
#     Define the clear_screen function to
#     clear the screen before the simulation.
#     '''
#     current_platform = platform.system()        # Find the platform
#     if 'windows' in current_platform.lower():
#         get_ipython().magic('clear')
#     elif 'Darwin' in current_platform.lower():
#         get_ipython().magic('clear')
#     elif 'linux' in current_platform.lower():
#         get_ipython().magic('clear')  
# # end of def clear_screen()
# =============================================================================

