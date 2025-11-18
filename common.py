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
        self.job_counter_list = []
        self.sampling_rate_list = []
# end class PerfStatics

# Instantiate the object that will store the performance statistics
global results

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
# end class PETypeManager

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
        self.runtime = -1                       # Represents the runtime of the task - once a PE has been selected
        self.laxity = -1                        # Indicates the Laxity of the Task, once a PE has been selected 
        self.head = False                       # If head is true, this task is the leading (the first) element in a task graph
        self.tail = False                       # If tail is true, this task is the end (the last) element in a task graph
        self.jobID = -1                         # This task belongs to job with this ID
        self.jobDeadline = -1                   # The deadline of the overarching Job - TODO implement in job_generator
        self.jobname = ''                       # This task belongs to job with this name
        self.base_ID = -1                       # This ID will be used to calculate the data volume from one task to another
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
memory_writeback = {}               # Dictionary of identifiers and timestamps for data being written back to memory
executable = {}                    # Dictionary of per-PE executable queues: {PE_ID: [task_list]} 

# no actual self, but we need access to the jobs and env variables from wherever it is being called from (scheduler or dash sim core)
def calculate_memory_movement_latency(caller, executable_task, PE_ID, canAllocate):

    # The rest of this is similar to what goes on in update_execution_queue, since we begin pulling data.
    # Get the Job Name
    for ind, job in enumerate(caller.jobs.list):
        if job.name == executable_task.jobname:
            job_ID = ind

    # Get the list of tasks 
    for task in caller.jobs.list[job_ID].task_list:
        if executable_task.base_ID == task.ID:
            
            wait_times = []

            if executable_task.head == True:
                if canAllocate and DEBUG_SIM:
                    print('task %d is a head task' %(executable_task.ID))
                # if a task is the leading task of a job
                # then it can start immediately since it has no predecessor
                wait_times.append(caller.env.now)
                wait_times.append(caller.env.now)
                
            # end of if ready_task.head == True:
            if canAllocate and DEBUG_SIM:
                print('task %d num predecessors %d' %(executable_task.ID, len(task.predecessors)))

            for predecessor in task.predecessors:

                # data required from the predecessor for $ready_task
                comm_vol = caller.jobs.list[job_ID].comm_vol[predecessor, executable_task.base_ID]

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

                # Decide communication timing mode (PE_to_PE or memory)
                if predecessor_task is not None:
                    comm_timing = decide_comm_timing(caller, executable_task, predecessor_task, predecessor_task.PE_ID, canAllocate)
                    executable_task.comm_timing_mode = comm_timing
                else:
                    # No predecessor found (shouldn't happen, but default to legacy behavior)
                    assert False

                if comm_timing == 'PE_to_PE' or PE_to_PE:
                    # Use PE-to-PE communication timing (includes forwarding in forwarding mode)
                    comm_band = ResourceManager.comm_band[predecessor_PE_ID, PE_ID]
                    PE_to_PE_comm_time = int(comm_vol/comm_band)
                    wait_times.append(PE_to_PE_comm_time + caller.env.now)

                    if (DEBUG_SIM) and canAllocate:
                        mode_str = "forwarding" if comm_mode == 'forwarding' else "PE-to-PE"
                        print('[D] Time %d: Data transfer (%s) from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                            %(caller.env.now, mode_str, predecessor_PE_ID, PE_ID,
                                executable_task.ID, real_predecessor_ID, wait_times[-1]))

                    # If we are colocating, there is no need to allocate more space (self-to-self bandwidth is set at a high number so there is no latency overhead either)
                    if canAllocate and predecessor_PE_ID == PE_ID:
                        target_PE = caller.PEs[executable_task.PE_ID]
                        if target_PE.forwarding_enabled:
                            data_id = f"{predecessor_task.ID}_output"
                            target_PE.allocate_scratchpad(data_id, comm_vol, predecessor_task.ID)
                # end of if comm_timing == 'PE_to_PE':

                if comm_timing == 'memory' or shared_memory:
                    # Use memory communication timing
                    comm_band = ResourceManager.comm_band[caller.resource_matrix.list[-1].ID, PE_ID]
                    from_memory_comm_time = int(comm_vol/comm_band)
                    if f"{predecessor_task.ID}_output" in memory_writeback:
                        from_memory_comm_time += memory_writeback[f"{predecessor_task.ID}_output"]
                    wait_times.append(from_memory_comm_time + caller.env.now)
                    if (DEBUG_SIM and canAllocate):
                        print('[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                            %(caller.env.now, executable_task.ID, real_predecessor_ID, PE_ID, wait_times[-1]))
                # end of if comm_timing == 'memory'
            # end of for predecessor in task.predecessors:


            if (INFO_SIM) and canAllocate:
                print('[I] Time %d: Task %d execution ready time(s) due to communication:'
                    %(caller.env.now, executable_task.ID))
                print('%12s'%(''), wait_times)

            # Populate all ready tasks in executable with a time stamp
            # which will show when a task is ready for execution

            # Set the time stamp for when the task is ready for execution
            if (wait_times):
                return max(wait_times)
            else:
                return 0
            # end else
        # end if executable_task.base_ID == task.ID:
    # end  for task in self.jobs.list[job_ID].task_list:

            
def decide_comm_timing(caller, task:Tasks, predecessor_task, predecessor_PE_ID, canAllocate):
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

