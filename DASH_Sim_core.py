'''!
@brief This file contains the simulation core that handles the simulation events.
'''
import sys
import numpy as np

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
import DTPM
import DTPM_policies
from scheduler import Scheduler
import RELIEF_Sim_helpers
from typing import List

# Define the core of the simulation engine
# This function calls the scheduler, starts/interrupts the tasks,
# and manages collection of all the statistics

class SimulationManager:
    '''!
    Define the SimulationManager class to handle the simulation events.
    '''
    def __init__(self, env, sim_done, job_gen, scheduler:Scheduler, PE_list, jobs, resource_matrix):
        '''!
        @param env: Pointer to the current simulation environment
        @param sim_done: Simpy event object to indicate whether the simulation must be finished
        @param job_gen: JobGenerator object
        @param scheduler: Pointer to the DASH_scheduler
        @param PE_list: The PEs available in the current SoC
        @param jobs: The list of all jobs given to DASH-Sim
        @param resource_matrix: The data structure that defines power/performance characteristics of the PEs for each supported task
        '''
        self.env = env
        self.sim_done = sim_done
        self.job_gen = job_gen
        self.scheduler = scheduler
        self.PEs = PE_list
        self.jobs = jobs
        self.resource_matrix = resource_matrix

        self.action = env.process(self.run())  # starts the run() method as a SimPy process


    # I want to delete update_ready_queue and update_execution_queue
    def update_ready_queue(self,completed_task):
        '''!
        This function updates the common.TaskQueues.ready after one task is completed.

        As the simulation proceeds, tasks are being processed.
        We need to update the ready tasks queue after completion of each task.

        @param completed_task: Object for the task that just completed execution
        '''

        # completed_task is the task whose processing is just completed
        # Add completed task to the completed tasks queue
        common.completed.append(completed_task)

        # Remove the completed task from the queue of the PE
        for task in self.PEs[completed_task.PE_ID].queue:
            if task.ID == completed_task.ID:
                self.PEs[task.PE_ID].queue.remove(task)

        # unlock the PE:
        PE = self.PEs[completed_task.PE_ID]
        PE.lock = False

        # Remove the completed task from the currently running queue
        common.running.remove(completed_task)

        # Remove the completed task from the current DAG representation
        if completed_task.ID in common.current_dag:
            common.current_dag.remove_node(completed_task.ID)
        
        # Initialize $remove_from_outstanding_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_outstanding_queue = []

        # Initialize $to_memory_comm_time which will be communication time to
        # memory for data from a predecessor task to a outstanding task
        to_memory_comm_time = -1
        
        job_ID = -1
        for ind, job in enumerate(self.jobs.list):
            if job.name == completed_task.jobname:
                job_ID = ind


        # Check if the dependency of any outstanding task is cleared
        # We need to move them to the ready queue
        
        for i, outstanding_task in enumerate(common.outstanding):                       # Go over each outstanding task
            if (completed_task.ID in outstanding_task.predecessors):                                    # if the completed task is one of the predecessors
                outstanding_task.predecessors.remove(completed_task.ID)                                 # Clear this predecessor
                PE.scratchpad[f"{completed_task.ID}_output"]['dependencies'].append(outstanding_task)   # the output of the predecessor will save back to memory when ejected if the outstanding task has not finished computing

                if (common.shared_memory):
                    # Get the communication time to memory for data from a
                    # predecessor task to a outstanding task
                    comm_vol = self.jobs.list[job_ID].comm_vol[completed_task.base_ID , outstanding_task.base_ID]
                    comm_band = common.ResourceManager.comm_band[completed_task.PE_ID, self.resource_matrix.list[-1].ID]
                    to_memory_comm_time = int(comm_vol/comm_band)                                           # Communication time from a PE to memory

                    if (common.DEBUG_SIM):
                        print('[D] Time %d: Data from task %d for task %d will be sent to memory in %d us'
                              %(self.env.now, completed_task.ID, outstanding_task.ID, to_memory_comm_time))

                    # Based on this communication time, this outstanding task
                    # will be added to the ready queue. That is why, keep track of
                    # all communication times required for a task in the list
                    # $ready_wait_times
                    outstanding_task.ready_wait_times.append(to_memory_comm_time + self.env.now)
                # end of if (common.shared_memory):
            # end of if (completed_task.ID in outstanding_task.predecessors):

            no_predecessors = (len(outstanding_task.predecessors) == 0)                            # Check if this was the last dependency
            currently_running = (outstanding_task in                                               # if the task is in the running queue,
                                 common.running)                                   # We should not put it back to the ready queue
            not_in_ready_queue = not(outstanding_task in                                           # If this task is already in the ready queue,
                                  common.ready)                                    # We should not append another copy

            if (no_predecessors and not(currently_running) and not_in_ready_queue):
                if (common.forwarding_enabled):
                    common.ready.append(common.outstanding[i]) # if we're forwarding - writing back to memory is done upon ejection from a scratchpad, not when the task stops running
                elif (common.PE_to_PE):                                                              # if PE to PE communication is utilized
                    common.ready.append(common.outstanding[i])     # Add the task to the ready queue immediately

                elif (common.shared_memory):
                    # if shared memory is utilized for communication, then
                    # the outstanding task will wait for a certain amount time
                    # (till the $time_stamp)for being added into the ready queue
                    common.wait_ready.append(outstanding_task)
                    if (common.INFO_SIM) and (common.shared_memory):
                            print('[I] Time %d: Task %d ready times due to memory communication of its predecessors are'
                                  %(self.env.now, outstanding_task.ID))
                            print('%12s'%(''), outstanding_task.ready_wait_times)
                    common.wait_ready[-1].time_stamp = max(outstanding_task.ready_wait_times)

                remove_from_outstanding_queue.append(outstanding_task)
        # end of for i, outstanding_task in...

        # Remove the tasks from outstanding queue that have been moved to ready queue
        for task in remove_from_outstanding_queue:
            common.outstanding.remove(task)

        print([obj.ID for obj in common.outstanding])
        # At the end of this function:
            # Newly processed $completed_task is added to the completed tasks
            # outstanding tasks with no dependencies are added to the ready queue
            # based on the communication mode and then, they are removed from
            # the outstanding queue
    #end def update_ready_queue(completed_task)


    def update_execution_queue(self, ready_list):
        '''!
        This function updates the common.TaskQueues.executable if one task is ready
        for execution but waiting for the communication time, either between
        memory and a PE, or between two PEs (based on the communication mode)

        @param ready_list: List of tasks that are ready to be executed
        '''
        # Initialize $remove_from_ready_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_ready_queue = []
        
        # Initialize $from_memory_comm_time which will be communication time 
        # for data from memory to a PE
        from_memory_comm_time = -1

        # Initialize $PE_to_PE_comm_time which will be communication time
        # for data from a PE to another PE
        PE_to_PE_comm_time = -1

        job_ID = -1
        for ready_task in ready_list:
            # If other communication modes are used (PE_to_PE or shared_memory)
            for ind, job in enumerate(self.jobs.list):
                if job.name == ready_task.jobname:
                    job_ID = ind

            for i, task in enumerate(self.jobs.list[job_ID].task_list):
                if ready_task.base_ID == task.ID:
                    if ready_task.head == True:
                        print('task %d is a head task' %(ready_task.ID))
                        # if a task is the leading task of a job
                        # then it can start immediately since it has no predecessor
                        ready_task.PE_to_PE_wait_time.append(self.env.now)
                        ready_task.execution_wait_times.append(self.env.now)
                        continue
                    # end of if ready_task.head == True:

                    print('task %d num predecessors %d' %(ready_task.ID, len(task.predecessors)))

                    for predecessor in task.predecessors:

                        # data required from the predecessor for $ready_task
                        comm_vol = self.jobs.list[job_ID].comm_vol[predecessor, ready_task.base_ID]

                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + ready_task.ID - ready_task.base_ID

                        # Initialize following two variables which will be used if
                        # PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1
                        predecessor_task = None

                        # Find the predecessor task to get its PE and finish time
                        for completed in common.completed:
                            if completed.ID == real_predecessor_ID:
                                predecessor_PE_ID = completed.PE_ID
                                predecessor_finish_time = completed.finish_time
                                predecessor_task = completed
                                break

                        # Decide communication timing mode (PE_to_PE or memory)
                        if predecessor_task is not None:
                            comm_timing = common.decide_comm_timing(ready_task, predecessor_task, predecessor_task.PE_ID)
                            ready_task.comm_timing_mode = comm_timing
                        else:
                            assert False

                        if comm_timing == 'PE_to_PE' or common.PE_to_PE:
                            # Use PE-to-PE communication timing (includes forwarding in forwarding mode)
                            comm_band = common.ResourceManager.comm_band[predecessor_PE_ID, ready_task.PE_ID]
                            PE_to_PE_comm_time = int(comm_vol/comm_band)
                            ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)

                            if (common.DEBUG_SIM):
                                mode_str = "forwarding" if common.comm_mode == 'forwarding' else "PE-to-PE"
                                print('[D] Time %d: Data transfer (%s) from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                      %(self.env.now, mode_str, predecessor_PE_ID, ready_task.PE_ID,
                                        ready_task.ID, real_predecessor_ID, ready_task.PE_to_PE_wait_time[-1]))

                            # In forwarding mode, allocate scratchpad space for the data
                            if common.comm_mode == 'forwarding' and comm_timing == 'PE_to_PE':
                                assert False # should never be here
                        # end of if comm_timing == 'PE_to_PE':

                        if comm_timing == 'memory' or common.shared_memory:
                            # Use memory communication timing
                            comm_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, ready_task.PE_ID]
                            from_memory_comm_time = int(comm_vol/comm_band)
                            if (common.DEBUG_SIM):
                                print('[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                                      %(self.env.now, ready_task.ID, real_predecessor_ID, ready_task.PE_ID, from_memory_comm_time))
                            ready_task.execution_wait_times.append(from_memory_comm_time + self.env.now)
                        # end of if comm_timing == 'memory'
                    # end of for predecessor in task.predecessors:

                    if (common.INFO_SIM) and (common.PE_to_PE):
                        print('[I] Time %d: Task %d execution ready times due to communication between PEs are'
                              %(self.env.now, ready_task.ID))
                        print('%12s'%(''), ready_task.PE_to_PE_wait_time)

                    if (common.INFO_SIM) and (common.shared_memory):
                        print('[I] Time %d: Task %d execution ready time(s) due to communication between memory and PE-%s are'
                              %(self.env.now, ready_task.ID, ready_task.PE_ID))
                        print('%12s'%(''), ready_task.execution_wait_times)

                    # Populate all ready tasks in executable with a time stamp
                    # which will show when a task is ready for execution

                    # Set the time stamp for when the task is ready for execution
                    if (common.PE_to_PE or (common.comm_mode == 'forwarding' and comm_timing == 'PE_to_PE')):
                        ready_task.time_stamp = max(ready_task.PE_to_PE_wait_time)
                    else:
                        ready_task.time_stamp = max(ready_task.execution_wait_times)

                    # Add to per-PE queue in common.executable dictionary
                    if ready_task.PE_ID != -1 and ready_task.PE_ID in common.executable:
                        common.executable[ready_task.PE_ID].append(ready_task)

                    remove_from_ready_queue.append(ready_task)
                # end of ready_task.base_ID == task.ID:
            # end of i, task in enumerate(self.jobs.list[job_ID].task_list):    
        # end of for ready_task in ready_list:
        
        # Remove the tasks from ready queue that have been moved to executable queue
        for task in remove_from_ready_queue:
            common.ready.remove(task)

        # Sort each PE's executable queue by job ID
        for pe_id in common.executable:
            common.executable[pe_id].sort(key=lambda task: task.jobID, reverse=False)

        
    def update_completed_queue(self):
        '''!
        This function updates the common.TaskQueues.completed 
        '''  
        ## Be careful about this function when there are diff jobs in the system
        # reorder tasks based on their job IDs
        common.completed.sort(key=lambda x: x.jobID, reverse=False)
        
        first_task_jobID =  common.completed[0].jobID
        last_task_jobID = common.completed[-1].jobID
        
        if ((last_task_jobID - first_task_jobID) > 15):
            for i,task in enumerate(common.completed):
                if (task.jobID == first_task_jobID):
                    del common.completed[i]
    
    # PEs call this to write values ejected from their scratchpads back to memory
    def writeback_handler(self,data_id, size, PE_ID):
        comm_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, PE_ID]
        memory_comm_time = int((size/comm_band) * common.get_congestion_factor(self))
        common.memory_writeback[data_id] = self.env.now + memory_comm_time

        common.active_noc_transfers.append({
            'end_time': self.env.now + memory_comm_time,
            'src_PE': PE_ID,
            'dst_PE': -1,  # memory
            'task_ID': -1
        })
        
    #
    def run(self):
        '''!
        Implement the basic run method that will be called periodically in each simulation "tick".

        This function takes the next ready tasks and run on the specific PE and update the common.TaskQueues.ready list accordingly.
        '''
        DTPM_module = DTPM.DTPMmodule(self.env, self.resource_matrix, self.PEs)

        for cluster in common.ClusterManager.cluster_list:
            DTPM_policies.initialize_frequency(cluster)

        while (True):                                                           # Continue till the end of the simulation

            if self.env.now % common.sampling_rate == 0:
                #common.results.job_counter_list.append(common.results.job_counter)
                #common.results.sampling_rate_list.append(self.env.now)
                # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                DTPM_module.evaluate_idle_PEs()
            # end of if self.env.now % common.sampling_rate == 0:

            if (common.forwarding_enabled):
                remove_from_writeback = []
                for (identifier, finishTime) in common.memory_writeback.items():
                    if finishTime <= self.env.now:
                        remove_from_writeback.append(identifier)
                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data transfer for data %s to memory is completed'
                                %(self.env.now, identifier))
                for id in remove_from_writeback:
                    common.memory_writeback.pop(id)
            

            if (common.shared_memory):
                # this section is activated only if shared memory is used

                # Initialize $remove_from_wait_ready which will populate tasks
                # to be removed from the wait ready queue
                remove_from_wait_ready = []

                for waiting_task in common.wait_ready:
                    if waiting_task.time_stamp <= self.env.now:
                        common.ready.append(waiting_task)
                        remove_from_wait_ready.append(waiting_task)
                # at the end of this loop, all the waiting tasks with a time stamp
                # equal or smaller than the simulation time will be added to
                # the ready queue list
                #end of for i, waiting_task in...

                # Remove the tasks from wait ready queue that have been moved to ready queue
                for task in remove_from_wait_ready:
                    common.wait_ready.remove(task)
            # end of if (common.shared_memory):

            if (common.INFO_SIM) and len(common.ready) > 0:
                print('[I] Time %s: DASH-Sim ticks with %d task ready for being assigned to a PE'
                      % (self.env.now, len(common.ready)))

            if (not len(common.ready) == 0):
                # give all tasks in ready_list to the chosen scheduler
                # and scheduler will assign the tasks to a PE
                if self.scheduler.name == 'CPU_only':
                    self.scheduler.CPU_only(common.ready)
                elif self.scheduler.name == 'MET':
                    self.scheduler.MET(common.ready)
                elif self.scheduler.name == 'EFT':
                    self.scheduler.EFT(common.ready)
                elif self.scheduler.name == 'STF':
                    self.scheduler.STF(common.ready)
                elif self.scheduler.name == 'ETF':
                    self.scheduler.ETF(common.ready)
                elif self.scheduler.name == 'ETF_LB':
                    self.scheduler.ETF_LB(common.ready)
                elif self.scheduler.name == 'CP':
                    self.scheduler.CP(common.ready)
                elif self.scheduler.name == 'RELIEF_BASE':
                    self.scheduler.RELIEF_BASIC(common.ready)
                else:
                    print('[E] Could not find the requested scheduler')
                    print('[E] Please check "config_file.ini" and enter a proper name')
                    print('[E] or check "scheduler.py" if the scheduler exist')
                    sys.exit()
                # end of if self.scheduler.name
                if (self.scheduler.name != 'RELIEF_BASE'):
                    self.update_execution_queue(common.ready)
            # end of if not len(common.ready) == 0:

            # Initialize $remove_from_executable which will populate tasks
            # to be removed from the executable queue
            remove_from_executable = {}  # {PE_ID: [tasks_to_remove]}

            if (self.scheduler.name == 'RELIEF_BASE'):
                for pe_id, pe_queue in common.executable.items():

                    P = self.PEs[pe_id]
                    if len(pe_queue) == 0:
                        continue

                    if self.env.now >= common.warmup_period:
                        if not P.idle:
                            ready_tasks_count = sum(1 for task in pe_queue if task.time_stamp <= self.env.now)
                            if ready_tasks_count > 0:
                                P.blocking += 1
                    
                    # In RELIEF mode, PEs have three states
                    # 1) P.idle = True, P.lock = False: There is nothing going on for this PE, it is fully idle. 
                    #    This is the only state in which scheduling is allowed on the PE
                    # 2) P.idle = True, P.lock = True: The PE is locked because data is being transferred in. 
                    #    Need to check if the first task in its queue has had its data transferred so we can begin execution
                    # 3) P.idle = False, P.lock = True: The PE is running.
                    # For simplicity, each PE can ONLY operate on a single input at a time right now
                    tasks_to_remove = []
                    executable_task = pe_queue[0]
                    if P.idle and P.lock:
                        dynamic_dependencies_met = True

                        dependencies_completed = []
                        for dynamic_dependency in executable_task.dynamic_dependencies:
                            dependencies_completed = dependencies_completed + list(filter(lambda completed_task: completed_task.ID == dynamic_dependency, common.completed))

                        if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                            dynamic_dependencies_met = False

                        task_has_assignment = (executable_task.PE_ID == pe_id)  # Should always be true by design

                        if (executable_task.time_stamp <= self.env.now and dynamic_dependencies_met and task_has_assignment): # if it's time to execute
                            P.queue.append(executable_task)

                            if (common.INFO_SIM):
                                print('[I] Time %s: Task %s is ready for execution by PE-%s'
                                    % (self.env.now, executable_task.ID, pe_id))

                            current_resource = self.resource_matrix.list[pe_id]
                            self.env.process(P.run(  # Send the current task and a handle for this simulation manager (self)
                                self, executable_task, current_resource, DTPM_module))  # This handle is used by the PE to call the update_ready_queue function

                            remove_from_executable[pe_id] = [executable_task]
                        # end if (executable_task.time_stamp <= self.env.now and dynamic_dependencies_met and task_has_assignment):
                        
                    # end if P.idle and P.lock:
                    elif P.idle and not P.lock:
                        # lock PE, begin pulling value from input
                        P.lock = True

                        executable_task.time_stamp = common.calculate_memory_movement_latency(self,executable_task,executable_task.PE_ID,True)
                    # end elif P.idle and not P.lock:
                    elif not P.idle and P.lock: 
                        continue # PE is running when locked and not idle
                    else:
                        assert(False) #should be impossible
            # end of if (self.scheduler.name == 'RELIEF_BASE'):
            else:   
                # Go over each PE's queue from common.executable dictionary
                for pe_id, pe_queue in common.executable.items():
                    if len(pe_queue) == 0:
                        continue

                    PE = self.PEs[pe_id]

                    # for PE blocking data collection
                    if self.env.now >= common.warmup_period:
                        if not PE.idle:
                            ready_tasks_count = sum(1 for task in pe_queue if task.time_stamp <= self.env.now)
                            if ready_tasks_count > 0:
                                PE.blocking += 1

                    # Process tasks in this PE's queue
                    tasks_to_remove = []
                    for executable_task in pe_queue:
                        is_time_to_execute = (executable_task.time_stamp <= self.env.now)
                        PE_has_capacity = (len(PE.queue) < PE.capacity)  # capacity is the number of jobs a PE can have waiting?
                        task_has_assignment = (executable_task.PE_ID == pe_id)  # Should always be true by design

                        dynamic_dependencies_met = True

                        dependencies_completed = []
                        for dynamic_dependency in executable_task.dynamic_dependencies:
                            dependencies_completed = dependencies_completed + list(filter(lambda completed_task: completed_task.ID == dynamic_dependency, common.completed))
                        if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                            dynamic_dependencies_met = False

                        if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met and task_has_assignment:
                            PE.queue.append(executable_task)

                            if (common.INFO_SIM):
                                print('[I] Time %s: Task %s is ready for execution by PE-%s'
                                    % (self.env.now, executable_task.ID, pe_id))

                            current_resource = self.resource_matrix.list[pe_id]
                            self.env.process(PE.run(  # Send the current task and a handle for this simulation manager (self)
                                self, executable_task, current_resource, DTPM_module))  # This handle is used by the PE to call the update_ready_queue function

                            tasks_to_remove.append(executable_task)
                        # end of if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met

                    # Remove executed tasks from this PE's queue
                    if tasks_to_remove:
                        remove_from_executable[pe_id] = tasks_to_remove
                # end of for pe_id, pe_queue in common.executable.items():

            # Remove the tasks from executable queue that have been executed by a resource
            for pe_id, tasks in remove_from_executable.items():
                for task in tasks:
                    common.executable[pe_id].remove(task)

            # If DRL scheduler is active, tasks waiting in the executable queue will be redirected to the ready queue
            if (self.scheduler.name == 'DRL'):
                # Pop tasks from all PE queues and move to ready
                for pe_id in list(common.executable.keys()):
                    while len(common.executable[pe_id]) > 0:
                        task = common.executable[pe_id].pop(-1)
                        common.ready.append(task)
                        
            # The simulation tick is completed. Wait till the next interval
            yield self.env.timeout(common.simulation_clk)

            if self.env.now > common.simulation_length and common.inject_fixed_num_jobs is False:
                self.sim_done.succeed()
        #end while (True)
