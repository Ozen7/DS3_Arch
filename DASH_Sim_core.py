'''!
@brief This file contains the simulation core that handles the simulation events.
'''
import sys
import numpy as np

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
import DTPM
import DTPM_policies
from scheduler import Scheduler
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
        PE.task = None # reset PE ID
        PE.dependencies = [] #reset dependencies
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
                PE.scratchpad[f"{completed_task.ID}_output"]['dependencies'].append(outstanding_task)   # Dependencies for scratchpad values are handled here

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

        #schedule
        if (self.scheduler.name in common.new_schedulers):
            #runs every cycle
            common.cleanup_noc_transfers(self.env.now, caller=self)

        if (not len(common.ready) == 0):
            # give all tasks in ready_list to the chosen scheduler
            # and scheduler will assign the tasks to a PE
            if self.scheduler.name == 'RELIEF':
                self.scheduler.RELIEF(common.ready)
            elif self.scheduler.name == 'LL':
                self.scheduler.LL(common.ready)
            elif self.scheduler.name == 'GEDF_D':
                self.scheduler.GEDF_D(common.ready)
            elif self.scheduler.name == 'GEDF_N':
                self.scheduler.GEDF_N(common.ready)
            elif self.scheduler.name == 'HetSched':
                self.scheduler.HetSched(common.ready)
            elif self.scheduler.name == 'FCFS':
                self.scheduler.FCFS(common.ready)
            else:
                print('[E] Could not find the requested scheduler')
                print('[E] Please check "config_file.ini" and enter a proper name')
                print('[E] or check "scheduler.py" if the scheduler exist')
                sys.exit()
        # end of if not len(common.ready) == 0:

        # Remove the tasks from outstanding queue that have been moved to ready queue
        for task in remove_from_outstanding_queue:
            common.outstanding.remove(task)

        #print([obj.ID for obj in common.outstanding])
        # At the end of this function:
            # Newly processed $completed_task is added to the completed tasks
            # outstanding tasks with no dependencies are added to the ready queue
            # based on the communication mode and then, they are removed from
            # the outstanding queue
    #end def update_ready_queue(completed_task)

        
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
    def writeback_handler(self,data_id, size, PE):
        comm_band = int(common.ResourceManager.comm_band[PE.ID, self.resource_matrix.list[-1].ID])
      
        common.increase_congestion(self.env.now, [size], [PE.ID], -1, [data_id], [comm_band], self)
    #
    def run(self):
        '''!
        Implement the basic run method that will be called periodically in each simulation "tick".

        This function takes the next ready tasks and run on the specific PE and update the common.TaskQueues.ready list accordingly.
        '''
        DTPM_module = DTPM.DTPMmodule(self.env, self.resource_matrix, self.PEs)

        for cluster in common.ClusterManager.cluster_list:
            DTPM_policies.initialize_frequency(cluster)

        while (True):  
            if self.env.now % common.sampling_rate == 0:
                #common.results.job_counter_list.append(common.results.job_counter)
                #common.results.sampling_rate_list.append(self.env.now)
                # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                DTPM_module.evaluate_idle_PEs()
            # end of if self.env.now % common.sampling_rate == 0:

            
            if (self.scheduler.name in common.new_schedulers):
                #runs every cycle
                common.cleanup_noc_transfers(self.env.now, caller=self)

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

            # Initialize $remove_from_executable which will populate tasks
            # to be removed from the executable queue
            remove_from_executable = {}  # {PE_ID: [tasks_to_remove]}

            for family in common.TypeManager.get_all_families():
                for pe_id in common.TypeManager.get_PEs_of_family(family):
                    P = self.PEs[pe_id]
                    pe_queue = common.executable[family]
                    if len(pe_queue) == 0 and not P.lock:
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

                    if P.allocating:
                        i = 0
                        
                        executable_task = P.task
                        out = P.allocate_scratchpad(f"{executable_task.ID}_output",executable_task.output_packet_size*common.packet_size,executable_task.ID)                 # allocate room in the scratchpad for the output of this task.
                        out2 = False
                        if out == True:
                            out2 = common.calculate_memory_movement_latency(self,executable_task,executable_task.PE_ID,True)

                            
                        if out == True and out2 == True:
                            P.allocating = False
                        continue

                    executable_task = None
                    if P.task == None:
                        i = 0
                        #find index position of first non negative task for certain scheduling algorithms
                        if self.scheduler.name in common.deprioritize_negative_laxity: 
                            while i < len(pe_queue)-1 and pe_queue[i].laxity < 0 and pe_queue[i].isForwarded == False:
                                i += 1

                        executable_task = pe_queue[i]
                    else:
                        executable_task = P.task
                    assert executable_task != None

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

                                # immediately allocate room in the scratchpad for output - done so that any ejection of data from the scratchpad is
                                # taken into consideration when calculating memory latency

                        # end if (executable_task.time_stamp <= self.env.now and dynamic_dependencies_met and task_has_assignment):
                        
                    # end if P.idle and P.lock:
                    elif P.idle and not P.lock:
                        # lock PE, begin pulling value from input
                        P.lock = True
                        # this function also sets the task's timestamp
                        P.task = executable_task
                        executable_task.PE_ID = P.ID
                        # remove it from the executable queue
                        common.executable[family].remove(executable_task)
                        
                        out = P.allocate_scratchpad(f"{executable_task.ID}_output",executable_task.output_packet_size*common.packet_size,executable_task.ID)                 # allocate room in the scratchpad for the output of this task.
                        out2 = False
                        if out == True:
                            out2 = common.calculate_memory_movement_latency(self,executable_task,executable_task.PE_ID,True)

                            
                        if out == False or out2 == False:
                            P.allocating = True
                            continue



                    # end elif P.idle and not P.lock:
                    elif not P.idle and P.lock: 
                        continue # PE is running when locked and not idle
                    else:
                        assert(False) #should be impossible
                # end for pe_id in common.TypeManager.get_PEs_of_family(family):
            # end for family in common.TypeManager.get_all_types():
                        
            # The simulation tick is completed. Wait till the next interval
            yield self.env.timeout(common.simulation_clk)

            if self.env.now > common.simulation_length and common.inject_fixed_num_jobs is False:
                self.sim_done.succeed()
        #end while (True)
