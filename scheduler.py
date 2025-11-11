'''!
@brief This file contains all schedulers in DASH-Sim

Scheduler class is defined in this file which contains different types of scheduler as a member function.
Developers can add thier own algorithms here and implement in DASH-Sim by add a function caller in DASH_Sim_core.py
'''
import networkx as nx
import numpy as np
import copy
import bisect

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
from common import Tasks, ApplicationManager, ResourceManager
from processing_element import PE
import DTPM_power_models
import DASH_Sim_core

import pickle
from typing import List
class Scheduler:
    '''!
    The Scheduler class constains all schedulers implemented in DASH-Sim
    '''
    def __init__(self, env, resource_matrix:ResourceManager, name:str, PE_list: List['PE'], jobs: ApplicationManager):
        '''!
        @param env: Pointer to the current simulation environment
        @param resource_matrix: The data structure that defines power/performance characteristics of the PEs for each supported task
        @param name : The name of the requested scheduler
        @param PE_list: The PEs available in the current SoCs
        @param jobs: The list of all jobs given to DASH-Sim
        '''
        self.env = env
        self.resource_matrix = resource_matrix
        self.name = name
        self.PEs = PE_list
        self.jobs = jobs
        self.assigned = [0] * (len(self.PEs))

        # At the end of this function, the scheduler class has a copy of the
        # the power/performance characteristics of the resource matrix and
        # name of the requested scheduler name


    # end  def __init__(self, env, resource_matrix, scheduler_name)

    def populate_execution_queue(self, ready_list):
        DASH_Sim_core.SimulationManager.update_execution_queue()
        pass

    # Specific scheduler instances can be defined below
    def CPU_only(self, list_of_ready):
        '''!
        This scheduler always select the resource with ID 0 (CPU) to execute all outstanding tasks without any comparison between
        available resources
        @param list_of_ready: The list of ready tasks
        '''
        for task in list_of_ready:
            task.PE_ID = 0

    # end def CPU_only(list_of_ready):


    def MET(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and returns the ID of the resource
        with minimum execution time for the current task.
        @param list_of_ready: The list of ready tasks
        '''
        # Initialize a list to record number of assigned tasks to a PE
        # for every scheduling instance
        assigned = [0]*(len(self.PEs))

        # go over all ready tasks for scheduling and make a decision
        for task in list_of_ready:

            exec_times = [np.inf]*(len(self.PEs))                                             # Initialize a list to keep execution times of task for each PE

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):

                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        exec_times[i] = self.resource_matrix.list[i].performance[ind]

            min_of_exec_times = min(exec_times)                                                 # $min_of_exec_times is the minimum of execution time of the task among all PEs
            count_minimum = exec_times.count(min_of_exec_times)                                 # also, record how many times $min_of_exec_times is seen in the list
            #print(count_minimum)

            # if there are two or more PEs satisfying minimum execution
            # then we should try to utilize all those PEs
            if (count_minimum > 1):

                # if there are tow or more PEs satisfying minimum execution
                # populate the IDs of those PEs into a list
                min_PE_IDs = [i for i, x in enumerate(exec_times) if x == min_of_exec_times]

                # then check whether those PEs are busy or idle
                PE_check_list = [True if not self.PEs[index].idle else False for i, index in enumerate(min_PE_IDs)]

                # assign tasks to the idle PEs instead of the ones that are currently busy
                if (True in PE_check_list) and (False in PE_check_list):
                    for PE in PE_check_list:
                        # if a PE is currently busy remove that PE from $min_PE_IDs list
                        # to schedule the task to a idle PE
                        if (PE == True):
                            min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])

                # then compare the number of the assigned tasks to remaining PEs
                # and choose the one with the lowest number of assigned tasks
                assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

# =============================================================================
#                 # assign tasks to the idle PEs instead of the ones that are currently busy
#                 if (True in PE_check_list) and (False in PE_check_list):
#                     for PE in PE_check_list:
#                         # if a PE is currently busy remove that PE from $min_PE_IDs list
#                         # to schedule the task to a idle PE
#                         if (PE == True):
#                             min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])
#
#
#                 # then compare the number of the assigned tasks to remaining PEs
#                 # and choose the one with the lowest number of assigned tasks
#                 assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
#                 PE_ID_index = assigned_tasks.index(min(assigned_tasks))
# =============================================================================


                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

            else:
                task.PE_ID = exec_times.index(min_of_exec_times)
            # end of if count_minimum >1:
            # since one task is just assigned to a PE, increase the number by 1
            assigned[task.PE_ID] += 1

            if (task.PE_ID == -1):
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,task.name))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print ('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                           %(self.env.now, task.ID, task.PE_ID,
                             self.resource_matrix.list[task.PE_ID].type))
            # end of if task.PE_ID == -1:
        # end of for task in list_of_ready:
        # At the end of this loop, we should have a valid (non-negative ID)
        # that can run next_task

    # end of MET(list_of_ready)

    def EFT(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the earliest finish time for the task
        @param list_of_ready: The list of ready tasks
        '''

        for task in list_of_ready:

            comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector
            comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE

            if (common.DEBUG_SCH):
                print ('[D] Time %s: The scheduler function is called with task %s'
                       %(self.env.now, task.ID))

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    # if the task is supported by the resource, retrieve the index of the task
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)


                        # $PE_comm_wait_times is a list to store the estimated communication time
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []

                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []

                        job_ID = -1                                                     # Initialize the job ID

                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii

                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID

                            # Initialize following two variables which will be used if
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1


                            for completed in common.completed:
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    #print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)

                                PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between PE %s to PE %s from task %s to task %s is %d'
                                          %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready

                                #PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol/memory_to_PE_band)

                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between memory to PE %s from task %s to task %s is %d'
                                          %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            # $comm_ready contains the estimated communication time
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource.
                            comm_ready[i] = (max(PE_comm_wait_times))
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))


                        # update the comparison vector accordingly
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i], PE_wait_time[-1])
                    # end of if (task.name in...
            # end of for i in range(len(self.resource_matrix.list)):

            # after going over each resource, choose the one which gives the minimum result
            task.PE_ID = comparison.index(min(comparison))

            if task.PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,task.ID))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                              %(self.env.now, task.ID))
                    print('%12s'%(''), comparison)
                    print ('[D] Time %s: The scheduler assigns task %s to resource %s: %s'
                           %(self.env.now, task.ID, task.PE_ID, self.resource_matrix.list[task.PE_ID].type))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[task.PE_ID].available_time = self.env.now + comparison[task.PE_ID]

            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of for task in list_of_ready:

    #end of EFT(list_of_ready)

    def STF(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and returns the ID of the resource
        with minimum execution time for the current task. The only difference between STF and MET is the order in which the tasks 
        are scheduled onto resources
        @param list_of_ready: The list of ready tasks
        '''

        ready_list = copy.deepcopy(list_of_ready)

        # Iterate through the list of ready tasks until all of them are scheduled
        while (len(ready_list) > 0) :

            shortest_task_exec_time = np.inf
            shortest_task_pe_id     = -1

            for task in ready_list:

                min_time = np.inf                                                                   # Initialize the best performance found so far as a large number

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)


                            if (self.resource_matrix.list[i].performance[ind] < min_time):              # Found resource with smaller execution time
                                min_time    = self.resource_matrix.list[i].performance[ind]                # Update the best time found so far
                                resource_id = self.resource_matrix.list[i].ID                           # Record the ID of the resource
                                #task.PE_ID = i                                                          # Record the corresponding resource

                #print('[INFO] Task - %d, Resource - %d, Time - %d' %(task.ID, resource_id, min_time))
                # Obtain the ID and resource for the shortest task in the current iteration
                if (min_time < shortest_task_exec_time) :
                    shortest_task_exec_time = min_time
                    shortest_task_pe_id     = resource_id
                    shortest_task           = task
                # end of if (min_time < shortest_task_exec_time)

            # end of for task in list_of_ready:
            # At the end of this loop, we should have the minimum execution time
            # of a task across all resources

            # Assign PE ID of the shortest task
            index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            shortest_task.PE_ID        = shortest_task_pe_id

            if (common.DEBUG_SCH):
                print ('[I] Time %s: The scheduler function found task %d to be shortest on resource %d with %.1f'
                       %(self.env.now, shortest_task.ID, shortest_task.PE_ID, shortest_task_exec_time))

            if list_of_ready[index].PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                       % (self.env.now,shortest_task.name))
                print ('[E] or job_**.txt file')
                assert(shortest_task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print ('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                           %(self.env.now, shortest_task.ID, shortest_task.PE_ID,
                             self.resource_matrix.list[shortest_task.PE_ID].type))
            # end of if shortest_task.PE_ID == -1:

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list) :
                if task.ID == shortest_task.ID :
                    ready_list.remove(task)

        # end of for task in list_of_ready:
        # At the end of this loop, all ready tasks are assigned to the resources
        # on which the execution times are minimum. The tasks will execute
        # in the order of increasing execution times


    # end of STF(list_of_ready)

    def ETF_LB(self, list_of_ready):
        '''!
        This scheduler compares the execution times of the current task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the earliest finish time for the task. Additionally, the task 
        with the lowest earliest finish time is scheduled first
        @param list_of_ready: The list of ready tasks
        '''

        ready_list = copy.deepcopy(list_of_ready)

        task_counter = 0
        assigned = self.assigned

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1
            shortest_comparison = [np.inf] * len(self.PEs)

            for task in ready_list:

                comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
                comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE

                if (common.DEBUG_SCH):
                    print('[D] Time %s: The scheduler function is called with task %s'
                          % (self.env.now, task.ID))

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        # if the task is supported by the resource, retrieve the index of the task
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                            # $PE_comm_wait_times is a list to store the estimated communication time
                            # (or the remaining communication time) of all predecessors of a task for a PE
                            # As simulation forwards, relevant data is being sent after a task is completed
                            # based on the time instance, one should consider either whole communication
                            # time or the remaining communication time for scheduling
                            PE_comm_wait_times = []

                            # $PE_wait_time is a list to store the estimated wait times for a PE
                            # till that PE is available if the PE is currently running a task
                            PE_wait_time = []

                            job_ID = -1  # Initialize the job ID

                            # Retrieve the job ID which the current task belongs to
                            for ii, job in enumerate(self.jobs.list):
                                if job.name == task.jobname:
                                    job_ID = ii

                            for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                                # data required from the predecessor for $ready_task
                                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                                # retrieve the real ID  of the predecessor based on the job ID
                                real_predecessor_ID = predecessor + task.ID - task.base_ID

                                # Initialize following two variables which will be used if
                                # PE to PE communication is utilized
                                predecessor_PE_ID = -1
                                predecessor_finish_time = -1

                                for completed in common.completed:
                                    if completed.ID == real_predecessor_ID:
                                        predecessor_PE_ID = completed.PE_ID
                                        predecessor_finish_time = completed.finish_time
                                        # print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                        break

                                if (common.PE_to_PE):
                                    # Compute the PE to PE communication time
                                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                    PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                    PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d'
                                              % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                                if (common.shared_memory):
                                    # Compute the communication time considering the shared memory
                                    # only consider memory to PE communication time
                                    # since the task passed the 1st phase (PE to memory communication)
                                    # and its status changed to ready

                                    # PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                    memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                    shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                    PE_comm_wait_times.append(shared_memory_comm_time)
                                    if (common.DEBUG_SCH):
                                        print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d'
                                              % (self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                                # $comm_ready contains the estimated communication time
                                # for the resource in consideration for scheduling
                                # maximum value is chosen since it represents the time required for all
                                # data becomes available for the resource.
                                comm_ready[i] = max(PE_comm_wait_times)
                            # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                            # if a resource currently is executing a task, then the estimated remaining time
                            # for the task completion should be considered during scheduling
                            PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                            # update the comparison vector accordingly
                            comparison[i] = self.resource_matrix.list[i].performance[ind] * (1 + DTPM_power_models.compute_DVFS_performance_slowdown(common.ClusterManager.cluster_list[self.PEs[i].cluster_ID])) + max(comm_ready[i], PE_wait_time[-1])
                        # end of if (task.name in...
                # end of for i in range(len(self.resource_matrix.list)):

                if min(comparison) < shortest_task_exec_time:
                    resource_id = comparison.index(min(comparison))
                    shortest_task_exec_time = min(comparison)
                    #                    print(shortest_task_exec_time, comparison)
                    count_minimum = comparison.count(shortest_task_exec_time)  # also, record how many times $min_of_exec_times is seen in the list
                    # if there are two or more PEs satisfying minimum execution
                    # then we should try to utilize all those PEs
                    if (count_minimum > 1):
                        # if there are two or more PEs satisfying minimum execution
                        # populate the IDs of those PEs into a list
                        min_PE_IDs = [i for i, x in enumerate(comparison) if x == shortest_task_exec_time]
                        # then compare the number of the assigned tasks to remaining PEs
                        # and choose the one with the lowest number of assigned tasks
                        assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                        PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                        # finally, choose the best available PE for the task
                        task.PE_ID = min_PE_IDs[PE_ID_index]
                    #   print(count_minimum, task.PE_ID)
                    else:
                        task.PE_ID = comparison.index(shortest_task_exec_time)
                    # end of if count_minimum >1:

                    # since one task is just assigned to a PE, increase the number by 1
                    assigned[task.PE_ID] += 1

                    resource_id = task.PE_ID
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                    shortest_comparison = copy.deepcopy(comparison)

            # assign PE ID of the shortest task
            index = [i for i, x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID = shortest_task_pe_id

            if shortest_task.PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check SoC.**.txt file'
                      % (self.env.now, shortest_task.ID))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                          % (self.env.now, shortest_task.ID))
                    print('%12s' % (''), comparison)
                    print('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                          % (self.env.now, shortest_task.ID, shortest_task.PE_ID, self.resource_matrix.list[shortest_task.PE_ID].name))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            index_min_available_time = self.PEs[shortest_task.PE_ID].available_time_list.index(min(self.PEs[shortest_task.PE_ID].available_time_list))
            self.PEs[shortest_task.PE_ID].available_time_list[index_min_available_time] = self.env.now + shortest_comparison[shortest_task.PE_ID]

            self.PEs[shortest_task.PE_ID].available_time = min(self.PEs[shortest_task.PE_ID].available_time_list)

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

    # end of ETF_LB( list_of_ready)

    def ETF(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task. Additionally, the task with the
        lowest earliest finish time  is scheduled first
        '''
        ready_list = copy.deepcopy(list_of_ready)
    
        task_counter = 0
    
        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0 :
    
            shortest_task_exec_time = np.inf
            shortest_task_pe_id     = -1
            shortest_comparison     = [np.inf] * len(self.PEs)
    
            for task in ready_list:
                
                comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector 
                comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE
                
                if (common.DEBUG_SCH):
                    print ('[D] Time %s: The scheduler function is called with task %s'
                           %(self.env.now, task.ID))
                    
                for i in range(len(self.resource_matrix.list)):
                    # if the task is supported by the resource, retrieve the index of the task
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        
                            
                        # $PE_comm_wait_times is a list to store the estimated communication time 
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []
                        
                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []
                          
                        job_ID = -1                                                     # Initialize the job ID
                        
                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii
                                
                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                            
                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID
                            
                            # Initialize following two variables which will be used if 
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1
                            
                            
                            for completed in common.completed:
                                if (completed.ID == real_predecessor_ID):
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                    
                            
                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                #PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
                                
                                PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                                
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                                          %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                                
                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready 
                                
                                #PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol/memory_to_PE_band)
                                
                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
                                          %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                            
                            # $comm_ready contains the estimated communication time 
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource. 
                            comm_ready[i] = (max(PE_comm_wait_times))
                            
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
                        
                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))
                        
                        # update the comparison vector accordingly    
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i], PE_wait_time[-1])
                        
    
                        # after going over each resource, choose the one which gives the minimum result
                        resource_id = comparison.index(min(comparison))
                        #print('aa',comparison)
                    # end of if (task.name in self.resource_matrix.list[i]...
                    
                # obtain the task ID, resource for the task with earliest finish time 
                # based on the computation 
                #print('bb',comparison)
                if min(comparison) < shortest_task_exec_time :
                    shortest_task_exec_time = min(comparison)
                    shortest_task_pe_id     = resource_id
                    shortest_task           = task
                    shortest_comparison     = comparison
    
                    
                # end of for i in range(len(self.resource_matrix.list)):
            # end of for task in ready_list:
            
            # assign PE ID of the shortest task 
            index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID        = shortest_task_pe_id
    
            if shortest_task.PE_ID == -1:
                print ('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                       % (self.env.now, shortest_task.ID))
                print ('[E] or job_**.txt file')
                assert(task.PE_ID >= 0)           
            else: 
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively' 
                              %(self.env.now, shortest_task.ID))
                    print('%12s'%(''), comparison)
                    print ('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                           %(self.env.now, shortest_task.ID, shortest_task.PE_ID, 
                             self.resource_matrix.list[shortest_task.PE_ID].name))
            
            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[shortest_task.PE_ID].available_time = self.env.now + shortest_comparison[shortest_task.PE_ID]
            
            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list) :
                if task.ID == shortest_task.ID :
                    ready_list.remove(task)
            
            
            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :
    #end of ETF( list_of_ready) 
    
  
    
    def CP(self, list_of_ready):
        '''!
        This scheduler utilizes a look-up table for scheduling tasks to a particular processor
        @param list_of_ready: The list of ready tasks
        '''
        for task in list_of_ready:    
            ind = 0
            base =  0
            for item in common.ilp_job_list:
                if item[0] == task.jobID:
                    ind = common.ilp_job_list.index(item)
                    break
            
            previous_job_list = list(range(ind))
            for job in previous_job_list:
                selection = common.ilp_job_list[job][1]
                num_of_tasks = len(self.jobs.list[selection].task_list)
                base += num_of_tasks
            
            #print(task.jobID, base, task.base_ID)
            
            for i, schedule in enumerate(common.table):
            
                if len(common.table) > base:
                    if (task.base_ID + base) == i:
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]
                else:
                    if ( task.ID%num_of_tasks == i):
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]        
         
            
        list_of_ready.sort(key=lambda x: x.order, reverse=False) 
    # def CP_(self, list_of_ready): 
    """
    This scheduler is twofold:

    1) Choose which accelerator type to use
    - find minimum runtime of each task on each PE, while taking into account forwarding (maybe, look into this more. Did sudanshu take into account forwarding/not when calculating laxity?)
    - find least laxity on each PE type using critical path deadline and runtime
    - 
    
    """

    def RELIEF_BASIC(self, list_of_ready: List[Tasks]):
        '''!
        RELIEF (Resource-Limited Execution with Intelligent Forwarding) Scheduler.

        This scheduler implements a two-phase algorithm:
        Phase 1: Map each task to its fastest PE and calculate laxity
        Phase 2: Attempt to forward tasks to idle PEs when feasible

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Phase 1: Map tasks to PEs and organize by laxity
        # Create per-PE buckets to hold tasks sorted by laxity (smallest to largest)
        fwd_nodes = [[] for _ in range(len(self.PEs))]

        for task in list_of_ready:
            # Find the PE with minimum execution time for this task
            best_pe_id = -1
            best_pe_type = ""
            min_exec_time = -1

            for i, resource in enumerate(self.resource_matrix.list):
                if task.name in resource.supported_functionalities:
                    # Get execution time for this task on this PE
                    func_index = resource.supported_functionalities.index(task.name)
                    exec_time = resource.performance[func_index]

                    # Update if this is the fastest PE found so far
                    if min_exec_time == -1 or exec_time < min_exec_time:
                        min_exec_time = int(exec_time)
                        best_pe_id = resource.ID
                        best_pe_type = resource.type

            # Ensure task can be executed on at least one PE
            assert best_pe_id != -1, f"Task {task.name} cannot be executed on any PE"

            # Calculate task laxity: deadline - runtime
            # Laxity represents slack time before deadline is violated
            task.laxity = task.deadline - min_exec_time
            task.PE_ID = best_pe_id
            task.runtime = min_exec_time

            # Insert task into PE's bucket sorted by laxity (ascending order)
            # Smaller laxity = less slack = higher priority
            insert_index = 0
            while insert_index < len(fwd_nodes[best_pe_id]) and task.laxity > fwd_nodes[best_pe_id][insert_index].laxity:
                insert_index += 1
            fwd_nodes[best_pe_id].insert(insert_index, task)

        # Phase 2: Schedule tasks, forwarding to idle PEs when feasible
        for pe_id, PE in enumerate(self.PEs):
            # Check if this PE is idle and can accept forwarded tasks
            can_forward = PE.idle and (common.forwarding_enabled if hasattr(common, 'forwarding_enabled') else False)

            # Process all tasks assigned to this PE (highest laxity first - pop from end)
            while len(fwd_nodes[pe_id]) > 0:
                # Get next task to schedule (highest laxity)
                task = fwd_nodes[pe_id].pop()

                # Set execution readiness timestamp
                task.time_stamp = self.env.now

                # Find insertion point in this PE's executable queue based on laxity
                # Insert after tasks with lower or equal laxity
                insert_index = 0
                if pe_id in common.executable:
                    for exec_task in common.executable[pe_id]:
                        if exec_task.laxity <= task.laxity:
                            insert_index += 1
                        else:
                            break

                # Decide: forward to idle PE or schedule normally
                if can_forward and self.is_feasible(pe_id, task, insert_index):
                    # Forward task - insert at front of queue for immediate execution
                    if task.PE_ID in common.executable:
                        common.executable[task.PE_ID].insert(0, task)

                    # Mark task as forwarded and update metadata
                    task.isForwarded = True
                    task.forwarded_from_PE = pe_id
                    can_forward = False  # Can only forward one task per idle PE per scheduling round
                    self.update_forward_metadata(task)
                else:
                    # Normal scheduling - insert at calculated position
                    if task.PE_ID in common.executable:
                        common.executable[task.PE_ID].insert(insert_index, task)

        # Clear processed tasks from ready queue
        for task in list_of_ready:
            if task in common.ready:
                common.ready.remove(task)

    def is_feasible(self, accelerator_id:int, task:Tasks, index:int):
        '''!
        Determine if forwarding a task to an idle PE is feasible without violating laxity constraints.

        Forwarding is feasible if adding the task's runtime in front of the queue does not
        cause any non-forwarded tasks to violate their laxity constraints.

        @param accelerator_id: ID of the PE to which the task is assigned
        @param task: The task being considered for forwarding
        @param index: Position where task would be inserted in executable queue
        @return: True if forwarding is feasible, False otherwise
        '''
        can_forward = True

        # Check if any task already in this PE's queue would be delayed past its laxity
        if accelerator_id in common.executable:
            for i, executable_task in enumerate(common.executable[accelerator_id]):
                # Stop checking once we reach the insertion point
                if i == index:
                    break

                # Calculate current laxity relative to simulation time
                curr_laxity = executable_task.laxity - self.env.now

                # Check if a non-forwarded task with positive laxity would be violated
                if not executable_task.isForwarded and curr_laxity > 0:
                    # Forwarding is feasible only if this task's laxity exceeds the forwarded task's runtime
                    can_forward = curr_laxity > task.runtime
                    break

            # If forwarding is feasible, update laxity values for tasks that will be delayed
            # This accounts for the additional delay caused by the forwarded task executing first
            if can_forward:
                for i, executable_task in enumerate(common.executable[accelerator_id]):
                    if i == index:
                        break
                    # Reduce laxity by the forwarded task's runtime
                    executable_task.laxity -= task.runtime

        return can_forward

    def update_forward_metadata(self, task:Tasks):
        '''!
        Update metadata for a task that has been forwarded to an idle PE.

        This function handles all the bookkeeping required when a task is forwarded,
        including scratchpad management, data location tracking, and communication mode setup.

        @param task: The task that has been forwarded to an idle PE

        TODO: Implement the following functionality:

        1. SCRATCHPAD ALLOCATION FOR FORWARDED TASK'S INPUTS:
           - Identify all predecessor tasks whose outputs this task needs
           - For each predecessor:
             a) Check if predecessor's output is already in the target PE's scratchpad
                - Use: self.PEs[task.PE_ID].has_data_in_scratchpad(f"{pred_id}_output")
             b) If not in scratchpad, allocate space for the data
                - Get data size from task.data_sizes[pred_id] or job.comm_vol matrix
                - Use: self.PEs[task.PE_ID].allocate_scratchpad(data_id, size, pred_task_id)
                - This will automatically evict old data if scratchpad is full (LRU policy)
             c) Track data location in task.data_locations[pred_id] = task.PE_ID

        2. SCRATCHPAD RESERVATION FOR TASK'S OUTPUT:
           - Calculate output data size for this task
           - Reserve scratchpad space for task's output so successors can be forwarded
           - Use: self.PEs[task.PE_ID].allocate_scratchpad(f"{task.ID}_output", output_size, task.ID)
           - Consider: May need to check if there's sufficient space for output before forwarding

        3. UPDATE COMMUNICATION TIMING METADATA:
           - Set task.comm_timing_mode = 'PE_to_PE' (already handled by decide_comm_timing)
           - Update task.data_locations dictionary with locations of all input data
           - This metadata will be used by DASH_Sim_core.decide_comm_timing() to determine
             communication costs when the task executes

        4. TRACK PREDECESSOR PE LOCATIONS:
           - For each predecessor task, record which PE it executed on
           - Loop through common.completed to find predecessors
           - Store in task.data_locations[pred_task_ID] = pred_PE_ID
           - This enables calculating PE-to-PE communication bandwidth and timing

        5. CALCULATE ACTUAL COMMUNICATION COSTS:
           - For each predecessor:
             a) Get communication volume from job.comm_vol[pred_base_id, task.base_id]
             b) Get communication bandwidth:
                - If data in scratchpad: Use PE-to-PE bandwidth from comm_band matrix
                - If data in memory: Use memory-to-PE bandwidth
             c) Update task's expected ready time based on communication delays
             d) Consider: May need to adjust task.time_stamp based on actual comm delays

        6. HANDLE PARTIAL FORWARDING:
           - Some inputs may be in scratchpad (can forward), others in memory (cannot)
           - Track which inputs can use PE_to_PE timing vs memory timing
           - Store in task.input_comm_modes[pred_id] = 'PE_to_PE' or 'memory'
           - This allows hybrid communication: forward what's possible, fetch rest from memory

        7. UPDATE SUCCESSOR TASK OPPORTUNITIES:
           - Mark this task's output as potentially forwardable for successor tasks
           - When successors become ready, they can check if this task's output is in scratchpad
           - May want to maintain a global or per-PE data structure tracking available data

        8. STATISTICS AND DEBUGGING:
           - Track number of successful forwards vs failed forwards
           - Record scratchpad utilization over time
           - Log forwarding decisions for analysis
           - If common.DEBUG_SCH, print forwarding metadata details

        9. EDGE CASES TO HANDLE:
           - Task has no predecessors (head task): No inputs to forward
           - Scratchpad full: Need to evict data, may impact future forwarding opportunities
           - Multiple predecessors on different PEs: Some may be forwardable, others not
           - Predecessor output too large for scratchpad: Cannot forward, must use memory

        10. INTEGRATION WITH EXISTING COMMUNICATION MODES:
            - In forwarding mode, this function sets up the metadata
            - DASH_Sim_core.decide_comm_timing() will use task.isForwarded flag and metadata
            - DASH_Sim_core.update_execution_queue() will use metadata to calculate delays
            - Processing_element.run() may need to free scratchpad after task completes

        IMPLEMENTATION NOTES:
        - Access job information via: self.jobs.list[job_id] where job matches task.jobname
        - Communication volume matrix: self.jobs.list[job_id].comm_vol[pred_base_id, task_base_id]
        - Communication bandwidth: common.ResourceManager.comm_band[src_PE_id, dst_PE_id]
        - Scratchpad methods: PE.allocate_scratchpad(), PE.free_scratchpad(), PE.has_data_in_scratchpad()
        - Task metadata: task.data_locations, task.data_sizes, task.comm_timing_mode, task.forwarded_from_PE
        '''
        # TODO: Implement the functionality described above
        pass

    def RELIEF_ALT(self, list_of_ready: List[Tasks]):
        # Note: list_of_ready is a surface copy

        fwd_nodes = [[]*len(self.PEs)]                                          # Keeps track of sorted forwarding candidates for each PE

        for task in list_of_ready:
            laxity = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector
            job_ID = -1                                                         # Initialize the job ID

            # Retrieve the job ID which the current task belongs to
            # Jobs are stored in self.jobs.list, meaning we need to do a 
            # name lookup in that list to get the DAG structure. Why??
            for ii, job in enumerate(self.jobs.list):
                if job.name == task.jobname:
                    job_ID = ii

            # TODO - calculate critical path deadline of the task

            if (common.DEBUG_SCH):
                print ('[D] Time %s: The scheduler function is called with task %s'
                        %(self.env.now, task.ID))
            
            # Now, we compute the laxity of the task on each PE.
            for i in range(len(self.resource_matrix.list)):
                can_fwd_all = True                                             # indicates whether or not all resources can be forwarded
                # if the task is supported by the resource, retrieve the index of the task
                if (task.name in self.resource_matrix.list[i].supported_functionalities):
                    ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                    PE_FWD_LIST = []                                            # A list of predecessor values to be forwarded, used in runtime calculations

                    # Gather data from each predecessor to determine communication times
                    for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                        # data required from the predecessor for $ready_task
                        c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                        
                        # retrieve the real ID  of the predecessor based on the job ID - TODO What is the difference between real id and predecessor ID?
                        # Why does each task have 15 different types of ID
                        real_predecessor_ID = predecessor + task.ID - task.base_ID
                    
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1

                        # TODO - There has to be a better way to do this. Maybe we turn the completed list into a set? 
                        for completed in common.completed:
                            if (completed.ID == real_predecessor_ID):
                                predecessor_PE_ID = completed.PE_ID
                                predecessor_finish_time = completed.finish_time
                                break
                                #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                        
                        if (common.FWD_ENABLED): 
                            PE_FWD_LIST.append((predecessor,predecessor_PE_ID,predecessor_finish_time,c_vol))
                            # TODO - find a way to indicate which inputs can be forwarded. Forwarding can ONLY happen to idle PEs, but some inputs can be forwarded and others not.
                            if can_fwd_all: # no point in computing can_fwd if it can't forward
                                can_fwd_all = can_fwd_all and can_fwd(task,predecessor_PE_ID,i)
                    # End of for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:

                    # $PE_comm_wait_times is a list to store the estimated communication time 
                    # (or the remaining communication time) of all predecessors of a task for a PE
                    # As simulation forwards, relevant data is being sent after a task is completed
                    # based on the time instance, one should consider either whole communication
                    # time or the remaining communication time for scheduling
                    PE_comm_wait_times = []

                    if not can_fwd_all: # use memory
                        for (predecessor, predecessor_PE_ID, predecessor_finish_time, c_vol) in PE_FWD_LIST:
                            memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
                            shared_memory_comm_time = int(c_vol/memory_to_PE_band)
                        
                            PE_comm_wait_times.append(shared_memory_comm_time)
                            if (common.DEBUG_SCH):
                                print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
                                %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                    else:  #directly to PE scratchpad -> scratchpad
                        for (predecessor, predecessor_PE_ID, predecessor_finish_time, c_vol) in PE_FWD_LIST:
                        # Compute the PE to PE communication time
                        # PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                            PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                            PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
                            
                            PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                            
                            if (common.DEBUG_SCH):
                                print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
                                        %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
                                
                    
                    # comparison[i] is how long a task will take to fully run on PE i.
                    # latency + max(input wait time, PE's remaining runtime for all tasks)
                    task_finish_time_on_PE = self.resource_matrix.list[i].performance[ind] + max(max(PE_comm_wait_times), max((self.PEs[i].available_time - self.env.now), 0))

                    # TODO- now we need to find the critical path. requires an overhaul of job_generator.py, job_parser.py to include job-level deadlines, I believe

                    task_critical_path_deadline = task.deadline

                    # find least laxity!

                    laxity[i] = task_critical_path_deadline - task_finish_time_on_PE
                # end if (task.name in self.resource_matrix.list[i].supported_functionalities): - checks if accelerator supports computing this task
            # end for i in range(len(self.resource_matrix.list)) - computes execution time on each potential accelerator
            
            # we schedule on the PE with the highest laxity:
            max_laxity_pe_index = max(range(len(laxity)), key=laxity.__getitem__)
            task.laxity = laxity[max_laxity_pe_index]

            # find the index of fwd_nodes to insert into.
            index = 0
            while index < len(fwd_nodes[max_laxity_pe_index]) and fwd_nodes[max_laxity_pe_index][index].laxity < task.laxity:
                index+=1

            fwd_nodes.insert(index, task)

            # TODO - In our can_fwd logic,  we already perform the idle accelerator check, as well as indicate which inputs can be forwarded or not.
            # we also 
            task.forwarded = False
        # end for task in list_of_ready:



        # So, in our first loop, we perform checks on which inputs can potentially be forwarded or colocated when running, and our second loop will check whether a given PE is idle and whether it's list of nodes 
        # has room for forwarding. If it does, we forward. If not, we don't and all values are saved to memory.

        # Be very careful - each task can have many children, and each task can also have many parents




        #will be used to pull memory when PE begins working on it. Adjust based on whether the final selected PE is being forwarded to

    def LL(self,list_of_ready):
        ""

    def PE_fwdable(PE_ID):
        # first, we check whether the PE is idle. If it is, we can forward.

        # Then, we check the PE's input queues to see if forwarding on this accelerator at all is feasible.
        return True


    def can_fwd(task,predPE,newPE):

        # Then, performs a check to see if the task is still alive in the producer's output scratchpad
        # if so, it checks if there is a forwarding opportunity between producer and consumer.
        # finally, if there is a forwarding opportunity (laxity allows for it), it returns the memory speed 
        # of the forward (Based on PE-to-PE communication times) or -1 if it cant be forwarded

        #also check for colocation!
        return True

