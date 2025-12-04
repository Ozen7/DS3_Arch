'''!
@brief This file contains all schedulers in DASH-Sim

Scheduler class is defined in this file which contains different types of scheduler as a member function.
Developers can add thier own algorithms here and implement in DASH-Sim by add a function caller in DASH_Sim_core.py
'''
import networkx as nx
import numpy as np
import copy
import bisect
import random

import common                                                                   # The common parameters used in DASH-Sim are defined in common_parameters.py
from common import Tasks, ApplicationManager, ResourceManager
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
        self.lastUpdate = 0

        # At the end of this function, the scheduler class has a copy of the
        # the power/performance characteristics of the resource matrix and
        # name of the requested scheduler name


    # end  def __init__(self, env, resource_matrix, scheduler_name)

    def populate_execution_queue(self, ready_list):
        DASH_Sim_core.SimulationManager.update_execution_queue()
        pass

    def find_best_PE(self, task:Tasks, fwd_nodes):
        best_pe_id = -1
        min_finish_time = -1
        best_family = ""
        isColcoated = False
        # Find the fastest PE type for this task
        for resource in self.resource_matrix.list:
            if task.name in resource.supported_functionalities:
                # Get execution time for this task on this PE
                func_index = resource.supported_functionalities.index(task.name)
                isColcoated, latency = common.calculate_memory_movement_latency(self,task,resource.ID, False)
                finish_time = int(resource.performance[func_index]) + latency
                
                # Update if this is the fastest PE found so far - or, if it is colocated 
                if min_finish_time == -1 or finish_time < min_finish_time or (finish_time == min_finish_time and isColcoated):
                    min_finish_time = finish_time
                    best_pe_id = resource.ID
                    best_family = resource.accelerator_family

        # Ensure task can be executed on at least one PE
        assert best_pe_id != -1, f"Task {task.name} cannot be executed on any PE"

        task.PE_Family = best_family
        task.runtime = min_finish_time
        task.laxity = task.deadline - task.runtime

    def RELIEF(self, list_of_ready: List[Tasks]):
        '''!
        RELIEF Scheduler.

        This scheduler implements a two-phase algorithm:
        Phase 1: Map each task to its fastest PE and calculate laxity
        Phase 2: Attempt to forward tasks to idle PEs when feasible

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Phase 1: Map tasks to PEs and organize by laxity
        # Create per-PE buckets to hold tasks sorted by laxity (smallest to largest)
        fwd_nodes = {family:[] for family in common.TypeManager.get_all_families()}

        for task in list_of_ready:
            self.find_best_PE(task,fwd_nodes)

            # Insert task into PE's bucket sorted by laxity (smallest to largest)
            # Smaller laxity = less slack = higher priority
            insert_index = 0
            while insert_index < len(fwd_nodes[task.PE_Family]) and task.laxity > (fwd_nodes[task.PE_Family][insert_index].laxity):
                insert_index += 1
            fwd_nodes[task.PE_Family].insert(insert_index, task)
        print("LAXITY", task.laxity)

        # Phase 2: Schedule tasks, forwarding to idle PEs when feasible
        for family in common.TypeManager.get_all_families():
            idle_PEs = 0
            for PID in common.TypeManager.get_PEs_of_family(family):
                P = self.PEs[PID]
                if P.idle and not P.lock:
                    idle_PEs +=1
            num_forwards = 0
            while len(fwd_nodes[family]) > 0:
                # Get next task to schedule (least laxity)
                task = fwd_nodes[family].pop(0)
                try_forward = num_forwards < idle_PEs
                can_forward = False
                # Find insertion point in this PE's executable queue based on laxity
                # Insert after tasks with lower or equal laxity
                insert_index = 0
                for exec_task in common.executable[family]:
                    if task.laxity >= exec_task.laxity:
                        insert_index += 1
                    else:
                        break


                if try_forward:
                    for task_iter in common.executable[family][:insert_index]:
                        laxity = task_iter.laxity - self.env.now
                        if laxity > 0:
                            can_forward = task_iter.laxity >= task.runtime
                            break
            
                # if we are forwarding
                if can_forward:
                    new_insert_index = 0
                    task.isForwarded = True
                    common.results.num_RELIEF_forwards += 1
                    num_forwards += 1 # Can only forward one task per idle PE per scheduling round
                    # iterate through the tasks AFTER where the forwarded task will be and BEFORE its original location
                    for task_iter in common.executable[family][:insert_index]:
                        task_iter.laxity -= task.runtime
                        if task_iter.isForwarded:
                            new_insert_index += 1
                    insert_index = new_insert_index
                common.executable[family].insert(insert_index, task)

        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return

                        

    def LL(self,list_of_ready):
        '''!
        Least Laxity Scheduler.
        
        Maps each task to its fastest PE and calculates laxity

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Map tasks to PEs and organize by laxity

        for task in list_of_ready:
            self.find_best_PE(task,None)

            # Insert task into PE sorted by laxity (smallest to largest)
            # Smaller laxity = less slack = higher priority
            insert_index = 0
            while insert_index < len(common.executable[task.PE_Family]) and task.laxity >= common.executable[task.PE_Family][insert_index].laxity:
                insert_index += 1
            common.executable[task.PE_Family].insert(insert_index, task)
        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return
    

    def GEDF_D(self,list_of_ready):
        '''!
        Global Earliest Deadline First - Dag
        
        Maps each task to its fastest PE and sorts based on DAG deadline

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Map tasks to PEs and organize by dag deadline

        for task in list_of_ready:
            self.find_best_PE(task,None)

            for ii, job in enumerate(self.jobs.list):
                if job.name == task.jobname:
                    job_ID = ii

            task.jobDeadline = self.jobs.list[job_ID].deadline

            insert_index = 0
            while insert_index < len(common.executable[task.PE_Family]) and task.jobDeadline >= common.executable[task.PE_Family][insert_index].jobDeadline:
                insert_index += 1
            common.executable[task.PE_Family].insert(insert_index, task)
        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return
    
    def GEDF_N(self,list_of_ready):
        '''!
        Global Earliest Deadline First - Node
        
        Maps each task to its fastest PE and sorts based on Node deadline

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Map tasks to PEs and organize by deadline

        for task in list_of_ready:
            self.find_best_PE(task,None)

            insert_index = 0
            while insert_index < len(common.executable[task.PE_Family]) and task.deadline >= common.executable[task.PE_Family][insert_index].deadline:
                insert_index += 1
            common.executable[task.PE_Family].insert(insert_index, task)
        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return
    
    def FCFS(self,list_of_ready):
        '''!
        First Come First Served
        
        Tasks are ordered based on arrival rate. Simply push a task to the end of the queue.

        @param list_of_ready: List of tasks ready to be scheduled
        '''
        # Map tasks to PEs

        for task in list_of_ready:
            self.find_best_PE(task,None)

            # just add tasks to the end of the queue. 
            common.executable[task.PE_Family].append(task)
        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return
    

    def HetSched(self,list_of_ready):
        '''!
        HetSched (Quality-of-Mission Aware Scheduling for Autonomous Vehicle SoCs)

        https://doi.org/10.48550/arXiv.2203.13396
     
        Maps each task to its fastest PE and sorts based on Node deadline

        @param list_of_ready: List of tasks ready to be scheduled
        '''

        fwd_nodes = [[] for _ in range(len(self.PEs))]

        for task in list_of_ready:
            self.find_best_PE(task,fwd_nodes)

            task.laxity = task.sd - task.runtime # use SD to calculate laxity
            print("1) TASK LAXITY", task.ID,  task.laxity)

            insert_index = 0
            while insert_index < len(common.executable[task.PE_Family]) and task.laxity >= common.executable[task.PE_Family][insert_index].laxity:
                insert_index += 1
            print("INSERT AT", insert_index)
            for x in common.executable[task.PE_Family]:
                print("TASK LAX", x.ID, x.laxity)
            common.executable[task.PE_Family].insert(insert_index, task)
            
        
        # now that they are scheduled (put into the execution queue), we need to delete them from the ready list
        rm = []
        for task in list_of_ready:
            rm.append(task)
        for task in rm:
            common.ready.remove(task)
        
        return