import common



# Set forwarding memory movement
# Memory should only move when going into the PE's Queues, not when entering the execution queue. Double buffering?
def update_execution_queue_relief(self, ready_list):
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
                    # if a task is the leading task of a job
                    # then it can start immediately since it has no predecessor
                    ready_task.PE_to_PE_wait_time.append(self.env.now)
                    ready_task.execution_wait_times.append(self.env.now)
                # end of if ready_task.head == True:

                for predecessor in task.predecessors:
                    if(task.ID==ready_task.ID):
                        ready_task.predecessors = task.predecessors

                    # data required from the predecessor for $ready_task
                    comm_vol = self.jobs.list[job_ID].comm_vol[predecessor, ready_task.base_ID]

                    # retrieve the real ID  of the predecessor based on the job ID
                    real_predecessor_ID = predecessor + ready_task.ID - ready_task.base_ID

                    # Initialize following two variables which will be used if
                    # PE to PE communication is utilized
                    predecessor_PE_ID = -1
                    predecessor_finish_time = -1

                    # TODO - implement conditional forwarding for each potential input. Create another list that contains predecessor_is_forwarding

                    if (common.PE_to_PE):
                        # Compute the PE to PE communication time
                        for completed in common.completed:
                            if completed.ID == real_predecessor_ID:
                                predecessor_PE_ID = completed.PE_ID
                                predecessor_finish_time = completed.finish_time
                        comm_band = common.ResourceManager.comm_band[predecessor_PE_ID, ready_task.PE_ID]
                        PE_to_PE_comm_time = int(comm_vol/comm_band)
                        ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)

                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data transfer from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                    %(self.env.now, predecessor_PE_ID, ready_task.PE_ID,
                                    ready_task.ID, real_predecessor_ID, ready_task.PE_to_PE_wait_time[-1]))
                    # end of if (common.PE_to_PE):

                    if (common.shared_memory):
                        # Compute the memory to PE communication time
                        comm_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, ready_task.PE_ID]
                        from_memory_comm_time = int(comm_vol/comm_band)
                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                                    %(self.env.now, ready_task.ID, real_predecessor_ID, ready_task.PE_ID, from_memory_comm_time))
                        ready_task.execution_wait_times.append(from_memory_comm_time + self.env.now)
                    # end of if (common.shared_memory)
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
                # common.executable.append(ready_task) this is done in the scheduling algorithm.
                remove_from_ready_queue.append(ready_task)
                if (common.PE_to_PE):
                    common.executable[-1].time_stamp = max(ready_task.PE_to_PE_wait_time)
                else:
                    common.executable[-1].time_stamp = max(ready_task.execution_wait_times)
            # end of ready_task.base_ID == task.ID:
        # end of i, task in enumerate(self.jobs.list[job_ID].task_list):    
    # end of for ready_task in ready_list:
    
    # Remove the tasks from ready queue that have been moved to executable queue
    for task in remove_from_ready_queue:
        common.ready.remove(task)
    
    common.executable.sort(key=lambda task: task.jobID, reverse=False) # why would we sort based on which Job comes first - Nebil