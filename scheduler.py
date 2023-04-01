import math
import config
from tqdm import tqdm
from workflow import Task, Workflow
from vm import Vm
import sys
MAX_INT = sys.maxsize

class Scheduler(object):
    def __init__(self, workflow_list, vm_info) -> None:
        self.name = 'Default'
        self.workflow_list: list[Workflow] = workflow_list

        self.current_time = 0
        self.next_worklfow_arrival_time: float = MAX_INT
       
        self.next_event_time: float = MAX_INT
        self.next_event_vm: Vm = None
        self.next_event_type: float = None

        self.vm_info = vm_info
        self.active_vm_list: list[Vm] = []
        self.inactive_vm_list: list[Vm] = []
        self.cost = None

        self.ready_task_list: list[Task] = []
        self.wait_task_list: list[Task] = []
        self.finish_task_list: list[Task] = []
        
        self.time_out = False
    
    def preprocess(self, workflow: Workflow):
        self.calculateTaskConfidenceTime(workflow)
        self.calculateTaskDeadline(workflow)
        self.sortTaskList(workflow)
    
    def calculateTaskConfidenceTime(self, workflow: Workflow):
        for task in workflow.task_list:
            task.base_confidence_execute_time = task.base_execute_time * (1 + config.STANDARD_DEVIATION_OF_EXECUTE_TIME)
            for edge in task.succ_edge_list:
                edge.confidence_data_size = edge.base_data_size * (1 + config.STANDARD_DEVIATION_OF_DATA_SIZE)
                edge.confidence_data_transfer_time = edge.confidence_data_size / config.BAND_WIDTH
                
    
    # get earliest_start_time & earliest_finish_time & laxity_execute_time & deadline
    def calculateTaskDeadline(self, workflow: Workflow):
        vm_type_list = sorted(self.vm_info.items(), key = lambda x:x[1]['factor'], reverse=True)
        for vm_type in vm_type_list:
            factor = vm_type[1]['factor']
            
            # get earliest_*_time by base_execute_time from entry task to exit task
            earliest_workflow_finish_time = -1
            for task in workflow.task_list:
                task.earliest_start_time = self.current_time
                for pred_edge in task.pred_edge_list:
                    task.earliest_start_time = max(pred_edge.src_task.earliest_finish_time + pred_edge.base_data_transfer_time, task.earliest_start_time)
                task.earliest_finish_time = task.earliest_start_time + task.base_execute_time * factor
                earliest_workflow_finish_time = max(task.earliest_finish_time, earliest_workflow_finish_time)
            
            # vm of this type cannot finish workflow
            if earliest_workflow_finish_time > workflow.deadline:
                if factor == vm_type_list[-1][1]['factor']:
                    for task in workflow.task_list:
                        task.deadline = task.earliest_finish_time
                continue
                
            laxity = workflow.deadline - earliest_workflow_finish_time
            # print('laxity = {} - {} = {}'.format(workflow.deadline, earliest_workflow_finish_time, laxity))
            for task in workflow.task_list:
                task.laxity_execute_time = task.base_execute_time * factor + laxity * (task.base_confidence_execute_time * factor/(earliest_workflow_finish_time - self.current_time))
                # print('laxity time of {} : {}'.format(task.task_id, task.laxity_execute_time))
            
            # get deadline by laxity_execute_time from entry task to exit task
            for task in workflow.task_list:
                task.deadline = self.current_time + task.laxity_execute_time
                for pred_edge in task.pred_edge_list:
                    task.deadline = max(pred_edge.src_task.deadline + pred_edge.base_data_transfer_time + task.laxity_execute_time, task.deadline)
            break
        
        # for t in workflow.task_list:
        #     print('@@@ : ', t.task_id, t.deadline)
    
    def sortTaskList(self, workflow: Workflow):
        workflow.task_list.sort(key=lambda x: x.deadline)
        
    def sortReadyTaskList(self):
        self.ready_task_list.sort(key=lambda x: x.earliest_start_time)

    def updateTaskQueue(self, new_workflow: Workflow=None, finish_task: Task=None):
        assert new_workflow is not None or finish_task is not None
        if new_workflow is not None:
            for task in new_workflow.task_list:
                if len(task.pred_edge_list) == 0:
                    self.ready_task_list.append(task)
                else:
                    self.wait_task_list.append(task)

        else:
            assert not self.finish_task_list.__contains__(finish_task)
            for succ_edge in finish_task.succ_edge_list:
                tmp_ready = True
                for succ_pred in succ_edge.dst_task.pred_edge_list:
                    if succ_pred.src_task.finish_flag != True:
                        tmp_ready = False
                if tmp_ready:
                    self.ready_task_list.append(succ_edge.dst_task)
                    self.wait_task_list.remove(succ_edge.dst_task)
        
        if config.DEBUG:
            print('[Time {:.2f}] : length of ready_task_list: {}, length of wait_task_list: {}'.format(
                self.current_time,
                len(self.ready_task_list),
                len(self.wait_task_list),))
        
        self.sortReadyTaskList()
            

    def scheduleReadyTaskToMachine(self):
        for task in self.ready_task_list:
            self.scheduleSingleReadyTaskToMachine(task)
        # end for
        self.ready_task_list.clear()
        
    def scheduleSingleReadyTaskToMachine(self, task:Task):
        min_cost = MAX_INT
        real_start_time = MAX_INT
        confidence_start_time = MAX_INT
        target_vm: Vm = None
        
        # consider active vm
        for vm in self.active_vm_list:
            
            tmp_real_start_time = None
            tmp_confidence_start_time = None
            real_data_trans_time = 0
            confidence_data_trans_time = 0
            for pred_edge in task.pred_edge_list:
                assert pred_edge.src_task.allocated_vm != None
                # consider data transfer time when not on the same vm
                if pred_edge.src_task.allocated_vm is not vm:
                    real_data_trans_time = max(pred_edge.real_data_size/config.BAND_WIDTH, real_data_trans_time)
                    confidence_data_trans_time = max(pred_edge.confidence_data_size/config.BAND_WIDTH, confidence_data_trans_time)
            
            real_vm_available_time = self.current_time
            confidence_vm_available_time = self.current_time
            if vm.current_task is not None:
                confidence_vm_available_time = vm.current_task.confidence_finish_time
                real_vm_available_time = vm.current_task.real_finish_time
                if len(vm.wait_task_list) != 0 :
                    confidence_vm_available_time = vm.wait_task_list[-1].confidence_finish_time
                    real_vm_available_time = vm.wait_task_list[-1].real_finish_time
                    
            tmp_confidence_start_time = max(self.current_time + confidence_data_trans_time, confidence_vm_available_time)
            tmp_real_start_time = max(self.current_time + real_data_trans_time, real_vm_available_time)
            
            # case: -->-- current_time --->--- vm_available_time --->--- data_ready_time ---->    
            if tmp_confidence_start_time - confidence_vm_available_time > config.MAX_VM_IDLE_TIME:
                continue
            
            confidence_execute_time = task.base_confidence_execute_time * vm.factor
            confidence_finish_time = tmp_confidence_start_time + confidence_execute_time
            
            after_vm_time = confidence_finish_time - vm.start_time
            after_vm_unit = math.ceil(after_vm_time/3600)
            
            before_vm_time = confidence_vm_available_time - vm.start_time
            before_vm_unit = math.ceil(before_vm_time/3600)
            
            tmp_cost = (after_vm_unit - before_vm_unit)*vm.price
            
            if confidence_finish_time <= task.deadline and tmp_cost < min_cost:
                min_cost = tmp_cost
                target_vm = vm
                confidence_start_time = tmp_confidence_start_time
                real_start_time = tmp_real_start_time
        
        # if target_vm is not None:
        #     print('----- if {} to {}, cost = {}, confidence_start_time = {:.2f}'.format(task.task_id, target_vm.id, min_cost, confidence_start_time))
                
        # consider new vm
        tmp_cost = MAX_INT
        new_vm_type = None
        if min_cost > 0 or target_vm == None:
            real_data_trans_time = 0
            confidence_data_trans_time = 0
            for pred_edge in task.pred_edge_list:
                real_data_trans_time = max(pred_edge.real_data_size/config.BAND_WIDTH, real_data_trans_time)
                confidence_data_trans_time = max(pred_edge.confidence_data_size/config.BAND_WIDTH, confidence_data_trans_time)
            
            vm_type_list = sorted(self.vm_info.items(), key = lambda x:x[1]['factor'], reverse=True)
            for vm_type in vm_type_list:
                if self.current_time + confidence_data_trans_time + task.base_confidence_execute_time * vm_type[1]['factor'] <= task.deadline:
                    vm_unit = math.ceil(task.base_confidence_execute_time * vm_type[1]['factor'] / 3600)
                    tmp_cost = vm_unit * vm_type[1]['price']
                    new_vm_type = vm_type
                    break
            
            # Use the best vm to calculate the task in the worst case
            if new_vm_type == None:
                new_vm_type = vm_type_list[-1]
            
            # adding new vm is better
            if min_cost > tmp_cost or target_vm == None:
                target_vm = Vm(len(self.active_vm_list) + len(self.inactive_vm_list),
                                new_vm_type[0],
                                new_vm_type[1]['price'],
                                new_vm_type[1]['factor']
                            )
                target_vm.start_time = self.current_time
                self.active_vm_list.append(target_vm)
                if config.DEBUG:
                    print('[Time {:.2f}] : add a vm of type {}, id: {}'.format(self.current_time, target_vm.type, target_vm.id))
                # print('[Time {:.2f}] : add a vm of type {}, id: {}'.format(self.current_time, target_vm.type, target_vm.id))
                confidence_start_time = self.current_time + confidence_data_trans_time
                real_start_time = self.current_time + real_data_trans_time
        
        # schedule to target vm
        task.real_start_time = real_start_time
        assert real_start_time >= self.current_time
        task.real_execute_time = task.base_real_execute_time * target_vm.factor
        task.real_finish_time = task.real_start_time + task.real_execute_time
        
        task.confidence_start_time = confidence_start_time
        task.confidence_execute_time = task.base_confidence_execute_time * target_vm.factor
        task.confidence_finish_time = task.confidence_start_time + task.confidence_execute_time
        
        task.allocated_vm = target_vm
        
        target_vm.work_time += task.real_execute_time
        if target_vm.current_task is None:
            assert len(target_vm.wait_task_list) == 0
            target_vm.current_task = task
        else:
            target_vm.wait_task_list.append(task)
        
        if config.DEBUG:
            print('[Time {:.2f}] : schedule task {}({}) to vm {}, real_start_time: {:.2f}, real_finish_time: {:.2f}, vm.current_task: {}, vm.wait_task_list length: {}'.format(
                self.current_time, 
                task.task_id, 
                task.workflow.workflow_id, 
                target_vm.id,
                task.real_start_time,
                task.real_finish_time,
                target_vm.current_task.task_id,
                len(target_vm.wait_task_list)))
        # print(task.task_id, '->', task.allocated_vm.id, ':', task.real_start_time, task.real_finish_time)
        

    def updateEventTime(self):

        self.next_event_time: float = MAX_INT
        self.next_event_type = 'nothing'    
        self.next_event_vm = None    
        
        for vm in self.active_vm_list:
            # if busy vm
            if vm.current_task is not None:
                
                if self.next_event_time >= vm.current_task.real_finish_time:
                    self.next_event_time = vm.current_task.real_finish_time
                    self.next_event_type = 'finish'
                    self.next_event_vm = vm

            # if free vm
            else:
                tmp_next_vm_due_time = vm.start_time + math.ceil((self.current_time - vm.start_time) / 3600) * 3600

                if self.next_event_time >= tmp_next_vm_due_time:
                    self.next_event_time = tmp_next_vm_due_time
                    self.next_event_type = 'expire'
                    self.next_event_vm = vm
        
        if config.DEBUG:
            if self.next_event_type == 'nothing':
                print('[Time {:.2f}] : next event time : {:.2f}, event type: {}'.format(self.current_time, self.next_event_time, self.next_event_type))
            else:
                print('[Time {:.2f}] : next event time : {:.2f}, event type: {}, event vm : {}, and next_worklfow_arrival_time: {:.2f}'.format(self.current_time, self.next_event_time, self.next_event_type, self.next_event_vm.id, self.next_worklfow_arrival_time))
    
    def simulation(self, print_res=True):
        
        if print_res:
            print('{} scheduler is started'.format(self.name))
        
        skip_count = 0
        # for workflow_index in tqdm(range(0, config.WORKFLOW_NUM)):
        for workflow_index in range(0, len(self.workflow_list)):
            if skip_count > 0:
                skip_count -= 1
                continue
            
            tmp_workflow_list: list[Workflow] = []
            tmp_workflow_list.append(self.workflow_list[workflow_index])
            
            assert self.current_time <= self.workflow_list[workflow_index].arrival_time
            self.current_time = self.workflow_list[workflow_index].arrival_time

            # 同一时刻到达的工作流
            for tmp_workflow_index in range(workflow_index + 1, len(self.workflow_list)):
                if self.workflow_list[tmp_workflow_index].arrival_time == self.current_time:
                    tmp_workflow_list.append(self.workflow_list[tmp_workflow_index])
                    skip_count += 1
                else:
                    break

            # the time of next workflow arriving
            if workflow_index + skip_count != len(self.workflow_list) - 1:
                self.next_worklfow_arrival_time = self.workflow_list[workflow_index + skip_count + 1].arrival_time
            else:
                self.next_worklfow_arrival_time = MAX_INT

            for workflow in tmp_workflow_list:
                if config.DEBUG:
                    print('[Time {:.2f}] : workflow {} arrived'.format(self.current_time, workflow.workflow_id))
                self.preprocess(workflow)
                self.updateTaskQueue(new_workflow=workflow)

            self.scheduleReadyTaskToMachine()
            self.updateEventTime()

            # 按照事件时间来推进
            
            while self.next_event_time < self.next_worklfow_arrival_time:
                # 如果任务完成事件先发生
                if self.next_event_type == 'finish':
                
                    assert self.current_time <= self.next_event_time
                    self.current_time = self.next_event_time
                    
                    tmp_finish_task = self.next_event_vm.current_task
                    self.next_event_vm.current_task = None
                    tmp_finish_task.finish_flag = True
                    
                    self.time_out = False
                    tmp_finish_task.workflow.finish_time = max(tmp_finish_task.real_finish_time, tmp_finish_task.workflow.finish_time)
                    self.time_out = tmp_finish_task.workflow.finish_time > tmp_finish_task.workflow.deadline
                    
                    if config.DEBUG:
                        print('[Time {:.2f}] : vm {} completes the calculation of task {}({}), and the len of wait_task_list is {}'.format(
                            self.current_time, 
                            self.next_event_vm.id, 
                            tmp_finish_task.task_id, 
                            tmp_finish_task.workflow.workflow_id,
                            len(self.next_event_vm.wait_task_list)))
                    if len(self.next_event_vm.wait_task_list) != 0:
                        self.next_event_vm.current_task = self.next_event_vm.wait_task_list[0]
                        self.next_event_vm.wait_task_list.pop(0)

                    self.updateTaskQueue(finish_task=tmp_finish_task)
                    self.scheduleReadyTaskToMachine()
                    
                    self.finish_task_list.append(tmp_finish_task)
                
                # 如果是机器到期事件先发生
                elif self.next_event_type == 'expire':
                    
                    assert self.current_time <= self.next_event_time
                    self.current_time = self.next_event_time
                    expired_vm = self.next_event_vm
                    
                    expired_vm.end_time = self.current_time
                    expired_vm.use_time = expired_vm.end_time - expired_vm.start_time
                    expired_vm.total_cost = round(expired_vm.use_time/3600) * expired_vm.price
                    
                    expired_vm.active_flag = False
                    self.active_vm_list.remove(expired_vm)
                    self.inactive_vm_list.append(expired_vm)
                    if config.DEBUG:
                        print('[Time {:.2f}] : vm {} expired, work time: {:.0f}, total cost: {:.2f}'.format(
                            self.current_time, 
                            expired_vm.id, 
                            expired_vm.use_time,
                            expired_vm.total_cost))

                self.updateEventTime()
                
            # while end
                
        # while end
        assert len(self.active_vm_list) == 0
        return self.countUpExperimentResult(print_res=print_res)
    
    def countUpExperimentResult(self, print_res=True):
        total_cost = 0
        total_work_time = 0
        success_num = 0
        total_use_time = 0
        map_vm_type = {}
        for key in self.vm_info:
            map_vm_type[key] = 0
        for vm in self.inactive_vm_list:
            total_cost += vm.total_cost
            total_use_time += vm.use_time
            total_work_time += vm.work_time
            map_vm_type[vm.type] += round(vm.use_time/3600)
        
        for wf in self.workflow_list:
            if wf.finish_time <= wf.deadline:
                success_num += 1
        
        if print_res:
            print('algorithm name : {}'.format(self.name))
            print('success ratio : {:.2f}'.format(success_num/len(self.workflow_list)))
            print('totol cost : {:.2f}'.format(total_cost))
            print('utilization : {:.2f}'.format(total_work_time/total_use_time))
            print('vm count : {}'.format(map_vm_type))
            print()
        
        return [self.name, round(success_num/len(self.workflow_list), 2), round(total_cost, 2), round(total_work_time/total_use_time, 2), map_vm_type]