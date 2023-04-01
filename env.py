from scheduler import Scheduler
from workflow import Task, Workflow
import random
from vm import Vm
import config
import numpy as np
import math
import torch
import copy
import sys
MAX_INT = sys.maxsize

from typing import (
    Tuple,
)

class ScheduleEnv(Scheduler):
    def __init__(self, workflow_list, vm_info) -> None:
        super().__init__(workflow_list, vm_info)
                
        self.copy_workflow_list = copy.deepcopy(workflow_list)
        self.copy_vm_info = copy.deepcopy(vm_info)
        self.name = 'Rl'
        
        self.task_vm_record = []
        
        self.state = []
        self.action = -1
        
        self.current_workflow_index = 0
        
        self.print_res = False
    
    def seed(self, seed):
        np.random.seed(seed)
        
    def close():
        pass
    
    def reset(self) -> any:
        self.__init__(self.copy_workflow_list, self.copy_vm_info)
        self.copy_workflow_list = copy.deepcopy(self.copy_workflow_list)
        self.copy_vm_info = copy.deepcopy(self.copy_vm_info) 
        self.task_vm_record = []
        self.state = []
        self.action = -1
        self.current_workflow_index = 0
        self.print_res = False
        
        return self.nextEvent()

    def step(self, action) -> Tuple[any, float, bool, bool, dict]:
        # 继续调度就绪任务
        assert len(self.ready_task_list) > 0
        task = self.ready_task_list[0]
        self.ready_task_list.remove(task)
        return self.scheduleSingleReadyTaskToMachine(task, action)
        
    def nextEvent(self) -> any:
        # 没有就绪任务时按照事件时间推进
        self.updateEventTime()
        while self.next_event_time < self.next_worklfow_arrival_time:
            # 如果任务完成事件先发生
            if self.next_event_type == 'finish':
            
                assert self.current_time <= self.next_event_time
                self.current_time = self.next_event_time
                
                tmp_finish_task = self.next_event_vm.current_task
                self.next_event_vm.current_task = None
                tmp_finish_task.finish_flag = True
                tmp_finish_task.workflow.finish_num += 1
                if tmp_finish_task.workflow.finish_num == len(tmp_finish_task.workflow.task_list):
                    self.workflow_finish = 1
                
                if self.print_res: 
                    print('[{}] Task {} is finished at {}'.format(self.current_time, tmp_finish_task.task_id, self.current_time))
                
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
                self.finish_task_list.append(tmp_finish_task)
                if len(self.ready_task_list) > 0:
                    return self.getCurrentState(self.ready_task_list[0])
                
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
        
        # next_worklfow_arrival_time > next_event_time
        return self.nextWorkflow()

    def nextWorkflow(self) -> any:
        skip_count = 0
        for workflow_index in range(self.current_workflow_index, config.WORKFLOW_NUM):
            
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
            
            self.current_workflow_index = workflow_index + 1 + skip_count
            break
        
        if len(self.ready_task_list) > 0:
            return self.getCurrentState(self.ready_task_list[0])
        else:
            # print('{} scheduler is Done'.format(self.name))
            if self.print_res:
                self.countUpExperimentResult()
            return self.getCurrentState(None)
    
    def getCurrentState(self, task) -> any:
        
        denominator = 3600
        
        if task is None:
            return [[0, 0, 0, 0], self.vms_state]
        
        self.vms_state = []
        self.task_vm_record = []
        
        confidence_data_trans_time_to_newvm = 0
        
        for vm in self.active_vm_list:
            real_start_time = None
            confidence_start_time = None
            real_data_trans_time = 0
            confidence_data_trans_time = 0
            
            for pred_edge in task.pred_edge_list:
                assert pred_edge.src_task.allocated_vm != None
                # consider data transfer time when not on the same vm
                confidence_data_trans_time_to_newvm = max(pred_edge.confidence_data_size/config.BAND_WIDTH, confidence_data_trans_time_to_newvm)
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
            vm.confidence_vm_available_time = confidence_vm_available_time
                    
            confidence_start_time = max(self.current_time + confidence_data_trans_time, confidence_vm_available_time)
            confidence_execute_time = task.base_confidence_execute_time * vm.factor
            confidence_finish_time = confidence_start_time + confidence_execute_time
            
            real_start_time = max(self.current_time + real_data_trans_time, real_vm_available_time)
            
            after_vm_time = confidence_finish_time - vm.start_time
            after_vm_unit = math.ceil(after_vm_time/3600)
            
            before_vm_time = confidence_vm_available_time - vm.start_time
            before_vm_unit = math.ceil(before_vm_time/3600)
            
            unit_num = after_vm_unit - before_vm_unit
            cost = (after_vm_unit - before_vm_unit)*vm.price
            
            self.task_vm_record.append([real_start_time, confidence_start_time, cost, unit_num])
            
            
            # one-hot, 
            # finish time, # 相对于 current time
            # cost unit, 
            # remain
            
            a1 = 1 if vm.factor == 8.0 else 0
            a2 = 1 if vm.factor == 4.0 else 0
            a3 = 1 if vm.factor == 2.0 else 0
            a4 = 1 if vm.factor == 1.0 else 0
            
            self.vms_state.append([a1, a2, a3, a4,
                                (confidence_finish_time - self.current_time)/denominator, 
                                unit_num, 
                                (3600 - (after_vm_time % 3600))/denominator
                               ])
        
        # 't2.small' : {'price' : 0.023, 'factor' : 8.0 },
        # 't2.medium' : {'price' : 0.0464, 'factor' : 4.0 },
        # 't2.large' : {'price' : 0.0928, 'factor' : 2.0 },
        # 't2.xlarge' : {'price' : 0.1856, 'factor' : 1.0 }
        
        # new_finish_time = [(self.current_time + confidence_data_trans_time_to_newvm + task.base_confidence_execute_time * factor) for factor in [8,4,2,1]]
        new_cost_time = [(confidence_data_trans_time_to_newvm + task.base_confidence_execute_time * factor) for factor in [8,4,2,1]]
        
        
        self.vms_state.append([1,0,0,0, 
                                new_cost_time[0]/denominator, 
                                math.ceil(new_cost_time[0]/denominator), 
                                (3600 - (new_cost_time[0] % 3600))/denominator]
                              )     
        self.vms_state.append([0,1,0,0, 
                                new_cost_time[1]/denominator, 
                                math.ceil(new_cost_time[1]/denominator), 
                                (3600 - (new_cost_time[1] % 3600))/denominator]
                              )
        self.vms_state.append([0,0,1,0, 
                                new_cost_time[2]/denominator, 
                                math.ceil(new_cost_time[2]/denominator), 
                                (3600 - (new_cost_time[2] % 3600))/denominator]
                              )
        self.vms_state.append([0,0,0,1, 
                                new_cost_time[3]/denominator, 
                                math.ceil(new_cost_time[3]/denominator), 
                                (3600 - (new_cost_time[3] % 3600))/denominator]
                              )
        
        task_state = [
                # task.base_execute_time/denominator, # todo
                (task.workflow.deadline - self.current_time)/denominator, 
                task.rank_mpeft/denominator, 
                task.cp/denominator
            ]
        
        self.state = [
            # task state
            task_state,
            # vm state
            self.vms_state
        ]
        # print('caculate task-vm for task {}'.format(task.task_id))
        return self.state
    
    def scheduleSingleReadyTaskToMachine(self, task:Task, action):
        assert self.task_vm_record is not None
        
        if self.print_res:
            print('[{}] Scheduling Task {}, exe time {}'.format(self.current_time, task.task_id, task.base_confidence_execute_time))
        
        # schedule a task
        real_start_time = None
        confidence_start_time = None
        target_vm = None
        predict_cost = 0
        info_pre = ''
        
        # add new
        if action >= len(self.active_vm_list):
            info_pre = '[new] '
            if action == len(self.active_vm_list):
                target_vm = Vm(len(self.active_vm_list) + len(self.inactive_vm_list),
                                    't2.small', 0.0230, 8)
            if action == len(self.active_vm_list)+1:
                target_vm = Vm(len(self.active_vm_list) + len(self.inactive_vm_list),
                                    't2.medium', 0.0464, 4)
            if action == len(self.active_vm_list)+2:
                target_vm = Vm(len(self.active_vm_list) + len(self.inactive_vm_list),
                                    't2.large', 0.0928, 2)
            if action == len(self.active_vm_list)+3:
                target_vm = Vm(len(self.active_vm_list) + len(self.inactive_vm_list),
                                    't2.xlarge', 0.1856, 1)
                
            unit_num = math.ceil((task.base_confidence_execute_time * target_vm.factor)/3600)
            predict_cost = target_vm.price * unit_num
            predict_cost_second = task.base_confidence_execute_time * target_vm.factor / 3600
            
            confidence_data_trans_time = 0
            real_data_trans_time = 0
            for pred_edge in task.pred_edge_list:
                real_data_trans_time = max(pred_edge.real_data_size/config.BAND_WIDTH, real_data_trans_time)
                confidence_data_trans_time = max(pred_edge.confidence_data_size/config.BAND_WIDTH, confidence_data_trans_time)
                
            target_vm.start_time = self.current_time
            self.active_vm_list.append(target_vm)
            confidence_start_time = self.current_time + confidence_data_trans_time
            real_start_time = self.current_time + real_data_trans_time
            
        else:
            target_vm = self.active_vm_list[action]
            real_start_time = self.task_vm_record[action][0]
            confidence_start_time = self.task_vm_record[action][1]
            predict_cost = self.task_vm_record[action][2]
            predict_cost_second = task.base_confidence_execute_time * target_vm.factor / 3600
            unit_num = self.task_vm_record[action][3]
            
        task.real_start_time = real_start_time
        assert real_start_time >= self.current_time
        task.real_execute_time = task.base_real_execute_time * target_vm.factor
        task.real_finish_time = task.real_start_time + task.real_execute_time
        
        task.confidence_start_time = confidence_start_time
        task.confidence_execute_time = task.base_confidence_execute_time * target_vm.factor
        task.confidence_finish_time = task.confidence_start_time + task.confidence_execute_time
        target_vm.confidence_vm_available_time = task.confidence_finish_time
        
        task.allocated_vm = target_vm
        
        target_vm.work_time += task.real_execute_time
        if target_vm.current_task is None:
            assert len(target_vm.wait_task_list) == 0
            target_vm.current_task = task
        else:
            target_vm.wait_task_list.append(task)
        
        
        punish = 0
        done = False
        # if task.confidence_finish_time > task.workflow.deadline:
        #     # punish = len(task.workflow.task_list) - task.workflow.finish_num
        #     # punish = 1
        #     punish = (task.confidence_finish_time - task.workflow.deadline) / 3600
        #     # done = True

        if task.confidence_finish_time > task.deadline:
            # punish = len(task.workflow.task_list) - task.workflow.finish_num
            punish = 1
            # punish = (task.confidence_finish_time - task.workflow.deadline) / 3600
            # done = True
            
        wf_finish = 0
        if task is task.workflow.task_list[-1]:
            wf_finish = 1
        
        # reward = self.workflow_finish - (punish + predict_cost_second)
        reward = wf_finish - (punish + predict_cost)
        
        info = ['schedule task {} to {}vm {}({}), predict cost: {}({}*{})'.format(
                    task.task_id, info_pre, id(target_vm), target_vm.type, predict_cost, unit_num, target_vm.price),
                'task start time: {}, caculate time: {}, finish time: {}, time out: {}'.format(
                    task.confidence_start_time, task.confidence_execute_time, task.confidence_finish_time, self.time_out),
                'vm start time: {}, vm avalible time: {}, vm remain time: {}'.format(
                    target_vm.start_time, target_vm.confidence_vm_available_time, 3600 - (task.confidence_finish_time - target_vm.start_time)%3600),
                'reward = wf_finish({}) - (punish({}) + cost({})) = {}'.format(
                    wf_finish, punish, predict_cost, reward)
                ]
        
        if self.print_res:
            print('active vm num: {}, avalible vm num: {}, target: {}'.format(len(self.active_vm_list), len(self.state[1]), action))
            for i in range(len(self.state[1])):
                s = '*' if i == action else ''
                print('    {}vm id: {}, finish time: {}, cost num: {}, remain:{}'.format(
                    s, i, self.state[1][i][4], self.state[1][i][5], self.state[1][i][6]))
            
            for i in info:
                print(i)
            print('----------------------')
                
        self.task_vm_record = None
        
        
        info = ['schedule task {} to {}vm {}({}), predict cost: {}({}*{})'.format(
                    task.task_id, info_pre, id(target_vm), target_vm.type, predict_cost, unit_num, target_vm.price),
                'task start time: {}, caculate time: {}, finish time: {}, time out: {}'.format(
                    task.confidence_start_time, task.confidence_execute_time, task.confidence_finish_time, self.time_out),
                'vm start time: {}, vm avalible time: {}, vm remain time: {}'.format(
                    target_vm.start_time, target_vm.confidence_vm_available_time, 3600 - (task.confidence_finish_time - target_vm.start_time)%3600),
                'reward = wf_finish({}) - (punish({}) + cost({})) = {}'.format(
                    wf_finish, punish, predict_cost, reward)
                ]
        
        # update state
        if len(self.ready_task_list) == 0:
            observation = self.nextEvent()
        else:
            observation = self.getCurrentState(self.ready_task_list[0])
        if self.print_res and len(self.ready_task_list) > 0:
            print('next task {} to schedule'.format(self.ready_task_list[0].task_id))
        
        terminated = (len(self.active_vm_list) == 0) or done
        truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def sortReadyTaskList(self):
        self.ready_task_list.sort(key=lambda x: x.rank_mpeft)
    
    def preprocess(self, workflow: Workflow):
        self.calculateTaskRankOfMpeft(workflow)
        self.calculateTaskCp(workflow)
        self.sortTaskList(workflow)
        self.calculateTaskConfidenceTime(workflow)
        self.calculateTaskDeadline(workflow)
    
    def sortTaskList(self, workflow: Workflow):
        workflow.task_list.sort(key=lambda x: x.rank_mpeft, reverse=True)
    
    def calculateTaskRankOfMpeft(self, workflow: Workflow):
        for task in reversed(workflow.task_list):
            task.all_succ = set([task])
            task.direct_time: float = task.base_execute_time
            
            for succ_edge in task.succ_edge_list:
                task.direct_time += succ_edge.base_data_transfer_time
                task.all_succ |= succ_edge.dst_task.all_succ
        
        for task in workflow.task_list:
            task.rank_mpeft = 0
            for t in task.all_succ:
                task.rank_mpeft += t.direct_time
    
    def calculateTaskCp(self, workflow: Workflow):
        for task in reversed(workflow.task_list):
            task.cp = 0
            for succ_edge in task.succ_edge_list:
                task.cp = max(succ_edge.base_data_transfer_time + succ_edge.dst_task.base_execute_time, task.cp)
    
    # core
    def calculateTaskDeadline(self, workflow: Workflow):
        entry_pur = -1
        for task in reversed(workflow.task_list):
            task.probabilistic_upward_rank = task.base_execute_time
            prefix = 0
            for succ_edge in task.succ_edge_list:
                eta = 0 if math.pow(1.5, -(task.base_execute_time/succ_edge.base_data_transfer_time)) < random.random() else 1
                prefix = max(succ_edge.dst_task.probabilistic_upward_rank + eta * succ_edge.base_data_transfer_time, prefix)
            task.probabilistic_upward_rank += prefix
            if len(task.pred_edge_list) == 0:
                entry_pur = max(task.probabilistic_upward_rank, entry_pur)
            
        for task in workflow.task_list:
            task.deadline = workflow.arrival_time + (workflow.deadline - workflow.arrival_time) * (entry_pur - task.probabilistic_upward_rank + task.base_execute_time) / entry_pur