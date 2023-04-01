import yaml
import numpy
import copy
import config
from tqdm import tqdm

class Task():
    def __init__(self, task_id, workflow, base_execute_time, base_real_execute_time) -> None:
        
        # init
        self.task_id = task_id   
        self.workflow: Workflow = workflow
        self.base_execute_time: float = base_execute_time
        self.base_real_execute_time: float = base_real_execute_time
        
        self.pred_edge_list: list[Edge] = []
        self.succ_edge_list: list[Edge] = []
        
        # set by algorithm
        self.base_confidence_execute_time: float = None
        
        
        self.laxity_execute_time: float = None
        self.deadline: float = None

        self.earliest_start_time: float = None
        self.earliest_finish_time: float = None
        self.latest_start_time: float = None
        self.latest_finish_time: float = None
        
        self.rank = None

        
        self.finish_flag = False
           
        self.allocated_vm = None
        self.confidence_start_time: float = None
        self.confidence_execute_time: float = None
        self.confidence_finish_time: float = None
        self.real_start_time: float = None
        self.real_execute_time: float = None
        self.real_finish_time: float = None
        pass

class Edge(object):
    def __init__(self, src_task_id, dst_task_id, base_data_size) -> None:
        # init
        self.base_data_size = base_data_size
        self.base_data_transfer_time = self.base_data_size / config.BAND_WIDTH
        
        self.src_task_id = src_task_id
        self.dst_task_id = dst_task_id
        
        self.src_task: Task = None
        self.dst_task: Task = None
        
        # set by algorithm
        self.confidence_data_size = None
        self.confidence_data_transfer_time = None
        self.real_data_size = None
        

class Workflow(object):
    def __init__(self, workflow_id, workflow_name, arrival_time, makespan, deadline) -> None:
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.arrival_time: float = arrival_time
        
        # the max EFT of tasks
        self.makespan: float = makespan
        # deadline = arrival_time + makespan * factor
        self.deadline: float = deadline
        
        self.task_list: list[Task] = []
        
        self.finish_num = 0
        self.finish_time = -1
        
    def topySortTaskList(self):
        sorted_task_list = []
        sorted_task_map = {}
        task_num = len(self.task_list)
        while len(sorted_task_list) < task_num:
            for task in self.task_list:
                ready_flag = True
                if len(task.pred_edge_list) != 0:
                    for pred_edge in task.pred_edge_list:
                        if not sorted_task_map.__contains__(pred_edge.src_task_id):
                            ready_flag = False
                            break
                if ready_flag:
                    sorted_task_list.append(task)
                    sorted_task_map[task.task_id] = True
                    self.task_list.remove(task)
                    break
        
        assert len(self.task_list) == 0
        self.task_list = sorted_task_list
    
    def calculateBaseMakespan(self):
        for task in self.task_list:
            task.deadline = task.base_execute_time
            for pred in task.pred_edge_list:
                task.deadline = max(pred.src_task.deadline + pred.base_data_transfer_time + task.base_execute_time, task.deadline)
        self.makespan = self.task_list[-1].deadline


def WorkflowProducer(template_path=None, 
                     WORKFLOW_NUM=config.WORKFLOW_NUM, 
                     ARRIVAL_LAMBDA=config.ARRIVAL_LAMBDA, 
                     DEADLINE_FACTOR=config.DEADLINE_FACTOR):
    
    if template_path is None:
        with open('template.yml','r',encoding='utf-8') as file:
            template_yaml_workflow_list = yaml.load(stream=file,Loader=yaml.FullLoader)
    else:
        with open(template_path,'r',encoding='utf-8') as file:
            template_yaml_workflow_list = yaml.load(stream=file,Loader=yaml.FullLoader)
    
    # templateWorkflow {'workflowName': ..., 'taskList': [task...]}
    # task {
        # 'taskId': '...', 
        # 'taskWorkFlowId': '...', 
        # 'parentTaskList': [{'taskId': '...', 'dataSize': '...'}...], 
        # 'taskRunTime': 0.06, 
        # 'successorTaskList': []
    #  }
    # print(template_yaml_workflow_list[0]['taskList'][0])
    
    template_workflow_list: list[Workflow] = []
    template_workflow_dict = {}
    for template_yaml_workflow in template_yaml_workflow_list:
        
        # remove Epigenomics
        if 'Epigenomics' in template_yaml_workflow['workflowName']:
            continue
        
        new_workflow = Workflow(-1, template_yaml_workflow['workflowName'], -1, -1, -1)
        task_list: list[Task] = []
        template_yaml_task_list = template_yaml_workflow['taskList']
        
        for template_yaml_task in template_yaml_task_list:
            new_task = Task(template_yaml_task['taskId'], 
                            new_workflow, 
                            template_yaml_task['taskRunTime'],
                            template_yaml_task['taskRunTime'])
            
            for succ in template_yaml_task['successorTaskList']:
                new_task.succ_edge_list.append(
                    Edge(template_yaml_task['taskId'], succ['taskId'], succ['dataSize']))
        
            task_list.append(new_task)
        
        for i in range(0, len(task_list)):
            task = task_list[i]
            for succ_edge in task.succ_edge_list:
                for j in range(0, len(task_list)):
                    succ_task = task_list[j]
                    if succ_task.task_id == succ_edge.dst_task_id:
                        succ_task.pred_edge_list.append(succ_edge)
                        succ_edge.src_task = task
                        succ_edge.dst_task = succ_task
                        break
    
        
        new_workflow.task_list = task_list
        template_workflow_list.append(new_workflow)
        template_workflow_dict[new_workflow.workflow_name] = 0
        # print(new_workflow.workflow_name)
    
    assert len(template_workflow_list) == 12  # 12
    
    for template_workflow in template_workflow_list:
        template_workflow.topySortTaskList()
        template_workflow.calculateBaseMakespan()
        
    generated_workflow_list = []
    arrival_time = -1
    skip_count = 0
    #for _ in tqdm(range(0, WORKFLOW_NUM)):
    for _ in range(0, WORKFLOW_NUM):
        if skip_count > 0:
            skip_count -= 1
            continue
    
        arrival_time += 1
        arrival_num = numpy.random.poisson(ARRIVAL_LAMBDA)
        while arrival_num == 0:
            arrival_time += 1
            arrival_num = numpy.random.poisson(ARRIVAL_LAMBDA)
        
        skip_count += (arrival_num - 1)
        for i in range(0, arrival_num):
            if len(generated_workflow_list) == WORKFLOW_NUM:
                break
            template_workflow = template_workflow_list[numpy.random.randint(0,len(template_workflow_list))]
            # template_workflow = template_workflow_list[numpy.random.randint(3,6)]
            template_workflow_dict[template_workflow.workflow_name] += 1
            generated_workflow = Workflow(len(generated_workflow_list), 
                                          template_workflow.workflow_name, 
                                          arrival_time, 
                                          template_workflow.makespan, 
                                          template_workflow.makespan * DEADLINE_FACTOR + arrival_time)
            generated_task_list: list[Task] = []
            for template_task in template_workflow.task_list:
                
                tmp_base_real_execute_time = numpy.random.normal(template_task.base_execute_time, template_task.base_execute_time * config.STANDARD_DEVIATION_OF_EXECUTE_TIME)
                if tmp_base_real_execute_time <= 0:
                    tmp_base_real_execute_time = template_task.base_execute_time
                
                generated_task = Task(template_task.task_id, 
                                      generated_workflow,
                                      template_task.base_execute_time,
                                      tmp_base_real_execute_time)
                
                for succ_edge in template_task.succ_edge_list:
                    generated_task.succ_edge_list.append(copy.deepcopy(succ_edge))
                
                generated_task_list.append(generated_task)
            
            # get pred list of each task by succ list
            for i in range(0, len(generated_task_list)):
                task = generated_task_list[i]
                for succ_edge in task.succ_edge_list:
                    for j in range(i+1, len(generated_task_list)):
                        succ_task = generated_task_list[j]
                        if succ_task.task_id == succ_edge.dst_task_id:
                            succ_task.pred_edge_list.append(succ_edge)
                            succ_edge.src_task = task
                            succ_edge.dst_task = succ_task
                            succ_edge.real_data_size = numpy.random.normal(succ_edge.base_data_size, 
                                                                           succ_edge.base_data_size * config.STANDARD_DEVIATION_OF_DATA_SIZE)
                            if succ_edge.real_data_size <= 0:
                                succ_edge.real_data_size = succ_edge.base_data_size
                            break
            
            generated_workflow.task_list = generated_task_list
            generated_workflow_list.append(generated_workflow)
    assert len(generated_workflow_list) == WORKFLOW_NUM
    # print(template_workflow_dict)
    return generated_workflow_list

def WorkflowProducerExample():
    
    new_workflow = Workflow(
        workflow_id=0, 
        workflow_name='example', 
        arrival_time=0,
        makespan=-1,
        deadline=-1
    )
    task_list: list[Task] = []
    
    for index in range(5):
        new_task = Task(task_id=index, 
                        workflow=new_workflow, 
                        base_execute_time=1000 * (index+1),
                        base_real_execute_time=1000 * (index+1))
        
        if index == 0:
            new_task.succ_edge_list.append(Edge(index, index+1, (index+1) * 100000000))
            new_task.succ_edge_list.append(Edge(index, index+2, (index+1) * 100000000))
            new_task.succ_edge_list.append(Edge(index, index+3, (index+1) * 100000000))
        elif index != 4:
            new_task.succ_edge_list.append(Edge(index, 4, index * 100))
            
        task_list.append(new_task)
    
    for i in range(0, len(task_list)):
        task = task_list[i]
        for succ_edge in task.succ_edge_list:
            for j in range(0, len(task_list)):
                succ_task = task_list[j]
                if succ_task.task_id == succ_edge.dst_task_id:
                    succ_task.pred_edge_list.append(succ_edge)
                    succ_edge.src_task = task
                    succ_edge.dst_task = succ_task
                    succ_edge.real_data_size = succ_edge.base_data_size
                    break
    
    # for t in task_list:
    #     print(t.task_id, ' succ :')
    #     for e in t.succ_edge_list:
    #         print(e.src_task_id, '->', e.dst_task_id, e.base_data_size, e.confidence_data_size)
        
    #     print(t.task_id, ' pred :')
    #     for e in t.pred_edge_list:
    #         print(e.src_task_id, '->', e.dst_task_id, e.base_data_size, e.confidence_data_size)
    
    new_workflow.task_list = task_list
    new_workflow.topySortTaskList()
    new_workflow.calculateBaseMakespan()
    new_workflow.deadline = config.DEADLINE_FACTOR * new_workflow.makespan
    
    return [new_workflow]