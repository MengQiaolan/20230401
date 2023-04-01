from workflow import Task

class Vm(object):
    def __init__(self, id: int, type: str, price: float, factor: float) -> None:
        self.id = id
        self.type = type
        self.price = price
        self.factor = factor
        
        
        self.wait_task_list: list[Task] = []
        self.current_task: Task = None

        self.start_time: float = None
        self.end_time: float = None

        self.confidence_vm_available_time = None

        self.work_time: float = 0
        self.use_time: float = 0
        self.total_cost: float = None

        self.active_flag = None
        pass