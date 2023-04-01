from scheduler import Scheduler

class ROSA(Scheduler):
    def __init__(self, workflow_list, vm_info) -> None:
        super().__init__(workflow_list, vm_info)
        self.name = 'ROSA'