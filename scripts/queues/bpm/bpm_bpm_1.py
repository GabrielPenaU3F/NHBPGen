from domain.processes.bpm_process import BPMProcess
from domain.queues.queue_system import QueueSystem

arrivals_process = BPMProcess(1, 2)
service_process = BPMProcess(2, 2)
serv_number = 1

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(1000)
