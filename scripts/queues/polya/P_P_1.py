from domain.processes.polya_process import PolyaProcess
from domain.queues.queue_system import QueueSystem

arrivals_process = PolyaProcess(1, 0.9)
service_process = PolyaProcess(1, 0.9)
serv_number = 8

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(2000)
