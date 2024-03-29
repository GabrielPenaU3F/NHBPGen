from domain.processes.polya_process import PolyaProcess
from domain.queues.queue_system import QueueSystem

arrivals_process = PolyaProcess(1, 2)
service_process = PolyaProcess(3, 6)
serv_number = 1

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(1000)
