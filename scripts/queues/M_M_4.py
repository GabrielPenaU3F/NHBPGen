from domain.processes.poisson_process import PoissonProcess
from domain.queue_system import QueueSystem

arrivals_process = PoissonProcess(0.7)
service_process = PoissonProcess(0.2)
serv_number = 4

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(20)
