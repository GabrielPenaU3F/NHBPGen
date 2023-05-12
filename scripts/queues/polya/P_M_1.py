from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queue_system import QueueSystem

arrivals_process = PolyaProcess(1, 0.9)
service_process = PoissonProcess(0.8)
serv_number = 1

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(200)
