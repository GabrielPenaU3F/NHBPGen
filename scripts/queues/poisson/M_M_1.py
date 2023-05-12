from domain.processes.poisson_process import PoissonProcess
from domain.queue_system import QueueSystem

arrivals_process = PoissonProcess(0.8)
service_process = PoissonProcess(0.9)
serv_number = 1

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(20)

internal_history = q.export_states()



