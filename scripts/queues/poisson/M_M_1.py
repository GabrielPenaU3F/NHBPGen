from domain.processes.poisson_process import PoissonProcess
from domain.queues.queue_system import QueueSystem

arrivals_process = PoissonProcess(0.7)
service_process = PoissonProcess(0.7)
serv_number = 1

q = QueueSystem(arrivals_process, service_process, serv_number)
q.plot_queue(1000)

internal_history = q.export_states()



