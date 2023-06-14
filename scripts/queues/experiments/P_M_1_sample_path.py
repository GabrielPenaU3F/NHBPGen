from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.queues.birth_death_queue import BirthDeathQueue

n_queues = 1000
beta, gamma = 2, 0.8
arrival_process = PolyaProcess(gamma, beta)
service_process = PoissonProcess(1)
n_servers = 1
time = 1000

q = BirthDeathQueue(arrival_process, service_process, n_servers)
q.simulate_queue(time)
q_states = q.export_states()