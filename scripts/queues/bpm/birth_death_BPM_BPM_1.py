from domain.queues.birth_death.bpm_bpm_n import BDBPMBPMNQueue
from domain.queues.birth_death.polya_polya_n import BDPolyaPolyaNQueue

arrival_gamma = 12
arrival_beta = 10
service_gamma = 1.2
service_beta = 1

q = BDBPMBPMNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, 1)
q.plot_queue(10000)
