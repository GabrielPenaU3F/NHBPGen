from domain.queues.birth_death.bpm_bpm_n import BDBPMBPMNQueue

arrival_gamma = 2
arrival_beta = 5
service_gamma = 1
service_beta = 1

q = BDBPMBPMNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, 1)
q.plot_queue(1000)
