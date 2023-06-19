from domain.queues.birth_death.polya_polya_n import PolyaPolyaNQueue

arrival_gamma = 10.8
arrival_beta = 10
service_gamma = 1.2
service_beta = 1

q = PolyaPolyaNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, 1)
q.plot_queue(1000)
