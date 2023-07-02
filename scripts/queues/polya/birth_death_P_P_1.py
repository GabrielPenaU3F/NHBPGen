from domain.queues.birth_death.polya_polya_n import BDPolyaPolyaNQueue

arrival_gamma = 1
arrival_beta = 1
service_gamma = 2
service_beta = 2

q = BDPolyaPolyaNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, 1)
q.plot_queue(1000)
