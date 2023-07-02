from domain.queues.arrival_adaptive.polya_polya_n import ArrivalAdaptivePolyaPolyaNQueue

arrival_gamma = 0.05
arrival_beta = 0.1
service_gamma = 0.1
service_beta = 1

q = ArrivalAdaptivePolyaPolyaNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, 1)
q.plot_queue(2000)
