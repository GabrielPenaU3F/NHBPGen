from domain.queues.arrival_adaptive.polya_fendick_n import ArrivalAdaptivePolyaFendickNQueue
from domain.queues.arrival_adaptive.polya_polya_n import ArrivalAdaptivePolyaPolyaNQueue

arrival_gamma = 0.1
arrival_beta = 1
service_gamma = 0.1
service_beta = 1
fendick_rho = 0.5

for i in range(1, 100):
    q = ArrivalAdaptivePolyaFendickNQueue(arrival_gamma, arrival_beta, service_gamma, service_beta, fendick_rho, 1)
    q.simulate_queue(1000)
    print(q.get_final_size())
