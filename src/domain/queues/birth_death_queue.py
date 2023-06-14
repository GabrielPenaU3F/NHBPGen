from domain.queues.queue_system import QueueSystem


class BirthDeathQueue(QueueSystem):

    def __init__(self, arrival_process, service_process, n_servers=1):
        super().__init__(arrival_process, service_process, n_servers)

    def simulate_queue(self, time):

