from abc import abstractmethod

import numpy as np

from domain.queues.queue_system import QueueSystem


class BirthDeathQueue(QueueSystem):

    def __init__(self, arrival_process, service_process, n_servers):
        super().__init__(arrival_process, service_process, n_servers)

    def simulate_queue(self, time):
        initial_state = 0
        queue_states = [tuple((0, initial_state))]
        present_time = 0
        queue_state = 0
        while present_time < time:
            # First of all update the time
            present_time = self.generate_next_transition_time(queue_state, present_time)
            # The conditional probability of a birth must be calculated before updating state, because the system still
            # does not know if it is a birth or a death
            current_lambda = self.arrival_process.intensity_function(queue_state, present_time)
            current_mu = self.service_process.intensity_function(queue_state, present_time)
            prob_birth = current_lambda / (current_lambda + current_mu)
            event_indicator = np.random.binomial(1, prob_birth)
            if event_indicator == 1 or queue_state == 0:    # Birth
                queue_state += 1
            else:   # Death
                queue_state -= 1
            queue_states.append(tuple((present_time, queue_state)))
            self.states = list(zip(*queue_states))

    @abstractmethod
    def generate_next_transition_time(self, current_state, present_time):
        pass
