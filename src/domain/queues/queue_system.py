import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from exceptions import UninitializedQueueException


def queue_times(queue_tuple):
    return queue_tuple[0]


class QueueSystem:

    def __init__(self, arrival_process, service_process, n_servers=1):
        self.arrival_process = arrival_process
        self.service_process = service_process
        self.n_servers = n_servers
        self.states = None

    def simulate_queue(self, time):
        arrival_times = self.arrival_process.generate_arrivals(time)
        service_times = self.determine_service_times(time)
        arr_list = ['arrival' for i in range(len(arrival_times))]
        dep_list = ['departure' for i in range(len(service_times))]
        arrivals = tuple(zip(arrival_times, arr_list))
        departures = tuple(zip(service_times, dep_list))
        queue_events = sorted(arrivals + departures, key=queue_times)
        initial_state = 0
        queue_states = [tuple((0, initial_state))]
        for i in range(len(queue_events)):
            current_state = queue_states[-1][1]
            if queue_events[i][1] == 'arrival':
                new_state = current_state + 1
            elif queue_events[i][1] == 'departure' and current_state > 0:
                new_state = current_state - 1
            elif queue_events[i][1] == 'departure' and current_state == 0:
                new_state = current_state
            queue_states.append(tuple((queue_events[i][0], new_state)))

        self.states = list(zip(*queue_states))

    def plot_queue(self, time):
        if self.states is None:  # A.K.A. queue is not built yet
            self.simulate_queue(time)
        time_points, queue_state = self.states[0], self.states[1]
        plt.step(time_points, queue_state)
        plt.show()

    def determine_service_times(self, time):
        departures = []
        for k in range(self.n_servers):
            k_server_departure = self.service_process.generate_arrivals(time)
            departures = np.concatenate((departures, k_server_departure))
        return departures

    def get_final_size(self):
        if self.states is None:
            raise UninitializedQueueException('Queue has to be simulated first')
        return self.states[1][-1]

    def export_states(self, filename=None):
        if filename is not None:
            data = pd.DataFrame({'time': self.states[0],
                                 'state': self.states[1]})
            data.to_csv(filename, index=False)
        return self.states

