import numpy as np
from matplotlib import pyplot as plt

from domain.processes.polya_process import PolyaProcess

total_time = 1000

# A message is sent every 100 seconds on a regular basis
periodic_messages = np.arange(0, total_time, 100)

# We'll simulate 3 accidents
# Random accidents will occur at a rate of 1 each 500 seconds
n_accidents = 3
accident_rate = 1/500
accident_times = np.random.exponential(scale=1/accident_rate, size=n_accidents)
cumulative_accident_times = np.cumsum(accident_times)

# The duration of the safe failure regime is exponential with mean 50 seconds
safe_failure_duration_rate = 1/50
safe_failure_durations = np.random.exponential(scale=1/safe_failure_duration_rate, size=n_accidents)

arrival_matrix = []
for time in safe_failure_durations:
    # Whenever an accident occurs, a Polya(1, 1) process is triggered,
    # i.e., there is a cascade of messages according to a Polya regime
    proc = PolyaProcess(1, 1)
    arrivals = proc.generate_arrivals(time, show=False)
    arrival_matrix.append(arrivals)
arrival_matrix = np.array(arrival_matrix)

# Now add all the traffic together
traffic = periodic_messages
for k in range(len(cumulative_accident_times)):
    accident_time = cumulative_accident_times[k]
    message_times = arrival_matrix[k] + accident_time
    message_times = np.insert(message_times, 0, accident_time)
    traffic = np.concatenate((traffic, message_times))

# Sort the times
traffic = np.sort(traffic)
# Omit times that happened over the bound
traffic = list(filter(lambda x: x <= total_time, traffic))
ones = np.ones(len(traffic))

fig, ax = plt.subplots()
plt.stem(traffic, ones)
plt.show()
