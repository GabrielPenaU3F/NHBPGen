def generate_equally_spaced_observations(sample_path, step_length, number_of_steps):
    states = []
    for i in range(0, number_of_steps):
        time_marker = i * step_length
        states.append(count_events_until_time(sample_path, time_marker))
    return states


def count_events_until_time(sample_path, time_marker):
    return len([time for time in sample_path if time <= time_marker])


def chunk_array(arr, chunk_size):
    n = len(arr) // chunk_size  # Largest integer <= chunk_size
    trimmed_length = n * chunk_size
    trimmed_array = arr[:trimmed_length]
    return [trimmed_array[i:i + chunk_size] for i in range(0, trimmed_length, chunk_size)]

