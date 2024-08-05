import numpy as np
from matplotlib import pyplot as plt

from domain.processes.bpm_3p_process import BPM3pProcess
from domain.processes.bpm_process import BPMProcess
from domain.processes.fendick_process import FendickProcess
from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from domain.processes.yule_process import YuleProcess
from exceptions import SimulationException


class Sampler:

    def generate_arrivals_sample_path(self, model, time, plot=True):
        if not time > 0:
            raise SimulationException('Duration of the simulation must be a positive number')
        arrivals = model.generate_arrivals(time)
        if plot is True:
            x_times = np.concatenate(([0], arrivals, [time]))
            fig, axes = plt.subplots(figsize=(12, 5))
            steps = np.arange(model.get_initial_state(), model.get_initial_state() + len(arrivals) + 1)
            steps = np.append(steps, steps[-1])
            axes.step(x_times, steps, where='post')
            plt.show()
        return arrivals
