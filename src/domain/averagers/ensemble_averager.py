import numpy as np
from scipy.stats import linregress

from domain.averagers.averaging_strategies.ensemble_averaging_strategies import RegularEnsembleAveragingStrategy, \
    SquareEnsembleAveragingStrategy
from domain.sampler.sampler import Sampler


class EnsembleAverager:

    strategies = {
        'regular': RegularEnsembleAveragingStrategy,
        'sq': SquareEnsembleAveragingStrategy,
    }

    # N is the total of simulations to run
    # T is the total time units to simulate
    # Time step is the timestep between sucessive observations, in time units. Default is 1
    def average(self, ensemble, average_type='regular'):
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        return strategy_class().calculate(ensemble)

    def estimate_hurst(self, ensemble, T, time_step):
        average = self.average(ensemble, average_type='sq')
        t = np.linspace(0, T, int(T / time_step))
        log_t = np.log(t[1:])
        log_avg = np.log(average[1:int(T / time_step)])
        slope, intercept, r_value, p_value, std_err = linregress(log_t, log_avg)
        H = slope/2
        print(f'Hurst parameter: H={H}')
        return H
