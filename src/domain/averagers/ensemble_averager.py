import numpy as np

from domain.averagers.averaging_strategies.ensemble_averaging_strategies import RegularEnsembleAveragingStrategy, \
    SquareEnsembleAveragingStrategy
from domain.sampler import Sampler


class EnsembleAverager:

    strategies = {
        'regular': RegularEnsembleAveragingStrategy,
        'sq': SquareEnsembleAveragingStrategy,
    }

    # N is the total of simulations to run
    # T is the total time units to simulate
    # Time step is the timestep between sucessive observations, in time units. Default is 1
    def average(self, model, N, T, time_step=1, average_type='regular'):
        ensemble = self.generate_ensemble(model, N, T, time_step)
        strategy_class = self.strategies.get(average_type)
        if not strategy_class:
            raise ValueError(f"Unknown average type: {average_type}")

        return strategy_class().calculate(ensemble)

    def generate_ensemble(self, model, N, T, time_step):
        ensemble = []
        for i in range(N):
            sample_path = Sampler().generate_observations_sample_path(model, T, time_step, plot=False)
            ensemble.append(sample_path)
        return ensemble
