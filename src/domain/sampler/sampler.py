import numpy as np
from matplotlib import pyplot as plt

from domain.extra_functions import extra_functions
from domain.processes.nhbp import NHBP
from domain.sampler.sampler_strategies import ArrivalsSamplePathStrategy, ObservationsSamplePathStrategy
from exceptions import SimulationException


class Sampler:

    strategies = {
        'arrivals': ArrivalsSamplePathStrategy,
        'observations': ObservationsSamplePathStrategy
    }

    def simulate_sample_path(self, model: NHBP, time: float, path_type: str, *args, **kwargs):
        strategy = self.strategies.get(path_type)()
        full_args = model, time, *args
        return strategy.generate_path(*full_args, **kwargs)

    def generate_ensemble(self, model: NHBP, N: int, time: float, path_type: str, *args, **kwargs):
        strategy = self.strategies.get(path_type)()
        full_args = model, N, time, *args
        return strategy.generate_ensemble(*full_args, **kwargs)
