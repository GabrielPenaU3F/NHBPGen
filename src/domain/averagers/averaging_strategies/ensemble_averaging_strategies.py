from abc import abstractmethod, ABC
from typing import List

import numba
import numpy as np


class EnsembleAveragingStrategy(ABC):
    @abstractmethod
    def calculate(self, ensemble: List[List[float]]) -> List[float]:
        pass


class RegularEnsembleAveragingStrategy(EnsembleAveragingStrategy):

    @numba.njit
    def calculate(self, ensemble):
        average = np.mean(np.array(ensemble), axis=0)
        return average.tolist()


class SquareEnsembleAveragingStrategy(EnsembleAveragingStrategy):

    def calculate(self, ensemble):
        transposed_trajectories = list(zip(*ensemble))

        # Calculate the average squared value for each position
        avg_squared = [sum(x ** 2 for x in position_ensemble) / len(position_ensemble)
                       for position_ensemble in transposed_trajectories]

        return avg_squared

