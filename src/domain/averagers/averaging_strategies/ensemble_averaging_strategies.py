from abc import abstractmethod, ABC
from typing import List

import numba
import numpy as np


class EnsembleAveragingStrategy(ABC):
    @abstractmethod
    def calculate(self, ensemble: np.typing.NDArray) -> List[float]:
        pass


class RegularEnsembleAveragingStrategy(EnsembleAveragingStrategy):

    @numba.njit
    def calculate(self, ensemble):
        average = np.mean(np.array(ensemble), axis=0)
        return average.tolist()


class SquareEnsembleAveragingStrategy(EnsembleAveragingStrategy):

    def calculate(self, ensemble):
        avg_squared = np.mean(ensemble ** 2, axis=0)
        return avg_squared

