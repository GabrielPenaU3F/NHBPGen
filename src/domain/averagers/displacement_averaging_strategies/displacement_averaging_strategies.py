from abc import abstractmethod, ABC
from typing import List

import numpy as np


class DisplacementAveragingStrategy(ABC):
    @abstractmethod
    def calculate(self, X: List[float], k: int, delta: int) -> float:
        pass


class RegularAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, k, delta):
        return X[k + delta] - X[k]


class AbsoluteAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, k, delta):
        return np.abs(X[k + delta] - X[k])


class AbsoluteVelocityAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, k, delta):
        return np.abs((X[k + delta] - X[k])/delta)


class SquaredAveragingStrategy(DisplacementAveragingStrategy):
    def calculate(self, X, k, delta):
        return (X[k + delta] - X[k])**2
