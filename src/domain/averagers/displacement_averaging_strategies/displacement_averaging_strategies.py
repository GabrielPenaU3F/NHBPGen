from abc import abstractmethod, ABC
from typing import List

import numpy as np


class DisplacementAveragingStrategy(ABC):
    @abstractmethod
    def calculate(self, X: List[float], T: float, delta: int) -> float:
        pass


class RegularAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, T, delta):
        pass


class AbsoluteAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, T, delta):
        pass


class AbsoluteVelocityAveragingStrategy(DisplacementAveragingStrategy):

    def calculate(self, X, T, delta):
        sum = 0
        for j in range(1, int(T/delta)):
            sum += X[j * delta] - X[(j-1) * delta]

        return sum/(T - delta)
