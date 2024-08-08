from abc import abstractmethod, ABC
from typing import List

import numpy as np


class TimeAveragingStrategy(ABC):
    @abstractmethod
    def calculate(self, X: List[float], T: float, delta: int) -> float:
        pass


class RegularTimeAveragingStrategy(TimeAveragingStrategy):

    def calculate(self, X, T, delta):
        pass


class AbsoluteTimeAveragingStrategy(TimeAveragingStrategy):

    def calculate(self, X, T, delta):
        pass


class AbsoluteVelocityTimeAveragingStrategy(TimeAveragingStrategy):

    def calculate(self, X, T, delta):
        n = int(T/delta)
        sum = 0
        for j in range(1, n):
            sum += np.abs(X[j * delta] - X[(j-1) * delta])/delta

        # return n * sum
        return sum/T
