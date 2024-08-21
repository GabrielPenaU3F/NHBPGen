from abc import abstractmethod, ABC
from typing import List

import numpy as np


class TimeAveragingStrategy(ABC):

    def calculate(self, X: List[float], T: float, delta: int, time_step: int) -> float:
        n = int(T / time_step)
        n_delta = int(delta / time_step)
        sum_increments = 0
        for j in range(n_delta, n):
            n1 = j * n_delta
            n2 = (j - 1) * n_delta
            if n1 < len(X):
                sum_increments += self.summation_term(X, n1, n2)
            else:
                break
        return sum_increments / n

    @abstractmethod
    def summation_term(self, X, n1, n2):
        pass


class AbsoluteTimeAveragingStrategy(TimeAveragingStrategy):

    def summation_term(self, X, n1, n2):
        return np.abs(X[n1] - X[n2])


class SquareTimeAveragingStrategy(TimeAveragingStrategy):

    def summation_term(self, X, n1, n2):
        return (X[n1] - X[n2]) ** 2
