from abc import abstractmethod, ABC
from typing import List

import numpy as np


class TimeAveragingStrategy(ABC):

    def calculate(self, X: List[float], min_T: float, max_T: float, delta: int, time_step: int) -> List[float]:
        results = [0]

        T_axis = np.arange(min_T + time_step, max_T, time_step)
        n_min = int(min_T / time_step)
        n_delta = int(delta / time_step)

        if n_delta > int(min_T/time_step):
            raise IndexError('The delta time gap must be smaller than min_T')

        sum_increments = 0
        for j in range(n_delta, n_min):
            n1 = j * n_delta
            n2 = (j - 1) * n_delta
            if n1 < len(X):
                sum_increments += self.summation_term(X, n1, n2)
            else:
                break
        results.append(sum_increments / (n_min - n_delta + 1))

        for T in T_axis:
            n = int(T / time_step)
            if n < len(X):
                sum_increments += self.summation_term(X, n, n - n_delta)
                results.append(sum_increments / (n - n_delta + 1) )
            else:
                break

        return results[1:]


    @abstractmethod
    def summation_term(self, X, n1, n2):
        pass



class AbsoluteTimeAveragingStrategy(TimeAveragingStrategy):

    def summation_term(self, X, n1, n2):
        return np.abs(X[n1] - X[n2])


class SquareTimeAveragingStrategy(TimeAveragingStrategy):

    def summation_term(self, X, n1, n2):
        return (X[n1] - X[n2]) ** 2
