from abc import abstractmethod, ABC
from typing import List

import numpy as np


class TimeAveragingStrategy(ABC):


    @abstractmethod
    def calculate(self, X: List[float], n1: int, n2: int):
        pass



class AbsoluteTimeAveragingStrategy(TimeAveragingStrategy):

    def calculate(self, X, n1, n2):
        return np.abs(X[n1] - X[n2])


class SquareTimeAveragingStrategy(TimeAveragingStrategy):

    def calculate(self, X, n1, n2):
        return (X[n1] - X[n2]) ** 2
