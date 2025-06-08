import os

import numpy as np
import pandas as pd

from domain.extra_functions.extra_functions import get_base_path


class DataManager:

    @classmethod
    def load_data(cls, path):
        path = os.path.join(get_base_path(), path)
        data = pd.read_csv(path)
        if 'cumul_packets' not in data.columns:
            data = cls.preprocess(data)
            data.to_csv(path)

        return data

    @classmethod
    def slice_data(cls, data, start=1, end=None, reset_index=False):

        sliced_data = data.iloc[start - 1 : end]
        if reset_index:
            correct_index = np.arange(1, len(sliced_data) + 1)
            sliced_data.set_index(correct_index, inplace=True, drop=True)

        return sliced_data

    @classmethod
    def preprocess(cls, data):
        cumul_packets = data['n_packets'].cumsum()
        t_idx = data.index
        data = pd.DataFrame({'t_idx': t_idx, 'cumul_packets': cumul_packets})
        return data
