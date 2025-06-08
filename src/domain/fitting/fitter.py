from sklearn.metrics import r2_score
from domain.data_management.data_manager import DataManager
from domain.fitting.fit import Fit


class Fitter:

    @classmethod
    def fit_model(cls, data, fitter_model, start, end, p0, initial_state=0, reset_index=False):
        data = DataManager.slice_data(data, start, end, reset_index)
        x = data['t_idx'].values
        y = data['cumul_packets'].values
        model_params, cov = fitter_model.fit(x, y, p0, initial_state=initial_state)
        pred_y = fitter_model.mean_value_function(x, *model_params)
        rsq = r2_score(y, pred_y)
        return Fit(x, y, model_params, cov, rsq)
