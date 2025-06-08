from domain.data_management.data_manager import DataManager
from domain.processes.pena_sigmoid_3p_process import PenaSigmoid3pProcess

data = DataManager.load_data('..\\..\\data\\190_agg_1d.csv')
t = data['t_idx'][1:]
y = data['cumul_packets'][1:]

gamma, l, M = 0.755, 117, -4295
model = PenaSigmoid3pProcess(gamma, l, M)

loglik = model.log_likelihood(y, t)

print(f'AIC = {-loglik + 6}:.4f')
