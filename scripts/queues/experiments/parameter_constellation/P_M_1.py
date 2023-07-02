from scipy import stats as st

from domain.processes.poisson_process import PoissonProcess
from domain.processes.polya_process import PolyaProcess
from scripts.queues.experiments.P_G_n_divergence.classified_dataset_generator import generate_dataset_of_queues
from scripts.queues.experiments.P_G_n_divergence.proportion_confidence_interval import calculate_proportion_ci

n_queues = 1000
gamma, beta = .25, .1
arrival_process = PolyaProcess(gamma, beta)
service_process = PoissonProcess(1)
n_servers = 1
time = 1000

# The labels are 0s and 1s, since they're obtained from the R2 of a linear regression
# low R2 values correspond to stable queues, high R2 values correspond to unstable queues,

queues, rsqs = generate_dataset_of_queues(n_queues, arrival_process, service_process,
                                          n_servers, time, classif=.5, plot=False)
confidence = .95
n = len(rsqs)

# Queues marked as 1
proportion_of_stable_queues = rsqs.mean()
ic_lower_ones, ic_upper_ones = calculate_proportion_ci(proportion_of_stable_queues, n, confidence)
print('Proportion of stable queues: ' + str(proportion_of_stable_queues))
print('Confidence interval: (' + str(ic_lower_ones) + ', ' + str(ic_upper_ones) + ')')

# Test for the probability of queue being stable

shape = beta/gamma
scale = gamma
gamma_rv = st.gamma(shape, scale=scale)
prob_stable = 1 - gamma_rv.sf(1)

print('Fendicks Gamma distribution probability of stability ' + str(prob_stable))
