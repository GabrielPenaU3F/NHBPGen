import numpy as np
from scipy import stats as st


def calculate_proportion_ci(proportion, n, confidence_level):
    standard_error = np.sqrt(proportion * (1 - proportion) / n)
    z_critical = abs(st.norm.ppf((1 - confidence_level) / 2))
    margin_of_error = z_critical * standard_error
    lower_bound = proportion - margin_of_error
    upper_bound = proportion + margin_of_error
    return lower_bound, upper_bound
