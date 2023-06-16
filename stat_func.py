from math import comb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def organize(df, xi_l):
    total_patients = df.groupby("dosage")["total_patients"].sum().tolist()
    infected_patients = df[df['infected'] == 'Yes']["total_patients"].tolist()

    organize_l = []
    for i in range(len(total_patients)):
        organize_l.extend([total_patients[i], infected_patients[i], xi_l[i]])
    return organize_l

def create_likelihood(args):
    args = [int(num) if num == int(num) else num for num in args]

    function_string = f"def likelihood(c):\n    p = 1"
    for i in range(0, len(args), 3):
        num_exp = args[i]
        num_success = args[i + 1]
        num_failure = num_exp - num_success
        success_prob = args[i + 2]
        failure_prob = 1 - success_prob

        binomial_coefficient = comb(num_exp, num_success)
        term_string = f" * ({binomial_coefficient} * (({success_prob}**c) ** {num_success} )*( (1-{success_prob}**c) ** {num_failure}))"
        function_string += term_string

    function_string += "\n    return - p"

    global_namespace = {}
    local_namespace = {}
    exec(function_string, global_namespace, local_namespace)
    return local_namespace['likelihood']


def find_curr_theta(like_func):
    result = minimize(like_func, x0=1)
    mle_theta = result.x[0]
    return mle_theta


def find_best_dosage(xi_l, curr_theta, m):
    rec_dosage = np.argmin(np.abs(np.array(xi_l) ** curr_theta - m))+1
    return rec_dosage


def main(df, xi_l, m):
    args_l = organize(df, xi_l)
    likelihood_func = create_likelihood(args_l)
    est_theta = find_curr_theta(likelihood_func)
    recommended_dosage = find_best_dosage(xi_l, est_theta, m)
    xi_l_updated = np.array(xi_l) ** est_theta

    return est_theta, recommended_dosage, xi_l_updated

