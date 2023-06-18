from math import comb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def organize(df, xi_l):
    """
    Organizes the data in the DataFrame and xi list into a flattened list.

    Args:
        df (pandas.DataFrame): DataFrame containing the experimental data.
        xi_l (list): List of xi values.

    Returns:
        list: Flattened list containing the total patients, infected patients, and xi values for each dosage.
    """
    total_patients = df.groupby("dosage")["total_patients"].sum().tolist()
    infected_patients = df[df['infected'] == 'Yes']["total_patients"].tolist()

    organize_l = []
    for i in range(len(total_patients)):
        organize_l.extend([total_patients[i], infected_patients[i], xi_l[i]])
    return organize_l

def create_likelihood(args):
    """
    Creates a likelihood function based on the provided arguments.

    Args:
        args (list): List of arguments containing the number of experiments, number of successes, and success probabilities.

    Returns:
        function: Likelihood function.
    """
    args = [int(num) if num == int(num) else num for num in args]

    function_string = f"def likelihood(c):\n    p = 1"
    for i in range(0, len(args), 3):
        num_exp = args[i]
        num_success = args[i + 1]
        num_failure = num_exp - num_success
        success_prob = args[i + 2]
        failure_prob = 1 - success_prob
        term_string = f" * ((({success_prob}**c) ** {num_success} )*( (1-{success_prob}**c) ** {num_failure}))"
        function_string += term_string

    function_string += "\n    return - p"

    global_namespace = {}
    local_namespace = {}
    print(function_string)
    exec(function_string, global_namespace, local_namespace)
    return local_namespace['likelihood']


def find_curr_theta(like_func):
    """
    Finds the current theta value by minimizing the likelihood function.

    Args:
        like_func (function): Likelihood function.

    Returns:
        float: Estimated theta value.
    """
    result = minimize(like_func, x0=1)
    mle_theta = result.x[0]
    return mle_theta


def find_best_dosage(xi_l, curr_theta, m):
    """
    Finds the best dosage based on the xi values, current theta, and target m-value.

    Args:
        xi_l (list): List of xi values.
        curr_theta (float): Current estimated theta value.
        m (float): Target m-value.

    Returns:
        int: Recommended dosage.
    """
    rec_dosage = np.argmin(np.abs(np.array(xi_l) ** curr_theta - m)) + 1
    return rec_dosage


def main(df, xi_l, m):
    """
    Main function to estimate theta, determine the recommended dosage, and update the xi values.

    Args:
        df (pandas.DataFrame): DataFrame containing the experimental data.
        xi_l (list): List of xi values.
        m (float): Target m-value.

    Returns:
        tuple: Estimated theta value, recommended dosage, and updated xi values.
    """
    args_l = organize(df, xi_l)
    likelihood_func = create_likelihood(args_l)
    est_theta = find_curr_theta(likelihood_func)
    recommended_dosage = find_best_dosage(xi_l, est_theta, m)
    xi_l_updated = np.array(xi_l) ** est_theta

    return est_theta, recommended_dosage, xi_l_updated

