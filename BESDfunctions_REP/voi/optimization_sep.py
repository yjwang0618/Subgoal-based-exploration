import numpy as np

import scipy.optimize

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement

from knowledge_gradient_sep import *
# import sql_util


def bfgs_optimization(start_pt_x, func_to_minimize, bounds):
    bounds = bounds[1:]
    result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=func_to_minimize, x0=start_pt_x, 
                                                              fprime=None, args=(), approx_grad=True,
                                                              bounds=bounds, m=10, factr=10.0, pgtol=1e-10,
                                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, 
                                                              disp=0, callback=None)
    return result_x, result_f


def negative_kg_given_x_prime(s, all_S_x_prime, noise_and_cost_func, gp, hyperparameters_noise, repQL):
    def negative_kg(x):
        s_x = np.hstack((s*np.ones(1), x))
        return -compute_kg_given_x_prime(s, x, all_S_x_prime, hyperparameters_noise, noise_and_cost_func(repQL, s_x)[1], gp)
#        return -compute_kg_given_x_prime(s, x, all_S_x_prime, noise_and_cost_func(s_x)[0], noise_and_cost_func(s_x)[1], gp)

    return negative_kg


def negative_mu_kg(s, gp):
    def result(x):
        return -1.0 * gp.compute_mean_of_points(numpy.concatenate(([s], x)).reshape((1,-1)))[0]
    return result


def compute_mu(gp):
    def result(x):
        return gp.compute_mean_of_points(x.reshape((1,-1)))[0]
    return result
