
import numpy
import scipy.linalg
import scipy.optimize
from joblib import Parallel, delayed
from operator import itemgetter

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain


'''
Optimization of the Hyperparameters of the MISO statistical model: 
signal variance and length scales of the SE kernel,
and observational noise parameter.
'''

# ================================================================================== #
#                               hyperparameters Prior                                #
# ================================================================================== #
class NormalPrior(object):
    ''' class for a multivariate normal prior '''
    def __init__(self, mu, sig):
        self._mu = mu
        self._sig_inv = numpy.linalg.inv(sig)

    def compute_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(x_mu.T, numpy.dot(self._sig_inv, x_mu))

    def compute_grad_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(self._sig_inv + self._sig_inv.T, x_mu).flatten()

        

def compute_hyper_prior(problem_search_domain, points_sampled, points_sampled_value):
        # print(problem_search_domain[1:])
        prior_mean = []
        # print points_sampled
        prior_mean = numpy.concatenate(( [max(0.01, numpy.var(points_sampled_value))],
                                         [ (d[1]-d[0]) for d in problem_search_domain[1:] ] ))
        (lb,ub) = problem_search_domain[0]
        for _ in range(4):
            prior_mean = numpy.concatenate(( prior_mean, [ub-lb] ))
        # print(prior_mean)
        prior_mean = numpy.array(prior_mean)
        prior_sig = numpy.diag(numpy.power(prior_mean/2., 2.0)) # variance
        hyper_prior = NormalPrior(prior_mean, prior_sig)

        return hyper_prior



# ================================================================================== #
#                               optimize hyperparameters                             #
# ================================================================================== #

def covariance(hyper_without_noise, lengths_sq_0, point_one, point_two): 
    '''Calculate the covariance between two points (not accounting for noise) under the MISO model'''
    # sigma = numpy.array([[ hyper_without_noise[-4], hyper_without_noise[-3] ],
    #                      [ hyper_without_noise[-2], hyper_without_noise[-1] ]])
    # phi_s1 = numpy.array([ 1, point_one[0] ])
    # phi_s2 = numpy.array([ 1, point_two[0] ])
    # first = numpy.dot(phi_s1, sigma)
    # ker_s = numpy.dot(first, phi_s2)
    ker_s = hyper_without_noise[-4] + hyper_without_noise[-3] * point_two[0] + \
            hyper_without_noise[-2] * point_one[0] + hyper_without_noise[-1] * point_one[0] * point_two[0]
    # print(point_two[0])
    # -------------------- Matern52 -------------------- #
    norm_x_sq = numpy.divide( numpy.power(point_two[1:] - point_one[1:], 2.0), 
                              lengths_sq_0[:lengths_sq_0.shape[0]-4] ).sum()
    matern_norm_x = numpy.sqrt(5) * numpy.sqrt(norm_x_sq)
    norm_x_sum = 1.0 + matern_norm_x + 5.0/3.0*norm_x_sq
    result = hyper_without_noise[0] * ker_s * norm_x_sum * numpy.exp(-matern_norm_x)
    # -------------------- exponential -------------------- #
    # exp_value = numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), lengths_sq_0[1:]).sum())
    # result = hyperparameters_without_noise[0] * exp_value * ker_s
    # -------------------- all exponential -------------------- #
    # exp_value = numpy.exp(-0.5 * numpy.divide(numpy.power(point_two - point_one, 2.0), lengths_sq_0).sum())
    # result = hyperparameters_without_noise[0] * exp_value
    return result


def compute_covariance_matrix(hyperparameters_without_noise, noise_hyperparameters, points_sampled_noise_variance, points_sampled):
    '''
    Compute the covariance matrix of the points that have been sampled so far
    Args:
        dim: 1 + the dimension of the search space that the points are from
        hyperparameters_without_noise: the signal variance followed by the length scales
        noise_hyperparameters: the hyperparameters of the observational noise
        points_sampled: the points that have been sampled so far
    Returns: the covariance matrix
    '''
    lengths_sq_0 = numpy.power(hyperparameters_without_noise[1:], 2.0)
    cov_mat = numpy.zeros((points_sampled.shape[0], points_sampled.shape[0]), order='F')
    for j, point_two in enumerate(points_sampled):
        for i, point_one in enumerate(points_sampled[j:, ...], start=j):
            cov_mat[i, j] = covariance(hyperparameters_without_noise, lengths_sq_0, point_one, point_two)
            if i != j:
                cov_mat[j, i] = cov_mat[i, j]

        # add noise on the main diagonal
        cov_mat[j, j] += points_sampled_noise_variance[j]
        # cov_mat[j, j] += numpy.power(noise_hyperparameters[0], 2.0)
        # We have to square the noise hyperparameter. This is how it is done in MOE:
        # noise_variance: the ``\sigma_n^2`` (noise variance) associated w/observation, points_sampled_value.
        # In build_covariance_matrix() in moe/optimal_learning/python/python_version/python_utils.py:
        # noise_variance: i-th entry is amt of noise variance to add to i-th diagonal entry; i.e., noise measuring i-th point
    return cov_mat


def compute_log_likelihood(K_chol, K_inv_y, points_sampled_value):
    log_marginal_term1 = -0.5 * numpy.inner(points_sampled_value, K_inv_y)
    # print(log_marginal_term1)

    log_marginal_term2 = -numpy.log(K_chol[0].diagonal()).sum()

    log_marginal_term3 = -0.5 * numpy.float64(points_sampled_value.size) * numpy.log(2.0 * numpy.pi)
    return log_marginal_term1 + log_marginal_term2 + log_marginal_term3


def compute_log_likelihood_K(K, points_sampled_value):
    yk = numpy.inner(points_sampled_value, scipy.linalg.inv(K + 1e-10 * numpy.random.rand(*K.shape)))
    log_marginal_term1 = -0.5 * numpy.inner(yk, points_sampled_value.T)
    # print(log_marginal_term1)

    log_marginal_term2 = -0.5 * numpy.log(K.diagonal()).sum()

    log_marginal_term3 = -0.5 * numpy.float64(points_sampled_value.size) * numpy.log(2.0 * numpy.pi)
    return log_marginal_term1 + log_marginal_term2 + log_marginal_term3


def hyper_opt(points_sampled, points_sampled_value, points_sampled_noise_variance, 
                init_hyper, hyper_bounds, approx_grad, hyper_prior=None):
    '''
    Hyperparameter optimization
    Args:
        dim: 1 + the dimension of the search space that the points are from
        points_sampled:
        points_sampled_value:
        init_hyper: starting point of hyperparameters
        hyper_bounds: list of (lower_bound, upper_bound)
        approx_grad: bool
        hyper_prior:
    Returns: (optimial hyper, optimal marginal loglikelihood, function output)
    '''
    # hyper_bounds = [(0.0,numpy.inf) for hyperparameter in init_hyper]

    def obj_func(x):
        '''
        The negative marginal loglikelihood for hyperparameters x
        x: the hyperparameters, including noise hyperparameters appended to the hyperparameters of the kernels
        Returns: The negated value of the marginal loglikelihood at hyperparameters x
        '''
        # split x into hyperparameters and noise_hyperparameters
        # there are dim signal variances and length scales
        hyperparameters_without_noise = x[:-1]
        noise_hyperparameters = x[[-1]]

        # # compute the parts of the marginal loglikelihood
        covariance_matrix = compute_covariance_matrix(hyperparameters_without_noise, noise_hyperparameters, 
                                                        points_sampled_noise_variance, points_sampled)
        eigen = scipy.linalg.eigvalsh(covariance_matrix)
        if numpy.any(eigen<1e-3):
            print('not positive definite')
            if hyper_prior is not None:
                # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood + log prior
                return -1.0 * ( compute_log_likelihood_K(covariance_matrix, points_sampled_value)
                               + hyper_prior.compute_log_likelihood(hyperparameters_without_noise) )
            else:
                # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood
                return -1.0 * compute_log_likelihood_K(covariance_matrix, points_sampled_value)
        else:
            print('positive definite, take K_chol')
            K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
            K_inv_y = scipy.linalg.cho_solve(K_chol, points_sampled_value)
            if hyper_prior is not None:
                # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood + log prior
                return -1.0 * ( compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)
                               + hyper_prior.compute_log_likelihood(hyperparameters_without_noise) )
            else:
                # This BFGS minimizes but we wish to maximize, thus negate the log marginal likelihood
                return -1.0 * compute_log_likelihood(K_chol, K_inv_y, points_sampled_value)

    best_hyper = scipy.optimize.fmin_l_bfgs_b(func=obj_func, x0=init_hyper, args=(), approx_grad=True,
                                              bounds=hyper_bounds, m=10, factr=10.0, pgtol=0.01,
                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=100, disp=0, callback=None)
    return best_hyper


def generate_hyperbounds(problem_search_domain, upper_bound_noise_variances, upper_bound_signal_variances=1000.):
    '''
    Bounds on each hyperparameter to speed up BFGS. Zero is a trivial lower bound. For the upper bound keep in mind that
    the parameters will be squared, hence in particular the bounds for variances should cover all reasonable values.
    Args:
        num_IS: number of IS, including truth as IS 0
        problem_search_domain: the search domain of the problem, usually problem.obj_func_min._search_domain
        upper_bound_signal_variances: an upper bound on the signal variance and the sample/noise variance
    Returns: bounds as an array of pairs (lower_bound, upper_bound)
    '''
    general_lower_bound = 1e-1 # set a general lower bound for hyperparameters -- allowing 0. seems to cause singularities
    noise_var_lower_bound = 1e-2 # SPEARMINT uses a tophat prior with lower bound 0.01 on the squared hyperparameters for length scales

    # arbitrary large value for signal variance
    hyper_bounds = [(general_lower_bound, upper_bound_signal_variances)]
    # trivial upper bounds on length scales. Remember that they will be squared.
    for (lb,ub) in problem_search_domain[1:]:
        hyper_bounds = numpy.append( hyper_bounds, [(general_lower_bound, (ub-lb))] )
    # bounds on hyper of s. 
    (lb,ub) = problem_search_domain[0]
    for _ in range(4):
        hyper_bounds = numpy.append( hyper_bounds, [(general_lower_bound, (ub-lb))] )

    # add bounds on noise/sample variances
    hyper_bounds = numpy.append(hyper_bounds, numpy.array([ [noise_var_lower_bound, upper_bound_noise_variances] ]))
    # print(hyper_bounds)

    return hyper_bounds.reshape(-1,2) # reshape so that the array becomes an array of pairs (lower_bound, upper_bound)


def optimize_hyperparameters(problem_search_domain, points_sampled, points_sampled_value, points_sampled_noise_variance,
                             upper_bound_noise_variances = 10., consider_small_variances = True,
                             hyper_prior = None, num_restarts = 32, num_jobs = 16):
    '''
    Fit hyperparameters from data using MLE or MAP (described in Poloczek, Wang, and Frazier 2016)
    :param problem_search_domain: The search domain of the benchmark, as provided by the benchmark
    :param points_sampled: An array that gives the points sampled so far. Each points has the form [IS dim0 dim1 ... dimn]
    :param points_sampled_value: An array that gives the values observed at the points in same ordering
    :param upper_bound_noise_variances: An upper bound on the search interval for the noise variance parameters (before squaring)
    :param consider_small_variances: If true, half of the BFGS starting points have entries for the noise parameters set to a small value
    :param hyper_prior: use prior for MAP estimate if supplied, and do MLE otherwise
    :param num_restarts: number of starting points for BFGS to find MLE/MAP
    :param num_jobs: number of parallelized BFGS instances
    :return: An array with the best found values for the hyperparameters
    '''
    approx_grad = True
    upper_bound_signal_variances = numpy.maximum(10., numpy.var(points_sampled_value)) # pick huge upper bounds
    hyper_bounds = generate_hyperbounds(problem_search_domain, upper_bound_noise_variances, upper_bound_signal_variances)
    hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bd[0], bd[1]) for bd in hyper_bounds])
    hyper_multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_restarts)

    for i in xrange(num_restarts):
        init_hyper = hyper_multistart_pts[i]

        # if optimization is enabled, make sure that small variances are checked despite multi-modality
        # this optimization seems softer than using a MAP estimate
        if consider_small_variances and (i % 2 == 0):
            init_hyper[-1] = 0.1 # use a small value as starting point for noise parameters in BFGS

        hyper_multistart_pts[i] = init_hyper

    parallel_results = Parallel(n_jobs=num_jobs)(delayed(hyper_opt)(points_sampled, points_sampled_value, points_sampled_noise_variance,
                                init_hyper, hyper_bounds, approx_grad, hyper_prior) for init_hyper in hyper_multistart_pts)
    # print min(parallel_results,key=itemgetter(1))
    best_hyper = min(parallel_results,key=itemgetter(1))[0] # recall that we negated the log marginal likelihood when passing it to BFGS
    return best_hyper


def create_array_points_sampled_noise_variance(points_sampled, noise_hyperparameters):
    points_sampled_noise_variance = []
    for point in points_sampled:
        points_sampled_noise_variance.append(noise_hyperparameters[0])
    return numpy.array(points_sampled_noise_variance)