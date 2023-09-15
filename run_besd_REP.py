import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import numpy as np
import pickle
from pickle import dump
from joblib import Parallel, delayed
import sys

from operator import itemgetter
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

from moe.optimal_learning.python.cpp_wrappers.covariance import PolynomialMaternNu2p5 as cppCovariance
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData

from BESDfunctions_REP.voi.optimization_sep import *
from BESDfunctions_REP.model.hyperparameter_optimization_sep import optimize_hyperparameters, \
                                                                        create_array_points_sampled_noise_variance, \
                                                                        compute_hyper_prior
from BESDfunctions_REP.besd_utils_rep import process_parallel_results, load_sample_data
from BESDfunctions_REP.besd_utils_rep import sample_initial_x_uniform, select_startpts_x_BFGS
from BESDfunctions_REP.besd_utils_rep import sample_initial_f1f2_closer_f1_further_f2, select_startpts_f1closer_BFGS
from BESDfunctions_REP.besd_utils_rep import sample_initial_f1f2_higher_f1_lower_f2, select_startpts_f1higher_BFGS

from problems.identifier_REP import identify_problem
from problems.define_basic_functions_REP import define_test_pts

np.set_printoptions(threshold=sys.maxint)


def define_test_pts_(prob_name):
    test_x, row_num, col_num, _maze_num, _maze_name, _noise_base, _noise_rand, \
        _flag_num, _maze_size, _repQL, _s, _S, _skip, episodeCap = define_test_pts(prob_name)
    ### grid-world
    if prob_name == 'gw10One1': num_per_var = 2.5
    elif prob_name == 'gw10Two1': num_per_var = 2.5
    elif prob_name == 'gw10Two2': num_per_var = 2.5
    elif prob_name == 'gw10Three1': num_per_var = 2.5
    elif prob_name == 'gw20Three1': num_per_var = 2.5
    elif prob_name == 'gw20Three2': num_per_var = 2.5
    ### grid-world with a pit near the goal
    elif prob_name == 'pt10': num_per_var = 2.5
    ### grid-world with key to open the door
    elif prob_name == 'ky10One': num_per_var = 2.5
    ### grid-world with subgoals
    elif prob_name == 'it10': num_per_var = 2.5
    ### puddle-world
    elif prob_name == 'pd10': num_per_var = 2.5  # 7x10 puddleworld
    ### mountain-car
    elif prob_name == 'mcf1': num_per_var = 10  # mountain-car with 1 flag
    elif prob_name == 'mcf2': num_per_var = 5  # mountain-car with 2 flags
    elif prob_name == 'mcf3': num_per_var = 4  # mountain-car with 3 flags
    else: print('prob_name is wrong, please check !!!!!!!!!!!!!!!!!!!!!!')
    return test_x, num_per_var, row_num, col_num


# ================================================================================================================ #
#                                           basic settings of the problem                                          #
# ================================================================================================================ #
argv = sys.argv[1:]
'''
python run_besd_REP.py besd_gw 0 0 gw10Two1
    - benchmark_name
        + besd_gw -- gw10Two1, gw20Three1
        + besd_pt -- pt10
        + besd_ky -- ky10One
        + besd_it -- it10
        + besd_pd -- pd10
        + besd_mc -- mcf2, mcf3
    - which_problem: 0
    - replication_no: 0,1,2,...
'''

problem = identify_problem(argv, None)
prob_name = argv[-1]
test_x, num_per_var, row_num, col_num = define_test_pts_(prob_name)
var_dim = int(problem.obj_func_min.getDim()) - 1

folder = 'REP_rlt_'+prob_name+'/'
init_result_path = folder+'initResults_besd_'+prob_name+'/rep'+str(problem.replication_no)+'/'
exp_path = init_result_path+'initial_samples/'
result_folder = folder+'Results_besd_'+prob_name
result_path = result_folder + '/rep' + str(problem.replication_no)
if not os.path.exists(result_path): os.makedirs(result_path)
print ('Results are written into file ' + result_path)

if os.path.isfile(result_path+'.pickle'):
    with open(result_path+'.pickle', 'rb') as file: f_dict = pickle.load(file)
    list_cost = f_dict.get('cost')
    if len(list_cost) >= problem.num_iterations:
        quit()

'''
Different cases have 3 differences:
        1. generate initial samples
        2. generate random x points 
        3. based on 2, generate random x points as start points to calculate KG
    
    Case0: no constraints
        - sample_initial_x_uniform
        - problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain()
        - select_startpts_x_BFGS
        
    Case1: flag1 is closer to start point (0,0) than flag2
        - sample_initial_f1f2_closer_f1_further_f2
        - problem.obj_func_min.get_moe_domain().generate_closer_f1_further_f2()
        - select_startpts_f1closer_BFGS
        
    Case2: flag1 is above flag2 (y1 <= y2)
        - sample_initial_f1f2_higher_f1_lower_f2
        - problem.obj_func_min.get_moe_domain().generate_higher_f1_lower_f2()
        - select_startpts_f1higher_BFGS
'''

# ------------------------------------------------------------------ #
#                 data containers for pickle storage                 #
# ------------------------------------------------------------------ #
repQL_max = problem.obj_func_min.repQL
# repQL_array = [int(repQL_max/4.0), int(repQL_max/2.0), int(3*repQL_max/4.), repQL_max]
# repQL_noise_rate = {int(repQL_max/4.0): 4, 
#                     int(repQL_max/2.0): 2, 
#                     int(3*repQL_max/4.):4./3, 
#                     repQL_max:          1}
if prob_name in ['mcf1', 'mcf2', 'mcf3']:
    repQL_array = [int(repQL_max/5.0), repQL_max]
    repQL_noise_rate = {int(repQL_max/5.0): 5, 
                        repQL_max:          1}
else:
    repQL_array = [int(repQL_max/4.0), repQL_max]
    repQL_noise_rate = {int(repQL_max/4.0): 4, 
                        repQL_max:          1}

s_min = problem.obj_func_min.getSearchDomain()[0, 0]
s_max = problem.obj_func_min.getSearchDomain()[0, 1]
### list of the prediction curve
# s_array = np.array([s_min, s_max])
s_array = np.linspace(s_min, s_max, num=problem.obj_func_min.getNums())
s_lc_num = 11
s_lc_array = np.linspace(s_min, s_max, s_lc_num)
test_x_num = test_x.shape[0]
learningcurve_pts = []
for pt_x in test_x:
    learningcurve_pt = np.zeros((s_lc_num, var_dim+1))
    s_od = -1
    for s in s_lc_array:
        s_od += 1
        learningcurve_pt[s_od,:] = np.hstack(([s], pt_x))
    learningcurve_pts.append(learningcurve_pt)
list_test_mu = []
### other lists
list_random_seeds = []
list_sampled_repQL = []
list_cost = []                   # cumulative cost so far
list_sampled_points = []         # list of all points sampled in chron. order
list_sampled_vals = []           # list of corresponding obs
list_sampled_noise_var = []              # list of noise variance of observations in chron. order
list_pending_mu_star_points = []
list_mu_star_points = []
list_mu_star_truth = []          # list of values at resp. mu_star_points under IS0
list_mu_star_var = []
list_raw_voi = []
list_best_hyper = []
list_kg_fails = []

# ------------------------------------------------------------------ #
#                              parameters                            #
# ------------------------------------------------------------------ #
### Fast demo mode
# num_x_prime = 2000
# num_multistart = 8 # number of starting points when searching for optimum of posterior mean and maximum KG factor
# num_threads = -1 # how many jobs to use in parallelization? This uses all CPUs, reduce for parallel runs of BESD
# num_parallel_inst = 1 # how many instances of the benchmark can be run in parallel?

### Experiment mode
num_x_prime = 3000
num_multistart = 32 # number of starting points when searching for optimum of posterior mean and maximum KG factor
num_threads = -1 # how many jobs to use in parallelization? This uses all CPUs, reduce for parallel runs of BESD
num_parallel_inst = -1 # how many instances of the benchmark can be run in parallel?

# ------------------------------------------------------------------ #
#                         sample initial points                      #
# ------------------------------------------------------------------ # 
if os.path.isfile(result_path+'_initial_samples.pickle'):
    print('initial samples exists')
    with open(result_path+'_initial_samples.pickle', 'rb') as file: list_init_pts_value = pickle.load(file)
    if len(list_init_pts_value) >= len(problem.obj_func_min.s_array):
        print('load from initial data', int(num_per_var * var_dim))
        problem.set_hist_data = load_sample_data(problem, num_per_var, exp_path, result_path)
        if os.path.isfile(result_path+'.pickle'):
            print('already iterated')
            with open(result_path+'.pickle', 'rb') as file: f_dict = pickle.load(file)
            list_random_seeds = f_dict.get('random_seeds').tolist()
            list_cost = f_dict.get('cost').tolist()

            list_sampled_points = f_dict.get('sampled_points').tolist()
            list_sampled_vals = f_dict.get('sampled_vals').tolist()
            list_sampled_noise_var = f_dict.get('sampled_var_noise').tolist()
            list_sampled_repQL = f_dict.get('sampled_repQL').tolist()

            list_mu_star_points = f_dict.get('mu_star_points').tolist()
            list_mu_star_truth = f_dict.get('mu_star_truth').tolist()
            list_mu_star_var = f_dict.get('mu_star_var').tolist()

            array_sampled_vals_inQL = f_dict.get('sampled_vals_inQL')
            array_mu_star_truth_inQL = f_dict.get('mu_star_truth_inQL')
            
            truth_at_init_best_sampled = f_dict.get('init_best_truth')
            list_best_hyper = f_dict.get('best_hyper').tolist()
            list_kg_fails = f_dict.get('list_kg_fails')
            list_raw_voi = f_dict.get('list_raw_voi')
            list_test_mu = f_dict.get('test_mu').tolist()

            print('start from iteration ', len(list_sampled_vals))
        else:
            print('start from 0')
    else:
        problem.set_hist_data = sample_initial_x_uniform(problem, num_per_var, exp_path, result_path)
else:
    print('initial samples does not exist')
    problem.set_hist_data = sample_initial_x_uniform(problem, num_per_var, exp_path, result_path)

### mkg begins
kg_gp_cpp = None
num_discretization = num_x_prime * 2
# if the KG/unit_cost drops below this, then sample at optimum of posterior mean.
# The exploitation IS is defined in problem object.
exploitation_threshold = 1e-7

init_best_idx = numpy.argmax(problem.hist_data._points_sampled_value[problem.hist_data._points_sampled[:,0] == s_max])
best_sampled_val = -1.0 * problem.hist_data._points_sampled_value[init_best_idx]
# minus sign is because vals in hist_data were obtained from obj_func_max, while all values to store are
# from obj_func_min, for consistency
truth_at_init_best_sampled = best_sampled_val + problem.obj_func_min._meanval
best_mu_star_truth = np.inf
total_cost = 0.0

# ------------------------------------------------------------------ #
# after how many iterations shall we re-optimize the hyperparameters #
# ------------------------------------------------------------------ #
hyper_learning_interval = int(numpy.maximum(problem.num_iterations/5, len(problem.hist_data._points_sampled_value)))
# hyper_learning_interval = 5

# ------------------------------------------------------------------ #
# The hyper prior is computed from the initial dataset,              #
# since it requires that all IS are evaluated at the same points     #
# ------------------------------------------------------------------ #
# hyper_prior = None
hyper_prior = compute_hyper_prior(problem.obj_func_min.getSearchDomain(), 
                                  problem.hist_data.points_sampled,
                                  problem.hist_data.points_sampled_value)

len_list_sampled_vals = len(list_sampled_vals)
for kg_iteration in xrange(len_list_sampled_vals, problem.num_iterations):
    exp_path=init_result_path+'iteration'+str(kg_iteration)+'/'
    list_random_seeds_iter = []
    
    # ================================================================================================================ #
    #                             Update hyper every hyper_learning_interval - many samples                            #
    # ================================================================================================================ #
    if kg_iteration == len_list_sampled_vals or kg_gp_cpp.num_sampled % hyper_learning_interval == 0:
        current_hist_data = kg_gp_cpp.get_historical_data_copy() if kg_gp_cpp else problem.hist_data
        best_hyper = optimize_hyperparameters(problem.obj_func_min.getSearchDomain(),
                                              current_hist_data.points_sampled, current_hist_data.points_sampled_value,
                                              current_hist_data.points_sampled_noise_variance,
                                              upper_bound_noise_variances=10., consider_small_variances=True,
                                              hyper_prior=hyper_prior,
                                              num_restarts=4, #16
                                              num_jobs=num_threads)
        list_best_hyper.append(best_hyper)

        ### update hyperparameters for noise for each observed value in historical data
        print "BESD: repl {0}, itr {1}, best hyper: {2}".format(problem.replication_no, kg_iteration, best_hyper)
        ### Format: IS 0: signal variance and length scales, IS 1: signal variance and length scales, etc.
        ###  Then observational noise for IS 0, IS 1 etc.
        hyperparameters_noise = numpy.power(best_hyper[-1:], 2.0)
        hypers_GP = best_hyper[:-1] # separate hypers for GP and for observational noise

        ### update noise in historical data
        updated_points_sampled_noise_variance = create_array_points_sampled_noise_variance(current_hist_data.points_sampled, 
                                                                                           hyperparameters_noise)
        ### create new Historical data object with updated values
        new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
        new_historical_data.append_historical_data(current_hist_data.points_sampled,
                                                   current_hist_data.points_sampled_value,
                                                   updated_points_sampled_noise_variance)
        # new_historical_data.append_historical_data(current_hist_data.points_sampled,
        #                                            current_hist_data.points_sampled_value,
        #                                            current_hist_data.points_sampled_noise_variance)

        ### Use new hyperparameters -- this requires instantiating a new GP object
        kg_cov_cpp = cppCovariance(hyperparameters=hypers_GP)
        kg_gp_cpp = GaussianProcess(kg_cov_cpp, new_historical_data)


    # ================================================================================================================ #
    #                                    Find s and point that maximize KG/cost                                        #
    # ================================================================================================================ #
    discrete_pts_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_discretization)
    discrete_pts = np.hstack(( problem.obj_func_min.getSearchDomain()[0,-1] * np.ones((num_discretization,1)), 
                               discrete_pts_x ))
    all_mu = kg_gp_cpp.compute_mean_of_points(discrete_pts)
    sorted_idx = np.argsort(all_mu)
    all_S_xprime = discrete_pts[sorted_idx[-num_x_prime:], :]  # select the last num_x_prime samples

    
    # ================================================================================================================ #
    #                  For every s, compute a point of maximum KG/cost (here: minimum -1.0 * KG/cost)                  #
    # ================================================================================================================ #
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def min_kg_unit(start_pt_x, s, repQL):
        func_to_min = negative_kg_given_x_prime(s, all_S_xprime, problem.obj_func_min.noise_and_cost_func, 
                                                kg_gp_cpp, hyperparameters_noise*repQL_noise_rate[repQL], repQL)
        return bfgs_optimization(start_pt_x, func_to_min, problem.obj_func_min._search_domain) # approximate gradient

    def compute_kg_unit(pt_sx, s, repQL):
        x = pt_sx[1:]
        ### fixed noise, estimated by the "hyperparameter_optimization_sep.py"
        return compute_kg_given_x_prime(s, x, all_S_xprime, hyperparameters_noise*repQL_noise_rate[repQL], 
                                        problem.obj_func_min.noise_and_cost_func(repQL, pt_sx)[1], kg_gp_cpp)
        ### fixed noise, manually pre-defined
        # return compute_kg_given_x_prime(s, x, all_S_xprime, problem.obj_func_min.noise_and_cost_func(pt_sx)[0], 
        #                                 problem.obj_func_min.noise_and_cost_func(pt_sx)[1], kg_gp_cpp)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    min_negative_kg = np.inf
    list_raw_kg_this_itr = []
    with Parallel(n_jobs=num_threads) as parallel:
        for repQL in repQL_array:
            for s in s_array:
                test_pts_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(1000)
                test_pts = np.hstack((s*np.ones((1000,1)), test_pts_x))
                kg_of_test_pts = parallel(delayed(compute_kg_unit)(pt, s, repQL) for pt in test_pts)

                test_pt_x_chosen = test_pts_x[np.argmax(kg_of_test_pts)]
                kg_at_test_pt_x_chosen = np.max(kg_of_test_pts)

                start_pts_x = select_startpts_x_BFGS(s, list_sampled_points, test_pt_x_chosen, num_multistart, problem)
                parallel_results = parallel(delayed(min_kg_unit)(pt_x, s, repQL) for pt_x in start_pts_x)

                ### add candidate point to list, remember to negate its KG/cost value since we are looking for the minimum
                parallel_results = np.concatenate((parallel_results, [[test_pt_x_chosen, -1.0*kg_at_test_pt_x_chosen]]), axis=0)

                inner_min, inner_min_point_x = process_parallel_results(parallel_results)
                inner_min_point_x = np.concatenate((inner_min_point_x, ))
                inner_min_point = np.hstack(([s], inner_min_point_x))

                list_raw_kg_this_itr.append(-inner_min * problem.obj_func_min.noise_and_cost_func(repQL, inner_min_point)[1])
                if inner_min < min_negative_kg:
                    min_negative_kg = inner_min
                    sample_point_sx = inner_min_point
                    sample_s, sample_repQL = s, repQL


    # ================================================================================================================ #
    #                          Is the KG (normalized to unit cost) below the threshold?                                #
    #                          Then exploit by sampling the point of maximum posterior mean                            #
    # ================================================================================================================ #
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### Recommendation: Search for point of optimal posterior mean for truth IS
    def find_mu_star(s_max, start_pt_x):
        # Find the optimum of the posterior mean. This is the point that BESD will recommend in this iteration.
        # :param start_pt_x: starting point for BFGS
        # :return: recommended point
        return bfgs_optimization(start_pt_x, negative_mu_kg(s_max, kg_gp_cpp), problem.obj_func_min._search_domain)

    def search_mu_star_point(kg_gp_cpp, list_sampled_points, sample_point_sx, num_multistart, num_threads, problem):
        '''Search for point of optimal posterior mean for S'''
        if len(list_sampled_points) == 0: 
            test_pts = numpy.array([sample_point_sx])
        else: 
            test_pts = np.concatenate(([sample_point_sx], numpy.array(list_sampled_points)), axis=0)
        test_pts_x = test_pts[:, 1:]
        num_random_pts = int(1e4)
        random_pts_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_random_pts)
        test_pts_x = np.concatenate((test_pts_x, random_pts_x), axis=0)
        ### all points must be extended by S, i.e. a leading S must be added to each point
        test_pts_x_with_S = numpy.insert(test_pts_x, 0, s_max, axis=1)

        ### Negate mean values, since KG's GP works with the max version of the problem
        means = -1.0 * kg_gp_cpp.compute_mean_of_points(test_pts_x_with_S)

        mu_star_x_candidate = test_pts_x[np.argmin(means)]
        mean_mu_star_candidate = np.min(means)

        start_pts_x = select_startpts_x_BFGS(s_max, list_sampled_points, mu_star_x_candidate, num_multistart, problem)
        with Parallel(n_jobs=num_threads) as parallel:
            parallel_results = parallel(delayed(find_mu_star)(s_max, pt_x) for pt_x in start_pts_x)

        parallel_results = np.concatenate((parallel_results, [[mu_star_x_candidate, mean_mu_star_candidate]]), axis=0)
        mu_star_point_x = min(parallel_results, key=itemgetter(1))[0]
        return mu_star_point_x
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if -min_negative_kg * problem.obj_func_min.noise_and_cost_func(sample_repQL, sample_point_sx)[1] < exploitation_threshold:
        print "KG search failed, do exploitation"
        list_kg_fails.append(kg_iteration)
        print(min_negative_kg, problem.obj_func_min.noise_and_cost_func(sample_repQL, sample_point_sx)[1])
        mu_star_point_x = search_mu_star_point(kg_gp_cpp, list_sampled_points, sample_point_sx, 
                                               num_multistart, num_threads, problem)
        mu_star_point = np.hstack(([s_max], mu_star_point_x))
        sample_point_sx = mu_star_point
        sample_s, sample_repQL = s_min, repQL_array[0]


    # ================================================================================================================ #
    #     If we are running the expensive repQL and S, use the time to perform other expensive queries in parallel     #
    # ================================================================================================================ #
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ### make next observation and update GP
    def parallel_func(repQL, pt, random_seed): 
        return problem.obj_func_min.evaluate(repQL, pt, random_seed, exp_path=exp_path)
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    if sample_s == s_max and sample_repQL==repQL_array[-1]:
        list_pending_mu_star_points.append(sample_point_sx)
        np.random.seed(kg_iteration)
        random_seeds = np.random.randint(900, size=len(list_pending_mu_star_points))
        list_random_seeds_iter.append(random_seeds)
        with Parallel(n_jobs=num_parallel_inst) as parallel:
            result = parallel(delayed(parallel_func)(repQL_array[-1], pt, random_seed) \
                                      for (pt, random_seed) in zip(list_pending_mu_star_points, random_seeds))

        vals_pending_mu_star_points = [item[0] for item in result]
        vars_pending_mu_star_points = [item[1] for item in result]

        sample_val = vals_pending_mu_star_points[-1]  # the last value belongs to this iteration
        sample_var = vars_pending_mu_star_points[-1]  # the last variance belongs to this iteration

        ### remove point and obs of this iteration from lists
        list_pending_mu_star_points = list_pending_mu_star_points[:-1]
        vals_pending_mu_star_points = vals_pending_mu_star_points[:-1]
        vars_pending_mu_star_points = vars_pending_mu_star_points[:-1]

        ### add evaluations of mu_star to our list
        list_mu_star_truth.extend(vals_pending_mu_star_points)
        list_pending_mu_star_points = []
    else:
        ### just do the cheap observation and defer the expensive one
        np.random.seed(kg_iteration)
        random_seed = np.random.randint(900)
        list_random_seeds_iter.append(random_seed)
        sample_val, sample_var = problem.obj_func_min.evaluate(sample_repQL, sample_point_sx, random_seed, exp_path=exp_path)


    # ================================================================================================================ #
    # add point and observation to GP                                                                                  #
    # NOTE: while we work everywhere with the values of the minimization problem in the computation,                   #
    #       we used the maximization obj values for the GP.                                                            #
    #       That is why here sample_val is multiplied by -1.0                                                          #
    # ================================================================================================================ #
    kg_gp_cpp.add_sampled_points([SamplePoint( sample_point_sx, -1.0*sample_val, sample_var )])

    mu_star_point_x = search_mu_star_point(kg_gp_cpp, list_sampled_points, sample_point_sx, 
                                           num_multistart, num_threads, problem)
    mu_star_point = np.hstack((s_max * np.ones(1), mu_star_point_x))
    list_pending_mu_star_points.append(mu_star_point)


    # ================================================================================================================ #
    #     perform batched evaluation of mu_star points at truthIS, since they can delay the iteration significantly    #
    # ================================================================================================================ #
    if len(list_pending_mu_star_points) >= num_parallel_inst or (kg_iteration+1 == problem.num_iterations):
        ### have we collected several points or is this the last iteration?
        np.random.seed(kg_iteration)
        random_seeds = np.random.randint(900, size=len(list_pending_mu_star_points))
        list_random_seeds_iter.append(random_seeds)
        with Parallel(n_jobs=num_parallel_inst) as parallel:
            result = parallel(delayed(parallel_func)(repQL_array[-1], pt, random_seed) \
                                      for (pt, random_seed) in zip(list_pending_mu_star_points, random_seeds))

        vals_pending_mu_star_points = [item[0] for item in result]
        vars_pending_mu_star_points = [item[1] for item in result]

        list_mu_star_truth.extend(vals_pending_mu_star_points)
        list_mu_star_var.extend(vars_pending_mu_star_points)
        list_pending_mu_star_points = []


    # ================================================================================================================ #
    #                                           Print progress to stdout                                               #
    # ================================================================================================================ #
    if len(list_mu_star_truth) > 0:
        print 'repl ' + str(problem.replication_no) + ', it ' + str(kg_iteration) + ', sample IS ' \
              + str(sample_s) + ' at ' + str(sample_point_sx) \
              + ', recommendation ' + str(mu_star_point) +' has (observed) value ' + str(list_mu_star_truth[-1]
                                                                              + problem.obj_func_min.get_meanval())
    else:
        print 'repl ' + str(problem.replication_no) + ', it ' + str(kg_iteration) + ', sample IS ' \
              + str(sample_s) + ' at ' + str(sample_point_sx)

    # ----------------------------------------------------------------- #
    #                      Collect data for pickle                      #
    # ----------------------------------------------------------------- #
    list_random_seeds.append(list_random_seeds_iter)
    total_cost += problem.obj_func_min.noise_and_cost_func(sample_repQL, sample_point_sx)[1]
    list_cost.append(total_cost)
    list_sampled_points.append(sample_point_sx)
    list_sampled_repQL.append(sample_repQL)
    list_sampled_vals.append(sample_val)
    list_sampled_noise_var.append(sample_var)
    list_mu_star_points.append(mu_star_point)
    list_raw_voi.append(list_raw_kg_this_itr)

    list_test_mu_ = []
    if problem.obj_func_min.getFuncName() in ['GridWorld', 'GridWorldKey', 'MountainCar']: # key_word = 'steps'
        array_sampled_vals_inQL = np.exp( np.array(list_sampled_vals) + problem.obj_func_min._meanval )
        array_mu_star_truth_inQL = np.exp( np.array(list_mu_star_truth) + problem.obj_func_min._meanval )
        for pt_od, pt_x in enumerate(test_x):
            test_mu = -1.0 * kg_gp_cpp.compute_mean_of_points(learningcurve_pts[pt_od])
            test_mu = np.exp(test_mu + problem.obj_func_min._meanval)
            list_test_mu_.append(test_mu)
    elif problem.obj_func_min.getFuncName() in ['Puddle', 'GridWorld_Item', 'GridWorldPit']: # key_word = 'return'
        array_sampled_vals_inQL = -1.0*np.array(list_sampled_vals) + problem.obj_func_min._meanval
        array_mu_star_truth_inQL = -1.0*np.array(list_mu_star_truth) + problem.obj_func_min._meanval
        for pt_od, pt_x in enumerate(test_x):
            test_mu = kg_gp_cpp.compute_mean_of_points(learningcurve_pts[pt_od])
            test_mu = test_mu + problem.obj_func_min._meanval
            list_test_mu_.append(test_mu)
    list_test_mu.append(list_test_mu_)

    result_to_file = {'random_seeds': np.array(list_random_seeds),
                      'cost': np.array(list_cost),
                      # sampled: point, value, variance_noise
                      'sampled_points': np.array(list_sampled_points),
                      'sampled_vals': np.array(list_sampled_vals),
                      'sampled_var_noise': np.array(list_sampled_noise_var),
                      'sampled_repQL': np.array(list_sampled_repQL), 
                      # mu_star: point, value, variance_noise
                      'mu_star_points': np.array(list_mu_star_points),
                      'mu_star_truth': np.array(list_mu_star_truth),
                      'mu_star_var': np.array(list_mu_star_var),
                      # the value transformed into the original one (no log, no normalized around 0)
                      'sampled_vals_inQL': array_sampled_vals_inQL,
                      'mu_star_truth_inQL': array_mu_star_truth_inQL, 
                      # others
                      'init_best_truth': truth_at_init_best_sampled,
                      'best_hyper': np.array(list_best_hyper),
                      'list_kg_fails': list_kg_fails, 
                      'list_raw_voi': list_raw_voi, 
                      # test_points' values
                      'test_mu': np.array(list_test_mu)}
    with open(result_path+'.txt', "w") as file: file.write(str(result_to_file))
    with open(result_path+'.pickle', "wb") as file: dump(result_to_file, file)
    # problem.obj_func_min.save_data(result_path)

    # ----------------------------------------------------------------- #
    #                      Plot the learning curve                      #
    # ----------------------------------------------------------------- #
    # fig, axes = plt.subplots(row_num, col_num, sharex=True, sharey=True, figsize=(col_num*4, row_num*3))
    # count = -1
    # for row in range(row_num):
    #     for col in range(col_num):
    #         count += 1
    #         axes[row, col].plot(s_lc_array*s_max, list_test_mu_[count])
    #         axes[row, col].grid(True)
    #         axes[row, col].set_ylim([np.amin(np.array(list_test_mu_)), np.amax(np.array(list_test_mu_))])
    # fig.tight_layout()
    # plt.savefig(result_path + '/prediction_curve' + str(kg_iteration) + '.pdf')