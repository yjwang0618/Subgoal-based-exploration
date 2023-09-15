import sys
from moe.optimal_learning.python.data_containers import HistoricalData
import numpy as np
import pickle
from pickle import dump
from joblib import Parallel, delayed

np.set_printoptions(threshold=sys.maxint)

# ================================================================================================= #
#                                      generate initial data                                        #
#                                           (single IS)                                             #
# ================================================================================================= #

# --------------------------------------------------------------------------- #
#                              load from file                                 #
# --------------------------------------------------------------------------- #
def load_sample_data(problem, num_per_var, exp_path, result_path):
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    with open(result_path+'_initial_samples.pickle', 'rb') as file: 
        list_init_pts_value_noise = pickle.load(file)
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
    count = -1
    repQL = problem.obj_func_min.repQL
    s_min = problem.obj_func_min.getSearchDomain()[0, 0]
    s_max = problem.obj_func_min.getSearchDomain()[0, 1]
    for s in np.linspace(s_min, s_max, num=problem.obj_func_min.getNums()):
        count += 1
        pts_value_noise = list_init_pts_value_noise[count]
        points = pts_value_noise[:, 0:-2]
        vals_array = pts_value_noise[:, -2]
        noise_array = pts_value_noise[:, -1]
        new_historical_data.append_historical_data(points, vals_array, noise_array)
    
    return new_historical_data

# def load_sample_data_numSubgoals(problem, num_per_var, exp_path, result_path):
#     var_dim = int(problem.obj_func_min.getDim()) - 1
#     num_initial_pts_per_s = int(num_per_var * var_dim)
#     with open(result_path+'_initial_samples.pickle', 'rb') as file: 
#         list_init_pts_value_noise = pickle.load(file)
#     new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
#     count = -1
#     repQL = problem.obj_func_min.repQL
#     s_min = problem.obj_func_min.getSearchDomain()[0, 0]
#     s_max = problem.obj_func_min.getSearchDomain()[0, 1]
#     for _ in range(2): # two choices of the number of subgoals
#         for s in np.linspace(s_min, s_max, num=problem.obj_func_min.getNums()):
#             count += 1
#             pts_value_noise = list_init_pts_value_noise[count]
#             points = pts_value_noise[:, 0:-2]
#             vals_array = pts_value_noise[:, -2]
#             noise_array = pts_value_noise[:, -1]
#             new_historical_data.append_historical_data(points, vals_array, noise_array)
    
#     return new_historical_data

# --------------------------------------------------------------------------- #
#                 general function for initial data generator                 #
# --------------------------------------------------------------------------- #
def sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path):
    list_init_pts_value_noise = []
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
    repQL = problem.obj_func_min.repQL
    s_min = problem.obj_func_min.getSearchDomain()[0, 0]
    s_max = problem.obj_func_min.getSearchDomain()[0, 1]
    for s in np.linspace(s_min, s_max, num=problem.obj_func_min.getNums()):
        random_seeds = np.random.randint(900, size=num_initial_pts_per_s)
        points = np.hstack((s * np.ones(num_initial_pts_per_s).reshape((-1, 1)), points_x))

        vals_array, noise_array = np.zeros(num_initial_pts_per_s), np.zeros(num_initial_pts_per_s)
        i = -1
        for (pt,random_seed) in zip(points, random_seeds):
            i += 1
            value, noise_array[i] = problem.obj_func_min.evaluate(repQL, pt, random_seed, exp_path)
            vals_array[i] = -1.0*value

        new_historical_data.append_historical_data(points, vals_array, noise_array)

        pts_value_noise = np.hstack(( points, vals_array.reshape((-1,1)), noise_array.reshape((-1,1)) ))
        list_init_pts_value_noise.append(pts_value_noise)
        with open(result_path+'_initial_samples.txt', "w") as file: 
            file.write(str(list_init_pts_value_noise))
        with open(result_path+'_initial_samples.pickle', "wb") as file: 
            dump(np.array(list_init_pts_value_noise), file)
    # print(list_init_pts_value_noise)
    return new_historical_data

# --------------------------------------------------------------------------- #
#              different constraints on generating initial data               #
# --------------------------------------------------------------------------- #
def sample_initial_x_uniform(problem, num_per_var, exp_path, result_path):
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

def sample_initial_f1f2_closer_f1_further_f2(problem, num_per_var, exp_path, result_path):
    ''' flag 1 is closer to the start than flag 2 '''
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_closer_f1_further_f2(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

def sample_initial_f1f2_higher_f1_lower_f2(problem, num_per_var, exp_path, result_path):
    ''' y(f1) <= y(f2) '''
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_higher_f1_lower_f2(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data



# ================================================================================================= #
#                                        select start points                                        #
# ================================================================================================= #

# --------------------------------------------------------------------------- #
#                 general function for selecting start points                 #
# --------------------------------------------------------------------------- #
def select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    '''
    create starting points for BFGS, first select points from previously sampled points,
    but not more than half of the starting points
    :return: numpy array with starting points for BFGS
    '''
    if len(list_sampled_points) > 0:
        indices_chosen = np.random.choice(len(list_sampled_points), 
                                          int(min(len(list_sampled_points), num_multistart/2.-1.)), 
                                          replace=False)
        start_pts_x = np.array(list_sampled_points)[:,1:][indices_chosen]
        start_pts_x = np.vstack((pt_x_to_start_from, start_pts_x)) # add the point that will be sampled next
    else:
        start_pts_x = [pt_x_to_start_from]
    return start_pts_x

# --------------------------------------------------------------------------- #
#               different constraints on selecting start points               #
# --------------------------------------------------------------------------- #
def select_startpts_x_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    # fill up with points from an LHS
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x

def select_startpts_f1closer_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_closer_f1_further_f2(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x

def select_startpts_f1higher_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_higher_f1_lower_f2(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x



# ================================================================================================= #
#                                      process_parallel_results                                     #
# ================================================================================================= #
def process_parallel_results(parallel_results):
    inner_min = np.inf
    for result in parallel_results:
        if inner_min > result[1]:
            inner_min = result[1]
            inner_min_point = result[0]
    return inner_min, inner_min_point