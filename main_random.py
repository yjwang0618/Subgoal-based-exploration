from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library

standard_library.install_aliases()
import sys
import os
import os.path
import pickle
import numpy as np
from collections import defaultdict
import scipy.io
from smt.sampling_methods import LHS
    
from problems.define_basic_functions import obj_parallel, define_test_pts

np.set_printoptions(threshold=sys.maxint)


def initialize(prob_name):
    _, _, _, maze_num, maze_name, noise_base, noise_rand, \
        flag_num, maze_size, repQL, s, S, skip, episodeCap = define_test_pts(prob_name)

    max_iter_ = None
    is_max, mult = None, None
    num_policy_checks, checks_per_policy = 1, 1

    if prob_name in ['gw10Two1', 'gw20Three1', 'it10', 'ky10One']:
        dimension = int(flag_num * 2)
        bounds = [[0, maze_size - 1]] * dimension
        if prob_name == 'gw10Two1':
            max_iter_ = 40
        if prob_name == 'gw20Three1':
            max_iter_ = 80
        if prob_name == 'it10':
            max_iter_ = 30
        if prob_name == 'ky10One':
            max_iter_ = 30
    elif prob_name in ['mcf1', 'mcf2', 'mcf3']:
        max_iter_ = 50
        position_min = -1.2
        position_max = 0.6
        if prob_name == 'mcf1':
            dimension = 1
        elif prob_name == 'mcf2':
            dimension = 2
        elif prob_name == 'mcf3':
            dimension = 3
        bounds = [[position_min, position_max]] * dimension

    if env in ['gw', 'ky', 'mc']:
        is_max, mult = False, -1  # minimize the steps
    elif env in ['it', 'pd']:
        is_max, mult = True, 1  # maximize the reward
    if env in ['mc']:
        checks_per_policy = 10

    return maze_num, maze_name, noise_base, noise_rand, \
        flag_num, maze_size, repQL, s, S, skip, episodeCap, \
        max_iter_, is_max, mult, num_policy_checks, checks_per_policy, \
        np.array(bounds)


def sample_x(bounds, num_pts, if_constraint=False):
    if if_constraint:
        sampled_x = None
        while True:
            sampling = LHS(xlimits=bounds)
            large_sample_x = sampling(int(num_pts * 2.5))
            pts_x_fit = None
            for pt_x in large_sample_x:
                if pt_x[0] <= pt_x[2]:
                    if pts_x_fit is None:
                        pts_x_fit = pt_x
                    else:
                        pts_x_fit = np.vstack((pts_x_fit, pt_x))
            if sampled_x is None:
                sampled_x = pts_x_fit
            else:
                sampled_x = np.vstack((sampled_x, pts_x_fit))
            if sampled_x.shape[0] >= num_pts:
                sampled_x = sampled_x[np.random.choice(sampled_x.shape[0], num_pts, replace=False), :]
                break
    else:
        sampling = LHS(xlimits=bounds)
        sampled_x = sampling(num_pts)
    return sampled_x


def obj_val(x):
    global random_seeds, obj_count, list_sampled_x, list_y, list_cost
    obj_count += 1
    a_seed = random_seeds[obj_count]
    # x = x[0, :]
    print(x)
    y_mean, _, _, _ = obj_parallel(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path,
                                   env=env, flag_num=flag_num, random_seed=a_seed, x=x,
                                   noise_base=noise_base, noise_rand=noise_rand, maze_num=maze_num, maze_name=maze_name)
    y_mean = y_mean[-1]

    list_sampled_x.append(x)
    list_y.append(y_mean)
    list_cost.append(max_steps * repQL)
    return y_mean


# ============================================ #
#                     main                     #
# ============================================ #
if __name__ == '__main__':

    argv = sys.argv[1:]
    '''
    python main_random.py ky10One 0 0
        argv[0] prob_name
            gw10Two1
            gw20Three1
            it10
            ky10One
            mcf2
        argv[1] if_constraint
            0, 1
        argv[2] random seed
            0,1,2,3,...
    '''
    prob_name = argv[0]
    env = prob_name[:2]
    if_constraint = int(argv[1])
    random_seed = int(argv[2])
    np.random.seed(random_seed)
    random_seeds = np.random.randint(900, size=500)

    result_path = './REP_rlt_' + prob_name + '/Results_random_' + str(if_constraint) + '_' + prob_name + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    exp_path = result_path + 'sample' + str(random_seed) + '/'
    txt_path = result_path + 'sample' + str(random_seed)

    maze_num, maze_name, noise_base, noise_rand, \
        flag_num, maze_size, repQL, s, S, skip, episodeCap, \
        max_iter_, is_max, mult, num_policy_checks, checks_per_policy, \
        bounds = initialize(prob_name)
    s_num = int((S - s) / skip)
    max_steps = S
    list_x = sample_x(bounds, max_iter_, if_constraint=if_constraint)

    list_sampled_x, list_y, list_cost = [], [], []
    init_iter = 0
    max_iter = max_iter_
    obj_count = -1
    if os.path.isfile(txt_path + '_result.pickle'):
        with open(txt_path + '_result.pickle', 'rb') as file:
            f_dict = pickle.load(file)
        array_sampled_x = f_dict.get('sampled_x')
        array_y = f_dict.get('observed_y')
        array_cost = f_dict.get('cost')
        for ind, cost in enumerate(array_cost):
            list_sampled_x.append(array_sampled_x[ind, :])
            list_y.append(array_y[ind])
            list_cost.append(array_cost[ind])
        init_iter = len(list_y)
        obj_count = len(list_y) - 1

    for iter in range(init_iter, max_iter):
        y_mean = obj_val(list_x[iter])

    result = {
        'sampled_x': np.array(list_sampled_x),
        'observed_y': np.array(list_y),
        'cost': np.array(list_cost)
    }
    print(list_cost[-1])
    with open(txt_path + '_result.txt', "w") as file:
        file.write(str(result))
    with open(txt_path + '_result.pickle', "wb") as file:
        pickle.dump(result, file)
