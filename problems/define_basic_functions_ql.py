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
from pickle import dump
from joblib import Parallel, delayed
import numpy as np
from collections import defaultdict
import scipy.io
import matplotlib.pyplot as plt

from .rlpy.Agents.TDControlAgent import Q_Learning
from .rlpy.Policies.eGreedy import eGreedy
from .rlpy.Representations import Tabular, IncrementalTabular, RBF, TileCoding
from .rlpy.Representations_ql import Tabular as Tabular_ql
from .rlpy.Representations_ql import RBF as RBF_ql
from .rlpy.Experiments.Experiment import Experiment, Experiment_MountainCar, Experiment_ql, \
                                         Experiment_it, Experiment_it_ql
from .rlpy.Domains.GridWorld import GridWorld, GridWorld_Flag
from .rlpy.Domains.GridWorld_Key import GridWorld_Key, GridWorld_Key_Flag
from .rlpy.Domains.GridWorld_Items import GridWorld_Items, GridWorld_Items_Flag
from .rlpy.Domains.GridWorld_PuddleWindy import GridWorld_PuddleWindy, GridWorld_PuddleWindy_Flag
from .rlpy.Domains.MountainCar import MountainCar, MountainCar_flag

from .define_basic_functions import *

np.set_printoptions(threshold=sys.maxint)


#=====================================================================================================#
#||=================================================================================================||#
#||                            run the QL algorithm in given environment                            ||#
#||=================================================================================================||#
#=====================================================================================================#
def make_experiment_ql(max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_id, seed, exp_path, 
                    env, flag_num, key_word, x, noise, mapname, items_pos, WINDY, start, weight_vec_old):
    if x is not None: 
        return make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_id, seed, exp_path, 
                               env, flag_num, key_word, x, noise, mapname, items_pos, WINDY, goal)
    opt = {}
    opt["path"] = exp_path
    opt["exp_id"] = exp_id
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks
    opt["checks_per_policy"] = checks_per_policy
    weight_vec_new = None

    lambda_ = 0.0 # eligibility trace
    epsilon = 0.2 # epsilon-greedy
    initial_learn_rate = 0.11
    boyan_N0 = 100

    # ---- the position of the agent is represented by the (y,x) coordinates ---- #
    if env in ['gw', 'ky', 'it', 'pd']:
        if env=='gw':
            domain = GridWorld(mapname, noise=noise, episodeCap=episodeCap)
            domain.discount_factor = 1
        elif env=='ky':
            domain = GridWorld_Key(mapname, noise=noise, episodeCap=episodeCap)
            domain.discount_factor = 0.999
        elif env=='it':
            domain = GridWorld_Items(mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos)
            domain.discount_factor = 0.98
        elif env=='pd':
            domain = GridWorld_PuddleWindy(mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY)
            domain.discount_factor = 0.98

        opt["domain"] = domain
        opt["performance_domain"] = domain
        if weight_vec_old is None: representation = Tabular(domain)
        else: representation = Tabular_ql(domain, weight_vec_old)
        policy = eGreedy(representation, epsilon=epsilon, seed=seed)
        opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                  lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                  learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        if weight_vec_old is None: 
            if env in ['gw', 'ky', 'pd']:
                experiment = Experiment(**opt)
            elif env in ['it']:
                experiment = Experiment_it(**opt)
            experiment.run(visualize_steps=False,  # should each learning step be shown?
                       visualize_learning=False,  # show policy / value function?
                       visualize_performance=False)  # show performance runs?
            experiment.save()
        else:
            if env in ['gw', 'ky', 'pd']:
                experiment = Experiment_ql(**opt)
            elif env in ['it']:
                experiment = Experiment_it_ql(**opt)
            weight_vec_new = experiment.run(visualize_steps=False,  # should each learning step be shown?
                                            visualize_learning=False,  # show policy / value function?
                                            visualize_performance=False)  # show performance runs?
            experiment.save()

    # ---- the position of the agent can be represented by one coordinate (x) ---- #
    elif env=='mc':
        lambda_ = 0.2
        epsilon = 0.2 # epsilon-greedy
        gamm = 1
        initial_learn_rate = 0.5
        iter_min = max_steps / episodeCap
        iter_max = max_steps / 130
        iter_around = np.round((iter_min+iter_max)/2)
        boyan_N0 = 0.11 * iter_around / 0.9

        domain = MountainCar(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap)
        domain.discount_factor = 1
        opt["domain"] = domain
        opt["performance_domain"] = domain
        if weight_vec_old is None: representation = RBF(domain, num_rbfs=1000)
        else: representation = RBF_ql(domain, weight_vec_old, num_rbfs=1000)
        policy = eGreedy(representation, epsilon=epsilon, seed=seed)
        opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                  lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                  learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        if weight_vec_old is None: 
            experiment = Experiment(**opt)
            experiment.run(visualize_steps=False,  # should each learning step be shown?
                       visualize_learning=False,  # show policy / value function?
                       visualize_performance=False)  # show performance runs?
            experiment.save()
        else:
            experiment = Experiment_ql(**opt)
            weight_vec_new = experiment.run(visualize_steps=False,  # should each learning step be shown?
                                            visualize_learning=False,  # show policy / value function?
                                            visualize_performance=False)  # show performance runs?
            experiment.save()

    # ========================================================================= #
    #                       read the output from pickle                         #
    # ========================================================================= #
    if exp_id < 10: path_name = exp_path+'00'
    elif exp_id < 100: path_name = exp_path+'0'
    else: path_name = exp_path
    f = open(path_name+str(exp_id)+'-results.txt', 'r').read()
    f = f.replace('defaultdict(<type \'list\'>,', '')
    f = f.replace(')', '')
    f_dict = eval(f)
    f_output = f_dict.get(key_word)
    output_list = []
    k = -1
    for output in f_output:
        k += 1
        if k>0: output_list.append(output)

    return weight_vec_new, np.array(output_list)



#=====================================================================================================#
#||=================================================================================================||#
#||          returns the mean, std and raw data of multiple replications of RL experiments          ||#
#||=================================================================================================||#
#=====================================================================================================#

def obj_ql(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    weight_old = np.copy(weight_vec_old)
    curve = np.zeros((repQL, num_policy_checks))
    list_weight_vec_new = []
    for j in range(repQL):
        exp_id = exp_ids[j]
        seed = 1
        # print(weight_old)
        weight_vec_new, curve[j,:] = make_experiment_ql(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start, weight_old)
        list_weight_vec_new.append(weight_vec_new)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)
    if weight_vec_new is not None:
        Weight_vec_new = np.mean(list_weight_vec_new, axis=0)
    else:
        Weight_vec_new = None
    return Weight_vec_new, y_mean, y_std, exp_ids, curve


#=====================================================================================================#
#||=================================================================================================||#
#||                      test the performance of x, return the whole y_mean                         ||#
#||=================================================================================================||#
#=====================================================================================================#

def _ql_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
          env, flag_num, key_word, x, noise_base, noise_rand, maze_num, maze_name, noise_num, weight_old):

    mapname, items_pos, WINDY, start = None, None, None, None
    seed = 1
    curve = []

    if env=='gw':
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    exp_id = j+1
                    weight_vec_new, ys = make_experiment_ql(max_steps, episodeCap, num_policy_checks,  
                                     checks_per_policy, exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start, weight_old)
                    curve.append(ys)

    elif env=='ky':
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    exp_id = j+1
                    weight_vec_new, ys = make_experiment_ql(max_steps, episodeCap, num_policy_checks, 
                                     checks_per_policy, exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start, weight_old)
                    curve.append(ys)

    if env=='it':
        key_word='discounted_return'
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                for item1_y in range(0,3):
                    for item1_x in range(7,10):
                        items_pos = np.array([[item1_y, item1_x]])
                        print( 'x = {0}, maze = {1}, noise = {2}, items_pos = {3}'.format(
                                                            x, maze_order, noise, items_pos) )
                        for j in range(repQL):
                            exp_id = j+1
                            weight_vec_new, ys = make_experiment_ql(max_steps, episodeCap, num_policy_checks, 
                                     checks_per_policy, exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start, weight_old)
                            curve.append(ys)

    def num_environment(maze_num, noise_num):
        item_pos_num = 1
        WINDY_num = 1
        start_num = 1
        if env=='it':
            item_pos_num = 3*3
        if env=='pd':
            WINDY_num = 1
        if env=='mc':
            maze_num = 1
            start_num, noise_num = 10, 5
        return start_num, noise_num, maze_num*(noise_num+1)*item_pos_num*WINDY_num*(start_num+1)

    if env=='mc':
        start_num, noise_num, _ = num_environment(maze_num, noise_num)
        for start_order in range(start_num+1):
            start = -0.6 + 0.20 * start_order / start_num # [-0.6,-0.4]
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                for j in range(repQL):
                    exp_id = j+1
                    weight_vec_new, ys = make_experiment_ql(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start, weight_old)
                    curve.append(ys)

    _, _, num_environment = num_environment(maze_num, noise_num)

    return np.array(curve), num_environment


def obj_ql_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, index,
        env='gw', flag_num=2, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    random_seed = 0
    key_word, _, _, _, _, _ = obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    repQL = 1
    noise_num = 5
    weight_old = np.copy(weight_vec_old)
    exp_path = exp_path+str(index)+'/'
    curve, num_environment = _ql_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, env, 
                                    flag_num, key_word, x, noise_base, noise_rand, maze_num, maze_name, noise_num, weight_old)
    y_mean = np.mean(curve, 0)
    y_noise = np.var(curve, 0) / num_environment
    return y_mean[-1]
    