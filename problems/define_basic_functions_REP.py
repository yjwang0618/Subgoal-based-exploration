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
from .rlpy.Experiments.Experiment import Experiment, Experiment_MountainCar, Experiment_ql, Experiment_it
from .rlpy.Domains.GridWorld import GridWorld, GridWorld_Flag
from .rlpy.Domains.GridWorld_Key import GridWorld_Key, GridWorld_Key_Flag
from .rlpy.Domains.GridWorld_Items import GridWorld_Items, GridWorld_Items_Flag
from .rlpy.Domains.GridWorld_PuddleWindy import GridWorld_PuddleWindy, GridWorld_PuddleWindy_Flag
from .rlpy.Domains.MountainCar import MountainCar, MountainCar_flag

from .define_basic_functions import define_test_pts, make_experiment

np.set_printoptions(threshold=sys.maxint)

#=====================================================================================================#
#||=================================================================================================||#
#||          returns the mean, var and raw data of multiple replications of RL experiments          ||#
#||=================================================================================================||#
#=====================================================================================================#
def obj_parent(repQL, env, random_seed, noise_base, noise_rand, maze_num, maze_name):
    key_word='steps'
    np.random.seed(random_seed)
    noises = noise_base + noise_rand*np.random.random(repQL)
    mapnames = [None for _ in range(repQL)]
    items_positions = [None for _ in range(repQL)]
    WINDYs = [None for _ in range(repQL)]
    starts = [None for _ in range(repQL)]
    if env in ['gw', 'ky', 'it', 'pd', 'pt']:
        maze_orders = np.random.randint(maze_num, size=repQL)
        mapnames = []
        for maze_order in maze_orders:
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            mapnames.append(mapname)
        if env in ['it', 'pt']:
            key_word='discounted_return'
            item1_y = np.random.randint(low=0, high=3, size=repQL)
            item1_x = np.random.randint(low=7, high=10, size=repQL)
            for i in range(repQL):
                items_positions[i] = np.array([[item1_y[i], item1_x[i]]])
        if env=='pd':
            key_word='return'
            WINDY_ALL = [np.array([0,0,0,1,1,1,2,2,1,0]), 
                         np.array([0,0,0,1,1,1,2,1,1,0]),
                         np.array([0,0,0,0,1,1,2,2,1,0]),
                         np.array([0,0,0,1,1,2,2,2,1,0]),
                         np.array([0,0,0,1,1,1,2,2,1,0]),
                         np.array([0,0,0,1,1,1,2,1,1,0]),
                         np.array([0,0,0,1,1,1,2,0,1,0]),
                         np.array([0,0,0,1,1,1,1,2,1,0]),
                         np.array([0,0,0,0,1,1,1,1,1,0]),
                         np.array([0,0,0,0,1,1,2,1,1,0])]
            WINDY_order = np.random.randint(len(WINDY_ALL), size=repQL)
            WINDYs = WINDY_ALL[WINDY_order]
    if env=='mc':
        starts = -0.6 + 0.20 * np.random.random(repQL) # [-0.6, -0.4]
    return key_word, noises, mapnames, items_positions, WINDYs, starts


def obj(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noises, mapnames, items_positions, WINDYs, starts = \
                    obj_parent(repQL, env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    # print('noises', noises)
    # print('mapnames', mapnames)
    # print('items_positions', items_positions)
    # print('WINDYs', WINDYs)
    # print('starts', starts)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    seed = 1
    curve = np.zeros((repQL, num_policy_checks))
    for j in range(repQL):
        noise, mapname, exp_id = noises[j], mapnames[j], exp_ids[j]
        items_pos, WINDY, start = items_positions[j], WINDYs[j], starts[j]
        curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start)
    if key_word=='steps':
        y_mean = np.mean(np.log(curve), 0)
        y_noise = np.var(np.log(curve), 0) / repQL
    else:
        y_mean = np.mean(curve, 0)
        y_noise = np.var(curve, 0) / repQL
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(key_word)
    print(curve, y_mean, y_noise)
    return y_mean, y_noise, exp_ids, curve


# def obj_value(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
#         env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
#         noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
#     key_word, noises, mapnames, items_positions, WINDYs, starts = \
#                     obj_parent(repQL, env, random_seed, noise_base, noise_rand, maze_num, maze_name)
#     np.random.seed(random_seed)
#     seed = 1
#     curve = np.zeros((repQL, num_policy_checks))
#     for j in range(repQL):
#         noise, mapname, exp_id = noises[j], mapnames[j], exp_ids[j]
#         items_pos, WINDY, start = items_positions[j], WINDYs[j], starts[j]
#         curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
#                                      exp_id, seed, exp_path, env, flag_num, key_word, 
#                                      x, noise, mapname, items_pos, WINDY, start)
#     if key_word=='steps':
#         y_mean = np.mean(np.log(curve), 0)
#         y_noise = np.var(np.log(curve), 0) / repQL
#     else:
#         y_mean = np.mean(curve, 0)
#         y_noise = np.var(curve, 0) / repQL
#     return y_mean