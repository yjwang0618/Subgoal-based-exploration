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
from .rlpy.Experiments.Experiment import Experiment, Experiment_MountainCar, Experiment_ql
from .rlpy.Domains.GridWorld import GridWorld, GridWorld_Flag
from .rlpy.Domains.GridWorld_Items import GridWorld_Items_Flag
from .rlpy.Domains.GridWorld_PuddleWindy import GridWorld_PuddleWindy_Flag
from .rlpy.Domains.MountainCar import MountainCar, MountainCar_flag, MountainCar_flag2

np.set_printoptions(threshold=sys.maxint)


#=====================================================================================================#
#||=================================================================================================||#
#||                     define the points whose values are predicted in misoKG                      ||#
#||=================================================================================================||#
#=====================================================================================================#
def define_test_pts(prob_name):
    test_x = np.zeros((18, 4))
    row_num, col_num = 0, 0
    maze_num, maze_name = None, None
    noise_base, noise_rand = 0, 0
    flag_num = 2
    maze_size = None

    if prob_name == 'mcf2': # mountain-car with 2 flags
        noise_base, noise_rand = 0, 0.01
        p1,p2,v1,v2 = 0,0,0,0
        test_x = np.array([[-0.8,0.45], [-0.7,0.45], [-0.7,0.45],
                           [-0.8,0.45], [-0.8,0.4], [-0.8,0.3],
                           [-0.9,0.45], [-0.9,0.4], [-0.9,0.3]])
        # test_x = np.array([[-0.8,-0.01,0.45,0.01], [-0.7,-0.01,0.45,0.01], [-0.7,-0.02,0.45,0.01],
        #                    [-0.8,-0.01,0.45,0.02], [-0.8,-0.01,0.4,0.02], [-0.8,-0.00,0.3,0.02],
        #                    [-0.9,-0.01,0.45,0.02], [-0.9,-0.01,0.4,0.02], [-0.9,-0.00,0.3,0.02]])
                           
        row_num, col_num = 3, 3
        flag_num = 2
        repQL = 16
        s, S, skip, episodeCap = 4000, 2e4, 3000, 200

    elif prob_name == 'mcf3': # mountain-car with 3 flags
        noise_base, noise_rand = 0, 0.01
        p1,p2,p3,v1,v2,v3 = 0,0,0,0,0,0
        test_x = np.array([[-0.7,-0.3,0.45], [-0.7,-0.3,0.4], [-0.7,-0.3,0.3],
                           [-0.8,-0.3,0.45], [-0.8,-0.3,0.4], [-0.8,-0.3,0.3],
                           [-0.9,-0.3,0.45], [-0.9,-0.3,0.4], [-0.9,-0.3,0.3]])
        # test_x = np.array([[-0.7,-0.01, -0.03,0.04, 0.45,0.01], 
        #                      [-0.7,-0.01, -0.03,0.04, 0.4,0.01], 
        #                      [-0.7,-0.01, -0.03,0.04, 0.3,0.02],
        #                    [-0.8,-0.00, -0.03,0.04, 0.45,0.01], 
        #                      [-0.8,-0.00, -0.03,0.04, 0.4,0.01], 
        #                      [-0.8,-0.00, -0.03,0.04, 0.3,0.02],
        #                    [-0.9,-0.00, -0.03,0.04, 0.45,0.01], 
        #                      [-0.9,-0.00, -0.03,0.04, 0.4,0.01], 
        #                      [-0.9,-0.00, -0.03,0.04, 0.3,0.01]])
        row_num, col_num = 3, 4
        flag_num = 3
        repQL = 16
        s, S, skip, episodeCap = 4000, 2e4, 3000, 200

    return test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap



#=====================================================================================================#
#||=================================================================================================||#
#||                            run the RL algorithm in given environment                            ||#
#||=================================================================================================||#
#=====================================================================================================#
def make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_id, seed, exp_path, 
                    prob_name, lambda_, env, flag_num, key_word, 
                    x, noise, mapname, items_pos, WINDY, start):
    opt = {}
    opt["path"] = exp_path
    opt["exp_id"] = exp_id
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks
    opt["checks_per_policy"] = checks_per_policy

    epsilon = 0.2 # epsilon-greedy
    initial_learn_rate = 0.11
    boyan_N0 = 100

    if env=='mc':
        epsilon = 0.2 # epsilon-greedy
        gamm = 1
        initial_learn_rate = 0.5
        iter_min = max_steps / episodeCap
        iter_max = max_steps / 130
        iter_around = np.round((iter_min+iter_max)/2)
        boyan_N0 = 0.11 * iter_around / 0.9

        # if flag_num==2:
        #     FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
        #     FlagWid = np.array([2, 2])
        #     FlagHeight = np.array([1, 1])
        #     FlagHeight0 = np.array([0, 0])
        # elif flag_num==3:
        #     FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
        #     FlagWid = np.array([2, 2, 2])
        #     FlagHeight = np.array([1, 1, 1])
        #     FlagHeight0 = np.array([0, 0, 0])
        # elif flag_num==0: # Q-learning without flags
        #     if prob_name=='mcf2':
        #         FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
        #         FlagWid = np.array([2, 2])
        #         FlagHeight = np.array([0, 0])
        #         FlagHeight0 = np.array([0, 0])
        #     elif prob_name=='mcf3':
        #         FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
        #         FlagWid = np.array([2, 2, 2])
        #         FlagHeight = np.array([0, 0, 0])
        #         FlagHeight0 = np.array([0, 0, 0])

        if flag_num==2:
            FlagPos = np.array([x[0], x[1]])
            FlagWid = np.array([2, 2])
            FlagHeight = np.array([100, 100])
            FlagHeight0 = np.array([0, 0])
        elif flag_num==3:
            FlagPos = np.array([x[0], x[1], x[2]])
            FlagWid = np.array([2, 2, 2])
            FlagHeight = np.array([100, 100, 100])
            FlagHeight0 = np.array([0, 0, 0])
        elif flag_num==0: # Q-learning without flags
            if prob_name=='mcf2':
                FlagPos = np.array([x[0], x[1]])
                FlagWid = np.array([2, 2])
                FlagHeight = np.array([0, 0])
                FlagHeight0 = np.array([0, 0])
            elif prob_name=='mcf3':
                FlagPos = np.array([x[0], x[1], x[2]])
                FlagWid = np.array([2, 2, 2])
                FlagHeight = np.array([0, 0, 0])
                FlagHeight0 = np.array([0, 0, 0])
            
        domain = MountainCar_flag(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                  FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
        performance_domain = MountainCar_flag(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                              FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
        # domain = MountainCar(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap)
        # performance_domain = MountainCar(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap)
        # print(domain.statespace_limits)
        opt["domain"] = domain
        opt["performance_domain"] = performance_domain
        
        # representation = RBF(domain, grid_bins=np.array([20,10,5]), include_border=False)
        representation = RBF(domain, num_rbfs=1000)
        # print('representation.num_rbfs = ', representation.num_rbfs)
        print('dim', representation.dims)

        # representation = TileCoding(domain, memory=3000, num_tilings=[4,8], resolutions=[25,5], dimensions=[[0,1], [0,1,2]])

        # representation = Tabular(domain, discretization=[20,20,3])
        # print('representation.features_num = ', representation.features_num)
        # print('representation.bins_per_dim = ', representation.bins_per_dim)

        policy = eGreedy(representation, epsilon=epsilon, seed=seed)
        opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                  lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                  learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        # experiment = Experiment(**opt)
        experiment = Experiment_MountainCar(**opt)
        experiment.run(visualize_steps=False,  # should each learning step be shown?
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

    return np.array(output_list)



#=====================================================================================================#
#||=================================================================================================||#
#||          returns the mean, std and raw data of multiple replications of RL experiments          ||#
#||=================================================================================================||#
#=====================================================================================================#
def obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name):
    key_word='steps'
    np.random.seed(random_seed)
    noise = noise_base + noise_rand*np.random.random()
    mapname = None
    items_pos = None
    WINDY = None
    start = None
    if env=='mc':
        start = -0.6 + 0.20 * np.random.rand() # [-0.6, -0.4]
    return key_word, noise, mapname, items_pos, WINDY, start


def obj(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        prob_name, lambda_, env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    print('random_seed = ', random_seed)
    # print('maze_num = ', maze_num)
    # print('noise_base = ', noise_base)
    # print('noise_rand = ', noise_rand)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    curve = np.zeros((repQL, num_policy_checks))
    for j in range(repQL):
        exp_id = exp_ids[j]
        seed = 1
        curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, prob_name, lambda_, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)
    return y_mean, y_std, exp_ids, curve


def obj_parallel(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        prob_name, lambda_, env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    print('random_seed = ', random_seed)
    np.random.seed(random_seed)
    exp_ids = np.random.choice(np.arange(1,900), size=repQL, replace=False)
    with Parallel(n_jobs=-1) as parallel:
        curve = parallel(delayed(make_experiment)(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                 exp_ids[j], 1, exp_path, prob_name, lambda_, env, flag_num, key_word, 
                                 x, noise, mapname, items_pos, WINDY, start) for j in np.arange(repQL))
    y_mean = np.mean(np.array(curve), 0)
    y_std = np.std(np.array(curve), 0)
    return y_mean, y_std, exp_ids, np.array(curve)
