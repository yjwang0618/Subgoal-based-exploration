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

np.set_printoptions(threshold=sys.maxint)


#=====================================================================================================#
#||=================================================================================================||#
#||                     define the points whose values are predicted in misoKG                      ||#
#||=================================================================================================||#
#=====================================================================================================#

def define_test_pts(prob_name, default_flag_num=None):
    test_x = np.zeros((18, 4))
    row_num, col_num = 0, 0
    maze_num, maze_name = None, None
    noise_base, noise_rand = 0, 0
    if default_flag_num: flag_num = default_flag_num
    else: flag_num = 2
    maze_size = None

    # ========================================================================= #
    #                              GridWorld 10x10                              #
    # ========================================================================= #
    # -------- One room (0 wall) -------- #
    if prob_name == 'gw10One1':
        maze_num, maze_name = 1, '10x10_'
        noise_base, noise_rand = 0, 0
        y1, x1, y2, x2 = 4, 4, 8, 8
        test_x = np.array([[4,4,8,8], [3,7,8,8], [5,5,9,7], [3,7,9,7], [4,4,8,7], [3,7,8,7], # good
                           [4,4,9,6], [3,7,9,6], [5,5,8,6], [3,7,8,6], [4,4,7,7], [3,7,7,7], # ok
                           [4,4,9,5], [4,4,9,4], [4,4,8,5], [4,4,7,6], [4,4,6,6], [4,4,7,5], # bad
                           [1,8,9,3], [2,7,9,4], [3,6,9,2], [3,7,7,6], [4,4,8,1], [3,7,7,0]])# so bad
        row_num, col_num = 4, 6
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 200, 1000, 400, 500

    # -------- Two rooms (1 wall) -------- #
    elif prob_name in ['gw10Two1']: # 0-4,5-9
        maze_num = 5
        maze_name = '10x10_TwoRooms1_'

        noise_base, noise_rand = 0, 0.02
        y1, x1, y2, x2 = 3, 7, 8, 1
        test_x = np.array([[3,7,8,1], [2,7,8,1], [4,7,8,1], [4,8,8,1], [5,7,8,1], [6,8,8,1], # good
                           [1,8,9,2], [2,9,9,2], [0,7,9,2], [1,8,8,2], [2,9,8,2], [0,7,8,2], # ok
                           [1,8,8,5], [2,6,8,6], [0,7,8,7], [1,8,7,5], [2,6,7,8], [0,7,9,6]])# bad
        row_num, col_num = 3, 6
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 200, 1000, 400, 500


    elif prob_name == 'gw10Two2': # 0-8,-14
        maze_num, maze_name = 9, '10x10_TwoRooms2_'
        noise_base, noise_rand = 0, 0.02
        y1, x1, y2, x2 = 4, 7, 8, 1
        test_x = np.array([[4,7,8,1], [4,6,8,1], [5,7,8,1], [4,7,9,2], [4,7,7,0], [4,7,7,1], # good
                           [1,8,8,2], [2,9,8,2], [0,7,8,2], [1,8,9,3], [2,9,9,3], [0,7,9,3], # ok
                           [1,8,8,8], [2,9,8,7], [0,7,8,6], [1,8,9,8], [2,9,9,7], [0,7,9,6]])# bad
        row_num, col_num = 3, 6
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 200, 1000, 400, 500

    # -------- Three rooms (2 walls) -------- #
    elif prob_name == 'gw10Three1':
        maze_num, maze_name = None, '10x10_ThreeRooms1_'
        noise_base, noise_rand = 0, 0.02
        y1, x1, y2, x2 = 0, 0, 0, 0
        test_x = np.zeros((18, 4))
        row_num, col_num = 3, 6
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 200, 2000, 600, 500

    # -------- Two rooms with a pit -------- #
    elif prob_name in ['pt10']: # 0-4,5-9
        maze_num = 5
        maze_name = '10x10_TwoRooms1_pit_'

        noise_base, noise_rand = 0, 0.0
        y1, x1, y2, x2 = 3, 7, 8, 1
        test_x = np.array([[3,7,8,1], [2,7,8,1], [4,7,8,1], [4,8,8,1], [5,7,8,1], [6,8,8,1], # good
                           [1,8,9,2], [2,9,9,2], [0,7,9,2], [1,8,8,2], [2,9,8,2], [0,7,8,2], # ok
                           [1,8,8,5], [2,6,8,6], [0,7,8,7], [1,8,7,5], [2,6,7,8], [0,7,9,6]])# bad
        row_num, col_num = 3, 6
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 200, 1000, 400, 500

    # ========================================================================= #
    #                              GridWorld 20x20                              #
    # ========================================================================= #
    elif prob_name == 'gw20Three1':
        maze_num, maze_name = 18, '20x20_ThreeRooms1_'
        noise_base, noise_rand = 0.02, 0.02
        y1, x1, y2, x2 = 7, 15, 18, 18
        test_x = np.array([[5,15,18,18], [5,15,18,17], [5,15,18,16], [5,13,18,18], [5,13,18,17], [5,13,18,16], # good
                           [7,15,16,15], [7,15,15,15], [7,15,16,14], [7,13,16,15], [7,13,15,15], [7,13,16,14], # ok
                           [9,12,14,15], [9,12,16,12], [9,12,16,8], [7,13,14,15], [7,13,16,12], [7,13,16,8]])# bad
        row_num, col_num = 3, 6
        maze_size = 20
        repQL = 10
        s, S, skip, episodeCap = 4000, 10000, 3000, 1000

    elif prob_name == 'gw20Three2':
        maze_num, maze_name = 18, '20x20_ThreeRooms2_'
        noise_base, noise_rand = 0.02, 0.02
        y1, x1, y2, x2 = 7, 15, 18, 18
        test_x = np.array([[5,15,18,13], [5,15,17,13], [5,15,19,13], [5,13,18,13], [5,13,19,13], [5,13,17,13], # good
                           [7,15,16,13], [7,15,15,13], [7,15,16,13], [7,13,16,13], [7,13,15,13], [7,13,16,13], # ok
                           [9,12,14,15], [9,12,16,12], [9,12,16,8], [7,13,14,15], [7,13,16,12], [7,13,16,8]])# bad
        row_num, col_num = 3, 6
        maze_size = 20
        repQL = 20
        s, S, skip, episodeCap = 4000, 10000, 3000, 1000

    # ========================================================================= #
    #                            GridWorld with keys                            #
    # ========================================================================= #
    if prob_name == 'ky10One':
        # maze_num, maze_name = 12, '10x10_TwoRooms_i'
        maze_num, maze_name = 2, '10x10_TwoRooms_ii'
        noise_base, noise_rand = 0.0, 0
        # flag_num = 2
        flag_num = 3
        test_x = np.array([[1,9,9,1], [0,9,9,1], [1,9,9,2], [0,9,9,2], # good
                           [1,7,9,1], [0,7,9,1], [1,7,9,2], [0,7,9,2], # ok
                           [3,4,9,4], [4,6,9,4], [4,8,8,2], [3,8,7,0]])# bad
        row_num, col_num = 3, 4
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 400, 1000, 300, 500

    # ========================================================================= #
    #                          GridWorld with Subgoals                          #
    # ========================================================================= #
    if prob_name == 'it10':
        maze_num, maze_name = 1, '10x10_SmallRoom1_'
        noise_base, noise_rand = 0, 0
        test_x = np.array([[1,9,9,6], [0,9,9,6], [1,9,8,6], [0,9,8,6], # good
                           [1,7,9,7], [0,7,9,7], [1,7,9,5], [0,7,9,5], # ok
                           [3,4,7,4], [4,6,9,4], [4,8,8,2], [3,8,7,0]])# bad
        row_num, col_num = 3, 4
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 400, 2000, 800, 500

    # ========================================================================= #
    #                               Puddle World                                #
    # ========================================================================= #
    elif prob_name == 'pd10': # 7x10 puddleworld
        maze_num, maze_name = 12, '7x10_'
        noise_base, noise_rand = 0, 0.02
        y1, x1, y2, x2, y3, x3 = 2, 1, 1, 9, 3, 7
        test_x = np.array([[0,9,4,9,4,8], [0,9,4,9,4,8], [0,8,4,9,4,8], [0,8,4,9,4,8], [1,9,4,9,4,8], [1,9,4,9,4,8], # good
                           [2,1,1,9,4,8], [2,1,2,9,4,8], [2,1,0,9,4,8], [0,3,1,9,4,8], [0,3,2,9,4,8], [0,3,0,9,4,8], # ok
                           [1,3,1,9,4,9], [1,3,2,9,4,9], [2,1,2,8,4,9], [2,1,3,8,4,9], [2,4,1,8,4,9], [2,4,1,8,4,9]])# bad
        row_num, col_num = 3, 6
        flag_num = 3
        maze_size = 10
        repQL = 20
        s, S, skip, episodeCap = 800, 2000, 600, 1000

    # ========================================================================= #
    #                               Mountain-car                                #
    # ========================================================================= #
    elif prob_name in ['mcf1', 'mcf2', 'mcf3']:
        noise_base, noise_rand = 0, 0.01
        p1,p2,p3,v1,v2,v3 = 0,0,0,0,0,0
        row_num, col_num = 3, 3
        repQL = 50
        s, S, skip, episodeCap = 4000, 10000, 3000, 200

        if prob_name == 'mcf1': # mountain-car with 2 flags
            flag_num = 1
            test_x = np.array([[-0.7], [-0.65], [-0.75], # good
                               [0.4], [0.4], [0.4], # ok
                               [0.3], [0.3], [0.3]])# bad 
        elif prob_name == 'mcf2': # mountain-car with 2 flags
            flag_num = 2
            test_x = np.array([[-0.7, 0.5], [-0.65, 0.5], [-0.75, 0.5], # good
                               [-0.7, 0.4], [-0.65, 0.4], [-0.75, 0.4], # ok
                               [-0.7, 0.3], [-0.65, 0.3], [-0.75, 0.3]])# bad 
        elif prob_name == 'mcf3': # mountain-car with 3 flags
            flag_num = 3
            test_x = np.array([[-0.7, -0.5, 0.5], [-0.65, -0.5, 0.5], [-0.75, -0.5, 0.5], # good
                               [-0.7, -0.5, 0.4], [-0.65, -0.5, 0.4], [-0.75, -0.5, 0.4], # ok
                               [-0.7, -0.5, 0.3], [-0.65, -0.5, 0.3], [-0.75, -0.5, 0.3]])# bad 
        
    return test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap


def define_test_pts1(prob_name):
    test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap = define_test_pts(prob_name)
    if prob_name == 'gw10Two1':     s, S, skip, episodeCap = 300, 500, 200, 500
    return test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap


def define_test_pts2(prob_name):
    test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap = define_test_pts(prob_name)
    if prob_name == 'gw10Two1':     s, S, skip, episodeCap = 500, 1000, 500, 500
    return test_x, row_num, col_num, maze_num, maze_name, noise_base, noise_rand, \
           flag_num, maze_size, repQL, s, S, skip, episodeCap



#=====================================================================================================#
#||=================================================================================================||#
#||                            run the RL algorithm in given environment                            ||#
#||=================================================================================================||#
#=====================================================================================================#

def make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_id, seed, exp_path, 
                    env, flag_num, key_word, x, noise, mapname, items_pos, WINDY, start):
    opt = {}
    opt["path"] = exp_path
    opt["exp_id"] = exp_id
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks
    opt["checks_per_policy"] = checks_per_policy

    lambda_ = 0.0 # eligibility trace
    epsilon = 0.2 # epsilon-greedy
    initial_learn_rate = 0.11
    boyan_N0 = 100

    # ========================================================================= #
    #     the position of the agent is represented by the (y,x) coordinates     #
    # ========================================================================= #
    if env in ['gw', 'ky', 'it', 'pd', 'pt']:
        if env in ['gw', 'ky', 'it', 'pt']:
            if flag_num == 1:
                FlagPos = np.array([[x[0], x[1]]])
                FlagWid = np.array([[10]])
                FlagHeight = np.array([[0.2]])
                FlagHeight0 = np.array([[0]])
            if flag_num == 2:
                FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
                FlagWid = np.array([[10], [10]])
                FlagHeight = np.array([[0.2], [0.2]])
                FlagHeight0 = np.array([[0], [0]])
            if flag_num == 3:
                FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
                FlagWid = np.array([[10], [10], [10]])
                FlagHeight = np.array([[0.2], [0.2], [0.2]])
                FlagHeight0 = np.array([[0], [0], [0]])
            if flag_num == 0: # Q-learning without flags
                # FlagHeight = np.array([[0]])
                FlagHeight = np.array([[0], [0]])
                # FlagHeight = np.array([[0], [0], [0]])
        elif env == 'pd':
            FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
            FlagWid = np.array([[10], [10], [10]])
            FlagHeight = np.array([[0.2], [0.2], [0.2]])
            FlagHeight0 = np.array([[0], [0], [0]])
            if flag_num == 0: # Q-learning without flags
                FlagHeight = np.array([[0], [0], [0]])

        ### domains in different environments
        if env=='gw':
            # domain = GridWorld(mapname, noise=noise, episodeCap=episodeCap)
            # performance_domain = GridWorld(mapname, noise=noise, episodeCap=episodeCap)
            domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                                FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = performance_domain.discount_factor = 1

        elif env=='pt':
            domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                                FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = performance_domain.discount_factor = 1

        elif env=='ky':
            domain = GridWorld_Key_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                        FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = GridWorld_Key_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = performance_domain.discount_factor = 0.999

        elif env=='it':
            domain = GridWorld_Items_Flag(mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos, 
                                          FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = GridWorld_Items_Flag(mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos, 
                                                      FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = performance_domain.discount_factor = 0.98

        elif env=='pd':
            domain = GridWorld_PuddleWindy_Flag(mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY,
                                                FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = GridWorld_PuddleWindy_Flag(mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY,
                                                            FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = performance_domain.discount_factor = 0.98
            # print(domain.statespace_limits)
        
        ### run the experiment
        opt["domain"] = domain
        opt["performance_domain"] = performance_domain
        representation = Tabular(domain)
        # if representation.features_num > 400:
        #     representation = IncrementalTabular(domain)

        policy = eGreedy(representation, epsilon=epsilon, seed=seed)
        opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                  lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                  learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        if env in ['gw', 'ky', 'pd', 'pt']:
            experiment = Experiment(**opt)
        elif env in ['it']:
            experiment = Experiment_it(**opt)
        experiment.run(visualize_steps=False,  # should each learning step be shown?
                       visualize_learning=False,  # show policy / value function?
                       visualize_performance=False)  # show performance runs?
        experiment.save()

    # ========================================================================= #
    #    the position of the agent can be represented by one coordinate (x)     #
    # ========================================================================= #
    elif env=='mc':
        lambda_ = 0.2
        epsilon = 0.2 # epsilon-greedy
        gamm = 1
        initial_learn_rate = 0.5
        iter_min = max_steps / episodeCap
        iter_max = max_steps / 130
        iter_around = np.round((iter_min+iter_max)/2)
        boyan_N0 = 0.11 * iter_around / 0.9

        if flag_num==1:
            FlagPos = np.array([x[0]])
            FlagWid = np.array([2])
            FlagHeight = np.array([500])
            FlagHeight0 = np.array([0])
        elif flag_num==2:
            FlagPos = np.array([x[0], x[1]])
            # FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
            FlagWid = np.array([2, 2])
            FlagHeight = np.array([500, 500])
            FlagHeight0 = np.array([0, 0])
        elif flag_num==3:
            FlagPos = np.array([x[0], x[1], x[2]])
            FlagWid = np.array([2, 2, 2])
            FlagHeight = np.array([500, 500, 500])
            FlagHeight0 = np.array([0, 0, 0])
        elif flag_num==0: # Q-learning without flags
            FlagPos = np.array([x[0], x[1]])
            # FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
            FlagWid = np.array([2, 2])
            FlagHeight = np.array([0, 0])
            FlagHeight0 = np.array([0, 0])
            # FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
            # FlagWid = np.array([2, 2, 2])
            # FlagHeight = np.array([0, 0, 0])
            # FlagHeight0 = np.array([0, 0, 0])
            
        domain = MountainCar_flag(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                  FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
        performance_domain = MountainCar_flag(start=start, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                              FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
        # print(domain.statespace_limits)
        opt["domain"] = domain
        opt["performance_domain"] = performance_domain
        
        # representation = RBF(domain, grid_bins=np.array([20,10,5]), include_border=False)
        representation = RBF(domain, num_rbfs=1000)
        # print('representation.num_rbfs = ', representation.num_rbfs)
        # print('dim', representation.dims)
        # representation = TileCoding(domain, memory=1000, num_tilings=[8,4], resolutions=[10,5], dimensions=[[0,1], [0,1,2]])

        policy = eGreedy(representation, epsilon=epsilon, seed=seed)
        opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                  lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                  learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        experiment = Experiment(**opt)
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
    if env in ['gw', 'ky', 'it', 'pd', 'pt']:
        maze_order = np.random.randint(maze_num)
        mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
        print('mapname = ', mapname)
        if env in ['it', 'pt']:
            key_word='discounted_return'
            item1_y = np.random.randint(low=0,high=3)
            item1_x = np.random.randint(low=7,high=10)
            # item2_y = np.random.randint(low=8,high=10)
            # item2_x = np.random.randint(low=2,high=5)
            # items_pos = np.array([[item1_y, item1_x], [item2_y, item2_x]])
            items_pos = np.array([[item1_y, item1_x]])
            print('item position = ', items_pos)
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
            WINDY_order = np.random.randint(len(WINDY_ALL))
            WINDY = WINDY_ALL[WINDY_order]
    if env=='mc':
        start = -0.6 + 0.20 * np.random.rand() # [-0.6, -0.4]
    return key_word, noise, mapname, items_pos, WINDY, start


def obj(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    print('random_seed = ', random_seed)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    curve = np.zeros((repQL, num_policy_checks))
    for j in range(repQL):
        exp_id = exp_ids[j]
        seed = 1
        curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)
    return y_mean, y_std, exp_ids, curve


def obj_value(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    print('random_seed = ', random_seed)
    print('exp_ids = ', exp_ids)
    curve = np.zeros((repQL, num_policy_checks))
    for j in range(repQL):
        exp_id = exp_ids[j]
        seed = 1
        curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)
    return y_mean


def obj_value1(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    np.random.seed(random_seed)
    exp_ids = np.random.randint(low=1, high=900, size=repQL)
    seed = 1
    curve = np.zeros((repQL, num_policy_checks))
    for j in range(repQL):
        exp_id = exp_ids[j]
        curve[j,:] = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                     exp_id, seed, exp_path, env, flag_num, key_word, 
                                     x, noise, mapname, items_pos, WINDY, start)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)
    return y_mean[-1]


def obj_parallel(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, noise, mapname, items_pos, WINDY, start = \
                    obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    print('random_seed = ', random_seed)
    np.random.seed(random_seed)
    exp_ids = np.random.choice(np.arange(1,900), size=repQL, replace=False)
    with Parallel(n_jobs=-1) as parallel:
        curve = parallel(delayed(make_experiment)(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                 exp_ids[j], 1, exp_path, env, flag_num, key_word, 
                                 x, noise, mapname, items_pos, WINDY, start) for j in np.arange(repQL))
    y_mean = np.mean(np.array(curve), 0)
    y_std = np.std(np.array(curve), 0)
    return y_mean, y_std, exp_ids, np.array(curve)



#=====================================================================================================#
#||=================================================================================================||#
#||                      test the performance of x, return the whole y_mean                         ||#
#||=================================================================================================||#
#=====================================================================================================#

def _test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
          env, flag_num, key_word, x, noise_base, noise_rand, maze_num, maze_name, noise_num):
    
    def num_environment(maze_num,noise_num):
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
                    ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                         exp_id, seed, exp_path, env, flag_num, key_word, 
                                         x, noise, mapname, items_pos, WINDY, start)
                    curve.append(ys)

    elif env=='pt':
        key_word='discounted_return'
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    exp_id = j+1
                    ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                         exp_id, seed, exp_path, env, flag_num, key_word, 
                                         x, noise, mapname, items_pos, WINDY, start)
                    curve.append(ys)

    elif env=='ky':
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    exp_id = j+1
                    ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                         exp_id, seed, exp_path, env, flag_num, key_word, 
                                         x, noise, mapname, items_pos, WINDY, start)
                    curve.append(ys)

    elif env=='it':
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
                            ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                                 exp_id, seed, exp_path, env, flag_num, key_word, 
                                                 x, noise, mapname, items_pos, WINDY, start)
                            curve.append(ys)

    elif env=='pd':
        key_word='return'
        for maze_order in range(maze_num):
            mapname = os.path.join(num_environmentGridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                WINDY = np.array([0,0,0,1,1,1,2,2,1,0])
                # WINDY = np.zeros(10)
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    # print(x, maze_order, noise, j)
                    exp_id = j+1
                    ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                         exp_id, seed, exp_path, env, flag_num, key_word, 
                                         x, noise, mapname, items_pos, WINDY, start)
                    curve.append(ys)

    elif env=='mc':
        start_num, noise_num, _ = num_environment(maze_num,noise_num)
        for start_order in range(start_num+1):
            start = -0.6 + 0.20 * start_order / start_num # [-0.6, -0.4]
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                for j in range(repQL):
                    exp_id = j+1
                    ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                         exp_id, seed, exp_path, env, flag_num, key_word, 
                                         x, noise, mapname, items_pos, WINDY, start)
                    curve.append(ys)

    _, _, num_environment = num_environment(maze_num,noise_num)

    return np.array(curve), num_environment

# ========================================================================= #
#                         return the whole y_mean                           #
# ========================================================================= #
def obj_all_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', random_seed=0, x=None, 
        noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, _, _, _, _, _ = obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    repQL = 1
    noise_num = 5
    curve, num_environment = _test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
                                   env, flag_num, key_word, x, noise_base, noise_rand, maze_num, maze_name, noise_num)
    y_mean = np.mean(curve, 0)
    y_noise = np.var(curve, 0) / num_environment
    return y_mean

# ========================================================================= #
#                         return the last y_mean                            #
# ========================================================================= #
def obj_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
             env='gw', flag_num=2, key_word='steps', random_seed=0, X=None, index=0, 
             noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word, _, _, _, _, _ = obj_parent(env, random_seed, noise_base, noise_rand, maze_num, maze_name)
    repQL = 1
    noise_num = 5
    x = X[index]
    exp_path = exp_path+str(index)+'/'
    curve, num_environment = _test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
                                   env, flag_num, key_word, x, noise_base, noise_rand, maze_num, maze_name, noise_num)
    y_mean = np.mean(curve, 0)
    y_noise = np.var(curve, 0) / num_environment
    return y_mean[-1]

    # key_word='steps'
    # repQL = 1
    # noise_num = 5
    # mapname = None
    # items_pos = None
    # WINDY = None
    # start = None
    # curve = []
    # x = X[index]
    # exp_path = exp_path+str(index)+'/'
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # if env=='gw':
    #     for maze_order in range(maze_num):
    #         mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
    #         for noise_order in range(noise_num+1):
    #             noise = noise_base + noise_rand * noise_order / noise_num
    #             print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
    #             for j in range(repQL):
    #                 exp_id = j+1
    #                 seed = 1
    #                 ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
    #                                      exp_id, seed, exp_path, env, flag_num, key_word, 
    #                                      x, noise, mapname, items_pos, WINDY, start)
    #                 curve.append(ys)

    # elif env=='ky':
    #     for maze_order in range(maze_num):
    #         mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
    #         for noise_order in range(noise_num+1):
    #             noise = noise_base + noise_rand * noise_order / noise_num
    #             print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
    #             for j in range(repQL):
    #                 exp_id = j+1
    #                 seed = 1
    #                 ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
    #                                      exp_id, seed, exp_path, env, flag_num, key_word, 
    #                                      x, noise, mapname, items_pos, WINDY, start)
    #                 curve.append(ys)

    # if env=='it':
    #     key_word='discounted_return'
    #     for maze_order in range(maze_num):
    #         mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
    #         for noise_order in range(noise_num+1):
    #             noise = noise_base + noise_rand * noise_order / noise_num
    #             for item1_y in range(0,3):
    #                 for item1_x in range(7,10):
    #                     items_pos = np.array([[item1_y, item1_x]])
    #                     print( 'x = {0}, maze = {1}, noise = {2}, items_pos = {3}'.format(
    #                                                         x, maze_order, noise, items_pos) )
    #                     for j in range(repQL):
    #                         exp_id = j+1
    #                         seed = 1
    #                         ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
    #                                              exp_id, seed, exp_path, env, flag_num, key_word, 
    #                                              x, noise, mapname, items_pos, WINDY, start)
    #                         curve.append(ys)

    # if env=='pd':
    #     key_word='return'
    #     for maze_order in range(maze_num):
    #         mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
    #         for noise_order in range(noise_num+1):
    #             noise = noise_base + noise_rand * noise_order / noise_num
    #             WINDY = np.array([0,0,0,1,1,1,2,2,1,0])
    #             # WINDY = np.zeros(10)
    #             print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
    #             for j in range(repQL):
    #                 print(x, maze_order, noise, j)
    #                 exp_id = j+1
    #                 seed = 1
    #                 ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
    #                                      exp_id, seed, exp_path, env, flag_num, key_word, 
    #                                      x, noise, mapname, items_pos, WINDY, start)
    #                 curve.append(ys)

    # if env=='mc':
    #     start_num, noise_num = 2, 3
    #     for start_order in range(start_num+1):
    #         start = -0.6 + 0.20 * start_order / start_num # [-0.6, -0.4]
    #         for noise_order in range(noise_num+1):
    #             noise = noise_base + noise_rand * noise_order / noise_num

    #             for j in range(repQL):
    #                 exp_id = j+1
    #                 seed = 1
    #                 ys = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
    #                                      exp_id, seed, exp_path, env, flag_num, key_word, 
    #                                      x, noise, mapname, items_pos, WINDY, start)
    #                 curve.append(ys)

    # y_mean = np.mean(np.array(curve), 0)
    # y_std = np.std(np.array(curve), 0)

    # return y_mean[-1]
