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
from .rlpy.Domains.MountainCar import MountainCar, MountainCar_flag

np.set_printoptions(threshold=sys.maxint)



#=====================================================================================================#
#||=================================================================================================||#
#||                            run the RL algorithm in given environment                            ||#
#||=================================================================================================||#
#=====================================================================================================#
def make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_id, seed, exp_path, 
                    env, flag_num, key_word, 
                    x, noise, mapname, items_pos, WINDY, goal, weight_vec_old):
    opt = {}
    opt["path"] = exp_path
    opt["exp_id"] = exp_id
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks
    opt["checks_per_policy"] = checks_per_policy
    weight_vec_new = None
    if weight_vec_old is not None: flag_num = 0

    # ========================================================================= #
    #     the position of the agent is represented by the (y,x) coordinates     #
    # ========================================================================= #
    if env in ['gw', 'it', 'pd']:
        if env in ['gw', 'it']:
            FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
            FlagWid = np.array([[10], [10]])
            FlagHeight = np.array([[0.2], [0.2]])
            FlagHeight0 = np.array([[0], [0]])
            if flag_num == 0: # Q-learning without flags
                FlagHeight = np.array([[0], [0]])
        if env == 'pd':
            FlagPos = np.array([[x[0], x[1]], [x[2], x[3]], [x[4], x[5]]])
            FlagWid = np.array([[10], [10], [10]])
            FlagHeight = np.array([[0.2], [0.2], [0.2]])
            FlagHeight0 = np.array([[0], [0], [0]])
            if flag_num == 0: # Q-learning without flags
                FlagHeight = np.array([[0], [0], [0]])

        # ----------------------------------------------- #
        #    Q-learning with inheritting weight vector    #
        # -----------------------------------------  ---- #
        if weight_vec_old is not None:
            if env=='gw':
                domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                        FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = 1
            elif env=='it':
                domain = GridWorld_Items_Flag(mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos, 
                                              FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = 0.999
            elif env=='pd':
                domain = GridWorld_PuddleWindy_Flag(mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY,
                                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = 0.98

            opt["domain"] = domain
            opt["performance_domain"] = domain
            representation = Tabular_ql(domain, weight_vec_old)
            policy = eGreedy(representation, epsilon=epsilon, seed=seed)
            opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                      lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
            experiment = Experiment_ql(**opt)
            weight_vec_new = experiment.run(visualize_steps=False,  # should each learning step be shown?
                                            visualize_learning=False,  # show policy / value function?
                                            visualize_performance=False)  # show performance runs?
            experiment.save()

        # ----------------------------------------------- #
        #   training without inheritting weight vector    #
        # ----------------------------------------------- #
        else:
            ### domains in different environments
            if env=='gw':
                domain = GridWorld_Flag(mapname, noise=noise, episodeCap=episodeCap, 
                                        FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
                performance_domain = GridWorld_Flag(
                                        mapname, noise=noise, episodeCap=episodeCap, 
                                        FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = performance_domain.discount_factor = 1

            elif env=='it':
                domain = GridWorld_Items_Flag(mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos, 
                                              FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
                performance_domain = GridWorld_Items_Flag(
                                              mapname, noise=noise, episodeCap=episodeCap, items_pos=items_pos, 
                                              FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = performance_domain.discount_factor = 0.999

            elif env=='pd':
                domain = GridWorld_PuddleWindy_Flag(mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY,
                                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
                performance_domain = GridWorld_PuddleWindy_Flag(
                                                    mapname, noise=noise, episodeCap=episodeCap, WINDY=WINDY,
                                                    FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
                domain.discount_factor = performance_domain.discount_factor = 0.98
            
            ### run the experiment
            opt["domain"] = domain
            opt["performance_domain"] = performance_domain
            representation = Tabular(domain)
            if representation.features_num > 300:
                representation = IncrementalTabular(domain)

            policy = eGreedy(representation, epsilon=epsilon, seed=seed)
            opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                      lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
            experiment = Experiment(**opt)
            experiment.run(visualize_steps=False,  # should each learning step be shown?
                           visualize_learning=False,  # show policy / value function?
                           visualize_performance=False)  # show performance runs?
            experiment.save()

    # ========================================================================= #
    #    the position of the agent can be represented by one coordinate (x)     #
    # ========================================================================= #
    elif env=='mc':
        lambda_ = 0.0
        epsilon = 0.1
        gamm = 1
        initial_learn_rate = 0.5
        iter_min = max_steps / episodeCap
        iter_max = max_steps / 130
        iter_around = np.round((iter_min+iter_max)/2)
        boyan_N0 = 0.11 * iter_around / 0.9

        if flag_num==1:
            FlagPos = np.array([x[0]])
            FlagWid = np.array([1])
            FlagHeight = np.array([1])
            FlagHeight0 = np.array([0])
        elif flag_num==2:
            FlagPos = np.array([x[0], x[1]])
            FlagWid = np.array([1, 1])
            FlagHeight = np.array([1, 1])
            FlagHeight0 = np.array([0, 0])
        elif flag_num==3:
            FlagPos = np.array([x[0], x[1], x[3]])
            FlagWid = np.array([1, 1, 1])
            FlagHeight = np.array([1, 1, 1])
            FlagHeight0 = np.array([0, 0, 0])
        elif flag_num==0: # Q-learning without flags
            FlagPos = np.array([x[0], x[1]])
            FlagWid = np.array([1, 1])
            FlagHeight = np.array([0, 0])
            FlagHeight0 = np.array([0, 0])
        # ----------------------------------------------- #
        #    Q-learning with inheritting weight vector    #
        # -----------------------------------------  ---- #
        if weight_vec_old is not None:
            domain = MountainCar_flag(goal=goal, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                      FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            domain.discount_factor = 1
            opt["domain"] = domain
            opt["performance_domain"] = domain
            representation = RBF_ql(domain, weight_vec_old, num_rbfs=features_num)
            policy = eGreedy(representation, epsilon=epsilon, seed=seed)
            opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                      lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
            experiment = Experiment_ql(**opt)
            weight_vec_new = experiment.run(visualize_steps=False,  # should each learning step be shown?
                                            visualize_learning=False,  # show policy / value function?
                                            visualize_performance=False)  # show performance runs?
            experiment.save()
        # ----------------------------------------------- #
        #   training without inheritting weight vector    #
        # ----------------------------------------------- #
        else:
            domain = MountainCar_flag(goal=goal, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                      FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
            performance_domain = MountainCar_flag(goal=goal, noise=noise, discount_factor=gamm, episodeCap=episodeCap, 
                                                  FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight0)
            opt["domain"] = domain
            opt["performance_domain"] = performance_domain
            representation = RBF(domain, num_rbfs=100)
            policy = eGreedy(representation, epsilon=epsilon, seed=seed)
            opt["agent"] = Q_Learning(policy, representation, discount_factor=domain.discount_factor,
                                      lambda_=lambda_, initial_learn_rate=initial_learn_rate,
                                      learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
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

    return np.array(output_list), weight_vec_new



#=====================================================================================================#
#||=================================================================================================||#
#||          returns the mean, std and raw data of multiple replications of RL experiments          ||#
#||=================================================================================================||#
#=====================================================================================================#
def obj(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
        env='gw', flag_num=2, key_word='steps', 
        random_seed=0, x=None, noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word='steps'
    np.random.seed(random_seed)
    noise = noise_base + noise_rand*np.random.random()
    mapname = None
    items_pos = None
    WINDY = None
    goal = None
    # ---------------------------------------------------- #
    #          the randomness in each environment          #
    # ---------------------------------------------------- #
    if env=='gw' or env=='it' or env=='pd':
        maze_order = np.random.randint(maze_num)
        mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
        print('mapname = ', mapname)
        if env=='it':
            key_word='discounted_return'
            item1_y = np.random.randint(low=1,high=3)
            item1_x = np.random.randint(low=3,high=6)
            item2_y = np.random.randint(low=6,high=8)
            item2_x = np.random.randint(low=2,high=5)
            items_pos = np.array([[item1_y, item1_x], [item2_y, item2_x]])
            print('item position = ', items_pos)
        if env=='pd':
            key_word='return'
            WINDY = np.array([0,0,0,1,1,1,2,2,1,0])
            # WINDY = np.zeros(10)
    if env=='mc':
        goal = 0.45 + 0.10 * np.random.rand() # [0.45, 0.55]

    # ---------------------------------------------------- #
    #         run the experiment and get raw data          #
    # ---------------------------------------------------- #
    print('random_seed = ', random_seed)
    # print('maze_num = ', maze_num)
    # print('noise_base = ', noise_base)
    # print('noise_rand = ', noise_rand)
        
    curve = np.zeros((repQL, num_policy_checks))
    exp_ids = np.zeros(repQL)
    for j in range(repQL):
        exp_id = j+1
        # exp_id = np.random.randint(low=1, high=900)
        exp_ids[j] = exp_id
        seed = 1
        curve[j,:], weight_vec_new = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                                     exp_id, seed, exp_path, 
                                                     env, flag_num, key_word, 
                                                     x, noise, 
                                                     mapname, items_pos, WINDY, goal, weight_vec_old)
    y_mean = np.mean(curve, 0)
    y_std = np.std(curve, 0)

    if weight_vec_old is None:
        return y_mean, y_std, exp_ids, curve
    else:
        return weight_vec_new, y_mean, y_std, exp_ids, curve



#=====================================================================================================#
#||=================================================================================================||#
#||                                   test the performance of x                                     ||#
#||=================================================================================================||#
#=====================================================================================================#
def obj_test(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
             env='gw', flag_num=2, key_word='steps', 
             random_seed=0, X=None, index=0, noise_base=0, noise_rand=0, maze_num=None, maze_name=None, weight_vec_old=None):
    key_word='steps'
    repQL = 1
    noise_num = 5
    mapname = None
    items_pos = None
    WINDY = None
    goal = None
    curve = []
    x = X[index]
    exp_path = exp_path+str(index)+'/'
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if env=='gw':
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    exp_id = j+1
                    seed = 1
                    ys, weight_vec_new = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                                         exp_id, seed, exp_path, 
                                                         env, flag_num, key_word, 
                                                         x, noise, 
                                                         mapname, items_pos, WINDY, goal, weight_vec_old)
                    curve.append(ys)

    if env=='it':
        key_word='discounted_return'
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                for item1_y in range(1,3):
                    for item1_x in range(3,6):
                        for item2_y in range(6,8):
                            for item2_x in range(2,5):
                                items_pos = np.array([[item1_y, item1_x], [item2_y, item2_x]])
                                print( 'x = {0}, maze = {1}, noise = {2}, items_pos = {3}'.format(x, maze_order, noise, items_pos) )
                                for j in range(repQL):
                                    exp_id = j+1
                                    seed = 1
                                    ys, weight_vec_new = make_experiment(max_steps, episodeCap, num_policy_checks, 
                                                                         checks_per_policy, 
                                                                         exp_id, seed, exp_path, 
                                                                         env, flag_num, key_word, 
                                                                         x, noise, 
                                                                         mapname, items_pos, WINDY, goal, weight_vec_old)
                                    curve.append(ys)

    if env=='pd':
        key_word='return'
        for maze_order in range(maze_num):
            mapname = os.path.join(GridWorld.default_map_dir, maze_name+str(maze_order)+'.txt')
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                WINDY = np.array([0,0,0,1,1,1,2,2,1,0])
                # WINDY = np.zeros(10)
                print( 'x = {0}, maze = {1}, noise = {2}'.format(x, maze_order, noise) )
                for j in range(repQL):
                    print(x, maze_order, noise, j)
                    exp_id = j+1
                    seed = 1
                    ys, weight_vec_new = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                                         exp_id, seed, exp_path, 
                                                         env, flag_num, key_word, 
                                                         x, noise, 
                                                         mapname, items_pos, WINDY, goal, weight_vec_old)
                    curve.append(ys)

    if env=='mc':
        goal_num = 10
        for goal_order in range(goal_num+1):
            goal = 0.45 + 0.10 * goal_order / goal_num # [0.45, 0.55]
            for noise_order in range(noise_num+1):
                noise = noise_base + noise_rand * noise_order / noise_num
                for j in range(repQL):
                    exp_id = j+1
                    seed = 1
                    ys, weight_vec_new = make_experiment(max_steps, episodeCap, num_policy_checks, checks_per_policy, 
                                                         exp_id, seed, exp_path, 
                                                         env, flag_num, key_word, 
                                                         x, noise, 
                                                         mapname, items_pos, WINDY, goal, weight_vec_old)
                    curve.append(ys)

    y_mean = np.mean(np.array(curve), 0)
    y_std = np.std(np.array(curve), 0)

    return y_mean[-1]