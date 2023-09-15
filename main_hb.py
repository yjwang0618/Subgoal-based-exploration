from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import sys
import cPickle as pickle
import os
from joblib import Parallel, delayed
import numpy as np
import pickle
import scipy.io
from collections import defaultdict
from random import random
from math import log, ceil
from time import time, ctime

from problems.define_basic_functions import obj_value1, define_test_pts

np.set_printoptions(threshold=sys.maxint)

#=====================================================================================================#
#||=================================================================================================||#
#||                                 the general class for hyperband                                 ||#
#||=================================================================================================||#
#=====================================================================================================#
class Hyperband(object):
    num_policy_checks = 1

    def __init__(self, prob_name, max_iter, eta, folder, sample, if_constraint):
        _test_x, _row_num, _col_num, self.maze_num, self.maze_name, self.noise_base, self.noise_rand, \
            self.flag_num, self.maze_size, self.repQL, self.s, self.S, _skip, self.episodeCap = define_test_pts(prob_name)
        self.env = prob_name[:2]
        if self.env in ['gw', 'ky', 'mc']: self.is_max = False # minimize the steps
        if self.env in ['it', 'pd']: self.is_max = True # maximize the reward
        
        self.steps_per_iter = int(self.S/2)
        self.max_iter = max_iter  	# maximum iterations per configuration
        self.eta = eta				# defines configuration downsampling rate (default = 3)
        self.folder = folder
        self.sample = sample
        self.if_constraint = if_constraint
        self.random_seeds = np.random.randint(900, size=100000)
        
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter
		
        self.list_max_steps = []
        self.list_cost = []
        self.list_params_sampled_x = []
        self.list_value = []
        self.list_counter, self.list_best_counter = [], []
        self.list_configurations, self.list_iterations = [], []
        self.list_s, self.list_n, self.list_r, self.list_i = [], [], [], []

        self.counter = 0
        self.best_counter, self.best_loss = -1, np.inf
        self.obj_visited, self.get_params_visit, self.try_params_visit = 0, 0, 0

    def run(self, skip_last=0):
        for s in reversed(range(0, self.s_max+1)):
            # -------------------------------------------------------------------------------------------- #
            #                               initial settings for each round                                #
            # -------------------------------------------------------------------------------------------- #
            n = int( ceil( self.B / self.max_iter / (s+1) * self.eta ** s ))	# initial number of configurations
            r = self.max_iter * self.eta ** (-s)	# initial number of iterations per config
            T = self.get_params(n) # n random configurations
            print(s)
            # -------------------------------------------------------------------------------------------- #
            #          decrease the # of T (n_configs), increase the # of max_steps (n_iterations)         #
            # -------------------------------------------------------------------------------------------- #
            for i in range((s+1) - int(skip_last)):	# changed from s + 1
                self.result_path = self.folder+'_experiment/'
                if not os.path.exists(self.result_path): os.makedirs(self.result_path)
                self.obj_visited = 0
                # ------- Run each of the n configs for <n_iterations>, ------- #
                # ------- and keep best (n_configs / eta) configurations ------ #
                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)
                max_steps = n_iterations * self.steps_per_iter
                print("\n*** {} configurations x {:.1f} iterations each".format(n_configs, n_iterations))
                # ------- settings for the current T ------- #
                t_num = len(T)
                counter0 = self.counter+1
                counters = np.arange(counter0, counter0 + t_num)
                costs = np.ones(t_num) * max_steps*self.repQL
                configurations = np.ones(t_num) * n_configs
                iterations = np.ones(t_num) * n_iterations
                s_s = np.ones(t_num) * s
                n_s = np.ones(t_num) * n
                r_s = np.ones(t_num) * r
                i_s = np.ones(t_num) * i
                self.list_max_steps.extend( np.ones(t_num) * max_steps )
                self.list_cost.extend(costs)
                self.list_params_sampled_x.extend(T)
                self.list_counter.extend(counters)
                self.list_configurations.extend(configurations)
                self.list_iterations.extend(iterations)
                self.list_s.extend(s_s)
                self.list_n.extend(n_s)
                self.list_r.extend(r_s)
                self.list_i.extend(i_s)
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('self.counter', self.counter)
                print('counters', counters)
                print('T', np.arange(len(T)))
                # ------- test over each t in T ------- #
                with Parallel(n_jobs=7) as parallel:
                    values = parallel( delayed(obj_value1)(self.repQL, max_steps, self.episodeCap, self.num_policy_checks, 
                                               self.checks_per_policy, self.result_path+'para'+str(counters[index])+'/', 
                                               env=self.env, flag_num=self.flag_num, random_seed=self.random_seeds[self.counter+index], 
                                               x=T[index], noise_base=self.noise_base, noise_rand=self.noise_rand, 
                                               maze_num=self.maze_num, maze_name=self.maze_name) for index in np.arange(len(T)) )
                self.list_value.extend(values)
                self.counter += t_num
                # ------- find the best counter ------- #
                val_losses = []
                for index, value in enumerate(values):
                    if self.is_max: loss = -1.0 * value
                    else: loss = value	
                    val_losses.append(loss)
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = counters[index]
                        self.list_best_counter.append(self.best_counter)
                # ------- save the data ------- #
                list_results = {"max_steps": np.array(self.list_max_steps),
                                "cost": np.array(self.list_cost),
                                'sampled_x': np.array(self.list_params_sampled_x),
                                "value": np.array(self.list_value),
                                'counter': np.array(self.list_counter),
                                'best_counter': np.array(self.list_best_counter),
                                'n_configs': np.array(self.list_configurations),
                                'n_iterations': np.array(self.list_iterations),
                                's': np.array(self.list_s),
                                'n': np.array(self.list_n),
                                'r': np.array(self.list_r),
                                'i': np.array(self.list_i)}
                with open(self.folder+'_list_results.pickle', "wb") as file: pickle.dump(list_results, file)
                with open(self.folder+'_list_results.txt', "w") as file: file.write(str(list_results))
                # ------- select a number of best configurations for the next loop ------- #
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices]
                T = T[0 : int(n_configs / self.eta)]
	
    def get_params(self, init_point_num):
        from GPyOpt.core.task.space import Design_space
        from GPyOpt.experiment_design import LatinDesign
        self.get_params_visit += 1
        np.random.seed(self.sample*10 + self.get_params_visit)
        space = Design_space(self.domain)
        paramsDesign = LatinDesign(space)
        if self.if_constraint:
            PTS_X = None
            while True:
                pts_x = paramsDesign.get_samples(int(init_point_num*2.5))
                pts_x_fit = None
                for pt_x in pts_x:
                    if pt_x[0] <= pt_x[2]:
                        if pts_x_fit is None: pts_x_fit = pt_x
                        else: pts_x_fit = np.vstack((pts_x_fit, pt_x))
                if PTS_X is None: PTS_X = pts_x_fit
                else: PTS_X = np.vstack((PTS_X, pts_x_fit))
                if PTS_X.shape[0] >= init_point_num:
                    PTS_X = PTS_X[np.random.choice(PTS_X.shape[0], init_point_num, replace=False), :]
                    break
        else:
            PTS_X = paramsDesign.get_samples(init_point_num)
        return PTS_X
    


#=====================================================================================================#
#||=================================================================================================||#
#||                                      the child classes                                          ||#
#||=================================================================================================||#
#=====================================================================================================#

class Hyperband_gw(Hyperband):
    def __init__(self, prob_name, max_iter, eta, folder, sample, if_constraint):
        super(Hyperband_gw, self).__init__(prob_name, max_iter, eta, folder, sample, if_constraint)
        wid_min = 0.1
        wid_max = self.maze_size*2
        self.domain = [{'name':'var_1', 'type':'continuous', 'domain':(0,self.maze_size-1), 'dimensionality':1},
                        {'name':'var_2', 'type':'continuous', 'domain':(0,self.maze_size-1), 'dimensionality':1},
                        {'name':'var_3', 'type':'continuous', 'domain':(0,self.maze_size-1), 'dimensionality':1},
                        {'name':'var_4', 'type':'continuous', 'domain':(0,self.maze_size-1), 'dimensionality':1}
                        # {'name':'var_5', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1},
                        # {'name':'var_6', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1}
                        ]
        self.checks_per_policy = 1


class Hyperband_mc(Hyperband):
    def __init__(self, prob_name, max_iter, eta, folder, sample, if_constraint):
        super(Hyperband_mc, self).__init__(prob_name, max_iter, eta, folder, sample, if_constraint)
        position_min = -1.2
        position_max = 0.6
        x_size = 2
        wid_min = 0.1
        wid_max = x_size*5
        domain1 = [{'name':'var_1', 'type':'continuous', 'domain':(position_min,position_max), 'dimensionality':1}]
		           # {'name':'var_2', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1}]
        domain2 = [{'name':'var_1', 'type':'continuous', 'domain':(position_min,position_max), 'dimensionality':1},
                   {'name':'var_2', 'type':'continuous', 'domain':(position_min,position_max), 'dimensionality':1}]
                   # {'name':'var_3', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1},
                   # {'name':'var_4', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1}]
        if self.flag_num==1: self.domain = domain1
        if self.flag_num==2: self.domain = domain2
        self.checks_per_policy = 10


class Hyperband_pd(Hyperband):
    def __init__(self, prob_name, max_iter, eta, folder, sample, if_constraint):
        super(Hyperband_pd, self).__init__(prob_name, max_iter, eta, folder, sample, if_constraint)
        wid_min = 0.1
        wid_max = self.maze_size*2
        self.domain = [{'name':'var_1', 'type':'continuous', 'domain':(0,6), 'dimensionality':1},
                       {'name':'var_2', 'type':'continuous', 'domain':(0,9), 'dimensionality':1},
                       {'name':'var_3', 'type':'continuous', 'domain':(0,6), 'dimensionality':1},
                       {'name':'var_4', 'type':'continuous', 'domain':(0,9), 'dimensionality':1}]
                       # {'name':'var_5', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1},
                       # {'name':'var_6', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1}]
        self.checks_per_policy = 1


#=====================================================================================================#
#||=================================================================================================||#
#||                                              main                                               ||#
#||=================================================================================================||#
#=====================================================================================================#
if __name__ == '__main__':
    argv = sys.argv[1:]
    '''
    python main_hb.py gw10Two1 0 i
        argv[0] prob_name
            gw10One1, gw10Two1, gw10Two2, gw10Three1; gw20Three1
            ky10One
            it10
            pd10
            mcf1, mcf2
        argv[1] constraint
            0, 1
        argv[2] sample_num
            0,1,2,3,4,...
	'''
    prob_name = argv[0]
    env = prob_name[:2]
    if_constraint = int(argv[1])
    sample = int(argv[2])
    
    max_iter = 81  	# maximum iterations per configuration
    eta = 3			# defines configuration downsampling rate (default = 3)
    
    folder = './REP_rlt_'+prob_name+'/Results_hb_'+prob_name+'/sample'+str(sample)
    np.random.seed(sample)
    if env in ['gw', 'ky', 'it']:
        hb = Hyperband_gw(prob_name, max_iter, eta, folder, sample, if_constraint)
    elif env == 'pd':
        hb = Hyperband_pd(prob_name, max_iter, eta, folder, sample, if_constraint)
    elif env == 'mc':
        hb = Hyperband_mc(prob_name, max_iter, eta, folder, sample, if_constraint)
    hb.run(skip_last=0)