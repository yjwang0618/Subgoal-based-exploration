from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from abc import ABCMeta, abstractproperty
from future import standard_library
standard_library.install_aliases()
import os
import pickle
from pickle import dump
import numpy as np
import math

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

from .define_basic_functions import define_test_pts, obj


class misoPuddle_(object):
    num_policy_checks = 1
    checks_per_policy = 1
    all_data = []
    all_data_raw = []

    def __init__(self, prob_name, version, mult=-1.0):
        _test_x, _row_num, _col_num, \
            self.maze_num, self.maze_name, self.noise_base, self.noise_rand, \
            self.flag_num, self.maze_size, self.repQL, \
            self.s, self.S, self.skip, self.episodeCap = define_test_pts(prob_name)
        self.s_min = self.s/self.S
        self.s_max = 1
        self.skip /= self.S
        self.s_array = np.linspace(self.s_min, self.s_max, num=self.getNums())
        print(self.s_array)

        self.cost_fix = self.S*2
        self.version = version
        self._mult = mult
        self.evaluate_called = 0

        self._dim = self.flag_num*2+1
        self._search_domain = np.zeros((self._dim, 2))
        self._search_domain[0,:] = [self.s_min, self.s_max]
        self._search_domain[1,:] = [0, 6]
        self._search_domain[2,:] = [0, 9]
        self._search_domain[3,:] = [0, 6]
        self._search_domain[4,:] = [0, 9]
        # for row in range(self.flag_num*2+1, self.flag_num*3+1):
        #     self._search_domain[row,:] = [0.1, self.maze_size*2]

        # self._meanval = -100 # S, episodeCap = 1000, 500
        self._meanval = -3.0666 # S, episodeCap = 2000, 500
        self.noise = []
        self.y_min = None
        self.y_max = None
        
    def evaluate(self, x, random_seed, exp_path='./initResults/'):
        max_steps = int(x[0] * self.S)
        if self.version == 1: repQL = self.repQL
        elif self.version == 2: repQL = int(self.cost_fix / max_steps) 
        x_true = x[1:]
        self.evaluate_called += 1
        np.random.seed(self.evaluate_called)
        random_seed = np.random.randint(900)
        y_mean, y_std, _, raw_data = obj(repQL, max_steps, self.episodeCap, 
                                         self.num_policy_checks, self.checks_per_policy, exp_path, 
                                         env='pd', flag_num=self.flag_num, 
                                         random_seed=random_seed, x=x_true, 
                                         noise_base=self.noise_base, noise_rand=self.noise_rand, 
                                         maze_num=self.maze_num, maze_name=self.maze_name)
        y_mean = y_mean[-1]
        y_std = y_std[-1]
        current_data = np.hstack((x, [y_mean], [y_std]))
        self.all_data.append(current_data)
        self.all_data_raw.append(raw_data)
        val = y_mean - self._meanval
        return self._mult * val # negate val because we want to maximize rewards

    def noise_and_cost_func(self, x):
        if self.version == 1:
            max_steps = int(x[0] * self.S)
            cost = self.repQL * max_steps
        elif self.version == 2: 
            cost = self.cost_fix
        for ind, s in enumerate(self.s_array):
            if np.absolute(x[0]-s)<1e-5:
                noise = self.noise[ind]
        return (noise, cost)

    def save_data(self, result_path):
        with open(result_path+'/all_data.txt', "w") as file: file.write(str(self.all_data))
        with open(result_path+'/all_data.pickle', "wb") as file: dump(self.all_data, file)
        with open(result_path+'/all_data_raw.pickle', "wb") as file: dump(self.all_data_raw, file)

    def getSearchDomain(self):
        return self._search_domain
    
    def getFuncName(self):
        return 'Puddle'

    def getDim(self):
        return self._dim

    def getNums(self):
        return int((self.s_max-self.s_min)/self.skip)+1

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])

    def get_meanval(self):
        return self._meanval



class MisoPuddle(object):
    
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, version, obj_func_idx, bucket, prob_name):
        self.replication_no = replication_no
        self.version = version
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func = [misoPuddle_(prob_name, version, mult=-1.0)]
        self._obj_func_idx = obj_func_idx

    @property
    def obj_func_min(self):
        return self._obj_func[self._obj_func_idx]

    @property
    def num_iterations(self):
        return 60
        # increase for more steps of the BO algorithm

    @property
    def hist_data(self):
        return self._hist_data

    @hist_data.setter
    def set_hist_data(self, data):
        self._hist_data = data


class MisoPuddleMkg(MisoPuddle):

    def __init__(self, replication_no, version, obj_func_idx, bucket, prob_name):
        super(MisoPuddleMkg, self).__init__(replication_no, "mkg", version, obj_func_idx, bucket, prob_name)
        self._hist_data = None


class_collection = {
    "miso_pd": MisoPuddleMkg,
}
