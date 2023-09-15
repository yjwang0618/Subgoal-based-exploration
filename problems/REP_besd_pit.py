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

from .define_basic_functions_REP import define_test_pts, obj


class besdGridWorld_pit(object):
    num_policy_checks = 1
    checks_per_policy = 1
    all_data = []
    all_data_raw = []
    
    def __init__(self, prob_name, mult=-1.0):
        self.env = prob_name[:2]
        _test_x, _row_num, _col_num, self.maze_num, self.maze_name, \
            self.noise_base, self.noise_rand, self.flag_num, self.maze_size, self.repQL, \
            self.s, self.S, self.skip, self.episodeCap = define_test_pts(prob_name)
        self.s_min = self.s/self.S
        self.s_max = 1
        self.skip /= self.S
        self.s_array = np.linspace(self.s_min, self.s_max, num=self.getNums())
        print(self.s_array)

        self._mult = mult
        self.evaluate_called = 0

        self._dim = self.flag_num*2+1
        self._search_domain = np.zeros((self._dim, 2))
        self._search_domain[0,:] = [self.s_min, self.s_max]
        self._search_domain[1,:] = [0, self.maze_size-1]
        self._search_domain[2,:] = [0, self.maze_size-1]
        self._search_domain[3,:] = [0, self.maze_size-1]
        self._search_domain[4,:] = [0, self.maze_size-1]
            
        self._meanval, self.noise = 0, [0.1, 0.1, 0.1]
        self.y_min, self.y_max = None, None
            
    def evaluate(self, repQL, s_x, random_seed, exp_path='./initResults/'):
        max_steps = int(s_x[0] * self.S)
        x_true = s_x[1:]
        self.evaluate_called += 1
        np.random.seed(self.evaluate_called)
        random_seed = np.random.randint(900)
        y_mean, y_noise, _, raw_data = obj(repQL, max_steps, self.episodeCap, 
                                         self.num_policy_checks, self.checks_per_policy, exp_path, 
                                         env=self.env, flag_num=self.flag_num, 
                                         random_seed=random_seed, x=x_true, 
                                         noise_base=self.noise_base, noise_rand=self.noise_rand, 
                                         maze_num=self.maze_num, maze_name=self.maze_name)
        y_mean = y_mean[-1]
        y_noise = y_noise[-1]
        current_data = np.hstack((s_x, [y_mean], [y_noise]))
        self.all_data.append(current_data)
        self.all_data_raw.append(raw_data)
        
        val = y_mean - self._meanval
        return self._mult*val, y_noise

    def noise_and_cost_func(self, repQL, s_x):
        max_steps = int(s_x[0] * self.S)
        cost = repQL * max_steps
        for ind, s in enumerate(self.s_array):
            if np.absolute(s_x[0]-s)<1e-5:
                noise = self.noise[ind]
        return (noise, cost)

    def save_data(self, result_path):
        with open(result_path+'/all_data.txt', "w") as file: file.write(str(self.all_data))
        with open(result_path+'/all_data.pickle', "wb") as file: dump(self.all_data, file)
        with open(result_path+'/all_data_raw.pickle', "wb") as file: dump(self.all_data_raw, file)

    def getSearchDomain(self):
        return self._search_domain
    
    def getFuncName(self):
        return 'GridWorldPit'

    def getDim(self):
        return self._dim

    def getNums(self):
        return int((self.s_max-self.s_min)/self.skip)+1

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])

    def get_meanval(self):
        return self._meanval



class BesdGridWorldPit(object):
    
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, obj_func_idx, bucket, prob_name):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self.prob_name = prob_name
        self._obj_func = [besdGridWorld_pit(prob_name, mult=-1.0)]
        self._obj_func_idx = obj_func_idx

    @property
    def obj_func_min(self): return self._obj_func[self._obj_func_idx]

    @property
    def num_iterations(self): return 100

    @property
    def hist_data(self): return self._hist_data

    @hist_data.setter
    def set_hist_data(self, data): self._hist_data = data

    # comment the following when there is no multi information source
    # @abstractproperty
    # def num_is_in(self): None

    # @property
    # def truth_is(self): return 0

    # @property
    # def exploitation_is(self): return 1

    # @property
    # def list_sample_is(self): return range(4)



class BesdGridWorldPitMkg(BesdGridWorldPit):

    def __init__(self, replication_no, obj_func_idx, bucket, prob_name):
        super(BesdGridWorldPitMkg, self).__init__(replication_no, "besd", obj_func_idx, bucket, prob_name)

    # @property
    # def num_is_in(self): return 3   # This should be idx of the last IS



class_collection = {
    "besd_pt": BesdGridWorldPitMkg,
}
