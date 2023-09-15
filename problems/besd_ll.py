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

from .deep.run_ll import EPISODE_CAP, MAX_STEPS_MIN, MAX_STEPS_MAX, SKIP
from .deep.run_ll import main


STATE_RANGE = [[-1, 1], [-0.4, 2], [-2.15, 1.66], [-1.73, 0.55], [-3, 3.2], [-5.63, 5.73]]


class besdLunarLander_(object):
    num_policy_checks = 1
    checks_per_policy = 1
    
    def __init__(self, prob_name, mult=1.0):
        self.s_min = MAX_STEPS_MIN / MAX_STEPS_MAX
        self.s_max = 1
        self.skip = SKIP / MAX_STEPS_MAX
        self.s_array = np.linspace(self.s_min, self.s_max, num=self.getNums())
        print(self.s_array)

        self._mult = mult
        self.evaluate_called = 0

        self._dim = 5
        self._search_domain = np.zeros((self._dim, 2))
        self._search_domain[0,:] = [self.s_min, self.s_max]

        self._search_domain[1,:] = STATE_RANGE[0]
        self._search_domain[2,:] = STATE_RANGE[1]

        self._search_domain[3,:] = STATE_RANGE[0]
        self._search_domain[4,:] = STATE_RANGE[1]
            
        self._meanval = 70
        self.noise = [0.01, 0.01]
            
    def evaluate(self, x, random_seed, exp_path='./experiment_data/', for_initial=False):
        max_steps = int(x[0] * MAX_STEPS_MAX)
        x_true = x[1:]
        self.evaluate_called += 1
        np.random.seed(self.evaluate_called)
        random_seed = np.random.randint(900)
        y_mean_s, y_mean_S = main(x=x_true, max_steps=max_steps, exp_path=exp_path, aseed=random_seed)
        y_mean_s += 200
        y_mean_S += 200
        if for_initial:
            val_s = y_mean_s - self._meanval
            val_S = y_mean_S - self._meanval
            return self._mult * val_s, self._mult * val_S
        y_mean = y_mean_s if x[0] < 1 else y_mean_S
        val = y_mean - self._meanval
        return self._mult * val

    def noise_and_cost_func(self, x):
        max_steps = int(x[0] * MAX_STEPS_MAX)
        cost = max_steps
        for idx, max_steps_min in enumerate(self.s_array):
            if np.absolute(x[0] - max_steps_min) < 1e-5:
                noise = self.noise[idx]
        return (noise, cost)

    def getSearchDomain(self): return self._search_domain
    
    def getFuncName(self): return 'LunarLander'

    def getDim(self): return self._dim

    def getNums(self): return int((self.s_max - self.s_min) / self.skip) + 1

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])

    def get_meanval(self): return self._meanval



class BesdLunarLander(object):
    
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, obj_func_idx, bucket, prob_name):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self.prob_name = prob_name
        self._obj_func = [besdLunarLander_(prob_name, mult=1.0)]
        self._obj_func_idx = obj_func_idx

    @property
    def obj_func_min(self): return self._obj_func[self._obj_func_idx]

    @property
    def num_iterations(self): return 100

    @property
    def hist_data(self): return self._hist_data

    @hist_data.setter
    def set_hist_data(self, data): self._hist_data = data



class BesdLunarLanderMkg(BesdLunarLander):

    def __init__(self, replication_no, obj_func_idx, bucket, prob_name):
        super(BesdLunarLanderMkg, self).__init__(replication_no, "besd", obj_func_idx, bucket, prob_name)



class_collection = {
    "besd_ll": BesdLunarLanderMkg,
}
