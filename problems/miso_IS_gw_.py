from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import numpy as np
import math

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

from .rlpy.Domains.GridWorld import GridWorld, GridWorld_Flag
from .rlpy.Agents.TDControlAgent import Q_Learning
from .rlpy.Representations.Tabular import Tabular
from .rlpy.Policies.eGreedy import eGreedy
from .rlpy.Experiments.Experiment import Experiment
    

class misoGridWorld_xs(object):
    s = 200
    S = 2000
    skip = 300
    episodeCap = 500
    mean_y = episodeCap - 50
    repQL = 100
    # -----------------------------------
    flag_num = 2
    maze_size = 10
    _num_s = 2
    num_policy_checks = 1
    cost_fix = S*2

    def __init__(self, version, mult=1.0):
        self._dim = self.flag_num*2
        self._search_domain = np.zeros((self._dim, 2))

        self._search_domain[0,:] = [0, 9]
        self._search_domain[1,:] = [0, 9]
        self._search_domain[2,:] = [0, 9]
        self._search_domain[3,:] = [0, 9]

        self.version = version
        self._mult = mult
        self._meanval = self.mean_y
        
    def noise_and_cost_func(self, IS, x):
        if self.version == 1:
            repQL = 10 # math.ceil(x[0]/2)
            if IS==0: max_steps = self.S
            else: max_steps = int(self.step_value_list[IS-1])
            cost = repQL * max_steps
        elif self.version == 2: cost = self.cost_fix
        return (10, cost)

    def evaluate(self, IS, x, exp_path='./initResults/'):
        if self.version == 1: 
            repQL = 10 # math.ceil(x[0]/2)
            if IS==0: max_steps = self.S
            else: max_steps = int(self.step_value_list[IS-1])
        elif self.version == 2: 
            repQL = int(self.cost_fix / max_steps) 
        val = self.obj(x, max_steps, repQL, exp_path) - self._meanval
        return self._mult * val
    
    def obj(self, x, max_steps, repQL, exp_path):
        curve = np.zeros((repQL, self.num_policy_checks))
        ''' random MDP '''
        noise = 0 + 0.01 * np.random.random()
        maze_order = 0
        # maze_order = np.random.randint(9)
        # mapname = os.path.join(GridWorld.default_map_dir, '10x10_TwoRooms1_'+str(maze_order)+'.txt')
        # maze_order = np.random.randint(15)
        mapname = os.path.join(GridWorld.default_map_dir, '10x10_TwoRooms2_'+str(maze_order)+'.txt')
        maze = np.loadtxt(mapname, dtype=np.uint8)
        if maze.ndim == 1:
            maze = maze[np.newaxis, :]
        for j in range(repQL):
            exp_id = np.random.randint(low=1, high=900)
            seed = 1
            curve[j,:] = self.make_experiment(x, maze, noise, max_steps, exp_id, exp_path, seed)
        y = np.mean(curve, 0)
        y = y[-1]
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return y

    def getFuncName(self):
        return 'GridWorld'

    def getSearchDomain(self):
        return self._search_domain

    def getDim(self):
        return self._dim

    def getNumIS(self):
        return self._num_IS

    def getList_IS_to_query(self):
        return np.arange(self._num_IS)

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])

    def get_meanval(self):
        return self._meanval
    
    def make_experiment(self, x, maze, noise, max_steps, exp_id, exp_path, seed):
        opt = {}
        opt["path"] = exp_path
        opt["exp_id"] = exp_id
        opt["max_steps"] = max_steps
        opt["num_policy_checks"] = self.num_policy_checks
        opt["checks_per_policy"] = 2
        gamm = 1
        FlagPos = np.array([[x[0], x[1]], [x[2], x[3]]])
        FlagWid = np.array([[5], [10]])
        FlagHeight = np.array([[0.5], [1]])
        domain = GridWorld_Flag(mapname, noise=noise, episodeCap=self.episodeCap, 
                                FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight)
        performance_domain =domain
        domain.discount_factor = gamm
        opt["domain"] = domain
        opt["performance_domain"] = performance_domain
        lambda_ = 0 # TD(lambda): control the eligibility traces
        initial_learn_rate = 0.11
        boyan_N0 = 100
        representation = Tabular(domain)
        policy = eGreedy(representation, epsilon=0.1, seed=seed)
        opt["agent"] = Q_Learning(
            policy, representation, discount_factor=domain.discount_factor,
            lambda_=lambda_, initial_learn_rate=initial_learn_rate,
            learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
        experiment = Experiment(**opt)
        experiment.run()
        experiment.save()
        steps_array = np.zeros((1,self.num_policy_checks+1))
        if exp_id < 10: path_name = exp_path+'00'
        elif exp_id < 100: path_name = exp_path+'0'
        else: path_name = exp_path
        f = open(path_name+str(exp_id)+'-results.txt', 'r').read()
        f = f.replace('defaultdict(', '')
        f = f.replace(')', '')
        f = f.replace('<type \'list\'>,', '')
        f_dict = eval(f)
        f_steps = f_dict.get('steps')
        k = -1
        for step in f_steps:
            k += 1
            steps_array[0,k] = step
        curve = np.delete(steps_array, 0)
        return curve


if __name__ == "__main__":
    misoGW = misoGridWorld_xs()
    python_search_domain = TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in misoGW._search_domain])
    pts = python_search_domain.generate_uniform_random_points_in_domain(1000)
    print ("misoGridWorld_xs")
    print ("IS0: {0}".format(np.mean([misoGW.evaluate(x) for x in pts])))
