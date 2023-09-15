#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import json
import numpy as np

from .problems.rlpy.Domains.MountainCar import MountainCar
from .problems.rlpy.Domains.MountainCar_flag import MountainCar_flag
from .problems.rlpy.Agents import Q_Learning
from .problems.rlpy.Representations import RBF, TileCoding
from .problems.rlpy.Policies import eGreedy
from .problems.rlpy.Experiments import Experiment


def make_experiment(x, max_steps=10000, num_policy_checks=10, 
                    exp_id=1, path="./Results/Temp", seed=1):
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = max_steps
    opt["num_policy_checks"] = num_policy_checks
    gamm = 1
    FlagPos = np.array([x[0], x[1]])
    FlagWid = np.array([x[2], x[3]])
    FlagHeight = np.array([1, 2])
    #domain = MountainCar(noise=0.1)
    domain = MountainCar_flag(FlagPos=FlagPos, FlagWid=FlagWid, FlagHeight=FlagHeight, noise=0.1)
    performance_domain =domain
    domain.discount_factor = gamm
    opt["domain"] = domain
    opt["performance_domain"] = performance_domain
    lambda_ = 0
    initial_learn_rate = 0.5
    boyan_N0 = 100
    representation = RBF(domain, num_rbfs=100)
    policy = eGreedy(representation, epsilon=0.1, seed=seed)
    opt["agent"] = Q_Learning(
        policy, representation, discount_factor=domain.discount_factor,
        lambda_=lambda_, initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan", boyan_N0=boyan_N0)
    experiment = Experiment(**opt)
    experiment.run(visualize_steps=True,
                   visualize_learning=True,
                   visualize_performance=True)
    
    return experiment
make_experiment(np.array([-1.2, 0.6, 3, 5]))