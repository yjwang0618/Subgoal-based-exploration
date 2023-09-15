from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from .Experiment import Experiment, Experiment_MountainCar, Experiment_ql, Experiment_it, Experiment_it_ql
from .MDPSolverExperiment import MDPSolverExperiment
# for backward compatibility with existing experiment scripts
OnlineExperiment = Experiment
