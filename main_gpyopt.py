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
import numpy as np
from collections import defaultdict
import scipy.io
from GPyOpt.methods import BayesianOptimization
from problems.define_basic_functions import obj_parallel, define_test_pts

np.set_printoptions(threshold=sys.maxint)


def obj_val(x):
    global random_seeds, obj_count, list_sampled_x, list_y, list_cost, list_opt_x
    obj_count += 1
    random_seed = random_seeds[obj_count]
    x = x[0, :]
    print(x)
    y_mean, _, _, _ = obj_parallel(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path,
                                   env=env, flag_num=flag_num, random_seed=random_seed, x=x,
                                   noise_base=noise_base, noise_rand=noise_rand, maze_num=maze_num, maze_name=maze_name)
    y_mean = y_mean[-1]

    list_sampled_x.append(x)
    list_y.append(y_mean)
    list_cost.append(max_steps * repQL)
    return y_mean


# ============================================ #
#                     main                     #
# ============================================ #
if __name__ == '__main__':

    argv = sys.argv[1:]
    '''
    python main_gpyopt.py mcf2 EI 0 0
        argv[0] prob_name
            gw10One1, gw10Two1, gw10Two2, gw10Three1; gw20Three1
            ky10One
            it10
            pd10
            mcf1, mcf2
        argv[1] algorithm
            EI, LCB
        argv[2] constraint (for ei and lcb)
            0, 1
        argv[3] sample_num
            0,1,2,3,...
    '''
    prob_name = argv[0]
    env = prob_name[:2]
    acq = argv[1]
    constraint = int(argv[2])
    sample = int(argv[3])

    result_path = './REP_rlt_' + prob_name + '/Results_gpyopt_' + prob_name + acq + '/'
    if not os.path.exists(result_path): os.makedirs(result_path)

    _test_x, _row_num, _col_num, maze_num, maze_name, noise_base, noise_rand, \
        flag_num, maze_size, repQL, s, S, skip, episodeCap = define_test_pts(prob_name)
    num_policy_checks = 1
    checks_per_policy = 1
    s_num = int((S - s) / skip)
    max_steps = S

    if prob_name in ['gw10One1', 'gw10Two1', 'gw10Two2', 'gw20Three1', 'ky10One', 'it10']:
        max_iter_ = 70
        wid_min = 0.1
        wid_max = maze_size * 2
        if flag_num == 2:
            domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1}
                      # {'name':'var_5', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1},
                      # {'name':'var_6', 'type':'continuous', 'domain':(wid_min,wid_max), 'dimensionality':1}
                      ]
        if flag_num == 3:
            domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_2', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_3', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_4', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_5', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1},
                      {'name': 'var_6', 'type': 'continuous', 'domain': (0, maze_size - 1), 'dimensionality': 1}
                      ]
        if constraint:
            constraints = [{'name': 'const_1', 'constraint': 'x[:,0] - x[:,2]'}]
        else:
            constraints = None
    elif prob_name in ['mcf1', 'mcf2', 'mcf3']:
        max_iter_ = 100
        position_min = -1.2
        position_max = 0.6
        constraints = None
        if prob_name == 'mcf1':
            domain = [
                {'name': 'var_1', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1}]
        elif prob_name == 'mcf2':
            domain = [
                {'name': 'var_1', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1},
                {'name': 'var_2', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1}]
            if constraint:
                constraints = [{'name': 'const_1', 'constraint': 'x[:,0] - x[:,1]'}]
            else:
                constraints = None
        elif prob_name == 'mcf3':
            domain = [
                {'name': 'var_1', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1},
                {'name': 'var_2', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1},
                {'name': 'var_3', 'type': 'continuous', 'domain': (position_min, position_max), 'dimensionality': 1}]

    if env in ['gw', 'ky', 'mc']:
        is_max, mult = False, -1  # minimize the steps
    elif env in ['it', 'pd']:
        is_max, mult = True, 1  # maximize the reward
    if env in ['mc']: checks_per_policy = 10

    initial_design_numdata = 10
    if prob_name == 'mcf3': initial_design_numdata = 12
    exp_path = result_path + 'sample' + str(sample) + '/'
    txt_path = result_path + 'sample' + str(sample)
    np.random.seed(sample)
    random_seeds = np.random.randint(900, size=(initial_design_numdata + max_iter_) * 2)
    list_sampled_x, list_y, list_cost = [], [], []
    list_opt_x = []
    X_init, Y_init = None, None
    max_iter = max_iter_
    obj_count = -1
    if os.path.isfile(txt_path + '_result.pickle'):
        with open(txt_path + '_result.pickle', 'rb') as file:
            f_dict = pickle.load(file)
        array_sampled_x = f_dict.get('sampled_x')
        array_y = f_dict.get('observed_y')
        array_cost = f_dict.get('cost')
        X_init = array_sampled_x
        Y_init = np.reshape(np.array(array_y), (-1, 1))
        for ind, cost in enumerate(array_cost):
            list_sampled_x.append(array_sampled_x[ind, :])
            list_y.append(array_y[ind])
            list_cost.append(array_cost[ind])
        max_iter = max_iter_ - len(list_y)
        obj_count = len(list_y) - 1

    myBO = BayesianOptimization(f=obj_val, domain=domain,
                                constraints=constraints,
                                X=X_init, Y=Y_init,
                                initial_design_numdata=initial_design_numdata,
                                initial_design_type='latin',
                                acquisition_type=acq, maximize=is_max)
    myBO.run_optimization(max_iter=max_iter)
    myBO.save_report(report_file=txt_path + '_report.txt')
    myBO.save_evaluations(evaluations_file=txt_path + '_evaluations.txt')
    myBO.save_models(txt_path + '_models.txt')
    result = {'opt_x': np.array(list_opt_x),
              'sampled_x': np.array(list_sampled_x),
              'observed_y': np.array(list_y),
              'cost': np.array(list_cost),
              'random_seeds': random_seeds}
    with open(txt_path + '_result.txt', "w") as file:
        file.write(str(result))
    with open(txt_path + '_result.pickle', "wb") as file:
        pickle.dump(result, file)