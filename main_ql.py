from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import os
import sys
import json
import pickle
from pickle import dump
import numpy as np
from collections import defaultdict
import scipy.io
from problems.define_basic_functions import define_test_pts
from problems.define_basic_functions_ql import obj_ql

np.set_printoptions(threshold=sys.maxint)


if __name__ == '__main__':

	argv = sys.argv[1:]
	'''
	python main_ql.py it10 2
		gw10Two1; gw20Three1
		ky10One
		it10
		pd10
		mcf2
	'''
	prob_name = argv[0]
	env = prob_name[:2]
	transfer_weight = int(argv[1])
	_test_x, _row_num, _col_num, maze_num, maze_name, noise_base, noise_rand, \
		flag_num, maze_size, repQL, s, S, skip, episodeCap = define_test_pts(prob_name)
	repQL = 1
	max_steps = S
	num_policy_checks = 1
	checks_per_policy = 1
	if env == 'mc': checks_per_policy = 10
	sample_num = 50
	iter_num = 120

	if prob_name[0:4]=='gw10': features_num = 100
	elif prob_name[0:4]=='gw20': features_num = 400
	elif prob_name[0:4]=='ky10': features_num = 100*2 # if got the key
	elif prob_name[0:4]=='it10': features_num = 100*2 # if got the item
	elif prob_name[0:4]=='pd10': features_num = 70
	elif env=='mc': features_num = 1000

	if env in ['gw', 'ky', 'it', 'pd']: actions_num = 4
	elif env=='mc': actions_num = 3

	if not transfer_weight: 
		result_path = './REP_rlt_'+prob_name+'/Results_ql_'+prob_name+'/'
		weight_vec_init = None
	elif transfer_weight==1: # transfer Q-learning
		result_path = './REP_rlt_'+prob_name+'/Results_ql_'+prob_name+'_transfer/'
	elif transfer_weight==2: # every time learns a new one
		result_path = './REP_rlt_'+prob_name+'/Results_ql_'+prob_name+'_independent/'
	else: print('transfer_weight must be 0, 1, or 2')

	list_y_all = []
	list_cost_all = []
	list_weight_vec = []
	for sample in range(sample_num):
		exp_path = result_path+'exp/sample'+str(sample)+'/'
		txt_path = result_path+'sample'+str(sample)

		np.random.seed(sample)
		if transfer_weight==1:
			if env in ['gw']: weight_vec_init = np.random.uniform(low=0.0, high=0.01, size=features_num*actions_num)
			elif env in ['ky', 'pd']: weight_vec_init = np.random.uniform(low=0.0, high=1, size=features_num*actions_num)
			elif env in ['it']: weight_vec_init = np.random.uniform(low=0.0, high=1, size=features_num*actions_num)
			elif env=='mc': weight_vec_init = np.random.uniform(low=-1, high=0, size=(features_num+1)*actions_num)
			weight_vec_old = np.copy(weight_vec_init)
		elif transfer_weight==0:
			weight_vec_old = np.copy(weight_vec_init)

		list_s = []
		list_y = []
		list_cost = []
		list_weight = []
		
		for iteration in range(iter_num):
			random_seed = iteration+sample*iter_num
			if transfer_weight < 2:
				weight_vec_new, y_mean, y_std, exp_ids, curve = \
						obj_ql(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
								env=env, flag_num=0, random_seed=random_seed, x=None, 
								noise_base=noise_base, noise_rand=noise_rand, 
								maze_num=maze_num, maze_name=maze_name, weight_vec_old=weight_vec_old)
				weight_vec_old = weight_vec_new
			elif transfer_weight==2:
				np.random.seed(sample*iter_num+iteration)
				if env in ['gw']: weight_vec_init = np.random.uniform(low=0.0, high=0.001, size=features_num*actions_num)
				elif env in ['ky', 'pd']: weight_vec_init = np.random.uniform(low=0.0, high=0.01, size=features_num*actions_num)
				elif env in ['it']: weight_vec_init = np.random.uniform(low=0.0, high=0.01, size=features_num*actions_num)
				elif env=='mc': weight_vec_init = np.random.uniform(low=-1, high=0, size=(features_num+1)*actions_num)

				weight_vec_new, y_mean, y_std, exp_ids, curve = \
						obj_ql(repQL, max_steps, episodeCap, num_policy_checks, checks_per_policy, exp_path, 
								env=env, flag_num=0, random_seed=random_seed, x=None, 
								noise_base=noise_base, noise_rand=noise_rand, 
								maze_num=maze_num, maze_name=maze_name, weight_vec_old=weight_vec_init)

			print(y_mean, y_std, exp_ids)
			y_mean = y_mean[-1]
			list_s.append(max_steps)
			list_y.append(y_mean)
			list_cost.append(repQL*max_steps)
			list_weight.append(weight_vec_new)

		list_y_all.append(np.array(list_y))
		list_cost_all.append(np.array(list_cost))
		list_weight_vec.append(list_weight)
        
		result_all = {"list_y_all": list_y_all,
						"list_cost_all": list_cost_all,
						'list_weight_vec':list_weight_vec}
		with open(result_path+'all_result.txt', "w") as file: file.write(str(result_all))
		with open(result_path+'all_result.pickle', "wb") as file: pickle.dump(result_all, file)