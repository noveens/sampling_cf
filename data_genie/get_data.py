import os
import random
from tqdm import tqdm
from collections import defaultdict

from data_genie.data_genie_config import *
from data_genie.data_genie_utils import TRAINING_DATA_PATH, CACHED_KENDALL_TAU_PATH, load_obj, save_obj
from data_genie.data_genie_utils import count_performance_retained, get_best_results

from utils import INF

def get_data_pointwise(dataset):
	PATH = TRAINING_DATA_PATH(dataset, "pointwise")
	if not os.path.exists(PATH + ".pkl"): prep_data(dataset)
	return load_obj(PATH)

def get_data_pairwise(dataset):
	PATH = TRAINING_DATA_PATH(dataset, "pairwise")
	if not os.path.exists(PATH + ".pkl"): prep_data(dataset) 
	return load_obj(PATH)

def prep_data(dataset):
	# Get model runs
	results = get_results(dataset)

	# Build train, val, and test data	
	val_data = [ [], [], [], [], [] ]
	test_data = copy.deepcopy(val_data)
	train_data_pointwise = copy.deepcopy(val_data)
	train_data_pairwise = copy.deepcopy(val_data)

	for task, metrics in scenarios:
		all_options = []
		for m in metrics:
			for sampling_percent in percent_rns_options:
				all_options.append([ m, sampling_percent ])
		random.shuffle(all_options)

		val_indices  = [ all_options[0] ]
		if len(metrics) == 1: test_indices, train_indices = [ all_options[1] ], all_options[2:]
		else: test_indices, train_indices = all_options[1:4], all_options[4:]

		# Validation/testing data
		for container, indices in [ (val_data, val_indices), (test_data, test_indices) ]:
			for m, sampling_percent in indices:
				for sampling in all_samplers:
					container[0].append(get_embedding_id(task, 'complete_data', 0))
					container[1].append(get_embedding_id(task, sampling, sampling_percent))
					container[2].append(task_map[task])
					container[3].append(metric_map[m])
					container[4].append(
						count_performance_retained(results[task][m][sampling_percent][sampling], m, scaled = False)
					)

		# Training data
		for m, sampling_percent in train_indices:
			y = [ count_performance_retained(
				results[task][m][sampling_percent][sampling], m, scaled = False
			) for sampling in all_samplers ]

			# Pointwise
			for at, sampling in enumerate(all_samplers):
				if y[at] in [ INF, -INF ]: continue

				train_data_pointwise[0].append(get_embedding_id(task, 'complete_data', 0))
				train_data_pointwise[1].append(get_embedding_id(task, sampling, sampling_percent))
				train_data_pointwise[2].append(task_map[task])
				train_data_pointwise[3].append(metric_map[m])
				train_data_pointwise[4].append(y[at])

			# Pairwise
			for i in range(len(all_samplers)):
				for j in range(i+1, len(all_samplers)):
					if y[i] in [ INF, -INF ]: continue
					if y[j] in [ INF, -INF ]: continue
					if y[i] == y[j]: continue

					if y[i] > y[j]: better, lower = i, j
					else: better, lower = j, i

					train_data_pairwise[0].append(get_embedding_id(task, 'complete_data', 0))
					train_data_pairwise[1].append(get_embedding_id(task, all_samplers[better], sampling_percent))
					train_data_pairwise[2].append(get_embedding_id(task, all_samplers[lower], sampling_percent))
					train_data_pairwise[3].append(task_map[task])
					train_data_pairwise[4].append(metric_map[m])

	save_obj([ train_data_pointwise, val_data, test_data ], TRAINING_DATA_PATH(dataset, "pointwise"))
	save_obj([ train_data_pairwise, val_data, test_data ], TRAINING_DATA_PATH(dataset, "pairwise"))

def get_results(dataset):
	PATH = CACHED_KENDALL_TAU_PATH(dataset)
	if os.path.exists(PATH + ".pkl"): return load_obj(PATH)

	loop = tqdm(
		total = len(scenarios) * ((len(svp_methods) * len(sampling_svp)) + len(sampling_kinds)) * \
		len(methods_to_compare) * len(percent_rns_options)
	)

	y = {}
	for task, metrics_to_return in scenarios:
		
		# Structure of `y`
		y[task] = {}
		for m in metrics_to_return: 
			y[task][m] = {}
			for percent_rns in percent_rns_options: 
				y[task][m][percent_rns] = defaultdict(list)
		
		# Random/graph-based sampling
		for sampling_kind in sampling_kinds:
			for method in methods_to_compare:
				
				complete_data_metrics = get_best_results(
					dataset, 0, 'complete_data', method, task, metrics_to_return
				)

				for percent_rns in percent_rns_options:

					loop.update(1)
					metrics = get_best_results(
						dataset, percent_rns, sampling_kind, method, task, metrics_to_return
					)
					if metrics is None: continue

					for at, m in enumerate(metrics_to_return):
						y[task][m][percent_rns][sampling_kind].append([
							metrics[at], complete_data_metrics[at]
						])
		
		# SVP sampling
		for svp_method in svp_methods:
			for sampling_kind in sampling_svp:
				for method in methods_to_compare:

					complete_data_metrics = get_best_results(
						dataset, 0, 'complete_data', method, task, metrics_to_return
					)

					for percent_rns in percent_rns_options:

						loop.update(1)
						metrics = get_best_results(
							dataset, percent_rns, "svp_{}".format(svp_method), method, task, metrics_to_return,
							sampling_svp = sampling_kind
						)
						if metrics is None: continue

						for at, m in enumerate(metrics_to_return):
							y[task][m][percent_rns]["svp_{}_{}".format(svp_method, sampling_kind)].append([
								metrics[at], complete_data_metrics[at]
							])

	save_obj(y, PATH)
	return y

