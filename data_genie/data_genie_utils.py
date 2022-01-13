import os
import re
import pickle
import numpy as np
from scipy import stats

from utils import get_common_path, INF
from data_path_constants import BASE_DATA_GENIE_PATH, get_log_base_path

# NOTE: Below is the definition of the directory-structure of data-genie data folder
def append(path):
	# Append all relative paths to the base data-genie path
	final = BASE_DATA_GENIE_PATH + path
	# Create intermediate directories if they don't exist
	os.makedirs(os.path.dirname(final), exist_ok = True)
	return final

TRAINING_DATA_PATH = lambda dataset, pointwise_or_pairwise: append("/train_test_splits/{}/{}".format(
	pointwise_or_pairwise, 
	dataset
))
CACHED_KENDALL_TAU_PATH = lambda dataset: append("/cached/taus/{}".format(dataset))
EMBEDDINGS_ROOT = lambda dataset: append("/embeddings/{}/".format(dataset))
EMBEDDINGS_PATH_GCN = lambda dataset, hid_dim, layers: "{}/unsupervised_gcn_dim_{}_layers_{}".format(EMBEDDINGS_ROOT(dataset), hid_dim, layers)
EMBEDDINGS_PATH_HANDCRAFTED = lambda dataset: "{}/handcrafted".format(EMBEDDINGS_ROOT(dataset))

INFOGRAPH_MODEL_PATH = lambda hid_dim, layers: append("/models/unsupervised_gcn_dim_{}_layers_{}.pt".format(hid_dim, layers))
INFOGRAPH_CACHED_GRAPHS = lambda dataset: append("/cached/{}_graphs.bin".format(dataset))
INFOGRAPH_CACHED_DATA_STATS = lambda dataset: append("/cached/{}_data_stats".format(dataset))

TENSORBOARD_BASE = append("/tensorboard/")
LOGS_BASE = append("/logs/")
MODELS_BASE = append("/models/")

def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def load_numpy(path):
	with open(path + ".npy", 'rb') as f: return np.load(f)

def save_numpy(path, data):
	with open(path + ".npy", 'wb') as f: np.save(f, data)

def get_best_results(dataset, percent_sampling, sampling_kind, method, task, metrics_to_return, sampling_svp = None):
	BASE_PATH = get_log_base_path()
	all_logs = os.listdir(BASE_PATH)

	search_key = get_common_path({
		"dataset": dataset,
		"model_type": method,
		"sampling": sampling_kind,
		"sampling_percent": percent_sampling,
		"sampling_svp": sampling_svp,
		"task": task,
	}, star_match = True)
	if search_key is None: return

	relevant_logs = []
	for log in all_logs:
		if re.search(search_key, log): 
			f = open(BASE_PATH + log, "r") ; lines = f.readlines() ; f.close()
			relevant_logs.append(lines)

	smaller = [  INF, lambda x, y: x < y ]
	larger  = [ -INF, lambda x, y: x > y ]
	metric_to_validate, best, better = {
		'explicit': 	[ 'MSE' ] 	 + smaller,
		'implicit': 	[ 'AUC' ] 	 + larger,
		'sequential': 	[ 'HR@100' ] + larger,
	}[task]

	# Amongst all logs, search for best AUC/MSE/HR@100 on the validation set.
	# Return best log's test metrics
	ret, all_metrics = [], {}
	for log in relevant_logs:
		best, all_metrics = get_best_test_result(log, metric_to_validate, best, all_metrics, better)
	
	for metric in metrics_to_return + [ 'time' ]: 
		if metric in all_metrics: ret.append(float(all_metrics[metric]))
		else: ret.append(None)
	return ret

def get_best_test_result(lines, metric, best, all_metrics, better):
	for line in lines:
		line = line.strip()
		if line.endswith("(TEST)"):
			this_metrics = {}
			for m in line[:-7].split(" | "): # removing " (TEST)"
				if "=" not in m: continue
				key, val = m.split(" = ")
				this_metrics[key] = val
			if better(float(this_metrics[metric]), best): 
				best, all_metrics = float(this_metrics[metric]), this_metrics
	
	return best, all_metrics

def has_inf(arr):
	return sum(map(lambda x: x in [ INF, -INF, None ], arr)) > 0

def count_performance_retained(arr, metric, scaled = False):
	complete_data_logits = np.array(list(map(lambda x: x[1], arr)))
	subset_logits = np.array(list(map(lambda x: x[0], arr)))
	if has_inf(complete_data_logits) or has_inf(subset_logits): return -INF

	# AUC, HR, PSP, NDCG -- higher is better
	if metric not in [ 'MSE' ]: complete_data_logits, subset_logits = -complete_data_logits, -subset_logits

	if scaled: return 100.0 * stats.kendalltau(complete_data_logits, subset_logits)[0] # value, p-value
	return stats.kendalltau(complete_data_logits, subset_logits)[0] # value, p-value
