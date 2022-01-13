import os
from utils import get_common_path

# Sampling experiments' constants
BASE_SAMPLING_PATH = "./experiments/sampling_runs/"

# Data-genie experiments' constants
BASE_DATA_GENIE_PATH = "./experiments/data_genie/"

def get_svp_log_file_path(hyper_params):
    return BASE_SAMPLING_PATH + "/results/logs/SVP/{}.txt".format(get_common_path(hyper_params))

def get_svp_model_file_path(hyper_params):
    return BASE_SAMPLING_PATH + "/results/models/SVP/{}.pt".format(get_common_path(hyper_params))

def get_log_base_path():
	return BASE_SAMPLING_PATH + "/results/logs/trained/"

def get_log_file_path(hyper_params):
	return get_log_base_path() + get_common_path(hyper_params) + ".txt"

def get_model_file_path(hyper_params):
	return BASE_SAMPLING_PATH + "/results/models/trained/" + get_common_path(hyper_params) + ".pt"

def get_data_path(hyper_params):
    dataset = hyper_params
    if type(dataset) != str: dataset = hyper_params['dataset']
    return "./datasets/{}/".format(dataset)

def get_index_path(hyper_params):
    train_test_split = {
        'explicit':    '20_percent_hist',
        'implicit':    '20_percent_hist',
        'sequential':  'leave_2',
    }[hyper_params['task']]

    ret = get_data_path(hyper_params['dataset']) + "/{}/".format(train_test_split)

    if hyper_params['sampling'][:3] == 'svp':
        ret += "{}_{}/{}_perc_{}/".format(
            hyper_params['sampling'], hyper_params['task'],
            hyper_params['sampling_percent'], hyper_params['sampling_svp']
        )
    elif hyper_params['sampling'] == 'complete_data':
        ret += "complete_data/"
    else:
        ret += "{}_perc_{}/".format(
            hyper_params['sampling_percent'], hyper_params['sampling'],
        )

    return ret
