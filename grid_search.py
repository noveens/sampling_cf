import os
import gc
import copy
import time
import json
import datetime
import traceback
from tqdm import tqdm
import multiprocessing

from main import main
from utils import get_common_path
from data_path_constants import get_index_path, get_log_file_path

# NOTE: Specify all possible combinations of hyper-parameters you want to search on.
# NOTE: A list of values for a hyper-parameter means that you want to train all possible combinations of them
hyper_params = {
    'dataset': [ 
		'magazine', 
        # 'ml-100k',
    ],
    'task': [ 
        'explicit', 
        'implicit', 
        'sequential',
    ],
    'sampling': [
        'complete_data',
        'user_rns',
        'interaction_rns',
        'freq_user_rns',
        'temporal',
        'tail_user_remove',
        'svp_bias_only',
        'svp_MF_dot',
        'pagerank',
        'random_walk',
        'forest_fire',
    ], 
    'sampling_svp': [
        'forgetting_events', 
        'forgetting_events_propensity',
        'forgetting_events_user', 
        'forgetting_events_user_propensity',
    ],
    'sampling_percent': [ 20, 40, 60, 80, 90, 99 ],

    'latent_size': [ 8, 16, 32, 50 ], # Latent size in all algos
    'lr': 0.006, # LR for ADAM # 0.01 for ml-100k ; 0.003 for ml-25m
    'dropout': [ 0.3 ], # 0.3/4 works good for 0-core, 0.6/8 for 5-core

    'model_type': [ 
        'pop_rec',
        'bias_only',
        'MF_dot',
        'NeuMF',
        'MVAE',
        'SVAE',
        'SASRec',
    ],

    'num_heads': 1, ## SASRec
    'num_blocks': 2, ## SASRec
    'num_next': [ 2 ], ## SVAE

    'num_train_negs': 1,
    'num_test_negs': 100,

    #### Below hyper-params will be re-set from `data_hyperparams`
    #### But adding just because we need them to compute common path
    #### while counting the number of unique tasks
    'weight_decay': float(1e-6),
}

# NOTE: Entering multiple of the same GPU-ID will launch multiple runs on the SAME GPU
# NOTE: Entering -1 or an invalid GPU-ID will run a corresponding run on the CPU
gpu_ids = [ 0, 0,     1, 1 ]

################## CONFIGURATION INPUT ENDS ###################

# STEP-1: Count processes 
def get_all_jobs(task):
    ret, single_proc = [], True

    for key in task:
        if type(task[key]) != list: continue

        single_proc = False
        for val in task[key]:
            send = copy.deepcopy(task) ; send[key] = val
            ret += get_all_jobs(send)

        break # All sub-jobs are already counted

    return ret if not single_proc else [ task ]

duplicate_tasks = get_all_jobs(hyper_params)
print("Total processes before unique:", len(duplicate_tasks))

def enough_users_items(task):
    data_stats_file = get_index_path(task) + "data_stats.json"
    with open(data_stats_file) as f: stats = json.load(f)
    return stats['num_users'] >= 50 and stats['num_items'] >= 50 and stats['num_train_interactions'] >= 100

temp = set()
covered_tasks, all_tasks = set(), []
for task in tqdm(duplicate_tasks):
    log_file = get_common_path(task)

    if log_file is None: continue
    if log_file in covered_tasks: continue
    if not enough_users_items(task): continue

    temp.add(log_file)

    ##### TEMP: Checking if job has already been done
    if os.path.exists(get_log_file_path(task)):
        f = open(get_log_file_path(task), 'r')
        lines = f.readlines() ; f.close()
        exists = sum(map(lambda x: int('TEST' in x.strip()), lines))
        if exists != 0: continue

    all_tasks.append(task)
    covered_tasks.add(log_file)
print("Total processes after unique:", len(temp))
print("Total processes after removing already finished jobs:", len(all_tasks))
print(set(list(map(lambda x: x['model_type'], all_tasks))))
# exit()

# STEP-2: Assign individual GPU processes
gpu_jobs = [ [] for _ in range(len(gpu_ids)) ]
for i, task in enumerate(all_tasks): gpu_jobs[i % len(gpu_ids)].append(task)

# Step-3: Spawn jobs parallely
def file_write(log_file, s):
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def run_tasks(hyper_params, tasks, gpu_id):
    start_time = time.time()
    for num, task in enumerate(tasks):
        percent_done = max(0.00001, float(num) / float(len(tasks)))
        time_elapsed = time.time() - start_time
        file_write(
            "experiments/grid_search_log.txt", 
            str(task) + "\nGPU_ID = " + str(gpu_id) + "; dataset = " + task['dataset'] + "; [{} / {}] ".format(num, len(tasks)) +
            str(round(100.0 * percent_done, 2)) + "% done; " +
            "ETA = " + str(datetime.timedelta(seconds=int((time_elapsed / percent_done) - time_elapsed)))
        )
        try: main(task, gpu_id = gpu_id)
        except Exception as e:
            file_write(
                "experiments/grid_search_log.txt", "GPU_ID = " + str(gpu_id) + \
                "; ERROR [" + str(num) + "/" + str(len(tasks)) + "]\nJOB: " + str(task) + "\n" + str(traceback.format_exc())
            )
        gc.collect()

for gpu in range(len(gpu_ids)):
    p = multiprocessing.Process(target=run_tasks, args=(hyper_params, gpu_jobs[gpu], gpu_ids[gpu], ))
    p.start()