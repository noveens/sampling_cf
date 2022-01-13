import copy

TEST_SPLIT   = 0.2 # Of total 100%
VAL_SPLIT    = 0.15 # Of (100 - TEST_SPLIT)%
NUM_SAMPLES  = 10 # Number of samples per degree features e.g. user/item degree, hop-plot distr.
NUM_PURE_FEAUTRES = 7 * NUM_SAMPLES # Graph-based features
NUM_FEAUTRES = NUM_PURE_FEAUTRES + 5 # The remaining 5 are generic features: #users, #items, etc.

# NOTE: No need to change, these are just maps which create a unique id for each metric/task
task_map   = { t : at for at, t in enumerate([ 'explicit', 'implicit', 'sequential' ]) }
metric_map = { m : at for at, m in enumerate([ 'MSE', 'AUC', 'HR@100', 'NDCG@10' ]) }

# NOTE: This is a list of features (in order) for the handcrafted data features
first_map = [ '# Users', '# Items', '# Train', '# Val', '# Test' ]
second_map = [ 'User degree', 'Item degree', 'Node degree', 'Connected comp', 'Hop plot', 'Eigen Vals', 'Clustering coeff' ]

datasets = [ 
	'magazine', 
	# 'ml-100k',
	# 'luxury', 
	# 'beeradvocate',
	# 'goodreads_comics',
	# 'video_games'
]
methods_to_compare = [ 
	'bias_only',
	'MF_dot',
	'NeuMF',
	'pop_rec',
	'MVAE',
	'SVAE',
	'SASRec',
]
sampling_kinds = [ 
	'user_rns', 
	'interaction_rns', 
	'freq_user_rns', 
	'temporal',
    'tail_user_remove',
    'pagerank',
    'random_walk',
    'forest_fire',
]
svp_methods = [ 
	'bias_only',
	'MF_dot'
] 
sampling_svp = [ 
	'forgetting_events', 
    'forgetting_events_propensity',
    'forgetting_events_user', 
    'forgetting_events_user_propensity',
]
scenarios = [
    [ 'explicit',   [ 'MSE'    					 ] ],
    [ 'implicit',   [ 'AUC', 'HR@100', 'NDCG@10' ] ],
    [ 'sequential', [ 'AUC', 'HR@100', 'NDCG@10' ] ],
]
percent_rns_options = [ 20, 40, 60, 80, 90, 99 ]

# NOTE: Below code creates `subset_order` creates a unique embedding ID for each subset
# 		To make fetching from a common dataset embedding matrix
# 		It has `task` x `percent rns options` x `sampler` number of IDs
all_samplers = copy.deepcopy(sampling_kinds) 
for svp_method in svp_methods: 
	all_samplers += [ "svp_{}_{}".format(svp_method, sampling) for sampling in sampling_svp ]

subset_order, total_embeddings = {}, 0
for task, metrics in scenarios:
	subset_order[task] = {}
	
	subset_order[task]['complete_data'] = total_embeddings
	total_embeddings += 1

	for sampling_percent in percent_rns_options:
		subset_order[task][sampling_percent] = {}
		for sampler in all_samplers:
			subset_order[task][sampling_percent][sampler] = total_embeddings
			total_embeddings += 1

def get_embedding_id(task, sampler, sampling_percent):
	if sampler == 'complete_data': return subset_order[task][sampler] 
	return subset_order[task][sampling_percent][sampler]
