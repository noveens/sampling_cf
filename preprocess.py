from initial_data_prep_code import movielens, amazon, goodreads, beeradvocate
from data_path_constants import get_data_path
from svp_handler import SVPHandler

percent_sample = [ 20, 40, 60, 80, 90, 99 ]

# Which datasets to prep?
for dataset in [
	'magazine',
	'ml-100k',

	## Did not download & preprocess the following in 
	## the included code, but feel free to download and uncomment
	# 'luxury',
	# 'video_games',
	# 'beeradvocate',
	# 'goodreads_comics',
]:

	print("\n\n\n!!!!!!!! STARTED PROCESSING {} !!!!!!!!\n\n\n".format(dataset))

	if dataset in [ 'ml-100k' ]: total_data = movielens.prep(dataset)
	elif dataset in [ 'luxury', 'magazine', 'video_games' ]: total_data = amazon.prep(dataset)
	elif dataset in [ 'goodreads_comics' ]: total_data = goodreads.prep(dataset)
	elif dataset in [ 'beeradvocate' ]: total_data = beeradvocate.prep(dataset)

	# Store original data
	total_data.save_data(get_data_path(dataset))

	# Sampling
	for train_test_split in [ '20_percent_hist', 'leave_2' ]:

		total_data.complete_data_stats = None # Since task changed
		path_uptil_now = get_data_path(dataset) + "/" + train_test_split + "/"

		# Make full-data (No sampling)
		total_data.train_test_split(train_test_split)
		print("\n{} split, Overall:".format(train_test_split))
		total_data.save_index(path_uptil_now + "/complete_data/")

		# Frequency sample from user hist (Stratified)
		print("\n{} split, user history random sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.frequency_sample(percent, 0)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_freq_user_rns")

		# Sample users randomly
		print("\n{} split, user random sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.user_random_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_user_rns")

		# Sample interactions randomly
		print("\n{} split, interaction random sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.interaction_random_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_interaction_rns")

		# Temporal sampling
		print("\n{} split, user history temporal sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.temporal_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_temporal")

		# Remove tail users sampling
		print("\n{} split, tail user sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.tail_user_remove(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_tail_user_remove")

		# Pagerank based sampling
		print("\n{} split, pagerank sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.pagerank_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_pagerank")

		# RW based sampling
		print("\n{} split, random walk sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.random_walk_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_random_walk")

		# Forest-fire based sampling
		print("\n{} split, forest fire sampling".format(train_test_split))
		for percent in percent_sample:
			total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
			total_data.forest_fire_sample(percent)
			total_data.save_index(path_uptil_now + str(percent) + "_perc_forest_fire")

		# Sample interactions according to SVP
		hyper_params = {}
		hyper_params['dataset'] = dataset
		hyper_params['sampling'] = 'complete_data' # While training the proxy model

		for proxy_model in [ 'bias_only', 'MF_dot' ]:
			scenarios = [ 'sequential' ] if train_test_split == 'leave_2' else [ 'implicit', 'explicit' ]

			for loss_type in scenarios:
				print() ; svp_handler = SVPHandler(proxy_model, loss_type, hyper_params)

				for sampling in [ 
					'forgetting_events', 
					'forgetting_events_propensity',
					'forgetting_events_user', 
					'forgetting_events_user_propensity',
				]:
					print("\n{} split, SVP: {}_{}, {} loss".format(train_test_split, proxy_model, sampling, loss_type))
					for percent in percent_sample:
						total_data.load_index(path_uptil_now + "/complete_data/") # Re-load index map
						total_data.svp_sample(percent, svp_handler, sampling)
						total_data.save_index(path_uptil_now + "svp_{}_{}/{}_perc_{}".format(proxy_model, loss_type, percent, sampling))
