import gc
import os
import dgl
import snap
import torch 
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

from data_genie.data_genie_config import *
from data_genie.data_genie_utils import save_numpy, load_numpy
from data_genie.data_genie_utils import EMBEDDINGS_PATH_GCN, EMBEDDINGS_PATH_HANDCRAFTED, INFOGRAPH_MODEL_PATH

from load_data import DataHolder
from data_path_constants import get_data_path, get_index_path

from data_genie.InfoGraph.infograph_model import InfoGraph
from data_genie.InfoGraph.infograph_dataset import SyntheticDataset
from data_genie.InfoGraph.train_infograph import train_infograph, argument

def get_embeddings(dataset, embedding_type):
	return {
		'unsupervise': get_embeddings_gcn,
		'handcrafted': get_embeddings_handcrafted,
	}[embedding_type[:11]](dataset, embedding_type)

def get_embeddings_gcn(dataset, model_file):
	# Extract which InfoGraph model we are looking for from the specified `model_file`
	dim, layers = None, None
	splitted = model_file.split("_")
	for i, word in enumerate(splitted):
		if word == "dim": dim = int(splitted[i+1])
		if word == "layers": layers = int(splitted[i+1])

	PATH = EMBEDDINGS_PATH_GCN(dataset, dim, layers)
	if not os.path.exists(PATH + ".npy"): prep_gcn_embeddings(dim, layers)
	return load_numpy(PATH)

def prep_gcn_embeddings(dim, layers):
	# Step-1: Check if the unsupervised GCN Model has been trained?
	if not os.path.exists(INFOGRAPH_MODEL_PATH(dim, layers)):
		print("Specified InfoGraph configuration not trained yet, training now..")
		infograph_args = argument()
		infograph_args.n_layers = layers
		infograph_args.hid_dim = dim
		train_infograph(infograph_args)
		
	print("Loading best InfoGraph model..")
	model = InfoGraph(dim, layers)
	model.load_state_dict(torch.load(INFOGRAPH_MODEL_PATH(dim, layers)))
	model.eval()

	print("Loading data..")
	# Keep the dimension of node features fixed to a reasonable value.
	# If you want to change this, please also change at `InfoGraph/infograph_model.py` line 134
	dataset = SyntheticDataset(feature_dimension = 32)
	graphs, _ = map(list, zip(*dataset))
	num_graphs = len(graphs)

	# Embeddings
	print("Getting GCN embeddings..")
	print(len(graphs))

	BSZ = 32 # Batch-size for predicting dataset embeddings
	emb = np.zeros([ len(graphs), (dim * layers) + 5 ])
	for b in tqdm(range(0, len(graphs), BSZ)):
		wholegraph = dgl.batch(graphs[b:b+BSZ])
		wholegraph.ndata['attr'] = wholegraph.ndata['attr'].to(torch.float32)
		emb[b:b+BSZ] = np.hstack([
			model.get_embedding(wholegraph).cpu().numpy(),
			np.array(dataset.basic_data_stats_features[b:b+BSZ], dtype = np.float32)
		])

	del graphs, dataset, model
	gc.collect()
	
	# NOTE: Since we'll anyways be training on ALL datasets, we'll prep embeddings for ALL datasets at once
	at = 0
	for d in datasets:
		final = np.zeros([ total_embeddings, (dim * layers) + 5 ])

		for task, metrics in scenarios:
			final[get_embedding_id(task, 'complete_data', 0)] = emb[at] ; at += 1
			
			for sampling_percent in percent_rns_options:					
				for sampling in all_samplers:
					final[get_embedding_id(task, sampling, sampling_percent)] = emb[at] ; at += 1

		save_numpy(EMBEDDINGS_PATH_GCN(d, dim, layers), final)

	assert at == num_graphs

def get_embeddings_handcrafted(dataset, model_file):
	PATH = EMBEDDINGS_PATH_HANDCRAFTED(dataset)
	if not os.path.exists(PATH + ".npy"): prep_handcrafted_embeddings(dataset, PATH)
	return load_numpy(PATH)

def prep_handcrafted_embeddings(dataset, save_path):
	final = np.zeros([ total_embeddings, NUM_FEAUTRES ])
	
	print("Getting handcrafted embeddings..")
	loop = tqdm(total = len(all_samplers) * len(scenarios) * len(percent_rns_options))
	for task, metrics in scenarios:
		final[get_embedding_id(task, 'complete_data', 0)] = get_single_feature(dataset, task, 'complete_data', 0)
		
		for sampling_percent in percent_rns_options:
			for sampling in sampling_kinds:
				final[get_embedding_id(task, sampling, sampling_percent)] = get_single_feature(dataset, task, sampling, sampling_percent)
				loop.update(1)

			for svp_method in svp_methods: 
				for sampling in sampling_svp:
					name = "svp_{}_{}".format(svp_method, sampling)
					final[get_embedding_id(task, name, sampling_percent)] = get_single_feature(
						dataset, task, "svp_{}".format(svp_method), sampling_percent, svp = sampling
					)
					loop.update(1)

	loop.close()
	save_numpy(save_path, final)

def get_single_feature(dataset, task, sampling, sampling_percent, svp = None):
	hyper_params = {
		'dataset': dataset,
		'task': task,
		'sampling': sampling,
		'sampling_percent': sampling_percent,
		'sampling_svp': svp
	}

	data = DataHolder(get_data_path(hyper_params), get_index_path(hyper_params))

	MIL = float(1e6)
	return list([ float(data.num_users) / MIL, float(data.num_items) / MIL ] + \
				[ float(data.num_train_interactions) / MIL ] + \
				[ float(data.num_val_interactions) / MIL ] + \
				[ float(data.num_test_interactions) / MIL ] + \
				degree_distribution(data) + \
				graph_characteristics(data)) 

def degree_distribution(data):
	user_degree, item_degree = defaultdict(int), defaultdict(int)

	for at, (u, i, r) in enumerate(data.data):
		if data.index[at] != -1:
			user_degree[u] += 1
			item_degree[i] += 1

	user_degree = sorted([ user_degree[u] for u in user_degree ])
	item_degree = sorted([ item_degree[i] for i in item_degree ])

	return sample_distribution(user_degree) + sample_distribution(item_degree)

def graph_characteristics(data):
	g, snap_g = create_nx_graph(data)

	return \
	sample_distribution([ g.degree(n) for n in g.nodes() ]) + \
	sample_distribution([ len(c) for c in nx.connected_components(g) ]) + \
	sample_distribution(hop_plot(snap_g)) + \
	sample_distribution(snap_g.GetEigVals(100), reverse = True) + \
	sample_distribution(clustering_coefficient(snap_g)) 

def hop_plot(g):
	g.PlotHops("temp", "Random Graph hop plot", False, 128)
	f = open("hop.temp.tab", 'r') ; lines = f.readlines() ; f.close()
	for f in [ "hop.temp.png", "hop.temp.tab", "hop.temp.plt" ]: os.remove(f)
	return list(map(lambda x: float(x.strip().split("\t")[1]), lines[4:]))	

def clustering_coefficient(g):
	Cf, CfVec = g.GetClustCf(True, -1)
	return [ pair.GetVal2() for pair in CfVec ]

def sample_distribution(distribution, reverse = False):
	distribution = sorted(distribution, reverse = reverse) ; n = len(distribution)
	to_pick = np.round(np.linspace(0, len(distribution) - 1, NUM_SAMPLES)).astype(int)
	
	return [ distribution[i] for i in to_pick ]

def create_nx_graph(data):
	g = nx.Graph()
	snap_g = snap.TUNGraph.New()

	# Add nodes & edges
	user_map, item_map, node_num = {}, {}, 0

	for at, (u, i, r) in enumerate(data.data):
		if data.index[at] != -1: 
			if u not in user_map:
				user_map[u] = node_num
				g.add_node(node_num)
				snap_g.AddNode(node_num)
				node_num += 1
			if i not in item_map:
				item_map[i] = node_num
				g.add_node(node_num)
				snap_g.AddNode(node_num)
				node_num += 1
			g.add_edge(user_map[u], item_map[i])
			snap_g.AddEdge(user_map[u], item_map[i])
	assert node_num == g.number_of_nodes()
	return g, snap_g
