from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import dgl
import os

from load_data import DataHolder
from data_path_constants import get_data_path, get_index_path
from data_genie.data_genie_config import *
from data_genie.data_genie_utils import save_numpy, load_numpy
from data_genie.data_genie_utils import INFOGRAPH_CACHED_GRAPHS, INFOGRAPH_CACHED_DATA_STATS

''' 
This is a wrapper class which contains ALL the CF datasets (and their sampled subsets) 
that we want to train Data-Genie on 
'''
class SyntheticDataset(DGLDataset):
	def __init__(self, feature_dimension): 
		self.feature_dimension = feature_dimension
		super().__init__(name='synthetic')

	def process(self):
		self.graphs = []
		self.basic_data_stats_features = []

		for dataset in datasets: 
			single_dataset = SingleDataset(self.feature_dimension, dataset)
			self.graphs += single_dataset.graphs
			self.basic_data_stats_features += single_dataset.basic_data_stats_features

		self.orig_total = len(self.graphs) # Including None
		assert len(self.basic_data_stats_features) == len(self.graphs)
		
		# Remove very small subsets
		self.graphs = list(filter(lambda x: x is not None, self.graphs))
		self.basic_data_stats_features = list(filter(lambda x: x is not None, self.basic_data_stats_features))
		assert len(self.basic_data_stats_features) == len(self.graphs)

	def __getitem__(self, i): 
		# NOTE: Return 0 as the graph-label
		# We don't care about the label since this is an unsupervised task
		return self.graphs[i], 0

	def __len__(self): return len(self.graphs)

class SingleDataset(DGLDataset):
	def __init__(self, feature_dimension, dataset): 
		self.feature_dimension = feature_dimension
		self.dataset = dataset
		super().__init__(name='synthetic')

	# Randomly init all node features (no implicit representation of user/item nodes)
	def get_random_node_features(self, graph):
		graph.ndata['attr'] = torch.empty(graph.num_nodes(), self.feature_dimension)
		torch.nn.init.xavier_uniform_(graph.ndata['attr'], gain=torch.nn.init.calculate_gain('relu'))
		return graph

	# Make the user-item interaction graph of the given CF dataset
	def add_graph(self, hyper_params):
		data = DataHolder(get_data_path(hyper_params), get_index_path(hyper_params))

		g = nx.Graph()

		# Add nodes & edges
		user_map, item_map, node_num = {}, {}, 0

		for at, (u, i, r) in enumerate(data.data):
			if data.index[at] != -1: 
				if u not in user_map:
					user_map[u] = node_num
					g.add_node(node_num)
					node_num += 1
				if i not in item_map:
					item_map[i] = node_num
					g.add_node(node_num)
					node_num += 1
				g.add_edge(user_map[u], item_map[i])
		
		assert node_num == g.number_of_nodes()

		# If nodes are too less return None
		if node_num < 50: 
			self.graphs.append(None)
			self.basic_data_stats_features.append(None)
			return None
		
		self.graphs.append(self.get_random_node_features(dgl.from_networkx(g)))
		
		MIL = float(1e6)
		self.basic_data_stats_features.append([
			float(data.num_users) / MIL, 
			float(data.num_items) / MIL,
			float(data.num_train_interactions) / MIL,
			float(data.num_val_interactions) / MIL,
			float(data.num_test_interactions) / MIL
		])

	def process(self):
		self.graphs = []
		self.basic_data_stats_features = []
		
		total_samplers = (len(svp_methods) * len(sampling_svp)) + len(sampling_kinds)
		loop = tqdm(total = len(scenarios) * (1 + (len(percent_rns_options) * total_samplers)))
		for task, metrics in scenarios:
			# Full dataset
			self.add_graph({
				'dataset': self.dataset,
				'task': task,
				'sampling': 'complete_data',
			}) ; loop.update(1)
			
			# Sub-sampled dataset
			for sampling_percent in percent_rns_options:					
				for sampling in sampling_kinds:
					self.add_graph({
						'dataset': self.dataset,
						'task': task,
						'sampling': sampling,
						'sampling_percent': sampling_percent,
					}) ; loop.update(1)

				for svp_method in svp_methods:
					for sampling in sampling_svp:
						self.add_graph({
							'dataset': self.dataset,
							'task': task,
							'sampling': "svp_{}".format(svp_method),
							'sampling_svp': sampling,
							'sampling_percent': sampling_percent,
						}) ; loop.update(1)
					
		loop.close()
		assert len(self.graphs) == len(self.basic_data_stats_features)

	def save(self): 
		save_graphs(self.graph_path, self.graphs)
		save_numpy(
			self.basic_data_stats_features_path, 
			np.array(self.basic_data_stats_features)
		)
		
	def load(self): 
		self.graphs, _ = load_graphs(self.graph_path)
		self.basic_data_stats_features = load_numpy(self.basic_data_stats_features_path).tolist()

	def has_cache(self): return os.path.exists(self.graph_path)

	@property
	def graph_path(self): return INFOGRAPH_CACHED_GRAPHS(self.dataset)

	@property
	def basic_data_stats_features_path(self): return INFOGRAPH_CACHED_DATA_STATS(self.dataset)
