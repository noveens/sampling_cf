import torch
import numpy as np

from torch_utils import LongTensor, FloatTensor, is_cuda_available
from data_genie.data_genie_config import *
from data_genie.get_data import get_data_pointwise, get_data_pairwise
from data_genie.get_embeddings import get_embeddings
from data_genie.InfoGraph.infograph_dataset import SyntheticDataset

# NOTE: Helper functions
def sel(d, i): return list(map(lambda x: x[i], d))
def sel2(d, i): return list(map(lambda x: list(map(lambda y: y[i], x)), d))

# Main class for manipulating and iterating over training data for DataGenie
class OracleData:
	def __init__(self, datasets, feats_to_keep, embedding_type, bsz, pointwise):
		self.feats_to_keep, self.bsz, self.embedding_type = feats_to_keep, bsz, embedding_type
		get_final_data = get_data_pointwise if pointwise else get_data_pairwise

		# Get all embeddings
		self.embeddings = {}
		for dataset in datasets: self.embeddings[dataset] = get_embeddings(dataset, embedding_type)
		self.embeddings = self.normalize_embeddings(self.embeddings)

		# Create data
		self.dataset_indices, train_data, val_data, test_data = [], None, None, None
		for dataset in datasets:
			self.dataset_indices.append(0 if train_data is None else len(train_data))

			this_dataset = self.join(get_final_data(dataset), self.embeddings[dataset], pointwise)
			if train_data is None: train_data, val_data, test_data = this_dataset
			else:
				for at, container in enumerate([ train_data, val_data, test_data ]):
					for i in range(5): container[i] += this_dataset[at][i]

		print("\n{} data:".format("Pointwise" if pointwise else "Pairwise"))
		print("# of training points:", len(train_data[0]))
		print("# of validation points:", len(val_data[0]) // len(all_samplers))
		print("# of testing points:", len(test_data[0]) // len(all_samplers))

		# NOTE: Each entry inside `train_data`, `val_data`, `test_data` is a 5-tuple:
		# 0. Full-data embedding ID
		# 1. Subset embedding ID
		# 2. Task ID
		# 3. Metric ID
		# 4. Y: Kendall's Tau between the ranked list of recommendation algorithms \
		#       trained on full vs. the data-subset (what we want to predict w/ Data-Genie)

		# Move data to GPU
		self.val = [ FloatTensor(val_data[0]), FloatTensor(val_data[1]), LongTensor(val_data[2]), LongTensor(val_data[3]), val_data[4] ]
		self.test = [ FloatTensor(test_data[0]), FloatTensor(test_data[1]), LongTensor(test_data[2]), LongTensor(test_data[3]), test_data[4] ]

		if pointwise: 
			self.train = [
				FloatTensor(train_data[0]), FloatTensor(train_data[1]), LongTensor(train_data[2]), 
				LongTensor(train_data[3]), FloatTensor(train_data[4])
			]
		else:
			self.train = [
				FloatTensor(train_data[0]), FloatTensor(train_data[1]), FloatTensor(train_data[2]), 
				LongTensor(train_data[3]), LongTensor(train_data[4])
			]

	def normalize_embeddings(self, embeddings):
		combined_embeddings = np.concatenate(list(embeddings.values()), axis=0)
		
		# GCN-embeddings
		if self.feats_to_keep is None: pass
		# Handcrafted embeddings
		else:
			mean, std = self.get_mean_std(combined_embeddings)
			for dataset in embeddings:
				embeddings[dataset] = self.select_and_norm(embeddings[dataset], mean, std)
		
		return embeddings

	def join(self, all_data, embeddings, pointwise):
		train, val, test = all_data
		
		def join(data, indices): 
			for i in indices: data[i] = list(map(lambda x: embeddings[x], data[i]))
			return data
		
		test, val = map(lambda x: join(x, [ 0, 1 ]), [ test, val ])

		if pointwise: train = join(train, [ 0, 1 ])
		else: train = join(train, [ 0, 1, 2 ])

		return [ train, val, test ]

	def shuffle_train(self):
		num_train_interactions = len(self.train[0])
		rand_indices = np.arange(num_train_interactions) ; np.random.shuffle(rand_indices)
		rand_indices_tensor = LongTensor(rand_indices)

		for i in range(len(self.train)):
			self.train[i] = self.train[i][rand_indices_tensor]

		return self.train

	# Convert task/metric index to one-hot vector
	def one_hot(self, index, total):
		if index.shape[0] > 1:
			ret = torch.zeros(index.shape[0], total)
			if is_cuda_available: ret = ret.cuda()
			ret.scatter_(1, index, 1.0)
			return ret

		ret = torch.zeros(total)
		if is_cuda_available: ret = ret.cuda()
		ret.scatter_(0, index, 1.0)
		return ret

	def sklearn_regression_feature(self, complete, subset, task, metric):
		return torch.cat([ complete, subset, self.one_hot(task, 3), self.one_hot(metric, 4) ], axis = -1).cpu().numpy()

	def sklearn_bce_feature(self, complete, pos, neg, task, metric):
		return torch.cat([ complete, pos, neg, self.one_hot(task, 3), self.one_hot(metric, 4) ], axis = -1).cpu().numpy()

	@property
	def sklearn_regression_data(self):		
		shuffled_train = self.shuffle_train()

		x, y = [], []
		for b in range(shuffled_train[0].shape[0]):
			x.append(self.sklearn_regression_feature(
				shuffled_train[0][b], shuffled_train[1][b],
				shuffled_train[2][b].unsqueeze(-1), shuffled_train[3][b].unsqueeze(-1)
			))

			y.append(float(shuffled_train[4][b]))
		
		return np.asarray(x), np.asarray(y)

	@property
	def sklearn_bce_data(self):		
		shuffled_train = self.shuffle_train()

		x, y = [], []
		for b in range(shuffled_train[0].shape[0]):
			pos_or_neg = np.random.uniform()

			if pos_or_neg > 0.5:
				x.append(self.sklearn_bce_feature(
					shuffled_train[0][b], shuffled_train[1][b], shuffled_train[2][b],
					shuffled_train[3][b].unsqueeze(-1), shuffled_train[4][b].unsqueeze(-1)
				))
				y.append(1)
			else:
				x.append(self.sklearn_bce_feature(
					shuffled_train[0][b], shuffled_train[2][b], shuffled_train[1][b],
					shuffled_train[3][b].unsqueeze(-1), shuffled_train[4][b].unsqueeze(-1)
				))
				y.append(0)
		
		return np.asarray(x), np.asarray(y)

	def train_iter(self):
		num_train_interactions = len(self.train[0])
		
		shuffled_train = self.shuffle_train()

		for b in range(0, num_train_interactions, self.bsz):
			l, r = b, b + self.bsz

			yield shuffled_train[0][l:r], shuffled_train[1][l:r], shuffled_train[2][l:r], \
			shuffled_train[3][l:r], shuffled_train[4][l:r]

	def test_iter(self, data):
		for b in range(0, len(data[0]), self.bsz):
			l, r = b, b + self.bsz

			yield data[0][l:r], data[1][l:r], data[2][l:r], \
			data[3][l:r], data[4][l:r]

	def select_and_norm(self, data, mean, std):
		data = np.asarray(data)
		assert data.shape[-1] == NUM_FEAUTRES

		indices = []
		for f in self.feats_to_keep: indices += list(range(NUM_SAMPLES * f, NUM_SAMPLES * (f+1)))
		
		mean = mean[np.array(indices)] ; std = std[np.array(indices)]
		
		indices = list(range(5)) + list(map(lambda x: x+5, indices))
		indices = np.array(indices)

		if len(data.shape) == 2:
			data = data[:, indices]
			data[:, 5:] -= mean
			data[:, 5:] /= std
		else:
			data = data[:, :, indices]
			data[:, :, 5:] -= mean
			data[:, :, 5:] /= std
			
		return data
	
	def get_mean_std(self, combined_data):
		assert combined_data.shape[1] == NUM_FEAUTRES
		temp_data = combined_data[:, 5:NUM_FEAUTRES]
		
		std = np.array(list(map(lambda x: max(x, float(1e-6)), np.std(temp_data, axis = 0))))
		return np.mean(temp_data, axis = 0), std
