import dgl
import torch
import torch.nn as nn

from torch_utils import is_cuda_available
from data_genie.data_genie_loss import PointwiseLoss, PairwiseLoss

# NOTE: Below two are the training classes for data-genie: pointwise/pairwise
class PointwiseDataGenie:
	def __init__(self, hyper_params, writer, xavier_init):
		self.hyper_params = hyper_params
		self.writer = writer

		self.model = DataGenie(hyper_params)
		if is_cuda_available: self.model.cuda()
		
		xavier_init(self.model) ; print(self.model)
		
		self.criterion = PointwiseLoss()
		self.optimizer = torch.optim.Adam(
			self.model.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.98),
			weight_decay=hyper_params['weight_decay']
		)

	def step(self, data_batch):
		complete, subset, task, metric, y = data_batch

		# Forward pass
		output = torch.tanh(self.model(complete, subset, task, metric)) # Since in range [ -1, 1 ]
		loss = self.criterion(output, y, return_mean = True)

		return torch.mean(loss), float(torch.sum(loss)), float(output.shape[0])

	def get_metric_dict(self, mse, total): return { 'MSE': round(mse / total, 4) }

	def save(self): torch.save(self.model.state_dict(), self.hyper_params['model_file'])

	def load(self): self.model.load_state_dict(torch.load(self.hyper_params['model_file']))

class PairwiseDataGenie:
	def __init__(self, hyper_params, writer, xavier_init):
		self.hyper_params = hyper_params
		self.writer = writer

		self.model = DataGenie(hyper_params)
		if is_cuda_available: self.model.cuda()

		xavier_init(self.model) ; print(self.model)
		
		self.criterion = PairwiseLoss()
		self.optimizer = torch.optim.Adam(
			self.model.parameters(), lr=hyper_params['lr'], betas=(0.9, 0.98),
			weight_decay=hyper_params['weight_decay']
		)
		
	def step(self, data_batch):
		complete, pos, neg, task, metric = data_batch

		# Forward pass
		pos_output = self.model(complete, pos, task, metric)
		neg_output = self.model(complete, neg, task, metric)
		loss = self.criterion(pos_output, neg_output, return_mean = True)

		# Metric
		auc = float(torch.sum(pos_output > neg_output))
		return loss, auc, float(pos_output.shape[0])

	def get_metric_dict(self, auc, total): return { 'AUC': round(auc / total, 4) }

	def save(self): torch.save(self.model.state_dict(), self.hyper_params['model_file'])

	def load(self): self.model.load_state_dict(torch.load(self.hyper_params['model_file']))

# NOTE: Below is the actual data-genie pytorch model
class DataGenie(nn.Module):
	def __init__(self, hyper_params):
		super(DataGenie, self).__init__()

		self.feats = hyper_params['feats']

		if self.feats is None:
			dim, layers = None, None
			splitted = hyper_params['embedding_type'].split("_")
			for i, word in enumerate(splitted):
				if word == "dim": dim = int(splitted[i+1])
				if word == "layers": layers = int(splitted[i+1])
			self.feats = (dim * layers) + 5

		self.task_embedding = nn.Embedding(3, 3)
		self.metric_embedding = nn.Embedding(4, 4)
		self.final = nn.Sequential(
			nn.Linear((2 * self.feats) + 3 + 4, hyper_params['dim']),
			nn.Dropout(hyper_params['dropout']),
			nn.ReLU(),
			nn.Linear(hyper_params['dim'], 1),
		)

	def forward(self, complete_feats, subset_feats, task, metric):	
		return self.final(torch.cat([
			complete_feats, 
			subset_feats,
			self.task_embedding(task), self.metric_embedding(metric)
		], axis = -1))[:, 0]
