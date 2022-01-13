import torch
import torch.nn as nn
import torch.nn.functional as F

class PointwiseLoss(nn.Module):
	def __init__(self): super(PointwiseLoss, self).__init__()

	def forward(self, output, y, return_mean = True):
		loss = torch.pow(output - y, 2)
		if return_mean: return torch.mean(loss)
		return loss

class PairwiseLoss(nn.Module):
	def __init__(self): super(PairwiseLoss, self).__init__()

	def forward(self, pos_output, neg_output, return_mean = True):
		loss = -F.logsigmoid(pos_output - neg_output)
		if return_mean: return torch.mean(loss)
		return loss
