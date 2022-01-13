import time
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import Ridge, LogisticRegression

from data_genie.data_genie_config import *
from data_genie.data_genie_model import PointwiseDataGenie
from data_genie.data_genie_utils import LOGS_BASE, TENSORBOARD_BASE, MODELS_BASE

from torch_utils import xavier_init
from utils import log_end_epoch, INF

def get_metrics(output, y, pointwise):
	output = np.array(output).reshape([ len(output) // len(all_samplers), len(all_samplers) ])
	y = np.array(y).reshape([ len(y) // len(all_samplers), len(all_samplers) ])

	metrics = defaultdict(list)
	for i in range(len(output)):
		pred, true = np.array(output[i]), np.array(y[i])

		keep_indices = []
		for j in range(len(true)):
			if true[j] >= -1 and true[j] <= 1: keep_indices.append(j)
		keep_indices = np.array(keep_indices)

		if len(keep_indices) == 0: continue
		pred, true = pred[keep_indices], true[keep_indices]

		if pointwise: metrics['MSE'] += ((pred - true) ** 2).tolist()

		if len(keep_indices) < 5: continue
		metrics['P@1'].append(float(true[np.argmax(pred)] == np.max(true))) # Since there can be many with same kendall's tau
		metrics['Rand P@1'].append(float(true[np.random.randint(0, len(true))] == np.max(true)))
	
	return { k: round(100.0 * np.mean(metrics[k]), 2) if k != 'MSE' else round(np.mean(metrics[k]), 4) for k in metrics }

def validate(model, data, which_data, pointwise):
	model.model.eval()

	output, y = [], []
	with torch.no_grad():
		for complete, subset, task, metric, tau in data.test_iter(which_data):
			output_batch = model.model(complete, subset, task, metric)
			if pointwise: output_batch = torch.tanh(output_batch)
			output += output_batch.cpu().numpy().tolist()
			y += tau

	return get_metrics(output, y, pointwise)

def train_one_epoch(model, data):
	model.model.train()
		
	metric, total = 0.0, 0.0
	for data_batch in data.train_iter():
		# Empty the gradients
		model.model.zero_grad()
		model.optimizer.zero_grad()
	
		# Forward pass
		loss, temp_metric, temp_total = model.step(data_batch)
		metric += temp_metric ; total += temp_total

		# Backward pass
		loss.backward()
		model.optimizer.step()
	
	return model.get_metric_dict(metric, total)

def train_pytorch(data, Analyzer, feats_to_keep, lr, wd, dim, dropout, embedding_type, graph_dim, gcn_layers, EPOCHS, VALIDATE_EVERY):
	common_path = "{}_{}_dim_{}_drop_{}_lr_{}_wd_{}".format(
		"pointwise" if Analyzer == PointwiseDataGenie else "pairwise",
		embedding_type, dim, dropout, lr, wd
	)
	if feats_to_keep is not None: common_path += "_feats_" + "-".join(map(str, feats_to_keep))
	if embedding_type != "handcrafted": common_path += "_graph_dim_{}_gcn_layers_{}".format(graph_dim, gcn_layers)

	writer = SummaryWriter(log_dir = TENSORBOARD_BASE + common_path)
	
	model = Analyzer({
		'lr': lr,
		'weight_decay': float(wd),
		'dim': dim,
		'dropout': dropout,
		'feats': 5 + (len(feats_to_keep) * NUM_SAMPLES) if feats_to_keep is not None else None,
		'log_file': LOGS_BASE + '{}.txt'.format(common_path),
		'model_file': MODELS_BASE + '{}.pt'.format(common_path),
		'embedding_type': embedding_type,
		'graph_dim': graph_dim, 
		'gcn_layers': gcn_layers
	}, writer, xavier_init)

	# Train model
	start_time, best_metric = time.time(), -INF

	avg_p1 = 0.0
	for epoch in tqdm(range(1, EPOCHS + 1)):
		epoch_start_time = time.time()

		train_metrics = train_one_epoch(model, data)
		for m in train_metrics: writer.add_scalar('Train/' + m, train_metrics[m], epoch)
		log_end_epoch(model.hyper_params, train_metrics, epoch, time.time() - epoch_start_time, metrics_on = '(TRAIN)', dont_print = True)

		if epoch % VALIDATE_EVERY == 0:
			val_metrics = validate(model, data, data.val, pointwise = Analyzer == PointwiseDataGenie)
			for m in val_metrics: writer.add_scalar('Validation/' + m, val_metrics[m], epoch)
			log_end_epoch(model.hyper_params, val_metrics, epoch, time.time() - epoch_start_time, metrics_on = '(VAL)', dont_print = True)

			test_metrics = validate(model, data, data.test, pointwise = Analyzer == PointwiseDataGenie)
			for m in test_metrics: writer.add_scalar('Test/' + m, test_metrics[m], epoch)
			log_end_epoch(model.hyper_params, test_metrics, epoch, time.time() - epoch_start_time, metrics_on = '(TEST)', dont_print = True)

			avg_p1 += test_metrics['Rand P@1']

			if test_metrics["P@1"] > best_metric: 
				model.save() ; best_metric = test_metrics["P@1"]

	model.load()
	test_metrics = validate(model, data, data.test, pointwise = Analyzer == PointwiseDataGenie)
	test_metrics['Rand P@1'] = round(avg_p1 / float(EPOCHS), 2)
	for m in test_metrics: writer.add_scalar('Test/' + m, test_metrics[m], EPOCHS + 1)
	log_end_epoch(model.hyper_params, test_metrics, "final", time.time() - start_time, metrics_on = '(TEST)')

def train_linear_regression(data, embedding_type, feats_to_keep, C):
	start_time = time.time()
	
	log_file = LOGS_BASE + "linear_regression_{}_C_{}".format(embedding_type, C)
	if feats_to_keep is not None: log_file += "_feats_" + "-".join(map(str, feats_to_keep))
	log_file += '.txt'

	x, y = data.sklearn_regression_data
	
	########## Initially
	# model = Ridge(alpha = C).fit(x, y)

	########## Backward selection
	model = RFE(Ridge(alpha = C, normalize = True), n_features_to_select = 10, step = 1).fit(x, y)

	train_mse = round(np.mean((y - model.predict(x)) ** 2), 4)
	train_var = round(np.var(y), 4)

	output, y = [], []
	for complete, subset, task, metric, tau in data.test_iter(data.test):
		output += model.predict(
			data.sklearn_regression_feature(complete, subset, task.unsqueeze(-1), metric.unsqueeze(-1))
		).tolist()
		y += tau

	test_metrics = get_metrics(output, y, pointwise = True)
	test_metrics['Train MSE'] = train_mse
	test_metrics['Train Var'] = train_var
	log_end_epoch({ 'log_file': log_file }, test_metrics, "final", time.time() - start_time, metrics_on = '(TEST)')

def train_logistic_regression(data, embedding_type, feats_to_keep, C):
	start_time = time.time()

	log_file = LOGS_BASE + "logistic_regression_{}_C_{}".format(embedding_type, C)
	if feats_to_keep is not None: log_file += "_feats_" + "-".join(map(str, feats_to_keep))
	log_file += '.txt'

	x, y = data.sklearn_bce_data
	model = LogisticRegression(C = C, max_iter = 3000).fit(x, y)
	train_auc = round(roc_auc_score(y, model.predict_proba(x)[:, 1]), 4)

	output, y = [], []
	for complete, subset, task, metric, tau in data.test_iter(data.test):
		output += model.predict_proba(
			data.sklearn_bce_feature(complete, subset, subset, task.unsqueeze(-1), metric.unsqueeze(-1))
		)[:, 1].tolist()
		y += tau

	test_metrics = get_metrics(output, y, pointwise = False)
	test_metrics['Train AUC'] = train_auc
	log_end_epoch({ 'log_file': log_file }, test_metrics, "final", time.time() - start_time, metrics_on = '(TEST)')

def train_xgboost_regression(data, embedding_type, feats_to_keep, max_depth):
	start_time = time.time()

	common_path = "xgboost_regression_{}_depth_{}".format(embedding_type, max_depth)
	if feats_to_keep is not None: common_path += "_feats_" + "-".join(map(str, feats_to_keep))

	x, y = data.sklearn_regression_data
	model = XGBRegressor(max_depth = max_depth).fit(x, y)
	train_mse = round(np.mean((y - model.predict(x)) ** 2), 4)
	train_var = round(np.var(y), 4)

	output, y = [], []
	for complete, subset, task, metric, tau in data.test_iter(data.test):
		output += model.predict(
			data.sklearn_regression_feature(complete, subset, task.unsqueeze(-1), metric.unsqueeze(-1))
		).tolist()
		y += tau

	test_metrics = get_metrics(output, y, pointwise = True)
	test_metrics['Train MSE'] = train_mse
	test_metrics['Train Var'] = train_var
	log_end_epoch({ 'log_file': LOGS_BASE + "{}.txt".format(common_path) }, test_metrics, "final", time.time() - start_time, metrics_on = '(TEST)')

def train_xgboost_bce(data, embedding_type, feats_to_keep, max_depth):
	start_time = time.time()

	common_path = "xgboost_bce_{}_depth_{}".format(embedding_type, max_depth)
	if feats_to_keep is not None: common_path += "_feats_" + "-".join(map(str, feats_to_keep))

	x, y = data.sklearn_bce_data
	model = XGBClassifier(max_depth = max_depth, use_label_encoder=False, eval_metric = "logloss").fit(x, y)
	train_auc = round(roc_auc_score(y, model.predict(x)), 4)

	output, y = [], []
	for complete, subset, task, metric, tau in data.test_iter(data.test):
		output += model.predict(
			data.sklearn_bce_feature(complete, subset, subset, task.unsqueeze(-1), metric.unsqueeze(-1))
		).tolist()
		y += tau

	test_metrics = get_metrics(output, y, pointwise = False)
	test_metrics['Train AUC'] = train_auc
	log_end_epoch({ 'log_file': LOGS_BASE + "{}.txt".format(common_path) }, test_metrics, "final", time.time() - start_time, metrics_on = '(TEST)')
