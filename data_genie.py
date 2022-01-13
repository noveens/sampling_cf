from data_genie.data_genie_config import *
from data_genie.data_genie_trainers import *
from data_genie.data_genie_data import OracleData
from data_genie.data_genie_model import PointwiseDataGenie, PairwiseDataGenie

# NOTE: Please edit the config in `data_genie/data_genie_config.py` before \
# 		running this trainer script

print("Datasets:", ", ".join(datasets))

for embedding_type in [
	'handcrafted',
	
	'unsupervised_gcn_dim_8_layers_1',
	'unsupervised_gcn_dim_8_layers_3',
	'unsupervised_gcn_dim_16_layers_1',
	'unsupervised_gcn_dim_16_layers_3',
	'unsupervised_gcn_dim_32_layers_1',
]:

	# If you want to try out different combinations of handcrafted features
	options = [
		# [0, 1],
		# [0, 1, 3],
		# [0, 1, 4],
		# [0, 1, 5],
		[0, 1, 3, 4, 5],
		# [0, 1, 2, 3, 4, 5, 6],
	] if embedding_type == 'handcrafted' else [ None ]

	# Create model
	for feats_to_keep in options:
		# Load data
		print("\n\n{} Using {} embedding {}\n\n".format("="*30, embedding_type, "="*30))
		pointwise_data = OracleData(datasets, feats_to_keep, embedding_type, bsz = 128, pointwise = True)
		pairwise_data = OracleData(datasets, feats_to_keep, embedding_type, bsz = 128, pointwise = False)

		########### Linear Regression
		print("\n\n{} Linear Regression {}\n\n".format("="*30, "="*30))
		for C in [ 1e-4, 1e-2, 1, 1e2, 1e4 ]: train_linear_regression(pointwise_data, embedding_type, feats_to_keep, C = float(C))

		########### Logistic Regression
		print("\n\n{} Logistic Regression {}\n\n".format("="*30, "="*30))
		for C in [ 1e-2, 1, 1e2, 1e4 ]: train_logistic_regression(pairwise_data, embedding_type, feats_to_keep, C = float(C))

		########### XGBoost
		print("\n\n{} XGBoost Regression {}\n\n".format("="*30, "="*30))
		for max_depth in [ 2, 4, 6, 8, 10 ]:
			train_xgboost_regression(pointwise_data, embedding_type, feats_to_keep, max_depth = max_depth)
		
		print("\n\n{} XGBoost Classification {}\n\n".format("="*30, "="*30))
		for max_depth in [ 2, 4, 6, 8, 10 ]:
			train_xgboost_bce(pairwise_data, embedding_type, feats_to_keep, max_depth = max_depth)

		########## Neural network
		print("\n\n{} Data-Genie {}\n\n".format("="*30, "="*30))
		for Analyzer in [ 
			PointwiseDataGenie, 
			PairwiseDataGenie 
		]:

			print("\nPointwise:" if Analyzer == PointwiseDataGenie else "\nPairwise:")

			# NOTE: These hyper-parameters were estimated using a basic grid-search						
			LR = 0.001
			WD = float(1e-6)
			DIM = 64
			DROPOUT = 0.2
			GRAPH_DIM = 64
			GCN_LAYERS = 2
			EPOCHS = 200 if embedding_type != 'handcrafted' else 20

			train_pytorch(
				pointwise_data if Analyzer == PointwiseDataGenie else pairwise_data,
				Analyzer, feats_to_keep, LR, WD, DIM, 
				DROPOUT, embedding_type, GRAPH_DIM, GCN_LAYERS,
				EPOCHS = EPOCHS, VALIDATE_EVERY = 1
			)
