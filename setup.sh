#!/bin/bash

download_amazon_magazine() {
	mkdir datasets/magazine/

	link="http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Magazine_Subscriptions.csv"
	wget $link -P datasets/magazine/ -q --show-progress

	mv datasets/magazine/Magazine_Subscriptions.csv datasets/magazine/data.csv
}

download_ml_100k() {
	link="https://files.grouplens.org/datasets/movielens/ml-100k.zip"

	wget $link -P datasets/ -q --show-progress
	unzip -qq datasets/ml-100k.zip -d datasets/ 
	rm datasets/ml-100k.zip
}

# Where to store the datasets?
mkdir -p datasets/

# Where to store the logs/models of trained models
mkdir -p experiments/sampling_runs/results/logs/trained/
mkdir -p experiments/sampling_runs/results/models/trained/

# Where to store the logs/models of trained proxy models for SVP
mkdir -p experiments/sampling_runs/results/logs/SVP/
mkdir -p experiments/sampling_runs/results/models/SVP/

# Base directory for all Data-Genie experiments
mkdir -p experiments/data_genie/

# Download the 0-core version of an amazon-dataset & the MovieLens dataset
echo "============= Downloading datasets ============="
download_amazon_magazine
download_ml_100k

echo "============= Preprocessing datasets ============="
python preprocess.py
