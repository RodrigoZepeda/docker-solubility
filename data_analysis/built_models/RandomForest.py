#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:23:37 2019

@author: rod

Based on
https://deepchem.io/docs/2.0.0/_modules/deepchem/molnet/load_function/delaney_datasets.html
https://github.com/deepchem/deepchem/blob/master/examples/delaney/delaney_graph_conv.py

Check, for inspiration
https://pchanda.github.io/Deepchem-GraphConvolutions/
https://www.deepchem.io/docs/notebooks/graph_convolutional_networks_for_tox21.html
https://www.oreilly.com/library/view/deep-learning-for/9781492039822/ch04.html

For interpretability
https://deepchem.io/docs/notebooks/Explaining_Tox21.html

Random forest:
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
#os.chdir("/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores")

import numpy as np
np.random.seed(1342341)

import sys
sys.stderr = None            # suppress stderr
import deepchem as dc
sys.stderr = sys.__stderr__  # restore stderr

from deepchem.models.sklearn_models import RandomForestRegressor, SklearnModel
from deepchem.utils.save import load_from_disk

import pandas as pd
import subprocess

"""
DATASETS
"""

#Read dataset
dataset_file= "delaney-processed.csv"
dataset = load_from_disk(dataset_file)

#Establish featurizer
featurizer = dc.feat.fingerprints.CircularFingerprint(size=1024)

#Read CSV with featurizer
loader = dc.data.CSVLoader(
      tasks=["measured log solubility in mols per litre"],
      smiles_field="smiles",
      featurizer=featurizer)

#dataset = loader.featurize(dataset_file,  shard_size=8192)
dataset = loader.featurize(dataset_file)

# Initialize transformers
transformers = [
      dc.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset)
]

for transformer in transformers:
   dataset = transformer.transform(dataset)

#Split dataset according to index (why?)
splitter = dc.splits.IndexSplitter(dataset_file)
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset)

"""
MODEL BUILDING
"""

# Fit
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Do setup for optimal parameter searching
n_estimators       = 2**np.linspace(0,20,21, endpoint=True, dtype=int)
max_features       = "auto" #Empirical

#Empty for dataframe
estimators    = []
train_results = []
test_results  = []
depths        = []
splits        = []
leafs         = []

for estimator in n_estimators:

        print('n_estimators = {0}'.format(estimator))
        #Create model
        sklmodel = RandomForestRegressor(n_estimators=estimator,
                                         criterion = "mse",
                                         max_features = max_features,
                                         bootstrap = True,
                                         oob_score = False,
                                         n_jobs = 32)
        model = SklearnModel(sklmodel, "RandomForest/")
        model.fit(train_dataset)

        #Append trains cores and results
        train_scores = model.evaluate(train_dataset, [metric])
        train_results.append(list(train_scores.values())[0])
        valid_scores = model.evaluate(valid_dataset, [metric])
        test_results.append(list(valid_scores.values())[0])

        #Append measures
        estimators.append(estimator)

# Fit trained model
model.save()
print("Dataframe for n_estimator selection")
df = pd.DataFrame({'nestimators': estimators,
                   'train_scores': train_results,
                   'result_scores': test_results})
df.to_csv("RandomForestAnalysis.csv")

from notifyending import *
notify_ending("Random Forest")
