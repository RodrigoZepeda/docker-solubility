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
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
os.chdir("/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores")

import numpy as np
np.random.seed(1342341)

import sys
sys.stderr = None            # suppress stderr
import deepchem as dc
sys.stderr = sys.__stderr__  # restore stderr
import xgboost
from deepchem.models.xgboost_models import XGBoostModel
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.save import load_from_disk

import tensorflow as tf
tf.set_random_seed(1342341)

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

"""
DATASETS
"""

#Read dataset
dataset_file= "/home/rod/Dropbox/Quimica/Bases/Raw/delaney-processed.csv"
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

# Do setup required for tf/keras models
nb_epoch = 10
n_estimators = 500
xgboost_model = xgboost.XGBRegressor(n_estimators=10)
model = dc.models.XGBoostModel(xgboost_model, "/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores/XGBoost/")

# Fit trained model
print("Fitting model")
model.fit(train_dataset, nb_epoch=nb_epoch)
model.save()
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric])
valid_scores = model.evaluate(valid_dataset, [metric])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

"""
With featurizer = dc.feat.ConvMolFeaturizer()
----------------------------------------
Train scores
{'mean-pearson_r2_score': 0.9637847589740351}
Validation scores
{'mean-pearson_r2_score': 0.8422518074477224}
"""

"""
PREDICTION
"""

"""
#Check model predictions
smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C',
          'CCOC(=O)CC',
          'CSc1nc(NC(C)C)nc(NC(C)C)n1',
          'CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1',
          'Cc1cc2ccccc2cc1C']
mols = [Chem.MolFromSmiles(s) for s in smiles]
x = featurizer.featurize(mols)
predicted_solubility = model.predict_on_batch(x)
"""
