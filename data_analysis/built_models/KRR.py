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
#os.chdir("/home/rod/Dropbox/Quimica/Docker/docker-solubility/data_analysis/built_models/")

import numpy as np
np.random.seed(1342341)

import sys
sys.stderr = None            # suppress stderr
import deepchem as dc
sys.stderr = sys.__stderr__  # restore stderr
from sklearn.kernel_ridge import KernelRidge
from deepchem.models.sklearn_models import SklearnModel
from deepchem.utils.save import load_from_disk

import tensorflow as tf
tf.set_random_seed(1342341)

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
"""
DATASETS
"""

#Read dataset
dataset_file= "Complete_dataset_without_duplicates.csv"
modeldir = "krr/"

dataset = load_from_disk(dataset_file)

#Establish featurizer
featurizer = dc.feat.fingerprints.CircularFingerprint(size=1024)

#Read CSV with featurizer
loader = dc.data.CSVLoader(
      tasks=["logS"],
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
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.median)

# Do setup required for tf/keras models
sklmodel = KernelRidge(kernel="rbf", alpha=1e-5, gamma=0.05)
model = SklearnModel(sklmodel, modeldir)

# Fit trained model
print("Fitting model")
model.fit(train_dataset)
model.save()
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric,  dc.metrics.Metric(dc.metrics.mae_score)])
 = model.evaluate(valid_dataset, [metric, dc.metrics.Metric(dc.metrics.mae_score)])

#Error kernel
predict_train = pd.DataFrame(model.predict(train_dataset), columns=['prediction']).to_csv(modeldir + "predict_train.csv")
predict_valid = pd.DataFrame(model.predict(valid_dataset), columns=['prediction']).to_csv(modeldir + "predict_validation.csv")

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)
metrics =   np.concatenate(([], ["r2","mse"]))
print("Dataframe for n_estimator selection")
df = pd.DataFrame({'metrics': metrics,
                   'train_scores': list(train_scores.values()),
                   'result_scores': list(valid_scores.values())})
df.to_csv(modeldir + "KRRAnalysis.csv", index= False)


 #Append trains cores and results
pd.DataFrame(train_dataset.y, columns=['prediction']).to_csv(modeldir + "train_original.csv")
pd.DataFrame(valid_dataset.y, columns=['prediction']).to_csv(modeldir + "valid_original.csv")
