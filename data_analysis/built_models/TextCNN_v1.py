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

import deepchem as dc
from deepchem.models import TextCNNModel
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
featurizer = dc.feat.RawFeaturizer()

#Read CSV with featurizer
loader = dc.data.CSVLoader(
      tasks=["measured log solubility in mols per litre"],
      smiles_field="smiles",
      featurizer=featurizer)
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
char_dict, length = dc.models.TextCNNModel.build_char_dict(train_dataset)

# Do setup required for tf/keras models
batch_size = 64        # Batch size of models
nb_epoch = 10
model = TextCNNModel(1, char_dict, seq_length=length,
                     mode='regression',
                     learning_rate=1e-3,
                     batch_size=batch_size,
                     use_queue=False,
                     model_dir="/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores/TextCNN/") #To prevent overfitting

# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)

model.save()
print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)


#predicted_solubility = model.predict(train_dataset)

