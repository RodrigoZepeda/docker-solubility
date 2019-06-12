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
https://github.com/PatWalters/solubility/blob/master/solubility_comparison.py

For interpretability
https://deepchem.io/docs/notebooks/Explaining_Tox21.html
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
os.chdir("/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores")


import pickle
import numpy as np
np.random.seed(1342341)

import sys
sys.stderr = None            # suppress stderr
import deepchem as dc
sys.stderr = sys.__stderr__  # restore stderr

from deepchem.models import DAGModel
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

#Establish featurizer
featurizer = dc.feat.ConvMolFeaturizer()

#Read CSV with featurizer
loader = dc.data.CSVLoader(
      tasks=["measured log solubility in mols per litre"],
      smiles_field="smiles",
      featurizer=featurizer)

#dataset = loader.featurize(dataset_file,  shard_size=8192)
dataset = loader.featurize(dataset_file)

#Split dataset according to index (why?)
splitter = dc.splits.IndexSplitter(dataset_file)
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset)

max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
max_atoms_valid = max([mol.get_num_atoms() for mol in valid_dataset.X])
max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
max_atoms = max([max_atoms_train, max_atoms_valid, max_atoms_test])

with open("DAGModel/maxatoms.pickle", "wb") as f:
    pickle.dump(max_atoms, f)


reshard_size = 512
transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
train_dataset.reshard(reshard_size)
train_dataset = transformer.transform(train_dataset)
valid_dataset.reshard(reshard_size)
valid_dataset = transformer.transform(valid_dataset)


"""
MODEL BUILDING
"""

# Fit
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Do setup required for tf/keras models
n_tasks = 1 #Only solubility to predict
n_atom_feat = 75
batch_size = 64
n_graph_feat = 30
nb_epoch = 10
model = DAGModel(n_tasks = n_tasks, max_atoms = max_atoms, n_atom_feat = n_atom_feat,
                   n_graph_feat = n_graph_feat, mode = "regression",
                   batch_size=batch_size, learning_rate=1e-3,
                   use_queue=False,
                   model_dir="/home/rod/Dropbox/Quimica/Analysis/ANalisis/Borradores/DAGModel/") #To prevent overfitting

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
#Check model predictions
from rdkit import Chem
smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C',
          'CCOC(=O)CC',
          'CSc1nc(NC(C)C)nc(NC(C)C)n1',
          'CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1',
          'Cc1cc2ccccc2cc1C']
mols = [Chem.MolFromSmiles(s) for s in smiles]
x = featurizer.featurize(mols)
x2 = dc.data.datasets.NumpyDataset(x)
y = transformer.transform(x2)
predicted_solubility = model.predict(y)
"""