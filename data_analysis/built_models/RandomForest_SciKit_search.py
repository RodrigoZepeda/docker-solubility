#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that generates the complete database
"""

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
import os

import pandas as pd
import numpy as np
from math import pow

np.random.seed(328765784)

"""
DATASETS
"""


#Count number of processors of server
import multiprocessing
cpus = multiprocessing.cpu_count()

dataset_file= "Complete_dataset_without_duplicates.csv"
modeldir = "random_forest_2/"
nestimators = int(pow(2,10))
fbits = 5   #Bits in fingerprint
radius = 10 #Fingerprint radius
niter = 50  #Search iter
cvfold = 3 
#Create directory if not exists
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

#Import database
database = pd.read_csv(dataset_file)

#Associate molecule to each smile and create fingerprint
mols = [Chem.MolFromSmiles(x) for x in  database["smiles"]]

#Get the fingerprints for the mols
fmols = [AllChem.GetMorganFingerprintAsBitVect(x,radius, nBits = int(pow(2, fbits))) for x in mols]
string_fpints = [x.ToBitString() for x in fmols]

#Add fingerprint to database
fpints = [np.fromstring(x,'u1') - ord('0') for x in string_fpints]
database =  pd.concat([database.reset_index(drop=True), pd.DataFrame(fpints)], axis=1)

# Split the data into training and testing sets
train, validate, test = np.split(database.sample(frac=1), [int(.8*len(database)), int(.9*len(database))])

#Run Sci-Kit learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1000, stop = nestimators, num = 50)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = niter, cv = cvfold, verbose=2, 
                               n_jobs =  int(cpus/2))
# Fit the random search model
rf_random.fit(train.drop(["smiles", "logS"], axis=1), train["logS"])

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(rf_random.best_params_)

# Instantiate model with 1000 decision trees
'''
rf = RandomForestRegressor(n_estimators = nestimators,
                             criterion = "mse",
                             max_features = "auto",
                             bootstrap = True,
                             oob_score = False,
                             n_jobs = int(cpus/2))

# Train the model on training data
rf.fit(train.drop(["smiles", "logS"], axis=1), train["logS"])


# Use the forest's predict method on the test data
predictions = rf.predict(test.drop(["smiles","logS"], axis = 1))

# Calculate the absolute errors
errors = abs(predictions - test["logS"])

#Get r^2
from sklearn.metrics import r2_score

print('R²:', round(r2_score(predictions, test["logS"]) , 2), ' in logS.')
print('Median Absolute Error:', round(np.median(errors), 2), ' in logS.')

try:
    from notifyending import *
    notify_ending("Finished fitting random forest")
except:
    print("Random forest")

#quit()
'''