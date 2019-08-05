#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that generates the complete database
"""

from __future__ import print_function
from rdkit import Chem
import os

import pandas as pd
import numpy as np
from math import pow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

import multiprocessing

np.random.seed(1342341)

"""
DATASETS
"""


#Count number of processors of server
cpus = multiprocessing.cpu_count()

dataset_file = "Complete_dataset_without_duplicates.csv"
train_file   = "train_deepchem.csv"
test_file    = "test_deepchem.csv"
modeldir     = "random_forest_2/"
nestimators  = int(pow(2,10)) #Deepchem with 1024=2^10 results in 0.97/0.94
fbits        = 11             #2^fbits Bits in fingerprint. Deepchem has 2048 = 2^11
radius       = 2              #Fingerprint radius. Deehcpem has 2
train_perc   = 0.8            #percent of data in train set. Deepchem has 0.8
 
#Create directory if not exists
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

#Import database
try:
    database = pd.read_csv(dataset_file)
except:
    print("Unable to locate file")

try: 
    trainset = pd.read_csv(train_file, header = None)
    testset  = pd.read_csv(test_file, header = None)
except: 
    print("Unable to locate files")
    
#Associate molecule to each smile and create fingerprint
mols = [Chem.MolFromSmiles(x) for x in  database["smiles"]]

#Get the fingerprints for the mols (same fingerprints as deepchem)
fmols = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol = x, 
                                               radius = int(radius), 
                                               useChirality = False, 
                                               useBondTypes = True,
                                               useFeatures = False,
                                               nBits = int(pow(2, fbits))) for x in mols]
string_fpints = [x.ToBitString() for x in fmols]

#Add fingerprint to database
fpints   = [np.fromstring(x,'u1') - ord('0') for x in string_fpints]
database =  pd.concat([database.reset_index(drop=True), pd.DataFrame(fpints)], axis=1)

#Normalize validation
mu = np.mean(database["logS"])
s  = np.std(database["logS"])
database["Normalized_logS"] = (database["logS"] - mu)/s #coincides with deepchem


# Split the data into training and testing sets
#train, validate, test = np.split(database.sample(frac=1), [int(train_perc*len(database)), int((0.5 + train_perc/2)*len(database))])
tlimit = int(train_perc*len(database))
vlimit = int((0.5 + train_perc/2)*len(database))


train = database[:tlimit]
test  = database[tlimit:vlimit] 

#Run Sci-Kit learn
#----> HASTA AQUÍ SALVO POR LOS SETS ES IGUAL QUE DEEPCHEM
# Instantiate model with nestimators decision trees
rf = RandomForestRegressor(n_estimators = nestimators,
                             criterion = "mse",
                             max_features = "auto",
                             bootstrap = True,
                             oob_score = False,
                             n_jobs = int(cpus/2),
                             verbose = 1)

# Train the model on training data
rf.fit(train.drop(["smiles", "logS","Normalized_logS"], axis=1), train["Normalized_logS"])

# Use the forest's predict method on the test data
predictions   = rf.predict(test.drop(["smiles", "logS","Normalized_logS"], axis = 1))
predictions_t = rf.predict(train.drop(["smiles", "logS","Normalized_logS"], axis = 1))


# Calculate the errors in the 
normalized_errors = predictions - test["Normalized_logS"]
logs_errors = s*normalized_errors
S_error = np.exp(s*predictions + mu) - np.exp(test["logS"])

#Get r^2


print('Train Sklearn R²:', round(r2_score(predictions_t, train["Normalized_logS"]) , 2))
print('Train Deepchems R²:', round(pearsonr(predictions_t, train["Normalized_logS"])[0]**2 , 2)) #This is what deepchem uses

print('Pred Sklearn R²:', round(r2_score(predictions, test["Normalized_logS"]) , 2))
print('Pred Deepchems R²:', round(pearsonr(predictions, test["Normalized_logS"])[0]**2 , 2)) #This is what deepchem uses


print('Train Sklearn R²:', round(r2_score(predictions_t, testset[0]) , 2))
print('Train Deepchems R²:', round(pearsonr(predictions_t, testset[0])[0]**2 , 2)) #This is what deepchem uses

print('Pred Sklearn R²:', round(r2_score(predictions, trainset[0]) , 2))
print('Pred Deepchems R²:', round(pearsonr(predictions, trainset[0])[0]**2 , 2)) #This is what deepchem uses


try:
    from notifyending import notify_ending
    notify_ending("Finished fitting random forest")
except:
    print("Random forest")

#quit()
