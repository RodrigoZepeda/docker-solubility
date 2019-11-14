#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that generates the complete database
"""

from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem

import os
import pandas as pd
import numpy as np
import multiprocessing
from math import pow, log
import pickle
np.random.seed(1342341)

"""
DATASETS
"""


#Count number of processors of server
cpus = multiprocessing.cpu_count()

dataset_file= "Complete_dataset_without_duplicates.csv"
modeldir = "random_forest_classifier/"
nestimators = int(pow(2,18)) #Deepchem with 1024=2^10 results in 0.97/0.94
fbits = 13 #2^fbits Bits in fingerprint. Deepchem has 2048 = 2^11
radius =3 #Fingerprint radius. Deehcpem has 2
train_perc = 0.6 #percent of data in train set. Deepchem has 0.8
logs_limit = log(0.3, 10)

#Create directory if not exists
if not os.path.exists(modeldir):
    os.makedirs(modeldir)


#Import database
database = pd.read_csv(dataset_file)

#Associate molecule to each smile and create fingerprint
mols = [Chem.MolFromSmiles(x) for x in  database["smiles"]]

#Get the fingerprints for the mols (same fingerprints as deepchem)
fmols = [AllChem.GetMorganFingerprintAsBitVect(mol = x, 
                                               radius = int(radius), 
                                               useChirality = False, 
                                               useBondTypes = True,
                                               useFeatures = False,
                                               nBits = int(pow(2, fbits))) for x in mols]
string_fpints = [x.ToBitString() for x in fmols]

#Add fingerprint to database
fpints = [np.array(list(x))  for x in string_fpints]
database =  pd.concat([database.reset_index(drop=True), pd.DataFrame(fpints)], axis=1)

#Count out the cases we are looking for classify
total_cases = sum(x > logs_limit for x in database["logS"])
print("Total cases are ", total_cases, " of sample representing ", round(100*total_cases/len(database),2), "% of cases" )

#Classify
database["Classification"] = True
database["Classification"] = [(x > logs_limit) for x in database["logS"]]

print("Total cases are ", 100*database.Classification.sum()/len(database),"% of cases" )

# Split the data into training and testing sets
train, validate, test = np.split(database.sample(frac=1), [int(train_perc*len(database)), int((0.5 + train_perc/2)*len(database))])

train_cases    = sum(x > logs_limit for x in train["logS"])
validate_cases = sum(x > logs_limit for x in validate["logS"])
test_cases     = sum(x > logs_limit for x in test["logS"])

print("Total train cases are ", 100*train.Classification.sum()/len(train), "% of cases" )
print("Total valid cases are ", 100*validate.Classification.sum()/len(validate), "% of cases" )
print("Total test cases are ", 100*test.Classification.sum()/len(test), "% of cases" )

correct_check = pd.concat([train, test, validate])
print("Total check cases are ", 100*correct_check.Classification.sum()/len(database), "% of cases" )



#Run Sci-Kit learn
from sklearn.ensemble import RandomForestClassifier

sum_positive = train.Classification.sum()
sum_negative = len(train) - sum_positive
scale_pos_weight = sum_negative / sum_positive

# Instantiate model with nestimators decision trees
rf = RandomForestClassifier(n_estimators = nestimators,
                             criterion = "gini",
                             max_features = "auto",
                             bootstrap = True,
                             oob_score = False,
                             n_jobs = int(cpus/2),
                             verbose = 1,
                             class_weight={0:1,1:scale_pos_weight})

# Train the model on training data
rf.fit(train.drop(["smiles", "logS","Classification"], axis=1), train["Classification"])

# Use the forest's predict method on the test data
predictions   = rf.predict(test.drop(["smiles", "logS","Classification"], axis = 1))
predictions_t = rf.predict(train.drop(["smiles", "logS","Classification"], axis = 1))
predictions_v = rf.predict(validate.drop(["smiles", "logS","Classification"], axis = 1))


#Get r^2
from sklearn.metrics import precision_score
print('Train Sklearn precision:', round(precision_score(predictions_t, train["Classification"]) , 2))
print('Pred Sklearn precision:', round(precision_score(predictions, test["Classification"]) , 2))
print('Validate Sklearn precision:', round(precision_score(predictions_v, validate["Classification"]) , 2))


with open('random_forest', 'wb') as f:
    pickle.dump(rf, f)


try:
    from notifyending import notify_ending
    notify_ending("Finished fitting random forest")
except:
    print("Random forest")

#quit()
