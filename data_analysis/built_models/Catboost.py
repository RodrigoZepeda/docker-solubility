#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that generates a random forest model using catboost
https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html
https://medium.com/@nilimeshhalder/binary-classification-using-catboost-in-python-manual-and-automatic-parameter-tuning-440ad08a867f
"""

from __future__ import print_function
import os

import pandas as pd
import numpy as np
from math import pow

from catboost import Pool, CatBoostClassifier

import multiprocessing
from rdkitfeaturize import *

np.random.seed(1342341)

"""
DATASETS
"""


#Count number of processors of server
cpus = multiprocessing.cpu_count()
if cpus <= 1: #Default to one cpu
    cpus = 2;

inputfiledir  = "/home/rod/Dropbox/Quimica/Docker/docker-solubility/data_analysis/data/edited/"
train_file    = "TRAIN_Complete_dataset_without_duplicates_with_categories.csv"
test_file     = "TEST_Complete_dataset_without_duplicates_with_categories.csv"
validate_file = "VALIDATE_Complete_dataset_without_duplicates_with_categories.csv"
modeldir      = "random_forest_2/"
nestimators   = int(pow(2,10)) #Deepchem with 1024=2^10 results in 0.97/0.94
fbits         = 11             #2^fbits Bits in fingerprint. Deepchem has 2048 = 2^11
radius        = 2              #Fingerprint radius. Deehcpem has 2
 
#Create directory if not exists
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

try: 
    validset = pd.read_csv(inputfiledir + validate_file).drop(["id"],axis = 1)  
    trainset = pd.read_csv(inputfiledir + train_file).drop(["id"],axis = 1)  
    testset  = pd.read_csv(inputfiledir + test_file).drop(["id"],axis = 1)    
except: 
    print("Unable to locate valid, train and test files")
    

#Featurize all molecules
trainset = rdkfeaturization(trainset, radius, fbits)
testset  = rdkfeaturization(testset, radius, fbits)
validset = rdkfeaturization(validset, radius, fbits)

#Pool
numericize(trainset)
numericize(testset)
numericize(validset)


#https://catboost.ai/docs/concepts/python-reference_pool.html
cattrain = Pool(data = trainset.drop(["smiles", "logS","Normalized_logS","Category"], \
                             axis = 1), label = trainset["Category"])
cattest  = Pool(testset.drop(["smiles", "logS","Normalized_logS","Category"], \
                             axis = 1), label = testset["Category"])
catvalid = Pool(validset.drop(["smiles", "logS","Normalized_logS","Category"], \
                             axis = 1), label = validset["Category"])

#Run Sci-Kit learn
# specify the training parameters 
#https://catboost.ai/docs/concepts/python-reference_parameters-list.html
#scale_pos_weight for imbalanced datasets is sum_negative / sum_positive
#https://catboost.ai/docs/concepts/parameter-tuning.html
sum_positive = trainset['Category'].sum() 
sum_negative = len(trainset) - trainset['Category'].sum() 
scale_pos_weight = sum_negative / sum_positive

model = CatBoostClassifier(iterations = 100, 
                           #depth=10, 
                           #l2_leaf_reg = 10,
                           #border_count=254,
                           verbose = True,
                           use_best_model=True,
                           scale_pos_weight = scale_pos_weight,
                           eval_metric='Precision',
                           thread_count = int(cpus/2),
                           loss_function='Logloss')

# Train the model on training data
model.fit(cattrain, eval_set = cattest, plot = True)

#Pecision metrics
model.eval_metrics(cattrain,"Precision")
model.eval_metrics(cattest,"Precision")

model.save_model("Catboost_Sol")

#LOad model
"""
from_file = CatBoostClassifier()
from_file.load_model("Catboost_Sol")
"""

try:
    from notifyending import notify_ending
    notify_ending("Finished fitting random forest with catboost")
except:
    print("Random forest")

#quit()
