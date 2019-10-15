#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:43:54 2019

@author: rod
"""


from __future__ import print_function
import time
start = time.time()
import os

import pandas as pd
import numpy as np
from math import pow

from catboost import Pool, CatBoostClassifier

import multiprocessing
from rdkitfeaturize import * 

fname = "To_predict.csv"

#Load model
model = CatBoostClassifier()
model.load_model("Catboost_Sol")

#Load file to predtc
try: 
    readfile = pd.read_csv(fname)  
except: 
    print("Unable to locate valid, train and test files")
    
#Add characteristics
fbits         = 11  #2^fbits Bits in fingerprint. Deepchem has 2048 = 2^11
radius        = 2   #Fingerprint radius. Deehcpem has 2


#Featurize all molecules
readfile.columns = ["smiles"]
readfile   = readfile.drop(readfile.index[[141,142,317]]).reset_index(drop=True)
predictset = rdkfeaturization(readfile, radius, fbits, normalize = False)



#Pool
predictions = model.predict(predictset.drop(["smiles"], axis = 1))
predictions = pd.DataFrame(predictions, columns = ["Category"])
solubilize(predictions)

probabs     = model.predict_proba(predictset.drop(["smiles"], axis = 1))[:,1]
probabs     = pd.DataFrame(probabs, columns = ["Probability"])

pd.concat([predictions, probabs, readfile], axis = 1).to_csv("Predictions.csv")

end = time.time()
print("Predicted in:")
print(end - start)