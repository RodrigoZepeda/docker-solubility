#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:18:35 2019

@author: rod
"""
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def rdkfeaturization(database, radius, fbits, normalize = True, useChirality = False,
                     useBondTypes = True, useFeatures = False):
    
    #Associate molecule to each smile and create fingerprint
    mols = [Chem.MolFromSmiles(x) for x in  database["smiles"]]
    
    #Get the fingerprints for the mols (same fingerprints as deepchem)
    fmols = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol = x, 
                                                   radius = int(radius), 
                                                   useChirality = useChirality, 
                                                   useBondTypes = useBondTypes,
                                                   useFeatures  = useFeatures,
                                                   nBits = int(pow(2, fbits))) for x in mols]
    string_fpints = [x.ToBitString() for x in fmols]
    
    #Add fingerprint to database
    fpints   = [np.fromstring(x,'u1') - ord('0') for x in string_fpints]
    database =  pd.concat([database.reset_index(drop=True), pd.DataFrame(fpints)], axis=1)
    
    #Normalize validation
    if normalize:
        mu = np.mean(database["logS"])
        s  = np.std(database["logS"])
        database["Normalized_logS"] = (database["logS"] - mu)/s #coincides with deepchem

    return database

def numericize(data, newlabelvals = [0,1]):
    newcol = [newlabelvals[0]]*len(data)
    for i in range(0,len(data)):
        if data["Category"].values[i] == "soluble":
            newcol[i] = newlabelvals[1]
    data.drop(["Category"], axis = 1)
    data["Category"] = newcol
    return True

def solubilize(data, labelvals = [0,1]):
    newcol = ["no soluble"]*len(data)
    for i in range(0,len(data)):
        if data["Category"].values[i] == labelvals[1]:
            newcol[i] = "soluble"
    data.drop(["Category"], axis = 1)
    data["Category"] = newcol
    return True