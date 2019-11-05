#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File that generates the complete database
"""

from __future__ import print_function
import pandas
from rdkit import Chem
from rdkit.Chem import AllChem


#Import Hou dataset
hu_solubility = pandas.read_table('raw/data_set.dat', header = None,
                                  usecols = [0,2], sep ='\s+', 
                                  names=["smiles","logS"])
hu_solubility = hu_solubility.drop([175])

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  hu_solubility["smiles"]]
except:
    print("Hu error :)")

#Import wang's datasets
wang_t1 = pandas.read_excel("raw/wang/ci800406y_si_001.xls", usecols = [1,14], skiprows=2)
wang_t1 = wang_t1.drop(range(1708,wang_t1.shape[0])) #Last rows contain info on dataset
wang_t2 = pandas.read_excel("raw/wang/ci800406y_si_002.xls", usecols = [1,14], skiprows=2)
wang_t3 = pandas.read_excel("raw/wang/ci800406y_si_003.xls", usecols = [1,13], skiprows=2)
wang_t4 = pandas.read_excel("raw/wang/ci800406y_si_004.xls", usecols = [2,13], skiprows=2)
wang_t5 = pandas.read_excel("raw/wang/ci800406y_si_005.xls", usecols = [1,14], skiprows=2)
wang_t5 = wang_t5.drop(range(119,wang_t5.shape[0])) #Last rows contain info on dataset

wang = pandas.concat([wang_t1, wang_t2, wang_t3, wang_t4, wang_t5], axis=0)

wang["smiles"] = [AllChem.MolToSmiles(AllChem.MolFromSLN(x)) for x in wang["SLN"]]
wang = wang.drop(['SLN'], axis=1)
wang.columns = ["logS","smiles"]


"""    
#Import Huuskonen dataset
#changed line 20 removing extra value
huuskonen_t1 = pandas.read_table('edited/test1_edited.smi', header=None, 
                              usecols = [3,6], sep='\s+',
                              names=["logS","smiles"])
huuskonen_t1["dataset"] = "test1"

huuskonen_t1 = huuskonen_t1.drop([119])
#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  huuskonen_t1["smiles"]]
except:
    print("Huuskonen1 error :)")
    
huuskonen_t2 = pandas.read_table('raw/Huuskonen/train.smi', header=None, 
                              usecols = [3,6], sep='\s+',
                              names=["logS","smiles"])
huuskonen_t2["dataset"] = "train"

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  huuskonen_t2["smiles"]]
except:
    print("Huuskonen1 error :)")
"""
    
#Import Delaney's dataset
delaney = pandas.read_table('raw/delaney-processed.csv', 
                              usecols = [8,9], sep=',')
delaney.columns = ["logS","smiles"]

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  delaney["smiles"]]
except:
    print("delaney error :)")
    
#Import St Andrew's dataset
standrews = pandas.read_excel("raw/dls_100.xlsx", usecols = [2,5])
standrews.columns = ["logS","smiles"]
standrews = standrews.drop(standrews.index[range(100,standrews.shape[0])]) #Last rows contain info on dataset

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  standrews["smiles"]]
except:
    print("standrews error :)")
    

#Import Mariano's new datasets
nitro = pandas.read_table("raw/NitroMariano.csv", sep=",")
nitro.columns = ["logS","smiles"]

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  nitro["smiles"]]
except:
    print("Nitro error :)")
    
phospho = pandas.read_table("raw/FosfoMariano.csv", sep=",")
phospho.columns = ["logS","smiles"]

#Verify we are able to read:
try:
    [Chem.MolFromSmiles(x) for x in  phospho["smiles"]]
except:
    print("Fosfo error :)")    
    
#Create dataset
solubility_df = pandas.concat([wang, hu_solubility, delaney, standrews, nitro, phospho], axis=0)

#solubility_df = pandas.concat([wang, hu_solubility, delaney, standrews], axis=0)
solubility_df = solubility_df.reset_index(drop=True)

solubility_df.to_csv("edited/Complete_dataset_repeated.csv", index = False)

#Delete duplicates
#TODO improve duplicate drop to have a reason to drop some not the others
solubility_df_no_duplicates = solubility_df.drop_duplicates(subset=["smiles"])
solubility_df_no_duplicates = solubility_df_no_duplicates.reset_index(drop=True)


solubility_df_no_duplicates.to_csv("edited/Complete_dataset_without_duplicates.csv", index = False)

