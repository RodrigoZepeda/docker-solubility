# -*- coding: utf-8 -*-
'''
Selects which model files to run
'''
import time
import predictchem
import deepchem
from deepchem.models import GraphConvModel
from deepchem.models import WeaveModel
from deepchem.models import MPNNModel
import sys

#Docker's working directories
model_dir = "/usr/src/models/"
data_dir = "/data/"
newdir = "model " + time.ctime()

#Models to run are inputed through docker file
models = sys.argv
del models[0] #First argument of list is always the script name
flag_predicted = True   #Flag for running models
print('Running models')

#Predict between different models
if len(models) == 0 or "GraphConv" in models:
    print("Evaluating Graph Convolution Model")
    predictchem.predict_csv_from_model(
        featurizer = deepchem.feat.ConvMolFeaturizer(),
        transformers = 2,
        modelname = GraphConvModel,
        model_file = model_dir + "graph_convolution",
        dataset_file = '/data/To_predict.csv',
        fname = 'PredictedGraphConvolution.csv',
        parentdir = data_dir,
        newdir = newdir)
    flag_predicted = False;

if len(models) == 0 or "Weave" in models:
    print("Evaluating Weave Neural Network Model")
    predictchem.predict_csv_from_model(
        featurizer = deepchem.feat.WeaveFeaturizer(),
        transformers = 2,
        modelname = WeaveModel,
        model_file = model_dir + "weave_model",
        dataset_file = data_dir + 'To_predict.csv',
        fname = 'PredictedWeave.csv',
        parentdir = data_dir,
        newdir = newdir)
    flag_predicted = False;

if len(models) == 0 or "DAG" in models:
    print("Evaluating DAG Model")
    #exec(open("DAGModel.py").read());
    #flag_predicted = False;

if len(models) == 0 or "MPNN" in models:
    print("Evaluating Message Passing Neural Network Model")
    predictchem.predict_csv_from_model(
        featurizer = deepchem.feat.WeaveFeaturizer(),
        transformers = 2,
        modelname = MPNNModel,
        model_file = model_dir + "mpnn_model",
        dataset_file = data_dir + 'To_predict.csv',
        fname = 'PredictedMPNN.csv',
        parentdir = data_dir,
        newdir = newdir)
    flag_predicted = False;

if flag_predicted:
    sys.exit('ERROR: No adecquate options for models passed to main.py \n' +
    'Please leave empty to run all models:\n' +
    "'docker run --rm -v ~<PATHTODIRECTORY>:/data docker-solubility-v1'" +
    '\nor specify at least one of the following models:' +
    "\n > GraphConv" +
    "\n > Weave" +
    "\n > DAG" +
    "\n > MPNN" +
    "\nand run as:\n"
    "'docker run --rm -v ~<PATHTODIRECTORY>:/data docker-solubility-v1 MODELNAME'")

print('Process finished')
