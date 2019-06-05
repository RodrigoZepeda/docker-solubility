# -*- coding: utf-8 -*-
'''
Selects which model files to run
'''

import sys

#Models to run are inputed through docker file
models = sys.argv
flag = True #Flag for running models
del models[0] #First argument of list is always the script name
print(models)
print('Running docker')

#Choose which model to run
if len(models) > 0:
    #TODO If len(models)= 0 run all models
    if "GraphConv" in models or "all" in models:
        exec(open("GraphConvolutionModel.py").read()); #TODO: Make a cleaner approach with functions
        flag = False;
    if "Weave" in models  or "all" in models:
        exec(open("WeaveModel.py").read());
        flag = False;
    if "DAG" in models or "all" in models:
        exec(open("DAGModel.py").read());
        flag = False;
    if "MPNN" in models or "all" in models:
        exec(open("MPNNModel.py").read());
        flag = False;

if len(models) < 1 or flag:
    sys.exit('ERROR: No adecquate options for models passed to main.py \n' +
    'Please select "all" to run all models or ' +
    'specify at least one of the following:' +
    "\n > GraphConv" +
    "\n > Weave" +
    "\n > DAG" +
    "\n > MPNN" +
    "\nAnd run as:\n"
    "'docker run --rm -v ~<PATHTODIRECTORY>:/data docker-solubility-v1 MODELNAME'")

print('Process finished')
