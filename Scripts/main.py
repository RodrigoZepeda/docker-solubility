# -*- coding: utf-8 -*-
'''
Selects which model files to run
'''

import sys

#Models to run are inputed through docker file
models = sys.argv
del models[0] #First argument of list is always the script name
print(models)

print('Running docker')

#Choose which model to run
if len(models) < 1:
    sys.exit('ERROR: No options for models passed to main.py \n' +
    'Please specify at least one of the following:' +
    "\n > GraphConvolution" +
    "\n > Model2" +
    "\n > Model3" +
    "\nAnd run as:\n"
    "docker run -it --rm -v ~<PATHTODIRECTORY>:/data docker-solubility-v1 Model1 Model2")
else:
    if "GraphConvolution" in models:
        print("GC")
    if "Model2" in models:
        print("Model2")
    if "Model3" in models:
        print("Model3")
