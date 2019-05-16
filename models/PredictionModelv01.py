'''
Runs
'''
import time
import deepchem as dc
import pandas as pd
from deepchem.models import GraphConvModel
from deepchem.utils.save import load_from_disk
from rdkit import Chem

#Featurize molecules
featurizer = dc.feat.ConvMolFeaturizer()
model = GraphConvModel.load_from_dir("/usr/src/models/graph_convolution")

#Read dataset to predict
newsmiles = pd.read_csv('/data/To_predict.csv')
mols = [Chem.MolFromSmiles(s) for s in newsmiles.loc[:,"Smiles"]]

#Featurize data
x = featurizer.featurize(mols)

#Predict molecules
predicted_solubility = model.predict_on_batch(x)

#Convert to dataframe
mydf = pd.concat([newsmiles, pd.DataFrame(predicted_solubility)], axis = 1)
mydf.columns = ["Smile","Predicted Solubility"]
mydf.to_csv('/data/PredictedGraphConvolution ' + time.ctime() + '.csv', index=False)
