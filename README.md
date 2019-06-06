# docker-solubility-v1
Docker for reproducing our solubility model.

## Install Docker
To run model please [install docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

### Install from Docker
```
docker pull rodrigozepeda/docker-solubility-v1
```

### Install from Github

To install from Github either clone or manually download project
```
 git clone https://github.com/RodrigoZepeda/docker-solubility-v1
```

Go to project directory:
```
cd docker-solubility-v1
```

Then run Docker build command
```
docker build -t docker-solubility-v1 .
```
## Run

To run you need to setup a directory with the file you want to predict
```
docker run -it -v /PATH/TO_FILE/YOU_WANT_TO_WORK_ON/:/data docker-solubility-v1 all
```

where:

* ``/PATH/TO_FILE/YOU_WANT_TO_WORK_ON/`` is substituted by path to the csv file conaining the Smiles you want to predict (see [To_predict.csv on Github](https://github.com/RodrigoZepeda/docker-solubility-v1/blob/master/predict_files/To_predict.csv) for an example).
* Included models can be specified as: 
  + ``GraphConv``
  + ``Weave``
  + ``MPNN``
  + ``DAG``
  + ``RandomForest``
  + ``KRR``
  + ``XGBoost``

As an example, assuming the files to predict are included in ``~/Dropbox/predict_files`` and you want the Graph Convolution model you can:
```
sudo docker run --rm -v ~/Dropbox/predict_files:/data docker-solubility-v1 GraphConv
```

See [manual for further instructions](https://github.com/RodrigoZepeda/docker-solubility-v1/blob/master/Manual.md)  
