# docker-solubility_v1
Docker for reproducing our solubility prediction model.

## Description

## Installation
To run the model please [install docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/). After the installation our suggestion is that you use docker to pull from [dockerhub](https://cloud.docker.com/repository/docker/rodrigozepeda/docker-solubility-v1).

### Running from Docker
```
docker run --rm -v /PATH/TO_FILE/YOU_WANT_TO_WORK_ON/:/data rodrigozepeda/docker-solubility-v1
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
sudo docker run --rm -v ~/Dropbox/predict_files:/data rodrigozepeda/docker-solubility-v1 GraphConv
```

## Debugging

To run from bash:
```
docker run --rm --entrypoint bash -it -v ~/Dropbox/Quimica/Docker/docker-solubility-v2/predict_files:/data rodrigozepeda/docker-solubility-v1
```

To run from python:
```
docker run --rm --entrypoint python -it -v ~/Dropbox/Quimica/Docker/docker-solubility-v2/predict_files:/data rodrigozepeda/docker-solubility-v1
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

Run without specifying repository:
```
docker run --rm -v /PATH/TO_FILE/YOU_WANT_TO_WORK_ON/:/data docker-solubility-v1
```
