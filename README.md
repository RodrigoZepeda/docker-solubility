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

To run interactive session:
```
docker run -it -v <PATH TO FILE YOU WANT TO WORK ON>:/data --name container1 docker-solubility-v1 /bin/bash
```

where ``<PATH TO FILE YOU WANT TO WORK ON>`` is substituted by path to the csv file conaining the Smiles you want to predict. See **MANUAL**  
