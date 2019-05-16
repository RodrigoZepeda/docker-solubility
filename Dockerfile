# Dockerfile for building image with deepchem (cpu) installed
# https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263
# To build from scratch: docker build --no-cache -t docker-solubility-v1 .

#Download conda
FROM continuumio/miniconda3

#Set version
LABEL version="0.1"
LABEL maintainer="Rodrigo Zepeda <rzepeda17@gmail.com>"

# Set environment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en

#Set working directory
COPY input_files /usr/src/input_files
COPY models /usr/src/models
COPY scripts /usr/src/scripts

#Update conda
RUN conda update conda
RUN conda update --all

#Create environment in 3.6.6 (same as my ubuntu)
RUN yes "yes" | conda create -n env python=3.6.6

#Set docker to open in environment each time
#see https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

#Specify environment installation and install deepchem (and dependencies)
RUN yes "yes" | conda install -n env -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.1.0

#Clean conda after installation
RUN yes "yes" | conda clean --all

#Run Python scripts
CMD ["python","/usr/src/scripts/PredictionModelv01.py" ]
