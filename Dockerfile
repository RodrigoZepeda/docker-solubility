#Download conda
FROM continuumio/miniconda3

#Update conda
RUN conda update conda
RUN conda update --all

#Create environment in 3.6.6 (same as my ubuntu)
RUN yes "yes" | conda create -n env python=3.6.6

#Set docker to open in environment each time
#see https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

#To work in this environment we need to source it this time
RUN yes "yes" | conda install -n env -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.1.0
