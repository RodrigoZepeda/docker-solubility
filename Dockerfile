# Dockerfile for building image with deepchem (cpu) installed
# https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263
# To build from scratch: docker build --no-cache -t docker-solubility-v1 .

#Download conda
FROM rodrigozepeda/docker-deepchem:v1

# Labels.
#https://medium.com/@chamilad/lets-make-your-docker-image-better-than-90-of-existing-ones-8b1e5de950d
LABEL org.label-schema.schema-version="1.0"
#LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="rodrigozepeda/docker-solubility-v1"
LABEL org.label-schema.description="Machine Learning models for predicting molecular solubility."
LABEL org.label-schema.url="https://github.com/RodrigoZepeda/docker-solubility-v1"
LABEL org.label-schema.version="0.1"
LABEL org.label-schema.docker.cmd="docker run -it -v <PATH TO FILE YOU WANT TO WORK ON>:/data docker-solubility-v1"

# Set environment variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en

#Set files
COPY input_files /usr/src/input_files
COPY models /usr/src/models
COPY scripts /usr/src/scripts

#Run Python scripts
CMD ["python","/usr/src/scripts/GraphConvolutionModel.py" ]
