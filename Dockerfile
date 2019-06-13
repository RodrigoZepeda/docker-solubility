# Dockerfile for building image with deepchem (cpu) installed
# https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263
# To build from scratch: docker build --no-cache -t docker-solubility-v1 .

#Download conda
FROM rodrigozepeda/docker-deepchem:v1.01

# Labels.
#https://medium.com/@chamilad/lets-make-your-docker-image-better-than-90-of-existing-ones-8b1e5de950d
LABEL org.label-schema.schema-version="1.0" \
      org.label-schema.name="rodrigozepeda/docker-solubility"  \
      org.label-schema.description="Machine Learning models for predicting molecular solubility." \
      org.label-schema.url="https://github.com/RodrigoZepeda/docker-solubility" \
      org.label-schema.version="0.1" \
      org.label-schema.docker.cmd="docker run -it -v <PATH TO FILE YOU WANT TO WORK ON>:/data docker-solubility"

# Set environment variables
ENV LANG en_US.UTF-8 \
    LANGUAGE en_US:en

#Set files
#https://stackoverflow.com/questions/30256386/how-to-copy-multiple-files-in-one-layer-using-a-dockerfile
COPY models /usr/src/models
COPY scripts /usr/src/scripts

WORKDIR /usr/src/scripts

#Run Python scripts
ENTRYPOINT [ "python",  "main.py" ]
CMD []
