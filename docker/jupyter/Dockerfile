FROM jupyter/scipy-notebook

USER root

RUN apt-get update && apt-get install -y libpq-dev 
COPY requirements.txt /home/jovyan/work 
RUN pip install --no-cache-dir -r requirements.txt
RUN conda install -c blaze blaze

USER jovyan
