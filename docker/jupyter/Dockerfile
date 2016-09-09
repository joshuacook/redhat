FROM jupyter/all-spark-notebook

USER root

RUN apt-get update && apt-get install -y libpq-dev 
COPY requirements.txt /home/jovyan/work 
RUN pip install --no-cache-dir -r requirements.txt

USER jovyan
