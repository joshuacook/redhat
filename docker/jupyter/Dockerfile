FROM jupyter/all-spark-notebook

USER root

COPY requirements.txt /home/jovyan/work 
RUN pip install --no-cache-dir -r requirements.txt

USER jovyan
