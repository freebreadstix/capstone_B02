FROM ucsdets/scipy-ml-notebook:2021.2-stable

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    pip install gensim