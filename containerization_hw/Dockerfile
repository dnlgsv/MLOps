FROM ubuntu:21.04

LABEL maintainer="danilgusev9639@gmail.com"
ENV ADMIN="Daniil Gusev"

FROM python:3.8
RUN  apt-get update \ 
  && /usr/local/bin/python -m pip install --upgrade pip \
  && apt-get install -y git \
  && apt-get install -y wget \
  && apt-get install unzip
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.63.0/source/boost_1_63_0.zip \
  && unzip boost_1_63_0.zip \
  && mv boost_1_63_0 /usr/local/bin

RUN git clone https://github.com/facebookresearch/Starspace.git
WORKDIR /Starspace
COPY . .
RUN make

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

CMD ./starspace train -trainFile data/starspace_input_file.txt -model data/modelSaveFile
