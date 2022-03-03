FROM jupyter/minimal-notebook:latest

USER root

RUN apt-get update && apt-get install -qq --yes --no-install-recommends \
    curl


# install JVM
RUN apt-get update && apt-get install -qq --yes --no-install-recommends \
    default-jdk

# install TrustyAI python package...
RUN pip install trustyai