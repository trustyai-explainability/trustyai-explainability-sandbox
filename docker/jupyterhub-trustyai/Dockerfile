FROM jupyter/minimal-notebook:latest

USER root

RUN apt-get update && apt-get install -qq --yes --no-install-recommends \
    curl


# install
RUN apt-get update && apt-get install -qq --yes --no-install-recommends \
    default-jdk \
    maven

# install TrustyAI python package...
RUN pip install trustyai