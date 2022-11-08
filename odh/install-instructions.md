# instruction for deploying the explainable data science notebook on jupyterhub
1. log in an admin in ODH
2. go to _Builds > ImageStream_
3. click _Create ImageStream_ button
4. switch to _YAML view_
5. paste [explainable-data-science-notebook](explainable-data-science-notebook) YAML contents
6. save the _ImageStream_
7. wait for JH image to pick the new notebook image
8. you should see an _Explainable Data Science Notebook_ when logging into JupyterHub (_Netorking > Routes > jupyterhub_)

The JH image is defined within the [s2i-lab-trustyai](https://github.com/tteofili/s2i-lab-trustyai) repo.
