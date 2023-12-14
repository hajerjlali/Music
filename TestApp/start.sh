#!bin/bash
app="docker.testfront23"
docker build -t ${app} .
docker run -d -p 56738:80 \
    --name=${app} \
    -v $PWD:/app ${app}
