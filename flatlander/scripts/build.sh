#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../

docker build -t fl:latest -f ./Dockerfile.gpu --build-arg NB_USER=$USER .