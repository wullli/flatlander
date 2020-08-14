#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../

aicrowd-repo2docker \
        --no-run \
        --user-id 1007 \
        --user-name tcwullsc \
        --image-name flatland-docker \
        --debug \
        .




1