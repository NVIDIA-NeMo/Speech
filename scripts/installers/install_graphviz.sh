#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DOCKER=false
MAYBE_SUDO=""

GRAPHVIZ_REPO=https://gitlab.com/graphviz/graphviz.git
GRAPHVIZ_LATEST_RELEASE=9.0.0
GRAPHVIZ_PY_REPO=https://github.com/xflr6/graphviz
GRAPHVIZ_PY_LATEST_RELEASE=448d1a0  # Temporary fix until 0.20.2

if [[ $* == *--docker* ]]; then
    echo "Docker installation"
    DOCKER=true
else
    echo "Local installation"
    if [[ $(sudo -n -v 2) ]]; then
        MAYBE_SUDO="sudo"
    else
        echo "No sudo detected"
    fi
fi

{
    $MAYBE_SUDO apt-get update
    $MAYBE_SUDO apt-get remove -y graphviz
    pip uninstall -y graphviz
    if [[ $DOCKER == false ]]; then
        $MAYBE_SUDO apt-get install -y libtool libltdl-dev automake autoconf bison flex tcl \
            ghostscript libgd-dev fontconfig libcairo2-dev libpango1.0-dev libgts-dev
    fi
    git clone ${GRAPHVIZ_REPO} -b ${GRAPHVIZ_LATEST_RELEASE} && cd graphviz
    ./autogen.sh && ./configure --disable-python --disable-perl
    $MAYBE_SUDO make -j && $MAYBE_SUDO make install
    cd .. && $MAYBE_SUDO rm -rf graphviz
    pip install -v "git+${GRAPHVIZ_PY_REPO}@${GRAPHVIZ_PY_LATEST_RELEASE}#egg=graphviz"
} || { echo "graphviz installed with errors! Please check installation manually."; exit 1; }
echo "graphviz (re-) installed successfully!"
