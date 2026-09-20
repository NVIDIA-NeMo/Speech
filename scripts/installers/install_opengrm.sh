#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# This script install OpenFST and Ngram tools from OpenGRM library
# Optionally, you can specify a path where to install it as a first positional argument: scripts/installers/install_opengrm.sh /path/to/install/openfst .
# Alternatively, in the Linux Debian you can use: sudo apt install libngram-tools

DECODERS_PATH=/workspace/nemo/decoders  # Path to decoders folder: /workspace/nemo/decoders if you use NeMo/Dockerfile
if [ "$#" -eq 1 ]
then
  DECODERS_PATH=$1
fi
cd $DECODERS_PATH

# Install OpenGrm OpenFST
wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz && tar xvzf openfst-1.8.2.tar.gz && cd openfst-1.8.2 && ./configure --enable-grm && make -j4 && make -j4 install && cd ..

# Install OpenGrm Ngram
OPENFSTPREFIX=$DECODERS_PATH/openfst-1.8.2/src &&  wget https://www.opengrm.org/twiki/pub/GRM/NGramDownload/ngram-1.3.14.tar.gz && tar xvzf ngram-1.3.14.tar.gz && cd ngram-1.3.14 && LDFLAGS="-L${OPENFSTPREFIX}/lib" CXXFLAGS="-I${OPENFSTPREFIX}/include" ./configure --prefix ${OPENFSTPREFIX} && make -j4 && make -j4 install && cd ..
