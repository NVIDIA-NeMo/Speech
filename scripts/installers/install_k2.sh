#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

K2_REPO=https://github.com/k2-fsa/k2
LATEST_RELEASE=5735fa7 # fix for PyTorch 2.4.0
# uncomment the following line after the next k2 version is released (>1.24.4)
#LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
#    ls-remote --exit-code --refs --sort='version:refname' --tags ${K2_REPO} '*.*' \
#    | tail --lines=1 \
#    | cut -d '/' -f 3)
# "cut --delimiter '/' --fields 3" doesn't work on macOS, use "-d ... -f ..." instead

K2_MAKE_ARGS="-j" pip install -v "git+${K2_REPO}@${LATEST_RELEASE}#egg=k2" || { echo "k2 could not be installed!"; exit 1; }
python3 -m k2.version > /dev/null || { echo "k2 installed with errors! Please check installation manually."; exit 1; }
echo "k2 installed successfully!"
