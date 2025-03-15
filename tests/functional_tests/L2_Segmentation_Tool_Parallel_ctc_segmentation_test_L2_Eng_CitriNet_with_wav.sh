# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
cd tools/ctc_segmentation &&
    TIME=$(date +"%Y-%m-%d-%T") &&
    /bin/bash run_segmentation.sh \
        --MODEL_NAME_OR_PATH="stt_en_citrinet_512_gamma_0_25" \
        --DATA_DIR=/home/TestData/ctc_segmentation/eng \
        --OUTPUT_DIR=/tmp/ctc_seg_en/output${TIME} \
        --LANGUAGE=en \
        --USE_NEMO_NORMALIZATION="TRUE" &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo /home/TestData/ctc_segmentation/verify_alignment.py \
        -r /home/TestData/ctc_segmentation/eng/eng_valid_segments_1.7.txt \
        -g /tmp/ctc_seg_en/output${TIME}/verified_segments/nv_test_segments.txt
