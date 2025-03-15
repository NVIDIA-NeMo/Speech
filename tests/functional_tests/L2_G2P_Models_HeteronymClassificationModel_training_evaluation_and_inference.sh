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
cd examples/tts/g2p &&
    TIME=$(date +"%Y-%m-%d-%T") && OUTPUT_DIR=output_${TIME} &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_heteronym_classification_train_and_evaluate.py \
        train_manifest=/home/TestData/g2p/manifest.json \
        validation_manifest=/home/TestData/g2p/manifest.json \
        test_manifest=/home/TestData/g2p/manifest.json \
        model.wordids=/home/TestData/g2p/wordids.tsv \
        trainer.max_epochs=1 \
        model.max_seq_length=64 \
        do_training=True \
        do_testing=True \
        exp_manager.exp_dir=${OUTPUT_DIR} \
        +exp_manager.use_datetime_version=False +exp_manager.version=test &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_heteronym_classification_inference.py \
        manifest=/home/TestData/g2p/manifest.json \
        pretrained_model=${OUTPUT_DIR}/HeteronymClassification/test/checkpoints/HeteronymClassification.nemo \
        output_manifest=preds.json
