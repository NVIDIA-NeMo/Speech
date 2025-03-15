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
    TIME=$(date +"%Y-%m-%d-%T") && OUTPUT_DIR_CONFORMER=output_ctc_${TIME} &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_train_and_evaluate.py \
        train_manifest=/home/TestData/g2p/g2p.json \
        validation_manifest=/home/TestData/g2p/g2p.json \
        model.test_ds.manifest_filepath=/home/TestData/g2p/g2p.json \
        model.tokenizer.dir=/home/TestData/g2p/tokenizer_spe_unigram_v512 \
        trainer.max_epochs=1 \
        model.max_source_len=64 \
        trainer.devices=1 \
        do_training=True \
        do_testing=True \
        exp_manager.exp_dir=${OUTPUT_DIR_CONFORMER} \
        +exp_manager.use_datetime_version=False +exp_manager.version=test \
        --config-name=g2p_conformer_ctc &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo g2p_inference.py \
        pretrained_model=${OUTPUT_DIR_CONFORMER}/G2P-Conformer-CTC/test/checkpoints/G2P-Conformer-CTC.nemo \
        manifest_filepath=/home/TestData/g2p/g2p.json \
        phoneme_field=text
