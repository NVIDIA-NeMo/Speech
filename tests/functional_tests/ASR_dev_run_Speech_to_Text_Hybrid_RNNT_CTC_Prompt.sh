# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

# Functional test for Hybrid RNNT-CTC BPE Model with Prompt
# This test validates the model can train end-to-end with prompt conditioning

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe_prompt.py \
    --config-path="../conf/fastconformer/hybrid_transducer_ctc" --config-name="fastconformer_hybrid_transducer_ctc_bpe_prompt" \
    model.train_ds.manifest_filepath=/home/TestData/asr/prompt_parakeet/multilingual_train.json \
    model.validation_ds.manifest_filepath=/home/TestData/asr/prompt_parakeet/multilingual_dev.json \
    model.test_ds.manifest_filepath=/home/TestData/asr/prompt_parakeet/multilingual_dev.json \
    model.tokenizer.dir="/home/TestData/asr/prompt_parakeet/merged_universal_tokenizer/" \
    model.validation_ds.batch_size=4 \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_to_text_hybrid_rnnt_ctc_prompt_results

