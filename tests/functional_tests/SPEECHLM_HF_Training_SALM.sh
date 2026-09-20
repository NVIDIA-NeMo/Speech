# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES.
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

# Run training
torchrun --nproc-per-node 1 --no-python \
  coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/speechlm2/salm_train.py \
      model.pretrained_llm=/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1 \
      model.pretrained_asr=/home/TestData/speechlm/pretrained_models/canary-1b-flash.nemo \
      data.train_ds.input_cfg.0.cuts_path=/home/TestData/speechlm/lhotse/libri/librispeech_cuts_lower_train-clean-5.jsonl.gz \
      data.validation_ds.datasets.val_set_0.input_cfg.0.cuts_path=/home/TestData/speechlm/lhotse/libri/librispeech_cuts_lower_dev-clean-2.jsonl.gz \
      trainer.devices=1 \
      trainer.max_steps=10

# Convert to HF format
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  examples/speechlm2/to_hf.py \
  class_path=nemo.collections.speechlm2.models.SALM \
  ckpt_path=salm_results/checkpoints/step\\=10-last.ckpt \
  ckpt_config=salm_results/exp_config.yaml \
  output_dir=test_salm_hf_model

# Run generation
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  examples/speechlm2/salm_generate.py \
  pretrained_name=test_salm_hf_model \
  inputs=/home/TestData/speechlm/lhotse/libri/librispeech_cuts_lower_dev-clean-2-first10.jsonl.gz \
  batch_size=4 \
  output_manifest=generations.jsonl
head generations.jsonl

# Run generation + WER eval
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
  examples/speechlm2/salm_eval.py \
  pretrained_name=test_salm_hf_model \
  inputs=/home/TestData/speechlm/lhotse/libri/librispeech_cuts_lower_dev-clean-2-first10.jsonl.gz \
  batch_size=4 \
  output_manifest=generations_wer.jsonl
head generations_wer.jsonl
