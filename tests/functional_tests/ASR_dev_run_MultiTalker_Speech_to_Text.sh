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
#!/bin/bash
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_transducer/speech_to_text_mt_rnnt_bpe.py \
    --config-path="../conf/conformer" --config-name="conformer_transducer_bpe" \
    ++model.train_ds.input_cfg=/home/TestData/an4_tsasr/simulated_train/msasr_train_tiny.yaml \
    model.train_ds.manifest_filepath=null \
    ++model.train_ds.use_lhotse=true \
    ++model.validation_ds.input_cfg=/home/TestData/an4_tsasr/simulated_valid/msasr_valid_tiny.yaml \
    model.validation_ds.manifest_filepath=null \
    ++model.validation_ds.use_lhotse=true \
    ++model.test_ds.manifest_filepath=null \
    model.tokenizer.dir=/home/TestData/an4_tsasr/tokenizer_bpe_asr_phase1_en_v1024_beep \
    model.tokenizer.type=bpe \
    ++model.spk_kernel_type="ff" \
    ++model.spk_kernel_layers=[0] \
    ++model.add_bg_spk_kernel=true \
    trainer.devices=1 \
    trainer.accelerator="gpu" \
    +trainer.fast_dev_run=True \
    exp_manager.exp_dir=/tmp/speech_to_text_results
