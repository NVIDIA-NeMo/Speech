# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
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
HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/easy_magpietts.py \
    --config-name easy_magpietts \
    name="EasyMagpietts-OnlineCFGDistillation-FastDev" \
    +mode="online_cfg_distillation_train" \
    +init_from_ptl_ckpt="/home/TestData/tts/2603_EasyMagpieTTS/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay_126.ckpt" \
    +model.teacher_model_path="/home/TestData/tts/2603_EasyMagpieTTS/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay_126.ckpt" \
    model.phoneme_tokenizer.tokenizer_path="/home/TestData/tts/2603_EasyMagpieTTS/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json" \
    +train_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/ipa_manifests/an4_train_context_v1_ipa.json" \
    +train_ds_meta.an4.audio_dir="/" \
    +train_ds_meta.an4.tokenizer_names="[nemotron_nano_30b]" \
    +train_ds_meta.an4.feature_dir=null \
    +val_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/ipa_manifests/an4_val_context_v1_ipa.json" \
    +val_ds_meta.an4.audio_dir="/" \
    +val_ds_meta.an4.tokenizer_names="[nemotron_nano_30b]" \
    +val_ds_meta.an4.feature_dir=null \
    max_epochs=1 \
    batch_size=2 \
    model.codecmodel_path="/home/TestData/tts/25fps_spectral_codec_with_bandwidth_extension.nemo" \
    +model.vector_quantizer._target_="nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer" \
    +model.vector_quantizer.num_groups=8 \
    +model.vector_quantizer.num_levels_per_group="[4, 4, 4, 4, 4]" \
    ++model.add_language_to_context_text=true \
    '+model.ignore_phoneme_languages=[vi,zh]' \
    '+model.training_modes=[{text_input_mode:streaming,streaming_phonemes_delay:3,streaming_speech_delay:5}]' \
    +model.run_val_inference=false \
    +model.use_multilingual_asr=false \
    +model.use_utmos=false \
    +model.lt_distillation_start_step=0 \
    +model.lt_distillation_ramp_len=0 \
    model.optim.lr=5e-6 \
    ~model.optim.sched \
    trainer.log_every_n_steps=1 \
    trainer.precision=bf16-mixed \
    trainer.gradient_clip_val=0.0 \
    trainer.devices="[0]" \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    +trainer.val_check_interval=1 \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch
