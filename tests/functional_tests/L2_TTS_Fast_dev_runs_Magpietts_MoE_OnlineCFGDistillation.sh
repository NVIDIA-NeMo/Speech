# Copyright (c) 2026, NVIDIA CORPORATION.
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
HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts.py \
    --config-name magpietts \
    name="MagpieTTS-MoE-OnlineCFGDistillation" \
    +mode="online_cfg_distillation_train" \
    +init_from_nemo_model="/home/TestData/tts/2602_MoE/moe16_sinkhorn_top1_valLoss5.0469_step2625132_epoch524.nemo" \
    +model.teacher_model_path="/home/TestData/tts/2602_MoE/moe16_sinkhorn_top1_valLoss5.0469_step2625132_epoch524.nemo" \
    model.codecmodel_path="/home/TestData/tts/21fps_causal_codecmodel.nemo" \
    +model.use_moe=true \
    +model.decoder.use_moe=true \
    model.prior_scaling_factor=null \
    +train_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_train_context_v1.json" \
    +train_ds_meta.an4.audio_dir="/" \
    +train_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +train_ds_meta.an4.feature_dir=null \
    +val_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_val_context_v1.json" \
    +val_ds_meta.an4.audio_dir="/" \
    +val_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +val_ds_meta.an4.feature_dir=null \
    trainer.devices="[0]" \
    max_epochs=1 \
    batch_size=4 \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch \
    +model.router_load_balancing_loss_coeff=0.0 \
    +model.router_z_loss_coeff=0.001 \
    model.decoder.d_ffn=3072 \
    +model.decoder.num_experts=16 \
    +model.decoder.top_k_experts=1 \
    +model.decoder.routing_strategy="sinkhorn" \
    +model.decoder.router_jitter_noise=0.0 \
    model.local_transformer_type="none" \
    model.encoder.is_causal=false \
    model.model_type="decoder_context_tts"
