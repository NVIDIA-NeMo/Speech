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
rm -rf examples/multimodal/text_to_image/sd_train_results

coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/multimodal/text_to_image/stable_diffusion/sd_train.py \
    trainer.devices=1 \
    trainer.max_steps=3 \
    +trainer.val_check_interval=10 \
    trainer.limit_val_batches=2 \
    trainer.gradient_clip_val=0 \
    exp_manager.exp_dir=examples/multimodal/text_to_image/sd_train_results \
    exp_manager.create_checkpoint_callback=False \
    exp_manager.resume_if_exists=False \
    model.resume_from_checkpoint=null \
    model.precision=16 \
    model.micro_batch_size=1 \
    model.global_batch_size=1 \
    model.first_stage_key=moments \
    model.cond_stage_key=encoded \
    +model.load_vae=False \
    +model.load_unet=False \
    +model.load_encoder=False \
    model.parameterization=v \
    model.load_only_unet=False \
    model.text_embedding_dropout_rate=0.0 \
    model.inductor=True \
    model.inductor_cudagraphs=False \
    model.capture_cudagraph_iters=15 \
    +model.unet_config.num_head_channels=64 \
    +model.unet_config.use_linear_in_transformer=True \
    model.unet_config.context_dim=1024 \
    model.unet_config.use_flash_attention=null \
    model.unet_config.resblock_gn_groups=16 \
    model.unet_config.unet_precision=fp16 \
    +model.unet_config.timesteps=1000 \
    model.optim.name=megatron_fused_adam \
    +model.optim.capturable=True \
    +model.optim.master_weights=True \
    model.optim.weight_decay=0.01 \
    model.first_stage_config.from_pretrained=null \
    model.data.num_workers=16 \
    model.data.synthetic_data=True
