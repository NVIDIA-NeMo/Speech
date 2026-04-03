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
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/easy_magpietts.py \
    --config-name easy_magpietts \
    name="EasyMagpieTTS-OnlinePO-FastDev" \
    +mode="onlinepo_train" \
    +init_from_ptl_ckpt="/home/TestData/tts/2603_EasyMagpieTTS/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay_126.ckpt" \
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
    +model.reference_free=true \
    +model.loss_type=grpo \
    +model.scale_rewards=true \
    +model.grpo_beta=0.0 \
    ++model.normalize_whisper_transcript=true \
    ++model.reward_asr_model=whisper \
    ++model.speaker_verification_model_name=titanet_large \
    ++model.use_pesq=false \
    +model.n_generations_per_item=2 \
    +model.batch_size_for_chunked_tf=2 \
    +model.max_decoder_steps=300 \
    +model.min_valid_codes_len=4 \
    +model.max_valid_codes_len=490 \
    ++model.aux_phoneme_loss_weight=0.1 \
    ++model.best_cer_threshold=1.0 \
    ++model.worst_cer_threshold=1.0 \
    +model.inference_cfg_prob=0.5 \
    +model.inference_cfg_scale=2.5 \
    +model.gt_phoneme_input_prob=1.0 \
    +model.inference_temperature=0.7 \
    +model.inference_topk=80 \
    +model.inference_phoneme_sampling_method=argmax \
    +model.use_local_transformer_prob=1.0 \
    +model.cer_reward_weight=0.5 \
    +model.ssim_reward_weight=0.5 \
    +model.use_utmos=false \
    +model.utmos_reward_weight=0.0 \
    ++model.pesq_reward_weight=0.0 \
    ++model.val_n_generations_per_item=1 \
    model.optim.lr=5e-6 \
    ~model.optim.sched \
    trainer.log_every_n_steps=1 \
    trainer.precision=32 \
    trainer.gradient_clip_val=0.0 \
    trainer.devices="[0]" \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    +trainer.val_check_interval=1 \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch
