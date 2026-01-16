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
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts_inference.py \
    --codecmodel_path /home/TestData/tts/AudioCodec_21Hz_no_eliz_without_wavlm_disc.nemo \
    --datasets_json_path examples/tts/evalset_config.json \
    --datasets an4_val_ci \
    --out_dir ./mplf_zs_0 \
    --batch_size 4 \
    --longform_mode always \
    --longform_max_decoder_steps 50000 \
    --use_cfg \
    --cfg_scale 2.5 \
    --num_repeats 1 \
    --temperature 0.6 \
    --hparams_files /home/TestData/tts/2506_ZeroShot/lrhm_short_yt_prioralways_alignement_0.002_priorscale_0.1.yaml \
    --checkpoint_files /home/TestData/tts/2506_ZeroShot/dpo-T5TTS--val_loss=0.4513-epoch=3.ckpt \
    --legacy_codebooks \
    --legacy_text_conditioning \
    --apply_attention_prior \
    --run_evaluation \
    --clean_up_disk \
    --cer_target 0.25 \
    --ssim_target 0.7
