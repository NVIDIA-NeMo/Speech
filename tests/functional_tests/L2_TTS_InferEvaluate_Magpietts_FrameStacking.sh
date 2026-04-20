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

# Tests a 4x-stacked model with local transformer inference.

CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts_inference.py \
    --deterministic \
    --codecmodel_path /home/TestData/tts/21fps_causal_codecmodel.nemo \
    --datasets_json_path examples/tts/evalset_config.json \
    --datasets an4_val_ci \
    --out_dir ./mp_fs_4x_0 \
    --batch_size 4 \
    --use_cfg \
    --cfg_scale 2.5 \
    --num_repeats 1 \
    --temperature 0.6 \
    --hparams_files /home/TestData/tts/2602_FrameStacking4x/hparams.yaml \
    --checkpoint_files /home/TestData/tts/2602_FrameStacking4x/frame-stacking-4x-english-nanocodec.ckpt \
    --run_evaluation \
    --clean_up_disk \
    --cer_target 0.07 \
    --ssim_target 0.66 \
    --use_local_transformer \
    --asr_model_name /home/TestData/tts/pretrained_models/parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo \
    --eou_model_name /home/TestData/tts/pretrained_models/wav2vec2-base-960h
