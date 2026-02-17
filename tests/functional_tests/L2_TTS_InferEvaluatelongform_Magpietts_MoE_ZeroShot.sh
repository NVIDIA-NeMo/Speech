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
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts_inference.py \
    --nemo_files "/home/TestData/tts/2602_MoE/moe16_sinkhorn_top1_valLoss5.0469_step2625132_epoch524.nemo" \
    --codecmodel_path "/home/TestData/tts/21fps_causal_codecmodel.nemo" \
    --datasets_json_path "examples/tts/evalset_config.json" \
    --datasets "an4_val_ci_longform_tiny" \
    --out_dir "./mplf_moe_zs_0" \
    --batch_size 6 \
    --use_cfg \
    --cfg_scale 2.5 \
    --apply_attention_prior \
    --run_evaluation \
    --clean_up_disk \
    --cer_target 0.03 \
    --ssim_target 0.50
