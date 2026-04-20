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

# Not making this deterministic because the HuggingFace Qwen2 rotary embedding uses a CuBLAS matmul
# that is incompatible with torch.use_deterministic_algorithms(True) on CUDA >= 10.2.
# Setting CUBLAS_WORKSPACE_CONFIG is not sufficient since the error originates inside the
# transformers library (modeling_qwen2.py Qwen2RotaryEmbedding.forward).
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts_inference.py \
    --deterministic \
    --codecmodel_path /home/TestData/tts/25fps_spectral_codec_with_bandwidth_extension.nemo \
    --nemo_files /home/TestData/tts/2603_EasyMagpieTTS/EMTTS_Pretraining_Qwen_WithCrossLingual_3_5_Delay.nemo \
    --out_dir ./emp_zs_0 \
    --model_type easy_magpie \
    --batch_size 4 \
    --datasets_json_path examples/tts/evalset_config.json \
    --datasets an4_val_ci_nemotronTokenizer \
    --use_local_transformer \
    --topk 80 \
    --use_cfg \
    --cfg_scale 2.5 \
    --phoneme_input_type predicted \
    --run_evaluation \
    --disable_fcd \
    --phoneme_tokenizer_path /home/TestData/tts/2603_EasyMagpieTTS/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json \
    --temperature 0.6 \
    --clean_up_disk \
    --cer_target 0.10 \
    --ssim_target 0.70 \
    --asr_model_name /home/TestData/tts/pretrained_models/parakeet-tdt-1.1b/parakeet-tdt-1.1b.nemo \
    --eou_model_name /home/TestData/tts/pretrained_models/wav2vec2-base-960h
