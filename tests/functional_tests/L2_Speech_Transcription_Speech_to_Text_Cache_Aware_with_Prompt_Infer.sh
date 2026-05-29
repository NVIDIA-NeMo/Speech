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

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
    model_path="/home/TestData/asr/stt_ml_fastconformer_rnnt_xl_streaming_prompt.nemo" \
    dataset_manifest="/home/TestData/asr/prompt_parakeet/multilingual_dev_target_lang_field.json" \
    output_path="/tmp/stt_cache_aware_streaming_prompt_test_res" \
    target_lang=auto \
    att_context_size="[56,0]" \
    decoder_type=rnnt \
    pad_and_drop_preencoded=true \
    batch_size=8 \
    strip_lang_tags=false
