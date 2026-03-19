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

# TODO(vbataev): fix decoding with CUDA graphs on CI for this test
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo \
    examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py \
    asr_model=/home/TestData/an4_tsasr/multitalker_parakeet_v1_tiny.nemo \
    diar_model=/home/TestData/an4_diarizer/diar_streaming_sortformer_4spk-v2-tiny.nemo \
    audio_file=/home/TestData/an4_diarizer/simulated_valid/multispeaker_session_1.wav \
    max_num_of_spks=4 \
    masked_asr=False \
    mask_preencode=False \
    single_speaker_mode=False \
    parallel_speaker_strategy=True \
    sent_break_sec=2.0 \
    cache_gating=True \
    batch_size=2 \
    att_context_size=[70,13] \
    spk_supervision=diar \
    binary_diar_preds=False \
    output_path=/tmp/mt_parakeet.seglst.json \
    log=False generate_realtime_scripts=False 