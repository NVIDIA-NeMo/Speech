# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
dataset_meta_info = {
    'riva_hard_digits': {
        'manifest_path': '/Data/evaluation_manifests/hard-digits-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_letters': {
        'manifest_path': '/Data/evaluation_manifests/hard-letters-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_money': {
        'manifest_path': '/Data/evaluation_manifests/hard-money-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'riva_hard_short': {
        'manifest_path': '/Data/evaluation_manifests/hard-short-path-corrected.ndjson',
        'audio_dir': '/Data/RIVA-TTS',
        'feature_dir': '/Data/RIVA-TTS',
    },
    'vctk': {
        'manifest_path': '/Data/evaluation_manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths_silence_trimmed.json',
        'audio_dir': '/Data/VCTK-Corpus-0.92',
        'feature_dir': '/Data/VCTK-Corpus-0.92',
    },
    'libritts_seen': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_seen_evalset_from_testclean_v2.json',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    'libritts_test_clean': {
        'manifest_path': '/Data/evaluation_manifests/LibriTTS_test_clean_withContextAudioPaths.jsonl',
        'audio_dir': '/Data/LibriTTS',
        'feature_dir': '/Data/LibriTTS',
    },
    'an4_val_ci': {
        'manifest_path': '/home/TestData/an4_dataset/an4_val_context_v1.json',
        'audio_dir': '/',
        'feature_dir': None,
    },
    'local_longer_1': {
        'manifest_path': '/workspace/NeMo/long_manifests/filtered_test_longer_1_magpie_gemma2_audiowav.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'local_longer_4': {
        'manifest_path': '/workspace/NeMo/long_manifests/filtered_test_longer_4_magpie_gemma2_normalized_audiowav.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'local_longer_100': {
        'manifest_path': '/workspace/NeMo/long_manifests/test_long_100_claude4-5sonnet_normalized.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'local_longer_10': {
        'manifest_path': '/workspace/NeMo/long_manifests/test_long_10_claude4-5sonnet_emma_normalized.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'carlos': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_carlos_regular.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'rubby': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_rubby_regular.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'lindy': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_lindy_other.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'rodney': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_rodney_other.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'megan': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_megan_additional.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'samy': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_samy_neutral.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'virginie': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_virginie_regular.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'houzhen': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_houzhen_regular.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'siwei': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_siwei_regular.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'emma': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_emma_additional.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'sean': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_sean_additional.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
    'tom': {
        'manifest_path': '/workspace/NeMo/manifest_11_3_rel/en_tom_additional.json',
        'audio_dir': '/',
        'feature_dir': '/',
    },
}

