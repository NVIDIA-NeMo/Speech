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
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 NEMO_NUMBA_MINVER=0.53 CUDA_VISIBLE_DEVICES=0 coverage run -a --data-file=/workspace/.coverage --source=/workspace/ -m pytest \
    tests/collections/asr/numba \
    tests/collections/asr/test_asr_classification_model.py \
    tests/collections/asr/test_asr_context_biasing.py \
    tests/collections/asr/test_asr_ctc_encoder_model_bpe.py \
    tests/collections/asr/test_asr_ctcencdec_model.py \
    tests/collections/asr/test_asr_datasets.py \
    tests/collections/asr/test_asr_eou.py \
    tests/collections/asr/test_asr_exportables.py \
    tests/collections/asr/test_asr_filterbankfeatures_seq_len.py \
    tests/collections/asr/test_asr_hybrid_rnnt_ctc_model_bpe.py \
    tests/collections/asr/test_asr_hybrid_rnnt_ctc_model_bpe_prompt.py \
    tests/collections/asr/test_asr_hybrid_rnnt_ctc_model_char.py \
    tests/collections/asr/test_asr_interctc_models.py \
    tests/collections/asr/test_asr_lhotse_dataset.py \
    tests/collections/asr/test_asr_lhotse_speaker_dataset.py \
    tests/collections/asr/test_asr_local_attn.py \
    tests/collections/asr/test_asr_metrics.py \
    tests/collections/asr/test_asr_modules.py \
    tests/collections/asr/test_asr_multitalker_models.py \
    -m "not pleasefixme" --with_downloads
