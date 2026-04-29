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
CUDA_VISIBLE_DEVICES="" NEMO_NUMBA_MINVER=0.53 coverage run -a --data-file=/workspace/.coverage --source=/workspace/ -m pytest \
    tests/collections/asr/inference \
    tests/collections/asr/k2 \
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
    tests/collections/asr/test_asr_metrics.py \
    tests/collections/asr/test_asr_modules.py \
    tests/collections/asr/test_asr_multitalker_models.py \
    tests/collections/asr/test_asr_parts_submodules_batchnorm.py \
    tests/collections/asr/test_asr_regression_model.py \
    tests/collections/asr/test_asr_rnnt_encdec_model.py \
    tests/collections/asr/test_asr_rnnt_encoder_model_bpe.py \
    tests/collections/asr/test_asr_samplers.py \
    tests/collections/asr/test_asr_subsampling.py \
    tests/collections/asr/test_boosting_tree.py \
    tests/collections/asr/test_conformer_encoder.py \
    tests/collections/asr/test_custom_tokenizer.py \
    tests/collections/asr/test_jasper_block.py \
    tests/collections/asr/test_label_datasets.py \
    tests/collections/asr/test_ngram_lm.py \
    tests/collections/asr/test_padding_and_batch_size_invariance.py \
    tests/collections/asr/test_preprocessing_segment.py \
    tests/collections/asr/test_ssl_models.py \
    tests/collections/asr/test_text_to_text_dataset.py \
    tests/collections/asr/utils \
    -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat
