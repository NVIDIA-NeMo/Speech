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
    tests/collections/speechlm2/test_audio_placeholders.py \
    tests/collections/speechlm2/test_datamodule.py \
    tests/collections/speechlm2/test_datamodule_parallel.py \
    tests/collections/speechlm2/test_duplex_stt_dataset.py \
    tests/collections/speechlm2/test_early_interruption.py \
    tests/collections/speechlm2/test_force_align.py \
    tests/collections/speechlm2/test_freezing_params.py \
    tests/collections/speechlm2/test_init_from_checkpoint.py \
    tests/collections/speechlm2/test_label_prep.py \
    tests/collections/speechlm2/test_metrics.py \
    tests/collections/speechlm2/test_nemotron_voicechat.py \
    tests/collections/speechlm2/test_parallel.py \
    tests/collections/speechlm2/test_role_swap.py \
    tests/collections/speechlm2/test_salm_asr_decoder_multilayerproj.py \
    tests/collections/speechlm2/test_salm_asr_decoder_qformer.py \
    tests/collections/speechlm2/test_salm_automodel.py \
    tests/collections/speechlm2/test_salm_automodel_lora.py \
    tests/collections/speechlm2/test_salm_lora.py \
    tests/collections/speechlm2/test_salm.py \
    tests/collections/speechlm2/test_to_hf.py \
    -m "not pleasefixme" --cpu --with_downloads --relax_numba_compat
