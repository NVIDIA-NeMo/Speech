# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import pytest

from nemo.collections.asr.models import ASRModel
from tests.collections.asr.decoding.utils import make_preprocessor_deterministic, preserve_decoding_cfg_and_cpu_device

CHECKPOINTS_PATH = Path("/home/TestData/asr")

@pytest.fixture(scope="package")
def _stt_en_conformer_transducer_small_raw():
    if CHECKPOINTS_PATH.exists():
        model = ASRModel.restore_from(
            str(CHECKPOINTS_PATH / "stt_en_conformer_transducer_small.nemo"), map_location="cpu"
        )
    else:
        model_name = "stt_en_conformer_transducer_small"
        model = ASRModel.from_pretrained(model_name, map_location="cpu")
    make_preprocessor_deterministic(model)
    return model


@pytest.fixture(scope="package")
def _stt_en_fastconformer_transducer_large_raw():
    if CHECKPOINTS_PATH.exists():
        model = ASRModel.restore_from(
            str(CHECKPOINTS_PATH / "stt_en_fastconformer_transducer_large.nemo"), map_location="cpu"
        )
    else:
        model_name = "stt_en_fastconformer_transducer_large"
        model = ASRModel.from_pretrained(model_name, map_location="cpu")
    make_preprocessor_deterministic(model)
    return model


@pytest.fixture(scope="package")
def _stt_en_fastconformer_tdt_large_raw():
    if CHECKPOINTS_PATH.exists():
        model = ASRModel.restore_from(
            str(CHECKPOINTS_PATH / "stt_en_fastconformer_tdt_large.nemo"), map_location="cpu"
        )
    else:
        model_name = "nvidia/stt_en_fastconformer_tdt_large"
        model = ASRModel.from_pretrained(model_name, map_location="cpu")
    make_preprocessor_deterministic(model)
    return model


@pytest.fixture(scope="package")
def _canary_180m_flash_raw():
    model_name = "nvidia/canary-180m-flash"
    model = ASRModel.from_pretrained(model_name, map_location="cpu")
    make_preprocessor_deterministic(model)
    return model


@pytest.fixture
def stt_en_conformer_transducer_small(_stt_en_conformer_transducer_small_raw):
    """Function-level fixture for model. Guarantees to preserve decoding config and device"""
    model = _stt_en_conformer_transducer_small_raw
    with preserve_decoding_cfg_and_cpu_device(model):
        yield model


@pytest.fixture
def stt_en_fastconformer_transducer_large(_stt_en_fastconformer_transducer_large_raw):
    """Function-level fixture for model. Guarantees to preserve decoding config and device"""
    model = _stt_en_fastconformer_transducer_large_raw
    with preserve_decoding_cfg_and_cpu_device(model):
        yield model


@pytest.fixture
def stt_en_fastconformer_tdt_large(_stt_en_fastconformer_tdt_large_raw):
    """Function-level fixture for model. Guarantees to preserve decoding config and device"""
    model = _stt_en_fastconformer_tdt_large_raw
    with preserve_decoding_cfg_and_cpu_device(model):
        yield model


@pytest.fixture
def canary_180m_flash(_canary_180m_flash_raw):
    """Function-level fixture for model. Guarantees to preserve decoding config and device"""
    model = _canary_180m_flash_raw
    with preserve_decoding_cfg_and_cpu_device(model):
        yield model
