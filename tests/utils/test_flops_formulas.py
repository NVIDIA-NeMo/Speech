# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nemo.utils.flops_formulas import FLOPSConfig, bert, gpt3, llama2, llama3, mixtral, nemotron, transformer
from nemo.utils.hyena_flops_formulas import hyena


@pytest.fixture
def flops_config():
    return FLOPSConfig(
        gbs=1,
        enc_seq_len=128,
        hs=768,
        layers=12,
        ffn_hs=3072,
        attention_heads=12,
        moe_router_topk=2,
        query_groups=12,
        vocab_size=50257,
        model_pattern="SDH*",
    )


def test_gpt3(flops_config):
    expected_flops = 96334774272
    assert gpt3(flops_config) == expected_flops


def test_llama2(flops_config):
    expected_flops = 106753425408.0
    assert llama2(flops_config) == expected_flops


def test_llama3(flops_config):
    expected_flops = 163527524352.0
    assert llama3(flops_config) == expected_flops


def test_nemotron(flops_config):
    expected_flops = 217130729472.0
    assert nemotron(flops_config) == expected_flops


def test_mixtral(flops_config):
    expected_flops = 171983241216.0
    assert mixtral(flops_config) == expected_flops


def test_bert(flops_config):
    expected_flops = 84146651135.99998
    assert bert(flops_config) == expected_flops


def test_hyena(flops_config):
    expected_flops = 116883062784.0
    assert hyena(flops_config) == expected_flops


def test_transformer(flops_config):
    expected_flops = 118427811840.0
    assert transformer(flops_config) == expected_flops


def test_transformer_no_moe(flops_config):
    flops_config.moe_router_topk = 0
    expected_flops = 96684539904.0
    assert transformer(flops_config) == expected_flops
