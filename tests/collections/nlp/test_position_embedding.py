# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import random

import pytest
import torch

from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType
from nemo.collections.nlp.modules.common.megatron.position_embedding import (
    ALiBiRelativePositionEmbedding,
    KERPLERelativePositionEmbedding,
    RotaryEmbedding,
    SandwichRelativePositionEmbedding,
    T5RelativePositionEmbedding,
    XPOSPositionEmbedding,
)
from nemo.collections.nlp.modules.common.megatron.position_embedding.rotary_position_embedding import (
    apply_rotary_pos_emb,
)
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal


@pytest.fixture()
def cfg():
    cfg = {
        'max_seq_len': 8,
        'num_attention_heads': 2,
        'layer_type': LayerType.encoder,
        'hidden_size': 4,
        'rpe_init_method_std': 0.02,
        'rpe_num_buckets': 6,
        'rpe_max_distance': 16,
    }
    return cfg


@pytest.mark.unit
def test_alibi(cfg):
    # non-causal
    PE_nc = ALiBiRelativePositionEmbedding(
        bidirectional=True,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
    )

    # causal
    PE_c = ALiBiRelativePositionEmbedding(
        bidirectional=False,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
    )

    q_len = k_len = random.randint(1, cfg['max_seq_len'] * 2)

    bias_nc = PE_nc(q_len, k_len)
    assert bias_nc.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert torch.equal(bias_nc, bias_nc.transpose(2, 3))

    bias_c = PE_c(q_len, k_len)
    assert bias_c.shape == (1, cfg['num_attention_heads'], 1, k_len)
    assert torch.equal(bias_c, bias_nc[:, :, -1:, :])


@pytest.mark.unit
def test_sandwich(cfg):
    # non-causal
    PE_nc = SandwichRelativePositionEmbedding(
        bidirectional=True,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
        hidden_size=cfg['hidden_size'],
    )

    # causal
    PE_c = SandwichRelativePositionEmbedding(
        bidirectional=False,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
        hidden_size=cfg['hidden_size'],
    )

    q_len = k_len = random.randint(1, cfg['max_seq_len'] * 2)

    bias_nc = PE_nc(q_len, k_len)
    assert bias_nc.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert torch.equal(bias_nc, bias_nc.transpose(2, 3))

    bias_c = PE_c(q_len, k_len)
    assert bias_c.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert torch.all(torch.triu(bias_c, diagonal=0) == 0)


@pytest.mark.unit
def test_kerple(cfg):
    # non-causal
    PE_nc = KERPLERelativePositionEmbedding(
        bidirectional=True,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
    )

    # causal
    PE_c = KERPLERelativePositionEmbedding(
        bidirectional=False,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        max_seq_len=cfg['max_seq_len'],
    )

    q_len = k_len = random.randint(1, cfg['max_seq_len'] * 2)

    bias_nc = PE_nc(q_len, k_len)
    assert bias_nc.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert torch.equal(bias_nc, bias_nc.transpose(2, 3))

    bias_c = PE_c(q_len, k_len)
    assert bias_c.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert torch.all(torch.triu(bias_c, diagonal=0) == 0)


@pytest.mark.unit
def test_t5relative(cfg):
    # non-causal
    PE_nc = T5RelativePositionEmbedding(
        bidirectional=True,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        init_method=init_method_normal(cfg['rpe_init_method_std']),
        relative_position_num_buckets=cfg['rpe_num_buckets'],
        relative_position_max_distance=cfg['rpe_max_distance'],
    )

    # causal
    PE_c = T5RelativePositionEmbedding(
        bidirectional=False,
        num_attention_heads=cfg['num_attention_heads'],
        layer_type=cfg['layer_type'],
        init_method=init_method_normal(cfg['rpe_init_method_std']),
        relative_position_num_buckets=cfg['rpe_num_buckets'],
        relative_position_max_distance=cfg['rpe_max_distance'],
    )

    q_len = k_len = random.randint(1, cfg['max_seq_len'] * 2)

    bias_nc = PE_nc(q_len, k_len)
    assert bias_nc.shape == (1, cfg['num_attention_heads'], q_len, k_len)

    bias_c = PE_c(q_len, k_len)
    assert bias_c.shape == (1, cfg['num_attention_heads'], q_len, k_len)
    assert (
        len(torch.triu(bias_c, diagonal=0).unique()) == cfg['num_attention_heads'] + 1
        if q_len > 1
        else cfg['num_attention_heads']
    )


@pytest.mark.unit
def test_rotary(cfg):
    PE = RotaryEmbedding(dim=cfg['hidden_size'])
    rotary_embedding = PE(cfg['max_seq_len'])

    x = torch.rand(cfg['max_seq_len'], 1, cfg['num_attention_heads'], cfg['hidden_size'])
    x_rotary = apply_rotary_pos_emb(x, rotary_embedding)
    assert x_rotary.shape == x.shape

    hd = cfg['hidden_size'] // 2
    x_rotary_test = torch.cat(
        (
            x[..., :hd] * rotary_embedding[..., :hd].cos() + x[..., hd:] * rotary_embedding[..., hd:].sin() * -1,
            x[..., :hd] * rotary_embedding[..., :hd].sin() + x[..., hd:] * rotary_embedding[..., hd:].cos(),
        ),
        dim=-1,
    )
    assert torch.equal(x_rotary, x_rotary_test)

    offset = random.choice(range(1, cfg['max_seq_len']))
    rotary_embedding_offset = PE(cfg['max_seq_len'], offset=offset)
    x_rotary = apply_rotary_pos_emb(x[: offset + 1], rotary_embedding[: offset + 1])
    x_rotary_offset = apply_rotary_pos_emb(x[offset : offset + 1], rotary_embedding_offset[:1])
    assert torch.equal(x_rotary[-1], x_rotary_offset[0])


@pytest.mark.unit
def test_xpos(cfg):
    PE = XPOSPositionEmbedding(head_dim=cfg['hidden_size'])
    x = torch.rand(cfg['max_seq_len'], 1, cfg['num_attention_heads'], cfg['hidden_size'])

    x_rotary = PE(x)
    assert x_rotary.shape == x.shape

    offset = random.choice(range(1, cfg['max_seq_len']))
    x_rotary = PE(x[: offset + 1])
    x_rotary_offset = PE(x[offset : offset + 1], offset=offset)
    assert torch.equal(x_rotary[-1], x_rotary_offset[0])
