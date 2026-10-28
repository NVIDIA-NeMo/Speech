# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import torch
from omegaconf import DictConfig

from nemo.collections.speechlm2.modules.perception import AudioPerceptionModule, QformerConnector

PROMPT_SIZE = 3
LLM_HIDDEN_SIZE = 8


@pytest.fixture
def connector():
    torch.manual_seed(0)
    return QformerConnector(
        prompt_size=PROMPT_SIZE,
        target_layer_ids=[0],
        qformer_num_hidden_layers=1,
        encoder_config=DictConfig({"d_model": 16, "encoder_attention_heads": 2}),
        llm_config=DictConfig({"hidden_size": LLM_HIDDEN_SIZE}),
    ).eval()


@pytest.fixture
def perception():
    torch.manual_seed(0)
    cfg = DictConfig(
        {
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "features": 80,
            },
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 80,
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 2,
                "subsampling_factor": 4,
                "ff_expansion_factor": 2,
                "conv_kernel_size": 5,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.QformerConnector",
                "prompt_size": PROMPT_SIZE,
                "target_layer_ids": [0],
                "qformer_num_hidden_layers": 1,
                "encoder_config": {"d_model": 32, "encoder_attention_heads": 4},
                "llm_config": {"hidden_size": LLM_HIDDEN_SIZE},
            },
            "output_dim": LLM_HIDDEN_SIZE,
        }
    )
    return AudioPerceptionModule(cfg).eval()


def test_qformer_connector_returns_the_prompt_size_as_length(connector):
    hidden = [torch.randn(2, 16, 11), torch.randn(2, 16, 11)]
    lengths = torch.tensor([11, 5])

    with torch.no_grad():
        output, output_lens = connector(hidden, [lengths, lengths])

    assert output.shape == (2, LLM_HIDDEN_SIZE, PROMPT_SIZE)
    assert output_lens.tolist() == [PROMPT_SIZE, PROMPT_SIZE]


def test_qformer_connector_batched_rows_match_single_rows(connector):
    """The padded frames of a shorter row do not enter its cross-attention."""
    torch.manual_seed(0)
    hidden = [torch.randn(2, 16, 11), torch.randn(2, 16, 11)]
    lengths = torch.tensor([11, 5])

    with torch.no_grad():
        batched, _ = connector(hidden, [lengths, lengths])
        full_row, _ = connector([h[:1] for h in hidden], [lengths[:1], lengths[:1]])
        short_row, _ = connector([h[1:, :, :5] for h in hidden], [lengths[1:], lengths[1:]])

    torch.testing.assert_close(batched[0], full_row[0])
    torch.testing.assert_close(batched[1], short_row[0])


def test_perception_with_qformer_connector_batched_rows_match_single_rows(perception):
    """Through the perception module, a shorter row gives the same prompt embeddings as the row alone."""
    torch.manual_seed(0)
    audio = torch.randn(2, 16000)
    audio[1, 6000:] = 0.0
    audio_lens = torch.tensor([16000, 6000])

    with torch.no_grad():
        batched, batched_lens = perception(input_signal=audio, input_signal_length=audio_lens)
        full_row, _ = perception(input_signal=audio[:1], input_signal_length=audio_lens[:1])
        short_row, _ = perception(input_signal=audio[1:, :6000], input_signal_length=audio_lens[1:])

    assert batched.shape == (2, PROMPT_SIZE, LLM_HIDDEN_SIZE)
    torch.testing.assert_close(batched[0], full_row[0])
    torch.testing.assert_close(batched[1], short_row[0])
    assert batched_lens.tolist() == [PROMPT_SIZE, PROMPT_SIZE]
