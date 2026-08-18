# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo.collections.speechlm2.parts import dflash as salm_dflash


REPO_ROOT = Path(__file__).parents[3]


class _FakeMoEMesh:
    mesh_dim_names = ("ep_shard", "ep")

    def __init__(self, ep_mesh):
        self.ep_mesh = ep_mesh

    def __getitem__(self, name):
        assert name == "ep"
        return self.ep_mesh


def test_synchronize_ep_group_uses_ep_process_group(monkeypatch):
    group = object()
    ep_mesh = SimpleNamespace(size=lambda: 8, get_group=lambda: group)
    calls = []
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "barrier", lambda *, group: calls.append(group))

    salm_dflash._synchronize_ep_group_before_target_forward(_FakeMoEMesh(ep_mesh))

    assert calls == [group]


def test_synchronize_ep_group_is_noop_without_distributed_ep(monkeypatch):
    calls = []
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "barrier", lambda **kwargs: calls.append(kwargs))

    salm_dflash._synchronize_ep_group_before_target_forward(None)
    salm_dflash._synchronize_ep_group_before_target_forward(
        _FakeMoEMesh(SimpleNamespace(size=lambda: 1, get_group=lambda: object()))
    )

    assert calls == []


def test_expand_ids_with_audio_preserves_internal_pad_valued_tokens():
    input_ids = torch.tensor([[0, 0, 11, 99, 0, 12]])
    audio_embeddings = [torch.randn(3, 4)]

    expanded = salm_dflash._expand_ids_with_audio(
        input_ids,
        audio_embeddings,
        padding_id=0,
        placeholder_id=99,
        mask_token_id=18,
    )

    assert expanded.tolist() == [[11, 18, 18, 18, 0, 12]]


def test_expand_ids_with_audio_left_pads_rows_to_common_length():
    input_ids = torch.tensor([[0, 10, 99, 12], [20, 21, 22, 23]])
    audio_embeddings = [torch.randn(2, 4)]

    expanded = salm_dflash._expand_ids_with_audio(
        input_ids,
        audio_embeddings,
        padding_id=0,
        placeholder_id=99,
        mask_token_id=18,
    )

    assert expanded.tolist() == [[10, 18, 18, 12], [20, 21, 22, 23]]


def test_expand_ids_with_audio_requires_every_replacement_to_be_used():
    with pytest.raises(ValueError, match="Used 0 of 1"):
        salm_dflash._expand_ids_with_audio(
            torch.tensor([[1, 2, 3]]),
            [torch.randn(2, 4)],
            padding_id=0,
            placeholder_id=99,
            mask_token_id=18,
        )


def test_build_draft_config_applies_explicit_architecture_and_layer_taps():
    target_config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=8,
        head_dim=16,
        vocab_size=128,
    )

    draft_config, layer_ids = salm_dflash._build_draft_config(
        target_config,
        {
            "draft_num_hidden_layers": 2,
            "target_layer_ids": [1, 6],
            "draft_model_config": {"intermediate_size": 96},
        },
        block_size=8,
        mask_token_id=18,
    )

    assert layer_ids == [1, 6]
    assert draft_config.num_hidden_layers == 2
    assert draft_config.intermediate_size == 96
    assert draft_config.dflash_config == {"mask_token_id": 18, "target_layer_ids": [1, 6]}


def test_build_draft_config_rejects_managed_overrides():
    target_config = Qwen3Config(hidden_size=64, num_attention_heads=4, num_hidden_layers=8, vocab_size=128)

    with pytest.raises(ValueError, match="cannot override managed keys: block_size"):
        salm_dflash._build_draft_config(
            target_config,
            {"draft_model_config": {"block_size": 32}},
            block_size=8,
            mask_token_id=18,
        )


def test_salm_automodel_dflash_defaults_match_nemotron_3_5_lightning():
    cfg = OmegaConf.load(REPO_ROOT / "examples/speechlm2/conf/salm_automodel.yaml")
    dflash_cfg = OmegaConf.to_container(cfg.dflash, resolve=True)
    target_config = Qwen3Config(
        hidden_size=2688,
        intermediate_size=1856,
        num_attention_heads=32,
        num_key_value_heads=2,
        num_hidden_layers=52,
        head_dim=128,
        vocab_size=131072,
    )

    draft_config, target_layer_ids = salm_dflash._build_draft_config(
        target_config,
        dflash_cfg,
        block_size=dflash_cfg["block_size"],
        mask_token_id=dflash_cfg["mask_token_id"],
    )

    assert dflash_cfg["enabled"] is False
    assert draft_config.num_hidden_layers == 6
    assert draft_config.hidden_size == 2688
    assert draft_config.intermediate_size == 6144
    assert draft_config.num_attention_heads == 32
    assert draft_config.num_key_value_heads == 2
    assert draft_config.head_dim == 128
    assert draft_config.rms_norm_eps == pytest.approx(1.0e-6)
    assert draft_config.max_position_embeddings == 1048576
    assert draft_config.rope_parameters == {
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "rope_theta": 10000,
        "rope_type": "yarn",
    }
    assert target_layer_ids == [1, 5, 19, 29, 41, 51]
    assert draft_config.dflash_config == {
        "mask_token_id": 990,
        "target_layer_ids": [1, 5, 19, 29, 41, 51],
    }
    assert draft_config.block_size == 8


class _TargetLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.calls = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        output_hidden_states,
        use_cache,
        return_dict,
        compute_logits=True,
    ):
        self.calls.append(
            {
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
                "use_cache": use_cache,
                "return_dict": return_dict,
                "compute_logits": compute_logits,
            }
        )
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden + index)
        return SimpleNamespace(hidden_states=(hidden,))


class _TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = _TargetLLM()


def test_target_hidden_states_uses_audio_embeddings_and_skips_logits():
    module = salm_dflash.SALMDFlashModule(_TargetModel(), {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0, 2]
    inputs = {
        "input_embeddings": torch.randn(2, 5, 4),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    hidden = module._target_hidden_states(inputs)

    assert hidden.shape == (2, 5, 8)
    assert torch.allclose(hidden[..., :4], inputs["input_embeddings"] + 1)
    assert torch.allclose(hidden[..., 4:], inputs["input_embeddings"] + 6)
    assert module.target.llm.calls == [
        {
            "attention_mask": inputs["attention_mask"],
            "output_hidden_states": False,
            "use_cache": False,
            "return_dict": True,
            "compute_logits": False,
        }
    ]


def test_get_consolidated_state_dict_uses_plain_state_dict_without_distributed(monkeypatch):
    expected = {"weight": torch.tensor([1.0])}
    model = SimpleNamespace(state_dict=Mock(return_value=expected))
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: False)

    result = salm_dflash._get_consolidated_model_state_dict(model)

    assert result is expected
    model.state_dict.assert_called_once_with()


def test_state_dict_hook_keeps_only_draft_parameters():
    module = SimpleNamespace(_CHECKPOINT_STATE_PREFIX="draft_model.")
    state_dict = {
        "wrapper.draft_model.layer.weight": torch.ones(1),
        "wrapper.target.layer.weight": torch.ones(1),
        "wrapper.trainer_module.loss.weight": torch.ones(1),
    }

    salm_dflash.SALMDFlashModule._keep_draft_checkpoint_state(module, state_dict, "wrapper.", {})

    assert list(state_dict) == ["wrapper.draft_model.layer.weight"]
