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

pytest.importorskip("nemo_automodel")
pytestmark = pytest.mark.unit

from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss  # noqa: E402

from nemo.collections.speechlm2.parts import dflash as salm_dflash  # noqa: E402


REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    "loss_mask,block_size",
    [
        (torch.tensor([[1.0] * 8]), 8),
        (torch.tensor([[1.0] * 4]), 8),
        (torch.tensor([[0.0] * 8 + [1.0] * 8]), 8),
        (torch.tensor([[0.0] * 9 + [1.0] * 7]), 8),
        (torch.tensor([[0.0] * 16, [0.0] * 8 + [1.0] * 8]), 8),
    ],
)
def test_anchor_precheck_matches_automodel_unpacked_sampler(loss_mask, block_size):
    """The synchronized precheck must exactly predict Automodel's early raise."""
    trainer = SimpleNamespace(block_size=block_size, num_anchors=512, max_total_anchors=None)
    try:
        salm_dflash.DFlashTrainerModule._sample_anchor_positions(
            trainer,
            seq_len=loss_mask.shape[1],
            loss_mask=loss_mask,
            device=loss_mask.device,
        )
    except salm_dflash.NoValidAnchorsError:
        automodel_has_valid = False
    else:
        automodel_has_valid = True

    assert salm_dflash._has_valid_dflash_anchors(loss_mask, block_size) is automodel_has_valid


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


def test_validate_dflash_parallelism_rejects_tensor_parallelism():
    mesh_context = SimpleNamespace(tp_size=2, pp_size=1, cp_size=1)

    with pytest.raises(NotImplementedError, match="tp_size=2"):
        salm_dflash._validate_dflash_parallelism(mesh_context)


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


def test_expand_ids_with_audio_matches_unpad_behavior_for_all_padding_row():
    expanded = salm_dflash._expand_ids_with_audio(
        torch.tensor([[0, 0, 0], [0, 11, 12]]),
        [],
        padding_id=0,
        placeholder_id=99,
        mask_token_id=18,
    )

    assert expanded.tolist() == [[0, 0], [11, 12]]


def test_device_falls_back_before_draft_configuration():
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})

    assert module.draft_model is None
    assert module.device == torch.device("cpu")


class _BatchTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.cfg = {}
        self.text_pad_id = 0
        self.audio_locator_tag_id = 99

    def _embed_tokens(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4).clone()


class _CaptureDFlashTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return "dflash-result"


def test_prepare_batch_keeps_full_unshifted_ids_and_token_aligned_loss_mask(
    monkeypatch,
):
    module = salm_dflash.SALMDFlashModule(_BatchTarget(), {"dflash": {"mask_token_id": 990, "block_size": 2}})
    audio_embeddings = [
        torch.tensor(
            [
                [100.0, 100.0, 100.0, 100.0],
                [101.0, 101.0, 101.0, 101.0],
            ]
        )
    ]
    monkeypatch.setattr(module, "_audio_embeddings", Mock(return_value=audio_embeddings))
    batch = {
        "input_ids": torch.tensor([[0, 10, 99, 20, 21, 22]]),
        "loss_mask": torch.tensor([[False, False, False, False, True, True]]),
    }

    prepared = module._prepare_batch(batch)

    assert prepared["input_ids"].tolist() == [[10, 990, 990, 20, 21, 22]]
    assert prepared["loss_mask"].tolist() == [[False, False, False, False, True, True]]
    assert prepared["attention_mask"].tolist() == [[True, True, True, True, True, True]]
    assert prepared["input_embeddings"].shape == (1, 6, 4)
    assert prepared["input_embeddings"][0].tolist() == [
        [10.0, 10.0, 10.0, 10.0],
        [100.0, 100.0, 100.0, 100.0],
        [101.0, 101.0, 101.0, 101.0],
        [20.0, 20.0, 20.0, 20.0],
        [21.0, 21.0, 21.0, 21.0],
        [22.0, 22.0, 22.0, 22.0],
    ]

    captured_hidden = torch.randn(1, 6, 8)
    target_hidden_states = Mock(return_value=captured_hidden)
    monkeypatch.setattr(module, "_target_hidden_states", target_hidden_states)
    module.trainer_module = _CaptureDFlashTrainer()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))

    result = module._run_batch(batch)

    assert result == "dflash-result"
    target_inputs = target_hidden_states.call_args.args[0]
    assert target_inputs["input_ids"].tolist() == [[10, 990, 990, 20, 21, 22]]
    assert target_inputs["loss_mask"].tolist() == [[False, False, False, False, True, True]]
    assert module.trainer_module.kwargs["input_ids"].tolist() == [[10, 990, 990, 20, 21, 22]]
    assert module.trainer_module.kwargs["loss_mask"].tolist() == [[False, False, False, False, True, True]]
    assert module.trainer_module.kwargs["hidden_states"] is captured_hidden


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
    assert draft_config.dflash_config == {
        "mask_token_id": 18,
        "target_layer_ids": [1, 6],
    }


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
    assert dflash_cfg["block_size"] == 8
    assert dflash_cfg["num_anchors"] == 512
    assert dflash_cfg["max_total_anchors"] == 512
    assert dflash_cfg["loss_decay_gamma"] == pytest.approx(4.0)
    assert dflash_cfg["attention_backend"] == "flex_attention"
    assert dflash_cfg["activation_checkpointing"] is True
    assert dflash_cfg["use_fused_linear_ce"] is True
    assert dflash_cfg["linear_ce_chunk_size"] == 256
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


class _MinimalTargetLLM(nn.Module):
    """Target whose explicit forward rejects every optional HF-style kwarg."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.calls = []

    def forward(self, *, inputs_embeds, attention_mask):
        self.calls.append({"attention_mask": attention_mask})
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden + index)
        return hidden


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


def test_target_hidden_states_filters_unsupported_optional_forward_kwargs():
    target = _TargetModel()
    target.llm = _MinimalTargetLLM()
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0, 1]
    inputs = {
        "input_embeddings": torch.randn(1, 4, 3),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
    }

    hidden = module._target_hidden_states(inputs)

    assert hidden.shape == (1, 4, 6)
    assert target.llm.calls == [{"attention_mask": inputs["attention_mask"]}]


def test_get_consolidated_state_dict_uses_plain_state_dict_without_distributed(
    monkeypatch,
):
    expected = {"weight": torch.tensor([1.0])}
    model = SimpleNamespace(state_dict=Mock(return_value=expected))
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: False)

    result = salm_dflash._get_consolidated_model_state_dict(model)

    assert result is expected
    model.state_dict.assert_called_once_with()


def test_train_keeps_frozen_target_in_eval_mode_and_draft_in_requested_mode():
    target = nn.Sequential(nn.Dropout(p=0.5))
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    module.draft_model = nn.Sequential(nn.Dropout(p=0.5))

    module.train()

    assert module.training
    assert module.draft_model.training
    assert not module.target.training
    assert not module.target[0].training


def test_globally_normalized_loss_uses_draft_dp_weight(monkeypatch):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    module._draft_dp_size = 2
    module._draft_dp_group = object()
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value, *, op, group):
        assert op == salm_dflash.torch.distributed.ReduceOp.SUM
        assert group is module._draft_dp_group
        value.fill_(10.0)

    monkeypatch.setattr(salm_dflash.torch.distributed, "all_reduce", fake_all_reduce)
    local_loss = torch.tensor(2.0, requires_grad=True)
    metrics = SimpleNamespace(loss=local_loss, loss_weight=torch.tensor(3.0))

    loss = module._globally_normalized_loss(metrics)
    loss.backward()

    assert loss.item() == pytest.approx(1.2)
    assert local_loss.grad.item() == pytest.approx(0.6)


def test_dflash_loss_times_weight_recovers_decay_weighted_numerator():
    torch.manual_seed(7)
    block_size = 4
    logits = torch.randn(1, 6, 11)
    targets = torch.randint(0, 11, (1, 6))
    block_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]])
    loss_fn = DFlashDecayLoss(loss_gamma=4.0, normalize="mean")

    result = loss_fn(logits, targets, block_mask, block_size=block_size)

    nll = torch.nn.functional.cross_entropy(logits.view(-1, 11), targets.view(-1), reduction="none").view(1, 6)
    depth_weights = torch.exp(-torch.arange(block_size - 1, dtype=logits.dtype) / 4.0).repeat(2)
    effective_weights = block_mask * depth_weights.unsqueeze(0)
    expected_numerator = (nll * effective_weights).sum()
    torch.testing.assert_close(result.total_loss * effective_weights.sum(), expected_numerator)


def test_training_step_synchronizes_multi_dataset_skips(monkeypatch):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    module.draft_model = nn.Linear(1, 1)
    module._draft_dp_size = 1
    module._draft_dp_group = None
    monkeypatch.setattr(module, "log", Mock())
    monkeypatch.setattr(salm_dflash, "_max_rank_value", lambda _value, _device: 3)
    availability = []

    def agree(local_condition, _device):
        availability.append(local_condition)
        return local_condition

    monkeypatch.setattr(salm_dflash, "_all_ranks_agree", agree)
    monkeypatch.setattr(salm_dflash, "_all_ranks_report_same_value", lambda _value, _device: True)
    metrics = SimpleNamespace(
        loss=torch.tensor(2.0, requires_grad=True),
        loss_weight=torch.tensor(3.0),
        accuracy=torch.tensor(0.5),
        accept_len=torch.tensor(1.5),
    )
    run_batch = Mock(side_effect=[salm_dflash.NoValidAnchorsError("skip"), metrics])
    monkeypatch.setattr(module, "_run_batch", run_batch)
    batch = {
        "dataset_a": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        "dataset_b": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
    }

    loss = module.training_step(batch, batch_idx=0)

    torch.testing.assert_close(loss, metrics.loss)
    assert availability == [True, True, False]
    assert run_batch.call_count == 2


def test_training_step_returns_differentiable_zero_when_every_dataset_is_skipped(monkeypatch):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    module.draft_model = nn.Linear(1, 1)
    log = Mock()
    monkeypatch.setattr(module, "log", log)
    monkeypatch.setattr(salm_dflash, "_max_rank_value", lambda value, _device: value)
    monkeypatch.setattr(salm_dflash, "_all_ranks_agree", lambda condition, _device: condition)
    monkeypatch.setattr(salm_dflash, "_all_ranks_report_same_value", lambda _value, _device: True)
    monkeypatch.setattr(
        module,
        "_run_batch",
        Mock(side_effect=salm_dflash.NoValidAnchorsError("skip")),
    )

    loss = module.training_step({"input_ids": torch.ones(1, 2, dtype=torch.long)}, batch_idx=0)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert all(parameter.grad is None for parameter in module.draft_model.parameters())
    log.assert_any_call("train/dflash_skipped_step", 1.0, on_step=True)
    log.assert_any_call("train/dflash_skip/no_valid_anchors", 1.0, on_step=True)


def test_validation_step_accumulates_additive_metrics_in_float64(monkeypatch):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    module.draft_model = nn.Linear(1, 1)
    metrics = SimpleNamespace(
        loss=torch.tensor(2.0, dtype=torch.bfloat16),
        loss_weight=torch.tensor(3.0),
        correct_tokens=torch.tensor(2**24 + 1),
        valid_tokens=torch.tensor(2**24 + 3),
        accept_len_sum=torch.tensor(7.0),
        valid_blocks=torch.tensor(4),
    )
    monkeypatch.setattr(salm_dflash, "_max_rank_value", lambda value, _device: value)
    monkeypatch.setattr(salm_dflash, "_all_ranks_agree", lambda condition, _device: condition)
    monkeypatch.setattr(salm_dflash, "_all_ranks_report_same_value", lambda _value, _device: True)
    monkeypatch.setattr(module, "_run_batch", Mock(return_value=metrics))

    module.validation_step({"input_ids": torch.ones(1, 2, dtype=torch.long)}, batch_idx=0)

    stored = module._partial_val_metrics["validation"][0]
    assert stored.dtype == torch.float64
    assert stored[2].item() == 2**24 + 1
    assert stored[3].item() == 2**24 + 3


def test_aggregate_validation_accuracy_preserves_default_checkpoint_monitor(
    monkeypatch,
):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    log = Mock()
    monkeypatch.setattr(module, "log", log)

    module._log_validation_metrics(torch.tensor([8.0, 4.0, 3.0, 6.0, 5.0, 2.0]))

    log.assert_any_call("val/dflash_accuracy", torch.tensor(0.5), on_epoch=True)
    log.assert_any_call("val_acc", torch.tensor(0.5), on_epoch=True)


def test_state_dict_hook_keeps_only_draft_parameters():
    module = SimpleNamespace(_CHECKPOINT_STATE_PREFIX="draft_model.")
    state_dict = {
        "wrapper.draft_model.layer.weight": torch.ones(1),
        "wrapper.target.layer.weight": torch.ones(1),
        "wrapper.trainer_module.loss.weight": torch.ones(1),
    }

    salm_dflash.SALMDFlashModule._keep_draft_checkpoint_state(module, state_dict, "wrapper.", {})

    assert list(state_dict) == ["wrapper.draft_model.layer.weight"]
