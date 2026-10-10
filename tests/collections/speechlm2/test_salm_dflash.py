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
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel  # noqa: E402
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel  # noqa: E402

from nemo.collections.speechlm2.parts import dflash as salm_dflash  # noqa: E402
from nemo.collections.speechlm2.parts import packed_sequences  # noqa: E402

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
    trainer = SimpleNamespace(block_size=block_size, num_anchors=512)
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


def test_stateful_training_resume_does_not_skip_a_batch():
    from lightning.pytorch.loops.utilities import _select_data_fetcher
    from lightning.pytorch.trainer.states import RunningStage
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    class StatefulCursor:
        def __init__(self, position=0):
            self.position = position

        def __iter__(self):
            return self

        def __next__(self):
            batch = self.position
            self.position += 1
            return batch

    def setup(position=0):
        cursor = StatefulCursor(position)
        trainer = SimpleNamespace(lightning_module=salm_dflash.SALMDFlashModule)
        fetcher = _select_data_fetcher(trainer, RunningStage.TRAINING)
        fetcher.setup(CombinedLoader(cursor, "max_size_cycle"))
        iter(fetcher)
        return cursor, fetcher

    def next_batch(fetcher):
        batch = next(fetcher)
        if hasattr(batch, "__next__"):
            batch = next(batch)
        return batch[0]

    cursor, continuous = setup()
    assert next_batch(continuous) == 0
    saved_position = cursor.position
    expected_next = next_batch(continuous)
    _, resumed = setup(saved_position)

    assert next_batch(resumed) == expected_next == 1


def test_training_step_transfers_iterator_batch_before_forward(monkeypatch):
    from nemo.core.utils import lightning_utils

    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    events = []

    def transform(name):
        def apply(batch, **kwargs):
            events.append((name, batch, kwargs))
            return batch + 1

        return apply

    monkeypatch.setattr(lightning_utils, "_check_shutdown_before_next_batch", lambda _: events.append("shutdown"))
    module._trainer = SimpleNamespace(
        precision_plugin=SimpleNamespace(convert_input=transform("precision")),
        strategy=SimpleNamespace(batch_to_device=transform("device")),
    )
    monkeypatch.setattr(module, "_on_before_batch_transfer", transform("before_transfer"))
    forward = Mock(return_value="loss")
    monkeypatch.setattr(module, "_training_step_batch", forward, raising=False)

    def batches():
        events.append("fetch")
        yield 10, 7, 2

    assert module.training_step(iter(batches())) == "loss"
    assert events == [
        "shutdown",
        "fetch",
        ("precision", 10, {}),
        ("before_transfer", 11, {"dataloader_idx": 2}),
        ("device", 12, {"dataloader_idx": 2}),
    ]
    forward.assert_called_once_with(13, 7)


@pytest.mark.parametrize("variant", ["dflash", "dflash2"])
@pytest.mark.parametrize("skip_batch", [False, True])
def test_iterator_training_runs_with_lightning_logging(monkeypatch, tmp_path, skip_batch, variant):
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader

    from nemo.utils.callbacks.training_stats import TrainingStatsCallback

    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18, "variant": variant}})
    module.draft_model = nn.Linear(1, 1)
    module._draft_dp_size = 1
    module._draft_dp_group = None
    monkeypatch.setattr(module, "configure_model", lambda: None)
    monkeypatch.setattr(module, "on_train_end", lambda: None)

    def run_batch(batch):
        if skip_batch:
            raise salm_dflash.NoValidAnchorsError("skip")
        loss = module.draft_model(torch.ones(1, 1)).sum()
        return SimpleNamespace(
            loss=loss,
            loss_weight=torch.tensor(1.0),
            accuracy=torch.tensor(0.5),
            accept_len=torch.tensor(1.5),
            base_loss=loss,
            selector_loss=loss * 0.1,
            selector_loss_denominator=torch.tensor(1.0),
            base_accuracy=torch.tensor(0.5),
            base_accept_len=torch.tensor(1.5),
            candidate_recall=torch.tensor(0.75),
        )

    monkeypatch.setattr(module, "_run_batch", run_batch)
    stats = TrainingStatsCallback()
    trainer = Trainer(
        callbacks=[stats],
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=DataLoader([{"input_ids": torch.ones(2, dtype=torch.long)}] * 2))

    assert trainer.global_step == 2
    assert stats.num_examples_total == 2
    assert stats.num_tokens_total == 4
    assert trainer.logged_metrics["train/dflash_skipped_step"].item() == float(skip_batch)
    if not skip_batch:
        assert "train/dflash_loss" in trainer.logged_metrics


@pytest.mark.parametrize("axis", ["tp_size", "cp_size"])
def test_validate_dflash_parallelism_rejects_sequence_sharding(axis):
    sizes = {"tp_size": 1, "pp_size": 1, "cp_size": 1}
    sizes[axis] = 2
    mesh_context = SimpleNamespace(**sizes)

    with pytest.raises(NotImplementedError, match=rf"{axis}=2"):
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


def test_expand_ids_with_audio_handles_adjacent_and_edge_placeholders():
    expanded = salm_dflash._expand_ids_with_audio(
        torch.tensor([[0, 99, 99, 11, 99], [12, 13, 99, 14, 15]]),
        [torch.randn(length, 4) for length in (2, 1, 3, 2)],
        padding_id=0,
        placeholder_id=99,
        mask_token_id=18,
    )

    assert expanded.tolist() == [[18, 18, 18, 11, 18, 18, 18], [0, 12, 13, 18, 18, 14, 15]]


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


class _AudioTarget(_BatchTarget):
    def __init__(self):
        super().__init__()
        self.cfg.update({"encoder_chunk_size_seconds": 30.0, "encoder_chunk_batch_size": 8})
        self.perception = nn.Linear(1, 1)
        self.sampling_rate = 16000
        self._device_mesh = None

    def _uses_parallel_expert_encoder(self):
        return False


@pytest.mark.parametrize("uses_parallel_expert_encoder", [False, True])
@pytest.mark.parametrize("has_speaker_targets", [False, True])
def test_audio_embeddings_forwards_bounded_chunking_and_speaker_lengths(
    monkeypatch, uses_parallel_expert_encoder, has_speaker_targets
):
    target = _AudioTarget()
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    monkeypatch.setattr(target, "_uses_parallel_expert_encoder", lambda: uses_parallel_expert_encoder)
    expected = [torch.randn(2, 3)]
    encode = Mock(return_value=expected)
    monkeypatch.setattr(salm_dflash, "encode_audio_with_cp_distribution", encode)
    speaker_targets = torch.randn(1, 5, 8) if has_speaker_targets else None
    speaker_lengths = torch.tensor([5]) if has_speaker_targets else None
    batch = {
        "audios": torch.randn(1, 32000),
        "audio_lens": torch.tensor([32000]),
        "spk_targets": speaker_targets,
        "spk_target_length": speaker_lengths,
    }

    assert module._audio_embeddings(batch) is expected
    assert encode.call_args.kwargs["chunk_size_seconds"] == 30.0
    assert encode.call_args.kwargs["chunk_batch_size"] == 8
    assert encode.call_args.kwargs["spk_targets"] is (speaker_targets if uses_parallel_expert_encoder else None)
    assert encode.call_args.kwargs["spk_target_lengths"] is (speaker_lengths if uses_parallel_expert_encoder else None)


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


def test_pack_audio_for_dflash_builds_unshifted_boundary_metadata():
    input_ids = torch.tensor([[0, 10, 99, 20, 21], [30, 31, 99, 0, 32]])
    embeds = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2).clone()
    loss_mask = torch.tensor(
        [[False, False, False, True, True], [False, True, False, True, True]],
        dtype=torch.bool,
    )
    replacements = [
        torch.tensor([[100.0, 101.0], [102.0, 103.0]]),
        torch.tensor([[200.0, 201.0]]),
    ]

    packed = packed_sequences.pack_audio_for_dflash(
        input_ids=input_ids,
        embeds=embeds,
        loss_mask=loss_mask,
        replacements=replacements,
        padding_id=0,
        placeholder_id=99,
        mask_token_id=990,
    )

    assert packed["input_ids"].tolist() == [[10, 990, 990, 20, 21, 30, 31, 990, 0, 32]]
    assert packed["loss_mask"].tolist() == [[False, False, False, True, True, False, True, False, False, True]]
    assert packed["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    assert packed["seq_lens"].tolist() == [[5, 5]]
    assert packed["doc_remaining"].tolist() == [[4, 3, 2, 1, 0, 4, 3, 2, 1, 0]]
    assert packed["cu_seqlens"].dtype == torch.int32
    assert packed["cu_seqlens"].tolist() == [0, 5, 10]
    assert packed["max_seqlen"].item() == 5
    assert packed["qkv_format"] == "thd"
    assert packed["input_embeddings"].shape == (10, 2)
    assert packed["input_embeddings"][1:3].tolist() == replacements[0].tolist()
    assert packed["input_embeddings"][7:8].tolist() == replacements[1].tolist()


def test_validate_packed_dflash_rejects_non_int32_cu_seqlens():
    packed = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "input_embeddings": torch.randn(4, 2),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "seq_lens": torch.tensor([[2, 2]]),
        "doc_remaining": torch.tensor([[1, 0, 1, 0]]),
        "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int64),
        "max_seqlen": torch.tensor(2, dtype=torch.int32),
        "qkv_format": "thd",
    }

    with pytest.raises(ValueError, match="cu_seqlens must have dtype torch.int32"):
        packed_sequences._validate_packed_dflash_inputs(packed)


def test_pack_audio_for_dflash_one_document_matches_unpacked_preparation(monkeypatch):
    target = _BatchTarget()
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 990, "block_size": 2}})
    audio_embeddings = [torch.tensor([[100.0] * 4, [101.0] * 4])]
    monkeypatch.setattr(module, "_audio_embeddings", Mock(return_value=audio_embeddings))
    batch = {
        "input_ids": torch.tensor([[0, 10, 99, 20, 21, 22]]),
        "loss_mask": torch.tensor([[False, False, False, False, True, True]]),
    }
    unpacked = module._prepare_batch(batch)

    target.cfg["packed_sequences"] = True
    packed = module._prepare_batch(batch)

    assert packed["input_ids"].tolist() == unpacked["input_ids"].tolist()
    assert packed["loss_mask"].tolist() == unpacked["loss_mask"].tolist()
    torch.testing.assert_close(packed["input_embeddings"].unsqueeze(0), unpacked["input_embeddings"])
    assert packed["seq_lens"].tolist() == [[6]]
    assert packed["doc_remaining"].tolist() == [[5, 4, 3, 2, 1, 0]]


def test_pack_audio_for_dflash_rejects_malformed_inputs():
    with pytest.raises(ValueError, match=r"same \[B, S\] shape"):
        packed_sequences.pack_audio_for_dflash(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            embeds=torch.ones(1, 3, 2),
            loss_mask=torch.ones(1, 2, dtype=torch.bool),
            replacements=[],
            padding_id=0,
            placeholder_id=99,
            mask_token_id=990,
        )


@pytest.mark.parametrize("variant", ["dflash", "dflash2"])
def test_run_batch_forwards_all_packing_metadata(monkeypatch, variant):
    module = salm_dflash.SALMDFlashModule(
        _BatchTarget(),
        {"dflash": {"variant": variant, "mask_token_id": 990, "block_size": 2}},
    )
    packed = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "input_embeddings": torch.randn(4, 4),
        "attention_mask": None,
        "loss_mask": torch.tensor([[True, True, True, True]]),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "seq_lens": torch.tensor([[2, 2]]),
        "doc_remaining": torch.tensor([[1, 0, 1, 0]]),
        "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32),
        "max_seqlen": torch.tensor(2, dtype=torch.int32),
        "qkv_format": "thd",
    }
    monkeypatch.setattr(module, "_prepare_batch", Mock(return_value=packed))
    monkeypatch.setattr(module, "_target_hidden_states", Mock(return_value=torch.randn(1, 4, 8)))
    monkeypatch.setattr(salm_dflash, "_has_valid_dflash_anchors", lambda *args, **kwargs: True)
    module.trainer_module = _CaptureDFlashTrainer()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))

    module._run_batch({})

    assert module.trainer_module.kwargs["position_ids"] is packed["position_ids"]
    assert module.trainer_module.kwargs["seq_lens"] is packed["seq_lens"]
    assert module.trainer_module.kwargs["doc_remaining"] is packed["doc_remaining"]


def test_packed_anchor_precheck_requires_complete_block_in_document():
    loss_mask = torch.tensor([[False, True, True, True, True, True]])
    doc_remaining = torch.tensor([[2, 1, 0, 2, 1, 0]])

    assert not salm_dflash._has_valid_dflash_anchors(loss_mask, block_size=4, doc_remaining=doc_remaining)
    assert salm_dflash._has_valid_dflash_anchors(loss_mask, block_size=3, doc_remaining=doc_remaining)


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
        "block_size": 8,
        "mask_token_id": 18,
        "target_layer_ids": [1, 6],
    }
    assert draft_config.architectures == ["Qwen3DFlashDraftModel"]
    assert draft_config.is_causal is False


def test_build_dflash2_config_matches_automodel_recipe_fields():
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
            "variant": "dflash2",
            "draft_num_hidden_layers": 2,
            "target_layer_ids": [1, 6],
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 32,
            "selector_top_k": 8,
            "draft_sliding_window": 32,
        },
        block_size=8,
        mask_token_id=18,
    )

    assert layer_ids == [1, 6]
    assert draft_config.architectures == ["Qwen3DFlash2DraftModel"]
    assert draft_config.layer_types == ["sliding_attention", "sliding_attention"]
    assert draft_config.sliding_window == 32
    assert draft_config.use_sliding_window is True
    assert draft_config.dflash_config == {
        "block_size": 8,
        "mask_token_id": 18,
        "target_layer_ids": [1, 6],
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 32,
        "selector_top_k": 8,
    }
    draft = Qwen3DFlash2DraftModel(draft_config)
    assert draft.candidate_selector.top_k == 8


def test_dflash2_rejects_fused_linear_ce():
    with pytest.raises(ValueError, match="use_fused_linear_ce"):
        salm_dflash.SALMDFlashModule(
            nn.Linear(1, 1),
            {
                "dflash": {
                    "variant": "dflash2",
                    "mask_token_id": 18,
                    "use_fused_linear_ce": True,
                }
            },
        )


class _TrainerTargetLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=6,
            head_dim=8,
            vocab_size=64,
        )
        self.embed_tokens = nn.Embedding(64, 32)
        self.lm_head = nn.Linear(32, 64, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _TrainerTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = _TrainerTargetLLM()


@pytest.mark.parametrize(
    "variant,trainer_name",
    [("dflash", "DFlashTrainerModule"), ("dflash2", "DFlash2TrainerModule")],
)
def test_create_trainer_module_selects_configured_variant(monkeypatch, variant, trainer_name):
    base_factory = Mock(return_value=object())
    dflash2_factory = Mock(return_value=object())
    monkeypatch.setattr(salm_dflash, "DFlashTrainerModule", base_factory)
    monkeypatch.setattr(salm_dflash, "DFlash2TrainerModule", dflash2_factory)
    module = salm_dflash.SALMDFlashModule(
        _TrainerTarget(),
        {
            "dflash": {
                "variant": variant,
                "mask_token_id": 18,
                "selector_loss_weight": 0.25,
                "use_fused_linear_ce": False,
            }
        },
    )
    module.draft_model = nn.Linear(32, 32)

    result = module._create_trainer_module()

    selected = dflash2_factory if trainer_name == "DFlash2TrainerModule" else base_factory
    unselected = base_factory if trainer_name == "DFlash2TrainerModule" else dflash2_factory
    assert result is selected.return_value
    expected_draft_cls = Qwen3DFlash2DraftModel if variant == "dflash2" else Qwen3DFlashDraftModel
    assert module._draft_model_class() is expected_draft_cls
    unselected.assert_not_called()
    kwargs = selected.call_args.kwargs
    assert kwargs["draft_model"] is module.draft_model
    if variant == "dflash2":
        assert kwargs["selector_loss_weight"] == pytest.approx(0.25)
        assert "use_fused_linear_ce" not in kwargs
    else:
        assert "use_fused_linear_ce" not in kwargs


@pytest.mark.parametrize("variant", ["dflash", "dflash2"])
def test_draft_hf_warm_start_restores_weights(tmp_path, variant):
    target = _TrainerTarget()
    cfg = {
        "variant": variant,
        "mask_token_id": 63,
        "block_size": 4,
        "draft_num_hidden_layers": 2,
        "target_layer_ids": [1, 4],
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": 64,
        "attention_backend": "sdpa",
        "use_fused_linear_ce": False,
    }
    module = salm_dflash.SALMDFlashModule(target, {"dflash": cfg})
    draft_config, _ = salm_dflash._build_draft_config(target.llm.config, cfg, 4, 63)
    draft_config._attn_implementation = "sdpa"
    original = module._draft_model_class()(draft_config)
    original.save_pretrained(tmp_path)
    cfg["init_from_pretrained"] = str(tmp_path)
    module = salm_dflash.SALMDFlashModule(target, {"dflash": cfg})
    restored = module._initialize_draft_model(draft_config, torch.float32)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_draft_hf_warm_start_rejects_incomplete_checkpoint(monkeypatch):
    module = salm_dflash.SALMDFlashModule(
        _TrainerTarget(), {"dflash": {"mask_token_id": 63, "init_from_pretrained": "/missing-weights"}}
    )
    factory = Mock()
    factory.from_pretrained.return_value = (nn.Linear(1, 1), {"missing_keys": ["layer.weight"]})
    monkeypatch.setattr(module, "_draft_model_class", lambda: factory)
    with pytest.raises(RuntimeError, match="missing_keys"):
        module._initialize_draft_model(Qwen3Config(), torch.float32)


def test_dflash2_salm_components_run_forward_and_train_selector():
    torch.manual_seed(7)
    target = _TrainerTarget()
    dflash_config = {
        "variant": "dflash2",
        "mask_token_id": 63,
        "block_size": 4,
        "draft_num_hidden_layers": 2,
        "target_layer_ids": [1, 4],
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": 64,
        "num_anchors": 2,
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
    }
    module = salm_dflash.SALMDFlashModule(target, {"dflash": dflash_config})
    draft_config, module.target_layer_ids = salm_dflash._build_draft_config(
        target.llm.config,
        dflash_config,
        block_size=4,
        mask_token_id=63,
    )
    draft_config._attn_implementation = "sdpa"
    module.draft_model = module._draft_model_class()(draft_config)
    module.trainer_module = module._create_trainer_module()
    input_ids = torch.randint(0, 63, (1, 8))
    hidden_states = torch.randn(1, 8, 64)

    metrics = module.trainer_module(input_ids=input_ids, hidden_states=hidden_states, loss_mask=torch.ones(1, 8))
    metrics.loss.backward()

    assert torch.isfinite(metrics.loss)
    assert metrics.selector_loss.item() > 0
    assert module.draft_model.candidate_selector.successor_codebook.grad.abs().sum() > 0


def test_build_draft_config_rejects_managed_overrides():
    target_config = Qwen3Config(hidden_size=64, num_attention_heads=4, num_hidden_layers=8, vocab_size=128)

    with pytest.raises(ValueError, match="cannot override managed keys: block_size"):
        salm_dflash._build_draft_config(
            target_config,
            {"draft_model_config": {"block_size": 32}},
            block_size=8,
            mask_token_id=18,
        )


def test_salm_automodel_dflash2_defaults_match_nemotron_3_5_lightning():
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
    assert dflash_cfg["variant"] == "dflash2"
    assert dflash_cfg["block_size"] == 8
    assert dflash_cfg["num_anchors"] == 512
    assert dflash_cfg["loss_decay_gamma"] == pytest.approx(4.0)
    assert dflash_cfg["attention_backend"] == "flex_attention"
    assert dflash_cfg["activation_checkpointing"] is True
    assert dflash_cfg["use_fused_linear_ce"] is False
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
        "block_size": 8,
        "mask_token_id": 990,
        "target_layer_ids": [1, 5, 19, 29, 41, 51],
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
    }
    assert draft_config.block_size == 8
    assert draft_config.architectures == ["Qwen3DFlash2DraftModel"]


class _TargetLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
        self.norm = _AddConstant(10.0)
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
        hidden = self.norm(hidden)
        return SimpleNamespace(hidden_states=(hidden,))


class _TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = _TargetLLM()


class _AddConstant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, inputs):
        return inputs + self.value


class _MinimalTargetLLM(nn.Module):
    """Target whose explicit forward rejects every optional HF-style kwarg."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.norm = nn.Identity()
        self.calls = []

    def forward(self, *, inputs_embeds, attention_mask):
        self.calls.append({"attention_mask": attention_mask})
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden + index)
        return self.norm(hidden)


def test_target_hidden_states_uses_pre_final_norm_block_outputs_and_skips_logits():
    module = salm_dflash.SALMDFlashModule(_TargetModel(), {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0, 2]
    inputs = {
        "input_embeddings": torch.randn(2, 5, 4),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    hidden = module._target_hidden_states(inputs)

    assert hidden.shape == (2, 5, 8)
    assert torch.allclose(hidden[..., :4], inputs["input_embeddings"] + 1)
    # Block 2 contributes +3 after blocks 0 and 1 contributed +1 and +2.
    # The separate final norm contributes +10 to the model output, but it must
    # not alter DFlash's captured decoder-block feature.
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


class _PackedTargetLLM(nn.Module):
    """Small target that mixes causally inside, but never across, THD documents."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.layers = nn.ModuleList([nn.Identity()])
        self.calls = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        qkv_format=None,
        cu_seqlens=None,
        position_ids=None,
        max_seqlen=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        compute_logits=False,
    ):
        self.calls.append(
            {
                "qkv_format": qkv_format,
                "cu_seqlens": cu_seqlens,
                "position_ids": position_ids,
                "max_seqlen": max_seqlen,
                "compute_logits": compute_logits,
            }
        )
        if qkv_format == "thd":
            boundaries = cu_seqlens.tolist()
            mixed = torch.cat(
                [inputs_embeds[start:end].cumsum(dim=0) for start, end in zip(boundaries, boundaries[1:])],
                dim=0,
            )
        else:
            mixed = inputs_embeds.cumsum(dim=1)
        hidden = self.layers[0](mixed)
        return SimpleNamespace(hidden_states=(hidden,))


class _PackedTargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = _PackedTargetLLM()


def _packed_target_inputs(embeddings, seq_lens):
    lengths = torch.tensor([seq_lens], dtype=torch.long)
    positions = torch.cat([torch.arange(length) for length in seq_lens]).unsqueeze(0)
    remaining = torch.cat([torch.arange(length - 1, -1, -1) for length in seq_lens]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
    token_count = embeddings.shape[0]
    return {
        "input_ids": torch.arange(token_count).unsqueeze(0),
        "input_embeddings": embeddings,
        "attention_mask": None,
        "loss_mask": torch.ones(1, token_count, dtype=torch.bool),
        "position_ids": positions,
        "seq_lens": lengths,
        "doc_remaining": remaining,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": torch.tensor(max(seq_lens), dtype=torch.int32),
        "qkv_format": "thd",
    }


def test_target_hidden_states_one_document_packed_matches_unpacked():
    target = _PackedTargetModel()
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0]
    embeddings = torch.randn(5, 4)

    unpacked = module._target_hidden_states(
        {
            "input_embeddings": embeddings.unsqueeze(0),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        }
    )
    packed = module._target_hidden_states(_packed_target_inputs(embeddings, [5]))

    torch.testing.assert_close(packed, unpacked)
    assert packed.shape == (1, 5, 4)


def test_target_hidden_states_packed_isolates_documents_and_uses_thd_metadata():
    target = _PackedTargetModel()
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0]
    embeddings = torch.randn(6, 4)
    inputs = _packed_target_inputs(embeddings, [3, 3])

    reference = module._target_hidden_states(inputs)
    independent = torch.cat(
        [
            module._target_hidden_states(
                {
                    "input_embeddings": embeddings[start:end].unsqueeze(0),
                    "attention_mask": torch.ones(1, end - start, dtype=torch.bool),
                }
            )
            for start, end in ((0, 3), (3, 6))
        ],
        dim=1,
    )
    torch.testing.assert_close(reference, independent)

    perturbed_inputs = _packed_target_inputs(embeddings.clone(), [3, 3])
    perturbed_inputs["input_embeddings"][3:] += 100
    perturbed = module._target_hidden_states(perturbed_inputs)
    torch.testing.assert_close(reference[:, :3], perturbed[:, :3])
    assert not torch.allclose(reference[:, 3:], perturbed[:, 3:])

    reverse_perturbed_inputs = _packed_target_inputs(embeddings.clone(), [3, 3])
    reverse_perturbed_inputs["input_embeddings"][:3] += 100
    reverse_perturbed = module._target_hidden_states(reverse_perturbed_inputs)
    assert not torch.allclose(reference[:, :3], reverse_perturbed[:, :3])
    torch.testing.assert_close(reference[:, 3:], reverse_perturbed[:, 3:])

    assert target.llm.calls[-1]["qkv_format"] == "thd"
    assert target.llm.calls[-1]["cu_seqlens"].tolist() == [0, 3, 6]
    assert target.llm.calls[-1]["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2]]
    assert target.llm.calls[-1]["compute_logits"] is False


def test_target_hidden_states_rejects_partial_packing_metadata():
    module = salm_dflash.SALMDFlashModule(_PackedTargetModel(), {"dflash": {"mask_token_id": 18}})
    module.target_layer_ids = [0]

    with pytest.raises(ValueError, match="missing required fields"):
        module._target_hidden_states(
            {
                "input_embeddings": torch.randn(4, 3),
                "attention_mask": None,
                "qkv_format": "thd",
                "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
            }
        )


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


def test_dflash2_globally_normalizes_base_and_selector_terms_separately(monkeypatch):
    module = salm_dflash.SALMDFlashModule(
        nn.Linear(1, 1),
        {
            "dflash": {
                "variant": "dflash2",
                "mask_token_id": 18,
                "selector_loss_weight": 0.5,
            }
        },
    )
    module._draft_dp_size = 2
    module._draft_dp_group = object()
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: True)
    global_weights = iter((10.0, 8.0))

    def fake_all_reduce(value, *, op, group):
        assert op == salm_dflash.torch.distributed.ReduceOp.SUM
        assert group is module._draft_dp_group
        value.fill_(next(global_weights))

    monkeypatch.setattr(salm_dflash.torch.distributed, "all_reduce", fake_all_reduce)
    base_loss = torch.tensor(2.0, requires_grad=True)
    selector_loss = torch.tensor(4.0, requires_grad=True)
    metrics = SimpleNamespace(
        loss=base_loss + 0.5 * selector_loss,
        loss_weight=torch.tensor(3.0),
        base_loss=base_loss,
        selector_loss=selector_loss,
        selector_loss_denominator=torch.tensor(2.0),
    )

    loss = module._globally_normalized_loss(metrics)
    loss.backward()

    assert loss.item() == pytest.approx(2.2)
    assert base_loss.grad.item() == pytest.approx(0.6)
    assert selector_loss.grad.item() == pytest.approx(0.25)


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
    log = Mock()
    monkeypatch.setattr(module, "log", log)
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
        valid_tokens=torch.tensor(12),
        valid_blocks=torch.tensor(4),
    )
    run_batch = Mock(side_effect=[salm_dflash.NoValidAnchorsError("skip"), metrics])
    monkeypatch.setattr(module, "_run_batch", run_batch)
    batch = {
        "dataset_a": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        "dataset_b": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
    }

    loss = module._training_step_batch(batch, batch_idx=0)

    torch.testing.assert_close(loss, metrics.loss)
    assert availability == [True, True, False]
    assert run_batch.call_count == 2


@pytest.mark.parametrize("configured", [Path("outputs/draft"), Path("/durable/draft")])
def test_train_end_resolves_relative_export_under_log_dir(monkeypatch, tmp_path, configured):
    module = salm_dflash.SALMDFlashModule(
        nn.Linear(1, 1),
        {"dflash": {"mask_token_id": 18, "output_dir": str(configured)}},
    )
    module.draft_model = Mock()
    module._trainer = SimpleNamespace(is_global_zero=True, log_dir=str(tmp_path / "experiment"))
    state_dict = {"weight": torch.tensor([1.0])}
    monkeypatch.setattr(salm_dflash, "_get_consolidated_model_state_dict", lambda _model: state_dict)

    module.on_train_end()

    expected = configured if configured.is_absolute() else tmp_path / "experiment" / configured
    module.draft_model.save_pretrained.assert_called_once_with(expected, state_dict=state_dict)


def test_training_step_returns_differentiable_zero_when_every_dataset_is_skipped(
    monkeypatch,
):
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

    loss = module._training_step_batch({"input_ids": torch.ones(1, 2, dtype=torch.long)}, batch_idx=0)

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert all(parameter.grad is None for parameter in module.draft_model.parameters())
    log.assert_any_call("train/dflash_skipped_step", 1.0, on_step=True, batch_size=1)
    log.assert_any_call("train/dflash_skip/no_valid_anchors", 1.0, on_step=True, batch_size=1)


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


def test_dflash2_validation_uses_separate_loss_denominators_and_selector_metrics(
    monkeypatch,
):
    module = salm_dflash.SALMDFlashModule(
        nn.Linear(1, 1),
        {
            "dflash": {
                "variant": "dflash2",
                "mask_token_id": 18,
                "selector_loss_weight": 0.5,
            }
        },
    )
    log = Mock()
    monkeypatch.setattr(module, "log", log)

    module._log_validation_metrics(torch.tensor([8.0, 4.0, 6.0, 3.0, 3.0, 6.0, 5.0, 2.0, 2.0, 4.0, 5.0]))

    log.assert_any_call("val/dflash_loss", torch.tensor(3.0), on_epoch=True)
    log.assert_any_call("val/dflash_selector_loss", torch.tensor(2.0), on_epoch=True)
    log.assert_any_call("val/dflash_accuracy", torch.tensor(0.5), on_epoch=True)
    log.assert_any_call("val/dflash_base_accept_len", torch.tensor(2.0), on_epoch=True)
    log.assert_any_call("val/dflash_candidate_recall", torch.tensor(5.0 / 6.0), on_epoch=True)
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


def test_rejects_invalid_label_source_and_chunk_size():
    with pytest.raises(ValueError, match="label_source"):
        salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18, "label_source": "teacher"}})
    with pytest.raises(ValueError, match="target_argmax_chunk_size"):
        salm_dflash.SALMDFlashModule(
            nn.Linear(1, 1),
            {"dflash": {"mask_token_id": 18, "target_argmax_chunk_size": 0}},
        )


def test_target_argmax_labels_are_causally_shifted_and_do_not_cross_packed_boundaries():
    module = salm_dflash.SALMDFlashModule(
        _TargetModel(),
        {"dflash": {"mask_token_id": 18, "label_source": "target_argmax", "target_argmax_chunk_size": 2}},
    )
    weight = torch.eye(4)
    module._materialize_frozen_lm_head = Mock(return_value=(weight, None))
    inputs = {
        "input_ids": torch.tensor([[10, 11, 20, 21]]),
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "qkv_format": "thd",
    }

    labels = module._build_target_argmax_labels(torch.eye(4), inputs)

    assert labels.tolist() == [[10, 0, 20, 2]]
    assert module._materialize_frozen_lm_head.call_count == 1


def test_target_argmax_hidden_capture_uses_final_norm_and_skips_full_logits():
    module = salm_dflash.SALMDFlashModule(
        _TargetModel(),
        {"dflash": {"mask_token_id": 18, "label_source": "target_argmax", "target_argmax_chunk_size": 2}},
    )
    module.target_layer_ids = [0, 2]
    module._materialize_frozen_lm_head = Mock(return_value=(torch.eye(4), None))
    inputs = {
        "input_ids": torch.tensor([[7, 8, 9]]),
        "input_embeddings": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    hidden, labels = module._target_hidden_states(inputs)

    assert hidden.shape == (1, 3, 8)
    # Final norm adds ten after the layer stack adds six, so every row's
    # dominant class is derived from the final-normalized representation.
    expected_top1 = (inputs["input_embeddings"] + 16).argmax(dim=-1)
    assert labels.tolist() == [[7, expected_top1[0, 0].item(), expected_top1[0, 1].item()]]
    assert module.target.llm.calls[-1]["compute_logits"] is False


def test_run_batch_forwards_target_argmax_labels(monkeypatch):
    module = salm_dflash.SALMDFlashModule(
        _BatchTarget(),
        {"dflash": {"mask_token_id": 990, "block_size": 2, "label_source": "target_argmax"}},
    )
    prepared = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "input_embeddings": torch.randn(1, 3, 4),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[False, True, True]]),
    }
    label_ids = torch.tensor([[10, 20, 21]])
    monkeypatch.setattr(module, "_prepare_batch", Mock(return_value=prepared))
    monkeypatch.setattr(
        module,
        "_target_hidden_states",
        Mock(return_value=(torch.randn(1, 3, 8), label_ids)),
    )
    monkeypatch.setattr(salm_dflash, "_has_valid_dflash_anchors", lambda *args, **kwargs: True)
    module.trainer_module = _CaptureDFlashTrainer()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))

    module._run_batch({})

    assert module.trainer_module.kwargs["label_ids"] is label_ids


@pytest.mark.parametrize("variant,init_path", [("dflash", "draft"), ("dflash2", None)])
def test_projection_only_requires_pretrained_dflash2(variant, init_path):
    with pytest.raises(ValueError, match="projection_only"):
        salm_dflash.SALMDFlashModule(
            _TrainerTarget(),
            {
                "dflash": {
                    "mask_token_id": 63,
                    "variant": variant,
                    "projection_only": True,
                    "init_from_pretrained": init_path,
                }
            },
        )


@pytest.mark.parametrize("block_size", [8, 16])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_projection_only_warm_start_updates_only_fc_and_resumes(tmp_path, block_size, activation_checkpointing):
    torch.manual_seed(7)
    target = _TrainerTarget().requires_grad_(False)
    cfg = {
        "variant": "dflash2",
        "mask_token_id": 63,
        "block_size": block_size,
        "draft_num_hidden_layers": 2,
        "target_layer_ids": [1, 4],
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": 8,
        "num_anchors": 2,
        "attention_backend": "sdpa",
        "projection_only": True,
        "init_from_pretrained": str(tmp_path / "warm"),
        "use_fused_linear_ce": False,
    }
    warm_config, _ = salm_dflash._build_draft_config(target.llm.config, cfg, 8, 63)
    warm_config._attn_implementation = "sdpa"
    initial = Qwen3DFlash2DraftModel(warm_config)
    initial.save_pretrained(tmp_path / "warm")
    initial_state = {name: value.clone() for name, value in initial.state_dict().items()}
    target_state = {name: value.clone() for name, value in target.state_dict().items()}

    def build_module():
        module = salm_dflash.SALMDFlashModule(target, {"dflash": cfg})
        config, module.target_layer_ids = salm_dflash._build_draft_config(target.llm.config, cfg, block_size, 63)
        config._attn_implementation = "sdpa"
        module.draft_model = module._initialize_draft_model(config, torch.float32)
        module._configure_projection_only()
        if activation_checkpointing:
            module.draft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        module.trainer_module = module._create_trainer_module()
        return module

    module = build_module()
    optimizer = module.configure_optimizers()
    assert module.draft_model.config.block_size == block_size
    assert {n for n, p in module.draft_model.named_parameters() if p.requires_grad} == {"fc.weight"}
    assert [p for g in optimizer.param_groups for p in g["params"]] == [module.draft_model.fc.weight]
    input_ids = torch.randint(0, 63, (1, 32))
    features = torch.randn(1, 32, 64)
    labels = (input_ids + 1) % 63

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        metrics = model.trainer_module(
            input_ids=input_ids, hidden_states=features, loss_mask=torch.ones(1, 32), label_ids=labels
        )
        metrics.loss.backward()
        assert torch.isfinite(model.draft_model.fc.weight.grad).all()
        assert model.draft_model.fc.weight.grad.abs().sum() > 0
        assert all(p.grad is None for n, p in model.draft_model.named_parameters() if n != "fc.weight")
        opt.step()

    step(module, optimizer)
    assert not torch.equal(module.draft_model.fc.weight, initial_state["fc.weight"])
    for name, value in module.draft_model.state_dict().items():
        if name != "fc.weight":
            torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, target_state[name], rtol=0, atol=0)
    assert all(p.grad is None for p in target.parameters())
    module.draft_model.save_pretrained(tmp_path / "export")
    exported = Qwen3DFlash2DraftModel.from_pretrained(tmp_path / "export")
    assert exported.config.block_size == block_size
    for name, value in module.draft_model.state_dict().items():
        torch.testing.assert_close(exported.state_dict()[name], value, rtol=0, atol=0)

    import copy

    model_state = copy.deepcopy(module.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    resumed = build_module()
    resumed.load_state_dict(model_state)
    resumed_optimizer = resumed.configure_optimizers()
    resumed_optimizer.load_state_dict(optimizer_state)
    step(module, optimizer)
    step(resumed, resumed_optimizer)
    for name, value in module.draft_model.state_dict().items():
        torch.testing.assert_close(resumed.draft_model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
def test_target_argmax_matches_dense_logits_with_padding_and_mask(chunk_size):
    torch.manual_seed(19)
    module = salm_dflash.SALMDFlashModule(
        _TargetModel(), {"dflash": {"mask_token_id": 18, "target_argmax_chunk_size": chunk_size}}
    )
    weight, bias = torch.randn(7, 4), torch.randn(7)
    module._materialize_frozen_lm_head = Mock(return_value=(weight, bias))
    hidden = torch.randn(2, 5, 4)
    ids = torch.tensor([[0, 0, 4, 3, 2], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=torch.bool)
    inputs = {"input_ids": ids, "attention_mask": mask, "loss_mask": loss_mask}
    labels = module._build_target_argmax_labels(hidden, inputs)
    dense = torch.nn.functional.linear(hidden, weight, bias).argmax(-1)
    expected = ids.clone()
    expected[0, 3:] = dense[0, 2:4]
    expected[1, 1] = dense[1, 0]
    expected[1, 3] = dense[1, 2]
    torch.testing.assert_close(labels, expected)
    torch.testing.assert_close(inputs["input_ids"], ids)


def test_target_argmax_final_norm_changes_label_and_cleans_hooks_on_error():
    target = _TargetModel()
    target.llm.norm = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        target.llm.norm.weight.copy_(torch.eye(4).flip(0))
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18, "label_source": "target_argmax"}})
    module.target_layer_ids = [0, 2]
    module._materialize_frozen_lm_head = Mock(return_value=(torch.eye(4), None))
    inputs = {
        "input_ids": torch.tensor([[7, 8, 9]]),
        "input_embeddings": torch.eye(4)[:3].unsqueeze(0),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    hidden, labels = module._target_hidden_states(inputs)
    assert labels.tolist() == [[7, 3, 2]]
    torch.testing.assert_close(hidden[..., :4], inputs["input_embeddings"] + 1)
    assert not any(m._forward_hooks for m in target.llm.modules())
    module._materialize_frozen_lm_head.side_effect = RuntimeError("head unavailable")
    with pytest.raises(RuntimeError, match="head unavailable"):
        module._target_hidden_states(inputs)
    assert not any(m._forward_hooks for m in target.llm.modules())


def _projection_only_fsdp_worker(rank, rendezvous, dtype, separate_target_units):
    import copy
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(7)
        target = _TrainerTarget().to(device=device, dtype=dtype).requires_grad_(False)
        cfg = {
            "variant": "dflash2",
            "mask_token_id": 63,
            "block_size": 16,
            "draft_num_hidden_layers": 2,
            "target_layer_ids": [1, 4],
            "conv_group_size": 8,
            "selector_rank": 16,
            "selector_top_k": 8,
            "num_anchors": 2,
            "attention_backend": "sdpa",
            "projection_only": True,
            "init_from_pretrained": "already-loaded",
        }
        module = salm_dflash.SALMDFlashModule(target, {"dflash": cfg})
        config, _ = salm_dflash._build_draft_config(target.llm.config, cfg, 16, 63)
        config._attn_implementation = "sdpa"
        module.draft_model = Qwen3DFlash2DraftModel(config).to(device=device, dtype=dtype)
        module._configure_projection_only()
        module.trainer_module = module._create_trainer_module().to(device)
        reference = copy.deepcopy(module)
        ref_optimizer = reference.configure_optimizers()
        initial = {name: value.clone() for name, value in reference.draft_model.state_dict().items()}
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        fully_shard(module.draft_model, mesh=mesh)
        optimizer = module.configure_optimizers()
        # Test real sharded target-head gathering with no successor labels on rank 0.
        if separate_target_units:
            fully_shard(target.llm.lm_head, mesh=mesh)
            fully_shard(target.llm.embed_tokens, mesh=mesh)
        fully_shard(target.llm, mesh=mesh)
        ids = torch.tensor([[1, 2, 3, 4]], device=device)
        hidden = torch.randn(1, 4, 32, device=device, dtype=dtype)
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "loss_mask": torch.full_like(ids, rank, dtype=torch.bool),
        }
        labels = module._build_target_argmax_labels(hidden, inputs)
        expected = ids.clone()
        if rank:
            expected[:, 1:] = reference.target.llm.lm_head(hidden[:, :-1]).argmax(-1)
        torch.testing.assert_close(labels, expected)
        # Compare sharded gradients/updates with the average of two unsharded references.
        # Root-owned heads exercise explicit argmax gathering above. Separate
        # FSDP units match production and also exercise the real frozen head
        # and embedding modules during draft forward/backward below.
        if not separate_target_units:
            object.__setattr__(module.trainer_module, "lm_head", reference.target.llm.lm_head)
            object.__setattr__(module.trainer_module, "embed_tokens", reference.target.llm.embed_tokens)
        torch.manual_seed(100 + rank)
        ids = torch.randint(0, 63, (1, 32), device=device)
        features = torch.randn(1, 32, 64, device=device, dtype=dtype)
        for model in (reference, module):
            torch.manual_seed(300 + rank)
            result = model.trainer_module(
                input_ids=ids, hidden_states=features, loss_mask=torch.ones_like(ids), label_ids=(ids + 1) % 63
            )
            result.loss.backward()
        dist.all_reduce(reference.draft_model.fc.weight.grad)
        reference.draft_model.fc.weight.grad.div_(2)
        grad = module.draft_model.fc.weight.grad.full_tensor()
        torch.testing.assert_close(
            grad,
            reference.draft_model.fc.weight.grad,
            rtol=2e-2 if dtype == torch.bfloat16 else 2e-4,
            atol=5e-4 if dtype == torch.bfloat16 else 2e-6,
        )
        optimizer.step()
        ref_optimizer.step()
        for name, parameter in module.draft_model.named_parameters():
            expected = reference.draft_model.state_dict()[name] if name == "fc.weight" else initial[name]
            torch.testing.assert_close(
                parameter.full_tensor(),
                expected,
                rtol=2e-4 if name == "fc.weight" else 0,
                atol=2e-6 if name == "fc.weight" else 0,
            )
        assert all(p.grad is None for p in target.parameters())
        export_path = Path(rendezvous.removeprefix("file://")).parent / "export"
        module.output_dir = str(export_path)
        module._trainer = SimpleNamespace(is_global_zero=rank == 0, log_dir=str(export_path.parent))
        module.on_train_end()
        dist.barrier()
        if rank == 0:
            restored = Qwen3DFlash2DraftModel.from_pretrained(export_path, dtype=dtype)
            assert restored.config.block_size == 16
            for name, value in restored.state_dict().items():
                expected = reference.draft_model.state_dict()[name] if name == "fc.weight" else initial[name]
                torch.testing.assert_close(
                    value.to(device),
                    expected,
                    rtol=2e-4 if name == "fc.weight" else 0,
                    atol=2e-6 if name == "fc.weight" else 0,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("separate_target_units", [False, True])
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_projection_only_fsdp_and_empty_rank_argmax(tmp_path, dtype, separate_target_units):
    torch.multiprocessing.spawn(
        _projection_only_fsdp_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", dtype, separate_target_units),
        nprocs=2,
    )


@pytest.mark.parametrize("label_source", ["ground_truth", "target_argmax"])
def test_padded_draft_inputs_exclude_left_padding(monkeypatch, label_source):
    module = salm_dflash.SALMDFlashModule(
        _BatchTarget(), {"dflash": {"mask_token_id": 18, "block_size": 2, "label_source": label_source}}
    )
    ids = torch.tensor([[0, 0, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25]])
    mask = torch.tensor([[False, False, False, True, True, True], [False, True, True, True, True, True]])
    inputs = {"input_ids": ids, "loss_mask": mask, "attention_mask": ids.ne(0)}
    hidden = torch.arange(48).view(2, 6, 4).float()
    labels = ids + 1
    monkeypatch.setattr(module, "_prepare_batch", lambda batch: inputs)
    monkeypatch.setattr(
        module, "_target_hidden_states", lambda batch: (hidden, labels) if label_source == "target_argmax" else hidden
    )
    module.trainer_module = _CaptureDFlashTrainer()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))
    module._run_batch({})
    seen = module.trainer_module.kwargs
    assert seen["input_ids"][0].tolist() == [10, 11, 12, 13, 0, 0]
    torch.testing.assert_close(seen["hidden_states"][0, :4], hidden[0, 2:])
    assert seen["loss_mask"][0].tolist() == [False, True, True, True, False, False]
    torch.testing.assert_close(seen["input_ids"][1], ids[1])
    if label_source == "target_argmax":
        torch.testing.assert_close(seen["label_ids"][0, :4], labels[0, 2:])
    # Target input remains unchanged; only the draft's padding layout changes.
    assert inputs["input_ids"][0].tolist() == [0, 0, 10, 11, 12, 13]


def test_globally_normalized_bf16_loss_reduces_weights_in_float32(monkeypatch):
    module = salm_dflash.SALMDFlashModule(nn.Linear(1, 1), {"dflash": {"mask_token_id": 18}})
    module._draft_dp_size = 32
    module._draft_dp_group = object()
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(salm_dflash.torch.distributed, "is_initialized", lambda: True)

    def reduce(value, **kwargs):
        assert value.dtype == torch.float32
        assert value.item() == 1001.0
        value.fill_(100003.0)

    monkeypatch.setattr(salm_dflash.torch.distributed, "all_reduce", reduce)
    value = torch.tensor(2.0, dtype=torch.bfloat16, requires_grad=True)
    loss = module._globally_normalized_loss(SimpleNamespace(loss=value, loss_weight=torch.tensor(1001.0)))
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(2 * 1001 * 32 / 100003)
    loss.backward()
    assert torch.isfinite(value.grad)


def test_configure_model_checks_frozen_target_backend_before_draft_construction():
    target = _BatchTarget()
    target.configure_model = Mock()
    target._validate_parallelism_compatibility = Mock(side_effect=ValueError("unsupported target backend"))
    module = salm_dflash.SALMDFlashModule(target, {"dflash": {"mask_token_id": 18}})
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(distributed_setup=None))
    with pytest.raises(ValueError, match="unsupported target backend"):
        module.configure_model()
    target._validate_parallelism_compatibility.assert_called_once_with(check_backward=False)
    assert module.draft_model is None


def test_padded_anchor_precheck_uses_the_draft_layout(monkeypatch):
    module = salm_dflash.SALMDFlashModule(_BatchTarget(), {"dflash": {"mask_token_id": 18, "block_size": 4}})
    ids = torch.tensor([[0, 0, 0, 0, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25, 26, 27]])
    mask = torch.tensor([[False, False, False, False, False, True, True, True], [False] * 8])
    inputs = {"input_ids": ids, "loss_mask": mask, "attention_mask": ids.ne(0)}
    monkeypatch.setattr(module, "_prepare_batch", lambda batch: inputs)
    monkeypatch.setattr(module, "_target_hidden_states", lambda batch: torch.zeros(2, 8, 4))
    module.trainer_module = _CaptureDFlashTrainer()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))
    assert module._run_batch({}) == "dflash-result"
    assert salm_dflash._has_valid_dflash_anchors(module.trainer_module.kwargs["loss_mask"], 4)


def test_scheduler_can_use_default_optimizer():
    module = salm_dflash.SALMDFlashModule(
        nn.Linear(1, 1),
        {
            "dflash": {
                "mask_token_id": 18,
                "lr": 0.01,
                "lr_scheduler": {
                    "_target_": "nemo.core.optim.lr_scheduler.CosineAnnealing",
                    "warmup_steps": 3,
                    "max_steps": 10,
                },
            }
        },
    )
    module.draft_model = nn.Linear(2, 2)
    configured = module.configure_optimizers()
    assert isinstance(configured, dict)
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    optimizer.step()
    configured["lr_scheduler"]["scheduler"].step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


@pytest.mark.parametrize("label_source", ["ground_truth", "target_argmax"])
def test_batched_draft_loss_is_invariant_to_padding_features(monkeypatch, label_source):
    torch.manual_seed(7)
    target = _TrainerTarget()
    cfg = {
        "variant": "dflash2",
        "mask_token_id": 63,
        "block_size": 4,
        "draft_num_hidden_layers": 2,
        "target_layer_ids": [1, 4],
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": 8,
        "num_anchors": 2,
        "attention_backend": "sdpa",
        "label_source": label_source,
    }
    module = salm_dflash.SALMDFlashModule(target, {"dflash": cfg})
    config, _ = salm_dflash._build_draft_config(target.llm.config, cfg, 4, 63)
    config._attn_implementation = "sdpa"
    module.draft_model = Qwen3DFlash2DraftModel(config)
    module.trainer_module = module._create_trainer_module()
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(moe_mesh=None))
    ids = torch.tensor([[0, 0, 0, 0, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25, 26, 27]])
    mask = torch.tensor([[False, False, False, False, False, True, True, True], [False] + [True] * 7])
    inputs = {"input_ids": ids, "loss_mask": mask, "attention_mask": ids.ne(0)}
    features = torch.randn(2, 8, 64)
    monkeypatch.setattr(module, "_prepare_batch", lambda batch: inputs)
    monkeypatch.setattr(
        module,
        "_target_hidden_states",
        lambda batch: (features, (ids + 1) % 63) if label_source == "target_argmax" else features,
    )
    torch.manual_seed(123)
    baseline = module._run_batch({}).loss
    features[0, :4] = torch.randn(4, 64) * 100
    torch.manual_seed(123)
    torch.testing.assert_close(module._run_batch({}).loss, baseline, rtol=0, atol=0)
    features[0, 4:] = torch.randn(4, 64) * 100
    torch.manual_seed(123)
    assert not torch.allclose(module._run_batch({}).loss, baseline)
