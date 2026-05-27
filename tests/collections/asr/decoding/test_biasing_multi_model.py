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

import pytest
import torch

from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
    GPUBiasingMultiModel,
    GPUBiasingMultiModelReference,
)
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
if hasattr(torch, "mps") and torch.mps.is_available():
    DEVICES.append(torch.device("mps"))

# Triton only works on CUDA, so only test use_triton=True if Triton is available
USE_TRITON_OPTIONS = [False, True] if TRITON_AVAILABLE else [False]


def create_boosting_model(phrases: list[str], tokenizer, device: torch.device) -> GPUBoostingTreeModel:
    """Helper to create boosting model from phrases"""
    cfg = BoostingTreeModelConfig(key_phrases_list=phrases, context_score=1.0)
    model = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)
    return model.to(device)


class TestGPUBiasingMultiModel:
    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_models_incremental(self, stt_en_conformer_transducer_small, device: torch.device):
        """Test adding 2 boosting models one-by-one, verifying arcs and states are correctly merged."""
        tokenizer = stt_en_conformer_transducer_small.tokenizer
        vocab_size = tokenizer.vocab_size

        # Create empty multi-model
        multi_model = GPUBiasingMultiModel(vocab_size=vocab_size).to(device)

        # Initially empty
        assert multi_model.num_models == 0
        assert multi_model.has_models() is False
        assert multi_model.num_states_total == 0
        assert multi_model.num_arcs_extended_total == 0

        # Create and add first model
        model1 = create_boosting_model(["hello", "world"], tokenizer, device)
        model_id1 = multi_model.add_model(model1, alpha=1.0)

        # Verify after first model
        assert model_id1 == 0
        assert multi_model.num_models == 1
        assert multi_model.has_models() is True
        assert multi_model.model2active[model_id1].item() is True
        assert multi_model.num_states_total == model1.num_states
        assert multi_model.num_arcs_extended_total == model1.num_arcs_extended
        assert multi_model.model2num_states[model_id1].item() == model1.num_states
        assert multi_model.model2num_arcs_extended[model_id1].item() == model1.num_arcs_extended

        # Create and add second model
        model2 = create_boosting_model(["test", "one", "two"], tokenizer, device)
        model_id2 = multi_model.add_model(model2, alpha=1.5)

        # Verify after second model
        assert model_id2 == 1
        assert multi_model.num_models == 2
        assert multi_model.has_models() is True
        assert multi_model.model2active[model_id1].item() is True
        assert multi_model.model2active[model_id2].item() is True
        assert multi_model.num_states_total == model1.num_states + model2.num_states
        assert multi_model.num_arcs_extended_total == model1.num_arcs_extended + model2.num_arcs_extended

        # Verify offsets
        assert multi_model.model2states_offset[model_id1].item() == 0
        assert multi_model.model2states_offset[model_id2].item() == model1.num_states
        assert multi_model.model2arcs_offset[model_id1].item() == 0
        assert multi_model.model2arcs_offset[model_id2].item() == model1.num_arcs_extended

        # Verify init states work
        init_states = multi_model.get_init_states(batch_size=4, bos=True)
        assert init_states.shape == (4,)
        assert init_states.device.type == device.type

    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.parametrize("device", DEVICES)
    def test_add_then_remove_model(self, stt_en_conformer_transducer_small, device: torch.device):
        """Test adding 2 models then removing the first one."""
        tokenizer = stt_en_conformer_transducer_small.tokenizer
        vocab_size = tokenizer.vocab_size

        multi_model = GPUBiasingMultiModel(vocab_size=vocab_size).to(device)

        # Add two models
        model1 = create_boosting_model(["alpha", "beta"], tokenizer, device)
        model2 = create_boosting_model(["gamma", "delta"], tokenizer, device)

        model_id1 = multi_model.add_model(model1, alpha=1.0)
        model_id2 = multi_model.add_model(model2, alpha=2.0)

        # Store counts before removal
        model1_num_states = model1.num_states
        model1_num_arcs = model1.num_arcs_extended
        total_states_before = multi_model.num_states_total
        total_arcs_before = multi_model.num_arcs_extended_total

        assert multi_model.model2active[model_id1].item() is True
        assert multi_model.model2active[model_id2].item() is True

        # Remove first model
        multi_model.remove_model(model_id1)

        # Verify removal
        assert model_id1 in multi_model.free_ids
        assert multi_model.model2active[model_id1].item() is False
        assert multi_model.model2active[model_id2].item() is True
        assert multi_model.model2alpha[model_id1].item() == 0.0
        assert multi_model.model2alpha[model_id2].item() == 2.0

        # Verify state/arc counts decreased
        assert multi_model.num_states_total == total_states_before - model1_num_states
        assert multi_model.num_arcs_extended_total == total_arcs_before - model1_num_arcs

        # Verify model2 offset updated (shifted left)
        assert multi_model.model2states_offset[model_id2].item() == 0
        assert multi_model.model2arcs_offset[model_id2].item() == 0

    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.parametrize("device", DEVICES)
    def test_model_id_reuse(self, stt_en_conformer_transducer_small, device):
        """Test that removed model IDs are reused."""
        tokenizer = stt_en_conformer_transducer_small.tokenizer
        vocab_size = tokenizer.vocab_size

        multi_model = GPUBiasingMultiModel(vocab_size=vocab_size).to(device)

        # Add model1 -> id=0
        model1 = create_boosting_model(["first"], tokenizer, device)
        model_id1 = multi_model.add_model(model1)
        assert model_id1 == 0

        # Add model2 -> id=1
        model2 = create_boosting_model(["second"], tokenizer, device)
        model_id2 = multi_model.add_model(model2)
        assert model_id2 == 1

        # Remove model1
        multi_model.remove_model(model_id1)
        assert model_id1 in multi_model.free_ids

        # Add model3 -> should reuse id=0
        model3 = create_boosting_model(["third"], tokenizer, device)
        model_id3 = multi_model.add_model(model3)
        assert model_id3 == model_id1  # Reused ID
        assert model_id1 not in multi_model.free_ids  # No longer free

        # Verify model3 is active
        assert multi_model.model2active[model_id3].item() is True

    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("use_triton", USE_TRITON_OPTIONS)
    @pytest.mark.parametrize("bos", [True, False])
    def test_advance_matches_reference(
        self, stt_en_conformer_transducer_small, device: torch.device, batch_size: int, use_triton, bos: bool
    ):
        """Verify GPUBiasingMultiModel produces same scores/states as reference implementation."""
        tokenizer = stt_en_conformer_transducer_small.tokenizer
        vocab_size = tokenizer.vocab_size

        # Create both implementations
        multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=use_triton).to(device)
        reference = GPUBiasingMultiModelReference(vocab_size=vocab_size).to(device)

        # Create boosting models with same phrases
        phrases1 = ["hello world", "test"]
        phrases2 = ["neural", "network"]

        model1_mm = create_boosting_model(phrases1, tokenizer, device)
        model1_ref = create_boosting_model(phrases1, tokenizer, device)
        model2_mm = create_boosting_model(phrases2, tokenizer, device)
        model2_ref = create_boosting_model(phrases2, tokenizer, device)

        # Add models to both with same alpha values
        alpha1, alpha2 = 1.0, 1.5
        model_id1_mm = multi_model.add_model(model1_mm, alpha=alpha1)
        model_id1_ref = reference.add_model(model1_ref, alpha=alpha1)
        model_id2_mm = multi_model.add_model(model2_mm, alpha=alpha2)
        model_id2_ref = reference.add_model(model2_ref, alpha=alpha2)

        assert model_id1_mm == model_id1_ref
        assert model_id2_mm == model_id2_ref

        # Get initial states
        states_mm = multi_model.get_init_states(batch_size=batch_size, bos=bos)
        states_ref = reference.get_init_states(batch_size=batch_size, bos=bos)

        # Create model_ids tensor with alternating models
        model_ids = torch.tensor(
            [model_id1_mm if i % 2 == 0 else model_id2_mm for i in range(batch_size)],
            dtype=torch.long,
            device=device,
        )

        # Call advance on both
        scores_mm, next_states_mm = multi_model.advance(states_mm, model_ids)
        scores_ref, next_states_ref = reference.advance(states_ref, model_ids)

        # Verify shapes
        assert scores_mm.shape == (batch_size, vocab_size)
        assert next_states_mm.shape == (batch_size, vocab_size)
        assert scores_ref.shape == (batch_size, vocab_size)
        assert next_states_ref.shape == (batch_size, vocab_size)

        # Verify scores and states match
        assert torch.allclose(
            scores_mm, scores_ref, atol=1e-5
        ), f"Scores mismatch: max diff = {(scores_mm - scores_ref).abs().max()}"
        assert torch.equal(next_states_mm, next_states_ref), "Next states mismatch"

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_multi_model(self, device: torch.device):
        """Test behavior of empty multi-model."""
        vocab_size = 100
        multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=False).to(device)

        # Verify empty state
        assert multi_model.has_models() is False
        assert multi_model.num_models == 0
        assert multi_model.num_states_total == 0
        assert multi_model.num_arcs_extended_total == 0

        # get_init_states should work and return START_STATE
        init_states = multi_model.get_init_states(batch_size=4, bos=True)
        assert init_states.shape == (4,)
        assert (init_states == GPUBiasingMultiModel.START_STATE).all()
