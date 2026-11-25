# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.modules.magpietts_modules import (
    LocalTransformerHelper,
    SpecialAudioToken,
    clear_forbidden_logits,
    stacked_channel_order,
)
from nemo.collections.tts.modules.transformer_2501 import Transformer

NUM_CODEBOOKS = 4
STACKING = 2
NUM_CHANNELS = NUM_CODEBOOKS * STACKING
CODEBOOK_SIZE = 16
NUM_TOKENS = CODEBOOK_SIZE + len(SpecialAudioToken)
D_MODEL = 32
EOS_ID = SpecialAudioToken.get_index(SpecialAudioToken.AUDIO_EOS, CODEBOOK_SIZE)
MASK_ID = SpecialAudioToken.get_index(SpecialAudioToken.MASK_TOKEN, CODEBOOK_SIZE)
FRAME_MAJOR = stacked_channel_order(NUM_CODEBOOKS, STACKING, 'frame_major')


class _Modules(torch.nn.Module):
    """Local-transformer sub-modules as owned by EasyMagpie (one embedding and head per stacked channel)."""

    def __init__(self, is_causal=True):
        super().__init__()
        self.local_transformer = Transformer(
            n_layers=2,
            d_model=D_MODEL,
            d_ffn=D_MODEL * 4,
            sa_n_heads=2,
            kernel_size=1,
            is_causal=is_causal,
            max_length_causal_mask=NUM_CHANNELS + 2,
            use_learnable_pos_emb=True,
        )
        self.audio_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(NUM_TOKENS, D_MODEL) for _ in range(NUM_CHANNELS)]
        )
        self.audio_in_projection = torch.nn.Identity()
        self.local_transformer_in_projection = torch.nn.Identity()
        self.local_transformer_audio_out_projection = torch.nn.Identity()
        self.local_transformer_out_projections = torch.nn.ModuleList(
            [torch.nn.Linear(D_MODEL, NUM_TOKENS) for _ in range(NUM_CHANNELS)]
        )


def _helper(modules, channel_order=None, embeddings=None, heads=None):
    return LocalTransformerHelper(
        local_transformer=modules.local_transformer,
        audio_embeddings=modules.audio_embeddings if embeddings is None else embeddings,
        audio_in_projection=modules.audio_in_projection,
        local_transformer_in_projection=modules.local_transformer_in_projection,
        local_transformer_audio_out_projection=modules.local_transformer_audio_out_projection,
        local_transformer_out_projections=modules.local_transformer_out_projections if heads is None else heads,
        num_audio_codebooks=NUM_CHANNELS,
        frame_stacking_factor=1,
        audio_eos_id=EOS_ID,
        mask_token_id=MASK_ID,
        codebook_size=CODEBOOK_SIZE,
        channel_order=channel_order,
    )


def _permuted_reference(modules, order):
    """Default-order helper whose position i uses the embedding and head of channel order[i]."""
    embeddings = torch.nn.ModuleList([modules.audio_embeddings[c] for c in order])
    heads = torch.nn.ModuleList([modules.local_transformer_out_projections[c] for c in order])
    return _helper(modules, embeddings=embeddings, heads=heads)


def _canonical_logits(position_logits, order):
    """Reorder (B, T, C*V) logits from local-transformer position order to channel order."""
    by_position = position_logits.view(*position_logits.shape[:2], NUM_CHANNELS, NUM_TOKENS)
    by_channel = torch.empty_like(by_position)
    by_channel[:, :, list(order)] = by_position
    return by_channel.reshape(position_logits.shape)


@pytest.mark.unit
class TestStackedChannelOrder:
    def test_codebook_major_is_index_order(self):
        assert stacked_channel_order(8, 2, 'codebook_major') == tuple(range(16))

    def test_frame_major_predicts_all_codebooks_of_a_frame_first(self):
        # Channel c * S + f holds codebook c of frame f.
        assert stacked_channel_order(8, 2, 'frame_major') == (0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
        assert stacked_channel_order(2, 3, 'frame_major') == (0, 3, 1, 4, 2, 5)

    def test_unknown_order_raises(self):
        with pytest.raises(ValueError, match="Unsupported local_transformer_order"):
            stacked_channel_order(8, 2, 'interleaved')


@pytest.mark.unit
class TestLocalTransformerHelperChannelOrder:
    def test_default_order_matches_index_order(self):
        modules = _Modules().eval()
        assert _helper(modules).channel_order == tuple(range(NUM_CHANNELS))

    def test_rejects_non_permutation(self):
        modules = _Modules()
        with pytest.raises(ValueError, match="permutation"):
            _helper(modules, channel_order=[0, 0, 1, 2, 3, 4, 5, 6])

    def test_rejects_custom_order_with_frame_stacking(self):
        modules = _Modules()
        with pytest.raises(ValueError, match="frame_stacking_factor == 1"):
            LocalTransformerHelper(
                local_transformer=modules.local_transformer,
                audio_embeddings=modules.audio_embeddings,
                audio_in_projection=modules.audio_in_projection,
                local_transformer_in_projection=modules.local_transformer_in_projection,
                local_transformer_audio_out_projection=modules.local_transformer_audio_out_projection,
                local_transformer_out_projections=modules.local_transformer_out_projections,
                num_audio_codebooks=NUM_CODEBOOKS,
                frame_stacking_factor=STACKING,
                audio_eos_id=EOS_ID,
                mask_token_id=MASK_ID,
                codebook_size=CODEBOOK_SIZE,
                channel_order=[1, 0, 2, 3],
            )

    @pytest.mark.parametrize("targets_offset_by_one", [False, True])
    def test_frame_major_logits_equal_permuted_sequence(self, targets_offset_by_one):
        torch.manual_seed(0)
        modules = _Modules(is_causal=not targets_offset_by_one).eval()
        dec_out = torch.randn(2, 3, D_MODEL)
        targets = torch.randint(0, CODEBOOK_SIZE, (2, NUM_CHANNELS, 3))

        with torch.no_grad():
            frame_major = _helper(modules, channel_order=FRAME_MAJOR).compute_logits(
                dec_out, targets, targets_offset_by_one=targets_offset_by_one
            )
            reference = _permuted_reference(modules, FRAME_MAJOR).compute_logits(
                dec_out, targets[:, list(FRAME_MAJOR)], targets_offset_by_one=targets_offset_by_one
            )

        assert frame_major.shape == (2, 3, NUM_CHANNELS * NUM_TOKENS)
        torch.testing.assert_close(frame_major, _canonical_logits(reference, FRAME_MAJOR))

    def test_frame_major_logits_depend_only_on_earlier_channels(self):
        torch.manual_seed(1)
        modules = _Modules().eval()
        helper = _helper(modules, channel_order=FRAME_MAJOR)
        dec_out = torch.randn(1, 1, D_MODEL)
        targets = torch.randint(0, CODEBOOK_SIZE, (1, NUM_CHANNELS, 1))
        changed = targets.clone()
        last_channel = FRAME_MAJOR[-1]
        changed[:, last_channel] = (changed[:, last_channel] + 1) % CODEBOOK_SIZE

        with torch.no_grad():
            logits = helper.compute_logits(dec_out, targets).view(NUM_CHANNELS, NUM_TOKENS)
            logits_changed = helper.compute_logits(dec_out, changed).view(NUM_CHANNELS, NUM_TOKENS)

        # Only the input of the last predicted channel changed, so no channel's logits move.
        torch.testing.assert_close(logits, logits_changed)

        first_channel = FRAME_MAJOR[0]
        changed = targets.clone()
        changed[:, first_channel] = (changed[:, first_channel] + 1) % CODEBOOK_SIZE
        with torch.no_grad():
            logits_changed = helper.compute_logits(dec_out, changed).view(NUM_CHANNELS, NUM_TOKENS)
        torch.testing.assert_close(logits[first_channel], logits_changed[first_channel])
        later = list(FRAME_MAJOR[1:])
        assert not torch.allclose(logits[later], logits_changed[later])

    @pytest.mark.parametrize("use_kv_cache", [False, True])
    def test_frame_major_greedy_sampling_matches_teacher_forced_argmax(self, use_kv_cache):
        torch.manual_seed(2)
        modules = _Modules().eval()
        helper = _helper(modules, channel_order=FRAME_MAJOR)
        dec_out = torch.randn(3, D_MODEL)

        with torch.no_grad():
            codes = helper.sample_autoregressive(dec_out, temperature=0.0, topk=NUM_TOKENS, use_kv_cache=use_kv_cache)
            assert codes.shape == (3, NUM_CHANNELS, 1)
            modules.local_transformer.reset_cache(use_cache=False)
            logits = helper.compute_logits(dec_out.unsqueeze(1), codes)
        logits = clear_forbidden_logits(logits.view(3, NUM_CHANNELS, NUM_TOKENS), CODEBOOK_SIZE)
        torch.testing.assert_close(logits.argmax(dim=-1), codes.squeeze(-1))

    def test_frame_major_greedy_sampling_equals_permuted_sequence(self):
        torch.manual_seed(3)
        modules = _Modules().eval()
        dec_out = torch.randn(3, D_MODEL)

        with torch.no_grad():
            frame_major = _helper(modules, channel_order=FRAME_MAJOR).sample_autoregressive(
                dec_out, temperature=0.0, topk=NUM_TOKENS
            )
            reference = _permuted_reference(modules, FRAME_MAJOR).sample_autoregressive(
                dec_out, temperature=0.0, topk=NUM_TOKENS
            )

        torch.testing.assert_close(frame_major[:, list(FRAME_MAJOR)], reference)

    def test_frame_major_cfg_sampling_returns_conditional_batch(self):
        torch.manual_seed(4)
        modules = _Modules().eval()
        helper = _helper(modules, channel_order=FRAME_MAJOR)
        dec_out = torch.randn(4, D_MODEL)

        with torch.no_grad():
            codes = helper.sample_autoregressive(dec_out, temperature=0.0, topk=NUM_TOKENS, use_cfg=True, cfg_scale=1.0)
            conditional = helper.sample_autoregressive(dec_out[:2], temperature=0.0, topk=NUM_TOKENS)

        torch.testing.assert_close(codes, conditional)

    def test_frame_major_maskgit_returns_channel_order(self):
        torch.manual_seed(5)
        modules = _Modules(is_causal=False).eval()
        dec_out = torch.randn(2, D_MODEL)

        with torch.no_grad():
            torch.manual_seed(6)
            frame_major = _helper(modules, channel_order=FRAME_MAJOR).sample_maskgit(
                dec_out, temperature=1e-4, topk=NUM_TOKENS, n_steps=1
            )
            torch.manual_seed(6)
            reference = _permuted_reference(modules, FRAME_MAJOR).sample_maskgit(
                dec_out, temperature=1e-4, topk=NUM_TOKENS, n_steps=1
            )

        assert frame_major.shape == (2, NUM_CHANNELS, 1)
        torch.testing.assert_close(frame_major[:, list(FRAME_MAJOR)], reference)
