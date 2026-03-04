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

import importlib
import sys

import pytest
import torch

# Import the masking module directly to avoid triggering the full nemo.collections.asr import chain
_spec = importlib.util.spec_from_file_location(
    "masking",
    "nemo/collections/asr/modules/ssl_modules/masking.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
# Ensure nemo.core dependencies are available
sys.modules.setdefault("masking", _mod)
_spec.loader.exec_module(_mod)
RandomBlockMasking = _mod.RandomBlockMasking


class TestRandomBlockMasking:
    @pytest.fixture(params=[False, True], ids=["no_overlap", "overlap"])
    def masking_module(self, request):
        return RandomBlockMasking(
            feat_in=16,
            mask_prob=0.5,
            block_size=8,
            mask_value=0.0,
            allow_overlap=request.param,
        )

    def test_output_shapes(self, masking_module):
        B, D, T = 4, 16, 100
        feats = torch.randn(B, D, T)
        lengths = torch.tensor([100, 80, 60, 100])

        masked_feats, masks = masking_module(feats, lengths)

        assert masked_feats.shape == (B, D, T)
        assert masks.shape == (B, D, T)

    def test_mask_is_binary(self, masking_module):
        feats = torch.randn(4, 16, 100)
        lengths = torch.tensor([100, 80, 60, 100])

        _, masks = masking_module(feats, lengths)

        unique_vals = masks.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals)

    def test_mask_consistent_across_features(self, masking_module):
        """Mask should be the same for all feature dimensions at a given time step."""
        feats = torch.randn(4, 16, 100)
        lengths = torch.tensor([100, 80, 60, 100])

        _, masks = masking_module(feats, lengths)

        # All feature dims should have the same mask pattern
        for b in range(4):
            first_feat_mask = masks[b, 0, :]
            for d in range(1, 16):
                assert torch.equal(masks[b, d, :], first_feat_mask)

    def test_masked_positions_get_mask_value(self, masking_module):
        """Where mask is 1, features should equal mask_embedding."""
        feats = torch.randn(4, 16, 100)
        lengths = torch.tensor([100, 80, 60, 100])

        masked_feats, masks = masking_module(feats, lengths)

        mask_2d = masks[:, 0, :]  # (B, T) - same across features
        for b in range(4):
            masked_times = mask_2d[b].bool()
            if masked_times.any():
                # masked positions should have the mask_embedding value
                expected = masking_module.mask_embedding.unsqueeze(1).expand(-1, masked_times.sum())
                actual = masked_feats[b, :, masked_times]
                assert torch.allclose(actual, expected)

    def test_unmasked_positions_unchanged(self, masking_module):
        """Where mask is 0, features should be unchanged from input."""
        feats = torch.randn(4, 16, 100)
        lengths = torch.tensor([100, 80, 60, 100])

        masked_feats, masks = masking_module(feats, lengths)

        mask_2d = masks[:, 0, :]
        for b in range(4):
            unmasked_times = ~mask_2d[b].bool()
            if unmasked_times.any():
                assert torch.equal(masked_feats[b, :, unmasked_times], feats[b, :, unmasked_times])

    def test_some_positions_are_masked(self, masking_module):
        """With mask_prob=0.5, we should get some masked positions."""
        feats = torch.randn(4, 16, 200)
        lengths = torch.tensor([200, 200, 200, 200])

        _, masks = masking_module(feats, lengths)

        assert masks.sum() > 0

    def test_short_audio(self):
        """Audio shorter than block_size * max_mask_ratio should still work."""
        module = RandomBlockMasking(feat_in=16, block_size=48, mask_value=0.0)
        feats = torch.randn(2, 16, 20)
        lengths = torch.tensor([20, 15])

        masked_feats, masks = module(feats, lengths)

        assert masked_feats.shape == feats.shape
        assert masks.shape == feats.shape

    def test_batch_size_one(self, masking_module):
        feats = torch.randn(1, 16, 100)
        lengths = torch.tensor([100])

        masked_feats, masks = masking_module(feats, lengths)

        assert masked_feats.shape == feats.shape
        assert masks.shape == feats.shape

    def test_learnable_mask_embedding(self):
        """When mask_value is None, mask_embedding should be learnable."""
        module = RandomBlockMasking(feat_in=16, mask_value=None, freeze=False)
        feats = torch.randn(2, 16, 100)
        lengths = torch.tensor([100, 80])

        masked_feats, masks = module(feats, lengths)

        assert masked_feats.shape == feats.shape
        assert module.mask_embedding.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu(self, masking_module):
        masking_module = masking_module.cuda()
        feats = torch.randn(4, 16, 100, device='cuda')
        lengths = torch.tensor([100, 80, 60, 100], device='cuda')

        masked_feats, masks = masking_module(feats, lengths)

        assert masked_feats.device.type == 'cuda'
        assert masks.device.type == 'cuda'
