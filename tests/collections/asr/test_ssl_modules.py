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

# Import the quantizer module
_q_spec = importlib.util.spec_from_file_location(
    "quantizers",
    "nemo/collections/asr/modules/ssl_modules/quantizers.py",
    submodule_search_locations=[],
)
_q_mod = importlib.util.module_from_spec(_q_spec)
sys.modules.setdefault("quantizers", _q_mod)
_q_spec.loader.exec_module(_q_mod)
RandomProjectionVectorQuantizer = _q_mod.RandomProjectionVectorQuantizer


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


class TestRandomProjectionVectorQuantizer:
    @pytest.fixture(params=["cosine", "l2"])
    def quantizer(self, request):
        return RandomProjectionVectorQuantizer(
            feat_in=32,
            code_dim=8,
            num_classes=64,
            num_books=2,
            dist_fn=request.param,
        )

    def test_output_shapes(self, quantizer):
        B, D, T = 2, 32, 10
        x = torch.randn(B, D, T)
        xq, xid = quantizer(input_signal=x)
        assert xq.shape == (B, 8, T, 2)  # (B, code_dim, T, num_books)
        assert xid.shape == (B, T, 2)  # (B, T, num_books)

    def test_output_shapes_time_ahead(self):
        q = RandomProjectionVectorQuantizer(
            feat_in=32,
            code_dim=8,
            num_classes=64,
            num_books=2,
            dist_fn="l2",
            time_ahead=True,
        )
        B, T, D = 2, 10, 32
        x = torch.randn(B, T, D)
        xq, xid = q(input_signal=x)
        assert xq.shape == (B, T, 8, 2)  # (B, T, code_dim, num_books)
        assert xid.shape == (B, T, 2)

    def test_squeeze_single(self):
        q = RandomProjectionVectorQuantizer(
            feat_in=32,
            code_dim=8,
            num_classes=64,
            num_books=1,
            dist_fn="cosine",
            squeeze_single=True,
        )
        x = torch.randn(2, 32, 10)
        xq, xid = q(input_signal=x)
        assert xq.shape == (2, 8, 10)  # squeezed book dim
        assert xid.shape == (2, 10)

    def test_combine_time_steps(self):
        q = RandomProjectionVectorQuantizer(
            feat_in=16,
            code_dim=8,
            num_classes=64,
            num_books=1,
            dist_fn="l2",
            combine_time_steps=2,
        )
        x = torch.randn(2, 16, 10)  # T=10, will become T=5
        xq, xid = q(input_signal=x)
        assert xq.shape == (2, 8, 5, 1)
        assert xid.shape == (2, 5, 1)

    def test_codebooks_are_float32(self):
        q = RandomProjectionVectorQuantizer(feat_in=16, code_dim=8, num_classes=32, num_books=1)
        assert q.codebooks.dtype == torch.float32

    def test_l2_nearest_neighbor_correctness(self):
        """Verify L2 picks the closest codebook entry on a small example."""
        q = RandomProjectionVectorQuantizer(
            feat_in=4,
            code_dim=4,
            num_classes=3,
            num_books=1,
            dist_fn="l2",
            time_ahead=True,
            squeeze_single=True,
        )
        # Set projection to identity so x passes through unchanged
        with torch.no_grad():
            q.proj.weight.copy_(torch.eye(4))
            # Set codebook to known vectors (already normalized)
            cb = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ).unsqueeze(
                0
            )  # (1, 3, 4)
            q.codebooks.copy_(cb)

        # Input close to codebook entry 2 (z-axis)
        inp = torch.tensor([[[0.05, 0.05, 0.9, 0.0]]])  # (1, 1, 4)
        _, xid = q(input_signal=inp)
        assert xid.item() == 2

        # Input close to codebook entry 0 (x-axis)
        inp = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
        _, xid = q(input_signal=inp)
        assert xid.item() == 0

    def test_cosine_nearest_neighbor_correctness(self):
        """Verify cosine picks the most similar codebook entry."""
        q = RandomProjectionVectorQuantizer(
            feat_in=4,
            code_dim=4,
            num_classes=3,
            num_books=1,
            dist_fn="cosine",
            time_ahead=True,
            squeeze_single=True,
        )
        with torch.no_grad():
            q.proj.weight.copy_(torch.eye(4))
            cb = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ).unsqueeze(0)
            q.codebooks.copy_(cb)

        inp = torch.tensor([[[0.05, 0.05, 0.9, 0.0]]])
        _, xid = q(input_signal=inp)
        assert xid.item() == 2

        inp = torch.tensor([[[0.9, 0.1, 0.0, 0.0]]])
        _, xid = q(input_signal=inp)
        assert xid.item() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu(self, quantizer):
        quantizer = quantizer.cuda()
        x = torch.randn(2, 32, 10, device='cuda')
        xq, xid = quantizer(input_signal=x)
        assert xq.device.type == 'cuda'
        assert xid.device.type == 'cuda'
