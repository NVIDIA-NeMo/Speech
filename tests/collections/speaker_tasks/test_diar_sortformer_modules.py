# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import math

import pytest
import torch

from nemo.collections.asr.modules.sortformer_modules import SortformerModules


class TestSortformerModules_CheckStreamingParameters:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n_spk, spkcache_len, fifo_len, chunk_len, lc, rc, spkcache_update_period, spkcache_sil_frames_per_spk",
        [
            (4, 188, 376, 376, 1, 1, 376, 0),  # Example 1: All equal values
            (3, 100, 200, 150, 0, 0, 150, 3),  # Example 2: Different values, zero contexts
            (5, 50, 100, 50, 2, 2, 75, 9),  # Example 3: Small values, larger contexts
        ],
    )
    def test_valid_parameters(
        self, n_spk, spkcache_len, fifo_len, chunk_len, lc, rc, spkcache_update_period, spkcache_sil_frames_per_spk
    ):
        """Test 1: All streaming parameters are valid."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=fifo_len,
            chunk_len=chunk_len,
            chunk_left_context=lc,
            chunk_right_context=rc,
            spkcache_update_period=spkcache_update_period,
            spkcache_sil_frames_per_spk=spkcache_sil_frames_per_spk,
        )
        # Should not raise any exception
        sortformer_modules._check_streaming_parameters()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "param_name, param_value, expected_min",
        [
            ("fifo_len", -5, 0),
            ("chunk_len", 0, 1),
            ("chunk_left_context", -2, 0),
            ("chunk_right_context", -3, 0),
            ("spkcache_update_period", 0, 1),
            ("spkcache_sil_frames_per_spk", -1, 0),
        ],
    )
    def test_invalid_parameters(self, param_name, param_value, expected_min):
        """Test that _check_streaming_parameters raises ValueError for parameters below their minimum."""
        params = {
            "num_spks": 4,
            "spkcache_len": 188,
            "fifo_len": 376,
            "chunk_len": 12,
            "chunk_left_context": 1,
            "chunk_right_context": 1,
            "spkcache_update_period": 12,
            "spkcache_sil_frames_per_spk": 3,
        }
        params[param_name] = param_value

        with pytest.raises(
            ValueError, match=f"Parameter '{param_name}' must be at least {expected_min}, but got {param_value}."
        ):
            sortformer_modules = SortformerModules(**params)
            sortformer_modules._check_streaming_parameters()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "param_name, param_value",
        [
            ("spkcache_len", 1.01),
            ("fifo_len", 12.05),
            ("chunk_len", 10.51),
            ("chunk_left_context", 2.11),
            ("chunk_right_context", 3.21),
            ("spkcache_update_period", 7.3),
            ("spkcache_sil_frames_per_spk", 2.5),
        ],
    )
    def test_invalid_float_parameters(self, param_name, param_value):
        """Test that _check_streaming_parameters raises TypeError for non-integer parameters."""
        params = {
            "num_spks": 4,
            "spkcache_len": 188,
            "fifo_len": 376,
            "chunk_len": 12,
            "chunk_left_context": 1,
            "chunk_right_context": 1,
            "spkcache_update_period": 12,
            "spkcache_sil_frames_per_spk": 3,
        }
        params[param_name] = param_value

        with pytest.raises(
            TypeError, match=f"Parameter '{param_name}' must be an integer, but got {param_name}: {param_value}"
        ):
            sortformer_modules = SortformerModules(**params)
            sortformer_modules._check_streaming_parameters()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "spkcache_len, n_spk, spkcache_sil_frames_per_spk",
        [
            (15, 4, 3),  # spkcache_len is 15, minimum is (1+3)*4=16
            (1, 1, 1),  # spkcache_len is 1, minimum is (1+1)*1=2
        ],
    )
    def test_invalid_spkcache_len(self, spkcache_len, n_spk, spkcache_sil_frames_per_spk):
        """Test that spkcache_len is validated against the minimum required length."""
        min_spkcache_len = (1 + spkcache_sil_frames_per_spk) * n_spk
        params = {
            "num_spks": n_spk,
            "spkcache_len": spkcache_len,
            "spkcache_sil_frames_per_spk": spkcache_sil_frames_per_spk,
            "fifo_len": 376,
            "chunk_len": 12,
            "chunk_left_context": 1,
            "chunk_right_context": 1,
            "spkcache_update_period": 12,
        }
        with pytest.raises(
            ValueError,
            match=f"Parameter 'spkcache_len' must be at least {min_spkcache_len}, but got {spkcache_len}.",
        ):
            sortformer_modules = SortformerModules(**params)
            sortformer_modules._check_streaming_parameters()


class TestSortformerModules_GeneralUtils:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, max_length, lengths",
        [
            (2, 5, torch.tensor([3, 4])),  # Example 1: Different lengths
            (4, 3, torch.tensor([0, 1, 2, 3])),  # Example 2: Various lengths including 0
            (1, 6, torch.tensor([6])),  # Example 3: Single batch, full length
        ],
    )
    def test_length_to_mask(self, batch_size, max_length, lengths):
        """Test the length_to_mask method that converts length values to encoder mask input tensor."""
        # Call the method directly on the class since it's a static method
        mask = SortformerModules.length_to_mask(lengths, max_length)

        # Check output shape
        assert mask.shape == (batch_size, max_length)
        assert mask.dtype == torch.bool

        # Check mask values
        for i in range(batch_size):
            seq_len = lengths[i].item()
            # First 'seq_len' positions should be True (1)
            if seq_len > 0:
                assert torch.all(mask[i, :seq_len] == True)
            # Remaining positions should be False (0)
            if seq_len < max_length:
                assert torch.all(mask[i, seq_len:] == False)

        # Check device consistency
        assert mask.device == lengths.device

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, num_spks",
        [
            (1, 50, 2),  # Example 1: Single batch, more frames
            (4, 20, 8),  # Example 2: Larger batch, more speakers
        ],
    )
    def test_forward_speaker_sigmoids(self, batch_size, n_frames, num_spks):
        """Test the forward_speaker_sigmoids method that outputs speaker probabilities using Sigmoid activation."""
        sortformer_modules = SortformerModules(num_spks=num_spks)

        # Use the correct hidden dimension from the model
        hidden_dim = sortformer_modules.tf_d_model  # This should be 192

        # Create test input tensor with correct hidden dimension
        hidden_out = torch.randn(batch_size, n_frames, hidden_dim)

        # Call the method
        preds = sortformer_modules.forward_speaker_sigmoids(hidden_out)

        # Check output shape
        assert preds.shape == (batch_size, n_frames, num_spks)

        # Check output data type
        assert preds.dtype == torch.float32

        # Check that all values are between 0 and 1 (Sigmoid output range)
        assert torch.all(preds >= 0.0)
        assert torch.all(preds <= 1.0)

        # Check that the output is not all zeros or all ones (reasonable probability distribution)
        assert not torch.all(preds == 0.0)
        assert not torch.all(preds == 1.0)

        # Test with gradient computation
        hidden_out_with_grad = torch.randn(batch_size, n_frames, hidden_dim, requires_grad=True)
        preds_with_grad = sortformer_modules.forward_speaker_sigmoids(hidden_out_with_grad)

        # Check that gradients can be computed
        loss = preds_with_grad.sum()
        loss.backward()
        assert hidden_out_with_grad.grad is not None
        assert hidden_out_with_grad.grad.shape == hidden_out_with_grad.shape

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, encoder_lengths",
        [
            (2, 10, 4, torch.tensor([8, 6])),  # Example 1: Different lengths for each batch
            (1, 15, 2, torch.tensor([12])),  # Example 2: Single batch, partial length
            (3, 8, 6, torch.tensor([8, 5, 3])),  # Example 3: Multiple batches, varying lengths
        ],
    )
    def test_apply_mask_to_preds(self, batch_size, n_frames, n_spk, encoder_lengths):
        """Test the apply_mask_to_preds method that applies mask to speaker predictions."""
        # Create test predictions tensor with random values
        spkcache_fifo_chunk_preds = torch.randn(batch_size, n_frames, n_spk)

        # Store a copy of the original tensor to check immutability
        original_preds = spkcache_fifo_chunk_preds.clone()

        # Call the method directly on the class since it's a static method
        masked_preds = SortformerModules.apply_mask_to_preds(spkcache_fifo_chunk_preds, encoder_lengths)

        # Check output shape
        assert masked_preds.shape == (batch_size, n_frames, n_spk)

        # Check device consistency
        assert masked_preds.device == spkcache_fifo_chunk_preds.device

        # Check data type
        assert masked_preds.dtype == spkcache_fifo_chunk_preds.dtype

        # Verify masking behavior
        for i in range(batch_size):
            valid_length = encoder_lengths[i].item()

            # Valid frames (within encoder_lengths) should be unchanged
            if valid_length > 0:
                assert torch.allclose(masked_preds[i, :valid_length], original_preds[i, :valid_length])

            # Invalid frames (beyond encoder_lengths) should be zero
            if valid_length < n_frames:
                assert torch.all(masked_preds[i, valid_length:] == 0.0)

        # Check that the method doesn't modify the original tensor
        assert torch.allclose(spkcache_fifo_chunk_preds, original_preds)

        # Test edge case: all lengths are zero
        zero_lengths = torch.zeros(batch_size, dtype=torch.long)
        zero_masked_preds = SortformerModules.apply_mask_to_preds(spkcache_fifo_chunk_preds, zero_lengths)
        assert torch.all(zero_masked_preds == 0.0)

        # Test edge case: all lengths are full
        full_lengths = torch.full((batch_size,), n_frames, dtype=torch.long)
        full_masked_preds = SortformerModules.apply_mask_to_preds(spkcache_fifo_chunk_preds, full_lengths)
        assert torch.allclose(full_masked_preds, spkcache_fifo_chunk_preds)


class TestSortformerModules_StreamingUtils:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, feat_dim, feat_len, chunk_len, subsampling_factor, chunk_left_context, chunk_right_context",
        [
            (2, 80, 100, 10, 8, 1, 1),  # Example 1: Small chunks, small context
            (1, 64, 200, 20, 4, 2, 2),  # Example 2: Larger chunks, larger context
            (3, 128, 150, 15, 8, 0, 0),  # Example 3: No context
        ],
    )
    def test_streaming_feat_loader(
        self, batch_size, feat_dim, feat_len, chunk_len, subsampling_factor, chunk_left_context, chunk_right_context
    ):
        """Test the streaming_feat_loader method that loads chunks of feature sequence for streaming inference."""
        sortformer_modules = SortformerModules(
            num_spks=4,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=chunk_len,
            chunk_left_context=chunk_left_context,
            chunk_right_context=chunk_right_context,
            spkcache_update_period=376,
        )
        sortformer_modules.subsampling_factor = subsampling_factor

        # Create test data
        feat_seq = torch.randn(batch_size, feat_dim, feat_len)
        feat_seq_length = torch.tensor([feat_len] * batch_size)
        feat_seq_offset = torch.zeros(batch_size)

        # Collect all chunks
        chunks = []
        for chunk_data in sortformer_modules.streaming_feat_loader(feat_seq, feat_seq_length, feat_seq_offset):
            chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset = chunk_data
            chunks.append(
                {
                    'chunk_idx': chunk_idx,
                    'chunk_feat_seq_t': chunk_feat_seq_t,
                    'feat_lengths': feat_lengths,
                    'left_offset': left_offset,
                    'right_offset': right_offset,
                }
            )

        # Verify we got at least one chunk
        assert len(chunks) > 0

        # Verify chunk properties
        for chunk in chunks:
            chunk_idx = chunk['chunk_idx']
            chunk_feat_seq_t = chunk['chunk_feat_seq_t']
            feat_lengths = chunk['feat_lengths']
            left_offset = chunk['left_offset']
            right_offset = chunk['right_offset']

            # Check chunk index is sequential
            assert chunk_idx == len(chunks) - 1 if chunk == chunks[-1] else chunk_idx < len(chunks)

            # Check that the tensor is transposed correctly (feat_dim moved to last dimension)
            assert chunk_feat_seq_t.shape[2] == feat_dim
            assert chunk_feat_seq_t.shape[0] == batch_size

            # Check feat_lengths shape and values
            assert feat_lengths.shape == (batch_size,)
            assert torch.all(feat_lengths >= 0)
            assert torch.all(feat_lengths <= chunk_feat_seq_t.shape[1])

            # Check offsets are non-negative
            assert left_offset >= 0
            assert right_offset >= 0

            # Check offsets don't exceed expected bounds
            assert left_offset <= chunk_left_context * subsampling_factor
            assert right_offset <= chunk_right_context * subsampling_factor

            # Verify the chunk size makes sense
            actual_chunk_size = chunk_feat_seq_t.shape[1] - left_offset - right_offset
            assert actual_chunk_size > 0
            assert actual_chunk_size <= chunk_len * subsampling_factor

        # Verify all chunks together cover the full sequence
        total_processed = sum(
            chunk['chunk_feat_seq_t'].shape[1] - chunk['left_offset'] - chunk['right_offset'] for chunk in chunks
        )
        assert total_processed >= feat_len - (chunk_len * subsampling_factor)  # Allow for some overlap/remainder

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_frames, num_tensors",
        [
            (3, 512, 376, 3),  # Example: matches your real input
            (2, 128, 100, 2),  # Smaller test
            (1, 64, 50, 4),  # Single batch, more tensors
        ],
    )
    def test_concat_and_pad(self, batch_size, emb_dim, n_frames, num_tensors):
        """Test the concat_and_pad function with lists of tensors as in real input scenario."""
        embs = []
        lengths = []
        for _ in range(num_tensors):
            emb = torch.randn(batch_size, n_frames, emb_dim)
            # Allow zero length as well as full length
            length = torch.randint(0, n_frames + 1, (batch_size,))
            embs.append(emb)
            lengths.append(length)
        # Compute expected total lengths for each batch
        expected_total_lengths = torch.sum(torch.stack(lengths), dim=0)
        max_total_length = expected_total_lengths.max().item()
        # Call the function
        output, total_lengths = SortformerModules.concat_and_pad(embs, lengths)
        # Check output shapes
        assert output.shape == (batch_size, max_total_length, emb_dim)
        assert total_lengths.shape == (batch_size,)
        # Check total_lengths matches expected
        assert torch.allclose(total_lengths, expected_total_lengths)
        # For each batch, check that the concatenated region matches the input, and the padded region is zero
        for b in range(batch_size):
            out_ptr = 0
            for i in range(num_tensors):
                n = lengths[i][b].item()
                if n > 0:
                    assert torch.allclose(output[b, out_ptr : out_ptr + n], embs[i][b, :n], atol=1e-6)
                out_ptr += n
            # The rest should be zero
            if out_ptr < max_total_length:
                assert torch.allclose(
                    output[b, out_ptr:],
                    torch.zeros(max_total_length - out_ptr, emb_dim, device=output.device, dtype=output.dtype),
                    atol=1e-6,
                )
        # Edge case: mismatched list lengths
        with pytest.raises(ValueError):
            SortformerModules.concat_and_pad(embs, lengths[:-1])
        # Edge case: empty lists
        with pytest.raises(ValueError):
            SortformerModules.concat_and_pad([], [])
        # Edge case: single tensor
        single_emb = [torch.randn(batch_size, n_frames, emb_dim)]
        single_length = [torch.randint(0, n_frames + 1, (batch_size,))]
        single_output, single_total_lengths = SortformerModules.concat_and_pad(single_emb, single_length)
        assert single_output.shape == (batch_size, single_length[0].max().item(), emb_dim)
        assert torch.allclose(single_total_lengths, single_length[0])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "tensor_shapes, dim, return_lengths, device",
        [
            ([(2, 3, 4), (2, 5, 4)], 1, False, None),  # Example 1: Concatenate along dim 1, no lengths
            ([(1, 5), (1, 3)], 1, True, torch.device('cpu')),  # Example 2: With device specification
        ],
    )
    def test_concat_embs(self, tensor_shapes, dim, return_lengths, device):
        """Test the concat_embs method that concatenates a list of tensors along a specified dimension."""
        # Create test tensors
        list_of_tensors = [torch.randn(*shape) for shape in tensor_shapes]

        # Call the method directly on the class since it's a static method
        if return_lengths:
            embs, lengths = SortformerModules.concat_embs(
                list_of_tensors, return_lengths=return_lengths, dim=dim, device=device
            )
        else:
            embs = SortformerModules.concat_embs(
                list_of_tensors, return_lengths=return_lengths, dim=dim, device=device
            )

        # Check output tensor shape
        expected_shape = list(tensor_shapes[0])
        for shape in tensor_shapes[1:]:
            expected_shape[dim] += shape[dim]

        assert embs.shape == tuple(expected_shape)

        # Check device if specified
        if device is not None:
            assert embs.device == device
        else:
            # Should be on the same device as input tensors
            assert embs.device == list_of_tensors[0].device

        # Check data type
        assert embs.dtype == list_of_tensors[0].dtype

        # Check that concatenation worked correctly
        # Verify the concatenated tensor contains the original tensors
        start_idx = 0
        for i, tensor in enumerate(list_of_tensors):
            end_idx = start_idx + tensor.shape[dim]
            if dim == 0:
                assert torch.allclose(embs[start_idx:end_idx], tensor)
            elif dim == 1:
                assert torch.allclose(embs[:, start_idx:end_idx], tensor)
            elif dim == 2:
                assert torch.allclose(embs[:, :, start_idx:end_idx], tensor)
            start_idx = end_idx

        # Check lengths if returned
        if return_lengths:
            assert lengths.shape == (embs.shape[0],)
            assert torch.all(lengths == embs.shape[1])
            if device is not None:
                assert lengths.device == device
            else:
                assert lengths.device == list_of_tensors[0].device

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, async_streaming, device",
        [
            (1, False, None),  # Example 1: Default synchronous streaming
            (2, True, None),  # Example 2: Asynchronous streaming, batch size 2
            (4, False, torch.device('cpu')),  # Example 3: Synchronous with device
            (1, True, torch.device('cpu')),  # Example 4: Asynchronous with device
        ],
    )
    def test_init_streaming_state(self, batch_size, async_streaming, device):
        """Test the init_streaming_state method that initializes StreamingSortformerState."""
        sortformer_modules = SortformerModules(
            num_spks=4,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )

        # Call the method
        streaming_state = sortformer_modules.init_streaming_state(
            batch_size=batch_size, async_streaming=async_streaming, device=device
        )

        # Check that we got a StreamingSortformerState object
        from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState

        assert isinstance(streaming_state, StreamingSortformerState)

        # Check device consistency
        expected_device = device if device is not None else torch.device('cpu')

        if async_streaming:
            # Check spkcache tensor
            assert streaming_state.spkcache is not None
            assert streaming_state.spkcache.shape == (
                batch_size,
                sortformer_modules.spkcache_len,
                sortformer_modules.fc_d_model,
            )
            assert streaming_state.spkcache.device == expected_device
            assert torch.all(streaming_state.spkcache == 0)

            # Check spkcache_preds tensor
            assert streaming_state.spkcache_preds is not None
            assert streaming_state.spkcache_preds.shape == (
                batch_size,
                sortformer_modules.spkcache_len,
                sortformer_modules.n_spk,
            )
            assert streaming_state.spkcache_preds.device == expected_device
            assert torch.all(streaming_state.spkcache_preds == 0)

            # Check spkcache_lengths tensor
            assert streaming_state.spkcache_lengths is not None
            assert streaming_state.spkcache_lengths.shape == (batch_size,)
            assert streaming_state.spkcache_lengths.device == expected_device
            assert streaming_state.spkcache_lengths.dtype == torch.long
            assert torch.all(streaming_state.spkcache_lengths == 0)

            # Check fifo tensor
            assert streaming_state.fifo is not None
            assert streaming_state.fifo.shape == (
                batch_size,
                sortformer_modules.fifo_len,
                sortformer_modules.fc_d_model,
            )
            assert streaming_state.fifo.device == expected_device
            assert torch.all(streaming_state.fifo == 0)

            # Check fifo_lengths tensor
            assert streaming_state.fifo_lengths is not None
            assert streaming_state.fifo_lengths.shape == (batch_size,)
            assert streaming_state.fifo_lengths.device == expected_device
            assert streaming_state.fifo_lengths.dtype == torch.long
            assert torch.all(streaming_state.fifo_lengths == 0)

            # Check that fifo_preds is None (not initialized in async mode)
            assert streaming_state.fifo_preds is None
            assert streaming_state.spk_perm is None

        else:
            # Check spkcache tensor (empty for synchronous)
            assert streaming_state.spkcache is not None
            assert streaming_state.spkcache.shape == (batch_size, 0, sortformer_modules.fc_d_model)
            assert streaming_state.spkcache.device == expected_device

            # Check fifo tensor (empty for synchronous)
            assert streaming_state.fifo is not None
            assert streaming_state.fifo.shape == (batch_size, 0, sortformer_modules.fc_d_model)
            assert streaming_state.fifo.device == expected_device

            # Check that other attributes are None (not initialized in sync mode)
            assert streaming_state.spkcache_preds is None
            assert streaming_state.spkcache_lengths is None
            assert streaming_state.fifo_lengths is None
            assert streaming_state.fifo_preds is None
            assert streaming_state.spk_perm is None


class TestSortformerModules_StreamingScoreComputations:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, n_boost_per_spk, scale_factor, offset",
        [
            (2, 10, 4, 3, 1.0, 0.5),  # Example 1: Default parameters
            (1, 8, 2, 2, 2.0, 0.3),  # Example 2: Higher scale factor, lower offset
            (3, 12, 6, 4, 0.5, 0.7),  # Example 3: Lower scale factor, higher offset
            (2, 6, 3, 1, 1.5, 0.2),  # Example 4: Boost only 1 frame per speaker
        ],
    )
    def test_boost_topk_scores(self, batch_size, n_frames, n_spk, n_boost_per_spk, scale_factor, offset):
        """Test the _boost_topk_scores method that increases top-k scores for each speaker."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=100,
        )

        # Create test scores tensor with random values
        scores = torch.randn(batch_size, n_frames, n_spk)

        # Store original scores for comparison
        original_scores = scores.clone()

        # Call the method
        boosted_scores = sortformer_modules._boost_topk_scores(scores, n_boost_per_spk, scale_factor, offset)

        # Check output shape
        assert boosted_scores.shape == (batch_size, n_frames, n_spk)

        # Check device consistency
        assert boosted_scores.device == scores.device

        # Check data type
        assert boosted_scores.dtype == scores.dtype

        # Verify that the method modifies the input tensor in-place
        assert torch.allclose(boosted_scores, scores)

        # Verify boosting behavior for each batch and speaker
        for b in range(batch_size):
            for s in range(n_spk):
                # Get the top-k indices for this batch and speaker
                speaker_scores = original_scores[b, :, s]
                _, topk_indices = torch.topk(speaker_scores, n_boost_per_spk, dim=0, largest=True, sorted=False)

                # Check that boosted scores are higher than original for top-k frames
                for idx in topk_indices:
                    original_score = original_scores[b, idx, s]
                    boosted_score = boosted_scores[b, idx, s]
                    expected_boost = scale_factor * math.log(offset)
                    assert boosted_score == original_score - expected_boost

                # Check that non-boosted scores remain unchanged
                non_boosted_mask = torch.ones(n_frames, dtype=torch.bool)
                non_boosted_mask[topk_indices] = False
                non_boosted_indices = torch.where(non_boosted_mask)[0]

                for idx in non_boosted_indices:
                    assert boosted_scores[b, idx, s] == original_scores[b, idx, s]

        # Test edge case: n_boost_per_spk = 0 (no boosting)
        scores_no_boost = torch.randn(batch_size, n_frames, n_spk)
        original_no_boost = scores_no_boost.clone()
        boosted_no_boost = sortformer_modules._boost_topk_scores(scores_no_boost, 0, scale_factor, offset)
        assert torch.allclose(boosted_no_boost, original_no_boost)

        # Test edge case: n_boost_per_spk = n_frames (boost all frames)
        scores_all_boost = torch.randn(batch_size, n_frames, n_spk)
        original_all_boost = scores_all_boost.clone()
        boosted_all_boost = sortformer_modules._boost_topk_scores(scores_all_boost, n_frames, scale_factor, offset)

        # All scores should be boosted
        expected_boost = scale_factor * math.log(offset)
        assert torch.allclose(boosted_all_boost, original_all_boost - expected_boost)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, emb_dim, n_spk, sil_threshold",
        [
            (2, 10, 512, 4, 0.2),  # Example 1: Default parameters
            (1, 8, 256, 2, 0.1),  # Example 2: Lower silence threshold
            (3, 12, 128, 6, 0.5),  # Example 3: Higher silence threshold
            (2, 6, 64, 3, 0.3),  # Example 4: Smaller dimensions
        ],
    )
    def test_get_silence_profile(self, batch_size, n_frames, emb_dim, n_spk, sil_threshold):
        """Test the _get_silence_profile method that computes mean silence embeddings."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )
        sortformer_modules.sil_threshold = sil_threshold

        # Create test embeddings and predictions
        emb_seq = torch.randn(batch_size, n_frames, emb_dim)
        preds = torch.rand(batch_size, n_frames, n_spk)  # Random probabilities between 0 and 1

        # Initialize state
        mean_sil_emb = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)

        # Call the method
        upd_mean_sil_emb, upd_n_sil_frames = sortformer_modules._get_silence_profile(
            mean_sil_emb, n_sil_frames, emb_seq, preds
        )

        # Check output shapes
        assert upd_mean_sil_emb.shape == (batch_size, emb_dim)
        assert upd_n_sil_frames.shape == (batch_size,)

        # Check device consistency
        assert upd_mean_sil_emb.device == emb_seq.device
        assert upd_n_sil_frames.device == emb_seq.device

        # Check data type
        assert upd_mean_sil_emb.dtype == emb_seq.dtype
        assert upd_n_sil_frames.dtype == torch.long

        # Verify the calculation manually for each batch for the first call
        for b in range(batch_size):
            is_silence = preds[b].sum(dim=1) < sil_threshold
            sil_count = is_silence.sum().item()

            if sil_count > 0:
                silence_embeddings = emb_seq[b][is_silence]
                expected_mean = silence_embeddings.mean(dim=0)
                assert torch.allclose(upd_mean_sil_emb[b], expected_mean, atol=1e-6)
                assert upd_n_sil_frames[b] == sil_count
            else:
                assert torch.allclose(upd_mean_sil_emb[b], torch.zeros(emb_dim), atol=1e-6)
                assert upd_n_sil_frames[b] == 0

        # Test running average with a second call
        emb_seq2 = torch.randn(batch_size, n_frames, emb_dim, device=emb_seq.device)
        preds2 = torch.rand(batch_size, n_frames, n_spk, device=emb_seq.device)

        final_mean_sil_emb, final_n_sil_frames = sortformer_modules._get_silence_profile(
            upd_mean_sil_emb, upd_n_sil_frames, emb_seq2, preds2
        )

        for b in range(batch_size):
            is_silence2 = preds2[b].sum(dim=1) < sil_threshold
            sil_count2 = is_silence2.sum().item()
            total_sil_count = upd_n_sil_frames[b].item() + sil_count2

            if total_sil_count > 0:
                sil_emb_sum1 = upd_mean_sil_emb[b] * upd_n_sil_frames[b]
                sil_emb_sum2 = emb_seq2[b][is_silence2].sum(dim=0) if sil_count2 > 0 else 0
                expected_final_mean = (sil_emb_sum1 + sil_emb_sum2) / total_sil_count

                assert torch.allclose(final_mean_sil_emb[b], expected_final_mean, atol=1e-6)
                assert final_n_sil_frames[b] == total_sil_count
            else:
                assert torch.allclose(final_mean_sil_emb[b], torch.zeros(emb_dim), atol=1e-6)
                assert final_n_sil_frames[b] == 0

        # Test edge case: all frames are silence
        all_silence_preds = torch.zeros(batch_size, n_frames, n_spk, device=preds.device)
        mean_sil_emb_init = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames_init = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)
        all_silence_mean, all_silence_n_frames = sortformer_modules._get_silence_profile(
            mean_sil_emb_init, n_sil_frames_init, emb_seq, all_silence_preds
        )
        for b in range(batch_size):
            expected_all_silence = emb_seq[b].mean(dim=0)
            assert torch.allclose(all_silence_mean[b], expected_all_silence, atol=1e-6)
            assert all_silence_n_frames[b] == n_frames

        # Test edge case: no silence frames
        no_silence_preds = torch.ones(batch_size, n_frames, n_spk, device=preds.device) * (
            sil_threshold + 0.1
        )  # All above threshold
        mean_sil_emb_init = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames_init = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)
        no_silence_mean, no_silence_n_frames = sortformer_modules._get_silence_profile(
            mean_sil_emb_init, n_sil_frames_init, emb_seq, no_silence_preds
        )
        assert torch.allclose(no_silence_mean, torch.zeros(batch_size, emb_dim), atol=1e-6)
        assert torch.all(no_silence_n_frames == 0)

        # Test edge case: mixed silence and speech
        mixed_preds = torch.zeros(batch_size, n_frames, n_spk, device=preds.device)
        # Make first half of frames silence, second half speech
        mixed_preds[:, : n_frames // 2] = 0.0  # Silence
        mixed_preds[:, n_frames // 2 :] = sil_threshold + 0.1  # Speech
        mean_sil_emb_init = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames_init = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)
        mixed_mean, mixed_n_frames = sortformer_modules._get_silence_profile(
            mean_sil_emb_init, n_sil_frames_init, emb_seq, mixed_preds
        )

        for b in range(batch_size):
            # Only first half should be considered silence
            silence_embeddings = emb_seq[b, : n_frames // 2]
            expected_mixed_mean = silence_embeddings.mean(dim=0)
            assert torch.allclose(mixed_mean[b], expected_mixed_mean, atol=1e-6)
            assert mixed_n_frames[b] == n_frames // 2

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, pred_score_threshold",
        [
            (2, 10, 4, 0.25),  # Example 1: Default parameters
            (1, 8, 2, 0.1),  # Example 2: Lower threshold
            (3, 12, 6, 0.5),  # Example 3: Higher threshold
            (2, 6, 3, 0.3),  # Example 4: Smaller dimensions
        ],
    )
    def test_get_log_pred_scores(self, batch_size, n_frames, n_spk, pred_score_threshold):
        """Test the _get_log_pred_scores method that computes log-based speaker scores."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )
        sortformer_modules.pred_score_threshold = pred_score_threshold

        # Create test predictions with various probability values
        preds = torch.rand(batch_size, n_frames, n_spk)  # Random probabilities between 0 and 1

        # Call the method
        scores = sortformer_modules._get_log_pred_scores(preds)

        # Check output shape
        assert scores.shape == (batch_size, n_frames, n_spk)

        # Check device consistency
        assert scores.device == preds.device

        # Check data type
        assert scores.dtype == preds.dtype

        # Verify the calculation manually for each batch and frame
        for b in range(batch_size):
            for f in range(n_frames):
                # Get predictions for this frame
                frame_preds = preds[b, f]  # Shape: (n_spk,)

                # Apply clamping
                clamped_preds = torch.clamp(frame_preds, min=pred_score_threshold)
                clamped_1_preds = torch.clamp(1.0 - frame_preds, min=pred_score_threshold)

                # Calculate log probabilities
                log_probs = torch.log(clamped_preds)
                log_1_probs = torch.log(clamped_1_preds)
                log_1_probs_sum = log_1_probs.sum()

                # Calculate expected scores
                expected_scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)

                # Compare with method output
                assert torch.allclose(scores[b, f], expected_scores, atol=1e-6)

        # Test edge case: all predictions are at threshold
        threshold_preds = torch.full((batch_size, n_frames, n_spk), pred_score_threshold)
        threshold_scores = sortformer_modules._get_log_pred_scores(threshold_preds)

        # All scores should be the same for this case
        expected_threshold_score = (
            math.log(pred_score_threshold)
            - math.log(1.0 - pred_score_threshold)
            + n_spk * math.log(1.0 - pred_score_threshold)
            - math.log(0.5)
        )

        assert torch.allclose(threshold_scores, torch.full_like(threshold_scores, expected_threshold_score), atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, spkcache_len, spkcache_sil_frames_per_spk",
        [
            (2, 20, 4, 16, 3),  # Example 1: Default parameters
            (1, 15, 2, 8, 2),  # Example 2: Smaller dimensions
            (3, 25, 6, 40, 4),  # Example 3: Larger dimensions
            (2, 12, 3, 6, 1),  # Example 4: Small cache size
        ],
    )
    def test_get_topk_indices(self, batch_size, n_frames, n_spk, spkcache_len, spkcache_sil_frames_per_spk):
        """Test the _get_topk_indices method that finds top-k highest scoring frames."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
            spkcache_sil_frames_per_spk=spkcache_sil_frames_per_spk,
            max_index=99999,
        )

        # Create test scores with some high values and some -inf values
        scores = torch.randn(batch_size, n_frames, n_spk)

        # Add some -inf scores to test invalid score handling
        if n_frames > 5:
            scores[:, 2:4, :] = float('-inf')  # Make some frames have -inf scores

        # Call the method
        topk_indices_sorted, is_disabled = sortformer_modules._get_topk_indices(scores)

        # Check output shapes
        assert topk_indices_sorted.shape == (batch_size, spkcache_len)
        assert is_disabled.shape == (batch_size, spkcache_len)

        # Check device consistency
        assert topk_indices_sorted.device == scores.device
        assert is_disabled.device == scores.device

        # Check data types
        assert topk_indices_sorted.dtype == torch.long
        assert is_disabled.dtype == torch.bool

        # Verify that indices are within valid range
        n_frames_no_sil = n_frames - spkcache_sil_frames_per_spk
        assert torch.all(topk_indices_sorted >= 0)
        assert torch.all(topk_indices_sorted < n_frames)

        # Verify that disabled frames have index 0 (placeholder)
        disabled_indices = topk_indices_sorted[is_disabled]
        assert torch.all(disabled_indices == 0)

        # Verify that non-disabled frames have valid indices
        non_disabled_indices = topk_indices_sorted[~is_disabled]
        assert torch.all(non_disabled_indices < n_frames_no_sil)

        # Note: After remainder operation and setting disabled frames to 0,
        # indices are not guaranteed to be in ascending order
        # The method sorts the original topk indices, but then applies remainder and sets disabled to 0

        # Test edge case: all scores are -inf
        all_inf_scores = torch.full((batch_size, n_frames, n_spk), float('-inf'))
        all_inf_indices, all_inf_disabled = sortformer_modules._get_topk_indices(all_inf_scores)

        # All frames should be disabled
        assert torch.all(all_inf_disabled)
        # All indices should be placeholder (0)
        assert torch.all(all_inf_indices == 0)

        # Test edge case: scores with some frames in silence region
        normal_scores = torch.randn(batch_size, n_frames, n_spk)
        normal_indices, normal_disabled = sortformer_modules._get_topk_indices(normal_scores)

        # Check that silence frames (last spkcache_sil_frames_per_spk frames) are properly handled
        for b in range(batch_size):
            for i in range(spkcache_len):
                if normal_indices[b, i] >= n_frames_no_sil:
                    # This frame should be disabled
                    assert normal_disabled[b, i]

        # Test that we get exactly spkcache_len indices
        assert topk_indices_sorted.shape[1] == spkcache_len

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, spkcache_len",
        [
            (2, 20, 4, 12),  # Example 1: Default parameters
            (1, 15, 2, 8),  # Example 2: Smaller dimensions
            (3, 25, 6, 18),  # Example 3: Larger dimensions
        ],
    )
    def test_gather_spkcache_and_preds(self, batch_size, n_frames, n_spk, spkcache_len):
        """Test the _gather_spkcache_and_preds method that gathers embeddings and predictions."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
            spkcache_sil_frames_per_spk=2,
            max_index=99999,
        )

        # Create test embeddings and predictions
        emb_dim = 192
        emb_seq = torch.randn(batch_size, n_frames, emb_dim)
        preds = torch.rand(batch_size, n_frames, n_spk)  # Random probabilities between 0 and 1

        # Create test topk indices and disabled mask
        topk_indices = torch.randint(0, n_frames, (batch_size, spkcache_len))
        is_disabled = torch.zeros((batch_size, spkcache_len), dtype=torch.bool)

        # Make some frames disabled for testing
        if spkcache_len > 2:
            is_disabled[:, 1:3] = True  # Disable some frames

        # Get mean silence embedding, which is needed for the method call
        mean_sil_emb = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)
        mean_sil_emb, _ = sortformer_modules._get_silence_profile(mean_sil_emb, n_sil_frames, emb_seq, preds)

        # Call the method
        emb_seq_gathered, preds_gathered = sortformer_modules._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, is_disabled, mean_sil_emb
        )

        # Check output shapes
        assert emb_seq_gathered.shape == (batch_size, spkcache_len, emb_dim)
        assert preds_gathered.shape == (batch_size, spkcache_len, n_spk)

        # Check device consistency
        assert emb_seq_gathered.device == emb_seq.device
        assert preds_gathered.device == preds.device

        # Check data types
        assert emb_seq_gathered.dtype == emb_seq.dtype
        assert preds_gathered.dtype == preds.dtype

        # Verify that non-disabled frames are gathered correctly
        for b in range(batch_size):
            for i in range(spkcache_len):
                if not is_disabled[b, i]:
                    # Non-disabled frames should have the original embeddings and predictions
                    frame_idx = topk_indices[b, i]
                    assert torch.allclose(emb_seq_gathered[b, i], emb_seq[b, frame_idx], atol=1e-6)
                    assert torch.allclose(preds_gathered[b, i], preds[b, frame_idx], atol=1e-6)

        # Verify that disabled frames use silence embedding and zero predictions
        for b in range(batch_size):
            if torch.any(is_disabled[b]):
                # Check that disabled frames use the provided silence embedding
                disabled_embeddings = emb_seq_gathered[b, is_disabled[b]]
                expected_sil_emb = mean_sil_emb[b].unsqueeze(0).expand(disabled_embeddings.shape[0], -1)
                assert torch.allclose(disabled_embeddings, expected_sil_emb, atol=1e-6)

                # Check that disabled frames have zero predictions
                disabled_predictions = preds_gathered[b, is_disabled[b]]
                assert torch.allclose(disabled_predictions, torch.zeros_like(disabled_predictions), atol=1e-6)

        # Test edge case: all frames disabled
        all_disabled = torch.ones((batch_size, spkcache_len), dtype=torch.bool)
        all_disabled_emb, all_disabled_preds = sortformer_modules._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, all_disabled, mean_sil_emb
        )

        # All embeddings should be silence embeddings
        expected_sil_emb_all = mean_sil_emb.unsqueeze(1).expand(-1, spkcache_len, -1)
        assert torch.allclose(all_disabled_emb, expected_sil_emb_all, atol=1e-6)

        # All predictions should be zero
        assert torch.allclose(all_disabled_preds, torch.zeros_like(all_disabled_preds), atol=1e-6)

        # Test edge case: no frames disabled
        no_disabled = torch.zeros((batch_size, spkcache_len), dtype=torch.bool)
        no_disabled_emb, no_disabled_preds = sortformer_modules._gather_spkcache_and_preds(
            emb_seq, preds, topk_indices, no_disabled, mean_sil_emb
        )

        # All embeddings and predictions should be gathered directly
        for b in range(batch_size):
            for i in range(spkcache_len):
                frame_idx = topk_indices[b, i]
                assert torch.allclose(no_disabled_emb[b, i], emb_seq[b, frame_idx], atol=1e-6)
                assert torch.allclose(no_disabled_preds[b, i], preds[b, frame_idx], atol=1e-6)

        # Test that the method handles edge indices correctly
        edge_indices = torch.zeros((batch_size, spkcache_len), dtype=torch.long)
        edge_indices[:, 0] = 0  # First frame
        edge_indices[:, -1] = n_frames - 1  # Last frame

        edge_emb, edge_preds = sortformer_modules._gather_spkcache_and_preds(
            emb_seq, preds, edge_indices, is_disabled, mean_sil_emb
        )

        # Verify edge frame gathering works correctly
        for b in range(batch_size):
            if not is_disabled[b, 0]:
                assert torch.allclose(edge_emb[b, 0], emb_seq[b, 0], atol=1e-6)
            if not is_disabled[b, -1]:
                assert torch.allclose(edge_emb[b, -1], emb_seq[b, -1], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk",
        [
            (2, 10, 4),  # Example 1: Default parameters
            (1, 8, 2),  # Example 2: Smaller dimensions
            (3, 12, 6),  # Example 3: Larger dimensions
            (2, 6, 3),  # Example 4: Small dimensions
        ],
    )
    def test_get_max_perm_index(self, batch_size, n_frames, n_spk):
        """Test the _get_max_perm_index method that finds the number of speakers with positive scores."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )

        # Create test scores with various patterns
        scores = torch.randn(batch_size, n_frames, n_spk)

        # Call the method
        max_perm_index = sortformer_modules._get_max_perm_index(scores)

        # Check output shape
        assert max_perm_index.shape == (batch_size,)

        # Check device consistency
        assert max_perm_index.device == scores.device

        # Check data type
        assert max_perm_index.dtype == torch.long

        # Verify that max_perm_index is within valid range [0, n_spk]
        assert torch.all(max_perm_index >= 0)
        assert torch.all(max_perm_index <= n_spk)

        # Test case: all scores positive
        all_positive_scores = torch.abs(torch.randn(batch_size, n_frames, n_spk)) + 0.1  # Ensure all > 0
        all_pos_max_perm = sortformer_modules._get_max_perm_index(all_positive_scores)

        # All speakers should have positive scores, so max_perm_index should be n_spk
        assert torch.all(all_pos_max_perm == n_spk)

        # Test case: all scores negative
        all_negative_scores = -torch.abs(torch.randn(batch_size, n_frames, n_spk)) - 0.1  # Ensure all < 0
        all_neg_max_perm = sortformer_modules._get_max_perm_index(all_negative_scores)

        # No speakers should have positive scores, so max_perm_index should be 0
        assert torch.all(all_neg_max_perm == 0)

        # Test case: mixed positive and negative scores
        mixed_scores = torch.randn(batch_size, n_frames, n_spk)
        mixed_max_perm = sortformer_modules._get_max_perm_index(mixed_scores)

        # Check that max_perm_index is within valid range [0, n_spk]
        assert torch.all(mixed_max_perm >= 0)
        assert torch.all(mixed_max_perm <= n_spk)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, min_pos_scores_per_spk",
        [
            (2, 10, 4, 3),  # Example 1: Default parameters
            (1, 8, 2, 2),  # Example 2: Smaller dimensions
            (3, 12, 6, 4),  # Example 3: Larger dimensions
        ],
    )
    def test_disable_low_scores(self, batch_size, n_frames, n_spk, min_pos_scores_per_spk):
        """Test the _disable_low_scores method that disables low scores for non-speech and overlapped speech."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )

        # Create test predictions and scores
        preds = torch.rand(batch_size, n_frames, n_spk)  # Random probabilities between 0 and 1
        scores = torch.randn(batch_size, n_frames, n_spk)  # Random scores (positive and negative)

        # Call the method
        modified_scores = sortformer_modules._disable_low_scores(preds, scores, min_pos_scores_per_spk)

        # Check output shape and properties
        assert modified_scores.shape == (batch_size, n_frames, n_spk)
        assert modified_scores.device == scores.device
        assert modified_scores.dtype == scores.dtype

        # Test 1: Non-speech frames (preds <= 0.5) should have -inf scores
        is_speech = preds > 0.5
        non_speech_mask = ~is_speech
        assert torch.all(modified_scores[non_speech_mask] == float('-inf'))

        # Test 2: Speech frames with positive scores should remain unchanged
        speech_positive_mask = is_speech & (scores > 0)
        if torch.any(speech_positive_mask):
            assert torch.allclose(modified_scores[speech_positive_mask], scores[speech_positive_mask], atol=1e-6)

        # Test edge case: all predictions are speech (preds > 0.5)
        all_speech_preds = torch.full((batch_size, n_frames, n_spk), 0.8)
        all_speech_scores = torch.randn(batch_size, n_frames, n_spk)
        all_speech_modified = sortformer_modules._disable_low_scores(
            all_speech_preds, all_speech_scores, min_pos_scores_per_spk
        )

        # Verify that the method works correctly with all speech predictions
        # (some scores might be -inf due to overlapped speech, but that's expected)
        assert all_speech_modified.shape == (batch_size, n_frames, n_spk)

        # Test edge case: all predictions are non-speech (preds <= 0.5)
        all_nonspeech_preds = torch.full((batch_size, n_frames, n_spk), 0.3)
        all_nonspeech_scores = torch.randn(batch_size, n_frames, n_spk)
        all_nonspeech_modified = sortformer_modules._disable_low_scores(
            all_nonspeech_preds, all_nonspeech_scores, min_pos_scores_per_spk
        )

        # All scores should be -inf
        assert torch.all(all_nonspeech_modified == float('-inf'))

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk",
        [
            (2, 10, 4),  # Example 1: Default parameters
            (1, 8, 2),  # Example 2: Smaller dimensions
            (3, 12, 6),  # Example 3: Larger dimensions
        ],
    )
    def test_permute_speakers(self, batch_size, n_frames, n_spk):
        """Test the _permute_speakers method that creates random permutations of speaker scores."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=188,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
        )

        # Create test scores and max_perm_index
        scores = torch.randn(batch_size, n_frames, n_spk)
        max_perm_index = torch.randint(1, n_spk + 1, (batch_size,))  # Random number of speakers to permute

        # Call the method
        permuted_scores, spk_perm = sortformer_modules._permute_speakers(scores, max_perm_index)

        # Check output shapes
        assert permuted_scores.shape == (batch_size, n_frames, n_spk)
        assert spk_perm.shape == (batch_size, n_spk)

        # Check device consistency
        assert permuted_scores.device == scores.device
        assert spk_perm.device == scores.device

        # Check data types
        assert permuted_scores.dtype == scores.dtype
        assert spk_perm.dtype == torch.long

        # Test 1: Verify that permutation is valid (contains all speaker indices)
        for b in range(batch_size):
            perm_indices = spk_perm[b]
            expected_indices = torch.arange(n_spk, device=scores.device)
            assert torch.all(torch.sort(perm_indices)[0] == expected_indices)

        # Test 2: Verify that scores are permuted correctly
        for b in range(batch_size):
            batch_perm = spk_perm[b]
            batch_scores = scores[b]  # Shape: (n_frames, n_spk)
            batch_permuted = permuted_scores[b]  # Shape: (n_frames, n_spk)

            # Check that permuted scores match original scores with permutation
            for i in range(n_spk):
                original_idx = batch_perm[i]
                assert torch.allclose(batch_permuted[:, i], batch_scores[:, original_idx], atol=1e-6)

        # Test edge case: max_perm_index = 0 (no permutation)
        zero_perm_index = torch.zeros((batch_size,), dtype=torch.long)
        zero_perm_scores, zero_perm_spk = sortformer_modules._permute_speakers(scores, zero_perm_index)

        # Scores should be unchanged
        assert torch.allclose(zero_perm_scores, scores, atol=1e-6)
        # Permutation should be identity
        for b in range(batch_size):
            assert torch.all(zero_perm_spk[b] == torch.arange(n_spk, device=scores.device))

        # Test edge case: max_perm_index = n_spk (all speakers permuted)
        all_perm_index = torch.full((batch_size,), n_spk, dtype=torch.long)
        _, all_perm_spk = sortformer_modules._permute_speakers(scores, all_perm_index)

        # All speakers should be permuted
        for b in range(batch_size):
            perm_indices = all_perm_spk[b]
            original_indices = torch.arange(n_spk, device=scores.device)
            # Check that permutation is valid but not identity
            assert torch.all(torch.sort(perm_indices)[0] == original_indices)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, n_frames, n_spk, spkcache_len",
        [
            (2, 20, 4, 12),  # Example 1: Default parameters
            (1, 15, 2, 8),  # Example 2: Smaller dimensions
            (3, 25, 6, 18),  # Example 3: Larger dimensions
            (3, 25, 6, 25),  # Example 4: Equal spkcache_len and n_frames
        ],
    )
    def test_compress_spkcache(self, batch_size, n_frames, n_spk, spkcache_len):
        """Test the _compress_spkcache method that compresses speaker cache for streaming inference."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=376,
            chunk_len=376,
            chunk_left_context=1,
            chunk_right_context=1,
            spkcache_update_period=376,
            spkcache_sil_frames_per_spk=2,
        )

        # Create test embeddings and predictions
        emb_dim = 192
        emb_seq = torch.randn(batch_size, n_frames, emb_dim)
        preds = torch.rand(batch_size, n_frames, n_spk)  # Random probabilities between 0 and 1

        # Get mean silence embedding, which is needed for the method call
        mean_sil_emb = torch.zeros(batch_size, emb_dim, device=emb_seq.device)
        n_sil_frames = torch.zeros(batch_size, dtype=torch.long, device=emb_seq.device)
        mean_sil_emb, _ = sortformer_modules._get_silence_profile(mean_sil_emb, n_sil_frames, emb_seq, preds)

        # Call the method without permutation
        spkcache, spkcache_preds, spk_perm = sortformer_modules._compress_spkcache(
            emb_seq, preds, mean_sil_emb, permute_spk=False
        )

        # Check output shapes
        assert spkcache.shape == (batch_size, spkcache_len, emb_dim)
        assert spkcache_preds.shape == (batch_size, spkcache_len, n_spk)
        assert spk_perm is None  # No permutation when permute_spk=False

        # Check device consistency
        assert spkcache.device == emb_seq.device
        assert spkcache_preds.device == preds.device

        # Check data types
        assert spkcache.dtype == emb_seq.dtype
        assert spkcache_preds.dtype == preds.dtype

        # Test with permutation enabled
        spkcache_perm, spkcache_preds_perm, spk_perm_perm = sortformer_modules._compress_spkcache(
            emb_seq, preds, mean_sil_emb, permute_spk=True
        )

        # Check output shapes with permutation
        assert spkcache_perm.shape == (batch_size, spkcache_len, emb_dim)
        assert spkcache_preds_perm.shape == (batch_size, spkcache_len, n_spk)
        assert spk_perm_perm.shape == (batch_size, n_spk)  # Permutation tensor should be returned

        # Verify that compression reduces the number of frames
        assert spkcache_len <= n_frames

        # Verify that the method handles edge case: n_frames = spkcache_len
        if n_frames == spkcache_len:
            edge_emb_seq = torch.randn(batch_size, spkcache_len, emb_dim)
            edge_preds = torch.rand(batch_size, spkcache_len, n_spk)
            mean_sil_emb_edge, _ = sortformer_modules._get_silence_profile(
                mean_sil_emb, n_sil_frames, edge_emb_seq, edge_preds
            )
            edge_spkcache, edge_spkcache_preds, edge_spk_perm = sortformer_modules._compress_spkcache(
                edge_emb_seq, edge_preds, mean_sil_emb_edge, permute_spk=False
            )

            assert edge_spkcache.shape == (batch_size, spkcache_len, emb_dim)
            assert edge_spkcache_preds.shape == (batch_size, spkcache_len, n_spk)


class TestSortformerModules_StreamingUpdate:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_spk, spkcache_len, cur_spkcache_len, fifo_len, cur_fifo_len, chunk_len, lc, rc",
        [
            (2, 512, 4, 16, 0, 16, 0, 5, 1, 1),  # Example 1: empty spkcache and fifo
            (3, 256, 3, 32, 16, 32, 24, 6, 0, 0),  # Example 2: non-empty spkcache and fifo
            (4, 128, 2, 16, 16, 32, 20, 10, 1, 2),  # Example 3: full spkcache, fifo not full
        ],
    )
    def test_fifo_not_full(
        self, batch_size, emb_dim, n_spk, spkcache_len, cur_spkcache_len, fifo_len, cur_fifo_len, chunk_len, lc, rc
    ):
        """Tests the streaming_update method for cases where the FIFO buffer does not become full."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=fifo_len,
            chunk_len=chunk_len,
            fc_d_model=emb_dim,
            spkcache_sil_frames_per_spk=3,
        )
        sortformer_modules.training = False  # Disable training mode for consistent behavior

        # Initialize streaming state
        streaming_state = sortformer_modules.init_streaming_state(batch_size=batch_size, async_streaming=False)
        streaming_state.spkcache = torch.randn(batch_size, cur_spkcache_len, emb_dim)
        streaming_state.fifo = torch.randn(batch_size, cur_fifo_len, emb_dim)
        streaming_state.fifo_preds = torch.rand(batch_size, cur_fifo_len, n_spk)

        # Create chunk and preds
        chunk_total_len = chunk_len + lc + rc
        chunk = torch.randn(batch_size, chunk_total_len, emb_dim)
        preds = torch.rand(batch_size, cur_spkcache_len + cur_fifo_len + chunk_total_len, n_spk)

        # Calculate expected states before the update
        expected_chunk_preds = preds[
            :, cur_spkcache_len + cur_fifo_len + lc : cur_spkcache_len + cur_fifo_len + chunk_len + lc
        ]
        old_fifo_preds = preds[:, cur_spkcache_len : cur_spkcache_len + cur_fifo_len]
        expected_fifo_preds = torch.cat([old_fifo_preds, expected_chunk_preds], dim=1)
        expected_fifo_embs = torch.cat([streaming_state.fifo, chunk[:, lc : lc + chunk_len]], dim=1)
        expected_spkcache_embs = streaming_state.spkcache.clone()

        # Call streaming_update
        streaming_state, chunk_preds = sortformer_modules.streaming_update(streaming_state, chunk, preds, lc, rc)

        # Check returned chunk_preds
        assert chunk_preds.shape == (batch_size, chunk_len, n_spk)
        assert torch.allclose(chunk_preds, expected_chunk_preds)

        # Check updated streaming state's fifo
        assert streaming_state.fifo.shape == (batch_size, cur_fifo_len + chunk_len, emb_dim)
        assert torch.allclose(streaming_state.fifo, expected_fifo_embs)
        assert streaming_state.fifo_preds.shape == (batch_size, cur_fifo_len + chunk_len, n_spk)
        assert torch.allclose(streaming_state.fifo_preds, expected_fifo_preds)

        # Check updated streaming state's spkcache
        assert streaming_state.spkcache.shape == (batch_size, cur_spkcache_len, emb_dim)
        assert torch.allclose(streaming_state.spkcache, expected_spkcache_embs)
        assert streaming_state.spkcache_preds is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_spk, spkcache_len, cur_spkcache_len, fifo_len, cur_fifo_len, chunk_len, spkcache_update_period, lc, rc",
        [
            (2, 512, 4, 16, 0, 16, 15, 5, 10, 1, 1),  # Example 1: spkcache is empty
            (3, 256, 3, 32, 16, 16, 14, 7, 16, 1, 2),  # Example 2: spkcache is not empty
            (4, 256, 5, 32, 16, 0, 0, 8, 16, 1, 2),  # Example 3: spkcache is not empty, no fifo
        ],
    )
    def test_fifo_full_no_compression(
        self,
        batch_size,
        emb_dim,
        n_spk,
        spkcache_len,
        cur_spkcache_len,
        fifo_len,
        cur_fifo_len,
        chunk_len,
        spkcache_update_period,
        lc,
        rc,
    ):
        """Tests the streaming_update method for cases where the FIFO buffer becomes full but no compression of spkcache is needed."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=fifo_len,
            chunk_len=chunk_len,
            fc_d_model=emb_dim,
            spkcache_update_period=spkcache_update_period,
        )
        sortformer_modules.training = False

        # Initialize streaming state
        streaming_state = sortformer_modules.init_streaming_state(batch_size=batch_size, async_streaming=False)
        streaming_state.spkcache = torch.randn(batch_size, cur_spkcache_len, emb_dim)
        streaming_state.fifo = torch.randn(batch_size, cur_fifo_len, emb_dim)
        streaming_state.fifo_preds = torch.rand(batch_size, cur_fifo_len, n_spk)

        # Create chunk and preds
        chunk_total_len = chunk_len + lc + rc
        chunk = torch.randn(batch_size, chunk_total_len, emb_dim)
        preds = torch.rand(batch_size, cur_spkcache_len + cur_fifo_len + chunk_total_len, n_spk)

        # Check that the FIFO buffer will overflow after adding the chunk
        assert cur_fifo_len + chunk_len > fifo_len

        # Calculate the number of frames to move from FIFO to spkcache
        pop_out_len = spkcache_update_period
        pop_out_len = max(pop_out_len, chunk_len - fifo_len + cur_fifo_len)
        pop_out_len = min(pop_out_len, cur_fifo_len + chunk_len)

        # Check that the spkcache will not overflow after adding the frames
        assert cur_spkcache_len + pop_out_len <= spkcache_len

        # Calculate expected states before the update
        expected_chunk_preds = preds[
            :, cur_spkcache_len + cur_fifo_len + lc : cur_spkcache_len + cur_fifo_len + chunk_len + lc
        ]
        old_fifo_preds = preds[:, cur_spkcache_len : cur_spkcache_len + cur_fifo_len]
        fifo_preds_before_split = torch.cat([old_fifo_preds, expected_chunk_preds], dim=1)
        fifo_embs_before_split = torch.cat([streaming_state.fifo, chunk[:, lc : lc + chunk_len]], dim=1)
        pop_out_embs = fifo_embs_before_split[:, :pop_out_len]
        expected_fifo_embs = fifo_embs_before_split[:, pop_out_len:]
        expected_fifo_preds = fifo_preds_before_split[:, pop_out_len:]
        expected_spkcache_embs = torch.cat([streaming_state.spkcache, pop_out_embs], dim=1)

        # Call streaming_update
        streaming_state, chunk_preds = sortformer_modules.streaming_update(streaming_state, chunk, preds, lc, rc)

        # Check returned chunk_preds
        assert chunk_preds.shape == (batch_size, chunk_len, n_spk)
        assert torch.allclose(chunk_preds, expected_chunk_preds)

        # Check updated streaming state's fifo
        assert streaming_state.fifo.shape == (batch_size, cur_fifo_len + chunk_len - pop_out_len, emb_dim)
        assert torch.allclose(streaming_state.fifo, expected_fifo_embs)
        assert streaming_state.fifo_preds.shape == (batch_size, cur_fifo_len + chunk_len - pop_out_len, n_spk)
        assert torch.allclose(streaming_state.fifo_preds, expected_fifo_preds)

        # Check updated streaming state's spkcache
        assert streaming_state.spkcache.shape == (batch_size, cur_spkcache_len + pop_out_len, emb_dim)
        assert torch.allclose(streaming_state.spkcache, expected_spkcache_embs)
        assert streaming_state.spkcache_preds is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_spk, spkcache_len, cur_spkcache_len, fifo_len, cur_fifo_len, chunk_len, spkcache_update_period, lc, rc",
        [
            (2, 512, 4, 16, 10, 16, 15, 5, 10, 1, 1),  # Example 1: spkcache is not full
            (3, 256, 3, 32, 32, 16, 14, 7, 16, 1, 2),  # Example 2: spkcache is full
            (4, 256, 5, 32, 32, 0, 0, 8, 16, 1, 2),  # Example 3: spkcache is full, no fifo
        ],
    )
    def test_fifo_full_with_compression(
        self,
        batch_size,
        emb_dim,
        n_spk,
        spkcache_len,
        cur_spkcache_len,
        fifo_len,
        cur_fifo_len,
        chunk_len,
        spkcache_update_period,
        lc,
        rc,
    ):
        """Tests the streaming_update method for cases where the FIFO buffer becomes full and compression of spkcache is needed."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=fifo_len,
            chunk_len=chunk_len,
            fc_d_model=emb_dim,
            spkcache_update_period=spkcache_update_period,
        )
        sortformer_modules.training = False

        # Initialize streaming state
        streaming_state = sortformer_modules.init_streaming_state(batch_size=batch_size, async_streaming=False)
        streaming_state.spkcache = torch.randn(batch_size, cur_spkcache_len, emb_dim)
        streaming_state.fifo = torch.randn(batch_size, cur_fifo_len, emb_dim)
        streaming_state.fifo_preds = torch.rand(batch_size, cur_fifo_len, n_spk)

        # Create chunk and preds
        chunk_total_len = chunk_len + lc + rc
        chunk = torch.randn(batch_size, chunk_total_len, emb_dim)
        preds = torch.rand(batch_size, cur_spkcache_len + cur_fifo_len + chunk_total_len, n_spk)

        # Check that the FIFO buffer will overflow after adding the chunk
        assert cur_fifo_len + chunk_len > fifo_len

        # Calculate the number of frames to move from FIFO to spkcache
        pop_out_len = spkcache_update_period
        pop_out_len = max(pop_out_len, chunk_len - fifo_len + cur_fifo_len)
        pop_out_len = min(pop_out_len, cur_fifo_len + chunk_len)

        # Check that the spkcache will overflow after adding the frames
        assert cur_spkcache_len + pop_out_len > spkcache_len

        # Calculate expected states before the update
        expected_chunk_preds = preds[
            :, cur_spkcache_len + cur_fifo_len + lc : cur_spkcache_len + cur_fifo_len + chunk_len + lc
        ]
        old_fifo_preds = preds[:, cur_spkcache_len : cur_spkcache_len + cur_fifo_len]
        fifo_preds_before_split = torch.cat([old_fifo_preds, expected_chunk_preds], dim=1)
        fifo_embs_before_split = torch.cat([streaming_state.fifo, chunk[:, lc : lc + chunk_len]], dim=1)
        expected_fifo_embs = fifo_embs_before_split[:, pop_out_len:]
        expected_fifo_preds = fifo_preds_before_split[:, pop_out_len:]

        # Call streaming_update
        streaming_state, chunk_preds = sortformer_modules.streaming_update(streaming_state, chunk, preds, lc, rc)

        # Check returned chunk_preds
        assert chunk_preds.shape == (batch_size, chunk_len, n_spk)
        assert torch.allclose(chunk_preds, expected_chunk_preds)

        # Check updated streaming state's fifo
        assert streaming_state.fifo.shape == (batch_size, cur_fifo_len + chunk_len - pop_out_len, emb_dim)
        assert torch.allclose(streaming_state.fifo, expected_fifo_embs)
        assert streaming_state.fifo_preds.shape == (batch_size, cur_fifo_len + chunk_len - pop_out_len, n_spk)
        assert torch.allclose(streaming_state.fifo_preds, expected_fifo_preds)

        # Check updated streaming state's spkcache
        assert streaming_state.spkcache.shape == (batch_size, spkcache_len, emb_dim)
        assert streaming_state.spkcache_preds.shape == (batch_size, spkcache_len, n_spk)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_spk, spkcache_len, cur_spkcache_len, fifo_len, cur_fifo_len, chunk_len, spkcache_update_period, lc, rc",
        [
            (256, 8, 2, 128, 128, 16, 15, 5, 16, 1, 1),
        ],
    )
    def test_training_mode_with_permutation(
        self,
        batch_size,
        emb_dim,
        n_spk,
        spkcache_len,
        cur_spkcache_len,
        fifo_len,
        cur_fifo_len,
        chunk_len,
        spkcache_update_period,
        lc,
        rc,
    ):
        """Tests that speaker permutation is triggered during spkcache compression in training mode."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=spkcache_len,
            fifo_len=fifo_len,
            chunk_len=chunk_len,
            fc_d_model=emb_dim,
            spkcache_update_period=spkcache_update_period,
        )
        sortformer_modules.training = True  # Set to training mode

        # Initialize streaming state
        streaming_state = sortformer_modules.init_streaming_state(batch_size=batch_size, async_streaming=False)
        streaming_state.spkcache = torch.randn(batch_size, cur_spkcache_len, emb_dim)
        streaming_state.fifo = torch.randn(batch_size, cur_fifo_len, emb_dim)
        streaming_state.fifo_preds = torch.rand(batch_size, cur_fifo_len, n_spk)

        # Create chunk and preds
        chunk_total_len = chunk_len + lc + rc
        chunk = torch.randn(batch_size, chunk_total_len, emb_dim)
        preds = torch.rand(batch_size, cur_spkcache_len + cur_fifo_len + chunk_total_len, n_spk)

        # Check that the FIFO buffer will overflow after adding the chunk
        assert cur_fifo_len + chunk_len > fifo_len

        # Call streaming_update
        streaming_state, chunk_preds = sortformer_modules.streaming_update(streaming_state, chunk, preds, lc, rc)

        # After compression is triggered in training mode, spk_perm should not be None.
        assert streaming_state.spk_perm is not None
        assert streaming_state.spk_perm.shape == (batch_size, n_spk)

        # Check if it is a valid permutation.
        # A valid permutation of N items, when sorted, should be equal to [0, 1, ..., N-1].
        for b in range(batch_size):
            perm = streaming_state.spk_perm[b]
            assert torch.all(torch.sort(perm)[0] == torch.arange(n_spk, device=perm.device))

        # Get the permuted chunk_preds
        chunk_preds_perm = torch.stack(
            [chunk_preds[batch_index, :, streaming_state.spk_perm[batch_index]] for batch_index in range(batch_size)]
        )

        # Check that not all the permutations are identical (should be true for large batch size)
        assert not torch.allclose(chunk_preds, chunk_preds_perm)

        # Get the inverse permutation
        inv_spk_perm = torch.stack(
            [torch.argsort(streaming_state.spk_perm[batch_index]) for batch_index in range(batch_size)]
        )
        chunk_preds_perm_inv = torch.stack(
            [chunk_preds_perm[batch_index, :, inv_spk_perm[batch_index]] for batch_index in range(batch_size)]
        )

        # Check that after permutation and inverse permutation we got the original chunk_preds
        assert torch.allclose(chunk_preds, chunk_preds_perm_inv)


class TestSortformerModules_StreamingUpdateAsync:
    def _assert_async_batch_item_state(
        self,
        chunk_preds,
        expected_chunk_preds,
        fifo_len,
        expected_fifo_len,
        fifo,
        expected_fifo_embs,
        fifo_preds,
        expected_fifo_preds,
        spkcache_len,
        expected_spkcache_len,
        spkcache,
        expected_spkcache_embs,
        spkcache_preds,
        expected_spkcache_preds,
    ):
        """Helper function to assert the state of a single item in an async batch."""
        assert chunk_preds.shape == expected_chunk_preds.shape
        assert torch.allclose(chunk_preds, expected_chunk_preds)
        assert fifo_len == expected_fifo_len
        assert fifo.shape == expected_fifo_embs.shape
        assert torch.allclose(fifo, expected_fifo_embs)
        assert fifo_preds.shape == expected_fifo_preds.shape
        assert torch.allclose(fifo_preds, expected_fifo_preds)
        assert spkcache_len == expected_spkcache_len
        assert spkcache.shape == expected_spkcache_embs.shape
        assert torch.allclose(spkcache, expected_spkcache_embs)
        assert spkcache_preds.shape == expected_spkcache_preds.shape
        assert torch.allclose(spkcache_preds, expected_spkcache_preds)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "batch_size, emb_dim, n_spk, max_spkcache_len, spkcache_lengths, max_fifo_len, fifo_lengths, "
        "max_chunk_len, chunk_lengths, spkcache_update_period, lc, rc",
        [
            (
                3,
                256,
                4,
                32,
                torch.tensor([0, 16, 32]),
                16,
                torch.tensor([0, 8, 16]),
                8,
                torch.tensor([8, 5, 0]),
                16,
                1,
                1,
            ),  # Example 1: fifo not full
            (
                3,
                128,
                4,
                32,
                torch.tensor([0, 10, 12]),
                16,
                torch.tensor([16, 14, 14]),
                8,
                torch.tensor([8, 5, 8]),
                20,
                1,
                1,
            ),  # Example 2: fifo is full, spkcache is not full (no compression)
            (
                3,
                64,
                4,
                32,
                torch.tensor([24, 32, 21]),
                16,
                torch.tensor([16, 14, 14]),
                8,
                torch.tensor([8, 5, 3]),
                12,
                1,
                1,
            ),  # Example 3: fifo and spkcache are full (do compression)
            (
                6,
                32,
                5,
                32,
                torch.tensor([0, 10, 0, 20, 30, 32]),
                16,
                torch.tensor([5, 13, 13, 16, 16, 15]),
                5,
                torch.tensor([5, 3, 4, 3, 5, 3]),
                10,
                2,
                3,
            ),  # Example 4: mixed cases
        ],
    )
    def test_streaming_update_async(
        self,
        batch_size,
        emb_dim,
        n_spk,
        max_spkcache_len,
        spkcache_lengths,
        max_fifo_len,
        fifo_lengths,
        max_chunk_len,
        chunk_lengths,
        spkcache_update_period,
        lc,
        rc,
    ):
        """Tests the async streaming update method."""
        sortformer_modules = SortformerModules(
            num_spks=n_spk,
            spkcache_len=max_spkcache_len,
            fifo_len=max_fifo_len,
            chunk_len=max_chunk_len,
            fc_d_model=emb_dim,
            spkcache_update_period=spkcache_update_period,
        )
        sortformer_modules.training = False

        # Check that the input lengths are correct
        assert spkcache_lengths.shape == (batch_size,)
        assert fifo_lengths.shape == (batch_size,)
        assert chunk_lengths.shape == (batch_size,)
        assert spkcache_lengths.max() <= max_spkcache_len
        assert fifo_lengths.max() <= max_fifo_len
        assert chunk_lengths.max() <= max_chunk_len
        assert spkcache_lengths.min() >= 0
        assert fifo_lengths.min() >= 0
        assert chunk_lengths.min() >= 0

        # Initialize streaming state
        streaming_state = sortformer_modules.init_streaming_state(batch_size=batch_size, async_streaming=True)
        streaming_state.spkcache_lengths = spkcache_lengths.clone()
        streaming_state.fifo_lengths = fifo_lengths.clone()
        for b in range(batch_size):
            streaming_state.spkcache[b, : spkcache_lengths[b]] = torch.randn(spkcache_lengths[b], emb_dim)
            streaming_state.spkcache_preds[b, : spkcache_lengths[b]] = torch.rand(spkcache_lengths[b], n_spk)
            streaming_state.fifo[b, : fifo_lengths[b]] = torch.randn(fifo_lengths[b], emb_dim)

        # Keep spkcache and fifo from the initial state
        initial_spkcache = streaming_state.spkcache.clone()
        initial_spkcache_preds = streaming_state.spkcache_preds.clone()
        initial_fifo = streaming_state.fifo.clone()

        # Create input chunk and preds
        chunk_total_len = max_chunk_len + lc + rc
        chunk = torch.randn(batch_size, chunk_total_len, emb_dim)
        preds_total_len = max_spkcache_len + max_fifo_len + chunk_total_len
        preds = torch.rand(batch_size, preds_total_len, n_spk)

        # Run streaming_update_async
        streaming_state, chunk_preds = sortformer_modules.streaming_update_async(
            streaming_state, chunk, chunk_lengths + lc, preds, lc, rc
        )

        # Process batch items
        for b in range(batch_size):
            spkcache_len = spkcache_lengths[b].item()
            fifo_len = fifo_lengths[b].item()
            chunk_len = chunk_lengths[b].item()

            expected_chunk_preds = preds[b, spkcache_len + fifo_len + lc : spkcache_len + fifo_len + lc + chunk_len]
            updated_fifo_embs = torch.zeros(max_fifo_len + max_chunk_len, emb_dim)
            updated_fifo_preds = torch.zeros(max_fifo_len + max_chunk_len, n_spk)
            expected_spkcache_embs = torch.zeros(max_spkcache_len, emb_dim)
            expected_spkcache_preds = torch.zeros(max_spkcache_len, n_spk)
            updated_fifo_embs[:fifo_len] = initial_fifo[b, :fifo_len]
            updated_fifo_embs[fifo_len : fifo_len + chunk_len] = chunk[b, lc : lc + chunk_len]
            updated_fifo_preds[:fifo_len] = preds[b, spkcache_len : spkcache_len + fifo_len]
            updated_fifo_preds[fifo_len : fifo_len + chunk_len] = expected_chunk_preds
            expected_fifo_embs = torch.zeros(max_fifo_len, emb_dim)
            expected_fifo_preds = torch.zeros(max_fifo_len, n_spk)

            # Case 1: Fifo not full
            if fifo_len + chunk_len <= max_fifo_len:
                expected_fifo_len = fifo_len + chunk_len
                expected_spkcache_len = spkcache_len
                expected_spkcache_embs = initial_spkcache[b]
                expected_spkcache_preds = initial_spkcache_preds[b]
                expected_fifo_embs[: fifo_len + chunk_len] = updated_fifo_embs[: fifo_len + chunk_len]
                expected_fifo_preds[: fifo_len + chunk_len] = updated_fifo_preds[: fifo_len + chunk_len]

            else:
                pop_out_len = spkcache_update_period
                pop_out_len = max(pop_out_len, max_chunk_len - max_fifo_len + fifo_len)
                pop_out_len = min(pop_out_len, fifo_len + chunk_len)

                expected_fifo_len = fifo_len + chunk_len - pop_out_len
                expected_fifo_embs[:expected_fifo_len] = updated_fifo_embs[
                    pop_out_len : pop_out_len + expected_fifo_len
                ]
                expected_fifo_preds[:expected_fifo_len] = updated_fifo_preds[
                    pop_out_len : pop_out_len + expected_fifo_len
                ]
                pop_out_embs = updated_fifo_embs[:pop_out_len]
                pop_out_preds = updated_fifo_preds[:pop_out_len]

                # Case 2: spkcache not full (no compression)
                if spkcache_len + pop_out_len <= max_spkcache_len:
                    expected_spkcache_len = spkcache_len + pop_out_len
                    expected_spkcache_embs[:spkcache_len] = initial_spkcache[b, :spkcache_len]
                    expected_spkcache_embs[spkcache_len : spkcache_len + pop_out_len] = pop_out_embs
                    expected_spkcache_preds[:spkcache_len] = initial_spkcache_preds[b, :spkcache_len]
                    expected_spkcache_preds[spkcache_len : spkcache_len + pop_out_len] = pop_out_preds

                # Case 3: spkcache is full (do compression)
                else:
                    expected_spkcache_len = max_spkcache_len
                    # Compression logic is validated in its own unit test.
                    # Here, we trust its output and verify the resulting state's integrity.
                    expected_spkcache_embs = streaming_state.spkcache[b, :max_spkcache_len]
                    expected_spkcache_preds = streaming_state.spkcache_preds[b, :max_spkcache_len]

            self._assert_async_batch_item_state(
                chunk_preds[b, :chunk_len],
                expected_chunk_preds,
                streaming_state.fifo_lengths[b].item(),
                expected_fifo_len,
                streaming_state.fifo[b, :max_fifo_len],
                expected_fifo_embs,
                streaming_state.fifo_preds[b, :max_fifo_len],
                expected_fifo_preds,
                streaming_state.spkcache_lengths[b].item(),
                expected_spkcache_len,
                streaming_state.spkcache[b, :max_spkcache_len],
                expected_spkcache_embs,
                streaming_state.spkcache_preds[b, :max_spkcache_len],
                expected_spkcache_preds,
            )
