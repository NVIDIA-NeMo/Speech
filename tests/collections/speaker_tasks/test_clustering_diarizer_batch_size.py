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

"""Regression tests for the ClusteringDiarizer batch_size=None defect.

Reproduces the bug reported as ``TypeError: iteration over a 0-d tensor`` in
``audio_to_label._fixed_seq_collate_fn``. The underlying cause is that a missing
top-level ``batch_size`` in the diarizer config was forwarded to
``torch.utils.data.DataLoader`` as ``None``, which disables automatic batching and
feeds un-batched 0-d tensor samples to the collate function.
"""

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_label import _fixed_seq_collate_fn
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer


def _single_unbatched_sample():
    """Mimic AudioToSpeechLabelDataset.__getitem__: (feat, feat_len, label, label_len).

    ``feat_len``, ``label`` and ``label_len`` are 0-d scalar tensors, exactly the shape
    that triggered ``TypeError: iteration over a 0-d tensor`` under ``batch_size=None``.
    """
    feat = torch.randn(16000)
    feat_len = torch.tensor(16000).long()
    label = torch.tensor(0).long()
    label_len = torch.tensor(1).long()
    return feat, feat_len, label, label_len


@pytest.mark.unit
def test_zero_d_tensor_reproduces_original_bug():
    """A raw un-batched 4-tuple must raise the exact reported TypeError."""
    sample = _single_unbatched_sample()
    with pytest.raises(TypeError):
        # zip(*sample) iterates the 0-d feat_len tensor.
        _fixed_seq_collate_fn(self=None, batch=sample)


@pytest.mark.unit
def test_batched_input_collates_without_error():
    """A proper list-of-samples batch collates fine (sanity check the fix target)."""
    batch = [_single_unbatched_sample() for _ in range(3)]
    audio_signal, audio_lengths, tokens, tokens_lengths = _fixed_seq_collate_fn(self=None, batch=batch)
    assert audio_signal.shape[0] == 3
    assert audio_lengths.shape[0] == 3
    assert tokens.shape[0] == 3
    assert tokens_lengths.shape[0] == 3


@pytest.mark.unit
def test_get_batch_size_coalesces_none_to_positive_default():
    """When the config has no top-level batch_size, a positive int default is used."""
    diarizer = ClusteringDiarizer.__new__(ClusteringDiarizer)
    diarizer._cfg = OmegaConf.create({'sample_rate': 16000})  # no batch_size key

    resolved = diarizer._get_batch_size()

    assert resolved is not None
    assert isinstance(resolved, int)
    assert resolved > 0


@pytest.mark.unit
def test_get_batch_size_respects_explicit_value():
    """An explicit top-level batch_size is preserved unchanged."""
    diarizer = ClusteringDiarizer.__new__(ClusteringDiarizer)
    diarizer._cfg = OmegaConf.create({'sample_rate': 16000, 'batch_size': 7})

    assert diarizer._get_batch_size() == 7
