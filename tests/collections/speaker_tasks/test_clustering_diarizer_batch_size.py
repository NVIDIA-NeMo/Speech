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

"""Regression test for ClusteringDiarizer passing ``batch_size=None`` to its dataloaders.

A config without a top-level ``batch_size`` used to reach ``DataLoader`` as ``None``, which turns off
batching and fails in the collate function with ``TypeError: iteration over a 0-d tensor``.
"""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer


def _diarizer(extra_cfg):
    # __init__ loads the VAD and speaker models, so build the object without it.
    diarizer = ClusteringDiarizer.__new__(ClusteringDiarizer)
    diarizer._cfg = OmegaConf.create({'sample_rate': 16000, 'num_workers': 0, **extra_cfg})
    diarizer._vad_window_length_in_sec = 0.63
    diarizer._vad_shift_length_in_sec = 0.01
    diarizer._vad_model = MagicMock()
    diarizer._speaker_model = MagicMock()
    return diarizer


@pytest.mark.unit
@pytest.mark.parametrize(
    "extra_cfg, expected",
    [({}, 64), ({'batch_size': None}, 64), ({'batch_size': 7}, 7)],
    ids=["missing", "null", "explicit"],
)
def test_dataloader_configs_get_a_batch_size(extra_cfg, expected):
    diarizer = _diarizer(extra_cfg)

    diarizer._setup_vad_test_data('vad_manifest.json')
    diarizer._setup_spkr_test_data('spk_manifest.json')

    vad_config = diarizer._vad_model.setup_test_data.call_args.kwargs['test_data_config']
    spk_config = diarizer._speaker_model.setup_test_data.call_args.args[0]
    assert vad_config['batch_size'] == expected
    assert spk_config['batch_size'] == expected
