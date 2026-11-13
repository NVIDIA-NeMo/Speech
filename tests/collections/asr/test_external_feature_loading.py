# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest
import torch

from nemo.collections.asr.data.feature_to_label import FeatureToLabelDataset
from nemo.collections.asr.parts.preprocessing.feature_loader import ExternalFeatureLoader


@pytest.mark.unit
def test_feature_dataset_rejects_object_array(tmp_path):
    features = tmp_path / 'features.npy'
    np.save(features, np.array([object()], dtype=object))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'feature_file': str(features), 'duration': 1.0, 'label': '0'}) + '\n')
    dataset = FeatureToLabelDataset(manifest_filepath=str(manifest), labels=['0'])
    with pytest.raises(ValueError, match='Object arrays cannot be loaded'):
        dataset[0]


@pytest.mark.unit
@pytest.mark.parametrize('suffix', ['.npy', '.pt', '.pth'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16])
def test_numeric_feature_dataset_round_trip(tmp_path, suffix, dtype):
    array = np.arange(12, dtype=dtype).reshape(3, 4)
    features = tmp_path / f'features{suffix}'
    if suffix == '.npy':
        np.save(features, array)
    else:
        torch.save(torch.from_numpy(array), features)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(
        ''.join(json.dumps({'feature_file': str(features), 'duration': 1.0, 'label': '0'}) + '\n' for _ in range(2))
    )
    dataset = FeatureToLabelDataset(manifest_filepath=str(manifest), labels=['0'])
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    expected = array.astype(np.float32)
    if suffix == '.npy' and dtype == np.int16:
        expected /= 2**15
    torch.testing.assert_close(batch[0], torch.from_numpy(np.stack([expected, expected])))


@pytest.mark.unit
def test_feature_loader_rejects_numpy_archive(tmp_path):
    features = tmp_path / 'features.npz'
    np.savez(features, features=np.ones((3, 4)))
    with pytest.raises(ValueError, match='single NumPy array'):
        ExternalFeatureLoader().process(str(features))
