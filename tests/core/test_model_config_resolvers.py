# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tarfile

import pytest
from omegaconf import OmegaConf

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils.config_utils import resolve_model_config
from tests.core.test_save_restore import MockModel


@pytest.mark.unit
@pytest.mark.parametrize(
    'expression',
    [
        '${oc.env:MODEL_TEST_VALUE,unset}',
        '${oc.decode:"${oc.env:MODEL_TEST_VALUE,unset}"}',
        '${${resolver}:MODEL_TEST_VALUE,unset}',
        '${oc.select:missing,${oc.env:MODEL_TEST_VALUE,unset}}',
        r'\${oc.env:MODEL_TEST_VALUE,unset}',
        '\\' * 3 + '${oc.env:MODEL_TEST_VALUE,unset}',
        '\\' * 7 + '${oc.env:MODEL_TEST_VALUE,unset}',
        '${sum:${prefix},${suffix}}',
    ],
)
@pytest.mark.parametrize('return_config', [False, True])
def test_restore_rejects_external_config_resolvers(model_archive, monkeypatch, expression, return_config):
    monkeypatch.setenv('MODEL_TEST_VALUE', 'synthetic-value')
    archive, extracted = model_archive
    config_path = extracted / 'model_config.yaml'
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.metadata = {'values': [expression], 'alias': '${metadata.values.0}'}
    config.resolver = 'oc.env'
    config.prefix = '$'
    config.suffix = '{oc.env:MODEL_TEST_VALUE,unset}'
    OmegaConf.save(config, config_path)
    with tarfile.open(archive, 'w') as dest:
        for item in extracted.iterdir():
            dest.add(item, arcname=item.name)
    with pytest.raises(ValueError, match='resolver') as error:
        MockModel.restore_from(archive, return_config=return_config, map_location='cpu')
    assert 'synthetic-value' not in str(error.value)


@pytest.mark.unit
def test_restore_preserves_internal_config_references(tmp_path, model_archive):
    archive, extracted = model_archive
    connector = SaveRestoreConnector()
    config_path = extracted / 'model_config.yaml'
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)
    config.width = 4
    config.metadata = {'width': '${width}', 'double_width': '${multiply:${width},2}'}
    OmegaConf.save(config, config_path)
    connector.model_extracted_dir = str(extracted)
    restored = MockModel.restore_from(archive, save_restore_connector=connector, map_location='cpu')
    assert restored.cfg.metadata.width == 4
    assert restored.cfg.metadata.double_width == 8
    output = tmp_path / 'roundtrip.nemo'
    restored.save_to(output)
    assert MockModel.restore_from(output, return_config=True).metadata.double_width == 8


@pytest.mark.unit
def test_trusted_override_config_can_use_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_TEST_VALUE', 'synthetic-value')
    model = MockModel(OmegaConf.create({}))
    archive = tmp_path / 'model.nemo'
    model.save_to(archive)
    override = MockModel.restore_from(archive, return_config=True)
    OmegaConf.set_struct(override, False)
    override.metadata = '${oc.env:MODEL_TEST_VALUE}'
    restored = MockModel.restore_from(archive, override_config_path=override, map_location='cpu')
    assert restored.cfg.metadata == 'synthetic-value'


@pytest.fixture
def model_archive(tmp_path):
    model = MockModel(OmegaConf.create({}))
    archive = tmp_path / 'model.nemo'
    model.save_to(archive)
    extracted = tmp_path / 'extracted'
    extracted.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(extracted, filter='data')
    return archive, extracted


@pytest.mark.unit
@pytest.mark.parametrize('escapes', [1, 3, 7])
def test_model_config_helper_rejects_escaped_expressions(escapes):
    config = OmegaConf.create({'metadata': '\\' * escapes + '${oc.env:MODEL_TEST_VALUE,unset}'})
    with pytest.raises(ValueError, match='resolver'):
        resolve_model_config(config)


@pytest.mark.unit
def test_model_config_helper_preserves_arithmetic_and_literal_paths():
    config = OmegaConf.create(
        {'width': 4, 'double': '${multiply:${width},2}', 'total': '${sum:4,2}', 'path': r'C:\models\checkpoint'}
    )
    result = resolve_model_config(config)
    assert result.width == 4
    assert result.double == 8
    assert result.total == 6
    assert result.path == config.path
