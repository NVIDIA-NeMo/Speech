# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from scripts.tokenizers import get_hf_text_data


@pytest.mark.unit
@pytest.mark.parametrize(
    'filename', ['corpus.txt', 'corpus with spaces.txt', "corpus;'quoted'.txt", 'corpus$(name).txt']
)
def test_corpus_output_is_replaced_as_literal_path(tmp_path, monkeypatch, filename):
    output = tmp_path / filename
    output.write_text('old corpus\n')
    monkeypatch.setattr(os, 'system', Mock(side_effect=AssertionError('unexpected shell')))
    monkeypatch.setattr(get_hf_text_data.hf_datasets, 'load_dataset', lambda **kwargs: [{'text': kwargs['path']}])
    config = OmegaConf.create(
        {
            'output_file': str(output),
            'hf_data_cfg': [{'path': 'first'}, {'path': 'second'}],
            'num_workers': 1,
            'normalize_text': False,
            'text_key': 'text',
        }
    )
    get_hf_text_data.main.__wrapped__(config)
    assert output.read_text() == 'first\nsecond\n'


@pytest.mark.unit
def test_corpus_cleanup_preserves_symlink_target(tmp_path, monkeypatch):
    target = tmp_path / 'source.txt'
    target.write_text('source\n')
    output = tmp_path / 'corpus.txt'
    output.symlink_to(target)
    monkeypatch.setattr(os, 'system', Mock(side_effect=AssertionError('unexpected shell')))
    config = OmegaConf.create({'output_file': str(output), 'hf_data_cfg': []})
    get_hf_text_data.main.__wrapped__(config)
    assert not output.is_symlink()
    assert target.read_text() == 'source\n'
