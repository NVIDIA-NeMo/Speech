# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G2P tokenizer artifacts survive model archive round trips."""

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
from tests.core.test_save_restore import MockModel


@pytest.mark.unit
@pytest.mark.parametrize('configured_vocab', [False, True])
def test_grapheme_tokenizer_archive_round_trip(tmp_path, configured_vocab):
    vocab = tmp_path / 'graphemes.txt'
    vocab.write_text('"x"\n"y"\n')
    cfg = OmegaConf.create(
        {
            'tokenizer_grapheme': {
                'dataset': {'_target_': 'nemo.collections.common.tokenizers.char_tokenizer.CharTokenizer'},
                'vocab_file': str(vocab) if configured_vocab else None,
                'unk_token': None,
                'do_lower': True,
                'add_punctuation': False,
            }
        }
    )
    model = _GraphemeArtifactModel(cfg)
    expected = model.grapheme_tokenizer.vocab
    archive = tmp_path / 'original.nemo'
    model.save_to(str(archive))
    vocab.unlink()

    restored = _GraphemeArtifactModel.restore_from(str(archive), map_location='cpu')
    assert restored.grapheme_tokenizer.vocab == expected
    if configured_vocab:
        assert 'x' in expected and 'a' not in expected
    repacked = tmp_path / 'repacked.nemo'
    restored.save_to(str(repacked))
    assert _GraphemeArtifactModel.restore_from(str(repacked), map_location='cpu').grapheme_tokenizer.vocab == expected


class _GraphemeArtifactModel(MockModel):
    """Exercise the production tokenizer setup without constructing an acoustic encoder."""

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.mode = 'conformer_bpe'
        self.grapheme_tokenizer = CTCG2PModel.setup_grapheme_tokenizer(self, self.cfg)
