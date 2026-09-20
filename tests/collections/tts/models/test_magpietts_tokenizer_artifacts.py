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

"""
Unit tests for MagpieTTSModel._register_tokenizer_artifacts (.nemo packaging of tokenizer files).

A code-switching tokenizer (e.g. Hindi = hi_prondict + ipa_cmudict) stores ``g2p.phoneme_dict`` as
a LIST of files. These tests pin the two behaviors that make such a model round-trip through .nemo:

1. List elements are registered under a DOT-indexed config path (``phoneme_dict.{i}``), so the
   save/restore connector's ``OmegaConf.update`` writes each back into the list. An underscore
   (``phoneme_dict_{i}``) would instead create sibling keys ``phoneme_dict_0/_1`` (and leave
   ``phoneme_dict`` null), which ``IpaG2p`` rejects on restore.
2. A full ``save_to`` -> ``restore_from`` round-trip resolves every list entry to an existing,
   packaged file (and produces no ``phoneme_dict_{i}`` sibling keys) -- and the common string case
   still round-trips too.

The tests exercise the real ``MagpieTTSModel._register_tokenizer_artifacts`` without building a full
model (no codec / encoders / GPU): a mock ``self`` for the registration-path test, and a minimal
``ModelPT`` subclass that reuses the method for the save/restore round-trip.
"""

import os
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import ListConfig, OmegaConf

from nemo.collections.tts.models.magpietts import MagpieTTSModel
from nemo.core.classes import ModelPT

_IPA_TOKENIZER = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"
_IPA_G2P = "nemo.collections.tts.g2p.models.i18n_ipa.IpaG2p"


def _g2p_cfg(phoneme_dict):
    """Build a minimal text_tokenizers config whose single tokenizer carries `phoneme_dict`."""
    return OmegaConf.create(
        {
            "text_tokenizers": {
                "hindi_phoneme": {
                    "_target_": _IPA_TOKENIZER,
                    "g2p": {"_target_": _IPA_G2P, "phoneme_dict": phoneme_dict},
                }
            }
        }
    )


class _TokenizerArtifactModel(ModelPT):
    """Minimal ModelPT that reuses MagpieTTSModel's tokenizer-artifact registration, for round-trips.

    It deliberately builds no tokenizers/codec -- it only registers the tokenizer file artifacts, so
    save_to/restore_from exercises exactly the artifact packaging/resolution path under test.
    """

    def __init__(self, cfg, trainer=None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.w = torch.nn.Linear(1, 1)  # ensure a non-empty state dict to save
        MagpieTTSModel._register_tokenizer_artifacts(self, self.cfg)
        # Mirror real usage (the tokenizer reads the dict during __init__) and consume the artifact
        # now, while it is on disk. On restore NeMo extracts artifacts to a temp dir that is removed
        # once restore_from returns, so reading here is what proves the registered path resolved to a
        # real, packaged file. We record the basenames read so tests can assert on a restored model.
        self.loaded_phoneme_dicts = self._read_phoneme_dicts(self.cfg)

    @staticmethod
    def _read_phoneme_dicts(cfg):
        g2p = cfg.text_tokenizers.hindi_phoneme.g2p
        phoneme_dict = g2p.phoneme_dict
        paths = list(phoneme_dict) if isinstance(phoneme_dict, (list, ListConfig)) else [phoneme_dict]
        basenames = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                f.read()
            basenames.append(os.path.basename(path))
        return basenames

    def setup_training_data(self, train_data_config):
        self._train_dl = None

    def setup_validation_data(self, val_data_config):
        self._validation_dl = None

    def setup_test_data(self, test_data_config):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class TestRegisterTokenizerArtifacts:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_list_phoneme_dict_uses_dot_indexed_paths(self):
        """A list phoneme_dict registers each element under `phoneme_dict.{i}` (dot) and stays a list."""
        cfg = _g2p_cfg(["/data/hi_prondict.dict", "/data/ipa_cmudict.txt"])

        mock_model = MagicMock(spec=MagpieTTSModel)
        # Mimic register_artifact's contract: return an absolute path for a given src.
        mock_model.register_artifact.side_effect = lambda config_path, src, verify_src_exists=True: os.path.abspath(
            src
        )

        MagpieTTSModel._register_tokenizer_artifacts(mock_model, cfg)

        registered_paths = [call.args[0] for call in mock_model.register_artifact.call_args_list]
        assert "text_tokenizers.hindi_phoneme.g2p.phoneme_dict.0" in registered_paths
        assert "text_tokenizers.hindi_phoneme.g2p.phoneme_dict.1" in registered_paths
        # Regression guard: the underscore form is what broke .nemo restore.
        assert not any("phoneme_dict_0" in p or "phoneme_dict_1" in p for p in registered_paths)

        # phoneme_dict must remain a 2-element list (not collapsed to sibling keys / null).
        g2p = cfg.text_tokenizers.hindi_phoneme.g2p
        assert isinstance(g2p.phoneme_dict, ListConfig) and len(g2p.phoneme_dict) == 2
        assert "phoneme_dict_0" not in g2p and "phoneme_dict_1" not in g2p

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_list_phoneme_dict_survives_nemo_save_restore(self, tmp_path):
        """End-to-end: a list phoneme_dict resolves to existing packaged files after save_to/restore_from."""
        d1 = tmp_path / "hi_prondict.dict"
        d1.write_text("नमस्ते\tn a m a s t e\n", encoding="utf-8")
        d2 = tmp_path / "ipa_cmudict.txt"
        d2.write_text("HELLO\th ʌ l o\n", encoding="utf-8")

        model = _TokenizerArtifactModel(_g2p_cfg([str(d1), str(d2)]))
        # After registration the config is a 2-element list of absolute paths.
        pd = model.cfg.text_tokenizers.hindi_phoneme.g2p.phoneme_dict
        assert isinstance(pd, ListConfig) and len(pd) == 2

        nemo_path = str(tmp_path / "list_dict_model.nemo")
        model.save_to(nemo_path)
        restored = _TokenizerArtifactModel.restore_from(nemo_path, map_location="cpu")

        g2p = restored.cfg.text_tokenizers.hindi_phoneme.g2p
        rpd = g2p.phoneme_dict
        assert isinstance(rpd, ListConfig) and len(rpd) == 2
        # No sibling keys from the underscore bug, and the list is intact.
        assert "phoneme_dict_0" not in g2p and "phoneme_dict_1" not in g2p
        # Both list entries resolved to real packaged files at restore time (read in __init__).
        restored_basenames = " ".join(restored.loaded_phoneme_dicts)
        assert len(restored.loaded_phoneme_dicts) == 2
        assert "hi_prondict.dict" in restored_basenames
        assert "ipa_cmudict.txt" in restored_basenames

    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_string_phoneme_dict_survives_nemo_save_restore(self, tmp_path):
        """The common single-file (string) phoneme_dict still round-trips through .nemo."""
        d = tmp_path / "en.dict"
        d.write_text("HELLO\th ʌ l o\n", encoding="utf-8")

        model = _TokenizerArtifactModel(_g2p_cfg(str(d)))
        nemo_path = str(tmp_path / "string_dict_model.nemo")
        model.save_to(nemo_path)
        restored = _TokenizerArtifactModel.restore_from(nemo_path, map_location="cpu")

        rpd = restored.cfg.text_tokenizers.hindi_phoneme.g2p.phoneme_dict
        assert isinstance(rpd, str)
        # The single entry resolved to a real packaged file at restore time (read in __init__).
        assert restored.loaded_phoneme_dicts == [os.path.basename(rpd)]
        assert restored.loaded_phoneme_dicts[0].endswith("en.dict")
