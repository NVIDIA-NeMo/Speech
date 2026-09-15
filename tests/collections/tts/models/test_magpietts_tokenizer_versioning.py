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
Unit tests for the versioned tokenizer defaults that MagpieTTS configs carry.

``HindiCharsTokenizer`` / ``ArabicCharsTokenizer`` (``charset_version``, ``punct_version``) and the
pt-BR ``IPATokenizer`` (``locale_specific_punct``) each gained a second character/punctuation set after
models had already been trained and released. A config that omits those fields is therefore ambiguous:
it is either an archive that predates them (and whose vocabulary was built with the v1 values) or a
fresh training config that should get today's defaults. These tests pin both readings:

1. ``setup_tokenizers`` uses the current defaults for fresh configs, and the pre-versioning values only
   for configs stamped with a release that predates the fields -- so new training is never silently
   downgraded to v1.
2. The ``nemo_version`` stamp that ``ModelPT`` writes into every config it saves is what dates the
   config, and ``VERSIONED_TOKENIZER_FIELDS`` records the release each default changed in.
3. Whatever the resolved values are, they are written back into the config, so a model saved today
   restores to the same token-to-ID mapping no matter how the class defaults evolve later.
"""

import inspect
from unittest.mock import MagicMock, patch

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from packaging.version import Version

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    VERSIONED_TOKENIZER_FIELDS,
    resolve_versioned_tokenizer_defaults,
)
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import (
    check_text_embedding_matches_tokenizer,
    setup_tokenizers,
)
from nemo.core.classes import ModelPT

_HINDI_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.HindiCharsTokenizer"
_ARABIC_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.ArabicCharsTokenizer"
_IPA = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"

# A release that predates the versioned fields (both v2512 and v2602 are stamped exactly this) and one
# that postdates them. "2.8.0rc0" is deliberately the *current* side: see ``_predates_nemo_release``.
LEGACY_VERSION = "2.6.0rc0"
CURRENT_VERSION = "2.8.0rc0"


def _tokenizers_cfg(**tokenizer_fields):
    """A single-entry ``text_tokenizers`` config for the Hindi character tokenizer."""
    return OmegaConf.create({"hindi_chartokenizer": {"_target_": _HINDI_CHARS, **tokenizer_fields}})


class _TokenizerVersionModel(ModelPT):
    """Minimal stand-in for MagpieTTS that reproduces only its tokenizer/embedding sizing.

    It mirrors the two lines that matter -- resolve the tokenizer defaults according to the config's
    provenance, then size the text embedding from the resulting vocabulary -- so a ``save_to`` /
    ``restore_from`` round-trip exercises the real vocabulary-drift failure without building a codec,
    encoders, or downloading a released archive.
    """

    def __init__(self, cfg, trainer=None):
        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            cfg_nemo_version=cfg.get('nemo_version', None),
        )
        super().__init__(cfg=cfg, trainer=trainer)
        self.text_embedding = torch.nn.Embedding(len(self.tokenizer.tokens) + 2, 4)

    def setup_training_data(self, train_data_config):
        self._train_dl = None

    def setup_validation_data(self, val_data_config):
        self._validation_dl = None

    def setup_test_data(self, test_data_config):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class TestSetupTokenizersDefaults:
    """Covers the plumbing from ``setup_tokenizers`` down to the resolver. Which value each stamp
    resolves to is pinned by ``TestVersionDating``, and what each value *means* for the resulting
    vocabulary by ``test_tts_tokenizers.py`` at the tokenizer-class level."""

    @pytest.mark.unit
    def test_version_reaches_the_tokenizer_config(self):
        """The one line of plumbing between the model constructor and the resolver."""
        cfg = _tokenizers_cfg()

        setup_tokenizers(cfg, cfg_nemo_version=LEGACY_VERSION)

        assert cfg.hindi_chartokenizer.charset_version == 1
        assert cfg.hindi_chartokenizer.punct_version == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("cfg_nemo_version", [CURRENT_VERSION, LEGACY_VERSION])
    def test_explicit_values_are_never_overridden(self, cfg_nemo_version):
        """An explicitly configured version wins over both defaults -- that is what pins v2607."""
        cfg = _tokenizers_cfg(charset_version=1, punct_version=2)

        setup_tokenizers(cfg, cfg_nemo_version=cfg_nemo_version)

        assert cfg.hindi_chartokenizer.charset_version == 1
        assert cfg.hindi_chartokenizer.punct_version == 2


class TestResolveVersionedTokenizerDefaults:
    # The current direction for these two targets is covered by
    # ``test_current_backfill_matches_the_tokenizer_class_default``, which asserts something stronger.
    @pytest.mark.unit
    def test_arabic_charset_version_dates_back_to_v1(self):
        cfg = OmegaConf.create({"_target_": _ARABIC_CHARS})

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert cfg.charset_version == 1

    @pytest.mark.unit
    def test_pt_br_locale_specific_punct_dates_back_to_off(self):
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "pt-BR"})

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert cfg.locale_specific_punct is False

    @pytest.mark.unit
    def test_non_default_punct_list_suppresses_pt_br_backfill(self):
        """An explicit punctuation list already fixes the vocabulary; adding the flag would fight it."""
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "pt-BR", "non_default_punct_list": [".", ","]})

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert "locale_specific_punct" not in cfg

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_fields, field, expected",
        [
            ({"chars": None}, "charset_version", 1),
            ({"non_default_punct_list": None}, "punct_version", 1),
            ({"charset_version": None}, "charset_version", 1),
        ],
        ids=["null-chars", "null-punct-list", "null-field-itself"],
    )
    def test_explicit_null_counts_as_unset(self, cfg_fields, field, expected):
        """``chars: null`` must not read as "specified".

        The tokenizers gate their version branches on ``is None``, so a null suppressing the backfill
        would leave the branch running against the class default rather than the dated value.
        """
        cfg = OmegaConf.create({"_target_": _HINDI_CHARS, **cfg_fields})

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert cfg[field] == expected

    @pytest.mark.unit
    def test_other_ipa_locales_are_untouched(self):
        """Only pt-BR's punctuation set diverged from DEFAULT_PUNCTUATION, so only it is backfilled."""
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "es-ES"})

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert "locale_specific_punct" not in cfg

    @pytest.mark.unit
    @pytest.mark.parametrize("extra_cfg", [{}, {"_target_": None}], ids=["absent", "null"])
    def test_config_without_target_is_ignored(self, extra_cfg):
        """HuggingFace tokenizer nodes carry no resolvable ``_target_`` and must pass through untouched."""
        cfg = OmegaConf.create({"pretrained_model": "google-t5/t5-small", **extra_cfg})
        before = OmegaConf.to_container(cfg)

        resolve_versioned_tokenizer_defaults(cfg, LEGACY_VERSION)

        assert OmegaConf.to_container(cfg) == before

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "target, field, extra_cfg",
        [
            (_HINDI_CHARS, "charset_version", {}),
            (_HINDI_CHARS, "punct_version", {}),
            (_ARABIC_CHARS, "charset_version", {}),
            (_IPA, "locale_specific_punct", {"locale": "pt-BR"}),
        ],
    )
    def test_current_backfill_matches_the_tokenizer_class_default(self, target, field, extra_cfg):
        """What gets persisted for a current config must equal what the tokenizer would have chosen itself.

        ``VERSIONED_TOKENIZER_FIELDS`` and the tokenizer signatures read the same ``DEFAULT_*`` constants,
        so this holds by construction today. It is kept as the guard against someone re-introducing a
        literal on either side, which would silently pin every newly trained model a version behind.
        """
        class_default = inspect.signature(hydra.utils.get_class(target)).parameters[field].default
        cfg = OmegaConf.create({"_target_": target, **extra_cfg})

        resolve_versioned_tokenizer_defaults(cfg, CURRENT_VERSION)

        assert cfg[field] == class_default

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_fields, cfg_nemo_version, should_warn",
        [
            ({}, LEGACY_VERSION, True),
            ({"_target_": _IPA, "locale": "pt-BR"}, LEGACY_VERSION, True),
            ({}, CURRENT_VERSION, False),
            ({"punct_version": 1, "charset_version": 1}, LEGACY_VERSION, False),
        ],
        ids=["legacy-hindi", "legacy-pt-br", "current", "already-explicit"],
    )
    def test_legacy_backfill_is_never_silent(self, cfg_fields, cfg_nemo_version, should_warn):
        """Falling back to a pre-versioning vocabulary must say so.

        The tokenizers cannot be relied on for this: the pt-BR path emits nothing at all, and the
        Hindi/Arabic ``DeprecationWarning`` is swallowed by Python's default filters outside pytest.
        Warning only on the legacy direction keeps ordinary training runs quiet.
        """
        cfg = OmegaConf.create({"_target_": _HINDI_CHARS, **cfg_fields})

        with patch("nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.logging.warning") as mock_warning:
            resolve_versioned_tokenizer_defaults(cfg, cfg_nemo_version)

        assert mock_warning.called is should_warn
        if should_warn:
            # The message has to name the field, otherwise it is not actionable.
            assert any(field in mock_warning.call_args.args[0] for field in ("punct_version", "locale_specific_punct"))


class TestVersionDating:
    """How a config's ``nemo_version`` stamp maps onto the versioned fields."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_nemo_version, expect_legacy",
        [
            ("2.6.0rc0", True),  # v2512 and v2602 are both stamped exactly this
            ("2.7.3", True),
            ("2.8.0rc0", False),  # v2607; the rc window is broken toward the current defaults
            ("2.8.0", False),
            ("3.1.0", False),
            (None, False),  # hand-authored fresh training config
            ("not-a-version", False),
        ],
        ids=["v2512-v2602", "2.7.3", "v2607-rc", "2.8.0", "3.1.0", "unstamped", "garbage"],
    )
    def test_stamp_decides_the_backfilled_value(self, cfg_nemo_version, expect_legacy):
        cfg = OmegaConf.create({"_target_": _HINDI_CHARS})

        resolve_versioned_tokenizer_defaults(cfg, cfg_nemo_version)

        assert cfg.charset_version == (1 if expect_legacy else 2)

    @pytest.mark.unit
    @pytest.mark.parametrize("spec", VERSIONED_TOKENIZER_FIELDS, ids=lambda s: s.field)
    def test_every_field_documents_a_parseable_release(self, spec):
        """``changed_in`` is compared against real stamps, so it has to parse and differ from legacy."""
        assert Version(spec.changed_in).release
        assert spec.current != spec.legacy


class _StopAtTokenizerSetup(Exception):
    """Raised from the patched ``setup_tokenizers`` to end ``__init__`` once the call site has run."""


def _mock_codec():
    """An AudioCodecModel stand-in with the numeric attributes ``__init__`` reads before tokenizer setup."""
    codec = MagicMock()
    codec.sample_rate = 22050
    codec.output_sample_rate = 22050
    codec.samples_per_frame = 1024
    codec.num_codebooks = 8
    codec.codebook_size = 1000
    return codec


class TestProductionCallSites:
    """The wiring in the real model constructors, which is the whole fix.

    Without this, mutating either call site (deleting the kwarg, or negating it) leaves the entire TTS
    unit suite green while reintroducing the v2602 restore failure -- the tests below are the only thing
    that fails on such a mutation, because every other test drives ``setup_tokenizers`` directly.

    Each constructor is stopped at the tokenizer-setup call rather than run to completion, so no codec,
    encoders, or downloads are needed.
    """

    @staticmethod
    def _captured_kwargs(model_cls, module_path, cfg_extra, codec_attr):
        cfg = OmegaConf.create(
            {
                "codecmodel_path": "nvidia/fake-codec",
                "text_tokenizers": _tokenizers_cfg(),
                **cfg_extra,
            }
        )
        captured = {}
        with (
            patch(f"{module_path}.AudioCodecModel") as mock_codec,
            patch(f"{module_path}.setup_tokenizers") as mock_setup,
        ):
            getattr(mock_codec, codec_attr).return_value = _mock_codec()

            def _record(*_args, **kwargs):
                captured.update(kwargs)
                raise _StopAtTokenizerSetup()

            mock_setup.side_effect = _record
            with pytest.raises(_StopAtTokenizerSetup):
                model_cls(cfg=cfg)
        return captured

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_extra, expected",
        [({}, None), ({"nemo_version": LEGACY_VERSION}, LEGACY_VERSION)],
        ids=["fresh", "serialized"],
    )
    def test_magpietts_passes_config_nemo_version(self, cfg_extra, expected):
        from nemo.collections.tts.models.magpietts import MagpieTTSModel

        captured = self._captured_kwargs(
            MagpieTTSModel, "nemo.collections.tts.models.magpietts", cfg_extra, "from_pretrained"
        )

        assert captured["cfg_nemo_version"] == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_extra, expected",
        [({}, None), ({"nemo_version": LEGACY_VERSION}, LEGACY_VERSION)],
        ids=["fresh", "serialized"],
    )
    def test_easy_magpietts_passes_config_nemo_version(self, cfg_extra, expected):
        from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

        captured = self._captured_kwargs(
            EasyMagpieTTSInferenceModel,
            "nemo.collections.tts.models.easy_magpietts_inference",
            cfg_extra,
            "restore_from",
        )

        assert captured["cfg_nemo_version"] == expected


class TestTextEmbeddingMismatchError:
    """The diagnostic a user actually hits when a vocabulary is rebuilt the wrong way."""

    @staticmethod
    def _model_and_state_dict(ckpt_rows):
        model = MagicMock()
        model.num_tokens_per_tokenizer = {"hindi_chartokenizer": 191}
        return {"text_embedding.weight": torch.zeros(ckpt_rows, 4)}, model

    @pytest.mark.unit
    def test_matching_sizes_pass(self):
        state_dict, tokenizer = self._model_and_state_dict(100)

        check_text_embedding_matches_tokenizer(
            state_dict, text_embedding=torch.nn.Embedding(100, 4), tokenizer=tokenizer, model_cfg=OmegaConf.create({})
        )

    @pytest.mark.unit
    def test_mismatch_names_the_fields_to_pin(self):
        state_dict, tokenizer = self._model_and_state_dict(193)

        with pytest.raises(RuntimeError) as excinfo:
            check_text_embedding_matches_tokenizer(
                state_dict,
                text_embedding=torch.nn.Embedding(148, 4),
                tokenizer=tokenizer,
                model_cfg=OmegaConf.create({"nemo_version": LEGACY_VERSION}),
            )

        message = str(excinfo.value)
        assert "193" in message and "148" in message  # both sides of the mismatch
        assert LEGACY_VERSION in message  # the stamp that drove the decision
        assert "hindi_chartokenizer" in message  # which tokenizer to look at
        for field in ("charset_version", "punct_version", "locale_specific_punct"):
            assert field in message  # what to set

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "state_dict, text_embedding",
        [({}, torch.nn.Embedding(10, 4)), ({"text_embedding.weight": torch.zeros(10, 4)}, None)],
        ids=["no-weight-in-ckpt", "cas-encoder-variant-has-no-table"],
    )
    def test_absent_embedding_is_not_an_error(self, state_dict, text_embedding):
        """The CAS-encoder MagpieTTS variant has no text_embedding to compare; it must not trip here."""
        check_text_embedding_matches_tokenizer(
            state_dict, text_embedding=text_embedding, tokenizer=MagicMock(), model_cfg=OmegaConf.create({})
        )


class TestNemoRoundTrip:
    """End-to-end ``save_to``/``restore_from`` coverage of the vocabulary-drift failure."""

    @staticmethod
    def _save_legacy_archive(tmp_path):
        """Write a .nemo that looks like a pre-versioning release (v2512/v2602).

        Such archives were trained with the v1 charset/punctuation but their configs name neither, so
        the versioned fields are stripped back out after the model is built. The ``nemo_version`` stamp
        is what remains to date them -- both released archives carry exactly ``2.6.0rc0``. Setting it
        before ``super().__init__`` is also what a real restore does: ``ModelPT`` only stamps a config
        that has no version yet, so the original release's stamp survives every later save.
        """
        model = _TokenizerVersionModel(
            OmegaConf.create({"text_tokenizers": _tokenizers_cfg(), "nemo_version": LEGACY_VERSION})
        )
        model.text_embedding = torch.nn.Embedding(
            len(setup_tokenizers(_tokenizers_cfg(), cfg_nemo_version=LEGACY_VERSION).tokens) + 2, 4
        )
        with open_dict(model.cfg):
            del model.cfg.text_tokenizers.hindi_chartokenizer.charset_version
            del model.cfg.text_tokenizers.hindi_chartokenizer.punct_version
        path = str(tmp_path / "legacy.nemo")
        model.save_to(path)
        return path, model.text_embedding.num_embeddings

    @pytest.mark.unit
    def test_archive_without_version_fields_restores_with_v1_vocab(self, tmp_path):
        """Regression: a released checkpoint that predates the fields must not pick up the v2 charsets.

        Picking them up rebuilds a vocabulary of a different size than the checkpoint was trained with
        (v2 collapses case, so it is the *smaller* of the two), and ``restore_from`` then dies on a
        text_embedding size mismatch -- exactly how v2602 broke.
        """
        path, expected_num_embeddings = self._save_legacy_archive(tmp_path)

        restored = _TokenizerVersionModel.restore_from(path, map_location="cpu")

        assert restored.text_embedding.num_embeddings == len(restored.tokenizer.tokens) + 2
        assert restored.text_embedding.num_embeddings == expected_num_embeddings

    @pytest.mark.unit
    def test_newly_trained_model_round_trips_on_current_defaults(self, tmp_path):
        """A model trained today keeps its v2 vocabulary through a save/restore cycle.

        The archive is stamped with the current release, so nothing here depends on the dating rule:
        this pins that the versions persisted at save time are what decide the vocabulary, which is
        what makes every future release self-describing regardless of how the class defaults move.
        """
        model = _TokenizerVersionModel(OmegaConf.create({"text_tokenizers": _tokenizers_cfg()}))
        assert model.cfg.text_tokenizers.hindi_chartokenizer.charset_version == 2
        path = str(tmp_path / "current.nemo")
        model.save_to(path)

        restored = _TokenizerVersionModel.restore_from(path, map_location="cpu")

        assert restored.cfg.text_tokenizers.hindi_chartokenizer.charset_version == 2
        assert restored.cfg.text_tokenizers.hindi_chartokenizer.punct_version == 2
        assert restored.text_embedding.num_embeddings == model.text_embedding.num_embeddings
