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

"""Language-ID prompt conditioning in :class:`EncDecRNNTBPEModelWithPrompt`.

Covers the language-resolution surface that multilingual ("unified") checkpoints rely on and the
``prompt`` / ``prompt_indices`` forward arguments, including that the pre-existing
``prompt_indices`` behaviour is unchanged.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecRNNTModel
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt, RNNTPromptTranscribeConfig
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__ as NUMBA_MINIMUM_VERSION
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE, numba_cpu_is_supported

ENC_HIDDEN = 64
NUM_PROMPTS = 8
PROMPT_DICTIONARY = {'en-US': 0, 'de-DE': 3, 'ja-JP': 5, 'unk': 7}

pytestmark = pytest.mark.skipif(
    not numba_cpu_is_supported(NUMBA_MINIMUM_VERSION),
    reason=f'RNNTLoss has not been compiled with appropriate numba version. {NUMBA_INSTALLATION_MESSAGE}',
)


def build_config(tokenizer_dir: str, prompt_enabled: bool, prompt_dictionary=None, extra_defaults=None) -> DictConfig:
    """Build a tiny prompt-conditioned RNNT BPE config, optionally with prompting enabled."""
    model_defaults = {'enc_hidden': ENC_HIDDEN, 'pred_hidden': 32}
    if prompt_enabled:
        model_defaults.update(
            {
                'initialize_prompt_feature': True,
                'num_prompts': NUM_PROMPTS,
                'prompt_dictionary': PROMPT_DICTIONARY if prompt_dictionary is None else prompt_dictionary,
            }
        )
    if extra_defaults:
        model_defaults.update(extra_defaults)

    return DictConfig(
        {
            'preprocessor': DictConfig(
                {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': {}}
            ),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(
                {
                    'cls': 'nemo.collections.asr.modules.ConvASREncoder',
                    'params': {
                        'feat_in': 64,
                        'activation': 'relu',
                        'conv_mask': True,
                        'jasper': [
                            {
                                'filters': ENC_HIDDEN,
                                'repeat': 1,
                                'kernel': [1],
                                'stride': [1],
                                'dilation': [1],
                                'dropout': 0.0,
                                'residual': False,
                                'separable': True,
                                'se': True,
                                'se_context_size': -1,
                            }
                        ],
                    },
                }
            ),
            'decoder': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
                    'prednet': {'pred_hidden': 32, 'pred_rnn_layers': 1},
                }
            ),
            'joint': DictConfig(
                {
                    '_target_': 'nemo.collections.asr.modules.RNNTJoint',
                    'jointnet': {'joint_hidden': 32, 'activation': 'relu'},
                }
            ),
            'tokenizer': DictConfig(
                {'dir': os.path.join(tokenizer_dir, 'asr', 'tokenizers', 'an4_wpe_128'), 'type': 'wpe'}
            ),
            'decoding': DictConfig({'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}),
            'loss': DictConfig({'loss_name': 'default'}),
        }
    )


@pytest.fixture()
def build_prompt_model(test_data_dir):
    """Factory for tiny :class:`EncDecRNNTBPEModelWithPrompt` instances."""

    def _build(prompt_enabled=True, prompt_dictionary=None, extra_defaults=None):
        cfg = build_config(test_data_dir, prompt_enabled, prompt_dictionary, extra_defaults)
        # eval() disables the preprocessor's dither, so two forward passes over the same audio are
        # comparable; several tests below assert exact agreement between prompt paths.
        return EncDecRNNTBPEModelWithPrompt(cfg=cfg).eval()

    return _build


@pytest.fixture()
def prompt_model(build_prompt_model):
    """A model with language-ID prompt conditioning enabled."""
    return build_prompt_model(prompt_enabled=True)


@pytest.fixture()
def unconditioned_model(build_prompt_model):
    """The same class instantiated from a config that does not enable prompting."""
    return build_prompt_model(prompt_enabled=False)


def make_audio(batch_size=2, samples=4000):
    """A deterministic audio batch and its lengths."""
    generator = torch.Generator().manual_seed(0)
    signal = torch.randn(batch_size, samples, generator=generator)
    return signal, torch.full((batch_size,), samples, dtype=torch.long)


class TestForwardPromptArguments:
    """``forward`` accepts language indices or a pre-built one-hot, mirroring the hybrid variant."""

    @pytest.mark.unit
    def test_prompt_indices_condition_the_encoder(self, prompt_model):
        signal, signal_len = make_audio()
        with torch.no_grad():
            de, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([3, 3])
            )
            ja, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([5, 5])
            )

        assert not torch.allclose(de, ja), "different languages must produce different encoder output"

    @pytest.mark.unit
    def test_prebuilt_prompt_matches_indices(self, prompt_model):
        """The `prompt` path must agree with the `prompt_indices` path it back-fills."""
        signal, signal_len = make_audio()
        with torch.no_grad():
            from_indices, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([3, 3])
            )
            time_steps = from_indices.shape[2]
            one_hot = torch.zeros(2, time_steps, NUM_PROMPTS)
            one_hot[:, :, 3] = 1.0
            from_prompt, _ = prompt_model.forward(input_signal=signal, input_signal_length=signal_len, prompt=one_hot)

        assert torch.allclose(from_indices, from_prompt)

    @pytest.mark.unit
    def test_prompt_longer_than_encoder_output_is_trimmed(self, prompt_model):
        signal, signal_len = make_audio()
        with torch.no_grad():
            reference, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([3, 3])
            )
            padded = torch.zeros(2, reference.shape[2] + 7, NUM_PROMPTS)
            padded[:, :, 3] = 1.0
            trimmed, _ = prompt_model.forward(input_signal=signal, input_signal_length=signal_len, prompt=padded)

        assert torch.allclose(reference, trimmed)

    @pytest.mark.unit
    def test_per_row_languages_are_independent(self, prompt_model):
        """Each row uses its own language, so a mixed batch matches the single-language runs."""
        signal, signal_len = make_audio()
        with torch.no_grad():
            mixed, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([3, 5])
            )
            de, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([3, 3])
            )
            ja, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, prompt_indices=torch.tensor([5, 5])
            )

        assert torch.allclose(mixed[0], de[0])
        assert torch.allclose(mixed[1], ja[1])

    @pytest.mark.unit
    def test_missing_prompt_raises_when_conditioning_is_enabled(self, prompt_model):
        signal, signal_len = make_audio()
        with pytest.raises(ValueError, match="prompt or prompt_indices"):
            prompt_model.forward(input_signal=signal, input_signal_length=signal_len)

    @pytest.mark.unit
    def test_unconditioned_model_needs_no_prompt(self, unconditioned_model):
        """A config that does not enable prompting leaves `forward` unconditioned."""
        assert unconditioned_model.concat is False

        signal, signal_len = make_audio()
        with torch.no_grad():
            encoded, encoded_len = unconditioned_model.forward(input_signal=signal, input_signal_length=signal_len)

        assert encoded.shape[0] == 2
        assert encoded_len.shape == (2,)


class TestDefaultPromptLanguage:
    """The fallback language is taken from the model, never hardcoded."""

    @pytest.mark.unit
    def test_checkpoint_can_declare_its_default(self, build_prompt_model):
        model = build_prompt_model(extra_defaults={'default_prompt_language': 'unk'})
        assert model.default_prompt_language == 'unk'

    @pytest.mark.unit
    def test_declared_default_must_be_a_dictionary_key(self, build_prompt_model):
        model = build_prompt_model(extra_defaults={'default_prompt_language': 'kl-KL'})
        with pytest.raises(ValueError, match="not a key of"):
            _ = model.default_prompt_language

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "dictionary,expected",
        [
            ({'auto': 1, 'unk': 2, 'en-US': 0}, 'auto'),
            ({'unk': 2, 'en-US': 0}, 'unk'),
            ({'en-US': 0, 'de-DE': 3}, 'en-US'),
            ({'de-DE': 3, 'ja-JP': 5}, None),
        ],
    )
    def test_language_agnostic_keys_are_preferred(self, build_prompt_model, dictionary, expected):
        model = build_prompt_model(prompt_dictionary=dictionary)
        assert model.default_prompt_language == expected


class TestResolvePromptLanguage:
    """Resolution accepts locale keys and bare codes, and never aborts a run on an unknown one."""

    @pytest.mark.unit
    def test_exact_dictionary_key(self, prompt_model):
        assert prompt_model.resolve_prompt_language('de-DE') == 3

    @pytest.mark.unit
    def test_bare_language_code(self, prompt_model):
        """A manifest carrying "de" must reach the "de-DE" prompt."""
        assert prompt_model.resolve_prompt_language('de') == 3

    @pytest.mark.unit
    def test_none_uses_the_model_default(self, prompt_model):
        assert prompt_model.resolve_prompt_language(None) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_unknown_language_falls_back_to_the_default(self, prompt_model):
        assert prompt_model.resolve_prompt_language('kl-KL') == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_unknown_language_raises_without_a_default(self, build_prompt_model):
        model = build_prompt_model(prompt_dictionary={'de-DE': 3, 'ja-JP': 5})
        with pytest.raises(ValueError, match="no default language"):
            model.resolve_prompt_language('kl-KL')

    @pytest.mark.unit
    def test_ambiguous_code_is_resolved_deterministically(self, build_prompt_model):
        model = build_prompt_model(prompt_dictionary={'en-US': 0, 'en-GB': 1, 'unk': 7})
        assert model.resolve_prompt_language('en') == 1  # "en-GB" sorts first

    @pytest.mark.unit
    def test_empty_dictionary_raises(self, build_prompt_model):
        model = build_prompt_model(prompt_dictionary={})
        with pytest.raises(ValueError, match="Prompt dictionary is empty"):
            model.resolve_prompt_language('de-DE')


class TestTranscribeForward:
    """Offline transcription conditions on `target_lang`, else on the dataloader's indices."""

    @staticmethod
    def _batch(prompt_indices):
        signal, signal_len = make_audio()
        transcript = torch.zeros(2, 4, dtype=torch.long)
        transcript_len = torch.full((2,), 4, dtype=torch.long)
        return signal, signal_len, transcript, transcript_len, prompt_indices

    @pytest.mark.unit
    def test_target_lang_selects_the_language(self, prompt_model):
        batch = self._batch(prompt_indices=None)
        with torch.no_grad():
            de = prompt_model._transcribe_forward(batch, RNNTPromptTranscribeConfig(target_lang='de-DE'))
            ja = prompt_model._transcribe_forward(batch, RNNTPromptTranscribeConfig(target_lang='ja-JP'))

        assert not torch.allclose(de['encoded'], ja['encoded'])

    @pytest.mark.unit
    def test_explicit_target_lang_overrides_dataloader_indices(self, prompt_model):
        """`target_lang` names the language for the whole run, so it wins over manifest indices."""
        from_manifest = self._batch(prompt_indices=torch.tensor([5, 5]))
        with torch.no_grad():
            overridden = prompt_model._transcribe_forward(
                from_manifest, RNNTPromptTranscribeConfig(target_lang='de-DE')
            )
            expected = prompt_model._transcribe_forward(
                self._batch(prompt_indices=torch.tensor([3, 3])), RNNTPromptTranscribeConfig(target_lang=None)
            )

        assert torch.allclose(overridden['encoded'], expected['encoded'])

    @pytest.mark.unit
    def test_mixed_dataloader_indices_are_honoured_per_row(self, prompt_model):
        """A manifest naming a different language per utterance conditions each row on its own."""
        with torch.no_grad():
            mixed = prompt_model._transcribe_forward(
                self._batch(prompt_indices=torch.tensor([3, 5])), RNNTPromptTranscribeConfig(target_lang=None)
            )
            de = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='de-DE')
            )
            ja = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='ja-JP')
            )

        assert torch.allclose(mixed['encoded'][0], de['encoded'][0])
        assert torch.allclose(mixed['encoded'][1], ja['encoded'][1])

    @pytest.mark.unit
    def test_dataloader_indices_are_used_when_no_target_lang(self, prompt_model):
        with torch.no_grad():
            from_indices = prompt_model._transcribe_forward(
                self._batch(prompt_indices=torch.tensor([3, 3])), RNNTPromptTranscribeConfig(target_lang=None)
            )
            explicit = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='de-DE')
            )

        assert torch.allclose(from_indices['encoded'], explicit['encoded'])

    @pytest.mark.unit
    def test_default_language_is_used_when_nothing_is_given(self, prompt_model):
        """Neither a manifest index nor `target_lang`: the model's own default applies."""
        with torch.no_grad():
            implicit = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang=None)
            )
            explicit = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='unk')
            )

        assert torch.allclose(implicit['encoded'], explicit['encoded'])


def make_cut(language, prompt_mode=None):
    """A stand-in cut carrying just the fields prompt resolution reads."""
    custom = None if prompt_mode is None else {'prompt_mode': prompt_mode}
    return SimpleNamespace(supervisions=[SimpleNamespace(language=language)], custom=custom)


def make_prompt_dataset(default_prompt_mode='unified', default_lang=None):
    """The prompt-index dataset, built without audio since only prompt resolution is exercised."""
    cfg = {
        'prompt_dictionary': PROMPT_DICTIONARY,
        'num_prompts': NUM_PROMPTS,
        'default_prompt_mode': default_prompt_mode,
        'default_lang': default_lang,
    }
    return LhotseSpeechToTextBpeDatasetWithPromptIndex(tokenizer=None, cfg=cfg)


class TestPromptIndexDatasetLanguage:
    """Per-utterance language resolution in the dataset feeding ``prompt_indices``."""

    @pytest.mark.unit
    def test_langid_mode_uses_the_cut_language(self):
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        assert dataset._get_prompt_index_for_cut(make_cut('de-DE')) == 3
        assert dataset._get_prompt_index_for_cut(make_cut('ja-JP')) == 5

    @pytest.mark.unit
    def test_langid_mode_is_deterministic(self):
        """Transcription relies on this: "unified" would substitute `auto` at random instead."""
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        cut = make_cut('de-DE')
        assert {dataset._get_prompt_index_for_cut(cut) for _ in range(50)} == {3}

    @pytest.mark.unit
    def test_missing_language_falls_back_to_default(self):
        """A plain list of audio files carries no language, and must still decode."""
        dataset = make_prompt_dataset(default_prompt_mode='langID', default_lang='unk')
        assert dataset._get_prompt_index_for_cut(make_cut(None)) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_unknown_language_falls_back_to_default(self):
        dataset = make_prompt_dataset(default_prompt_mode='langID', default_lang='unk')
        assert dataset._get_prompt_index_for_cut(make_cut('kl-KL')) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_unknown_language_still_raises_without_a_default(self):
        """Training configures no default, so data whose language cannot be honoured fails loudly."""
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        with pytest.raises(ValueError, match="Unknown prompt key"):
            dataset._get_prompt_index_for_cut(make_cut('kl-KL'))

    @pytest.mark.unit
    def test_declared_default_lang_must_be_known(self):
        with pytest.raises(ValueError, match="Unknown prompt key"):
            make_prompt_dataset(default_lang='kl-KL')

    @pytest.mark.unit
    def test_per_cut_prompt_mode_still_overrides(self):
        """The training tag remains authoritative, so existing data blends are unaffected."""
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        assert dataset._get_prompt_index_for_cut(make_cut('de-DE', prompt_mode='auto')) == dataset.auto_index

    @pytest.mark.unit
    def test_unified_mode_still_randomises(self):
        """The training default is untouched: it mixes the real language with `auto`."""
        dataset = make_prompt_dataset(default_prompt_mode='unified')
        drawn = {dataset._get_prompt_index_for_cut(make_cut('de-DE')) for _ in range(200)}
        assert drawn == {3, dataset.auto_index}


class TestTranscribeDataloaderRequestsPerUtteranceLanguage:
    """The transcribe dataloader must ask for deterministic per-utterance languages."""

    @pytest.mark.unit
    def test_dataloader_config(self, prompt_model, monkeypatch, tmp_path):
        captured = {}

        def capture(config):
            captured.update(config)
            return None

        monkeypatch.setattr(prompt_model, '_setup_dataloader_from_config', capture)
        manifest = tmp_path / 'manifest.json'
        manifest.write_text('')
        prompt_model._setup_transcribe_dataloader({'manifest_filepath': str(manifest), 'batch_size': 2})

        assert captured['initialize_prompt_feature'] is True, "per-utterance languages must be read"
        assert captured['default_prompt_mode'] == 'langID', "'unified' would randomise the language"
        assert captured['default_lang'] == prompt_model.default_prompt_language
        assert captured['prompt_dictionary'] == prompt_model.cfg.model_defaults.prompt_dictionary
        assert captured['lang_field'] == 'lang'

    @pytest.mark.unit
    def test_lang_field_is_forwarded(self, prompt_model, monkeypatch, tmp_path):
        """A manifest naming the language field something else must still condition per utterance."""
        captured = {}
        monkeypatch.setattr(prompt_model, '_setup_dataloader_from_config', lambda config: captured.update(config))
        manifest = tmp_path / 'manifest.json'
        manifest.write_text('')
        prompt_model._setup_transcribe_dataloader(
            {'manifest_filepath': str(manifest), 'batch_size': 2, 'lang_field': 'source_lang'}
        )

        assert captured['lang_field'] == 'source_lang'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "target_lang,expected",
        [
            ('de-DE', 'de-DE'),  # exact key
            ('de', 'de-DE'),  # bare code
            (None, 'unk'),  # model default
            ('kl-KL', 'unk'),  # unknown: must not stop the dataloader from being built
        ],
    )
    def test_default_lang_prefers_target_lang(self, prompt_model, target_lang, expected):
        """`default_lang` agrees with the language actually applied, so no misleading fallback log."""
        assert prompt_model._default_transcribe_language(target_lang) == expected


class TestPipelineDefaultLanguage:
    """``BasePipeline`` asks the model for its default instead of assuming "auto" or "en-US"."""

    @staticmethod
    def _resolve(model, prompt_dict):
        stub = SimpleNamespace(
            _prompt_config={'prompt_dict': prompt_dict},
            asr_model=SimpleNamespace(asr_model=model),
        )
        return BasePipeline._resolve_default_language_code(stub)

    @pytest.mark.unit
    def test_model_declared_default_wins(self, build_prompt_model):
        model = build_prompt_model(extra_defaults={'default_prompt_language': 'unk'})
        assert self._resolve(model, PROMPT_DICTIONARY) == 'unk'

    @pytest.mark.unit
    def test_falls_back_to_dictionary_candidates(self):
        """A model without the attribute keeps the pre-existing "auto" / "en-US" preference."""
        assert self._resolve(SimpleNamespace(), {'auto': 1, 'en-US': 0}) == 'auto'
        assert self._resolve(SimpleNamespace(), {'en-US': 0, 'de-DE': 3}) == 'en-US'

    @pytest.mark.unit
    def test_none_when_no_candidate_exists(self):
        """None signals the caller to demand an explicit language rather than guessing."""
        assert self._resolve(SimpleNamespace(), {'de-DE': 3}) is None

    @pytest.mark.unit
    def test_none_without_prompt_config(self):
        assert BasePipeline._resolve_default_language_code(SimpleNamespace(_prompt_config=None)) is None


class TestBaseClassesAreUntouched:
    """Prompt conditioning stays in the prompt class; stock models gain no prompt surface."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "attribute",
        ['concat', 'prompt_kernel', 'resolve_prompt_language', 'default_prompt_language', 'set_inference_prompt'],
    )
    def test_plain_rnnt_has_no_prompt_surface(self, attribute):
        assert not hasattr(EncDecRNNTModel, attribute)
        assert not hasattr(EncDecRNNTBPEModel, attribute)

    @pytest.mark.unit
    def test_plain_model_forward_signature_has_no_prompt(self):
        import inspect

        parameters = inspect.signature(EncDecRNNTModel.forward).parameters
        assert 'prompt' not in parameters
        assert 'prompt_indices' not in parameters
