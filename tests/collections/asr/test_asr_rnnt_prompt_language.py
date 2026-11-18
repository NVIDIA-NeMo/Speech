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

Covers the shared language resolver used by the model, the prompt-index dataset and the streaming
pipelines, and the ``prompt`` / ``prompt_indices`` forward arguments.
"""

import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex
from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecRNNTModel
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt, RNNTPromptTranscribeConfig
from nemo.collections.asr.parts.utils import prompt_language_utils
from nemo.collections.asr.parts.utils.prompt_language_utils import resolve_prompt_index, resolve_prompt_language
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__ as NUMBA_MINIMUM_VERSION
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE, numba_cpu_is_supported

ENC_HIDDEN = 64
NUM_PROMPTS = 8
PROMPT_DICTIONARY = {'en-US': 0, 'en-GB': 1, 'de-DE': 3, 'ja-JP': 5, 'auto': 6, 'fr': 7}

pytestmark = pytest.mark.skipif(
    not numba_cpu_is_supported(NUMBA_MINIMUM_VERSION),
    reason=f'RNNTLoss has not been compiled with appropriate numba version. {NUMBA_INSTALLATION_MESSAGE}',
)


def build_config(tokenizer_dir: str, prompt_enabled: bool, prompt_dictionary=None) -> DictConfig:
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

    def _build(prompt_enabled=True, prompt_dictionary=None):
        cfg = build_config(test_data_dir, prompt_enabled, prompt_dictionary)
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
    def test_forward_declares_new_and_deprecated_names(self, prompt_model):
        """`typecheck` rejects undeclared keywords, so the deprecated names must stay declared."""
        assert {'language_prompt_features', 'language_prompt_indices', 'prompt', 'prompt_indices'} <= set(
            prompt_model.input_types
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "deprecated_name, new_name",
        [('prompt_indices', 'language_prompt_indices'), ('prompt', 'language_prompt_features')],
    )
    def test_deprecated_name_matches_new_name_and_warns(self, prompt_model, monkeypatch, deprecated_name, new_name):
        warnings = []
        monkeypatch.setattr(
            prompt_language_utils.logging, 'warning', lambda message, *args, **kwargs: warnings.append(message)
        )
        signal, signal_len = make_audio()
        with torch.no_grad():
            reference, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 3])
            )
            value = torch.tensor([3, 3])
            if deprecated_name == 'prompt':
                value = torch.zeros(2, reference.shape[2], NUM_PROMPTS)
                value[:, :, 3] = 1.0
            new, _ = prompt_model.forward(input_signal=signal, input_signal_length=signal_len, **{new_name: value})
            assert not warnings
            deprecated, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, **{deprecated_name: value}
            )

        assert torch.allclose(new, reference)
        assert torch.allclose(deprecated, reference)
        assert len(warnings) == 1
        assert f"`{deprecated_name}`" in warnings[0] and f"`{new_name}`" in warnings[0]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "deprecated_name, new_name",
        [('prompt_indices', 'language_prompt_indices'), ('prompt', 'language_prompt_features')],
    )
    def test_deprecated_and_new_name_together_raise(self, prompt_model, deprecated_name, new_name):
        signal, signal_len = make_audio()
        value = torch.zeros(2, 4, NUM_PROMPTS) if deprecated_name == 'prompt' else torch.tensor([3, 3])
        with pytest.raises(ValueError, match=f"either `{new_name}` or the deprecated `{deprecated_name}`"):
            prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, **{new_name: value, deprecated_name: value}
            )

    @pytest.mark.unit
    def test_prompt_indices_condition_the_encoder(self, prompt_model):
        signal, signal_len = make_audio()
        with torch.no_grad():
            de, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 3])
            )
            ja, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([5, 5])
            )

        assert not torch.allclose(de, ja), "different languages must produce different encoder output"

    @pytest.mark.unit
    def test_prebuilt_prompt_matches_indices(self, prompt_model):
        """The `prompt` path must agree with the `prompt_indices` path."""
        signal, signal_len = make_audio()
        with torch.no_grad():
            from_indices, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 3])
            )
            time_steps = from_indices.shape[2]
            one_hot = torch.zeros(2, time_steps, NUM_PROMPTS)
            one_hot[:, :, 3] = 1.0
            from_prompt, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_features=one_hot
            )

        assert torch.allclose(from_indices, from_prompt)

    @pytest.mark.unit
    def test_prompt_longer_than_encoder_output_is_trimmed(self, prompt_model):
        signal, signal_len = make_audio()
        with torch.no_grad():
            reference, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 3])
            )
            padded = torch.zeros(2, reference.shape[2] + 7, NUM_PROMPTS)
            padded[:, :, 3] = 1.0
            trimmed, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_features=padded
            )

        assert torch.allclose(reference, trimmed)

    @pytest.mark.unit
    def test_per_row_languages_are_independent(self, prompt_model):
        """Each row uses its own language, so a mixed batch matches the single-language runs."""
        signal, signal_len = make_audio()
        with torch.no_grad():
            mixed, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 5])
            )
            de, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([3, 3])
            )
            ja, _ = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, language_prompt_indices=torch.tensor([5, 5])
            )

        assert torch.allclose(mixed[0], de[0])
        assert torch.allclose(mixed[1], ja[1])

    @pytest.mark.unit
    def test_missing_prompt_raises_when_conditioning_is_enabled(self, prompt_model):
        signal, signal_len = make_audio()
        with pytest.raises(ValueError, match="Either language_prompt_features or language_prompt_indices"):
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


class TestSharedPromptLanguageResolver:
    """The one set of rules used by the models, the prompt-index dataset and the streaming pipelines."""

    @pytest.mark.unit
    def test_exact_dictionary_key(self):
        assert resolve_prompt_language('de-DE', PROMPT_DICTIONARY) == 'de-DE'
        assert resolve_prompt_language('fr', PROMPT_DICTIONARY) == 'fr'

    @pytest.mark.unit
    def test_bare_code_matching_one_locale(self):
        """A manifest carrying "de" must reach the "de-DE" prompt."""
        assert resolve_prompt_index('de', PROMPT_DICTIONARY) == 3
        assert resolve_prompt_index('DE', PROMPT_DICTIONARY) == 3

    @pytest.mark.unit
    def test_none_selects_auto(self):
        assert resolve_prompt_language(None, PROMPT_DICTIONARY) == 'auto'

    @pytest.mark.unit
    def test_none_raises_without_auto(self):
        with pytest.raises(ValueError, match="no 'auto' prompt"):
            resolve_prompt_language(None, {'de-DE': 3})

    @pytest.mark.unit
    def test_ambiguous_code_raises(self):
        """ "en" could mean "en-US" or "en-GB"; picking one silently would be a guess."""
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_prompt_language('en', PROMPT_DICTIONARY)

    @pytest.mark.unit
    def test_unsupported_language_raises_and_suggests_auto(self):
        with pytest.raises(ValueError, match="not supported by this model") as error:
            resolve_prompt_language('kl-KL', PROMPT_DICTIONARY)

        message = str(error.value)
        assert "de-DE" in message, "the supported languages must be listed"
        assert "'auto'" in message and "not guaranteed" in message

    @pytest.mark.unit
    def test_unsupported_language_does_not_suggest_a_missing_auto(self):
        with pytest.raises(ValueError, match="not supported by this model") as error:
            resolve_prompt_language('kl-KL', {'de-DE': 3})

        assert "'auto'" not in str(error.value)

    @pytest.mark.unit
    def test_empty_dictionary_raises(self):
        with pytest.raises(ValueError, match="prompt dictionary is empty"):
            resolve_prompt_language('de-DE', {})


class TestModelPromptLanguage:
    """The model exposes the shared resolver, with no separate fallback language."""

    @pytest.mark.unit
    def test_resolves_with_the_shared_rules(self, prompt_model):
        assert prompt_model.resolve_prompt_language('de-DE') == 3
        assert prompt_model.resolve_prompt_language('de') == 3
        assert prompt_model.resolve_prompt_language(None) == PROMPT_DICTIONARY['auto']

    @pytest.mark.unit
    def test_unsupported_language_raises(self, prompt_model):
        with pytest.raises(ValueError, match="not supported by this model"):
            prompt_model.resolve_prompt_language('kl-KL')

    @pytest.mark.unit
    def test_no_separate_default_language(self, prompt_model):
        """`auto` is the only language-agnostic prompt, so there is nothing else to configure."""
        assert not hasattr(prompt_model, 'default_prompt_language')

    @pytest.mark.unit
    def test_set_inference_prompt_uses_the_shared_rules(self, prompt_model):
        prompt_model.set_inference_prompt('de')
        assert prompt_model._inference_prompt_index == 3

        with pytest.raises(ValueError, match="not supported by this model"):
            prompt_model.set_inference_prompt('kl-KL')


class TestTranscribeForward:
    """Offline transcription conditions on the dataloader's indices, else on `target_lang`."""

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
    def test_dataloader_indices_are_used(self, prompt_model):
        """The dataloader already applies `target_lang` or each utterance's own language."""
        with torch.no_grad():
            from_indices = prompt_model._transcribe_forward(
                self._batch(prompt_indices=torch.tensor([3, 3])), RNNTPromptTranscribeConfig(target_lang='ja-JP')
            )
            explicit = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='de-DE')
            )

        assert torch.allclose(from_indices['encoded'], explicit['encoded'])

    @pytest.mark.unit
    def test_mixed_dataloader_indices_are_honoured_per_row(self, prompt_model):
        """A manifest naming a different language per utterance conditions each row on its own."""
        with torch.no_grad():
            mixed = prompt_model._transcribe_forward(
                self._batch(prompt_indices=torch.tensor([3, 5])), RNNTPromptTranscribeConfig()
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
    def test_auto_is_used_when_nothing_is_given(self, prompt_model):
        with torch.no_grad():
            implicit = prompt_model._transcribe_forward(self._batch(prompt_indices=None), RNNTPromptTranscribeConfig())
            explicit = prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='auto')
            )

        assert torch.allclose(implicit['encoded'], explicit['encoded'])

    @pytest.mark.unit
    def test_unsupported_target_lang_raises(self, prompt_model):
        with pytest.raises(ValueError, match="not supported by this model"):
            prompt_model._transcribe_forward(
                self._batch(prompt_indices=None), RNNTPromptTranscribeConfig(target_lang='kl-KL')
            )

    @pytest.mark.unit
    def test_default_config_reads_the_manifest(self):
        """Offline and streaming share the default: each utterance's own manifest language."""
        assert RNNTPromptTranscribeConfig().target_lang is None
        assert RNNTPromptTranscribeConfig().lang_field == 'lang'


def make_cut(language, prompt_mode=None):
    """A stand-in cut carrying just the fields prompt resolution reads."""
    custom = None if prompt_mode is None else {'prompt_mode': prompt_mode}
    return SimpleNamespace(id='cut-0', supervisions=[SimpleNamespace(language=language)], custom=custom)


def make_prompt_dataset(default_prompt_mode='unified', default_lang=None, target_lang=None):
    """The prompt-index dataset, built without audio since only prompt resolution is exercised."""
    cfg = {
        'prompt_dictionary': PROMPT_DICTIONARY,
        'num_prompts': NUM_PROMPTS,
        'default_prompt_mode': default_prompt_mode,
        'default_lang': default_lang,
        'target_lang': target_lang,
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
    def test_bare_code_uses_the_shared_rules(self):
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        assert dataset._get_prompt_index_for_cut(make_cut('de')) == 3

    @pytest.mark.unit
    def test_langid_mode_is_deterministic(self):
        """Transcription relies on this: "unified" would substitute `auto` at random instead."""
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        cut = make_cut('de-DE')
        assert {dataset._get_prompt_index_for_cut(cut) for _ in range(50)} == {3}

    @pytest.mark.unit
    def test_missing_language_uses_default_lang(self):
        """A plain list of audio files carries no language, and must still decode."""
        dataset = make_prompt_dataset(default_prompt_mode='langID', default_lang='auto')
        assert dataset._get_prompt_index_for_cut(make_cut(None)) == PROMPT_DICTIONARY['auto']

    @pytest.mark.unit
    def test_missing_language_raises_without_default_lang(self):
        """Training configures no default, so a cut without a language fails loudly."""
        dataset = make_prompt_dataset(default_prompt_mode='langID')
        with pytest.raises(ValueError, match="has no language"):
            dataset._get_prompt_index_for_cut(make_cut(None))

    @pytest.mark.unit
    def test_unsupported_language_raises_even_with_default_lang(self):
        """`default_lang` covers a missing language only; it never replaces an unsupported one."""
        dataset = make_prompt_dataset(default_prompt_mode='langID', default_lang='auto')
        with pytest.raises(ValueError, match="not supported by this model"):
            dataset._get_prompt_index_for_cut(make_cut('kl-KL'))

    @pytest.mark.unit
    def test_target_lang_replaces_every_cut_language(self):
        """A run-wide `target_lang` applies even to cuts whose own language is unsupported."""
        dataset = make_prompt_dataset(default_prompt_mode='langID', target_lang='de-DE')
        assert dataset._get_prompt_index_for_cut(make_cut('ja-JP')) == 3
        assert dataset._get_prompt_index_for_cut(make_cut('kl-KL')) == 3
        assert dataset._get_prompt_index_for_cut(make_cut(None)) == 3

    @pytest.mark.unit
    def test_unsupported_target_lang_raises(self):
        dataset = make_prompt_dataset(default_prompt_mode='langID', target_lang='kl-KL')
        with pytest.raises(ValueError, match="not supported by this model"):
            dataset._get_prompt_index_for_cut(make_cut('de-DE'))

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


class TestTranscribeDataloaderLanguage:
    """The transcribe dataloader applies `target_lang`, else each utterance's own language."""

    @staticmethod
    def _capture_dataloader_config(model, monkeypatch, tmp_path, **config):
        captured = {}
        monkeypatch.setattr(model, '_setup_dataloader_from_config', lambda config: captured.update(config))
        manifest = tmp_path / 'manifest.json'
        manifest.write_text('')
        model._setup_transcribe_dataloader({'manifest_filepath': str(manifest), 'batch_size': 2, **config})
        return captured

    @pytest.mark.unit
    def test_dataloader_config(self, prompt_model, monkeypatch, tmp_path):
        captured = self._capture_dataloader_config(prompt_model, monkeypatch, tmp_path)

        assert captured['initialize_prompt_feature'] is True, "per-utterance languages must be read"
        assert captured['default_prompt_mode'] == 'langID', "'unified' would randomise the language"
        assert captured['default_lang'] == 'auto'
        assert captured['target_lang'] is None
        assert captured['prompt_dictionary'] == prompt_model.cfg.model_defaults.prompt_dictionary
        assert captured['lang_field'] == 'lang'

    @pytest.mark.unit
    def test_target_lang_and_lang_field_are_forwarded(self, prompt_model, monkeypatch, tmp_path):
        captured = self._capture_dataloader_config(
            prompt_model, monkeypatch, tmp_path, target_lang='de', lang_field='source_lang'
        )

        assert captured['target_lang'] == 'de'
        assert captured['lang_field'] == 'source_lang'

    @pytest.mark.unit
    def test_manifest_processing_carries_target_lang(self, prompt_model, tmp_path):
        trcfg = RNNTPromptTranscribeConfig(target_lang='de')
        ds_config = prompt_model._transcribe_input_manifest_processing(['a.wav'], str(tmp_path), trcfg)

        assert ds_config['target_lang'] == 'de'


class TestPipelinePromptLanguage:
    """``BasePipeline`` resolves languages with the same rules as the model."""

    @staticmethod
    def _resolve(language_code, prompt_dict=PROMPT_DICTIONARY):
        stub = SimpleNamespace(_prompt_config={'prompt_dict': prompt_dict})
        return BasePipeline._resolve_prompt_index(stub, language_code)

    @pytest.mark.unit
    def test_exact_and_bare_codes(self):
        assert self._resolve('de-DE') == 3
        assert self._resolve('de') == 3

    @pytest.mark.unit
    @pytest.mark.parametrize("language_code", [None, ''])
    def test_no_language_selects_auto(self, language_code):
        assert self._resolve(language_code) == PROMPT_DICTIONARY['auto']

    @pytest.mark.unit
    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="not supported by this model"):
            self._resolve('kl-KL')

    @pytest.mark.unit
    def test_missing_prompt_config_raises(self):
        with pytest.raises(RuntimeError, match="Prompt configuration is missing"):
            BasePipeline._resolve_prompt_index(SimpleNamespace(_prompt_config=None), 'de-DE')


class TestEncodeWithPrompts:
    """The buffered wrapper hands the model indices, never a prompt sized by its own length math."""

    @staticmethod
    def _call(processed_signal, captured):
        def model(**kwargs):
            captured.update(kwargs)
            # Subsampling rounds each of its three stages up, so the encoder emits ceil(T / 8).
            time_steps = -(-kwargs['processed_signal'].shape[2] // 8)
            prompt = kwargs.get('language_prompt_features')
            if prompt is not None and prompt.shape[1] < time_steps:
                # The model only trims a prompt that is too long; a short one fails to concatenate.
                raise RuntimeError(f"prompt has {prompt.shape[1]} frames, encoder emitted {time_steps}")
            return torch.zeros(1, 4, time_steps), torch.tensor([time_steps])

        wrapper = SimpleNamespace(
            device=torch.device('cpu'),
            device_str='cpu',
            compute_dtype=torch.float32,
            cast_dtype=torch.float32,
            use_amp=False,
            asr_model=model,
        )
        wrapper.encode = partial(RNNTInferenceWrapper.encode, wrapper)
        return RNNTInferenceWrapper.encode_with_prompts(
            wrapper,
            processed_signal=processed_signal,
            processed_signal_length=torch.tensor([processed_signal.shape[2]]),
            language_prompt_indices=torch.tensor([3]),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("feature_frames", [768, 700])
    def test_indices_reach_the_model(self, feature_frames):
        """700 frames is the regression: floor(700/8) is 87 while the encoder emits 88."""
        captured = {}
        self._call(torch.zeros(1, 4, feature_frames), captured)

        assert torch.equal(captured['language_prompt_indices'], torch.tensor([3]))
        assert not {'language_prompt_features', 'prompt', 'prompt_indices'} & set(captured)


class TestBaseClassesAreUntouched:
    """Prompt conditioning stays in the prompt class; stock models gain no prompt surface."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "attribute",
        ['concat', 'prompt_kernel', 'resolve_prompt_language', 'set_inference_prompt'],
    )
    def test_plain_rnnt_has_no_prompt_surface(self, attribute):
        assert not hasattr(EncDecRNNTModel, attribute)
        assert not hasattr(EncDecRNNTBPEModel, attribute)

    @pytest.mark.unit
    def test_plain_model_forward_signature_has_no_prompt(self):
        import inspect

        parameters = inspect.signature(EncDecRNNTModel.forward).parameters
        assert not {'language_prompt_features', 'language_prompt_indices', 'prompt', 'prompt_indices'} & set(
            parameters
        )
