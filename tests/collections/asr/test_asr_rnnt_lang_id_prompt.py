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

import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecRNNTModel, EncDecUnifiedRNNTBPEModel
from nemo.collections.asr.models.unified_rnnt_bpe_models import UnifiedTranscribeConfig
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__ as NUMBA_MINIMUM_VERSION
from nemo.core.utils.numba_utils import NUMBA_INSTALLATION_MESSAGE, numba_cpu_is_supported

NUMBA_RNNT_LOSS_AVAILABLE = numba_cpu_is_supported(NUMBA_MINIMUM_VERSION)

ENC_HIDDEN = 64
NUM_PROMPTS = 8
PROMPT_DICTIONARY = {'en-US': 0, 'en': 0, 'de-DE': 3, 'ja-JP': 5, 'unk': 7}

requires_numba = pytest.mark.skipif(
    not NUMBA_RNNT_LOSS_AVAILABLE,
    reason=f'RNNTLoss has not been compiled with appropriate numba version. {NUMBA_INSTALLATION_MESSAGE}',
)


def build_config(tokenizer_dir: str, prompt_enabled: bool, prompt_dictionary=None, extra_defaults=None) -> DictConfig:
    """Build a tiny unified RNNT BPE config, optionally with language conditioning enabled."""
    model_defaults = {'enc_hidden': ENC_HIDDEN, 'pred_hidden': 32}
    if prompt_enabled:
        model_defaults.update(
            {
                'initialize_lang_id_prompt': True,
                'num_lang_id_prompts': NUM_PROMPTS,
                'language_dictionary': PROMPT_DICTIONARY if prompt_dictionary is None else prompt_dictionary,
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
                    'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
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
def build_unified(test_data_dir):
    """Factory for tiny :class:`EncDecUnifiedRNNTBPEModel` instances."""

    def _build(prompt_enabled=True, prompt_dictionary=None, extra_defaults=None, trainer=None):
        cfg = build_config(test_data_dir, prompt_enabled, prompt_dictionary, extra_defaults)
        return EncDecUnifiedRNNTBPEModel(cfg=cfg, trainer=trainer)

    return _build


@pytest.fixture()
def unified_model(build_unified):
    """A unified model with language conditioning enabled."""
    return build_unified(prompt_enabled=True)


@pytest.fixture()
def unconditioned_model(build_unified):
    """The unified class instantiated from a config that does not enable conditioning."""
    return build_unified(prompt_enabled=False)


@pytest.fixture()
def plain_model():
    """A stock RNN-T that knows nothing about languages."""
    cfg = DictConfig(
        {
            'labels': ListConfig([' ', 'a', 'b', 'c']),
            'preprocessor': DictConfig(
                {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': {}}
            ),
            'model_defaults': DictConfig({'enc_hidden': ENC_HIDDEN, 'pred_hidden': 32}),
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
            'decoding': DictConfig({'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}),
            'loss': DictConfig({'loss_name': 'default'}),
        }
    )
    return EncDecRNNTModel(cfg=cfg)


@requires_numba
class TestBaseModelIsUntouched:
    """Language conditioning lives on the dedicated class, so nothing leaks into the base RNN-T.

    An earlier revision put the mixin on the shared base, which made every capability probe answer
    true for every ASR model and broke feature detection in the cache-aware streaming script.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        'attribute',
        [
            'use_lang_id_prompt',
            'language_conditioning_enabled',
            'language_dictionary',
            'language_to_id',
            'set_inference_language',
            'apply_lang_id_prompt',
        ],
    )
    def test_no_language_surface_on_the_base_classes(self, plain_model, attribute):
        from nemo.collections.asr.models.asr_model import ASRModel

        assert not hasattr(plain_model, attribute)
        assert not hasattr(EncDecRNNTModel, attribute)
        assert not hasattr(ASRModel, attribute)

    @pytest.mark.unit
    def test_base_forward_and_input_types_are_unchanged(self, plain_model):
        assert 'lang_id_prompt' not in plain_model.input_types
        assert 'lang_id_prompt' not in EncDecRNNTModel.forward.__wrapped__.__code__.co_varnames

    @pytest.mark.unit
    def test_source_lang_stays_off_the_shared_transcribe_config(self):
        """``source_lang`` belongs to the unified config, not to every model's transcription API."""
        from nemo.collections.asr.parts.mixins import TranscribeConfig

        assert not hasattr(TranscribeConfig(), 'source_lang')
        assert UnifiedTranscribeConfig().source_lang is None
        assert isinstance(EncDecUnifiedRNNTBPEModel.get_transcribe_config(), UnifiedTranscribeConfig)

    @pytest.mark.unit
    def test_call_sites_must_use_a_guarded_read(self, plain_model, unified_model):
        """The capability check every pipeline uses must work for both kinds of model."""
        assert getattr(plain_model, 'language_conditioning_enabled', False) is False
        assert getattr(unified_model, 'language_conditioning_enabled', False) is True


@requires_numba
class TestLangIdPromptSetup:
    @pytest.mark.unit
    def test_disabled_without_the_config_flag(self, unconditioned_model):
        assert unconditioned_model.use_lang_id_prompt is False
        assert unconditioned_model.language_conditioning_enabled is False
        assert unconditioned_model.num_lang_id_prompts is None
        assert unconditioned_model.language_dictionary is None
        assert not hasattr(unconditioned_model, 'lang_id_prompt_kernel')
        assert 'lang_id_prompt' not in unconditioned_model.input_types

    @pytest.mark.unit
    def test_enabled_from_config(self, unified_model):
        assert unified_model.use_lang_id_prompt is True
        assert unified_model.language_conditioning_enabled is True
        assert unified_model.num_lang_id_prompts == NUM_PROMPTS
        assert unified_model.language_dictionary == PROMPT_DICTIONARY
        assert unified_model.lang_id_prompt_kernel[0].in_features == NUM_PROMPTS + ENC_HIDDEN
        assert unified_model.lang_id_prompt_kernel[-1].out_features == ENC_HIDDEN
        assert 'lang_id_prompt' in unified_model.input_types

    @pytest.mark.unit
    def test_prompt_kernel_weights_are_in_state_dict(self, unified_model):
        keys = [key for key in unified_model.state_dict() if key.startswith('lang_id_prompt_kernel.')]
        assert keys, "lang_id_prompt_kernel must be a registered submodule so checkpoints round-trip"

    @pytest.mark.unit
    def test_missing_language_dictionary_raises(self, build_unified):
        with pytest.raises(ValueError, match='language_dictionary'):
            build_unified(prompt_enabled=True, prompt_dictionary={})

    @pytest.mark.unit
    @pytest.mark.parametrize('other_scheme_key', ['initialize_prompt_feature', 'num_prompts', 'prompt_dictionary'])
    def test_prompt_streaming_config_keys_are_left_alone(self, build_unified, other_scheme_key):
        """``PromptStreamingMixin``'s keys are a live namespace, not a legacy one.

        They appear in the shipped ``fastconformer_*_prompt.yaml`` configs, so a model instantiated
        from such a config through a class without that mixin must still load rather than fail on
        keys it simply does not use.
        """
        value = PROMPT_DICTIONARY if other_scheme_key == 'prompt_dictionary' else True

        model = build_unified(prompt_enabled=False, extra_defaults={other_scheme_key: value})

        assert model.language_conditioning_enabled is False


@requires_numba
class TestCheckpointRoundTrip:
    """The class is named in the checkpoint, so restoring must reproduce the conditioning exactly."""

    @pytest.mark.unit
    def test_conditioning_survives_save_and_restore(self, unified_model, tmp_path):
        path = str(tmp_path / 'unified.nemo')
        unified_model.save_to(path)

        restored = EncDecUnifiedRNNTBPEModel.restore_from(path, map_location='cpu')

        assert restored.language_conditioning_enabled is True
        assert restored.language_dictionary == PROMPT_DICTIONARY
        assert restored.default_language == 'unk'
        for original, loaded in zip(
            unified_model.lang_id_prompt_kernel.parameters(), restored.lang_id_prompt_kernel.parameters()
        ):
            assert torch.equal(original, loaded)

    @pytest.mark.unit
    def test_restoring_an_unconditioned_checkpoint_stays_off(self, unconditioned_model, tmp_path):
        """Restoring a checkpoint that does not ask for conditioning must not invent any.

        This is the subclass-substitution case: a config without the prompt keys routed through the
        unified class, e.g. by fine-tuning from a plain checkpoint through this target.
        """
        path = str(tmp_path / 'unconditioned.nemo')
        unconditioned_model.save_to(path)

        restored = EncDecUnifiedRNNTBPEModel.restore_from(path, map_location='cpu')

        assert restored.language_conditioning_enabled is False
        assert not hasattr(restored, 'lang_id_prompt_kernel')


@requires_numba
class TestInferenceOnly:
    """The projection is trainable but no training step conditions on it, so training must refuse.

    Left unguarded, DDP fails on an unused parameter and a manual loop trains (and scores WER)
    without the language conditioning the model was built for.
    """

    @pytest.mark.unit
    def test_enabling_conditioning_with_a_trainer_attached_raises(self, build_unified):
        trainer = pytest.importorskip('lightning.pytorch').Trainer(accelerator='cpu', devices=1, logger=False)

        with pytest.raises(ValueError, match='inference-only'):
            build_unified(prompt_enabled=True, trainer=trainer)

    @pytest.mark.unit
    def test_models_without_conditioning_are_unaffected(self, build_unified):
        trainer = pytest.importorskip('lightning.pytorch').Trainer(accelerator='cpu', devices=1, logger=False)

        model = build_unified(prompt_enabled=False, trainer=trainer)

        assert model.language_conditioning_enabled is False


@requires_numba
class TestLanguageToId:
    """Strict resolution, mirroring the API of the encoder-conditioned unified model."""

    @pytest.mark.unit
    def test_known_language(self, unified_model):
        assert unified_model.language_to_id('de-DE') == 3
        assert unified_model.language_to_id('en') == 0

    @pytest.mark.unit
    def test_index_passes_through(self, unified_model):
        assert unified_model.language_to_id(5) == 5
        assert unified_model.language_to_id(torch.tensor(5)) == 5

    @pytest.mark.unit
    def test_unknown_language_raises(self, unified_model):
        with pytest.raises(ValueError, match="Unknown language 'kl-KL'"):
            unified_model.language_to_id('kl-KL')

    @pytest.mark.unit
    def test_raises_for_an_unconditioned_model(self, unconditioned_model):
        with pytest.raises(ValueError, match='not trained with language conditioning'):
            unconditioned_model.language_to_id('en-US')


@requires_numba
class TestResolveLanguage:
    """Lenient resolution: the projection always runs, so there is always an index to return."""

    @pytest.mark.unit
    def test_known_language(self, unified_model):
        assert unified_model.resolve_language('de-DE') == 3
        assert unified_model.resolve_language('en') == 0

    @pytest.mark.unit
    @pytest.mark.parametrize('requested', [None, 'kl-KL'])
    def test_falls_back_to_the_default_language(self, unified_model, requested):
        assert unified_model.resolve_language(requested) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_raises_when_no_candidate_default_exists(self, build_unified):
        model = build_unified(prompt_enabled=True, prompt_dictionary={'fr-FR': 0})
        with pytest.raises(ValueError, match='Cannot pick a default'):
            model.resolve_language(None)

    @pytest.mark.unit
    def test_raises_for_an_unconditioned_model(self, unconditioned_model):
        with pytest.raises(ValueError, match='not trained with language conditioning'):
            unconditioned_model.resolve_language('en-US')


@requires_numba
class TestSetInferenceLanguage:
    """A sticky language for inference modes with no call to thread one through."""

    @pytest.mark.unit
    def test_sets_the_language_used_when_none_is_requested(self, unified_model):
        assert unified_model.resolve_language(None) == PROMPT_DICTIONARY['unk']

        unified_model.set_inference_language('de-DE')

        assert unified_model.resolve_language(None) == 3

    @pytest.mark.unit
    def test_an_explicit_language_still_wins(self, unified_model):
        unified_model.set_inference_language('de-DE')

        assert unified_model.resolve_language('ja-JP') == 5

    @pytest.mark.unit
    def test_none_restores_the_default(self, unified_model):
        unified_model.set_inference_language('de-DE')
        unified_model.set_inference_language(None)

        assert unified_model.resolve_language(None) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_unknown_language_raises(self, unified_model):
        with pytest.raises(ValueError, match="Unknown language 'kl-KL'"):
            unified_model.set_inference_language('kl-KL')


@requires_numba
class TestDefaultLanguage:
    """The model advertises its own language-agnostic entry, so no call site hardcodes one."""

    @pytest.mark.unit
    def test_prefers_the_most_language_agnostic_key(self, unified_model):
        assert unified_model.default_language == 'unk'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        'dictionary, expected',
        [
            ({'en-US': 0, 'auto': 1, 'unk': 2}, 'unk'),
            ({'en-US': 0, 'auto': 1}, 'auto'),
            ({'en-US': 0, 'de-DE': 1}, 'en-US'),
            ({'de-DE': 0}, None),
        ],
    )
    def test_falls_through_the_preference_order(self, build_unified, dictionary, expected):
        assert build_unified(prompt_enabled=True, prompt_dictionary=dictionary).default_language == expected

    @pytest.mark.unit
    def test_is_none_without_conditioning(self, unconditioned_model):
        assert unconditioned_model.default_language is None


@requires_numba
class TestApplyLangIdPrompt:
    @pytest.mark.unit
    def test_onehot_prompt_shape_and_content(self, unified_model):
        prompt = unified_model.create_lang_id_prompt(3, 5, dtype=torch.float32, device=torch.device('cpu'))
        assert prompt.shape == (3, NUM_PROMPTS)
        assert torch.equal(prompt.sum(dim=-1), torch.ones(3))
        assert torch.all(prompt[:, 5] == 1.0)

    @pytest.mark.unit
    def test_batched_onehot_prompts_one_language_per_row(self, unified_model):
        """Per-sample prompting (e.g. per-utterance manifest languages) selects a language per row."""
        prompt = unified_model.create_lang_id_prompts(
            torch.tensor([0, 5, 3]), dtype=torch.float32, device=torch.device('cpu')
        )
        assert prompt.shape == (3, NUM_PROMPTS)
        assert torch.equal(prompt.sum(dim=-1), torch.ones(3))
        assert prompt[0, 0] == 1.0 and prompt[1, 5] == 1.0 and prompt[2, 3] == 1.0
        # A single-index batch must match the scalar builder.
        assert torch.equal(
            unified_model.create_lang_id_prompts([5, 5], dtype=torch.float32, device=torch.device('cpu')),
            unified_model.create_lang_id_prompt(2, 5, dtype=torch.float32, device=torch.device('cpu')),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('time_steps', [1, 7, 13, 91])
    def test_broadcasts_over_any_encoder_length(self, unified_model, time_steps):
        """A (B, num_lang_id_prompts) prompt must fit any encoder length.

        Regression: callers used to precompute the time dimension from the feature length, which is
        off by one whenever that length is not a multiple of the subsampling factor. The shortfall was
        silently zero-padded, conditioning the tail of every chunk on an all-zero prompt.
        """
        encoded = torch.randn(2, ENC_HIDDEN, time_steps)
        prompt = unified_model.create_lang_id_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)

        out = unified_model.apply_lang_id_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, time_steps)

    @pytest.mark.unit
    def test_per_frame_prompt_with_matching_length_is_accepted(self, unified_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, 9, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        out = unified_model.apply_lang_id_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, 9)
        # Broadcasting a 2D prompt must be equivalent to an explicit per-frame prompt.
        broadcast = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)
        )
        assert torch.allclose(out, broadcast)

    @pytest.mark.unit
    def test_per_row_prompts_are_independent(self, unified_model):
        """Batched streaming conditions each stream on its own language."""
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        mixed = torch.zeros(2, NUM_PROMPTS)
        mixed[0, 0] = 1.0
        mixed[1, 5] = 1.0

        out = unified_model.apply_lang_id_prompt(encoded, mixed)
        en_only = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 0, torch.float32, encoded.device)
        )
        ja_only = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 5, torch.float32, encoded.device)
        )

        assert torch.allclose(out[0], en_only[0])
        assert torch.allclose(out[1], ja_only[1])

    @pytest.mark.unit
    @pytest.mark.parametrize('prompt_time_steps', [8, 10])
    def test_per_frame_prompt_length_mismatch_raises(self, unified_model, prompt_time_steps):
        """Silently padding/truncating a per-frame prompt is what produced the tail-frame bug."""
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, prompt_time_steps, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        with pytest.raises(ValueError, match='time steps'):
            unified_model.apply_lang_id_prompt(encoded, prompt)

    @pytest.mark.unit
    def test_wrong_num_prompts_raises(self, unified_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='classes'):
            unified_model.apply_lang_id_prompt(encoded, torch.zeros(2, NUM_PROMPTS + 1))

    @pytest.mark.unit
    def test_different_languages_give_different_output(self, unified_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_en = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 0, torch.float32, encoded.device)
        )
        out_ja = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 5, torch.float32, encoded.device)
        )
        assert not torch.allclose(out_en, out_ja)

    @pytest.mark.unit
    def test_transcribe_helper_is_noop_without_conditioning(self, unconditioned_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        assert unconditioned_model.apply_lang_id_prompt_for_transcribe(encoded, 'en-US') is encoded

    @pytest.mark.unit
    def test_transcribe_helper_conditions_on_the_requested_language(self, unified_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_de = unified_model.apply_lang_id_prompt_for_transcribe(encoded, 'de-DE')
        out_ja = unified_model.apply_lang_id_prompt_for_transcribe(encoded, 'ja-JP')
        assert out_de.shape == encoded.shape
        assert not torch.allclose(out_de, out_ja)

    @pytest.mark.unit
    def test_prompt_rejected_by_an_unconditioned_model(self, unconditioned_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='not trained with language conditioning'):
            unconditioned_model.apply_lang_id_prompt(encoded, torch.zeros(2, NUM_PROMPTS))


@requires_numba
class TestCacheAwareSurfaceIsNotClaimed:
    """The unified model must not answer the cache-aware feature probes.

    ``speech_to_text_cache_aware_streaming_infer.py`` feature-detects prompt support by looking for
    ``set_inference_prompt``. This checkpoint was not trained in streaming conditions, so claiming
    that surface would offer a decoding mode the weights do not support. Cache-aware streaming
    belongs to ``PromptStreamingMixin``, whose models are trained for it.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize('attribute', ['set_inference_prompt', 'concat'])
    def test_probes_stay_false_for_a_plain_model(self, plain_model, attribute):
        assert not hasattr(plain_model, attribute)

    @pytest.mark.unit
    @pytest.mark.parametrize('attribute', ['set_inference_prompt', 'concat'])
    def test_probes_stay_false_for_a_unified_model(self, unified_model, attribute):
        assert not hasattr(unified_model, attribute)

    @pytest.mark.unit
    def test_the_stream_step_hook_is_left_as_the_base_no_op(self, unified_model):
        """``ASRModuleMixin``'s no-op must remain the implementation the MRO finds."""
        from nemo.collections.asr.parts.mixins.mixins import ASRModuleMixin

        owner = next(c for c in type(unified_model).__mro__ if '_apply_prompt_to_encoded' in c.__dict__)
        assert owner is ASRModuleMixin

    @pytest.mark.unit
    def test_prompt_streaming_models_keep_their_own_hook(self):
        """``PromptStreamingMixin`` stays the source of truth for the prompt-aware training models."""
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
        from nemo.collections.asr.parts.mixins.mixins import PromptStreamingMixin

        owner = next(c for c in EncDecRNNTBPEModelWithPrompt.__mro__ if '_apply_prompt_to_encoded' in c.__dict__)
        assert owner is PromptStreamingMixin


@requires_numba
class TestPromptPrecision:
    """The prompt projection must survive a model-wide cast to a low-precision dtype.

    The one-hot prompt is a unit-magnitude feature next to much larger encoder activations, so in
    bfloat16 its contribution falls below the mantissa and decoding silently ignores the requested
    language.
    """

    @pytest.mark.unit
    def test_kernel_stays_float32_after_model_cast(self, unified_model):
        unified_model.to(torch.bfloat16)

        assert all(param.dtype is torch.float32 for param in unified_model.lang_id_prompt_kernel.parameters())
        # ... while the rest of the model did get cast.
        assert unified_model.encoder.encoder[0].mconv[0].conv.weight.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_output_keeps_encoder_dtype(self, unified_model):
        unified_model.to(torch.bfloat16)
        encoded = torch.randn(2, ENC_HIDDEN, 9, dtype=torch.bfloat16)

        out = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 3, torch.bfloat16, encoded.device)
        )

        assert out.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_bfloat16_model_matches_float32_projection_exactly(self, unified_model):
        """After casting the model to bfloat16 the projection must still compute in float32.

        Feeding bfloat16-representable activations means the only difference left between a
        float32 model and a bfloat16 one is the projection's own arithmetic plus the final cast of
        the result. So the bfloat16 model's output must equal the float32 output rounded to
        bfloat16, bit for bit. A projection that followed the model-wide cast would not match:
        the hidden layer and ReLU would be evaluated in bfloat16 too.
        """
        torch.manual_seed(0)
        # Encoder activations an order of magnitude larger than the one-hot prompt, chosen to be
        # exactly representable in bfloat16 so they are not themselves a source of difference.
        encoded = (torch.randn(2, ENC_HIDDEN, 32) * 10.0).to(torch.bfloat16)

        def project(dtype):
            prompt = unified_model.create_lang_id_prompt(2, 3, dtype, encoded.device)
            return unified_model.apply_lang_id_prompt(encoded.to(dtype), prompt)

        reference = project(torch.float32)
        unified_model.to(torch.bfloat16)
        measured = project(torch.bfloat16)

        assert measured.dtype is torch.bfloat16
        assert torch.equal(measured, reference.to(torch.bfloat16))

    @pytest.mark.unit
    def test_languages_stay_distinguishable_in_bfloat16(self, unified_model):
        torch.manual_seed(0)
        unified_model.to(torch.bfloat16)
        encoded = (torch.randn(2, ENC_HIDDEN, 32) * 10.0).to(torch.bfloat16)

        out_en = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 0, torch.bfloat16, encoded.device)
        )
        out_ja = unified_model.apply_lang_id_prompt(
            encoded, unified_model.create_lang_id_prompt(2, 5, torch.bfloat16, encoded.device)
        )

        assert not torch.equal(out_en, out_ja)

    @pytest.mark.unit
    def test_device_moves_are_still_honoured(self, unified_model):
        unified_model.to(torch.device('cpu'))
        assert all(param.device.type == 'cpu' for param in unified_model.lang_id_prompt_kernel.parameters())


@requires_numba
class TestForwardIntegration:
    @pytest.mark.unit
    def test_forward_accepts_prompt(self, unified_model):
        unified_model.eval()
        signal = torch.randn(2, 16000)
        signal_len = torch.tensor([16000, 16000])
        prompt = unified_model.create_lang_id_prompt(2, 3, torch.float32, signal.device)

        with torch.no_grad():
            encoded, encoded_len = unified_model.forward(
                input_signal=signal, input_signal_length=signal_len, lang_id_prompt=prompt
            )
            baseline, _ = unified_model.forward(input_signal=signal, input_signal_length=signal_len)

        assert encoded.shape == baseline.shape
        assert encoded_len.shape == (2,)
        assert not torch.allclose(encoded, baseline)

    @pytest.mark.unit
    def test_forward_without_a_prompt_is_still_valid(self, unified_model):
        """Buffered and chunked streaming condition per chunk, so they call forward() prompt-free.

        Declaring the prompt as a required input would break those callers, which is exactly how the
        chunked script failed against a model whose forward demanded prompt indices.
        """
        unified_model.eval()

        with torch.no_grad():
            encoded, encoded_len = unified_model.forward(
                input_signal=torch.randn(2, 16000), input_signal_length=torch.tensor([16000, 16000])
            )

        assert encoded.shape[0] == 2
        assert encoded_len.shape == (2,)

    @pytest.mark.unit
    def test_transcribe_forward_conditions_on_the_config(self, unified_model):
        """``_transcribe_forward`` is the offline transcription entry point for ``source_lang``."""
        unified_model.eval()
        batch = (torch.randn(2, 16000), torch.tensor([16000, 16000]))

        with torch.no_grad():
            de = unified_model._transcribe_forward(batch, UnifiedTranscribeConfig(source_lang='de-DE'))
            ja = unified_model._transcribe_forward(batch, UnifiedTranscribeConfig(source_lang='ja-JP'))

        assert not torch.allclose(de['encoded'], ja['encoded'])


class TestBufferedWrapperPromptShape:
    """``encode_with_prompts`` must send each model family the prompt shape its forward expects."""

    @staticmethod
    def _capture_encode_call(language_conditioning_enabled: bool):
        from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper

        captured = {}

        class Stub:
            asr_model = SimpleNamespace(language_conditioning_enabled=language_conditioning_enabled)

            def get_subsampling_factor(self):
                return 8

            def encode(self, processed_signal, processed_signal_length, prompt_vectors=None):
                captured['shape'] = tuple(prompt_vectors.shape)
                return processed_signal, processed_signal_length

        # 100 feature frames is deliberately not a multiple of the subsampling factor.
        RNNTInferenceWrapper.encode_with_prompts(
            Stub(), torch.zeros(2, 80, 100), torch.tensor([100, 100]), torch.zeros(2, NUM_PROMPTS)
        )
        return captured['shape']

    @pytest.mark.unit
    def test_unified_model_gets_a_time_free_prompt(self):
        """Broadcasting inside the model avoids estimating the encoder length from the feature length."""
        assert self._capture_encode_call(language_conditioning_enabled=True) == (2, NUM_PROMPTS)

    @pytest.mark.unit
    def test_prompt_streaming_model_still_gets_an_expanded_prompt(self):
        """`concat` models declare their prompt as (B, T, D) and reject a 2D tensor."""
        assert self._capture_encode_call(language_conditioning_enabled=False) == (2, 100 // 8, NUM_PROMPTS)


@requires_numba
class TestCacheAwareDelegation:
    """Cache-aware streaming must not silently ignore a prompt the model was trained with.

    The path used to gate prompt injection on ``concat``, which unified models never set, so it
    skipped the trained projection entirely and returned plausible but wrong transcripts.
    """

    @pytest.mark.unit
    def test_prompt_vectors_are_delegated_to_the_model(self, unified_model):
        from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
            CacheAwareRNNTInferenceWrapper,
        )

        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt_vectors = unified_model.create_lang_id_prompt(2, 3, torch.float32, encoded.device)
        stub = SimpleNamespace(asr_model=unified_model)

        out = CacheAwareRNNTInferenceWrapper._apply_prompt_vectors(stub, encoded, prompt_vectors)

        assert torch.allclose(out, unified_model.apply_lang_id_prompt(encoded, prompt_vectors))

    @pytest.mark.unit
    def test_load_time_validation_accepts_a_unified_model(self, unified_model):
        """Unified models validate their own shapes, so the ``concat`` check must not fire."""
        from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
            CacheAwareRNNTInferenceWrapper,
        )

        CacheAwareRNNTInferenceWrapper._validate_prompt_support(SimpleNamespace(asr_model=unified_model))


@requires_numba
class TestPipelineDefaultLanguage:
    """Streaming pipelines take their default language from the model, not from a literal.

    The buffered and cache-aware pipelines used to hardcode different defaults, so the same
    checkpoint was conditioned on a different language depending on which pipeline ran it.
    """

    @staticmethod
    def _resolve(model, prompt_dict=None):
        from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline

        stub = SimpleNamespace(
            asr_model=SimpleNamespace(asr_model=model),
            _prompt_config={'prompt_dict': prompt_dict if prompt_dict is not None else {'en-US': 0}},
        )
        return BasePipeline._resolve_default_language_code(stub)

    @pytest.mark.unit
    def test_unified_model_uses_its_advertised_default(self, unified_model):
        assert self._resolve(unified_model) == unified_model.default_language == 'unk'

    @pytest.mark.unit
    @pytest.mark.parametrize(
        'prompt_dict, expected', [({'en-US': 0, 'auto': 1}, 'auto'), ({'en-US': 0}, 'en-US'), ({'de-DE': 0}, None)]
    )
    def test_prompt_streaming_model_prefers_its_auto_prompt(self, plain_model, prompt_dict, expected):
        """``concat`` models are matched against their own vocabulary, preferring ``auto``."""
        plain_model.concat = True
        assert self._resolve(plain_model, prompt_dict) == expected

    @pytest.mark.unit
    def test_returns_none_without_prompt_support(self):
        from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline

        assert BasePipeline._resolve_default_language_code(SimpleNamespace(_prompt_config=None)) is None
