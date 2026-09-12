# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import EncDecRNNTModel
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


def build_model(
    prompt_enabled: bool, prompt_dictionary=PROMPT_DICTIONARY, extra_defaults=None, trainer=None
) -> EncDecRNNTModel:
    """Build a tiny RNNT model, optionally with language-ID prompt conditioning enabled."""
    model_defaults = {'enc_hidden': ENC_HIDDEN, 'pred_hidden': 32}
    if prompt_enabled:
        model_defaults.update(
            {
                'initialize_lang_id_prompt': True,
                'num_lang_id_prompts': NUM_PROMPTS,
                'lang_id_prompt_dictionary': prompt_dictionary,
            }
        )
    if extra_defaults:
        model_defaults.update(extra_defaults)

    cfg = DictConfig(
        {
            'labels': ListConfig([' ', 'a', 'b', 'c']),
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
            'decoding': DictConfig({'strategy': 'greedy_batch', 'greedy': {'max_symbols': 5}}),
            'loss': DictConfig({'loss_name': 'default'}),
        }
    )
    return EncDecRNNTModel(cfg=cfg, trainer=trainer)


@pytest.fixture()
def prompt_model():
    return build_model(prompt_enabled=True)


@pytest.fixture()
def plain_model():
    return build_model(prompt_enabled=False)


@requires_numba
class TestLangIdPromptSetup:
    @pytest.mark.unit
    def test_disabled_by_default(self, plain_model):
        assert plain_model.use_lang_id_prompt is False
        assert plain_model.num_lang_id_prompts is None
        assert plain_model.lang_id_prompt_dictionary is None
        assert not hasattr(plain_model, 'lang_id_prompt_kernel')
        assert 'lang_id_prompt' not in plain_model.input_types

    @pytest.mark.unit
    def test_enabled_from_config(self, prompt_model):
        assert prompt_model.use_lang_id_prompt is True
        assert prompt_model.num_lang_id_prompts == NUM_PROMPTS
        assert prompt_model.lang_id_prompt_dictionary == PROMPT_DICTIONARY
        assert prompt_model.lang_id_prompt_kernel[0].in_features == NUM_PROMPTS + ENC_HIDDEN
        assert prompt_model.lang_id_prompt_kernel[-1].out_features == ENC_HIDDEN
        assert 'lang_id_prompt' in prompt_model.input_types

    @pytest.mark.unit
    def test_flag_is_readable_on_any_rnnt_model_without_getattr(self):
        """The flag lives on ``EncDecRNNTModel``, so RNN-T call sites can branch without a guard.

        It deliberately does not reach ``ASRModel``: that class is imported too early to see
        ``parts.mixins``, which imports ``nemo.collections.asr.models`` in turn.
        """
        from nemo.collections.asr.models.asr_model import ASRModel

        assert EncDecRNNTModel.use_lang_id_prompt is False
        assert not hasattr(ASRModel, 'use_lang_id_prompt')

    @pytest.mark.unit
    def test_prompt_kernel_weights_are_in_state_dict(self, prompt_model):
        keys = [key for key in prompt_model.state_dict() if key.startswith('lang_id_prompt_kernel.')]
        assert keys, "lang_id_prompt_kernel must be a registered submodule so checkpoints round-trip"

    @pytest.mark.unit
    def test_missing_prompt_dictionary_raises(self):
        with pytest.raises(ValueError, match='lang_id_prompt_dictionary'):
            build_model(prompt_enabled=True, prompt_dictionary={})

    @pytest.mark.unit
    @pytest.mark.parametrize('other_scheme_key', ['initialize_prompt_feature', 'num_prompts', 'prompt_dictionary'])
    def test_prompt_streaming_config_keys_are_left_alone(self, other_scheme_key):
        """``PromptStreamingMixin``'s keys are a live namespace, not a legacy one.

        They appear in the shipped ``fastconformer_*_prompt.yaml`` configs, so a model instantiated
        from such a config through a class without that mixin must still load rather than fail on
        keys it simply does not use.
        """
        value = PROMPT_DICTIONARY if other_scheme_key == 'prompt_dictionary' else True

        model = build_model(prompt_enabled=False, extra_defaults={other_scheme_key: value})

        assert model.use_lang_id_prompt is False


@requires_numba
class TestInferenceOnly:
    """The projection is trainable but no training step conditions on it, so training must refuse.

    Left unguarded, DDP fails on an unused parameter and a manual loop trains (and scores WER)
    without the language conditioning the model was built for.
    """

    @pytest.mark.unit
    def test_enabling_prompts_with_a_trainer_attached_raises(self):
        trainer = pytest.importorskip('lightning.pytorch').Trainer(accelerator='cpu', devices=1, logger=False)

        with pytest.raises(ValueError, match='inference-only'):
            build_model(prompt_enabled=True, trainer=trainer)

    @pytest.mark.unit
    def test_models_without_prompts_are_unaffected(self):
        trainer = pytest.importorskip('lightning.pytorch').Trainer(accelerator='cpu', devices=1, logger=False)

        model = build_model(prompt_enabled=False, trainer=trainer)

        assert model.use_lang_id_prompt is False


@requires_numba
class TestResolveLangIdPrompt:
    @pytest.mark.unit
    def test_known_language(self, prompt_model):
        assert prompt_model.resolve_lang_id_prompt('de-DE') == 3
        assert prompt_model.resolve_lang_id_prompt('en') == 0

    @pytest.mark.unit
    @pytest.mark.parametrize('target_lang', [None, 'kl-KL'])
    def test_falls_back_to_the_default_language(self, prompt_model, target_lang):
        assert prompt_model.resolve_lang_id_prompt(target_lang) == PROMPT_DICTIONARY['unk']

    @pytest.mark.unit
    def test_raises_when_no_candidate_default_exists(self):
        model = build_model(prompt_enabled=True, prompt_dictionary={'fr-FR': 0})
        with pytest.raises(ValueError, match='Cannot pick a default'):
            model.resolve_lang_id_prompt(None)

    @pytest.mark.unit
    def test_raises_for_non_prompt_model(self, plain_model):
        with pytest.raises(ValueError, match='not trained with language-ID prompt'):
            plain_model.resolve_lang_id_prompt('en-US')


@requires_numba
class TestDefaultLangIdPrompt:
    """The model advertises its own language-agnostic prompt, so no call site hardcodes one."""

    @pytest.mark.unit
    def test_prefers_the_most_language_agnostic_key(self, prompt_model):
        assert prompt_model.default_lang_id_prompt == 'unk'

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
    def test_falls_through_the_preference_order(self, dictionary, expected):
        assert build_model(prompt_enabled=True, prompt_dictionary=dictionary).default_lang_id_prompt == expected

    @pytest.mark.unit
    def test_is_none_without_prompt_support(self, plain_model):
        assert plain_model.default_lang_id_prompt is None


@requires_numba
class TestApplyLangIdPrompt:
    @pytest.mark.unit
    def test_onehot_prompt_shape_and_content(self, prompt_model):
        prompt = prompt_model.create_lang_id_prompt(3, 5, dtype=torch.float32, device=torch.device('cpu'))
        assert prompt.shape == (3, NUM_PROMPTS)
        assert torch.equal(prompt.sum(dim=-1), torch.ones(3))
        assert torch.all(prompt[:, 5] == 1.0)

    @pytest.mark.unit
    def test_batched_onehot_prompts_one_language_per_row(self, prompt_model):
        """Per-sample prompting (e.g. per-utterance manifest languages) selects a language per row."""
        prompt = prompt_model.create_lang_id_prompts(
            torch.tensor([0, 5, 3]), dtype=torch.float32, device=torch.device('cpu')
        )
        assert prompt.shape == (3, NUM_PROMPTS)
        assert torch.equal(prompt.sum(dim=-1), torch.ones(3))
        assert prompt[0, 0] == 1.0 and prompt[1, 5] == 1.0 and prompt[2, 3] == 1.0
        # A single-index batch must match the scalar builder.
        assert torch.equal(
            prompt_model.create_lang_id_prompts([5, 5], dtype=torch.float32, device=torch.device('cpu')),
            prompt_model.create_lang_id_prompt(2, 5, dtype=torch.float32, device=torch.device('cpu')),
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('time_steps', [1, 7, 13, 91])
    def test_broadcasts_over_any_encoder_length(self, prompt_model, time_steps):
        """A (B, num_lang_id_prompts) prompt must fit any encoder length.

        Regression: callers used to precompute the time dimension from the feature length, which is
        off by one whenever that length is not a multiple of the subsampling factor. The shortfall was
        silently zero-padded, conditioning the tail of every chunk on an all-zero prompt.
        """
        encoded = torch.randn(2, ENC_HIDDEN, time_steps)
        prompt = prompt_model.create_lang_id_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)

        out = prompt_model.apply_lang_id_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, time_steps)

    @pytest.mark.unit
    def test_per_frame_prompt_with_matching_length_is_accepted(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, 9, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        out = prompt_model.apply_lang_id_prompt(encoded, prompt)

        assert out.shape == (2, ENC_HIDDEN, 9)
        # Broadcasting a 2D prompt must be equivalent to an explicit per-frame prompt.
        broadcast = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 3, dtype=encoded.dtype, device=encoded.device)
        )
        assert torch.allclose(out, broadcast)

    @pytest.mark.unit
    def test_per_row_prompts_are_independent(self, prompt_model):
        """Batched streaming conditions each stream on its own language."""
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        mixed = torch.zeros(2, NUM_PROMPTS)
        mixed[0, 0] = 1.0
        mixed[1, 5] = 1.0

        out = prompt_model.apply_lang_id_prompt(encoded, mixed)
        en_only = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 0, torch.float32, encoded.device)
        )
        ja_only = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 5, torch.float32, encoded.device)
        )

        assert torch.allclose(out[0], en_only[0])
        assert torch.allclose(out[1], ja_only[1])

    @pytest.mark.unit
    @pytest.mark.parametrize('prompt_time_steps', [8, 10])
    def test_per_frame_prompt_length_mismatch_raises(self, prompt_model, prompt_time_steps):
        """Silently padding/truncating a per-frame prompt is what produced the tail-frame bug."""
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt = torch.zeros(2, prompt_time_steps, NUM_PROMPTS)
        prompt[:, :, 3] = 1.0

        with pytest.raises(ValueError, match='time steps'):
            prompt_model.apply_lang_id_prompt(encoded, prompt)

    @pytest.mark.unit
    def test_wrong_num_prompts_raises(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='classes'):
            prompt_model.apply_lang_id_prompt(encoded, torch.zeros(2, NUM_PROMPTS + 1))

    @pytest.mark.unit
    def test_different_languages_give_different_output(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_en = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 0, torch.float32, encoded.device)
        )
        out_ja = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 5, torch.float32, encoded.device)
        )
        assert not torch.allclose(out_en, out_ja)

    @pytest.mark.unit
    def test_transcribe_helper_is_noop_without_prompt_support(self, plain_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        assert plain_model.apply_lang_id_prompt_for_transcribe(encoded, 'en-US') is encoded

    @pytest.mark.unit
    def test_transcribe_helper_conditions_on_target_lang(self, prompt_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        out_de = prompt_model.apply_lang_id_prompt_for_transcribe(encoded, 'de-DE')
        out_ja = prompt_model.apply_lang_id_prompt_for_transcribe(encoded, 'ja-JP')
        assert out_de.shape == encoded.shape
        assert not torch.allclose(out_de, out_ja)

    @pytest.mark.unit
    def test_prompt_rejected_by_non_prompt_model(self, plain_model):
        encoded = torch.randn(2, ENC_HIDDEN, 9)
        with pytest.raises(ValueError, match='not trained with language-ID prompt'):
            plain_model.apply_lang_id_prompt(encoded, torch.zeros(2, NUM_PROMPTS))


@requires_numba
class TestCacheAwareSurfaceIsNotClaimed:
    """The mixin must not answer the cache-aware feature probes, on any model.

    ``speech_to_text_cache_aware_streaming_infer.py`` feature-detects prompt support. Because the
    mixin sits on ``ASRModel``, anything it defines is defined for every ASR model — so claiming
    ``set_inference_prompt`` made that probe true everywhere and broke the script for plain models.
    Cache-aware streaming belongs to ``PromptStreamingMixin``, whose models are trained for it.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize('attribute', ['set_inference_prompt', 'concat'])
    def test_probes_stay_false_for_a_plain_model(self, plain_model, attribute):
        assert not hasattr(plain_model, attribute)

    @pytest.mark.unit
    @pytest.mark.parametrize('attribute', ['set_inference_prompt', 'concat'])
    def test_probes_stay_false_for_a_unified_model(self, prompt_model, attribute):
        assert not hasattr(prompt_model, attribute)

    @pytest.mark.unit
    def test_the_stream_step_hook_is_left_as_the_base_no_op(self, prompt_model):
        """``ASRModuleMixin``'s no-op must remain the implementation the MRO finds."""
        from nemo.collections.asr.parts.mixins.mixins import ASRModuleMixin

        owner = next(c for c in type(prompt_model).__mro__ if '_apply_prompt_to_encoded' in c.__dict__)
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
    def test_kernel_stays_float32_after_model_cast(self, prompt_model):
        prompt_model.to(torch.bfloat16)

        assert all(param.dtype is torch.float32 for param in prompt_model.lang_id_prompt_kernel.parameters())
        # ... while the rest of the model did get cast.
        assert prompt_model.encoder.encoder[0].mconv[0].conv.weight.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_output_keeps_encoder_dtype(self, prompt_model):
        prompt_model.to(torch.bfloat16)
        encoded = torch.randn(2, ENC_HIDDEN, 9, dtype=torch.bfloat16)

        out = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 3, torch.bfloat16, encoded.device)
        )

        assert out.dtype is torch.bfloat16

    @pytest.mark.unit
    def test_bfloat16_model_matches_float32_projection_exactly(self, prompt_model):
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
            prompt = prompt_model.create_lang_id_prompt(2, 3, dtype, encoded.device)
            return prompt_model.apply_lang_id_prompt(encoded.to(dtype), prompt)

        reference = project(torch.float32)
        prompt_model.to(torch.bfloat16)
        measured = project(torch.bfloat16)

        assert measured.dtype is torch.bfloat16
        assert torch.equal(measured, reference.to(torch.bfloat16))

    @pytest.mark.unit
    def test_languages_stay_distinguishable_in_bfloat16(self, prompt_model):
        torch.manual_seed(0)
        prompt_model.to(torch.bfloat16)
        encoded = (torch.randn(2, ENC_HIDDEN, 32) * 10.0).to(torch.bfloat16)

        out_en = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 0, torch.bfloat16, encoded.device)
        )
        out_ja = prompt_model.apply_lang_id_prompt(
            encoded, prompt_model.create_lang_id_prompt(2, 5, torch.bfloat16, encoded.device)
        )

        assert not torch.equal(out_en, out_ja)

    @pytest.mark.unit
    def test_device_moves_are_still_honoured(self, prompt_model):
        prompt_model.to(torch.device('cpu'))
        assert all(param.device.type == 'cpu' for param in prompt_model.lang_id_prompt_kernel.parameters())


@requires_numba
class TestForwardIntegration:
    @pytest.mark.unit
    def test_forward_accepts_prompt(self, prompt_model):
        prompt_model.eval()
        signal = torch.randn(2, 16000)
        signal_len = torch.tensor([16000, 16000])
        prompt = prompt_model.create_lang_id_prompt(2, 3, torch.float32, signal.device)

        with torch.no_grad():
            encoded, encoded_len = prompt_model.forward(
                input_signal=signal, input_signal_length=signal_len, lang_id_prompt=prompt
            )
            baseline, _ = prompt_model.forward(input_signal=signal, input_signal_length=signal_len)

        assert encoded.shape == baseline.shape
        assert encoded_len.shape == (2,)
        assert not torch.allclose(encoded, baseline)


class TestBufferedWrapperPromptShape:
    """``encode_with_prompts`` must send each model family the prompt shape its forward expects."""

    @staticmethod
    def _capture_encode_call(use_lang_id_prompt: bool):
        from nemo.collections.asr.inference.model_wrappers.rnnt_inference_wrapper import RNNTInferenceWrapper

        captured = {}

        class Stub:
            asr_model = SimpleNamespace(use_lang_id_prompt=use_lang_id_prompt)

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
        assert self._capture_encode_call(use_lang_id_prompt=True) == (2, NUM_PROMPTS)

    @pytest.mark.unit
    def test_prompt_streaming_model_still_gets_an_expanded_prompt(self):
        """`concat` models declare their prompt as (B, T, D) and reject a 2D tensor."""
        assert self._capture_encode_call(use_lang_id_prompt=False) == (2, 100 // 8, NUM_PROMPTS)


@requires_numba
class TestCacheAwareDelegation:
    """Cache-aware streaming must not silently ignore a prompt the model was trained with.

    The path used to gate prompt injection on ``concat``, which unified models never set, so it
    skipped the trained projection entirely and returned plausible but wrong transcripts.
    """

    @pytest.mark.unit
    def test_prompt_vectors_are_delegated_to_the_model(self, prompt_model):
        from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
            CacheAwareRNNTInferenceWrapper,
        )

        encoded = torch.randn(2, ENC_HIDDEN, 9)
        prompt_vectors = prompt_model.create_lang_id_prompt(2, 3, torch.float32, encoded.device)
        stub = SimpleNamespace(asr_model=prompt_model)

        out = CacheAwareRNNTInferenceWrapper._apply_prompt_vectors(stub, encoded, prompt_vectors)

        assert torch.allclose(out, prompt_model.apply_lang_id_prompt(encoded, prompt_vectors))

    @pytest.mark.unit
    def test_load_time_validation_accepts_a_unified_model(self, prompt_model):
        """Unified models validate their own shapes, so the ``concat`` check must not fire."""
        from nemo.collections.asr.inference.model_wrappers.cache_aware_rnnt_inference_wrapper import (
            CacheAwareRNNTInferenceWrapper,
        )

        CacheAwareRNNTInferenceWrapper._validate_prompt_support(SimpleNamespace(asr_model=prompt_model))


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
    def test_unified_model_uses_its_advertised_default(self, prompt_model):
        assert self._resolve(prompt_model) == prompt_model.default_lang_id_prompt == 'unk'

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
