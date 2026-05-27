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

"""Unit tests for the vLLM NeMo Speech LM (SALM) plugin.

Covers plugin registration, config loading + escape-hatch wiring, special
token handling, and backend selection -- without requiring GPU or model
weights.
"""

import importlib.util
from types import SimpleNamespace

import pytest

try:
    from nemo.collections.speechlm2.vllm.salm import config as _config_module

    NeMoSpeechLMConfig = _config_module.NeMoSpeechLMConfig

    _HAS_CONFIG = True
except (ImportError, RuntimeError):
    _HAS_CONFIG = False

_HAS_VLLM = importlib.util.find_spec("vllm") is not None
_DEFAULT_CONFIG_KWARGS = {
    "pretrained_llm": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "pretrained_asr": "nvidia/canary-1b-v2",
    "audio_locator_tag": "<|audio|>",
    "prompt_format": "nemotron-nano-v3",
    "pretrained_weights": True,
}


@pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
class TestNeMoSpeechLMConfig:
    """Tests for NeMoSpeechLMConfig."""

    @pytest.fixture(autouse=True)
    def mock_backbone_config(self, monkeypatch):
        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            if "Nemotron" in model_name:
                return SimpleNamespace(
                    architectures=["NemotronHybridForCausalLM"],
                    hidden_size=2048,
                    vocab_size=131072,
                    num_hidden_layers=4,
                    num_key_value_heads=2,
                    layer_norm_epsilon=1e-5,
                )
            return SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                hidden_size=2048,
                vocab_size=151936,
                num_hidden_layers=4,
                rms_norm_eps=1e-6,
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

    def test_model_type(self):
        assert NeMoSpeechLMConfig.model_type == "nemo_speechlm"

    def test_default_construction_for_hf_serialization(self):
        """HF internally constructs a no-arg config when serializing configs."""
        cfg = NeMoSpeechLMConfig()
        assert cfg.pretrained_llm is None
        assert cfg.pretrained_asr is None
        assert cfg.audio_locator_tag is None
        assert cfg.prompt_format is None
        assert cfg.pretrained_weights is None
        assert cfg.llm_architectures == []
        assert cfg.get_text_config() is cfg.text_config

    def test_loads_text_config(self):
        """Config should load a text_config from the pretrained LLM."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.text_config is not None
        assert hasattr(cfg.text_config, "hidden_size")
        assert cfg.get_text_config() is cfg.text_config

    def test_hybrid_backbone_aliases_for_vllm(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is True
        assert cfg.llm_architectures == ["NemotronHForCausalLM"]
        assert cfg.text_config.total_num_kv_heads == cfg.text_config.num_key_value_heads
        assert cfg.text_config.rms_norm_eps == cfg.text_config.layer_norm_epsilon

    @pytest.mark.parametrize(
        "architectures, expected_is_hybrid",
        [
            (["NemotronHForCausalLM"], True),
            (["NemotronHybridForCausalLM"], True),
            (["Qwen3ForCausalLM"], False),
            (["LlamaForCausalLM"], False),
            (["Qwen2ForCausalLM"], False),
        ],
    )
    def test_is_hybrid_backend_helper(self, architectures, expected_is_hybrid):
        """``_is_hybrid_backend`` should match the documented hybrid allow-list."""
        from nemo.collections.speechlm2.vllm.salm.config import _is_hybrid_backend

        assert _is_hybrid_backend(architectures) is expected_is_hybrid

    @pytest.mark.parametrize(
        "backbone_archs, expected_is_hybrid",
        [
            (["NemotronHForCausalLM"], True),
            (["NemotronHybridForCausalLM"], True),
            (["Qwen3ForCausalLM"], False),
        ],
    )
    def test_is_hybrid_set_from_backbone_architectures(self, monkeypatch, backbone_archs, expected_is_hybrid):
        """``cfg.is_hybrid`` is driven by the backbone HF config's ``architectures``."""

        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            kwargs = dict(
                architectures=backbone_archs,
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
            )
            if expected_is_hybrid:
                kwargs.update(num_key_value_heads=2, layer_norm_epsilon=1e-5)
            else:
                kwargs.update(rms_norm_eps=1e-6)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is expected_is_hybrid

    def test_hybrid_backbone_does_not_set_layer_types_shim(self):
        """Hybrid backbones must NOT have layer_types overridden -- the runtime
        is_hybrid escape hatch only fires when every layer is 'attention'."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.is_hybrid is True
        assert getattr(cfg.text_config, "layer_types", None) is None

    def test_transformer_backbone_engages_layer_types_shim(self):
        """Non-hybrid backbones get layer_types=['attention']*N so vLLM's
        ModelConfig.is_hybrid property returns False at runtime even though
        the model class declares IsHybrid (needed for NemotronH path)."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        assert cfg.is_hybrid is False
        assert cfg.text_config.layer_types == ["attention"] * 4

    def test_custom_pretrained_llm(self):
        """Config should accept different LLM backbones."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        assert cfg.pretrained_llm == "Qwen/Qwen3-1.7B"
        assert cfg.text_config is not None
        assert cfg.llm_architectures == ["Qwen3ForCausalLM"]

    def test_audio_locator_tag_default_accepted(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.audio_locator_tag == "<|audio|>"

    def test_audio_locator_tag_custom_rejected(self):
        """Plugin only supports ``<|audio|>``; mismatched checkpoints fail at load time."""
        with pytest.raises(ValueError, match="audio_locator_tag"):
            NeMoSpeechLMConfig(
                **{
                    **_DEFAULT_CONFIG_KWARGS,
                    "audio_locator_tag": "<|custom_audio|>",
                }
            )

    @pytest.mark.parametrize(
        "field",
        [
            "pretrained_llm",
            "pretrained_asr",
            "audio_locator_tag",
            "prompt_format",
            "pretrained_weights",
        ],
    )
    def test_required_exported_fields(self, field):
        kwargs = dict(_DEFAULT_CONFIG_KWARGS)
        kwargs.pop(field)
        with pytest.raises(ValueError, match=field):
            NeMoSpeechLMConfig(**kwargs)

    def test_unknown_attr_raises(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent_attribute_xyz


@pytest.mark.skipif(not (_HAS_CONFIG and _HAS_VLLM), reason="NeMoSpeechLMConfig or vLLM not available")
class TestBackendSelection:
    """Tests for ``backends.make_backend`` dispatch on hybrid/transformer configs."""

    @pytest.fixture(autouse=True)
    def mock_backbone_config(self, monkeypatch):
        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            if "Nemotron" in model_name:
                return SimpleNamespace(
                    architectures=["NemotronHybridForCausalLM"],
                    hidden_size=2048,
                    vocab_size=131072,
                    num_hidden_layers=4,
                    num_key_value_heads=2,
                    layer_norm_epsilon=1e-5,
                )
            return SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                hidden_size=2048,
                vocab_size=151936,
                num_hidden_layers=4,
                rms_norm_eps=1e-6,
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

    def test_hybrid_config_picks_hybrid_backend(self):
        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend, make_backend

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        backend = make_backend(cfg)
        assert isinstance(backend, HybridBackend)
        assert backend.architectures() == ["NemotronHForCausalLM"]

    def test_transformer_config_picks_transformer_backend(self):
        from nemo.collections.speechlm2.vllm.salm.backends import TransformerBackend, make_backend

        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "Qwen/Qwen3-1.7B",
            }
        )
        backend = make_backend(cfg)
        assert isinstance(backend, TransformerBackend)
        assert backend.architectures() == ["Qwen3ForCausalLM"]


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestSpecialTokens:
    """Tests for special token handling."""

    def test_adds_missing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.salm.audio import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_called_once()

    def test_skips_existing_token(self):
        from unittest.mock import MagicMock

        from nemo.collections.speechlm2.vllm.salm.audio import _ensure_special_tokens

        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {"<|audio|>": 99}
        _ensure_special_tokens(tokenizer)
        tokenizer.add_special_tokens.assert_not_called()

    def test_placeholder_str(self):
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert NeMoSpeechLMForConditionalGeneration.get_placeholder_str("audio", 0) == "<|audio|>"
        assert NeMoSpeechLMForConditionalGeneration.get_placeholder_str("image", 0) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestAudioProcessing:
    """Tests for audio encoding with a tiny perception module."""

    def test_data_parser_normalizes_audio(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        info = object.__new__(NeMoSpeechLMProcessingInfo)
        monkeypatch.setattr(info, "_get_expected_hidden_size", lambda: 2048)

        parser = info.get_data_parser()

        assert parser.audio_resampler.target_sr == 16000
        assert parser.target_channels == 1

    def test_processing_info_has_no_audio_duration_limit(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        info = object.__new__(NeMoSpeechLMProcessingInfo)

        assert not hasattr(info, "get_max_audio_len")
        assert not hasattr(info, "get_max_audio_tokens")

    def test_dummy_inputs_use_profiling_audio_length(self):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            NeMoSpeechLMDummyInputsBuilder,
            NeMoSpeechLMProcessingInfo,
        )

        info = object.__new__(NeMoSpeechLMProcessingInfo)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = info

        result = builder.get_dummy_mm_data(seq_len=0, mm_counts={"audio": 1}, mm_options={})

        assert result["audio"][0].shape[-1] == 40 * 16000

    def test_call_hf_processor_requires_matching_placeholder_count(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_FakeTokenizer,
            _estimate_audio_tokens=lambda samples: 2,
        )

        with pytest.raises(ValueError, match="placeholders"):
            processor._call_hf_processor(
                prompt="Transcribe this audio",
                mm_data={"audios": [[0.0] * 16000]},
                mm_kwargs={},
                tok_kwargs={},
            )

    def test_call_hf_processor_emits_true_audio_lengths(self):
        import torch

        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_FakeTokenizer,
            _estimate_audio_tokens=lambda samples: 2,
        )

        result = processor._call_hf_processor(
            prompt="Transcribe: <|audio|>",
            mm_data={"audios": [[0.0] * 12345]},
            mm_kwargs={},
            tok_kwargs={},
        )

        assert len(result["audio_signal"]) == 1
        assert result["audio_signal"][0].shape[-1] == 12345
        assert torch.equal(result["audio_signal_length"], torch.tensor([12345]))

    def test_perception_forward(self):
        """A small NeMo perception module should encode dummy audio to embeddings."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from nemo.collections.speechlm2.vllm.salm.audio import _load_nemo_perception

        perception_cfg = {
            "output_dim": 256,
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "feat_in": 128,
                "feat_out": -1,
                "n_layers": 2,
                "d_model": 256,
                "subsampling": "dw_striding",
                "subsampling_factor": 8,
                "subsampling_conv_channels": 64,
                "ff_expansion_factor": 4,
                "self_attention_model": "rel_pos",
                "n_heads": 4,
                "conv_kernel_size": 9,
                "conv_norm_type": "batch_norm",
                "dropout": 0.0,
                "dropout_pre_encoder": 0.0,
                "dropout_emb": 0.0,
                "dropout_att": 0.0,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": 256,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 128,
                "n_fft": 512,
                "log": True,
                "frame_splicing": 1,
                "dither": 0.0,
                "pad_to": 0,
                "pad_value": 0.0,
            },
        }

        perception = _load_nemo_perception(perception_cfg)
        perception = perception.to("cuda", dtype=torch.float32)

        dummy_audio = torch.randn(1, 16000, device="cuda")
        audio_len = torch.tensor([16000], device="cuda")

        with torch.no_grad():
            embeds, embed_lens = perception(input_signal=dummy_audio, input_signal_length=audio_len)

        assert embeds.ndim == 3
        assert embeds.shape[0] == 1
        assert embeds.shape[2] == 256
        assert embed_lens[0] > 0


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestPluginRegistration:
    """Tests for plugin registration with vLLM."""

    def test_register_config(self, monkeypatch):
        """register() should add nemo_speechlm to vLLM's config registry."""
        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        register()

        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        assert "nemo_speechlm" in _CONFIG_REGISTRY

    def test_register_model(self, monkeypatch):
        """register() should make NeMoSpeechLMForConditionalGeneration importable.

        The plugin now registers a single architecture name; the obsolete
        ``NeMoSpeechLMHybridForConditionalGeneration`` no longer appears.
        """
        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        register()

        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert "NeMoSpeechLMForConditionalGeneration" in ModelRegistry.get_supported_archs()
        assert NeMoSpeechLMForConditionalGeneration is not None

    def test_register_does_not_patch_fast_tokenizer(self, monkeypatch):
        from transformers import AutoConfig, PreTrainedTokenizerFast

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig, "from_pretrained", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
        )

        assert "_orig_batch_encode_plus" not in PreTrainedTokenizerFast.__dict__
        register()
        assert "_orig_batch_encode_plus" not in PreTrainedTokenizerFast.__dict__

    def test_register_does_not_load_backbone_config(self, monkeypatch):
        from unittest.mock import Mock

        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        from_pretrained = Mock(side_effect=AssertionError("register() must not load remote backbone configs"))
        monkeypatch.setattr(AutoConfig, "from_pretrained", from_pretrained)

        register()

        from_pretrained.assert_not_called()


class _FakeTokenizer:
    def __init__(self):
        self.added_special_tokens = None

    def get_vocab(self):
        return {}

    def add_special_tokens(self, tokens):
        self.added_special_tokens = tokens

    def encode(self, prompt, add_special_tokens=True):
        return list(range(max(1, len(prompt.split()))))
