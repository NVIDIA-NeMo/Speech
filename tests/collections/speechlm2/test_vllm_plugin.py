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

"""Unit tests for the vLLM NeMo Speech LM (SALM) plugin.

Covers plugin registration, config loading + escape-hatch wiring, special
token handling, and backend selection -- without requiring GPU or model
weights.
"""

import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path
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


def test_full_stack_asr_exports_grouped_encoder_dependencies():
    repo_root = Path(__file__).parents[3]
    code = """
from omegaconf import OmegaConf
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.modules import (
    ConvASRDecoder,
    GGEMMTransformerEncoder,
    MoETransformerEncoder,
    ParallelExpertEncoder,
    TransformerEncoder,
)
from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT

cfg = OmegaConf.load(
    "examples/speaker_tasks/diarization/conf/neural_diarizer/streaming_sortformer_diarizer_4spk-v2.yaml"
)
for name in ("train_ds", "validation_ds", "test_ds"):
    cfg.model[name] = None
cfg.model.spec_augment = {
    "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
    "freq_masks": 2,
    "time_masks": 2,
    "freq_width": 10,
    "time_width": 0.05,
}
model = SortformerEncLabelModel(cfg.model).eval()
assert ASRModel is not None
assert all(
    dependency is not None
    for dependency in (
        SortformerEncLabelModel,
        ConvASRDecoder,
        GGEMMTransformerEncoder,
        MoETransformerEncoder,
        TransformerEncoder,
        ParallelExpertEncoder,
        ParallelExpertEncoderPT,
    )
)
assert model.cfg.get("spec_augment") is not None
assert model.spec_augmentation is not None
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


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
        assert cfg.llm_config is None
        assert cfg.pretrained_asr is None
        assert cfg.audio_locator_tag is None
        assert cfg.prompt_format is None
        assert cfg.pretrained_weights is None
        assert cfg.pe_encoder_path is None
        assert cfg.pe_encoder_config is None
        assert cfg.pe_encoder_overrides is None
        assert cfg.speaker_encoder is None
        assert cfg.llm_architectures == []
        assert cfg.get_text_config() is cfg.text_config

    def test_loads_text_config(self):
        """Config should load a text_config from the pretrained LLM."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.text_config is not None
        assert hasattr(cfg.text_config, "hidden_size")
        assert cfg.get_text_config() is cfg.text_config

    def test_uses_embedded_backbone_config(self, monkeypatch):
        """Bundled exports must not reload the stale training backbone reference."""
        embedded_config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 2048,
            "vocab_size": 151936,
            "num_hidden_layers": 4,
            "rms_norm_eps": 1e-6,
        }
        text_config = SimpleNamespace(**embedded_config)

        def fail_from_pretrained(*args, **kwargs):
            pytest.fail("must not load pretrained_llm when llm_config is embedded")

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", fail_from_pretrained)
        monkeypatch.setattr(
            _config_module.AutoConfig,
            "for_model",
            lambda model_type, **kwargs: text_config,
        )

        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "pretrained_llm": "llm_backbone",
                "llm_config": embedded_config,
            }
        )

        assert cfg.pretrained_llm == "llm_backbone"
        assert cfg.llm_config == embedded_config
        assert cfg.text_config is text_config

    def test_preserves_explicit_phpee_export_schema(self):
        pe_config = {
            "target": "nemo.collections.asr.modules.parallel_expert_encoder.ParallelExpertEncoderPT",
            "chunk_size_seconds": 30.0,
        }
        cfg = NeMoSpeechLMConfig(
            **_DEFAULT_CONFIG_KWARGS,
            pe_encoder_config=pe_config,
        )

        assert cfg.pe_encoder_path is None
        assert cfg.pe_encoder_config == pe_config

    def test_preserves_path_phpee_overrides(self):
        overrides = {
            "speaker_feature_config_version": 1,
            "speaker_feature_mode": "continuous",
            "speaker_activity_threshold": None,
            "diar_normalize_type": "per_feature",
        }
        cfg = NeMoSpeechLMConfig(
            **_DEFAULT_CONFIG_KWARGS,
            pe_encoder_path="/models/placeholderParallelExpertEncoder.nemo",
            pe_encoder_overrides=overrides,
        )

        assert cfg.pe_encoder_overrides == overrides

    def test_preserves_independent_speaker_encoder_export_schema(self):
        speaker_config = {
            "path": "/models/speaker-transformer",
            "frozen": True,
            "chunk_size_seconds": 120.0,
            "asr_chunk_size_seconds": None,
        }
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, speaker_encoder=speaker_config)

        assert cfg.speaker_encoder == speaker_config
        assert cfg.encoder_chunk_size_seconds is None

    def test_rejects_phpee_and_independent_speaker_encoder_together(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            NeMoSpeechLMConfig(
                **_DEFAULT_CONFIG_KWARGS,
                pe_encoder_config={"target": "ParallelExpertEncoderPT"},
                speaker_encoder={"path": "/models/speaker-transformer"},
            )

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

    def test_encoder_chunk_size_seconds_default_none(self):
        """Legacy checkpoints without a chunk size keep the single-pass encoder path."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.encoder_chunk_size_seconds is None

    def test_encoder_chunk_size_seconds_round_trips(self):
        """Chunk size set in config.json (e.g. SALMAutomodel default 30 s) survives load."""
        cfg = NeMoSpeechLMConfig(
            **{
                **_DEFAULT_CONFIG_KWARGS,
                "encoder_chunk_size_seconds": 30.0,
            }
        )
        assert cfg.encoder_chunk_size_seconds == 30.0

    def test_encoder_chunk_size_seconds_default_init_inert(self):
        """No-arg default init must still expose ``encoder_chunk_size_seconds=None``."""
        cfg = NeMoSpeechLMConfig()
        assert cfg.encoder_chunk_size_seconds is None


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
class TestHybridBackendWeightMapping:
    """Tests for the NeMo/Automodel -> vLLM NemotronH weight boundary."""

    @pytest.fixture
    def backend(self):
        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend

        config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=None))
        return HybridBackend(config)

    @pytest.mark.parametrize(
        ("holder_name", "vllm_name"),
        [
            ("_fp32_params.A_log", "A"),
            ("_fp32_params.dt_bias", "dt_bias"),
            ("_fp32_params.D", "D"),
        ],
    )
    def test_canonicalizes_fp32_param_holder_names_without_changing_tensors(self, backend, holder_name, vllm_name):
        import torch

        tensor = torch.tensor([-2.0, 0.5, 3.0])
        original_values = tensor.clone()
        [(mapped_name, mapped_tensor)] = backend.nemo_to_hf_llm_weights(
            [(f"llm.model.layers.0.mixer.{holder_name}", tensor)]
        )

        assert mapped_name == f"backbone.layers.0.mixer.{vllm_name}"
        assert mapped_tensor is tensor
        assert torch.equal(mapped_tensor, original_values)

    @pytest.mark.parametrize("holder_name", ["_fp32_params.A_log", "_fp32_params.dt_bias", "_fp32_params.D"])
    def test_does_not_canonicalize_non_mixer_fp32_param_holders(self, backend, holder_name):
        import torch

        source_name = f"llm.model.layers.0.other.{holder_name}"
        tensor = torch.tensor([1.0])
        [(mapped_name, mapped_tensor)] = backend.nemo_to_hf_llm_weights([(source_name, tensor)])

        assert mapped_name == f"backbone.layers.0.other.{holder_name}"
        assert mapped_tensor is tensor

    @pytest.mark.parametrize("param_name", ["A", "dt_bias", "D"])
    def test_already_canonical_fp32_param_names_are_unchanged(self, backend, param_name):
        import torch

        canonical_name = f"backbone.layers.0.mixer.{param_name}"
        tensor = torch.tensor([1.0])
        [(mapped_name, mapped_tensor)] = backend.nemo_to_hf_llm_weights([(canonical_name, tensor)])

        assert mapped_name == canonical_name
        assert mapped_tensor is tensor

    def test_a_log_reaches_vllm_loader_and_is_transformed_once(self, backend, monkeypatch):
        import torch
        from torch import nn
        from vllm.model_executor.layers.mamba import mamba_mixer2
        from vllm.model_executor.model_loader import weight_utils
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
        from vllm.model_executor.models.utils import AutoWeightsLoader

        a_log = torch.tensor([-2.0, 0.5, 3.0])
        model = nn.Module()
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.layers[0].mixer = nn.Module()
        model.model.layers[0].mixer.A = nn.Parameter(torch.empty_like(a_log))

        monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: 0)
        model.model.layers[0].mixer.A.weight_loader = mamba_mixer2.composed_weight_loader(
            mamba_mixer2.sharded_weight_loader(0),
            lambda tensor: -torch.exp(tensor.float()),
        )

        hf_weights = backend.nemo_to_hf_llm_weights([("llm.model.layers.0.mixer._fp32_params.A_log", a_log)])
        loaded = AutoWeightsLoader(model).load_weights(
            hf_weights,
            mapper=NemotronHForCausalLM.hf_to_vllm_mapper,
        )

        assert loaded == {"model.layers.0.mixer.A"}
        assert torch.equal(model.model.layers[0].mixer.A, -torch.exp(a_log))

    def test_ordinary_and_moe_mappings_are_unchanged(self, backend):
        import torch

        ordinary = torch.arange(6).reshape(2, 3)
        down_projs = torch.arange(12).reshape(2, 2, 3)
        gate_and_up_projs = torch.arange(24).reshape(2, 4, 3)
        mapped = list(
            backend.nemo_to_hf_llm_weights(
                [
                    ("llm.model.layers.1.mixer.in_proj.weight", ordinary),
                    ("llm.model.layers.2.mixer.experts.down_projs", down_projs),
                    (
                        "llm.model.layers.2.mixer.experts.gate_and_up_projs",
                        gate_and_up_projs,
                    ),
                ]
            )
        )

        assert [name for name, _ in mapped] == [
            "backbone.layers.1.mixer.in_proj.weight",
            "backbone.layers.2.mixer.experts.0.down_proj.weight",
            "backbone.layers.2.mixer.experts.1.down_proj.weight",
            "backbone.layers.2.mixer.experts.0.up_proj.weight",
            "backbone.layers.2.mixer.experts.1.up_proj.weight",
        ]
        assert mapped[0][1] is ordinary
        assert torch.equal(mapped[1][1], down_projs[0].t())
        assert torch.equal(mapped[2][1], down_projs[1].t())
        assert torch.equal(mapped[3][1], gate_and_up_projs[0].t())
        assert torch.equal(mapped[4][1], gate_and_up_projs[1].t())


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

    @staticmethod
    def _make_pe_mount_modules(torch, encoder_d_model=1024, pe_d_model=2048):
        class _Encoder(torch.nn.Module):
            def __init__(self, d_model, feat_in=None):
                super().__init__()
                self.d_model = d_model
                self._feat_in = feat_in
                self.weight = torch.nn.Parameter(torch.ones(1))

        perception = torch.nn.Module()
        perception.encoder = _Encoder(encoder_d_model)
        perception.preprocessor = SimpleNamespace(featurizer=SimpleNamespace(normalize="per_feature"))
        perception.proj = torch.nn.Linear(pe_d_model, 4096)
        perception.cfg = {
            "preprocessor": {"features": 128},
            "modality_adapter": {"d_model": pe_d_model},
        }
        pe_encoder = _Encoder(pe_d_model, feat_in=128)
        return perception, pe_encoder

    @staticmethod
    def _make_pe_processing_model(torch, *, fail_forward=False):
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.online_inference_enabled = False
                self.context_entries = 0
                self.context_exits = 0

            @contextlib.contextmanager
            def online_inference(self):
                self.context_entries += 1
                self.online_inference_enabled = True
                try:
                    yield
                finally:
                    self.online_inference_enabled = False
                    self.context_exits += 1

        class _Perception(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))
                self.encoder = _Encoder()
                self.online_enabled_during_forward = None

            def forward(self, input_signal, input_signal_length):
                self.online_enabled_during_forward = self.encoder.online_inference_enabled
                if fail_forward:
                    raise RuntimeError("synthetic PE forward failure")
                batch_size = input_signal.shape[0]
                return torch.ones(batch_size, 3, 4), torch.full((batch_size,), 3, dtype=torch.long)

        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.perception = _Perception()
        model._uses_pe_encoder = True
        return model

    def test_independent_speaker_encoder_mount_reconstructs_dual_encoder(self, monkeypatch, tmp_path):
        import torch

        from nemo.collections.speechlm2.modules.perception import IdentityConnector, IndependentDualEncoder
        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_independent_speaker_encoder

        class _Encoder(torch.nn.Module):
            supports_sequence_packed_output = True

            def __init__(self, width):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self._feat_in = 4
                self._feat_out = width
                self.subsampling_factor = 2

            def forward_sequence_packed(self, audio_signal, length, **kwargs):
                return audio_signal

        artifact = tmp_path / "speaker"
        artifact.mkdir()
        (artifact / "model_config.yaml").write_text("{}")
        (artifact / "model.safetensors").touch()

        asr = _Encoder(3)
        speaker = _Encoder(5)
        perception = torch.nn.Module()
        perception.encoder = asr
        perception.modality_adapter = IdentityConnector()
        perception.rote = None
        perception.preprocessor = SimpleNamespace(featurizer=SimpleNamespace(hop_length=160, sample_rate=16000))
        perception.proj = torch.nn.Linear(3, 7)
        perception.from_config_dict = lambda config: speaker

        monkeypatch.setattr("safetensors.torch.load_file", lambda *args, **kwargs: speaker.state_dict())

        mounted = _maybe_mount_independent_speaker_encoder(
            perception,
            {
                "path": str(artifact),
                "frozen": True,
                "chunk_size_seconds": 120.0,
                "asr_chunk_size_seconds": None,
            },
        )

        assert mounted is True
        assert isinstance(perception.encoder, IndependentDualEncoder)
        assert perception.encoder.asr_encoder is asr
        assert perception.encoder.auxiliary_encoder is speaker
        assert perception.encoder.d_model == 8
        assert perception.encoder.asr_chunk_size_seconds is None
        assert perception.encoder.auxiliary_chunk_size_seconds == 120.0
        assert perception.encoder.freeze_auxiliary is True
        assert all(not parameter.requires_grad for parameter in speaker.parameters())
        assert perception.proj.in_features == 8
        assert perception.proj.out_features == 7

    def test_independent_speaker_encoder_mount_fails_closed_on_global_chunking(self):
        import torch

        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_independent_speaker_encoder

        perception = torch.nn.Module()
        perception.encoder = torch.nn.Identity()
        with pytest.raises(ValueError, match="encoder_chunk_size_seconds=null"):
            _maybe_mount_independent_speaker_encoder(
                perception,
                {"path": "/models/speaker-transformer"},
                encoder_chunk_size_seconds=30.0,
            )

    def test_pe_processing_enters_and_exits_online_inference(self):
        import torch

        model = self._make_pe_processing_model(torch)
        audio_input = SimpleNamespace(
            audio_signal=torch.ones(1, 16),
            audio_signal_length=torch.tensor([16]),
        )

        result = model._process_audio(audio_input)

        assert len(result) == 1
        assert result[0].shape == (3, 4)
        assert model.perception.online_enabled_during_forward is True
        assert model.perception.encoder.context_entries == 1
        assert model.perception.encoder.context_exits == 1
        assert model.perception.encoder.online_inference_enabled is False

    def test_pe_processing_error_does_not_leak_online_inference_state(self):
        import torch

        model = self._make_pe_processing_model(torch, fail_forward=True)
        audio_input = SimpleNamespace(
            audio_signal=torch.ones(1, 16),
            audio_signal_length=torch.tensor([16]),
        )

        with pytest.raises(RuntimeError, match="synthetic PE forward failure"):
            model._process_audio(audio_input)

        assert model.perception.online_enabled_during_forward is True
        assert model.perception.encoder.context_entries == 1
        assert model.perception.encoder.context_exits == 1
        assert model.perception.encoder.online_inference_enabled is False

    def test_pe_perception_weight_loading_requires_exact_architecture(self):
        import torch

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        class _Perception(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(2, 2)

        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.perception = _Perception()
        model._uses_pe_encoder = True

        with pytest.raises(RuntimeError, match="does not exactly match"):
            model._load_perception_weights({})

        model._uses_pe_encoder = False
        assert model._load_perception_weights({}) == set()

    def test_pe_mount_allows_replacing_canary_encoder_with_wider_pee(self, monkeypatch):
        import torch

        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_pe_encoder

        perception, pe_encoder = self._make_pe_mount_modules(torch)
        monkeypatch.setattr(
            ParallelExpertEncoderPT,
            "load_from_nemo",
            lambda *args, **kwargs: pe_encoder,
        )

        assert _maybe_mount_pe_encoder(perception, "nvidia/ParallelExpertEncoder") is True
        assert perception.encoder is pe_encoder
        assert perception.preprocessor.featurizer.normalize is None

    def test_pe_mount_constructs_canonical_encoder_from_inline_config(self, monkeypatch):
        import torch

        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_pe_encoder

        perception, pe_encoder = self._make_pe_mount_modules(torch)
        observed = {}

        def from_inline_config(config, *, map_location):
            observed["config"] = config
            observed["map_location"] = map_location
            return pe_encoder

        monkeypatch.setattr(ParallelExpertEncoderPT, "from_inline_config", from_inline_config)
        config = {
            "asr_encoder_cfg": {"_target_": "example.TransformerEncoder"},
            "diarization_model_cfg": {"target": "example.Sortformer"},
        }

        assert _maybe_mount_pe_encoder(perception, None, config) is True
        assert perception.encoder is pe_encoder
        assert observed == {"config": config, "map_location": "cpu"}

    def test_pe_mount_rejects_ambiguous_or_unknown_inline_config(self):
        import torch

        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_pe_encoder

        perception, _ = self._make_pe_mount_modules(torch)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _maybe_mount_pe_encoder(
                perception,
                "/tmp/pee.nemo",
                {"asr_encoder_cfg": {}, "diarization_model_cfg": {}},
            )
        with pytest.raises(ValueError, match="missing required config sections"):
            _maybe_mount_pe_encoder(perception, None, {"unknown": "schema"})

    @pytest.mark.parametrize(
        "mismatch, expected_error",
        [
            ("mel", "expects 128 mel bins"),
            ("adapter", "modality_adapter.d_model=1024"),
            ("projection", "proj.in_features=1024"),
        ],
    )
    def test_pe_mount_rejects_unchanged_component_dimension_mismatches(self, monkeypatch, mismatch, expected_error):
        import torch

        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
        from nemo.collections.speechlm2.vllm.salm.audio import _maybe_mount_pe_encoder

        perception, pe_encoder = self._make_pe_mount_modules(torch)
        if mismatch == "mel":
            perception.cfg["preprocessor"]["features"] = 80
        elif mismatch == "adapter":
            perception.cfg["modality_adapter"]["d_model"] = 1024
        else:
            perception.proj = torch.nn.Linear(1024, 4096)
        monkeypatch.setattr(
            ParallelExpertEncoderPT,
            "load_from_nemo",
            lambda *args, **kwargs: pe_encoder,
        )

        with pytest.raises(ValueError, match=expected_error):
            _maybe_mount_pe_encoder(perception, "nvidia/ParallelExpertEncoder")

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

    def test_pe_processing_ignores_exported_generic_estimator_chunking(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        exported_estimator = {
            "chunk_size_seconds": 15.0,
            "preprocessor": {
                "n_fft": 512,
                "hop_length": 160,
                "stft_pad_amount": 256,
            },
            "subsampling": {
                "type": "conv",
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "repeat": 3,
                "ceil_mode": False,
            },
        }
        info = object.__new__(NeMoSpeechLMProcessingInfo)
        monkeypatch.setattr(
            info,
            "get_hf_config",
            lambda: SimpleNamespace(
                pe_encoder_path="/models/ParallelExpertEncoder.nemo",
                pe_encoder_config=None,
                encoder_chunk_size_seconds=15.0,
                audio_token_estimator=exported_estimator,
            ),
        )

        estimator_config = info._get_audio_token_estimator_config()

        assert info._get_encoder_chunk_size_seconds() is None
        assert estimator_config["chunk_size_seconds"] is None
        assert exported_estimator["chunk_size_seconds"] == 15.0
        assert info._estimate_audio_tokens(559_280, None, estimator_config) == 437
        assert info._estimate_audio_tokens(559_280, None, exported_estimator) == 438

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

    def test_dummy_inputs_use_requested_audio_length(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMDummyInputsBuilder

        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=0,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=12345)},
        )

        assert result["audio"][0].length == 12345

    def test_dummy_inputs_cap_requested_audio_length_to_text_budget(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            _DUMMY_AUDIO_TEXT_TOKEN_RESERVE,
            NeMoSpeechLMDummyInputsBuilder,
            NeMoSpeechLMProcessingInfo,
        )

        target_audio_tokens = 4
        max_audio_len = NeMoSpeechLMProcessingInfo._samples_for_audio_tokens(target_audio_tokens)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=_DUMMY_AUDIO_TEXT_TOKEN_RESERVE + target_audio_tokens,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=max_audio_len + 16000)},
        )

        assert result["audio"][0].length == max_audio_len

    def test_dummy_inputs_large_seq_len_uses_max_audio_cap(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm.audio import (
            _DUMMY_AUDIO_MAX_DURATION_S,
            _SAMPLING_RATE,
            NeMoSpeechLMDummyInputsBuilder,
        )

        max_audio_len = int(_DUMMY_AUDIO_MAX_DURATION_S * _SAMPLING_RATE)
        builder = object.__new__(NeMoSpeechLMDummyInputsBuilder)
        builder.info = SimpleNamespace(
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        monkeypatch.setattr(
            builder,
            "_get_dummy_audios",
            lambda length, num_audios: [SimpleNamespace(length=length) for _ in range(num_audios)],
        )

        result = builder.get_dummy_mm_data(
            seq_len=10_000_000,
            mm_counts={"audio": 1},
            mm_options={"audio": SimpleNamespace(length=max_audio_len + 16000)},
        )

        assert result["audio"][0].length == max_audio_len

    def test_call_hf_processor_requires_matching_placeholder_count(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_FakeTokenizer,
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: (2),
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
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
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: (2),
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
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
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
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
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert "NeMoSpeechLMForConditionalGeneration" in ModelRegistry.get_supported_archs()
        assert NeMoSpeechLMForConditionalGeneration is not None

    def test_register_model_config_hook(self, monkeypatch):
        """The outer Speech architecture must delegate Nemotron-H cache defaults."""
        from transformers import AutoConfig
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP

        from nemo.collections.speechlm2.vllm.salm import register
        from nemo.collections.speechlm2.vllm.salm.config_hook import NeMoSpeechLMForConditionalGenerationConfig

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        assert MODELS_CONFIG_MAP["NeMoSpeechLMForConditionalGeneration"] is NeMoSpeechLMForConditionalGenerationConfig

    @pytest.mark.parametrize(
        ("is_hybrid", "backbone_dtype", "initial_dtype", "expected_dtype"),
        [
            (True, None, "auto", "float32"),
            (True, "float16", "auto", "float16"),
            (True, "float32", "float16", "float16"),
            (False, None, "auto", "auto"),
        ],
    )
    def test_model_config_hook_sets_only_hybrid_auto_cache_dtype(
        self, is_hybrid, backbone_dtype, initial_dtype, expected_dtype
    ):
        from nemo.collections.speechlm2.vllm.salm.config_hook import NeMoSpeechLMForConditionalGenerationConfig

        text_config = SimpleNamespace()
        if backbone_dtype is not None:
            text_config.mamba_ssm_cache_dtype = backbone_dtype
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(mamba_ssm_cache_dtype=initial_dtype),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(is_hybrid=is_hybrid, text_config=text_config)),
        )

        NeMoSpeechLMForConditionalGenerationConfig.verify_and_update_config(vllm_config)

        assert vllm_config.cache_config.mamba_ssm_cache_dtype == expected_dtype

    def test_register_does_not_patch_fast_tokenizer(self, monkeypatch):
        from transformers import AutoConfig, PreTrainedTokenizerFast

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
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


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class _FakeTokenizer:
    def __init__(self):
        self.added_special_tokens = None

    def get_vocab(self):
        return {}

    def add_special_tokens(self, tokens):
        self.added_special_tokens = tokens

    def encode(self, prompt, add_special_tokens=True):
        return list(range(max(1, len(prompt.split()))))
