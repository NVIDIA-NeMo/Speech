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
from huggingface_hub.errors import StrictDataclassClassValidationError

try:
    import nemo.collections.speechlm2.vllm.salm as _salm_module
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


def _identity_hf_config_override(hf_config):
    """Pickleable target override for vLLM's composed-override API."""
    return hf_config


def test_full_stack_asr_exports_grouped_encoder_dependencies():
    repo_root = Path(__file__).parents[3]
    code = """
from omegaconf import OmegaConf
from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
from nemo.collections.asr.modules import (
    ConvASRDecoder,
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


@pytest.mark.skipif(not _HAS_CONFIG, reason="SpeechLM vLLM plugin not available")
@pytest.mark.parametrize(
    "architecture",
    ("Qwen3DFlash2DraftModel", "DFlashQwen3DFlash2DraftModel"),
)
def test_normalizes_automodel_dflash2_architecture(architecture):
    """Normalization does not need vLLM installed or model weights loaded."""
    config = SimpleNamespace(architectures=[architecture])

    result = _salm_module._normalize_dflash2_architecture(config, supported_archs={"DFlash2DraftModel"})

    assert result is config
    assert config.architectures == ["DFlash2DraftModel"]


@pytest.mark.skipif(not _HAS_CONFIG, reason="SpeechLM vLLM plugin not available")
@pytest.mark.parametrize(
    "architectures",
    (None, [], ["DFlash2DraftModel"], ["Qwen3DFlashDraftModel"], ["Qwen3DFlash2DraftModel", "Other"]),
)
def test_dflash2_normalization_preserves_unrelated_architectures(architectures):
    config = SimpleNamespace(architectures=architectures)

    _salm_module._normalize_dflash2_architecture(config, supported_archs={"DFlash2DraftModel"})

    assert config.architectures == architectures


@pytest.mark.skipif(not _HAS_CONFIG, reason="SpeechLM vLLM plugin not available")
@pytest.mark.parametrize(
    "supported_archs",
    (
        set(),
        {"Qwen3DFlash2DraftModel"},
        {"Qwen3DFlash2DraftModel", "DFlash2DraftModel"},
    ),
)
def test_dflash2_normalization_requires_canonical_only_runtime(supported_archs):
    config = SimpleNamespace(architectures=["Qwen3DFlash2DraftModel"])

    _salm_module._normalize_dflash2_architecture(config, supported_archs=supported_archs)

    assert config.architectures == ["Qwen3DFlash2DraftModel"]


@pytest.mark.skipif(not _HAS_CONFIG, reason="SpeechLM vLLM plugin not available")
def test_dflash_alias_registration_supports_class_backed_registry():
    class NativeModel:
        pass

    class FakeRegistry:
        models = {"DFlashDraftModel": SimpleNamespace(model_cls=NativeModel)}

        @classmethod
        def get_supported_archs(cls):
            return list(cls.models)

        @classmethod
        def register_model(cls, architecture, model_ref):
            cls.models[architecture] = model_ref

    _salm_module._register_model_aliases(
        FakeRegistry,
        "DFlashDraftModel",
        ("AutomodelDraft",),
    )

    assert FakeRegistry.models["AutomodelDraft"] is NativeModel


@pytest.mark.skipif(not _HAS_CONFIG, reason="SpeechLM vLLM plugin not available")
@pytest.mark.parametrize(
    ("model_cfg", "backbone_architectures", "expected"),
    [
        pytest.param(
            {"pe_encoder_path": "/models/pee.nemo"},
            ["NemotronHForCausalLM"],
            "Nemotron3_5TranscribeForConditionalGeneration",
            id="training-config",
        ),
        pytest.param(
            {"pe_encoder_path": None, "pe_encoder_config": {"target": "ParallelExpertEncoderPT"}},
            ["NemotronHForCausalLM"],
            "Nemotron3_5TranscribeForConditionalGeneration",
            id="exported-config",
        ),
        pytest.param({}, ["NemotronHForCausalLM"], "NeMoSpeechLMForConditionalGeneration", id="no-pe-encoder"),
        pytest.param(
            {"pe_encoder_path": "/models/pee.nemo"},
            ["Qwen3ForCausalLM"],
            "NeMoSpeechLMForConditionalGeneration",
            id="transformer-backbone",
        ),
    ],
)
def test_checkpoint_architecture_names_transcribe_by_structure(model_cfg, backbone_architectures, expected):
    from nemo.collections.speechlm2.vllm.salm.config import checkpoint_architecture

    assert checkpoint_architecture(model_cfg, backbone_architectures) == expected


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
        assert cfg.image_token_index is None
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

    def test_derives_multibranch_mtp_contract_from_backbone(self, monkeypatch):
        """Legacy compute_mtp exports should derive the exact *E physical head."""

        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            return SimpleNamespace(
                architectures=["NemotronHForCausalLM"],
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
                num_key_value_heads=2,
                layer_norm_epsilon=1e-5,
                num_nextn_predict_layers=1,
                mtp_hybrid_override_pattern=None,
                mtp_layers_block_type=["attention", "moe"],
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, compute_mtp=True, mtp=None)

        assert cfg.mtp == {
            "enabled": True,
            "num_nextn_predict_layers": 1,
            "use_repeated_layer": False,
            "hybrid_override_pattern": "*E",
        }
        assert cfg.mtp_hybrid_override_pattern == "*E"

    def test_compute_mtp_false_does_not_enable_backbone_head(self, monkeypatch):
        """A backbone MTP head must not opt an export into speculative decoding."""

        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            return SimpleNamespace(
                architectures=["NemotronHForCausalLM"],
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
                num_key_value_heads=2,
                layer_norm_epsilon=1e-5,
                num_nextn_predict_layers=1,
                mtp_layers_block_type=["attention", "moe"],
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, compute_mtp=False)

        assert cfg.mtp is None

    def test_unsupported_mtp_topology_fails_closed(self, monkeypatch):
        """vLLM 0.23 cannot instantiate Mamba or MLP MTP sublayers."""

        def from_pretrained(model_name: str, trust_remote_code: bool = True):
            return SimpleNamespace(
                architectures=["NemotronHForCausalLM"],
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
                num_key_value_heads=2,
                layer_norm_epsilon=1e-5,
                num_nextn_predict_layers=1,
                mtp_layers_block_type=["mamba", "moe"],
            )

        monkeypatch.setattr(_config_module.AutoConfig, "from_pretrained", from_pretrained)

        with pytest.raises(ValueError, match="does not support block type 'mamba'"):
            NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, compute_mtp=True)

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

    @pytest.mark.parametrize("audio_token_id", [None, 0, 151935, 151936, 151945])
    def test_audio_token_id_reaches_target_and_draft_configs(self, audio_token_id):
        cfg = NeMoSpeechLMConfig(
            **{**_DEFAULT_CONFIG_KWARGS, "pretrained_llm": "Qwen/Qwen3-1.7B"},
            audio_token_id=audio_token_id,
            speechlm_runtime_added_token_ids=[] if audio_token_id is None else [audio_token_id],
        )
        assert cfg.speechlm_runtime_added_token_ids == cfg.get_text_config().speechlm_runtime_added_token_ids
        assert cfg.audio_token_id == audio_token_id
        assert cfg.get_text_config().audio_token_id == audio_token_id
        assert cfg.speechlm_output_vocab_size == 151936

    def test_audio_token_id_survives_config_serialization(self, tmp_path):
        cfg = NeMoSpeechLMConfig(
            **_DEFAULT_CONFIG_KWARGS,
            audio_token_id=4,
            speechlm_runtime_added_token_ids=[4, 5],
            llm_config={
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": 100,
            },
        )
        assert cfg.layer_types == ["attention"] * cfg.num_hidden_layers
        assert cfg.get_text_config().layer_types == ["attention"] * cfg.num_hidden_layers
        cfg.save_pretrained(tmp_path)
        assert cfg.layer_types == ["attention"] * cfg.num_hidden_layers
        restored = NeMoSpeechLMConfig.from_pretrained(tmp_path)
        assert restored.audio_token_id == restored.get_text_config().audio_token_id == 4
        assert restored.speechlm_runtime_added_token_ids == [4, 5]
        assert restored.get_text_config().speechlm_runtime_added_token_ids == [4, 5]
        assert restored.speechlm_output_vocab_size == 100
        assert restored.image_token_index == 100
        assert restored.layer_types == cfg.layer_types
        assert restored.get_text_config().layer_types == cfg.get_text_config().layer_types

    @pytest.mark.parametrize(
        "field, layers, error",
        [
            ("layer_types", ["invalid"] * 2, "entries must be in"),
            ("layer_types", ["attention"], "num_hidden_layers"),
            ("mlp_layer_types", ["invalid"] * 2, "entries must be in"),
            ("layer_types", 2, "not iterable"),
        ],
    )
    def test_config_serialization_rejects_invalid_layer_metadata(self, tmp_path, field, layers, error):
        cfg = NeMoSpeechLMConfig(
            **_DEFAULT_CONFIG_KWARGS,
            llm_config={
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "num_hidden_layers": 2,
            },
        )
        # Inject malformed metadata to exercise save-time validation directly.
        vars(cfg.text_config)[field] = layers
        with pytest.raises(StrictDataclassClassValidationError, match=error):
            cfg.save_pretrained(tmp_path)
        assert getattr(cfg, field) == layers
        assert getattr(cfg.text_config, field) == layers

    @pytest.mark.parametrize("audio_token_id", [True, -1, 151946, "1"])
    def test_invalid_audio_token_id_rejected(self, audio_token_id):
        with pytest.raises(ValueError, match="audio_token_id"):
            NeMoSpeechLMConfig(
                **{**_DEFAULT_CONFIG_KWARGS, "pretrained_llm": "Qwen/Qwen3-1.7B"},
                audio_token_id=audio_token_id,
            )

    @pytest.mark.parametrize("token_ids", [[True], [-1], [151946], ["1"], "1"])
    def test_invalid_runtime_added_token_ids_rejected(self, token_ids):
        with pytest.raises(ValueError, match="speechlm_runtime_added_token_ids"):
            NeMoSpeechLMConfig(
                **{**_DEFAULT_CONFIG_KWARGS, "pretrained_llm": "Qwen/Qwen3-1.7B"},
                speechlm_runtime_added_token_ids=token_ids,
            )

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

    @pytest.mark.parametrize(
        ("source_name", "hf_name"),
        [
            ("llm.model.embed_tokens.weight", "backbone.embed_tokens.weight"),
            # ModelOpt exports already use NemotronH's own embedding name.
            ("backbone.embeddings.weight", "backbone.embeddings.weight"),
            ("llm.lm_head.weight", "lm_head.weight"),
        ],
    )
    def test_pads_vocabulary_tensors_to_served_vocab(self, source_name, hf_name):
        import torch

        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend

        backend = HybridBackend(SimpleNamespace(text_config=SimpleNamespace(vocab_size=6)))
        tensor = torch.ones(4, 3)
        [(mapped_name, mapped_tensor)] = backend.nemo_to_hf_llm_weights([(source_name, tensor)])

        assert mapped_name == hf_name
        assert mapped_tensor.shape == (6, 3)
        assert torch.equal(mapped_tensor[:4], tensor)
        assert not mapped_tensor[4:].any()

    @staticmethod
    def _lora_weights():
        import torch

        prefix = "llm.model.layers.0.mixer.in_proj"
        w = torch.randn(4, 3)
        a = torch.randn(2, 3)
        b = torch.randn(4, 2)
        return (
            prefix,
            w,
            a,
            b,
            [(f"{prefix}.weight", w), (f"{prefix}.lora_A.weight", a), (f"{prefix}.lora_B.weight", b)],
        )

    @pytest.mark.parametrize(
        ("lora_cfg", "scaling"),
        [({"dim": 2, "alpha": 4}, 2.0), ({"r": 2, "lora_alpha": 1}, 0.5)],
    )
    def test_preprocess_merges_lora_adapters(self, lora_cfg, scaling):
        import torch

        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend

        backend = HybridBackend(SimpleNamespace(text_config=SimpleNamespace(vocab_size=None), lora=lora_cfg))
        prefix, w, a, b, weights = self._lora_weights()
        out = dict(backend.preprocess_llm_weights(weights))
        assert set(out) == {f"{prefix}.weight"}
        torch.testing.assert_close(out[f"{prefix}.weight"], w + scaling * (b @ a))

    def test_preprocess_rejects_unrecognized_lora_config(self):
        from nemo.collections.speechlm2.vllm.salm.backends import HybridBackend

        backend = HybridBackend(SimpleNamespace(text_config=SimpleNamespace(vocab_size=None), lora={"rank": 2}))
        with pytest.raises(ValueError, match="rank/alpha"):
            backend.preprocess_llm_weights(self._lora_weights()[-1])

    def test_preprocess_passes_through_weights_without_lora(self, backend):
        import torch

        weights = [("llm.model.layers.0.mixer.in_proj.weight", torch.randn(4, 3))]
        assert backend.preprocess_llm_weights(weights) == weights

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

    def test_pe_processing_uses_encoder_graphs_and_falls_back_to_eager(self):
        import torch

        model = self._make_pe_processing_model(torch)
        audio_input = SimpleNamespace(audio_signal=torch.ones(2, 16), audio_signal_length=torch.tensor([16, 8]))
        graph_calls = []

        def graphs(audio_signal, audio_lengths):
            graph_calls.append(audio_lengths.tolist())
            return [torch.zeros(5, 4), torch.zeros(2, 4)]

        model._encoder_graphs = graphs
        result = model._process_audio(audio_input)
        assert graph_calls == [[16, 8]]
        assert [r.shape for r in result] == [(5, 4), (2, 4)]
        assert model.perception.online_enabled_during_forward is None  # eager forward did not run

        model._encoder_graphs = lambda audio_signal, audio_lengths: None  # e.g. audio too long to graph
        result = model._process_audio(audio_input)
        assert [r.shape for r in result] == [(3, 4), (3, 4)]
        assert model.perception.online_enabled_during_forward is True

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
        model._perception_weights_initialized = False

        with pytest.raises(RuntimeError, match="does not exactly match"):
            model._load_perception_weights({})

        assert not model._perception_weights_initialized
        seed = {key: value.clone() for key, value in model.perception.state_dict().items()}
        model._load_perception_weights(seed)
        assert model._perception_weights_initialized
        before_bias = model.perception.encoder.bias.detach().clone()
        update = torch.full_like(model.perception.encoder.weight, 0.5)
        assert model._load_perception_weights({"encoder.weight": update}) == {"perception.encoder.weight"}
        torch.testing.assert_close(model.perception.encoder.weight, update)
        torch.testing.assert_close(model.perception.encoder.bias, before_bias)
        assert model._load_perception_weights({}) == set()
        with pytest.raises(RuntimeError, match="unexpected"):
            model._load_perception_weights({"wrong.weight": update})

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
            processor._call_hf_processor_with_prompt(
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

        result = processor._call_hf_processor_with_prompt(
            prompt="Transcribe: <|audio|>",
            mm_data={"audios": [[0.0] * 12345]},
            mm_kwargs={},
            tok_kwargs={},
        )

        assert len(result["audio_signal"]) == 1
        assert result["audio_signal"][0].shape[-1] == 12345
        assert torch.equal(result["audio_signal_length"], torch.tensor([12345]))

    @pytest.mark.parametrize("capture", [False, True])
    def test_call_hf_processor_audio_only_packages_audio_without_prompt(self, capture):
        """vLLM 0.32 passes only the audio; the processor must not tokenize or expand any prompt."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(get_tokenizer=_FakeTokenizer)

        result = processor._call_hf_processor_audio_only(
            hf_data={"audio": [[0.0] * 12345, torch.zeros(1, 800)]},
            hf_kwargs={"capture_ctc_timestamps": capture},
        )

        assert "input_ids" not in result
        assert [audio.shape for audio in result["audio_signal"]] == [torch.Size([12345]), torch.Size([800])]
        assert torch.equal(result["audio_signal_length"], torch.tensor([12345, 800]))
        assert torch.equal(result["capture_ctc_timestamps"], torch.tensor([capture, capture]))

    def test_call_hf_processor_audio_only_without_audio_is_empty(self):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)

        result = processor._call_hf_processor_audio_only(hf_data={}, hf_kwargs={})

        assert dict(result) == {}

    @pytest.mark.parametrize("takes_prompt", [False, True])
    def test_call_hf_processor_follows_the_installed_vllm_contract(self, monkeypatch, takes_prompt):
        from nemo.collections.speechlm2.vllm.salm import audio
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        calls = []
        monkeypatch.setattr(audio, "_PROCESSOR_TAKES_PROMPT", takes_prompt)
        monkeypatch.setattr(processor, "_call_hf_processor_with_prompt", lambda *a, **k: calls.append(("prompt", a)))
        monkeypatch.setattr(processor, "_call_hf_processor_audio_only", lambda *a, **k: calls.append(("audio", a)))

        if takes_prompt:
            processor._call_hf_processor("p", {"audios": []}, {}, {})
            assert calls == [("prompt", ("p", {"audios": []}, {}, {}))]
        else:
            processor._call_hf_processor({"audio": []}, {})
            assert calls == [("audio", ({"audio": []}, {}))]

    def test_prompt_updates_expand_the_placeholder_token_id(self, monkeypatch):
        """vLLM 0.32 applies prompt updates on token ids only."""
        from nemo.collections.speechlm2.vllm.salm import audio
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        class _PlaceholderTokenizer(_FakeTokenizer):
            def encode(self, prompt, add_special_tokens=True):
                return [42] if prompt == "<|audio|>" else [1, 2]

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_PlaceholderTokenizer,
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: samples // 100,
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        items = SimpleNamespace(get=lambda i: SimpleNamespace(shape=(300 * (i + 1),)))
        mm_items = SimpleNamespace(get_items=lambda modality, typ: items)
        monkeypatch.setattr(audio, "_PROMPT_UPDATES_TAKE_TEXT", False)

        (update,) = processor._get_prompt_updates(mm_items, {}, None)

        assert update.target == [42]
        assert update.replacement(0) == [42] * 3
        assert update.replacement(1) == [42] * 6

    def test_prompt_updates_without_uncached_audio(self, monkeypatch):
        """vLLM asks for prompt updates of the items missing from its processor cache only; with every audio item
        cached (as for the second startup profiling input) there are no audio items at all."""
        from nemo.collections.speechlm2.vllm.salm import audio
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        class _PlaceholderTokenizer(_FakeTokenizer):
            def encode(self, prompt, add_special_tokens=True):
                return [42] if prompt == "<|audio|>" else [1, 2]

        def no_items(modality, typ):
            raise KeyError(f"Modality {modality!r} not found. Available modalities: set()")

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_PlaceholderTokenizer,
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        mm_items = SimpleNamespace(get_items=no_items)
        monkeypatch.setattr(audio, "_PROMPT_UPDATES_TAKE_TEXT", False)

        (update,) = processor._get_prompt_updates(mm_items, {}, None)

        assert update.modality == "audio"
        assert update.target == [42]

    def test_prompt_updates_require_a_single_placeholder_token(self, monkeypatch):
        from nemo.collections.speechlm2.vllm.salm import audio
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        class _SplitPlaceholderTokenizer(_FakeTokenizer):
            def encode(self, prompt, add_special_tokens=True):
                return [7, 8]

        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=_SplitPlaceholderTokenizer,
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        mm_items = SimpleNamespace(get_items=lambda modality, typ: None)
        monkeypatch.setattr(audio, "_PROMPT_UPDATES_TAKE_TEXT", False)

        with pytest.raises(ValueError, match="exactly one token"):
            processor._get_prompt_updates(mm_items, {}, None)

    def test_prompt_updates_expand_the_placeholder_text_on_older_vllm(self):
        from nemo.collections.speechlm2.vllm.salm import audio
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        if not audio._PROMPT_UPDATES_TAKE_TEXT:
            pytest.skip("the installed vLLM applies prompt updates on token ids only")
        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: samples // 100,
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        items = SimpleNamespace(get=lambda i: SimpleNamespace(shape=(300,)))
        mm_items = SimpleNamespace(get_items=lambda modality, typ: items)

        (update,) = processor._get_prompt_updates(mm_items, {}, None)

        assert update.target == "<|audio|>"
        assert update.replacement(0).full == "<|audio|>" * 3

    @pytest.mark.parametrize("separable", [True, False])
    def test_call_hf_processor_expands_the_cached_template_like_direct_tokenization(self, separable):
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMMultiModalProcessor

        tokenizer = _SpecialTokenTokenizer(separable=separable)
        processor = object.__new__(NeMoSpeechLMMultiModalProcessor)
        processor.info = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            _estimate_audio_tokens=lambda samples, chunk_size_seconds=None, estimator_config=None: samples // 100,
            _get_encoder_chunk_size_seconds=lambda: None,
            _get_audio_token_estimator_config=lambda: None,
        )
        prompt = "Transcribe this: <|audio|> then <|audio|> please"

        reference = _SpecialTokenTokenizer(separable=separable)
        # The first request repeats no placeholder, so its own expansion cannot tell a nonseparable tokenizer apart.
        for lengths in ([100, 100], [300, 500], [700, 200], [100, 900]):
            result = processor._call_hf_processor_with_prompt(
                prompt=prompt, mm_data={"audios": [[0.0] * n for n in lengths]}, mm_kwargs={}, tok_kwargs={}
            )
            first, second = ("<|audio|>" * (n // 100) for n in lengths)
            expanded = f"Transcribe this: {first} then {second} please"
            assert result["input_ids"][0].tolist() == reference.encode(expanded)

        # Separable: the template and a probe expansion are tokenized once; every request expands the cached ids.
        # Otherwise every request is tokenized directly.
        assert len(tokenizer.encoded) == (2 if separable else 2 + 4)
        assert tokenizer.encoded[0] == prompt

    def test_encoder_graph_runner_splits_batches_into_buckets(self):
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.batch_sizes, runner.window = [8, 16, 32, 64, 128], 4000
        runner.max_rows_x_frames = 128 * runner.window
        assert runner._split(1, 4000) == [8]
        assert runner._split(8, 4000) == [8]
        assert runner._split(80, 4000) == [64, 16]
        assert runner._split(71, 4000) == [64, 8]
        assert runner._split(300, 4000) == [128, 128, 32, 8, 8]
        # two-window audio: graphs hold at most half the rows
        assert runner._split(80, 8000) == [64, 16]
        assert runner._split(130, 8000) == [64, 64, 8]

    def test_encoder_graph_runner_defaults_cover_low_concurrency(self):
        """One request replays a one-row graph, not a padded batch."""
        import inspect

        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.batch_sizes = list(inspect.signature(EncoderGraphRunner).parameters["batch_sizes"].default)
        runner.window = 4000
        runner.max_rows_x_frames = runner.batch_sizes[-1] * runner.window
        assert runner.batch_sizes[:3] == [1, 2, 4]
        assert runner._split(1, 4000) == [1]
        assert runner._split(3, 4000) == [4]
        assert runner._split(4, 4000) == [4]
        assert runner._split(5, 4000) == [8]
        # Rows past the large graphs are padded into one small graph, not split over several.
        assert runner._split(13, 4000) == [8, 8]
        assert runner._split(12, 4000) == [8, 4]
        assert runner._split(37, 4000) == [32, 8]

    def test_encoder_graph_capture_raises_the_recompile_limit_only_while_capturing(self):
        """Capturing every bucket compiles more attention variants than dynamo's default limit allows."""
        import torch

        from nemo.collections.speechlm2.vllm.salm import encoder_graphs
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        config = torch._dynamo.config
        name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
        before = getattr(config, name)
        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.perception = SimpleNamespace(
            maybe_preprocess_audio=lambda audio, lengths, *_: (audio[:, None], lengths)
        )
        runner.max_frames, runner.graphs, runner.enabled, runner.ctc_timestamp_inputs = 800, {}, True, False
        seen = []

        def capture_all(n_feat, dtype):
            seen.append(getattr(config, name))
            raise RuntimeError("stop after the limit is checked")

        runner._capture_all = capture_all
        assert runner(torch.zeros(1, 300), torch.tensor([300])) is None
        assert seen == [max(before, encoder_graphs._CAPTURE_RECOMPILE_LIMIT)]
        assert getattr(config, name) == before

    def test_shared_graph_buffers_hand_out_views_of_one_buffer_per_slot(self):
        import torch

        from nemo.collections.common.parts.optional_cuda_graphs import SharedGraphBuffers

        buffers = SharedGraphBuffers()
        big = buffers.view("x", (4, 3, 5), torch.float32, "cpu")
        small = buffers.view("x", (2, 3, 4), torch.float32, "cpu")
        other = buffers.view("y", (2, 3, 4), torch.float32, "cpu")
        assert small.is_contiguous() and small.data_ptr() == big.data_ptr()
        assert other.data_ptr() != big.data_ptr()
        # A larger request gets a new buffer; the earlier views keep the old one.
        bigger = buffers.view("x", (8, 3, 5), torch.float32, "cpu")
        assert bigger.data_ptr() != big.data_ptr() and big.numel() == 60

    @pytest.mark.parametrize(
        "frames,expected", [(8001, 8400), (8400, 8400), (8401, 8800), (11999, 12000), (12000, 12000)]
    )
    def test_long_audio_pads_only_the_last_window_to_its_bucket(self, frames, expected):
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.window, runner.last_window_bucket = 4000, 400
        assert runner._long_frames(frames) == expected

    def test_long_audio_steps_pad_rows_by_at_most_half(self):
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.step_batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 32, 48, 64, 96, 128]
        for rows in range(1, 129):
            size = runner._bucket(rows)
            assert rows <= size <= max(16, 1.5 * rows)
        assert runner._bucket(129) is None

    def test_encoder_graph_options_come_from_the_model_config(self):
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import encoder_graph_options

        assert encoder_graph_options(SimpleNamespace(encoder_cuda_graphs=True)) == {}
        config = SimpleNamespace(
            encoder_cuda_graph_batch_sizes=[1, 4, 32],
            encoder_cuda_graph_frame_bucket=128,
            encoder_cuda_graph_max_windows=1,
            encoder_cuda_graph_long_audio=False,
            encoder_cuda_graph_last_window_bucket=500,
        )
        assert encoder_graph_options(config) == {
            "batch_sizes": (1, 4, 32),
            "frame_bucket": 128,
            "max_windows": 1,
            "long_audio": False,
            "last_window_bucket": 500,
        }

    @pytest.mark.parametrize(
        "kwargs", [{"batch_sizes": ()}, {"batch_sizes": (0, 8)}, {"frame_bucket": 0}, {"max_windows": 0}]
    )
    def test_encoder_graph_runner_rejects_invalid_buckets(self, kwargs):
        import torch

        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        with pytest.raises(ValueError):
            EncoderGraphRunner(torch.nn.Module(), **kwargs)

    @pytest.mark.parametrize("window,frame_bucket,max_windows", [(4000, 256, 1), (4000, 256, 2), (1024, 128, 2)])
    def test_encoder_graph_runner_captures_every_reachable_bucket(self, window, frame_bucket, max_windows):
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.window, runner.frame_bucket, runner.max_frames = window, frame_bucket, window * max_windows
        runner.batch_sizes, runner.max_rows_x_frames = [8, 16], 16 * window
        captured = set()
        runner._capture = lambda rows, frames, n_feat, dtype: captured.add((rows, frames))
        runner._capture_all(n_feat=128, dtype=None)
        for frames in range(1, runner.max_frames + 1):
            bucket = runner._frame_bucket(frames)
            assert bucket >= frames
            assert {(rows, bucket) for rows in runner._split(17, bucket)} <= captured

    def test_encoder_graph_runner_falls_back_to_eager_when_capture_fails(self):
        """A failed capture, even after some buckets were captured, makes this and every later call return None."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        runner = object.__new__(EncoderGraphRunner)
        runner.long_audio = False
        runner.perception = SimpleNamespace(
            maybe_preprocess_audio=lambda audio, lengths, *_: (audio[:, None], lengths)
        )
        runner.window, runner.frame_bucket, runner.max_frames = 400, 128, 800
        runner.batch_sizes, runner.max_rows_x_frames = [8], 8 * 400
        runner.graphs, runner.pool, runner.enabled = {}, object(), True
        attempts = []

        def capture(rows, frames, n_feat, dtype):
            attempts.append((rows, frames))
            if len(attempts) > 1:
                raise RuntimeError("operation not permitted when stream is capturing")
            runner.graphs[rows, frames] = None

        runner._capture = capture
        audio, lengths = torch.zeros(2, 300), torch.tensor([300, 200])
        assert runner(audio, lengths) is None
        assert runner(audio, lengths) is None
        assert len(attempts) == 2 and runner.graphs == {} and runner.pool is None and not runner.enabled

    def test_encoder_graph_runner_failed_capture_leaves_eager_inference_working(self, monkeypatch):
        """A capture that syncs the host fails; the eager forward still runs on the same device afterwards."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        torch.manual_seed(0)
        perception = _toy_perception().cuda().eval()
        runner = EncoderGraphRunner(perception, batch_sizes=(4,), frame_bucket=128, max_windows=1)
        forward = runner._forward
        monkeypatch.setattr(runner, "_forward", lambda x, x_len: (x_len.max().item(), forward(x, x_len))[1])
        audio = torch.randn(2, 1500, device="cuda")
        lengths = torch.tensor([1500, 900], device="cuda")
        assert runner(audio, lengths) is None
        assert not runner.enabled and runner.graphs == {}
        with torch.no_grad(), perception.encoder.online_inference():
            emb, emb_len = perception(input_signal=audio, input_signal_length=lengths)
        assert torch.isfinite(emb).all() and emb_len.tolist()[0] > 0

    def test_encoder_graph_runner_matches_eager_forward(self):
        """Graph replays over padded buckets return the eager forward's per-row embeddings."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from nemo.collections.speechlm2.vllm.salm.encoder_graphs import EncoderGraphRunner

        torch.manual_seed(0)
        perception = _toy_perception().cuda().eval()
        runner = EncoderGraphRunner(perception, batch_sizes=(4, 8, 16), frame_bucket=128, max_windows=2)
        for batch, max_len in ((3, 150), (13, 600), (21, 1000)):
            lengths = torch.randint(200, max_len * 10 + 1, (batch,), device="cuda")
            lengths[0] = max_len * 10
            audio = torch.randn(batch, max_len * 10, device="cuda")
            got = runner(audio, lengths)
            with torch.no_grad():
                emb, emb_len = perception(input_signal=audio, input_signal_length=lengths)
            assert len(got) == batch
            for i in range(batch):
                torch.testing.assert_close(got[i], emb[i, : emb_len[i]])
        # longer than max_windows windows: left to the eager path
        assert runner(torch.randn(1, 30000, device="cuda"), torch.tensor([30000], device="cuda")) is None
        assert {b for b, _ in runner.graphs} <= {4, 8, 16}

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

    def test_register_transcribe_architecture(self, monkeypatch):
        """The dedicated PE-encoder/NemotronH architecture name resolves to the SpeechLM class and config hook."""
        from transformers import AutoConfig
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP
        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm import register
        from nemo.collections.speechlm2.vllm.salm.config_hook import NeMoSpeechLMForConditionalGenerationConfig

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        speechlm = ModelRegistry.models["NeMoSpeechLMForConditionalGeneration"]
        transcribe = ModelRegistry.models["Nemotron3_5TranscribeForConditionalGeneration"]
        assert (transcribe.module_name, transcribe.class_name) == (speechlm.module_name, speechlm.class_name)
        assert (
            MODELS_CONFIG_MAP["Nemotron3_5TranscribeForConditionalGeneration"]
            is NeMoSpeechLMForConditionalGenerationConfig
        )

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
class TestDFlashPlugin:
    """Tests for the target-model contract required by DFlash and DFlash2."""

    @pytest.fixture(autouse=True)
    def restore_plugin_state(self):
        from vllm.config.speculative import SpeculativeConfig
        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm import salm as salm_module

        original_models = dict(ModelRegistry.models)
        original_override = SpeculativeConfig.hf_config_override
        original_nemo_override = salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE
        yield
        ModelRegistry.models.clear()
        ModelRegistry.models.update(original_models)
        SpeculativeConfig.hf_config_override = staticmethod(original_override)
        salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE = original_nemo_override

    def test_registers_automodel_dflash_architecture_alias(self, monkeypatch):
        """An untouched Automodel draft config should resolve to vLLM's native DFlash model."""
        from transformers import AutoConfig
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        from nemo.collections.speechlm2.vllm.salm import register

        if "DFlashDraftModel" not in ModelRegistry.get_supported_archs():
            pytest.skip("installed vLLM does not provide native DFlash support")

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        native_model = ModelRegistry.models["DFlashDraftModel"]
        automodel_config = AutoConfig.for_model("qwen3", architectures=["Qwen3DFlashDraftModel"])
        runtime_config = EAGLEConfig(automodel_config, method="dflash", model_type="eagle")
        assert runtime_config.architectures == ["DFlashQwen3DFlashDraftModel"]

        for alias in ("Qwen3DFlashDraftModel", *runtime_config.architectures):
            alias_model = ModelRegistry.models[alias]
            assert (alias_model.module_name, alias_model.class_name) == (
                native_model.module_name,
                native_model.class_name,
            )

    @pytest.mark.parametrize(
        "automodel_arch",
        ("Qwen3DFlash2DraftModel", "DFlashQwen3DFlash2DraftModel"),
    )
    def test_routes_automodel_dflash2_to_candidate_selector_runtime(self, monkeypatch, automodel_arch):
        """Automodel exports must retain DFlash2 semantics after vLLM config wrapping."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        from nemo.collections.speechlm2.vllm.salm import register

        if "DFlash2DraftModel" not in ModelRegistry.get_supported_archs():
            pytest.skip("installed vLLM does not provide the DFlash2 candidate-selector runtime")

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        native_model = ModelRegistry.models["DFlash2DraftModel"]
        automodel_config = AutoConfig.for_model("qwen3", architectures=[automodel_arch])
        normalized_config = SpeculativeConfig.hf_config_override(automodel_config)
        runtime_config = EAGLEConfig(normalized_config, method="dflash", model_type="eagle")

        assert normalized_config.architectures == ["DFlash2DraftModel"]
        assert runtime_config.architectures == ["DFlash2DraftModel"]
        runtime_model = ModelRegistry.models[runtime_config.architectures[0]]
        assert (runtime_model.module_name, runtime_model.class_name) == (
            native_model.module_name,
            native_model.class_name,
        )

    def test_model_advertises_eagle3_support(self):
        from vllm.model_executor.models.interfaces import SupportsEagle3, supports_eagle3

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert supports_eagle3(NeMoSpeechLMForConditionalGeneration)
        for method_name in ("set_aux_hidden_state_layers", "get_eagle3_default_aux_hidden_state_layers"):
            assert method_name in SupportsEagle3.__dict__
            assert method_name in NeMoSpeechLMForConditionalGeneration.__dict__

    @pytest.mark.parametrize("architecture", ("Qwen3ForCausalLM", "NemotronHForCausalLM"))
    def test_supported_backbone_implements_installed_eagle3_protocol(self, architecture):
        from vllm.model_executor.models.interfaces import supports_eagle3
        from vllm.model_executor.models.registry import ModelRegistry

        if architecture not in ModelRegistry.get_supported_archs():
            pytest.skip(f"installed vLLM does not provide {architecture}")

        model_cls = ModelRegistry.models[architecture].load_model_cls()
        assert supports_eagle3(model_cls)

    def test_get_language_model_exposes_wrapped_decoder(self):
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        language_model = object()
        object.__setattr__(model, "language_model", language_model)

        assert model.get_language_model() is language_model

    def test_aux_hidden_state_methods_delegate_to_wrapped_decoder(self):
        from unittest.mock import Mock

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        layers = (2, 6, 20, 30, 42, 52)
        language_model = Mock(
            spec=[
                "set_aux_hidden_state_layers",
                "get_eagle3_default_aux_hidden_state_layers",
            ]
        )
        language_model.get_eagle3_default_aux_hidden_state_layers.return_value = layers

        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        object.__setattr__(model, "language_model", language_model)

        model.set_aux_hidden_state_layers(layers)

        language_model.set_aux_hidden_state_layers.assert_called_once_with(layers)
        assert model.get_eagle3_default_aux_hidden_state_layers() == layers

    def test_aux_hidden_state_methods_reject_unsupported_decoder(self):
        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        object.__setattr__(model, "language_model", object())
        config = SimpleNamespace(text_config=SimpleNamespace(architectures=["UnsupportedForCausalLM"]))
        object.__setattr__(model, "config", config)

        with pytest.raises(NotImplementedError, match="UnsupportedForCausalLM.*set_aux_hidden_state_layers"):
            model.set_aux_hidden_state_layers((1,))
        with pytest.raises(NotImplementedError, match="UnsupportedForCausalLM.*get_eagle3_default"):
            model.get_eagle3_default_aux_hidden_state_layers()

    def test_forward_preserves_auxiliary_hidden_state_output(self):
        from unittest.mock import Mock

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        output = (object(), [object(), object()])
        language_model = Mock(return_value=output)
        model = object.__new__(NeMoSpeechLMForConditionalGeneration)
        object.__setattr__(model, "language_model", language_model)

        input_ids = object()
        positions = object()
        assert model.forward(input_ids, positions) is output
        language_model.assert_called_once_with(input_ids, positions, None, None)


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not installed")
class TestMTPPlugin:
    """Tests for NeMo SpeechLM MTP speculative-decoding support."""

    @pytest.fixture(autouse=True)
    def restore_original_override(self):
        """Keep the process-local fallback hook isolated between tests."""
        from nemo.collections.speechlm2.vllm import salm as salm_module

        original_override = salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE
        yield
        salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE = original_override

    @pytest.fixture(autouse=True)
    def mock_backbone_config(self, monkeypatch):
        """Keep registration tests independent of Hugging Face network access."""
        if not _HAS_CONFIG:
            return

        monkeypatch.setattr(
            _config_module.AutoConfig,
            "from_pretrained",
            lambda *args, **kwargs: SimpleNamespace(
                architectures=["NemotronHybridForCausalLM"],
                hidden_size=2048,
                vocab_size=131072,
                num_hidden_layers=4,
                num_key_value_heads=2,
                layer_norm_epsilon=1e-5,
            ),
        )

    class _HFConfigLike:
        """Minimal stand-in for a HuggingFace PretrainedConfig."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def update(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    def test_mtp_patch_registers_model(self, monkeypatch):
        """register() should add NeMoSpeechLMMTPModel to the model registry."""
        from transformers import AutoConfig
        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        assert "NeMoSpeechLMMTPModel" in ModelRegistry.get_supported_archs()

    def test_mtp_patch_extends_mtp_model_types(self, monkeypatch):
        """register() should add both SpeechLM draft model types to vLLM's MTPModelTypes Literal."""
        from typing import get_args

        import vllm.config.speculative as _spec_mod
        from transformers import AutoConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )

        register()

        assert "nemo_speechlm_mtp" in get_args(_spec_mod.MTPModelTypes)
        assert "nemotron3_5_transcribe_mtp" in get_args(_spec_mod.MTPModelTypes)

    def test_patched_override_routes_nemo_mtp_config(self, monkeypatch):
        """hf_config_override should rewrite nemo_speechlm configs with MTP heads."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={
                "enabled": True,
                "num_nextn_predict_layers": 1,
                "use_repeated_layer": True,
            },
        )
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.model_type == "nemo_speechlm_mtp"
        assert result.architectures == ["NeMoSpeechLMMTPModel"]
        assert result.n_predict == 1
        assert result.num_nextn_predict_layers == 1

    @pytest.mark.parametrize(
        ("architecture", "mtp_model_type", "mtp_architecture"),
        [
            ("NeMoSpeechLMForConditionalGeneration", "nemo_speechlm_mtp", "NeMoSpeechLMMTPModel"),
            (
                "Nemotron3_5TranscribeForConditionalGeneration",
                "nemotron3_5_transcribe_mtp",
                "Nemotron3_5TranscribeMTPModel",
            ),
        ],
    )
    def test_patched_override_names_draft_after_target(
        self, monkeypatch, architecture, mtp_model_type, mtp_architecture
    ):
        """The dedicated architecture's MTP draft uses vLLM's native model type and name; both resolve to one class."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig
        from vllm.model_executor.models.registry import ModelRegistry

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            architectures=[architecture],
            mtp={
                "enabled": True,
                "num_nextn_predict_layers": 2,
                "use_repeated_layer": True,
            },
        )
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.model_type == mtp_model_type
        assert result.architectures == [mtp_architecture]
        draft = ModelRegistry.models[mtp_architecture]
        assert (draft.module_name, draft.class_name) == ("nemo.collections.speechlm2.vllm.salm.mtp", "NeMoSpeechLMMTP")

    def test_patched_override_is_pickleable_for_spawned_engine(self, monkeypatch):
        """The callable retained on draft ModelConfig must survive multiprocessing spawn."""
        import pickle

        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module
        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        restored = pickle.loads(pickle.dumps(SpeculativeConfig.hf_config_override))
        assert restored is salm_module._nemo_speechlm_mtp_hf_config_override

        overrides = [restored]
        if hasattr(SpeculativeConfig, "compose_draft_hf_overrides"):
            composed = SpeculativeConfig.compose_draft_hf_overrides(_identity_hf_config_override)
            assert composed is not SpeculativeConfig.hf_config_override
            overrides.append(pickle.loads(pickle.dumps(composed)))

        for override in overrides:
            hf_cfg = self._HFConfigLike(
                model_type="nemo_speechlm",
                mtp={
                    "enabled": True,
                    "num_nextn_predict_layers": 4,
                    "use_repeated_layer": True,
                },
            )
            result = override(hf_cfg)
            assert result.model_type == "nemo_speechlm_mtp"
            assert result.n_predict == 1

    def test_patched_override_lazily_captures_native_hook_in_spawn_child(self, monkeypatch):
        """A fresh spawn import should delegate unrelated configs to vLLM's native hook."""
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module

        original_calls = []

        def _recording_native(cfg):
            original_calls.append(cfg)
            return cfg

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(
            SpeculativeConfig,
            "hf_config_override",
            staticmethod(_recording_native),
        )

        hf_cfg = self._HFConfigLike(model_type="unrelated")
        result = salm_module._nemo_speechlm_mtp_hf_config_override(hf_cfg)

        assert result is hf_cfg
        assert original_calls == [hf_cfg]
        assert salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE is _recording_native

    def test_patched_override_rejects_missing_native_hook(self, monkeypatch):
        """A corrupted install must fail clearly instead of recursing into our override."""
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(
            SpeculativeConfig,
            "hf_config_override",
            staticmethod(salm_module._nemo_speechlm_mtp_hf_config_override),
        )

        with pytest.raises(RuntimeError, match="without preserving vLLM's original hook"):
            salm_module._nemo_speechlm_mtp_hf_config_override(self._HFConfigLike(model_type="unrelated"))

    def test_patched_override_enabled_mtp_defaults_to_one_head(self, monkeypatch):
        """An enabled training block without an explicit depth constructs one head."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        hf_cfg = self._HFConfigLike(model_type="nemo_speechlm", mtp={"enabled": True})
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.model_type == "nemo_speechlm_mtp"
        assert result.n_predict == 1
        assert result.num_nextn_predict_layers == 1

    def test_patched_override_repeated_layer_exposes_one_reusable_head(self, monkeypatch):
        """Repeated-layer training depth must not constrain inference-time K."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={
                "enabled": True,
                "num_nextn_predict_layers": 4,
                "use_repeated_layer": True,
            },
        )
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.n_predict == 1
        assert result.num_nextn_predict_layers == 1

    def test_patched_override_no_mtp_falls_through(self, monkeypatch):
        """hf_config_override should not alter non-MTP configs."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        original_calls = []

        def _recording_orig(cfg):
            original_calls.append(cfg)
            return cfg

        from nemo.collections.speechlm2.vllm import salm as salm_module

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(SpeculativeConfig, "hf_config_override", staticmethod(_recording_orig))
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={"enabled": False, "num_nextn_predict_layers": 1},
        )
        SpeculativeConfig.hf_config_override(hf_cfg)

        assert len(original_calls) == 1

    def test_patched_override_depth_without_enabled_flag_falls_through(self, monkeypatch):
        """A retained recipe depth must not enable a head that training did not construct."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module
        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        original_calls = []

        def _recording_orig(cfg):
            original_calls.append(cfg)
            return cfg

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(SpeculativeConfig, "hf_config_override", staticmethod(_recording_orig))
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={"num_nextn_predict_layers": 4, "use_repeated_layer": True},
        )
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.model_type == "nemo_speechlm"
        assert original_calls == [hf_cfg]

    def test_patched_override_explicitly_disabled_mtp_falls_through(self, monkeypatch):
        """An exported recipe depth must not override mtp.enabled=false."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module
        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        original_calls = []

        def _recording_orig(cfg):
            original_calls.append(cfg)
            return cfg

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(SpeculativeConfig, "hf_config_override", staticmethod(_recording_orig))
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={
                "enabled": False,
                "num_nextn_predict_layers": 4,
                "use_repeated_layer": True,
            },
        )
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result.model_type == "nemo_speechlm"
        assert original_calls == [hf_cfg]

    def test_patched_override_multi_head_without_repeated_layer_raises(self, monkeypatch):
        """hf_config_override should raise for multi-head checkpoints without use_repeated_layer."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()

        hf_cfg = self._HFConfigLike(
            model_type="nemo_speechlm",
            mtp={
                "enabled": True,
                "num_nextn_predict_layers": 3,
                "use_repeated_layer": False,
            },
        )
        with pytest.raises(ValueError, match="use_repeated_layer"):
            SpeculativeConfig.hf_config_override(hf_cfg)

    def test_mtp_override_registration_is_idempotent(self, monkeypatch):
        """register() should not repeatedly wrap the config override."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        register()
        first_override = SpeculativeConfig.hf_config_override
        register()

        assert SpeculativeConfig.hf_config_override is first_override

    def test_mtp_override_reregistration_preserves_first_native_hook(self, monkeypatch):
        """A later wrapper that delegates to us must not become our fallback."""
        from transformers import AutoConfig
        from vllm.config.speculative import SpeculativeConfig

        from nemo.collections.speechlm2.vllm import salm as salm_module
        from nemo.collections.speechlm2.vllm.salm import register

        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError()),
        )
        native_calls = []

        def _recording_native(cfg):
            native_calls.append(cfg)
            return cfg

        monkeypatch.setattr(salm_module, "_ORIGINAL_VLLM_HF_CONFIG_OVERRIDE", None)
        monkeypatch.setattr(
            SpeculativeConfig,
            "hf_config_override",
            staticmethod(_recording_native),
        )
        register()
        first_override = SpeculativeConfig.hf_config_override

        def _third_party_wrapper(cfg):
            return first_override(cfg)

        monkeypatch.setattr(
            SpeculativeConfig,
            "hf_config_override",
            staticmethod(_third_party_wrapper),
        )
        register()

        hf_cfg = self._HFConfigLike(model_type="unrelated")
        result = SpeculativeConfig.hf_config_override(hf_cfg)

        assert result is hf_cfg
        assert native_calls == [hf_cfg]
        assert salm_module._ORIGINAL_VLLM_HF_CONFIG_OVERRIDE is _recording_native

    def test_mtp_declares_external_multimodal_embedding_support(self):
        """The vLLM proposer should forward target-produced audio embeddings."""
        from vllm.model_executor.models import supports_multimodal_embeddings

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        assert supports_multimodal_embeddings(NeMoSpeechLMMTP)

    @pytest.mark.parametrize(
        "name, unused",
        [
            ("llm.mtp.layers.0.mixer.q_proj.weight", False),
            ("llm.lm_head.weight", False),
            ("llm.model.embed_tokens.weight", False),
            ("llm.backbone.embeddings.weight", False),
            ("llm.model.layers.3.mixer.in_proj.weight", True),
            ("llm.model.norm.weight", True),
        ],
    )
    def test_mtp_reads_only_draft_tensors_from_the_checkpoint(self, name, unused):
        """vLLM 0.32 skips checkpoint tensors before reading them; the ``llm.``-prefixed draft head must survive."""
        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)

        assert model.is_unused_checkpoint_weight(name) is unused

    def test_mtp_defers_remapped_names_to_vllm(self, monkeypatch):
        """Names already remapped by load_weights go through vLLM's own NemotronHMTP rule."""
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        monkeypatch.setattr(
            NemotronHMTP, "is_unused_checkpoint_weight", lambda self, name: name == "unused", raising=False
        )
        model = object.__new__(NeMoSpeechLMMTP)

        assert model.is_unused_checkpoint_weight("unused") is True
        assert model.is_unused_checkpoint_weight("mtp.layers.0.mixer.q_proj.weight") is False

    def test_mtp_keeps_every_remapped_name_on_older_vllm(self, monkeypatch):
        """vLLM 0.28 has no NemotronHMTP rule; nothing is skipped there."""
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        monkeypatch.delattr(NemotronHMTP, "is_unused_checkpoint_weight", raising=False)
        model = object.__new__(NeMoSpeechLMMTP)

        assert model.is_unused_checkpoint_weight("mtp.layers.0.mixer.q_proj.weight") is False

    def test_model_declares_replayssm_support(self):
        """vLLM refuses --use-replayssm for architectures that do not declare it."""
        from vllm.model_executor.models import interfaces

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        assert NeMoSpeechLMForConditionalGeneration.supports_replayssm is True
        if hasattr(interfaces, "supports_replayssm"):
            assert interfaces.supports_replayssm(NeMoSpeechLMForConditionalGeneration)

    def test_mtp_warms_every_triton_variant_of_the_eagle_prepare_kernel(self, monkeypatch):
        """Loading the draft should JIT every num_reqs variant, so no request pays for it later."""
        import vllm.v1.spec_decode.utils as spec_utils

        from nemo.collections.speechlm2.vllm.salm.mtp import warm_eagle_prepare_inputs_kernel

        launched = []

        class _Kernel:
            def __getitem__(self, grid):
                return lambda *args: launched.append((grid, args[-1]))

        monkeypatch.setattr(spec_utils, "eagle_prepare_inputs_padded_kernel", _Kernel())
        _pretend_cuda_is_available(monkeypatch)
        warm_eagle_prepare_inputs_kernel(device="cpu")

        # Triton specializes integer arguments on ``== 1`` and ``% 16 == 0``.
        assert [num_reqs for _, num_reqs in launched] == [1, 2, 16]
        assert all(grid == (num_reqs,) for grid, num_reqs in launched)

    def test_mtp_kernel_warmup_never_fails_loading(self, monkeypatch):
        import vllm.v1.spec_decode.utils as spec_utils

        from nemo.collections.speechlm2.vllm.salm.mtp import warm_eagle_prepare_inputs_kernel

        class _Broken:
            def __getitem__(self, grid):
                raise RuntimeError("compile failed")

        monkeypatch.setattr(spec_utils, "eagle_prepare_inputs_padded_kernel", _Broken())
        _pretend_cuda_is_available(monkeypatch)
        warm_eagle_prepare_inputs_kernel(device="cpu")

    def test_embed_input_ids_text_only(self):
        """embed_input_ids with no audio embeddings should return plain text embeddings."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        m = object.__new__(NeMoSpeechLMMTP)
        base_embeds = torch.arange(6, dtype=torch.float).reshape(3, 2)
        m.model = SimpleNamespace(get_input_embeddings=lambda ids: base_embeds.clone())

        result = m.embed_input_ids(torch.tensor([1, 2, 3]), multimodal_embeddings=None)

        assert result.shape == (3, 2)
        assert torch.equal(result, base_embeds)

    def test_embed_input_ids_fuses_audio(self):
        """embed_input_ids should replace placeholder positions with audio embeddings."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        m = object.__new__(NeMoSpeechLMMTP)
        base_embeds = torch.zeros(4, 2)
        m.model = SimpleNamespace(get_input_embeddings=lambda ids: base_embeds.clone())

        audio_feat = torch.ones(2, 2) * 9.0
        is_audio = torch.tensor([False, True, True, False])
        result = m.embed_input_ids(
            torch.tensor([0, 1, 2, 3]),
            multimodal_embeddings=[audio_feat],
            is_multimodal=is_audio,
        )

        assert torch.equal(result[0], torch.zeros(2))
        assert torch.equal(result[1], torch.ones(2) * 9.0)
        assert torch.equal(result[2], torch.ones(2) * 9.0)
        assert torch.equal(result[3], torch.zeros(2))

    def test_embed_input_ids_fuses_multiple_audio_chunks(self):
        """embed_input_ids should preserve vLLM's nested embedding order."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        base_embeds = torch.zeros(5, 2)
        model.model = SimpleNamespace(get_input_embeddings=lambda ids: base_embeds.clone())

        audio_feats = [[torch.ones(1, 2) * 3.0], [torch.ones(2, 2) * 7.0]]
        is_audio = torch.tensor([True, False, True, True, False])
        result = model.embed_input_ids(
            torch.tensor([0, 1, 2, 3, 4]),
            multimodal_embeddings=audio_feats,
            is_multimodal=is_audio,
        )

        assert torch.equal(result[0], torch.ones(2) * 3.0)
        assert torch.equal(result[1], torch.zeros(2))
        assert torch.equal(result[2], torch.ones(2) * 7.0)
        assert torch.equal(result[3], torch.ones(2) * 7.0)
        assert torch.equal(result[4], torch.zeros(2))

    def test_embed_input_ids_requires_multimodal_mask(self):
        """Audio embeddings without placeholder positions should fail clearly."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        model.model = SimpleNamespace(get_input_embeddings=lambda ids: torch.zeros(2, 2))

        with pytest.raises(ValueError, match="is_multimodal"):
            model.embed_input_ids(
                torch.tensor([0, 1]),
                multimodal_embeddings=[torch.ones(1, 2)],
            )

    def test_embed_input_ids_ignores_embeddings_without_placeholder_positions(self):
        """vLLM's merge semantics leave embeddings unchanged for an all-text mask."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        model.model = SimpleNamespace(get_input_embeddings=lambda ids: torch.zeros(2, 2))

        result = model.embed_input_ids(
            torch.tensor([0, 1]),
            multimodal_embeddings=[torch.ones(1, 2)],
            is_multimodal=torch.tensor([False, False]),
        )

        assert torch.equal(result, torch.zeros(2, 2))

    def test_target_weight_split_excludes_salm_mtp_and_rejects_bare_mtp(self):
        """The target loader should route SALM draft weights, leave a bundled CTC head, and reject native names."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.model import NeMoSpeechLMForConditionalGeneration

        perception_tensor = torch.ones(1)
        llm_tensor = torch.ones(2)
        perception, llm = NeMoSpeechLMForConditionalGeneration._split_perception_llm(
            [
                ("perception.encoder.weight", perception_tensor),
                ("llm.mtp.layers.0.weight", torch.ones(4)),
                ("llm.model.layers.0.weight", llm_tensor),
                ("llm.model.layers.0._extra_state", torch.ones(5)),
                ("ctc_timestamp.decoder.decoder_layers.0.weight", torch.ones(6)),
                ("ctc_timestamp.tokenizer.model_proto", torch.ones(7, dtype=torch.uint8)),
            ]
        )

        assert perception == {"encoder.weight": perception_tensor}
        assert [name for name, _ in llm] == ["llm.model.layers.0.weight"]
        assert llm[0][1] is llm_tensor

        with pytest.raises(ValueError, match=r"llm\.mtp\.\*"):
            NeMoSpeechLMForConditionalGeneration._split_perception_llm([("mtp.layers.0.weight", torch.ones(3))])

    def test_mtp_weight_remap_uses_vllm_embedding_alias(self):
        """Exported SpeechLM embeddings must pass NemotronHMTP's name filter."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import _remap_nemo_mtp_weights

        tensor = torch.ones(2, 3)
        remapped = dict(
            _remap_nemo_mtp_weights(
                [
                    ("llm.model.embed_tokens.weight", tensor),
                    ("llm.mtp.layers.0.enorm.weight", tensor),
                    ("llm.lm_head.weight", tensor),
                ]
            )
        )

        assert set(remapped) == {
            "backbone.embeddings.weight",
            "mtp.layers.0.enorm.weight",
            "lm_head.weight",
        }
        assert remapped["backbone.embeddings.weight"] is tensor

        padded = dict(
            _remap_nemo_mtp_weights(
                [
                    ("llm.model.embed_tokens.weight", tensor),
                    ("llm.lm_head.weight", tensor),
                ],
                target_vocab=5,
            )
        )
        assert padded["backbone.embeddings.weight"].shape == (5, 3)
        assert padded["lm_head.weight"].shape == (5, 3)

    def test_mtp_weight_remap_rejects_distinct_head_layers(self):
        """Unsupported distinct heads in a SpeechLM export should fail loudly."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import _remap_nemo_mtp_weights

        tensor = torch.ones(2, 3)
        with pytest.raises(ValueError, match="distinct multi-head"):
            list(
                _remap_nemo_mtp_weights(
                    [("llm.mtp.layers.1.enorm.weight", tensor)],
                    expected_layer_modules=1,
                )
            )

    def test_mtp_weight_remap_ignores_non_salm_names(self):
        """The draft loader supports SpeechLM exports, not bare backbone weights."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import _remap_nemo_mtp_weights

        assert list(_remap_nemo_mtp_weights([("mtp.layers.0.norm.weight", torch.ones(1))])) == []

    def test_mtp_weight_remap_allows_all_sublayers_in_hybrid_pattern(self):
        """Hybrid-pattern sublayers belong to one reusable prediction step."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import _remap_nemo_mtp_weights

        tensor = torch.ones(2, 3)
        remapped = dict(
            _remap_nemo_mtp_weights(
                [(f"llm.mtp.layers.{i}.norm.weight", tensor) for i in range(3)],
                expected_layer_modules=3,
            )
        )
        assert set(remapped) == {f"mtp.layers.{i}.norm.weight" for i in range(3)}

    def test_mtp_load_weights_bounds_layers_from_serialized_pattern(self, monkeypatch):
        """The loader should derive its physical-layer bound from config."""
        import torch
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        torch.nn.Module.__init__(model)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(mtp_hybrid_override_pattern="*E", vocab_size=3),
        )
        captured = []

        def _capture_weights(self, weights):
            captured.extend(weights)
            return set()

        monkeypatch.setattr(NemotronHMTP, "load_weights", _capture_weights)
        tensor = torch.ones(1, 2)
        NeMoSpeechLMMTP.load_weights(
            model,
            [
                ("llm.model.embed_tokens.weight", tensor),
                ("llm.mtp.layers.0.norm.weight", tensor),
                ("llm.mtp.layers.1.norm.weight", tensor),
                ("llm.lm_head.weight", tensor),
            ],
        )
        assert [name for name, _ in captured] == [
            "backbone.embeddings.weight",
            "mtp.layers.0.norm.weight",
            "mtp.layers.1.norm.weight",
            "lm_head.weight",
        ]
        captured_tensors = dict(captured)
        assert captured_tensors["backbone.embeddings.weight"].shape == (3, 2)
        assert captured_tensors["lm_head.weight"].shape == (3, 2)
        assert captured_tensors["mtp.layers.0.norm.weight"] is tensor
        assert captured_tensors["mtp.layers.1.norm.weight"] is tensor

        with pytest.raises(ValueError, match="distinct multi-head"):
            NeMoSpeechLMMTP.load_weights(model, [("llm.mtp.layers.2.norm.weight", tensor)])

    @pytest.mark.parametrize("pattern", [None, "", 3])
    def test_mtp_load_weights_rejects_invalid_serialized_pattern(self, monkeypatch, pattern):
        import torch
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        torch.nn.Module.__init__(model)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(mtp_hybrid_override_pattern=pattern, vocab_size=3),
        )
        monkeypatch.setattr(NemotronHMTP, "load_weights", lambda self, weights: set())

        with pytest.raises(ValueError, match="non-empty string"):
            NeMoSpeechLMMTP.load_weights(model, [])

    def test_mtp_load_weights_rejects_checkpoint_without_draft_head(self, monkeypatch):
        """A quantized export that dropped llm.mtp.* must fail instead of drafting from uninitialized weights."""
        import torch
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        torch.nn.Module.__init__(model)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(mtp_hybrid_override_pattern="*E", vocab_size=3),
        )
        monkeypatch.setattr(NemotronHMTP, "load_weights", lambda self, weights: {name for name, _ in weights})
        tensor = torch.ones(1, 2)

        with pytest.raises(ValueError, match="no 'llm.mtp"):
            NeMoSpeechLMMTP.load_weights(
                model,
                [("backbone.layers.0.mixer.in_proj.weight", tensor), ("lm_head.weight", tensor)],
            )

    def test_mtp_load_weights_accepts_checkpoint_with_draft_head(self, monkeypatch):
        """A checkpoint carrying llm.mtp.* passes the guard and hands the head tensors to vLLM's loader."""
        import torch
        from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP

        from nemo.collections.speechlm2.vllm.salm.mtp import NeMoSpeechLMMTP

        model = object.__new__(NeMoSpeechLMMTP)
        torch.nn.Module.__init__(model)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(mtp_hybrid_override_pattern="*E", vocab_size=3),
        )
        monkeypatch.setattr(NemotronHMTP, "load_weights", lambda self, weights: {name for name, _ in weights})
        tensor = torch.ones(1, 2)

        loaded = NeMoSpeechLMMTP.load_weights(
            model,
            [("backbone.layers.0.mixer.in_proj.weight", tensor), ("llm.mtp.layers.0.eh_proj.weight", tensor)],
        )

        assert loaded == {"mtp.layers.0.eh_proj.weight"}

    def test_mtp_weight_remap_splits_packed_experts(self):
        """Packed Automodel MTP experts must become vLLM per-expert weights."""
        import torch

        from nemo.collections.speechlm2.vllm.salm.mtp import _remap_nemo_mtp_weights

        down_projs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        up_projs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        remapped = dict(
            _remap_nemo_mtp_weights(
                [
                    ("llm.mtp.layers.1.mixer.experts.down_projs", down_projs),
                    ("llm.mtp.layers.1.mixer.experts.gate_and_up_projs", up_projs),
                    ("llm.mtp.layers.1.mixer.experts._extra_state", torch.tensor(1)),
                ]
            )
        )

        assert set(remapped) == {
            "mtp.layers.1.mixer.experts.0.down_proj.weight",
            "mtp.layers.1.mixer.experts.1.down_proj.weight",
            "mtp.layers.1.mixer.experts.0.up_proj.weight",
            "mtp.layers.1.mixer.experts.1.up_proj.weight",
        }
        for expert_idx in range(2):
            down = remapped[f"mtp.layers.1.mixer.experts.{expert_idx}.down_proj.weight"]
            up = remapped[f"mtp.layers.1.mixer.experts.{expert_idx}.up_proj.weight"]
            assert down.shape == (3, 4)
            assert up.shape == (4, 3)
            assert down.dtype == down_projs.dtype
            assert up.dtype == up_projs.dtype
            assert torch.equal(down, down_projs[expert_idx].t())
            assert torch.equal(up, up_projs[expert_idx].t())

    @pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
    def test_mtp_hybrid_override_pattern_from_config(self):
        """mtp_hybrid_override_pattern should read hybrid_override_pattern from mtp config dict."""
        cfg = NeMoSpeechLMConfig(
            **_DEFAULT_CONFIG_KWARGS,
            mtp={
                "enabled": True,
                "num_nextn_predict_layers": 1,
                "hybrid_override_pattern": "*E",
            },
        )
        assert cfg.mtp_hybrid_override_pattern == "*E"

    @pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
    def test_mtp_hybrid_override_pattern_default_all_attention(self):
        """mtp_hybrid_override_pattern should default to '*' (all-attention) when absent."""
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        assert cfg.mtp_hybrid_override_pattern == "*"

    @pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
    def test_image_token_index_is_unserialized_backbone_vocab_boundary(self):
        """vLLM's compatibility property should identify the first padded row."""
        import importlib

        config_mod = importlib.import_module("nemo.collections.speechlm2.vllm.salm.config")
        extra_rows = config_mod._SPEECHLM_EMBED_EXTRA_ROWS

        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS)
        base_vocab = cfg.text_config.vocab_size - extra_rows
        assert cfg.image_token_index == base_vocab
        # vLLM copies the target value onto the draft config. The setter accepts
        # that runtime-only assignment without adding serialized state.
        cfg.image_token_index = base_vocab
        assert cfg.image_token_index == base_vocab
        assert "audio_token_index" not in cfg.to_dict()
        assert "image_token_index" not in cfg.to_dict()

    @pytest.mark.skipif(not _HAS_CONFIG, reason="NeMoSpeechLMConfig not available")
    def test_legacy_image_token_index_is_validated_after_backbone_load(self):
        cfg = NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, image_token_index=131072)
        assert cfg.image_token_index == 131072
        assert "image_token_index" not in cfg.to_dict()

        with pytest.raises(ValueError, match="backbone vocabulary boundary"):
            NeMoSpeechLMConfig(**_DEFAULT_CONFIG_KWARGS, image_token_index=42)


class _SpecialTokenTokenizer:
    """Whitespace tokenizer with ``<|audio|>`` as a special token (split out of the text before tokenizing).

    With ``separable=False`` the special token's id depends on the preceding token, so the template ids cannot
    be expanded.
    """

    def __init__(self, separable=True):
        self.separable = separable
        self.encoded = []

    def get_vocab(self):
        return {"<|audio|>": 1}

    def add_special_tokens(self, tokens):
        pass

    def encode(self, prompt, add_special_tokens=True):
        import re

        self.encoded.append(prompt)
        ids = [0] if add_special_tokens else []
        for piece in re.split(r"(<\|audio\|>)", prompt):
            if piece == "<|audio|>":
                ids.append(1 if self.separable or ids[-1] != 1 else 2)
            else:
                ids.extend(10 + len(w) for w in piece.split())
        return ids


def _pretend_cuda_is_available(monkeypatch):
    """Let the MTP kernel warmup run its launch loop on a machine without a CUDA device."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)


def _toy_perception():
    """Stand-in for the perception module: a framewise 'preprocessor', a 2x subsampling encoder with a
    length-masked mean (so padding would leak into valid frames if it were not masked), and the online-inference
    API."""
    import torch
    from torch import nn

    class _ToyPerception(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(10, 8)
            self.encoder = SimpleNamespace(chunk_feat_len=1024, online_inference=contextlib.nullcontext)

        def maybe_preprocess_audio(self, input_signal, input_signal_length, processed_signal, processed_signal_length):
            if processed_signal is not None:
                return processed_signal, processed_signal_length
            b, n = input_signal.shape
            return input_signal[:, : n // 10 * 10].view(b, -1, 10).transpose(1, 2), input_signal_length // 10

        def forward(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
        ):
            x, x_len = self.maybe_preprocess_audio(
                input_signal, input_signal_length, processed_signal, processed_signal_length
            )
            mask = (torch.arange(x.shape[-1], device=x.device) < x_len[:, None]).unsqueeze(1)
            mean = (x * mask).sum(-1, keepdim=True) / x_len[:, None, None]
            h = self.proj(((x - mean) * mask).transpose(1, 2))
            return h[:, ::2], (x_len + 1) // 2

    return _ToyPerception()


class _FakeTokenizer:
    def __init__(self):
        self.added_special_tokens = None

    def get_vocab(self):
        return {}

    def add_special_tokens(self, tokens):
        self.added_special_tokens = tokens

    def encode(self, prompt, add_special_tokens=True):
        return list(range(max(1, len(prompt.split()))))
