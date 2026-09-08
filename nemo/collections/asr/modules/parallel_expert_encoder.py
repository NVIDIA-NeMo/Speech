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

"""Parallel Expert Speech Encoder.

Runs a Sortformer speaker-diarization branch and either an ASR FastConformer or
native Transformer encoder on the same mel input, then fuses their outputs with
a sinusoidal speaker kernel. The encoder expects unnormalized mels; the ASR and
Sortformer branches independently reapply ``normalize_batch`` internally. I/O
matches :class:`ConformerEncoder`.

Only self-contained bundles with inline ``asr_encoder_cfg`` and
``diarization_model_cfg`` sections are supported.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tarfile
from collections.abc import Mapping
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, Serialization
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    "ParallelExpertEncoder",
    "ParallelExpertEncoderPT",
    "TransformerCTCDecoder",
]

_ASR_ENCODER_TYPES = {
    "fastconformer": ConformerEncoder,
    "transformer": TransformerEncoder,
}
_SPEAKER_FEATURE_CONFIG_VERSION = 1
_SPEAKER_FEATURE_MODE_CONTINUOUS = "continuous"
_SPEAKER_FEATURE_MODE_THRESHOLD = "thresholded"
_SPEAKER_FEATURE_MODES = frozenset({_SPEAKER_FEATURE_MODE_CONTINUOUS, _SPEAKER_FEATURE_MODE_THRESHOLD})
_BUNDLE_CONFIG_OVERRIDE_KEYS = frozenset(
    {
        "asr_normalize_type",
        "diar_normalize_type",
        "missing_rttm_target",
        "speaker_activity_threshold",
        "speaker_feature_config_version",
        "speaker_feature_mode",
        "spk_kernel_scale",
        "sync_max_audio_length",
    }
)


def _disable_max_seq_length_sync(module: nn.Module) -> None:
    """Disable feature-length collectives in every encoder below ``module``."""
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def _normalize_asr_encoder_type(asr_encoder_type: Optional[str]) -> str:
    """Validate and normalize the ASR architecture selector."""
    normalized = "fastconformer" if asr_encoder_type is None else str(asr_encoder_type).lower()
    if normalized not in _ASR_ENCODER_TYPES:
        supported = ", ".join(sorted(_ASR_ENCODER_TYPES))
        raise ValueError(f"asr_encoder_type must be one of {{{supported}}}, got {asr_encoder_type!r}.")
    return normalized


def _normalize_speaker_feature_contract(
    speaker_feature_mode: Optional[str],
    speaker_activity_threshold: Optional[float],
) -> tuple[str, Optional[float]]:
    """Validate one explicit speaker-feature fusion contract.

    ``None`` for ``speaker_feature_mode`` is supported only by the inner-module
    constructor, where it derives the mode from the threshold for API
    compatibility. Bundle configs are resolved separately and always become
    explicit before the inner module is constructed.
    """
    if speaker_feature_mode is None:
        speaker_feature_mode = (
            _SPEAKER_FEATURE_MODE_CONTINUOUS if speaker_activity_threshold is None else _SPEAKER_FEATURE_MODE_THRESHOLD
        )
    normalized_mode = str(speaker_feature_mode).lower()
    if normalized_mode not in _SPEAKER_FEATURE_MODES:
        supported = ", ".join(sorted(_SPEAKER_FEATURE_MODES))
        raise ValueError(f"speaker_feature_mode must be one of {{{supported}}}, got {speaker_feature_mode!r}.")
    if normalized_mode == _SPEAKER_FEATURE_MODE_CONTINUOUS:
        if speaker_activity_threshold is not None:
            raise ValueError("speaker_feature_mode='continuous' requires speaker_activity_threshold=None.")
        return normalized_mode, None

    if speaker_activity_threshold is None:
        raise ValueError("speaker_feature_mode='thresholded' requires a non-null speaker_activity_threshold.")
    threshold = float(speaker_activity_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"speaker_activity_threshold must be in [0, 1], got {speaker_activity_threshold!r}.")
    return normalized_mode, threshold


def _resolve_speaker_feature_contract(cfg: DictConfig) -> tuple[str, Optional[float]]:
    """Resolve the versioned speaker-feature contract and fail closed when ambiguous."""
    config_version = cfg.get("speaker_feature_config_version", None)
    speaker_feature_mode = cfg.get("speaker_feature_mode", None)
    has_threshold = "speaker_activity_threshold" in cfg
    speaker_activity_threshold = cfg.get("speaker_activity_threshold", None)

    if config_version is not None and int(config_version) != _SPEAKER_FEATURE_CONFIG_VERSION:
        raise ValueError(
            "Unsupported speaker_feature_config_version="
            f"{config_version!r}; expected {_SPEAKER_FEATURE_CONFIG_VERSION}."
        )
    if speaker_feature_mode is not None:
        return _normalize_speaker_feature_contract(speaker_feature_mode, speaker_activity_threshold)
    if config_version is not None:
        raise ValueError("speaker_feature_config_version requires an explicit speaker_feature_mode.")
    if has_threshold:
        return _normalize_speaker_feature_contract(None, speaker_activity_threshold)

    raise ValueError(
        "Unversioned canonical ParallelExpertEncoder bundle has no speaker-feature contract. "
        "Historical canonical bundles were used with both continuous and thresholded activity, "
        "so this cannot be inferred safely. Supply explicit config_overrides with "
        "speaker_feature_config_version=1, speaker_feature_mode, and speaker_activity_threshold."
    )


def _merge_bundle_config_overrides(cfg: DictConfig, config_overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    """Merge the small, runtime-semantic PEE override surface into a bundle config."""
    merged = _clone_config(cfg)
    if config_overrides in (None, {}):
        return merged
    if not isinstance(config_overrides, Mapping):
        raise TypeError(
            f"ParallelExpertEncoder config_overrides must be a mapping, got {type(config_overrides).__name__}."
        )
    unknown = sorted(set(config_overrides) - _BUNDLE_CONFIG_OVERRIDE_KEYS)
    if unknown:
        supported = ", ".join(sorted(_BUNDLE_CONFIG_OVERRIDE_KEYS))
        raise ValueError(
            f"Unsupported ParallelExpertEncoder config_overrides keys {unknown}; supported keys: {supported}."
        )
    return OmegaConf.merge(merged, OmegaConf.create(dict(config_overrides)))


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype."""
    previous = torch.get_default_dtype()
    if dtype == previous or not dtype.is_floating_point:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _disable_dist_feature_sync():
    """Temporarily make ``torch.distributed`` look uninitialized.

    Sortformer's streaming path synchronizes feature lengths across ranks. A
    generation worker processes one recording, so that synchronization is both
    unnecessary and unsafe there.
    """
    if not (hasattr(dist, "is_initialized") and dist.is_initialized()):
        yield
        return
    original_is_initialized = dist.is_initialized
    dist.is_initialized = lambda: False
    try:
        yield
    finally:
        dist.is_initialized = original_is_initialized


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations."""
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _read_bundle_members(nemo_path: str) -> tuple[DictConfig, dict[str, torch.Tensor]]:
    """Read a local PE bundle's config and state dictionary."""
    config_bytes = None
    weights_bytes = None
    try:
        with tarfile.open(nemo_path, mode="r") as archive:
            for member in archive.getmembers():
                basename = os.path.basename(member.name)
                if basename not in {"model_config.yaml", "model_weights.ckpt"}:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                if basename == "model_config.yaml":
                    config_bytes = stream.read()
                else:
                    weights_bytes = stream.read()
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"Could not read ParallelExpertEncoder bundle {nemo_path!r}: {error}") from error

    if config_bytes is None:
        raise RuntimeError(f"{nemo_path!r} is missing model_config.yaml.")
    if weights_bytes is None:
        raise RuntimeError(f"{nemo_path!r} is missing model_weights.ckpt.")
    config = OmegaConf.create(config_bytes.decode("utf-8"))
    state = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    return config, state


@experimental
class ParallelExpertEncoderPT(ModelPT):
    """ModelPT shell for saving and restoring a PE ``.nemo`` archive."""

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        self._validate_bundle_schema(cfg)
        super().__init__(cfg=cfg, trainer=trainer)
        speaker_feature_mode, speaker_activity_threshold = _resolve_speaker_feature_contract(self._cfg)
        self.encoder = ParallelExpertEncoder(
            asr_encoder_cfg=self._cfg.get("asr_encoder_cfg", None),
            diarization_model_cfg=self._cfg.get("diarization_model_cfg", None),
            asr_encoder_type=self._cfg.get("asr_encoder_type", "fastconformer"),
            asr_normalize_type=self._cfg.get("asr_normalize_type", None),
            diar_normalize_type=self._cfg.get("diar_normalize_type", None),
            freeze_diar=self._cfg.get("freeze_diar", True),
            freeze_asr=self._cfg.get("freeze_asr", False),
            online_inference_length=self._cfg.get("online_inference_length", 500),
            chunk_left_context=self._cfg.get("chunk_left_context", 50),
            chunk_right_context=self._cfg.get("chunk_right_context", 50),
            diar_fifo_len=self._cfg.get("diar_fifo_len", 40),
            diar_spkcache_update_period=self._cfg.get("diar_spkcache_update_period", 300),
            diar_spkcache_len=self._cfg.get("diar_spkcache_len", 188),
            missing_rttm_target=self._cfg.get("missing_rttm_target", -1.0),
            speaker_feature_mode=speaker_feature_mode,
            speaker_activity_threshold=speaker_activity_threshold,
            spk_kernel_scale=self._cfg.get("spk_kernel_scale", 1.0),
            sync_max_audio_length=self._cfg.get("sync_max_audio_length", False),
        )
        # Keep the architecture-only bundle config beside the inner module.
        # SpeechLM HF export embeds this small config in config.json so the
        # consolidated checkpoint can reconstruct phPEE without carrying a
        # second, multi-GB copy of its initialization bundle.
        self.encoder._bundle_config = _clone_config(self._cfg)
        self.encoder._bundle_config.diar_normalize_type = self.encoder.diar_normalize_type
        self.encoder._bundle_config.speaker_feature_config_version = _SPEAKER_FEATURE_CONFIG_VERSION
        self.encoder._bundle_config.speaker_feature_mode = self.encoder.speaker_feature_mode
        self.encoder._bundle_config.speaker_activity_threshold = self.encoder.speaker_activity_threshold
        self.encoder._bundle_config.sync_max_audio_length = self.encoder.sync_max_audio_length

    @staticmethod
    def _validate_bundle_schema(cfg: DictConfig) -> None:
        """Require the self-contained ParallelExpertEncoder bundle schema."""
        missing = [key for key in ("asr_encoder_cfg", "diarization_model_cfg") if cfg.get(key, None) in (None, {}, "")]
        if missing:
            raise ValueError(f"ParallelExpertEncoder bundle is missing required config sections {missing}.")
        _normalize_asr_encoder_type(cfg.get("asr_encoder_type", "fastconformer"))

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass

    @classmethod
    def is_pe_nemo(cls, nemo_path: str) -> bool:
        """Return whether a local archive declares a ParallelExpertEncoderPT target."""
        if not (isinstance(nemo_path, str) and nemo_path.endswith(".nemo") and os.path.isfile(nemo_path)):
            return False
        try:
            with tarfile.open(nemo_path, mode="r") as archive:
                for member in archive.getmembers():
                    if os.path.basename(member.name) != "model_config.yaml":
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        return False
                    cfg = OmegaConf.create(stream.read().decode("utf-8"))
                    if not str(cfg.get("target", "")).endswith("ParallelExpertEncoderPT"):
                        return False
                    # Keep the released public probe target-based. Runtime loading
                    # uses the schema resolver and remains strict.
                    return True
        except (tarfile.TarError, OSError) as error:
            logging.warning("[ParallelExpertEncoder] Could not inspect %s: %s", nemo_path, error)
            return False
        return False

    @classmethod
    def load_from_nemo(
        cls,
        model_path_or_name: str,
        *,
        map_location: Union[str, torch.device] = "cpu",
        strict: bool = True,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> ParallelExpertEncoder:
        """Load a PE bundle and return its inner encoder.

        config_overrides is intentionally restricted to runtime-semantic fields.
        It resolves legacy bundle ambiguity without allowing a recipe to replace
        the saved encoder architecture accidentally.
        """
        if (
            isinstance(model_path_or_name, str)
            and model_path_or_name.endswith(".nemo")
            and os.path.isfile(model_path_or_name)
        ):
            cfg, state = _read_bundle_members(model_path_or_name)
            if not str(cfg.get("target", "")).endswith("ParallelExpertEncoderPT"):
                raise ValueError(f"{model_path_or_name!r} is not a ParallelExpertEncoderPT .nemo bundle.")
            cfg = _merge_bundle_config_overrides(cfg, config_overrides)
            cls._validate_bundle_schema(cfg)
            shell = cls(cfg=cfg, trainer=None)
            prefix = "encoder."
            encoder_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            if not encoder_state:
                raise RuntimeError(
                    f"No '{prefix}*' tensors found in {model_path_or_name!r}; the archive is not a saved PE bundle."
                )
            incompatible = shell.encoder.load_state_dict(encoder_state, strict=strict)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                logging.warning(
                    "[ParallelExpertEncoder] load_from_nemo(%s): %d missing / %d unexpected keys.",
                    model_path_or_name,
                    len(incompatible.missing_keys),
                    len(incompatible.unexpected_keys),
                )
            return shell.encoder.to(map_location)

        if config_overrides not in (None, {}):
            raise ValueError(
                "ParallelExpertEncoder config_overrides currently require a local .nemo bundle path; "
                f"got pretrained model identifier {model_path_or_name!r}."
            )
        bundle = cls.from_pretrained(
            model_name=model_path_or_name,
            map_location=map_location,
            strict=strict,
        )
        return bundle.encoder

    @classmethod
    def from_inline_config(
        cls,
        cfg: Union[DictConfig, dict],
        *,
        map_location: Union[str, torch.device] = "cpu",
    ) -> ParallelExpertEncoder:
        """Construct phPEE architecture without loading standalone weights.

        This is intended for consolidated SpeechLM checkpoints, whose root
        state dict supplies every phPEE tensor after construction.
        """
        shell = cls(cfg=OmegaConf.create(cfg), trainer=None)
        return shell.encoder.to(map_location)

    @classmethod
    def save_to_nemo(
        cls,
        encoder: ParallelExpertEncoder,
        output_nemo_path: str,
        *,
        template_bundle_path: str,
    ) -> None:
        """Save ``encoder`` using a compatible PE bundle config as a template."""
        if not isinstance(encoder, ParallelExpertEncoder):
            raise TypeError(f"save_to_nemo expects a ParallelExpertEncoder, got {type(encoder).__name__}")
        if not os.path.isfile(template_bundle_path):
            raise FileNotFoundError(f"template_bundle_path does not exist: {template_bundle_path}")

        template_cfg = None
        with tarfile.open(template_bundle_path, mode="r") as archive:
            for member in archive.getmembers():
                if os.path.basename(member.name) != "model_config.yaml":
                    continue
                stream = archive.extractfile(member)
                if stream is not None:
                    template_cfg = OmegaConf.create(stream.read().decode("utf-8"))
                break
        if template_cfg is None:
            raise RuntimeError(f"Could not read model_config.yaml from template bundle: {template_bundle_path}")
        cls._validate_bundle_schema(template_cfg)

        template_d_model = int(template_cfg.asr_encoder_cfg.get("d_model", -1))
        template_n_spk = int(template_cfg.diarization_model_cfg.get("sortformer_modules", {}).get("num_spks", -1))
        template_asr_encoder_type = _normalize_asr_encoder_type(template_cfg.get("asr_encoder_type", "fastconformer"))
        if template_asr_encoder_type != encoder.asr_encoder_type:
            raise ValueError(
                f"Template asr_encoder_type={template_asr_encoder_type!r} does not match "
                f"encoder.asr_encoder_type={encoder.asr_encoder_type!r}; "
                "the saved bundle would instantiate the wrong ASR encoder architecture."
            )
        if template_d_model != int(encoder.d_model):
            raise ValueError(
                f"Template asr_encoder_cfg.d_model={template_d_model} does not match "
                f"encoder.d_model={encoder.d_model}; the saved bundle would fail strict reload."
            )
        if template_n_spk != int(encoder.n_spk):
            raise ValueError(
                "Template diarization_model_cfg.sortformer_modules.num_spks="
                f"{template_n_spk} does not match encoder.n_spk={encoder.n_spk}; "
                "the saved bundle would fail strict reload."
            )

        shell = cls(cfg=template_cfg, trainer=None)
        shell.encoder = encoder
        template_cfg.diar_normalize_type = encoder.diar_normalize_type
        template_cfg.speaker_feature_config_version = _SPEAKER_FEATURE_CONFIG_VERSION
        template_cfg.speaker_feature_mode = encoder.speaker_feature_mode
        template_cfg.speaker_activity_threshold = encoder.speaker_activity_threshold
        template_cfg.sync_max_audio_length = encoder.sync_max_audio_length
        shell._cfg = template_cfg
        shell.save_to(output_nemo_path)


@experimental
class ParallelExpertEncoder(nn.Module):
    """Sortformer diarizer plus a selectable ASR encoder with Conformer-compatible I/O.

    ``asr_encoder_type='fastconformer'`` preserves legacy bundle behavior and
    expects ``asr_encoder_cfg`` to instantiate :class:`ConformerEncoder`.
    ``asr_encoder_type='transformer'`` selects the native
    :class:`TransformerEncoder` used by Transformer AED ASR checkpoints.
    """

    supports_external_speaker_targets = True

    def __init__(
        self,
        asr_encoder_cfg: DictConfig,
        diarization_model_cfg: DictConfig,
        asr_normalize_type: Optional[str] = None,
        diar_normalize_type: Optional[str] = None,
        freeze_diar: bool = True,
        freeze_asr: bool = False,
        online_inference_length: int = 500,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: int = 40,
        diar_spkcache_update_period: int = 300,
        diar_spkcache_len: int = 188,
        asr_encoder_type: str = "fastconformer",
        missing_rttm_target: float = -1.0,
        speaker_feature_mode: Optional[str] = None,
        speaker_activity_threshold: Optional[float] = None,
        spk_kernel_scale: float = 1.0,
        sync_max_audio_length: bool = False,
    ):
        super().__init__()

        # Lazy import: SortformerEncLabelModel imports from asr.modules.
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        if asr_encoder_cfg is None or diarization_model_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires both asr_encoder_cfg and diarization_model_cfg; "
                "self-contained PE bundles supply them inline in model_config.yaml."
            )

        self.asr_encoder_type = _normalize_asr_encoder_type(asr_encoder_type)
        self.asr_encoder = Serialization.from_config_dict(_clone_config(asr_encoder_cfg))
        expected_encoder_class = _ASR_ENCODER_TYPES[self.asr_encoder_type]
        if not isinstance(self.asr_encoder, expected_encoder_class):
            raise TypeError(
                f"asr_encoder_type={self.asr_encoder_type!r} requires asr_encoder_cfg._target_ "
                f"to instantiate {expected_encoder_class.__name__}, got {type(self.asr_encoder).__name__}."
            )
        self.asr_normalize_type = asr_normalize_type or "per_feature"
        self._feat_in = self.asr_encoder._feat_in

        diarization_model_cfg = _clone_config(diarization_model_cfg)
        if diar_normalize_type is None:
            diar_normalize_type = diarization_model_cfg.get("preprocessor", {}).get("normalize", None)
        self.diar_normalize_type = diar_normalize_type
        configured_diar_subsampling = int(diarization_model_cfg.encoder.get("subsampling_factor", -1))
        if configured_diar_subsampling != self.asr_encoder.subsampling_factor:
            raise ValueError(
                "ParallelExpertEncoder requires the diarization output subsampling factor and embedded diarization encoder subsampling factor "
                f"({configured_diar_subsampling}) to equal the ASR encoder "
                f"subsampling factor ({self.asr_encoder.subsampling_factor})."
            )
        diarization_model_cfg.output_subsampling_factor = self.asr_encoder.subsampling_factor
        self.diarization_model = SortformerEncLabelModel.from_config_dict(diarization_model_cfg)
        diarization_subsampling_factor = int(self.diarization_model.encoder.subsampling_factor)
        if diarization_subsampling_factor != self.asr_encoder.subsampling_factor:
            raise ValueError(
                "ParallelExpertEncoder instantiated a diarization encoder with subsampling factor "
                f"({diarization_subsampling_factor}) instead of the ASR encoder factor "
                f"({self.asr_encoder.subsampling_factor})."
            )

        # The ASR and diarization experts are called from data-dependent paths
        # in both training and replicated inference. Their positional
        # buffers are local state, so synchronizing the longest feature length
        # on the default process group is unnecessary and can deadlock when
        # ranks process different request shapes.
        self.sync_max_audio_length = bool(sync_max_audio_length)
        if not self.sync_max_audio_length:
            _disable_max_seq_length_sync(self)

        self.freeze_diar = bool(freeze_diar)
        self.freeze_asr = bool(freeze_asr)
        self.online_inference_length = int(online_inference_length)
        self.online_inference_enabled: Optional[bool] = None
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        self.chunk_feat_len = self.online_inference_length * self.asr_encoder.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.asr_encoder.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.asr_encoder.subsampling_factor
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)

        self.missing_rttm_target = float(missing_rttm_target)
        self.speaker_feature_mode, self.speaker_activity_threshold = _normalize_speaker_feature_contract(
            speaker_feature_mode, speaker_activity_threshold
        )
        self.spk_kernel_scale = float(spk_kernel_scale)
        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = int(self.asr_encoder.d_model)

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.register_buffer(
            "diar_kernel",
            self._build_sinusoid_position_encoding(self.n_spk, self.asr_d_model),
            persistent=False,
        )
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        if self.freeze_diar:
            self.diarization_model.requires_grad_(False)
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.requires_grad_(False)
            self.asr_encoder.eval()

    def train(self, mode: bool = True) -> ParallelExpertEncoder:
        """Set mode while keeping frozen branches in evaluation mode."""
        super().train(mode)
        if self.freeze_diar:
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.eval()
        return self

    @property
    def d_model(self) -> int:
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        return self.asr_encoder.subsampling_factor

    @property
    def pre_encode(self):
        return self.asr_encoder.pre_encode

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Wrap trainable ASR stages before FSDP2 sharding.

        The frozen Sortformer branch is deliberately excluded. Per-layer wrappers
        preserve FSDP2 boundaries, unlike a checkpoint around the entire encoder
        call.
        """
        if not enabled or self.freeze_asr:
            return
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        pre_encode = getattr(self.asr_encoder, "pre_encode", None)
        if (
            pre_encode is not None
            and not isinstance(pre_encode, nn.Linear)
            and getattr(pre_encode, "_checkpoint_wrapped_module", None) is None
        ):
            self.asr_encoder.pre_encode = checkpoint_wrapper(pre_encode)

        layers = getattr(self.asr_encoder, "layers", None)
        if layers is not None:
            for index, layer in enumerate(layers):
                if getattr(layer, "_checkpoint_wrapped_module", None) is None:
                    layers[index] = checkpoint_wrapper(layer)

    def _asr_output_frame_boundary(self, input_frame_boundary: int) -> int:
        """Map an input-frame boundary to the selected ASR encoder's output grid."""
        if getattr(self, "asr_encoder_type", "fastconformer") == "transformer":
            return (input_frame_boundary + self.subsampling_factor - 1) // self.subsampling_factor
        return round(input_frame_boundary / self.subsampling_factor)

    def freeze(self) -> None:
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        unfreeze(self, partial=partial)

    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(max_position, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    @staticmethod
    def _align_diar_frames(spk_targets: torch.Tensor, target_len: int) -> torch.Tensor:
        if spk_targets.ndim != 3:
            raise ValueError(f"spk_targets must have shape (B, T, n_spk), got {tuple(spk_targets.shape)}.")
        current_len = spk_targets.shape[1]
        if current_len == 0 and target_len:
            raise ValueError("spk_targets cannot have an empty time dimension when encoder output is non-empty.")
        if current_len < target_len:
            last = spk_targets[:, -1:, :]
            spk_targets = torch.cat([spk_targets, last.repeat(1, target_len - current_len, 1)], dim=1)
        elif current_len > target_len:
            spk_targets = spk_targets[:, :target_len, :]
        return spk_targets

    @staticmethod
    def _match_module_io(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        parameter = next(module.parameters(), None)
        if parameter is None:
            return tensor
        return tensor.to(device=parameter.device, dtype=parameter.dtype)

    def _check_spk_targets(self, spk_targets: Optional[torch.Tensor], batch_size: int) -> None:
        if spk_targets is None:
            return
        n_spk = int(getattr(self, "n_spk", self.diar_kernel.shape[0]))
        if spk_targets.ndim != 3 or spk_targets.shape[0] != batch_size:
            raise ValueError(
                f"spk_targets must have shape ({batch_size}, T, {n_spk}), got {tuple(spk_targets.shape)}."
            )
        if spk_targets.shape[-1] != n_spk:
            raise ValueError(
                f"spk_targets carry {spk_targets.shape[-1]} speaker slots, but this encoder uses n_spk={n_spk}."
            )

    def _missing_target_rows(self, spk_targets: torch.Tensor) -> torch.Tensor:
        missing_rttm_target = getattr(self, "missing_rttm_target", None)
        if missing_rttm_target is None:
            return torch.zeros(spk_targets.shape[0], dtype=torch.bool, device=spk_targets.device)
        return (spk_targets == missing_rttm_target).all(dim=(1, 2))

    def _should_run_diarization(
        self,
        spk_targets: Optional[torch.Tensor],
        use_diarization: Optional[torch.Tensor] = None,
    ) -> bool:
        """Run a uniform training/distributed path while retaining the local eval fast path."""
        if spk_targets is None or self.training:
            return True
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return True
        if use_diarization is None:
            use_diarization = self._missing_target_rows(spk_targets)
        return bool(use_diarization.any().item())

    def _speaker_features(self, targets: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Apply the bundle's explicit speaker-feature fusion contract."""
        mode = getattr(self, "speaker_feature_mode", None)
        if mode == _SPEAKER_FEATURE_MODE_CONTINUOUS:
            return targets.to(dtype)
        if mode != _SPEAKER_FEATURE_MODE_THRESHOLD:
            raise RuntimeError(f"Invalid speaker_feature_mode at runtime: {mode!r}.")
        threshold = getattr(self, "speaker_activity_threshold", None)
        if threshold is None:
            raise RuntimeError("Thresholded speaker features require speaker_activity_threshold.")
        return (targets > threshold).to(dtype)

    def _fuse_diar_and_asr(
        self,
        asr_encoded: torch.Tensor,
        spk_targets: torch.Tensor,
        *,
        diarization_preds: Optional[torch.Tensor] = None,
        use_diarization: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse ASR states with continuous or explicitly thresholded speaker activity."""
        states = asr_encoded.transpose(1, 2)
        spk_targets = self._downsample_high_resolution_diarization_for_fusion(spk_targets, states.shape[1])
        spk_targets = self._align_diar_frames(spk_targets, states.shape[1]).to(
            device=states.device, dtype=states.dtype
        )
        if use_diarization is not None:
            if diarization_preds is None:
                raise ValueError("diarization_preds are required when use_diarization is provided.")
            if use_diarization.numel() != states.shape[0]:
                raise ValueError("use_diarization must contain one value per batch row.")
            diarization_preds = self._downsample_high_resolution_diarization_for_fusion(
                diarization_preds, states.shape[1]
            )
            diarization_preds = self._align_diar_frames(diarization_preds, states.shape[1]).to(
                device=states.device, dtype=states.dtype
            )
            spk_targets = torch.where(
                use_diarization.to(device=states.device, dtype=torch.bool).view(-1, 1, 1),
                diarization_preds,
                spk_targets,
            )

        speaker_features = self._speaker_features(spk_targets, states.dtype)
        normalized_states = self.asr_norm(states)
        normalized_targets = self.diar_norm(speaker_features)
        infusion = torch.matmul(normalized_targets, self.diar_kernel.to(normalized_targets.dtype))
        return (normalized_states + getattr(self, "spk_kernel_scale", 1.0) * infusion).transpose(1, 2)

    @contextlib.contextmanager
    def online_inference(self, enabled: bool = True):
        """Route ``forward`` through the windowed generation path inside this scope."""
        previous = getattr(self, "online_inference_enabled", None)
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    def forward(self, audio_signal, length, spk_targets=None):
        """Encode mels and fuse RTTM or Sortformer speaker activity."""
        if spk_targets is not None:
            use_online = False
        elif getattr(self, "online_inference_enabled", None) is not None:
            use_online = bool(self.online_inference_enabled) and self.online_inference_length > 0
        elif self.online_inference_length > 0 and not self.training:
            use_online = audio_signal.shape[-1] > self.chunk_feat_len
        else:
            use_online = False
        runner = self._forward_online if use_online else self._forward
        return runner(audio_signal=audio_signal, length=length, spk_targets=spk_targets)

    def _align_diarization_output_resolution(
        self, predictions: torch.Tensor, embedding_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Map native Sortformer probabilities onto the ASR fusion grid."""
        model = self.diarization_model
        native_factor = 1 if model.high_resolution else int(model.encoder.subsampling_factor)
        downsample_factor = int(model.output_subsampling_factor) // native_factor
        if downsample_factor <= 1:
            return predictions
        native_lengths = embedding_lengths * (int(model.encoder.subsampling_factor) // native_factor)
        return model.sortformer_modules.downsample_preds(predictions, downsample_factor, lengths=native_lengths)

    def _downsample_high_resolution_diarization_for_fusion(
        self, predictions: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Pool unaligned high-resolution Sortformer probabilities exactly once."""
        model = getattr(self, "diarization_model", None)
        if model is None:
            return predictions
        if not model.high_resolution or predictions.shape[1] <= target_len:
            return predictions
        downsample_factor = int(model.output_subsampling_factor)
        predictions = model.sortformer_modules.downsample_preds(predictions, downsample_factor)
        if predictions.shape[1] != target_len:
            raise RuntimeError(
                "High-resolution Sortformer predictions did not align with the ASR grid after "
                f"{downsample_factor}x downsampling: diar={predictions.shape[1]} "
                f"asr={target_len}."
            )
        return predictions

    def _run_diarization(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        if self.diar_normalize_type:
            audio_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.diar_normalize_type)
        diar_signal = self._match_module_io(audio_signal, self.diarization_model)
        diar_length = length.to(device=diar_signal.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_diar):
            embeddings, embedding_lengths = self.diarization_model.frontend_encoder(
                processed_signal=diar_signal,
                processed_signal_length=diar_length,
                bypass_pre_encode=False,
            )
            predictions = self.diarization_model.forward_infer(
                emb_seq=embeddings,
                emb_seq_length=embedding_lengths,
            )
            return self._align_diarization_output_resolution(predictions, embedding_lengths)

    def _run_asr(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.asr_normalize_type:
            audio_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.asr_normalize_type)
        audio_signal = self._match_module_io(audio_signal, self.asr_encoder)
        length = length.to(device=audio_signal.device)

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_asr):
            return self.asr_encoder(audio_signal=audio_signal, length=length)

    def _forward(self, audio_signal, length, spk_targets=None):
        """Single-pass forward used by training and validation."""
        self._check_spk_targets(spk_targets, audio_signal.shape[0])
        use_diarization = None if spk_targets is None else self._missing_target_rows(spk_targets)
        needs_diarization = self._should_run_diarization(spk_targets, use_diarization)
        diarization_preds = self._run_diarization(audio_signal, length) if needs_diarization else None
        asr_encoded, asr_encoded_len = self._run_asr(audio_signal, length)

        if spk_targets is None:
            spk_targets = diarization_preds
        elif not needs_diarization:
            use_diarization = None
        output = self._fuse_diar_and_asr(
            asr_encoded,
            spk_targets,
            diarization_preds=diarization_preds,
            use_diarization=use_diarization,
        )
        return output, asr_encoded_len

    def _forward_online(self, audio_signal, length, spk_targets=None):
        """Run both branches over context-extended long-form windows."""
        self._check_spk_targets(spk_targets, audio_signal.shape[0])
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / self.chunk_feat_len))

        if self.asr_normalize_type:
            asr_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.asr_normalize_type)
        else:
            asr_signal = audio_signal
        asr_signal = self._match_module_io(asr_signal, self.asr_encoder)
        asr_length = length.to(device=asr_signal.device)

        run_streaming_diar = spk_targets is None
        use_diarization = None
        if spk_targets is not None:
            use_diarization = self._missing_target_rows(spk_targets)
            run_streaming_diar = bool(use_diarization.any().item())
            if not run_streaming_diar:
                use_diarization = None

        if run_streaming_diar:
            if self.diar_normalize_type:
                diar_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.diar_normalize_type)
            else:
                diar_signal = audio_signal
            streaming_state, stream_dtype, diar_signal, diar_length = self._init_streaming_diar(
                diar_signal, length, batch_size=audio_signal.shape[0]
            )
            total_preds = torch.zeros(
                (diar_signal.shape[0], 0, self.n_spk),
                device=diar_signal.device,
                dtype=stream_dtype,
            )

        asr_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        # The window loop uses the longest row to keep every batch tensor
        # rectangular. Report each row's actual output length independently;
        # adding the longest row's scalar core length to every row would expose
        # padded frames as valid for shorter audios.
        valid_feat_lengths = length.clamp(max=audio_signal.shape[-1])
        encoded_len = torch.as_tensor(
            [self._asr_output_frame_boundary(int(row_len)) for row_len in valid_feat_lengths.detach().cpu()],
            dtype=asr_length.dtype,
            device=asr_length.device,
        )
        for chunk_index in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, "_suppress_online_pbar", False),
        ):
            start = chunk_index * self.chunk_feat_len
            end = min(start + self.chunk_feat_len, total_feat_len)
            context_start = max(start - self.left_ctx_feat_len, 0)
            context_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset = start - context_start
            right_offset = context_end - end

            asr_chunk = asr_signal[:, :, context_start:context_end]
            chunk_length = (asr_length - context_start).clamp(min=0, max=context_end - context_start)
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_asr):
                encoded_context, _ = self.asr_encoder(audio_signal=asr_chunk, length=chunk_length)
            left_drop = left_offset // self.subsampling_factor
            core_len = self._asr_output_frame_boundary(end) - self._asr_output_frame_boundary(start)
            core_len = max(0, min(core_len, encoded_context.shape[-1] - left_drop))
            asr_chunks.append(encoded_context[:, :, left_drop : left_drop + core_len])

            if run_streaming_diar:
                previous_len = total_preds.shape[1]
                # Sortformer's streaming boundary is time-major for every
                # supported pre-encoder. Its internal adapter performs any
                # FeatureStacking-specific channel-first conversion.
                diar_chunk = diar_signal[:, :, context_start:context_end].transpose(1, 2)
                diar_chunk_length = (diar_length - context_start).clamp(min=0, max=context_end - context_start)
                with (
                    torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_diar),
                    _disable_dist_feature_sync(),
                    _default_dtype(stream_dtype),
                ):
                    streaming_state, total_preds = self.diarization_model.forward_streaming_step(
                        processed_signal=diar_chunk,
                        processed_signal_length=diar_chunk_length,
                        streaming_state=streaming_state,
                        total_preds=total_preds,
                        left_offset=left_offset,
                        right_offset=right_offset,
                    )
                diar_chunks.append(self._align_diar_frames(total_preds[:, previous_len:], core_len))

        asr_encoded = torch.cat(asr_chunks, dim=2)
        diarization_preds = torch.cat(diar_chunks, dim=1) if run_streaming_diar else None
        if spk_targets is None:
            spk_targets = diarization_preds
            use_diarization = None
        output = self._fuse_diar_and_asr(
            asr_encoded,
            spk_targets,
            diarization_preds=diarization_preds,
            use_diarization=use_diarization,
        )
        return output, encoded_len

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        modules = self.diarization_model.sortformer_modules
        modules.chunk_len = self.online_inference_length
        modules.fifo_len = self.diar_fifo_len
        modules.spkcache_update_period = self.diar_spkcache_update_period
        modules.spkcache_len = self.diar_spkcache_len
        check_streaming_parameters = getattr(
            self.diarization_model,
            "_check_streaming_parameters",
            modules._check_streaming_parameters,
        )
        check_streaming_parameters()

        parameter = next(self.diarization_model.parameters(), None)
        if parameter is None:
            device = audio_signal.device
            stream_dtype = torch.get_default_dtype()
        else:
            device = parameter.device
            stream_dtype = parameter.dtype
            # Refresh the nested ModelPT/Lightning device tracker. Sortformer's
            # streaming path uses ``self.device`` when assembling chunk state,
            # which can otherwise remain stale after moving the parent encoder.
            self.diarization_model.to(device)
        diar_signal = audio_signal.to(device=device, dtype=stream_dtype)
        diar_length = length.to(device=device)
        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            state = modules.init_streaming_state(
                batch_size=batch_size,
                async_streaming=self.diarization_model.async_streaming,
                device=device,
            )
        return state, stream_dtype, diar_signal, diar_length


class TransformerCTCDecoder(ConvASRDecoder):
    """CTC decoder with an optional Transformer bridge before the CTC convolution.

    The inherited ConvASRDecoder supplies the final 1x1 convolution and CTC
    log-softmax. Enabling use_transformer inserts a length-aware dense
    TransformerEncoder between encoder states and that convolution. Disabling it
    produces the equivalent Conv-only CTC head, which makes the two alternatives
    directly comparable.

    d_model is intentionally constrained to feat_in. This keeps the head strictly
    Transformer+Conv (or Conv-only), without a separate Linear projection layer.
    """

    requires_encoded_lengths = False

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        init_mode: str = "xavier_uniform",
        vocabulary: Optional[List[str]] = None,
        add_blank: bool = True,
        use_transformer: bool = True,
        d_model: Optional[int] = None,
        n_heads: int = 8,
        n_layers: int = 2,
        drop_rate: float = 0.1,
        dropout_pre_encoder: Optional[float] = None,
        dropout_emb: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        ff_expansion: float = 4.0,
        pre_block_norm: bool = True,
        self_attention_model: Optional[str] = "rope",
        rope_base: float = 10000.0,
        rotary_fraction: float = 1.0,
        pos_emb_max_len: int = 5000,
        xscaling: bool = False,
        attn_mode: str = "full",
        sync_max_audio_length: bool = True,
        residual: bool = False,
        residual_scale: float = 1.0,
        learnable_residual_scale: bool = False,
    ):
        if residual and not use_transformer:
            raise ValueError("TransformerCTCDecoder residual connections require use_transformer=True.")

        super().__init__(
            feat_in=feat_in,
            num_classes=num_classes,
            init_mode=init_mode,
            vocabulary=vocabulary,
            add_blank=add_blank,
        )

        self.use_transformer = use_transformer
        self.requires_encoded_lengths = use_transformer
        self.transformer = None
        self.residual = residual

        if self.use_transformer:
            transformer_d_model = feat_in if d_model is None else int(d_model)
            if transformer_d_model != feat_in:
                raise ValueError(
                    "TransformerCTCDecoder requires d_model to equal feat_in so the head remains "
                    "Transformer+Conv without a Linear projection. "
                    f"Received d_model={transformer_d_model} and feat_in={feat_in}."
                )

            self.transformer = TransformerEncoder(
                feat_in=feat_in,
                d_model=feat_in,
                n_heads=n_heads,
                n_layers=n_layers,
                subsampling=None,
                subsampling_factor=1,
                drop_rate=drop_rate,
                dropout_pre_encoder=dropout_pre_encoder,
                dropout_emb=dropout_emb,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                ff_expansion=ff_expansion,
                pre_block_norm=pre_block_norm,
                self_attention_model=self_attention_model,
                rope_base=rope_base,
                rotary_fraction=rotary_fraction,
                pos_emb_max_len=pos_emb_max_len,
                xscaling=xscaling,
                attn_mode=attn_mode,
                sync_max_audio_length=sync_max_audio_length,
            )
            # The bridge consumes already encoded frame states through
            # TransformerEncoder's bypass_pre_encode path, so it has no
            # pre-encoder to train or checkpoint.
            self.transformer.pre_encode = nn.Identity()

            if residual:
                if learnable_residual_scale:
                    self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
                else:
                    self.register_buffer("residual_scale", torch.tensor(float(residual_scale)), persistent=True)

    def forward(
        self, encoder_output: torch.Tensor, encoded_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return CTC log probabilities from channels-first encoder states.

        Args:
            encoder_output: Acoustic states shaped (B, D, T).
            encoded_lengths: Valid frame counts shaped (B,). Required when
                use_transformer=True so padded frames cannot affect the Transformer bridge.

        Returns:
            Log probabilities shaped (B, T, num_classes_with_blank).
        """
        if encoder_output.ndim != 3:
            raise ValueError(
                "TransformerCTCDecoder expects encoder_output with shape (B, D, T), "
                f"but got {tuple(encoder_output.shape)}."
            )
        if encoder_output.shape[1] != self._feat_in:
            raise ValueError(
                f"TransformerCTCDecoder expected {self._feat_in} encoder features, "
                f"but got {encoder_output.shape[1]}."
            )

        encoded = encoder_output
        if self.use_transformer:
            if encoded_lengths is None:
                raise ValueError("TransformerCTCDecoder requires encoded_lengths when use_transformer=True.")

            residual_input = encoded
            encoded, _ = self.transformer(
                audio_signal=encoded.transpose(1, 2),
                length=encoded_lengths,
                bypass_pre_encode=True,
            )

            if self.residual:
                encoded = residual_input + self.residual_scale.to(dtype=encoded.dtype) * encoded

        return super().forward(encoder_output=encoded)
