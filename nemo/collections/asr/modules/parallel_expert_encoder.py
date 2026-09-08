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
import re
import tarfile
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    "PEETransformerCTCTimestampExtractor",
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


_UNSET_PARALLEL_SPEAKER_GATE = object()


class PEETransformerCTCTimestampExtractor:
    """Extract speaker-specific word timestamps from PEE and a CTC timestamp head.

    This is intentionally an inference helper, not a new PEE encoder.  It keeps the
    supplied :class:`ParallelExpertEncoder` and :class:`TransformerCTCDecoder`
    untouched, runs the PEE ASR expert, and combines three already synchronized
    signals:

    * CTC log-probabilities from ``TransformerCTCDecoder``;
    * raw (pre-threshold) Sortformer speaker sigmoids; and
    * a Nemotron-Transcribe t-SOT string containing ``<spk:N>`` tags.

    CTC alignment follows the usual blank-expanded Viterbi dynamic program used by
    NeMo Forced Aligner. ``serialized`` is the default: the original t-SOT word
    order shares one monotonic CTC path. In explicit ``parallel`` mode, each tagged
    speaker stream is independently aligned against the same CTC frames, so
    overlapping timestamps are possible. Model inference is still shared once per
    audio record and the independent speaker DPs are evaluated in one padded batch.

    Args:
        encoder: Archive-compatible two-branch PEE module, or its
            ``ParallelExpertEncoderPT`` wrapper. The extractor runs the raw ASR and
            Sortformer branches directly and does not use PEE's fused forward output.
        ctc_decoder: The separately loaded :class:`TransformerCTCDecoder` head.
        tokenizer: The BPE tokenizer used to train the CTC head.  It must expose
            ``text_to_ids(str)``.
        blank_id: CTC blank index.  Defaults to the decoder's last class.
        input_frame_seconds: Input mel frame shift, used with the PEE subsampling
            factor when an audio duration is not supplied.  PEE normally uses 0.01.
        ctc_frame_seconds: Optional explicit CTC frame shift.
        sortformer_frame_seconds: Optional explicit Sortformer frame shift.
        speaker_activity_threshold: Threshold used only to expose optional speaker
            activity subspans; raw sigmoid values remain in the DP.
        speaker_logprob_weight: Weight of ``log(sigmoid)`` added to CTC token-state
            emissions. A value of one is the literal CTC-probability ×
            ``max(activity, epsilon)`` product; zero disables the soft gate.
        parallel_speaker_gate_threshold: Optional hard Sortformer gate for parallel
            alignment. Non-blank token emissions below this probability are disabled
            while CTC blank emissions remain available. Set ``None`` to use only the
            soft gate.
        alignment_mode: ``'serialized'`` (default) or ``'parallel'``.
        speaker_assignment_mode: ``'optimal'`` learns a per-audio one-to-one mapping
            from t-SOT tags to Sortformer columns using an initial CTC alignment;
            ``'identity'`` uses tag ``N`` as Sortformer column ``N``.
    """

    _SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        ctc_decoder: Optional[TransformerCTCDecoder] = None,
        tokenizer: Optional[Any] = None,
        *,
        blank_id: Optional[int] = None,
        input_frame_seconds: float = 0.01,
        ctc_frame_seconds: Optional[float] = None,
        sortformer_frame_seconds: Optional[float] = None,
        speaker_activity_threshold: float = 0.5,
        speaker_logprob_weight: float = 0.25,
        parallel_speaker_gate_threshold: Optional[float] = 0.5,
        alignment_mode: str = 'serialized',
        speaker_assignment_mode: str = 'optimal',
        epsilon: float = 1.0e-6,
    ):
        if input_frame_seconds <= 0:
            raise ValueError(f"input_frame_seconds must be positive, got {input_frame_seconds}.")
        if ctc_frame_seconds is not None and ctc_frame_seconds <= 0:
            raise ValueError(f"ctc_frame_seconds must be positive, got {ctc_frame_seconds}.")
        if sortformer_frame_seconds is not None and sortformer_frame_seconds <= 0:
            raise ValueError(f"sortformer_frame_seconds must be positive, got {sortformer_frame_seconds}.")
        if not 0.0 <= speaker_activity_threshold <= 1.0:
            raise ValueError("speaker_activity_threshold must be between zero and one.")
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        if parallel_speaker_gate_threshold is not None and not 0.0 <= float(parallel_speaker_gate_threshold) <= 1.0:
            raise ValueError("parallel_speaker_gate_threshold must be between zero and one or None.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.encoder = encoder
        self.ctc_decoder = ctc_decoder
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.input_frame_seconds = float(input_frame_seconds)
        self.ctc_frame_seconds = ctc_frame_seconds
        self.sortformer_frame_seconds = sortformer_frame_seconds
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.speaker_logprob_weight = float(speaker_logprob_weight)
        self.parallel_speaker_gate_threshold = (
            None if parallel_speaker_gate_threshold is None else float(parallel_speaker_gate_threshold)
        )
        self.alignment_mode = self._validate_alignment_mode(alignment_mode)
        self.speaker_assignment_mode = self._validate_assignment_mode(speaker_assignment_mode)
        self.epsilon = float(epsilon)

    @classmethod
    def parse_sot_words(cls, sot_transcript: str) -> List[Dict[str, Any]]:
        """Parse a t-SOT transcript without sending speaker tags to the tokenizer.

        A tag remains active until the next tag.  Text that precedes the first tag
        is retained with ``speaker_tag=None`` rather than being silently assigned to
        speaker zero.
        """
        if not isinstance(sot_transcript, str):
            raise TypeError(f"sot_transcript must be a string, got {type(sot_transcript).__name__}.")

        words: List[Dict[str, Any]] = []
        active_speaker: Optional[int] = None
        cursor = 0

        def append_words(text: str, speaker_tag: Optional[int]) -> None:
            for word in re.findall(r"\S+", text):
                words.append(
                    {
                        'word': word,
                        'speaker_tag': speaker_tag,
                        'word_index': len(words),
                    }
                )

        for match in cls._SPEAKER_TAG_RE.finditer(sot_transcript):
            append_words(sot_transcript[cursor : match.start()], active_speaker)
            active_speaker = int(match.group(1))
            cursor = match.end()
        append_words(sot_transcript[cursor:], active_speaker)
        return words

    def extract_ctc_and_sortformer(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run the raw v3 PEE experts and the Transformer CTC timestamp head.

        The CTC head was trained on the ASR expert states, not on PEE's fused
        output. Therefore this intentionally calls the two PEE branches
        directly: _run_asr supplies acoustic states to the CTC decoder and
        _run_diarization supplies unthresholded Sortformer sigmoid activity.
        The public PEE forward method is not used because it fuses those two
        signals for the language-model encoder.

        Timestamp extraction operates on one recording at a time, so its t-SOT
        transcript and Sortformer columns cannot be associated with another row.
        """
        if self.encoder is None:
            raise ValueError("encoder is required to run PEE inference.")
        if self.ctc_decoder is None:
            raise ValueError("ctc_decoder is required to run PEE inference.")
        if not isinstance(processed_signal, torch.Tensor) or processed_signal.ndim != 3:
            raise ValueError("processed_signal must have shape (1, features, frames).")
        if not isinstance(processed_signal_length, torch.Tensor) or processed_signal_length.numel() != 1:
            raise ValueError("processed_signal_length must contain exactly one recording length.")
        if processed_signal.shape[0] != 1:
            raise ValueError(
                "PEETransformerCTCTimestampExtractor aligns one recording per call; "
                f"got batch size {processed_signal.shape[0]}."
            )

        # ParallelExpertEncoderPT exposes its inner module as .encoder; direct
        # PEE instances are already the callable object required here.
        pee_encoder = getattr(self.encoder, "encoder", self.encoder)
        if not isinstance(pee_encoder, nn.Module):
            raise TypeError(
                "encoder must be a ParallelExpertEncoder or ParallelExpertEncoderPT wrapper; "
                f"got {type(pee_encoder).__name__}."
            )
        if not hasattr(pee_encoder, "_run_asr") or not hasattr(pee_encoder, "_run_diarization"):
            raise TypeError(
                "PEETransformerCTCTimestampExtractor requires the two-branch "
                "ParallelExpertEncoder API with _run_asr() and _run_diarization()."
            )

        modules: List[nn.Module] = [pee_encoder, self.ctc_decoder]
        previous_modes = [(module, module.training) for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.inference_mode():
                # Do not call pee_encoder(...): it returns the fused PEE state,
                # whereas this CTC decoder expects the raw ASR expert state.
                speech_states, speech_lengths = pee_encoder._run_asr(
                    processed_signal, processed_signal_length
                )
                sortformer_sigmoids = pee_encoder._run_diarization(
                    processed_signal, processed_signal_length
                )
                ctc_log_probs = self.ctc_decoder(
                    speech_states,
                    encoded_lengths=speech_lengths,
                )
        finally:
            for module, was_training in previous_modes:
                module.train(was_training)

        if not isinstance(speech_states, torch.Tensor) or not isinstance(speech_lengths, torch.Tensor):
            raise RuntimeError("PEE _run_asr() must return (states, lengths) tensors.")
        if not isinstance(sortformer_sigmoids, torch.Tensor):
            raise RuntimeError("PEE _run_diarization() must return Sortformer sigmoid probabilities.")
        if sortformer_sigmoids.ndim != 3:
            raise RuntimeError(
                "PEE _run_diarization() must return (batch, frames, speakers), "
                f"got {tuple(sortformer_sigmoids.shape)}."
            )
        if sortformer_sigmoids.shape[0] != speech_states.shape[0]:
            raise RuntimeError(
                "ASR and Sortformer branches returned different batch sizes: "
                f"{speech_states.shape[0]} vs {sortformer_sigmoids.shape[0]}."
            )
        if sortformer_sigmoids.shape[1] == 0:
            raise RuntimeError("PEE _run_diarization() returned no Sortformer frames.")

        # Sortformer is normally aligned onto the ASR grid by PEE, but its padded
        # length can differ by one rounding frame. Preserve its actual frame count;
        # the DP layer resamples it onto the CTC grid when necessary.
        sortformer_lengths = torch.full_like(
            speech_lengths,
            fill_value=int(sortformer_sigmoids.shape[1]),
        )
        return {
            "ctc_log_probs": ctc_log_probs,
            "ctc_lengths": speech_lengths,
            "sortformer_sigmoids": sortformer_sigmoids,
            "sortformer_lengths": sortformer_lengths,
        }

    def extract_from_audio(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        preprocessor: nn.Module,
        sot_transcript: str,
        *,
        audio_duration: Optional[float] = None,
        time_offset: float = 0.0,
        **alignment_kwargs: Any,
    ) -> Dict[str, Any]:
        """Preprocess one waveform, run PEE + CTC, and return word timestamps.

        ``time_offset`` should be the JSONL record's ``offset`` when timestamps are
        required on the original recording timeline; the default returns chunk-local
        seconds.
        """
        if not isinstance(preprocessor, nn.Module):
            raise TypeError(f"preprocessor must be an nn.Module, got {type(preprocessor).__name__}.")

        was_training = preprocessor.training
        try:
            preprocessor.eval()
            with torch.inference_mode():
                preprocessor_result = preprocessor(input_signal=input_signal, length=input_signal_length)
        finally:
            preprocessor.train(was_training)

        if not isinstance(preprocessor_result, tuple) or len(preprocessor_result) != 2:
            raise RuntimeError("preprocessor must return (processed_signal, processed_signal_length).")
        processed_signal, processed_signal_length = preprocessor_result
        model_outputs = self.extract_ctc_and_sortformer(processed_signal, processed_signal_length)

        if audio_duration is None:
            sample_rate = getattr(preprocessor, '_sample_rate', getattr(preprocessor, 'sample_rate', None))
            if sample_rate is not None:
                audio_duration = self._scalar_length(input_signal_length, 'input_signal_length') / float(sample_rate)

        return self.extract_from_outputs(
            sot_transcript=sot_transcript,
            audio_duration=audio_duration,
            time_offset=time_offset,
            **model_outputs,
            **alignment_kwargs,
        )

    def extract_from_outputs(
        self,
        ctc_log_probs: torch.Tensor,
        sortformer_sigmoids: Optional[torch.Tensor],
        sot_transcript: str,
        *,
        ctc_lengths: Optional[torch.Tensor] = None,
        sortformer_lengths: Optional[torch.Tensor] = None,
        audio_duration: Optional[float] = None,
        time_offset: float = 0.0,
        alignment_mode: Optional[str] = None,
        speaker_assignment_mode: Optional[str] = None,
        speaker_logprob_weight: Optional[float] = None,
        parallel_speaker_gate_threshold: Any = _UNSET_PARALLEL_SPEAKER_GATE,
    ) -> Dict[str, Any]:
        """Force-align a single t-SOT transcript from precomputed model outputs.

        Args:
            ctc_log_probs: CTC ``log_softmax`` output with shape ``(T, V)`` or
                ``(1, T, V)``.  It is cast to CPU fp32 for numerically stable DP.
            sortformer_sigmoids: Raw Sortformer sigmoid output with shape ``(T, S)``
                or ``(1, T, S)``.  It may have a different frame count from CTC;
                it is interpolated onto the CTC timeline for alignment.
            sot_transcript: Nemotron-Transcribe output containing ``<spk:N>`` tags.
            ctc_lengths: Valid CTC frames.  Batch size greater than one is rejected
                deliberately so every result is tied to one audio record.
            sortformer_lengths: Valid Sortformer frames.
            audio_duration: Chunk duration in seconds.  When supplied, it controls
                the timestamp grid exactly; otherwise the PEE frame shift is used.
            time_offset: Added to output seconds, e.g. JSONL ``offset``.
            alignment_mode: Per-call ``'parallel'`` or ``'serialized'`` override.
            speaker_assignment_mode: Per-call ``'optimal'`` or ``'identity'``
                Sortformer-column mapping override.
            speaker_logprob_weight: Per-call Sortformer DP-prior weight override.
            parallel_speaker_gate_threshold: Per-call hard token-emission gate for
                ``parallel`` alignment. Omit it to use the extractor default; pass
                ``None`` to disable the hard gate for this call.

        Returns:
            A dictionary whose ``speaker_word_timestamps`` contains one ordered list
            per t-SOT speaker tag.  Word ``start`` / ``end`` are CTC-derived seconds;
            Sortformer details are supplied as activity/confidence metadata.
        """
        mode = self._validate_alignment_mode(alignment_mode or self.alignment_mode)
        assignment_mode = self._validate_assignment_mode(
            speaker_assignment_mode or self.speaker_assignment_mode
        )
        speaker_weight = self.speaker_logprob_weight if speaker_logprob_weight is None else speaker_logprob_weight
        if speaker_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        parallel_gate_threshold = (
            self.parallel_speaker_gate_threshold
            if parallel_speaker_gate_threshold is _UNSET_PARALLEL_SPEAKER_GATE
            else parallel_speaker_gate_threshold
        )
        if parallel_gate_threshold is not None and not 0.0 <= float(parallel_gate_threshold) <= 1.0:
            raise ValueError("parallel_speaker_gate_threshold must be between zero and one or None.")

        ctc = self._single_recording_tensor(ctc_log_probs, 'ctc_log_probs', expected_ndim=2)
        if ctc.shape[0] == 0 or ctc.shape[1] < 2:
            raise ValueError(f"ctc_log_probs must have at least one frame and two classes, got {tuple(ctc.shape)}.")
        ctc_length = self._select_length(ctc_lengths, ctc.shape[0], 'ctc_lengths')
        ctc = ctc[:ctc_length].detach().to(device='cpu', dtype=torch.float32)
        if torch.isnan(ctc).any():
            raise ValueError("ctc_log_probs contains NaN values.")
        ctc_log_normalizer_error = float(torch.logsumexp(ctc, dim=-1).abs().max().item())
        if ctc_log_normalizer_error > 0.05:
            raise ValueError(
                "ctc_log_probs does not appear to be log-softmax output: maximum "
                f"log-normalization error is {ctc_log_normalizer_error:.4f}."
            )
        blank_id = self._resolve_blank_id(ctc.shape[1])

        sortformer: Optional[torch.Tensor] = None
        sortformer_length: Optional[int] = None
        if sortformer_sigmoids is not None:
            sortformer = self._single_recording_tensor(
                sortformer_sigmoids,
                'sortformer_sigmoids',
                expected_ndim=2,
            )
            if sortformer.shape[1] == 0:
                raise ValueError("sortformer_sigmoids must contain at least one speaker column.")
            sortformer_length = self._select_length(
                sortformer_lengths,
                sortformer.shape[0],
                'sortformer_lengths',
            )
            sortformer = sortformer[:sortformer_length].detach().to(device='cpu', dtype=torch.float32)
            if torch.isnan(sortformer).any():
                raise ValueError("sortformer_sigmoids contains NaN values.")
            if float(sortformer.min().item()) < -1.0e-3 or float(sortformer.max().item()) > 1.001:
                raise ValueError("sortformer_sigmoids must contain sigmoid probabilities in [0, 1].")
            sortformer = sortformer.clamp(min=0.0, max=1.0)
            sortformer_on_ctc = self._resample_speaker_probs(sortformer, ctc_length)
        else:
            sortformer_on_ctc = None

        if audio_duration is not None:
            audio_duration = float(audio_duration)
            if audio_duration <= 0:
                raise ValueError(f"audio_duration must be positive, got {audio_duration}.")
        time_offset = float(time_offset)
        ctc_step_seconds = self._resolve_ctc_frame_seconds(ctc_length, audio_duration)
        sortformer_step_seconds = self._resolve_sortformer_frame_seconds(
            sortformer_length,
            audio_duration,
            ctc_step_seconds,
        )

        tokenized_words = self._tokenize_words(
            self.parse_sot_words(sot_transcript),
            blank_id,
            alignment_mode=mode,
        )
        speaker_tags = self._speaker_tags_in_order(tokenized_words)
        if not tokenized_words:
            return {
                'speaker_word_timestamps': {},
                'speaker_tag_to_sortformer_column': {},
                'alignment_mode': mode,
                'speaker_assignment_mode': assignment_mode,
                'ctc_frame_seconds': ctc_step_seconds,
                'sortformer_frame_seconds': sortformer_step_seconds,
                'time_offset': time_offset,
                'num_ctc_frames': ctc_length,
                'num_sortformer_frames': sortformer_length,
                'ctc_log_normalizer_error': ctc_log_normalizer_error,
                'alignment_diagnostics': {},
            }

        # Learn the t-SOT-tag-to-Sortformer-column mapping from pure CTC paths.
        # In parallel mode these preliminary paths are also independent, avoiding
        # any artificial ordering constraint for overlapping speaker transcripts.
        preliminary_scores: Dict[Optional[int], float] = {}
        if mode == 'serialized':
            preliminary_rows, preliminary_score = self._align_word_sequence(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=None,
                speaker_mapping={},
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=0.0,
            )
            preliminary_scores[None] = preliminary_score
        else:
            preliminary_rows, preliminary_scores = self._align_parallel_word_streams_batched(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=None,
                speaker_mapping={},
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=0.0,
                speaker_gate_threshold=None,
            )

        speaker_mapping, assignment_scores = self._resolve_speaker_mapping(
            speaker_tags=speaker_tags,
            preliminary_rows=preliminary_rows,
            speaker_probs=sortformer_on_ctc,
            assignment_mode=assignment_mode,
        )

        alignment_scores: Dict[Optional[int], float] = {}
        if mode == 'serialized':
            rows, path_score = self._align_word_sequence(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=sortformer_on_ctc,
                speaker_mapping=speaker_mapping,
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=float(speaker_weight),
                speaker_gate_threshold=None,
            )
            alignment_scores[None] = path_score
        else:
            rows, alignment_scores = self._align_parallel_word_streams_batched(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=sortformer_on_ctc,
                speaker_mapping=speaker_mapping,
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=float(speaker_weight),
                speaker_gate_threshold=parallel_gate_threshold,
            )

        speaker_word_timestamps: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for row in rows:
            speaker_word_timestamps.setdefault(row['speaker_tag'], []).append(row)

        return {
            'speaker_word_timestamps': speaker_word_timestamps,
            'speaker_tag_to_sortformer_column': speaker_mapping,
            'alignment_mode': mode,
            'speaker_assignment_mode': assignment_mode,
            'ctc_frame_seconds': ctc_step_seconds,
            'sortformer_frame_seconds': sortformer_step_seconds,
            'time_offset': time_offset,
            'num_ctc_frames': ctc_length,
            'num_sortformer_frames': sortformer_length,
            'ctc_log_normalizer_error': ctc_log_normalizer_error,
            'alignment_diagnostics': {
                'preliminary_ctc_path_scores': preliminary_scores,
                'final_path_scores': alignment_scores,
                'speaker_assignment_scores': assignment_scores,
                'parallel_speaker_gate_threshold': parallel_gate_threshold if mode == 'parallel' else None,
            },
        }

    def _tokenize_words(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
        *,
        alignment_mode: str,
    ) -> List[Dict[str, Any]]:
        """Tokenize each CTC target stream while retaining exact word ownership.

        SentencePiece can choose different subword decompositions when a word is
        encoded alone versus in its transcript. We therefore tokenize a complete
        target stream once, use SentencePiece character offsets to assign its exact
        IDs to words, and never send t-SOT speaker tags to the tokenizer. Parallel
        mode tokenizes each speaker transcript independently; serialized mode
        tokenizes the original t-SOT word order as one stream.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to force-align t-SOT words.")
        if not hasattr(self.tokenizer, "text_to_ids"):
            raise TypeError("tokenizer must expose a text_to_ids(str) method.")
        if not words:
            return []

        if alignment_mode == "parallel":
            streams = self._group_words_by_speaker(words)
        elif alignment_mode == "serialized":
            streams = {None: list(words)}
        else:
            raise ValueError(f"Unsupported alignment_mode for tokenization: {alignment_mode!r}.")

        tokenized_by_word_index: Dict[int, Dict[str, Any]] = {}
        for stream_words in streams.values():
            stream_token_ids = self._tokenize_word_stream(stream_words, blank_id)
            if len(stream_token_ids) != len(stream_words):
                raise RuntimeError("Tokenizer stream assignment returned the wrong number of words.")
            for word_record, token_ids in zip(stream_words, stream_token_ids):
                item = dict(word_record)
                item["token_ids"] = token_ids
                word_index = int(item["word_index"])
                if word_index in tokenized_by_word_index:
                    raise RuntimeError(f"Duplicate t-SOT word index {word_index}.")
                tokenized_by_word_index[word_index] = item

        return [tokenized_by_word_index[int(word["word_index"])] for word in words]

    def _tokenize_word_stream(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[List[int]]:
        """Return exact full-stream SentencePiece IDs grouped by whitespace word."""
        text = " ".join(str(word["word"]) for word in words)
        if not text:
            return []

        token_ids = self._text_to_valid_ctc_ids(text, blank_id, "t-SOT speaker stream")
        sentencepiece = getattr(self.tokenizer, "tokenizer", None)
        encode_with_offsets = getattr(sentencepiece, "encode_as_immutable_proto", None)
        if not callable(encode_with_offsets):
            encode_with_offsets = getattr(sentencepiece, "EncodeAsImmutableProto", None)
        if not callable(encode_with_offsets):
            raise TypeError(
                "tokenizer must expose SentencePiece immutable-proto offsets to assign "
                "full-stream CTC tokens to individual words."
            )

        proto = encode_with_offsets(text)
        pieces = list(proto.pieces)
        proto_ids = [int(piece.id) for piece in pieces]
        if token_ids != proto_ids:
            raise RuntimeError(
                "tokenizer.text_to_ids() and its SentencePiece offset API produced different IDs; "
                "cannot safely assign CTC tokens to words."
            )

        word_spans: List[Tuple[int, int]] = []
        cursor = 0
        for word_record in words:
            word = str(word_record["word"])
            start = cursor
            end = start + len(word)
            word_spans.append((start, end))
            cursor = end + 1

        ids_by_word: List[List[int]] = [[] for _ in words]
        for piece in pieces:
            begin = int(piece.begin)
            end = int(piece.end)
            overlapping_words = [
                index
                for index, (word_start, word_end) in enumerate(word_spans)
                if begin < word_end and end > word_start
            ]
            if len(overlapping_words) > 1:
                raise RuntimeError(
                    "A SentencePiece token crosses a whitespace word boundary, so "
                    "word-level CTC alignment would be ambiguous."
                )
            if overlapping_words:
                word_index = overlapping_words[0]
            else:
                # A leading SentencePiece separator has zero width at the start of
                # the first word or spans only the inter-word whitespace. It belongs
                # to the following word, which preserves the original ID sequence.
                word_index = next(
                    (
                        index
                        for index, (word_start, _) in enumerate(word_spans)
                        if word_start >= end
                    ),
                    len(word_spans) - 1,
                )
            ids_by_word[word_index].append(int(piece.id))

        for word_record, word_ids in zip(words, ids_by_word):
            if not word_ids:
                raise RuntimeError(
                    f"SentencePiece produced no tokens for word {word_record['word']!r} in the full transcript."
                )
        if [token_id for word_ids in ids_by_word for token_id in word_ids] != token_ids:
            raise RuntimeError("SentencePiece word assignment did not preserve the full-stream token order.")
        return ids_by_word

    def _text_to_valid_ctc_ids(self, text: str, blank_id: int, context: str) -> List[int]:
        """Tokenize one text stream and validate that all IDs are non-blank CTC IDs."""
        try:
            token_ids = self.tokenizer.text_to_ids(text)
        except TypeError as error:
            raise TypeError(
                "tokenizer.text_to_ids(text) failed. Supply the single-language BPE "
                "tokenizer used by the CTC head."
            ) from error
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        token_ids = [int(token_id) for token_id in token_ids]
        if not token_ids:
            raise ValueError(f"Tokenizer produced no CTC tokens for {context}.")
        invalid_ids = [token_id for token_id in token_ids if token_id < 0 or token_id >= blank_id]
        if invalid_ids:
            raise ValueError(
                f"Tokenizer produced CTC-invalid IDs {invalid_ids} for {context}; "
                f"valid non-blank IDs are [0, {blank_id})."
            )
        return token_ids

    @staticmethod
    def _speaker_tags_in_order(tokenized_words: Sequence[Dict[str, Any]]) -> List[int]:
        """Return distinct explicit t-SOT tags in first-occurrence order."""
        tags: List[int] = []
        for word in tokenized_words:
            tag = word['speaker_tag']
            if tag is not None and tag not in tags:
                tags.append(tag)
        return tags

    @staticmethod
    def _group_words_by_speaker(
        tokenized_words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """Preserve t-SOT stream order within each speaker-specific transcript."""
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for word in tokenized_words:
            grouped.setdefault(word['speaker_tag'], []).append(word)
        return grouped

    @staticmethod
    def _build_ctc_target(
        tokenized_words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> Tuple[List[int], List[Optional[int]], List[int]]:
        """Build ``[blank, token, blank, ...]`` labels and token-state ownership."""
        labels: List[int] = [blank_id]
        state_to_word: List[Optional[int]] = [None]
        flat_tokens: List[int] = []
        for word_index, word in enumerate(tokenized_words):
            for token_id in word['token_ids']:
                labels.append(token_id)
                state_to_word.append(word_index)
                labels.append(blank_id)
                state_to_word.append(None)
                flat_tokens.append(token_id)
        return labels, state_to_word, flat_tokens

    @staticmethod
    def _minimum_ctc_frames(token_ids: Sequence[int]) -> int:
        """Minimum CTC frames, including mandatory blanks for equal neighbours."""
        if not token_ids:
            return 0
        repeated_neighbours = sum(
            previous == current for previous, current in zip(token_ids[:-1], token_ids[1:])
        )
        return len(token_ids) + repeated_neighbours

    def _align_word_sequence(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        ctc_log_probs: torch.Tensor,
        blank_id: int,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step_seconds: float,
        time_offset: float,
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        labels, state_to_word, flat_tokens = self._build_ctc_target(tokenized_words, blank_id)
        minimum_frames = self._minimum_ctc_frames(flat_tokens)
        if minimum_frames > ctc_log_probs.shape[0]:
            raise ValueError(
                "CTC target is infeasible: it needs at least "
                f"{minimum_frames} frames (including repeated-token blanks), but only "
                f"{ctc_log_probs.shape[0]} valid CTC frames are available."
            )

        state_speaker_columns: List[Optional[int]] = [None] * len(labels)
        for state_index, local_word_index in enumerate(state_to_word):
            if local_word_index is None:
                continue
            speaker_tag = tokenized_words[local_word_index]['speaker_tag']
            state_speaker_columns[state_index] = speaker_mapping.get(speaker_tag)

        path, path_score = self._ctc_viterbi_align(
            ctc_log_probs=ctc_log_probs,
            labels=labels,
            blank_id=blank_id,
            state_speaker_columns=state_speaker_columns,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
        )
        rows = self._word_rows_from_path(
            tokenized_words=tokenized_words,
            labels=labels,
            state_to_word=state_to_word,
            path=path,
            ctc_log_probs=ctc_log_probs,
            speaker_probs=speaker_probs,
            speaker_mapping=speaker_mapping,
            ctc_step_seconds=ctc_step_seconds,
            time_offset=time_offset,
        )
        return rows, path_score

    def _align_parallel_word_streams_batched(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        ctc_log_probs: torch.Tensor,
        blank_id: int,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step_seconds: float,
        time_offset: float,
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float],
    ) -> Tuple[List[Dict[str, Any]], Dict[Optional[int], float]]:
        """Align independent speaker streams in one padded CTC Viterbi batch.

        All streams share the same CTC frames. Only the target-state axis is padded;
        no ``(speakers, frames, vocabulary)`` tensor is materialized.
        """
        grouped_words = self._group_words_by_speaker(tokenized_words)
        streams: List[Tuple[Optional[int], List[Dict[str, Any]], List[int], List[Optional[int]], List[Optional[int]]]] = []
        for speaker_tag, speaker_words in grouped_words.items():
            speaker_words = list(speaker_words)
            labels, state_to_word, flat_tokens = self._build_ctc_target(speaker_words, blank_id)
            minimum_frames = self._minimum_ctc_frames(flat_tokens)
            if minimum_frames > ctc_log_probs.shape[0]:
                raise ValueError(
                    "CTC target is infeasible for speaker "
                    f"{speaker_tag!r}: it needs at least {minimum_frames} frames but only "
                    f"{ctc_log_probs.shape[0]} are available."
                )
            state_speaker_columns: List[Optional[int]] = [None] * len(labels)
            for state_index, local_word_index in enumerate(state_to_word):
                if local_word_index is not None:
                    state_speaker_columns[state_index] = speaker_mapping.get(
                        speaker_words[local_word_index]['speaker_tag']
                    )
            streams.append((speaker_tag, speaker_words, labels, state_to_word, state_speaker_columns))

        if not streams:
            return [], {}

        num_streams = len(streams)
        max_states = max(len(labels) for _, _, labels, _, _ in streams)
        state_lengths = torch.tensor([len(labels) for _, _, labels, _, _ in streams], dtype=torch.long)
        labels_batch = torch.full((num_streams, max_states), blank_id, dtype=torch.long)
        columns_batch = torch.full((num_streams, max_states), -1, dtype=torch.long)
        for stream_index, (_, _, labels, _, columns) in enumerate(streams):
            labels_batch[stream_index, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            columns_batch[stream_index, : len(columns)] = torch.tensor(
                [-1 if column is None else int(column) for column in columns], dtype=torch.long
            )

        paths, scores = self._ctc_viterbi_align_batched(
            ctc_log_probs=ctc_log_probs,
            labels=labels_batch,
            state_lengths=state_lengths,
            blank_id=blank_id,
            state_speaker_columns=columns_batch,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
        )

        rows: List[Dict[str, Any]] = []
        score_by_speaker: Dict[Optional[int], float] = {}
        for (speaker_tag, speaker_words, labels, state_to_word, _), path, score in zip(streams, paths, scores):
            rows.extend(
                self._word_rows_from_path(
                    tokenized_words=speaker_words,
                    labels=labels,
                    state_to_word=state_to_word,
                    path=path,
                    ctc_log_probs=ctc_log_probs,
                    speaker_probs=speaker_probs,
                    speaker_mapping=speaker_mapping,
                    ctc_step_seconds=ctc_step_seconds,
                    time_offset=time_offset,
                )
            )
            score_by_speaker[speaker_tag] = score
        return rows, score_by_speaker


    def _ctc_viterbi_align(
        self,
        *,
        ctc_log_probs: torch.Tensor,
        labels: Sequence[int],
        blank_id: int,
        state_speaker_columns: Sequence[Optional[int]],
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Run blank-expanded CTC Viterbi DP with an optional Sortformer prior.

        The recurrence is the one used by NeMo Forced Aligner: each state can stay,
        advance one state, or skip a blank when the two surrounding non-blank labels
        differ.  All arithmetic is fp32 on CPU because the PEE/CTC inference path is
        commonly bf16.
        """
        if ctc_log_probs.ndim != 2:
            raise ValueError(f"ctc_log_probs must have shape (T, V), got {tuple(ctc_log_probs.shape)}.")
        if len(labels) != len(state_speaker_columns):
            raise ValueError("labels and state_speaker_columns must have the same length.")
        if not labels:
            raise ValueError("Cannot align an empty CTC target.")

        log_probs = ctc_log_probs.detach().to(device='cpu', dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if int(labels_tensor.max().item()) >= log_probs.shape[1] or int(labels_tensor.min().item()) < 0:
            raise ValueError("CTC target contains labels outside the CTC vocabulary.")
        emissions = log_probs.index_select(dim=1, index=labels_tensor)

        if speaker_gate_threshold is not None and not 0.0 <= float(speaker_gate_threshold) <= 1.0:
            raise ValueError("speaker_gate_threshold must be between zero and one or None.")
        if speaker_probs is not None and (speaker_logprob_weight > 0.0 or speaker_gate_threshold is not None):
            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != log_probs.shape[0]:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
            state_columns = torch.tensor(
                [-1 if column is None else int(column) for column in state_speaker_columns],
                dtype=torch.long,
            )
            valid_states = torch.nonzero(state_columns >= 0, as_tuple=False).flatten()
            if valid_states.numel() > 0:
                speaker_columns = state_columns.index_select(0, valid_states)
                if int(speaker_columns.max().item()) >= speaker_probs.shape[1]:
                    raise ValueError("speaker mapping references a missing Sortformer column.")
                activity = speaker_probs[:, speaker_columns].to(dtype=torch.float32)
                gated_emissions = emissions[:, valid_states]
                if speaker_logprob_weight > 0.0:
                    # This is probability-domain multiplication by activity**weight,
                    # expressed safely in log space. Do not multiply log-probabilities.
                    gated_emissions = gated_emissions + float(speaker_logprob_weight) * torch.log(
                        activity.clamp_min(self.epsilon)
                    )
                if speaker_gate_threshold is not None:
                    gated_emissions = gated_emissions.masked_fill(
                        activity < float(speaker_gate_threshold), -float('inf')
                    )
                emissions[:, valid_states] = gated_emissions

        num_frames, num_states = emissions.shape
        if num_states < 2:
            raise ValueError("CTC target must contain at least blank and one token state.")
        neg_inf = -float('inf')
        previous_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
        previous_scores[0] = emissions[0, 0]
        previous_scores[1] = emissions[0, 1]
        backpointers = torch.full((num_frames, num_states), -1, dtype=torch.long)
        backpointers[0, 0] = 0
        backpointers[0, 1] = 1
        state_indices = torch.arange(num_states, dtype=torch.long)

        for frame_index in range(1, num_frames):
            best_scores = previous_scores.clone()
            best_previous_states = state_indices.clone()

            advance_one_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
            advance_one_scores[1:] = previous_scores[:-1]
            take_advance_one = advance_one_scores > best_scores
            best_scores = torch.where(take_advance_one, advance_one_scores, best_scores)
            best_previous_states = torch.where(take_advance_one, state_indices - 1, best_previous_states)

            if num_states > 2:
                skip_positions = torch.arange(2, num_states, dtype=torch.long)
                can_skip = (labels_tensor[skip_positions] != blank_id) & (
                    labels_tensor[skip_positions] != labels_tensor[skip_positions - 2]
                )
                if can_skip.any():
                    allowed_positions = skip_positions[can_skip]
                    skip_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
                    skip_scores[allowed_positions] = previous_scores[allowed_positions - 2]
                    take_skip = skip_scores > best_scores
                    best_scores = torch.where(take_skip, skip_scores, best_scores)
                    best_previous_states = torch.where(take_skip, state_indices - 2, best_previous_states)

            previous_scores = best_scores + emissions[frame_index]
            backpointers[frame_index] = best_previous_states

        final_state = num_states - 1
        if previous_scores[num_states - 2] > previous_scores[final_state]:
            final_state = num_states - 2
        final_score = previous_scores[final_state]
        if not torch.isfinite(final_score):
            raise ValueError("No valid CTC Viterbi path exists for this transcript and audio.")

        path = torch.empty((num_frames,), dtype=torch.long)
        state = int(final_state)
        for frame_index in range(num_frames - 1, -1, -1):
            path[frame_index] = state
            if frame_index > 0:
                state = int(backpointers[frame_index, state].item())
                if state < 0:
                    raise RuntimeError("CTC Viterbi backtrace reached an invalid state.")
        return path, float(final_score.item())

    def _ctc_viterbi_align_batched(
        self,
        *,
        ctc_log_probs: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        blank_id: int,
        state_speaker_columns: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Run independent CTC Viterbi paths in a padded speaker batch.

        ``labels`` has shape ``(num_speakers, max_states)``. Padded states are
        masked to ``-inf``; the recurrence remains vectorized across speakers and
        states while retaining the normal time-axis loop and exact CTC tie behavior.
        """
        if ctc_log_probs.ndim != 2:
            raise ValueError(f"ctc_log_probs must have shape (T, V), got {tuple(ctc_log_probs.shape)}.")
        if labels.ndim != 2 or state_speaker_columns.shape != labels.shape:
            raise ValueError("labels and state_speaker_columns must have matching shape (num_streams, max_states).")
        if state_lengths.ndim != 1 or state_lengths.shape[0] != labels.shape[0]:
            raise ValueError("state_lengths must have shape (num_streams,).")
        if speaker_gate_threshold is not None and not 0.0 <= float(speaker_gate_threshold) <= 1.0:
            raise ValueError("speaker_gate_threshold must be between zero and one or None.")

        log_probs = ctc_log_probs.detach().to(device='cpu', dtype=torch.float32)
        labels = labels.detach().to(device='cpu', dtype=torch.long)
        state_lengths = state_lengths.detach().to(device='cpu', dtype=torch.long)
        state_speaker_columns = state_speaker_columns.detach().to(device='cpu', dtype=torch.long)
        num_streams, max_states = labels.shape
        num_frames = log_probs.shape[0]
        if num_streams == 0 or max_states < 2 or int(state_lengths.min().item()) < 2:
            raise ValueError("Each CTC stream must contain at least blank and one token state.")
        if int(state_lengths.max().item()) > max_states:
            raise ValueError("state_lengths cannot exceed labels.shape[1].")

        state_mask = torch.arange(max_states, dtype=torch.long).unsqueeze(0) < state_lengths.unsqueeze(1)
        valid_labels = labels[state_mask]
        if valid_labels.numel() == 0 or int(valid_labels.min().item()) < 0 or int(valid_labels.max().item()) >= log_probs.shape[1]:
            raise ValueError("CTC target contains labels outside the CTC vocabulary.")
        emissions = log_probs[:, labels.reshape(-1)].reshape(num_frames, num_streams, max_states)
        emissions = emissions.permute(1, 0, 2).contiguous()
        neg_inf = -float('inf')
        emissions.masked_fill_(~state_mask.unsqueeze(1), neg_inf)

        if speaker_probs is not None and (speaker_logprob_weight > 0.0 or speaker_gate_threshold is not None):
            speaker_probs = speaker_probs.detach().to(device='cpu', dtype=torch.float32)
            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != num_frames:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
            token_states = state_mask & (state_speaker_columns >= 0)
            if token_states.any():
                speaker_columns = state_speaker_columns[token_states]
                if int(speaker_columns.max().item()) >= speaker_probs.shape[1]:
                    raise ValueError("speaker mapping references a missing Sortformer column.")
                safe_columns = state_speaker_columns.clamp_min(0)
                activity = speaker_probs[:, safe_columns.reshape(-1)]
                activity = activity.reshape(num_frames, num_streams, max_states).permute(1, 0, 2)
                if speaker_logprob_weight > 0.0:
                    soft_gate = torch.where(
                        token_states.unsqueeze(1),
                        float(speaker_logprob_weight) * torch.log(activity.clamp_min(self.epsilon)),
                        torch.zeros_like(activity),
                    )
                    emissions = emissions + soft_gate
                if speaker_gate_threshold is not None:
                    inactive_tokens = token_states.unsqueeze(1) & (activity < float(speaker_gate_threshold))
                    emissions.masked_fill_(inactive_tokens, neg_inf)

        previous_scores = torch.full((num_streams, max_states), neg_inf, dtype=torch.float32)
        previous_scores[:, 0] = emissions[:, 0, 0]
        previous_scores[:, 1] = emissions[:, 0, 1]
        backpointers = torch.full((num_frames, num_streams, max_states), -1, dtype=torch.long)
        backpointers[0, :, 0] = 0
        backpointers[0, :, 1] = 1
        state_indices = torch.arange(max_states, dtype=torch.long).unsqueeze(0).expand(num_streams, -1)

        for frame_index in range(1, num_frames):
            best_scores = previous_scores.clone()
            best_previous_states = state_indices.clone()

            advance_one_scores = torch.full_like(previous_scores, neg_inf)
            advance_one_scores[:, 1:] = previous_scores[:, :-1]
            take_advance_one = advance_one_scores > best_scores
            best_scores = torch.where(take_advance_one, advance_one_scores, best_scores)
            best_previous_states = torch.where(take_advance_one, state_indices - 1, best_previous_states)

            if max_states > 2:
                skip_scores = torch.full_like(previous_scores, neg_inf)
                can_skip = (
                    state_mask[:, 2:]
                    & (labels[:, 2:] != blank_id)
                    & (labels[:, 2:] != labels[:, :-2])
                )
                skip_scores[:, 2:] = torch.where(can_skip, previous_scores[:, :-2], skip_scores[:, 2:])
                take_skip = skip_scores > best_scores
                best_scores = torch.where(take_skip, skip_scores, best_scores)
                best_previous_states = torch.where(take_skip, state_indices - 2, best_previous_states)

            previous_scores = best_scores + emissions[:, frame_index, :]
            previous_scores.masked_fill_(~state_mask, neg_inf)
            backpointers[frame_index] = torch.where(state_mask, best_previous_states, -torch.ones_like(best_previous_states))

        last_blank_states = state_lengths - 1
        last_token_states = state_lengths - 2
        last_blank_scores = previous_scores.gather(1, last_blank_states.unsqueeze(1)).squeeze(1)
        last_token_scores = previous_scores.gather(1, last_token_states.unsqueeze(1)).squeeze(1)
        choose_token = last_token_scores > last_blank_scores
        final_states = torch.where(choose_token, last_token_states, last_blank_states)
        final_scores = torch.where(choose_token, last_token_scores, last_blank_scores)
        if not torch.isfinite(final_scores).all():
            failed_streams = torch.nonzero(~torch.isfinite(final_scores), as_tuple=False).flatten().tolist()
            gate_hint = (
                "; lower or disable the hard speaker gate threshold."
                if speaker_gate_threshold is not None
                else "."
            )
            raise ValueError(
                "No valid CTC Viterbi path exists for speaker stream(s) "
                f"{failed_streams}{gate_hint}"
            )

        paths = torch.empty((num_streams, num_frames), dtype=torch.long)
        states = final_states.clone()
        stream_indices = torch.arange(num_streams, dtype=torch.long)
        for frame_index in range(num_frames - 1, -1, -1):
            paths[:, frame_index] = states
            if frame_index > 0:
                states = backpointers[frame_index, stream_indices, states]
                if (states < 0).any():
                    raise RuntimeError("CTC Viterbi backtrace reached an invalid state.")
        return [paths[index] for index in range(num_streams)], [float(score.item()) for score in final_scores]


    def _word_rows_from_path(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        labels: Sequence[int],
        state_to_word: Sequence[Optional[int]],
        path: torch.Tensor,
        ctc_log_probs: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step_seconds: float,
        time_offset: float,
    ) -> List[Dict[str, Any]]:
        """Convert a Viterbi state path into CTC word intervals and speaker metadata."""
        frames_by_word: List[List[int]] = [[] for _ in tokenized_words]
        for frame_index, state_index in enumerate(path.tolist()):
            word_index = state_to_word[state_index]
            if word_index is not None:
                frames_by_word[word_index].append(frame_index)

        rows: List[Dict[str, Any]] = []
        for word, frames in zip(tokenized_words, frames_by_word):
            if not frames:
                raise RuntimeError(f"CTC path did not visit any token state for word {word['word']!r}.")
            start_frame, end_frame = frames[0], frames[-1]
            selected_log_probs = torch.tensor(
                [ctc_log_probs[frame, labels[int(path[frame].item())]].item() for frame in frames],
                dtype=torch.float32,
            )
            speaker_tag = word['speaker_tag']
            sortformer_column = speaker_mapping.get(speaker_tag)
            speaker_confidence: Optional[float] = None
            speaker_activity_start: Optional[float] = None
            speaker_activity_end: Optional[float] = None
            if speaker_probs is not None and sortformer_column is not None:
                activity = speaker_probs[start_frame : end_frame + 1, sortformer_column]
                speaker_confidence = float(activity.mean().item())
                active_indices = torch.nonzero(activity >= self.speaker_activity_threshold, as_tuple=False).flatten()
                if active_indices.numel() > 0:
                    activity_start = start_frame + int(active_indices[0].item())
                    activity_end = start_frame + int(active_indices[-1].item())
                    speaker_activity_start = time_offset + activity_start * ctc_step_seconds
                    speaker_activity_end = time_offset + (activity_end + 1) * ctc_step_seconds

            rows.append(
                {
                    'word': word['word'],
                    'word_index': word['word_index'],
                    'speaker_tag': speaker_tag,
                    'start': time_offset + start_frame * ctc_step_seconds,
                    'end': time_offset + (end_frame + 1) * ctc_step_seconds,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'ctc_confidence': float(torch.exp(selected_log_probs.mean()).item()),
                    'sortformer_column': sortformer_column,
                    'speaker_confidence': speaker_confidence,
                    'speaker_activity_start': speaker_activity_start,
                    'speaker_activity_end': speaker_activity_end,
                }
            )
        return rows

    def _resolve_speaker_mapping(
        self,
        *,
        speaker_tags: Sequence[int],
        preliminary_rows: Sequence[Dict[str, Any]],
        speaker_probs: Optional[torch.Tensor],
        assignment_mode: str,
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, List[float]]]:
        """Map t-SOT tags to raw Sortformer columns using preliminary CTC spans."""
        mapping: Dict[int, Optional[int]] = {tag: None for tag in speaker_tags}
        assignment_scores: Dict[int, List[float]] = {}
        if speaker_probs is None:
            return mapping, assignment_scores

        num_columns = speaker_probs.shape[1]
        if assignment_mode == 'identity':
            invalid_tags = [tag for tag in speaker_tags if not 0 <= tag < num_columns]
            if invalid_tags:
                raise ValueError(
                    "speaker_assignment_mode='identity' requires each t-SOT tag to match a "
                    f"Sortformer column; invalid tag(s): {invalid_tags}."
                )
            valid_tags = list(speaker_tags)
        else:
            valid_tags = list(speaker_tags)
        rows_by_tag: Dict[int, List[Dict[str, Any]]] = {tag: [] for tag in valid_tags}
        for row in preliminary_rows:
            tag = row['speaker_tag']
            if tag in rows_by_tag:
                rows_by_tag[tag].append(row)

        score_matrix: List[List[float]] = []
        for tag in valid_tags:
            column_scores: List[float] = []
            for column in range(num_columns):
                frame_scores: List[torch.Tensor] = []
                for row in rows_by_tag[tag]:
                    frame_scores.append(
                        torch.log(
                            speaker_probs[row['start_frame'] : row['end_frame'] + 1, column].clamp_min(self.epsilon)
                        )
                    )
                score = torch.cat(frame_scores).mean() if frame_scores else torch.tensor(-float('inf'))
                column_scores.append(float(score.item()))
            assignment_scores[tag] = column_scores
            score_matrix.append(column_scores)

        if assignment_mode == 'identity':
            for tag in valid_tags:
                mapping[tag] = tag
            return mapping, assignment_scores

        for tag, column in zip(valid_tags, self._maximum_weight_assignment(score_matrix)):
            mapping[tag] = column
        return mapping, assignment_scores

    @staticmethod
    def _maximum_weight_assignment(score_matrix: Sequence[Sequence[float]]) -> List[int]:
        """Solve a small rectangular one-to-one maximum-weight assignment exactly."""
        if not score_matrix:
            return []
        num_rows = len(score_matrix)
        num_columns = len(score_matrix[0])
        if num_rows > num_columns:
            raise ValueError("Cannot assign more t-SOT speakers than Sortformer columns.")
        if any(len(row) != num_columns for row in score_matrix):
            raise ValueError("speaker assignment score rows must have equal width.")

        # Sortformer has at most eight slots here, so an exact bitmask DP is clearer
        # and avoids adding a SciPy dependency just for this tiny assignment problem.
        states: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
        for row in score_matrix:
            next_states: Dict[int, Tuple[float, List[int]]] = {}
            for used_columns, (score_so_far, columns) in states.items():
                for column, score in enumerate(row):
                    if used_columns & (1 << column):
                        continue
                    next_mask = used_columns | (1 << column)
                    candidate = (score_so_far + score, columns + [column])
                    current = next_states.get(next_mask)
                    if current is None or candidate[0] > current[0]:
                        next_states[next_mask] = candidate
            states = next_states
        if not states:
            raise ValueError("No valid Sortformer speaker assignment exists.")
        return max(states.values(), key=lambda item: item[0])[1]

    @staticmethod
    def _resample_speaker_probs(speaker_probs: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Linearly resample raw Sortformer probabilities onto the CTC frame grid."""
        if speaker_probs.ndim != 2:
            raise ValueError(f"speaker_probs must have shape (T, S), got {tuple(speaker_probs.shape)}.")
        source_frames = speaker_probs.shape[0]
        if source_frames <= 0 or target_frames <= 0:
            raise ValueError("speaker and CTC frame counts must be positive.")
        if source_frames == target_frames:
            return speaker_probs
        if source_frames == 1:
            return speaker_probs.expand(target_frames, -1)

        positions = torch.linspace(0, source_frames - 1, target_frames, dtype=torch.float32)
        lower = positions.floor().to(dtype=torch.long)
        upper = positions.ceil().to(dtype=torch.long)
        fraction = (positions - lower.to(dtype=torch.float32)).unsqueeze(-1)
        return speaker_probs[lower] * (1.0 - fraction) + speaker_probs[upper] * fraction

    @staticmethod
    def _validate_alignment_mode(mode: str) -> str:
        mode = str(mode).lower()
        if mode not in {'parallel', 'serialized'}:
            raise ValueError(f"alignment_mode must be 'parallel' or 'serialized', got {mode!r}.")
        return mode

    @staticmethod
    def _validate_assignment_mode(mode: str) -> str:
        mode = str(mode).lower()
        if mode not in {'optimal', 'identity'}:
            raise ValueError(f"speaker_assignment_mode must be 'optimal' or 'identity', got {mode!r}.")
        return mode

    @staticmethod
    def _single_recording_tensor(tensor: torch.Tensor, name: str, expected_ndim: int) -> torch.Tensor:
        """Accept ``(T, D)`` or ``(1, T, D)``, rejecting ambiguous batched input."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.ndim == expected_ndim:
            return tensor
        if tensor.ndim == expected_ndim + 1:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{name} has batch size {tensor.shape[0]}; align one recording per call so "
                    "timestamps cannot be associated with the wrong transcript."
                )
            return tensor[0]
        raise ValueError(
            f"{name} must have shape (T, D) or (1, T, D), but got {tuple(tensor.shape)}."
        )

    @staticmethod
    def _scalar_length(value: Any, name: str) -> int:
        """Read an integral single-item tensor / scalar length without hidden batching."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one length, got shape {tuple(value.shape)}.")
            value = value.detach().cpu().item()
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be a scalar length, got {type(value).__name__}.") from error
        if not math.isfinite(numeric_value) or numeric_value != int(numeric_value):
            raise ValueError(f"{name} must be a finite integer, got {value!r}.")
        return int(numeric_value)

    @classmethod
    def _select_length(cls, lengths: Optional[Any], maximum: int, name: str) -> int:
        if maximum <= 0:
            raise ValueError(f"{name} maximum must be positive, got {maximum}.")
        if lengths is None:
            return maximum
        length = cls._scalar_length(lengths, name)
        if not 0 < length <= maximum:
            raise ValueError(f"{name} must be in [1, {maximum}], got {length}.")
        return length

    def _resolve_blank_id(self, ctc_vocab_size: int) -> int:
        """Validate the decoder/head blank convention against the supplied log-probs."""
        if self.blank_id is not None:
            blank_id = int(self.blank_id)
        elif self.ctc_decoder is not None:
            blank_id = int(self.ctc_decoder.num_classes_with_blank) - 1
        else:
            blank_id = ctc_vocab_size - 1
        if not 0 <= blank_id < ctc_vocab_size:
            raise ValueError(
                f"blank_id={blank_id} is outside the CTC class range [0, {ctc_vocab_size})."
            )
        if self.ctc_decoder is not None and self.ctc_decoder.num_classes_with_blank != ctc_vocab_size:
            raise ValueError(
                "CTC log-probability class count does not match the TransformerCTCDecoder: "
                f"{ctc_vocab_size} vs {self.ctc_decoder.num_classes_with_blank}."
            )
        tokenizer_vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        if callable(tokenizer_vocab_size):
            tokenizer_vocab_size = tokenizer_vocab_size()
        if tokenizer_vocab_size is not None and int(tokenizer_vocab_size) > blank_id:
            raise ValueError(
                "Tokenizer vocabulary is larger than the non-blank CTC vocabulary: "
                f"{tokenizer_vocab_size} vs blank_id={blank_id}."
            )
        return blank_id

    def _resolve_ctc_frame_seconds(self, ctc_length: int, audio_duration: Optional[float]) -> float:
        if audio_duration is not None:
            return float(audio_duration) / ctc_length
        if self.ctc_frame_seconds is not None:
            return float(self.ctc_frame_seconds)

        pee_encoder = getattr(self.encoder, 'encoder', self.encoder)
        subsampling_factor = getattr(pee_encoder, 'subsampling_factor', None)
        if subsampling_factor is None and hasattr(pee_encoder, 'pee'):
            speech_expert = getattr(pee_encoder.pee, 'experts', {}).get('speech', None)
            subsampling_factor = getattr(speech_expert, 'subsampling_factor', None)
        if subsampling_factor is None:
            subsampling_factor = 1
        return self.input_frame_seconds * float(subsampling_factor)

    def _resolve_sortformer_frame_seconds(
        self,
        sortformer_length: Optional[int],
        audio_duration: Optional[float],
        ctc_step_seconds: float,
    ) -> Optional[float]:
        if sortformer_length is None:
            return None
        if audio_duration is not None:
            return float(audio_duration) / sortformer_length
        if self.sortformer_frame_seconds is not None:
            return float(self.sortformer_frame_seconds)
        return ctc_step_seconds
