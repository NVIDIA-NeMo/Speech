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
    "MultiSpeakerSOTWordTimestampAligner",
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
        "chunk_size_seconds",
        "ctc_timestamp_model_path",
        "diar_normalize_type",
        "frame_shift_seconds",
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

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        """Initialize a serializable Parallel Expert Encoder wrapper.

        Args:
            cfg (DictConfig): Parallel Expert Encoder bundle configuration.
            trainer (Optional[Trainer]): Lightning trainer associated with the model.
        """
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
            frame_shift_seconds=self._cfg.get("frame_shift_seconds", 0.01),
            chunk_size_seconds=self._cfg.get("chunk_size_seconds", None),
            sync_max_audio_length=self._cfg.get("sync_max_audio_length", False),
            ctc_timestamp_model_path=self._cfg.get("ctc_timestamp_model_path", None),
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
        self.encoder._bundle_config.ctc_timestamp_model_path = self.encoder.ctc_timestamp_model_path

    @staticmethod
    def _validate_bundle_schema(cfg: DictConfig) -> None:
        """Require the self-contained ParallelExpertEncoder bundle schema."""
        missing = [key for key in ("asr_encoder_cfg", "diarization_model_cfg") if cfg.get(key, None) in (None, {}, "")]
        if missing:
            raise ValueError(f"ParallelExpertEncoder bundle is missing required config sections {missing}.")
        _normalize_asr_encoder_type(cfg.get("asr_encoder_type", "fastconformer"))

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
        template_cfg.frame_shift_seconds = encoder.frame_shift_seconds
        template_cfg.chunk_size_seconds = encoder.chunk_size_seconds
        template_cfg.sync_max_audio_length = encoder.sync_max_audio_length
        template_cfg.ctc_timestamp_model_path = encoder.ctc_timestamp_model_path
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
        frame_shift_seconds: float = 0.01,
        chunk_size_seconds: Optional[float] = None,
        sync_max_audio_length: bool = False,
        ctc_timestamp_model_path: Optional[str] = None,
    ):
        """Initialize the ASR and diarization experts and their fusion layers.

        Args:
            asr_encoder_cfg (DictConfig): Configuration used to construct the ASR encoder.
            diarization_model_cfg (DictConfig): Configuration used to construct the Sortformer model.
            asr_normalize_type (Optional[str]): Feature normalization applied before ASR inference.
            diar_normalize_type (Optional[str]): Feature normalization applied before diarization.
            freeze_diar (bool): Whether to freeze the diarization expert.
            freeze_asr (bool): Whether to freeze the ASR expert.
            online_inference_length (int): Core window length in encoded frames for online inference.
            chunk_left_context (int): Number of encoded left-context frames per window.
            chunk_right_context (int): Number of encoded right-context frames per window.
            diar_fifo_len (int): Sortformer streaming FIFO length.
            diar_spkcache_update_period (int): Sortformer speaker-cache update period.
            diar_spkcache_len (int): Sortformer speaker-cache length.
            asr_encoder_type (str): ASR encoder architecture name.
            missing_rttm_target (float): Sentinel identifying rows without external speaker targets.
            speaker_feature_mode (Optional[str]): Continuous or thresholded speaker-feature mode.
            speaker_activity_threshold (Optional[float]): Activity threshold for thresholded fusion.
            spk_kernel_scale (float): Scale applied to the sinusoidal speaker infusion.
            frame_shift_seconds (float): Duration represented by one input feature frame.
            chunk_size_seconds (Optional[float]): Optional independent-branch chunk duration.
            sync_max_audio_length (bool): Whether child encoders synchronize maximum sequence lengths.
            ctc_timestamp_model_path (Optional[str]): Local CTC timestamp adapter path.
        """
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
        self.frame_shift_seconds = float(frame_shift_seconds)
        if self.frame_shift_seconds <= 0:
            raise ValueError(f"frame_shift_seconds must be positive, got {frame_shift_seconds}.")
        self.chunk_size_seconds = self._validate_chunk_size("chunk_size_seconds", chunk_size_seconds)
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
        self.ctc_timestamp_model_path = ctc_timestamp_model_path
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
        """Apply the configured trainability and evaluation modes to both experts."""
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

    @staticmethod
    def _validate_chunk_size(name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive or None, got {value}.")
        return value

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
        """Build sinusoidal speaker embeddings.

        Args:
            max_position (int): Number of speaker positions.
            embedding_dim (int): Embedding width.

        Returns:
            torch.Tensor: Sinusoidal embeddings shaped ``(max_position, embedding_dim)``.
        """
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
        """Pad or truncate speaker activity to a target frame count.

        Args:
            spk_targets (torch.Tensor): Speaker activity shaped ``(B, T, S)``.
            target_len (int): Required number of time frames.

        Returns:
            torch.Tensor: Speaker activity shaped ``(B, target_len, S)``.
        """
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
        """Move a tensor to the device and dtype of a module.

        Args:
            tensor (torch.Tensor): Tensor to move.
            module (nn.Module): Module defining the destination device and dtype.

        Returns:
            torch.Tensor: Converted tensor, or the original tensor for a parameterless module.
        """
        parameter = next(module.parameters(), None)
        if parameter is None:
            return tensor
        return tensor.to(device=parameter.device, dtype=parameter.dtype)

    def _check_spk_targets(self, spk_targets: Optional[torch.Tensor], batch_size: int) -> None:
        """Validate optional external speaker-activity targets.

        Args:
            spk_targets (Optional[torch.Tensor]): Speaker targets shaped ``(B, T, S)``.
            batch_size (int): Expected batch size.
        """
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
        """Identify rows containing only the missing-target sentinel.

        Args:
            spk_targets (torch.Tensor): Speaker targets shaped ``(B, T, S)``.

        Returns:
            torch.Tensor: Boolean mask shaped ``(B,)``.
        """
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

    @contextlib.contextmanager
    def capture_ctc_timestamps(self, device: torch.device):
        """Capture chunk-synchronous CTC and speaker outputs for one generation call."""
        if self.__dict__.get("_ctc_timestamp_capture_state") is not None:
            raise RuntimeError("CTC timestamp capture does not support nested generation calls.")
        extractor = _get_ctc_timestamp_extractor(self, self.ctc_timestamp_model_path, device)
        self.__dict__["_ctc_timestamp_capture_state"] = {"extractor": extractor, "outputs": None}
        try:
            yield
        finally:
            self.__dict__.pop("_ctc_timestamp_capture_state", None)

    def generate_ctc_timestamps(
        self,
        sot_transcripts: Sequence[str],
        audio_durations: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """Align generated ASR text with outputs captured during the PEE generation pass."""
        state = self.__dict__.get("_ctc_timestamp_capture_state")
        if state is None or state["outputs"] is None:
            raise RuntimeError("No CTC outputs were captured during the current generation call.")
        return state["extractor"].extract_from_outputs_batch(
            sot_transcripts=sot_transcripts,
            audio_durations=audio_durations,
            **state["outputs"],
        )

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
        """Run the diarization expert on a mel-spectrogram batch.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.

        Returns:
            torch.Tensor: Speaker probabilities shaped ``(B, T_out, S)``.
        """
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
        """Run the ASR expert on a mel-spectrogram batch.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded states and their valid lengths.
        """
        if self.asr_normalize_type:
            audio_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.asr_normalize_type)
        audio_signal = self._match_module_io(audio_signal, self.asr_encoder)
        length = length.to(device=audio_signal.device)

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_asr):
            return self.asr_encoder(audio_signal=audio_signal, length=length)

    def _ctc_timestamp_decoder(self) -> Optional[nn.Module]:
        """Return the active timestamp decoder during capture.

        Returns:
            Optional[nn.Module]: CTC decoder, or ``None`` when capture is inactive.
        """
        state = self.__dict__.get("_ctc_timestamp_capture_state")
        return None if state is None else state["extractor"].ctc_decoder

    def _timestamp_speaker_probs(
        self,
        spk_targets: torch.Tensor,
        diarization_preds: Optional[torch.Tensor],
        use_diarization: Optional[torch.Tensor],
        target_len: int,
    ) -> torch.Tensor:
        """Select and align speaker probabilities for timestamp extraction.

        Args:
            spk_targets (torch.Tensor): External or predicted speaker activity.
            diarization_preds (Optional[torch.Tensor]): Predicted activity for missing-target rows.
            use_diarization (Optional[torch.Tensor]): Rows that should use diarization predictions.
            target_len (int): Required CTC frame count.

        Returns:
            torch.Tensor: Speaker probabilities aligned to the CTC frame grid.
        """
        speaker_probs = self._downsample_high_resolution_diarization_for_fusion(spk_targets, target_len)
        speaker_probs = self._align_diar_frames(speaker_probs, target_len)
        if use_diarization is not None:
            diarization_preds = self._downsample_high_resolution_diarization_for_fusion(diarization_preds, target_len)
            diarization_preds = self._align_diar_frames(diarization_preds, target_len)
            speaker_probs = torch.where(
                use_diarization.to(device=speaker_probs.device, dtype=torch.bool).view(-1, 1, 1),
                diarization_preds.to(speaker_probs.device),
                speaker_probs,
            )
        return speaker_probs

    def _store_ctc_timestamp_outputs(
        self,
        ctc_log_probs: torch.Tensor,
        ctc_lengths: torch.Tensor,
        speaker_probs: torch.Tensor,
        speaker_lengths: torch.Tensor,
    ) -> None:
        """Store detached timestamp inputs in the active capture state.

        Args:
            ctc_log_probs (torch.Tensor): CTC log probabilities shaped ``(B, T, V)``.
            ctc_lengths (torch.Tensor): Valid CTC frame counts shaped ``(B,)``.
            speaker_probs (torch.Tensor): Speaker probabilities shaped ``(B, T, S)``.
            speaker_lengths (torch.Tensor): Valid speaker frame counts shaped ``(B,)``.
        """
        state = self.__dict__.get("_ctc_timestamp_capture_state")
        if state is None:
            return
        state["outputs"] = {
            "ctc_log_probs": ctc_log_probs.detach().cpu(),
            "ctc_lengths": ctc_lengths.detach().cpu(),
            "sortformer_sigmoids": speaker_probs.detach().cpu(),
            "sortformer_lengths": speaker_lengths.detach().cpu(),
        }

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
        ctc_decoder = self._ctc_timestamp_decoder()
        if ctc_decoder is not None:
            ctc_log_probs = ctc_decoder(asr_encoded, encoded_lengths=asr_encoded_len)
            speaker_probs = self._timestamp_speaker_probs(
                spk_targets, diarization_preds, use_diarization, ctc_log_probs.shape[1]
            )
            self._store_ctc_timestamp_outputs(
                ctc_log_probs,
                asr_encoded_len.clamp(max=ctc_log_probs.shape[1]),
                speaker_probs,
                asr_encoded_len.clamp(max=speaker_probs.shape[1]),
            )
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

        ctc_decoder = self._ctc_timestamp_decoder()
        asr_chunks: List[torch.Tensor] = []
        ctc_chunks: List[torch.Tensor] = []
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
                encoded_context, encoded_context_len = self.asr_encoder(audio_signal=asr_chunk, length=chunk_length)
            left_drop = left_offset // self.subsampling_factor
            core_len = self._asr_output_frame_boundary(end) - self._asr_output_frame_boundary(start)
            core_len = max(0, min(core_len, encoded_context.shape[-1] - left_drop))
            asr_chunks.append(encoded_context[:, :, left_drop : left_drop + core_len])
            if ctc_decoder is not None:
                ctc_context = ctc_decoder(encoded_context, encoded_lengths=encoded_context_len)
                ctc_chunks.append(ctc_context[:, left_drop : left_drop + core_len].detach().cpu())

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
        if ctc_decoder is not None:
            ctc_log_probs = torch.cat(ctc_chunks, dim=1)
            speaker_probs = self._timestamp_speaker_probs(
                spk_targets, diarization_preds, use_diarization, ctc_log_probs.shape[1]
            )
            self._store_ctc_timestamp_outputs(
                ctc_log_probs,
                encoded_len.clamp(max=ctc_log_probs.shape[1]),
                speaker_probs,
                encoded_len.clamp(max=speaker_probs.shape[1]),
            )
        output = self._fuse_diar_and_asr(
            asr_encoded,
            spk_targets,
            diarization_preds=diarization_preds,
            use_diarization=use_diarization,
        )
        return output, encoded_len

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        """Initialize Sortformer state for windowed inference.

        Args:
            audio_signal (torch.Tensor): Input mel features.
            length (torch.Tensor): Valid input frame counts.
            batch_size (int): Number of input recordings.

        Returns:
            tuple: Streaming state, computation dtype, converted features, and converted lengths.
        """
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
        """Initialize the optional Transformer bridge and CTC projection.

        Args:
            feat_in (int): Number of input encoder features.
            num_classes (int): Number of non-blank CTC classes.
            init_mode (str): Convolution weight initialization mode.
            vocabulary (Optional[List[str]]): CTC output vocabulary.
            add_blank (bool): Whether to append a CTC blank class.
            use_transformer (bool): Whether to enable the Transformer bridge.
            d_model (Optional[int]): Transformer hidden dimension; must equal ``feat_in``.
            n_heads (int): Number of Transformer attention heads.
            n_layers (int): Number of Transformer layers.
            drop_rate (float): Transformer dropout rate.
            dropout_pre_encoder (Optional[float]): Pre-encoder dropout rate.
            dropout_emb (float): Embedding dropout rate.
            qkv_bias (bool): Whether attention projections use bias.
            qk_norm (bool): Whether to normalize query and key vectors.
            ff_expansion (float): Feed-forward expansion factor.
            pre_block_norm (bool): Whether to apply normalization before each block.
            self_attention_model (Optional[str]): Self-attention implementation.
            rope_base (float): Rotary-position-embedding base.
            rotary_fraction (float): Fraction of dimensions using rotary embeddings.
            pos_emb_max_len (int): Maximum positional-embedding length.
            xscaling (bool): Whether to apply attention input scaling.
            attn_mode (str): Attention masking mode.
            sync_max_audio_length (bool): Whether to synchronize maximum sequence lengths.
            residual (bool): Whether to add the bridge input to its output.
            residual_scale (float): Initial residual branch scale.
            learnable_residual_scale (bool): Whether the residual scale is trainable.
        """
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

    def forward(self, encoder_output: torch.Tensor, encoded_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class MultiSpeakerSOTWordTimestampAligner:
    """Align all speaker streams together with dense max-sum CTC DP and a soft Sortformer prior."""

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
        maximum_token_len: float = 1.0,
        epsilon: float = 1.0e-6,
    ) -> None:
        """Initialize multi-speaker SOT word timestamp alignment.

        Args:
            encoder (Optional[nn.Module]): Parallel Expert Encoder or its wrapper.
            ctc_decoder (Optional[TransformerCTCDecoder]): CTC timestamp decoder.
            tokenizer (Optional[Any]): Tokenizer matching the CTC decoder vocabulary.
            blank_id (Optional[int]): Explicit CTC blank class index.
            input_frame_seconds (float): Duration represented by one input feature frame.
            ctc_frame_seconds (Optional[float]): Explicit duration of one CTC frame.
            sortformer_frame_seconds (Optional[float]): Explicit duration of one Sortformer frame.
            speaker_activity_threshold (float): Threshold for speaker activity metadata.
            speaker_logprob_weight (float): Weight of the Sortformer prior in CTC alignment.
            maximum_token_len (float): Maximum emitted word duration in seconds.
            epsilon (float): Numerical floor for logarithms.
        """
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        if not 0 <= speaker_activity_threshold <= 1:
            raise ValueError("speaker_activity_threshold must be between zero and one.")
        if maximum_token_len <= 0:
            raise ValueError("maximum_token_len must be positive.")
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.input_frame_seconds = float(input_frame_seconds)
        self.ctc_frame_seconds = ctc_frame_seconds
        self.sortformer_frame_seconds = sortformer_frame_seconds
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.speaker_logprob_weight = float(speaker_logprob_weight)
        self.maximum_token_len = float(maximum_token_len)
        self.epsilon = float(epsilon)

    @classmethod
    def parse_sot_words(cls, transcript: str) -> List[Dict[str, Any]]:
        """Split a t-SOT transcript into words while retaining speaker turns."""
        words: List[Dict[str, Any]] = []
        parts = cls._SPEAKER_TAG_RE.split(transcript)
        speaker_tag: Optional[int] = 0 if len(parts) == 1 else None
        turn_index: Optional[int] = None
        for index, part in enumerate(parts):
            if index % 2:
                speaker_tag, turn_index = int(part), index // 2
                continue
            for word in part.split():
                words.append(dict(word=word, speaker_tag=speaker_tag, turn_index=turn_index, word_index=len(words)))
        return words

    def extract_ctc_and_sortformer_batch(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the raw PEE ASR/diarization branches and the CTC adapter."""
        pee = getattr(self.encoder, 'encoder', self.encoder)
        if pee is None or self.ctc_decoder is None:
            raise ValueError("encoder and ctc_decoder are required for audio inference.")

        modules = (pee, self.ctc_decoder)
        previous_modes = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            with torch.inference_mode():
                speech_states, speech_lengths = pee._run_asr(processed_signal, processed_signal_length)
                diar_signal = processed_signal
                if pee.diar_normalize_type:
                    diar_signal, _, _ = normalize_batch(
                        diar_signal,
                        processed_signal_length,
                        normalize_type=pee.diar_normalize_type,
                    )
                diar_signal = pee._match_module_io(diar_signal, pee.diarization_model)
                embeddings, embedding_lengths = pee.diarization_model.frontend_encoder(
                    processed_signal=diar_signal,
                    processed_signal_length=processed_signal_length.to(diar_signal.device),
                    bypass_pre_encode=False,
                )
                native_predictions = pee.diarization_model.forward_infer(
                    emb_seq=embeddings,
                    emb_seq_length=embedding_lengths,
                )
                speaker_probs = pee._align_diarization_output_resolution(native_predictions, embedding_lengths)
                ctc_log_probs = self.ctc_decoder(speech_states, encoded_lengths=speech_lengths)
        finally:
            for module, was_training in zip(modules, previous_modes):
                module.train(was_training)

        diar_model = pee.diarization_model
        native_factor = 1 if diar_model.high_resolution else int(diar_model.encoder.subsampling_factor)
        downsample_factor = int(diar_model.output_subsampling_factor) // native_factor
        if downsample_factor <= 1:
            speaker_lengths = embedding_lengths
        else:
            native_lengths = embedding_lengths * (int(diar_model.encoder.subsampling_factor) // native_factor)
            speaker_lengths = torch.div(
                native_lengths + downsample_factor - 1,
                downsample_factor,
                rounding_mode='floor',
            )
        return {
            'ctc_log_probs': ctc_log_probs,
            'ctc_lengths': speech_lengths.clamp(max=ctc_log_probs.shape[1]),
            'sortformer_sigmoids': speaker_probs,
            'sortformer_lengths': speaker_lengths.clamp(min=1, max=speaker_probs.shape[1]),
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
    ) -> Dict[str, Any]:
        """Thin one-record wrapper around :meth:`extract_from_audio_batch`."""
        return self.extract_from_audio_batch(
            input_signal,
            input_signal_length,
            preprocessor,
            [sot_transcript],
            audio_durations=None if audio_duration is None else [audio_duration],
            time_offsets=[time_offset],
        )[0]

    def extract_from_audio_batch(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        preprocessor: nn.Module,
        sot_transcripts: Sequence[str],
        *,
        audio_durations: Optional[Sequence[Optional[float]]] = None,
        time_offsets: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Preprocess a waveform batch once, then align every record in parallel mode."""
        batch_size = input_signal.shape[0]
        if input_signal_length.shape != (batch_size,) or len(sot_transcripts) != batch_size:
            raise ValueError("audio lengths and transcripts must match the waveform batch.")
        with torch.inference_mode():
            processed_signal, processed_signal_length = preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        outputs = self.extract_ctc_and_sortformer_batch(processed_signal, processed_signal_length)
        if audio_durations is None:
            sample_rate = getattr(preprocessor, '_sample_rate', getattr(preprocessor, 'sample_rate', None))
            if sample_rate is not None:
                audio_durations = [float(length) / float(sample_rate) for length in input_signal_length.cpu()]
        return self.extract_from_outputs_batch(
            sot_transcripts=sot_transcripts,
            audio_durations=audio_durations,
            time_offsets=time_offsets,
            **outputs,
        )

    def extract_from_outputs_batch(
        self,
        ctc_log_probs: torch.Tensor,
        sortformer_sigmoids: Optional[torch.Tensor],
        sot_transcripts: Sequence[str],
        *,
        ctc_lengths: Optional[torch.Tensor] = None,
        sortformer_lengths: Optional[torch.Tensor] = None,
        audio_durations: Optional[Sequence[Optional[float]]] = None,
        time_offsets: Optional[Sequence[float]] = None,
        speaker_logprob_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Align a padded output batch, always using independent speaker streams."""
        if ctc_log_probs.ndim != 3:
            raise ValueError("ctc_log_probs must have shape (batch, frames, classes).")
        batch_size, max_ctc_frames, vocab_size = ctc_log_probs.shape
        if len(sot_transcripts) != batch_size:
            raise ValueError("sot_transcripts must contain one string per batch item.")
        ctc_lengths_list = self._lengths(ctc_lengths, batch_size, max_ctc_frames)
        durations = list(audio_durations) if audio_durations is not None else [None] * batch_size
        offsets = list(time_offsets) if time_offsets is not None else [0.0] * batch_size
        if len(durations) != batch_size or len(offsets) != batch_size:
            raise ValueError("audio_durations and time_offsets must match the batch size.")

        speaker_lengths_list: List[Optional[int]] = [None] * batch_size
        if sortformer_sigmoids is not None:
            if sortformer_sigmoids.ndim != 3 or sortformer_sigmoids.shape[0] != batch_size:
                raise ValueError("sortformer_sigmoids must have shape (batch, frames, speakers).")
            speaker_lengths_list = self._lengths(
                sortformer_lengths,
                batch_size,
                sortformer_sigmoids.shape[1],
            )

        blank_id = self._resolve_blank_id(vocab_size)
        ctc_cpu = ctc_log_probs.detach().float().cpu()
        speaker_cpu = None if sortformer_sigmoids is None else sortformer_sigmoids.detach().float().cpu()
        weight = self.speaker_logprob_weight if speaker_logprob_weight is None else float(speaker_logprob_weight)
        if weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")

        results = []
        for index, transcript in enumerate(sot_transcripts):
            ctc = ctc_cpu[index, : ctc_lengths_list[index]]
            speaker_probs = None
            speaker_length = speaker_lengths_list[index]
            if speaker_cpu is not None and speaker_length is not None:
                speaker_probs = self._resample_speaker_probs(
                    speaker_cpu[index, :speaker_length].clamp(0, 1),
                    ctc.shape[0],
                )
            results.append(
                self._align_record(
                    ctc=ctc,
                    speaker_probs=speaker_probs,
                    transcript=transcript,
                    blank_id=blank_id,
                    audio_duration=durations[index],
                    time_offset=float(offsets[index]),
                    speaker_logprob_weight=weight,
                    sortformer_length=speaker_length,
                )
            )
        return results

    def _align_record(
        self,
        *,
        ctc: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        transcript: str,
        blank_id: int,
        audio_duration: Optional[float],
        time_offset: float,
        speaker_logprob_weight: float,
        sortformer_length: Optional[int],
    ) -> Dict[str, Any]:
        """Align one t-SOT transcript against precomputed model outputs.

        Args:
            ctc (torch.Tensor): CTC log probabilities shaped ``(T, V)``.
            speaker_probs (Optional[torch.Tensor]): Speaker probabilities shaped ``(T, S)``.
            transcript (str): Multi-speaker t-SOT transcript.
            blank_id (int): CTC blank class index.
            audio_duration (Optional[float]): Recording duration in seconds.
            time_offset (float): Offset added to returned timestamps.
            speaker_logprob_weight (float): Weight of the speaker activity prior.
            sortformer_length (Optional[int]): Number of valid Sortformer frames.

        Returns:
            Dict[str, Any]: Word timestamps, speaker mapping, and alignment metadata.
        """
        words = self._tokenize_words(self.parse_sot_words(transcript), blank_id)
        pee = getattr(self.encoder, 'encoder', self.encoder)
        ctc_step = self._frame_seconds(
            ctc.shape[0],
            audio_duration,
            self.ctc_frame_seconds,
            self.input_frame_seconds * float(getattr(pee, 'subsampling_factor', 1)),
        )
        sortformer_step = (
            None
            if sortformer_length is None
            else self._frame_seconds(
                sortformer_length, audio_duration, self.sortformer_frame_seconds or None, ctc_step
            )
        )
        rows, mapping, preliminary_scores, final_scores, assignment_scores = [], {}, {}, {}, {}
        if words:
            preliminary_rows, preliminary_scores = self._align_streams(
                words,
                ctc,
                blank_id,
                speaker_probs=None,
                speaker_mapping={},
                ctc_step=ctc_step,
                time_offset=time_offset,
                speaker_logprob_weight=0.0,
            )
            speaker_tags = list(
                dict.fromkeys(word['speaker_tag'] for word in words if word['speaker_tag'] is not None)
            )
            mapping, assignment_scores = self._resolve_speaker_mapping(speaker_tags, preliminary_rows, speaker_probs)
            rows, final_scores = self._align_streams(
                words,
                ctc,
                blank_id,
                speaker_probs=speaker_probs,
                speaker_mapping=mapping,
                ctc_step=ctc_step,
                time_offset=time_offset,
                speaker_logprob_weight=speaker_logprob_weight,
            )
        return self._result(
            rows=rows,
            mapping=mapping,
            ctc_step=ctc_step,
            sortformer_step=sortformer_step,
            time_offset=time_offset,
            ctc_frames=ctc.shape[0],
            sortformer_frames=sortformer_length,
            preliminary_scores=preliminary_scores,
            final_scores=final_scores,
            assignment_scores=assignment_scores,
        )

    @staticmethod
    def _result(
        *,
        rows: Sequence[Dict[str, Any]],
        mapping: Dict[int, Optional[int]],
        ctc_step: float,
        sortformer_step: Optional[float],
        time_offset: float,
        ctc_frames: int,
        sortformer_frames: Optional[int],
        preliminary_scores: Dict[Optional[int], float],
        final_scores: Dict[Optional[int], float],
        assignment_scores: Dict[int, List[float]],
    ) -> Dict[str, Any]:
        """Format one timestamp-alignment result.

        Args:
            rows (Sequence[Dict[str, Any]]): Aligned word records.
            mapping (Dict[int, Optional[int]]): t-SOT tag to Sortformer-column mapping.
            ctc_step (float): Duration of one CTC frame in seconds.
            sortformer_step (Optional[float]): Duration of one Sortformer frame.
            time_offset (float): Offset applied to returned timestamps.
            ctc_frames (int): Number of valid CTC frames.
            sortformer_frames (Optional[int]): Number of valid Sortformer frames.
            preliminary_scores (Dict[Optional[int], float]): CTC-only stream scores.
            final_scores (Dict[Optional[int], float]): Speaker-aware stream scores.
            assignment_scores (Dict[int, List[float]]): Speaker-column assignment scores.

        Returns:
            Dict[str, Any]: Public timestamp result and alignment diagnostics.
        """
        return {
            'speaker_word_timestamps': MultiSpeakerSOTWordTimestampAligner._group_words_by_speaker(rows),
            'speaker_tag_to_sortformer_column': mapping,
            'alignment_mode': 'parallel',
            'requested_alignment_mode': 'parallel',
            'speaker_assignment_mode': 'optimal',
            'ctc_frame_seconds': ctc_step,
            'sortformer_frame_seconds': sortformer_step,
            'time_offset': time_offset,
            'num_ctc_frames': ctc_frames,
            'num_sortformer_frames': sortformer_frames,
            'alignment_diagnostics': {
                'preliminary_ctc_path_scores': preliminary_scores,
                'final_path_scores': final_scores,
                'speaker_assignment_scores': assignment_scores,
            },
        }

    def _align_streams(
        self,
        words: Sequence[Dict[str, Any]],
        ctc: torch.Tensor,
        blank_id: int,
        *,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step: float,
        time_offset: float,
        speaker_logprob_weight: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[Optional[int], float]]:
        """Align independently grouped speaker streams in one padded DP batch.

        Args:
            words (Sequence[Dict[str, Any]]): Tokenized t-SOT word records.
            ctc (torch.Tensor): CTC log probabilities shaped ``(T, V)``.
            blank_id (int): CTC blank class index.
            speaker_probs (Optional[torch.Tensor]): Speaker probabilities shaped ``(T, S)``.
            speaker_mapping (Dict[int, Optional[int]]): t-SOT tag to speaker-column mapping.
            ctc_step (float): Duration of one CTC frame in seconds.
            time_offset (float): Offset added to returned timestamps.
            speaker_logprob_weight (float): Weight of the speaker activity prior.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[Optional[int], float]]: Word rows and scores by speaker.
        """
        grouped = self._group_words_by_speaker(words)
        streams = []
        for speaker_tag, speaker_words in grouped.items():
            labels, state_to_word = self._build_ctc_target(speaker_words, blank_id)
            tokens = labels[1::2]
            if len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:])) > ctc.shape[0]:
                raise ValueError(f"Speaker {speaker_tag!r} transcript is too long for the CTC timeline.")
            column = speaker_mapping.get(speaker_tag)
            columns = [None if word_index is None else column for word_index in state_to_word]
            streams.append((speaker_tag, speaker_words, labels, state_to_word, columns))

        max_states = max(len(stream[2]) for stream in streams)
        labels_batch = torch.full((len(streams), max_states), blank_id, dtype=torch.long)
        columns_batch = torch.full_like(labels_batch, -1)
        state_lengths = torch.tensor([len(stream[2]) for stream in streams], dtype=torch.long)
        for index, (_, _, labels, _, columns) in enumerate(streams):
            labels_batch[index, : len(labels)] = torch.tensor(labels)
            columns_batch[index, : len(columns)] = torch.tensor(
                [-1 if column is None else column for column in columns]
            )

        paths, scores = self._ctc_forced_align_batched(
            ctc,
            labels_batch,
            state_lengths,
            blank_id,
            columns_batch,
            speaker_probs,
            speaker_logprob_weight,
        )
        rows = [
            row
            for (_, speaker_words, labels, state_to_word, _), path in zip(streams, paths)
            for row in self._word_rows_from_path(
                speaker_words,
                labels,
                state_to_word,
                path,
                ctc,
                speaker_probs,
                speaker_mapping,
                ctc_step,
                time_offset,
            )
        ]
        scores_by_speaker = {stream[0]: score for stream, score in zip(streams, scores)}
        return rows, scores_by_speaker

    @staticmethod
    def _group_words_by_speaker(
        words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """Group word records by their t-SOT speaker tag.

        Args:
            words (Sequence[Dict[str, Any]]): Word records in transcript order.

        Returns:
            Dict[Optional[int], List[Dict[str, Any]]]: Words grouped by speaker tag.
        """
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for word in words:
            grouped.setdefault(word['speaker_tag'], []).append(word)
        return grouped

    def _tokenize_words(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[Dict[str, Any]]:
        """Attach CTC token IDs to parsed t-SOT words.

        Args:
            words (Sequence[Dict[str, Any]]): Parsed t-SOT word records.
            blank_id (int): CTC blank class index.

        Returns:
            List[Dict[str, Any]]: Word records containing ``token_ids``.
        """
        tokenized = {}
        for stream_words in self._group_words_by_speaker(words).values():
            for word, token_ids in zip(stream_words, self._tokenize_word_stream(stream_words, blank_id)):
                tokenized[word['word_index']] = {**word, 'token_ids': token_ids}
        return [tokenized[word['word_index']] for word in words]

    def _tokenize_word_stream(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[List[int]]:
        """Tokenize one speaker stream while preserving word ownership.

        Args:
            words (Sequence[Dict[str, Any]]): Words belonging to one speaker stream.
            blank_id (int): CTC blank class index.

        Returns:
            List[List[int]]: CTC token IDs grouped by word.
        """
        text = ' '.join(word['word'] for word in words)
        token_ids = [int(token_id) for token_id in self.tokenizer.text_to_ids(text)]
        if not token_ids or min(token_ids) < 0 or max(token_ids) >= blank_id:
            raise ValueError("Tokenizer produced IDs outside the non-blank CTC vocabulary.")
        ids_by_word = [[int(token_id) for token_id in self.tokenizer.text_to_ids(word['word'])] for word in words]
        if any(not token_ids for token_ids in ids_by_word):
            raise ValueError("SentencePiece produced an empty word tokenization.")
        if [token_id for word_ids in ids_by_word for token_id in word_ids] != token_ids:
            raise ValueError("Per-word and full-stream SentencePiece tokenization disagree.")
        return ids_by_word

    @staticmethod
    def _build_ctc_target(
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> Tuple[List[int], List[Optional[int]]]:
        """Build a blank-expanded CTC target and word-ownership mapping.

        Args:
            words (Sequence[Dict[str, Any]]): Tokenized word records.
            blank_id (int): CTC blank class index.

        Returns:
            Tuple[List[int], List[Optional[int]]]: Expanded labels and state-to-word indices.
        """
        labels = [blank_id]
        state_to_word: List[Optional[int]] = [None]
        for word_index, word in enumerate(words):
            for token_id in word['token_ids']:
                labels.extend((token_id, blank_id))
                state_to_word.extend((word_index, None))
        return labels, state_to_word

    def _ctc_forced_align_batched(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        blank_id: int,
        state_speaker_columns: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Run dense max-sum CTC DP for every padded speaker stream together."""
        num_streams, max_states = labels.shape
        num_frames = log_probs.shape[0]
        state_mask = torch.arange(max_states).unsqueeze(0) < state_lengths.unsqueeze(1)
        emissions = log_probs[:, labels].permute(1, 0, 2).contiguous()
        emissions.masked_fill_(~state_mask.unsqueeze(1), -float('inf'))

        token_states = state_mask & (state_speaker_columns >= 0)
        if speaker_probs is not None and speaker_logprob_weight > 0 and token_states.any():
            columns = state_speaker_columns.clamp_min(0)
            activity = speaker_probs[:, columns].permute(1, 0, 2)
            emissions += torch.where(
                token_states.unsqueeze(1),
                speaker_logprob_weight * torch.log(activity.clamp_min(self.epsilon)),
                torch.zeros_like(activity),
            )

        previous = torch.full((num_streams, max_states), -float('inf'))
        previous[:, 0] = emissions[:, 0, 0]
        previous[:, 1] = emissions[:, 0, 1]
        backpointers = torch.full((num_frames, num_streams, max_states), -1, dtype=torch.long)
        states = torch.arange(max_states).unsqueeze(0).expand(num_streams, -1)

        for frame in range(1, num_frames):
            best = previous
            previous_states = states
            advance = torch.full_like(previous, -float('inf'))
            advance[:, 1:] = previous[:, :-1]
            take = advance > best
            best = torch.where(take, advance, best)
            previous_states = torch.where(take, states - 1, previous_states)
            if max_states > 2:
                skip = torch.full_like(previous, -float('inf'))
                can_skip = state_mask[:, 2:] & (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
                skip[:, 2:] = torch.where(can_skip, previous[:, :-2], skip[:, 2:])
                take = skip > best
                best = torch.where(take, skip, best)
                previous_states = torch.where(take, states - 2, previous_states)
            previous = best + emissions[:, frame]
            previous.masked_fill_(~state_mask, -float('inf'))
            backpointers[frame] = previous_states

        last_blank = state_lengths - 1
        last_token = state_lengths - 2
        blank_scores = previous.gather(1, last_blank[:, None]).squeeze(1)
        token_scores = previous.gather(1, last_token[:, None]).squeeze(1)
        final_states = torch.where(token_scores > blank_scores, last_token, last_blank)
        final_scores = torch.maximum(token_scores, blank_scores)
        if not torch.isfinite(final_scores).all():
            failed = torch.nonzero(~torch.isfinite(final_scores)).flatten().tolist()
            raise ValueError(f"No valid CTC forced-alignment path for speaker stream(s) {failed}.")

        paths = torch.empty((num_streams, num_frames), dtype=torch.long)
        current = final_states
        stream_indices = torch.arange(num_streams)
        for frame in range(num_frames - 1, -1, -1):
            paths[:, frame] = current
            if frame:
                current = backpointers[frame, stream_indices, current]
        return list(paths), [float(score) for score in final_scores]

    def _word_rows_from_path(
        self,
        words: Sequence[Dict[str, Any]],
        labels: Sequence[int],
        state_to_word: Sequence[Optional[int]],
        path: torch.Tensor,
        ctc: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_mapping: Dict[int, Optional[int]],
        ctc_step: float,
        time_offset: float,
    ) -> List[Dict[str, Any]]:
        """Convert CTC state paths into timestamped word records.

        Args:
            words (Sequence[Dict[str, Any]]): Tokenized words for one speaker stream.
            labels (Sequence[int]): Blank-expanded CTC target labels.
            state_to_word (Sequence[Optional[int]]): Target-state to word-index mapping.
            path (torch.Tensor): Best CTC target state at every frame.
            ctc (torch.Tensor): CTC log probabilities shaped ``(T, V)``.
            speaker_probs (Optional[torch.Tensor]): Speaker probabilities shaped ``(T, S)``.
            speaker_mapping (Dict[int, Optional[int]]): t-SOT tag to speaker-column mapping.
            ctc_step (float): Duration of one CTC frame in seconds.
            time_offset (float): Offset added to returned timestamps.

        Returns:
            List[Dict[str, Any]]: Timestamp and confidence metadata for each word.
        """
        frames_by_word: List[List[Tuple[int, int]]] = [[] for _ in words]
        for frame, state in enumerate(path.tolist()):
            word_index = state_to_word[state]
            if word_index is not None:
                frames_by_word[word_index].append((frame, state))

        rows = []
        for word, word_frames in zip(words, frames_by_word):
            if not word_frames:
                raise ValueError(f"CTC path did not visit word {word['word']!r}.")
            frames = [frame for frame, _ in word_frames]
            start = time_offset + frames[0] * ctc_step
            end = min(time_offset + (frames[-1] + 1) * ctc_step, start + self.maximum_token_len)
            end_frame = min(frames[-1], max(frames[0], math.ceil((end - time_offset) / ctc_step) - 1))
            selected = torch.stack([ctc[frame, labels[state]] for frame, state in word_frames])
            column = speaker_mapping.get(word['speaker_tag'])
            speaker_confidence = None
            activity_start = activity_end = None
            if speaker_probs is not None and column is not None:
                activity = speaker_probs[frames, column]
                speaker_confidence = float(activity.mean())
                active = torch.nonzero(activity >= self.speaker_activity_threshold).flatten()
                if active.numel():
                    first, last = frames[int(active[0])], frames[int(active[-1])]
                    activity_start = time_offset + first * ctc_step
                    activity_end = time_offset + (last + 1) * ctc_step
            rows.append(
                {
                    **word,
                    'start': start,
                    'end': end,
                    'start_frame': frames[0],
                    'end_frame': end_frame,
                    'ctc_confidence': float(torch.exp(selected.mean())),
                    'sortformer_column': column,
                    'speaker_confidence': speaker_confidence,
                    'speaker_activity_start': activity_start,
                    'speaker_activity_end': activity_end,
                }
            )
        return rows

    def _resolve_speaker_mapping(
        self,
        speaker_tags: Sequence[int],
        preliminary_rows: Sequence[Dict[str, Any]],
        speaker_probs: Optional[torch.Tensor],
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, List[float]]]:
        """Map t-SOT speaker tags to Sortformer output columns.

        Args:
            speaker_tags (Sequence[int]): Distinct t-SOT speaker tags.
            preliminary_rows (Sequence[Dict[str, Any]]): CTC-only aligned word records.
            speaker_probs (Optional[torch.Tensor]): Speaker probabilities shaped ``(T, S)``.

        Returns:
            Tuple[Dict[int, Optional[int]], Dict[int, List[float]]]: Mapping and assignment scores.
        """
        mapping = {tag: None for tag in speaker_tags}
        if speaker_probs is None or len(speaker_tags) > speaker_probs.shape[1]:
            return mapping, {}
        rows_by_tag = self._group_words_by_speaker(preliminary_rows)
        scores = []
        diagnostics = {}
        for tag in speaker_tags:
            column_scores = []
            for column in range(speaker_probs.shape[1]):
                values = [
                    torch.log(speaker_probs[row['start_frame'] : row['end_frame'] + 1, column].clamp_min(self.epsilon))
                    for row in rows_by_tag[tag]
                ]
                column_scores.append(float(torch.cat(values).mean()))
            diagnostics[tag] = column_scores
            scores.append(column_scores)
        for tag, column in zip(speaker_tags, self._maximum_weight_assignment(scores)):
            mapping[tag] = column
        return mapping, diagnostics

    @staticmethod
    def _maximum_weight_assignment(scores: Sequence[Sequence[float]]) -> List[int]:
        """Find the maximum-weight one-to-one speaker assignment.

        Args:
            scores (Sequence[Sequence[float]]): Score matrix indexed by tag and speaker column.

        Returns:
            List[int]: Selected column index for each score row.
        """
        states: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
        for row in scores:
            next_states = {}
            for used, (total, columns) in states.items():
                for column, score in enumerate(row):
                    if used & (1 << column):
                        continue
                    mask = used | (1 << column)
                    candidate = (total + score, columns + [column])
                    if mask not in next_states or candidate[0] > next_states[mask][0]:
                        next_states[mask] = candidate
            states = next_states
        return max(states.values(), key=lambda item: item[0])[1] if states else []

    @staticmethod
    def _resample_speaker_probs(speaker_probs: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Linearly resample speaker probabilities onto the CTC frame grid.

        Args:
            speaker_probs (torch.Tensor): Speaker probabilities shaped ``(T, S)``.
            target_frames (int): Required number of output frames.

        Returns:
            torch.Tensor: Resampled probabilities shaped ``(target_frames, S)``.
        """
        if speaker_probs.shape[0] == target_frames:
            return speaker_probs
        if speaker_probs.shape[0] == 1:
            return speaker_probs.expand(target_frames, -1)
        positions = torch.linspace(0, speaker_probs.shape[0] - 1, target_frames)
        lower = positions.floor().long()
        upper = positions.ceil().long()
        fraction = (positions - lower).unsqueeze(1)
        return speaker_probs[lower] * (1 - fraction) + speaker_probs[upper] * fraction

    @staticmethod
    def _lengths(values: Optional[torch.Tensor], batch_size: int, maximum: int) -> List[int]:
        """Normalize optional padded-sequence lengths.

        Args:
            values (Optional[torch.Tensor]): Explicit valid lengths.
            batch_size (int): Expected number of lengths.
            maximum (int): Maximum allowed sequence length.

        Returns:
            List[int]: Validated lengths for every batch row.
        """
        if values is None:
            return [maximum] * batch_size
        lengths = [int(value) for value in torch.as_tensor(values).reshape(-1).cpu()]
        if len(lengths) != batch_size or any(length <= 0 or length > maximum for length in lengths):
            raise ValueError("Padded sequence lengths must match the batch and time dimension.")
        return lengths

    def _resolve_blank_id(self, vocab_size: int) -> int:
        """Resolve and validate the CTC blank class index.

        Args:
            vocab_size (int): Number of CTC output classes including blank.

        Returns:
            int: CTC blank class index.
        """
        if self.blank_id is not None:
            blank_id = int(self.blank_id)
        elif self.ctc_decoder is not None:
            blank_id = int(self.ctc_decoder.num_classes_with_blank) - 1
        else:
            blank_id = vocab_size - 1
        if not 0 <= blank_id < vocab_size:
            raise ValueError(f"blank_id={blank_id} is outside the CTC vocabulary.")
        return blank_id

    @staticmethod
    def _frame_seconds(
        length: int,
        audio_duration: Optional[float],
        configured: Optional[float],
        default: float,
    ) -> float:
        """Resolve the duration represented by one model frame.

        Args:
            length (int): Number of valid model frames.
            audio_duration (Optional[float]): Recording duration in seconds.
            configured (Optional[float]): Explicit frame duration.
            default (float): Fallback frame duration.

        Returns:
            float: Duration of one frame in seconds.
        """
        if audio_duration is not None:
            return float(audio_duration) / length
        return float(configured) if configured is not None else default


def _get_ctc_timestamp_extractor(
    encoder: ParallelExpertEncoder,
    model_path: Optional[str],
    device: torch.device,
) -> MultiSpeakerSOTWordTimestampAligner:
    """Load and cache only the CTC adapter components needed for alignment."""
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("ctc_timestamp_model_path must be a non-empty local .nemo path.")
    resolved_path = os.path.realpath(os.path.expanduser(model_path))
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"CTC timestamp model does not exist: {resolved_path}")

    cached = encoder.__dict__.get("_ctc_timestamp_extractor_cache")
    if cached is None or cached[0] != resolved_path:
        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

        adapter = EncDecCTCModelBPE.restore_from(resolved_path, map_location="cpu")
        extractor = MultiSpeakerSOTWordTimestampAligner(
            encoder=encoder,
            ctc_decoder=adapter.decoder,
            tokenizer=adapter.tokenizer,
        )
        # This is an inference-only cache. Keep it out of nn.Module registration so
        # loading an adapter does not mutate the SALM checkpoint state dictionary.
        encoder.__dict__["_ctc_timestamp_extractor_cache"] = (resolved_path, extractor)
        cached = (resolved_path, extractor)

    extractor = cached[1]
    parameter = next(encoder.parameters(), None)
    dtype = parameter.dtype if parameter is not None and device.type != "cpu" else torch.float32
    extractor.ctc_decoder.to(device=device, dtype=dtype).eval()
    return extractor
