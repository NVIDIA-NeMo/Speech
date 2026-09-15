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


class _NoValidCTCViterbiPathError(ValueError):
    """Expose failed padded-stream indices when a CTC Viterbi path is impossible."""

    def __init__(self, failed_stream_indices: Sequence[int], *, active_region_restricted: bool) -> None:
        indices = tuple(int(index) for index in failed_stream_indices)
        if not indices:
            raise ValueError("A CTC Viterbi path error requires at least one failed stream index.")
        self.failed_stream_indices = indices
        self.active_region_restricted = bool(active_region_restricted)
        region_hint = " after restricting to the selected active regions" if self.active_region_restricted else ""
        super().__init__(
            "No valid CTC Viterbi path exists for speaker stream(s) "
            f"{list(indices)}{region_hint}. Lower the active-region threshold, increase the region "
            "padding or merge gap, or use serialized alignment."
        )


class _ParallelActiveRegionError(ValueError):
    """Expose speaker streams whose compact Sortformer region cannot be planned."""

    def __init__(
        self,
        failed_speaker_tags: Sequence[Optional[int]],
        *,
        reason: str,
        details: Optional[Mapping[Optional[int], Mapping[str, Any]]] = None,
    ) -> None:
        speaker_tags = tuple(failed_speaker_tags)
        if not speaker_tags:
            raise ValueError("An active-region error requires at least one speaker tag.")
        self.failed_speaker_tags = speaker_tags
        self.reason = str(reason)
        self.details = {speaker_tag: dict(value) for speaker_tag, value in (details or {}).items()}
        super().__init__(
            "Parallel active-region planning failed for speaker tag(s) "
            f"{list(speaker_tags)} ({self.reason}). Lower the active-region threshold or use serialized alignment."
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
        parallel_speaker_gate_threshold: Optional Sortformer threshold used to
            select each speaker's padded CTC regions in ``parallel`` mode. Frames
            outside those regions are excluded; the padded onset/offset collar
            remains available to CTC token emissions and is weighted by the soft
            Sortformer prior. Long inactive gaps become explicit word-boundary
            separators. Set ``None`` to use the unrestricted CTC timeline.
        parallel_speaker_gate_min_threshold: Lowest automatic retry threshold for
            a parallel active region. A failing speaker retries at progressively
            lower thresholds down to this floor; if it remains infeasible, the
            entire recording falls back to serialized t-SOT CTC alignment.
        parallel_active_region_padding_seconds: Context added on both sides of a
            selected Sortformer speech region; it remains available for CTC
            onset/offset evidence at active-region boundaries.
        parallel_active_region_merge_gap_seconds: Maximum remaining gap between
            padded active regions to merge. Longer gaps remain hard word boundaries.
        coarse_alignment_band_size: Optional half-width, in blank-expanded CTC
            target states, for a coarse-to-fine Viterbi search. When enabled, a
            temporally coarsened first pass supplies a per-frame center state and
            the fine pass evaluates only ``center - N`` through ``center + N``.
            ``None`` (the default) preserves dense, exact Viterbi alignment.
            This is an opt-in approximate accelerator: it falls back to dense DP
            when the coarse path fails, the band is infeasible, or the recovered
            path reaches an artificial band edge.
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
        parallel_speaker_gate_min_threshold: float = 0.20,
        parallel_active_region_padding_seconds: float = 0.16,
        parallel_active_region_merge_gap_seconds: float = 0.40,
        coarse_alignment_band_size: Optional[int] = None,
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
        if not 0.0 <= float(parallel_speaker_gate_min_threshold) <= 1.0:
            raise ValueError("parallel_speaker_gate_min_threshold must be between zero and one.")
        if parallel_active_region_padding_seconds < 0:
            raise ValueError("parallel_active_region_padding_seconds must be non-negative.")
        if parallel_active_region_merge_gap_seconds < 0:
            raise ValueError("parallel_active_region_merge_gap_seconds must be non-negative.")
        coarse_alignment_band_size = self._normalize_coarse_alignment_band_size(coarse_alignment_band_size)
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
        self.parallel_speaker_gate_min_threshold = float(parallel_speaker_gate_min_threshold)
        self.parallel_active_region_padding_seconds = float(parallel_active_region_padding_seconds)
        self.parallel_active_region_merge_gap_seconds = float(parallel_active_region_merge_gap_seconds)
        self.coarse_alignment_band_size = coarse_alignment_band_size
        self.alignment_mode = self._validate_alignment_mode(alignment_mode)
        self.speaker_assignment_mode = self._validate_assignment_mode(speaker_assignment_mode)
        self.epsilon = float(epsilon)

    @classmethod
    def parse_sot_words(cls, sot_transcript: str) -> List[Dict[str, Any]]:
        """Parse a t-SOT transcript without sending speaker tags to the tokenizer.

        A tag remains active until the next tag. Each tag occurrence also gets a
        monotonically increasing ``turn_index`` so reappearing speakers retain their
        original turn boundaries. Text that precedes the first tag is retained with
        ``speaker_tag=None`` rather than being silently assigned to speaker zero.
        """
        if not isinstance(sot_transcript, str):
            raise TypeError(f"sot_transcript must be a string, got {type(sot_transcript).__name__}.")

        words: List[Dict[str, Any]] = []
        active_speaker: Optional[int] = None
        # A t-SOT tag occurrence denotes a turn, even when a speaker tag
        # reappears later. Keep that identity so parallel alignment can retain
        # same-speaker turn order instead of flattening all of a speaker's text.
        active_turn_index: Optional[int] = None
        next_turn_index = 0
        cursor = 0

        def append_words(text: str, speaker_tag: Optional[int], turn_index: Optional[int]) -> None:
            for word in re.findall(r"\S+", text):
                words.append(
                    {
                        'word': word,
                        'speaker_tag': speaker_tag,
                        'turn_index': turn_index,
                        'word_index': len(words),
                    }
                )

        for match in cls._SPEAKER_TAG_RE.finditer(sot_transcript):
            append_words(sot_transcript[cursor : match.start()], active_speaker, active_turn_index)
            active_speaker = int(match.group(1))
            active_turn_index = next_turn_index
            next_turn_index += 1
            cursor = match.end()
        append_words(sot_transcript[cursor:], active_speaker, active_turn_index)
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

    def extract_ctc_and_sortformer_batch(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run the raw PEE experts and CTC head for a padded recording batch.

        The model tensors retain their batch dimension.  Pass the returned
        dictionary to :meth:`extract_from_outputs_batch` with one t-SOT
        transcript per item; that method evaluates the alignment DPs in one
        padded stream batch.  The single-record method remains unchanged for
        backward compatibility.
        """
        if self.encoder is None:
            raise ValueError("encoder is required to run PEE inference.")
        if self.ctc_decoder is None:
            raise ValueError("ctc_decoder is required to run PEE inference.")
        if not isinstance(processed_signal, torch.Tensor) or processed_signal.ndim != 3:
            raise ValueError("processed_signal must have shape (batch, features, frames).")
        if not isinstance(processed_signal_length, torch.Tensor) or processed_signal_length.ndim != 1:
            raise ValueError("processed_signal_length must have shape (batch,).")
        if processed_signal.shape[0] == 0 or processed_signal_length.shape[0] != processed_signal.shape[0]:
            raise ValueError("processed_signal and processed_signal_length must contain the same non-empty batch.")

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
                speech_states, speech_lengths = pee_encoder._run_asr(processed_signal, processed_signal_length)

                # This is the existing PEE diarization inference path with its
                # native embedding lengths retained.  PEE normally returns a
                # padded Sortformer batch, so those lengths are essential to
                # avoid resampling padded predictions for short recordings.
                diar_signal = processed_signal
                if pee_encoder.diar_normalize_type:
                    diar_signal, _, _ = normalize_batch(
                        diar_signal, processed_signal_length, normalize_type=pee_encoder.diar_normalize_type
                    )
                diar_signal = pee_encoder._match_module_io(diar_signal, pee_encoder.diarization_model)
                diar_length = processed_signal_length.to(device=diar_signal.device)
                embeddings, embedding_lengths = pee_encoder.diarization_model.frontend_encoder(
                    processed_signal=diar_signal,
                    processed_signal_length=diar_length,
                    bypass_pre_encode=False,
                )
                native_predictions = pee_encoder.diarization_model.forward_infer(
                    emb_seq=embeddings,
                    emb_seq_length=embedding_lengths,
                )
                sortformer_sigmoids = pee_encoder._align_diarization_output_resolution(
                    native_predictions, embedding_lengths
                )
                ctc_log_probs = self.ctc_decoder(speech_states, encoded_lengths=speech_lengths)
        finally:
            for module, was_training in previous_modes:
                module.train(was_training)

        if not isinstance(speech_states, torch.Tensor) or not isinstance(speech_lengths, torch.Tensor):
            raise RuntimeError("PEE _run_asr() must return (states, lengths) tensors.")
        if not isinstance(ctc_log_probs, torch.Tensor) or ctc_log_probs.ndim != 3:
            raise RuntimeError("TransformerCTCDecoder must return (batch, frames, classes) log-probabilities.")
        if not isinstance(sortformer_sigmoids, torch.Tensor) or sortformer_sigmoids.ndim != 3:
            raise RuntimeError("PEE _run_diarization() must return (batch, frames, speakers) probabilities.")
        if ctc_log_probs.shape[0] != processed_signal.shape[0] or sortformer_sigmoids.shape[0] != processed_signal.shape[0]:
            raise RuntimeError("PEE expert branches and CTC head returned inconsistent batch sizes.")
        if sortformer_sigmoids.shape[1] == 0:
            raise RuntimeError("PEE _run_diarization() returned no Sortformer frames.")

        speech_lengths = speech_lengths.detach().to(device=ctc_log_probs.device, dtype=torch.long)
        embedding_lengths = embedding_lengths.detach().to(device=ctc_log_probs.device, dtype=torch.long)
        if speech_lengths.ndim != 1 or speech_lengths.shape[0] != processed_signal.shape[0]:
            raise RuntimeError("PEE _run_asr() returned invalid batch lengths.")
        if (speech_lengths < 1).any() or (speech_lengths > ctc_log_probs.shape[1]).any():
            raise RuntimeError("PEE _run_asr() returned lengths outside the CTC output time dimension.")

        diar_model = pee_encoder.diarization_model
        native_factor = 1 if diar_model.high_resolution else int(diar_model.encoder.subsampling_factor)
        downsample_factor = int(diar_model.output_subsampling_factor) // native_factor
        if downsample_factor <= 1:
            sortformer_lengths = embedding_lengths
        else:
            native_lengths = embedding_lengths * (int(diar_model.encoder.subsampling_factor) // native_factor)
            sortformer_lengths = torch.div(
                native_lengths + downsample_factor - 1,
                downsample_factor,
                rounding_mode='floor',
            )
        sortformer_lengths = sortformer_lengths.clamp(min=1, max=sortformer_sigmoids.shape[1])
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

    def extract_from_audio_batch(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
        preprocessor: nn.Module,
        sot_transcripts: Sequence[str],
        *,
        audio_durations: Optional[Sequence[Optional[float]]] = None,
        time_offsets: Optional[Sequence[float]] = None,
        **alignment_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Batch preprocess, PEE inference, and timestamp alignment.

        ``input_signal`` is a padded waveform batch and ``sot_transcripts`` has
        exactly one generated t-SOT transcript per waveform.  PEE runs once on
        the waveform batch; :meth:`extract_from_outputs_batch` then flattens
        every serialized or per-speaker stream across recordings into one padded
        Viterbi batch.
        """
        if not isinstance(preprocessor, nn.Module):
            raise TypeError(f"preprocessor must be an nn.Module, got {type(preprocessor).__name__}.")
        if not isinstance(input_signal, torch.Tensor) or input_signal.ndim < 2:
            raise ValueError("input_signal must have a leading batch dimension and waveform samples.")
        if not isinstance(input_signal_length, torch.Tensor) or input_signal_length.ndim != 1:
            raise ValueError("input_signal_length must have shape (batch,).")
        batch_size = input_signal.shape[0]
        if batch_size == 0 or input_signal_length.shape[0] != batch_size:
            raise ValueError("input_signal and input_signal_length must contain the same non-empty batch.")
        self._validate_sot_transcripts(sot_transcripts, batch_size)

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
        model_outputs = self.extract_ctc_and_sortformer_batch(processed_signal, processed_signal_length)

        if audio_durations is None:
            sample_rate = getattr(preprocessor, '_sample_rate', getattr(preprocessor, 'sample_rate', None))
            if sample_rate is not None:
                input_lengths = self._select_batch_lengths(
                    input_signal_length,
                    batch_size,
                    input_signal.shape[-1],
                    'input_signal_length',
                )
                audio_durations = [length / float(sample_rate) for length in input_lengths]

        return self.extract_from_outputs_batch(
            sot_transcripts=sot_transcripts,
            audio_durations=audio_durations,
            time_offsets=time_offsets,
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
            parallel_speaker_gate_threshold: Per-call Sortformer threshold that
                selects padded active CTC regions. Omit it to use the extractor
                default; pass ``None`` to align the unrestricted CTC timeline.

        Returns:
            A dictionary whose ``speaker_word_timestamps`` contains one ordered list
            per t-SOT speaker tag.  Word ``start`` / ``end`` are CTC-derived seconds;
            Sortformer details are supplied as activity/confidence metadata.
        """
        requested_mode = self._validate_alignment_mode(alignment_mode or self.alignment_mode)
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

        parsed_sot_words = self.parse_sot_words(sot_transcript)
        sot_speaker_tags = self._speaker_tags_in_order(parsed_sot_words)
        speaker_count_policy = self._resolve_sot_sortformer_count_policy(
            requested_alignment_mode=requested_mode,
            speaker_assignment_mode=assignment_mode,
            speaker_tags=sot_speaker_tags,
            speaker_probs=sortformer_on_ctc,
        )
        mode = speaker_count_policy['effective_alignment_mode']
        sortformer_for_alignment = (
            sortformer_on_ctc if speaker_count_policy['use_sortformer_for_alignment'] else None
        )
        candidate_sortformer_columns = speaker_count_policy['selected_sortformer_columns']
        tokenized_words = self._tokenize_words(
            parsed_sot_words,
            blank_id,
            alignment_mode=mode,
        )
        needs_parallel_turn_fences = mode == 'parallel' and self._has_repeated_speaker_turns(tokenized_words)
        serialized_turn_anchor_words = (
            self._tokenize_words(parsed_sot_words, blank_id, alignment_mode='serialized')
            if needs_parallel_turn_fences
            else None
        )
        speaker_tags = self._speaker_tags_in_order(tokenized_words)
        if not tokenized_words:
            return {
                'speaker_word_timestamps': {},
                'speaker_tag_to_sortformer_column': {},
                'alignment_mode': mode,
                'requested_alignment_mode': requested_mode,
                'speaker_assignment_mode': assignment_mode,
                'ctc_frame_seconds': ctc_step_seconds,
                'sortformer_frame_seconds': sortformer_step_seconds,
                'time_offset': time_offset,
                'num_ctc_frames': ctc_length,
                'num_sortformer_frames': sortformer_length,
                'ctc_log_normalizer_error': ctc_log_normalizer_error,
                'alignment_diagnostics': {
                    'speaker_count_policy': speaker_count_policy,
                    'parallel_speaker_gate_min_threshold': self.parallel_speaker_gate_min_threshold,
                    'alignment_fallback': None,
                    'coarse_alignment_band_size': self.coarse_alignment_band_size,
                },
            }

        # Learn the t-SOT-tag-to-Sortformer-column mapping from pure CTC paths.
        # In parallel mode these preliminary paths are also independent, avoiding
        # any artificial ordering constraint for overlapping speaker transcripts.
        preliminary_scores: Dict[Optional[int], float] = {}
        preliminary_coarse_diagnostics: Dict[Optional[int], Dict[str, Any]] = {}
        if mode == 'serialized':
            preliminary_rows, preliminary_score, preliminary_coarse_diagnostic = self._align_word_sequence(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=None,
                speaker_mapping={},
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=0.0,
                use_coarse_alignment=False,
            )
            preliminary_scores[None] = preliminary_score
            preliminary_coarse_diagnostics[None] = preliminary_coarse_diagnostic
        else:
            (
                preliminary_rows,
                preliminary_scores,
                preliminary_coarse_diagnostics,
            ) = self._align_parallel_word_streams_batched(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=None,
                speaker_mapping={},
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=0.0,
                speaker_gate_threshold=None,
                use_coarse_alignment=False,
            )

        parallel_turn_word_bounds: Dict[int, Tuple[int, int]] = {}
        parallel_turn_diagnostics: Dict[Optional[int], List[Dict[str, Any]]] = {}
        serialized_turn_anchor_score: Optional[float] = None
        serialized_turn_anchor_coarse_diagnostic: Optional[Dict[str, Any]] = None
        if serialized_turn_anchor_words is not None:
            serialized_turn_anchor_rows, serialized_turn_anchor_score, serialized_turn_anchor_coarse_diagnostic = (
                self._align_word_sequence(
                    tokenized_words=serialized_turn_anchor_words,
                    ctc_log_probs=ctc,
                    blank_id=blank_id,
                    speaker_probs=None,
                    speaker_mapping={},
                    ctc_step_seconds=ctc_step_seconds,
                    time_offset=time_offset,
                    speaker_logprob_weight=0.0,
                    use_coarse_alignment=False,
                )
            )
            parallel_turn_word_bounds, parallel_turn_diagnostics = self._build_parallel_turn_frame_bounds(
                tokenized_words=tokenized_words,
                serialized_anchor_rows=serialized_turn_anchor_rows,
                ctc_num_frames=ctc.shape[0],
            )

        speaker_mapping, assignment_scores = self._resolve_speaker_mapping(
            speaker_tags=speaker_tags,
            preliminary_rows=preliminary_rows,
            speaker_probs=sortformer_for_alignment,
            assignment_mode=assignment_mode,
            candidate_columns=candidate_sortformer_columns,
        )

        alignment_scores: Dict[Optional[int], float] = {}
        final_coarse_diagnostics: Dict[Optional[int], Dict[str, Any]] = {}
        parallel_active_region_diagnostics: Dict[Optional[int], Dict[str, Any]] = {}
        alignment_fallback: Optional[Dict[str, Any]] = None
        parallel_retry_state: Optional[Dict[str, Any]] = None
        if mode == 'serialized':
            rows, path_score, final_coarse_diagnostic = self._align_word_sequence(
                tokenized_words=tokenized_words,
                ctc_log_probs=ctc,
                blank_id=blank_id,
                speaker_probs=sortformer_for_alignment,
                speaker_mapping=speaker_mapping,
                ctc_step_seconds=ctc_step_seconds,
                time_offset=time_offset,
                speaker_logprob_weight=float(speaker_weight),
                speaker_gate_threshold=None,
            )
            alignment_scores[None] = path_score
            final_coarse_diagnostics[None] = final_coarse_diagnostic
        else:
            parallel_retry_state = self._new_parallel_gate_retry_state(
                tokenized_words,
                parallel_gate_threshold,
            )
            parallel_timelines, parallel_active_region_diagnostics, terminal_failure = (
                self._build_parallel_active_timelines_with_retries(
                    tokenized_words=tokenized_words,
                    speaker_mapping=speaker_mapping,
                    speaker_probs=sortformer_for_alignment,
                    ctc_num_frames=ctc.shape[0],
                    ctc_step_seconds=ctc_step_seconds,
                    blank_id=blank_id,
                    retry_state=parallel_retry_state,
                )
            )
            parallel_stream_tags = list(self._group_words_by_speaker(tokenized_words))
            while parallel_timelines is not None:
                try:
                    rows, alignment_scores, final_coarse_diagnostics = self._align_parallel_word_streams_batched(
                        tokenized_words=tokenized_words,
                        ctc_log_probs=ctc,
                        blank_id=blank_id,
                        speaker_probs=sortformer_for_alignment,
                        speaker_mapping=speaker_mapping,
                        ctc_step_seconds=ctc_step_seconds,
                        time_offset=time_offset,
                        speaker_logprob_weight=float(speaker_weight),
                        # The compact timeline excludes frames outside the padded
                        # Sortformer-active regions. Do not hard-mask token emissions
                        # again inside its collar: genuine onset/offset phones can fall
                        # just below the activity threshold, while the soft prior and
                        # t-SOT turn fences still constrain the path.
                        speaker_gate_threshold=None,
                        speaker_timelines=parallel_timelines,
                        word_source_frame_bounds=parallel_turn_word_bounds or None,
                    )
                    terminal_failure = None
                    break
                except _NoValidCTCViterbiPathError as error:
                    failed_speaker_tags: List[Optional[int]] = []
                    for failed_stream_index in error.failed_stream_indices:
                        if not 0 <= failed_stream_index < len(parallel_stream_tags):
                            raise RuntimeError("CTC Viterbi reported an invalid parallel stream index.") from error
                        failed_speaker_tags.append(parallel_stream_tags[failed_stream_index])
                    if not failed_speaker_tags:
                        raise
                    exhausted = self._advance_parallel_gate_retry_state(
                        parallel_retry_state,
                        failed_speaker_tags,
                    )
                    if exhausted:
                        terminal_failure = {
                            'reason': 'no_valid_ctc_viterbi_path_in_active_regions',
                            'failed_speaker_tags': failed_speaker_tags,
                        }
                        parallel_timelines = None
                        break
                    (
                        parallel_timelines,
                        parallel_active_region_diagnostics,
                        terminal_failure,
                    ) = self._build_parallel_active_timelines_with_retries(
                        tokenized_words=tokenized_words,
                        speaker_mapping=speaker_mapping,
                        speaker_probs=sortformer_for_alignment,
                        ctc_num_frames=ctc.shape[0],
                        ctc_step_seconds=ctc_step_seconds,
                        blank_id=blank_id,
                        retry_state=parallel_retry_state,
                    )

            if parallel_timelines is None:
                if terminal_failure is None:
                    raise RuntimeError("Parallel active-region retry ended without an outcome.")
                alignment_fallback = self._parallel_serialized_fallback_diagnostic(
                    retry_state=parallel_retry_state,
                    reason=str(terminal_failure['reason']),
                    failed_speaker_tags=terminal_failure.get('failed_speaker_tags', []),
                    capacity_failures=terminal_failure.get('capacity_failures'),
                )
                # The t-SOT transcript is authoritative. Do not reuse the
                # speaker-wise tokenization or any Sortformer mapping from the
                # failed parallel attempt: serialized CTC must see the original
                # word order and no speaker prior.
                mode = 'serialized'
                tokenized_words = self._tokenize_words(
                    parsed_sot_words,
                    blank_id,
                    alignment_mode='serialized',
                )
                speaker_mapping = {
                    speaker_tag: None for speaker_tag in self._speaker_tags_in_order(tokenized_words)
                }
                assignment_scores = {}
                rows, path_score, final_coarse_diagnostic = self._align_word_sequence(
                    tokenized_words=tokenized_words,
                    ctc_log_probs=ctc,
                    blank_id=blank_id,
                    speaker_probs=None,
                    speaker_mapping=speaker_mapping,
                    ctc_step_seconds=ctc_step_seconds,
                    time_offset=time_offset,
                    speaker_logprob_weight=0.0,
                    speaker_gate_threshold=None,
                )
                alignment_scores = {None: path_score}
                final_coarse_diagnostics = {None: final_coarse_diagnostic}

        speaker_word_timestamps: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for row in rows:
            speaker_word_timestamps.setdefault(row['speaker_tag'], []).append(row)

        return {
            'speaker_word_timestamps': speaker_word_timestamps,
            'speaker_tag_to_sortformer_column': speaker_mapping,
            'alignment_mode': mode,
            'requested_alignment_mode': requested_mode,
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
                'speaker_count_policy': speaker_count_policy,
                'parallel_speaker_gate_threshold': parallel_gate_threshold if mode == 'parallel' else None,
                'parallel_speaker_gate_min_threshold': self.parallel_speaker_gate_min_threshold,
                'parallel_active_regions': parallel_active_region_diagnostics if mode == 'parallel' else {},
                'alignment_fallback': alignment_fallback,
                'parallel_active_region_padding_seconds': self.parallel_active_region_padding_seconds,
                'parallel_active_region_merge_gap_seconds': self.parallel_active_region_merge_gap_seconds,
                'parallel_turn_fences': parallel_turn_diagnostics if mode == 'parallel' else {},
                'parallel_turn_anchor_path_score': serialized_turn_anchor_score,
                'parallel_turn_anchor_coarse_alignment': serialized_turn_anchor_coarse_diagnostic,
                'coarse_alignment_band_size': self.coarse_alignment_band_size,
                # Speaker-column assignment must remain globally exact: a narrow
                # coarse-to-fine path can alter the evidence used by that mapping.
                'preliminary_forced_dense_for_speaker_mapping': self.coarse_alignment_band_size is not None,
                'coarse_alignment': {
                    'preliminary': preliminary_coarse_diagnostics,
                    'final': final_coarse_diagnostics,
                },
            },
        }

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
        alignment_mode: Optional[str] = None,
        speaker_assignment_mode: Optional[str] = None,
        speaker_logprob_weight: Optional[float] = None,
        parallel_speaker_gate_threshold: Any = _UNSET_PARALLEL_SPEAKER_GATE,
    ) -> List[Dict[str, Any]]:
        """Force-align a padded batch of independent t-SOT recordings together.

        Unlike a Python loop over :meth:`extract_from_outputs`, this method
        concatenates the valid CTC grids into a private global grid, flattens
        every record's serialized target or independent speaker target into one
        padded stream collection, and executes batched preliminary and final
        Viterbi passes. When a speaker has multiple t-SOT turns, one additional
        batched serialized CTC-only anchor pass provides same-speaker turn fences.
        Per-stream source-frame indices retain record-local ownership, so no path
        can cross between recordings.

        Args mirror :meth:`extract_from_outputs`, except tensor inputs carry a
        leading batch dimension and sequence arguments contain one entry per
        recording.  Record lengths are required whenever the model tensors are
        padded.  The returned list preserves input batch order.
        """
        if not isinstance(ctc_log_probs, torch.Tensor) or ctc_log_probs.ndim != 3:
            raise ValueError("ctc_log_probs must have shape (batch, frames, classes).")
        batch_size, max_ctc_frames, ctc_vocab_size = ctc_log_probs.shape
        if batch_size == 0 or max_ctc_frames == 0 or ctc_vocab_size < 2:
            raise ValueError("ctc_log_probs must contain a non-empty batch, time grid, and CTC vocabulary.")
        self._validate_sot_transcripts(sot_transcripts, batch_size)
        requested_mode = self._validate_alignment_mode(alignment_mode or self.alignment_mode)
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

        ctc_lengths_list = self._select_batch_lengths(
            ctc_lengths, batch_size, max_ctc_frames, 'ctc_lengths'
        )
        blank_id = self._resolve_blank_id(ctc_vocab_size)
        ctc_cpu = ctc_log_probs.detach().to(device='cpu', dtype=torch.float32)

        sortformer_cpu: Optional[torch.Tensor] = None
        sortformer_lengths_list: List[Optional[int]]
        if sortformer_sigmoids is None:
            sortformer_lengths_list = [None] * batch_size
        else:
            if not isinstance(sortformer_sigmoids, torch.Tensor) or sortformer_sigmoids.ndim != 3:
                raise ValueError("sortformer_sigmoids must have shape (batch, frames, speakers).")
            if sortformer_sigmoids.shape[0] != batch_size or sortformer_sigmoids.shape[1] == 0:
                raise ValueError("sortformer_sigmoids must have the same non-empty batch and time dimensions.")
            if sortformer_sigmoids.shape[2] == 0:
                raise ValueError("sortformer_sigmoids must contain at least one speaker column.")
            sortformer_lengths_list = [
                int(length)
                for length in self._select_batch_lengths(
                    sortformer_lengths,
                    batch_size,
                    int(sortformer_sigmoids.shape[1]),
                    'sortformer_lengths',
                )
            ]
            sortformer_cpu = sortformer_sigmoids.detach().to(device='cpu', dtype=torch.float32)

        audio_duration_list = self._select_optional_float_sequence(
            audio_durations, batch_size, 'audio_durations'
        )
        time_offset_list = self._select_float_sequence(time_offsets, batch_size, 'time_offsets', default=0.0)

        records: List[Dict[str, Any]] = []
        for record_index in range(batch_size):
            ctc_length = ctc_lengths_list[record_index]
            ctc = ctc_cpu[record_index, :ctc_length]
            if torch.isnan(ctc).any():
                raise ValueError(f"ctc_log_probs[{record_index}] contains NaN values.")
            ctc_log_normalizer_error = float(torch.logsumexp(ctc, dim=-1).abs().max().item())
            if ctc_log_normalizer_error > 0.05:
                raise ValueError(
                    f"ctc_log_probs[{record_index}] does not appear to be log-softmax output: maximum "
                    f"log-normalization error is {ctc_log_normalizer_error:.4f}."
                )

            sortformer: Optional[torch.Tensor] = None
            sortformer_on_ctc: Optional[torch.Tensor] = None
            sortformer_length: Optional[int] = None
            if sortformer_cpu is not None:
                sortformer_length = sortformer_lengths_list[record_index]
                assert sortformer_length is not None
                sortformer = sortformer_cpu[record_index, :sortformer_length]
                if torch.isnan(sortformer).any():
                    raise ValueError(f"sortformer_sigmoids[{record_index}] contains NaN values.")
                if float(sortformer.min().item()) < -1.0e-3 or float(sortformer.max().item()) > 1.001:
                    raise ValueError(f"sortformer_sigmoids[{record_index}] must contain sigmoid probabilities in [0, 1].")
                sortformer_on_ctc = self._resample_speaker_probs(sortformer.clamp(min=0.0, max=1.0), ctc_length)

            audio_duration = audio_duration_list[record_index]
            if audio_duration is not None and audio_duration <= 0:
                raise ValueError(f"audio_durations[{record_index}] must be positive, got {audio_duration}.")
            time_offset = time_offset_list[record_index]
            ctc_step_seconds = self._resolve_ctc_frame_seconds(ctc_length, audio_duration)
            sortformer_step_seconds = self._resolve_sortformer_frame_seconds(
                sortformer_length,
                audio_duration,
                ctc_step_seconds,
            )
            parsed_sot_words = self.parse_sot_words(sot_transcripts[record_index])
            sot_speaker_tags = self._speaker_tags_in_order(parsed_sot_words)
            speaker_count_policy = self._resolve_sot_sortformer_count_policy(
                requested_alignment_mode=requested_mode,
                speaker_assignment_mode=assignment_mode,
                speaker_tags=sot_speaker_tags,
                speaker_probs=sortformer_on_ctc,
            )
            record_mode = speaker_count_policy['effective_alignment_mode']
            sortformer_for_alignment = (
                sortformer_on_ctc if speaker_count_policy['use_sortformer_for_alignment'] else None
            )
            candidate_sortformer_columns = speaker_count_policy['selected_sortformer_columns']
            tokenized_words = self._tokenize_words(
                parsed_sot_words,
                blank_id,
                alignment_mode=record_mode,
            )
            needs_parallel_turn_fences = (
                record_mode == 'parallel' and self._has_repeated_speaker_turns(tokenized_words)
            )
            serialized_turn_anchor_words = (
                self._tokenize_words(parsed_sot_words, blank_id, alignment_mode='serialized')
                if needs_parallel_turn_fences
                else None
            )
            records.append(
                {
                    'record_index': record_index,
                    'ctc': ctc,
                    'sortformer_on_ctc': sortformer_on_ctc,
                    'sortformer_for_alignment': sortformer_for_alignment,
                    'sortformer_length': sortformer_length,
                    'ctc_step_seconds': ctc_step_seconds,
                    'sortformer_step_seconds': sortformer_step_seconds,
                    'time_offset': time_offset,
                    'ctc_log_normalizer_error': ctc_log_normalizer_error,
                    'requested_alignment_mode': requested_mode,
                    'alignment_mode': record_mode,
                    'parsed_sot_words': parsed_sot_words,
                    'speaker_count_policy': speaker_count_policy,
                    'alignment_fallback': None,
                    'parallel_timelines': None,
                    'parallel_gate_retry_state': None,
                    'candidate_sortformer_columns': candidate_sortformer_columns,
                    'tokenized_words': tokenized_words,
                    'serialized_turn_anchor_words': serialized_turn_anchor_words,
                    'parallel_turn_word_bounds': {},
                    'parallel_turn_diagnostics': {},
                    'serialized_turn_anchor_score': None,
                    'serialized_turn_anchor_coarse_diagnostic': None,
                    'speaker_tags': self._speaker_tags_in_order(tokenized_words),
                }
            )

        # No active streams are needed for empty transcripts, but retain the
        # same result schema as the single-record public method.
        results: List[Optional[Dict[str, Any]]] = [None] * batch_size
        active_records = [record for record in records if record['tokenized_words']]
        for record in records:
            if record['tokenized_words']:
                continue
            results[record['record_index']] = self._build_batched_alignment_result(
                record=record,
                alignment_mode=record['alignment_mode'],
                speaker_assignment_mode=assignment_mode,
                speaker_mapping={},
                rows=[],
                preliminary_scores={},
                final_scores={},
                assignment_scores={},
                preliminary_coarse_diagnostics={},
                final_coarse_diagnostics={},
                parallel_active_region_diagnostics={},
                parallel_gate_threshold=parallel_gate_threshold,
            )
        if not active_records:
            return [result for result in results if result is not None]

        # Each global CTC-frame index belongs to exactly one record.  The stream
        # DP accepts arbitrary source-frame indices, so this retains one shared
        # padded trellis without changing the existing per-record row formatter.
        ctc_offset = 0
        global_ctc_parts: List[torch.Tensor] = []
        global_speaker_parts: List[torch.Tensor] = []
        for record in active_records:
            record['ctc_offset'] = ctc_offset
            ctc_offset += int(record['ctc'].shape[0])
            global_ctc_parts.append(record['ctc'])
            if sortformer_cpu is not None:
                speaker_probs = record['sortformer_on_ctc']
                if speaker_probs is None:
                    raise RuntimeError("Batch Sortformer packing lost a recording's speaker probabilities.")
                global_speaker_parts.append(speaker_probs)
        global_ctc = torch.cat(global_ctc_parts, dim=0)
        global_speaker_probs = torch.cat(global_speaker_parts, dim=0) if global_speaker_parts else None

        preliminary_streams: List[Dict[str, Any]] = []
        for record in active_records:
            if record['alignment_mode'] == 'serialized':
                stream_groups = [(None, record['tokenized_words'])]
            else:
                stream_groups = list(self._group_words_by_speaker(record['tokenized_words']).items())
            for stream_key, stream_words in stream_groups:
                num_frames = int(record['ctc'].shape[0])
                preliminary_streams.append(
                    self._build_batched_record_stream(
                        record=record,
                        stream_key=stream_key,
                        tokenized_words=stream_words,
                        blank_id=blank_id,
                        speaker_mapping={},
                        source_frame_indices=torch.arange(num_frames, dtype=torch.long),
                        active_region_ids=torch.zeros(num_frames, dtype=torch.long),
                    )
                )
        preliminary_alignment = self._align_record_streams_batched(
            streams=preliminary_streams,
            ctc_log_probs=global_ctc,
            speaker_probs=None,
            blank_id=blank_id,
            speaker_logprob_weight=0.0,
            speaker_gate_threshold=None,
            use_coarse_alignment=False,
        )

        preliminary_rows: Dict[int, List[Dict[str, Any]]] = {
            int(record['record_index']): [] for record in active_records
        }
        preliminary_scores: Dict[int, Dict[Optional[int], float]] = {
            int(record['record_index']): {} for record in active_records
        }
        preliminary_coarse_diagnostics: Dict[int, Dict[Optional[int], Dict[str, Any]]] = {
            int(record['record_index']): {} for record in active_records
        }
        for aligned in preliminary_alignment:
            stream = aligned['stream']
            record = stream['record']
            record_index = int(record['record_index'])
            preliminary_rows[record_index].extend(
                self._word_rows_from_path(
                    tokenized_words=stream['tokenized_words'],
                    labels=stream['labels'],
                    state_to_word=stream['state_to_word'],
                    path=aligned['path'],
                    ctc_log_probs=record['ctc'],
                    speaker_probs=None,
                    speaker_mapping={},
                    ctc_step_seconds=record['ctc_step_seconds'],
                    time_offset=record['time_offset'],
                    source_frame_indices=stream['source_frame_indices'],
                    active_region_ids=stream['active_region_ids'],
                )
            )
            preliminary_scores[record_index][stream['stream_key']] = aligned['score']
            preliminary_coarse_diagnostics[record_index][stream['stream_key']] = aligned['diagnostic']

        # A separate serialized CTC-only guide retains the original t-SOT turn
        # order. Its same-speaker midpoints become per-token source-frame fences
        # for the final parallel streams. All guides share one padded DP batch.
        serialized_turn_anchor_streams: List[Dict[str, Any]] = []
        for record in active_records:
            anchor_words = record['serialized_turn_anchor_words']
            if anchor_words is None:
                continue
            num_frames = int(record['ctc'].shape[0])
            serialized_turn_anchor_streams.append(
                self._build_batched_record_stream(
                    record=record,
                    stream_key=None,
                    tokenized_words=anchor_words,
                    blank_id=blank_id,
                    speaker_mapping={},
                    source_frame_indices=torch.arange(num_frames, dtype=torch.long),
                    active_region_ids=torch.zeros(num_frames, dtype=torch.long),
                )
            )
        if serialized_turn_anchor_streams:
            serialized_turn_anchor_alignment = self._align_record_streams_batched(
                streams=serialized_turn_anchor_streams,
                ctc_log_probs=global_ctc,
                speaker_probs=None,
                blank_id=blank_id,
                speaker_logprob_weight=0.0,
                speaker_gate_threshold=None,
                use_coarse_alignment=False,
            )
            for aligned in serialized_turn_anchor_alignment:
                stream = aligned['stream']
                record = stream['record']
                anchor_rows = self._word_rows_from_path(
                    tokenized_words=stream['tokenized_words'],
                    labels=stream['labels'],
                    state_to_word=stream['state_to_word'],
                    path=aligned['path'],
                    ctc_log_probs=record['ctc'],
                    speaker_probs=None,
                    speaker_mapping={},
                    ctc_step_seconds=record['ctc_step_seconds'],
                    time_offset=record['time_offset'],
                    source_frame_indices=stream['source_frame_indices'],
                    active_region_ids=stream['active_region_ids'],
                )
                bounds, diagnostics = self._build_parallel_turn_frame_bounds(
                    tokenized_words=record['tokenized_words'],
                    serialized_anchor_rows=anchor_rows,
                    ctc_num_frames=int(record['ctc'].shape[0]),
                )
                record['parallel_turn_word_bounds'] = bounds
                record['parallel_turn_diagnostics'] = diagnostics
                record['serialized_turn_anchor_score'] = aligned['score']
                record['serialized_turn_anchor_coarse_diagnostic'] = aligned['diagnostic']

        for record in active_records:
            record_index = int(record['record_index'])
            speaker_mapping, assignment_scores = self._resolve_speaker_mapping(
                speaker_tags=record['speaker_tags'],
                preliminary_rows=preliminary_rows[record_index],
                speaker_probs=record['sortformer_for_alignment'],
                assignment_mode=assignment_mode,
                candidate_columns=record['candidate_sortformer_columns'],
            )
            record['speaker_mapping'] = speaker_mapping
            record['assignment_scores'] = assignment_scores

        parallel_active_region_diagnostics: Dict[int, Dict[Optional[int], Dict[str, Any]]] = {
            int(record['record_index']): {} for record in active_records
        }
        for record in active_records:
            if record['alignment_mode'] != 'parallel':
                continue
            retry_state = self._new_parallel_gate_retry_state(
                record['tokenized_words'],
                parallel_gate_threshold,
            )
            record['parallel_gate_retry_state'] = retry_state
            timelines, diagnostics, terminal_failure = self._build_parallel_active_timelines_with_retries(
                tokenized_words=record['tokenized_words'],
                speaker_mapping=record['speaker_mapping'],
                speaker_probs=record['sortformer_on_ctc'],
                ctc_num_frames=int(record['ctc'].shape[0]),
                ctc_step_seconds=record['ctc_step_seconds'],
                blank_id=blank_id,
                retry_state=retry_state,
            )
            record_index = int(record['record_index'])
            parallel_active_region_diagnostics[record_index] = diagnostics
            if terminal_failure is not None:
                self._switch_batched_record_to_serialized_fallback(
                    record=record,
                    blank_id=blank_id,
                    retry_state=retry_state,
                    terminal_failure=terminal_failure,
                )
                continue
            if timelines is None:
                raise RuntimeError("Parallel active-region planning ended without a timeline or fallback.")
            record['parallel_timelines'] = timelines

        final_streams = self._build_final_batched_streams(
            records=active_records,
            blank_id=blank_id,
        )
        while True:
            try:
                final_alignment = self._align_record_streams_batched(
                    streams=final_streams,
                    ctc_log_probs=global_ctc,
                    speaker_probs=global_speaker_probs,
                    blank_id=blank_id,
                    speaker_logprob_weight=float(speaker_weight),
                    # The compact timeline itself is the hard active-region selection.
                    # Keep its padded collar available for CTC onset/offset tokens.
                    speaker_gate_threshold=None,
                    use_coarse_alignment=True,
                )
                break
            except _NoValidCTCViterbiPathError as error:
                failed_tags_by_record: Dict[int, List[Optional[int]]] = {}
                failed_records: Dict[int, Dict[str, Any]] = {}
                for failed_stream_index in error.failed_stream_indices:
                    if not 0 <= failed_stream_index < len(final_streams):
                        raise RuntimeError("CTC Viterbi reported an invalid batched parallel stream index.") from error
                    stream = final_streams[failed_stream_index]
                    record = stream['record']
                    if record['alignment_mode'] != 'parallel':
                        # Serialized CTC is the terminal fallback. If it has no
                        # valid path, the transcript itself cannot be aligned and
                        # there is no less-constrained mode left to try.
                        raise
                    record_index = int(record['record_index'])
                    failed_records[record_index] = record
                    failed_tags_by_record.setdefault(record_index, []).append(stream['stream_key'])
                if not failed_records:
                    raise

                for record_index, record in failed_records.items():
                    retry_state = record.get('parallel_gate_retry_state')
                    if not isinstance(retry_state, dict):
                        raise RuntimeError("Parallel batch record is missing its adaptive gate retry state.")
                    failed_speaker_tags = failed_tags_by_record[record_index]
                    exhausted = self._advance_parallel_gate_retry_state(
                        retry_state,
                        failed_speaker_tags,
                    )
                    if exhausted:
                        terminal_failure: Dict[str, Any] = {
                            'reason': 'no_valid_ctc_viterbi_path_in_active_regions',
                            'failed_speaker_tags': failed_speaker_tags,
                        }
                        self._switch_batched_record_to_serialized_fallback(
                            record=record,
                            blank_id=blank_id,
                            retry_state=retry_state,
                            terminal_failure=terminal_failure,
                        )
                        continue

                    timelines, diagnostics, terminal_failure = self._build_parallel_active_timelines_with_retries(
                        tokenized_words=record['tokenized_words'],
                        speaker_mapping=record['speaker_mapping'],
                        speaker_probs=record['sortformer_on_ctc'],
                        ctc_num_frames=int(record['ctc'].shape[0]),
                        ctc_step_seconds=record['ctc_step_seconds'],
                        blank_id=blank_id,
                        retry_state=retry_state,
                    )
                    parallel_active_region_diagnostics[record_index] = diagnostics
                    if terminal_failure is not None:
                        self._switch_batched_record_to_serialized_fallback(
                            record=record,
                            blank_id=blank_id,
                            retry_state=retry_state,
                            terminal_failure=terminal_failure,
                        )
                        continue
                    if timelines is None:
                        raise RuntimeError("Parallel active-region retry ended without a timeline or fallback.")
                    record['parallel_timelines'] = timelines

                # Repack the shared DP so healthy records remain batched while
                # only the failing record(s) use their relaxed regions or one
                # serialized t-SOT stream.
                final_streams = self._build_final_batched_streams(
                    records=active_records,
                    blank_id=blank_id,
                )

        final_rows: Dict[int, List[Dict[str, Any]]] = {
            int(record['record_index']): [] for record in active_records
        }
        final_scores: Dict[int, Dict[Optional[int], float]] = {
            int(record['record_index']): {} for record in active_records
        }
        final_coarse_diagnostics: Dict[int, Dict[Optional[int], Dict[str, Any]]] = {
            int(record['record_index']): {} for record in active_records
        }
        for aligned in final_alignment:
            stream = aligned['stream']
            record = stream['record']
            record_index = int(record['record_index'])
            final_rows[record_index].extend(
                self._word_rows_from_path(
                    tokenized_words=stream['tokenized_words'],
                    labels=stream['labels'],
                    state_to_word=stream['state_to_word'],
                    path=aligned['path'],
                    ctc_log_probs=record['ctc'],
                    speaker_probs=record['sortformer_on_ctc'],
                    speaker_mapping=record['speaker_mapping'],
                    ctc_step_seconds=record['ctc_step_seconds'],
                    time_offset=record['time_offset'],
                    source_frame_indices=stream['source_frame_indices'],
                    active_region_ids=stream['active_region_ids'],
                )
            )
            final_scores[record_index][stream['stream_key']] = aligned['score']
            final_coarse_diagnostics[record_index][stream['stream_key']] = aligned['diagnostic']

        for record in active_records:
            record_index = int(record['record_index'])
            results[record_index] = self._build_batched_alignment_result(
                record=record,
                alignment_mode=record['alignment_mode'],
                speaker_assignment_mode=assignment_mode,
                speaker_mapping=record['speaker_mapping'],
                rows=final_rows[record_index],
                preliminary_scores=preliminary_scores[record_index],
                final_scores=final_scores[record_index],
                assignment_scores=record['assignment_scores'],
                preliminary_coarse_diagnostics=preliminary_coarse_diagnostics[record_index],
                final_coarse_diagnostics=final_coarse_diagnostics[record_index],
                parallel_active_region_diagnostics=parallel_active_region_diagnostics[record_index],
                parallel_gate_threshold=parallel_gate_threshold,
            )
        if any(result is None for result in results):
            raise RuntimeError("Batch timestamp alignment did not produce one result per input record.")
        return [result for result in results if result is not None]

    def _build_final_batched_streams(
        self,
        *,
        records: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[Dict[str, Any]]:
        """Build final serialized or compact-parallel streams after per-record planning."""
        streams: List[Dict[str, Any]] = []
        for record in records:
            num_frames = int(record['ctc'].shape[0])
            if record['alignment_mode'] == 'serialized':
                streams.append(
                    self._build_batched_record_stream(
                        record=record,
                        stream_key=None,
                        tokenized_words=record['tokenized_words'],
                        blank_id=blank_id,
                        speaker_mapping=record['speaker_mapping'],
                        source_frame_indices=torch.arange(num_frames, dtype=torch.long),
                        active_region_ids=torch.zeros(num_frames, dtype=torch.long),
                    )
                )
                continue

            timelines = record.get('parallel_timelines')
            if not isinstance(timelines, Mapping):
                raise RuntimeError("A parallel batch record has no planned active-region timelines.")
            for stream_key, stream_words in self._group_words_by_speaker(record['tokenized_words']).items():
                timeline = timelines.get(stream_key)
                if not isinstance(timeline, Mapping):
                    raise RuntimeError(f"Missing parallel timeline for speaker tag {stream_key!r}.")
                streams.append(
                    self._build_batched_record_stream(
                        record=record,
                        stream_key=stream_key,
                        tokenized_words=stream_words,
                        blank_id=blank_id,
                        speaker_mapping=record['speaker_mapping'],
                        source_frame_indices=timeline['source_frame_indices'],
                        active_region_ids=timeline['region_ids'],
                        word_source_frame_bounds=record['parallel_turn_word_bounds'] or None,
                    )
                )
        return streams

    def _build_batched_record_stream(
        self,
        *,
        record: Dict[str, Any],
        stream_key: Optional[int],
        tokenized_words: Sequence[Dict[str, Any]],
        blank_id: int,
        speaker_mapping: Mapping[Optional[int], Optional[int]],
        source_frame_indices: torch.Tensor,
        active_region_ids: torch.Tensor,
        word_source_frame_bounds: Optional[Mapping[int, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Build one globally addressable CTC stream for a batch-recording DP."""
        labels, state_to_word, flat_tokens = self._build_ctc_target(tokenized_words, blank_id)
        source_frame_indices = source_frame_indices.detach().to(device='cpu', dtype=torch.long)
        active_region_ids = active_region_ids.detach().to(device='cpu', dtype=torch.long)
        if source_frame_indices.ndim != 1 or active_region_ids.ndim != 1:
            raise ValueError("Batch stream timeline tensors must be one-dimensional.")
        if source_frame_indices.numel() == 0 or source_frame_indices.shape != active_region_ids.shape:
            raise ValueError("Batch stream timeline is empty or has mismatched region IDs.")
        num_ctc_frames = int(record['ctc'].shape[0])
        invalid_source = (source_frame_indices < -1) | (source_frame_indices >= num_ctc_frames)
        invalid_region = ((source_frame_indices < 0) & (active_region_ids != -1)) | (
            (source_frame_indices >= 0) & (active_region_ids < 0)
        )
        if invalid_source.any() or invalid_region.any():
            raise ValueError("Batch stream timeline contains an invalid CTC frame or region ID.")
        if source_frame_indices[0] < 0 or source_frame_indices[-1] < 0:
            raise ValueError("Each compact CTC timeline must begin and end on an acoustic frame.")
        actual_frame_count = int((source_frame_indices >= 0).sum().item())
        minimum_frames = self._minimum_ctc_frames(flat_tokens)
        if minimum_frames > actual_frame_count:
            raise ValueError(
                "CTC target is infeasible in selected active regions for batch record "
                f"{record['record_index']}, stream {stream_key!r}: it needs at least {minimum_frames} acoustic "
                f"CTC frames but only {actual_frame_count} are available. Lower the active-region threshold, "
                "increase the region padding or merge gap, or use serialized alignment."
            )

        state_speaker_columns: List[Optional[int]] = [None] * len(labels)
        for state_index, local_word_index in enumerate(state_to_word):
            if local_word_index is not None:
                state_speaker_columns[state_index] = speaker_mapping.get(
                    tokenized_words[local_word_index]['speaker_tag']
                )
        separator_state_mask = torch.tensor(
            [
                label == blank_id
                and (
                    state_index == 0
                    or state_index == len(labels) - 1
                    or state_to_word[state_index - 1] != state_to_word[state_index + 1]
                )
                for state_index, label in enumerate(labels)
            ],
            dtype=torch.bool,
        )
        state_min_source_frames, state_max_source_frames = self._state_source_frame_bounds(
            tokenized_words=tokenized_words,
            state_to_word=state_to_word,
            word_source_frame_bounds=word_source_frame_bounds,
            local_ctc_num_frames=num_ctc_frames,
            source_frame_offset=int(record['ctc_offset']),
        )
        global_source_frame_indices = source_frame_indices.clone()
        acoustic_frames = global_source_frame_indices >= 0
        global_source_frame_indices[acoustic_frames] += int(record['ctc_offset'])
        return {
            'record': record,
            'stream_key': stream_key,
            'tokenized_words': list(tokenized_words),
            'labels': labels,
            'state_to_word': state_to_word,
            'state_speaker_columns': state_speaker_columns,
            'state_min_source_frames': state_min_source_frames,
            'state_max_source_frames': state_max_source_frames,
            'source_frame_indices': source_frame_indices,
            'global_source_frame_indices': global_source_frame_indices,
            'active_region_ids': active_region_ids,
            'separator_state_mask': separator_state_mask,
        }

    def _align_record_streams_batched(
        self,
        *,
        streams: Sequence[Dict[str, Any]],
        ctc_log_probs: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        blank_id: int,
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float],
        use_coarse_alignment: bool,
    ) -> List[Dict[str, Any]]:
        """Pack independent record streams and run one padded CTC Viterbi call."""
        if not streams:
            return []
        num_streams = len(streams)
        max_states = max(len(stream['labels']) for stream in streams)
        max_time = max(int(stream['global_source_frame_indices'].numel()) for stream in streams)
        state_lengths = torch.tensor([len(stream['labels']) for stream in streams], dtype=torch.long)
        time_lengths = torch.tensor(
            [int(stream['global_source_frame_indices'].numel()) for stream in streams], dtype=torch.long
        )
        labels_batch = torch.full((num_streams, max_states), blank_id, dtype=torch.long)
        columns_batch = torch.full((num_streams, max_states), -1, dtype=torch.long)
        state_min_frames_batch = torch.zeros((num_streams, max_states), dtype=torch.long)
        state_max_frames_batch = torch.full(
            (num_streams, max_states), ctc_log_probs.shape[0] - 1, dtype=torch.long
        )
        source_frames_batch = torch.full((num_streams, max_time), -1, dtype=torch.long)
        separator_states_batch = torch.zeros((num_streams, max_states), dtype=torch.bool)
        for stream_index, stream in enumerate(streams):
            labels = stream['labels']
            columns = stream['state_speaker_columns']
            source_frame_indices = stream['global_source_frame_indices']
            labels_batch[stream_index, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            columns_batch[stream_index, : len(columns)] = torch.tensor(
                [-1 if column is None else int(column) for column in columns], dtype=torch.long
            )
            state_min_frames_batch[stream_index, : len(labels)] = torch.tensor(
                stream['state_min_source_frames'], dtype=torch.long
            )
            state_max_frames_batch[stream_index, : len(labels)] = torch.tensor(
                stream['state_max_source_frames'], dtype=torch.long
            )
            source_frames_batch[stream_index, : source_frame_indices.numel()] = source_frame_indices
            separator_states_batch[stream_index, : len(labels)] = stream['separator_state_mask']

        paths, scores, diagnostics = self._ctc_viterbi_align_batched(
            ctc_log_probs=ctc_log_probs,
            labels=labels_batch,
            state_lengths=state_lengths,
            blank_id=blank_id,
            state_speaker_columns=columns_batch,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
            source_frame_indices=source_frames_batch,
            time_lengths=time_lengths,
            separator_state_mask=separator_states_batch,
            state_min_source_frames=state_min_frames_batch,
            state_max_source_frames=state_max_frames_batch,
            use_coarse_alignment=use_coarse_alignment,
        )
        return [
            {'stream': stream, 'path': path, 'score': score, 'diagnostic': diagnostic}
            for stream, path, score, diagnostic in zip(streams, paths, scores, diagnostics)
        ]

    def _build_batched_alignment_result(
        self,
        *,
        record: Mapping[str, Any],
        alignment_mode: str,
        speaker_assignment_mode: str,
        speaker_mapping: Dict[int, Optional[int]],
        rows: Sequence[Dict[str, Any]],
        preliminary_scores: Mapping[Optional[int], float],
        final_scores: Mapping[Optional[int], float],
        assignment_scores: Mapping[int, List[float]],
        preliminary_coarse_diagnostics: Mapping[Optional[int], Dict[str, Any]],
        final_coarse_diagnostics: Mapping[Optional[int], Dict[str, Any]],
        parallel_active_region_diagnostics: Mapping[Optional[int], Dict[str, Any]],
        parallel_gate_threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Format one record from a globally batched alignment run."""
        speaker_word_timestamps: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for row in rows:
            speaker_word_timestamps.setdefault(row['speaker_tag'], []).append(row)
        return {
            'speaker_word_timestamps': speaker_word_timestamps,
            'speaker_tag_to_sortformer_column': speaker_mapping,
            'alignment_mode': alignment_mode,
            'requested_alignment_mode': record.get('requested_alignment_mode', alignment_mode),
            'speaker_assignment_mode': speaker_assignment_mode,
            'ctc_frame_seconds': record['ctc_step_seconds'],
            'sortformer_frame_seconds': record['sortformer_step_seconds'],
            'time_offset': record['time_offset'],
            'num_ctc_frames': int(record['ctc'].shape[0]),
            'num_sortformer_frames': record['sortformer_length'],
            'ctc_log_normalizer_error': record['ctc_log_normalizer_error'],
            'alignment_diagnostics': {
                'preliminary_ctc_path_scores': dict(preliminary_scores),
                'final_path_scores': dict(final_scores),
                'speaker_assignment_scores': dict(assignment_scores),
                'speaker_count_policy': dict(record.get('speaker_count_policy', {})),
                'parallel_speaker_gate_threshold': parallel_gate_threshold if alignment_mode == 'parallel' else None,
                'parallel_speaker_gate_min_threshold': self.parallel_speaker_gate_min_threshold,
                'parallel_active_regions': (
                    dict(parallel_active_region_diagnostics) if alignment_mode == 'parallel' else {}
                ),
                'alignment_fallback': record.get('alignment_fallback'),
                'parallel_active_region_padding_seconds': self.parallel_active_region_padding_seconds,
                'parallel_active_region_merge_gap_seconds': self.parallel_active_region_merge_gap_seconds,
                'parallel_turn_fences': (
                    dict(record.get('parallel_turn_diagnostics', {})) if alignment_mode == 'parallel' else {}
                ),
                'parallel_turn_anchor_path_score': record.get('serialized_turn_anchor_score'),
                'parallel_turn_anchor_coarse_alignment': record.get('serialized_turn_anchor_coarse_diagnostic'),
                'coarse_alignment_band_size': self.coarse_alignment_band_size,
                'preliminary_forced_dense_for_speaker_mapping': self.coarse_alignment_band_size is not None,
                'coarse_alignment': {
                    'preliminary': dict(preliminary_coarse_diagnostics),
                    'final': dict(final_coarse_diagnostics),
                },
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

    def _resolve_sot_sortformer_count_policy(
        self,
        *,
        requested_alignment_mode: str,
        speaker_assignment_mode: str,
        speaker_tags: Sequence[int],
        speaker_probs: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Treat t-SOT speakers as authoritative and Sortformer as auxiliary.

        A Sortformer stream counts as active when it has at least one frame at
        ``speaker_activity_threshold``. If t-SOT has more speakers than those
        active streams, a requested parallel alignment is downgraded to pure
        serialized CTC. Otherwise, optimal assignment is restricted to the most
        active raw Sortformer columns, leaving extra streams unused.
        """
        tags = list(speaker_tags)
        if len(tags) != len(set(tags)):
            raise ValueError('speaker_tags must contain distinct t-SOT speaker tags.')

        policy: Dict[str, Any] = {
            'requested_alignment_mode': requested_alignment_mode,
            'effective_alignment_mode': requested_alignment_mode,
            'sot_speaker_count': len(tags),
            'sot_speaker_tags': tags,
            'sortformer_column_count': None,
            'sortformer_active_column_count': None,
            'sortformer_active_columns': [],
            'sortformer_column_activity_mass': [],
            'selected_sortformer_columns': [],
            'ignored_sortformer_columns': [],
            'use_sortformer_for_alignment': speaker_probs is not None,
            'reason': 'no_sortformer_output',
        }
        if speaker_probs is None:
            return policy
        if speaker_probs.ndim != 2 or speaker_probs.shape[1] == 0:
            raise ValueError('speaker_probs must have shape (frames, non-empty speakers).')

        probs = speaker_probs.detach().to(device='cpu', dtype=torch.float32)
        num_columns = int(probs.shape[1])
        activity_mass = probs.sum(dim=0)
        activity_values = [float(value) for value in activity_mass.tolist()]
        active_columns = torch.nonzero(
            (probs >= self.speaker_activity_threshold).any(dim=0), as_tuple=False
        ).flatten().tolist()
        policy['sortformer_column_count'] = num_columns
        policy['sortformer_active_column_count'] = len(active_columns)
        policy['sortformer_active_columns'] = active_columns
        policy['sortformer_column_activity_mass'] = activity_values

        if not tags:
            policy['reason'] = 'no_explicit_sot_speaker_tags'
            return policy

        if len(tags) > len(active_columns):
            # There cannot be a reliable one-to-one t-SOT-to-Sortformer mapping.
            # Keep the complete transcript and rely only on its serialized CTC path.
            policy.update(
                {
                    'effective_alignment_mode': 'serialized',
                    'selected_sortformer_columns': [],
                    'ignored_sortformer_columns': list(range(num_columns)),
                    'use_sortformer_for_alignment': False,
                    'reason': 'sot_speakers_exceed_active_sortformer_columns',
                }
            )
            return policy

        if speaker_assignment_mode == 'identity':
            # Identity is an explicit user override: tag N remains column N, but
            # every unreferenced column is still ignored.
            selected_columns = [tag for tag in tags if 0 <= tag < num_columns]
            policy['reason'] = 'identity_assignment_preserves_tag_columns'
        else:
            ranked_columns = sorted(active_columns, key=lambda column: (-activity_values[column], column))
            selected_columns = ranked_columns[: len(tags)]
            policy['reason'] = (
                'sot_and_active_sortformer_speaker_counts_match'
                if len(tags) == len(active_columns)
                else 'sot_speakers_fewer_than_active_sortformer_columns'
            )

        selected_set = set(selected_columns)
        policy['selected_sortformer_columns'] = selected_columns
        policy['ignored_sortformer_columns'] = [
            column for column in range(num_columns) if column not in selected_set
        ]
        return policy


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
    def _has_repeated_speaker_turns(tokenized_words: Sequence[Dict[str, Any]]) -> bool:
        """Return whether any explicit t-SOT speaker has more than one turn."""
        turns_by_speaker: Dict[Optional[int], set[int]] = {}
        for word in tokenized_words:
            turn_index = word.get('turn_index')
            if turn_index is None:
                continue
            if isinstance(turn_index, bool) or not isinstance(turn_index, int):
                raise TypeError("turn_index must be an integer or None.")
            turns = turns_by_speaker.setdefault(word['speaker_tag'], set())
            turns.add(turn_index)
            if len(turns) > 1:
                return True
        return False

    @staticmethod
    def _build_parallel_turn_frame_bounds(
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        serialized_anchor_rows: Sequence[Dict[str, Any]],
        ctc_num_frames: int,
    ) -> Tuple[Dict[int, Tuple[int, int]], Dict[Optional[int], List[Dict[str, Any]]]]:
        """Build same-speaker t-SOT turn fences from a serialized CTC anchor.

        The serialized anchor preserves global t-SOT order. For each speaker, a
        midpoint between consecutive occurrences of that speaker becomes a hard
        source-frame fence for the token states in the adjacent turns. Different
        speakers intentionally retain independent, potentially overlapping windows.
        """
        if ctc_num_frames <= 0:
            raise ValueError("ctc_num_frames must be positive.")

        anchor_by_word_index: Dict[int, Dict[str, Any]] = {}
        for row in serialized_anchor_rows:
            try:
                word_index = int(row['word_index'])
                start_frame = int(row['start_frame'])
                end_frame = int(row['end_frame'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Serialized turn-anchor rows must carry integer word and frame indices.") from error
            if not 0 <= start_frame <= end_frame < ctc_num_frames:
                raise ValueError("Serialized turn-anchor row has a frame outside the CTC timeline.")
            if word_index in anchor_by_word_index:
                raise ValueError(f"Serialized turn-anchor has duplicate word_index {word_index}.")
            anchor_by_word_index[word_index] = row

        turns_by_speaker: Dict[Optional[int], List[Dict[str, Any]]] = {}
        seen_word_indices: set[int] = set()
        for word in tokenized_words:
            try:
                word_index = int(word['word_index'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Tokenized words must carry integer word_index values.") from error
            if word_index in seen_word_indices:
                raise ValueError(f"Tokenized words have duplicate word_index {word_index}.")
            seen_word_indices.add(word_index)
            if word_index not in anchor_by_word_index:
                raise ValueError(f"Serialized turn-anchor omitted word_index {word_index}.")

            turn_index = word.get('turn_index')
            if turn_index is not None and (isinstance(turn_index, bool) or not isinstance(turn_index, int)):
                raise TypeError("turn_index must be an integer or None.")
            speaker_turns = turns_by_speaker.setdefault(word['speaker_tag'], [])
            if not speaker_turns or speaker_turns[-1]['turn_index'] != turn_index:
                speaker_turns.append({'turn_index': turn_index, 'words': []})
            speaker_turns[-1]['words'].append(word)

        if len(anchor_by_word_index) != len(seen_word_indices):
            unexpected = sorted(set(anchor_by_word_index).difference(seen_word_indices))
            raise ValueError(f"Serialized turn-anchor has unknown word_index values: {unexpected}.")

        bounds_by_word_index: Dict[int, Tuple[int, int]] = {}
        diagnostics: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for speaker_tag, turns in turns_by_speaker.items():
            if len(turns) <= 1:
                continue
            previous_anchor_start = -1
            for turn in turns:
                anchor_rows = [anchor_by_word_index[int(word['word_index'])] for word in turn['words']]
                anchor_start = min(int(row['start_frame']) for row in anchor_rows)
                anchor_end = max(int(row['end_frame']) for row in anchor_rows)
                if anchor_start < previous_anchor_start:
                    raise ValueError(
                        "Serialized turn-anchor is not monotonic within a speaker; cannot create turn fences."
                    )
                turn['anchor_start_frame'] = anchor_start
                turn['anchor_end_frame'] = anchor_end
                previous_anchor_start = anchor_start

            lower = 0
            speaker_diagnostics: List[Dict[str, Any]] = []
            for turn_index, turn in enumerate(turns):
                if turn_index + 1 == len(turns):
                    upper = ctc_num_frames - 1
                else:
                    next_turn = turns[turn_index + 1]
                    upper = (int(turn['anchor_end_frame']) + int(next_turn['anchor_start_frame'])) // 2
                    upper = min(ctc_num_frames - 1, max(lower, upper))
                if upper < lower:
                    raise RuntimeError("Parallel turn fence construction produced an empty CTC interval.")
                bounds = (lower, upper)
                for word in turn['words']:
                    bounds_by_word_index[int(word['word_index'])] = bounds
                speaker_diagnostics.append(
                    {
                        'turn_index': turn['turn_index'],
                        'first_word_index': int(turn['words'][0]['word_index']),
                        'last_word_index': int(turn['words'][-1]['word_index']),
                        'anchor_start_frame': int(turn['anchor_start_frame']),
                        'anchor_end_frame': int(turn['anchor_end_frame']),
                        'min_source_frame': lower,
                        'max_source_frame': upper,
                    }
                )
                lower = upper + 1
            diagnostics[speaker_tag] = speaker_diagnostics
        return bounds_by_word_index, diagnostics

    @staticmethod
    def _state_source_frame_bounds(
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        state_to_word: Sequence[Optional[int]],
        word_source_frame_bounds: Optional[Mapping[int, Tuple[int, int]]],
        local_ctc_num_frames: int,
        source_frame_offset: int = 0,
    ) -> Tuple[List[int], List[int]]:
        """Return inclusive source-frame bounds for token states in one target."""
        if local_ctc_num_frames <= 0:
            raise ValueError("local_ctc_num_frames must be positive.")
        default_min = int(source_frame_offset)
        default_max = default_min + int(local_ctc_num_frames) - 1
        state_min_frames = [default_min] * len(state_to_word)
        state_max_frames = [default_max] * len(state_to_word)
        if word_source_frame_bounds is None:
            return state_min_frames, state_max_frames

        for state_index, local_word_index in enumerate(state_to_word):
            if local_word_index is None:
                continue
            word = tokenized_words[local_word_index]
            word_index = int(word['word_index'])
            bounds = word_source_frame_bounds.get(word_index)
            if bounds is None:
                continue
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError("word_source_frame_bounds values must be (min_frame, max_frame) tuples.")
            lower, upper = int(bounds[0]), int(bounds[1])
            if not 0 <= lower <= upper < local_ctc_num_frames:
                raise ValueError(
                    f"Turn frame bounds [{lower}, {upper}] for word_index {word_index} are outside "
                    f"the local CTC grid [0, {local_ctc_num_frames - 1}]."
                )
            state_min_frames[state_index] = lower + int(source_frame_offset)
            state_max_frames[state_index] = upper + int(source_frame_offset)
        return state_min_frames, state_max_frames

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

    def _parallel_gate_threshold_schedule(self, initial_threshold: Optional[float]) -> List[Optional[float]]:
        """Return the bounded automatic threshold schedule for one speaker stream."""
        if initial_threshold is None:
            # ``None`` is an explicit request for the unrestricted timeline, so
            # there is no active-region threshold to relax.
            return [None]
        initial = float(initial_threshold)
        floor = min(initial, self.parallel_speaker_gate_min_threshold)
        schedule: List[Optional[float]] = []
        for candidate in (initial, 0.40, 0.30, 0.25, floor):
            candidate = float(candidate)
            if floor <= candidate <= initial and not any(
                existing is not None and math.isclose(float(existing), candidate, abs_tol=1.0e-8)
                for existing in schedule
            ):
                schedule.append(candidate)
        return schedule

    def _new_parallel_gate_retry_state(
        self,
        tokenized_words: Sequence[Dict[str, Any]],
        initial_threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Initialize independent adaptive threshold schedules for each t-SOT speaker."""
        schedules = {
            speaker_tag: self._parallel_gate_threshold_schedule(initial_threshold)
            for speaker_tag in self._group_words_by_speaker(tokenized_words)
        }
        return {
            'schedules': schedules,
            'positions': {speaker_tag: 0 for speaker_tag in schedules},
            'attempted_thresholds': {
                speaker_tag: [schedule[0]] for speaker_tag, schedule in schedules.items()
            },
            'gate_floor': self.parallel_speaker_gate_min_threshold,
        }

    @staticmethod
    def _parallel_gate_thresholds_for_retry_state(
        retry_state: Mapping[str, Any],
    ) -> Dict[Optional[int], Optional[float]]:
        schedules = retry_state['schedules']
        positions = retry_state['positions']
        return {
            speaker_tag: schedules[speaker_tag][positions[speaker_tag]]
            for speaker_tag in schedules
        }

    @staticmethod
    def _advance_parallel_gate_retry_state(
        retry_state: Dict[str, Any],
        speaker_tags: Sequence[Optional[int]],
    ) -> List[Optional[int]]:
        """Relax each requested stream once and return streams already at their floor."""
        schedules = retry_state['schedules']
        positions = retry_state['positions']
        attempted_thresholds = retry_state['attempted_thresholds']
        exhausted: List[Optional[int]] = []
        for speaker_tag in dict.fromkeys(speaker_tags):
            schedule = schedules.get(speaker_tag)
            if not schedule:
                exhausted.append(speaker_tag)
                continue
            position = int(positions[speaker_tag])
            if position + 1 >= len(schedule):
                exhausted.append(speaker_tag)
                continue
            position += 1
            positions[speaker_tag] = position
            attempted_thresholds[speaker_tag].append(schedule[position])
        return exhausted

    def _parallel_timeline_capacity_failures(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        blank_id: int,
        timelines: Mapping[Optional[int], Mapping[str, torch.Tensor]],
    ) -> Dict[Optional[int], Dict[str, int]]:
        """Find target streams with fewer selected acoustic frames than CTC requires."""
        failures: Dict[Optional[int], Dict[str, int]] = {}
        for speaker_tag, speaker_words in self._group_words_by_speaker(tokenized_words).items():
            timeline = timelines.get(speaker_tag)
            if timeline is None:
                raise RuntimeError(f"Missing active-region timeline for speaker tag {speaker_tag!r}.")
            source_frame_indices = timeline.get('source_frame_indices')
            if not isinstance(source_frame_indices, torch.Tensor):
                raise TypeError("Parallel active-region timeline is missing source_frame_indices.")
            _, _, flat_tokens = self._build_ctc_target(speaker_words, blank_id)
            minimum_frames = self._minimum_ctc_frames(flat_tokens)
            available_frames = int((source_frame_indices >= 0).sum().item())
            if minimum_frames > available_frames:
                failures[speaker_tag] = {
                    'minimum_ctc_frames': minimum_frames,
                    'available_ctc_frames': available_frames,
                }
        return failures

    def _annotate_parallel_gate_retry_diagnostics(
        self,
        diagnostics: Dict[Optional[int], Dict[str, Any]],
        retry_state: Mapping[str, Any],
    ) -> None:
        """Attach the per-speaker retry history to final active-region diagnostics."""
        thresholds = self._parallel_gate_thresholds_for_retry_state(retry_state)
        for speaker_tag, diagnostic in diagnostics.items():
            diagnostic['adaptive_gate_retry'] = {
                'attempted_thresholds': list(retry_state['attempted_thresholds'].get(speaker_tag, [])),
                'selected_threshold': thresholds.get(speaker_tag),
                'gate_floor': retry_state['gate_floor'],
            }

    def _build_parallel_active_timelines_with_retries(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        speaker_mapping: Dict[int, Optional[int]],
        speaker_probs: Optional[torch.Tensor],
        ctc_num_frames: int,
        ctc_step_seconds: float,
        blank_id: int,
        retry_state: Dict[str, Any],
    ) -> Tuple[
        Optional[Dict[Optional[int], Dict[str, torch.Tensor]]],
        Dict[Optional[int], Dict[str, Any]],
        Optional[Dict[str, Any]],
    ]:
        """Plan compact timelines, relaxing only streams that cannot hold their CTC target."""
        last_diagnostics: Dict[Optional[int], Dict[str, Any]] = {}
        while True:
            try:
                timelines, diagnostics = self._build_parallel_active_timelines(
                    tokenized_words=tokenized_words,
                    speaker_mapping=speaker_mapping,
                    speaker_probs=speaker_probs,
                    ctc_num_frames=ctc_num_frames,
                    ctc_step_seconds=ctc_step_seconds,
                    active_threshold=self._parallel_gate_thresholds_for_retry_state(retry_state),
                )
            except _ParallelActiveRegionError as error:
                exhausted = self._advance_parallel_gate_retry_state(retry_state, error.failed_speaker_tags)
                if exhausted:
                    return None, last_diagnostics, {
                        'reason': error.reason,
                        'failed_speaker_tags': list(error.failed_speaker_tags),
                    }
                continue

            last_diagnostics = diagnostics
            capacity_failures = self._parallel_timeline_capacity_failures(
                tokenized_words=tokenized_words,
                blank_id=blank_id,
                timelines=timelines,
            )
            if capacity_failures:
                failed_speaker_tags = list(capacity_failures)
                exhausted = self._advance_parallel_gate_retry_state(retry_state, failed_speaker_tags)
                if exhausted:
                    return None, diagnostics, {
                        'reason': 'insufficient_ctc_frames_in_active_regions',
                        'failed_speaker_tags': failed_speaker_tags,
                        'capacity_failures': capacity_failures,
                    }
                continue

            self._annotate_parallel_gate_retry_diagnostics(diagnostics, retry_state)
            return timelines, diagnostics, None

    def _parallel_serialized_fallback_diagnostic(
        self,
        *,
        retry_state: Mapping[str, Any],
        reason: str,
        failed_speaker_tags: Sequence[Optional[int]],
        capacity_failures: Optional[Mapping[Optional[int], Mapping[str, int]]] = None,
    ) -> Dict[str, Any]:
        """Describe a terminal parallel failure without silently opening a full speaker timeline."""
        diagnostic: Dict[str, Any] = {
            'from_alignment_mode': 'parallel',
            'to_alignment_mode': 'serialized',
            'reason': reason,
            'failed_speaker_tags': list(failed_speaker_tags),
            'attempted_gate_thresholds': {
                str(speaker_tag): list(thresholds)
                for speaker_tag, thresholds in retry_state['attempted_thresholds'].items()
            },
            'gate_floor': retry_state['gate_floor'],
        }
        if capacity_failures:
            diagnostic['capacity_failures'] = {
                str(speaker_tag): dict(details) for speaker_tag, details in capacity_failures.items()
            }
        return diagnostic

    def _switch_batched_record_to_serialized_fallback(
        self,
        *,
        record: Dict[str, Any],
        blank_id: int,
        retry_state: Mapping[str, Any],
        terminal_failure: Mapping[str, Any],
    ) -> None:
        """Make a single batch record use authoritative serialized t-SOT CTC."""
        tokenized_words = self._tokenize_words(
            record['parsed_sot_words'],
            blank_id,
            alignment_mode='serialized',
        )
        record['tokenized_words'] = tokenized_words
        record['speaker_tags'] = self._speaker_tags_in_order(tokenized_words)
        record['speaker_mapping'] = {
            speaker_tag: None for speaker_tag in record['speaker_tags']
        }
        record['assignment_scores'] = {}
        record['alignment_mode'] = 'serialized'
        record['parallel_turn_word_bounds'] = {}
        record['parallel_turn_diagnostics'] = {}
        record['parallel_timelines'] = None
        record['alignment_fallback'] = self._parallel_serialized_fallback_diagnostic(
            retry_state=retry_state,
            reason=str(terminal_failure['reason']),
            failed_speaker_tags=terminal_failure.get('failed_speaker_tags', []),
            capacity_failures=terminal_failure.get('capacity_failures'),
        )

    def _build_parallel_active_timelines(
        self,
        *,
        tokenized_words: Sequence[Dict[str, Any]],
        speaker_mapping: Dict[int, Optional[int]],
        speaker_probs: Optional[torch.Tensor],
        ctc_num_frames: int,
        ctc_step_seconds: float,
        active_threshold: Union[Optional[float], Mapping[Optional[int], Optional[float]]],
    ) -> Tuple[Dict[Optional[int], Dict[str, torch.Tensor]], Dict[Optional[int], Dict[str, Any]]]:
        """Build compact CTC timelines for independently aligned speaker streams.

        A timeline contains original CTC frame indices and ``-1`` virtual frames.
        A virtual frame is inserted only between disjoint active regions and can
        emit a blank state at a complete-word boundary. This prevents a word from
        spanning a long Sortformer-inactive gap while preserving one CTC DP per
        speaker transcript.
        """
        if ctc_num_frames <= 0:
            raise ValueError("ctc_num_frames must be positive.")
        full_source_frames = torch.arange(ctc_num_frames, dtype=torch.long)
        full_region_ids = torch.zeros(ctc_num_frames, dtype=torch.long)
        timelines: Dict[Optional[int], Dict[str, torch.Tensor]] = {}
        diagnostics: Dict[Optional[int], Dict[str, Any]] = {}
        padding_frames = int(round(self.parallel_active_region_padding_seconds / ctc_step_seconds))
        merge_gap_frames = int(round(self.parallel_active_region_merge_gap_seconds / ctc_step_seconds))

        for speaker_tag in self._group_words_by_speaker(tokenized_words):
            column = speaker_mapping.get(speaker_tag)
            speaker_threshold = (
                active_threshold.get(speaker_tag)
                if isinstance(active_threshold, Mapping)
                else active_threshold
            )
            constrained = (
                speaker_tag is not None
                and speaker_probs is not None
                and column is not None
                and speaker_threshold is not None
            )
            if not constrained:
                timelines[speaker_tag] = {
                    'source_frame_indices': full_source_frames.clone(),
                    'region_ids': full_region_ids.clone(),
                }
                diagnostics[speaker_tag] = {
                    'constrained_to_active_regions': False,
                    'sortformer_column': column,
                    'selected_ctc_frames': ctc_num_frames,
                    'virtual_separator_count': 0,
                }
                continue

            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != ctc_num_frames:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
            column = int(column)
            if not 0 <= column < speaker_probs.shape[1]:
                raise ValueError(
                    f"Speaker tag {speaker_tag!r} maps to unavailable Sortformer column {column}."
                )
            activity = speaker_probs[:, column]
            active_frames = torch.nonzero(activity >= float(speaker_threshold), as_tuple=False).flatten().tolist()
            if not active_frames:
                raise _ParallelActiveRegionError(
                    [speaker_tag],
                    reason='no_active_ctc_frames',
                    details={
                        speaker_tag: {
                            'sortformer_column': column,
                            'active_threshold': float(speaker_threshold),
                        }
                    },
                )

            raw_regions: List[Tuple[int, int]] = []
            start = previous = int(active_frames[0])
            for frame in active_frames[1:]:
                frame = int(frame)
                if frame == previous + 1:
                    previous = frame
                    continue
                raw_regions.append((start, previous))
                start = previous = frame
            raw_regions.append((start, previous))

            padded_regions = [
                (max(0, start - padding_frames), min(ctc_num_frames - 1, end + padding_frames))
                for start, end in raw_regions
            ]
            merged_regions: List[Tuple[int, int]] = []
            for start, end in padded_regions:
                if merged_regions and start <= merged_regions[-1][1] + 1 + merge_gap_frames:
                    merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
                else:
                    merged_regions.append((start, end))

            source_frames: List[int] = []
            region_ids: List[int] = []
            for region_id, (start, end) in enumerate(merged_regions):
                if source_frames:
                    source_frames.append(-1)
                    region_ids.append(-1)
                source_frames.extend(range(start, end + 1))
                region_ids.extend([region_id] * (end - start + 1))
            timelines[speaker_tag] = {
                'source_frame_indices': torch.tensor(source_frames, dtype=torch.long),
                'region_ids': torch.tensor(region_ids, dtype=torch.long),
            }
            diagnostics[speaker_tag] = {
                'constrained_to_active_regions': True,
                'sortformer_column': column,
                'active_threshold': float(speaker_threshold),
                'padding_frames': padding_frames,
                'merge_gap_frames': merge_gap_frames,
                'raw_active_regions': [
                    {'start_frame': start, 'end_frame': end} for start, end in raw_regions
                ],
                'selected_active_regions': [
                    {'start_frame': start, 'end_frame': end} for start, end in merged_regions
                ],
                'selected_ctc_frames': len(source_frames) - max(0, len(merged_regions) - 1),
                'virtual_separator_count': max(0, len(merged_regions) - 1),
            }
        return timelines, diagnostics

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
        use_coarse_alignment: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
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

        path, path_score, coarse_diagnostic = self._ctc_viterbi_align(
            ctc_log_probs=ctc_log_probs,
            labels=labels,
            blank_id=blank_id,
            state_speaker_columns=state_speaker_columns,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
            use_coarse_alignment=use_coarse_alignment,
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
        return rows, path_score, coarse_diagnostic

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
        speaker_timelines: Optional[Mapping[Optional[int], Mapping[str, torch.Tensor]]] = None,
        word_source_frame_bounds: Optional[Mapping[int, Tuple[int, int]]] = None,
        use_coarse_alignment: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[Optional[int], float], Dict[Optional[int], Dict[str, Any]]]:
        """Align independent speaker streams in one padded CTC Viterbi batch.

        Each stream can provide a compact timeline containing original CTC frames
        plus virtual separators between disjoint Sortformer-active regions. The
        DP evaluates all streams together without materializing a
        ``(speakers, frames, vocabulary)`` tensor.
        """
        grouped_words = self._group_words_by_speaker(tokenized_words)
        full_source_frames = torch.arange(ctc_log_probs.shape[0], dtype=torch.long)
        full_region_ids = torch.zeros(ctc_log_probs.shape[0], dtype=torch.long)
        streams: List[Dict[str, Any]] = []
        for speaker_tag, speaker_words in grouped_words.items():
            speaker_words = list(speaker_words)
            labels, state_to_word, flat_tokens = self._build_ctc_target(speaker_words, blank_id)
            timeline = None if speaker_timelines is None else speaker_timelines.get(speaker_tag)
            if timeline is None:
                source_frame_indices = full_source_frames.clone()
                active_region_ids = full_region_ids.clone()
            else:
                source_frame_indices = timeline.get('source_frame_indices')
                active_region_ids = timeline.get('region_ids')
                if not isinstance(source_frame_indices, torch.Tensor) or not isinstance(active_region_ids, torch.Tensor):
                    raise TypeError(
                        "speaker_timelines entries must contain tensor-valued "
                        "'source_frame_indices' and 'region_ids'."
                    )
                source_frame_indices = source_frame_indices.detach().to(device='cpu', dtype=torch.long)
                active_region_ids = active_region_ids.detach().to(device='cpu', dtype=torch.long)
            if source_frame_indices.ndim != 1 or active_region_ids.ndim != 1:
                raise ValueError("speaker timeline tensors must be one-dimensional.")
            if source_frame_indices.numel() == 0 or source_frame_indices.shape != active_region_ids.shape:
                raise ValueError(f"Speaker timeline for {speaker_tag!r} is empty or has mismatched region IDs.")
            invalid_source = (source_frame_indices < -1) | (source_frame_indices >= ctc_log_probs.shape[0])
            invalid_region = ((source_frame_indices < 0) & (active_region_ids != -1)) | (
                (source_frame_indices >= 0) & (active_region_ids < 0)
            )
            if invalid_source.any() or invalid_region.any():
                raise ValueError(f"Speaker timeline for {speaker_tag!r} contains invalid CTC frame or region IDs.")
            if source_frame_indices[0] < 0 or source_frame_indices[-1] < 0:
                raise ValueError(f"Speaker timeline for {speaker_tag!r} must begin and end on an acoustic CTC frame.")
            actual_frame_count = int((source_frame_indices >= 0).sum().item())
            minimum_frames = self._minimum_ctc_frames(flat_tokens)
            if minimum_frames > actual_frame_count:
                raise ValueError(
                    "CTC target is infeasible in selected active regions for speaker "
                    f"{speaker_tag!r}: it needs at least {minimum_frames} acoustic CTC frames but only "
                    f"{actual_frame_count} are available. Lower the active-region threshold, increase "
                    "the region padding or merge gap, or use serialized alignment."
                )

            state_speaker_columns: List[Optional[int]] = [None] * len(labels)
            for state_index, local_word_index in enumerate(state_to_word):
                if local_word_index is not None:
                    state_speaker_columns[state_index] = speaker_mapping.get(
                        speaker_words[local_word_index]['speaker_tag']
                    )
            separator_state_mask = torch.tensor(
                [
                    label == blank_id
                    and (
                        state_index == 0
                        or state_index == len(labels) - 1
                        or state_to_word[state_index - 1] != state_to_word[state_index + 1]
                    )
                    for state_index, label in enumerate(labels)
                ],
                dtype=torch.bool,
            )
            state_min_source_frames, state_max_source_frames = self._state_source_frame_bounds(
                tokenized_words=speaker_words,
                state_to_word=state_to_word,
                word_source_frame_bounds=word_source_frame_bounds,
                local_ctc_num_frames=ctc_log_probs.shape[0],
            )
            streams.append(
                {
                    'speaker_tag': speaker_tag,
                    'speaker_words': speaker_words,
                    'labels': labels,
                    'state_to_word': state_to_word,
                    'state_speaker_columns': state_speaker_columns,
                    'state_min_source_frames': state_min_source_frames,
                    'state_max_source_frames': state_max_source_frames,
                    'source_frame_indices': source_frame_indices,
                    'active_region_ids': active_region_ids,
                    'separator_state_mask': separator_state_mask,
                }
            )

        if not streams:
            return [], {}, {}

        num_streams = len(streams)
        max_states = max(len(stream['labels']) for stream in streams)
        max_time = max(int(stream['source_frame_indices'].numel()) for stream in streams)
        state_lengths = torch.tensor([len(stream['labels']) for stream in streams], dtype=torch.long)
        time_lengths = torch.tensor(
            [int(stream['source_frame_indices'].numel()) for stream in streams], dtype=torch.long
        )
        labels_batch = torch.full((num_streams, max_states), blank_id, dtype=torch.long)
        columns_batch = torch.full((num_streams, max_states), -1, dtype=torch.long)
        state_min_frames_batch = torch.zeros((num_streams, max_states), dtype=torch.long)
        state_max_frames_batch = torch.full(
            (num_streams, max_states), ctc_log_probs.shape[0] - 1, dtype=torch.long
        )
        source_frames_batch = torch.full((num_streams, max_time), -1, dtype=torch.long)
        separator_states_batch = torch.zeros((num_streams, max_states), dtype=torch.bool)
        for stream_index, stream in enumerate(streams):
            labels = stream['labels']
            columns = stream['state_speaker_columns']
            source_frame_indices = stream['source_frame_indices']
            labels_batch[stream_index, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            columns_batch[stream_index, : len(columns)] = torch.tensor(
                [-1 if column is None else int(column) for column in columns], dtype=torch.long
            )
            state_min_frames_batch[stream_index, : len(labels)] = torch.tensor(
                stream['state_min_source_frames'], dtype=torch.long
            )
            state_max_frames_batch[stream_index, : len(labels)] = torch.tensor(
                stream['state_max_source_frames'], dtype=torch.long
            )
            source_frames_batch[stream_index, : source_frame_indices.numel()] = source_frame_indices
            separator_states_batch[stream_index, : len(labels)] = stream['separator_state_mask']

        paths, scores, coarse_diagnostics = self._ctc_viterbi_align_batched(
            ctc_log_probs=ctc_log_probs,
            labels=labels_batch,
            state_lengths=state_lengths,
            blank_id=blank_id,
            state_speaker_columns=columns_batch,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
            source_frame_indices=source_frames_batch,
            time_lengths=time_lengths,
            separator_state_mask=separator_states_batch,
            state_min_source_frames=state_min_frames_batch,
            state_max_source_frames=state_max_frames_batch,
            use_coarse_alignment=use_coarse_alignment,
        )

        rows: List[Dict[str, Any]] = []
        score_by_speaker: Dict[Optional[int], float] = {}
        coarse_diagnostics_by_speaker: Dict[Optional[int], Dict[str, Any]] = {}
        for stream, path, score, coarse_diagnostic in zip(streams, paths, scores, coarse_diagnostics):
            rows.extend(
                self._word_rows_from_path(
                    tokenized_words=stream['speaker_words'],
                    labels=stream['labels'],
                    state_to_word=stream['state_to_word'],
                    path=path,
                    ctc_log_probs=ctc_log_probs,
                    speaker_probs=speaker_probs,
                    speaker_mapping=speaker_mapping,
                    ctc_step_seconds=ctc_step_seconds,
                    time_offset=time_offset,
                    source_frame_indices=stream['source_frame_indices'],
                    active_region_ids=stream['active_region_ids'],
                )
            )
            score_by_speaker[stream['speaker_tag']] = score
            coarse_diagnostics_by_speaker[stream['speaker_tag']] = coarse_diagnostic
        return rows, score_by_speaker, coarse_diagnostics_by_speaker


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
        use_coarse_alignment: bool = True,
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Run blank-expanded CTC Viterbi DP with an optional Sortformer prior.

        The dense path remains the exact implementation. When
        ``coarse_alignment_band_size`` is configured, an approximate coarse pass
        first proposes a narrow target-state corridor for the fine DP; the method
        automatically falls back to dense Viterbi if that corridor is unsuitable.
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

        # Route serialized alignment through the same narrow batched kernel when
        # enabled. This prevents the old eager (T, target-state) emission
        # materialization even for the default one-stream alignment mode.
        if use_coarse_alignment and self.coarse_alignment_band_size is not None:
            labels_batch = labels_tensor.unsqueeze(0)
            state_columns_batch = torch.tensor(
                [[-1 if column is None else int(column) for column in state_speaker_columns]],
                dtype=torch.long,
            )
            full_source_frames = torch.arange(log_probs.shape[0], dtype=torch.long).unsqueeze(0)
            paths, scores, diagnostics = self._ctc_viterbi_align_batched(
                ctc_log_probs=log_probs,
                labels=labels_batch,
                state_lengths=torch.tensor([labels_tensor.numel()], dtype=torch.long),
                blank_id=blank_id,
                state_speaker_columns=state_columns_batch,
                speaker_probs=speaker_probs,
                speaker_logprob_weight=speaker_logprob_weight,
                speaker_gate_threshold=speaker_gate_threshold,
                source_frame_indices=full_source_frames,
                time_lengths=torch.tensor([log_probs.shape[0]], dtype=torch.long),
                separator_state_mask=torch.zeros_like(labels_batch, dtype=torch.bool),
                use_coarse_alignment=True,
            )
            return paths[0], scores[0], diagnostics[0]

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

        return self._ctc_viterbi_from_emissions(
            emissions=emissions,
            labels=labels_tensor,
            blank_id=blank_id,
            use_coarse_alignment=use_coarse_alignment,
        )


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
        source_frame_indices: Optional[torch.Tensor] = None,
        time_lengths: Optional[torch.Tensor] = None,
        separator_state_mask: Optional[torch.Tensor] = None,
        state_min_source_frames: Optional[torch.Tensor] = None,
        state_max_source_frames: Optional[torch.Tensor] = None,
        use_coarse_alignment: bool = True,
    ) -> Tuple[List[torch.Tensor], List[float], List[Dict[str, Any]]]:
        """Run independent CTC Viterbi paths in a padded speaker batch.

        ``source_frame_indices`` selects a compact per-stream CTC timeline. A
        value of ``-1`` denotes a virtual separator: only a blank state at a
        complete-word boundary is allowed there. This makes a long Sortformer
        silence an actual alignment boundary rather than a cheap CTC blank run.
        Optional ``state_min_source_frames`` and ``state_max_source_frames``
        provide inclusive original-CTC-frame bounds for each target state.
        They constrain non-blank token emissions on acoustic frames, while
        leaving CTC blank transitions and virtual separators unrestricted.
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
        num_source_frames = log_probs.shape[0]
        if num_streams == 0 or max_states < 2 or int(state_lengths.min().item()) < 2:
            raise ValueError("Each CTC stream must contain at least blank and one token state.")
        if int(state_lengths.max().item()) > max_states:
            raise ValueError("state_lengths cannot exceed labels.shape[1].")

        if source_frame_indices is None:
            source_frame_indices = torch.arange(num_source_frames, dtype=torch.long).unsqueeze(0).expand(num_streams, -1)
        else:
            source_frame_indices = source_frame_indices.detach().to(device='cpu', dtype=torch.long)
        if source_frame_indices.ndim != 2 or source_frame_indices.shape[0] != num_streams:
            raise ValueError("source_frame_indices must have shape (num_streams, max_time).")
        max_time = source_frame_indices.shape[1]
        if max_time == 0:
            raise ValueError("source_frame_indices must contain at least one time step.")
        if time_lengths is None:
            time_lengths = torch.full((num_streams,), max_time, dtype=torch.long)
        else:
            time_lengths = time_lengths.detach().to(device='cpu', dtype=torch.long)
        if time_lengths.ndim != 1 or time_lengths.shape[0] != num_streams:
            raise ValueError("time_lengths must have shape (num_streams,).")
        if int(time_lengths.min().item()) < 1 or int(time_lengths.max().item()) > max_time:
            raise ValueError("time_lengths must be in [1, source_frame_indices.shape[1]].")

        time_mask = torch.arange(max_time, dtype=torch.long).unsqueeze(0) < time_lengths.unsqueeze(1)
        invalid_source = time_mask & ((source_frame_indices < -1) | (source_frame_indices >= num_source_frames))
        if invalid_source.any():
            raise ValueError("source_frame_indices contains an invalid original CTC frame index.")
        start_frames = source_frame_indices[:, 0]
        end_frames = source_frame_indices.gather(1, (time_lengths - 1).unsqueeze(1)).squeeze(1)
        if (start_frames < 0).any() or (end_frames < 0).any():
            raise ValueError("Each compact CTC timeline must begin and end on an acoustic frame.")
        actual_time_mask = time_mask & (source_frame_indices >= 0)
        virtual_time_mask = time_mask & (source_frame_indices < 0)

        state_mask = torch.arange(max_states, dtype=torch.long).unsqueeze(0) < state_lengths.unsqueeze(1)
        valid_labels = labels[state_mask]
        if valid_labels.numel() == 0 or int(valid_labels.min().item()) < 0 or int(valid_labels.max().item()) >= log_probs.shape[1]:
            raise ValueError("CTC target contains labels outside the CTC vocabulary.")
        if separator_state_mask is None:
            if virtual_time_mask.any():
                raise ValueError("Virtual compact-timeline frames require separator_state_mask.")
            separator_state_mask = torch.zeros((num_streams, max_states), dtype=torch.bool)
        else:
            separator_state_mask = separator_state_mask.detach().to(device='cpu', dtype=torch.bool)
        if separator_state_mask.shape != labels.shape:
            raise ValueError("separator_state_mask must have shape (num_streams, max_states).")
        separator_state_mask = separator_state_mask & state_mask
        state_min_source_frames, state_max_source_frames = self._normalize_state_source_frame_bounds(
            state_min_source_frames=state_min_source_frames,
            state_max_source_frames=state_max_source_frames,
            labels=labels,
            state_lengths=state_lengths,
            blank_id=blank_id,
            num_source_frames=num_source_frames,
        )

        # Do not construct the dense (streams, time, target-state) emission
        # tensor when a coarse band is requested. The helper emits full target
        # trellises only for the short coarse pass and for individual fallbacks;
        # the fine pass gathers exactly the states inside each stream's band.
        if use_coarse_alignment and self.coarse_alignment_band_size is not None:
            return self._ctc_viterbi_align_batched_coarse_to_fine(
                log_probs=log_probs,
                labels=labels,
                state_lengths=state_lengths,
                blank_id=blank_id,
                state_speaker_columns=state_speaker_columns,
                speaker_probs=speaker_probs,
                speaker_logprob_weight=speaker_logprob_weight,
                speaker_gate_threshold=speaker_gate_threshold,
                source_frame_indices=source_frame_indices,
                time_lengths=time_lengths,
                separator_state_mask=separator_state_mask,
                virtual_time_mask=virtual_time_mask,
                state_min_source_frames=state_min_source_frames,
                state_max_source_frames=state_max_source_frames,
            )

        safe_source_frames = source_frame_indices.clamp_min(0)
        emissions = log_probs[safe_source_frames.unsqueeze(-1), labels.unsqueeze(1)]
        neg_inf = -float('inf')
        emissions.masked_fill_(~state_mask.unsqueeze(1), neg_inf)
        emissions.masked_fill_(~time_mask.unsqueeze(-1), neg_inf)

        # Per-state source-frame bounds apply only to non-blank tokens on
        # actual acoustic frames. Blank states must remain available for the
        # normal CTC transitions, including compact-timeline separators.
        token_states = state_mask & (labels != blank_id)
        out_of_bounds_tokens = (
            actual_time_mask.unsqueeze(-1)
            & token_states.unsqueeze(1)
            & (
                (safe_source_frames.unsqueeze(-1) < state_min_source_frames.unsqueeze(1))
                | (safe_source_frames.unsqueeze(-1) > state_max_source_frames.unsqueeze(1))
            )
        )
        emissions.masked_fill_(out_of_bounds_tokens, neg_inf)

        if speaker_probs is not None and (speaker_logprob_weight > 0.0 or speaker_gate_threshold is not None):
            speaker_probs = speaker_probs.detach().to(device='cpu', dtype=torch.float32)
            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != num_source_frames:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
            token_states = state_mask & (state_speaker_columns >= 0)
            if token_states.any():
                speaker_columns = state_speaker_columns[token_states]
                if int(speaker_columns.max().item()) >= speaker_probs.shape[1]:
                    raise ValueError("speaker mapping references a missing Sortformer column.")
                safe_columns = state_speaker_columns.clamp_min(0)
                activity = speaker_probs[safe_source_frames.unsqueeze(-1), safe_columns.unsqueeze(1)]
                token_emissions = actual_time_mask.unsqueeze(-1) & token_states.unsqueeze(1)
                if speaker_logprob_weight > 0.0:
                    soft_gate = torch.where(
                        token_emissions,
                        float(speaker_logprob_weight) * torch.log(activity.clamp_min(self.epsilon)),
                        torch.zeros_like(activity),
                    )
                    emissions = emissions + soft_gate
                if speaker_gate_threshold is not None:
                    inactive_tokens = token_emissions & (activity < float(speaker_gate_threshold))
                    emissions.masked_fill_(inactive_tokens, neg_inf)

        if virtual_time_mask.any():
            separator_emissions = torch.where(
                separator_state_mask.unsqueeze(1),
                torch.zeros_like(emissions),
                torch.full_like(emissions, neg_inf),
            )
            emissions = torch.where(virtual_time_mask.unsqueeze(-1), separator_emissions, emissions)

        previous_scores = torch.full((num_streams, max_states), neg_inf, dtype=torch.float32)
        previous_scores[:, 0] = emissions[:, 0, 0]
        previous_scores[:, 1] = emissions[:, 0, 1]
        backpointers = torch.full((max_time, num_streams, max_states), -1, dtype=torch.long)
        backpointers[0, :, 0] = 0
        backpointers[0, :, 1] = 1
        state_indices = torch.arange(max_states, dtype=torch.long).unsqueeze(0).expand(num_streams, -1)

        for frame_index in range(1, max_time):
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

            updated_scores = best_scores + emissions[:, frame_index, :]
            updated_scores.masked_fill_(~state_mask, neg_inf)
            valid_time = time_mask[:, frame_index]
            previous_scores = torch.where(valid_time.unsqueeze(1), updated_scores, previous_scores)
            backpointers[frame_index] = torch.where(
                valid_time.unsqueeze(1) & state_mask,
                best_previous_states,
                -torch.ones_like(best_previous_states),
            )

        last_blank_states = state_lengths - 1
        last_token_states = state_lengths - 2
        last_blank_scores = previous_scores.gather(1, last_blank_states.unsqueeze(1)).squeeze(1)
        last_token_scores = previous_scores.gather(1, last_token_states.unsqueeze(1)).squeeze(1)
        choose_token = last_token_scores > last_blank_scores
        final_states = torch.where(choose_token, last_token_states, last_blank_states)
        final_scores = torch.where(choose_token, last_token_scores, last_blank_scores)
        if not torch.isfinite(final_scores).all():
            failed_streams = torch.nonzero(~torch.isfinite(final_scores), as_tuple=False).flatten().tolist()
            raise _NoValidCTCViterbiPathError(
                failed_streams,
                active_region_restricted=bool(virtual_time_mask.any().item()),
            )

        paths = torch.full((num_streams, max_time), -1, dtype=torch.long)
        states = final_states.clone()
        stream_indices = torch.arange(num_streams, dtype=torch.long)
        for frame_index in range(max_time - 1, -1, -1):
            valid_time = time_mask[:, frame_index]
            paths[valid_time, frame_index] = states[valid_time]
            if frame_index > 0:
                previous_states = backpointers[frame_index, stream_indices, states]
                if (previous_states[valid_time] < 0).any():
                    raise RuntimeError("CTC Viterbi backtrace reached an invalid state.")
                states = torch.where(valid_time, previous_states, states)
        return (
            [paths[index, : int(time_lengths[index].item())] for index in range(num_streams)],
            [float(score.item()) for score in final_scores],
            [
                {
                    'requested_band_size': None,
                    'coarse_num_frames': None,
                    'coarse_stride': None,
                    'used_coarse_band': False,
                    'fallback_reason': None,
                }
                for _ in range(num_streams)
            ],
        )

    @staticmethod
    def _normalize_state_source_frame_bounds(
        *,
        state_min_source_frames: Optional[torch.Tensor],
        state_max_source_frames: Optional[torch.Tensor],
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        blank_id: int,
        num_source_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate optional inclusive token-state source-frame bounds."""
        num_streams, max_states = labels.shape
        expected_shape = (num_streams, max_states)
        if state_min_source_frames is None:
            state_min_source_frames = torch.zeros(expected_shape, dtype=torch.long)
        else:
            state_min_source_frames = state_min_source_frames.detach().to(device='cpu', dtype=torch.long)
        if state_max_source_frames is None:
            state_max_source_frames = torch.full(
                expected_shape,
                num_source_frames - 1,
                dtype=torch.long,
            )
        else:
            state_max_source_frames = state_max_source_frames.detach().to(device='cpu', dtype=torch.long)
        if state_min_source_frames.shape != expected_shape:
            raise ValueError("state_min_source_frames must have shape (num_streams, max_states).")
        if state_max_source_frames.shape != expected_shape:
            raise ValueError("state_max_source_frames must have shape (num_streams, max_states).")

        state_mask = torch.arange(max_states, dtype=torch.long).unsqueeze(0) < state_lengths.unsqueeze(1)
        token_states = state_mask & (labels != blank_id)
        invalid_bounds = token_states & (
            (state_min_source_frames < 0)
            | (state_max_source_frames >= num_source_frames)
            | (state_min_source_frames > state_max_source_frames)
        )
        if invalid_bounds.any():
            raise ValueError(
                "state source-frame bounds must be inclusive valid CTC frame indices with min <= max "
                "for every non-blank target state."
            )
        return state_min_source_frames, state_max_source_frames


    def _ctc_viterbi_align_batched_coarse_to_fine(
        self,
        *,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        blank_id: int,
        state_speaker_columns: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float],
        source_frame_indices: torch.Tensor,
        time_lengths: torch.Tensor,
        separator_state_mask: torch.Tensor,
        virtual_time_mask: torch.Tensor,
        state_min_source_frames: Optional[torch.Tensor] = None,
        state_max_source_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[float], List[Dict[str, Any]]]:
        """Align a padded speaker batch without a dense fine emission tensor.

        Each stream first receives a short, max-pooled coarse CTC path. The
        resulting per-frame center state defines a fixed-width band for the
        fine recurrence. The coarse pass is only a soft-prior guide: it keeps
        the Sortformer log-probability prior but deliberately disables the hard
        speaker gate. The final banded emissions below retain and enforce the
        caller's hard speaker gate. Unlike the legacy coarse path, the fine
        emissions are indexed directly from ``log_probs`` as
        ``(streams, time, 2 * N + 1)`` rather than materializing
        ``(streams, time, target_states)``.  Streams whose coarse/fine paths
        are unsuitable are rerun exactly through the existing dense batch path.
        """
        requested_band_size = self.coarse_alignment_band_size
        if requested_band_size is None:
            raise RuntimeError("Coarse-to-fine batch helper requires a positive band size.")

        num_streams, max_states = labels.shape
        max_time = source_frame_indices.shape[1]
        state_min_source_frames, state_max_source_frames = self._normalize_state_source_frame_bounds(
            state_min_source_frames=state_min_source_frames,
            state_max_source_frames=state_max_source_frames,
            labels=labels,
            state_lengths=state_lengths,
            blank_id=blank_id,
            num_source_frames=log_probs.shape[0],
        )
        if speaker_probs is not None and (speaker_logprob_weight > 0.0 or speaker_gate_threshold is not None):
            speaker_probs = speaker_probs.detach().to(device='cpu', dtype=torch.float32)
            if speaker_probs.ndim != 2 or speaker_probs.shape[0] != log_probs.shape[0]:
                raise ValueError("speaker_probs must have shape (T_ctc, num_speakers).")
        else:
            speaker_probs = None

        # Construct each stream's compact coarse timeline independently. The
        # grouping is necessarily stream-specific because active regions,
        # virtual separators, and transcript lengths differ, but the pooled
        # emissions and the actual coarse Viterbi recurrence below are batched
        # across every usable stream.
        state_centers = torch.zeros((num_streams, max_time), dtype=torch.long)
        coarse_num_frames: List[Optional[int]] = [None] * num_streams
        coarse_strides: List[Optional[int]] = [None] * num_streams
        coarse_target_acoustic_frames: List[Optional[int]] = [None] * num_streams
        coarse_grouping_modes: List[Optional[str]] = [None] * num_streams
        fallback_reasons: List[Optional[str]] = [None] * num_streams
        coarse_ready = torch.zeros(num_streams, dtype=torch.bool)
        coarse_groups: List[Optional[List[Tuple[int, int]]]] = [None] * num_streams

        for stream_index in range(num_streams):
            stream_time = int(time_lengths[stream_index].item())
            stream_states = int(state_lengths[stream_index].item())
            stream_virtual = virtual_time_mask[stream_index, :stream_time]
            token_labels = labels[stream_index, :stream_states]
            token_labels = token_labels[token_labels != blank_id].tolist()
            minimum_frames = self._minimum_ctc_frames(token_labels)

            groups: Optional[List[Tuple[int, int]]] = None
            for candidate_stride in range(4, 1, -1):
                candidate_groups = self._coarse_time_groups(
                    num_frames=stream_time,
                    stride=candidate_stride,
                    virtual_time_mask=stream_virtual,
                )
                acoustic_group_count = sum(
                    not bool(stream_virtual[start].item()) for start, _ in candidate_groups
                )
                if acoustic_group_count >= minimum_frames:
                    groups = candidate_groups
                    coarse_strides[stream_index] = candidate_stride
                    coarse_target_acoustic_frames[stream_index] = acoustic_group_count
                    coarse_grouping_modes[stream_index] = 'fixed_stride'
                    break

            # A uniform stride of two can still be too aggressive for dense
            # BPE targets: CTC needs at least one acoustic coarse frame per
            # token (plus repeated-token blanks). In that case, retain the
            # required number plus a modest slack by mixing singleton and
            # two-frame groups. Virtual separators and each active region's
            # first/last acoustic frame remain singleton groups.
            if groups is None:
                acoustic_frame_count = int((~stream_virtual).sum().item())
                target_slack = min(
                    acoustic_frame_count - minimum_frames,
                    max(8, int(math.ceil(0.05 * minimum_frames))),
                )
                target_acoustic_groups = minimum_frames + target_slack
                coarse_target_acoustic_frames[stream_index] = target_acoustic_groups
                candidate_groups = self._coarse_time_groups_target_aware(
                    num_frames=stream_time,
                    target_acoustic_groups=target_acoustic_groups,
                    virtual_time_mask=stream_virtual,
                )
                acoustic_group_count = sum(
                    not bool(stream_virtual[start].item()) for start, _ in candidate_groups
                )
                if acoustic_group_count >= minimum_frames:
                    groups = candidate_groups
                    # This construction deliberately mixes singleton and
                    # two-frame groups, so it has no uniform coarse stride.
                    coarse_grouping_modes[stream_index] = 'target_aware_mixed_1_2'

            if groups is None or len(groups) >= stream_time:
                fallback_reasons[stream_index] = 'insufficient_coarse_compression'
                if groups is not None:
                    coarse_num_frames[stream_index] = len(groups)
                continue

            coarse_num_frames[stream_index] = len(groups)
            coarse_groups[stream_index] = groups

        # Group construction above uses Python lists, but all expensive CTC
        # operations below are batched. Bucket pooled groups by width (at most
        # four for the current coarse strategies) so we never materialize a
        # full fine (stream_time, target_states) trellis just to form the
        # shorter coarse pass.
        ready_stream_indices = [
            stream_index for stream_index, groups in enumerate(coarse_groups) if groups is not None
        ]
        if ready_stream_indices:
            ready_index_tensor = torch.tensor(ready_stream_indices, dtype=torch.long)
            typed_ready_group_lists: List[List[Tuple[int, int]]] = []
            for stream_index in ready_stream_indices:
                groups = coarse_groups[stream_index]
                if groups is None:
                    raise RuntimeError("Coarse-ready stream is missing its group timeline.")
                typed_ready_group_lists.append(groups)

            max_coarse_time = max(len(groups) for groups in typed_ready_group_lists)
            coarse_time_lengths = torch.tensor(
                [len(groups) for groups in typed_ready_group_lists], dtype=torch.long
            )
            coarse_emissions = torch.full(
                (len(ready_stream_indices), max_coarse_time, max_states),
                -float('inf'),
                dtype=log_probs.dtype,
            )
            groups_by_width: Dict[int, List[Tuple[int, int]]] = {}
            for ready_stream_index, groups in enumerate(typed_ready_group_lists):
                for group_index, (start, end) in enumerate(groups):
                    groups_by_width.setdefault(end - start, []).append((ready_stream_index, group_index))

            for group_width, group_locations in groups_by_width.items():
                group_ready_indices = torch.tensor(
                    [ready_stream_index for ready_stream_index, _ in group_locations], dtype=torch.long
                )
                group_coarse_indices = torch.tensor(
                    [group_index for _, group_index in group_locations], dtype=torch.long
                )
                original_stream_indices = ready_index_tensor[group_ready_indices]
                group_sources = torch.stack(
                    [
                        source_frame_indices[
                            ready_stream_indices[ready_stream_index],
                            typed_ready_group_lists[ready_stream_index][group_index][0] : typed_ready_group_lists[
                                ready_stream_index
                            ][group_index][1],
                        ]
                        for ready_stream_index, group_index in group_locations
                    ],
                    dim=0,
                )
                group_count = len(group_locations)
                group_states = (
                    torch.arange(max_states, dtype=torch.long)
                    .view(1, 1, max_states)
                    .expand(group_count, group_width, max_states)
                )
                group_emissions = self._gather_ctc_emissions_for_states(
                    log_probs=log_probs,
                    labels=labels[original_stream_indices],
                    state_lengths=state_lengths[original_stream_indices],
                    source_frame_indices=group_sources,
                    time_lengths=torch.full((group_count,), group_width, dtype=torch.long),
                    state_speaker_columns=state_speaker_columns[original_stream_indices],
                    speaker_probs=speaker_probs,
                    speaker_logprob_weight=speaker_logprob_weight,
                    # This coarse path only guides the state corridor. A hard
                    # gate would make that guide brittle when pooled frames
                    # straddle a Sortformer boundary; the fine pass below
                    # retains the caller's gate and enforces the actual mask.
                    speaker_gate_threshold=None,
                    separator_state_mask=separator_state_mask[original_stream_indices],
                    state_indices=group_states,
                    blank_id=blank_id,
                    state_min_source_frames=state_min_source_frames[original_stream_indices],
                    state_max_source_frames=state_max_source_frames[original_stream_indices],
                )
                # Virtual separators are singleton groups, so pooling never
                # crosses one. Advanced indexing scatters each pooled row back
                # into the padded coarse batch.
                coarse_emissions[group_ready_indices, group_coarse_indices] = group_emissions.amax(dim=1)

            coarse_paths, _, coarse_valid = self._ctc_viterbi_dp_dense_batched(
                emissions=coarse_emissions,
                labels=labels[ready_index_tensor],
                state_lengths=state_lengths[ready_index_tensor],
                time_lengths=coarse_time_lengths,
                blank_id=blank_id,
            )
            for ready_stream_index, original_stream_index in enumerate(ready_stream_indices):
                if not bool(coarse_valid[ready_stream_index].item()):
                    fallback_reasons[original_stream_index] = 'coarse_path_unavailable'
                    continue
                stream_time = int(time_lengths[original_stream_index].item())
                state_centers[original_stream_index, :stream_time] = self._expand_coarse_state_path(
                    coarse_path=coarse_paths[
                        ready_stream_index, : int(coarse_time_lengths[ready_stream_index].item())
                    ],
                    groups=typed_ready_group_lists[ready_stream_index],
                    num_frames=stream_time,
                )
                coarse_ready[original_stream_index] = True

        # Shift a fixed-width interval at target edges rather than shrinking it:
        # this keeps the fine tensor rectangular and vectorizable across streams.
        max_band_width = min(max_states, 2 * requested_band_size + 1)
        band_widths = torch.minimum(state_lengths, torch.full_like(state_lengths, max_band_width))
        max_band_starts = (state_lengths - band_widths).clamp_min(0)
        band_starts = (state_centers - requested_band_size).clamp_min(0)
        band_starts = torch.minimum(band_starts, max_band_starts.unsqueeze(1))
        local_band_states = torch.arange(max_band_width, dtype=torch.long).view(1, 1, max_band_width)
        band_state_indices = band_starts.unsqueeze(-1) + local_band_states

        band_emissions = self._gather_ctc_emissions_for_states(
            log_probs=log_probs,
            labels=labels,
            state_lengths=state_lengths,
            source_frame_indices=source_frame_indices,
            time_lengths=time_lengths,
            state_speaker_columns=state_speaker_columns,
            speaker_probs=speaker_probs,
            speaker_logprob_weight=speaker_logprob_weight,
            speaker_gate_threshold=speaker_gate_threshold,
            separator_state_mask=separator_state_mask,
            state_indices=band_state_indices,
            blank_id=blank_id,
            state_min_source_frames=state_min_source_frames,
            state_max_source_frames=state_max_source_frames,
        )
        fine_paths, fine_scores, fine_valid, fine_touched_edge = self._ctc_viterbi_dp_banded_batched(
            emissions=band_emissions,
            labels=labels,
            state_lengths=state_lengths,
            time_lengths=time_lengths,
            blank_id=blank_id,
            band_starts=band_starts,
            band_widths=band_widths,
        )

        # Decide all fallbacks before running them. A single dense padded call
        # keeps exceptional streams vectorized too, rather than recursively
        # running one Viterbi recurrence per speaker.
        for stream_index in range(num_streams):
            fallback_reason = fallback_reasons[stream_index]
            if fallback_reason is None and not bool(coarse_ready[stream_index].item()):
                fallback_reason = 'coarse_path_unavailable'
            if fallback_reason is None and not bool(fine_valid[stream_index].item()):
                fallback_reason = 'band_has_no_valid_path'
            if fallback_reason is None and bool(fine_touched_edge[stream_index].item()):
                fallback_reason = 'path_touched_band_edge'
            fallback_reasons[stream_index] = fallback_reason

        fallback_stream_indices = [
            stream_index for stream_index, fallback_reason in enumerate(fallback_reasons) if fallback_reason is not None
        ]
        fallback_paths: Dict[int, torch.Tensor] = {}
        fallback_scores: Dict[int, float] = {}
        if fallback_stream_indices:
            fallback_index_tensor = torch.tensor(fallback_stream_indices, dtype=torch.long)
            try:
                dense_paths, dense_scores, _ = self._ctc_viterbi_align_batched(
                    ctc_log_probs=log_probs,
                    labels=labels[fallback_index_tensor],
                    state_lengths=state_lengths[fallback_index_tensor],
                    blank_id=blank_id,
                    state_speaker_columns=state_speaker_columns[fallback_index_tensor],
                    speaker_probs=speaker_probs,
                    speaker_logprob_weight=speaker_logprob_weight,
                    speaker_gate_threshold=speaker_gate_threshold,
                    source_frame_indices=source_frame_indices[fallback_index_tensor],
                    time_lengths=time_lengths[fallback_index_tensor],
                    separator_state_mask=separator_state_mask[fallback_index_tensor],
                    state_min_source_frames=state_min_source_frames[fallback_index_tensor],
                    state_max_source_frames=state_max_source_frames[fallback_index_tensor],
                    use_coarse_alignment=False,
                )
            except _NoValidCTCViterbiPathError as error:
                mapped_failed_streams: List[int] = []
                for failed_stream_index in error.failed_stream_indices:
                    if not 0 <= failed_stream_index < len(fallback_stream_indices):
                        raise RuntimeError("Dense coarse fallback reported an invalid stream index.") from error
                    mapped_failed_streams.append(fallback_stream_indices[failed_stream_index])
                raise _NoValidCTCViterbiPathError(
                    mapped_failed_streams,
                    active_region_restricted=error.active_region_restricted,
                ) from error
            for fallback_batch_index, stream_index in enumerate(fallback_stream_indices):
                fallback_paths[stream_index] = dense_paths[fallback_batch_index]
                fallback_scores[stream_index] = dense_scores[fallback_batch_index]

        paths: List[torch.Tensor] = []
        scores: List[float] = []
        diagnostics: List[Dict[str, Any]] = []
        for stream_index in range(num_streams):
            stream_time = int(time_lengths[stream_index].item())
            fallback_reason = fallback_reasons[stream_index]
            if fallback_reason is None:
                paths.append(fine_paths[stream_index, :stream_time].clone())
                scores.append(float(fine_scores[stream_index].item()))
                diagnostics.append(
                    {
                        'requested_band_size': requested_band_size,
                        'coarse_num_frames': coarse_num_frames[stream_index],
                        'coarse_stride': coarse_strides[stream_index],
                        'used_coarse_band': True,
                        'fallback_reason': None,
                        'fine_band_max_states': int(band_widths[stream_index].item()),
                        'coarse_frame_selection': 'max_pool',
                        'coarse_hard_speaker_gate': (
                            'disabled_for_guide' if speaker_gate_threshold is not None else 'not_requested'
                        ),
                        'coarse_target_acoustic_frames': coarse_target_acoustic_frames[stream_index],
                        'coarse_grouping_mode': coarse_grouping_modes[stream_index],
                    }
                )
                continue

            paths.append(fallback_paths[stream_index])
            scores.append(fallback_scores[stream_index])
            diagnostics.append(
                {
                    'requested_band_size': requested_band_size,
                    'coarse_num_frames': coarse_num_frames[stream_index],
                    'coarse_stride': coarse_strides[stream_index],
                    'used_coarse_band': False,
                    'fallback_reason': fallback_reason,
                    'fine_band_max_states': int(band_widths[stream_index].item()),
                    'coarse_frame_selection': 'max_pool',
                    'coarse_hard_speaker_gate': (
                        'disabled_for_guide' if speaker_gate_threshold is not None else 'not_requested'
                    ),
                    'coarse_target_acoustic_frames': coarse_target_acoustic_frames[stream_index],
                    'coarse_grouping_mode': coarse_grouping_modes[stream_index],
                }
            )
        return paths, scores, diagnostics

    def _gather_ctc_emissions_for_states(
        self,
        *,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        source_frame_indices: torch.Tensor,
        time_lengths: torch.Tensor,
        state_speaker_columns: torch.Tensor,
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
        speaker_gate_threshold: Optional[float],
        separator_state_mask: torch.Tensor,
        state_indices: torch.Tensor,
        blank_id: int,
        state_min_source_frames: Optional[torch.Tensor] = None,
        state_max_source_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather CTC emissions for arbitrary target-state indices.

        This is the key memory-pruning primitive for batched coarse-to-fine
        alignment. ``state_indices`` has the desired final state dimension, so
        no intermediate ``(streams, time, all_target_states)`` tensor is built.
        """
        if state_indices.ndim != 3:
            raise ValueError("state_indices must have shape (num_streams, max_time, num_states).")
        num_streams, max_states = labels.shape
        if state_indices.shape[0] != num_streams or source_frame_indices.shape[0] != num_streams:
            raise ValueError("State and source-frame batches must have the same number of streams.")
        if state_indices.shape[1] != source_frame_indices.shape[1]:
            raise ValueError("state_indices and source_frame_indices must have the same time dimension.")

        _, max_time, _ = state_indices.shape
        state_min_source_frames, state_max_source_frames = self._normalize_state_source_frame_bounds(
            state_min_source_frames=state_min_source_frames,
            state_max_source_frames=state_max_source_frames,
            labels=labels,
            state_lengths=state_lengths,
            blank_id=blank_id,
            num_source_frames=log_probs.shape[0],
        )
        state_indices = state_indices.detach().to(device='cpu', dtype=torch.long)
        time_mask = torch.arange(max_time, dtype=torch.long).unsqueeze(0) < time_lengths.unsqueeze(1)
        valid_states = (
            time_mask.unsqueeze(-1)
            & (state_indices >= 0)
            & (state_indices < state_lengths.view(num_streams, 1, 1))
        )
        safe_state_indices = state_indices.clamp(min=0, max=max_states - 1)
        state_labels = labels.gather(1, safe_state_indices.reshape(num_streams, -1)).reshape_as(safe_state_indices)
        safe_source_frames = source_frame_indices.clamp_min(0)
        emissions = log_probs[safe_source_frames.unsqueeze(-1), state_labels]
        neg_inf = -float('inf')
        emissions.masked_fill_(~valid_states, neg_inf)

        actual_time = time_mask & (source_frame_indices >= 0)
        gathered_min_frames = state_min_source_frames.gather(
            1, safe_state_indices.reshape(num_streams, -1)
        ).reshape_as(safe_state_indices)
        gathered_max_frames = state_max_source_frames.gather(
            1, safe_state_indices.reshape(num_streams, -1)
        ).reshape_as(safe_state_indices)
        out_of_bounds_tokens = (
            valid_states
            & actual_time.unsqueeze(-1)
            & (state_labels != blank_id)
            & (
                (safe_source_frames.unsqueeze(-1) < gathered_min_frames)
                | (safe_source_frames.unsqueeze(-1) > gathered_max_frames)
            )
        )
        emissions.masked_fill_(out_of_bounds_tokens, neg_inf)

        if speaker_probs is not None:
            state_columns = state_speaker_columns.gather(
                1, safe_state_indices.reshape(num_streams, -1)
            ).reshape_as(safe_state_indices)
            token_states = valid_states & actual_time.unsqueeze(-1) & (state_columns >= 0)
            if token_states.any():
                referenced_columns = state_columns[token_states]
                if int(referenced_columns.max().item()) >= speaker_probs.shape[1]:
                    raise ValueError("speaker mapping references a missing Sortformer column.")
                safe_columns = state_columns.clamp_min(0)
                activity = speaker_probs[safe_source_frames.unsqueeze(-1), safe_columns]
                if speaker_logprob_weight > 0.0:
                    emissions = emissions + torch.where(
                        token_states,
                        float(speaker_logprob_weight) * torch.log(activity.clamp_min(self.epsilon)),
                        torch.zeros_like(activity),
                    )
                if speaker_gate_threshold is not None:
                    emissions.masked_fill_(
                        token_states & (activity < float(speaker_gate_threshold)),
                        neg_inf,
                    )

        virtual_time = time_mask & (source_frame_indices < 0)
        if virtual_time.any():
            separator_states = separator_state_mask.gather(
                1, safe_state_indices.reshape(num_streams, -1)
            ).reshape_as(safe_state_indices)
            virtual_emissions = torch.where(
                separator_states & valid_states,
                torch.zeros_like(emissions),
                torch.full_like(emissions, neg_inf),
            )
            emissions = torch.where(virtual_time.unsqueeze(-1), virtual_emissions, emissions)
        return emissions

    @staticmethod
    def _ctc_viterbi_dp_banded_batched(
        *,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        time_lengths: torch.Tensor,
        blank_id: int,
        band_starts: torch.Tensor,
        band_widths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a vectorized CTC Viterbi recurrence in variable state bands.

        Backpointers store only the relative CTC transition (stay, +1, +2),
        using ``int8`` instead of full global-state indices.  A per-frame band
        start converts the local backtrace position back to a global target
        state without allocating a dense target-state trellis.
        """
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape (num_streams, max_time, max_band_states).")
        num_streams, max_time, max_band_width = emissions.shape
        if labels.ndim != 2 or labels.shape[0] != num_streams:
            raise ValueError("labels must have shape (num_streams, max_target_states).")
        if band_starts.shape != (num_streams, max_time):
            raise ValueError("band_starts must have shape (num_streams, max_time).")
        if state_lengths.shape != (num_streams,) or time_lengths.shape != (num_streams,):
            raise ValueError("state_lengths and time_lengths must have shape (num_streams,).")
        if band_widths.shape != (num_streams,) or int(band_widths.min().item()) < 1:
            raise ValueError("band_widths must be positive and have shape (num_streams,).")

        neg_inf = -float('inf')
        invalid_step = -128
        local_states = torch.arange(max_band_width, dtype=torch.long).unsqueeze(0)
        local_state_mask = local_states < band_widths.unsqueeze(1)
        initial_global_states = band_starts[:, 0].unsqueeze(1) + local_states
        initial_valid = local_state_mask & ((initial_global_states == 0) | (initial_global_states == 1))
        previous_scores = torch.where(
            initial_valid,
            emissions[:, 0, :],
            torch.full_like(emissions[:, 0, :], neg_inf),
        )
        backpointers = torch.full(
            (max_time, num_streams, max_band_width),
            invalid_step,
            dtype=torch.int8,
        )
        backpointers[0] = torch.where(
            initial_valid,
            torch.zeros((num_streams, max_band_width), dtype=torch.int8),
            torch.full((num_streams, max_band_width), invalid_step, dtype=torch.int8),
        )
        previous_starts = band_starts[:, 0]

        for frame_index in range(1, max_time):
            current_starts = band_starts[:, frame_index]
            current_global_states = current_starts.unsqueeze(1) + local_states
            current_valid = local_state_mask & (current_global_states < state_lengths.unsqueeze(1))

            def previous_at(offset: int) -> torch.Tensor:
                previous_local_states = current_global_states + offset - previous_starts.unsqueeze(1)
                previous_valid = (previous_local_states >= 0) & (
                    previous_local_states < band_widths.unsqueeze(1)
                )
                gathered = previous_scores.gather(
                    1, previous_local_states.clamp(min=0, max=max_band_width - 1)
                )
                return gathered.masked_fill(~previous_valid, neg_inf)

            best_scores = previous_at(0)
            best_steps = torch.zeros((num_streams, max_band_width), dtype=torch.int8)
            advance_one_scores = previous_at(-1)
            take_advance_one = advance_one_scores > best_scores
            best_scores = torch.where(take_advance_one, advance_one_scores, best_scores)
            best_steps = torch.where(
                take_advance_one,
                torch.full_like(best_steps, -1),
                best_steps,
            )

            safe_current_states = current_global_states.clamp(min=0, max=labels.shape[1] - 1)
            safe_two_back_states = (current_global_states - 2).clamp(min=0, max=labels.shape[1] - 1)
            current_labels = labels.gather(1, safe_current_states)
            two_back_labels = labels.gather(1, safe_two_back_states)
            can_skip = (
                current_valid
                & (current_global_states >= 2)
                & (current_labels != blank_id)
                & (current_labels != two_back_labels)
            )
            skip_scores = previous_at(-2).masked_fill(~can_skip, neg_inf)
            take_skip = skip_scores > best_scores
            best_scores = torch.where(take_skip, skip_scores, best_scores)
            best_steps = torch.where(take_skip, torch.full_like(best_steps, -2), best_steps)

            updated_scores = (best_scores + emissions[:, frame_index, :]).masked_fill(~current_valid, neg_inf)
            valid_time = frame_index < time_lengths
            previous_scores = torch.where(valid_time.unsqueeze(1), updated_scores, previous_scores)
            previous_starts = torch.where(valid_time, current_starts, previous_starts)
            backpointers[frame_index] = torch.where(
                valid_time.unsqueeze(1) & current_valid,
                best_steps,
                torch.full_like(best_steps, invalid_step),
            )

        last_time_indices = (time_lengths - 1).unsqueeze(1)
        last_band_starts = band_starts.gather(1, last_time_indices).squeeze(1)
        final_blank_states = state_lengths - 1
        final_token_states = state_lengths - 2

        def final_score_for(states: torch.Tensor) -> torch.Tensor:
            local = states - last_band_starts
            valid = (local >= 0) & (local < band_widths)
            gathered = previous_scores.gather(1, local.clamp(min=0, max=max_band_width - 1).unsqueeze(1)).squeeze(1)
            return gathered.masked_fill(~valid, neg_inf)

        final_blank_scores = final_score_for(final_blank_states)
        final_token_scores = final_score_for(final_token_states)
        choose_token = final_token_scores > final_blank_scores
        final_states = torch.where(choose_token, final_token_states, final_blank_states)
        final_scores = torch.where(choose_token, final_token_scores, final_blank_scores)
        valid_paths = torch.isfinite(final_scores)

        paths = torch.full((num_streams, max_time), -1, dtype=torch.long)
        touched_band_edge = torch.zeros(num_streams, dtype=torch.bool)
        states = final_states.clone()
        for frame_index in range(max_time - 1, -1, -1):
            valid_time = frame_index < time_lengths
            current_starts = band_starts[:, frame_index]
            local = states - current_starts
            in_band = (local >= 0) & (local < band_widths)
            valid_paths &= ~valid_time | in_band
            write_mask = valid_time & in_band
            paths[write_mask, frame_index] = states[write_mask]
            band_ends = current_starts + band_widths - 1
            touched_band_edge |= write_mask & (
                ((current_starts > 0) & (states == current_starts))
                | ((band_ends < state_lengths - 1) & (states == band_ends))
            )
            if frame_index > 0:
                steps = backpointers[frame_index].gather(
                    1, local.clamp(min=0, max=max_band_width - 1).unsqueeze(1)
                ).squeeze(1)
                valid_step = steps != invalid_step
                valid_paths &= ~valid_time | valid_step
                update_mask = write_mask & valid_step
                states = torch.where(update_mask, states + steps.to(dtype=torch.long), states)
        return paths, final_scores, valid_paths, touched_band_edge


    @staticmethod
    def _ctc_viterbi_dp_dense_batched(
        *,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        state_lengths: torch.Tensor,
        time_lengths: torch.Tensor,
        blank_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run exact full-state CTC Viterbi for a variable-length stream batch.

        This is the coarse-pass counterpart to ``_ctc_viterbi_dp_banded_batched``.
        It uses the same vectorized recurrence with a zero band start and a
        band width equal to each stream's full CTC target length, so there is
        no state-space approximation. Padding masks allow coarse timelines and
        blank-expanded targets to have different lengths in the same batch.
        """
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape (num_streams, max_time, max_target_states).")
        num_streams, max_time, max_states = emissions.shape
        if num_streams == 0 or max_time < 1 or max_states < 2:
            raise ValueError("Batched CTC emissions must contain streams, frames, and at least two states.")
        if labels.shape != (num_streams, max_states):
            raise ValueError("labels must have shape (num_streams, max_target_states).")
        if state_lengths.shape != (num_streams,) or time_lengths.shape != (num_streams,):
            raise ValueError("state_lengths and time_lengths must have shape (num_streams,).")
        if int(state_lengths.min().item()) < 2 or int(state_lengths.max().item()) > max_states:
            raise ValueError("state_lengths must be in [2, max_target_states].")
        if int(time_lengths.min().item()) < 1 or int(time_lengths.max().item()) > max_time:
            raise ValueError("time_lengths must be in [1, max_time].")

        band_starts = torch.zeros((num_streams, max_time), dtype=torch.long)
        paths, scores, valid, _ = PEETransformerCTCTimestampExtractor._ctc_viterbi_dp_banded_batched(
            emissions=emissions,
            labels=labels,
            state_lengths=state_lengths,
            time_lengths=time_lengths,
            blank_id=blank_id,
            band_starts=band_starts,
            band_widths=state_lengths,
        )
        return paths, scores, valid


    def _ctc_viterbi_from_emissions(
        self,
        *,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        blank_id: int,
        virtual_time_mask: Optional[torch.Tensor] = None,
        use_coarse_alignment: bool = True,
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """Run exact dense Viterbi on an already materialized single trellis.

        Coarse-enabled callers route through the direct-gather batched kernel
        before creating ``emissions``. This retained helper is therefore the
        exact dense path used when coarse alignment is disabled (including the
        preliminary speaker-column mapping).
        """
        if emissions.ndim != 2:
            raise ValueError(f"emissions must have shape (T, S), got {tuple(emissions.shape)}.")
        if labels.ndim != 1 or labels.shape[0] != emissions.shape[1]:
            raise ValueError("labels must have shape (S,) matching emissions.")
        emissions = emissions.detach().to(device='cpu', dtype=torch.float32)
        labels = labels.detach().to(device='cpu', dtype=torch.long)
        num_frames, num_states = emissions.shape
        if num_frames < 1 or num_states < 2:
            raise ValueError("CTC emissions must contain at least one frame and two states.")
        if virtual_time_mask is not None:
            virtual_time_mask = virtual_time_mask.detach().to(device='cpu', dtype=torch.bool)
            if virtual_time_mask.ndim != 1 or virtual_time_mask.numel() != num_frames:
                raise ValueError("virtual_time_mask must have shape (T,).")
        # Keep this private argument to make the exact preliminary call explicit.
        _ = use_coarse_alignment
        path, score = self._ctc_viterbi_dp_dense(
            emissions=emissions,
            labels=labels,
            blank_id=blank_id,
        )
        return path, score, {
            'requested_band_size': None,
            'coarse_num_frames': None,
            'coarse_stride': None,
            'used_coarse_band': False,
            'fallback_reason': None,
        }


    @staticmethod
    def _coarse_time_groups(
        *,
        num_frames: int,
        stride: int,
        virtual_time_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """Pool acoustic compact-time regions while retaining their endpoints."""
        if num_frames < 1 or stride < 1:
            raise ValueError("num_frames and stride must both be positive.")
        if virtual_time_mask.ndim != 1 or virtual_time_mask.numel() != num_frames:
            raise ValueError("virtual_time_mask must have shape (T,).")

        groups: List[Tuple[int, int]] = []
        cursor = 0
        while cursor < num_frames:
            if bool(virtual_time_mask[cursor].item()):
                groups.append((cursor, cursor + 1))
                cursor += 1
                continue

            region_start = cursor
            while cursor < num_frames and not bool(virtual_time_mask[cursor].item()):
                cursor += 1
            region_end = cursor
            region_length = region_end - region_start
            if region_length <= 2:
                groups.extend((frame, frame + 1) for frame in range(region_start, region_end))
                continue

            groups.append((region_start, region_start + 1))
            middle_end = region_end - 1
            middle_start = region_start + 1
            while middle_start < middle_end:
                middle_stop = min(middle_start + stride, middle_end)
                groups.append((middle_start, middle_stop))
                middle_start = middle_stop
            groups.append((region_end - 1, region_end))
        return groups

    @staticmethod
    def _coarse_time_groups_target_aware(
        *,
        num_frames: int,
        target_acoustic_groups: int,
        virtual_time_mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """Build a compressed CTC timeline retaining a target acoustic count.

        This is the fallback grouping strategy when even a uniform two-frame
        coarse pass would leave fewer frames than the CTC target's minimum
        legal duration. It starts from the uncompressed compact timeline and
        spends the available compression budget as two-frame acoustic groups.
        The caller chooses ``target_acoustic_groups`` at or above the target's
        minimum legal CTC duration. Virtual separators and the first and last
        acoustic frame of every active region always remain singleton groups.

        The pair budget is distributed proportionally over active regions, then
        pairs in each region are spread deterministically through its interior.
        This avoids concentrating representative-frame loss in an early region
        of a multi-region speaker timeline.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be positive.")
        if isinstance(target_acoustic_groups, bool) or not isinstance(target_acoustic_groups, int):
            raise TypeError("target_acoustic_groups must be an integer.")
        if target_acoustic_groups < 0:
            raise ValueError("target_acoustic_groups must be non-negative.")
        if virtual_time_mask.ndim != 1 or virtual_time_mask.numel() != num_frames:
            raise ValueError("virtual_time_mask must have shape (T,).")

        virtual_time_mask = virtual_time_mask.detach().to(device='cpu', dtype=torch.bool)
        acoustic_frame_count = int((~virtual_time_mask).sum().item())
        if target_acoustic_groups > acoustic_frame_count:
            raise ValueError(
                "target_acoustic_groups cannot exceed the number of acoustic compact-time frames."
            )

        # Split the compact timeline into contiguous acoustic regions and
        # singleton virtual separators. Region endpoints are deliberately kept
        # unpooled so each active region has CTC context at both boundaries.
        segments: List[Tuple[bool, int, int]] = []
        acoustic_regions: List[Tuple[int, int]] = []
        cursor = 0
        while cursor < num_frames:
            if bool(virtual_time_mask[cursor].item()):
                segments.append((True, cursor, cursor + 1))
                cursor += 1
                continue
            start = cursor
            while cursor < num_frames and not bool(virtual_time_mask[cursor].item()):
                cursor += 1
            acoustic_regions.append((start, cursor))
            segments.append((False, start, cursor))

        pair_capacities = [max(0, (end - start - 2) // 2) for start, end in acoustic_regions]
        max_pair_count = sum(pair_capacities)
        # A two-frame group replaces two singleton groups and therefore spends
        # one unit of the acoustic-group compression budget.
        pair_budget = min(acoustic_frame_count - target_acoustic_groups, max_pair_count)

        pair_counts = [0] * len(acoustic_regions)
        if pair_budget and max_pair_count:
            # Give each region a proportional base allocation, then resolve the
            # remaining pairs by largest fractional remainder (stable tie break
            # by region order). This is deterministic and cannot exceed a
            # region's pair capacity.
            remainders: List[Tuple[int, int]] = []
            allocated = 0
            for region_index, capacity in enumerate(pair_capacities):
                scaled = pair_budget * capacity
                count = scaled // max_pair_count
                pair_counts[region_index] = count
                allocated += count
                remainders.append((scaled % max_pair_count, region_index))
            for _, region_index in sorted(remainders, key=lambda item: (-item[0], item[1])):
                if allocated >= pair_budget:
                    break
                if pair_counts[region_index] < pair_capacities[region_index]:
                    pair_counts[region_index] += 1
                    allocated += 1
            if allocated != pair_budget:
                raise RuntimeError("Unable to distribute the target-aware CTC coarse-frame budget.")

        groups: List[Tuple[int, int]] = []
        region_index = 0
        for is_virtual, start, end in segments:
            if is_virtual:
                groups.append((start, end))
                continue

            pair_count = pair_counts[region_index]
            region_index += 1
            region_length = end - start
            if region_length <= 2:
                groups.extend((frame, frame + 1) for frame in range(start, end))
                continue

            groups.append((start, start + 1))
            interior_length = region_length - 2
            interior_group_count = interior_length - pair_count
            interior_cursor = start + 1
            for group_index in range(interior_group_count):
                # Bresenham-style placement of the pair groups across this
                # region's interior. A pair adds one extra acoustic frame.
                use_pair = (
                    ((group_index + 1) * pair_count) // interior_group_count
                    > (group_index * pair_count) // interior_group_count
                )
                group_size = 2 if use_pair else 1
                groups.append((interior_cursor, interior_cursor + group_size))
                interior_cursor += group_size
            if interior_cursor != end - 1:
                raise RuntimeError("Target-aware CTC coarse groups did not partition an acoustic region.")
            groups.append((end - 1, end))

        if region_index != len(acoustic_regions):
            raise RuntimeError("Target-aware CTC coarse groups lost an acoustic region.")
        if sum(not bool(virtual_time_mask[start].item()) for start, _ in groups) < target_acoustic_groups:
            raise RuntimeError("Target-aware CTC coarse groups retained too few acoustic frames.")
        return groups

    @staticmethod
    def _expand_coarse_state_path(
        *,
        coarse_path: torch.Tensor,
        groups: Sequence[Tuple[int, int]],
        num_frames: int,
    ) -> torch.Tensor:
        """Expand one monotonic coarse path back to every compact CTC frame."""
        if coarse_path.ndim != 1 or coarse_path.numel() != len(groups):
            raise ValueError("coarse_path and groups must have the same length.")
        state_centers = torch.empty(num_frames, dtype=torch.long)
        expected_start = 0
        for state, (start, end) in zip(coarse_path.tolist(), groups):
            if start != expected_start or not start < end <= num_frames:
                raise ValueError("Coarse timeline groups must partition the compact CTC timeline.")
            state_centers[start:end] = int(state)
            expected_start = end
        if expected_start != num_frames:
            raise ValueError("Coarse timeline groups do not cover the compact CTC timeline.")
        return state_centers

    @staticmethod
    def _ctc_viterbi_dp_dense(
        *,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        blank_id: int,
    ) -> Tuple[torch.Tensor, float]:
        """Run the original exact CTC Viterbi recurrence on one emission trellis."""
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
        can_skip = torch.zeros(num_states, dtype=torch.bool)
        if num_states > 2:
            can_skip[2:] = (labels[2:] != blank_id) & (labels[2:] != labels[:-2])

        for frame_index in range(1, num_frames):
            best_scores = previous_scores.clone()
            best_previous_states = state_indices.clone()

            advance_one_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
            advance_one_scores[1:] = previous_scores[:-1]
            take_advance_one = advance_one_scores > best_scores
            best_scores = torch.where(take_advance_one, advance_one_scores, best_scores)
            best_previous_states = torch.where(take_advance_one, state_indices - 1, best_previous_states)

            if can_skip.any():
                skip_scores = torch.full((num_states,), neg_inf, dtype=torch.float32)
                skip_scores[can_skip] = previous_scores[state_indices[can_skip] - 2]
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
        source_frame_indices: Optional[torch.Tensor] = None,
        active_region_ids: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a Viterbi state path into CTC word intervals and speaker metadata."""
        path = path.detach().to(device='cpu', dtype=torch.long)
        if path.ndim != 1:
            raise ValueError("path must be one-dimensional.")
        if source_frame_indices is None:
            source_frame_indices = torch.arange(path.numel(), dtype=torch.long)
        else:
            source_frame_indices = source_frame_indices.detach().to(device='cpu', dtype=torch.long)
        if source_frame_indices.ndim != 1 or source_frame_indices.numel() != path.numel():
            raise ValueError("source_frame_indices must be one-dimensional and match path length.")
        if active_region_ids is not None:
            active_region_ids = active_region_ids.detach().to(device='cpu', dtype=torch.long)
            if active_region_ids.ndim != 1 or active_region_ids.shape != source_frame_indices.shape:
                raise ValueError("active_region_ids must match source_frame_indices.")

        frames_by_word: List[List[Tuple[int, int, Optional[int]]]] = [[] for _ in tokenized_words]
        for compact_frame, state_index in enumerate(path.tolist()):
            if not 0 <= state_index < len(state_to_word):
                raise RuntimeError("CTC path contains a state outside the CTC target.")
            word_index = state_to_word[state_index]
            if word_index is None:
                continue
            source_frame = int(source_frame_indices[compact_frame].item())
            if source_frame < 0:
                raise RuntimeError("A CTC token state was assigned to a virtual active-region separator.")
            region_id: Optional[int] = None
            if active_region_ids is not None:
                region_id = int(active_region_ids[compact_frame].item())
                if region_id < 0:
                    raise RuntimeError("A CTC token state was assigned to an invalid active-region ID.")
            frames_by_word[word_index].append((source_frame, state_index, region_id))

        rows: List[Dict[str, Any]] = []
        for word, word_frames in zip(tokenized_words, frames_by_word):
            if not word_frames:
                raise RuntimeError(f"CTC path did not visit any token state for word {word['word']!r}.")
            frames = [frame for frame, _, _ in word_frames]
            state_indices = [state_index for _, state_index, _ in word_frames]
            region_ids = {region_id for _, _, region_id in word_frames if region_id is not None}
            if active_region_ids is not None and len(region_ids) != 1:
                raise RuntimeError(
                    f"CTC word {word['word']!r} crossed multiple active regions; this violates parallel alignment."
                )
            if frames != sorted(frames):
                raise RuntimeError("Compact CTC timeline did not map token frames to increasing original frames.")
            start_frame, end_frame = frames[0], frames[-1]
            selected_log_probs = torch.tensor(
                [ctc_log_probs[frame, labels[state_index]].item() for frame, state_index in zip(frames, state_indices)],
                dtype=torch.float32,
            )
            speaker_tag = word['speaker_tag']
            sortformer_column = speaker_mapping.get(speaker_tag)
            speaker_confidence: Optional[float] = None
            speaker_activity_start: Optional[float] = None
            speaker_activity_end: Optional[float] = None
            if speaker_probs is not None and sortformer_column is not None:
                frame_tensor = torch.tensor(frames, dtype=torch.long)
                activity = speaker_probs.index_select(0, frame_tensor)[:, sortformer_column]
                speaker_confidence = float(activity.mean().item())
                active_indices = torch.nonzero(activity >= self.speaker_activity_threshold, as_tuple=False).flatten()
                if active_indices.numel() > 0:
                    activity_start = frames[int(active_indices[0].item())]
                    activity_end = frames[int(active_indices[-1].item())]
                    speaker_activity_start = time_offset + activity_start * ctc_step_seconds
                    speaker_activity_end = time_offset + (activity_end + 1) * ctc_step_seconds

            row = {
                'word': word['word'],
                'word_index': word['word_index'],
                'turn_index': word.get('turn_index'),
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
            rows.append(row)
        return rows

    def _resolve_speaker_mapping(
        self,
        *,
        speaker_tags: Sequence[int],
        preliminary_rows: Sequence[Dict[str, Any]],
        speaker_probs: Optional[torch.Tensor],
        assignment_mode: str,
        candidate_columns: Optional[Sequence[int]] = None,
    ) -> Tuple[Dict[int, Optional[int]], Dict[int, List[float]]]:
        """Map t-SOT tags to selected raw Sortformer columns.

        ``candidate_columns`` restricts the optimal one-to-one assignment without
        renumbering Sortformer outputs, so emitted ``sortformer_column`` metadata
        always remains a raw model column index.
        """
        mapping: Dict[int, Optional[int]] = {tag: None for tag in speaker_tags}
        assignment_scores: Dict[int, List[float]] = {}
        if speaker_probs is None:
            return mapping, assignment_scores

        num_columns = int(speaker_probs.shape[1])
        if candidate_columns is None:
            candidates = list(range(num_columns))
        else:
            candidates = [int(column) for column in candidate_columns]
            if len(candidates) != len(set(candidates)):
                raise ValueError('candidate_columns must not contain duplicates.')
            invalid_columns = [column for column in candidates if not 0 <= column < num_columns]
            if invalid_columns:
                raise ValueError(
                    f'candidate_columns contains invalid Sortformer column(s): {invalid_columns}.'
                )
        if len(speaker_tags) > len(candidates):
            raise ValueError(
                'Cannot assign more t-SOT speakers than selected Sortformer columns. '
                'Use serialized alignment or provide enough candidate columns.'
            )

        if assignment_mode == 'identity':
            invalid_tags = [tag for tag in speaker_tags if not 0 <= tag < num_columns]
            if invalid_tags:
                raise ValueError(
                    "speaker_assignment_mode='identity' requires each t-SOT tag to match a "
                    f"Sortformer column; invalid tag(s): {invalid_tags}."
                )
            excluded_tags = [tag for tag in speaker_tags if tag not in candidates]
            if excluded_tags:
                raise ValueError(
                    "speaker_assignment_mode='identity' conflicts with the selected Sortformer columns; "
                    f'tag(s) {excluded_tags} are unavailable.'
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

        candidate_score_matrix = [[row[column] for column in candidates] for row in score_matrix]
        for tag, candidate_index in zip(valid_tags, self._maximum_weight_assignment(candidate_score_matrix)):
            mapping[tag] = candidates[candidate_index]
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
    def _normalize_coarse_alignment_band_size(value: Optional[int]) -> Optional[int]:
        """Normalize an optional coarse-to-fine CTC target-state radius."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("coarse_alignment_band_size must be an integer or None, not a boolean.")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise TypeError("coarse_alignment_band_size must be an integer or None.") from error
        if normalized != value:
            raise ValueError("coarse_alignment_band_size must be an integer or None.")
        if normalized < 0:
            raise ValueError("coarse_alignment_band_size must be non-negative or None.")
        return normalized or None

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

    @staticmethod
    def _validate_sot_transcripts(sot_transcripts: Sequence[str], batch_size: int) -> None:
        """Validate one textual t-SOT target per padded tensor row."""
        if isinstance(sot_transcripts, (str, bytes)) or not isinstance(sot_transcripts, Sequence):
            raise TypeError("sot_transcripts must be a sequence of one string per batch item.")
        if len(sot_transcripts) != batch_size:
            raise ValueError(
                f"sot_transcripts must contain {batch_size} entries, got {len(sot_transcripts)}."
            )
        invalid = [index for index, transcript in enumerate(sot_transcripts) if not isinstance(transcript, str)]
        if invalid:
            raise TypeError(f"sot_transcripts entries must be strings; invalid index/indices: {invalid}.")

    @classmethod
    def _select_batch_lengths(
        cls,
        values: Optional[Any],
        batch_size: int,
        maximum: int,
        name: str,
    ) -> List[int]:
        """Validate a padded-batch length vector and return host integers."""
        if batch_size <= 0 or maximum <= 0:
            raise ValueError(f"{name} requires positive batch_size and maximum.")
        if values is None:
            return [maximum] * batch_size
        if isinstance(values, torch.Tensor):
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape ({batch_size},), got {tuple(values.shape)}.")
            raw_values = values.detach().cpu().tolist()
        else:
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != batch_size:
                raise ValueError(f"{name} must contain exactly {batch_size} lengths.")
            raw_values = list(values)
        lengths = [cls._scalar_length(value, f"{name}[{index}]") for index, value in enumerate(raw_values)]
        invalid = [length for length in lengths if not 0 < length <= maximum]
        if invalid:
            raise ValueError(f"{name} entries must be in [1, {maximum}], got {invalid}.")
        return lengths

    @staticmethod
    def _select_optional_float_sequence(
        values: Optional[Sequence[Optional[float]]],
        batch_size: int,
        name: str,
    ) -> List[Optional[float]]:
        """Normalize optional per-record metadata while rejecting NaNs."""
        if values is None:
            return [None] * batch_size
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != batch_size:
            raise ValueError(f"{name} must contain exactly {batch_size} entries.")
        normalized: List[Optional[float]] = []
        for index, value in enumerate(values):
            if value is None:
                normalized.append(None)
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name}[{index}] must be a float or None.") from error
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name}[{index}] must be finite, got {value!r}.")
            normalized.append(numeric_value)
        return normalized

    @classmethod
    def _select_float_sequence(
        cls,
        values: Optional[Sequence[float]],
        batch_size: int,
        name: str,
        *,
        default: float,
    ) -> List[float]:
        """Normalize required numeric batch metadata with a scalar default."""
        optional_values = cls._select_optional_float_sequence(values, batch_size, name)
        return [float(default) if value is None else value for value in optional_values]

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
