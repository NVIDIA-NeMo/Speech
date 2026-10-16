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
import copy
import io
import math
import os
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.packed_sequence import (
    PackedEncoderActivations,
    pack_encoder_output,
    unpack_encoder_output,
)
from nemo.collections.asr.parts.preprocessing.features import normalize_batch, normalize_packed_batch
from nemo.collections.common.parts.optional_cuda_graphs import SharedGraphBuffers, capture_cuda_graph, copy_padded_rows
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, Serialization
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    "CTCTimestampInputs",
    "ParallelExpertEncoder",
    "ParallelExpertEncoderPT",
]


@dataclass(frozen=True)
class CTCTimestampInputs:
    """Request-owned inputs retained until post-generation CTC alignment.

    ``asr_encoded`` keeps detached BF16 or FP16 ASR states in channels-first
    form ``(B, D, T)`` expected by
    :class:`~nemo.collections.speechlm2.parts.ctc_timestamp_utils.TransformerCTCDecoder`.
    All tensors are detached from the perception graph before this value is returned.
    """

    asr_encoded: torch.Tensor
    asr_encoded_lengths: torch.Tensor
    sortformer_sigmoids: torch.Tensor
    sortformer_lengths: torch.Tensor
    diarization_labels: Optional[torch.Tensor] = None
    diarization_lengths: Optional[torch.Tensor] = None
    diarization_frame_seconds: float = 0.01


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
        "diar_normalize_type",
        "frame_shift_seconds",
        "max_speaker_count",
        "missing_rttm_target",
        "speaker_activity_threshold",
        "speaker_feature_config_version",
        "speaker_feature_mode",
        "spk_kernel_scale",
        "sync_max_audio_length",
    }
)


def _normalize_asr_encoder_type(asr_encoder_type: Optional[str]) -> str:
    """
    Validate and normalize the ASR architecture selector.
    ``None`` defaults to ``'fastconformer'``; any other value is lowercased and checked against
    the supported encoder types.

    Args:
        asr_encoder_type (Optional[str]): Requested ASR encoder type, or ``None``.

    Returns:
        str: Lowercased encoder type, either ``'fastconformer'`` or ``'transformer'``.
    """
    normalized = "fastconformer" if asr_encoder_type is None else str(asr_encoder_type).lower()
    if normalized not in _ASR_ENCODER_TYPES:
        supported = ", ".join(sorted(_ASR_ENCODER_TYPES))
        raise ValueError(f"asr_encoder_type must be one of {{{supported}}}, got {asr_encoder_type!r}.")
    return normalized


def _disable_max_seq_length_sync(module: nn.Module) -> None:
    """
    Disable expert collectives without importing optional SpeechLM dependencies.
    Clears ``sync_max_audio_length`` on every submodule that exposes it, so experts stop
    all-reducing the batch audio length across ranks.

    Args:
        module (nn.Module): Root module whose submodules are updated in place.
    """
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def _normalize_speaker_feature_contract(
    speaker_feature_mode: Optional[str],
    speaker_activity_threshold: Optional[float],
) -> tuple[str, Optional[float]]:
    """
    Validate one explicit speaker-feature fusion contract.
    ``None`` for ``speaker_feature_mode`` is supported only by the inner-module constructor, where
    the mode is derived from the threshold for API compatibility. Bundle configs are resolved
    separately and always become explicit before the inner module is constructed.

    Args:
        speaker_feature_mode (Optional[str]): ``'continuous'``, ``'thresholded'``, or ``None``.
        speaker_activity_threshold (Optional[float]): Activity threshold in ``[0, 1]``; must be ``None``
            for continuous mode and set for thresholded mode.

    Returns:
        tuple[str, Optional[float]]: Normalized mode and threshold (``None`` for continuous mode).
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
    """
    Resolve the versioned speaker-feature contract and fail closed when ambiguous.
    Uses an explicit ``speaker_feature_mode`` when present, otherwise derives the mode from a
    ``speaker_activity_threshold`` key. Raises when neither is present or the version is unsupported.

    Args:
        cfg (DictConfig): Bundle configuration holding the speaker-feature fields.

    Returns:
        tuple[str, Optional[float]]: Normalized speaker-feature mode and activity threshold.
    """
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
    """
    Merge the small, runtime-semantic PEE override surface into a bundle config.
    Only keys in ``_BUNDLE_CONFIG_OVERRIDE_KEYS`` are accepted; the input config is cloned, not mutated.

    Args:
        cfg (DictConfig): Bundle configuration to merge into.
        config_overrides (Optional[Mapping[str, Any]]): Override values, or ``None``/empty for no overrides.

    Returns:
        DictConfig: Cloned config with the overrides merged in.
    """
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
    """
    Temporarily set the global default float dtype.
    Does nothing when ``dtype`` already is the default or is not a floating-point type.

    Args:
        dtype (torch.dtype): Default dtype to use inside the scope.
    """
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
    """
    Temporarily make ``torch.distributed`` look uninitialized.
    Sortformer's streaming path synchronizes feature lengths across ranks. A generation worker
    processes one recording, so that synchronization is both unnecessary and unsafe there.
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
    """
    Deep-copy a ``DictConfig`` without resolving interpolations.

    Args:
        config (Optional[DictConfig]): Config to copy, or ``None``.

    Returns:
        Optional[DictConfig]: Independent copy of ``config``, or ``None`` when ``config`` is ``None``.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _read_bundle_members(nemo_path: str) -> tuple[DictConfig, dict[str, torch.Tensor]]:
    """
    Read a local PE bundle's config and state dictionary.
    Extracts ``model_config.yaml`` and ``model_weights.ckpt`` from the tar archive and loads the
    weights on CPU with ``weights_only=True``.

    Args:
        nemo_path (str): Path to the local ``.nemo`` archive.

    Returns:
        tuple[DictConfig, dict[str, torch.Tensor]]: Bundle config and its state dictionary.
    """
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
        """
        List pretrained Parallel Expert Encoder bundles; none are published.

        Returns:
            List[PretrainedModelInfo]: Always an empty list.
        """
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        """
        No-op; this shell only serializes the encoder and owns no training data.

        Args:
            train_data_config (Union[DictConfig, dict]): Ignored training data configuration.
        """
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        """
        No-op; this shell only serializes the encoder and owns no validation data.

        Args:
            val_data_config (Union[DictConfig, dict]): Ignored validation data configuration.
        """
        pass

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        """
        Initialize a serializable Parallel Expert Encoder wrapper.
        Validates the bundle schema, resolves the speaker-feature contract, builds the inner
        :class:`ParallelExpertEncoder`, and stores an architecture-only copy of the config on it
        as ``_bundle_config``. The CTC timestamp artifact path is always cleared.

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
            max_speaker_count=self._cfg.get("max_speaker_count", 8),
            # Timestamp artifacts are deployment inputs, not portable PEE architecture.
            ctc_timestamp_model_path=None,
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
        self.encoder._bundle_config.max_speaker_count = self.encoder.max_speaker_count
        self.encoder._bundle_config.ctc_timestamp_model_path = None

    @staticmethod
    def _validate_bundle_schema(cfg: DictConfig) -> None:
        """
        Require the self-contained ParallelExpertEncoder bundle schema.
        Both ``asr_encoder_cfg`` and ``diarization_model_cfg`` must be non-empty, and
        ``asr_encoder_type`` must be a supported value.

        Args:
            cfg (DictConfig): Bundle configuration to validate.
        """
        missing = [key for key in ("asr_encoder_cfg", "diarization_model_cfg") if cfg.get(key, None) in (None, {}, "")]
        if missing:
            raise ValueError(f"ParallelExpertEncoder bundle is missing required config sections {missing}.")
        _normalize_asr_encoder_type(cfg.get("asr_encoder_type", "fastconformer"))

    @classmethod
    def is_pe_nemo(cls, nemo_path: str) -> bool:
        """
        Return whether a local archive declares a ParallelExpertEncoderPT target.
        Only the ``target`` field of ``model_config.yaml`` is inspected; unreadable archives return ``False``.

        Args:
            nemo_path (str): Path to a candidate local ``.nemo`` file.

        Returns:
            bool: ``True`` when the archive's ``target`` ends with ``ParallelExpertEncoderPT``.
        """
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
        """
        Load a PE bundle and return its inner encoder.
        Local ``.nemo`` paths are read directly and their ``encoder.*`` tensors are loaded into a fresh
        shell; other identifiers go through :meth:`from_pretrained`. ``config_overrides`` is restricted
        to runtime-semantic fields so it resolves legacy bundle ambiguity without allowing a recipe to
        replace the saved encoder architecture accidentally, and it requires a local path.

        Args:
            model_path_or_name (str): Local ``.nemo`` path or pretrained model name.
            map_location (Union[str, torch.device]): Device the returned encoder is moved to.
            strict (bool): Whether to load the state dict strictly.
            config_overrides (Optional[Mapping[str, Any]]): Runtime-semantic config overrides.

        Returns:
            ParallelExpertEncoder: Loaded inner encoder.
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
        """
        Construct phPEE architecture without loading standalone weights.
        This is intended for consolidated SpeechLM checkpoints, whose root state dict supplies
        every phPEE tensor after construction.

        Args:
            cfg (Union[DictConfig, dict]): Inline bundle configuration.
            map_location (Union[str, torch.device]): Device the returned encoder is moved to.

        Returns:
            ParallelExpertEncoder: Randomly initialized inner encoder.
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
        """
        Save ``encoder`` using a compatible PE bundle config as a template.
        The template's ASR encoder type, ``d_model`` and speaker count must match ``encoder``. The
        encoder's runtime-semantic fields are written into the template config before saving.

        Args:
            encoder (ParallelExpertEncoder): Encoder whose weights are saved.
            output_nemo_path (str): Destination ``.nemo`` path.
            template_bundle_path (str): Existing PE bundle whose ``model_config.yaml`` is reused.
        """
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
        template_cfg.max_speaker_count = encoder.max_speaker_count
        template_cfg.ctc_timestamp_model_path = None
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
    supports_sequence_packed_output = True
    # Online inference encodes up to this many ASR windows (rows) per forward.
    _ASR_WINDOW_ROWS = 128
    # Replay the ASR windows and streaming Sortformer steps of online inference from CUDA graphs.
    online_cuda_graphs = False
    # Optional ``split(rows, frames) -> sizes``: covers a batch of ASR windows with graph batch sizes (the last piece
    # padded). Without it, windows are graphed at their exact row count.
    online_graph_split = None
    # Optional ``bucket(rows) -> size``: the graph batch size a streaming Sortformer step's rows are padded to, or
    # ``None`` to run it eagerly. Without it, steps are graphed at their exact row count.
    online_graph_bucket = None
    # CUDA graph memory pool handle and input/output buffers (SharedGraphBuffers) shared by the online graphs, and
    # the most graphs each graph cache keeps.
    online_graph_pool = None
    online_graph_buffers = None
    online_max_graphs = 16
    _diar_step_graphs = None
    _asr_window_graphs = None
    _capture_online_graphs_now = False

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
        max_speaker_count: int = 8,
        ctc_timestamp_model_path: Optional[str] = None,
        ctc_timestamp_alignment: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ASR and diarization experts and their fusion layers.
        Builds the ASR encoder and Sortformer model from inline configs, requiring both to share one
        subsampling factor, disables cross-rank max-length sync unless requested, and registers the
        sinusoidal speaker kernel used for fusion. Configured branches are frozen at the end.

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
            max_speaker_count (int): Fixed speaker dimension retained for boolean diarization timestamps.
            ctc_timestamp_model_path (Optional[str]): Runtime-only lightweight CTC timestamp artifact path.
            ctc_timestamp_alignment (Optional[Dict[str, Any]]): Timestamp aligner options, including backend selection.
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
        self.ctc_timestamp_alignment = dict(ctc_timestamp_alignment or {})
        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.max_speaker_count = int(max_speaker_count)
        if self.max_speaker_count < self.n_spk:
            raise ValueError(
                f"max_speaker_count={self.max_speaker_count} must be at least the diarization model's "
                f"n_spk={self.n_spk}."
            )
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
        """
        Apply the configured trainability and evaluation modes to both experts.
        Frozen branches get ``requires_grad_(False)`` and are put in evaluation mode.
        """
        if self.freeze_diar:
            self.diarization_model.requires_grad_(False)
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.requires_grad_(False)
            self.asr_encoder.eval()

    def train(self, mode: bool = True) -> ParallelExpertEncoder:
        """
        Set mode while keeping frozen branches in evaluation mode.

        Args:
            mode (bool): ``True`` for training mode, ``False`` for evaluation mode.

        Returns:
            ParallelExpertEncoder: This module.
        """
        super().train(mode)
        if self.freeze_diar:
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.eval()
        return self

    @property
    def d_model(self) -> int:
        """
        Output feature width, equal to the ASR encoder's ``d_model``.

        Returns:
            int: Width of the fused encoder states.
        """
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        """
        Time subsampling factor of the ASR encoder, shared by the diarization branch.

        Returns:
            int: Input frames per encoder output frame.
        """
        return self.asr_encoder.subsampling_factor

    @property
    def pre_encode(self):
        """
        Pre-encoder (subsampling) module of the ASR encoder.

        Returns:
            nn.Module: ``asr_encoder.pre_encode``, possibly wrapped for activation checkpointing.
        """
        return self.asr_encoder.pre_encode

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """
        Wrap trainable ASR stages before FSDP2 sharding.
        Wraps the non-linear ``pre_encode`` and each ASR layer once with ``checkpoint_wrapper``. The
        frozen Sortformer branch is deliberately excluded, and nothing is wrapped when ASR is frozen.
        Per-layer wrappers preserve FSDP2 boundaries, unlike a checkpoint around the entire encoder call.

        Args:
            enabled (bool): Whether to enable activation checkpointing; ``False`` is a no-op.
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
        """
        Validate an optional chunk duration.

        Args:
            name (str): Parameter name used in the error message.
            value (Optional[float]): Chunk duration in seconds, or ``None`` to disable chunking.

        Returns:
            Optional[float]: ``value`` as a positive float, or ``None``.
        """
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive or None, got {value}.")
        return value

    def _chunk_size_tokens(self, chunk_size_seconds: Optional[float]) -> Optional[int]:
        """
        Convert a chunk duration to a token count on the subsampled encoder grid.

        Args:
            chunk_size_seconds (Optional[float]): Chunk duration in seconds, or ``None``.

        Returns:
            Optional[int]: Tokens per chunk (at least one), or ``None`` when chunking is disabled.
        """
        if chunk_size_seconds is None:
            return None
        token_seconds = self.frame_shift_seconds * self.subsampling_factor
        return max(1, round(chunk_size_seconds / token_seconds))

    @staticmethod
    def _chunk_metadata(packed: PackedEncoderActivations, max_tokens: int) -> PackedEncoderActivations:
        """
        Re-describe packed utterances as independent chunks of at most ``max_tokens``.
        Only the sequence metadata changes; ``data`` is shared, so each utterance becomes
        consecutive full chunks followed by one partial remainder.

        Args:
            packed (PackedEncoderActivations): Packed utterances with data shaped ``(T_total, D)``.
            max_tokens (int): Maximum tokens per chunk.

        Returns:
            PackedEncoderActivations: Same data with per-chunk ``lengths`` and ``cu_seqlens``.
        """
        chunk_lengths = []
        for sequence_length in packed.lengths.detach().cpu().tolist():
            chunk_lengths.extend([max_tokens] * (sequence_length // max_tokens))
            if sequence_length % max_tokens:
                chunk_lengths.append(sequence_length % max_tokens)
        lengths = torch.as_tensor(chunk_lengths, dtype=torch.int64, device=packed.data.device)
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=packed.data.device),
                lengths.cumsum(0, dtype=torch.int32),
            ]
        ).contiguous()
        return PackedEncoderActivations(
            data=packed.data,
            lengths=lengths,
            cu_seqlens=cu_seqlens,
            max_seqlen=min(max_tokens, packed.max_seqlen),
            padding_value=packed.padding_value,
            padded_length=None,
        )

    @staticmethod
    def _match_packed_module_io(packed: PackedEncoderActivations, module: nn.Module) -> PackedEncoderActivations:
        """
        Cast packed data to a module's parameter dtype, requiring the same device.

        Args:
            packed (PackedEncoderActivations): Packed input activations.
            module (nn.Module): Module defining the expected device and dtype.

        Returns:
            PackedEncoderActivations: Converted activations, or the input for a parameterless module.
        """
        parameter = next(module.parameters(), None)
        if parameter is None:
            return packed
        if packed.data.device != parameter.device:
            raise ValueError(
                f"Packed input is on {packed.data.device}, but {type(module).__name__} is on {parameter.device}."
            )
        if packed.data.dtype == parameter.dtype:
            return packed
        return packed.with_data(packed.data.to(dtype=parameter.dtype))

    def _forward_packed_branch(
        self,
        encoder: nn.Module,
        features: PackedEncoderActivations,
        chunk_size_seconds: Optional[float],
    ) -> PackedEncoderActivations:
        """
        Run an encoder token-flat, optionally splitting after feature stacking.
        Encoders without ``forward_sequence_packed`` fall back to a padded forward and are
        re-packed. When a chunk size is set and an utterance exceeds it, feature stacking runs
        once over each full utterance, and the encoder layers then see each chunk as an
        independent sequence. Chunking requires ``subsampling='feature_stacking'``.

        Args:
            encoder (nn.Module): ASR or Sortformer encoder.
            features (PackedEncoderActivations): Packed mel features shaped ``(T_total, D)``.
            chunk_size_seconds (Optional[float]): Independent chunk duration, or ``None``.

        Returns:
            PackedEncoderActivations: Packed encoder states with per-utterance metadata.
        """
        max_tokens = self._chunk_size_tokens(chunk_size_seconds)
        packed_forward = getattr(encoder, "forward_sequence_packed", None)
        if not callable(packed_forward):
            if max_tokens is not None and features.max_seqlen > max_tokens:
                raise TypeError(f"{type(encoder).__name__} does not support packed independent chunking.")
            padded = unpack_encoder_output(features, total_length=features.padded_length).transpose(1, 2)
            encoded, encoded_lengths = encoder(audio_signal=padded, length=features.lengths)
            return pack_encoder_output(encoded.transpose(1, 2), encoded_lengths)
        if max_tokens is None or features.max_seqlen <= max_tokens:
            return packed_forward(features, features.lengths)

        pre_encode = getattr(encoder, "pre_encode", None)
        unwrapped_pre_encode = getattr(pre_encode, "_checkpoint_wrapped_module", pre_encode)
        if type(unwrapped_pre_encode).__name__ != "FeatureStacking":
            raise TypeError(
                "Independent post-stacking chunking requires subsampling='feature_stacking'; "
                f"got {type(unwrapped_pre_encode).__name__} for {type(encoder).__name__}."
            )
        pre_encoded = pre_encode(features)
        chunked = self._chunk_metadata(pre_encoded, max_tokens)
        encoded_chunks = packed_forward(
            chunked,
            chunked.lengths,
            bypass_pre_encode=True,
        )
        return pre_encoded.with_data(encoded_chunks.data)

    def _asr_output_frame_boundary(self, input_frame_boundary: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """
        Map an input-frame boundary to the selected ASR encoder's output grid.
        Transformer encoders use ceiling division by the subsampling factor; FastConformer rounds.
        A tensor boundary stays on its device.

        Args:
            input_frame_boundary (Union[int, torch.Tensor]): Boundary position in input feature frames.

        Returns:
            Union[int, torch.Tensor]: Corresponding boundary in encoder output frames.
        """
        if getattr(self, "asr_encoder_type", "fastconformer") == "transformer":
            return (input_frame_boundary + self.subsampling_factor - 1) // self.subsampling_factor
        if isinstance(input_frame_boundary, torch.Tensor):
            # torch.round, like Python's round, rounds halves to even.
            return torch.round(input_frame_boundary.double() / self.subsampling_factor).to(input_frame_boundary.dtype)
        return round(input_frame_boundary / self.subsampling_factor)

    def freeze(self) -> None:
        """
        Freeze all parameters and switch to evaluation mode via :func:`nemo.core.classes.module.freeze`.
        The prior ``requires_grad`` state is snapshotted so ``unfreeze(partial=True)`` can restore it.
        """
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        """
        Unfreeze parameters and switch to training mode via :func:`nemo.core.classes.module.unfreeze`.

        Args:
            partial (bool): Restore each parameter's pre-:meth:`freeze` ``requires_grad`` instead of
                enabling gradients on every parameter.
        """
        unfreeze(self, partial=partial)

    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        """
        Build sinusoidal speaker embeddings.

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
        """
        Pad or truncate speaker activity to a target frame count.
        Padding repeats the last frame.

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
        """
        Move a tensor to the device and dtype of a module.

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
        """
        Validate optional external speaker-activity targets.
        Checks the batch size and that the speaker dimension equals ``n_spk``; ``None`` is accepted.

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
        """
        Identify rows containing only the missing-target sentinel.
        Returns all ``False`` when no ``missing_rttm_target`` is configured.

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
        """
        Run a uniform training/distributed path while retaining the local eval fast path.
        Diarization always runs without external targets, in training mode, or with more than one
        distributed rank. Otherwise it runs only when some row holds the missing-target sentinel.

        Args:
            spk_targets (Optional[torch.Tensor]): External speaker targets shaped ``(B, T, S)``.
            use_diarization (Optional[torch.Tensor]): Precomputed missing-target row mask shaped ``(B,)``.

        Returns:
            bool: Whether the Sortformer branch must run.
        """
        if spk_targets is None or self.training:
            return True
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return True
        if use_diarization is None:
            use_diarization = self._missing_target_rows(spk_targets)
        return bool(use_diarization.any().item())

    def _speaker_features(self, targets: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Apply the bundle's explicit speaker-feature fusion contract.
        Continuous mode casts activity as is; thresholded mode binarizes it with
        ``speaker_activity_threshold``.

        Args:
            targets (torch.Tensor): Speaker activity with speakers on the last dimension.
            dtype (torch.dtype): Output dtype.

        Returns:
            torch.Tensor: Speaker features with the same shape as ``targets``.
        """
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
        """
        Fuse ASR states with continuous or explicitly thresholded speaker activity.
        Speaker activity is aligned to the ASR frame grid; rows flagged by ``use_diarization`` take
        ``diarization_preds`` instead. The output is ``LayerNorm(states)`` plus ``spk_kernel_scale``
        times the normalized speaker features projected through the sinusoidal speaker kernel.

        Args:
            asr_encoded (torch.Tensor): ASR states shaped ``(B, D, T)``.
            spk_targets (torch.Tensor): External or predicted speaker activity shaped ``(B, T', S)``.
            diarization_preds (Optional[torch.Tensor]): Sortformer predictions shaped ``(B, T', S)``.
            use_diarization (Optional[torch.Tensor]): Rows that should use ``diarization_preds``, shaped ``(B,)``.

        Returns:
            torch.Tensor: Speaker-infused states shaped ``(B, D, T)``.
        """
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
        """
        Route ``forward`` through the windowed generation path inside this scope.
        The previous ``online_inference_enabled`` value is restored on exit.

        Args:
            enabled (bool): Whether to force (``True``) or disable (``False``) windowed inference.
        """
        previous = getattr(self, "online_inference_enabled", None)
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    supports_ctc_timestamp_inputs = True

    @torch.no_grad()
    def generate_ctc_timestamps(
        self,
        timestamp_inputs: CTCTimestampInputs,
        sot_transcripts: Sequence[str],
        audio_durations: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """
        Align generated ASR text with this encoder's CTC timestamp aligner.
        The aligner owns the deferred CTC head, its windows and the alignment
        (``MultiSpeakerSOTWordTimestampAligner.generate_from_inputs`` in
        ``nemo.collections.speechlm2.parts.ctc_timestamp_utils``).

        Args:
            timestamp_inputs (CTCTimestampInputs): Inputs retained during the encoder forward pass.
            sot_transcripts (Sequence[str]): Generated serialized-output-training transcripts, one per recording.
            audio_durations (Sequence[float]): Audio duration of each recording in seconds.

        Returns:
            List[Dict[str, Any]]: Per-recording alignment results produced by the aligner.
        """
        from nemo.collections.speechlm2.parts.ctc_timestamp_utils import get_ctc_timestamp_aligner

        if not isinstance(timestamp_inputs, CTCTimestampInputs):
            raise TypeError("timestamp_inputs must be a CTCTimestampInputs instance.")
        aligner = get_ctc_timestamp_aligner(
            self,
            self.ctc_timestamp_model_path,
            timestamp_inputs.asr_encoded.device,
        )
        return aligner.generate_from_inputs(timestamp_inputs, sot_transcripts, audio_durations)

    def forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
        return_ctc_timestamp_inputs: bool = False,
    ):
        """
        Encode mels and fuse RTTM or Sortformer speaker activity.
        External ``spk_targets`` always use the single-pass path. Otherwise an active
        :meth:`online_inference` scope decides, and in evaluation mode inputs longer than one
        window (``chunk_feat_len``) use the windowed path when ``online_inference_length > 0``.

        Args:
            audio_signal (torch.Tensor): Unnormalized mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``.
            return_ctc_timestamp_inputs (bool): Whether to also return :class:`CTCTimestampInputs`.

        Returns:
            tuple: Fused states shaped ``(B, D, T_out)`` and their lengths shaped ``(B,)``, followed by
                :class:`CTCTimestampInputs` when ``return_ctc_timestamp_inputs`` is ``True``.
        """
        if spk_targets is not None:
            use_online = False
        elif getattr(self, "online_inference_enabled", None) is not None:
            use_online = bool(self.online_inference_enabled) and self.online_inference_length > 0
        elif self.online_inference_length > 0 and not self.training:
            use_online = audio_signal.shape[-1] > self.chunk_feat_len
        else:
            use_online = False
        if return_ctc_timestamp_inputs:
            runner = (
                self._forward_online_with_ctc_timestamp_inputs
                if use_online
                else self._forward_with_ctc_timestamp_inputs
            )
        else:
            runner = self._forward_online if use_online else self._forward
        return runner(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
        )

    def forward_sequence_packed(self, audio_signal, length=None, spk_targets=None) -> PackedEncoderActivations:
        """
        Run Sortformer first and ASR second while keeping encoder states token-flat.
        Offline counterpart of :meth:`forward` that never pads between utterances. Sortformer
        is skipped when external speaker targets make it unnecessary. Speaker activity is
        fused into the ASR states the same way as on the padded path.

        Args:
            audio_signal (Union[torch.Tensor, PackedEncoderActivations]): Packed mel features,
                or padded mel features shaped ``(B, D, T)``.
            length (Optional[torch.Tensor]): Valid input frame counts shaped ``(B,)``; required
                for padded input and must match ``audio_signal.lengths`` for packed input.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T, S)``.

        Returns:
            PackedEncoderActivations: Speaker-infused ASR states shaped ``(T_total, D)``.
        """
        if bool(getattr(self, "online_inference_enabled", False)):
            raise RuntimeError("forward_sequence_packed is an offline API and cannot run inside online_inference().")
        if isinstance(audio_signal, PackedEncoderActivations):
            if length is not None and not torch.equal(length.to(audio_signal.lengths), audio_signal.lengths):
                raise ValueError("length must match audio_signal.lengths for packed input.")
            features = audio_signal
        else:
            if length is None:
                raise ValueError("length is required for padded input.")
            features = pack_encoder_output(audio_signal.transpose(1, 2), length)

        self._check_spk_targets(spk_targets, features.batch_size)
        needs_diarization = self._should_run_diarization(spk_targets)
        diarization_preds = self._run_diarization_packed(features) if needs_diarization else None
        asr_encoded = self._run_asr_packed(features)
        if diarization_preds is not None and not (
            torch.equal(diarization_preds.lengths, asr_encoded.lengths)
            and torch.equal(diarization_preds.cu_seqlens, asr_encoded.cu_seqlens)
        ):
            raise RuntimeError(
                "Sortformer and ASR output metadata diverged: "
                f"diar={diarization_preds.lengths.detach().cpu().tolist()} "
                f"asr={asr_encoded.lengths.detach().cpu().tolist()}."
            )
        return self._fuse_diar_and_asr_packed(
            asr_encoded,
            spk_targets if spk_targets is not None else diarization_preds,
            diarization_preds=diarization_preds,
        )

    def _align_diarization_output_resolution(
        self, predictions: torch.Tensor, embedding_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Map native Sortformer probabilities onto the ASR fusion grid.
        Downsamples by ``output_subsampling_factor`` divided by the native output factor; returns the
        input unchanged when that ratio is at most one.

        Args:
            predictions (torch.Tensor): Native Sortformer probabilities shaped ``(B, T_native, S)``.
            embedding_lengths (torch.Tensor): Valid encoder embedding frame counts shaped ``(B,)``.

        Returns:
            torch.Tensor: Speaker probabilities on the ASR frame grid shaped ``(B, T_out, S)``.
        """
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
        """
        Pool unaligned high-resolution Sortformer probabilities exactly once.
        Only applies to a high-resolution model when ``predictions`` is longer than ``target_len``;
        otherwise the input is returned unchanged.

        Args:
            predictions (torch.Tensor): Speaker probabilities shaped ``(B, T, S)``.
            target_len (int): ASR frame count the result must match.

        Returns:
            torch.Tensor: Probabilities shaped ``(B, target_len, S)`` after pooling, or the input.
        """
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

    def _run_diarization_packed(self, features: PackedEncoderActivations) -> PackedEncoderActivations:
        """
        Normalize each utterance, then run the frozen streaming-trained Sortformer.
        Without post-encoder layers or high-resolution output, the speaker head runs
        token-flat. Otherwise the embeddings are unpacked for ``forward_infer`` and the
        predictions are re-packed.

        Args:
            features (PackedEncoderActivations): Packed mel features shaped ``(T_total, D)``.

        Returns:
            PackedEncoderActivations: Packed speaker probabilities shaped ``(T_total, S)``.
        """
        if self.diar_normalize_type:
            features = normalize_packed_batch(features, self.diar_normalize_type)
        features = self._match_packed_module_io(features, self.diarization_model.encoder)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_diar):
            embeddings = self._forward_packed_branch(
                self.diarization_model.encoder,
                features,
                self.chunk_size_seconds,
            )
            modules = self.diarization_model.sortformer_modules
            projected = embeddings.data
            if modules.encoder_proj is not None:
                projected = modules.encoder_proj(projected)

            post_encoder = self.diarization_model.transformer_encoder
            has_post_layers = post_encoder is not None and len(post_encoder.layers) > 0
            if has_post_layers or self.diarization_model.high_resolution:
                padded = unpack_encoder_output(embeddings)
                if modules.encoder_proj is not None:
                    padded = modules.encoder_proj(padded)
                predictions = self.diarization_model.forward_infer(padded, embeddings.lengths)
                predictions = self._align_diarization_output_resolution(predictions, embeddings.lengths)
                return pack_encoder_output(predictions, embeddings.lengths)
            if post_encoder is not None and post_encoder.final_layer_norm is not None:
                projected = post_encoder.final_layer_norm(projected)
            predictions = modules.forward_speaker_sigmoids(projected)
            return embeddings.with_data(predictions)

    def _run_asr_packed(self, features: PackedEncoderActivations) -> PackedEncoderActivations:
        """
        Normalize once per utterance, then run the trainable ASR packed path.

        Args:
            features (PackedEncoderActivations): Packed mel features shaped ``(T_total, D)``.

        Returns:
            PackedEncoderActivations: Packed ASR encoder states shaped ``(T_total, D)``.
        """
        if self.asr_normalize_type:
            features = normalize_packed_batch(features, self.asr_normalize_type)
        features = self._match_packed_module_io(features, self.asr_encoder)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_asr):
            return self._forward_packed_branch(
                self.asr_encoder,
                features,
                self.chunk_size_seconds,
            )

    def _fuse_diar_and_asr_packed(
        self,
        asr_encoded: PackedEncoderActivations,
        spk_targets: Union[torch.Tensor, PackedEncoderActivations],
        *,
        diarization_preds: Optional[PackedEncoderActivations] = None,
    ) -> PackedEncoderActivations:
        """
        Infuse packed speaker activity into packed ASR states.
        Padded external targets are aligned to the ASR grid and packed. Rows holding only
        the missing-target sentinel fall back to ``diarization_preds``.

        Args:
            asr_encoded (PackedEncoderActivations): Packed ASR states shaped ``(T_total, D)``.
            spk_targets (Union[torch.Tensor, PackedEncoderActivations]): Packed speaker
                predictions, or padded external targets shaped ``(B, T, S)``.
            diarization_preds (Optional[PackedEncoderActivations]): Predictions for missing-target rows.

        Returns:
            PackedEncoderActivations: Speaker-infused ASR states with the ASR metadata.
        """
        if isinstance(spk_targets, PackedEncoderActivations):
            packed_targets = spk_targets
        else:
            use_diarization = self._missing_target_rows(spk_targets)
            targets = self._align_diar_frames(spk_targets, asr_encoded.max_seqlen).to(
                device=asr_encoded.data.device, dtype=asr_encoded.data.dtype
            )
            if bool(use_diarization.any().item()):
                if diarization_preds is None:
                    raise ValueError("diarization_preds are required for missing speaker-target rows.")
                padded_preds = unpack_encoder_output(diarization_preds)
                targets = torch.where(
                    use_diarization.to(device=targets.device, dtype=torch.bool).view(-1, 1, 1),
                    padded_preds.to(device=targets.device, dtype=targets.dtype),
                    targets,
                )
            packed_targets = pack_encoder_output(targets, asr_encoded.lengths)

        if not torch.equal(packed_targets.lengths, asr_encoded.lengths):
            raise RuntimeError("Packed speaker attributions must match ASR output lengths.")
        speaker_features = self._speaker_features(packed_targets.data, asr_encoded.data.dtype)
        normalized_states = self.asr_norm(asr_encoded.data)
        normalized_targets = self.diar_norm(speaker_features)
        infusion = torch.matmul(normalized_targets, self.diar_kernel.to(normalized_targets.dtype))
        return asr_encoded.with_data(normalized_states + getattr(self, "spk_kernel_scale", 1.0) * infusion)

    def _run_diarization(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        *,
        return_native_resolution: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Run the diarization expert on a mel-spectrogram batch.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            return_native_resolution (bool): Whether to also return native 10 ms predictions and their
                lengths; requires a high-resolution Sortformer model.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: Speaker probabilities
                shaped ``(B, T_out, S)``, or a tuple of those probabilities, native predictions shaped
                ``(B, T_native, S)``, and native valid lengths shaped ``(B,)``.
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
            native_predictions = self.diarization_model.forward_infer(
                emb_seq=embeddings,
                emb_seq_length=embedding_lengths,
            )
            predictions = self._align_diarization_output_resolution(native_predictions, embedding_lengths)
        if not return_native_resolution:
            return predictions
        if not self.diarization_model.high_resolution:
            raise RuntimeError("Diarization timestamps require a high-resolution Sortformer model with 10 ms output.")
        native_lengths = embedding_lengths * int(self.diarization_model.encoder.subsampling_factor)
        native_lengths = torch.minimum(native_lengths, length.to(native_lengths.device)).clamp(
            max=native_predictions.shape[1]
        )
        return predictions, native_predictions, native_lengths

    def _run_asr(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the ASR expert on a mel-spectrogram batch.

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

    def _timestamp_speaker_probs(
        self,
        spk_targets: torch.Tensor,
        diarization_preds: Optional[torch.Tensor],
        use_diarization: Optional[torch.Tensor],
        target_len: int,
    ) -> torch.Tensor:
        """
        Select and align speaker probabilities for timestamp extraction.

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

    def _build_ctc_timestamp_inputs(
        self,
        asr_encoded: torch.Tensor,
        asr_encoded_lengths: torch.Tensor,
        speaker_probs: torch.Tensor,
        speaker_lengths: torch.Tensor,
        diarization_probs: torch.Tensor,
        diarization_lengths: torch.Tensor,
    ) -> CTCTimestampInputs:
        """
        Detach reduced-precision request-owned inputs needed after text decoding.
        ASR states are stored in BF16 unless already BF16 or FP16. Native diarization probabilities
        are thresholded at 0.5 into boolean labels padded to ``max_speaker_count`` speakers.

        Args:
            asr_encoded (torch.Tensor): ASR states shaped ``(B, D, T)``.
            asr_encoded_lengths (torch.Tensor): Valid ASR frame counts shaped ``(B,)``.
            speaker_probs (torch.Tensor): Speaker probabilities shaped ``(B, T, S)``.
            speaker_lengths (torch.Tensor): Valid speaker frame counts shaped ``(B,)``.
            diarization_probs (torch.Tensor): Native 10 ms Sortformer probabilities shaped ``(B, T, S)``.
            diarization_lengths (torch.Tensor): Valid native diarization frame counts shaped ``(B,)``.

        Returns:
            CTCTimestampInputs: Detached inputs for deferred CTC decoding.
        """
        storage_dtype = asr_encoded.dtype
        if storage_dtype not in (torch.bfloat16, torch.float16):
            storage_dtype = torch.bfloat16
        if diarization_probs.shape[-1] > self.max_speaker_count:
            raise ValueError(
                f"Diarization output has {diarization_probs.shape[-1]} speakers, exceeding "
                f"max_speaker_count={self.max_speaker_count}."
            )
        diarization_labels = torch.zeros(
            (diarization_probs.shape[0], self.max_speaker_count, diarization_probs.shape[1]),
            dtype=torch.bool,
            device=diarization_probs.device,
        )
        diarization_labels[:, : diarization_probs.shape[-1]] = (diarization_probs > 0.5).transpose(1, 2)
        return CTCTimestampInputs(
            asr_encoded=asr_encoded.detach().to(dtype=storage_dtype),
            asr_encoded_lengths=asr_encoded_lengths.detach(),
            sortformer_sigmoids=speaker_probs.detach(),
            sortformer_lengths=speaker_lengths.detach(),
            diarization_labels=diarization_labels.detach(),
            diarization_lengths=diarization_lengths.detach(),
            diarization_frame_seconds=self.frame_shift_seconds,
        )

    def _forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the single-pass encoder without retaining timestamp inputs.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused states shaped ``(B, D, T_out)`` and their lengths.
        """
        output, encoded_len, _ = self._forward_impl(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            return_ctc_timestamp_inputs=False,
        )
        return output, encoded_len

    def _forward_with_ctc_timestamp_inputs(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ) -> tuple[torch.Tensor, torch.Tensor, CTCTimestampInputs]:
        """
        Run the single-pass encoder and retain request-owned timestamp inputs.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, CTCTimestampInputs]: Fused states shaped ``(B, D, T_out)``,
                their lengths, and the detached CTC timestamp inputs.
        """
        output, encoded_len, timestamp_inputs = self._forward_impl(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            return_ctc_timestamp_inputs=True,
        )
        if timestamp_inputs is None:
            raise RuntimeError("CTC timestamp input capture completed without returning timestamp inputs.")
        return output, encoded_len, timestamp_inputs

    def _forward_impl(
        self,
        audio_signal,
        length,
        spk_targets,
        return_ctc_timestamp_inputs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[CTCTimestampInputs]]:
        """
        Single-pass forward used by training and validation.
        Runs Sortformer when fusion needs it or timestamp inputs are requested, then the ASR encoder,
        and fuses external targets (with Sortformer fallback for missing-target rows) or Sortformer
        predictions into the ASR states.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``, or ``None``.
            return_ctc_timestamp_inputs (bool): Whether to build :class:`CTCTimestampInputs`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[CTCTimestampInputs]]: Fused states shaped
                ``(B, D, T_out)``, their lengths, and timestamp inputs or ``None``.
        """
        self._check_spk_targets(spk_targets, audio_signal.shape[0])
        use_diarization = None if spk_targets is None else self._missing_target_rows(spk_targets)
        needs_diarization_for_fusion = self._should_run_diarization(spk_targets, use_diarization)
        needs_diarization = needs_diarization_for_fusion or return_ctc_timestamp_inputs
        native_diarization_preds = native_diarization_lengths = None
        if needs_diarization and return_ctc_timestamp_inputs:
            diarization_preds, native_diarization_preds, native_diarization_lengths = self._run_diarization(
                audio_signal,
                length,
                return_native_resolution=True,
            )
        elif needs_diarization:
            diarization_preds = self._run_diarization(audio_signal, length)
        else:
            diarization_preds = None
        asr_encoded, asr_encoded_len = self._run_asr(audio_signal, length)

        if spk_targets is None:
            spk_targets = diarization_preds
        elif not needs_diarization_for_fusion:
            use_diarization = None
        timestamp_inputs = None
        if return_ctc_timestamp_inputs:
            speaker_probs = self._timestamp_speaker_probs(
                spk_targets, diarization_preds, use_diarization, asr_encoded.shape[-1]
            )
            timestamp_inputs = self._build_ctc_timestamp_inputs(
                asr_encoded,
                asr_encoded_len.clamp(max=asr_encoded.shape[-1]),
                speaker_probs,
                asr_encoded_len.clamp(max=speaker_probs.shape[1]),
                native_diarization_preds,
                native_diarization_lengths,
            )
        output = self._fuse_diar_and_asr(
            asr_encoded,
            spk_targets,
            diarization_preds=diarization_preds,
            use_diarization=use_diarization,
        )
        return output, asr_encoded_len, timestamp_inputs

    def _forward_online(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the windowed encoder without retaining timestamp inputs.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused states shaped ``(B, D, T_out)`` and their lengths.
        """
        output, encoded_len, _ = self._forward_online_impl(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            return_ctc_timestamp_inputs=False,
        )
        return output, encoded_len

    def _forward_online_with_ctc_timestamp_inputs(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ) -> tuple[torch.Tensor, torch.Tensor, CTCTimestampInputs]:
        """
        Run the windowed encoder and retain request-owned timestamp inputs.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, CTCTimestampInputs]: Fused states shaped ``(B, D, T_out)``,
                their lengths, and the detached CTC timestamp inputs.
        """
        output, encoded_len, timestamp_inputs = self._forward_online_impl(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            return_ctc_timestamp_inputs=True,
        )
        if timestamp_inputs is None:
            raise RuntimeError("CTC timestamp input capture completed without returning timestamp inputs.")
        return output, encoded_len, timestamp_inputs

    def _forward_online_impl(
        self,
        audio_signal,
        length,
        spk_targets,
        return_ctc_timestamp_inputs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[CTCTimestampInputs]]:
        """
        Run both branches over context-extended long-form windows.
        The input is split into ``chunk_feat_len`` core windows extended by left and right context.
        Each window runs the ASR encoder and keeps only its core output frames. Unless every row has
        external targets, Sortformer runs in streaming mode with its FIFO and speaker cache, and its
        per-window predictions are aligned to the ASR core length. Each row reports its own output
        length. Native-resolution diarization is kept for timestamp inputs, or computed with a full
        :meth:`_run_diarization` pass when streaming diarization was skipped.

        Args:
            audio_signal (torch.Tensor): Mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
            spk_targets (Optional[torch.Tensor]): External speaker activity shaped ``(B, T', S)``, or ``None``.
            return_ctc_timestamp_inputs (bool): Whether to build :class:`CTCTimestampInputs`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, Optional[CTCTimestampInputs]]: Fused states shaped
                ``(B, D, T_out)``, per-row output lengths, and timestamp inputs or ``None``.
        """
        self._check_spk_targets(spk_targets, audio_signal.shape[0])
        # Windows cover the padded width: the window count must be known on the host without reading ``length``
        # (no device sync, CUDA-graph capturable). Frames past a row's length are masked like any shorter row's
        # padding, so extra padding costs compute but does not change the outputs.
        total_feat_len = audio_signal.shape[-1]
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
            if return_ctc_timestamp_inputs and not self.diarization_model.high_resolution:
                raise RuntimeError(
                    "Diarization timestamps require a high-resolution Sortformer model with 10 ms output."
                )
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

        windows = []
        for chunk_index in range(num_chunks):
            start = chunk_index * self.chunk_feat_len
            end = min(start + self.chunk_feat_len, total_feat_len)
            context_start = max(start - self.left_ctx_feat_len, 0)
            context_end = min(end + self.right_ctx_feat_len, total_feat_len)
            windows.append((start, end, context_start, context_end))
        # ASR windows do not depend on each other, so all of them are encoded together; only the Sortformer
        # below carries state from window to window.
        asr_chunks = self._encode_asr_windows(asr_signal, asr_length, windows)
        diar_chunks: List[torch.Tensor] = []
        # The window loop uses the longest row to keep every batch tensor
        # rectangular. Report each row's actual output length independently;
        # adding the longest row's scalar core length to every row would expose
        # padded frames as valid for shorter audios.
        valid_feat_lengths = length.clamp(max=audio_signal.shape[-1])
        encoded_len = self._asr_output_frame_boundary(valid_feat_lengths.detach()).to(dtype=asr_length.dtype)
        for chunk_index in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, "_suppress_online_pbar", False),
        ):
            start, end, context_start, context_end = windows[chunk_index]
            left_offset = start - context_start
            right_offset = context_end - end
            core_len = asr_chunks[chunk_index].shape[-1]

            if run_streaming_diar:
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
                    streaming_state, chunk_preds = self._streaming_diar_step(
                        diar_chunk,
                        diar_chunk_length,
                        streaming_state,
                        empty_preds=total_preds[:, :0],
                        left_offset=left_offset,
                        right_offset=right_offset,
                        return_native_resolution=return_ctc_timestamp_inputs,
                        # The state after the last window is discarded.
                        update_state=chunk_index < num_chunks - 1,
                    )
                total_preds = torch.cat([total_preds, chunk_preds], dim=1)
                if return_ctc_timestamp_inputs:
                    chunk_preds = self.diarization_model.sortformer_modules.downsample_preds(
                        chunk_preds,
                        int(self.diarization_model.output_subsampling_factor),
                    )
                diar_chunks.append(self._align_diar_frames(chunk_preds, core_len))

        asr_encoded = torch.cat(asr_chunks, dim=2)
        diarization_preds = torch.cat(diar_chunks, dim=1) if run_streaming_diar else None
        native_diarization_preds = native_diarization_lengths = None
        if return_ctc_timestamp_inputs:
            if run_streaming_diar:
                native_diarization_preds = total_preds
                native_diarization_lengths = valid_feat_lengths.to(total_preds.device).clamp(max=total_preds.shape[1])
            else:
                _, native_diarization_preds, native_diarization_lengths = self._run_diarization(
                    audio_signal,
                    length,
                    return_native_resolution=True,
                )
        if spk_targets is None:
            spk_targets = diarization_preds
            use_diarization = None
        timestamp_inputs = None
        if return_ctc_timestamp_inputs:
            speaker_probs = self._timestamp_speaker_probs(
                spk_targets, diarization_preds, use_diarization, asr_encoded.shape[-1]
            )
            timestamp_inputs = self._build_ctc_timestamp_inputs(
                asr_encoded,
                encoded_len.clamp(max=asr_encoded.shape[-1]),
                speaker_probs,
                encoded_len.clamp(max=speaker_probs.shape[1]),
                native_diarization_preds,
                native_diarization_lengths,
            )
        output = self._fuse_diar_and_asr(
            asr_encoded,
            spk_targets,
            diarization_preds=diarization_preds,
            use_diarization=use_diarization,
        )
        return output, encoded_len, timestamp_inputs

    def _encode_asr_windows(
        self, asr_signal: torch.Tensor, asr_length: torch.Tensor, windows: List[tuple]
    ) -> List[torch.Tensor]:
        """
        Encode context-extended ASR windows, many per forward, and return each window's core frames.
        Windows of equal width share a forward (at most ``_ASR_WINDOW_ROWS`` rows), so no window is padded and each
        sees exactly the input it would see alone. Only the first and last windows differ in width.

        Args:
            asr_signal (torch.Tensor): Normalized mel features shaped ``(B, D, T)``.
            asr_length (torch.Tensor): Valid frame counts shaped ``(B,)``.
            windows (List[tuple]): ``(start, end, context_start, context_end)`` input frames per window.

        Returns:
            List[torch.Tensor]: Per window, the encoded core frames shaped ``(B, D_out, T_core)``.
        """
        batch_size = asr_signal.shape[0]
        by_width = {}
        for index, (_, _, context_start, context_end) in enumerate(windows):
            by_width.setdefault(context_end - context_start, []).append(index)
        group_size = max(1, self._ASR_WINDOW_ROWS // batch_size)
        cores: List[Optional[torch.Tensor]] = [None] * len(windows)
        for width, indices in by_width.items():
            for first in range(0, len(indices), group_size):
                group = indices[first : first + group_size]
                context_starts = [windows[i][2] for i in group]
                rows = torch.cat([asr_signal[:, :, cs : cs + width] for cs in context_starts])
                lengths = torch.cat([(asr_length - cs).clamp(min=0, max=width) for cs in context_starts])
                with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_asr):
                    encoded = self._asr_window_forward(rows, lengths)
                for k, i in enumerate(group):
                    start, end, context_start, _ = windows[i]
                    left_drop = (start - context_start) // self.subsampling_factor
                    core_len = self._asr_output_frame_boundary(end) - self._asr_output_frame_boundary(start)
                    core_len = max(0, min(core_len, encoded.shape[-1] - left_drop))
                    cores[i] = encoded[k * batch_size : (k + 1) * batch_size, :, left_drop : left_drop + core_len]
        return cores

    def _asr_window_forward(self, rows: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of equal-width ASR windows; with ``online_cuda_graphs`` set, inference replays CUDA graphs.

        Args:
            rows (torch.Tensor): Window mel features shaped ``(N, D, T)``.
            lengths (torch.Tensor): Valid window frame counts shaped ``(N,)``.

        Returns:
            torch.Tensor: Encoded windows shaped ``(N, D_out, T_out)``.
        """
        if not self._online_graphable(rows):
            return self.asr_encoder(audio_signal=rows, length=lengths)[0]
        if self._asr_window_graphs is None:
            self._asr_window_graphs = _ASRWindowGraphs(
                self.online_max_graphs, self.online_graph_pool, self.online_graph_buffers
            )
        return self._asr_window_graphs(
            self.asr_encoder, rows, lengths, self.online_graph_split, capture_now=self._capture_online_graphs_now
        )

    def _online_graphable(self, tensor: torch.Tensor) -> bool:
        """Whether online inference on ``tensor`` may replay CUDA graphs (graphs enabled, inference on CUDA)."""
        return (
            self.online_cuda_graphs
            and tensor.is_cuda
            and not torch.is_grad_enabled()
            # A graph fixes kernel and dtype choices: autocast and an enclosing capture run eagerly.
            and not torch.is_autocast_enabled("cuda")
            and not torch.cuda.is_current_stream_capturing()
        )

    @contextlib.contextmanager
    def capturing_online_graphs(self):
        """Capture every new online graph key at its first sight (normally the second), e.g. during a warmup."""
        previous = self._capture_online_graphs_now
        self._capture_online_graphs_now = True
        try:
            yield
        finally:
            self._capture_online_graphs_now = previous

    def complete_online_asr_graphs(self, batch_sizes) -> None:
        """
        Capture the ASR window graphs at every batch size for every window shape captured so far.

        Args:
            batch_sizes (Callable): ``batch_sizes(frames) -> sizes`` for windows of ``frames`` input frames.
        """
        if self._asr_window_graphs is not None:
            self._asr_window_graphs.capture_batch_sizes(self.asr_encoder, batch_sizes)

    def online_graph_count(self) -> int:
        """Number of captured online-inference CUDA graphs (ASR windows and streaming Sortformer steps)."""
        caches = (self._asr_window_graphs, self._diar_step_graphs)
        return sum(len(cache.graphs) for cache in caches if cache is not None)

    def reset_online_graphs(self) -> None:
        """Turn ``online_cuda_graphs`` off and drop every captured online graph."""
        self.online_cuda_graphs = False
        self._asr_window_graphs = self._diar_step_graphs = None

    def _streaming_diar_step(
        self,
        chunk: torch.Tensor,
        chunk_length: torch.Tensor,
        streaming_state: Any,
        empty_preds: torch.Tensor,
        **step_kwargs,
    ) -> tuple[Any, torch.Tensor]:
        """
        Run one streaming Sortformer step and return the new state and this window's predictions.
        With ``online_cuda_graphs`` set, steps of inference with synchronous streaming replay CUDA graphs, keyed by
        their input shapes; any other step runs eagerly.

        Args:
            chunk (torch.Tensor): Window mel features, time-major, shaped ``(B, T, D)``.
            chunk_length (torch.Tensor): Valid window frame counts shaped ``(B,)``.
            streaming_state (Any): Sortformer streaming state from the previous step.
            empty_preds (torch.Tensor): Empty prediction buffer passed to the step.
            **step_kwargs: Extra keyword arguments of ``forward_streaming_step``.

        Returns:
            tuple[Any, torch.Tensor]: Updated streaming state and this window's speaker predictions.
        """

        def step(state, signal, signal_length):
            return self.diarization_model.forward_streaming_step(
                processed_signal=signal,
                processed_signal_length=signal_length,
                streaming_state=state,
                # Sized from the window, whose rows a graph may have padded.
                total_preds=empty_preds.new_empty((signal.shape[0], 0, empty_preds.shape[-1])),
                **step_kwargs,
            )

        graphable = (
            self._online_graphable(chunk)
            and not self.diarization_model.async_streaming
            and streaming_state.spk_perm is None
        )
        if not graphable:
            return step(streaming_state, chunk, chunk_length)
        if self._diar_step_graphs is None:
            self._diar_step_graphs = _StreamingStepGraphs(
                self.online_max_graphs, self.online_graph_pool, self.online_graph_buffers
            )
        return self._diar_step_graphs(
            step,
            streaming_state,
            chunk,
            chunk_length,
            tuple(sorted(step_kwargs.items())),
            bucket=self.online_graph_bucket,
            capture_now=self._capture_online_graphs_now,
        )

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        """
        Initialize Sortformer state for windowed inference.
        Applies this encoder's window, FIFO and speaker-cache settings to the Sortformer modules,
        validates them, and moves the inputs to the diarization model's device and dtype.

        Args:
            audio_signal (torch.Tensor): Input mel features shaped ``(B, D, T)``.
            length (torch.Tensor): Valid input frame counts shaped ``(B,)``.
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
            # .to() walks every submodule; only call it when the tracker is actually stale.
            if getattr(self.diarization_model, "device", device) != device:
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


class _GraphCache:
    """CUDA graphs keyed by input shapes, captured lazily (inference only).

    A key gets a graph the second time it is seen (or at once with ``capture_now``), so shapes that occur only once
    stay eager. Past ``max_graphs`` graphs, new keys run eagerly. If a capture fails, everything runs eagerly from
    then on.

    Args:
        max_graphs (int): Maximum number of captured graphs.
        pool: CUDA graph memory pool handle to capture into, or ``None`` for a private pool.
        buffers (Optional[SharedGraphBuffers]): Buffers for the graphs' static inputs and outputs shared with other
            graphs that never replay concurrently, or ``None`` to give every graph its own.
    """

    def __init__(self, max_graphs: int = 16, pool=None, buffers: Optional[SharedGraphBuffers] = None):
        self.max_graphs = max_graphs
        self.enabled = True
        self.seen = set()
        self.graphs = {}
        self.pool = pool
        self.buffers = buffers if buffers is not None else _PrivateGraphBuffers()

    def _lookup(self, key, capture, capture_now: bool = False):
        """
        Return the graph entry for ``key``, capturing it with ``capture()`` when due, or ``None`` to run eagerly.

        Args:
            key: Hashable input signature.
            capture (Callable): Captures the graph and returns its entry.
            capture_now (bool): Capture at the first sight of ``key``.

        Returns:
            The cached entry, or ``None``.
        """
        if key in self.graphs:
            return self.graphs[key]
        if not self.enabled or len(self.graphs) >= self.max_graphs or (key not in self.seen and not capture_now):
            self.seen.add(key)
            return None
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        try:
            self.graphs[key] = capture()
        except Exception as error:  # noqa: BLE001 - a failed capture must not break inference
            logging.warning(f"CUDA graph capture of {type(self).__name__} failed; running eagerly: {error}")
            self.enabled = False
            self.graphs.clear()
            return None
        return self.graphs[key]


class _ASRWindowGraphs(_GraphCache):
    """CUDA graphs of the ASR encoder over batches of equal-width windows.

    ``split(rows, frames)`` covers a batch with graph batch sizes; each piece is padded to its size with copies of its
    first row, so a few graphs serve any window count. Outputs are cloned, so callers own them.
    """

    def __call__(self, encoder, rows: torch.Tensor, lengths: torch.Tensor, split=None, capture_now: bool = False):
        """
        Encode ``rows`` with ``encoder``, replaying a graph per piece where one is available.

        Args:
            encoder (nn.Module): ASR encoder taking ``audio_signal`` and ``length``.
            rows (torch.Tensor): Window mel features shaped ``(N, D, T)``.
            lengths (torch.Tensor): Valid window frame counts shaped ``(N,)``.
            split (Optional[Callable]): ``split(rows, frames) -> sizes``; ``None`` graphs the exact row count.
            capture_now (bool): Capture new keys at their first sight.

        Returns:
            torch.Tensor: Encoded windows shaped ``(N, D_out, T_out)``.
        """
        sizes = split(rows.shape[0], rows.shape[-1]) if split is not None else [rows.shape[0]]
        outputs, start = [], 0
        for size in sizes:
            piece, piece_lengths = rows[start : start + size], lengths[start : start + size]
            n = piece.shape[0]
            start += n
            key = (size, tuple(piece.shape[1:]), piece.dtype, lengths.dtype)
            entry = self._lookup(key, lambda: self._capture(encoder, size, piece, piece_lengths), capture_now)
            if entry is None:
                outputs.append(encoder(audio_signal=piece, length=piece_lengths)[0])
                continue
            graph, static_rows, static_lengths, encoded = entry
            copy_padded_rows(static_rows, piece)
            copy_padded_rows(static_lengths, piece_lengths)
            graph.replay()
            outputs.append(encoded[:n].clone())
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def capture_batch_sizes(self, encoder, batch_sizes) -> None:
        """
        Capture every window shape seen so far at every graph batch size.

        Args:
            encoder (nn.Module): ASR encoder taking ``audio_signal`` and ``length``.
            batch_sizes (Callable): ``batch_sizes(frames) -> sizes`` for windows of ``frames`` input frames.
        """
        for (_, shape, rows_dtype, lengths_dtype), entry in list(self.graphs.items()):
            device = entry[1].device
            for size in batch_sizes(shape[-1]):
                rows = torch.zeros(size, *shape, device=device, dtype=rows_dtype)
                lengths = torch.full((size,), shape[-1], device=device, dtype=lengths_dtype)
                self(encoder, rows, lengths, capture_now=True)

    def _capture(self, encoder, size: int, piece: torch.Tensor, piece_lengths: torch.Tensor):
        """Capture ``encoder`` on static ``size``-row buffers shaped like ``piece``."""
        static_rows = self.buffers.view("asr_windows", (size, *piece.shape[1:]), piece.dtype, piece.device)
        static_rows.copy_(piece[:1].expand_as(static_rows))
        static_lengths = piece_lengths[:1].expand(size).clone()

        def run():
            encoded, _ = encoder(audio_signal=static_rows, length=static_lengths)
            return self.buffers.store("asr_encoded", encoded)

        graph, encoded = capture_cuda_graph(run, pool=self.pool)
        return graph, static_rows, static_lengths, encoded


class _StreamingStepGraphs(_GraphCache):
    """CUDA graphs of one synchronous streaming Sortformer step, keyed by every input shape (inference only).

    The state shapes change while the speaker cache fills, then stay fixed, so a few keys cover any number of steps.
    With a ``bucket`` function, the rows are padded to a graph batch size with copies of the first row, so a few
    batch sizes cover any number of rows. A replay copies the state and the window into the graph's input buffers and
    clones its outputs, so callers own every returned tensor.
    """

    _STATE_FIELDS = (
        "spkcache",
        "spkcache_lengths",
        "spkcache_preds",
        "fifo",
        "fifo_lengths",
        "fifo_preds",
        "mean_sil_emb",
        "n_sil_frames",
    )

    def __call__(
        self,
        step,
        state,
        chunk: torch.Tensor,
        chunk_length: torch.Tensor,
        extra_key: tuple,
        bucket=None,
        capture_now: bool = False,
    ):
        """
        Run ``step`` eagerly or replay its graph for this input signature.

        Args:
            step (Callable): Function mapping ``(state, chunk, chunk_length)`` to ``(state, predictions)``.
            state (Any): Sortformer streaming state.
            chunk (torch.Tensor): Window mel features, time-major, shaped ``(B, T, D)``.
            chunk_length (torch.Tensor): Valid window frame counts shaped ``(B,)``.
            extra_key (tuple): Hashable step options that are part of the graph key.
            bucket (Optional[Callable]): ``bucket(rows) -> size`` to pad the rows to, or ``None`` to run eagerly;
                without it, the exact row count is graphed.
            capture_now (bool): Capture a new key at its first sight.

        Returns:
            tuple[Any, torch.Tensor]: Updated streaming state and this window's speaker predictions.
        """
        rows = chunk.shape[0]
        size = bucket(rows) if bucket is not None else rows
        if size is None:
            return step(state, chunk, chunk_length)
        tensors = {name: getattr(state, name) for name in self._STATE_FIELDS if getattr(state, name, None) is not None}
        if size != rows:
            tensors = {name: _pad_rows(t, size) for name, t in tensors.items()}
            chunk, chunk_length = _pad_rows(chunk, size), _pad_rows(chunk_length, size)
        key = (
            extra_key,
            bool(state.spkcache_compressed),
            tuple((name, tuple(t.shape), t.dtype) for name, t in tensors.items()),
            tuple(chunk.shape),
            chunk.dtype,
            tuple(chunk_length.shape),
        )
        entry = self._lookup(key, lambda: self._capture(step, state, tensors, chunk, chunk_length), capture_now)
        if entry is None:
            return step(state, chunk[:rows], chunk_length[:rows])
        graph, static_tensors, static_chunk, static_length, out_state, out_preds = entry
        for name, t in tensors.items():
            static_tensors[name].copy_(t)
        static_chunk.copy_(chunk)
        static_length.copy_(chunk_length)
        graph.replay()
        new_state = copy.copy(out_state)
        for name, value in vars(out_state).items():
            if isinstance(value, torch.Tensor):
                setattr(new_state, name, (value[:rows] if value.dim() and value.shape[0] == size else value).clone())
        return new_state, out_preds[:rows].clone()

    def _capture(self, step, state, tensors, chunk, chunk_length):
        """
        Capture ``step`` on static copies of its inputs.

        Args:
            step (Callable): Function mapping ``(state, chunk, chunk_length)`` to ``(state, predictions)``.
            state (Any): Sortformer streaming state used as the template.
            tensors (dict[str, torch.Tensor]): State tensors to give static input buffers.
            chunk (torch.Tensor): Window mel features, time-major, shaped ``(B, T, D)``.
            chunk_length (torch.Tensor): Valid window frame counts shaped ``(B,)``.

        Returns:
            tuple: The graph, its static state tensors, chunk and length buffers, and its output state and predictions.
        """
        static_tensors = {name: t.clone() for name, t in tensors.items()}
        static_chunk = self.buffers.view("diar_chunk", tuple(chunk.shape), chunk.dtype, chunk.device).copy_(chunk)
        static_length = chunk_length.clone()

        def run():
            # The step reassigns state attributes, so each run gets its own shallow copy of the state.
            fresh = copy.copy(state)
            for name, t in static_tensors.items():
                setattr(fresh, name, t)
            out_state, preds = step(fresh, static_chunk, static_length)
            # Unchanged state tensors are the static inputs; the rest move to shared buffers.
            static = {id(t) for t in static_tensors.values()}
            for name, value in list(vars(out_state).items()):
                if isinstance(value, torch.Tensor) and id(value) not in static:
                    setattr(out_state, name, self.buffers.store(f"diar_state.{name}", value))
            return out_state, self.buffers.store("diar_preds", preds)

        graph, (out_state, out_preds) = capture_cuda_graph(run, pool=self.pool)
        return graph, static_tensors, static_chunk, static_length, out_state, out_preds


def _pad_rows(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pad ``tensor`` to ``size`` rows with copies of its first row."""
    return torch.cat([tensor, tensor[:1].expand(size - tensor.shape[0], *tensor.shape[1:])])


class _PrivateGraphBuffers(SharedGraphBuffers):
    """SharedGraphBuffers that give every graph its own buffers."""

    def view(self, slot: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)
