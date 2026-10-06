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

"""Parallel Expert Speech Encoder.

Runs a Sortformer speaker-diarization expert and an ASR Conformer encoder on the
same mel input, then fuses their outputs (LayerNorm + sinusoidal speaker-kernel +
ADD). Expects un-normalised mels; the ASR branch re-applies ``normalize_batch``
internally, and so does the diarizer branch, as ``diar_normalize_type`` says (by
default, as the diarizer's own preprocessor did). I/O matches
:class:`ConformerEncoder` (drop-in). Only self-contained PE bundles (inline
``asr_encoder_cfg`` + ``diarization_model_cfg`` in ``model_config.yaml``) are
supported.
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
import yaml
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'ParallelExpertEncoder',
    'ParallelExpertEncoderPT',
]


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype.

    Makes ``SortformerModules.init_streaming_state`` allocate its dtype-less
    speaker-cache / FIFO buffers in the diarizer's dtype, avoiding fp32/bf16 mismatch.
    """
    prev = torch.get_default_dtype()
    if dtype == prev or not dtype.is_floating_point:
        yield
        return
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@contextlib.contextmanager
def _disable_dist_feature_sync():
    """Temporarily make ``torch.distributed`` look uninitialized.

    Skips the cross-rank ``all_reduce`` in ``SortformerEncLabelModel.forward_streaming``,
    which is unnecessary and unsafe for single-recording inference (e.g. a vLLM worker).
    The original ``dist.is_initialized`` is always restored.
    """
    if not (hasattr(dist, "is_initialized") and dist.is_initialized()):
        yield
        return
    orig_is_initialized = dist.is_initialized
    dist.is_initialized = lambda: False
    try:
        yield
    finally:
        dist.is_initialized = orig_is_initialized


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations.

    ``from_config_dict`` mutates its input in place, so sub-target builders get a copy.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


# Distinguishes "key absent -> reference default of per_feature" from "explicitly disabled".
# `None` and the checkpoint-contract string 'NA' both mean *no* normalization; a plain
# `or 'per_feature'` fallback would silently re-enable it (encoders whose preprocessor is
# already `normalize: NA` must not be normalized twice).
_NORMALIZE_UNSET = object()

# `asr_normalize_type: auto` asks `from_checkpoints` to replay the ASR checkpoint's own preprocessor
# `normalize` instead of the `per_feature` default. Only that route has the ASR preprocessor config,
# so the constructor refuses the value. These are the checkpoint values it can follow.
_NORMALIZE_AUTO = 'auto'
_NORMALIZE_AUTO_FOLLOWED = ('per_feature', 'all_features')

# The Sortformer branch is frozen during SpeechLM training, so its streaming knobs should stay
# exactly as its checkpoint was trained with unless a caller deliberately overrides them. The
# previous defaults (fifo 40 / update 300 / cache 188) came from the placeholder bundle's model
# card, which pairs a DIFFERENT Sortformer -- applying them to any other checkpoint silently
# changes a frozen branch's behaviour (e.g. sortformer-8spk ships fifo 0 / chunk 264 / cache 264).
_DIAR_UNSET = object()

# On the offline path both branches encode the same mels with the same subsampling factor, so
# their frame counts differ by convolution rounding at most. A larger gap means the diarizer's
# predictions are on another time grid (e.g. a high-resolution diarizer's unpooled 10 ms output),
# which `_align_diar_frames` would otherwise truncate or stretch without a word.
_MAX_DIAR_ASR_FRAME_MISMATCH = 2

# Whether a stream of `StreamingParallelExpertEncoder.cache_aware_stream_step` steps its diarizer.
_DIAR_STREAM_STEPPED = "stepped"
_DIAR_STREAM_SKIPPED = "skipped"


# --- Speaker-feature fusion contract -------------------------------------------------------
# Ported from the SALM-side encoder so a bundle states, rather than implies, how its speaker
# activity is consumed. Historical canonical bundles were used with BOTH continuous and
# thresholded activity, so an unversioned bundle carrying neither a mode nor a threshold is
# ambiguous and is rejected instead of guessed at.
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
# Override keys that SALM-side recipes set but that nothing in this encoder consumes. Accepting them
# would drop the value without a word, so each is refused with the reason and what to use instead.
_UNCONSUMED_BUNDLE_OVERRIDE_KEYS = {
    "chunk_size_seconds": (
        "this encoder does not chunk its input by it; the SpeechLM mounts replace it with "
        "model.encoder_chunk_size_seconds, so set that key instead"
    ),
    "frame_shift_seconds": (
        "this encoder converts no durations to frames; its frame rate is fixed by the mel input and "
        "the ASR branch's subsampling factor"
    ),
}
# `diar_normalize_type`: how the diarizer branch normalises the un-normalised mels it receives. They bypass the
# diarizer's own preprocessor, so unset, the encoder replays that preprocessor's `normalize` and keeps the input the
# diarizer was trained on; an explicit value overrides it. The streaming encoder's step normalises each chunk on its
# own, as it does the ASR input. `None` and 'NA' mean no normalisation; these are the normalisations the branch can
# apply.
_DIAR_NORMALIZE_TYPES = ("per_feature", "all_features")


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
    unconsumed = sorted(set(config_overrides) & set(_UNCONSUMED_BUNDLE_OVERRIDE_KEYS))
    if unconsumed:
        reasons = "; ".join(f"{key!r}: {_UNCONSUMED_BUNDLE_OVERRIDE_KEYS[key]}" for key in unconsumed)
        raise ValueError(f"ParallelExpertEncoder config_overrides keys {unconsumed} are not supported. {reasons}.")
    unknown = sorted(set(config_overrides) - _BUNDLE_CONFIG_OVERRIDE_KEYS)
    if unknown:
        supported = ", ".join(sorted(_BUNDLE_CONFIG_OVERRIDE_KEYS))
        raise ValueError(
            f"Unsupported ParallelExpertEncoder config_overrides keys {unknown}; supported keys: {supported}."
        )
    for key in sorted(config_overrides):
        logging.info("[ParallelExpertEncoder] Applying bundle config override %s=%r.", key, config_overrides[key])
    return OmegaConf.merge(merged, OmegaConf.create(dict(config_overrides)))


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
    """ModelPT shell so a :class:`ParallelExpertEncoder` can be saved/restored as a
    ``.nemo`` archive (inline ``asr_encoder_cfg`` + ``diarization_model_cfg``).
    """

    # Subclasses override this to mount a different encoder flavour (e.g. the streaming variant)
    # while reusing the whole save/restore path.
    _ENCODER_CLS = None

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        encoder_cls = type(self)._ENCODER_CLS or ParallelExpertEncoder
        try:
            speaker_feature_mode, speaker_activity_threshold = _resolve_speaker_feature_contract(self._cfg)
        except ValueError as error:
            # Upstream fails closed here. This fork keeps loading such a bundle at its historical
            # default, because bundles predating the contract were all written by this branch with
            # thresholding at 0.5 -- refusing them outright would strand working checkpoints. The
            # warning is the point: the bundle is ambiguous and should be re-exported with an
            # explicit contract.
            if "no speaker-feature contract" not in str(error):
                raise
            logging.warning(
                "[ParallelExpertEncoder] %s Falling back to speaker_feature_mode='thresholded' with "
                "speaker_activity_threshold=0.5, this branch's historical default.",
                error,
            )
            speaker_feature_mode, speaker_activity_threshold = _SPEAKER_FEATURE_MODE_THRESHOLD, 0.5
        self.encoder = encoder_cls(
            asr_encoder_cfg=self._cfg.get('asr_encoder_cfg', None),
            diarization_model_cfg=self._cfg.get('diarization_model_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', _NORMALIZE_UNSET),
            diar_normalize_type=self._cfg.get('diar_normalize_type', _NORMALIZE_UNSET),
            freeze_diar=self._cfg.get('freeze_diar', True),
            freeze_asr=self._cfg.get('freeze_asr', False),
            online_inference_length=self._cfg.get('online_inference_length', 500),
            chunk_left_context=self._cfg.get('chunk_left_context', 50),
            chunk_right_context=self._cfg.get('chunk_right_context', 50),
            diar_fifo_len=self._cfg.get('diar_fifo_len', _DIAR_UNSET),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', _DIAR_UNSET),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', _DIAR_UNSET),
            diar_chunk_len=self._cfg.get('diar_chunk_len', _DIAR_UNSET),
            speaker_activity_threshold=speaker_activity_threshold,
            speaker_feature_mode=speaker_feature_mode,
            chunk_size_seconds=self._cfg.get('chunk_size_seconds', None),
            spk_kernel_scale=self._cfg.get('spk_kernel_scale', 1.0),
            spk_kernel_row_stride=self._cfg.get('spk_kernel_row_stride', 1),
            spk_kernel_calibrate=self._cfg.get('spk_kernel_calibrate', False),
            speaker_row_offset=self._cfg.get('speaker_row_offset', 0),
            missing_rttm_target=self._cfg.get('missing_rttm_target', -1.0),
            sync_max_audio_length=self._cfg.get('sync_max_audio_length', None),
        )
        # Architecture-only snapshot, so an exported SpeechLM checkpoint can rebuild this encoder
        # without the original bundle. Stamped with the RESOLVED contract, not the raw config, so a
        # re-export is never ambiguous even when the source bundle was.
        self.encoder._bundle_config = _clone_config(self._cfg)
        self.encoder._bundle_config.speaker_feature_config_version = _SPEAKER_FEATURE_CONFIG_VERSION
        self.encoder._bundle_config.speaker_feature_mode = self.encoder.speaker_feature_mode
        self.encoder._bundle_config.speaker_activity_threshold = self.encoder.speaker_activity_threshold
        self.encoder._bundle_config.chunk_size_seconds = self.encoder.chunk_size_seconds
        self.encoder._bundle_config.diar_normalize_type = _persisted_diar_normalize_type(self.encoder)

    @staticmethod
    def _validate_bundle_schema(cfg: DictConfig) -> None:
        """Require the self-contained ParallelExpertEncoder bundle schema."""
        missing = [key for key in ("asr_encoder_cfg", "diarization_model_cfg") if cfg.get(key, None) in (None, {}, "")]
        if missing:
            raise ValueError(f"ParallelExpertEncoder bundle is missing required config sections {missing}.")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        pass

    @staticmethod
    def is_pe_nemo(nemo_path: str) -> bool:
        """Detect whether a ``.nemo`` archive is a :class:`ParallelExpertEncoderPT` bundle.

        Reads only ``model_config.yaml`` and checks its ``target:``.

        Args:
            nemo_path (str): Path to a ``.nemo`` archive.

        Returns:
            ``True`` if ``target`` ends with ``ParallelExpertEncoderPT``, else ``False``.
        """
        if not (isinstance(nemo_path, str) and nemo_path.endswith('.nemo') and os.path.isfile(nemo_path)):
            return False
        try:
            with tarfile.open(nemo_path, mode='r') as tf:
                for member in tf.getmembers():
                    if os.path.basename(member.name) == 'model_config.yaml':
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            return False
                        cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                        return str(cfg.get('target', '')).endswith('ParallelExpertEncoderPT')
        except (tarfile.TarError, OSError) as exc:
            logging.warning("[ParallelExpertEncoder] Could not inspect %s: %s", nemo_path, exc)
            return False
        return False

    @classmethod
    def load_from_nemo(
        cls,
        model_path_or_name: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> ParallelExpertEncoder:
        """Load a self-contained PE bundle and return its inner encoder.

        Follows the standard NeMo :class:`~nemo.core.classes.common.Model`
        convention for resolving a checkpoint reference:

        * a local ``.nemo`` file is restored with :meth:`ModelPT.restore_from`;
        * otherwise ``model_path_or_name`` is treated as a pretrained model
          identifier -- a HuggingFace Hub repo id (``{repo}/{name}``) or an NGC
          alias -- and resolved with :meth:`Model.from_pretrained`, which
          downloads/caches the ``.nemo`` (honouring the HuggingFace cache and
          ``HF_HUB_OFFLINE``, so a prefetched cache works on offline nodes).

        This mirrors ``speechlm2.parts.pretrained.load_pretrained_nemo`` so PE
        bundles load uniformly from local files or model cards.

        Args:
            model_path_or_name (str): Local ``.nemo`` path or pretrained model id.
            map_location (str | torch.device): Device to map weights onto.
            strict (bool): Enforce exact state-dict match.
            config_overrides (Mapping, optional): Runtime-semantic bundle fields to override --
                deliberately a small allow-list (see ``_BUNDLE_CONFIG_OVERRIDE_KEYS``) so a recipe
                can resolve a legacy bundle's speaker-feature ambiguity without silently swapping
                the saved architecture. Local ``.nemo`` paths only. Each applied override is logged.
                ``chunk_size_seconds`` and ``frame_shift_seconds`` are refused because nothing here
                consumes them. ``diar_normalize_type`` overrides the diarizer's own normalization on both
                encoder classes, as the encoder argument of that name does.

        Returns:
            The restored :class:`ParallelExpertEncoder`.
        """
        if (
            isinstance(model_path_or_name, str)
            and model_path_or_name.endswith('.nemo')
            and os.path.isfile(model_path_or_name)
        ):
            # Read and merge before constructing: `restore_from` builds the shell from the archive's
            # own config, which leaves no seam to apply overrides through.
            cfg, state = _read_bundle_members(model_path_or_name)
            if not str(cfg.get('target', '')).endswith('ParallelExpertEncoderPT'):
                raise ValueError(f"{model_path_or_name!r} is not a ParallelExpertEncoderPT .nemo bundle.")
            cfg = _merge_bundle_config_overrides(cfg, config_overrides)
            cls._validate_bundle_schema(cfg)
            shell = cls(cfg=cfg, trainer=None)
            prefix = 'encoder.'
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

        # A pretrained id resolves through the HF/NGC cache, where there is no local archive to
        # rewrite. Refuse rather than accept overrides and quietly drop them.
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
        cfg: Union[DictConfig, Mapping[str, Any]],
        *,
        map_location: Union[str, torch.device] = 'cpu',
    ) -> ParallelExpertEncoder:
        """Construct the encoder architecture from an embedded bundle config, without its weights.

        An HF export of a SpeechLM model stores the mounted bundle's ``model_config.yaml``, with its
        runtime values, as ``pe_encoder_config``, and the encoder tensors in the root state dict,
        which the caller loads after construction. The encoder class follows the loader class
        (:class:`StreamingParallelExpertEncoderPT` builds a :class:`StreamingParallelExpertEncoder`),
        not the config's ``target``, as in :meth:`load_from_nemo`.

        Args:
            cfg: The bundle config, e.g. ``pe_encoder_config`` of an exported model config.
            map_location: Device to move the constructed encoder to.

        Returns:
            The constructed encoder, with freshly initialised weights.
        """
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg)
        cls._validate_bundle_schema(cfg)
        shell = cls(cfg=cfg, trainer=None)
        return shell.encoder.to(map_location)

    @classmethod
    def save_to_nemo(
        cls,
        encoder: ParallelExpertEncoder,
        output_nemo_path: str,
        *,
        template_bundle_path: str,
    ) -> None:
        """Save ``encoder`` as a self-contained PE ``.nemo``, reusing ``model_config.yaml``
        from ``template_bundle_path``.

        The template must describe the same architecture (``d_model``, ``n_spk``);
        mismatches raise :class:`ValueError` fail-fast. The saved config is the template's, except
        ``diar_normalize_type``, which is set to the encoder's effective value (``'NA'`` for none), so a
        reload normalises the diarizer input as ``encoder`` does.

        Args:
            encoder (ParallelExpertEncoder): The encoder whose weights are persisted.
            output_nemo_path (str): Destination ``.nemo`` path.
            template_bundle_path (str): Existing PE ``.nemo`` whose ``model_config.yaml`` is reused.
        """
        if not isinstance(encoder, ParallelExpertEncoder):
            raise TypeError(f"save_to_nemo expects a ParallelExpertEncoder, " f"got {type(encoder).__name__}")
        if not os.path.isfile(template_bundle_path):
            raise FileNotFoundError(f"template_bundle_path does not exist: {template_bundle_path}")

        template_cfg: Optional[DictConfig] = None
        with tarfile.open(template_bundle_path, mode='r') as tf:
            for member in tf.getmembers():
                if os.path.basename(member.name) == 'model_config.yaml':
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        template_cfg = OmegaConf.create(fobj.read().decode('utf-8'))
                    break
        if template_cfg is None:
            raise RuntimeError(f"Could not read 'model_config.yaml' from template bundle: {template_bundle_path}")

        tmpl_asr = template_cfg.get('asr_encoder_cfg', None)
        tmpl_diar = template_cfg.get('diarization_model_cfg', None)
        if tmpl_asr in (None, {}, '') or tmpl_diar in (None, {}, ''):
            raise ValueError(
                f"Template bundle {template_bundle_path} is not self-contained "
                "(asr_encoder_cfg / diarization_model_cfg missing); it cannot be "
                "used as a save template."
            )

        tmpl_d_model = int(tmpl_asr.get('d_model', -1))
        tmpl_n_spk = int(tmpl_diar.get('sortformer_modules', {}).get('num_spks', -1))
        enc_d_model = int(encoder.d_model)
        enc_n_spk = int(encoder.n_spk)
        if tmpl_d_model != enc_d_model:
            raise ValueError(
                f"Template asr_encoder_cfg.d_model={tmpl_d_model} does not match "
                f"encoder.d_model={enc_d_model}; the saved bundle would fail "
                "strict reload."
            )
        if tmpl_n_spk != enc_n_spk:
            raise ValueError(
                f"Template diarization_model_cfg.sortformer_modules.num_spks="
                f"{tmpl_n_spk} does not match encoder.n_spk={enc_n_spk}; the "
                "saved bundle would fail strict reload."
            )

        # The saved config is the template with one change: the encoder's effective `diar_normalize_type`, which the
        # template need not state or may state otherwise. Set it first, so the shell below resolves the value being
        # saved; the template's own may not resolve (unset, over a diarizer `normalize` the encoder had to override).
        template_cfg.diar_normalize_type = _persisted_diar_normalize_type(encoder)
        # Fresh PT shell from that cfg to reuse NeMo's save_to; swap in encoder.
        shell = cls(cfg=template_cfg, trainer=None)
        shell.encoder = encoder
        # Pin `_cfg` to that cfg so save_to round-trips it.
        shell._cfg = template_cfg

        shell.save_to(output_nemo_path)
        logging.info(
            "[ParallelExpertEncoder] Saved PE bundle to %s using template config from %s",
            output_nemo_path,
            template_bundle_path,
        )


@experimental
class ParallelExpertEncoder(nn.Module):
    """Sortformer-diarizer + ASR Conformer encoder; I/O identical to :class:`ConformerEncoder`.

    Reconstructed from inline configs in the PE bundle's ``model_config.yaml``.

    Args:
        asr_encoder_cfg (DictConfig): Inline config for the ASR-side cache-aware encoder
            (:class:`ConformerEncoder`, :class:`StreamingTransformerEncoder`, ...).
        diarization_model_cfg (DictConfig): Inline config for the :class:`SortformerEncLabelModel`.
        asr_normalize_type (str, optional): Normalization replayed on the ASR branch. Defaults to
            ``per_feature`` when unset; pass ``None`` or ``'NA'`` to disable it entirely (required
            for ASR branches whose own preprocessor already uses ``normalize: NA``). ``'auto'`` is
            resolved by :meth:`from_checkpoints` only and is rejected here.
        diar_normalize_type (str, optional): Normalization applied to the diarizer branch's input.
            Unset (default) keeps the diarizer's own: its config's ``preprocessor.normalize``, which
            its preprocessor applied in training. ``'per_feature'`` or ``'all_features'`` overrides
            it, ``None`` or ``'NA'`` disables it, and any other value raises. The same holds for
            :class:`StreamingParallelExpertEncoder`. The offline forward and online inference normalise
            each whole utterance; the streaming class's ``cache_aware_stream_step`` normalises each
            chunk on its own, as it does the ASR input.
        freeze_diar (bool): Freeze the Sortformer parameters. Defaults to ``True``. A frozen branch
            always runs without autograd. A trainable one runs in the caller's grad mode, so it builds
            no graph under ``torch.no_grad()`` or ``torch.inference_mode()``.
        freeze_asr (bool): Freeze the wrapped ASR ConformerEncoder. Defaults to ``False``. The ASR
            branch's grad mode follows the same rule.
        online_inference_length (int): Online-inference window in encoder output frames
            (default ``500`` ~= 40s); ``<= 0`` disables it.
        chunk_left_context (int): Left context (output frames) per online window, shared by
            both branches. Default ``50``.
        chunk_right_context (int): Right context (output frames) per online window, shared by
            both branches. Default ``50``.
        diar_fifo_len (int, optional): Override the Sortformer's streaming ``fifo_len``.
            Unset (default) keeps the diarizer checkpoint's own value.
        diar_spkcache_update_period (int, optional): Override ``spkcache_update_period``; unset keeps
            the checkpoint's value.
        diar_spkcache_len (int, optional): Override ``spkcache_len``; unset keeps the checkpoint's value.
        diar_chunk_len (int, optional): Override the Sortformer's streaming ``chunk_len``; unset keeps
            the checkpoint's value. Distinct from ``online_inference_length``, which is PE's *ASR*
            long-form window.
        missing_rttm_target (float): Rows whose targets are all ``<=`` this value had no RTTM and
            are filled from the diarizer. Default ``-1.0``.
        sync_max_audio_length (bool, optional): Set ``sync_max_audio_length`` on every encoder in
            both branches (the ASR branch and the diarizer's encoder). ``False`` stops them from
            all-reducing the longest input length on every forward, which deadlocks when ranks run
            a data-dependent number of forwards. Unset (default) keeps each branch's own value.
    """

    def __init__(
        self,
        asr_encoder_cfg: DictConfig,
        diarization_model_cfg: DictConfig,
        asr_normalize_type: Optional[str] = _NORMALIZE_UNSET,
        diar_normalize_type: Optional[str] = _NORMALIZE_UNSET,
        freeze_diar: bool = True,
        freeze_asr: bool = False,
        online_inference_length: int = 500,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: Optional[int] = _DIAR_UNSET,
        diar_spkcache_update_period: Optional[int] = _DIAR_UNSET,
        diar_spkcache_len: Optional[int] = _DIAR_UNSET,
        diar_chunk_len: Optional[int] = _DIAR_UNSET,
        speaker_activity_threshold: Optional[float] = 0.5,
        speaker_feature_mode: Optional[str] = None,
        chunk_size_seconds: Optional[float] = None,
        spk_kernel_scale: float = 1.0,
        spk_kernel_row_stride: int = 1,
        spk_kernel_calibrate: bool = False,
        speaker_row_offset: int = 0,
        missing_rttm_target: float = -1.0,
        att_context_size: Optional[list] = None,
        sync_max_audio_length: Optional[bool] = None,
    ):
        super().__init__()

        # Lazy import: SortformerEncLabelModel imports from asr.modules (circular).
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        if asr_encoder_cfg is None or diarization_model_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires both `asr_encoder_cfg` and "
                "`diarization_model_cfg`; self-contained PE bundles supply "
                "these inline in their model_config.yaml."
            )

        # `from_config_dict` dispatches on `_target_`, so this builds whatever the config names.
        self.asr_encoder = ConformerEncoder.from_config_dict(_clone_config(asr_encoder_cfg))
        _require_asr_encoder_interface(self.asr_encoder)
        if asr_normalize_type is _NORMALIZE_UNSET:
            asr_normalize_type = 'per_feature'
        elif isinstance(asr_normalize_type, str) and asr_normalize_type == _NORMALIZE_AUTO:
            raise ValueError(
                "asr_normalize_type='auto' follows the ASR checkpoint's preprocessor, so only "
                "ParallelExpertEncoder.from_checkpoints (model.parallel_expert_encoder) resolves it. "
                "Set an explicit value here: 'per_feature', 'all_features', or null/'NA' for none."
            )
        self.asr_normalize_type = None if asr_normalize_type in (None, 'NA') else asr_normalize_type
        self._feat_in = self.asr_encoder._feat_in

        diarization_model_cfg = _clone_config(diarization_model_cfg)
        diarization_model_cfg.output_subsampling_factor = self.asr_encoder.subsampling_factor
        self.diarization_model = SortformerEncLabelModel.from_config_dict(diarization_model_cfg)
        if self.diarization_model.output_subsampling_factor != self.asr_encoder.subsampling_factor:
            raise ValueError(
                "ParallelExpertEncoder requires the diarization output subsampling factor "
                f"({self.diarization_model.output_subsampling_factor}) to equal the ASR encoder subsampling factor "
                f"({self.asr_encoder.subsampling_factor})."
            )
        self.diar_normalize_type = _resolve_diar_normalize_type(
            diar_normalize_type, _diarizer_own_normalize(diarization_model_cfg, self.diarization_model)
        )

        # None = each branch keeps its own flag. Recorded so the SpeechLM mounts can tell an explicit
        # PE-level setting, which wins, from an unset one.
        self.explicit_sync_max_audio_length = None if sync_max_audio_length is None else bool(sync_max_audio_length)
        if self.explicit_sync_max_audio_length is not None:
            self.sync_max_audio_length = self.explicit_sync_max_audio_length

        self.freeze_diar = freeze_diar
        self.freeze_asr = freeze_asr

        # Long-form / online inference configuration.
        self.online_inference_length = int(online_inference_length)
        # None = fall back to the length/training heuristic in `forward`.
        self.online_inference_enabled: Optional[bool] = None
        # Overlap-and-trim context (output frames) shared by both branches.
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        # Online-inference window + context in input mel frames (constant per session).
        self.chunk_feat_len = self.online_inference_length * self.asr_encoder.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.asr_encoder.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.asr_encoder.subsampling_factor
        # Only knobs the caller set explicitly are pushed onto the frozen diarizer; the rest keep
        # the values its checkpoint was trained with. See `_DIAR_UNSET`.
        self._diar_streaming_overrides = {
            'fifo_len': diar_fifo_len,
            'spkcache_update_period': diar_spkcache_update_period,
            'spkcache_len': diar_spkcache_len,
            'chunk_len': diar_chunk_len,
        }
        # Binarize speaker activity before fusion. The bundle records
        # `speaker_feature_mode: thresholded` / `speaker_activity_threshold: 0.5`, i.e. the
        # `diar_kernel` weights were FIT against binarised input -- and the reference fusion
        # thresholds unconditionally. Applying it here (rather than at the call site) means oracle
        # RTTM (already {0,1}, so a no-op) and Sortformer sigmoids reach the kernel as the same
        # distribution, so training on oracle and inferring on predictions do not diverge.
        # `None` opts into the soft-target experiment.
        # Name the contract that was previously implicit: a threshold means "thresholded", its
        # absence means "continuous". Same behaviour, but a bundle now STATES which it wants
        # rather than leaving a reader to infer it from whether a field happens to be set.
        self.speaker_feature_mode, self.speaker_activity_threshold = _normalize_speaker_feature_contract(
            speaker_feature_mode, speaker_activity_threshold
        )
        # Consumed by the packed-sequence branches only, which this fork does not carry yet; stored
        # and validated so a bundle round-trips it and the SALM-side contract is honoured.
        self.chunk_size_seconds = self._validate_chunk_size("chunk_size_seconds", chunk_size_seconds)
        self.spk_kernel_scale = float(spk_kernel_scale)

        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = self.asr_encoder.d_model

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        # Rows whose targets are entirely <= this sentinel had no RTTM; they fall back to the
        # embedded diarizer instead of being fused with meaningless targets. Without it, the
        # dataset's `no_rttm_to_ones` placeholder becomes "all speakers active at every frame"
        # after thresholding, and the kernel is trained on confidently-wrong supervision.
        self.missing_rttm_target = float(missing_rttm_target)
        self.spk_kernel_row_stride = int(spk_kernel_row_stride)
        self.spk_kernel_calibrate = bool(spk_kernel_calibrate)
        self.speaker_row_offset = int(speaker_row_offset)
        self.register_buffer(
            "diar_kernel",
            self._build_tag_kernel(
                self.n_spk,
                self.speaker_row_offset,
                self.asr_d_model,
                stride=self.spk_kernel_row_stride,
                calibrate=self.spk_kernel_calibrate,
            ),
            persistent=False,
        )

        if any(v is not _DIAR_UNSET for v in self._diar_streaming_overrides.values()):
            # Apply an EXPLICIT override once, here, so it does not depend on whether streaming is
            # ever set up. With nothing set (the default) the diarizer keeps its checkpoint's values
            # and this is skipped entirely -- constructing PE must not retune a frozen branch.
            self._apply_diar_streaming_overrides()

        if att_context_size is not None:
            # Applied AFTER the branch is built, and via set_att_context_size, so the multi-context
            # choice set is collapsed too -- see that method for why assignment alone is not enough.
            self.set_att_context_size(att_context_size)

        self.apply_internal_freeze()

    @classmethod
    def from_checkpoints(
        cls,
        asr_model: str,
        diar_model: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        **kwargs,
    ) -> "ParallelExpertEncoder":
        """Assemble a PE encoder from a standalone ASR ``.nemo`` and a standalone Sortformer ``.nemo``.

        Both the architecture *and* the weights of each branch come from its own checkpoint, so ASR
        and diarizer can be swapped independently without pre-building a fused bundle. Equivalent
        to :meth:`load_from_nemo` on a bundle assembled from the same two files, except for the
        fusion LayerNorms (see below).

        Weight mapping:

        * ASR ``encoder.*`` -> ``asr_encoder.*``. The ASR checkpoint's ``decoder.*`` / ``joint.*``
          (RNNT head) and ``preprocessor.*`` have no destination and are dropped -- PE consumes mels
          from the caller and owns no ASR-side preprocessor.
        * Diarizer: the **entire** state dict, verbatim, under ``diarization_model.*``. It is a
          whole ``SortformerEncLabelModel``, so its own ``preprocessor.*`` buffers and
          ``sortformer_modules.*`` parameters are all real destinations.

        ``asr_norm`` / ``diar_norm`` (4 tensors) exist in **neither** source and stay at PyTorch
        init. Note that ``LayerNorm(weight=1, bias=0)`` is *not* identity -- it standardizes over
        the feature dim -- so a freshly assembled encoder does not reproduce the standalone ASR
        encoder's activations. Those two norms are part of what the fusion has to learn.

        ASR normalization: the ASR branch receives un-normalized mels and replays
        ``asr_normalize_type`` itself. An absent ``asr_normalize_type`` keeps the historical
        ``per_feature`` and logs a warning, because it ignores the ASR checkpoint's own
        ``preprocessor.normalize`` (a checkpoint trained with ``normalize: NA`` is then fed
        re-normalized mels). ``asr_normalize_type='auto'`` follows that checkpoint value instead:
        ``NA``/``null`` disables the replay, and ``per_feature``/``all_features`` are replayed as
        given. Any other checkpoint value, or none, raises. The resolved value is the encoder's
        ``asr_normalize_type`` attribute.

        Diarizer normalization: an absent ``diar_normalize_type`` keeps the diarizer checkpoint's own
        ``preprocessor.normalize``, and an explicit one overrides it, on both encoder classes (see
        ``diar_normalize_type`` in the class docstring).

        Each source may be a local ``.nemo`` path or a pretrained model id (a HuggingFace Hub
        ``{repo}/{name}`` or an NGC alias), resolved the same way :meth:`load_from_nemo` resolves
        bundles. Local files are read straight out of the archive; ids go through the model class's
        ``from_pretrained``, which downloads and caches.

        Args:
            asr_model (str): Local ``.nemo`` path or pretrained id for the ASR branch.
            diar_model (str): Local ``.nemo`` path or pretrained id for the speaker branch.
            map_location: Device to map the loaded tensors onto.
            **kwargs: Forwarded to ``__init__`` (``asr_normalize_type``, ``freeze_diar``, ...), except
                that ``asr_normalize_type='auto'`` is first resolved from the ASR checkpoint.

        Returns:
            A ``cls`` instance with both branches populated from their checkpoints.
        """
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel

        asr_cfg, asr_state = _resolve_branch_source(asr_model, ASRModel, map_location)
        diar_cfg, diar_state = _resolve_branch_source(diar_model, SortformerEncLabelModel, map_location)
        kwargs = dict(kwargs)
        if 'asr_normalize_type' not in kwargs:
            logging.warning(
                "ParallelExpertEncoder.from_checkpoints: asr_normalize_type is not set, so the ASR branch "
                "re-normalizes its mels with 'per_feature' (the historical default). The ASR checkpoint %s "
                "was trained with preprocessor.normalize=%r. Set asr_normalize_type explicitly, or to 'auto' "
                "to follow the checkpoint.",
                asr_model,
                _checkpoint_normalize(asr_cfg, default='<not set>'),
            )
        elif isinstance(kwargs['asr_normalize_type'], str) and kwargs['asr_normalize_type'] == _NORMALIZE_AUTO:
            kwargs['asr_normalize_type'] = _asr_normalize_type_from_checkpoint(asr_cfg, asr_model)
            logging.info(
                "ParallelExpertEncoder.from_checkpoints: asr_normalize_type='auto' resolved to %r from "
                "the preprocessor.normalize of %s.",
                kwargs['asr_normalize_type'],
                asr_model,
            )
        encoder = cls(
            asr_encoder_cfg=OmegaConf.create(asr_cfg['encoder']),
            diarization_model_cfg=OmegaConf.create(diar_cfg),
            **kwargs,
        )

        remapped = {}
        for key, value in asr_state.items():
            if key.startswith('encoder.'):
                remapped['asr_encoder.' + key[len('encoder.') :]] = value
        for key, value in diar_state.items():
            remapped['diarization_model.' + key] = value

        missing, unexpected = encoder.load_state_dict(remapped, strict=False)
        # Only the fusion norms may be missing; anything else means the checkpoints do not match
        # the configs they were built from, and a silent partial load would train from noise.
        unsourced = {'asr_norm.weight', 'asr_norm.bias', 'diar_norm.weight', 'diar_norm.bias'}
        unexplained = set(missing) - unsourced
        if unexplained or unexpected:
            raise RuntimeError(
                f"ParallelExpertEncoder.from_checkpoints: state-dict mismatch.\n"
                f"  missing (beyond the fusion norms): {sorted(unexplained)[:10]}\n"
                f"  unexpected: {sorted(unexpected)[:10]}\n"
                f"Check that {asr_model} and {diar_model} match the architectures they declare."
            )
        n_asr = sum(1 for k in remapped if k.startswith('asr_encoder.'))
        n_diar = sum(1 for k in remapped if k.startswith('diarization_model.'))
        logging.info(
            "ParallelExpertEncoder.from_checkpoints: loaded %d ASR + %d diarizer tensors "
            "(%d ASR tensors in the checkpoint had no destination: RNNT head / preprocessor); "
            "asr_norm + diar_norm (4 tensors) start from init and must be trained.",
            n_asr,
            n_diar,
            len(asr_state) - n_asr,
        )
        return encoder

    @property
    def att_context_size(self):
        """The ASR branch's attention context — PE has no attention of its own.

        Forwarded rather than stored: speechlm2's ``_set_encoder_att_context`` assigns
        ``encoder.att_context_size = [left, right]`` per batch to match the chunk size, and on a
        plain ``nn.Module`` that would silently create a dead attribute the branch never reads,
        leaving the look-ahead at whatever the checkpoint was built with.
        """
        return self.asr_encoder.att_context_size

    @att_context_size.setter
    def att_context_size(self, value) -> None:
        self.set_att_context_size(value)

    def set_att_context_size(self, att_context_size) -> None:
        """Pin the ASR branch to exactly ``att_context_size``.

        Assigning ``att_context_size`` alone is **not** enough to pin the look-ahead. A
        multi-context checkpoint (this repo's streaming Conformer ships
        ``[[70, 13], [70, 6], [70, 1], [70, 0]]``) leaves ``att_context_size_all`` longer than one,
        and ``ConformerEncoder.forward`` then *randomly samples* a context on every training step
        (``conformer_encoder.py:663``), silently overriding the value that was set. Measured: six
        forwards of one input under ``.train()`` gave six different outputs.

        For a plain encoder speechlm2 avoids this by writing ``att_context_size`` into the encoder
        *config* before construction, which collapses the list. PE builds its branch from the
        checkpoint's own config, so it has to collapse the list here instead — which also keeps
        per-batch updates from ``_set_encoder_att_context`` deterministic.

        Use :meth:`ConformerEncoder.set_default_att_context_size` on ``self.asr_encoder`` directly
        if you want the validated, list-preserving behaviour instead.
        """
        att_context_size = list(att_context_size)
        self.asr_encoder.att_context_size = att_context_size
        # Collapse the choice set so training cannot sample around the value just set.
        self.asr_encoder.att_context_size_all = [att_context_size]
        if getattr(self.asr_encoder, 'att_context_probs', None) is not None:
            self.asr_encoder.att_context_probs = [1.0]

    @property
    def sync_max_audio_length(self) -> bool:
        """Whether any encoder in either branch all-reduces the longest input length per forward.

        PE has no flag of its own: the collective lives in each branch's encoder
        (``ConformerEncoder.update_max_seq_length``), so this reports, and sets, theirs.
        """
        return any(bool(module.sync_max_audio_length) for module in self._length_syncing_modules())

    @sync_max_audio_length.setter
    def sync_max_audio_length(self, value: bool) -> None:
        for module in self._length_syncing_modules():
            module.sync_max_audio_length = bool(value)

    def _length_syncing_modules(self) -> List[nn.Module]:
        """Every module in the ASR and diarizer branches that carries ``sync_max_audio_length``."""
        return [
            module
            for branch in (self.asr_encoder, self.diarization_model)
            for module in branch.modules()
            if hasattr(module, 'sync_max_audio_length')
        ]

    def _apply_diar_streaming_overrides(self) -> None:
        """Push only explicitly-configured streaming knobs onto the frozen Sortformer.

        Anything left unset keeps the value from the diarizer's own checkpoint. Previously
        ``chunk_len`` was taken from ``online_inference_length`` -- PE's *ASR* long-form window,
        which is a different quantity -- and the other three from constructor defaults tuned for a
        different Sortformer, so mounting any other checkpoint silently retuned a frozen branch.
        """
        sm = self.diarization_model.sortformer_modules
        for name, value in self._diar_streaming_overrides.items():
            if value is not _DIAR_UNSET and value is not None:
                setattr(sm, name, int(value))
        self.diarization_model._check_streaming_parameters()

    def _normalize_diar_input(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Apply ``diar_normalize_type`` to the diarizer branch's mels, with the statistics of each row given.

        The offline forward and online inference pass whole utterances; the streaming step of
        :class:`StreamingParallelExpertEncoder` passes one chunk.

        Args:
            audio_signal (Tensor): Un-normalised mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.

        Returns:
            The normalized mels, or ``audio_signal`` itself when ``diar_normalize_type`` is ``None``.
        """
        if not self.diar_normalize_type:
            return audio_signal
        normalized, _, _ = normalize_batch(
            audio_signal, length.to(device=audio_signal.device), normalize_type=self.diar_normalize_type
        )
        return normalized

    def apply_internal_freeze(self) -> None:
        """(Re-)apply this encoder's own ``freeze_diar`` / ``freeze_asr`` policy.

        Callers that freeze or unfreeze the perception encoder wholesale (e.g. speechlm2's
        ``freeze_speech_encoder``) operate on the module tree and cannot know that a *branch* of it
        is meant to stay frozen -- an outer ``unfreeze_module(perception.encoder)`` would put all
        357 Sortformer tensors back in the optimizer. They should call this afterwards to hand
        ownership of the internal split back to the encoder.
        """
        if self.freeze_diar:
            self.diarization_model.eval()
            for param in self.diarization_model.parameters():
                param.requires_grad = False
        if self.freeze_asr:
            self.asr_encoder.eval()
            for param in self.asr_encoder.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "ParallelExpertEncoder":
        """Set training mode, but keep frozen sub-branches in eval.

        The parent ``model.train()`` recurses into every sub-module, which would re-enable
        dropout / BatchNorm stat updates in a frozen branch. This re-asserts ``eval()`` on
        the frozen Sortformer (and ASR encoder) so their outputs stay deterministic.

        Args:
            mode (bool): Whether to set training mode (``True``) or eval mode (``False``).

        Returns:
            ParallelExpertEncoder: ``self``, matching ``nn.Module.train``d.
        """
        super().train(mode)
        if self.freeze_diar:
            self.diarization_model.eval()
        if self.freeze_asr:
            self.asr_encoder.eval()
        return self

    # ConformerEncoder-compatible properties (drop-in for SALM perception).
    @property
    def d_model(self) -> int:
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        return self.asr_encoder.subsampling_factor

    @property
    def pre_encode(self):
        return self.asr_encoder.pre_encode

    # freeze/unfreeze parity (plain nn.Module re-exposing the standalone helpers).
    def freeze(self) -> None:
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        unfreeze(self, partial=partial)

    # Fusion helpers
    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        """Mirror of ``MSEncDecMultiTaskModel.get_sinusoid_position_encoding``."""
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_position, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @classmethod
    def _build_tag_kernel(
        cls,
        n_tags: int,
        row_offset: int,
        embedding_dim: int,
        stride: int = 1,
        calibrate: bool = False,
    ) -> torch.Tensor:
        """Take ``n_tags`` sinusoid rows from ``row_offset``, spaced ``stride``, and optionally calibrate.

        Ported from the reference PE encoder. The defaults (``row_offset=0``, ``stride=1``,
        ``calibrate=False``) reproduce ``_build_sinusoid_position_encoding(n_tags, dim)`` exactly,
        which is what the published bundle uses -- so behaviour is unchanged unless a caller opts in.

        Why the knobs matter: adjacent sinusoid rows are highly correlated, so at ``stride=1``
        different speakers inject nearly the same direction into the ASR states. ``calibrate``
        rescales so one active tag injects norm ``sqrt(embedding_dim)``, matching the scale of the
        LayerNorm'd ASR states it is added to; uncalibrated, the infusion may be far too large or
        too small relative to the acoustics.

        Args:
            n_tags (int): Number of tag identities (rows in the kernel).
            row_offset (int): First sinusoid row reserved for this tag family.
            embedding_dim (int): ASR state dimension the kernel projects into.
            stride (int): Spacing between consecutive tag rows.
            calibrate (bool): Rescale so one active tag injects norm ``sqrt(embedding_dim)``.

        Returns:
            torch.Tensor: Tag kernel. Shape: ``(n_tags, embedding_dim)``.
        """
        rows = [row_offset + stride * i for i in range(n_tags)]
        table = cls._build_sinusoid_position_encoding(rows[-1] + 1, embedding_dim)
        kernel = table[rows].contiguous()
        if not calibrate:
            return kernel

        # The mean single-tag code is computed with LayerNorm's initial affine values. The norms
        # remain learnable, so training may move away from this starting point.
        eye = torch.eye(n_tags, dtype=kernel.dtype)
        centred = (eye - eye.mean(dim=1, keepdim=True)) / eye.std(dim=1, unbiased=False, keepdim=True)
        mean_norm = (centred @ kernel).norm(dim=1).mean()
        return (kernel * (math.sqrt(embedding_dim) / mean_norm)).contiguous()

    @staticmethod
    def _align_diar_frames(spk_targets: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad-by-repeat or truncate ``spk_targets`` to ``target_len`` along time."""
        cur_len = spk_targets.shape[1]
        if cur_len < target_len:
            last = spk_targets[:, -1:, :]
            spk_targets = torch.cat([spk_targets, last.repeat(1, target_len - cur_len, 1)], dim=1)
        elif cur_len > target_len:
            spk_targets = spk_targets[:, :target_len, :]
        return spk_targets

    @staticmethod
    def _require_diar_on_asr_grid(diar_preds: torch.Tensor, asr_len: int) -> None:
        """Raise if offline diarizer predictions are not on the ASR branch's frame grid.

        Args:
            diar_preds (Tensor): Diarizer predictions. Shape ``(B, T_diar, n_spk)``.
            asr_len (int): Number of frames the ASR branch produced for the same mels.

        Raises:
            ValueError: If ``T_diar`` and ``asr_len`` differ by more than ``_MAX_DIAR_ASR_FRAME_MISMATCH``.
        """
        diar_len = diar_preds.shape[1]
        if abs(diar_len - asr_len) > _MAX_DIAR_ASR_FRAME_MISMATCH:
            raise ValueError(
                f"The diarizer predicted {diar_len} frames but the ASR branch produced {asr_len} for the same "
                f"audio; they may differ by at most {_MAX_DIAR_ASR_FRAME_MISMATCH}. The predictions are on a "
                "different time grid (a high-resolution diarizer whose output was not pooled?), and aligning "
                "them would truncate or stretch the speaker activity."
            )

    @staticmethod
    def _match_module_io(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Cast ``tensor`` to ``module``'s parameter device & dtype (mels arrive fp32, experts run bf16).

        Args:
            tensor (Tensor): Input to align (e.g. mel features).
            module (nn.Module): Module whose first parameter sets the target device/dtype.

        Returns:
            ``tensor`` moved to the module's device/dtype, or unchanged if it has no parameters.
        """
        param = next(module.parameters(), None)
        if param is None:
            return tensor
        return tensor.to(device=param.device, dtype=param.dtype)

    def missing_rttm_rows(self, spk_targets: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Per-row mask of samples whose speaker targets are the missing-RTTM sentinel.

        Returns a ``(B,)`` bool tensor, or ``None`` when ``spk_targets`` is absent. Per-row, not
        all-or-nothing: a batch normally mixes cuts that have an RTTM with cuts that do not, and
        each should get the right speaker source.
        """
        if spk_targets is None:
            return None
        return (spk_targets <= self.missing_rttm_target).flatten(start_dim=1).all(dim=1)

    def _missing_target_rows(self, spk_targets: torch.Tensor) -> torch.Tensor:
        """Upstream's name for :meth:`missing_rttm_rows`, kept so SALM code calls one API.

        Delegates rather than duplicating: upstream compares ``== missing_rttm_target`` while this
        uses ``<=``, which is the same set for the -1.0 sentinel but survives a target that has
        been through a float cast or a collation pad. Non-optional argument to match the caller.
        """
        return self.missing_rttm_rows(spk_targets)

    @staticmethod
    def _validate_chunk_size(name: str, value: Optional[float]) -> Optional[float]:
        """Positive-or-None, rejected at construction rather than at the first forward."""
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive or None, got {value}.")
        return value

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

    def _should_run_diarization(
        self,
        spk_targets: Optional[torch.Tensor],
        use_diarization: Optional[torch.Tensor] = None,
    ) -> bool:
        """Whether to run the diarizer for this batch. Uniform in training and under DDP/FSDP.

        The obvious rule -- run it only when some row carries the missing-RTTM sentinel -- makes
        the decision depend on the DATA each rank happened to receive. Mix a corpus that has RTTMs
        with one that does not and ranks disagree: a rank whose batch is entirely RTTM-backed skips
        the diarizer's forward while another rank runs it. The two then disagree about executing a
        module, which desynchronises collectives and hangs the job rather than failing.

        So the skip is only ever a single-process inference optimisation. Training and any
        world_size > 1 take the uniform path, whatever the batch happens to hold.

        Args:
            spk_targets: ``(B, T, n_spk)`` oracle targets, or ``None`` when there are none.
            use_diarization: precomputed per-row mask; recomputed from the sentinel when omitted.

        Returns:
            bool: whether to run the diarizer branch.
        """
        if spk_targets is None or self.training:
            return True
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return True
        if use_diarization is None:
            use_diarization = self.missing_rttm_rows(spk_targets)
        return bool(use_diarization.any())

    def _fuse_diar_and_asr(self, asr_encoded: torch.Tensor, spk_targets: torch.Tensor) -> torch.Tensor:
        """Fuse ASR states with speaker-activity preds (LayerNorm + sinusoidal kernel + ADD).

        Args:
            asr_encoded (Tensor): ASR encoder output. Shape ``(B, D, T_asr)``.
            spk_targets (Tensor): Speaker-activity predictions. Shape ``(B, T_diar, n_spk)``.

        Returns:
            Fused encoder output. Shape ``(B, D, T_asr)``.
        """
        asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
        spk_targets = self._align_diar_frames(spk_targets, asr_enc_states.shape[1]).to(asr_enc_states.dtype)

        spk_targets = self._speaker_features(spk_targets, asr_enc_states.dtype)
        asr_enc_states = self.asr_norm(asr_enc_states)
        spk_targets = self.diar_norm(spk_targets)
        speaker_infusion = torch.matmul(spk_targets, self.diar_kernel.to(spk_targets.dtype))
        fused = self.spk_kernel_scale * speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

    # Forward — identical signature to ConformerEncoder.forward
    @contextlib.contextmanager
    def online_inference(self, enabled: bool = True):
        """Route ``forward`` through the windowed generation path inside this scope.

        Restores the previous value on exit, so nesting and early exceptions cannot leave the
        encoder stuck in generation mode for a subsequent training step.
        """
        previous = getattr(self, "online_inference_enabled", None)
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    def forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ):
        """Encode ``audio_signal``, optionally fusing diarization.

        Dispatches to :meth:`_forward` (offline) or :meth:`_forward_online` (long-form,
        inference-only, when the input exceeds one window).

        Args:
            audio_signal (Tensor): Un-normalised mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` speaker-activity override (RTTM/oracle);
                when ``None`` the wrapped Sortformer is run.

        Returns:
            Tuple ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T_asr)``.
        """
        if spk_targets is not None:
            use_online = False
        elif getattr(self, "online_inference_enabled", None) is not None:
            # An explicit `online_inference()` scope wins over the length heuristic below, and
            # deliberately does NOT re-check the audio against `chunk_feat_len`: a caller that
            # opened the scope is generating, and knows it wants the windowed path.
            use_online = bool(self.online_inference_enabled) and self.online_inference_length > 0
        elif self.online_inference_length > 0 and not self.training:
            # Even if spk_targets is None, use offline if audio is short enough
            use_online = audio_signal.shape[-1] > self.chunk_feat_len
        else:
            use_online = False

        if use_online:
            return self._forward_online(audio_signal=audio_signal, length=length, spk_targets=spk_targets)

        return self._forward(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
        )

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

    def _forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics."""
        # Rows with no RTTM carry the sentinel; they need diarizer predictions even though the
        # batch as a whole supplied `spk_targets`. Run the diarizer if ANY row needs it, then
        # splice per row below.
        missing_rows = self.missing_rttm_rows(spk_targets)
        needs_diar = self._should_run_diarization(spk_targets, missing_rows)
        diar_preds = None
        if needs_diar:
            # Normalise as `diar_normalize_type` says, then cast the fp32 mels to the diarizer's
            # device/dtype before its conv subsampling.
            diar_signal = self._match_module_io(
                self._normalize_diar_input(audio_signal, length), self.diarization_model
            )
            diar_length = length.to(device=diar_signal.device)
            with _branch_grad_mode(self.freeze_diar):
                emb_seq, emb_seq_length = self.diarization_model.frontend_encoder(
                    processed_signal=diar_signal,
                    processed_signal_length=diar_length,
                    bypass_pre_encode=False,
                )
                diar_preds = self.diarization_model.forward_infer(
                    emb_seq=emb_seq,
                    emb_seq_length=emb_seq_length,
                )
            if isinstance(diar_preds, tuple):
                diar_preds = diar_preds[0]
            # `forward_infer` returns a high-resolution diarizer's predictions every 10 ms. Pool
            # them onto the ASR grid, as `SortformerEncLabelModel.forward` and the streaming step
            # do; a no-op for a diarizer that already predicts at the ASR frame rate.
            diar_preds = self._align_diarization_output_resolution(diar_preds, emb_seq_length)

        if self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal,
                length,
                normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal
        # Cast fp32 mels to the ASR encoder's device/dtype before its conv subsampling.
        asr_audio_signal = self._match_module_io(asr_audio_signal, self.asr_encoder)
        asr_length = length.to(device=asr_audio_signal.device)

        with _branch_grad_mode(self.freeze_asr):
            asr_encoded, asr_encoded_len = self.asr_encoder(
                audio_signal=asr_audio_signal,
                length=asr_length,
            )

        if diar_preds is not None:
            self._require_diar_on_asr_grid(diar_preds, asr_encoded.shape[-1])
            if spk_targets is None:
                spk_targets = diar_preds
            else:
                # Per-row substitution: keep real RTTM targets, replace only sentinel rows. Splice at
                # the longer of the two lengths, so a sentinel row keeps every diarizer frame (an
                # all-sentinel batch then equals passing no targets) and a real row is only padded by
                # repeat, as the fusion would pad it anyway.
                splice_len = max(spk_targets.shape[1], diar_preds.shape[1])
                spk_targets = self._align_diar_frames(spk_targets, splice_len)
                diar_preds = self._align_diar_frames(diar_preds, splice_len).to(spk_targets.dtype)
                spk_targets = torch.where(missing_rows.view(-1, 1, 1), diar_preds, spk_targets)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(asr_encoded, spk_targets)
        else:
            outputs = asr_encoded

        return outputs, asr_encoded_len

    def _forward_online(self, audio_signal, length, spk_targets=None):
        """Long-form online inference: a lock-step loop over fixed windows.

        Walks the recording in non-overlapping windows of ``online_inference_length``
        output frames. Both experts run on the same context-extended slice
        ``[stt - left : end + right]`` (differing only in normalization): the ASR
        encoder uses overlap-and-trim, while the streaming Sortformer carries its
        speaker-cache / FIFO state across windows and trims context internally.
        Per-window diar outputs are aligned to the ASR frame count, then both buffers
        are concatenated and fused once.

        Args:
            audio_signal (Tensor): Un-normalised mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            spk_targets (Tensor, optional): ``(B, T, n_spk)`` override; when given, only ASR is chunked.

        Returns:
            Tuple ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T_asr)``.
        """
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / self.chunk_feat_len))

        # Normalise the whole utterance once (not per chunk) to match offline stats.
        if self.asr_normalize_type:
            asr_audio_signal, _, _ = normalize_batch(
                audio_signal,
                length,
                normalize_type=self.asr_normalize_type,
            )
        else:
            asr_audio_signal = audio_signal

        # Match the ASR encoder's device/dtype (mels arrive fp32, encoder runs bf16).
        asr_audio_signal = self._match_module_io(asr_audio_signal, self.asr_encoder)
        length = length.to(device=asr_audio_signal.device)

        run_streaming_diar = spk_targets is None
        if run_streaming_diar:
            # Like the ASR branch, the diarizer's normalization uses the whole utterance's statistics.
            streaming_state, stream_dtype, diar_audio_signal, diar_length = self._init_streaming_diar(
                self._normalize_diar_input(audio_signal, length),
                length,
                batch_size=audio_signal.shape[0],
            )
            n_spk = self.diarization_model.sortformer_modules.n_spk
            total_preds = torch.zeros(
                (diar_audio_signal.shape[0], 0, n_spk),
                device=diar_audio_signal.device,
                dtype=stream_dtype,
            )

        asr_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        asr_encoded_len = torch.zeros_like(length)

        for chunk_idx in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, '_suppress_online_pbar', False),
        ):
            stt = chunk_idx * self.chunk_feat_len
            end = min(stt + self.chunk_feat_len, total_feat_len)

            # Shared context-extended window (input mel frames) for both branches.
            enc_stt = max(stt - self.left_ctx_feat_len, 0)
            enc_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset = stt - enc_stt
            right_offset = enc_end - end

            asr_chunk = asr_audio_signal[:, :, enc_stt:enc_end]
            chunk_length = (length - enc_stt).clamp(min=0, max=enc_end - enc_stt)
            with _branch_grad_mode(self.freeze_asr):
                enc_ctx, _ = self.asr_encoder(audio_signal=asr_chunk, length=chunk_length)
            # Trim context off in output-frame space using rounded cumulative positions.
            left_drop = left_offset // self.subsampling_factor
            core_len = round(end / self.subsampling_factor) - round(stt / self.subsampling_factor)
            core_len = max(0, min(core_len, enc_ctx.shape[-1] - left_drop))
            enc_chunk = enc_ctx[:, :, left_drop : left_drop + core_len]
            asr_chunks.append(enc_chunk)
            asr_encoded_len += core_len
            align_target = enc_chunk.shape[-1]

            # Diar branch: stream the same window; Sortformer trims context internally.
            if run_streaming_diar:
                prev_len = total_preds.shape[1]
                diar_chunk = diar_audio_signal[:, :, enc_stt:enc_end].transpose(1, 2)  # (B, t, feat_in)
                diar_chunk_length = (diar_length - enc_stt).clamp(min=0, max=enc_end - enc_stt)
                with (
                    _branch_grad_mode(self.freeze_diar),
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
                diar_raw = total_preds[:, prev_len:]
                # Newly emitted frames, aligned to the ASR chunk (frame-parallel).
                new_preds = self._align_diar_frames(diar_raw, align_target)
                diar_chunks.append(new_preds)

        asr_encoded = torch.cat(asr_chunks, dim=2)  # (B, D, T_asr)
        if run_streaming_diar:
            spk_targets = torch.cat(diar_chunks, dim=1)  # (B, T_asr, n_spk)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(asr_encoded, spk_targets)
        else:
            outputs = asr_encoded

        return outputs, asr_encoded_len

    def _init_streaming_diar(self, audio_signal: torch.Tensor, length: torch.Tensor, batch_size: int):
        """Configure the wrapped Sortformer for streaming and build its initial state.

        Args:
            audio_signal (Tensor): Input mel features. Shape ``(B, feat_in, n_frames)``.
            length (Tensor): Per-sample feature lengths. Shape ``(B,)``.
            batch_size (int): Batch size for the streaming state.

        Returns:
            ``(streaming_state, stream_dtype, diar_audio_signal, diar_length)`` cast onto
            the diarizer's device & dtype.
        """
        self._apply_diar_streaming_overrides()
        sm = self.diarization_model.sortformer_modules

        diar_param = next(self.diarization_model.parameters(), None)
        if diar_param is not None:
            self.diarization_model.to(diar_param.device)
            diar_device, stream_dtype = diar_param.device, diar_param.dtype
        else:
            diar_device, stream_dtype = audio_signal.device, torch.get_default_dtype()

        diar_audio_signal = audio_signal.to(device=diar_device, dtype=stream_dtype)
        diar_length = length.to(device=diar_device)

        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            streaming_state = sm.init_streaming_state(
                batch_size=batch_size,
                async_streaming=self.diarization_model.async_streaming,
                device=diar_device,
            )
        return streaming_state, stream_dtype, diar_audio_signal, diar_length


@experimental
class StreamingParallelExpertEncoder(ParallelExpertEncoder, StreamingEncoder):
    """:class:`ParallelExpertEncoder` that also speaks the cache-aware streaming interface.

    The base class already requires the ASR branch to implement ``cache_aware_stream_step`` /
    ``get_initial_cache_state`` / ``setup_streaming_params``, so this subclass adds **no new
    capability** -- it exposes the branch's existing machinery through the wrapper and steps the
    Sortformer in lock-step on the same mel chunk so the fusion stays frame-aligned.

    Offline behaviour is inherited unchanged, including the normalisation of the diarizer input
    (``diar_normalize_type``: by default the diarizer's own), over the whole utterance. The streaming
    step applies it to each chunk on its own, as it does ``asr_normalize_type`` to the ASR input, so a
    stream then feeds the diarizer differently normalised mels than the offline (training) forward
    does; the first normalisation logs a warning saying so, once. Mount this class instead of the base
    whenever the perception stack will be driven chunk-by-chunk.
    """

    # ------------------------------------------------------------------
    # StreamingEncoder interface — delegated to the cache-aware ASR branch
    # ------------------------------------------------------------------
    # The ASR branch is a real cache-aware encoder, so all the machinery already exists; what this
    # adds is (a) exposing it through the wrapper and (b) stepping the Sortformer in lock-step on
    # the same mel chunk so the fusion stays frame-aligned.

    # Class-level defaults: `nn.Module.__getattr__` raises AttributeError for a missing instance
    # attribute, which would mask the actionable "call get_initial_cache_state() first" guard below.
    _diar_streaming_state = None
    _diar_total_preds = None
    _diar_stream_dtype = None
    # Whether the current stream steps the diarizer; ``None`` until its first chunk.
    _diar_stream_source = None
    # Whether the one-time warning about normalising the diarizer input has been logged.
    _diar_normalization_warned = False

    def _normalize_diar_input(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Normalise as the base class does; the first time there is a normalisation to apply, warn once.

        The streaming step normalises each chunk with that chunk's statistics, while the offline (training)
        forward and online inference use the whole utterance's, so the diarizer input differs between them.
        """
        if self.diar_normalize_type and not self._diar_normalization_warned:
            self._diar_normalization_warned = True
            logging.warning(
                "[StreamingParallelExpertEncoder] Normalizing the diarizer input with %r (diar_normalize_type). "
                "The streaming step (cache_aware_stream_step) normalizes each chunk on its own, while the offline "
                "(training) forward and online inference normalize the whole utterance, so the diarizer receives "
                "differently normalized mels when streaming than in training. Logged once.",
                self.diar_normalize_type,
            )
        return super()._normalize_diar_input(audio_signal, length)

    @property
    def streaming_cfg(self):
        """The ASR branch's streaming config (frame accounting, pre-encode cache sizes)."""
        return self.asr_encoder.streaming_cfg

    def setup_streaming_params(self, **kwargs) -> None:
        """Configure the ASR branch for cache-aware streaming, and the diarizer to match."""
        self.asr_encoder.setup_streaming_params(**kwargs)
        self._apply_diar_streaming_overrides()

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
        """Fresh ASR cache, and reset the diarizer's streaming state for a new stream."""
        diar_param = next(self.diarization_model.parameters(), None)
        diar_device = diar_param.device if diar_param is not None else device
        self._diar_stream_dtype = diar_param.dtype if diar_param is not None else dtype
        if diar_param is not None:
            # `forward_streaming_step` builds tensors on the diarizer's LightningModule `.device`,
            # which an outer `enc.to('cuda')` does NOT update (nn.Module.to does not re-enter a
            # child's `to`). Without this the embeddings are moved back to CPU mid-step and the
            # first LayerNorm dies with a device mismatch. `_init_streaming_diar` already does this.
            self.diarization_model.to(diar_device)
        sm = self.diarization_model.sortformer_modules
        with _disable_dist_feature_sync(), _default_dtype(self._diar_stream_dtype):
            self._diar_streaming_state = sm.init_streaming_state(
                batch_size=batch_size,
                async_streaming=self.diarization_model.async_streaming,
                device=diar_device,
            )
        self._diar_total_preds = torch.zeros(
            (batch_size, 0, sm.n_spk), device=diar_device, dtype=self._diar_stream_dtype
        )
        self._diar_stream_source = None
        return self.asr_encoder.get_initial_cache_state(
            batch_size=batch_size, dtype=dtype, device=device, max_dim=max_dim
        )

    def cache_aware_stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        keep_all_outputs=True,
        drop_extra_pre_encoded=None,
        spk_targets=None,
    ):
        """One streaming step: ASR chunk + diarizer chunk, fused.

        ``spk_targets`` (``(B, n_frames_out, n_spk)`` for THIS chunk) overrides the diarizer; when
        omitted the embedded Sortformer is stepped on the same mel chunk and its newly emitted
        frames are used. Both branches see the identical chunk and ``drop_extra_pre_encoded``, which
        is what keeps the fusion frame-aligned (the same rule ``SpeakerTaggedASR`` follows).

        Rows of ``spk_targets`` that carry the missing-RTTM sentinel get the diarizer's predictions
        instead, as on the offline path: the Sortformer is stepped on the full batch and only those
        rows are replaced. A stream whose rows all have real targets never steps it.

        The diarizer keeps one batched streaming state, so within a stream it is stepped on every
        chunk or on none. The first chunk after :meth:`get_initial_cache_state` decides. Once
        stepped, it keeps being stepped, also on chunks without sentinel rows. A chunk that needs it
        after chunks that skipped it raises, because its state has not seen the start of the audio.
        """
        if self.asr_normalize_type:
            asr_signal, _, _ = normalize_batch(
                processed_signal, processed_signal_length, normalize_type=self.asr_normalize_type
            )
        else:
            asr_signal = processed_signal
        asr_signal = self._match_module_io(asr_signal, self.asr_encoder)

        asr_kwargs = dict(
            processed_signal=asr_signal,
            processed_signal_length=processed_signal_length.to(asr_signal.device),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
        )
        if drop_extra_pre_encoded is not None:
            asr_kwargs["drop_extra_pre_encoded"] = drop_extra_pre_encoded
        with _branch_grad_mode(self.freeze_asr):
            asr_out = self.asr_encoder.cache_aware_stream_step(**asr_kwargs)
        asr_encoded, asr_encoded_len = asr_out[0], asr_out[1]
        rest = tuple(asr_out[2:])

        sentinel_rows = None
        if spk_targets is not None:
            sentinel_rows = self.missing_rttm_rows(spk_targets)
            if spk_targets.shape[1] == 0:
                # A zero-width slice has no frames to be sentinel; `.all()` over it is vacuously True.
                sentinel_rows = torch.zeros_like(sentinel_rows)
        if self._should_step_stream_diarizer(spk_targets is None or bool(sentinel_rows.any())):
            # The ASR branch drops `drop_extra_pre_encoded` frames of pre-encode cache from its
            # output; the diarizer must drop the same, or it emits N+2 frames per N ASR frames and
            # the fusion consumes STALE predictions -- the speaker signal ends up one chunk behind
            # the frames it is added to. When the caller does not say (perception does not), fall
            # back to the ASR branch's own streaming config rather than to 0.
            diar_drop = drop_extra_pre_encoded
            if diar_drop is None:
                diar_drop = getattr(self.asr_encoder.streaming_cfg, 'drop_extra_pre_encoded', None)
            diar_preds = self._stream_diarizer(
                processed_signal, processed_signal_length, asr_encoded.shape[-1], diar_drop
            )
            if spk_targets is None:
                spk_targets = diar_preds
            elif bool(sentinel_rows.any()):
                # `diar_preds` is already on the ASR chunk's frame grid. The oracle slice can be
                # shorter (the last chunk of a stream), so align it the way the fusion would: the
                # real rows then fuse exactly as they do without the splice.
                spk_targets = self._align_diar_frames(spk_targets, asr_encoded.shape[-1])
                diar_preds = diar_preds.to(device=spk_targets.device, dtype=spk_targets.dtype)
                spk_targets = torch.where(sentinel_rows.view(-1, 1, 1), diar_preds, spk_targets)
        if spk_targets is not None:
            asr_encoded = self._fuse_diar_and_asr(asr_encoded, spk_targets)
        return (asr_encoded, asr_encoded_len) + rest

    def _should_step_stream_diarizer(self, needed: bool) -> bool:
        """Whether to step the diarizer on this chunk, so that its state follows the whole stream.

        Args:
            needed (bool): this chunk needs diarizer predictions (no ``spk_targets``, or a sentinel row).

        Returns:
            bool: ``True`` if the stream already steps the diarizer, or if ``needed`` and no earlier
            chunk skipped it; ``False`` otherwise.

        Raises:
            RuntimeError: if ``needed`` after chunks of the same stream that did not step the diarizer.
        """
        if self._diar_stream_source == _DIAR_STREAM_STEPPED:
            return True
        if needed and self._diar_stream_source == _DIAR_STREAM_SKIPPED:
            raise RuntimeError(
                "This chunk needs the embedded diarizer (it has no `spk_targets`, or a row carries the "
                "missing-RTTM sentinel), but earlier chunks of this stream had real targets in every row. "
                "The diarizer was not stepped on them, so its streaming state has not seen the start of "
                "the audio. Mark a row without an RTTM as sentinel on every chunk of the stream (the "
                "dataset fills such a row across its whole length), or call get_initial_cache_state() to "
                "start a new stream."
            )
        self._diar_stream_source = _DIAR_STREAM_STEPPED if needed else _DIAR_STREAM_SKIPPED
        return needed

    def _stream_diarizer(self, processed_signal, processed_signal_length, align_target, drop_extra_pre_encoded):
        """Advance the Sortformer by one chunk and return its NEW frames, ASR-aligned."""
        state_batch = None
        if self._diar_total_preds is not None:
            state_batch = self._diar_total_preds.shape[0]
        if state_batch is not None and state_batch != processed_signal.shape[0]:
            raise RuntimeError(
                f"The diarizer's streaming state was allocated for batch {state_batch} but this step "
                f"has batch {processed_signal.shape[0]}. The embedded Sortformer keeps ONE batched "
                "state on the module, so it cannot follow a caller that steps a subset of streams "
                "(as `_generate_dynamic_streaming` does when streams desynchronise).\n"
                "Workarounds: pass `spk_targets` with real targets in every row so the diarizer is not "
                "stepped, or use the chunked streaming path, which always steps the full batch."
            )
        if self._diar_streaming_state is None:
            raise RuntimeError(
                "ParallelExpertEncoder.cache_aware_stream_step requires get_initial_cache_state() first "
                "-- it is what allocates the diarizer's streaming state."
            )
        # Normalise this chunk on its own, as `cache_aware_stream_step` does the ASR input, then cast.
        diar_signal = (
            self._normalize_diar_input(processed_signal, processed_signal_length)
            .to(device=self._diar_total_preds.device, dtype=self._diar_stream_dtype)
            .transpose(1, 2)
        )  # (B, t, feat_in)
        diar_len = processed_signal_length.to(device=diar_signal.device)
        prev_len = self._diar_total_preds.shape[1]
        step_kwargs = {}
        if drop_extra_pre_encoded is not None:
            step_kwargs["drop_extra_pre_encoded"] = drop_extra_pre_encoded
        with (
            _branch_grad_mode(self.freeze_diar),
            _disable_dist_feature_sync(),
            _default_dtype(self._diar_stream_dtype),
        ):
            self._diar_streaming_state, self._diar_total_preds = self.diarization_model.forward_streaming_step(
                processed_signal=diar_signal,
                processed_signal_length=diar_len,
                streaming_state=self._diar_streaming_state,
                total_preds=self._diar_total_preds,
                **step_kwargs,
            )
        new_preds = self._diar_total_preds[:, prev_len:]
        # Fail here rather than downstream. `_align_diar_frames` pads-by-repeat, and repeating a
        # zero-width tensor yields another zero-width tensor -- so an empty diarizer step would
        # propagate silently into `_fuse_diar_and_asr` and surface as a shape error deep in the
        # fusion, naming neither the diarizer nor the chunk that produced nothing. Not currently
        # reachable: the diarizer emits exactly as many frames as the ASR branch in every measured
        # configuration, which is precisely why this is cheap to assert.
        if new_preds.shape[1] == 0:
            raise RuntimeError(
                f"The diarizer produced no new frames for a chunk of {processed_signal.shape[-1]} "
                f"mel frames (valid lengths {processed_signal_length.tolist()}, "
                f"drop_extra_pre_encoded={drop_extra_pre_encoded}). Its feature stacking consumed "
                "the whole chunk, so there is nothing to align to the ASR branch's "
                f"{align_target} frames. Use a larger `inference_chunk_size`."
            )
        if align_target <= 0:
            raise RuntimeError(
                f"The ASR branch produced {align_target} frames for this chunk, so there is no "
                "target width to align the diarizer's predictions to. This means the chunk was "
                "consumed entirely by the pre-encode cache; use a larger `inference_chunk_size`."
            )
        return self._align_diar_frames(new_preds, align_target)


@experimental
class StreamingParallelExpertEncoderPT(ParallelExpertEncoderPT):
    """``.nemo`` shell that mounts a :class:`StreamingParallelExpertEncoder`.

    Identical archive layout to :class:`ParallelExpertEncoderPT`; only the encoder class differs, so
    a bundle built for one can be re-targeted at the other by changing ``target`` in its
    ``model_config.yaml``. Both classes read the bundle's ``diar_normalize_type`` the same way (unset,
    the diarizer's own ``preprocessor.normalize``; set, an override, also through ``config_overrides``
    of :meth:`load_from_nemo`), and this one also applies it to each chunk in its streaming step.
    """

    _ENCODER_CLS = StreamingParallelExpertEncoder


def _nemo_member(archive: "tarfile.TarFile", basename: str):
    """Find an archive member by basename.

    The two source checkpoints do not agree on member naming -- the ASR ``.nemo`` stores
    ``./model_config.yaml`` while the Sortformer one stores a bare ``model_config.yaml`` -- so
    match on the basename rather than the full path.
    """
    for member in archive.getmembers():
        if os.path.basename(member.name) == basename:
            return member
    raise FileNotFoundError(f"{basename!r} not found in the .nemo archive")


def _config_from_nemo(nemo_path: str) -> dict:
    """Read ``model_config.yaml`` out of a ``.nemo`` archive."""
    with tarfile.open(nemo_path) as archive:
        return yaml.safe_load(archive.extractfile(_nemo_member(archive, 'model_config.yaml')).read())


def _weights_from_nemo(nemo_path: str, map_location) -> dict:
    """Read ``model_weights.ckpt`` out of a ``.nemo`` archive as a plain state dict."""
    with tarfile.open(nemo_path) as archive:
        handle = archive.extractfile(_nemo_member(archive, 'model_weights.ckpt'))
        state = torch.load(io.BytesIO(handle.read()), map_location=map_location, weights_only=True)
    return state.get('state_dict', state) if isinstance(state, dict) else state


def _resolve_branch_source(path_or_name: str, model_cls, map_location):
    """Return ``(config_dict, state_dict)`` for a PE branch source.

    A local ``.nemo`` is read straight out of the archive -- cheap, and it never instantiates the
    model's unused heads. Anything else is treated as a pretrained id (HuggingFace Hub
    ``{repo}/{name}`` or NGC alias) and resolved via ``model_cls.from_pretrained``, which handles
    download and caching; its ``cfg``/``state_dict`` have the same layout as the archive's.
    """
    if isinstance(path_or_name, str) and path_or_name.endswith('.nemo') and os.path.isfile(path_or_name):
        return _config_from_nemo(path_or_name), _weights_from_nemo(path_or_name, map_location)
    logging.info("Resolving PE branch %r via %s.from_pretrained", path_or_name, model_cls.__name__)
    model = model_cls.from_pretrained(model_name=path_or_name, map_location=map_location).eval()
    return OmegaConf.to_container(model.cfg, resolve=True), model.state_dict()


def _checkpoint_normalize(cfg, default=_NORMALIZE_UNSET):
    """Return an ASR or diarizer config's ``preprocessor.normalize``, or ``default`` when it has none."""
    preprocessor = cfg.get('preprocessor', None) if isinstance(cfg, Mapping) else None
    if not isinstance(preprocessor, Mapping) or 'normalize' not in preprocessor:
        return default
    return preprocessor['normalize']


def _resolve_diar_normalize_type(diar_normalize_type, diarizer_normalize) -> Optional[str]:
    """Resolve ``diar_normalize_type`` to the normalization of the diarizer input, and log the result.

    Unset keeps the diarizer's own normalization; an explicit value overrides it. Both encoder classes
    resolve it this way.

    Args:
        diar_normalize_type: The constructor argument, or ``_NORMALIZE_UNSET``.
        diarizer_normalize: The ``normalize`` of the diarizer's own preprocessor.

    Returns:
        ``'per_feature'``, ``'all_features'``, or ``None`` for no normalization.

    Raises:
        ValueError: If the value to apply is not a normalization the diarizer branch can apply.
    """
    if diar_normalize_type is _NORMALIZE_UNSET:
        resolved = _diar_normalize_type_from_value(
            diarizer_normalize,
            "The diarizer's preprocessor.normalize",
            "; set diar_normalize_type explicitly",
        )
        source = f"the diarizer's own preprocessor.normalize={diarizer_normalize!r}"
    else:
        resolved = _diar_normalize_type_from_value(diar_normalize_type, "diar_normalize_type")
        source = (
            f"diar_normalize_type={diar_normalize_type!r}, over the diarizer's own "
            f"preprocessor.normalize={diarizer_normalize!r}"
        )
    logging.info(
        "[ParallelExpertEncoder] The diarizer branch input is %s, from %s.",
        f"normalized with {resolved!r}" if resolved else "not normalized",
        source,
    )
    return resolved


def _diarizer_own_normalize(diarization_model_cfg, diarization_model):
    """The ``normalize`` of the diarizer's own preprocessor, which it was trained with.

    Read from the config's ``preprocessor.normalize``. A config without it gets the preprocessor's own
    default, so that is read off the preprocessor built from it.
    """
    normalize = _checkpoint_normalize(diarization_model_cfg)
    if normalize is _NORMALIZE_UNSET:
        featurizer = getattr(getattr(diarization_model, 'preprocessor', None), 'featurizer', None)
        normalize = getattr(featurizer, 'normalize', None)
    return normalize


def _diar_normalize_type_from_value(value, source: str, hint: str = "") -> Optional[str]:
    """Map a diarizer normalization value to the one the diarizer branch applies, ``None`` for none.

    ``NA``, ``null`` or an empty value mean no normalization; ``per_feature`` and ``all_features`` are
    applied as given. Anything else raises, because the diarizer branch applies only these, and
    ``normalize_batch`` would skip an unknown string without a word. A ``fixed_mean``/``fixed_std`` dict,
    which ``normalize_batch`` does apply, is rejected as well.
    """
    if not value or value == 'NA':
        return None
    if isinstance(value, str) and value in _DIAR_NORMALIZE_TYPES:
        return value
    raise ValueError(
        f"{source}={value!r} is not a normalization the ParallelExpertEncoder diarizer branch can apply: use "
        f"one of {list(_DIAR_NORMALIZE_TYPES)}, or null/'NA' for none{hint}."
    )


def _persisted_diar_normalize_type(encoder) -> str:
    """``encoder``'s effective ``diar_normalize_type`` as this module's bundle configs state it, ``'NA'`` for none.

    That is the bundle config the encoder keeps and the one ``save_to_nemo`` writes. ``'NA'`` and not null: a
    serializer that drops null values would turn it into an absent key, which means "the diarizer's own
    normalization" and could then normalise a diarizer this encoder does not. An HF export (``to_hf.py``)
    overwrites it with the encoder's ``diar_normalize_type`` attribute, null for none; both reload as no
    normalization.
    """
    return encoder.diar_normalize_type or 'NA'


def _asr_normalize_type_from_checkpoint(asr_cfg, source):
    """Resolve ``asr_normalize_type='auto'``: the ASR branch replays its checkpoint's own normalization.

    ``NA``, ``null`` or an empty value mean the checkpoint's preprocessor does not normalize, so the
    branch must not either (``None``). ``per_feature`` and ``all_features`` are replayed as given.
    Anything else, or a config without ``preprocessor.normalize``, raises: guessing would feed the
    branch mels it was not trained on.
    """
    normalize = _checkpoint_normalize(asr_cfg)
    if normalize is _NORMALIZE_UNSET:
        raise ValueError(
            f"asr_normalize_type='auto' cannot follow the ASR checkpoint {source!r}: its config has no "
            "preprocessor.normalize. Set asr_normalize_type explicitly ('per_feature', 'all_features', or null/'NA')."
        )
    if not normalize or normalize == 'NA':
        return None
    if isinstance(normalize, str) and normalize in _NORMALIZE_AUTO_FOLLOWED:
        return normalize
    raise ValueError(
        f"asr_normalize_type='auto' cannot follow the ASR checkpoint {source!r}: its preprocessor.normalize="
        f"{normalize!r} is not one of {list(_NORMALIZE_AUTO_FOLLOWED)} or NA/null. Set asr_normalize_type explicitly."
    )


# Everything the PE reads off its ASR branch. Checked by name rather than with `isinstance` so any
# cache-aware encoder can be dropped in -- `ConformerEncoder` and `StreamingTransformerEncoder` both
# satisfy it, and they differ in ways the PE never touches (convolutional vs `feature_stacking`
# subsampling, `rel_pos` vs `rope`).
_ASR_ENCODER_INTERFACE = (
    '_feat_in',
    'att_context_size',
    'att_context_size_all',
    'cache_aware_stream_step',
    'd_model',
    'get_initial_cache_state',
    'pre_encode',
    'setup_streaming_params',
    'subsampling_factor',
)


def _require_asr_encoder_interface(encoder) -> None:
    """Raise if ``encoder`` cannot stand in as a PE ASR branch."""
    missing = [name for name in _ASR_ENCODER_INTERFACE if not hasattr(encoder, name)]
    if missing:
        raise TypeError(
            f"`asr_encoder_cfg._target_` instantiated {type(encoder).__name__}, which is missing "
            f"{missing} and so cannot serve as a ParallelExpertEncoder ASR branch. The branch must "
            f"be a cache-aware encoder (e.g. ConformerEncoder, StreamingTransformerEncoder)."
        )


def _branch_grad_mode(frozen: bool) -> torch.set_grad_enabled:
    """Grad mode for running one branch: the caller's, and always off for a frozen branch.

    It never turns autograd on. Under an outer ``torch.no_grad()`` (validation, generation), a trainable
    branch would otherwise build a graph nobody uses and hold its activations. Under an outer
    ``torch.inference_mode()`` (Lightning's default for ``validate``), an autograd Function in the branch,
    such as the fused Triton subsampling, would then try to save inference tensors for backward and raise.

    Args:
        frozen (bool): The branch is frozen (``freeze_asr`` / ``freeze_diar``).

    Returns:
        torch.set_grad_enabled: The context manager to run the branch in.
    """
    return torch.set_grad_enabled(torch.is_grad_enabled() and not frozen)
