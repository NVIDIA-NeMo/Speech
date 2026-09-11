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

"""Parallel Expert Speech Encoder (PEE).

Combines speech, speaker, and sound encoders inside a
:class:`GGEMMTransformerEncoder`. Expert widths, attention layouts, layer counts,
and feed-forward dimensions are read from their configs rather than assumed by
this module. The grouped path combines compatible operations and structurally
pads narrower feed-forward units when required, preserving each expert's native
result.

Generic grouped-GEMM execution, padding, and shape bucketing live in
``ggemm_transformer_encoder.py``. This module owns PEE-specific roles, task
heads, state fusion, streaming behavior, and checkpoint interpretation.

The speaker expert's states go through the Sortformer head
(``encoder_proj`` -> ``forward_speaker_sigmoids``) to produce per-frame speaker
activities, which are fused into the speech states (LayerNorm + sinusoidal
speaker kernel + ADD).

The sound expert is merged in per ``merge_sound_expert_to_asr``:

* ``False`` -- the SoundToken route. The sound expert's CTC
  head reads per-frame ``<ev:...>`` event and ``<sty:stt|end:...>`` style-span
  probabilities out of its states, and those are thresholded, LayerNorm-ed and
  injected through sinusoidal kernels: the direct analogue of the speaker branch, on
  disjoint sets of sinusoid rows. What reaches the ASR states is only the tags, a
  signal of rank <= ``n_sound_events + n_sound_styles``.
* ``True`` -- the whole sound representation instead: its encoder states are
  LayerNorm-ed, scaled and added onto the ASR states.

Speakers, events, and styles are separate families. Each has its own LayerNorm,
sinusoid-row block, and scale so one family's activity does not alter another's
representation or gain.

Every family's rows are strided and its kernel calibrated so a
single active tag injects a vector of norm ``sqrt(d_model)``. That makes each ``*_scale``
a plain fraction of the ASR state magnitude, unaffected by how many tags the family holds
or which rows it was given -- so widening a family, or moving it, no longer silently
changes how loudly it speaks.

Order matters: sound joins the ASR states FIRST, so speech + sound together form
the backbone that ``asr_norm`` normalizes, and the speaker kernel is then added on
top of that normalized sum.

The speaker kernel is built from thresholded activities, so gradients do not pass
through that fusion operation. Expert freezing remains independently configurable.

Offline and online encoding use the same fusion:

* :meth:`ParallelExpertEncoder._forward` -- one pass over the whole utterance.
  All experts share ``T``, so there is no streaming prefix.
* :meth:`ParallelExpertEncoder._forward_online` -- windowed long-form decoding
  where the speaker expert additionally attends over its streaming cache. The
  cache is passed as a ``prefix`` to ``forward_packed``, which right-pads the
  speech and sound experts to the speaker's longer ``T`` and masks each expert
  with its own length.

I/O matches :class:`ConformerEncoder` (drop-in). The shared mel input is normalized
once before packed execution so every expert receives identical features. Only self-contained PE bundles
(inline ``speech_expert_cfg`` / ``speaker_expert_cfg`` / ``sound_expert_cfg`` /
``sortformer_modules_cfg`` in ``model_config.yaml``) are supported.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
import os
import re
import shutil
import tarfile
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from nemo.collections.asr.modules.ggemm_transformer_encoder import GGEMMTransformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.module import freeze, unfreeze
from nemo.utils import logging
from nemo.utils.decorators import experimental

__all__ = [
    'GGEMMParallelExpertEncoder',
    'GGEMMParallelExpertEncoderPT',
    'ParallelExpertEncoder',
    'ParallelExpertEncoderPT',
]

# Roles follow packed-attention order.
EXPERT_ROLES = ('speech', 'speaker', 'sound')
# Encoder/head role represented by each branch.
EXPERT_TASKS = {'speech': 'asr_encoder', 'speaker': 'diarization', 'sound': 'sound_ctc'}


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    """Temporarily set the global default float dtype.

    Makes ``SortformerModules.init_streaming_state`` allocate its dtype-less
    speaker-cache / FIFO buffers in the speaker expert's dtype, avoiding fp32/bf16
    mismatch.

    Args:
        dtype (torch.dtype): Floating-point dtype to set as the global default.
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

    Skips cross-rank ``all_reduce``s in the Sortformer streaming helpers, which are
    unnecessary and unsafe for single-recording inference (e.g. a vLLM worker).
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


def _disable_max_seq_length_sync(module: nn.Module) -> None:
    """Turn off ``sync_max_audio_length`` on every encoder under ``module``.

    That flag makes ``update_max_seq_length`` issue an ``all_reduce`` on the
    **default** process group. Any such collective inside a data-dependent branch lets
    ranks emit different numbers of default-PG collectives on the same step, which
    NCCL cannot match positionally -- the run then deadlocks until the watchdog
    aborts it.

    Args:
        module (nn.Module): Root module whose encoder descendants are scanned.
    """
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def _clone_config(config: Optional[DictConfig]) -> Optional[DictConfig]:
    """Deep-copy a ``DictConfig`` without resolving interpolations.

    ``from_config_dict`` mutates its input in place, so sub-target builders get a copy.

    Args:
        config (DictConfig, optional): Config to deep-copy, or ``None``.

    Returns:
        A deep copy of ``config``, or ``None`` if the input was ``None``.
    """
    if config is None:
        return None
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _unwrap_cls(cls):
    """Step through a ``wrapt`` proxy chain to the underlying class.

    ``inspect.unwrap`` is not enough: NeMo's ``@experimental`` wraps the *class* in a
    ``wrapt`` proxy that forwards ``__repr__``/``__class__``, so the object
    ``inspect.unwrap`` returns still resolves ``__init__`` to the proxy's
    ``(*args, **kwargs)``.

    Args:
        cls (type): Class object that may be wrapped by ``@experimental``/``wrapt``.

    Returns:
        The underlying class after stepping through proxy wrappers.
    """
    seen = {id(cls)}
    cur = cls
    for _ in range(8):
        nxt = getattr(cur, "__wrapped__", None)
        if nxt is None or id(nxt) in seen:
            return cur
        seen.add(id(nxt))
        cur = nxt
    return cur


def _resolve_target(target: str) -> type:
    """Resolve a Hydra ``_target_`` string to a class.

    Args:
        target (str): Hydra ``_target_`` string (``module.path.ClassName``).

    Returns:
        The resolved class object.
    """
    module_path, _, cls_name = target.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import _target_ '{target}'.") from exc


def _init_param_names(cls) -> set:
    """Names ``cls.__init__`` accepts.

    Refuses to fall back to a ``**kwargs`` signature: filtering config keys against
    one would drop *every* key and silently build a default-shaped encoder.

    Args:
        cls (type): Class whose ``__init__`` signature is introspected.

    Returns:
        Set of parameter names accepted by ``cls.__init__`` (excluding ``self``).
    """
    for cand in (cls, _unwrap_cls(cls)):
        try:
            params = inspect.signature(cand.__init__).parameters
        except (TypeError, ValueError):
            continue
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return set(params) - {"self"}
    raise TypeError(
        f"Could not introspect a concrete __init__ signature for {cls!r}; refusing to "
        "filter config keys against a **kwargs signature (would build a default module)."
    )


def _build_from_cfg(cfg: DictConfig, what: str) -> nn.Module:
    """Instantiate a module from a config section carrying ``_target_``.

    Keys the installed class does not accept are reported rather than silently
    ignored -- tuning-only extras (e.g. ``flex_kernel_options``) are harmless, but
    anything architectural showing up here means a branch mismatch.

    Args:
        cfg (DictConfig): Config mapping carrying a ``_target_`` key.
        what (str): Label used in error messages (e.g. ``"speech_expert_cfg"``).

    Returns:
        Instantiated ``nn.Module``.
    """
    if cfg is None or '_target_' not in cfg:
        raise ValueError(f"{what} config must be a mapping carrying a `_target_` key.")
    cls = _resolve_target(str(cfg['_target_']))
    raw = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items() if k != '_target_'}
    accepted = _init_param_names(cls)
    kwargs = {k: v for k, v in raw.items() if k in accepted}
    dropped = sorted(set(raw) - accepted)
    if dropped:
        logging.warning(
            "[ParallelExpertEncoder] %s: ignoring config keys not accepted by %s: %s",
            what,
            getattr(_unwrap_cls(cls), '__name__', cls),
            dropped,
        )
    return cls(**kwargs)


@experimental
class ParallelExpertEncoderPT(ModelPT):
    """ModelPT shell so a :class:`ParallelExpertEncoder` can be saved/restored as a
    ``.nemo`` archive (inline ``speech_expert_cfg`` / ``speaker_expert_cfg`` /
    ``sound_expert_cfg`` / ``sortformer_modules_cfg``).
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        """Build the inner :class:`ParallelExpertEncoder` from a PE bundle config.

        Args:
            cfg (DictConfig): Self-contained PE bundle config (inline expert configs).
            trainer (Trainer, optional): Lightning trainer when used as a :class:`ModelPT` shell.
        """
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            speech_expert_cfg=self._cfg.get('speech_expert_cfg', None),
            speaker_expert_cfg=self._cfg.get('speaker_expert_cfg', None),
            sound_expert_cfg=self._cfg.get('sound_expert_cfg', None),
            sortformer_modules_cfg=self._cfg.get('sortformer_modules_cfg', None),
            sound_ctc_head_cfg=self._cfg.get('sound_ctc_head_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', 'per_feature'),
            freeze_speaker=self._cfg.get('freeze_speaker', True),
            freeze_speech=self._cfg.get('freeze_speech', False),
            freeze_sound=self._cfg.get('freeze_sound', True),
            online_inference_length=self._cfg.get('online_inference_length', 375),
            chunk_left_context=self._cfg.get('chunk_left_context', 50),
            chunk_right_context=self._cfg.get('chunk_right_context', 50),
            diar_fifo_len=self._cfg.get('diar_fifo_len', 0),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', 375),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', 200),
            missing_rttm_target=self._cfg.get('missing_rttm_target', -1.0),
            speaker_activity_threshold=self._cfg.get('speaker_activity_threshold', 0.5),
            spk_kernel_scale=self._cfg.get('spk_kernel_scale', None),
            spk_kernel_row_stride=self._cfg.get('spk_kernel_row_stride', 1),
            spk_kernel_calibrate=self._cfg.get('spk_kernel_calibrate', False),
            sound_event_token_prefix=self._cfg.get('sound_event_token_prefix', '<ev:'),
            sound_event_token_pattern=self._cfg.get('sound_event_token_pattern', r'^<ev:[^>]+>$'),
            sound_style_token_prefix=self._cfg.get('sound_style_token_prefix', '<sty:'),
            sound_style_token_pattern=self._cfg.get('sound_style_token_pattern', r'^<sty:(?:stt|end):[^>]+>$'),
            tag_row_stride=self._cfg.get('tag_row_stride', 16),
            speaker_row_offset=self._cfg.get('speaker_row_offset', 0),
            sound_event_row_offset=self._cfg.get('sound_event_row_offset', 512),
            sound_style_row_offset=self._cfg.get('sound_style_row_offset', 1024),
            legacy_spk_kernel_scale=self._cfg.get('legacy_spk_kernel_scale', 1.0),
            calibrated_spk_kernel_scale=self._cfg.get('calibrated_spk_kernel_scale', 0.75),
            sync_max_audio_length=self._cfg.get('sync_max_audio_length', False),
            always_run_diarization=self._cfg.get('always_run_diarization', True),
            moe_mode=self._cfg.get('moe_mode', 'dense'),
            fused_forward_in_training=self._cfg.get('fused_forward_in_training', False),
            ggemm_backend=self._cfg.get('ggemm_backend', 'baddbmm'),
            online_prefix_mode=self._cfg.get('online_prefix_mode', 'replace'),
            merge_sound_expert_to_asr=self._cfg.get('merge_sound_expert_to_asr', False),
            sound_merge_scale=self._cfg.get('sound_merge_scale', 0.3),
            sound_event_threshold=self._cfg.get('sound_event_threshold', 0.5),
            sound_kernel_scale=self._cfg.get('sound_kernel_scale', 0.75),
            inject_sound_styles=self._cfg.get('inject_sound_styles', True),
            sound_style_scale=self._cfg.get('sound_style_scale', 0.75),
        )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """Return pretrained PE bundles (none registered by default).

        Returns:
            Empty list; PE bundles are loaded from user-supplied ``.nemo`` archives.
        """
        return []

    def setup_training_data(self, train_data_config: Union[DictConfig, dict]):
        """No-op placeholder for :class:`ModelPT` compatibility.

        Args:
            train_data_config (DictConfig | dict): Training data configuration (unused).
        """
        pass

    def setup_validation_data(self, val_data_config: Union[DictConfig, dict]):
        """No-op placeholder for :class:`ModelPT` compatibility.

        Args:
            val_data_config (DictConfig | dict): Validation data configuration (unused).
        """
        pass

    @staticmethod
    def extract_encoder_state_dict(
        full_state_dict: dict[str, torch.Tensor], encoder_attr: str = 'encoder'
    ) -> dict[str, torch.Tensor]:
        """Extract one PEE expert encoder from a complete model state dict.

        PEE experts originate from full model checkpoints, whose state dicts also
        contain preprocessors, decoders, and task heads. This selects keys below
        ``<encoder_attr>.`` and removes that prefix so the result can be loaded into
        the matching encoder stored in :class:`GGEMMTransformerEncoder`.

        Args:
            full_state_dict (dict[str, torch.Tensor]): State dict of the source expert model.
            encoder_attr (str): Attribute that owns the required encoder. Use ``encoder``
                for ASR experts and ``transformer_encoder`` for the Sortformer
                speaker expert; its ``encoder`` attribute is the upstream
                FastConformer rather than the PEE Transformer encoder.

        Returns:
            dict[str, torch.Tensor]: The selected encoder state dict with ``encoder_attr`` removed.
        """
        prefix = f"{encoder_attr}."
        return {key[len(prefix) :]: value for key, value in full_state_dict.items() if key.startswith(prefix)}

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
    ) -> ParallelExpertEncoder:
        """Load a self-contained PE bundle and return its inner encoder.

        Follows the standard NeMo :class:`~nemo.core.classes.common.Model`
        convention for resolving a checkpoint reference:

        * a local ``.nemo`` file is read directly by
          :meth:`_load_encoder_from_archive` -- ``ModelPT.restore_from`` cannot
          instantiate an ``@experimental`` target (see that method's docstring);
        * otherwise ``model_path_or_name`` is treated as a pretrained model
          identifier -- a HuggingFace Hub repo id (``{repo}/{name}``) or an NGC
          alias -- and resolved with :meth:`Model.from_pretrained`, which
          downloads/caches the ``.nemo`` (honouring the HuggingFace cache and
          ``HF_HUB_OFFLINE``, so a prefetched cache works on offline nodes).

        Args:
            model_path_or_name (str): Local ``.nemo`` path or pretrained model id.
            map_location (str | torch.device): Device to map weights onto.
            strict (bool): Enforce exact state-dict match.

        Returns:
            The restored :class:`ParallelExpertEncoder`.
        """
        if (
            isinstance(model_path_or_name, str)
            and model_path_or_name.endswith('.nemo')
            and os.path.isfile(model_path_or_name)
        ):
            return cls._load_encoder_from_archive(model_path_or_name, map_location=map_location, strict=strict)
        bundle = cls.from_pretrained(
            model_name=model_path_or_name,
            map_location=map_location,
            strict=strict,
        )
        return bundle.encoder

    @classmethod
    def _load_encoder_from_archive(
        cls,
        nemo_path: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
    ) -> ParallelExpertEncoder:
        """Build the encoder straight from a ``.nemo`` archive, bypassing ``restore_from``.

        ``ModelPT.restore_from`` routes the bundle's ``target:`` through NeMo's
        config-instantiation allow-list, which rejects any ``@experimental``-decorated
        class: the decorator wraps the class in a ``wrapt`` proxy, ``issubclass()``
        raises ``TypeError`` on it, and the allow-list turns that into "unsafe target".
        Both :class:`ParallelExpertEncoderPT` and the flex ``TransformerEncoder`` the
        experts are built from are decorated, so a PE bundle can never be restored the
        normal way. Fixing that belongs in ``nemo/core/classes/common.py``, which is
        deliberately out of scope here.

        Args:
            nemo_path (str): Local ``.nemo`` bundle.
            map_location (str | torch.device): Device to map weights onto.
            strict (bool): Enforce exact state-dict match.

        Returns:
            ParallelExpertEncoder: The restored inner encoder.
        """
        import tempfile

        if not cls.is_pe_nemo(nemo_path):
            raise ValueError(f"{nemo_path!r} is not a ParallelExpertEncoderPT .nemo bundle.")

        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(nemo_path, mode='r') as tf:
                members = {os.path.basename(m.name): m for m in tf.getmembers() if m.isfile()}
                for name in ('model_config.yaml', 'model_weights.ckpt'):
                    if name not in members:
                        raise RuntimeError(f"{nemo_path} is missing '{name}'.")
                    member = members[name]
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        raise RuntimeError(f"Could not read '{name}' from {nemo_path}.")
                    with open(os.path.join(td, name), 'wb') as out:
                        shutil.copyfileobj(fobj, out)

            cfg = OmegaConf.load(os.path.join(td, 'model_config.yaml'))
            shell = cls(cfg=cfg, trainer=None)
            state = torch.load(
                os.path.join(td, 'model_weights.ckpt'),
                map_location=map_location,
                weights_only=True,
            )

        # The bundle stores the PT shell's state dict, so the encoder's own tensors
        # sit under an `encoder.` prefix. Anything else belongs to the shell and has
        # no counterpart on the bare encoder.
        prefix = 'encoder.'
        enc_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if not enc_state:
            raise RuntimeError(
                f"No '{prefix}*' tensors found in {nemo_path}; the bundle does not look "
                "like a saved ParallelExpertEncoderPT."
            )
        missing, unexpected = shell.encoder.load_state_dict(enc_state, strict=strict)
        if missing or unexpected:
            logging.warning(
                "[ParallelExpertEncoder] load_from_nemo(%s): %d missing / %d unexpected keys.",
                nemo_path,
                len(missing),
                len(unexpected),
            )
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

        The template must describe the same architecture (speech ``d_model``,
        ``num_spks``); mismatches raise :class:`ValueError` fail-fast.

        Args:
            encoder (ParallelExpertEncoder): The encoder whose weights are persisted.
            output_nemo_path (str): Destination ``.nemo`` path.
            template_bundle_path (str): Existing PE ``.nemo`` whose ``model_config.yaml`` is reused.
        """
        if not isinstance(encoder, ParallelExpertEncoder):
            raise TypeError(f"save_to_nemo expects a ParallelExpertEncoder, got {type(encoder).__name__}")
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

        missing = [
            key
            for key in ('speech_expert_cfg', 'speaker_expert_cfg', 'sound_expert_cfg', 'sortformer_modules_cfg')
            if template_cfg.get(key, None) in (None, {}, '')
        ]
        if missing:
            raise ValueError(
                f"Template bundle {template_bundle_path} is not self-contained (missing {missing}); "
                "it cannot be used as a save template."
            )

        # Only required on the SoundToken path, so it is not in `missing` above -- but
        # without it the saved bundle would carry sound_ctc_head weights that its own
        # config never builds, and fail strict reload.
        if not encoder.merge_sound_expert_to_asr and template_cfg.get('sound_ctc_head_cfg', None) in (None, {}, ''):
            raise ValueError(
                f"Encoder uses merge_sound_expert_to_asr=False (SoundToken injection) but template "
                f"bundle {template_bundle_path} has no sound_ctc_head_cfg; the saved bundle would "
                "not reload. Use a self-contained template with the required sound head config."
            )

        tmpl_d_model = int(template_cfg.speech_expert_cfg.get('d_model', -1))
        tmpl_n_spk = int(template_cfg.sortformer_modules_cfg.get('num_spks', -1))
        enc_d_model = int(encoder.d_model)
        enc_n_spk = int(encoder.n_spk)
        if tmpl_d_model != enc_d_model:
            raise ValueError(
                f"Template speech_expert_cfg.d_model={tmpl_d_model} does not match "
                f"encoder.d_model={enc_d_model}; the saved bundle would fail strict reload."
            )
        if tmpl_n_spk != enc_n_spk:
            raise ValueError(
                f"Template sortformer_modules_cfg.num_spks={tmpl_n_spk} does not match "
                f"encoder.n_spk={enc_n_spk}; the saved bundle would fail strict reload."
            )

        # New bundles use an unambiguous module path; schema dispatch still accepts historical targets.
        template_cfg.target = 'nemo.collections.asr.modules.parallel_expert_encoder_ggemm.ParallelExpertEncoderPT'

        # Fresh PT shell from the template cfg to reuse NeMo's save_to; swap in encoder.
        shell = cls(cfg=template_cfg, trainer=None)
        shell.encoder = encoder
        # Pin `_cfg` to the verbatim template so save_to round-trips it exactly.
        shell._cfg = template_cfg

        shell.save_to(output_nemo_path)
        logging.info(
            "[ParallelExpertEncoder] Saved PE bundle to %s using template config from %s",
            output_nemo_path,
            template_bundle_path,
        )


@experimental
class ParallelExpertEncoder(nn.Module):
    """Parallel expert encoder with :class:`ConformerEncoder`-compatible I/O.

    Reconstructed from inline configs in the PE bundle's ``model_config.yaml``.
    Expert dimensions and attention layouts are derived from those configs.

    Args:
        speech_expert_cfg (DictConfig): Inline config for the speech MoE expert
            whose output forms the ASR backbone.
        speaker_expert_cfg (DictConfig): Inline config for the Sortformer speaker
            expert.
        sound_expert_cfg (DictConfig): Inline config for the sound expert
            that participates in packed execution.
        sortformer_modules_cfg (DictConfig): Inline config for
            :class:`SortformerModules`, which supplies the speaker head
            (``encoder_proj`` + ``forward_speaker_sigmoids``) and the streaming
            speaker-cache logic. Projection dimensions must match the checkpoint.
        sound_ctc_head_cfg (DictConfig, optional): Inline config for the sound expert's
            CTC head (a :class:`ConvASRDecoder`), lifted from the sound checkpoint's
            ``decoder:`` block. Like the Sortformer head this is NOT part of the
            expert's encoder, so it is built here and loaded separately. Its
            ``vocabulary`` is what the ``<ev:...>`` column indices are read from.
            Required when ``merge_sound_expert_to_asr=False``, unused otherwise.
        asr_normalize_type (str, optional): Normalization applied to the shared mel
            input. Every expert receives the same normalized tensor. Pass ``None``
            to feed raw mels.
        freeze_speaker (bool): Freeze the speaker expert and head.
            The speaker kernel is built from a hard threshold on the speaker
            activities, so no gradient reaches this branch through the fusion
            operation.
        freeze_speech (bool): Freeze the speech expert.
        freeze_sound (bool): Freeze the sound expert.
        online_inference_length (int): Generation-time window in encoder output
            frames. Non-positive values disable windowing. Training and validation
            encode in one pass.
        chunk_left_context (int): Left context in encoder output frames.
        chunk_right_context (int): Right context in encoder output frames.
        diar_fifo_len (int): Sortformer streaming FIFO length.
        diar_spkcache_update_period (int): Sortformer streaming
            speaker-cache update period. The effective period cannot be shorter
            than the current chunk.
        diar_spkcache_len (int): Sortformer streaming speaker-cache length.
        missing_rttm_target (float): Sentinel marking rows that should use diarization
            predictions.
        speaker_activity_threshold (float): Binarization threshold applied to RTTM and
            diarization targets before speaker-kernel fusion.
        spk_kernel_scale (float, optional): Weight of the speaker-kernel contribution.
        spk_kernel_row_stride (int): Spacing between the sinusoid rows the speaker kernel
            uses. The saved value determines whether a bundle uses the legacy
            contiguous layout or a strided layout.
        spk_kernel_calibrate (bool): Rescale the speaker kernel so one active speaker
            injects ``sqrt(d_model)``, making ``spk_kernel_scale`` a *fraction of the ASR
            state magnitude* directly comparable to ``sound_kernel_scale`` and independent
            of ``n_spk``. The saved calibration setting must be preserved when
            reloading trained bundles.
        sync_max_audio_length (bool): Let the experts all-reduce their maximum sequence
            length on the default process group. Enable only when every rank is
            guaranteed to run the encoder on every step.
        always_run_diarization (bool): Run the speaker head on every single-pass forward
            instead of gating it on batch content.
        moe_mode (str): ``'dense'`` or ``'topk'`` for the speech MoE inside the grouped
            FFN. Only used on the fused path.
        fused_forward_in_training (bool): Use the fused packed path while training too.
            The per-expert path generally uses less activation memory, while
            inference always uses grouped execution.
        ggemm_backend (str): Grouped-GEMM backend.
        online_prefix_mode (str): How the speaker's streaming cache is spliced in the
            windowed path. ``'replace'`` lets the cache stand in for the speaker's
            leading frames; ``'extend'`` lengthens the speaker sequence and pads the
            other experts. Replacement falls back to extension when insufficient
            preceding context is available.
        merge_sound_expert_to_asr (bool): How the sound expert reaches the ASR states.
            ``False`` selects the **SoundToken** path: the CTC
            head reads per-frame ``<ev:...>`` and ``<sty:stt|end:...>`` probabilities
            out of the sound states, and those are thresholded at
            ``sound_event_threshold``, LayerNorm-ed per family and injected through
            ``sound_token_kernel`` / ``sound_style_kernel`` -- exactly as the speaker
            sigmoids go through ``diar_kernel``. Requires ``sound_ctc_head_cfg``.
            ``True`` instead adds the sound expert's **encoder states**, LayerNorm-ed
            and scaled by ``sound_merge_scale``; this route needs no CTC head.
        sound_merge_scale (float): Relative weight of the sound stream in the merged
            backbone. Both streams are normalized before scaling so this acts as a
            relative gain rather than depending on their native magnitudes.
            Ignored when ``merge_sound_expert_to_asr=False``.
        sound_event_threshold (float): Probability above which an event or style tag
            counts as present before kernel injection. Only read on the SoundToken path.
        sound_kernel_scale (float): Weight of the event-kernel contribution, the sound
            twin of ``spk_kernel_scale`` and on the same calibrated footing: a fraction of
            the ASR state magnitude, independent of how many event tags there are.
            Only read on the SoundToken path.
        inject_sound_styles (bool): Whether to inject ``<sty:stt|end:...>`` span
            delimiters as a separate tag family. When disabled, only event tags
            are injected. Only read on the SoundToken path.
        sound_style_scale (float): Weight of the style-kernel contribution, on the same
            calibrated footing as ``sound_kernel_scale``. It remains independent so
            event and style contributions can be tuned separately. Only read on the
            SoundToken path.
        sound_event_token_prefix (str): Event-token prefix shown in validation errors.
        sound_event_token_pattern (str): Regular expression selecting event tokens from
            the sound-head vocabulary.
        sound_style_token_prefix (str): Style-token prefix shown in validation errors.
        sound_style_token_pattern (str): Regular expression selecting style boundary
            tokens from the sound-head vocabulary.
        tag_row_stride (int): Spacing between sinusoid rows for sound event and style tags.
        speaker_row_offset (int): First sinusoid row reserved for speaker identities.
        sound_event_row_offset (int): First sinusoid row reserved for sound events.
        sound_style_row_offset (int): First sinusoid row reserved for sound styles.
        legacy_spk_kernel_scale (float): Default speaker gain for uncalibrated kernels.
        calibrated_spk_kernel_scale (float): Default speaker gain for calibrated kernels.
    """

    supports_external_speaker_targets = True
    parallel_expert_encoder_kind = "ggemm"

    def __init__(
        self,
        speech_expert_cfg: DictConfig,
        speaker_expert_cfg: DictConfig,
        sound_expert_cfg: DictConfig,
        sortformer_modules_cfg: DictConfig,
        sound_ctc_head_cfg: Optional[DictConfig] = None,
        asr_normalize_type: Optional[str] = 'per_feature',
        freeze_speaker: bool = True,
        freeze_speech: bool = False,
        freeze_sound: bool = True,
        online_inference_length: int = 375,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: int = 0,
        diar_spkcache_update_period: int = 375,
        diar_spkcache_len: int = 200,
        missing_rttm_target: float = -1.0,
        speaker_activity_threshold: float = 0.5,
        spk_kernel_scale: Optional[float] = None,
        spk_kernel_row_stride: int = 1,
        spk_kernel_calibrate: bool = False,
        sync_max_audio_length: bool = False,
        always_run_diarization: bool = True,
        moe_mode: str = 'dense',
        fused_forward_in_training: bool = False,
        ggemm_backend: str = 'baddbmm',
        online_prefix_mode: str = 'replace',
        merge_sound_expert_to_asr: bool = False,
        sound_merge_scale: float = 0.3,
        sound_event_threshold: float = 0.5,
        sound_kernel_scale: float = 0.75,
        inject_sound_styles: bool = True,
        sound_style_scale: float = 0.75,
        sound_event_token_prefix: str = '<ev:',
        sound_event_token_pattern: str = r'^<ev:[^>]+>$',
        sound_style_token_prefix: str = '<sty:',
        sound_style_token_pattern: str = r'^<sty:(?:stt|end):[^>]+>$',
        tag_row_stride: int = 16,
        speaker_row_offset: int = 0,
        sound_event_row_offset: int = 512,
        sound_style_row_offset: int = 1024,
        legacy_spk_kernel_scale: float = 1.0,
        calibrated_spk_kernel_scale: float = 0.75,
    ):
        """Construct experts, fusion heads, and streaming configuration.

        Args:
            speech_expert_cfg, speaker_expert_cfg, sound_expert_cfg,
            sortformer_modules_cfg, and remaining keyword arguments: see the class
            docstring for the full argument list and descriptions.
        """
        super().__init__()

        cfgs = {
            'speech': speech_expert_cfg,
            'speaker': speaker_expert_cfg,
            'sound': sound_expert_cfg,
        }
        absent = [role for role, cfg in cfgs.items() if cfg is None]
        if absent or sortformer_modules_cfg is None:
            raise ValueError(
                "ParallelExpertEncoder requires speech/speaker/sound expert configs and "
                f"sortformer_modules_cfg; missing: {absent + (['sortformer_modules_cfg'] if sortformer_modules_cfg is None else [])}. "
                "Self-contained PE bundles supply these inline in their model_config.yaml."
            )

        experts = {role: _build_from_cfg(_clone_config(cfgs[role]), f"{role}_expert_cfg") for role in EXPERT_ROLES}
        self.pee = GGEMMTransformerEncoder(experts)
        self.expert_tasks = dict(EXPERT_TASKS)

        # The Sortformer head. `ParallelExpertEncoderPT.extract_encoder_state_dict`
        # pulls only the speaker encoder.
        self.sortformer_modules = _build_from_cfg(_clone_config(sortformer_modules_cfg), 'sortformer_modules_cfg')

        speaker_d_model = self.pee.experts['speaker'].d_model
        if int(self.sortformer_modules.fc_d_model) != int(speaker_d_model):
            raise ValueError(
                f"sortformer_modules.fc_d_model={self.sortformer_modules.fc_d_model} must match the "
                f"speaker expert d_model={speaker_d_model}; the head projects the speaker states and "
                "the streaming cache is allocated at fc_d_model."
            )

        self.asr_normalize_type = asr_normalize_type
        self._feat_in = self.pee.experts['speech']._feat_in

        # The experts' `update_max_seq_length` all-reduces on the default process
        # group, and neither the online loop (data-dependent window count) nor a
        # content-gated head is reached uniformly by every rank.
        self.sync_max_audio_length = bool(sync_max_audio_length)
        if not self.sync_max_audio_length:
            _disable_max_seq_length_sync(self.pee)
        self.always_run_diarization = bool(always_run_diarization)

        self.freeze_speaker = freeze_speaker
        self.freeze_speech = freeze_speech
        self.freeze_sound = freeze_sound

        self.moe_mode = moe_mode
        self.fused_forward_in_training = bool(fused_forward_in_training)
        # PEE owns this opt-in so the native TransformerEncoder path remains
        # unchanged. When enabled, the per-expert training forwards are
        # recomputed during backward at the expert boundary.
        self.activation_checkpointing = False
        self.ggemm_backend = ggemm_backend
        if online_prefix_mode not in ('replace', 'extend'):
            raise ValueError(f"online_prefix_mode must be 'replace' or 'extend', got {online_prefix_mode!r}.")
        self.online_prefix_mode = online_prefix_mode

        # Long-form / online inference configuration.
        self.online_inference_length = int(online_inference_length)
        # Opt-in, and never on during training or validation: see `online_inference`.
        self.online_inference_enabled = False
        # Overlap-and-trim context (output frames), shared by every expert.
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        # Online-inference window + context in input mel frames (constant per session).
        self.chunk_feat_len = self.online_inference_length * self.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.subsampling_factor
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)
        self.missing_rttm_target = float(missing_rttm_target)
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.spk_kernel_calibrate = bool(spk_kernel_calibrate)
        self.sound_event_token_prefix = str(sound_event_token_prefix)
        self.sound_style_token_prefix = str(sound_style_token_prefix)
        self.sound_event_token_pattern = str(sound_event_token_pattern)
        self.sound_style_token_pattern = str(sound_style_token_pattern)
        try:
            self.sound_event_token_re = re.compile(self.sound_event_token_pattern)
            self.sound_style_token_re = re.compile(self.sound_style_token_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid sound token regular expression: {exc}") from exc

        self.tag_row_stride = max(1, int(tag_row_stride))
        self.speaker_row_offset = int(speaker_row_offset)
        self.sound_event_row_offset = int(sound_event_row_offset)
        self.sound_style_row_offset = int(sound_style_row_offset)
        if min(self.speaker_row_offset, self.sound_event_row_offset, self.sound_style_row_offset) < 0:
            raise ValueError("Speaker, event, and style sinusoid row offsets must be non-negative.")

        self.legacy_spk_kernel_scale = float(legacy_spk_kernel_scale)
        self.calibrated_spk_kernel_scale = float(calibrated_spk_kernel_scale)
        # The default scale follows the saved layout because calibrated and legacy
        # kernels express gain in different units.
        if spk_kernel_scale is None:
            spk_kernel_scale = (
                self.calibrated_spk_kernel_scale if self.spk_kernel_calibrate else self.legacy_spk_kernel_scale
            )
        self.spk_kernel_scale = float(spk_kernel_scale)

        self.n_spk = int(self.sortformer_modules.n_spk)
        # The speech MoE expert is the backbone the speaker kernel is fused into.
        self.asr_d_model = int(self.pee.experts['speech'].d_model)

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.spk_kernel_row_stride = max(1, int(spk_kernel_row_stride))
        spk_last_row = self.speaker_row_offset + self.spk_kernel_row_stride * (self.n_spk - 1)
        if spk_last_row >= self.sound_event_row_offset:
            raise ValueError(
                f"n_spk={self.n_spk} at spk_kernel_row_stride={self.spk_kernel_row_stride} "
                f"reaches sinusoid row {spk_last_row}, which is inside the sound event "
                f"block at row {self.sound_event_row_offset}. Speakers and sound tags would "
                "share rows and become indistinguishable; lower the stride or move the "
                "sound blocks."
            )
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

        # --- sound expert -> ASR merge -------------------------------------
        self.merge_sound_expert_to_asr = bool(merge_sound_expert_to_asr)
        self.sound_merge_scale = float(sound_merge_scale)
        self.sound_event_threshold = float(sound_event_threshold)
        self.sound_kernel_scale = float(sound_kernel_scale)
        self.inject_sound_styles = bool(inject_sound_styles)
        self.sound_style_scale = float(sound_style_scale)
        self.sound_ctc_head = None
        self.n_sound_events = 0
        self.n_sound_styles = 0
        if not self.merge_sound_expert_to_asr:
            # SoundToken path: the sound expert reaches ASR as discrete EVENT TOKENS
            # rather than as encoder states. Exactly the speaker branch's shape --
            # posteriors -> threshold -> LayerNorm -> matmul(kernel) -> scaled add --
            # with the CTC head standing in for forward_speaker_sigmoids.
            if sound_ctc_head_cfg is None:
                raise ValueError(
                    "merge_sound_expert_to_asr=False needs sound_ctc_head_cfg: the event "
                    "tokens come from the sound expert's CTC head, which is not part of "
                    "its encoder. Self-contained bundles carry it as the sound "
                    "checkpoint's `decoder:` block. Pass merge_sound_expert_to_asr="
                    "True to use the encoder-state merge instead, which needs no head."
                )
            self.sound_ctc_head = _build_from_cfg(_clone_config(sound_ctc_head_cfg), 'sound_ctc_head_cfg')
            vocabulary = list(sound_ctc_head_cfg.get('vocabulary') or ())
            if not vocabulary:
                raise ValueError(
                    "sound_ctc_head_cfg has no `vocabulary`; the tag column indices are "
                    "read from it and cannot be guessed from num_classes."
                )
            # Read the tag ids out of the vocabulary rather than hard-coding them: they
            # are an artifact of how the sound expert's tokenizer was built and will move
            # the next time it is retrained.
            event_ids = [i for i, tok in enumerate(vocabulary) if self.sound_event_token_re.match(str(tok))]
            if not event_ids:
                raise ValueError(
                    f"sound_ctc_head_cfg.vocabulary has no {self.sound_event_token_prefix}... "
                    f"tokens among its {len(vocabulary)} entries, so there is nothing to "
                    "inject. Is this the CTC-head sound expert?"
                )
            self.sound_event_tokens = tuple(str(vocabulary[i]) for i in event_ids)
            self.n_sound_events = len(event_ids)
            self.register_buffer("sound_event_token_ids", torch.tensor(event_ids, dtype=torch.long), persistent=False)

            style_ids = []
            if self.inject_sound_styles:
                style_ids = [i for i, tok in enumerate(vocabulary) if self.sound_style_token_re.match(str(tok))]
                if not style_ids:
                    raise ValueError(
                        f"inject_sound_styles=True but the vocabulary has no "
                        f"{self.sound_style_token_prefix}stt|end:... tokens among its "
                        f"{len(vocabulary)} entries. Pass inject_sound_styles=False to "
                        "inject only the event tags."
                    )
            self.sound_style_tokens = tuple(str(vocabulary[i]) for i in style_ids)
            self.n_sound_styles = len(style_ids)
            self.register_buffer("sound_style_token_ids", torch.tensor(style_ids, dtype=torch.long), persistent=False)

            # Events and styles are separate families, each with its own LayerNorm,
            # gain, and disjoint block of sinusoid rows.
            event_last_row = self.sound_event_row_offset + self.tag_row_stride * (self.n_sound_events - 1)
            if self.n_sound_styles and event_last_row >= self.sound_style_row_offset:
                raise ValueError(
                    f"n_sound_events={self.n_sound_events} at tag_row_stride={self.tag_row_stride} "
                    f"reaches sinusoid row {event_last_row}, which is inside the sound style "
                    f"block at row {self.sound_style_row_offset}. Lower the stride or move "
                    "the style block."
                )

            self.sound_token_norm = nn.LayerNorm(self.n_sound_events)
            self.register_buffer(
                "sound_token_kernel",
                self._build_tag_kernel(
                    self.n_sound_events,
                    self.sound_event_row_offset,
                    self.asr_d_model,
                    stride=self.tag_row_stride,
                ),
                persistent=False,
            )
            if self.n_sound_styles:
                self.sound_style_norm = nn.LayerNorm(self.n_sound_styles)
                self.register_buffer(
                    "sound_style_kernel",
                    self._build_tag_kernel(
                        self.n_sound_styles,
                        self.sound_style_row_offset,
                        self.asr_d_model,
                        stride=self.tag_row_stride,
                    ),
                    persistent=False,
                )

            # Frozen, like the speaker head. The threshold below is a hard binarization,
            # so no gradient would survive the fusion anyway; freezing makes that
            # explicit and keeps the head out of the optimizer.
            self.sound_ctc_head.eval()
            for p in self.sound_ctc_head.parameters():
                p.requires_grad = False

            # The same argument applies to the expert behind the head: on this route it
            # reaches ASR only through those binarized tags, so it cannot receive
            # gradient either. Leaving it unfrozen is not wrong, just inert -- and it
            # allocates optimizer state for parameters that cannot move. Warn rather
            # than override, since an auxiliary sound loss may train it separately.
            if not freeze_sound:
                logging.warning(
                    "merge_sound_expert_to_asr=False with freeze_sound=False: the sound "
                    "expert reaches the ASR states only through hard-thresholded event "
                    "tags, so it will receive NO gradient from this path and its "
                    "optimizer state is wasted. Set freeze_sound=True unless an "
                    "auxiliary sound loss trains it separately."
                )

        else:
            sound_d_model = int(self.pee.experts['sound'].d_model)
            if sound_d_model != self.asr_d_model:
                raise ValueError(
                    f"merge_sound_expert_to_asr requires the sound expert d_model "
                    f"({sound_d_model}) to match the speech expert d_model ({self.asr_d_model}); "
                    "the merge is an elementwise add onto the ASR states."
                )
            # Normalize both streams before the add so `sound_merge_scale` is a
            # relative gain rather than depending on checkpoint-specific activation
            # magnitudes.
            #
            # The speech-side norm is affine-free: it exists only to fix the scale, and
            # `asr_norm` right after the merge already carries a learnable affine. The
            # sound-side norm keeps its affine so training can still adjust the sound gain.
            #
            # Built only on this branch: on the SoundToken branch they would be
            # parameters in the state dict that no forward pass ever reads.
            self.merge_speech_norm = nn.LayerNorm(self.asr_d_model, elementwise_affine=False)
            self.sound_norm = nn.LayerNorm(self.asr_d_model)

        self._apply_freezing()

    def freeze_experts(self, *roles: str) -> None:
        """Permanently freeze selected expert branches for this encoder instance.

        The matching ``freeze_<role>`` flags are set as well as ``requires_grad`` so
        later calls to :meth:`train` or :meth:`unfreeze` preserve the policy.

        Args:
            roles (str): Expert roles to freeze (``'speech'``, ``'speaker'``, or ``'sound'``).
        """
        unknown = set(roles) - set(EXPERT_ROLES)
        if unknown:
            raise ValueError(f"Unknown expert roles {sorted(unknown)}; expected roles from {EXPERT_ROLES}.")
        for role in roles:
            setattr(self, f'freeze_{role}', True)
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Put each frozen branch in eval and drop its grads."""
        frozen = {
            'speech': self.freeze_speech,
            'speaker': self.freeze_speaker,
            'sound': self.freeze_sound,
        }
        for role, is_frozen in frozen.items():
            if not is_frozen:
                continue
            expert = self.pee.experts[role]
            expert.eval()
            for p in expert.parameters():
                p.requires_grad_(False)
        if self.freeze_speaker:
            # The head travels with the speaker expert.
            self.sortformer_modules.eval()
            for p in self.sortformer_modules.parameters():
                p.requires_grad_(False)
        if self.sound_ctc_head is not None:
            # Unconditionally frozen, unlike the speaker head: the event tags are
            # binarized before the kernel, so nothing could train it through the fusion.
            self.sound_ctc_head.eval()
            for p in self.sound_ctc_head.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True) -> "ParallelExpertEncoder":
        """Set training mode, but keep frozen experts in eval.

        The parent ``model.train()`` recurses into every sub-module, which would
        re-enable dropout in a frozen branch. This re-asserts ``eval()`` on the frozen
        experts so their outputs stay deterministic.

        Args:
            mode (bool): Whether to set training mode (``True``) or eval mode (``False``).

        Returns:
            ParallelExpertEncoder: ``self``, matching ``nn.Module.train``.
        """
        super().train(mode)
        self._apply_freezing()
        return self

    # ConformerEncoder-compatible properties (drop-in for SALM perception).
    @property
    def d_model(self) -> int:
        """Return the speech expert model dimension."""
        return self.asr_d_model

    @property
    def subsampling_factor(self) -> int:
        """Return the speech expert subsampling factor."""
        return self.pee.experts['speech'].subsampling_factor

    @property
    def pre_encode(self):
        """Return the speech expert pre-encoder."""
        return self.pee.experts['speech'].pre_encode

    def get_expert_task(self, expert_name: str) -> Optional[str]:
        """Return the role identifier recorded for ``expert_name``.

        Args:
            expert_name (str): Expert role name (``'speech'``, ``'speaker'``, or ``'sound'``).

        Returns:
            Task identifier string (e.g. ``'diarization'``), or ``None`` if unmapped.
        """
        if expert_name not in self.pee.experts:
            raise KeyError(f"Unknown PEE expert '{expert_name}'. Available: {list(self.pee.experts)}.")
        return self.expert_tasks.get(expert_name)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Recompute each trainable expert in backward instead of storing activations.

        SALM's generic helper wraps ``encoder.layers[i]`` in ``checkpoint_wrapper``,
        which finds nothing here: this module holds no ``layers`` of its own, they live
        one level down in ``pee.experts[role]``. PEE therefore owns an explicit,
        default-off policy and checkpoints each trainable expert's native forward as a
        unit. This keeps ``TransformerEncoder`` behavior and state-dict structure
        untouched. The fused inference path
        (:meth:`GGEMMTransformerEncoder.forward_packed`) is deliberately left alone.

        Args:
            enabled (bool): Turn activation checkpointing on (``True``) or off (``False``).
        """
        self.activation_checkpointing = bool(enabled)
        if enabled and self.fused_forward_in_training:
            logging.warning(
                "set_activation_checkpointing(True) with fused_forward_in_training=True has "
                "no effect: the fused path runs the layers itself instead of calling the "
                "experts' native forwards, so there is nothing to recompute. Leave "
                "fused_forward_in_training at its default False to get the memory back."
            )

    def _forward_all_training(self, audio_signal, length):
        """Run native expert forwards, optionally checkpointed at the PEE boundary.

        Args:
            audio_signal (torch.Tensor): Shared mel input for every expert.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Valid feature length per sample.
                Shape: (B,)

        Returns:
            dict: Per-expert ``(encoded, encoded_len)`` tuples from
            :meth:`GGEMMTransformerEncoder.forward_all`.
        """
        if not self.activation_checkpointing or not torch.is_grad_enabled():
            return self.pee.forward_all(audio_signal, length)

        outputs = {}
        for name in self.pee.expert_names:
            expert = self.pee.experts[name]
            if any(parameter.requires_grad for parameter in expert.parameters()):
                outputs[name] = checkpoint(
                    expert,
                    audio_signal,
                    length,
                    bypass_pre_encode=False,
                    use_reentrant=False,
                )
            else:
                outputs[name] = expert(audio_signal, length, bypass_pre_encode=False)
        return outputs

    # freeze/unfreeze parity (plain nn.Module re-exposing the standalone helpers).
    def freeze(self) -> None:
        """Freeze module parameters."""
        freeze(self)

    def unfreeze(self, partial: bool = False) -> None:
        """Unfreeze trainable branches while preserving expert freeze policy.

        Args:
            partial (bool): If ``True``, unfreeze only parameters that were partially frozen.
        """
        unfreeze(self, partial=partial)
        self._apply_freezing()

    # Fusion helpers
    @staticmethod
    def _build_sinusoid_position_encoding(max_position: int, embedding_dim: int) -> torch.Tensor:
        """Mirror of ``MSEncDecMultiTaskModel.get_sinusoid_position_encoding``.

        Args:
            max_position (int): Number of position rows to generate.
            embedding_dim (int): Embedding width for each row.

        Returns:
            torch.Tensor: Sinusoidal position table.
                Shape: (max_position, embedding_dim)
        """
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
        stride: int = 16,
        calibrate: bool = True,
    ) -> torch.Tensor:
        """Take ``n_tags`` sinusoid rows from ``row_offset``, spaced ``stride``, and calibrate.

        Every tag family slices the one shared table. Configurable row offsets and
        strides keep their identity ranges separate.

        Args:
            n_tags (int): Number of tag identities (rows in the kernel).
            row_offset (int): Starting row index in the shared sinusoid table.
            embedding_dim (int): ASR state dimension the kernel projects into.
            stride (int): Spacing between consecutive tag rows.
            calibrate (bool): Rescale so one active tag injects norm ``sqrt(embedding_dim)``.

        Returns:
            torch.Tensor: Tag kernel.
                Shape: (n_tags, embedding_dim)
        """
        rows = [row_offset + stride * i for i in range(n_tags)]
        table = cls._build_sinusoid_position_encoding(rows[-1] + 1, embedding_dim)
        kernel = table[rows].contiguous()
        if not calibrate:
            return kernel

        # The mean single-tag code is computed with LayerNorm's initial affine values.
        # The norms remain learnable, so training may move away from this starting point.
        eye = torch.eye(n_tags, dtype=kernel.dtype)
        centred = (eye - eye.mean(dim=1, keepdim=True)) / eye.std(dim=1, unbiased=False, keepdim=True)
        mean_norm = (centred @ kernel).norm(dim=1).mean()
        return (kernel * (math.sqrt(embedding_dim) / mean_norm)).contiguous()

    def _check_spk_target_width(self, spk_targets: Optional[torch.Tensor]) -> None:
        """Reject speaker targets whose speaker axis disagrees with ``n_spk``.

        ``n_spk`` is baked into the bundle: it fixes the row count of ``diar_kernel`` and
        the normalized shape of ``diar_norm``, so it cannot adapt to the batch. Left
        unchecked, a narrower target broadcasts against the predictions inside
        :meth:`_fuse_diar_and_asr` and reports a bare size mismatch several frames away
        from the setting that caused it.

        Args:
            spk_targets (torch.Tensor, optional): Speaker activity targets.
                Shape: (B, T, n_spk)
        """
        if spk_targets is None:
            return
        n_given = int(spk_targets.shape[-1])
        if n_given != self.n_spk:
            raise ValueError(
                f"spk_targets carry {n_given} speaker slots but this ParallelExpertEncoder "
                f"was built with n_spk={self.n_spk}. Set the data-side speaker count to "
                f"{self.n_spk} (in SALM: data.multispeaker_cfg.num_speakers) so unused "
                "speakers arrive as empty columns, or export a bundle whose speaker expert "
                f"has n_spk={n_given}."
            )

    @staticmethod
    def _align_diar_frames(spk_targets: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad-by-repeat or truncate ``spk_targets`` to ``target_len`` along time.

        Args:
            spk_targets (torch.Tensor): Speaker activity along time.
                Shape: (B, T, n_spk)
            target_len (int): Desired number of time frames.

        Returns:
            torch.Tensor: Padded or truncated targets.
                Shape: (B, target_len, n_spk)
        """
        cur_len = spk_targets.shape[1]
        if cur_len < target_len:
            last = spk_targets[:, -1:, :]
            spk_targets = torch.cat([spk_targets, last.repeat(1, target_len - cur_len, 1)], dim=1)
        elif cur_len > target_len:
            spk_targets = spk_targets[:, :target_len, :]
        return spk_targets

    def _match_module_io(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast ``tensor`` to the experts' device and parameter dtype.

        Args:
            tensor (torch.Tensor): Input tensor to cast.
                Shape: arbitrary

        Returns:
            torch.Tensor: ``tensor`` on the experts' device and parameter dtype.
                Shape: same as ``tensor``
        """
        param = next(self.pee.parameters(), None)
        if param is None:
            return tensor
        return tensor.to(device=param.device, dtype=param.dtype)

    def _speaker_head(self, speaker_encoded: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Speaker states -> per-frame speaker activity sigmoids.

        Applies the configured ``encoder_proj`` when present, then calls
        ``forward_speaker_sigmoids`` and masks predictions to valid frames. All
        projection dimensions come from ``sortformer_modules_cfg``.

        Args:
            speaker_encoded (torch.Tensor): Speaker expert output.
                Shape: (B, D, T)
            length (torch.Tensor): Valid frame counts.
                Shape: (B,)

        Returns:
            torch.Tensor: Speaker activity probabilities.
                Shape: (B, T, n_spk)
        """
        emb_seq = speaker_encoded.transpose(1, 2)  # (B, T, fc_d_model)
        if self.sortformer_modules.encoder_proj is not None:
            emb_seq = self.sortformer_modules.encoder_proj(emb_seq)  # (B, T, tf_d_model)
        preds = self.sortformer_modules.forward_speaker_sigmoids(emb_seq)  # (B, T, n_spk)
        mask = self.sortformer_modules.length_to_mask(length, emb_seq.shape[1])
        return preds * mask.unsqueeze(-1).to(preds.dtype)

    def _sound_tag_posteriors(self, sound_encoded: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sound states -> per-frame event and style tag probabilities.

        The analogue of :meth:`_speaker_head`: run the frozen CTC head once and keep the
        ``<ev:...>`` and ``<sty:stt|end:...>`` columns.

        The softmax spans the FULL CTC vocabulary before the columns are selected -- the
        head is trained as a distribution over all 13k tokens, so a bare softmax over the
        tag columns alone would renormalize away the (usually dominant) mass on blank and
        on ordinary word pieces, and report near-certainty on the most likely tag for
        every frame including silence.

        Args:
            sound_encoded (torch.Tensor): Sound expert output.
                Shape: (B, D, T)

        Returns:
            tuple: ``(events, styles)`` where ``events`` has shape ``(B, T, n_sound_events)``
            and ``styles`` has shape ``(B, T, n_sound_styles)`` or is ``None``.
        """
        log_probs = self.sound_ctc_head(encoder_output=sound_encoded)  # (B, T, V+1)
        events = log_probs.index_select(-1, self.sound_event_token_ids).exp()
        styles = None
        if self.n_sound_styles:
            styles = log_probs.index_select(-1, self.sound_style_token_ids).exp()
        return events, styles

    def _inject_sound_tokens(self, asr_encoded: torch.Tensor, sound_encoded: torch.Tensor) -> torch.Tensor:
        """Fuse sound tags into the ASR states (threshold + LayerNorm + kernel + ADD).

        Deliberately the same shape as :meth:`_fuse_diar_and_asr`, on disjoint sets of
        sinusoid rows, so an event, a style and a speaker never push the states the same
        way. The result is low rank: at most ``n_sound_events + n_sound_styles``
        directions into ``d_model``.

        Events and styles are normalized and projected separately, then summed. Sharing a
        LayerNorm would make each family's code depend on how many tags of the other
        family happen to be firing -- see the note where the two norms are built.

        Args:
            asr_encoded (torch.Tensor): Speech states.
                Shape: (B, D, T)
            sound_encoded (torch.Tensor): Sound expert states.
                Shape: (B, D, T)

        Returns:
            torch.Tensor: States with the tag signal added.
                Shape: (B, D, T)
        """
        if sound_encoded.shape[-1] != asr_encoded.shape[-1]:
            raise ValueError(
                f"sound expert produced {sound_encoded.shape[-1]} frames but the ASR states "
                f"have {asr_encoded.shape[-1]}; the expert outputs must share a frame grid."
            )
        with torch.no_grad():
            events, styles = self._sound_tag_posteriors(sound_encoded)
            # Hard threshold, matching the speaker branch: the kernels are defined over
            # tag PRESENCE, so the same binary signal is injected at train and at
            # inference time regardless of how confident the head happens to be.
            events = (events > self.sound_event_threshold).to(asr_encoded.dtype)
            if styles is not None:
                styles = (styles > self.sound_event_threshold).to(asr_encoded.dtype)

        states = asr_encoded.transpose(1, 2)  # (B, T, D)
        events = self.sound_token_norm(events)
        infusion = self.sound_kernel_scale * torch.matmul(events, self.sound_token_kernel.to(events.dtype))
        if styles is not None:
            styles = self.sound_style_norm(styles)
            infusion = infusion + self.sound_style_scale * torch.matmul(
                styles, self.sound_style_kernel.to(styles.dtype)
            )
        fused = infusion.to(states.dtype) + states
        return fused.transpose(1, 2).to(asr_encoded.dtype)  # (B, D, T)

    def _merge_sound_and_asr(self, asr_encoded: torch.Tensor, sound_encoded: torch.Tensor) -> torch.Tensor:
        """Add the sound expert's encoder states onto the ASR states.

        The alternative to :meth:`_inject_sound_tokens`, selected by
        ``merge_sound_expert_to_asr=True``: the whole sound representation is added,
        rather than only the discrete event tags the CTC head reads out of it.

        Both experts run over the same frames in the same packed group, so the two
        tensors are already aligned in time and need no resampling.

        Args:
            asr_encoded (torch.Tensor): Speech states before speaker fusion.
                Shape: (B, D, T)
            sound_encoded (torch.Tensor): Sound expert states.
                Shape: (B, D, T)

        Returns:
            torch.Tensor: Merged states.
                Shape: (B, D, T)
        """
        if not self.merge_sound_expert_to_asr:
            raise RuntimeError(
                "_merge_sound_and_asr is the encoder-state path, but "
                "merge_sound_expert_to_asr is False; _inject_sound_tokens is the one to call."
            )
        if sound_encoded.shape[-1] != asr_encoded.shape[-1]:
            raise ValueError(
                f"sound expert produced {sound_encoded.shape[-1]} frames but the ASR states "
                f"have {asr_encoded.shape[-1]}; the expert outputs must share a frame grid."
            )
        # Normalize BOTH streams so the scale is a relative weight (see __init__).
        speech_states = self.merge_speech_norm(asr_encoded.transpose(1, 2))  # (B, T, D)
        sound_states = self.sound_norm(sound_encoded.transpose(1, 2))  # (B, T, D)
        merged = speech_states + self.sound_merge_scale * sound_states.to(speech_states.dtype)
        return merged.transpose(1, 2).to(asr_encoded.dtype)  # (B, D, T)

    def _fuse_diar_and_asr(
        self,
        asr_encoded: torch.Tensor,
        spk_targets: torch.Tensor,
        *,
        diarization_preds: Optional[torch.Tensor] = None,
        use_diarization: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse sound-enhanced ASR states with speaker activity.

        Args:
            asr_encoded (torch.Tensor): ASR states after sound merge or injection.
                Shape: (B, D, T)
            spk_targets (torch.Tensor): RTTM or Sortformer speaker activity.
                Shape: (B, T, n_spk)
            diarization_preds (torch.Tensor, optional): Diarization predictions used for
                rows selected by ``use_diarization``. Callers must supply these whenever
                ``use_diarization`` can select a row; :meth:`_forward` guarantees it.
                Shape: (B, T, n_spk)
            use_diarization (torch.Tensor, optional): Bool mask per batch row.
                Shape: (B,)

        Returns:
            torch.Tensor: Fused encoder output.
                Shape: (B, D, T)
        """
        asr_enc_states = asr_encoded.transpose(1, 2)  # (B, T, D)
        spk_targets = self._align_diar_frames(spk_targets, asr_enc_states.shape[1]).to(asr_enc_states.dtype)
        # Select per row on device. Reading `use_diarization.any()` here to skip the select
        # would block the host on a device transfer, and with activation checkpointing this
        # runs again inside the backward pass, where stalling the host reorders the
        # surrounding FSDP gradient reduce-scatters relative to peer ranks.
        if use_diarization is not None and diarization_preds is not None:
            if use_diarization.numel() != spk_targets.shape[0]:
                raise ValueError(
                    f"use_diarization size ({use_diarization.numel()}) must match "
                    f"the speaker-target batch size ({spk_targets.shape[0]})."
                )
            diarization_preds = self._align_diar_frames(diarization_preds, asr_enc_states.shape[1]).to(
                asr_enc_states.dtype
            )
            spk_targets = torch.where(
                use_diarization.to(device=spk_targets.device, dtype=torch.bool).view(-1, 1, 1),
                diarization_preds,
                spk_targets,
            )

        # RTTM targets and Sortformer sigmoid outputs must produce the same
        # binary speaker kernel at train and inference time.
        spk_targets = (spk_targets > self.speaker_activity_threshold).to(asr_enc_states.dtype)

        asr_enc_states = self.asr_norm(asr_enc_states)
        spk_targets = self.diar_norm(spk_targets)
        speaker_infusion = torch.matmul(spk_targets, self.diar_kernel.to(spk_targets.dtype))
        fused = self.spk_kernel_scale * speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

    @contextlib.contextmanager
    def online_inference(self, enabled: bool = True):
        """Route :meth:`forward` through the windowed long-form path inside this block.

        Generation always opens this; training and validation never do. Off by default, and
        deliberately not inferred from ``self.training``, because validation also runs in
        eval mode and must stay on the single-pass path. The windowed loop calls the packed
        encoder once per window, so the number of collectives it emits tracks each rank's
        own audio length -- fine in one process, a deadlock in a distributed step.

        Args:
            enabled (bool): When ``True``, route :meth:`forward` through the windowed path.
        """
        previous = self.online_inference_enabled
        self.online_inference_enabled = bool(enabled)
        try:
            yield
        finally:
            self.online_inference_enabled = previous

    def _prepare_input(self, audio_signal, length):
        """Normalize and cast the shared mel input every expert consumes.

        Packed execution uses a shared input tensor, so normalization runs once and
        every expert receives the same features.

        Args:
            audio_signal (torch.Tensor): Raw mel features.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Valid feature length per sample.
                Shape: (B,)

        Returns:
            tuple: ``(audio_signal, length)`` normalized, cast, and on the experts' device.
        """
        if self.asr_normalize_type:
            audio_signal, _, _ = normalize_batch(audio_signal, length, normalize_type=self.asr_normalize_type)
        audio_signal = self._match_module_io(audio_signal)
        return audio_signal, length.to(device=audio_signal.device)

    # Forward — identical signature to ConformerEncoder.forward
    def forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
        return_experts: bool = False,
    ):
        """Encode ``audio_signal``, add sound information, and fuse speaker activity.

        Speaker fusion is per row and consistent across modes. Rows with RTTM use their
        ``spk_targets``; rows marked by ``missing_rttm_target`` use the Sortformer
        prediction.

        Encoding mode depends on the caller:

        * Training and validation take :meth:`_forward`, a single pass over the
          utterance, so every rank issues the same collectives.
        * Generation may open :meth:`online_inference` and take
          :meth:`_forward_online`, which walks long-form audio with a live speaker cache.

        Args:
            audio_signal (torch.Tensor): Un-normalised mel features; `per_feature` normalization
                is re-applied internally.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Per-sample feature lengths.
                Shape: (B,)
            spk_targets (torch.Tensor, optional): RTTM/oracle speaker activity.
                Shape: (B, T, n_spk)
                ``None`` predicts for the whole batch; rows marked by
                ``missing_rttm_target`` use predictions, allowing mixed target
                availability within a batch.
            return_experts (bool): Also return the per-expert outputs (including the
                unfused sound expert and the speaker activity predictions). Off by
                default so the standard return stays compatible with
                :class:`ConformerEncoder`.

        Returns:
            tuple: ``(outputs, encoded_lengths)`` with ``outputs`` of shape ``(B, D, T)``,
            or ``(outputs, encoded_lengths, experts)`` when ``return_experts`` is ``True``.
        """
        # Off unless a generation call opened `online_inference()`, so training and
        # validation both take the single-pass path no matter what the batch holds.
        use_online = self.online_inference_enabled and self.online_inference_length > 0

        runner = self._forward_online if use_online else self._forward
        outputs, lengths, experts = runner(audio_signal=audio_signal, length=length, spk_targets=spk_targets)
        if return_experts:
            return outputs, lengths, experts
        return outputs, lengths

    def _forward(self, audio_signal, length, spk_targets=None):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics.

        Inference calls :meth:`GGEMMTransformerEncoder.forward_packed` for all experts.
        They share the same input and therefore the same ``T``, so no streaming prefix
        or cross-expert length padding is needed.

        Training takes the per-expert path instead (see ``fused_forward_in_training``).
        Fusing costs memory that only matters once activations have to be kept for
        backward: it stacks every expert's FFN input into one ``(E_total, N, target_d)``
        tensor, and evaluates all ``moe_num_experts`` experts on every token so it can
        weight them by a router matrix that is zero outside the top ``k``. Both are a good
        trade when nothing is stored and a bad one when everything is.

        Args:
            audio_signal (torch.Tensor): Un-normalised mel features; `per_feature` normalization
                is re-applied internally.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Per-sample feature lengths.
                Shape: (B,)
            spk_targets (torch.Tensor, optional): RTTM/oracle speaker activity.
                Shape: (B, T, n_spk)

        Returns:
            tuple: ``(outputs, encoded_lengths, experts)`` with ``outputs`` of shape
            ``(B, D, T)`` and per-expert side outputs in ``experts``.
        """
        self._check_spk_target_width(spk_targets)
        use_diarization = (
            None if spk_targets is None else (spk_targets <= self.missing_rttm_target).flatten(start_dim=1).any(dim=1)
        )

        if spk_targets is None or self.always_run_diarization:
            run_diarization = True
        else:
            # Opt-out path. Reads batch content, so it both syncs the host and lets ranks
            # disagree; only safe in a single process.
            run_diarization = bool(use_diarization.any())

        signal, signal_length = self._prepare_input(audio_signal, length)
        # Respect an enclosing no_grad/inference_mode context. PEE only narrows
        # gradient tracking when every expert is frozen; it must never turn
        # gradients back on behind the caller's back.
        track_gradients = torch.is_grad_enabled() and not (
            self.freeze_speech and self.freeze_speaker and self.freeze_sound
        )
        with torch.set_grad_enabled(track_gradients):
            if self.training and not self.fused_forward_in_training:
                # Each expert follows its native path with no cross-expert stacking;
                # the speech MoE dispatches only its routed pairs.
                packed = self._forward_all_training(signal, signal_length)
            else:
                packed = self.pee.forward_packed(
                    signal, signal_length, backend=self.ggemm_backend, moe_mode=self.moe_mode
                )

        asr_encoded, asr_encoded_len = packed['speech']
        sound_encoded, sound_encoded_len = packed['sound']

        diarization_preds = None
        if run_diarization:
            speaker_encoded, speaker_len = packed['speaker']
            with torch.set_grad_enabled(track_gradients and not self.freeze_speaker):
                diarization_preds = self._speaker_head(speaker_encoded, speaker_len)
            if spk_targets is None:
                spk_targets = diarization_preds

        asr_states = asr_encoded
        if self.merge_sound_expert_to_asr:
            asr_states = self._merge_sound_and_asr(asr_states, sound_encoded)
        else:
            asr_states = self._inject_sound_tokens(asr_states, sound_encoded)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(
                asr_states,
                spk_targets,
                diarization_preds=diarization_preds,
                use_diarization=use_diarization,
            )
        else:
            outputs = asr_states

        experts = {
            'speech': (asr_encoded, asr_encoded_len),
            'sound': (sound_encoded, sound_encoded_len),
            'speaker_preds': diarization_preds,
        }
        return outputs, asr_encoded_len, experts

    def _forward_online(self, audio_signal, length, spk_targets=None):
        """Long-form generation path: dispatches to the offline pass or the windowed loop.

        Args:
            audio_signal (torch.Tensor): Un-normalised mel features; `per_feature` normalization
                is re-applied internally.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Per-sample feature lengths.
                Shape: (B,)
            spk_targets (torch.Tensor, optional): RTTM/oracle speaker activity override.
                Shape: (B, T, n_spk)
                Rows marked by ``missing_rttm_target`` still get a streaming Sortformer
                prediction; the rest keep their targets and only the encoder is chunked
                for them.

        Returns:
            tuple: ``(outputs, encoded_lengths, experts)``; ``outputs`` has shape ``(B, D, T)``.
        """
        total_feat_len = min(audio_signal.shape[-1], int(length.max().item()))
        num_chunks = max(1, math.ceil(total_feat_len / self.chunk_feat_len))

        if num_chunks == 1:
            # The whole batch fits one window, so every part of the streaming
            # apparatus is dead weight: the cache and context are empty, and
            # `streaming_update` reduces to selecting the current chunk.
            return self._forward(
                audio_signal=audio_signal[:, :, :total_feat_len],
                length=length.clamp(max=total_feat_len),
                spk_targets=spk_targets,
            )

        return self._forward_windowed(
            audio_signal=audio_signal,
            length=length,
            spk_targets=spk_targets,
            total_feat_len=total_feat_len,
            num_chunks=num_chunks,
        )

    def _forward_windowed(self, audio_signal, length, spk_targets, total_feat_len, num_chunks):
        """The windowed long-form loop proper: one ``forward_packed`` per window with a
        live speaker cache. Split out from :meth:`_forward_online` so the single-window
        fast path and this path can each be exercised directly.

        Args:
            audio_signal (torch.Tensor): Un-normalised mel features.
                Shape: (B, feat_in, n_frames)
            length (torch.Tensor): Per-sample feature lengths.
                Shape: (B,)
            spk_targets (torch.Tensor, optional): RTTM/oracle speaker activity.
                Shape: (B, T, n_spk)
            total_feat_len (int): Total mel frames to encode (clamped to batch max).
            num_chunks (int): Number of non-overlapping windows.

        Returns:
            tuple: ``(outputs, encoded_lengths, experts)``; ``outputs`` has shape ``(B, D, T)``.
        """
        # Normalise the whole utterance once (not per chunk) to match offline stats.
        signal, signal_length = self._prepare_input(audio_signal, length)
        self._check_spk_target_width(spk_targets)
        use_diarization = (
            None if spk_targets is None else (spk_targets <= self.missing_rttm_target).flatten(start_dim=1).any(dim=1)
        )
        run_streaming_diar = spk_targets is None or bool(use_diarization.any())

        streaming_state, stream_dtype = None, signal.dtype
        if run_streaming_diar:
            streaming_state, stream_dtype = self._init_streaming_diar(batch_size=signal.shape[0])

        asr_chunks: List[torch.Tensor] = []
        sound_chunks: List[torch.Tensor] = []
        diar_chunks: List[torch.Tensor] = []
        asr_encoded_len = torch.zeros_like(signal_length)
        track_gradients = torch.is_grad_enabled() and not (
            self.freeze_speech and self.freeze_speaker and self.freeze_sound
        )

        for chunk_idx in tqdm(
            range(num_chunks),
            total=num_chunks,
            desc="PEE online inference",
            disable=getattr(self, '_suppress_online_pbar', False),
        ):
            stt = chunk_idx * self.chunk_feat_len
            end = min(stt + self.chunk_feat_len, total_feat_len)

            # Shared context-extended window (input mel frames) for every expert.
            enc_stt = max(stt - self.left_ctx_feat_len, 0)
            enc_end = min(end + self.right_ctx_feat_len, total_feat_len)
            left_offset = stt - enc_stt
            right_offset = enc_end - end

            # The speaker attends over [cache | chunk].
            prefix, cache_len, extra = None, 0, 0
            if run_streaming_diar:
                cache = torch.cat([streaming_state.spkcache, streaming_state.fifo], dim=1)
                cache_len = cache.shape[1]
                prefix = {'speaker': cache.to(dtype=signal.dtype)}
                if self.online_prefix_mode == 'replace':
                    # Only as far back as the recording actually goes.
                    extra = min(cache_len * self.subsampling_factor, enc_stt) // self.subsampling_factor

            # 'replace' needs at least `cache_len` leading window frames to replace
            # with cache embeddings. Near the recording start, fall back to the
            # zero-padded 'extend' path when that context is unavailable.
            prefix_mode = 'replace' if (self.online_prefix_mode == 'replace' and extra == cache_len) else 'extend'
            ext_stt = enc_stt - (extra * self.subsampling_factor if prefix_mode == 'replace' else 0)

            window = signal[:, :, ext_stt:enc_end]
            window_length = (signal_length - ext_stt).clamp(min=0, max=enc_end - ext_stt)

            with torch.set_grad_enabled(track_gradients):
                packed, pre_encode = self.pee.forward_packed(
                    window,
                    window_length,
                    backend=self.ggemm_backend,
                    moe_mode=self.moe_mode,
                    prefix=prefix,
                    return_pre_encode=True,
                    prefix_mode=prefix_mode,
                )

            # Trim context off in output-frame space using rounded cumulative positions.
            # Speech/sound now also carry `extra` frames of extra left context up front.
            n_extra = extra if prefix_mode == 'replace' else 0
            left_drop = n_extra + left_offset // self.subsampling_factor
            right_drop = right_offset // self.subsampling_factor
            core_len = round(end / self.subsampling_factor) - round(stt / self.subsampling_factor)

            enc_ctx, _ = packed['speech']
            core = max(0, min(core_len, enc_ctx.shape[-1] - left_drop))
            asr_chunks.append(enc_ctx[:, :, left_drop : left_drop + core])
            snd_ctx, _ = packed['sound']
            sound_chunks.append(snd_ctx[:, :, left_drop : left_drop + core])
            asr_encoded_len += core
            align_target = core

            if run_streaming_diar:
                speaker_encoded, speaker_len = packed['speaker']
                with torch.set_grad_enabled(track_gradients and not self.freeze_speaker):
                    # Predictions span [spkcache | fifo | lc + chunk + rc], which is
                    # exactly the layout `streaming_update` documents.
                    preds = self._speaker_head(speaker_encoded, speaker_len)
                # The speaker's chunk excludes the frames the cache stood in for, so it
                # spans exactly [lc | chunk | rc] of the ORIGINAL window -- its lc is
                # `left_offset`, not the `left_drop` speech/sound use (which also has
                # to skip the extra left context those two received).
                chunk_embs, _ = pre_encode['speaker']  # (B, lc+chunk+rc, fc_d_model)
                with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
                    streaming_state, chunk_preds = self.sortformer_modules.streaming_update(
                        streaming_state,
                        chunk=chunk_embs.to(stream_dtype),
                        preds=preds.to(stream_dtype),
                        lc=left_offset // self.subsampling_factor,
                        rc=right_drop,
                    )
                # Newly emitted frames, aligned to the speech chunk (frame-parallel).
                diar_chunks.append(self._align_diar_frames(chunk_preds, align_target))

        asr_encoded = torch.cat(asr_chunks, dim=2)  # (B, D, T_asr)
        sound_encoded = torch.cat(sound_chunks, dim=2)
        diarization_preds = torch.cat(diar_chunks, dim=1) if run_streaming_diar else None
        if spk_targets is None:
            spk_targets = diarization_preds

        # Same order as the offline path: sound joins the ASR backbone first, then the
        # speaker kernel is fused on top. The windowed sound chunks were trimmed with
        # the identical offsets, so they stay frame-aligned with the ASR states.
        asr_states = asr_encoded
        if self.merge_sound_expert_to_asr:
            asr_states = self._merge_sound_and_asr(asr_states, sound_encoded)
        else:
            asr_states = self._inject_sound_tokens(asr_states, sound_encoded)

        if spk_targets is not None:
            outputs = self._fuse_diar_and_asr(
                asr_states,
                spk_targets,
                diarization_preds=diarization_preds,
                use_diarization=use_diarization,
            )
        else:
            outputs = asr_states

        experts = {
            'speech': (asr_encoded, asr_encoded_len),
            'sound': (sound_encoded, asr_encoded_len),
            'speaker_preds': diarization_preds,
        }
        return outputs, asr_encoded_len, experts

    def _init_streaming_diar(self, batch_size: int) -> Tuple[object, torch.dtype]:
        """Configure the Sortformer streaming params and build the initial state.

        Args:
            batch_size (int): Batch size for the streaming state.

        Returns:
            ``(streaming_state, stream_dtype)`` on the speaker expert's device & dtype.
        """
        sm = self.sortformer_modules
        sm.chunk_len = self.online_inference_length
        sm.chunk_left_context = self.chunk_left_context
        sm.chunk_right_context = self.chunk_right_context
        sm.fifo_len = self.diar_fifo_len
        sm.spkcache_update_period = self.diar_spkcache_update_period
        sm.spkcache_len = self.diar_spkcache_len
        sm._check_streaming_parameters()

        param = next(self.pee.experts['speaker'].parameters(), None)
        if param is not None:
            device, stream_dtype = param.device, param.dtype
        else:
            device, stream_dtype = self.diar_kernel.device, torch.get_default_dtype()

        with _disable_dist_feature_sync(), _default_dtype(stream_dtype):
            # Synchronous update: the cache starts empty and grows to `spkcache_len`,
            # so the prefix (and therefore T) is shorter on the first few windows.
            streaming_state = sm.init_streaming_state(batch_size=batch_size, async_streaming=False, device=device)
        return streaming_state, stream_dtype


# Public names distinguish this architecture from the two-branch encoder on main.
GGEMMParallelExpertEncoder = ParallelExpertEncoder
GGEMMParallelExpertEncoderPT = ParallelExpertEncoderPT
