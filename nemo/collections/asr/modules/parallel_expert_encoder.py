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
internally. I/O matches :class:`ConformerEncoder` (drop-in). Only self-contained PE
bundles (inline ``asr_encoder_cfg`` + ``diarization_model_cfg`` in
``model_config.yaml``) are supported.
"""

from __future__ import annotations

import contextlib
import math
import os
import tarfile
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
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


@experimental
class ParallelExpertEncoderPT(ModelPT):
    """ModelPT shell so a :class:`ParallelExpertEncoder` can be saved/restored as a
    ``.nemo`` archive (inline ``asr_encoder_cfg`` + ``diarization_model_cfg``).
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)
        self.encoder = ParallelExpertEncoder(
            asr_encoder_cfg=self._cfg.get('asr_encoder_cfg', None),
            diarization_model_cfg=self._cfg.get('diarization_model_cfg', None),
            asr_normalize_type=self._cfg.get('asr_normalize_type', None),
            freeze_diar=self._cfg.get('freeze_diar', True),
            freeze_asr=self._cfg.get('freeze_asr', False),
            online_inference_length=self._cfg.get('online_inference_length', 500),
            chunk_left_context=self._cfg.get('chunk_left_context', 50),
            chunk_right_context=self._cfg.get('chunk_right_context', 50),
            diar_fifo_len=self._cfg.get('diar_fifo_len', 40),
            diar_spkcache_update_period=self._cfg.get('diar_spkcache_update_period', 300),
            diar_spkcache_len=self._cfg.get('diar_spkcache_len', 188),
        )

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
        nemo_path: str,
        *,
        map_location: Union[str, torch.device] = 'cpu',
        strict: bool = True,
    ) -> ParallelExpertEncoder:
        """Load a self-contained PE ``.nemo`` bundle and return its inner encoder.

        Args:
            nemo_path (str): Path to a self-contained PE ``.nemo`` bundle.
            map_location (str | torch.device): Device to map weights onto.
            strict (bool): Enforce exact state-dict match.

        Returns:
            The restored :class:`ParallelExpertEncoder`.
        """
        bundle = cls.restore_from(
            restore_path=nemo_path,
            map_location=map_location,
            strict=strict,
        )
        return bundle.encoder

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
        mismatches raise :class:`ValueError` fail-fast.

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
    """Sortformer-diarizer + ASR Conformer encoder; I/O identical to :class:`ConformerEncoder`.

    Reconstructed from inline configs in the PE bundle's ``model_config.yaml``.

    Args:
        asr_encoder_cfg (DictConfig): Inline config for the ASR-side :class:`ConformerEncoder`.
        diarization_model_cfg (DictConfig): Inline config for the :class:`SortformerEncLabelModel`.
        asr_normalize_type (str, optional): Normalization replayed on the ASR branch. Defaults to ``per_feature``.
        freeze_diar (bool): Freeze the Sortformer parameters. Defaults to ``True``.
        freeze_asr (bool): Freeze the wrapped ASR ConformerEncoder. Defaults to ``False``.
        online_inference_length (int): Online-inference window in encoder output frames
            (default ``500`` ~= 40s); ``<= 0`` disables it.
        chunk_left_context (int): Left context (output frames) per online window, shared by
            both branches. Default ``50``.
        chunk_right_context (int): Right context (output frames) per online window, shared by
            both branches. Default ``50``.
        diar_fifo_len (int): Sortformer streaming ``fifo_len``. Default ``40``.
        diar_spkcache_update_period (int): Sortformer streaming ``spkcache_update_period``. Default ``300``.
        diar_spkcache_len (int): Sortformer streaming ``spkcache_len``. Default ``188``.
    """

    def __init__(
        self,
        asr_encoder_cfg: DictConfig,
        diarization_model_cfg: DictConfig,
        asr_normalize_type: Optional[str] = None,
        freeze_diar: bool = True,
        freeze_asr: bool = False,
        online_inference_length: int = 500,
        chunk_left_context: int = 50,
        chunk_right_context: int = 50,
        diar_fifo_len: int = 40,
        diar_spkcache_update_period: int = 300,
        diar_spkcache_len: int = 188,
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

        self.asr_encoder = ConformerEncoder.from_config_dict(_clone_config(asr_encoder_cfg))
        if not isinstance(self.asr_encoder, ConformerEncoder):
            raise TypeError(
                f"Expected `asr_encoder_cfg._target_` to instantiate a "
                f"ConformerEncoder, got {type(self.asr_encoder).__name__} instead."
            )
        self.asr_normalize_type = asr_normalize_type or 'per_feature'
        self._feat_in = self.asr_encoder._feat_in

        self.diarization_model = SortformerEncLabelModel.from_config_dict(_clone_config(diarization_model_cfg))

        self.freeze_diar = freeze_diar
        self.freeze_asr = freeze_asr

        # Long-form / online inference configuration.
        self.online_inference_length = int(online_inference_length)
        # Overlap-and-trim context (output frames) shared by both branches.
        self.chunk_left_context = max(0, int(chunk_left_context))
        self.chunk_right_context = max(0, int(chunk_right_context))
        # Online-inference window + context in input mel frames (constant per session).
        self.chunk_feat_len = self.online_inference_length * self.asr_encoder.subsampling_factor
        self.left_ctx_feat_len = self.chunk_left_context * self.asr_encoder.subsampling_factor
        self.right_ctx_feat_len = self.chunk_right_context * self.asr_encoder.subsampling_factor
        self.diar_fifo_len = int(diar_fifo_len)
        self.diar_spkcache_update_period = int(diar_spkcache_update_period)
        self.diar_spkcache_len = int(diar_spkcache_len)

        self.n_spk = int(self.diarization_model.sortformer_modules.n_spk)
        self.asr_d_model = self.asr_encoder.d_model

        self.asr_norm = nn.LayerNorm(self.asr_d_model)
        self.diar_norm = nn.LayerNorm(self.n_spk)
        self.register_buffer(
            "diar_kernel",
            self._build_sinusoid_position_encoding(self.n_spk, self.asr_d_model),
            persistent=False,
        )

        if self.freeze_diar:
            self.diarization_model.eval()
            for p in self.diarization_model.parameters():
                p.requires_grad = False
        if self.freeze_asr:
            self.asr_encoder.eval()
            for p in self.asr_encoder.parameters():
                p.requires_grad = False

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

        asr_enc_states = self.asr_norm(asr_enc_states)
        spk_targets = self.diar_norm(spk_targets)
        speaker_infusion = torch.matmul(spk_targets, self.diar_kernel.to(spk_targets.dtype))
        fused = speaker_infusion + asr_enc_states

        return fused.transpose(1, 2)  # (B, D, T)

    # Forward — identical signature to ConformerEncoder.forward
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

    def _forward(
        self,
        audio_signal,
        length,
        spk_targets=None,
    ):
        """Offline (non-chunked) forward pass. See :meth:`forward` for argument semantics."""
        if spk_targets is None:
            # Cast fp32 mels to the diarizer's device/dtype before its conv subsampling.
            diar_signal = self._match_module_io(audio_signal, self.diarization_model)
            diar_length = length.to(device=diar_signal.device)
            with torch.set_grad_enabled(not self.freeze_diar):
                emb_seq, emb_seq_length = self.diarization_model.frontend_encoder(
                    processed_signal=diar_signal,
                    processed_signal_length=diar_length,
                    bypass_pre_encode=False,
                )
                spk_targets = self.diarization_model.forward_infer(
                    emb_seq=emb_seq,
                    emb_seq_length=emb_seq_length,
                )

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

        with torch.set_grad_enabled(not self.freeze_asr):
            asr_encoded, asr_encoded_len = self.asr_encoder(
                audio_signal=asr_audio_signal,
                length=asr_length,
            )

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
            streaming_state, stream_dtype, diar_audio_signal, diar_length = self._init_streaming_diar(
                audio_signal,
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
            with torch.set_grad_enabled(not self.freeze_asr):
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
                    torch.set_grad_enabled(not self.freeze_diar),
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
        sm = self.diarization_model.sortformer_modules
        sm.chunk_len = self.online_inference_length
        sm.fifo_len = self.diar_fifo_len
        sm.spkcache_update_period = self.diar_spkcache_update_period
        sm.spkcache_len = self.diar_spkcache_len
        sm._check_streaming_parameters()

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
