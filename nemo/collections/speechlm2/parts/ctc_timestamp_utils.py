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

"""CTC timestamp alignment and lightweight artifact handling."""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import tempfile
import unicodedata
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.core.utils.optional_libs import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    from numba import jit

__all__ = [
    "CTCTimestampArtifact",
    "CTC_TIMESTAMP_ARTIFACT_FORMAT",
    "MultiSpeakerSOTWordTimestampAligner",
    "TransformerCTCDecoder",
    "export_ctc_timestamp_artifact",
    "get_ctc_timestamp_aligner",
    "load_ctc_timestamp_artifact",
    "save_ctc_timestamp_artifact",
]


CTC_TIMESTAMP_ARTIFACT_FORMAT = "nemo_ctc_timestamp_artifact_v1"
CTC_TIMESTAMP_BUNDLE_PREFIX = "ctc_timestamp."


class TransformerCTCDecoder(ConvASRDecoder):
    """CTC decoder with an optional Transformer bridge before the CTC convolution.

    The inherited ConvASRDecoder supplies the final 1x1 convolution and CTC
    log-softmax. Enabling use_transformer inserts a length-aware dense
    TransformerEncoder between encoder states and that convolution. Disabling it
    produces the equivalent Conv-only CTC head, which makes the two alternatives
    directly comparable.

    d_model is intentionally constrained to feat_in. This keeps the head strictly
    Transformer+Conv (or Conv-only), without a separate Linear projection layer.
    ``upsampling_factor=2`` adds the checkpoint's learned ConvTranspose1d before
    the bridge, producing native 40 ms CTC frames from 80 ms encoder states.
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
        upsampling_factor: int = 1,
        upsampling_mode: str = "conv_transpose",
    ):
        """
        Initialize the optional Transformer bridge and CTC projection.

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
            upsampling_factor (int): Output frames per encoder frame; 2 gives 40 ms CTC from 80 ms states.
            upsampling_mode (str): Learned upsampling layer; only ``conv_transpose`` is supported.
        """
        if residual and not use_transformer:
            raise ValueError("TransformerCTCDecoder residual connections require use_transformer=True.")
        if isinstance(upsampling_factor, bool) or not isinstance(upsampling_factor, int) or upsampling_factor < 1:
            raise ValueError("upsampling_factor must be a positive integer.")
        if upsampling_mode != "conv_transpose":
            raise ValueError("Only conv_transpose upsampling is supported.")

        super().__init__(
            feat_in=feat_in,
            num_classes=num_classes,
            init_mode=init_mode,
            vocabulary=vocabulary,
            add_blank=add_blank,
        )

        self.use_transformer = use_transformer
        self.requires_encoded_lengths = use_transformer or upsampling_factor > 1
        self.transformer = None
        self.residual = residual
        self.upsampling_factor = upsampling_factor
        if upsampling_factor > 1:
            self.register_buffer("upsample_factor", torch.tensor(upsampling_factor), persistent=True)
            self.upsample = nn.ConvTranspose1d(
                feat_in, feat_in, kernel_size=upsampling_factor, stride=upsampling_factor
            )

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
        """
        Return CTC log probabilities from channels-first encoder states.
        With ``use_transformer``, the states first pass through the Transformer bridge, whose scaled
        output is added to its input when ``residual`` is set; the inherited convolution and
        log-softmax follow.

        Args:
            encoder_output (torch.Tensor): Acoustic states shaped ``(B, D, T)``.
            encoded_lengths (Optional[torch.Tensor]): Valid frame counts shaped ``(B,)``. Required when
                ``use_transformer=True`` or ``upsampling_factor > 1``.

        Returns:
            torch.Tensor: Log probabilities shaped ``(B, T * upsampling_factor, num_classes_with_blank)``.
        """
        return self.classify(self.trunk(encoder_output, encoded_lengths))

    def trunk(self, encoder_output: torch.Tensor, encoded_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """:meth:`forward` up to the convolution: the upsampled, bridged states shaped ``(B, D, T')``."""
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
        if self.upsampling_factor > 1:
            if encoded_lengths is None:
                raise ValueError("Upsampling requires encoded_lengths.")
            encoded = self.upsample(encoded)
            encoded_lengths = self.output_lengths(encoded_lengths)
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
        return encoded

    def classify(self, encoded: torch.Tensor) -> torch.Tensor:
        """The rest of :meth:`forward`: log probabilities from :meth:`trunk` states, each frame on its own."""
        return super().forward(encoder_output=encoded)

    def output_lengths(self, encoded_lengths: torch.Tensor) -> torch.Tensor:
        """
        Convert valid encoder lengths to valid CTC output lengths.

        Args:
            encoded_lengths (torch.Tensor): Valid encoder frame counts shaped ``(B,)``.

        Returns:
            torch.Tensor: Valid CTC frame counts shaped ``(B,)``.
        """
        return encoded_lengths * self.upsampling_factor


class MultiSpeakerSOTWordTimestampAligner:
    """Align speaker streams with single-pass banded max-sum CTC DP and word-boundary refinement.

    Speaker mapping uses greedy token distributions, followed by speaker-inactive frame removal
    and anchor-guided DP with byte backpointers. The speaker prior defaults to weight 1.0. Every
    result receives diarization-assisted word refinement, capped at 12 frames by default.

    The aligner takes model outputs, not audio: ASR states for the CTC head
    (:meth:`generate_from_inputs`) or CTC log probabilities (:meth:`extract_from_outputs_batch`).
    To start from audio, run the encoder with ``return_ctc_timestamp_inputs=True`` and pass the
    returned ``CTCTimestampInputs`` to :meth:`generate_from_inputs`.

    Each result keeps word timestamps (``speaker_word_timestamps``) apart from the 10 ms
    Sortformer activity segments (``diarization_timestamps``). Score diarization (DER) with
    the segments only: word timestamps mark where each word is, not continuous speech
    activity, so a DER computed from them is poor.
    """

    _SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
    # Bound vocabulary-wide output per decoder call (64 MiB at FP32), except when
    # a single context window already exceeds this budget. Never alter its context.
    _CTC_WINDOW_MAX_ELEMENTS = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        ctc_decoder: Optional[TransformerCTCDecoder] = None,
        tokenizer: Optional[Any] = None,
        blank_id: Optional[int] = None,
        ctc_frame_seconds: Optional[float] = None,
        sortformer_frame_seconds: Optional[float] = None,
        speaker_activity_threshold: float = 0.5,
        speaker_logprob_weight: float = 1.0,
        maximum_token_len: float = 1.0,
        max_word_frames: int = 12,
        timestamp_language: Optional[str] = None,
        epsilon: float = 1.0e-6,
        online_inference_length: int = 0,
        chunk_left_context: int = 0,
        chunk_right_context: int = 0,
    ) -> None:
        """
        Initialize multi-speaker SOT word timestamp alignment.

        Args:
            ctc_decoder (Optional[TransformerCTCDecoder]): CTC timestamp decoder.
            tokenizer (Optional[Any]): Tokenizer matching the CTC decoder vocabulary.
            blank_id (Optional[int]): Explicit CTC blank class index.
            ctc_frame_seconds (Optional[float]): Duration of one CTC frame, for records without an
                audio duration.
            sortformer_frame_seconds (Optional[float]): Explicit duration of one Sortformer frame.
            speaker_activity_threshold (float): Threshold for speaker activity metadata.
            speaker_logprob_weight (float): Weight of the Sortformer prior in CTC alignment; defaults to 1.0.
            maximum_token_len (float): Maximum emitted word duration in seconds.
            max_word_frames (int): Word duration and onset-snap cap in encoder frames, scaled for upsampling.
            timestamp_language (Optional[str]): Explicit alignment-unit policy: zh/ja characters,
                th newmm words, or unchanged whitespace words for other languages. None selects
                per-transcript script detection.
            epsilon (float): Numerical floor for logarithms.
            online_inference_length (int): Core window, in encoder frames, of the deferred CTC
                head over stored states; 0 runs it over the whole sequence at once.
            chunk_left_context (int): Encoder frames of left context per deferred head window.
            chunk_right_context (int): Encoder frames of right context per deferred head window.
        """
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")
        if not 0 <= speaker_activity_threshold <= 1:
            raise ValueError("speaker_activity_threshold must be between zero and one.")
        if maximum_token_len <= 0:
            raise ValueError("maximum_token_len must be positive.")
        self.ctc_decoder = ctc_decoder
        if isinstance(max_word_frames, bool) or not isinstance(max_word_frames, int) or max_word_frames < 1:
            raise ValueError("max_word_frames must be a positive integer.")
        self.max_word_frames = max_word_frames
        self.timestamp_language = timestamp_language
        self.tokenizer = tokenizer
        self.blank_id = blank_id
        self.ctc_frame_seconds = ctc_frame_seconds
        self.sortformer_frame_seconds = sortformer_frame_seconds
        self.speaker_activity_threshold = float(speaker_activity_threshold)
        self.speaker_logprob_weight = float(speaker_logprob_weight)
        self.maximum_token_len = float(maximum_token_len)
        self.epsilon = float(epsilon)
        self.online_inference_length = int(online_inference_length)
        self.chunk_left_context = int(chunk_left_context)
        self.chunk_right_context = int(chunk_right_context)

    @classmethod
    def clean_sot_transcript(cls, transcript: str, special_tokens: Sequence[str]) -> str:
        """
        Remove decoder control tokens, preserving speaker, event and style tags.
        Decode with special tokens kept, then clean the text here. Speaker tags mark streams;
        event/style tags remain acoustic targets. EOS, padding and other control tokens are removed.

        Args:
            transcript (str): Decoded t-SOT transcript.
            special_tokens (Sequence[str]): The decoding tokenizer's special tokens.

        Returns:
            str: The transcript without those tokens, its words separated by single spaces.
        """
        for token in special_tokens:
            if cls._SPEAKER_TAG_RE.fullmatch(token) is None and re.fullmatch(r'<(?:ev|sty):[^<>]*>', token) is None:
                transcript = transcript.replace(token, " ")
        return " ".join(transcript.split())

    @classmethod
    def parse_sot_words(cls, transcript: str) -> List[Dict[str, Any]]:
        """
        Split a t-SOT transcript into words, treating em dashes as word separators.
        Each ``<spk:N>`` tag starts a new speaker turn. Words before the first tag have no speaker,
        and a transcript without any tag is assigned speaker 0.

        Args:
            transcript (str): Cleaned t-SOT transcript.

        Returns:
            List[Dict[str, Any]]: Word records with ``word``, ``speaker_tag``, ``turn_index`` (the
                0-based tag count, or None before the first tag) and ``word_index``.
        """
        transcript = transcript.replace("—", " ")
        words: List[Dict[str, Any]] = []
        parts = cls._SPEAKER_TAG_RE.split(transcript)
        speaker_tag: Optional[int] = 0 if len(parts) == 1 else None
        turn_index: Optional[int] = None
        for index, part in enumerate(parts):
            if index % 2:
                speaker_tag, turn_index = int(part), index // 2
                continue
            for word in part.split():
                words.append(
                    dict(
                        word=word,
                        speaker_tag=speaker_tag,
                        turn_index=turn_index,
                        word_index=len(words),
                    )
                )
        return words

    @torch.no_grad()
    def ctc_log_probs_from_inputs(self, timestamp_inputs: Any) -> torch.Tensor:
        """
        Run the deferred CTC head over stored ASR states in bounded windows.
        Windows follow the encoder's online inference: ``online_inference_length`` core
        frames, each decoded with ``chunk_left_context`` and ``chunk_right_context`` frames of
        context, refreshed by :func:`get_ctc_timestamp_aligner` on each call. This diagnostic
        API assembles full-vocabulary output; :meth:`prepare_from_inputs` retains only target columns.

        Args:
            timestamp_inputs (Any): ``CTCTimestampInputs`` whose ``asr_encoded`` is shaped ``(B, D, T)``.

        Returns:
            torch.Tensor: CTC log probabilities shaped ``(B, T * upsampling_factor, classes)``.
        """
        output = None
        for batch_start, start, chunk in self._iter_ctc_windows(timestamp_inputs):
            if output is None:
                states = timestamp_inputs.asr_encoded
                output = chunk.new_empty((states.shape[0], self._ctc_frames(states.shape[-1]), chunk.shape[-1]))
            output[batch_start : batch_start + chunk.shape[0], start : start + chunk.shape[1]] = chunk
            del chunk
        return output

    @torch.no_grad()
    def prepare_from_inputs(
        self,
        timestamp_inputs: Any,
        sot_transcripts: Sequence[str],
        audio_durations: Sequence[float],
    ) -> Dict[str, Any]:
        """
        Run the deferred CTC head over request-owned inputs and prepare their alignment.
        The half of :meth:`generate_from_inputs` that needs the CTC head and the tokenizer; finish it
        with :meth:`align_prepared`, or with :func:`align_prepared_batch` in a process without a model.
        It prepares what :meth:`prepare_alignment` prepares without ever holding the head's whole output:
        transcripts are tokenized first, and each window keeps only their columns after the softmax.

        Args:
            timestamp_inputs (Any): ``CTCTimestampInputs`` kept from the generation forward.
            sot_transcripts (Sequence[str]): Generated t-SOT transcripts, one per record.
            audio_durations (Sequence[float]): Audio durations in seconds, one per record.

        Returns:
            Dict[str, Any]: The prepared batch described in :meth:`prepare_alignment`.
        """
        states = timestamp_inputs.asr_encoded
        num_frames = self._ctc_frames(states.shape[-1])
        num_classes = int(self.ctc_decoder.num_classes_with_blank)

        return self._prepare(
            (states.shape[0], num_frames, num_classes),
            partial(_read_deferred_ctc_columns, self, timestamp_inputs, num_classes),
            timestamp_inputs.sortformer_sigmoids,
            sot_transcripts,
            ctc_lengths=self._ctc_frames(timestamp_inputs.asr_encoded_lengths).clamp(max=num_frames),
            # Speaker frames past the CTC timeline are never aligned.
            sortformer_lengths=timestamp_inputs.sortformer_lengths.clamp(max=states.shape[-1]),
            audio_durations=audio_durations,
            diarization_labels=timestamp_inputs.diarization_labels,
            diarization_lengths=timestamp_inputs.diarization_lengths,
            diarization_frame_seconds=timestamp_inputs.diarization_frame_seconds,
        )

    @torch.no_grad()
    def generate_from_inputs(
        self,
        timestamp_inputs: Any,
        sot_transcripts: Sequence[str],
        audio_durations: Sequence[float],
    ) -> List[Dict[str, Any]]:
        """
        Run the deferred CTC head over request-owned inputs and align generated ASR text.
        Equivalent to :meth:`prepare_from_inputs` followed by :meth:`align_prepared`.

        Args:
            timestamp_inputs (Any): ``CTCTimestampInputs`` kept from the generation forward.
            sot_transcripts (Sequence[str]): Generated t-SOT transcripts, one per record.
            audio_durations (Sequence[float]): Audio durations in seconds, one per record.

        Returns:
            List[Dict[str, Any]]: One alignment result per record.
        """
        return self.align_prepared(self.prepare_from_inputs(timestamp_inputs, sot_transcripts, audio_durations))

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
        diarization_labels: Optional[torch.Tensor] = None,
        diarization_lengths: Optional[torch.Tensor] = None,
        diarization_frame_seconds: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Align every record and independent speaker stream in shared DP/backtrace batches.
        Equivalent to :meth:`prepare_alignment` followed by :meth:`align_prepared`.

        Args:
            ctc_log_probs (torch.Tensor): CTC log probabilities shaped ``(B, T, classes)``.
            sortformer_sigmoids (Optional[torch.Tensor]): Sortformer speaker probabilities shaped
                ``(B, T_spk, speakers)``, or None to align without speaker information.
            sot_transcripts (Sequence[str]): t-SOT transcripts, one per record.
            ctc_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            sortformer_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            audio_durations (Optional[Sequence[Optional[float]]]): As in :meth:`prepare_alignment`.
            time_offsets (Optional[Sequence[float]]): As in :meth:`prepare_alignment`.
            speaker_logprob_weight (Optional[float]): As in :meth:`prepare_alignment`.
            diarization_labels (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            diarization_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            diarization_frame_seconds (float): As in :meth:`prepare_alignment`.

        Returns:
            List[Dict[str, Any]]: One alignment result per record.
        """
        return self.align_prepared(
            self.prepare_alignment(
                ctc_log_probs,
                sortformer_sigmoids,
                sot_transcripts,
                ctc_lengths=ctc_lengths,
                sortformer_lengths=sortformer_lengths,
                audio_durations=audio_durations,
                time_offsets=time_offsets,
                speaker_logprob_weight=speaker_logprob_weight,
                diarization_labels=diarization_labels,
                diarization_lengths=diarization_lengths,
                diarization_frame_seconds=diarization_frame_seconds,
            )
        )

    def prepare_alignment(
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
        diarization_labels: Optional[torch.Tensor] = None,
        diarization_lengths: Optional[torch.Tensor] = None,
        diarization_frame_seconds: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Turn transcripts into CTC targets and keep only the log-prob columns they use.
        The half of :meth:`extract_from_outputs_batch` that needs the tokenizer and the full
        CTC output. Per record it keeps the blank column and the columns of the transcript's
        tokens, renumbered so that blank is column 0, and moves everything to the host. The
        DP only compares labels and reads their columns, so :meth:`align_prepared` aligns
        exactly as on the full output, with no model, tokenizer or device.

        Args:
            ctc_log_probs (torch.Tensor): CTC log probabilities shaped ``(B, T, classes)``.
            sortformer_sigmoids (Optional[torch.Tensor]): Sortformer speaker probabilities shaped
                ``(B, T_spk, speakers)``, or None to align without speaker information.
            sot_transcripts (Sequence[str]): t-SOT transcripts, one per record.
            ctc_lengths (Optional[torch.Tensor]): Valid CTC frame counts shaped ``(B,)``; None uses
                every frame.
            sortformer_lengths (Optional[torch.Tensor]): Valid Sortformer frame counts shaped ``(B,)``;
                None uses every frame.
            audio_durations (Optional[Sequence[Optional[float]]]): Audio durations in seconds, one per
                record; a missing duration falls back to ``ctc_frame_seconds``.
            time_offsets (Optional[Sequence[float]]): Seconds added to each record's timestamps;
                defaults to 0.
            speaker_logprob_weight (Optional[float]): Weight of the Sortformer prior; None uses the
                aligner's ``speaker_logprob_weight``.
            diarization_labels (Optional[torch.Tensor]): Boolean speaker activity shaped
                ``(B, speakers, diarization_frames)``, used for the returned diarization segments.
            diarization_lengths (Optional[torch.Tensor]): Valid diarization frame counts shaped ``(B,)``.
            diarization_frame_seconds (float): Duration of one diarization label frame, in seconds.

        Returns:
            Dict[str, Any]: ``config``, the hyperparameters alignment uses, from which
                :func:`align_prepared_batch` rebuilds an aligner; ``num_speaker_columns``;
                ``diarization_frame_seconds``; ``diarization_max_speaker_count``; and ``records``,
                per record: ``words``, speaker ``streams`` with renumbered ``labels``,
                ``ctc_log_probs`` shaped ``(ctc_length, columns)``, ``ctc_length``, ``ctc_step``,
                ``speaker_probs`` shaped ``(speaker_length, speakers)``, ``speaker_length``,
                ``sortformer_step``, ``diarization_labels`` shaped ``(speakers, diarization_length)``,
                ``audio_duration`` and ``time_offset``. :func:`merge_prepared_batches` merges
                batches whose batch-level values match.
        """
        if ctc_log_probs.ndim != 3:
            raise ValueError("ctc_log_probs must have shape (batch, frames, classes).")

        return self._prepare(
            ctc_log_probs.shape,
            partial(_read_ctc_columns, ctc_log_probs),
            sortformer_sigmoids,
            sot_transcripts,
            ctc_lengths=ctc_lengths,
            sortformer_lengths=sortformer_lengths,
            audio_durations=audio_durations,
            time_offsets=time_offsets,
            speaker_logprob_weight=speaker_logprob_weight,
            diarization_labels=diarization_labels,
            diarization_lengths=diarization_lengths,
            diarization_frame_seconds=diarization_frame_seconds,
        )

    def _prepare(
        self,
        ctc_shape: Sequence[int],
        read_columns: Callable[[List[List[int]], List[int]], List[torch.Tensor]],
        sortformer_sigmoids: Optional[torch.Tensor],
        sot_transcripts: Sequence[str],
        *,
        ctc_lengths: Optional[torch.Tensor] = None,
        sortformer_lengths: Optional[torch.Tensor] = None,
        audio_durations: Optional[Sequence[Optional[float]]] = None,
        time_offsets: Optional[Sequence[float]] = None,
        speaker_logprob_weight: Optional[float] = None,
        diarization_labels: Optional[torch.Tensor] = None,
        diarization_lengths: Optional[torch.Tensor] = None,
        diarization_frame_seconds: float = 0.01,
    ) -> Dict[str, Any]:
        """
        :meth:`prepare_alignment` for a CTC output that is read only through ``read_columns``.
        Every transcript becomes CTC targets before any log probability is read, so a
        transcript the aligner rejects costs no CTC head run.

        Args:
            ctc_shape (Sequence[int]): ``(batch, frames, classes)`` of the CTC output.
            read_columns (Callable[[List[List[int]], List[int]], List[torch.Tensor]]): Given each
                record's columns, blank first, and its valid CTC length, returns its log
                probabilities shaped ``(ctc_length, columns)``, as float32 on the host.
            sortformer_sigmoids (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            sot_transcripts (Sequence[str]): As in :meth:`prepare_alignment`.
            ctc_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            sortformer_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            audio_durations (Optional[Sequence[Optional[float]]]): As in :meth:`prepare_alignment`.
            time_offsets (Optional[Sequence[float]]): As in :meth:`prepare_alignment`.
            speaker_logprob_weight (Optional[float]): As in :meth:`prepare_alignment`.
            diarization_labels (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            diarization_lengths (Optional[torch.Tensor]): As in :meth:`prepare_alignment`.
            diarization_frame_seconds (float): As in :meth:`prepare_alignment`.

        Returns:
            Dict[str, Any]: The prepared batch described in :meth:`prepare_alignment`.
        """
        batch_size, max_ctc_frames, vocab_size = ctc_shape
        if len(sot_transcripts) != batch_size:
            raise ValueError("sot_transcripts must contain one string per batch item.")
        ctc_lengths_list = self._lengths(ctc_lengths, batch_size, max_ctc_frames)
        durations = list(audio_durations) if audio_durations is not None else [None] * batch_size
        offsets = list(time_offsets) if time_offsets is not None else [0.0] * batch_size
        if len(durations) != batch_size or len(offsets) != batch_size:
            raise ValueError("audio_durations and time_offsets must match the batch size.")

        diarization_lengths_list: List[Optional[int]] = [None] * batch_size
        if diarization_labels is not None:
            if diarization_labels.ndim != 3 or diarization_labels.shape[0] != batch_size:
                raise ValueError("diarization_labels must have shape (batch, speakers, 10ms_frames).")
            if diarization_labels.dtype != torch.bool:
                raise TypeError("diarization_labels must use boolean storage.")
            diarization_lengths_list = self._lengths(
                diarization_lengths,
                batch_size,
                diarization_labels.shape[2],
            )
        elif diarization_lengths is not None:
            raise ValueError("diarization_lengths require diarization_labels.")
        if diarization_frame_seconds <= 0:
            raise ValueError("diarization_frame_seconds must be positive.")

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
        speaker_logprob_weight = (
            self.speaker_logprob_weight if speaker_logprob_weight is None else float(speaker_logprob_weight)
        )
        if speaker_logprob_weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")

        records, columns_per_record = [], []
        for index, transcript in enumerate(sot_transcripts):
            ctc_length = ctc_lengths_list[index]
            speaker_length = speaker_lengths_list[index]
            if durations[index] is not None:
                ctc_step = float(durations[index]) / ctc_length
            elif self.ctc_frame_seconds is not None:
                ctc_step = float(self.ctc_frame_seconds)
            else:
                raise ValueError(f"Record {index} has no audio duration, and ctc_frame_seconds is not set.")
            sortformer_step = (
                None
                if speaker_length is None
                else self._frame_seconds(
                    speaker_length,
                    durations[index],
                    self.sortformer_frame_seconds or None,
                    ctc_step,
                )
            )
            words = self._tokenize_words(
                _timestamp_units(
                    self.parse_sot_words(transcript),
                    self.timestamp_language or 'auto',
                ),
                blank_id,
            )
            streams = []
            for speaker_tag, speaker_words in self._group_words_by_speaker(words).items():
                labels, state_to_word = self._build_ctc_target(speaker_words, blank_id)
                tokens = labels[1::2]
                minimum_frames = len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))
                if minimum_frames > ctc_length:
                    raise ValueError(f"Speaker {speaker_tag!r} transcript is too long for the CTC timeline.")
                streams.append(
                    {
                        "speaker_tag": speaker_tag,
                        "words": speaker_words,
                        "labels": labels,
                        "state_to_word": state_to_word,
                    }
                )
            columns = [blank_id] + sorted({label for stream in streams for label in stream["labels"]} - {blank_id})
            column_of = {label: column for column, label in enumerate(columns)}
            for stream in streams:
                stream["labels"] = [column_of[label] for label in stream["labels"]]
            columns_per_record.append(columns)
            records.append(
                {
                    "words": words,
                    "streams": streams,
                    "ctc_log_probs": None,
                    "ctc_length": ctc_length,
                    "ctc_step": ctc_step,
                    "speaker_probs": (
                        None
                        if sortformer_sigmoids is None
                        else sortformer_sigmoids[index, :speaker_length].detach().float().cpu()
                    ),
                    "speaker_length": speaker_length,
                    "sortformer_step": sortformer_step,
                    "diarization_labels": (
                        None
                        if diarization_labels is None
                        else diarization_labels[index, :, : diarization_lengths_list[index]].detach().cpu()
                    ),
                    "audio_duration": durations[index],
                    "time_offset": float(offsets[index]),
                }
            )
        for record, log_probs in zip(records, read_columns(columns_per_record, ctc_lengths_list) if records else ()):
            record["ctc_log_probs"] = log_probs

        return {
            "config": {
                "speaker_activity_threshold": self.speaker_activity_threshold,
                "speaker_logprob_weight": speaker_logprob_weight,
                "maximum_token_len": self.maximum_token_len,
                "max_word_frames": self._ctc_frames(self.max_word_frames),
                "timestamp_language": self.timestamp_language,
                "epsilon": self.epsilon,
            },
            "num_speaker_columns": None if sortformer_sigmoids is None else sortformer_sigmoids.shape[-1],
            "diarization_frame_seconds": None if diarization_labels is None else float(diarization_frame_seconds),
            "diarization_max_speaker_count": None if diarization_labels is None else diarization_labels.shape[1],
            "records": records,
        }

    def align_prepared(self, prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Align a batch from :meth:`prepare_alignment` with single-pass banded CTC and word refinement.
        Reads only the batch and this aligner's hyperparameters, never the model, so it can
        run in another process than the one that prepared the batch.

        Args:
            prepared (Dict[str, Any]): A batch from :meth:`prepare_alignment`.

        Returns:
            List[Dict[str, Any]]: One alignment result per record.
        """
        return _align_banded_ctc(self, prepared)

    @staticmethod
    def diarization_segments(
        labels: torch.Tensor,
        *,
        frame_seconds: float,
        time_offset: float,
        audio_duration: Optional[float],
    ) -> List[Dict[str, Any]]:
        """
        Convert boolean ``(speakers, frames)`` activity into DER-ready segments.
        Each contiguous active run becomes one segment; ends are clipped to the audio duration and
        empty segments are dropped.

        Args:
            labels (torch.Tensor): Boolean speaker activity shaped ``(speakers, frames)``.
            frame_seconds (float): Duration of one label frame, in seconds.
            time_offset (float): Time of the first frame, in seconds.
            audio_duration (Optional[float]): Recording duration in seconds; segments end there.

        Returns:
            List[Dict[str, Any]]: ``speaker`` (the label row), ``start`` and ``end``, in start-time order.
        """
        segments = []
        for speaker, activity in enumerate(labels):
            padded = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int8),
                    activity.to(torch.int8),
                    torch.zeros(1, dtype=torch.int8),
                ]
            )
            transitions = padded[1:] - padded[:-1]
            starts = torch.nonzero(transitions == 1).flatten().tolist()
            ends = torch.nonzero(transitions == -1).flatten().tolist()
            for start_frame, end_frame in zip(starts, ends):
                start = time_offset + start_frame * frame_seconds
                end = time_offset + end_frame * frame_seconds
                if audio_duration is not None:
                    end = min(end, time_offset + float(audio_duration))
                if end > start:
                    segments.append(
                        {
                            "speaker": speaker,
                            "start": start,
                            "end": end,
                        }
                    )
        return sorted(segments, key=lambda segment: (segment["start"], segment["end"], segment["speaker"]))

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
        diarization_timestamps: Sequence[Dict[str, Any]],
        diarization_frame_seconds: Optional[float],
        diarization_max_speaker_count: Optional[int],
    ) -> Dict[str, Any]:
        """
        Format one timestamp-alignment result.

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
            diarization_timestamps (Sequence[Dict[str, Any]]): Native diarization activity segments.
            diarization_frame_seconds (Optional[float]): Native diarization frame duration.
            diarization_max_speaker_count (Optional[int]): Fixed boolean-label speaker dimension.

        Returns:
            Dict[str, Any]: Public timestamp result and alignment diagnostics.
        """
        return {
            "speaker_word_timestamps": MultiSpeakerSOTWordTimestampAligner._public_word_timestamps(rows),
            "diarization_timestamps": list(diarization_timestamps),
            "diarization_frame_seconds": diarization_frame_seconds,
            "diarization_max_speaker_count": diarization_max_speaker_count,
            "diarization_activity_threshold": 0.5 if diarization_frame_seconds is not None else None,
            "speaker_tag_to_sortformer_column": mapping,
            "alignment_mode": "banded",
            "requested_alignment_mode": "banded",
            "speaker_assignment_mode": "optimal",
            "ctc_frame_seconds": ctc_step,
            "sortformer_frame_seconds": sortformer_step,
            "time_offset": time_offset,
            "num_ctc_frames": ctc_frames,
            "num_sortformer_frames": sortformer_frames,
            "alignment_diagnostics": {
                "preliminary_ctc_path_scores": preliminary_scores,
                "final_path_scores": final_scores,
                "speaker_assignment_scores": assignment_scores,
            },
        }

    def _iter_ctc_windows(
        self, timestamp_inputs: Any, *, trunk_only: bool = False
    ) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """
        Yield one normalized core at a time, bounding batches by context/vocabulary size.
        Each core of ``online_inference_length`` frames is decoded with its left and right
        context, then trimmed back to the core. The records are split into sub-batches so a
        decoder call stays within ``_CTC_WINDOW_MAX_ELEMENTS``, unless one context window is
        already larger than that.

        Args:
            timestamp_inputs (Any): ``CTCTimestampInputs`` whose ``asr_encoded`` is shaped ``(B, D, T)``.
            trunk_only (bool): Yield :meth:`TransformerCTCDecoder.trunk` states ``(B, D, T)`` instead.

        Yields:
            Tuple[int, int, torch.Tensor]: First record index, first CTC frame index, and the
                core's log probs shaped ``(B_window, T_core, classes)``.
        """
        states = timestamp_inputs.asr_encoded
        lengths = timestamp_inputs.asr_encoded_lengths
        parameter = next(self.ctc_decoder.parameters(), None)
        decoder_dtype = parameter.dtype if parameter is not None else states.dtype
        core_length = self.online_inference_length
        core_length = core_length if core_length > 0 else states.shape[-1]
        width = states.shape[1] if trunk_only else getattr(self.ctc_decoder, "num_classes_with_blank", 1)
        decode = self.ctc_decoder.trunk if trunk_only else self.ctc_decoder
        frame_axis = -1 if trunk_only else 1
        for start in range(0, states.shape[-1], core_length):
            end = min(start + core_length, states.shape[-1])
            context_start = max(start - self.chunk_left_context, 0)
            context_end = min(end + self.chunk_right_context, states.shape[-1])
            context_frames = context_end - context_start
            batch_size = max(1, self._CTC_WINDOW_MAX_ELEMENTS // (self._ctc_frames(context_frames) * width))
            for batch_start in range(0, states.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, states.shape[0])
                context_lengths = (lengths[batch_start:batch_end] - context_start).clamp(min=0, max=context_frames)
                context_states = states[batch_start:batch_end, :, context_start:context_end].to(dtype=decoder_dtype)
                single = context_states.shape[0] == 1 and context_states.is_cuda
                if single:
                    # Two copies of a lone record, so the decoder's compiled attention, which torch.compile
                    # specializes for a batch of one, never needs a variant of its own for that case.
                    context_states, context_lengths = context_states.repeat(2, 1, 1), context_lengths.repeat(2)
                context_output = decode(context_states, encoded_lengths=context_lengths)
                if single:
                    context_output = context_output[:1]
                if context_output.shape[frame_axis] != self._ctc_frames(context_frames):
                    raise ValueError(
                        f"The CTC head returned {context_output.shape[frame_axis]} frames for {context_frames} "
                        "input frames."
                    )
                left_drop = self._ctc_frames(start - context_start)
                core_frames = self._ctc_frames(end - start)
                yield batch_start, self._ctc_frames(start), context_output.narrow(frame_axis, left_drop, core_frames)
                del context_output

    def _ctc_frames(self, encoder_frames: Union[int, torch.Tensor]) -> Union[int, torch.Tensor]:
        """
        Convert encoder frame counts to CTC frame counts.

        Args:
            encoder_frames (Union[int, torch.Tensor]): Encoder frame count(s).

        Returns:
            Union[int, torch.Tensor]: Counts scaled by the decoder's upsampling factor (1 by default).
        """
        return encoder_frames * int(getattr(self.ctc_decoder, "upsampling_factor", 1))

    @staticmethod
    def _group_words_by_speaker(
        words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """
        Group word records by their t-SOT speaker tag.

        Args:
            words (Sequence[Dict[str, Any]]): Word records in transcript order.

        Returns:
            Dict[Optional[int], List[Dict[str, Any]]]: Words grouped by speaker tag.
        """
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for word in words:
            grouped.setdefault(word["speaker_tag"], []).append(word)
        return grouped

    @staticmethod
    def _public_word_timestamps(
        words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """
        Return the minimal public CTC word-timestamp schema grouped by speaker.

        Args:
            words (Sequence[Dict[str, Any]]): Aligned word records with ``start`` and ``end``.

        Returns:
            Dict[Optional[int], List[Dict[str, Any]]]: ``word``, ``speaker``, ``start`` and ``end``
                records grouped by speaker tag.
        """
        grouped: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for word in words:
            speaker = word["speaker_tag"]
            grouped.setdefault(speaker, []).append(
                {
                    "word": word["word"],
                    "speaker": speaker,
                    "start": word["start"],
                    "end": word["end"],
                }
            )
        return grouped

    def _tokenize_words(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Attach CTC token IDs to parsed t-SOT words.
        Each speaker stream is tokenized separately; the output keeps the input word order.

        Args:
            words (Sequence[Dict[str, Any]]): Parsed t-SOT word records.
            blank_id (int): CTC blank class index.

        Returns:
            List[Dict[str, Any]]: Word records containing ``token_ids``.
        """
        tokenized = {}
        for stream_words in self._group_words_by_speaker(words).values():
            for word, token_ids in zip(stream_words, self._tokenize_word_stream(stream_words, blank_id)):
                tokenized[word["word_index"]] = {**word, "token_ids": token_ids}
        return [tokenized[word["word_index"]] for word in words]

    def _tokenize_word_stream(
        self,
        words: Sequence[Dict[str, Any]],
        blank_id: int,
    ) -> List[List[int]]:
        """
        Tokenize one speaker stream while preserving word ownership.
        When any word is a character unit, each word is tokenized on its own, characters with
        :func:`_character_token_ids`. Otherwise the joined stream is tokenized, and it must match
        the concatenated per-word tokenizations.

        Args:
            words (Sequence[Dict[str, Any]]): Words belonging to one speaker stream.
            blank_id (int): CTC blank class index.

        Returns:
            List[List[int]]: CTC token IDs grouped by word.
        """
        if any(word.get('alignment_unit') == 'character' for word in words):
            return [
                (
                    _character_token_ids(self.tokenizer.tokenizer, word['word'], blank_id)
                    if word.get('alignment_unit') == 'character'
                    else self._tokenize_word_stream([word], blank_id)[0]
                )
                for word in words
            ]
        text = " ".join(word["word"] for word in words)
        token_ids = [int(token_id) for token_id in self.tokenizer.text_to_ids(text)]
        if not token_ids or min(token_ids) < 0 or max(token_ids) >= blank_id:
            raise ValueError("Tokenizer produced IDs outside the non-blank CTC vocabulary.")
        ids_by_word = [[int(token_id) for token_id in self.tokenizer.text_to_ids(word["word"])] for word in words]
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
        """
        Build a blank-expanded CTC target and word-ownership mapping.

        Args:
            words (Sequence[Dict[str, Any]]): Tokenized word records.
            blank_id (int): CTC blank class index.

        Returns:
            Tuple[List[int], List[Optional[int]]]: Expanded labels and state-to-word indices.
        """
        labels = [blank_id]
        state_to_word: List[Optional[int]] = [None]
        for word_index, word in enumerate(words):
            for token_id in word["token_ids"]:
                labels.extend((token_id, blank_id))
                state_to_word.extend((word_index, None))
        return labels, state_to_word

    def _word_rows_from_path(
        self,
        words: Sequence[Dict[str, Any]],
        state_to_word: Sequence[Optional[int]],
        path: np.ndarray,
        ctc_step: float,
        time_offset: float,
    ) -> List[Dict[str, Any]]:
        """
        Convert CTC state paths into timestamped word records.
        A word spans the frames whose path state is one of its tokens; negative path states are
        skipped, and its end is capped at ``maximum_token_len`` seconds after its start.

        Args:
            words (Sequence[Dict[str, Any]]): Tokenized words for one speaker stream.
            state_to_word (Sequence[Optional[int]]): Target-state to word-index mapping.
            path (np.ndarray): Best CTC target state at every frame, ``-1`` where none.
            ctc_step (float): Duration of one CTC frame in seconds.
            time_offset (float): Offset added to returned timestamps.

        Returns:
            List[Dict[str, Any]]: Each word with its ``start`` and ``end`` in seconds.
        """
        frames_by_word: List[List[int]] = [[] for _ in words]
        for frame, state in enumerate(path.tolist()):
            if state < 0:
                continue
            word_index = state_to_word[state]
            if word_index is not None:
                frames_by_word[word_index].append(frame)

        rows = []
        for word, frames in zip(words, frames_by_word):
            if not frames:
                raise ValueError(f"CTC path did not visit word {word['word']!r}.")
            start = time_offset + frames[0] * ctc_step
            end = min(time_offset + (frames[-1] + 1) * ctc_step, start + self.maximum_token_len)
            rows.append({**word, "start": start, "end": end})
        return rows

    @staticmethod
    def _resample_speaker_probs(speaker_probs: torch.Tensor, target_frames: int) -> torch.Tensor:
        """
        Linearly resample speaker probabilities onto the CTC frame grid.

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
        """
        Normalize optional padded-sequence lengths.
        None yields ``maximum`` for every row.

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
        """
        Resolve and validate the CTC blank class index.
        Uses the explicit ``blank_id``, else the decoder's last class, else ``vocab_size - 1``.

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
        """
        Resolve the duration represented by one model frame.
        Prefers ``audio_duration / length``, then ``configured``, then ``default``.

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


def align_prepared_batch(prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Finish aligning a batch from :meth:`MultiSpeakerSOTWordTimestampAligner.prepare_alignment`.
    Builds an aligner from the hyperparameters the batch records, so it runs in any process:
    it needs no model, tokenizer or GPU.

    Args:
        prepared (Dict[str, Any]): A batch from ``prepare_alignment`` or ``prepare_from_inputs``.

    Returns:
        List[Dict[str, Any]]: One alignment result per record.
    """
    return MultiSpeakerSOTWordTimestampAligner(**prepared["config"]).align_prepared(prepared)


def merge_prepared_batches(batches: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Merge batches from :meth:`MultiSpeakerSOTWordTimestampAligner.prepare_alignment` into one.
    Each record retains its own frame support and alignment band. Batches merge only when every
    batch-level value, everything but ``records``, matches.

    Args:
        batches (Sequence[Dict[str, Any]]): One or more prepared batches.

    Returns:
        Optional[Dict[str, Any]]: One batch with every record, in order, or ``None`` when a
            batch-level value differs and the batches must be aligned separately.
    """
    batch_values = [{key: value for key, value in batch.items() if key != "records"} for batch in batches]
    if any(values != batch_values[0] for values in batch_values[1:]):
        return None
    return {**batch_values[0], "records": [record for batch in batches for record in batch["records"]]}


@dataclass(frozen=True)
class CTCTimestampArtifact:
    """Loaded CTC decoder and matching tokenizer."""

    decoder: TransformerCTCDecoder
    tokenizer: SentencePieceTokenizer
    decoder_config: dict[str, Any]


def _plain_mapping(value: Any, label: str) -> dict[str, Any]:
    """
    Convert a mapping or OmegaConf config into a plain ``dict``.

    Args:
        value (Any): Mapping or OmegaConf config; configs are resolved.
        label (str): Name of the value used in the error message.

    Returns:
        dict[str, Any]: Shallow ``dict`` copy of the mapping.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _decoder_init_config(value: Any) -> dict[str, Any]:
    """
    Keep only the decoder config fields accepted by :class:`TransformerCTCDecoder`.
    ``feat_in`` and ``num_classes`` must be present.

    Args:
        value (Any): Decoder config as a mapping or OmegaConf config.

    Returns:
        dict[str, Any]: Keyword arguments for ``TransformerCTCDecoder.__init__``.
    """
    config = _plain_mapping(value, "decoder_config")
    accepted = set(inspect.signature(TransformerCTCDecoder.__init__).parameters) - {"self"}
    init_config = {key: item for key, item in config.items() if key in accepted}
    missing = {"feat_in", "num_classes"} - set(init_config)
    if missing:
        raise ValueError(f"decoder_config is missing required field(s): {sorted(missing)}.")
    return init_config


def _tokenizer_payload(tokenizer: Any) -> dict[str, Any]:
    """
    Serialize a SentencePiece tokenizer's model proto and construction options.

    Args:
        tokenizer (Any): Tokenizer wrapping a SentencePiece processor in ``tokenizer.tokenizer``.

    Returns:
        dict[str, Any]: ``model_proto`` bytes plus ``legacy``, ``ignore_extra_whitespaces``,
            ``trim_spm_separator_after_special_token`` and ``spm_separator``.
    """
    processor = getattr(tokenizer, "tokenizer", None)
    serialize = getattr(processor, "serialized_model_proto", None)
    if not callable(serialize):
        raise TypeError("The CTC timestamp artifact requires a SentencePiece tokenizer.")
    return {
        "model_proto": bytes(serialize()),
        "legacy": bool(getattr(tokenizer, "legacy", False)),
        "ignore_extra_whitespaces": bool(getattr(tokenizer, "ignore_extra_whitespaces", True)),
        "trim_spm_separator_after_special_token": bool(
            getattr(tokenizer, "trim_spm_separator_after_special_token", True)
        ),
        "spm_separator": str(getattr(tokenizer, "spm_separator", "▁")),
    }


def save_ctc_timestamp_artifact(
    destination: Union[str, Path],
    decoder: TransformerCTCDecoder,
    tokenizer: Any,
    decoder_config: Any,
) -> Path:
    """
    Write the decoder, tokenizer, and construction config as one lightweight artifact.
    A ``.safetensors`` destination stores tensor data in safetensors and construction
    settings in its metadata. Other suffixes retain the original torch serialization.

    Args:
        destination (Union[str, Path]): Artifact path; parent directories are created.
        decoder (TransformerCTCDecoder): Decoder whose state dict is saved on the CPU.
        tokenizer (Any): SentencePiece tokenizer matching the decoder vocabulary.
        decoder_config (Any): Decoder construction config as a mapping or OmegaConf config.

    Returns:
        Path: Resolved artifact path.
    """
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _decoder_init_config(decoder_config)
    state_dict = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    tokenizer_payload = _tokenizer_payload(tokenizer)
    if path.suffix == ".safetensors":
        _save_ctc_timestamp_safetensors(path, state_dict, tokenizer_payload, config)
        return path
    payload = {
        "format": CTC_TIMESTAMP_ARTIFACT_FORMAT,
        "decoder_config": config,
        "decoder_state_dict": state_dict,
        "tokenizer": tokenizer_payload,
    }
    torch.save(payload, path)
    return path


def _load_sentencepiece(payload: Any) -> SentencePieceTokenizer:
    """
    Rebuild a SentencePiece tokenizer from a serialized payload.
    The model proto is written to a temporary ``.model`` file, which is deleted afterwards.

    Args:
        payload (Any): Mapping from :func:`_tokenizer_payload`.

    Returns:
        SentencePieceTokenizer: The restored tokenizer.
    """
    config = _plain_mapping(payload, "tokenizer")
    model_proto = config.get("model_proto")
    if not isinstance(model_proto, bytes) or not model_proto:
        raise ValueError("tokenizer.model_proto must contain serialized SentencePiece model bytes.")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as temporary:
            temporary.write(model_proto)
            temporary_path = temporary.name
        return SentencePieceTokenizer(
            model_path=temporary_path,
            legacy=bool(config.get("legacy", False)),
            ignore_extra_whitespaces=bool(config.get("ignore_extra_whitespaces", True)),
            trim_spm_separator_after_special_token=bool(config.get("trim_spm_separator_after_special_token", True)),
            spm_separator=str(config.get("spm_separator", "▁")),
        )
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def _validate_vocabulary(tokenizer: SentencePieceTokenizer, decoder_config: Mapping[str, Any]) -> None:
    """
    Check that the tokenizer's size and pieces match the decoder's ``vocabulary``, if one is configured.

    Args:
        tokenizer (SentencePieceTokenizer): Tokenizer loaded from the artifact.
        decoder_config (Mapping[str, Any]): Decoder construction config.
    """
    vocabulary = decoder_config.get("vocabulary")
    if vocabulary is None:
        return
    vocabulary = list(vocabulary)
    if tokenizer.vocab_size != len(vocabulary):
        raise ValueError(
            "Tokenizer vocabulary size does not match the decoder configuration: "
            f"{tokenizer.vocab_size} != {len(vocabulary)}."
        )
    actual = tokenizer.ids_to_tokens(list(range(len(vocabulary))))
    mismatch = next(
        (index for index, pair in enumerate(zip(actual, vocabulary)) if pair[0] != pair[1]),
        None,
    )
    if mismatch is not None:
        raise ValueError(
            "Tokenizer pieces do not match the decoder vocabulary at ID "
            f"{mismatch}: {actual[mismatch]!r} != {vocabulary[mismatch]!r}."
        )


def load_ctc_timestamp_artifact(
    source: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> CTCTimestampArtifact:
    """
    Load one self-contained timestamp artifact without restoring an ASR model.
    Reads either the safetensors or the torch format, loads the decoder state strictly, puts the
    decoder in eval mode with sequence-length sync disabled, and validates the tokenizer vocabulary.

    Args:
        source (Union[str, Path]): Artifact path.
        map_location (Union[str, torch.device]): Device for the decoder.
        dtype (Optional[torch.dtype]): Decoder dtype; None keeps the stored dtype.

    Returns:
        CTCTimestampArtifact: Decoder, tokenizer and decoder construction config.
    """
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CTC timestamp artifact does not exist: {path}")
    payload = (
        _load_ctc_timestamp_safetensors(path)
        if _is_safetensors_file(path)
        else torch.load(path, map_location="cpu", weights_only=True)
    )
    if not isinstance(payload, Mapping) or payload.get("format") != CTC_TIMESTAMP_ARTIFACT_FORMAT:
        raise ValueError(
            f"Expected a {CTC_TIMESTAMP_ARTIFACT_FORMAT!r} artifact. Convert the original .nemo adapter first."
        )
    config = _decoder_init_config(payload.get("decoder_config"))
    state_dict = payload.get("decoder_state_dict")
    if (
        not isinstance(state_dict, Mapping)
        or not state_dict
        or not all(isinstance(value, torch.Tensor) for value in state_dict.values())
    ):
        raise ValueError("decoder_state_dict must be a non-empty tensor mapping.")
    decoder = TransformerCTCDecoder(**config)
    decoder.load_state_dict(dict(state_dict), strict=True)
    decoder.to(device=map_location, dtype=dtype).eval()
    _disable_max_seq_length_sync(decoder)
    tokenizer = _load_sentencepiece(payload.get("tokenizer"))
    _validate_vocabulary(tokenizer, config)
    return CTCTimestampArtifact(decoder=decoder, tokenizer=tokenizer, decoder_config=config)


def _save_ctc_timestamp_safetensors(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    tokenizer_payload: Mapping[str, Any],
    decoder_config: Mapping[str, Any],
) -> None:
    """
    Write a CTC timestamp artifact as safetensors.
    Decoder tensors are stored under ``decoder.``, the tokenizer model proto as a uint8 tensor, and
    the format, decoder config and remaining tokenizer options as JSON metadata.

    Args:
        path (Path): Destination file.
        state_dict (Mapping[str, torch.Tensor]): Decoder state dict.
        tokenizer_payload (Mapping[str, Any]): Mapping from :func:`_tokenizer_payload`.
        decoder_config (Mapping[str, Any]): Decoder construction config.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("Saving a .safetensors CTC timestamp artifact requires safetensors.") from error

    model_proto = tokenizer_payload.get("model_proto")
    if not isinstance(model_proto, bytes) or not model_proto:
        raise ValueError("tokenizer.model_proto must contain serialized SentencePiece model bytes.")
    tokenizer_config = {key: value for key, value in tokenizer_payload.items() if key != "model_proto"}
    config = OmegaConf.to_container(OmegaConf.create(dict(decoder_config)), resolve=True)
    tensors = {f"decoder.{key}": value.detach().cpu().contiguous().clone() for key, value in state_dict.items()}
    tensors["tokenizer.model_proto"] = torch.frombuffer(bytearray(model_proto), dtype=torch.uint8).clone()
    metadata = {
        "format": CTC_TIMESTAMP_ARTIFACT_FORMAT,
        "decoder_config": json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        "tokenizer_config": json.dumps(tokenizer_config, ensure_ascii=False, separators=(",", ":")),
    }
    save_file(tensors, str(path), metadata=metadata)


def _is_safetensors_file(path: Path) -> bool:
    """
    Whether a file holds safetensors: an 8-byte little-endian header length, then the JSON header.
    Decided by content as well as suffix: a Hugging Face cache keeps ``model.safetensors`` as a
    symlink to a blob named by its hash, and callers pass the resolved blob.

    Args:
        path (Path): File to inspect.

    Returns:
        bool: True for a ``.safetensors`` suffix or a safetensors header.
    """
    if path.suffix == ".safetensors":
        return True
    with path.open("rb") as handle:
        prefix = handle.read(9)
    return len(prefix) == 9 and prefix[8:9] == b"{" and int.from_bytes(prefix[:8], "little") < path.stat().st_size


def _load_ctc_timestamp_safetensors(path: Path) -> dict[str, Any]:
    """
    Read a safetensors CTC timestamp artifact into the torch-format payload layout.
    Accepts a standalone artifact or a bundle whose keys and metadata use the ``ctc_timestamp.``
    prefix; a standalone artifact may hold no tensors besides the decoder and tokenizer.

    Args:
        path (Path): Safetensors file.

    Returns:
        dict[str, Any]: ``format``, ``decoder_config``, ``decoder_state_dict`` and ``tokenizer`` payload.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("Loading a .safetensors CTC timestamp artifact requires safetensors.") from error

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") == CTC_TIMESTAMP_ARTIFACT_FORMAT:
            prefix = ""
            decoder_config_key = "decoder_config"
            tokenizer_config_key = "tokenizer_config"
        elif metadata.get("ctc_timestamp_format") == CTC_TIMESTAMP_ARTIFACT_FORMAT:
            prefix = CTC_TIMESTAMP_BUNDLE_PREFIX
            decoder_config_key = "ctc_timestamp_decoder_config"
            tokenizer_config_key = "ctc_timestamp_tokenizer_config"
        else:
            raise ValueError(
                f"Expected a {CTC_TIMESTAMP_ARTIFACT_FORMAT!r} artifact. Convert the original .nemo adapter first."
            )
        try:
            decoder_config = _plain_mapping(json.loads(metadata[decoder_config_key]), "decoder_config")
            tokenizer_payload = _plain_mapping(json.loads(metadata[tokenizer_config_key]), "tokenizer_config")
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("Invalid CTC timestamp safetensors metadata.") from error
        decoder_prefix = f"{prefix}decoder."
        tensor_keys = list(handle.keys())
        if not prefix:
            unexpected = sorted(
                key for key in tensor_keys if key != "tokenizer.model_proto" and not key.startswith(decoder_prefix)
            )
            if unexpected:
                raise ValueError(f"Unexpected tensor(s) in CTC timestamp artifact: {unexpected}.")
        decoder_keys = [key for key in tensor_keys if key.startswith(decoder_prefix)]
        state_dict = {key.removeprefix(decoder_prefix): handle.get_tensor(key) for key in decoder_keys}
        model_proto = handle.get_tensor(f"{prefix}tokenizer.model_proto")

    if not isinstance(model_proto, torch.Tensor) or model_proto.dtype != torch.uint8 or model_proto.numel() == 0:
        raise ValueError("tokenizer.model_proto must be a non-empty uint8 tensor.")
    tokenizer_payload["model_proto"] = model_proto.contiguous().numpy().tobytes()
    return {
        "format": CTC_TIMESTAMP_ARTIFACT_FORMAT,
        "decoder_config": decoder_config,
        "decoder_state_dict": state_dict,
        "tokenizer": tokenizer_payload,
    }


def export_ctc_timestamp_artifact(source_nemo: Union[str, Path], destination: Union[str, Path]) -> Path:
    """
    Convert a training-time EncDecCTCModelBPE archive into the lightweight runtime artifact.
    :class:`TransformerCTCDecoder` is temporarily registered in ``nemo.collections.asr.modules`` so
    the archive can be restored.

    Args:
        source_nemo (Union[str, Path]): ``.nemo`` archive of the CTC timestamp adapter.
        destination (Union[str, Path]): Output artifact path.

    Returns:
        Path: Resolved artifact path from :func:`save_ctc_timestamp_artifact`.
    """
    import nemo.collections.asr.modules as asr_modules
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    source = Path(source_nemo).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CTC timestamp adapter does not exist: {source}")
    previous_decoder = getattr(asr_modules, "TransformerCTCDecoder", None)
    asr_modules.TransformerCTCDecoder = TransformerCTCDecoder
    try:
        adapter = EncDecCTCModelBPE.restore_from(str(source), map_location="cpu")
    finally:
        if previous_decoder is None:
            del asr_modules.TransformerCTCDecoder
        else:
            asr_modules.TransformerCTCDecoder = previous_decoder
    if not isinstance(adapter.decoder, TransformerCTCDecoder):
        raise TypeError(
            "CTC timestamp adapter decoder must be TransformerCTCDecoder, " f"got {type(adapter.decoder).__name__}."
        )
    return save_ctc_timestamp_artifact(destination, adapter.decoder, adapter.tokenizer, adapter.cfg.decoder)


def _disable_max_seq_length_sync(module: nn.Module) -> None:
    """
    Disable feature-length collectives in every inference-only decoder submodule.

    Args:
        module (nn.Module): Module whose submodules with ``sync_max_audio_length`` set are updated.
    """
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def get_ctc_timestamp_aligner(
    owner: nn.Module,
    artifact_path: Optional[str],
    device: torch.device,
) -> MultiSpeakerSOTWordTimestampAligner:
    """
    Load and cache an inference-only aligner without registering it in ``owner``'s module tree.
    The cache is keyed by the resolved path, and the aligner is rebuilt when ``owner``'s
    ``ctc_timestamp_alignment`` options change. Every call copies the encoder's deferred-head windows
    onto the cached aligner, so a change to them reaches it, and moves the decoder to ``device`` in
    ``owner``'s parameter dtype (float32 on CPU). The aligner keeps no reference to the encoder.

    Args:
        owner (nn.Module): Encoder module that holds the cache and the alignment and window settings.
        artifact_path (Optional[str]): Path to the lightweight CTC timestamp artifact.
        device (torch.device): Device for the CTC decoder.

    Returns:
        MultiSpeakerSOTWordTimestampAligner: The cached aligner.
    """
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("ctc_timestamp_model_path must be a non-empty lightweight artifact path.")
    resolved_path = os.path.realpath(os.path.expanduser(artifact_path))
    cached = owner.__dict__.get("_ctc_timestamp_extractor_cache")
    if cached is None or cached[0] != resolved_path:
        parameter = next(owner.parameters(), None)
        dtype = parameter.dtype if parameter is not None and device.type != "cpu" else torch.float32
        artifact = load_ctc_timestamp_artifact(resolved_path, map_location=device, dtype=dtype)
        aligner = MultiSpeakerSOTWordTimestampAligner(ctc_decoder=artifact.decoder, tokenizer=artifact.tokenizer)
        owner.__dict__["_ctc_timestamp_extractor_cache"] = (resolved_path, aligner)
        cached = (resolved_path, aligner)
        owner.__dict__.pop("_ctc_timestamp_alignment_options", None)
    aligner = cached[1]
    options = dict(getattr(owner, "ctc_timestamp_alignment", {}) or {})
    if options != owner.__dict__.get("_ctc_timestamp_alignment_options", {}):
        aligner = MultiSpeakerSOTWordTimestampAligner(
            ctc_decoder=aligner.ctc_decoder, tokenizer=aligner.tokenizer, **options
        )
        owner.__dict__["_ctc_timestamp_extractor_cache"] = (resolved_path, aligner)
        owner.__dict__["_ctc_timestamp_alignment_options"] = options
    aligner.online_inference_length = int(getattr(owner, "online_inference_length", 0))
    aligner.chunk_left_context = int(getattr(owner, "chunk_left_context", 0))
    aligner.chunk_right_context = int(getattr(owner, "chunk_right_context", 0))
    parameter = next(owner.parameters(), None)
    dtype = parameter.dtype if parameter is not None and device.type != "cpu" else torch.float32
    aligner.ctc_decoder.to(device=device, dtype=dtype).eval()
    _disable_max_seq_length_sync(aligner.ctc_decoder)
    return aligner


def _one_pass_speaker_mapping(ctc, streams, speaker, threshold, greedy):
    """
    Match token distributions, not alignment paths, to diarization columns.
    Greedy non-blank token onsets are counted per column while that column is active, and compared
    with each stream's target token counts by IDF-weighted cosine similarity of square-root counts,
    minus a penalty on the log ratio of their totals. A linear assignment then maps streams to columns;
    unassigned streams map to None.

    Args:
        ctc (np.ndarray): Compact CTC log probabilities shaped ``(T, columns)``, blank at column 0.
        streams (Sequence[Dict[str, Any]]): Speaker streams with ``speaker_tag`` and renumbered ``labels``.
        speaker (np.ndarray): Speaker probabilities on the CTC grid shaped ``(T, speakers)``.
        threshold (float): Activity threshold for counting a token under a column.
        greedy (np.ndarray): Greedy CTC column per frame shaped ``(T,)``.

    Returns:
        Dict[Optional[int], Optional[int]]: Sortformer column per speaker tag.
    """
    if not streams:
        return {}
    if speaker.shape[1] == 0:
        return {stream['speaker_tag']: None for stream in streams}
    from scipy.optimize import linear_sum_assignment

    peaks = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = np.stack(
        [
            np.bincount(greedy[peaks & (speaker[:, c] >= threshold)], minlength=ctc.shape[1])
            for c in range(speaker.shape[1])
        ]
    )
    expected = np.stack([np.bincount(np.asarray(s['labels'])[1::2], minlength=ctc.shape[1]) for s in streams])
    idf = np.log1p(len(observed) / (1 + (observed > 0).sum(axis=0)))
    a = np.sqrt(expected * idf)
    b = np.sqrt(observed * idf)
    similarity = a @ b.T / np.maximum(1e-8, np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :])
    # Penalize grossly incompatible amounts of speech, without a timing alignment.
    similarity -= 0.1 * np.abs(np.log((expected.sum(axis=1)[:, None] + 1) / (observed.sum(axis=1)[None, :] + 1)))
    rows, cols = linear_sum_assignment(-similarity)
    mapping = {s['speaker_tag']: None for s in streams}
    mapping.update({streams[r]['speaker_tag']: int(c) for r, c in zip(rows, cols)})
    return mapping


def _banded_ctc_search(logp, labels, positions, activity_prior, greedy=None):
    """
    Anchor-guided max-sum DP with byte backpointers; ties prefer stay, step, skip.
    The DP runs in a band of states. Under 1000 frames the band covers every state; otherwise its
    half-width is about 10% of the states plus 100. The band center interpolates from the first to the
    last state through anchors: trigrams that occur once among the greedy token onsets and once in the
    target, kept as the longest chain increasing in both time and state.

    Args:
        logp (np.ndarray): Compact CTC log probabilities shaped ``(T, columns)``, blank at column 0.
        labels (np.ndarray): Blank-expanded target labels shaped ``(states,)``.
        positions (np.ndarray): Indices of the frames searched, shaped ``(frames,)``.
        activity_prior (Optional[np.ndarray]): Log speaker activity per frame shaped ``(T,)``, added to
            non-blank states.
        greedy (Optional[np.ndarray]): Greedy CTC column per frame shaped ``(T,)``; None computes it
            from ``logp``.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: State per searched frame (all ``-1`` when no path
            exists), the path score (``-inf`` when none), and the first band state per frame.
    """
    frames, states = len(positions), len(labels)
    half = states if frames < 1000 else min(states, math.ceil(0.10 * states + 100))
    width = min(states, 2 * half + 1)
    starts = np.clip(np.rint(np.linspace(0, states - 1, frames)).astype(np.int64) - half, 0, states - width)
    target = labels[1::2]
    if frames < len(target) + np.count_nonzero(target[1:] == target[:-1]):
        return np.full(frames, -1, np.int64), -np.inf, starts
    if not frames:
        return np.full(frames, -1, np.int64), -np.inf, starts
    greedy = logp[positions].argmax(axis=1) if greedy is None else greedy[positions]
    starts = _ctc_anchor_band_starts(greedy, target, states, half, width)
    previous, pointers = _ctc_banded_forward(logp, labels, positions, activity_prior, starts, width)
    path, score = _ctc_banded_backtrace(previous, pointers, starts, states)
    return path, score, starts


def _ctc_anchor_band_starts(greedy, target, states, half, width):
    """
    Return band starts interpolated through a monotonic chain of unique trigram anchors.

    Args:
        greedy (np.ndarray): Greedy CTC columns for the searched frames, shaped ``(frames,)``;
            column 0 is blank.
        target (np.ndarray): Target token columns without inserted blanks, shaped ``(tokens,)``.
        states (int): Number of blank-expanded target states.
        half (int): Band half-width in states before clipping to the target boundaries.
        width (int): Number of states scored per frame, at most ``states``.

    Returns:
        np.ndarray: First band state per searched frame, shaped ``(frames,)`` with int64 dtype.
            Empty input returns an empty array.
    """
    frames = len(greedy)
    if not frames:
        return np.empty(0, dtype=np.int64)
    keep = (greedy != 0) & np.r_[True, greedy[1:] != greedy[:-1]]
    observed = greedy[keep]
    times = np.flatnonzero(keep)

    left, right = _ctc_trigram_positions(observed), _ctc_trigram_positions(target)
    pairs = sorted(
        (int(times[v[0] + 1]), 2 * (right[k][0] + 1) + 1)
        for k, v in left.items()
        if len(v) == 1 and len(right.get(k, [])) == 1
    )
    tails = []
    tail_ids = []
    parent = []
    for i, (_, state) in enumerate(pairs):
        j = bisect_left(tails, state)
        parent.append(tail_ids[j - 1] if j else -1)
        if j == len(tails):
            tails.append(state)
            tail_ids.append(i)
        else:
            tails[j] = state
            tail_ids[j] = i
    chosen = []
    k = tail_ids[-1] if tail_ids else -1
    while k >= 0:
        chosen.append(pairs[k])
        k = parent[k]
    chosen = chosen[::-1]
    points = [(0, 0)] + [(t, s) for t, s in chosen if 0 < t < frames - 1] + [(frames - 1, states - 1)]
    center = np.interp(np.arange(frames), [v[0] for v in points], [v[1] for v in points])
    starts = np.clip(np.rint(center).astype(np.int64) - half, 0, states - width)
    starts = np.minimum.accumulate(starts - 2 * np.arange(frames)) + 2 * np.arange(frames)
    starts = np.maximum(starts, np.maximum(0, states - width - 2 * (frames - 1 - np.arange(frames))))
    return starts


def _ctc_banded_forward(logp, labels, positions, activity_prior, starts, width):
    """
    Score the band using max-sum DP; ties prefer stay, step, then skip. Compiled with Numba when available.

    Args:
        logp (np.ndarray): Compact CTC log probabilities shaped ``(T, columns)``, blank at column 0.
        labels (np.ndarray): Blank-expanded target column indices shaped ``(states,)``.
        positions (np.ndarray): Nonempty indices of the frames searched, shaped ``(frames,)``.
        activity_prior (Optional[np.ndarray]): Log speaker activity shaped ``(T,)``, added only to
            non-blank states, or None to disable the prior.
        starts (np.ndarray): First band state per frame, shaped ``(frames,)``. Must start at zero
            and advance by zero, one, or two states per frame.
        width (int): Number of states scored per frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Final-frame scores shaped ``(width,)`` and uint8 backpointers
            shaped ``(frames, width)``, encoding stay (0), step (1), or skip (2).
    """
    if not NUMBA_AVAILABLE:
        return _ctc_banded_forward_numpy(logp, labels, positions, activity_prior, starts, width)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    return _ctc_banded_forward_compiled(
        np.ascontiguousarray(logp),
        labels,
        np.ascontiguousarray(positions, dtype=np.int64),
        np.zeros(1, np.float32) if activity_prior is None else np.ascontiguousarray(activity_prior),
        activity_prior is not None,
        np.ascontiguousarray(starts, dtype=np.int64),
        int(width),
        can_skip,
    )


def _ctc_banded_forward_numpy(logp, labels, positions, activity_prior, starts, width):
    """:func:`_ctc_banded_forward` without Numba: a NumPy loop over frames."""
    frames = len(positions)
    previous = np.full(width, -np.inf, np.float32)
    previous[: min(2, width)] = logp[positions[0], labels[: min(2, width)]]
    if activity_prior is not None:
        previous[: min(2, width)] += np.where(labels[: min(2, width)] != 0, activity_prior[positions[0]], 0)
    pointers = np.zeros((frames, width), np.uint8)
    padded = np.full(width + 4, -np.inf, np.float32)
    can_skip = (labels != 0) & (labels != np.r_[labels[:2], labels[:-2]])
    for t in range(1, frames):
        start = int(starts[t])
        delta = start - int(starts[t - 1])
        padded[2 : width + 2] = previous
        best = padded[2 + delta : 2 + delta + width].copy()
        one = padded[1 + delta : 1 + delta + width]
        two = padded[delta : delta + width]
        move = pointers[t]
        take = one > best
        np.copyto(best, one, where=take)
        move[take] = 1
        take = (two > best) & can_skip[start : start + width]
        np.copyto(best, two, where=take)
        move[take] = 2
        target = labels[start : start + width]
        previous = best + logp[positions[t], target]
        if activity_prior is not None:
            previous += np.where(target != 0, activity_prior[positions[t]], 0)
    return previous, pointers


if NUMBA_AVAILABLE:

    @jit(nopython=True, nogil=True, cache=True)
    def _ctc_banded_forward_compiled(logp, labels, positions, activity_prior, has_prior, starts, width, can_skip):
        """:func:`_ctc_banded_forward_numpy`, compiled, with the same arithmetic and tie order."""
        frames = positions.shape[0]
        previous = np.full(width, -np.inf, np.float32)
        current = np.empty(width, np.float32)
        pointers = np.zeros((frames, width), np.uint8)
        first = positions[0]
        for j in range(min(2, width)):
            previous[j] = logp[first, labels[j]]
            if has_prior and labels[j] != 0:
                previous[j] = previous[j] + activity_prior[first]
        for t in range(1, frames):
            start = starts[t]
            delta = start - starts[t - 1]
            position = positions[t]
            for j in range(width):
                # previous[k] is state start + j - (number of states moved) in the previous frame's band.
                k = j + delta
                best = previous[k] if k < width else np.float32(-np.inf)
                move = 0
                one = previous[k - 1] if 0 <= k - 1 < width else np.float32(-np.inf)
                if one > best:
                    best = one
                    move = 1
                two = previous[k - 2] if 0 <= k - 2 < width else np.float32(-np.inf)
                if two > best and can_skip[start + j]:
                    best = two
                    move = 2
                pointers[t, j] = move
                label = labels[start + j]
                current[j] = best + logp[position, label]
                if has_prior and label != 0:
                    current[j] = current[j] + activity_prior[position]
            previous, current = current, previous
        return previous, pointers


def _ctc_banded_backtrace(previous, pointers, starts, states):
    """
    Recover the best terminal path, or an all-minus-one path when no finite path exists.

    Args:
        previous (np.ndarray): Final-frame band scores shaped ``(width,)``.
        pointers (np.ndarray): Stay/step/skip backpointers shaped ``(frames, width)``; frames must
            be nonempty.
        starts (np.ndarray): First band state per frame, shaped ``(frames,)``. The final band must
            contain at least one of the two terminal states.
        states (int): Number of blank-expanded target states.

    Returns:
        Tuple[np.ndarray, float]: Target state per searched frame, shaped ``(frames,)``, and path
            score. An unreachable terminal state yields an all-minus-one path and ``-inf`` score.
    """
    frames, width = pointers.shape
    path = np.full(frames, -1, np.int64)
    ends = np.array([states - 1, states - 2])
    local = ends - starts[-1]
    valid = (local >= 0) & (local < width)
    ends = ends[valid]
    local = local[valid]
    best = previous[local].argmax()
    score = float(previous[local[best]])
    if np.isfinite(score):
        state = int(ends[best])
        for t in range(frames - 1, -1, -1):
            path[t] = state
            if t:
                state -= int(pointers[t, state - starts[t]])
    return path, score


def _refine_word_timestamps(result, max_frames=12):
    """
    Return a copy with same-speaker word boundaries refined inside VAD segments.
    Segment membership uses the original onset and half-open [start, end) VAD intervals. The first
    word of a segment snaps its onset back to the segment onset; other offsets extend to the next
    word in the segment or the segment end. Unmatched words retain their onset; their offset uses the
    next same-speaker onset, or the original offset for the final word. Onset moves and durations are
    capped at ``max_frames`` frames of ``min(ctc_frame_seconds, 0.08)`` seconds, so the cap respects
    both the actual CTC frame duration and nominal 80 ms frames.

    Args:
        result (Dict[str, Any]): Alignment result with ``speaker_word_timestamps``, ``ctc_frame_seconds``,
            ``diarization_timestamps`` and ``speaker_tag_to_sortformer_column``.
        max_frames (int): Positive cap in frames.

    Returns:
        Dict[str, Any]: The result with refined ``speaker_word_timestamps`` and
            ``word_timestamp_refinement`` statistics.
    """
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError('max_frames must be a positive integer')
    step = float(result['ctc_frame_seconds'])
    if not math.isfinite(step) or step <= 0:
        raise ValueError('ctc_frame_seconds must be positive and finite')
    limit = max_frames * min(step, 0.08)
    segments = defaultdict(list)
    for segment in result.get('diarization_timestamps', []):
        if segment['end'] > segment['start']:
            segments[int(segment['speaker'])].append((segment['start'], segment['end']))
    for intervals in segments.values():
        intervals.sort()
    mapping = result.get('speaker_tag_to_sortformer_column', {})
    refined = {}
    moved = unmatched = count = 0
    for tag, source in result['speaker_word_timestamps'].items():
        words = sorted((dict(w) for w in source), key=lambda w: w['start'])
        column = mapping.get(tag, mapping.get(str(tag)))
        intervals = segments.get(column, [])
        starts = [s for s, _ in intervals]
        memberships = []
        for word in words:
            index = bisect_right(starts, word['start']) - 1
            memberships.append(index if index >= 0 and word['start'] < intervals[index][1] else None)
        seen = set()
        for word, index in zip(words, memberships):
            if index is None:
                unmatched += 1
            elif index not in seen:
                onset = intervals[index][0]
                if 0 < word['start'] - onset <= limit:
                    word['start'] = onset
                    moved += 1
                seen.add(index)
        for i, (word, index) in enumerate(zip(words, memberships)):
            if index is None:
                end = words[i + 1]['start'] if i + 1 < len(words) else word['end']
            elif i + 1 < len(words) and memberships[i + 1] == index:
                end = words[i + 1]['start']
            else:
                end = intervals[index][1]
            word['end'] = max(word['start'], min(end, word['start'] + limit))
        refined[tag] = words
        count += len(words)
    return {
        **result,
        'speaker_word_timestamps': refined,
        'word_timestamp_refinement': {
            'method': 'diarization_boundary_refinement',
            'max_frames': max_frames,
            'max_seconds': limit,
            'words': count,
            'onsets_moved': moved,
            'unmatched_words': unmatched,
        },
    }


def _align_banded_ctc(aligner, prepared):
    """
    Align a prepared batch with the single-pass ``banded`` backend.
    Per record, speaker tags are mapped to Sortformer columns by :func:`_one_pass_speaker_mapping`,
    and each stream is searched by :func:`_banded_ctc_search` over frames whose mapped activity is at
    least 0.01, with the log activity as a prior. Streams with no legal path are reported in
    ``failed_streams``, and word timestamps are refined by :func:`_refine_word_timestamps`.

    Args:
        aligner (MultiSpeakerSOTWordTimestampAligner): Aligner supplying hyperparameters and helpers.
        prepared (Dict[str, Any]): A batch from :meth:`MultiSpeakerSOTWordTimestampAligner.prepare_alignment`.

    Returns:
        List[Dict[str, Any]]: One alignment result per record.
    """
    results = []
    for record in prepared['records']:
        ctc = record['ctc_log_probs'].numpy()
        greedy = ctc.argmax(axis=1)
        diagnostics = []
        frames = record['ctc_length']
        probs = record['speaker_probs']
        tags = [s['speaker_tag'] for s in record['streams']]
        if probs is None:
            speaker = None
            mapping = {tag: None for tag in tags}
        else:
            speaker = aligner._resample_speaker_probs(probs.clamp(0, 1), frames)
            mapping = _one_pass_speaker_mapping(
                ctc, record['streams'], speaker.numpy(), aligner.speaker_activity_threshold, greedy
            )
        rows, scores, failures = [], {}, []
        for stream in record['streams']:
            labels = np.asarray(stream['labels'], np.int64)
            column = mapping[stream['speaker_tag']]
            activity = (
                np.ones(frames)
                if speaker is None
                else (np.zeros(frames) if column is None else speaker[:, column].numpy())
            )
            # Conservative support selection is separate from the >0.5 diarization labels.
            positions = np.flatnonzero(activity >= 0.01)
            prior = prepared['config']['speaker_logprob_weight'] * np.log(np.maximum(activity, aligner.epsilon))
            path, score, _ = _banded_ctc_search(ctc, labels, positions, prior, greedy)
            failed = not np.isfinite(score)
            half = len(labels) if len(positions) < 1000 else min(len(labels), math.ceil(0.10 * len(labels) + 100))
            diagnostics.append(
                dict(
                    speaker=stream['speaker_tag'],
                    frames=frames,
                    retained_frames=len(positions),
                    states=len(labels),
                    half_width=half,
                    attempts=1,
                    fallback=False,
                    failed=failed,
                    mapping=column,
                    visited_cells=len(positions) * min(len(labels), 2 * half + 1),
                    full_cells=frames * len(labels),
                )
            )
            scores[stream['speaker_tag']] = score if not failed else None
            if failed:
                failures.append(
                    {
                        'speaker': stream['speaker_tag'],
                        'reason': 'no_legal_path_in_anchor_band',
                        'words': [w['word'] for w in stream['words']],
                    }
                )
                continue
            fullpath = np.full(frames, -1, np.int64)
            fullpath[positions] = path
            rows.extend(
                aligner._word_rows_from_path(
                    stream['words'],
                    stream['state_to_word'],
                    fullpath,
                    record['ctc_step'],
                    record['time_offset'],
                )
            )
        diarization = (
            []
            if record['diarization_labels'] is None
            else aligner.diarization_segments(
                record['diarization_labels'],
                frame_seconds=prepared['diarization_frame_seconds'],
                time_offset=record['time_offset'],
                audio_duration=record['audio_duration'],
            )
        )
        result = aligner._result(
            rows=rows,
            mapping=mapping,
            ctc_step=record['ctc_step'],
            sortformer_step=record['sortformer_step'],
            time_offset=record['time_offset'],
            ctc_frames=frames,
            sortformer_frames=record['speaker_length'],
            preliminary_scores={},
            final_scores=scores,
            assignment_scores={},
            diarization_timestamps=diarization,
            diarization_frame_seconds=prepared['diarization_frame_seconds'],
            diarization_max_speaker_count=prepared['diarization_max_speaker_count'],
        )
        result['speaker_assignment_mode'] = 'token_distribution'
        result['alignment_diagnostics']['failed_streams'] = failures
        result['alignment_diagnostics']['one_pass'] = True
        result['alignment_diagnostics']['banded'] = diagnostics
        results.append(_refine_word_timestamps(result, prepared['config']['max_word_frames']))
    return results


def _lexical_characters(text):
    """
    Return the letter, mark and number characters of NFC-normalized text.

    Args:
        text (str): Input text.

    Returns:
        List[str]: The lexical characters in order.
    """
    return [c for c in unicodedata.normalize('NFC', text) if unicodedata.category(c)[0] in 'LMN']


def _timestamp_units(words, language):
    """
    Choose units per transcript, without dataset metadata or shared mutable language state.
    ``auto`` picks ko, ja, zh or th from the Unicode names of non-tag characters. zh and ja words
    become lexical characters, and th text becomes pythainlp ``newmm`` words; any other language
    returns the words unchanged. Event/style tags stay atomic, and units never cross turns or tags.

    Args:
        words (List[Dict[str, Any]]): Word records from ``parse_sot_words``.
        language (Optional[str]): Language code, ``auto`` for script detection, or None.

    Returns:
        List[Dict[str, Any]]: Unit records with renumbered ``word_index``; split units also carry
            ``alignment_unit``.
    """
    if language == 'auto':
        names = [
            unicodedata.name(c, '')
            for word in words
            if not re.fullmatch(r'<[^<>]+>', word['word'])
            for c in word['word']
        ]
        # Script heuristic matching the evaluated policy; an explicit language overrides it.
        language = next(
            (
                lang
                for lang, prefixes in (
                    ('ko', ('HANGUL',)),
                    ('ja', ('HIRAGANA', 'KATAKANA')),
                    ('zh', ('CJK UNIFIED IDEOGRAPH', 'CJK COMPATIBILITY IDEOGRAPH')),
                    ('th', ('THAI',)),
                )
                if any(name.startswith(prefixes) for name in names)
            ),
            None,
        )
    if language not in ('zh', 'ja', 'th'):
        return words
    output = []
    if language == 'th':
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError as error:
            raise ImportError('Thai timestamp units require pythainlp==5.3.4.') from error
    # Preserve event/style tags as atomic targets and never segment across turns or tags.
    for (_, _, special), group in groupby(
        words,
        key=lambda w: (w.get('speaker_tag'), w.get('turn_index'), bool(re.fullmatch(r'<[^<>]+>', w['word']))),
    ):
        group = list(group)
        if special:
            for word in group:
                output.append({**word, 'word_index': len(output)})
        elif language == 'th':
            text = unicodedata.normalize('NFKC', ''.join(w['word'] for w in group)).casefold()
            text = ''.join(c for c in text if unicodedata.category(c)[0] in 'LMN')
            for unit in word_tokenize(text, engine='newmm', keep_whitespace=False):
                output.append({**group[0], 'word': unit, 'word_index': len(output), 'alignment_unit': 'word'})
        else:
            for word in group:
                for unit in _lexical_characters(word['word']):
                    output.append({**word, 'word': unit, 'word_index': len(output), 'alignment_unit': 'character'})
    return output


def _character_token_ids(processor, character, blank_id):
    """
    Represent a character with existing CTC pieces, without introducing word-boundary pieces.
    Uses the character's own piece when it exists, else its encoding without bare ``▁`` pieces. The
    IDs must be known, non-blank, and decode back to the character.

    Args:
        processor (Any): SentencePiece processor of the CTC tokenizer.
        character (str): Character to represent.
        blank_id (int): CTC blank class index; every ID must be below it.

    Returns:
        List[int]: CTC token IDs for the character.
    """
    index = processor.piece_to_id(character)
    ids = (
        [index]
        if index != processor.unk_id()
        else [token for token in processor.encode(character) if processor.id_to_piece(token) != '▁']
    )
    if (
        not ids
        or processor.unk_id() in ids
        or min(ids) < 0
        or max(ids) >= blank_id
        or ''.join(_lexical_characters(processor.decode(ids))) != character
    ):
        raise ValueError(f'Unrepresentable character: {character!r}')
    return ids


def _read_deferred_ctc_columns(
    aligner, timestamp_inputs, num_classes: int, columns: List[List[int]], lengths: List[int]
) -> List[torch.Tensor]:
    """
    Run the CTC head window by window and copy each record's kept columns to the host.
    The full-vocabulary log probabilities exist for one slice of frames at a time.

    Args:
        aligner (MultiSpeakerSOTWordTimestampAligner): Aligner owning the deferred CTC head.
        timestamp_inputs (Any): Request-owned encoder states and their valid lengths.
        num_classes (int): Expected CTC vocabulary size including blank.
        columns (List[List[int]]): Per-record CTC class indices to keep, blank first.
        lengths (List[int]): Per-record valid CTC frame counts.

    Returns:
        List[torch.Tensor]: Per-record float32 CPU log probabilities shaped ``(length, columns)``.
    """
    states = timestamp_inputs.asr_encoded
    width = max(len(record_columns) for record_columns in columns)
    # Records keep different numbers of columns; padding repeats each record's blank column and is dropped below.
    index = torch.tensor(
        [record_columns + record_columns[:1] * (width - len(record_columns)) for record_columns in columns],
        device=states.device,
    )
    # An explicit dtype: a vLLM server prepares on a thread of its own, while the speech encoder on the engine thread
    # sets the process-wide default dtype around its streaming steps.
    kept = torch.empty(
        (len(columns), aligner._ctc_frames(states.shape[-1]), width), dtype=torch.float32, device=states.device
    )
    decoder = aligner.ctc_decoder
    # A decoder without a separate trunk and classifier yields whole windows of log probabilities.
    split = callable(getattr(decoder, "trunk", None)) and callable(getattr(decoder, "classify", None))
    for batch_start, start, window in aligner._iter_ctc_windows(timestamp_inputs, trunk_only=split):
        rows = window.shape[0]
        rows_index = index[batch_start : batch_start + rows, None, :]
        if split:
            step = max(1, aligner._CTC_WINDOW_MAX_ELEMENTS // (rows * num_classes))
            slices = (
                (offset, decoder.classify(window[:, :, offset : offset + step]))
                for offset in range(0, window.shape[-1], step)
            )
        else:
            slices = ((0, window),)
        for offset, log_probs in slices:
            if log_probs.shape[-1] != num_classes:
                raise ValueError(f"The CTC head returned {log_probs.shape[-1]} classes, not {num_classes}.")
            span = log_probs.shape[1]
            kept[batch_start : batch_start + rows, start + offset : start + offset + span] = log_probs.gather(
                2, rows_index.expand(-1, span, -1)
            )
            del log_probs
        # The vocabulary-wide output must not outlive its window.
        del window, slices
    host = torch.empty(kept.shape, dtype=kept.dtype, pin_memory=kept.is_cuda)
    host.copy_(kept)
    return [
        host[row, :length, : len(record_columns)].clone()
        for row, (record_columns, length) in enumerate(zip(columns, lengths))
    ]


def _read_ctc_columns(ctc_log_probs: torch.Tensor, columns: List[List[int]], lengths: List[int]) -> List[torch.Tensor]:
    """
    Gather each record's kept columns from the full CTC output onto the host.

    Args:
        ctc_log_probs (torch.Tensor): Full CTC log probabilities shaped ``(B, T, classes)``.
        columns (List[List[int]]): Per-record CTC class indices to keep, blank first.
        lengths (List[int]): Per-record valid CTC frame counts.

    Returns:
        List[torch.Tensor]: Per-record float32 CPU log probabilities shaped ``(length, columns)``.
    """
    return [
        ctc_log_probs[index, :length]
        .detach()
        .index_select(1, torch.tensor(record_columns, device=ctc_log_probs.device))
        .float()
        .cpu()
        for index, (record_columns, length) in enumerate(zip(columns, lengths))
    ]


def _ctc_trigram_positions(tokens):
    """
    Index every token trigram by the positions where it starts.

    Args:
        tokens (np.ndarray): Token sequence.

    Returns:
        Dict[Tuple[int, int, int], List[int]]: Start positions of each trigram.
    """
    table = defaultdict(list)
    for i in range(len(tokens) - 2):
        table[tuple(tokens[i : i + 3])].append(i)
    return table
