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
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.asr.modules.conv_asr import ConvASRDecoder
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.parts.preprocessing.features import normalize_batch
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

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
    """Align speaker streams with exact max-sum CTC DP and packed two-bit backpointers."""

    _SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
    _EMISSION_BLOCK_FRAMES = 256

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
        speaker_logprob_weight: float = 0.0,
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
                words.append(
                    dict(
                        word=word,
                        speaker_tag=speaker_tag,
                        turn_index=turn_index,
                        word_index=len(words),
                    )
                )
        return words

    def extract_ctc_and_sortformer_batch(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the raw PEE ASR/diarization branches and the CTC adapter."""
        pee = getattr(self.encoder, "encoder", self.encoder)
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
                rounding_mode="floor",
            )
        return {
            "ctc_log_probs": ctc_log_probs,
            "ctc_lengths": speech_lengths.clamp(max=ctc_log_probs.shape[1]),
            "sortformer_sigmoids": speaker_probs,
            "sortformer_lengths": speaker_lengths.clamp(min=1, max=speaker_probs.shape[1]),
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
            sample_rate = getattr(preprocessor, "_sample_rate", getattr(preprocessor, "sample_rate", None))
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
        diarization_labels: Optional[torch.Tensor] = None,
        diarization_lengths: Optional[torch.Tensor] = None,
        diarization_frame_seconds: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Align every record and independent speaker stream in shared DP/backtrace batches."""
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

        diarization_lengths_list: List[Optional[int]] = [None] * batch_size
        diarization_cpu = None
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
            diarization_cpu = diarization_labels.detach().cpu()
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
        ctc_cpu = ctc_log_probs.detach().float().cpu()
        speaker_cpu = None if sortformer_sigmoids is None else sortformer_sigmoids.detach().float().cpu()
        weight = self.speaker_logprob_weight if speaker_logprob_weight is None else float(speaker_logprob_weight)
        if weight < 0:
            raise ValueError("speaker_logprob_weight must be non-negative.")

        speaker_by_record: List[Optional[torch.Tensor]] = [None] * batch_size
        aligned_speaker_cpu = None
        if speaker_cpu is not None:
            aligned_speaker_cpu = torch.zeros(
                (batch_size, max_ctc_frames, speaker_cpu.shape[-1]),
                dtype=speaker_cpu.dtype,
            )
            for index, speaker_length in enumerate(speaker_lengths_list):
                speaker_probs = self._resample_speaker_probs(
                    speaker_cpu[index, :speaker_length].clamp(0, 1),
                    ctc_lengths_list[index],
                )
                speaker_by_record[index] = speaker_probs
                aligned_speaker_cpu[index, : ctc_lengths_list[index]] = speaker_probs

        pee = getattr(self.encoder, "encoder", self.encoder)
        contexts = []
        streams = []
        for index, transcript in enumerate(sot_transcripts):
            speaker_length = speaker_lengths_list[index]
            ctc_step = self._frame_seconds(
                ctc_lengths_list[index],
                durations[index],
                self.ctc_frame_seconds,
                self.input_frame_seconds * float(getattr(pee, "subsampling_factor", 1)),
            )
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
            words = self._tokenize_words(self.parse_sot_words(transcript), blank_id)
            contexts.append(
                {
                    "words": words,
                    "ctc_step": ctc_step,
                    "sortformer_step": sortformer_step,
                    "mapping": {},
                    "assignment_scores": {},
                    "preliminary_scores": {},
                    "final_scores": {},
                    "rows": [],
                    "diarization_timestamps": (
                        []
                        if diarization_cpu is None
                        else self._diarization_segments(
                            diarization_cpu[index, :, : diarization_lengths_list[index]],
                            frame_seconds=float(diarization_frame_seconds),
                            time_offset=float(offsets[index]),
                            audio_duration=durations[index],
                        )
                    ),
                }
            )
            for speaker_tag, speaker_words in self._group_words_by_speaker(words).items():
                labels, state_to_word = self._build_ctc_target(speaker_words, blank_id)
                tokens = labels[1::2]
                minimum_frames = len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))
                if minimum_frames > ctc_lengths_list[index]:
                    raise ValueError(f"Speaker {speaker_tag!r} transcript is too long for the CTC timeline.")
                streams.append(
                    {
                        "record_index": index,
                        "speaker_tag": speaker_tag,
                        "words": speaker_words,
                        "labels": labels,
                        "state_to_word": state_to_word,
                    }
                )

        if streams:
            empty_mappings = [{} for _ in range(batch_size)]
            preliminary_paths, preliminary_path_scores = self._align_stream_batch(
                streams,
                ctc_cpu,
                ctc_lengths_list,
                blank_id,
                empty_mappings,
                speaker_probs=None,
                speaker_logprob_weight=0.0,
            )
            preliminary_rows = [[] for _ in range(batch_size)]
            for stream, path, score in zip(streams, preliminary_paths, preliminary_path_scores):
                record_index = stream["record_index"]
                context = contexts[record_index]
                rows = self._word_rows_from_path(
                    stream["words"],
                    stream["labels"],
                    stream["state_to_word"],
                    path,
                    ctc_cpu[record_index, : ctc_lengths_list[record_index]],
                    None,
                    {},
                    context["ctc_step"],
                    float(offsets[record_index]),
                )
                preliminary_rows[record_index].extend(rows)
                context["preliminary_scores"][stream["speaker_tag"]] = score

            for index, context in enumerate(contexts):
                speaker_tags = list(
                    dict.fromkeys(word["speaker_tag"] for word in context["words"] if word["speaker_tag"] is not None)
                )
                context["mapping"], context["assignment_scores"] = self._resolve_speaker_mapping(
                    speaker_tags,
                    preliminary_rows[index],
                    speaker_by_record[index],
                )

            use_speaker_dp = bool(
                aligned_speaker_cpu is not None
                and weight > 0
                and any(column is not None for context in contexts for column in context["mapping"].values())
            )
            if use_speaker_dp:
                final_paths, final_path_scores = self._align_stream_batch(
                    streams,
                    ctc_cpu,
                    ctc_lengths_list,
                    blank_id,
                    [context["mapping"] for context in contexts],
                    speaker_probs=aligned_speaker_cpu,
                    speaker_logprob_weight=weight,
                )
            else:
                final_paths, final_path_scores = preliminary_paths, preliminary_path_scores

            for stream, path, score in zip(streams, final_paths, final_path_scores):
                record_index = stream["record_index"]
                context = contexts[record_index]
                context["rows"].extend(
                    self._word_rows_from_path(
                        stream["words"],
                        stream["labels"],
                        stream["state_to_word"],
                        path,
                        ctc_cpu[record_index, : ctc_lengths_list[record_index]],
                        speaker_by_record[record_index],
                        context["mapping"],
                        context["ctc_step"],
                        float(offsets[record_index]),
                    )
                )
                context["final_scores"][stream["speaker_tag"]] = score

        return [
            self._result(
                rows=context["rows"],
                mapping=context["mapping"],
                ctc_step=context["ctc_step"],
                sortformer_step=context["sortformer_step"],
                time_offset=float(offsets[index]),
                ctc_frames=ctc_lengths_list[index],
                sortformer_frames=speaker_lengths_list[index],
                preliminary_scores=context["preliminary_scores"],
                final_scores=context["final_scores"],
                assignment_scores=context["assignment_scores"],
                diarization_timestamps=context["diarization_timestamps"],
                diarization_frame_seconds=(None if diarization_cpu is None else float(diarization_frame_seconds)),
                diarization_max_speaker_count=(None if diarization_cpu is None else diarization_cpu.shape[1]),
            )
            for index, context in enumerate(contexts)
        ]

    @staticmethod
    def _diarization_segments(
        labels: torch.Tensor,
        *,
        frame_seconds: float,
        time_offset: float,
        audio_duration: Optional[float],
    ) -> List[Dict[str, Any]]:
        """Convert boolean ``(speakers, frames)`` activity into DER-ready segments."""
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
            "alignment_mode": "parallel",
            "requested_alignment_mode": "parallel",
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

    def _align_stream_batch(
        self,
        streams: Sequence[Dict[str, Any]],
        ctc_log_probs: torch.Tensor,
        ctc_lengths: Sequence[int],
        blank_id: int,
        speaker_mappings: Sequence[Dict[int, Optional[int]]],
        speaker_probs: Optional[torch.Tensor],
        speaker_logprob_weight: float,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Run one padded DP/backtrace batch across every record and speaker stream."""
        max_states = max(len(stream["labels"]) for stream in streams)
        labels_batch = torch.full((len(streams), max_states), blank_id, dtype=torch.long)
        columns_batch = torch.full_like(labels_batch, -1)
        state_lengths = torch.tensor([len(stream["labels"]) for stream in streams], dtype=torch.long)
        record_indices = torch.tensor([stream["record_index"] for stream in streams], dtype=torch.long)
        frame_lengths = torch.tensor([ctc_lengths[index] for index in record_indices.tolist()], dtype=torch.long)
        for index, stream in enumerate(streams):
            labels = stream["labels"]
            mapping = speaker_mappings[stream["record_index"]]
            column = mapping.get(stream["speaker_tag"])
            labels_batch[index, : len(labels)] = torch.tensor(labels)
            if column is not None:
                token_states = [word_index is not None for word_index in stream["state_to_word"]]
                columns_batch[index, : len(token_states)] = torch.tensor(
                    [column if is_token else -1 for is_token in token_states]
                )

        return self._ctc_forced_align_batched(
            ctc_log_probs.index_select(0, record_indices),
            labels_batch,
            state_lengths,
            blank_id,
            columns_batch,
            None if speaker_probs is None else speaker_probs.index_select(0, record_indices),
            speaker_logprob_weight,
            frame_lengths=frame_lengths,
        )

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
            grouped.setdefault(word["speaker_tag"], []).append(word)
        return grouped

    @staticmethod
    def _public_word_timestamps(
        words: Sequence[Dict[str, Any]],
    ) -> Dict[Optional[int], List[Dict[str, Any]]]:
        """Return the minimal public CTC word-timestamp schema grouped by speaker."""
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
                tokenized[word["word_index"]] = {**word, "token_ids": token_ids}
        return [tokenized[word["word_index"]] for word in words]

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
            for token_id in word["token_ids"]:
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
        *,
        frame_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Run exact max-sum CTC DP for padded record/speaker stream batches.

        Only two score rows are live. Backtrace moves are encoded as ``stay=0``,
        ``advance=1``, or ``skip=2`` and packed four states per byte. This
        preserves the dense recurrence and tie-breaking exactly while reducing
        persistent backpointer storage from eight bytes to two bits per state.
        """
        num_streams, max_states = labels.shape
        if log_probs.ndim == 2:
            log_probs = log_probs.unsqueeze(0).expand(num_streams, -1, -1)
        elif log_probs.ndim != 3 or log_probs.shape[0] != num_streams:
            raise ValueError("log_probs must have shape (frames, classes) or (streams, frames, classes).")
        num_frames = log_probs.shape[1]
        if num_frames < 1 or max_states < 2:
            raise ValueError("CTC forced alignment requires at least one frame and two target states.")
        device = log_probs.device
        labels = labels.to(device=device)
        state_lengths = state_lengths.to(device=device)
        state_speaker_columns = state_speaker_columns.to(device=device)
        state_mask = torch.arange(max_states, device=device).unsqueeze(0) < state_lengths.unsqueeze(1)
        token_states = state_mask & (state_speaker_columns >= 0)
        columns = state_speaker_columns.clamp_min(0)
        if frame_lengths is None:
            frame_lengths = torch.full((num_streams,), num_frames, dtype=torch.long, device=device)
        else:
            frame_lengths = frame_lengths.to(device=device)
        if frame_lengths.shape != (num_streams,) or bool(((frame_lengths < 1) | (frame_lengths > num_frames)).any()):
            raise ValueError("frame_lengths must contain one valid length per CTC stream.")
        use_speaker_prior = bool(speaker_probs is not None and speaker_logprob_weight > 0 and token_states.any())
        if speaker_probs is not None:
            if speaker_probs.ndim == 2:
                speaker_probs = speaker_probs.unsqueeze(0).expand(num_streams, -1, -1)
            elif speaker_probs.ndim != 3 or speaker_probs.shape[0] != num_streams:
                raise ValueError("speaker_probs must have shape (frames, speakers) or (streams, frames, speakers).")
            speaker_probs = speaker_probs.to(device=device)

        def emission_block(start: int, end: int) -> torch.Tensor:
            block_frames = end - start
            gather_labels = labels.unsqueeze(1).expand(-1, block_frames, -1)
            emissions = torch.gather(log_probs[:, start:end], 2, gather_labels)
            if use_speaker_prior:
                gather_columns = columns.unsqueeze(1).expand(-1, block_frames, -1)
                activity = torch.gather(speaker_probs[:, start:end], 2, gather_columns)
                emissions = emissions + torch.where(
                    token_states.unsqueeze(1),
                    speaker_logprob_weight * torch.log(activity.clamp_min(self.epsilon)),
                    torch.zeros_like(activity),
                )
            return emissions.masked_fill(~state_mask.unsqueeze(1), -float("inf"))

        previous = torch.full((num_streams, max_states), -float("inf"), device=device)
        packed_states = (max_states + 3) // 4
        backpointers = torch.zeros(
            (max(0, num_frames - 1), num_streams, packed_states),
            dtype=torch.uint8,
            device=device,
        )

        for block_start in range(0, num_frames, self._EMISSION_BLOCK_FRAMES):
            block_end = min(block_start + self._EMISSION_BLOCK_FRAMES, num_frames)
            emissions = emission_block(block_start, block_end)
            local_start = 0
            if block_start == 0:
                previous[:, :2] = emissions[:, 0, :2]
                local_start = 1
            for local_frame in range(local_start, block_end - block_start):
                frame = block_start + local_frame
                best = previous.clone()
                moves = torch.zeros((num_streams, max_states), dtype=torch.uint8, device=device)
                take = previous[:, :-1] > best[:, 1:]
                best[:, 1:] = torch.where(take, previous[:, :-1], best[:, 1:])
                moves[:, 1:] = take.to(torch.uint8)
                if max_states > 2:
                    can_skip = state_mask[:, 2:] & (labels[:, 2:] != blank_id) & (labels[:, 2:] != labels[:, :-2])
                    take = can_skip & (previous[:, :-2] > best[:, 2:])
                    best[:, 2:] = torch.where(take, previous[:, :-2], best[:, 2:])
                    moves[:, 2:] = torch.where(take, 2, moves[:, 2:])
                updated = best + emissions[:, local_frame]
                updated.masked_fill_(~state_mask, -float("inf"))
                active_streams = frame < frame_lengths
                previous = torch.where(active_streams.unsqueeze(1), updated, previous)
                moves.masked_fill_(~active_streams.unsqueeze(1), 0)
                packed = backpointers[frame - 1]
                for offset in range(4):
                    values = moves[:, offset::4]
                    packed[:, : values.shape[1]] |= values << (2 * offset)

        last_blank = state_lengths - 1
        last_token = state_lengths - 2
        blank_scores = previous.gather(1, last_blank[:, None]).squeeze(1)
        token_scores = previous.gather(1, last_token[:, None]).squeeze(1)
        final_states = torch.where(token_scores > blank_scores, last_token, last_blank)
        final_scores = torch.maximum(token_scores, blank_scores)
        if not torch.isfinite(final_scores).all():
            failed = torch.nonzero(~torch.isfinite(final_scores)).flatten().tolist()
            raise ValueError(f"No valid CTC forced-alignment path for speaker stream(s) {failed}.")

        paths = torch.empty((num_streams, num_frames), dtype=torch.long, device=device)
        current = final_states
        stream_indices = torch.arange(num_streams, device=device)
        for frame in range(num_frames - 1, -1, -1):
            paths[:, frame] = current
            if frame:
                packed = backpointers[
                    frame - 1,
                    stream_indices,
                    torch.div(current, 4, rounding_mode="floor"),
                ]
                shifts = 2 * torch.remainder(current, 4)
                moves = torch.bitwise_right_shift(packed.to(torch.long), shifts) & 3
                current = torch.where(frame < frame_lengths, current - moves, current)
        paths_cpu = paths.cpu()
        return [path[: int(length)] for path, length in zip(paths_cpu, frame_lengths.cpu())], [
            float(score) for score in final_scores
        ]

    @staticmethod
    def estimate_dp_storage_bytes(num_frames: int, num_streams: int, max_states: int) -> Dict[str, int]:
        """Compare persistent dense and compact DP storage for a problem shape."""
        cells = int(num_frames) * int(num_streams) * int(max_states)
        return {
            "dense_emissions": cells * torch.tensor([], dtype=torch.float32).element_size(),
            "dense_backpointers": cells * torch.tensor([], dtype=torch.int64).element_size(),
            "compact_emission_block": min(
                int(num_frames),
                MultiSpeakerSOTWordTimestampAligner._EMISSION_BLOCK_FRAMES,
            )
            * int(num_streams)
            * int(max_states)
            * torch.tensor([], dtype=torch.float32).element_size(),
            "compact_backpointers": max(0, int(num_frames) - 1) * int(num_streams) * ((int(max_states) + 3) // 4),
        }

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
            end = min(
                time_offset + (frames[-1] + 1) * ctc_step,
                start + self.maximum_token_len,
            )
            end_frame = min(
                frames[-1],
                max(frames[0], math.ceil((end - time_offset) / ctc_step) - 1),
            )
            selected = torch.stack([ctc[frame, labels[state]] for frame, state in word_frames])
            column = speaker_mapping.get(word["speaker_tag"])
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
                    "start": start,
                    "end": end,
                    "start_frame": frames[0],
                    "end_frame": end_frame,
                    "ctc_confidence": float(torch.exp(selected.mean())),
                    "sortformer_column": column,
                    "speaker_confidence": speaker_confidence,
                    "speaker_activity_start": activity_start,
                    "speaker_activity_end": activity_end,
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
                    torch.log(speaker_probs[row["start_frame"] : row["end_frame"] + 1, column].clamp_min(self.epsilon))
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


@dataclass(frozen=True)
class CTCTimestampArtifact:
    """Loaded CTC decoder and matching tokenizer."""

    decoder: TransformerCTCDecoder
    tokenizer: SentencePieceTokenizer
    decoder_config: dict[str, Any]


def _plain_mapping(value: Any, label: str) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _decoder_init_config(value: Any) -> dict[str, Any]:
    config = _plain_mapping(value, "decoder_config")
    accepted = set(inspect.signature(TransformerCTCDecoder.__init__).parameters) - {"self"}
    init_config = {key: item for key, item in config.items() if key in accepted}
    missing = {"feat_in", "num_classes"} - set(init_config)
    if missing:
        raise ValueError(f"decoder_config is missing required field(s): {sorted(missing)}.")
    return init_config


def _tokenizer_payload(tokenizer: Any) -> dict[str, Any]:
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
    """Write the decoder, tokenizer, and construction config as one lightweight artifact."""
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = _decoder_init_config(decoder_config)
    state_dict = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    payload = {
        "format": CTC_TIMESTAMP_ARTIFACT_FORMAT,
        "decoder_config": config,
        "decoder_state_dict": state_dict,
        "tokenizer": _tokenizer_payload(tokenizer),
    }
    torch.save(payload, path)
    return path


def _load_sentencepiece(payload: Any) -> SentencePieceTokenizer:
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
    """Load one self-contained timestamp artifact without restoring an ASR model."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CTC timestamp artifact does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
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


def export_ctc_timestamp_artifact(source_nemo: Union[str, Path], destination: Union[str, Path]) -> Path:
    """Convert a training-time EncDecCTCModelBPE archive into the lightweight runtime artifact."""
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
    """Disable feature-length collectives in every inference-only decoder submodule."""
    for submodule in module.modules():
        if getattr(submodule, "sync_max_audio_length", False):
            submodule.sync_max_audio_length = False


def get_ctc_timestamp_aligner(
    owner: nn.Module,
    artifact_path: Optional[str],
    device: torch.device,
) -> MultiSpeakerSOTWordTimestampAligner:
    """Load and cache an inference-only aligner without registering it in ``owner``'s module tree."""
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("ctc_timestamp_model_path must be a non-empty lightweight artifact path.")
    resolved_path = os.path.realpath(os.path.expanduser(artifact_path))
    cached = owner.__dict__.get("_ctc_timestamp_extractor_cache")
    if cached is None or cached[0] != resolved_path:
        parameter = next(owner.parameters(), None)
        dtype = parameter.dtype if parameter is not None and device.type != "cpu" else torch.float32
        artifact = load_ctc_timestamp_artifact(resolved_path, map_location=device, dtype=dtype)
        aligner = MultiSpeakerSOTWordTimestampAligner(
            encoder=owner,
            ctc_decoder=artifact.decoder,
            tokenizer=artifact.tokenizer,
        )
        owner.__dict__["_ctc_timestamp_extractor_cache"] = (resolved_path, aligner)
        cached = (resolved_path, aligner)
    aligner = cached[1]
    parameter = next(owner.parameters(), None)
    dtype = parameter.dtype if parameter is not None and device.type != "cpu" else torch.float32
    aligner.ctc_decoder.to(device=device, dtype=dtype).eval()
    _disable_max_seq_length_sync(aligner.ctc_decoder)
    return aligner
