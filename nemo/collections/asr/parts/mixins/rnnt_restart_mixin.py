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

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor

from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import LabelLoopingStateItem
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class RNNTRestartMixin:
    """
    Restart of RNNT greedy decoding after the tokens in ``decoding.restart_after_tokens`` (e.g. ``<EOU>``).

    Used by ``transcribe()`` and ``conformer_stream_step()`` of RNNT models with a cache-aware streaming encoder and
    ``greedy_batch`` (label-looping) decoding. End-of-utterance models are trained on single utterances, so after
    such a token both the decoder state and the encoder left context are reset.
    """

    def decode_chunk_with_restart(
        self,
        encoded: Tensor,
        encoded_len: Tensor,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        hypotheses: Optional[List[Optional[Hypothesis]]],
        caches: tuple,
        restart_token_ids: List[int],
        keep_all_outputs: Union[bool, List[bool]] = True,
        drop_extra_pre_encoded: int = 0,
    ) -> tuple:
        """
        Greedy-decode one cache-aware streaming chunk, restarting from a fresh state after each token from
        ``decoding.restart_after_tokens`` (e.g. ``<EOU>``). End-of-utterance models are trained on single utterances
        and stop emitting tokens after ``<EOU>`` otherwise.

        For a stream that emits a restart token at chunk frame ``t``, the frames up to ``t`` are re-decoded from the
        previous decoder state, then the decoder state is reset and the rest of the chunk is re-encoded with a fresh
        encoder cache (repeated as needed). A restart token on the last frame resets the state for the next chunk.
        The ``length`` of the returned hypotheses is the total number of encoder frames decoded for the stream.

        Args:
            encoded: encoder output of the chunk [B, D, T].
            encoded_len: number of valid encoder frames per stream [B].
            processed_signal: chunk features [B, F, T_feat] (including the pre-encode cache).
            processed_signal_length: number of valid features per stream [B].
            hypotheses: raw hypotheses carried from the previous chunk (updated in place), or None.
            caches: (cache_last_channel, cache_last_time, cache_last_channel_len) after encoding the chunk.
            restart_token_ids: ids of the restart tokens.
            keep_all_outputs: ``keep_all_outputs`` of the encoder step, for the whole batch or per stream.
            drop_extra_pre_encoded: number of encoder frames dropped from the beginning of the chunk.

        Returns:
            Tuple of (raw hypotheses, updated caches).
        """
        decoding_computer = getattr(self.decoding.decoding, 'decoding_computer', None)
        if decoding_computer is None:
            raise NotImplementedError("`restart_after_tokens` requires `greedy_batch` decoding.")
        batch_size = encoded.shape[0]
        if isinstance(keep_all_outputs, bool):
            keep_all_outputs = [keep_all_outputs] * batch_size
        restart_token_ids = set(restart_token_ids)
        hypotheses = list(hypotheses) if hypotheses is not None else [None] * batch_size
        start_frames = _decoded_lengths(hypotheses)
        continuations = self._decode_continuation(encoded, encoded_len, hypotheses)
        num_frames_list, signal_len_list = encoded_len.tolist(), processed_signal_length.tolist()
        caches = list(caches)
        for b in range(batch_size):
            enc, num_frames, cont = encoded[b : b + 1], num_frames_list[b], continuations[b]
            signal, signal_len = processed_signal[b : b + 1], signal_len_list[b]
            start_frame, drop = start_frames[b], drop_extra_pre_encoded
            while True:
                prev = hypotheses[b]
                restart_frame = _find_restart_frame(cont, restart_token_ids, start_frame + num_frames)
                if restart_frame is not None:  # re-decode only up to the restart frame
                    restart_frame -= start_frame
                    cont = self._decode_continuation(enc[:, :, : restart_frame + 1], [restart_frame + 1], [prev])[0]
                if prev is None:
                    hypotheses[b] = cont
                else:
                    _append_hypothesis_segment(prev, cont)
                if restart_frame is None:
                    if any(int(token) in restart_token_ids for token in cont.y_sequence):  # on the last frame
                        _reset_hypothesis_decoder_state(hypotheses[b], decoding_computer)
                        caches = _replace_cache_row(caches, b)
                    break
                start_frame += restart_frame + 1
                _reset_hypothesis_decoder_state(hypotheses[b], decoding_computer)
                # causal subsampling: (pre-drop) output frame k ends at feature k * subsampling_factor
                tail_start = (restart_frame + 1 + drop) * self.encoder.subsampling_factor
                signal, signal_len = signal[:, :, tail_start:signal_len], signal_len - tail_start
                drop = 0  # the rest of the chunk is encoded as the beginning of a new stream
                enc, enc_len, new_cache = self._encode_fresh(
                    signal, torch.tensor([signal_len], device=enc.device), keep_all_outputs[b]
                )
                caches = _replace_cache_row(caches, b, new_cache)
                if not keep_all_outputs[b]:  # keep only the remaining frames of the chunk, as in the original stream
                    enc_len = enc_len.clamp(max=num_frames - restart_frame - 1)
                num_frames = int(enc_len[0])
                cont = self._decode_continuation(enc, enc_len, [hypotheses[b]])[0]
        return hypotheses, tuple(caches)

    def _decode_continuation(
        self, encoded: Tensor, encoded_len, hypotheses: List[Optional[Hypothesis]]
    ) -> List[Hypothesis]:
        """Raw greedy decoding of new frames, continuing from the decoder states of ``hypotheses`` (not modified)."""
        encoded_len = torch.as_tensor(encoded_len, device=encoded.device)
        states = [
            (
                None
                if hyp is None
                else Hypothesis(
                    score=0.0,
                    y_sequence=torch.empty(0, dtype=torch.long),
                    timestamp=torch.empty(0, dtype=torch.long),
                    dec_state=hyp.dec_state,
                    last_token=hyp.last_token,
                    biasing_cfg=hyp.biasing_cfg,
                )
            )
            for hyp in hypotheses
        ]
        return self.decoding.decoding(encoder_output=encoded, encoded_lengths=encoded_len, partial_hypotheses=states)[
            0
        ]

    def _encode_fresh(self, signal: Tensor, signal_length: Tensor, keep_all_outputs: bool):
        """Cache-aware encoding of one stream's features starting from the initial encoder cache."""
        cache = self.encoder.get_initial_cache_state(batch_size=1, dtype=signal.dtype, device=signal.device)
        encoded, encoded_len, *new_cache = self.encoder.cache_aware_stream_step(
            processed_signal=signal,
            processed_signal_length=signal_length,
            cache_last_channel=cache[0],
            cache_last_time=cache[1],
            cache_last_channel_len=cache[2],
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=0,
        )
        return self._apply_prompt_to_encoded(encoded), encoded_len, new_cache

    def cache_aware_transcribe_with_restart(
        self, processed_signal: Tensor, processed_signal_length: Tensor, restart_token_ids: List[int]
    ) -> List[Hypothesis]:
        """
        Offline decoding with restarts after ``decoding.restart_after_tokens``, run as batched cache-aware streaming
        over the features, so that the cost stays linear in the audio length. All utterances of the batch advance
        through each chunk together; an utterance leaves the batch after its last chunk.

        Returns:
            Raw (not post-processed) hypotheses.
        """
        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError("`restart_after_tokens` requires a cache-aware streaming encoder.")
        if self.encoder.streaming_cfg is None:
            self.encoder.setup_streaming_params()
        cfg = self.encoder.streaming_cfg

        def size(value, step):
            return (value[0] if step == 0 else value[1]) if isinstance(value, list) else value

        lengths = processed_signal_length.tolist()
        if min(lengths) <= 0:
            raise ValueError("`restart_after_tokens` requires non-empty audio.")
        hypotheses = [None] * len(lengths)
        active = list(range(len(lengths)))  # utterances that have not reached their last chunk yet
        caches = self.encoder.get_initial_cache_state(
            batch_size=len(active), dtype=processed_signal.dtype, device=processed_signal.device
        )
        position, step = 0, 0
        while active:
            chunk_start = position - size(cfg.pre_encode_cache_size, step)
            chunk_end = position + size(cfg.chunk_size, step)
            drop_extra_pre_encoded = 0 if step == 0 else cfg.drop_extra_pre_encoded
            chunk = processed_signal[active, :, max(0, chunk_start) : chunk_end]
            if chunk_start < 0:  # zero pre-encode cache at the beginning, as in streaming
                chunk = torch.nn.functional.pad(chunk, (-chunk_start, 0))
            chunk_len = [min(chunk_end, lengths[b]) - chunk_start for b in active]
            is_last = [chunk_end >= lengths[b] for b in active]
            encoded, encoded_len, *caches = self.encoder.cache_aware_stream_step(
                processed_signal=chunk,
                processed_signal_length=torch.tensor(chunk_len, device=chunk.device),
                cache_last_channel=caches[0],
                cache_last_time=caches[1],
                cache_last_channel_len=caches[2],
                keep_all_outputs=any(is_last),
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            if any(is_last) and not all(is_last) and cfg.valid_out_len > 0:
                # keep_all_outputs=False for utterances that continue after this chunk
                continues = torch.tensor([not last for last in is_last], device=encoded_len.device)
                encoded_len = torch.where(continues, encoded_len.clamp(max=cfg.valid_out_len), encoded_len)
            hyps, caches = self.decode_chunk_with_restart(
                encoded=self._apply_prompt_to_encoded(encoded),
                encoded_len=encoded_len,
                processed_signal=chunk,
                processed_signal_length=torch.tensor(chunk_len, device=chunk.device),
                hypotheses=[hypotheses[b] for b in active],
                caches=caches,
                restart_token_ids=restart_token_ids,
                keep_all_outputs=is_last,
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
            for b, hyp in zip(active, hyps):
                hypotheses[b] = hyp
            if any(is_last):  # the remaining chunks of these utterances would only repeat decoded frames
                keep = [i for i, last in enumerate(is_last) if not last]
                active = [active[i] for i in keep]
                keep = torch.tensor(keep, dtype=torch.long, device=caches[0].device)
                caches = [caches[0][:, keep], caches[1][:, keep], caches[2][keep]]
            position += size(cfg.shift_size, step)
            step += 1
        return hypotheses


def _replace_cache_row(caches: list, idx: int, row: Optional[list] = None) -> list:
    """Replace the encoder cache of stream ``idx`` with ``row`` (1-stream caches), or with the initial (zero) cache."""
    if row is None:
        row = [torch.zeros_like(caches[0][:, :1]), torch.zeros_like(caches[1][:, :1]), torch.zeros_like(caches[2][:1])]
    return [
        torch.cat((caches[0][:, :idx], row[0].to(caches[0].dtype), caches[0][:, idx + 1 :]), dim=1),
        torch.cat((caches[1][:, :idx], row[1].to(caches[1].dtype), caches[1][:, idx + 1 :]), dim=1),
        torch.cat((caches[2][:idx], row[2].to(caches[2].dtype), caches[2][idx + 1 :]), dim=0),
    ]


def _decoded_lengths(hypotheses: List[Optional[Hypothesis]]) -> List[int]:
    """Number of frames already decoded for each hypothesis (0 for a new stream), with a single device sync."""
    lengths = [
        hyp.dec_state.decoded_length if hyp is not None and hyp.dec_state is not None else None for hyp in hypotheses
    ]
    known = [length for length in lengths if length is not None]
    known = iter(torch.stack([torch.as_tensor(length) for length in known]).tolist() if known else [])
    return [0 if length is None else int(next(known)) for length in lengths]


def _reset_hypothesis_decoder_state(hypothesis: Hypothesis, decoding_computer) -> None:
    """Reset the decoder state of a partial hypothesis to start-of-sequence, keeping its tokens and time offset."""
    state = hypothesis.dec_state
    if isinstance(state, LabelLoopingStateItem):
        batched_state = decoding_computer.merge_to_batched_state([state])
        decoded_lengths = batched_state.decoded_lengths.clone()  # offsets the timestamps of the next step
        batched_state = decoding_computer.reset_state_by_mask(
            batched_state, torch.ones_like(decoded_lengths, dtype=torch.bool)
        )
        batched_state.decoded_lengths = decoded_lengths
        hypothesis.dec_state = decoding_computer.split_batched_state(batched_state)[0]
        hypothesis.dec_state.time_jump = state.time_jump
    else:
        hypothesis.dec_state = None
    hypothesis.last_token = None


def _find_restart_frame(hypothesis: Hypothesis, token_ids: Sequence[int], num_frames: int) -> Optional[int]:
    """Frame of the first token from ``token_ids`` emitted before the last frame, or None."""
    for token, frame in zip(_to_list(hypothesis.y_sequence), _to_list(hypothesis.timestamp)):
        if token in token_ids and frame < num_frames - 1:
            return int(frame)
    return None


def _append_hypothesis_segment(hypothesis: Hypothesis, segment: Hypothesis) -> None:
    """Append (in place) a raw hypothesis segment with absolute timestamps; ``length`` accumulates decoded frames."""
    token_duration = hypothesis.token_duration
    if isinstance(token_duration, torch.Tensor) and isinstance(segment.token_duration, torch.Tensor):
        token_duration = torch.cat((token_duration, segment.token_duration), dim=0)
    elif isinstance(token_duration, list) and isinstance(segment.token_duration, list):
        token_duration = token_duration + segment.token_duration
    hypothesis.merge_(segment)  # merge_ does not handle token_duration
    hypothesis.token_duration = token_duration


def _to_list(values) -> list:
    return values.tolist() if isinstance(values, torch.Tensor) else list(values)
