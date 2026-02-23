# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
True-streaming pipeline for StreamingSALM.

Unlike :class:`BufferedSALMPipeline` (Canary-Qwen), this pipeline processes
each audio chunk exactly once via KV cache — tokens are emitted incrementally
and never revised.  No buffering, no overlap, no LCS merging.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import Tensor

from nemo.collections.asr.inference.model_wrappers.streaming_salm_inference_wrapper import (
    StreamingSALMInferenceWrapper,
)
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.streaming_salm_state import StreamingSALMStreamingState
from nemo.collections.asr.inference.utils.enums import ASROutputGranularity, RequestType
from nemo.collections.speechlm2.modules.mimi_encoder import MimiEncoder
from nemo.utils.decorators import experimental


@experimental
class StreamingSALMPipeline(BasePipeline):
    """True-streaming pipeline for StreamingSALM (no overlap, no merging)."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: StreamingSALMInferenceWrapper,
    ):
        self.asr_model = asr_model
        self.init_parameters(cfg)
        super().__init__()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_parameters(self, cfg: DictConfig) -> None:
        self.sample_rate = cfg.streaming.sample_rate
        assert self.sample_rate == MimiEncoder.SAMPLE_RATE, (
            f"StreamingSALM requires sample_rate={MimiEncoder.SAMPLE_RATE}, got {self.sample_rate}"
        )
        self.batch_size = cfg.streaming.batch_size
        self.chunk_size_in_secs = cfg.streaming.chunk_size
        self.latency = cfg.streaming.get("latency", self.asr_model.latency)
        self.context = cfg.streaming.get("context", self.asr_model.context)
        self.device = self.asr_model.device
        self.asr_output_granularity = ASROutputGranularity.from_str(
            cfg.get("asr_output_granularity", "segment")
        )
        self.stop_history_eou_in_milliseconds = cfg.get("endpointing", {}).get("stop_history_eou", 0)
        # Mimi frame alignment: samples per Mimi frame
        self.mimi_frame_samples = int(MimiEncoder.SAMPLE_RATE * MimiEncoder.FRAME_SHIFT)  # 1920

    # ------------------------------------------------------------------
    # BasePipeline abstract method implementations
    # ------------------------------------------------------------------

    def create_state(self, options: ASRRequestOptions) -> StreamingSALMStreamingState:
        state = StreamingSALMStreamingState()
        state.set_global_offset(0)
        new_options = options.augment_with_defaults(
            default_enable_itn=False,
            default_enable_pnc=False,
            default_enable_nmt=False,
            default_source_language=None,
            default_target_language=None,
            default_stop_history_eou=self.stop_history_eou_in_milliseconds,
            default_asr_output_granularity=self.asr_output_granularity,
            default_language_code=None,
        )
        state.set_options(new_options)
        return state

    def get_sep(self) -> str:
        return " "

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """Process each frame independently (per-stream, no cross-stream batching)."""
        for frame in frames:
            state = self.get_state(frame.stream_id)
            self._process_frame(frame, state)

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        raise NotImplementedError("Feature buffer not supported for StreamingSALM")

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        return ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=RequestType.FRAME,
            pad_last_frame=True,
        )

    # ------------------------------------------------------------------
    # Core per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: Frame, state: StreamingSALMStreamingState) -> None:
        # 1. Accumulate audio samples (prepend residual from previous step)
        audio = self._accumulate_audio(frame, state)

        # 2. Align to Mimi frame boundary, store leftover in state.audio_residual
        aligned_audio, residual = self._align_to_mimi_frames(audio)

        if aligned_audio is not None:
            # 3. Encode with Mimi
            codes, code_lens = self.asr_model.encode_audio(
                aligned_audio.unsqueeze(0).to(self.device),
                torch.tensor([aligned_audio.shape[0]], device=self.device),
            )

            # 4. generate_streaming → (emitted_tokens, new_model_state)
            emitted, new_model_state = self.asr_model.generate_streaming(
                codes, state.model_state, self.latency, self.context
            )
            state.model_state = new_model_state

            # 5. Append emitted tokens (first batch element)
            new_tokens = emitted[0]
            state.tokens.extend(new_tokens)

        state.audio_residual = residual

        # 6. Update transcripts
        if frame.is_last:
            # Encode any leftover residual audio (pad to a full Mimi frame)
            if state.audio_residual is not None and state.audio_residual_len > 0:
                padded = torch.nn.functional.pad(
                    state.audio_residual,
                    (0, self.mimi_frame_samples - state.audio_residual.shape[0]),
                )
                codes, code_lens = self.asr_model.encode_audio(
                    padded.unsqueeze(0).to(self.device),
                    torch.tensor([state.audio_residual.shape[0]], device=self.device),
                )
                emitted, new_model_state = self.asr_model.generate_streaming(
                    codes, state.model_state, self.latency, self.context
                )
                state.model_state = new_model_state
                state.tokens.extend(emitted[0])
                state.audio_residual = None

            # Flush: call generate_streaming(None, state) to emit latency-buffered tokens
            if state.model_state is not None:
                emitted, new_model_state = self.asr_model.generate_streaming(
                    None, state.model_state, self.latency, self.context
                )
                state.model_state = new_model_state
                state.tokens.extend(emitted[0])
            state.final_transcript = self.asr_model.ids_to_text(state.tokens)
            state.partial_transcript = ""
        else:
            if len(state.tokens) > 0:
                state.partial_transcript = self.asr_model.ids_to_text(state.tokens)
            else:
                state.partial_transcript = ""

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _accumulate_audio(self, frame: Frame, state: StreamingSALMStreamingState) -> Tensor:
        """Prepend residual samples from previous step to current frame's audio."""
        audio = frame.samples
        if state.audio_residual is not None and state.audio_residual_len > 0:
            audio = torch.cat([state.audio_residual[: state.audio_residual_len], audio])
        return audio

    def _align_to_mimi_frames(self, audio: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """Split audio into Mimi-frame-aligned portion and residual."""
        n_samples = audio.shape[0]
        n_complete_frames = n_samples // self.mimi_frame_samples
        if n_complete_frames == 0:
            return None, audio  # Not enough for one frame
        aligned_len = n_complete_frames * self.mimi_frame_samples
        aligned = audio[:aligned_len]
        residual = audio[aligned_len:] if aligned_len < n_samples else None
        return aligned, residual
