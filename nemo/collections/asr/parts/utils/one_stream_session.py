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


import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel, EncDecRNNTModel
from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer


def get_default_device(allow_mps: bool = True) -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and allow_mps:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_default_compute_dtype(device: torch.device) -> torch.dtype:
    compute_dtype: torch.dtype
    can_use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if can_use_bfloat16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32
    return compute_dtype


@dataclass
class StreamingConfig:
    """Configuration for streaming parameters"""

    # model params
    model_name: str
    device: Optional[torch.device] = None
    compute_dtype: Optional[torch.dtype] = None
    # context params
    left_context_secs: float = 10.0
    chunk_secs: float = 2.0
    right_context_secs: float = 2.0

    def __post_init__(self):
        if self.device is None:
            self.device = get_default_device()
        if self.compute_dtype is None:
            self.compute_dtype = get_default_compute_dtype(self.device)


class StreamingSession:
    """Manages a streaming ASR session"""

    def __init__(self):
        self.config: Optional[StreamingConfig] = None
        self.model: Optional[ASRModel] = None
        self.sample_rate: Optional[int] = None
        self.hyp = None
        self.state: Optional[Any] = None
        self.is_active: bool = False

        self.batched_audio_buffer: Optional[StreamingBatchedAudioBuffer] = None
        self.audio_frames: np.ndarray = np.zeros([0], dtype=np.float32)
        self.first_chunk_processed: bool = False

        # Streaming parameters
        self.context_encoder_frames: Optional[ContextSize] = None
        self.context_samples: Optional[ContextSize] = None
        self.encoder_frame2audio_samples: Optional[int] = None

        self.fixed_transcription = ""
        self.temporary_transcription = ""
        self.rtfx: Optional[float] = None

    @property
    def transcription(self) -> str:
        if self.temporary_transcription:
            return f"{self.fixed_transcription} [{self.temporary_transcription}]"
        return self.fixed_transcription

    @staticmethod
    def load_model(model_name: str, device: str | torch.device, compute_dtype: torch.dtype) -> ASRModel:
        """Load and configure the ASR model"""
        print(f"Loading model: {model_name}, device {device}, dtype {compute_dtype}")

        model = ASRModel.from_pretrained(model_name=model_name)
        model.freeze()
        model.eval()
        model = model.to(device)
        model = model.to(compute_dtype)

        # Configure decoding for streaming
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy_batch"
            decoding_cfg.greedy.loop_labels = True
            decoding_cfg.greedy.preserve_alignments = False
            decoding_cfg.fused_batch_size = -1
            decoding_cfg.beam.return_best_hypothesis = True

        # Apply decoding configuration
        if hasattr(model, 'cur_decoder'):
            model.change_decoding_strategy(decoding_cfg, decoder_type='rnnt')
        elif isinstance(model, EncDecRNNTModel):
            model.change_decoding_strategy(decoding_cfg)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # Configure preprocessor for streaming
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0

        print("✓ Model loaded successfully!")
        return model

    def setup_streaming_session(self, config: StreamingConfig):
        """Setup streaming parameters based on model configuration"""
        self.is_active = False

        if self.config is None or (
            self.config.model_name != config.model_name
            or self.config.device != config.device
            or self.config.compute_dtype != config.compute_dtype
        ):
            self.model = self.load_model(
                model_name=config.model_name, device=config.device, compute_dtype=config.compute_dtype
            )
        assert self.model is not None
        self.config = config
        model_cfg = self.model.cfg

        # Audio parameters
        self.sample_rate = model_cfg.preprocessor['sample_rate']
        feature_stride_sec = model_cfg.preprocessor['window_stride']
        features_per_sec = 1.0 / feature_stride_sec
        encoder_subsampling_factor = self.model.encoder.subsampling_factor

        # Frame calculations
        features_frame2audio_samples = self._make_divisible_by(
            int(self.sample_rate * feature_stride_sec), factor=encoder_subsampling_factor
        )
        self.encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

        # Context sizes in encoder frames
        self.context_encoder_frames = ContextSize(
            left=int(self.config.left_context_secs * features_per_sec / encoder_subsampling_factor),
            chunk=int(self.config.chunk_secs * features_per_sec / encoder_subsampling_factor),
            right=int(self.config.right_context_secs * features_per_sec / encoder_subsampling_factor),
        )

        # Context sizes in audio samples
        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left * encoder_subsampling_factor * features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk * encoder_subsampling_factor * features_frame2audio_samples,
            right=self.context_encoder_frames.right * encoder_subsampling_factor * features_frame2audio_samples,
        )

        print(f"Streaming parameters configured:")
        print(f"  Sample rate: {self.sample_rate}")
        print(
            f"  Context (seconds): left={self.config.left_context_secs}, "
            f"chunk={self.config.chunk_secs}, right={self.config.right_context_secs}"
        )
        print(f"  Context (samples): {self.context_samples}")
        print(
            f"  Theoretical latency: "
            f"{(self.context_samples.chunk + self.context_samples.right) / self.sample_rate:.2f} seconds"
        )

        self.reset_buffer()
        self.audio_frames = np.ndarray([0], dtype=np.float32)
        self.hyp = None
        self.state = None
        self.is_active = True
        self.fixed_transcription = ""
        self.temporary_transcription = ""
        self.rtfx = None

    @staticmethod
    def _make_divisible_by(num: int, factor: int) -> int:
        """Make num divisible by factor"""
        return (num // factor) * factor

    def flush(self):
        self.process_audio_chunk(audio_chunk=None, is_last=True)

    def reset_buffer(self):
        self.batched_audio_buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self.config.device,
        )
        self.first_chunk_processed = False

    def process_audio_chunk(self, audio_chunk: Optional[np.ndarray], is_last: bool):
        """Process a single audio chunk"""
        if audio_chunk is not None:
            self.audio_frames = np.concatenate((self.audio_frames, audio_chunk))
        first_chunk_samples = self.context_samples.chunk + self.context_samples.right
        need_samples = self.context_samples.chunk if self.first_chunk_processed else first_chunk_samples
        while (self.audio_frames.shape[0] >= need_samples) or (is_last and self.audio_frames.shape[0] > 0):
            start_time = time.perf_counter_ns()
            cur_chunk = self.audio_frames[:need_samples]
            self._process_next_chunk(cur_chunk, is_last=is_last and self.audio_frames.shape[0] <= need_samples)
            end_time = time.perf_counter_ns()
            rtfx = (cur_chunk.shape[0] * 1_000_000_000 / self.sample_rate) / (end_time - start_time)
            if self.rtfx is not None:
                self.rtfx = self.rtfx * 0.9 + rtfx * 0.1
            else:
                self.rtfx = rtfx
            self.audio_frames = self.audio_frames[need_samples:].copy()
            need_samples = self.context_samples.chunk
        if not self.first_chunk_processed and self.audio_frames.shape[0] > 0:
            # TODO: improve first chunk processing
            self._process_first_temporary_chunk(self.audio_frames)

    @torch.inference_mode()
    def _process_first_temporary_chunk(self, audio_chunk: np.ndarray):
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.config.device)
        chunk_length = torch.tensor([len(audio_chunk)], device=self.config.device)
        # Get encoder output
        encoder_output, encoder_output_len = self.model(
            input_signal=audio_tensor,
            input_signal_length=chunk_length,
        )
        encoder_output = encoder_output.transpose(1, 2)

        chunk_batched_hyps, _, _ = self.model.decoding.decoding.decoding_computer(
            x=encoder_output,
            out_len=encoder_output_len,
            prev_batched_state=None,
        )
        chunk_hyp = batched_hyps_to_hypotheses(chunk_batched_hyps, batch_size=1)[0]
        self.temporary_transcription = self.model.tokenizer.ids_to_text(chunk_hyp.y_sequence.tolist())

    @torch.inference_mode()
    def _process_next_chunk(self, audio_chunk: np.ndarray, is_last: bool):
        assert self.is_active
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.config.device)
        chunk_length = torch.tensor([len(audio_chunk)], device=self.config.device)

        # Add to buffer
        self.batched_audio_buffer.add_audio_batch_(
            audio_tensor,
            audio_lengths=chunk_length,
            is_last_chunk=is_last,
            is_last_chunk_batch=torch.tensor([is_last], device=self.config.device),
        )

        # Get encoder output
        encoder_output, encoder_output_len = self.model(
            input_signal=self.batched_audio_buffer.samples,
            input_signal_length=self.batched_audio_buffer.context_size_batch.total(),
        )
        encoder_output = encoder_output.transpose(1, 2)

        # Remove left context
        encoder_context = self.batched_audio_buffer.context_size.subsample(factor=self.encoder_frame2audio_samples)
        encoder_context_batch = self.batched_audio_buffer.context_size_batch.subsample(
            factor=self.encoder_frame2audio_samples
        )
        encoder_output = encoder_output[:, encoder_context.left :]

        if encoder_context.chunk > 0:
            # Decode chunk
            chunk_batched_hyps, _, self.state = self.model.decoding.decoding.decoding_computer(
                x=encoder_output,
                out_len=encoder_context_batch.chunk,
                prev_batched_state=self.state,
            )
            chunk_hyp = batched_hyps_to_hypotheses(chunk_batched_hyps, batch_size=1)[0]

            # Merge hypotheses
            if self.hyp is None:
                self.hyp = chunk_hyp
            else:
                self.hyp.merge_(chunk_hyp)

            self.hyp.text = self.model.tokenizer.ids_to_text(self.hyp.y_sequence.tolist())

            # Update fixed_transcription with the decoded text
            self.fixed_transcription = self.hyp.text
        else:
            print(f"Unexpected context: {self.batched_audio_buffer.context_size} -> {encoder_context}")

        # decode right chunk
        if encoder_context.right > 0:
            encoder_output = encoder_output[:, encoder_context.chunk :]
            chunk_batched_temp_hyps, _, _ = self.model.decoding.decoding.decoding_computer(
                x=encoder_output,
                out_len=encoder_context_batch.right,
                prev_batched_state=self.state,
            )
            tmp_hyp = batched_hyps_to_hypotheses(chunk_batched_temp_hyps, batch_size=1)[0]
            self.temporary_transcription = self.model.tokenizer.ids_to_text(tmp_hyp.y_sequence.tolist())
        else:
            self.temporary_transcription = ""

        self.first_chunk_processed = True
        if is_last:
            self.reset_buffer()
