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


from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig

from nemo.collections.asr.inference.model_wrappers.salm_asr_inference_wrapper import SALMASRInferenceWrapper
from nemo.collections.asr.inference.pipelines.base_pipeline import BasePipeline
from nemo.collections.asr.inference.streaming.buffering.incremental_audio_bufferer import (
    BatchedIncrementalAudioBufferer,
)
from nemo.collections.asr.inference.streaming.framing.multi_stream import ContinuousBatchedRequestStreamer
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.inference.streaming.state.salm_state import SALMStreamingState
from nemo.collections.asr.inference.utils.enums import ASROutputGranularity, MergingStrategy, RequestType
from nemo.collections.asr.inference.utils.lcs_merge import lcs_merge
from nemo.utils.decorators import experimental

if TYPE_CHECKING:
    from nemo.collections.asr.inference.itn.inverse_normalizer import AlignmentPreservingInverseNormalizer
    from nemo.collections.asr.inference.nmt.llm_translator import LLMTranslator


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    """
    Parse the hypothesis. Extract the tokens before the EOS tokens.
    Args:
        answer: (torch.Tensor) Answer tensor.
        eos_tokens: (list[int]) EOS tokens.
    Returns:
        (torch.Tensor) Parsed hypothesis.
    """
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


@experimental
class BufferedSALMPipeline(BasePipeline):
    """Buffered SALM pipeline."""

    def __init__(
        self,
        cfg: DictConfig,
        asr_model: SALMASRInferenceWrapper,
        itn_model: AlignmentPreservingInverseNormalizer | None = None,
        nmt_model: LLMTranslator | None = None,
    ):
        """
        Initialize the BufferedSALMPipeline.
        Args:
            cfg: (DictConfig) Configuration parameters.
            asr_model: (SALMASRInferenceWrapper) ASR model.
            itn_model: (AlignmentPreservingInverseNormalizer | None) Inverse Text Normalization model.
            nmt_model: (LLMTranslator | None) LLM based translation model.
        """
        self.asr_model = asr_model
        self.init_parameters(cfg)
        self.init_nmt_model(nmt_model)
        super().__init__()

    def init_parameters(self, cfg: DictConfig) -> None:
        """
        Initialize the parameters.
        Args:
            cfg: (DictConfig) Configuration parameters.
        """
        self.sample_rate = cfg.streaming.sample_rate
        self.asr_output_granularity = ASROutputGranularity.from_str(cfg.asr_output_granularity)
        if self.asr_output_granularity is ASROutputGranularity.WORD:
            raise ValueError("Word level output granularity is not supported for SALM AED pipeline")

        self.batch_size = cfg.streaming.batch_size
        self.max_new_tokens = cfg.streaming.max_new_tokens
        self.device = self.asr_model.device
        self.stop_history_eou_in_milliseconds = cfg.endpointing.stop_history_eou

        self.chunk_size_in_secs = cfg.streaming.chunk_size
        self.buffer_size_in_secs = cfg.streaming.buffer_size
        self.overlap_size_in_secs = cfg.streaming.overlap_size
        self.overlap_ratio = self.overlap_size_in_secs / self.buffer_size_in_secs
        self.extra_overlap_tokens = 2  # extra tokens for better overlap detection
        self.merging_strategy = MergingStrategy.from_str(cfg.streaming.merging_strategy)

        self.audio_bufferer = BatchedIncrementalAudioBufferer(
            self.sample_rate,
            self.buffer_size_in_secs,
            self.chunk_size_in_secs,
            self.overlap_size_in_secs,
        )

        self.request_type = RequestType.from_str(cfg.streaming.request_type)
        if self.request_type is RequestType.FEATURE_BUFFER:
            raise ValueError("Feature buffer request type is not supported for SALM pipeline")

        self.prompts = [[{"role": "user", "content": f"Transcribe the following: {self.asr_model.audio_locator_tag}"}]]
        self.tokens = self.asr_model.preprocess_prompts(self.prompts)

    def create_state(self, options: ASRRequestOptions) -> SALMStreamingState:
        """
        Create new empty state.
        Args:
            options: (ASRRequestOptions) Request options for particular stream.
        Returns:
            (SALMStreamingState) New empty state.
        """
        state = SALMStreamingState()
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
        """Return the separator for the text processor."""
        return self.asr_model.word_separator

    def lcs_merge(self, state: SALMStreamingState, data: list[int]) -> None:
        """
        Merge the buffer and data using the LCS algorithm.
        Args:
            state: (SALMStreamingState) The state of the streaming pipeline.
            data: (list[int]) The new tokens to merge with the buffer.
        """
        if len(state.tokens) == 0:
            state.tokens.extend(data)
            return

        # extra overlap tokens for better overlap detection
        delay = int(self.overlap_ratio * len(data)) + self.extra_overlap_tokens
        state.tokens = lcs_merge(
            buffer=state.tokens,
            data=data[:delay],
            search_size=delay,
            sep_id=self.asr_model.word_separator_ids,
            min_lcs_length=1,
            merging_strategy=self.merging_strategy,
        )
        state.tokens.extend(data[delay:])

    def transcribe_step_for_frames(self, frames: list[Frame]) -> None:
        """
        Perform the transcribe step for frames.
        Args:
            frames: (list[Frame]) List of frames to transcribe.
        """
        buffers, paddings = self.audio_bufferer.update(frames)
        paddings = torch.tensor(paddings, dtype=torch.int64, device=self.device)

        # Right paddings for the final frames
        # Only for last frames frame.size is greater than frame.valid_size
        right_paddings = torch.tensor(
            [int(frame.size - frame.valid_size) for frame in frames], dtype=torch.int64, device=self.device
        ).clamp(min=0)

        # stack buffers
        audios = torch.cat([buffer.unsqueeze_(0) for buffer in buffers]).to(self.device)
        audio_lens = torch.tensor([audios.size(1)] * audios.size(0), dtype=torch.int64, device=self.device)
        audio_lens = audio_lens - paddings - right_paddings

        answer_ids = self.asr_model.generate(
            prompts=self.tokens.expand(len(audios), -1),
            audios=audios,
            audio_lens=audio_lens,
            max_new_tokens=self.max_new_tokens,
        ).cpu()

        for i, frame in enumerate(frames):
            state = self.get_state(frame.stream_id)
            new_tokens = parse_hyp(answer_ids[i], self.asr_model.eos_token_ids).tolist()
            state.incomplete_segment_tokens.clear()
            if self.audio_bufferer.is_full(frame.stream_id) or frame.is_last:
                self.lcs_merge(state, new_tokens)
            else:
                state.incomplete_segment_tokens.extend(new_tokens)

            if frame.is_last:
                state.final_transcript = self.asr_model.tokenizer.ids_to_text(state.tokens)
                state.partial_transcript = ""
            else:
                all_tokens = state.tokens.copy()
                if len(state.incomplete_segment_tokens) > 0:
                    # extra overlap tokens for better overlap detection
                    delay = int(self.overlap_ratio * len(state.incomplete_segment_tokens)) + self.extra_overlap_tokens
                    all_tokens = lcs_merge(
                        buffer=all_tokens,
                        data=state.incomplete_segment_tokens[:delay],
                        search_size=delay,
                        sep_id=self.asr_model.word_separator_ids,
                        min_lcs_length=1,
                        merging_strategy=self.merging_strategy,
                    )
                    all_tokens.extend(state.incomplete_segment_tokens[delay:])

                if len(all_tokens) > 0:
                    state.partial_transcript = self.asr_model.tokenizer.ids_to_text(all_tokens)
                else:
                    state.partial_transcript = ""

    def transcribe_step_for_feature_buffers(self, fbuffers: list[FeatureBuffer]) -> None:
        """
        Transcribe a step for feature buffers.
        Args:
            fbuffers: (list[FeatureBuffer]) List of feature buffers to transcribe.
        """
        raise NotImplementedError("Feature buffer request type is not supported for SALM pipeline")

    def get_request_generator(self) -> ContinuousBatchedRequestStreamer:
        """
        Initialize the request generator.
        Returns:
            (ContinuousBatchedRequestStreamer) Request generator.
        """
        request_generator = ContinuousBatchedRequestStreamer(
            n_frames_per_stream=1,
            frame_size_in_secs=self.chunk_size_in_secs,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            request_type=self.request_type,
            pad_last_frame=True,
        )
        return request_generator
