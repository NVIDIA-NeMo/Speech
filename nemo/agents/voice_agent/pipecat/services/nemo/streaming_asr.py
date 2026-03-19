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
# NOTE: This file will be deprecated in the future, as the new inference pipeline will replace it.

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.agents.voice_agent.pipecat.services.nemo.utils import CacheFeatureBufferer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


@dataclass
class ASRResult:
    text: str
    is_final: bool
    eou_prob: Optional[float] = None
    eob_prob: Optional[float] = None
    eou_latency: Optional[float] = None
    eob_latency: Optional[float] = None
    processing_time: Optional[float] = None


class NemoStreamingASRService:
    def __init__(
        self,
        model: str = "nvidia/parakeet_realtime_eou_120m-v1",
        att_context_size: List[int] = [70, 1],
        device: str = "cuda",
        eou_string: str = "<EOU>",
        eob_string: str = "<EOB>",
        decoder_type: str = None,
        chunk_size: int = -1,
        shift_size: int = -1,
        left_chunks: int = 2,
        sample_rate: int = 16000,
        frame_len_in_secs: float = 0.08,
        use_amp: bool = False,
        chunk_size_in_secs: float = 0.08,
    ):
        self.model = model
        self.eou_string = eou_string
        self.eob_string = eob_string
        self.device = device
        self.att_context_size = att_context_size
        self.decoder_type = decoder_type
        self.chunk_size = chunk_size
        self.shift_size = shift_size
        self.left_chunks = left_chunks
        self.asr_model = self._load_model(model)
        self.tokenizer: SentencePieceTokenizer = self.asr_model.tokenizer
        self.use_amp = use_amp
        self.pad_and_drop_preencoded = False
        self.blank_id = self.get_blank_id()
        self.chunk_size_in_secs = chunk_size_in_secs

        assert len(self.att_context_size) == 2, "Att context size must be a list of two integers"
        assert (
            self.att_context_size[0] >= 0
        ), f"Left att context size must be greater than 0: {self.att_context_size[0]}"
        assert (
            self.att_context_size[1] >= 0
        ), f"Right att context size must be greater than 0: {self.att_context_size[1]}"

        window_stride_in_secs = self.asr_model.cfg.preprocessor.window_stride
        model_stride = self.asr_model.cfg.encoder.subsampling_factor
        self.model_chunk_size = self.asr_model.encoder.streaming_cfg.chunk_size
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * window_stride_in_secs

        self.tokens_per_frame = math.ceil(np.trunc(self.chunk_size_in_secs / window_stride_in_secs) / model_stride)
        # overwrite the encoder streaming params with proper shift size for cache aware streaming
        self.asr_model.encoder.setup_streaming_params(
            chunk_size=self.model_chunk_size // model_stride, shift_size=self.tokens_per_frame
        )

        model_chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs

        self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs

        self._audio_buffer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.chunk_size_in_secs,
            preprocessor_cfg=self.asr_model.cfg.preprocessor,
            device=self.device,
        )
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._last_transcript_timestamp = time.time()
        print(f"NemoStreamingASRService initialized with model `{model}` on device `{self.device}`")

    def _reset_cache(self):
        (
            self._cache_last_channel,  # [17, B, 70, 512]
            self._cache_last_time,  # [17, B, 512, 8]
            self._cache_last_channel_len,  # B
        ) = self.asr_model.encoder.get_initial_cache_state(
            1
        )  # batch size is 1

    def _get_blank_hypothesis(self) -> List[Hypothesis]:
        blank_hypothesis = Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
        return [blank_hypothesis]

    @property
    def drop_extra_pre_encoded(self):
        return self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

    def get_blank_id(self):
        return len(self.tokenizer.vocab)

    def get_text_from_tokens(self, tokens: List[int]) -> str:
        sep = "\u2581"  # '▁'
        tokens = [int(t) for t in tokens if t != self.blank_id]
        if tokens:
            pieces = self.tokenizer.ids_to_tokens(tokens)
            text = "".join([p.replace(sep, ' ') if p.startswith(sep) else p for p in pieces])
        else:
            text = ""
        return text

    def _load_model(self, model: str):
        if model.endswith(".nemo"):
            asr_model = nemo_asr.models.ASRModel.restore_from(model, map_location=torch.device(self.device))
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model, map_location=torch.device(self.device))

        if self.decoder_type is not None and hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        elif isinstance(asr_model, nemo_asr.models.EncDecCTCModel):
            self.decoder_type = "ctc"
        elif isinstance(asr_model, nemo_asr.models.EncDecRNNTModel):
            self.decoder_type = "rnnt"
        else:
            raise ValueError("Decoder type not supported for this model.")

        if self.att_context_size is not None:
            if hasattr(asr_model.encoder, "set_default_att_context_size"):
                asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
            else:
                raise ValueError("Model does not support multiple lookaheads.")
        else:
            self.att_context_size = asr_model.cfg.encoder.att_context_size

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(asr_model, 'joint'):  # if an RNNT model
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            asr_model.change_decoding_strategy(decoding_cfg)

        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)

        # chunk_size is set automatically for models trained for streaming.
        # For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
        if self.chunk_size > 0:
            if self.shift_size < 0:
                shift_size = self.chunk_size
            else:
                shift_size = self.shift_size
            asr_model.encoder.setup_streaming_params(
                chunk_size=self.chunk_size, left_chunks=self.left_chunks, shift_size=shift_size
            )

        asr_model.eval()
        return asr_model

    def _get_best_hypothesis(self, encoded, encoded_len, partial_hypotheses=None):
        if self.decoder_type == "ctc":
            best_hyp = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
            )
        elif self.decoder_type == "rnnt":
            best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len, return_hypotheses=True, partial_hypotheses=partial_hypotheses
            )
        else:
            raise ValueError("Decoder type not supported for this model.")
        return best_hyp

    def _get_tokens_and_probs_from_alignments(self, alignments):
        tokens = []
        probs = []
        if self.decoder_type == "ctc":
            all_logits = alignments[0]
            all_tokens = alignments[1]
            for i in range(len(all_tokens)):
                token_id = int(all_tokens[i])
                if token_id != self.blank_id:
                    tokens.append(token_id)
                    logits = all_logits[i]  # shape (vocab_size,)
                    probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                    probs.append(probs_i)
        elif self.decoder_type == "rnnt":
            for t in range(len(alignments)):
                for u in range(len(alignments[t])):
                    logits, token_id = alignments[t][u]  # (logits, token_id)
                    token_id = int(token_id)
                    if token_id != self.blank_id:
                        tokens.append(token_id)
                        probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                        probs.append(probs_i)
        else:
            raise ValueError("Decoder type not supported for this model.")

        return tokens, probs

    def transcribe(self, audio: bytes, stream_id: str = "default") -> ASRResult:
        start_time = time.time()

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        self._audio_buffer.update(audio_array)

        features = self._audio_buffer.get_feature_buffer()
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)
        features = features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.asr_model.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
            )

        best_hyp = self._get_best_hypothesis(encoded, encoded_len, partial_hypotheses=self._previous_hypotheses)

        self._previous_hypotheses = best_hyp
        self._cache_last_channel = cache_last_channel
        self._cache_last_time = cache_last_time
        self._cache_last_channel_len = cache_last_channel_len

        tokens, probs = self._get_tokens_and_probs_from_alignments(best_hyp[0].alignments)

        text = self.get_text_from_tokens(tokens)

        is_final = False
        eou_latency = None
        eob_latency = None
        eou_prob = None
        eob_prob = None
        current_timestamp = time.time()
        if self.eou_string in text or self.eob_string in text:
            is_final = True
            if self.eou_string in text:
                eou_latency = (
                    current_timestamp - self._last_transcript_timestamp if text.strip() == self.eou_string else 0.0
                )
                eou_prob = self.get_eou_probability(tokens, probs, self.eou_string)
            if self.eob_string in text:
                eob_latency = (
                    current_timestamp - self._last_transcript_timestamp if text.strip() == self.eob_string else 0.0
                )
                eob_prob = self.get_eou_probability(tokens, probs, self.eob_string)
            self.reset_state(stream_id=stream_id)
        if text.strip():
            self._last_transcript_timestamp = current_timestamp

        processing_time = time.time() - start_time
        return ASRResult(
            text=text,
            is_final=is_final,
            eou_latency=eou_latency,
            eob_latency=eob_latency,
            eou_prob=eou_prob,
            eob_prob=eob_prob,
            processing_time=processing_time,
        )

    def reset_state(self, stream_id: str = "default"):
        self._audio_buffer.reset()
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._last_transcript_timestamp = time.time()

    def get_eou_probability(self, tokens: List[int], probs: List[float], eou_string: str = "<EOU>") -> float:
        text_tokens = self.tokenizer.ids_to_tokens(tokens)
        eou_index = text_tokens.index(eou_string)
        return probs[eou_index]
