#!/usr/bin/env python3
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
Adapter to use NeMo's native streaming pipelines with simulstream evaluation.

This adapter properly interfaces with NeMo's internal streaming API (transcribe_step)
rather than duplicating chunking/buffering logic. NeMo handles all buffering internally.

Key Insight:
    NeMo's pipelines already have complete streaming infrastructure:
    - Frame/FeatureBuffer creation
    - Buffering logic (BufferedPipeline / CacheAwarePipeline)
    - State management (StreamingState)
    - Translation integration (LLMTranslator)

    We just need to:
    1. Create Frame objects from audio chunks
    2. Call pipeline.transcribe_step()
    3. Convert TranscribeStepOutput → IncrementalOutput
"""

import atexit
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

from nemo.collections.asr.inference.nmt.prompts import EuroLLMTranslatorPromptTemplate
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_device, get_inference_dtype, setup_model
from nemo.utils import logging
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

try:
    from simulstream.server.speech_processors import SAMPLE_RATE, SpeechProcessor
    from simulstream.server.speech_processors.incremental_output import IncrementalOutput

    SIMULSTREAM_AVAILABLE = True
except ImportError:
    SIMULSTREAM_AVAILABLE = False
    SpeechProcessor = object
    SAMPLE_RATE = 16000

os.environ["HF_HOME"] = "/home/vbataev/hf_models"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


LANGUAGE_CODES = {
    "en": "English",
    "ru": "Russian",
    "da": "Danish",
    "it": "Italian",
    "de": "German",
    "zh": "Chinese",
}


def make_divisible_by(num, factor: int) -> int:
    """Make num divisible by factor"""
    return (num // factor) * factor

def get_local_model_path(repo_id):
    try:
        return snapshot_download(
            repo_id=repo_id,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return None


def get_llm_model(model_name: str = "Qwen/Qwen3-4B-Instruct-2507", model_params: dict[str, Any] | None = None):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if model_params is None:
        model_params = {
            "dtype": "auto",
            "seed": 42,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 8192,
        }
    # local_model_path = get_local_model_path(model_name)
    local_model_path = None
    if local_model_path:
        logging.info(f"Found model in local path {local_model_path}, will not download")
    else:
        logging.info(f"Will download model from HF: {model_name}")
    llm = LLM(local_model_path or model_name, **model_params)
    return llm


def get_asr_model(asr_cfg: DictConfig):
    # setup device
    map_location = get_inference_device(cuda=asr_cfg.device_id, allow_mps=True)
    compute_dtype = get_inference_dtype(asr_cfg.compute_dtype, device=map_location)

    logging.info(f"Inference will be done on device : {map_location} with compute_dtype: {compute_dtype}")

    if asr_cfg.model_name.lower().endswith(".nemo"):
        asr_cfg.model_path = asr_cfg.model_name
        asr_cfg.pretrained_name = None
    else:
        asr_cfg.pretrained_name = asr_cfg.model_name
        asr_cfg.model_path = None
    asr_model, model_name = setup_model(asr_cfg, map_location)

    model_cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(model_cfg.preprocessor, False)
    # some changes for streaming scenario
    model_cfg.preprocessor.dither = 0.0
    model_cfg.preprocessor.pad_to = 0

    if model_cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecRNNTBPEModel models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(model_cfg.preprocessor, True)

    asr_model.freeze()
    asr_model = asr_model.to(asr_model.device)
    asr_model.to(compute_dtype)

    if "max_symbols" in asr_cfg.decoding.greedy:
        # rename max_symbols -> max_symbols_per_step, as used in NeMo
        with open_dict(asr_cfg.decoding.greedy):
            asr_cfg.decoding.greedy.max_symbols_per_step = asr_cfg.decoding.greedy.max_symbols
            del asr_cfg.decoding.greedy.max_symbols

    decoding_cfg = OmegaConf.merge(OmegaConf.structured(RNNTDecodingConfig), asr_cfg.decoding)

    with open_dict(decoding_cfg):
        decoding_cfg.greedy.enable_per_stream_biasing = True
        decoding_cfg.beam.enable_per_stream_biasing = True
        if decoding_cfg.strategy != "greedy_batch" or decoding_cfg.greedy.loop_labels is not True:
            raise NotImplementedError(
                "This script currently supports only `greedy_batch` strategy with Label-Looping algorithm"
            )
        decoding_cfg.tdt_include_token_duration = True
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.fused_batch_size = -1  # temporarily stop fused batch during inference.
        decoding_cfg.beam.return_best_hypothesis = True  # return and write the best hypothsis only

    # Setup decoding strategy
    if hasattr(asr_model, 'change_decoding_strategy'):
        if not isinstance(asr_model, EncDecRNNTModel) and not isinstance(asr_model, EncDecHybridRNNTCTCModel):
            raise ValueError("The script supports rnnt model and hybrid model with rnnt decodng!")
        else:
            # rnnt model
            if isinstance(asr_model, EncDecRNNTModel):
                asr_model.change_decoding_strategy(decoding_cfg)

            # hybrid ctc rnnt model with decoder_type = rnnt
            if hasattr(asr_model, 'cur_decoder'):
                asr_model.change_decoding_strategy(decoding_cfg, decoder_type='rnnt')

    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()

    # decoding_computer = asr_model.decoding.decoding.decoding_computer
    return asr_model


def get_model_context(asr_model, cfg: DictConfig):
    audio_sample_rate = asr_model.cfg.preprocessor['sample_rate']
    assert audio_sample_rate == SAMPLE_RATE

    feature_stride_sec = asr_model.cfg.preprocessor['window_stride']
    features_per_sec = 1.0 / feature_stride_sec
    encoder_subsampling_factor = asr_model.encoder.subsampling_factor

    features_frame2audio_samples = make_divisible_by(
        int(audio_sample_rate * feature_stride_sec), factor=encoder_subsampling_factor
    )
    encoder_frame2audio_samples = features_frame2audio_samples * encoder_subsampling_factor

    context_encoder_frames = ContextSize(
        left=int(cfg.left_padding_size * features_per_sec / encoder_subsampling_factor),
        chunk=int(cfg.chunk_size * features_per_sec / encoder_subsampling_factor),
        right=int(cfg.right_padding_size * features_per_sec / encoder_subsampling_factor),
    )
    context_samples = ContextSize(
        left=context_encoder_frames.left * encoder_subsampling_factor * features_frame2audio_samples,
        chunk=context_encoder_frames.chunk * encoder_subsampling_factor * features_frame2audio_samples,
        right=context_encoder_frames.right * encoder_subsampling_factor * features_frame2audio_samples,
    )

    logging.info(
        "Corrected contexts (sec): "
        f"Left {context_samples.left / audio_sample_rate:.2f}, "
        f"Chunk {context_samples.chunk / audio_sample_rate:.2f}, "
        f"Right {context_samples.right / audio_sample_rate:.2f}"
    )
    logging.info(f"Corrected contexts (subsampled encoder frames): {context_encoder_frames}")
    logging.info(f"Corrected contexts (in audio samples): {context_samples}")
    latency_secs = (context_samples.chunk + context_samples.right) / audio_sample_rate
    logging.info(f"Theoretical latency: {latency_secs:.2f} seconds")
    return context_samples, context_encoder_frames, encoder_frame2audio_samples


def join_texts(texts: list[str]):
    result = ""
    for text in texts:
        text = text.strip()
        if text:
            if result:
                result += " " + text
            else:
                result = text
    return result


def get_common_prefix(sequence1: list, sequence2):
    common_prefix_len = 0
    for i in range(min(len(sequence1), len(sequence2))):
        if sequence1[i] == sequence2[i]:
            common_prefix_len += 1
        else:
            break
    return common_prefix_len


class NeMoStreamingPipelineAdapterV2(SpeechProcessor):
    """
    Adapter to use NeMo's streaming pipelines with simulstream evaluation.

    Architecture:
        audio_chunk → Frame → pipeline.transcribe_step() → TranscribeStepOutput → IncrementalOutput

    The pipeline internally handles:
        - Buffering (cache-aware or buffered mode)
        - Feature extraction
        - ASR decoding (CTC/RNN-T)
        - Translation (optional, via LLMTranslator)
        - State management per stream
    """

    asr_model = None
    nmt_model = None
    prompt_template = None
    context_samples = None
    context_encoder_frames = None
    encoder_frame2audio_samples = None
    asr_device = None
    output_manifest_path: Optional[str] = None
    wav_names: list[str] = []
    per_stream_boosting_requests: list[BiasingRequestItemConfig] | None = None
    detailed_log_path: str | None = None
    use_lcp: bool = True
    num_prev_sentences_for_translation: int = 5
    precomputed_asr_output: list[dict[str, Any]] | None = None

    def __init__(self, config: SimpleNamespace):
        """
        Initialize adapter.

        Args:
            config: Configuration from simulstream (SimpleNamespace)
                    Note: Will be converted to OmegaConf DictConfig for NeMo
        """
        if not SIMULSTREAM_AVAILABLE:
            raise ImportError("simulstream is required. Install with: pip install simulstream")

        super().__init__(config)

        # Stream state
        self.stream_id = 0
        self.frame_count = 0
        self.is_first_chunk = True
        # self._final_transcript_acc = ""
        # self._final_translation_acc = ""
        # self._last_partial_transcript = ""
        # self._last_partial_translation = ""
        # # Determine request type from config
        # self.request_type = getattr(config, 'request_type', 'frame')
        # if hasattr(config, 'streaming') and hasattr(config.streaming, 'request_type'):
        #     self.request_type = config.streaming.request_type
        self.latency_unit = getattr(config, 'latency_unit', 'word')
        if isinstance(self.latency_unit, str):
            self.latency_unit = self.latency_unit.lower()
        if self.latency_unit not in ("word", "char"):
            logging.warning(f"Unsupported latency_unit='{self.latency_unit}', defaulting to 'word'")
            self.latency_unit = "word"

        # Language settings (from runtime args)
        self.src_lang = None
        self.tgt_lang = None

        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self.asr_model.device,
        )
        self.decoding_state = None
        self.prev_tokens_rc = []
        self.prev_timestamps_rc = []
        self.accumulated_tokens = []
        self.accumulated_timestamps = []
        self.prev_sentences_asr = []
        self.prev_sentences_translated = []
        self.prev_partial_translation = ""
        self.prev_partial_translation_lcp = ""
        self.received_samples = 0
        self.precomputed_words_index = 0

    def reset_stream_state(self):
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self.asr_model.device,
        )
        self.decoding_state = None
        self.prev_tokens_rc = []
        self.prev_timestamps_rc = []
        self.accumulated_tokens = []
        self.accumulated_timestamps = []
        self.prev_sentences_asr = []
        self.prev_sentences_translated = []
        self.prev_partial_translation = ""
        self.prev_partial_translation_lcp = ""
        self.received_samples = 0
        self.precomputed_words_index = 0

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        """
        Load NeMo pipeline once (class-level, shared).

        Args:
            config: Configuration from simulstream
        """
        if cls.asr_model is not None or cls.nmt_model is not None:
            return  # Already loaded

        torch.set_float32_matmul_precision("high")

        # Convert SimpleNamespace to DictConfig
        # SimulStream uses SimpleNamespace for configuration, so we need to convert it to use in NeMo.
        cfg = OmegaConf.create(cls._namespace_to_dict(config))

        # setup LLM
        cls.nmt_model = get_llm_model(
            cfg.nmt.model_name, model_params=OmegaConf.to_container(cfg.nmt.llm_params, resolve=True)
        )
        cls.nmt_sampling_params = SamplingParams(**OmegaConf.to_container(cfg.nmt.sampling_params))
        cls.prompt_template = EuroLLMTranslatorPromptTemplate()

        if cfg.pipeline_v2.from_asr_manifest:
            cls.precomputed_asr_output = []
            manifest = read_manifest(cfg.pipeline_v2.from_asr_manifest)
            for record in manifest:
                cls.precomputed_asr_output.append(record["word"])

        # setup ASR model
        cls.asr_model = get_asr_model(cfg.asr)
        cls.asr_device = cls.asr_model.device
        # TODO: fix chunk masking
        cls.context_samples, cls.context_encoder_frames, cls.encoder_frame2audio_samples = get_model_context(
            cls.asr_model, cfg.streaming
        )
        if cls.context_samples.chunk != cls.context_samples.right:
            raise NotImplementedError

        # unified ASR model: use the att_context_size as chunk size (important for extra-low latency)
        if (
            cls.asr_model.cfg.encoder.att_context_style == "chunked_limited_with_rc"
            and cfg.pipeline_v2.att_context_size_as_chunk
        ):
            cls.asr_model.encoder.set_default_att_context_size(
                att_context_size=[
                    cls.context_encoder_frames.left,
                    cls.context_encoder_frames.chunk,
                    cls.context_encoder_frames.right,
                ]
            )

        cls.num_prev_sentences_for_translation = cfg.pipeline_v2.get("num_prev_sentences_for_translation", 5)
        cls.use_lcp = cfg.pipeline_v2.get("use_lcp", True)
        cls.detailed_log_path = getattr(config, "detailed_log_path", None)
        if cls.detailed_log_path is not None:
            # rewrite
            with open(cls.detailed_log_path, "w", encoding="utf-8") as f:
                f.write("")

        # Output manifest path (optional, but enabled by default when metrics_log_file is available).
        cls.output_manifest_path = getattr(config, 'output_manifest_file', None) or getattr(
            config, 'output_manifest', None
        )
        # cls.detailed_output_path = Path(cls.output_manifest_path).parent / "detailed_output.jsonl"
        if cls.output_manifest_path is None:
            metrics_log_file = getattr(config, 'metrics_log_file', None)
            if metrics_log_file:
                metrics_path = Path(metrics_log_file)
                cls.output_manifest_path = str(metrics_path.parent / f"{metrics_path.stem}_pred_manifest.jsonl")

        if cls.output_manifest_path:
            # Truncate at start of run.
            Path(cls.output_manifest_path).write_text("", encoding="utf-8")
            logging.info(f"Prediction manifest output: {cls.output_manifest_path}")

        # Load wav names from wav list if available.
        cls.wav_names = []
        wav_list_file = getattr(config, 'wav_list_file', None)
        if wav_list_file and Path(wav_list_file).exists():
            with open(wav_list_file, 'r', encoding='utf-8') as f:
                cls.wav_names = [line.strip() for line in f if line.strip()]

        # Register cleanup handler to properly shutdown vLLM on exit
        # Attempting to gracefully shut down vLLM engine, to get "ERROR 02-09 16:53:28 [core_client.py:610] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client."
        # Works for now, but returns warning.
        # TODO: Find a better way to gracefully shut down vLLM engine.
        atexit.register(cls.cleanup_model)

        # logging.info(f"  ASR model: {cfg.asr.model_name}")
        # if cfg.get('enable_nmt', False):
        #     logging.info(f"  NMT model: {cfg.nmt.model_name}")
        #     logging.info(f"  Translation: {cfg.nmt.source_language} → {cfg.nmt.target_language}")

        if cfg.get("per_stream_boosting") and cfg.per_stream_boosting.get("phrases_file"):
            boosting_model_alpha = cfg.per_stream_boosting.get("alpha", 1.0)
            with open(cfg.per_stream_boosting.phrases_file, "r", encoding="utf-8") as f:
                boosting_requests_raw = json.load(f)
                cls.per_stream_boosting_requests = [
                    BiasingRequestItemConfig(
                        BoostingTreeModelConfig(key_phrases_list=item["key_phrases_list"]),
                        boosting_model_alpha=boosting_model_alpha,
                        auto_manage_multi_model=False,
                    )
                    for item in boosting_requests_raw
                ]
            logging.info(
                f"Per-stream boosting enabled with weight {boosting_model_alpha:.2g}, "
                f"expected {len(cls.per_stream_boosting_requests)} ordered streams"
            )
        else:
            logging.info(
                "Per-stream boosting disabled; to enable, "
                "specify `per_stream_boosting.phrases_file` and `per_stream_boosting.alpha`"
            )

    @staticmethod
    def _namespace_to_dict(obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: NeMoStreamingPipelineAdapterV2._namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: NeMoStreamingPipelineAdapterV2._namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NeMoStreamingPipelineAdapterV2._namespace_to_dict(item) for item in obj]
        return obj

    def set_source_language(self, language: str) -> None:
        """Set source language (simulstream interface)."""
        self.src_lang = language

    def set_target_language(self, language: str) -> None:
        """Set target language (simulstream interface)."""
        self.tgt_lang = language

    @staticmethod
    def get_num_sentences(text: str) -> int:
        n_utt = text.count(".▁") + text.count("!▁") + text.count("?▁")
        if text.endswith((".", "!", "?")):
            n_utt += 1
        return n_utt

    @staticmethod
    def get_hyp_repr_with_temp(text: str, text_rc: str):
        text_repr = ""
        if text:
            text_repr = text
        if text_rc:
            if not text_repr:
                text_repr = f"[{text_rc}]"
            else:
                text_repr += f" [{text_rc}]"
        return text_repr

    def get_translation(self, text: str, translation_lcp="", verbose=False):
        llm_input = self.prompt_template.format(
            LANGUAGE_CODES[self.src_lang],
            LANGUAGE_CODES[self.tgt_lang],
            src_prefix=text,
            tgt_prefix=translation_lcp,
            src_context=" ".join(self.prev_sentences_asr[-self.num_prev_sentences_for_translation :]),
            tgt_context=" ".join(self.prev_sentences_translated[-self.num_prev_sentences_for_translation :]),
        )
        llm_output = self.nmt_model.generate([llm_input], self.nmt_sampling_params, use_tqdm=False)
        output_text = llm_output[0].outputs[0].text
        output_text = self.prompt_template.extract(output_text).strip()
        if verbose:
            print(f"Input: {llm_input}")
            print('-' * 15)
            print(f"Text: {text}")
            print(f"LCP: {translation_lcp}")
            print(f"Output: {output_text}")
        return output_text

    def _next_precomputed_asr_step(self, audio: np.ndarray):
        # time_start = self.received_samples / SAMPLE_RATE
        time_end = (self.received_samples + len(audio)) / SAMPLE_RATE
        words = []
        last_timestamp = 0
        while self.precomputed_words_index < len(self.precomputed_asr_output[self.stream_id]):
            word = self.precomputed_asr_output[self.stream_id][self.precomputed_words_index]
            if word["end"] > time_end:  # end - in seconds
                break
            words.append(word["word"])
            last_timestamp = word["end_offset"]  # end_offset - in frames
            self.precomputed_words_index += 1

        text = " ".join(words)
        tokens = self.asr_model.tokenizer.text_to_ids(text)
        # timestamps - currently not used
        # we are using here simple heuristic of assigning the last timestamps to all tokens,
        # since timestamps for tokens are not in manifest, unrecoverable
        timestamps = [last_timestamp for _ in range(len(tokens))]
        self.received_samples += len(audio)
        return (text, tokens, timestamps), ("", [], [])

    def _next_asr_step(self, audio: np.ndarray):
        if audio.ndim > 1:
            raise ValueError("Simulstream processes only one audio at a time (batch size 1).")

        audio_length = len(audio)
        is_last_chunk = audio_length < self.context_samples.chunk
        # Convert audio to torch tensor
        audio_tensor = torch.from_numpy(audio[None, :]).float().to(self.asr_device)

        if self.per_stream_boosting_requests is not None:
            biasing_request = self.per_stream_boosting_requests[self.stream_id]
        else:
            biasing_request = None

        if biasing_request is not None and (not biasing_request.is_empty()):
            if self.is_first_chunk:
                biasing_request.add_to_multi_model(
                    tokenizer=self.asr_model.tokenizer,
                    biasing_multi_model=self.asr_model.decoding.decoding.decoding_computer.biasing_multi_model,
                )

            multi_biasing_ids = torch.full(
                [1], fill_value=biasing_request.multi_model_id, dtype=torch.long, device=self.asr_device
            )
        else:
            multi_biasing_ids = None

        is_last_chunk_batch = torch.full([1], fill_value=is_last_chunk, device=self.asr_device)
        with torch.inference_mode(), torch.no_grad():
            self.buffer.add_audio_batch_(
                audio_tensor,
                audio_lengths=torch.full([1], fill_value=audio_length, device=self.asr_device),
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=is_last_chunk_batch,
            )

            # get encoder output using full buffer [left-chunk-right]
            encoder_output, encoder_output_len = self.asr_model(
                input_signal=self.buffer.samples,
                input_signal_length=self.buffer.context_size_batch.total(),
            )
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            # remove extra context from encoder_output (leave only frames corresponding to the chunk)
            encoder_context = self.buffer.context_size.subsample(factor=self.encoder_frame2audio_samples)
            encoder_context_batch = self.buffer.context_size_batch.subsample(factor=self.encoder_frame2audio_samples)
            # remove left context
            encoder_output = encoder_output[:, encoder_context.left :]
            encoder_output_len_to_decode = torch.where(
                is_last_chunk_batch,
                encoder_output_len - encoder_context_batch.left,
                encoder_context_batch.chunk,
            )
            batched_hyps_chunk, _, self.decoding_state = self.asr_model.decoding.decoding.decoding_computer(
                x=encoder_output,
                out_len=encoder_output_len_to_decode,
                prev_batched_state=self.decoding_state,
                multi_biasing_ids=multi_biasing_ids,
            )
        # get hyp for current chunk
        hyp_len = batched_hyps_chunk.current_lengths[0].cpu().item()
        if hyp_len:
            tokens = batched_hyps_chunk.transcript[0, :hyp_len].cpu().tolist()
            timestamps = batched_hyps_chunk.timestamps[0, :hyp_len].cpu().tolist()
            text = self.asr_model.tokenizer.ids_to_text(tokens)
        else:
            tokens = []
            timestamps = []
            text = ""

        # get hypothesis from right context (temporary ASR hypothesis part)
        text_rc = ""
        tokens_rc = []
        timestamps_rc = []
        if not is_last_chunk:
            # decode right context
            with torch.inference_mode(), torch.no_grad():
                decoded_len = encoder_output_len_to_decode[0].item()
                encoder_output = encoder_output[:, decoded_len:]
                # shift_indices = torch.arange(max_time, device=self.asr_device, dtype=torch.long)[None, :] + enc_lens_chunk[:, None]
                # # pad with zeros everything beyond needed context
                # shift_indices = torch.where(shift_indices < max_time, shift_indices, torch.zeros_like(shift_indices))
                batched_hyps_rc, _, _ = self.asr_model.decoding.decoding.decoding_computer(
                    # torch.gather(encs_dim_last, dim=1, index=shift_indices[:, :, None].expand(-1, -1, feat_dim)),
                    x=encoder_output,
                    out_len=encoder_context_batch.right,
                    prev_batched_state=self.decoding_state,
                    multi_biasing_ids=multi_biasing_ids,
                )
            hyp_len_rc = batched_hyps_rc.current_lengths[0].cpu().item()
            if hyp_len_rc > 0:
                tokens_rc = batched_hyps_rc.transcript[0, :hyp_len_rc].cpu().tolist()
                timestamps_rc = batched_hyps_rc.timestamps[0, :hyp_len_rc].cpu().tolist()
                text_rc = self.asr_model.tokenizer.ids_to_text(tokens_rc)
            else:
                tokens_rc = []
                timestamps_rc = []
                text_rc = ""
        return (text, tokens, timestamps), (text_rc, tokens_rc, timestamps_rc)

    def process_chunk(self, audio: np.ndarray) -> "IncrementalOutput":
        """
        Process audio chunk using NeMo's native streaming API.

        This creates a Frame or FeatureBuffer request (depending on config) and
        calls pipeline.transcribe_step(), which internally handles all buffering,
        feature extraction, and decoding.

        Auto-detects the last chunk by comparing chunk size to expected size.
        If chunk is smaller than expected, it's treated as the last chunk.
        NOTE: works only with batch size 1 (so does SimulStream).

        Args:
            audio: Audio chunk (numpy array, float32, mono, 16kHz)

        Returns:
            IncrementalOutput: Streaming results (partial/final ASR + translation)
        """
        # import pdb; pdb.set_trace()
        if self.precomputed_asr_output is not None:
            (text, tokens, timestamps), (text_rc, tokens_rc, timestamps_rc) = self._next_precomputed_asr_step(
                audio=audio
            )
        else:
            (text, tokens, timestamps), (text_rc, tokens_rc, timestamps_rc) = self._next_asr_step(audio=audio)
        # logging.info(f"Text: {self.get_hyp_repr_with_temp(text, text_rc)}")
        # add fixed part to accumulated ASR hypothesis (will not change in future)
        self.accumulated_tokens += tokens
        self.accumulated_timestamps += timestamps

        # split accumulated ASR hypothesis into fixed part (complete sentence) and non-fixed (incomplete sentence)
        # accumulated tokens should contain only non-fixed hypothesis
        accumulated_text = self.asr_model.tokenizer.ids_to_text(self.accumulated_tokens)
        if accumulated_text.endswith(".") or accumulated_text.endswith("?") or accumulated_text.endswith("!"):
            fixed_part = self.asr_model.tokenizer.ids_to_text(self.accumulated_tokens)
            self.accumulated_tokens = []
        else:
            tokens_repr = self.asr_model.tokenizer.ids_to_tokens(self.accumulated_tokens)
            incomplete_part_i = None
            for i in range(len(self.accumulated_tokens) - 1, 0, -1):
                if tokens_repr[i].startswith("▁") and (
                    tokens_repr[i - 1].endswith(".")
                    or tokens_repr[i - 1].endswith("?")
                    or tokens_repr[i - 1].endswith("!")
                ):
                    incomplete_part_i = i
                    break
            if incomplete_part_i is not None:
                fixed_part = self.asr_model.tokenizer.ids_to_text(self.accumulated_tokens[:incomplete_part_i])
                self.accumulated_tokens = self.accumulated_tokens[incomplete_part_i:]
            else:
                fixed_part = ""

        non_fixed_part = self.asr_model.tokenizer.ids_to_text(self.accumulated_tokens + tokens_rc)
        # logging.info(f"Split for translation: {self.get_hyp_repr_with_temp(fixed_part, non_fixed_part)}")

        if fixed_part or non_fixed_part:
            prev_partial_translation_initial = self.prev_partial_translation
            if fixed_part:
                # translate fixed part (will not be changed in future)
                fixed_part_translated = join_texts(
                    [
                        self.prev_partial_translation_lcp,
                        self.get_translation(fixed_part, translation_lcp=self.prev_partial_translation_lcp),
                    ]
                )
                self.prev_sentences_asr.append(fixed_part)
                self.prev_sentences_translated.append(fixed_part_translated)
                self.prev_partial_translation_lcp = ""
                self.prev_partial_translation = ""
            else:
                fixed_part_translated = ""

            if non_fixed_part:
                # translate non-fixed part, can be changed in future, but keep longest common prefix (lcp)
                non_fixed_part_translated = join_texts(
                    [
                        self.prev_partial_translation_lcp,
                        self.get_translation(non_fixed_part, translation_lcp=self.prev_partial_translation_lcp),
                    ]
                )

                if self.use_lcp and non_fixed_part_translated and self.prev_partial_translation:
                    # update lcp
                    tokens_non_fixed = self._tokenize_text(non_fixed_part_translated)
                    tokens_previous_partial = self._tokenize_text(self.prev_partial_translation)
                    common_non_fixed_prefix_len = get_common_prefix(tokens_non_fixed, tokens_previous_partial)
                    if common_non_fixed_prefix_len > 0:
                        self.prev_partial_translation_lcp = self._join_tokens(
                            tokens_non_fixed[:common_non_fixed_prefix_len]
                        )
                self.prev_partial_translation = non_fixed_part_translated
            else:
                non_fixed_part_translated = ""
            # logging.info(f"Translation: {self.get_hyp_repr_with_temp(fixed_part_translated, non_fixed_part_translated)}")

            # delete old invalid, emit new tokens
            full_translation_to_output = join_texts([fixed_part_translated, non_fixed_part_translated])
            curr_tokens = self._tokenize_text(full_translation_to_output)
            prev_tokens = self._tokenize_text(prev_partial_translation_initial)
            common_prefix_len = get_common_prefix(curr_tokens, prev_tokens)

            # Calculate deleted and generated token lists
            deleted_tokens = prev_tokens[common_prefix_len:]  # Tokens removed from previous
            generated_tokens = curr_tokens[common_prefix_len:]  # Tokens added in current
            # Construct strings from token lists
            deleted_string = self._join_tokens(deleted_tokens)
            generated_string = self._join_tokens(generated_tokens)
        else:
            fixed_part_translated = ""
            non_fixed_part_translated = ""
            generated_tokens = []
            generated_string = ""
            deleted_tokens = []
            deleted_string = ""

        if self.detailed_log_path is not None:
            with open(self.detailed_log_path, "a", encoding="utf-8") as f:
                print(
                    json.dumps(
                        {
                            "step_asr_text": text,
                            "step_asr_text_rc": text_rc,
                            "fixed_asr_text": fixed_part,
                            "non_fixed_asr_text": non_fixed_part,
                            "fixed_part_translated": fixed_part_translated,
                            "non_fixed_part_translated": non_fixed_part_translated,
                            "translation_lcp": self.prev_partial_translation_lcp,
                            "new_tokens": generated_tokens,
                            "new_string": generated_string,
                            "deleted_tokens": deleted_tokens,
                            "deleted_string": deleted_string,
                        }
                    ),
                    file=f,
                )

        self.is_first_chunk = False
        self.frame_count += 1

        return IncrementalOutput(
            new_tokens=generated_tokens,  # List of string tokens added
            new_string=generated_string,
            deleted_tokens=deleted_tokens,  # List of string tokens removed
            deleted_string=deleted_string,
        )

    def _convert_to_incremental_output(self, step_output) -> "IncrementalOutput":
        """
        Convert NeMo's TranscribeStepOutput to simulstream's IncrementalOutput.

        Calculate generated and deleted tokens by comparing previous and current partial outputs.
        Uses tokenization based on latency_unit:
          - word: split by whitespace
          - char: split into individual characters
        TODO: Think more on how actualyy this tokenization should be done.

        Args:
            step_output: NeMo's TranscribeStepOutput object with:
                - previous_partial_transcript: Previous step's partial transcript
                - previous_partial_translation: Previous step's partial translation
                - partial_transcript: Current step's partial transcript
                - partial_translation: Current step's partial translation (if NMT enabled)

        Returns:
            IncrementalOutput: Simulstream format with generated/deleted token lists
        """

        prev_partial = step_output.previous_partial_translation
        if step_output.final_translation:
            current_partial = step_output.final_translation
        elif step_output.partial_translation:
            current_partial = step_output.partial_translation
        else:
            current_partial = ""

        print(f"Current partial: {current_partial}")
        prev_tokens = self._tokenize_text(prev_partial)
        curr_tokens = self._tokenize_text(current_partial)

        # Find longest common prefix to identify what changed
        common_prefix_len = 0
        for i in range(min(len(prev_tokens), len(curr_tokens))):
            if prev_tokens[i] == curr_tokens[i]:
                common_prefix_len += 1
            else:
                break

        # Calculate deleted and generated token lists
        deleted_tokens = prev_tokens[common_prefix_len:]  # Tokens removed from previous
        generated_tokens = curr_tokens[common_prefix_len:]  # Tokens added in current

        # Construct strings from token lists
        deleted_string = self._join_tokens(deleted_tokens)
        generated_string = self._join_tokens(generated_tokens)

        return IncrementalOutput(
            new_tokens=generated_tokens,  # List of string tokens added
            new_string=generated_string,
            deleted_tokens=deleted_tokens,  # List of string tokens removed
            deleted_string=deleted_string,
        )

    def end_of_stream(self) -> "IncrementalOutput":
        """
        Called at the end of audio stream to finalize output.

        In most cases, the last chunk is auto-detected by size and processed with
        is_last=True in process_chunk(), so this returns empty output.

        This is kept as required by SpeechProcessor interface and serves as a
        fallback for edge cases where the last chunk has the same size as others.

        Returns:
            IncrementalOutput: Empty output in most cases
        """
        # pred_text = (self._final_transcript_acc + self._last_partial_transcript).strip()
        # pred_translation = (self._final_translation_acc + self._last_partial_translation).strip()
        # self._write_prediction_manifest_line(pred_text, pred_translation)

        # NOTE: Last chunk was already processed with is_last=False in process_chunk().
        # We only finalize stream state and emit empty incremental output here.
        return IncrementalOutput(
            new_tokens=[],
            new_string="",
            deleted_tokens=[],
            deleted_string="",
        )

    def clear(self) -> None:
        """
        Clear stream state and prepare for next audio (simulstream interface).

        This finalizes the current stream and resets state for a new one.
        """
        # Finalize current stream if we've processed anything
        if not self.is_first_chunk:
            self.end_of_stream()

        if self.per_stream_boosting_requests is not None:
            biasing_request = self.per_stream_boosting_requests[self.stream_id]
        else:
            biasing_request = None
        if biasing_request is not None and (not biasing_request.is_empty()):
            assert biasing_request.multi_model_id is not None
            self.asr_model.decoding.decoding.decoding_computer.biasing_multi_model.remove_model(
                biasing_request.multi_model_id
            )

        logging.info(f"Finished transcribing stream {self.stream_id}")

        # Reset for next stream
        self.stream_id += 1
        self.frame_count = 0
        self.is_first_chunk = True
        # self._final_transcript_acc = ""
        # self._final_translation_acc = ""
        # self._last_partial_transcript = ""
        # self._last_partial_translation = ""

        self.reset_stream_state()

    def _write_prediction_manifest_line(self, pred_text: str, pred_translation: str) -> None:
        """Write one NeMo-style manifest line with model predictions."""
        if not self.output_manifest_path:
            return

        audio_filepath = ""
        if self.stream_id < len(self.wav_names):
            audio_filepath = self.wav_names[self.stream_id]

        item = {
            "audio_filepath": audio_filepath,
            "pred_text": pred_text,
            "pred_translation": pred_translation,
            # Keep plural alias for compatibility with downstream scripts expecting this key.
            "pred_translations": pred_translation,
        }
        with open(self.output_manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def tokens_to_string(self, tokens: list[str]) -> str:
        """
        Convert tokens to string using NeMo's tokenizer.

        Args:
            tokens: list of token strings (BPE/SentencePiece tokens)

        Returns:
            Detokenized string
        """
        return self._join_tokens(tokens)

    def _tokenize_text(self, text: Optional[str]) -> list[str]:
        """Tokenize text according to configured latency unit. For char-level, removes
        all spaces so emitted token count matches simulstream eval (MWER path does
        .replace(" ", "") on resegmented text, so delay count must be non-space chars only)."""
        if not text:
            return []
        # for compatability with omnisteval
        text = text.replace("…", "")
        if self.latency_unit == "char":
            return list(text.strip())
        return text.strip().split()

    def _join_tokens(self, tokens: list[str]) -> str:
        """Join tokens according to configured latency unit."""
        if not tokens:
            return ""
        if self.latency_unit == "char":
            return "".join(tokens)
        return " ".join(tokens)

    @classmethod
    def cleanup_model(cls):
        """
        Explicitly cleanup vLLM and release resources.
        Call this when done with inference to properly shutdown vLLM engine.
        """
        if cls.nmt_model is not None:
            try:
                # vLLM cleanup - destroy the engine to release Ray resources
                if hasattr(cls.nmt_model, 'llm_engine'):
                    # Destroy the engine core
                    destroy_model_parallel()
            except Exception as e:
                print(f"[NeMo Adapter] Warning during vLLM cleanup: {e}")

    def __del__(self):
        """Cleanup when adapter is destroyed"""
        # Note: cleanup_model() is class-level, should be called explicitly
        # since multiple adapter instances share the same pipeline
        pass
