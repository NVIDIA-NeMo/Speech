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

"""
Adapter to use NeMo's native streaming pipelines with simulstream evaluation.

This adapter properly interfaces with NeMo's internal streaming API (transcribe_step)
rather than duplicating chunking/buffering logic. NeMo handles all buffering internally.

Key Insight:
    NeMo's pipelines already have complete streaming infrastructure:
    - Frame creation and buffering logic (BufferedRNNTPipeline / CacheAwareRNNTPipeline)
    - State management (StreamingState)
    - Translation integration (LLMTranslator)

    We just need to:
    1. Create Frame requests from audio chunks
    2. Call pipeline.transcribe_step()
    3. Convert TranscribeStepOutput -> IncrementalOutput
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import BiasingRequestItemConfig
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.utils import logging

try:
    from simulstream.server.speech_processors import SpeechProcessor
    from simulstream.server.speech_processors.incremental_output import IncrementalOutput

    SIMULSTREAM_AVAILABLE = True
except ImportError:
    SIMULSTREAM_AVAILABLE = False
    SpeechProcessor = object


class NeMoStreamingPipelineAdapter(SpeechProcessor):
    """
    Adapter to use NeMo's streaming pipelines with simulstream evaluation.

    Architecture:
        audio_chunk -> Frame -> pipeline.transcribe_step() -> TranscribeStepOutput -> IncrementalOutput

    The pipeline internally handles:
        - Buffering (cache-aware or buffered mode)
        - Feature extraction
        - ASR decoding (CTC/RNN-T)
        - Translation (optional, via LLMTranslator)
        - State management per stream
    """

    pipeline = None  # Class-level pipeline (shared across instances)
    output_manifest_path: Optional[str] = None
    wav_names: list[str] = []
    per_stream_boosting_requests: list[BiasingRequestItemConfig] | None = None
    _wer_hyps: list[str] = []
    _wer_refs: list[str] = []

    def __init__(self, config: SimpleNamespace):
        """
        Initialize adapter.

        Args:
            config: Configuration from simulstream (SimpleNamespace). Will be converted to an
                OmegaConf DictConfig for NeMo in `load_model`.
        """
        if not SIMULSTREAM_AVAILABLE:
            raise ImportError("simulstream is required. Install with: pip install simulstream")

        super().__init__(config)

        self.stream_id = 0
        self._reset_stream_state()

        self.latency_unit = getattr(config, 'latency_unit', 'word')
        if isinstance(self.latency_unit, str):
            self.latency_unit = self.latency_unit.lower()
        if self.latency_unit not in ("word", "char"):
            logging.warning(f"Unsupported latency_unit='{self.latency_unit}', defaulting to 'word'")
            self.latency_unit = "word"

        # Language settings (set at runtime by simulstream via set_source_language/set_target_language)
        self.src_lang = None
        self.tgt_lang = None

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        """
        Load the NeMo pipeline once (class-level, shared across all stream instances).

        Args:
            config: Configuration from simulstream.
        """
        if cls.pipeline is not None:
            return  # Already loaded

        import atexit

        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

        # SimulStream configs are SimpleNamespace objects; NeMo expects an OmegaConf DictConfig.
        cfg = OmegaConf.create(cls._namespace_to_dict(config))

        # This adapter always sends raw-audio Frame requests (see process_chunk), so the pipeline
        # must be built to match: `streaming.request_type` controls whether BufferedRNNTPipeline
        # builds a Frame- or feature_buffer-based bufferer, and mismatching it would silently
        # produce unpadded/wrong features. Force it here instead of relying on every config.
        if cfg.get('streaming', {}).get('request_type', 'frame') != 'frame':
            logging.warning(
                f"Overriding streaming.request_type='{cfg.streaming.request_type}' to 'frame' "
                f"({type(cls).__name__} only supports frame requests)."
            )
            cfg.streaming.request_type = 'frame'
        cls.cfg = cfg

        cls.pipeline = PipelineBuilder.build_pipeline(cfg)
        cls.pipeline.open_session()

        cls.detailed_log_path = getattr(config, "detailed_log_path", None)

        # Output manifest path (optional, but derived from metrics_log_file when not set explicitly).
        cls.output_manifest_path = getattr(config, 'output_manifest_file', None) or getattr(
            config, 'output_manifest', None
        )
        if cls.output_manifest_path is None:
            metrics_log_file = getattr(config, 'metrics_log_file', None)
            if metrics_log_file:
                metrics_path = Path(metrics_log_file)
                cls.output_manifest_path = str(metrics_path.parent / f"{metrics_path.stem}_pred_manifest.jsonl")

        if cls.output_manifest_path:
            Path(cls.output_manifest_path).write_text("", encoding="utf-8")  # truncate at start of run
            logging.info(f"Prediction manifest output: {cls.output_manifest_path}")
        cls._wer_calculated = False
        cls._wer_hyps = []
        cls._wer_refs = []

        cls.wav_names = []
        wav_list_file = getattr(config, 'wav_list_file', None)
        if wav_list_file and Path(wav_list_file).exists():
            with open(wav_list_file, 'r', encoding='utf-8') as f:
                cls.wav_names = [line.strip() for line in f if line.strip()]

        cls._load_reference_manifest(config)

        # vLLM (used by LLMTranslator) needs to be shut down explicitly, otherwise it can hang or
        # print noisy errors on process exit.
        atexit.register(cls.cleanup_model)

        logging.info(f"Loaded NeMo pipeline: {type(cls.pipeline).__name__}")
        logging.info(f"  ASR model: {cfg.asr.model_name}")
        if cfg.get('enable_nmt', False):
            logging.info(f"  NMT model: {cfg.nmt.model_name}")
            logging.info(f"  Translation: {cfg.nmt.source_language} -> {cfg.nmt.target_language}")

        if cfg.get("per_stream_boosting") and cfg.per_stream_boosting.get("phrases_file"):
            boosting_model_alpha = cfg.per_stream_boosting.get("alpha", 1.0)
            with open(cfg.per_stream_boosting.phrases_file, "r", encoding="utf-8") as f:
                boosting_requests_raw = json.load(f)
                cls.per_stream_boosting_requests = [
                    BiasingRequestItemConfig(
                        BoostingTreeModelConfig(key_phrases_list=item["key_phrases_list"]),
                        boosting_model_alpha=boosting_model_alpha,
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

    @classmethod
    def _load_reference_manifest(cls, config: SimpleNamespace) -> None:
        """Load optional input manifest to copy reference text fields and enable WER calculation."""
        cls.reference_manifest_by_audio = {}
        cls.reference_manifest_by_basename = {}
        cls.reference_manifest_items_ordered = []

        manifest_path = None
        for key in (
            "manifest",
            "manifest_file",
            "input_manifest",
            "input_manifest_file",
            "reference_manifest",
        ):
            value = getattr(config, key, None)
            if value:
                manifest_path = value
                break

        if not manifest_path:
            return

        manifest_path = str(manifest_path)
        if not Path(manifest_path).exists():
            logging.warning(f"Reference manifest path not found: {manifest_path}")
            return

        manifest_dir = Path(manifest_path).parent
        loaded = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                audio = item.get("audio_filepath", "")
                if not audio:
                    continue
                audio_path = Path(audio)
                if not audio_path.is_absolute():
                    audio_path = manifest_dir / audio_path
                audio_abs = str(audio_path.resolve())
                cls.reference_manifest_by_audio[audio_abs] = item
                cls.reference_manifest_by_basename[audio_path.name] = item
                cls.reference_manifest_items_ordered.append(item)
                loaded += 1

        logging.info(f"Loaded reference manifest entries: {loaded}")

    @staticmethod
    def _namespace_to_dict(obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: NeMoStreamingPipelineAdapter._namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [NeMoStreamingPipelineAdapter._namespace_to_dict(item) for item in obj]
        return obj

    def set_source_language(self, language: str) -> None:
        """Set source language (simulstream interface)."""
        self.src_lang = language

    def set_target_language(self, language: str) -> None:
        """Set target language (simulstream interface)."""
        self.tgt_lang = language

    def process_chunk(self, audio: np.ndarray) -> IncrementalOutput:
        """
        Process an audio chunk using NeMo's native streaming API.

        Creates a Frame request and calls pipeline.transcribe_step(), which internally
        handles all buffering, feature extraction, and decoding.

        NOTE: works only with batch size 1 (so does SimulStream).

        Args:
            audio: Audio chunk (numpy array, float32, mono, 16kHz)

        Returns:
            IncrementalOutput: Streaming results (partial/final ASR + translation)
        """
        if audio.ndim > 1:
            raise ValueError("Simulstream processes only one audio at a time (batch size 1).")

        expected_chunk_size = int(16000 * self.speech_chunk_size)
        audio_length = len(audio)
        if audio_length < expected_chunk_size:
            audio = np.concatenate([audio, np.zeros(expected_chunk_size - audio_length)])
        audio_tensor = torch.from_numpy(audio).float().to(self.pipeline.device)

        if self.is_first_chunk and self.per_stream_boosting_requests is not None:
            biasing_cfg = self.per_stream_boosting_requests[self.stream_id]
        else:
            biasing_cfg = None

        # simulstream doesn't tell us whether a chunk is the last one; is_last is always False here,
        # and any leftover right-context is handled by end_of_stream()/return_tail_result. The
        # pipeline's own internal bufferer accumulates the left/right padding sliding window
        # per stream from these raw-audio Frames (see __init__ for why "frame" is the only
        # supported request type).
        request = Frame(
            stream_id=self.stream_id,
            samples=audio_tensor,
            is_first=self.is_first_chunk,
            is_last=False,
            length=audio_length,
            options=ASRRequestOptions(biasing_cfg=biasing_cfg) if self.is_first_chunk else None,
        )

        # This internally handles: buffering -> encoding -> decoding -> translation
        step_outputs = self.pipeline.transcribe_step([request])
        step_output = step_outputs[0]

        # Snapshot the previous accumulated transcript before updating it below, so
        # _convert_to_incremental_output can diff against it when NMT is disabled.
        previous_transcript = self._final_transcript_acc + self._last_partial_transcript

        # Track final/latest-partial outputs to write a NeMo-style prediction manifest line.
        self._final_transcript_acc += step_output.final_transcript or ""
        self._final_translation_acc += step_output.final_translation or ""
        self._last_partial_transcript = step_output.partial_transcript or ""
        self._last_partial_translation = step_output.partial_translation or ""

        result = self._convert_to_incremental_output(step_output, previous_transcript)

        self.is_first_chunk = False

        if self.detailed_log_path is not None:
            with open(self.detailed_log_path, "a", encoding="utf-8") as f:
                print(
                    json.dumps(
                        {
                            "final_transcript": step_output.final_transcript,
                            "partial_transcript": step_output.partial_transcript,
                            "final_translation": step_output.final_translation,
                            "partial_translation": step_output.partial_translation,
                            "new_tokens": result.new_tokens,
                            "new_string": result.new_string,
                            "deleted_tokens": result.deleted_tokens,
                            "deleted_string": result.deleted_string,
                        }
                    ),
                    file=f,
                )

        return result

    def _convert_to_incremental_output(self, step_output, previous_transcript: str = "") -> IncrementalOutput:
        """
        Convert NeMo's TranscribeStepOutput to simulstream's IncrementalOutput.

        Computes generated/deleted tokens by diffing the previous and current partial (or final)
        output, tokenized according to `latency_unit` (word-split, or per-character for languages
        like Chinese). When NMT is enabled, the translation is diffed (since re-translation from
        the current transcript prefix can revise earlier text); otherwise the ASR transcript is
        diffed directly.

        Args:
            step_output: NeMo's TranscribeStepOutput for the current chunk.
            previous_transcript: Full accumulated transcript (final + partial) before this step,
                used as the diff baseline when NMT is disabled.

        Returns:
            IncrementalOutput: Simulstream format with generated/deleted token lists.
        """
        if self.pipeline.nmt_enabled:
            is_final = bool(step_output.final_transcript)
            prev_partial = self._prev_partial_translation
            current_partial = step_output.final_translation if is_final else step_output.partial_translation
            self._prev_partial_translation = "" if is_final else current_partial
        else:
            prev_partial = previous_transcript
            current_partial = self._final_transcript_acc + self._last_partial_transcript

        prev_tokens = self._tokenize_text(prev_partial)
        curr_tokens = self._tokenize_text(current_partial)

        common_prefix_len = 0
        for i in range(min(len(prev_tokens), len(curr_tokens))):
            if prev_tokens[i] == curr_tokens[i]:
                common_prefix_len += 1
            else:
                break

        deleted_tokens = prev_tokens[common_prefix_len:]
        generated_tokens = curr_tokens[common_prefix_len:]

        return IncrementalOutput(
            new_tokens=generated_tokens,
            new_string=self._join_tokens(generated_tokens),
            deleted_tokens=deleted_tokens,
            deleted_string=self._join_tokens(deleted_tokens),
        )

    def end_of_stream(self) -> IncrementalOutput:
        """
        Called at the end of the audio stream to finalize output.

        The last chunk was already processed with is_last=False in process_chunk() (simulstream
        doesn't signal which chunk is last), so this only finalizes stream state / writes the
        prediction manifest line and emits an empty incremental output. Required by the
        SpeechProcessor interface.
        """
        if self._finalized:
            return IncrementalOutput(new_tokens=[], new_string="", deleted_tokens=[], deleted_string="")

        pred_text = (self._final_transcript_acc + self._last_partial_transcript).strip()
        pred_translation = (self._final_translation_acc + self._last_partial_translation).strip()
        self._write_prediction_manifest_line(pred_text, pred_translation)

        self.pipeline.delete_state(self.stream_id)
        self._finalized = True
        return IncrementalOutput(new_tokens=[], new_string="", deleted_tokens=[], deleted_string="")

    def clear(self) -> None:
        """
        Clear stream state and prepare for the next audio stream (simulstream interface).
        """
        if not self.is_first_chunk and not self._finalized:
            self.end_of_stream()

        self.stream_id += 1
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        """Reset per-stream accumulators (used on init and between streams)."""
        self.is_first_chunk = True
        self._finalized = False
        self._final_transcript_acc = ""
        self._final_translation_acc = ""
        self._last_partial_transcript = ""
        self._last_partial_translation = ""
        self._prev_partial_translation = ""

    def _write_prediction_manifest_line(self, pred_text: str, pred_translation: str) -> None:
        """Write one NeMo-style manifest line with model predictions, and accumulate WER inputs."""
        audio_filepath = ""
        if self.stream_id < len(self.wav_names):
            audio_filepath = self.wav_names[self.stream_id]

        reference_item = self._get_reference_item(audio_filepath)
        if not audio_filepath and reference_item is not None:
            audio_filepath = str(reference_item.get("audio_filepath", "") or "")
        reference_text = ""
        reference_translation = ""
        if reference_item is not None:
            reference_text = reference_item.get("text", "")
            reference_translation = reference_item.get("answer", "")

        if reference_text:
            self._wer_hyps.append(pred_text)
            self._wer_refs.append(reference_text)

        if self.output_manifest_path:
            item = {
                "audio_filepath": audio_filepath,
                "text": reference_text,
                "translation": reference_translation,
                "pred_text": pred_text,
                "pred_translation": pred_translation,
            }
            with open(self.output_manifest_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Compute WER once, when the last stream finishes.
        if self.wav_names and self.stream_id == len(self.wav_names) - 1:
            self._calculate_and_log_wer()

    def _get_reference_item(self, audio_filepath: str) -> Optional[dict]:
        """Get reference manifest item by absolute path, basename, or stream order."""
        if not audio_filepath:
            if self.stream_id < len(self.reference_manifest_items_ordered):
                return self.reference_manifest_items_ordered[self.stream_id]
            return None
        try:
            audio_abs = str(Path(audio_filepath).resolve())
        except Exception:
            audio_abs = audio_filepath
        item = self.reference_manifest_by_audio.get(audio_abs)
        if item is not None:
            return item
        item = self.reference_manifest_by_basename.get(Path(audio_filepath).name)
        if item is not None:
            return item
        if self.stream_id < len(self.reference_manifest_items_ordered):
            return self.reference_manifest_items_ordered[self.stream_id]
        return None

    @classmethod
    def _calculate_and_log_wer(cls) -> None:
        """Calculate corpus-level WER from the accumulated (hypothesis, reference) pairs and log it."""
        if cls._wer_calculated:
            return
        cls._wer_calculated = True

        if not cls._wer_refs:
            logging.warning("WER calculation skipped because no reference text is available (use --manifest).")
            return

        try:
            wer, tokens, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=cls._wer_hyps, references=cls._wer_refs
            )
            total_res = {
                "samples": len(cls._wer_hyps),
                "tokens": tokens,
                "wer": wer,
                "ins_rate": ins_rate,
                "del_rate": del_rate,
                "sub_rate": sub_rate,
            }
            logging.info(f"WER: {total_res}")
        except Exception as e:
            logging.warning(f"Failed to calculate WER: {e}")

    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a token sequence into a human-readable string (SpeechProcessor interface)."""
        return self._join_tokens(tokens)

    def _tokenize_text(self, text: Optional[str]) -> List[str]:
        """Tokenize text according to the configured latency unit (word or char)."""
        if not text:
            return []
        text = text.replace("…", "")  # keep token counts consistent with simulstream's own eval path
        if self.latency_unit == "char":
            return list(text.strip())
        return text.strip().split()

    def _join_tokens(self, tokens: List[str]) -> str:
        """Join tokens according to the configured latency unit."""
        if not tokens:
            return ""
        if self.latency_unit == "char":
            return "".join(tokens)
        return " ".join(tokens)

    @classmethod
    def cleanup_model(cls):
        """
        Explicitly clean up vLLM (used by NMT) and release resources. Registered as an atexit
        handler so the vLLM engine shuts down gracefully instead of erroring on process exit.
        """
        if cls.pipeline is not None:
            cls._calculate_and_log_wer()
        if cls.pipeline is not None and cls.pipeline.nmt_model is not None:
            try:
                if hasattr(cls.pipeline.nmt_model, 'nmt_model'):
                    vllm_engine = cls.pipeline.nmt_model.nmt_model
                    if hasattr(vllm_engine, 'llm_engine'):
                        from vllm.distributed import destroy_model_parallel

                        destroy_model_parallel()
                    del vllm_engine
                    cls.pipeline.nmt_model.nmt_model = None
                    logging.info("vLLM engine cleaned up")
            except Exception as e:
                logging.warning(f"Error during vLLM cleanup: {e}")
